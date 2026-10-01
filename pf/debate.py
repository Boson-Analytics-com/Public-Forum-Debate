"""
Step 7: the argument generator.

Reads the knowledge base and produces debate speech. Two commands:

    case     build the constructive case from the strongest claim clusters
    respond  answer an opponent's argument

The core idea: HOW it answers depends on what evidence exists.

    strong  ->  argue with quotes and citations
    thin    ->  lead with reasoning, use weak evidence only as support
    none    ->  reason only, and CITE NOTHING

In "none" mode a citation guard scans the output for years, percentages and
phrases like "a study found". If any appear they are fabrications, because no
evidence was supplied to the model. The response is regenerated, then blocked.

Every "none" is written to gaps.jsonl. Add sources covering those gaps and rerun
the kb/ pipeline (fetch, parse, chunk, claims, index) to turn them into real
evidence for next time. This is why the tool never searches the web
mid-round: live snippets would bypass the quote verification that makes the
knowledge base trustworthy.

Usage:
    python debate.py case    --corpus ./corpus --side pro
    python debate.py respond --corpus ./corpus --side pro --query "opponent's argument"
    python debate.py respond --corpus ./corpus --side pro --query "..." --format crossfire
    python debate.py gaps    --corpus ./corpus
"""

import argparse
import json
import os
import re
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from dotenv import load_dotenv

from kb import index as kb

load_dotenv()

MODEL = os.environ.get("BEDROCK_MODEL_ID")
BEDROCK_REGION = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1"

# Multiplier on every call's max tokens. 1.0 = full budget (production).
# Lower it (e.g. 0.5) only to hold cost down while testing -- set the same
# value in debate.py, round.py, claims.py and claims_bedrock_opus.py.
MAX_TOKENS_SCALE = 1.0

# Word limit for each side's speech, per stage (keys match round.py's
# DEBATE_ORDER). The crossfire stages are deliberately absent: they have no
# word limit and keep their own conversational FORMAT_RULES["crossfire"].
SPEECH_WORD_LIMITS = {
    "constructive": 600,
    "rebuttal": 500,
    "summary": 400,
    "final_focus": 400,
}

# The word limit is enforced by the prompt; max_tokens is only a backstop so a
# speech that runs slightly long isn't cut off mid-sentence. English prose runs
# ~1.33 tokens/word, so 1.6 leaves ~20% headroom. It's divided by
# MAX_TOKENS_SCALE because the Bedrock adapters (here and in round.py, which
# must match) multiply it back down -- otherwise the cost-saving scale would
# truncate every speech below its word target.
TOKENS_PER_WORD = 1.6


def count_words(text: str) -> int:
    """Words as a reader counts them: whitespace-separated tokens."""
    return len(re.findall(r"\S+", text or ""))


def trim_to_words(text: str, limit: int) -> str:
    """`text` cut to at most `limit` words, ending on the last full sentence
    inside the limit (or at the limit itself if that would drop more than
    half). Only removes text, never rewrites it, so quotes that survive are
    still verbatim."""
    words = list(re.finditer(r"\S+", text or ""))
    if len(words) <= limit:
        return text
    cut = text[:words[limit - 1].end()]
    ends = list(re.finditer(r"[.!?][\"')\]]*(?=\s|$)", cut))
    if ends and ends[-1].end() >= len(cut) // 2:
        cut = cut[:ends[-1].end()]
    return cut.rstrip()


def word_limit_tokens(words: int) -> int:
    """max_tokens backstop for text that should run about `words` words."""
    return int(words * TOKENS_PER_WORD / MAX_TOKENS_SCALE)


FORMAT_RULES = {
    "crossfire": "2-4 sentences. Conversational, direct, spoken aloud. No signposting.",
    "rebuttal": "About 150 words. Signpost clearly. Group the argument, then answer it.",
    "summary": "About 100 words. Weigh the clash. Say why your side wins it.",
    "final": "About 80 words. One clear reason you win. No new arguments.",
}

BASE = """You are arguing the {side} side of this Public Forum resolution:
{resolution}

You are a strong, composed debater. You never sound evasive and never bluff.
Speak in first person, as if talking. Do not use headers or bullet points.

FORMAT: {format_rule}"""

STRONG = """
You have solid evidence. Use it.
- Lead with your strongest card. Name the source and the finding.
- Quote only from the EVIDENCE block. Never alter the words inside quotation marks.
- Say why it beats what they said -- do not just assert it.
"""

THIN = """
Your evidence here is weak, so do not lean on it.
- Lead with reasoning, not the card.
- You may reference the evidence below once, as support. Do not overstate it.
- Do not imply you have more evidence than you do.
"""

NONE = """
You have NO evidence on this point. That is fine -- debaters answer uncarded
arguments constantly. Pick whichever of these fits best:

1. REFRAME -- their question is not the one that decides the round; name the one that is.
2. EXTEND YOUR WARRANT -- you already argued why your contention holds. Stretch that
   logic to cover this, without new facts.
3. SHIFT THE BURDEN -- they asserted something too. Ask what supports it.
4. CONCEDE NARROWLY -- grant the small point, then show it does not touch your
   contention. This reads as confident, not weak.

ABSOLUTE RULE: name no study, report, survey, statistic, percentage, year,
organisation, author or dollar figure. Not one. You have no source for any of it.
Argue from logic alone. If you catch yourself about to cite something, reframe instead.
"""

# --- citation guard -----------------------------------------------------------

CITE_PATTERNS = [
    (r"\b(19|20)\d{2}\b", "a year"),
    (r"\d+(?:\.\d+)?\s*(?:%|percent|percentage)", "a percentage"),
    (r"[$£€]\s?\d", "a money figure"),
    (r"\b(?:a|the|one|recent|another)\s+(?:\w+\s+){0,2}"
     r"(?:study|report|survey|paper|analysis|poll|review)\b", "a study reference"),
    (r"\baccording to\b", "an attribution"),
    (r"\b(?:research|studies|data|statistics|evidence)\s+(?:show|shows|shown|found|"
     r"finds|indicate|indicates|suggest|suggests|confirm|confirms)\b", "a research claim"),
    (r"\b\d+\s+(?:out of|in)\s+\d+\b", "a ratio statistic"),
    (r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:out of|in)\s+"
     r"(?:one|two|three|four|five|six|seven|eight|nine|ten|\d+)\b", "a spelled-out statistic"),
    (r"\b(?:most|many|the majority of|nearly all|virtually all|the vast majority of)\s+"
     r"(?:\w+\s+){0,2}(?:experts|researchers|studies|scientists|cryptographers|economists|"
     r"analysts|officials|reports)\b", "an appeal to expert consensus"),
    (r"\b(?:FBI|DOJ|NSA|CIA|Europol|Interpol|GCHQ|EFF|ACLU|Congress|Senate)\b",
     "a named organisation"),
]


def citation_violations(text: str) -> list:
    hits = []
    for pat, label in CITE_PATTERNS:
        m = re.search(pat, text, re.I)
        if m:
            hits.append(f"{label} ('{m.group()[:40]}')")
    return hits


# --- prompt assembly ----------------------------------------------------------

def evidence_block(claims: list) -> str:
    out = []
    for i, c in enumerate(claims, 1):
        out.append(
            f"[{i}] CLAIM: {c['claim']}\n"
            f"    WARRANT: {c.get('warrant')}\n"
            f"    IMPACT: {c.get('impact')}\n"
            f"    QUOTE: \"{c.get('quote')}\"\n"
            f"    CITE: {c.get('cite')}")
    return "\n\n".join(out)


def threat_block(claims: list) -> str:
    return "\n".join(f"- {c['claim']}" for c in claims) or "(none identified)"


# Only accumulates calls made through THIS module's own adapter -- i.e. when
# debate.py is run standalone (`python debate.py case/respond ...`). Calls
# made via round.py's client (the normal debate-run path, which threads its
# own client through respond()/build_case()) accumulate in round.py's
# TOKEN_TOTALS instead, since that's whose adapter actually handles them.
TOKEN_TOTALS = {"input_tokens": 0, "output_tokens": 0, "calls": 0}


class BedrockMessagesAdapter:
    """Anthropic-compatible messages adapter backed by Amazon Bedrock.

    debate.py and round.py both use the same small client interface:
        client.messages.create(...)

    The actual model request is sent through the Bedrock Runtime Converse API,
    so the project uses AWS credentials instead of ANTHROPIC_API_KEY.
    """

    def __init__(self, runtime_client, model_id: str):
        self.runtime_client = runtime_client
        self.model_id = model_id

    def create(self, *, model=None, max_tokens=900, temperature=0.3,
               system=None, messages=None):
        max_tokens = max(1, int(max_tokens * MAX_TOKENS_SCALE))
        bedrock_messages = []
        for msg in messages or []:
            content = msg.get("content", "")
            if isinstance(content, str):
                content = [{"text": content}]
            bedrock_messages.append({"role": msg["role"], "content": content})

        kwargs = {
            "modelId": self.model_id,
            "messages": bedrock_messages,
            "inferenceConfig": {
                "maxTokens": max_tokens,
                "temperature": temperature,
            },
        }
        if system:
            kwargs["system"] = [{"text": system}]

        try:
            response = self.runtime_client.converse(**kwargs)
        except Exception as e:
            raise SystemExit(
                f"Bedrock call failed (model={self.model_id}): {e}\n"
                f"Common causes: request throttling (wait and retry), a "
                f"context-length error (the prompt + round history is too "
                f"large for this model), or an expired/invalid AWS "
                f"credential. Any round/debate progress already saved to "
                f"disk before this call is not lost -- resume the same "
                f"command to continue from there."
            ) from None
        blocks = response.get("output", {}).get("message", {}).get("content", [])
        text_blocks = [
            SimpleNamespace(type="text", text=b["text"])
            for b in blocks if isinstance(b, dict) and "text" in b
        ]
        usage = response.get("usage", {}) or {}
        TOKEN_TOTALS["input_tokens"] += usage.get("inputTokens", 0) or 0
        TOKEN_TOTALS["output_tokens"] += usage.get("outputTokens", 0) or 0
        TOKEN_TOTALS["calls"] += 1
        return SimpleNamespace(content=text_blocks, usage=usage)


class BedrockClientAdapter:
    def __init__(self, runtime_client, model_id: str):
        self.messages = BedrockMessagesAdapter(runtime_client, model_id)


def client_or_die(mock: bool):
    if mock:
        return None

    if not MODEL:
        raise SystemExit(
            "BEDROCK_MODEL_ID is not set. Export it (or set it in .env and "
            "export it into the shell) before running, or use --mock."
        )

    try:
        import boto3
        session = boto3.Session(region_name=BEDROCK_REGION)
        if session.get_credentials() is None:
            raise SystemExit(
                "AWS credentials not found. Set AWS_ACCESS_KEY_ID and "
                "AWS_SECRET_ACCESS_KEY, configure an AWS profile/role, or use --mock."
            )
        runtime = session.client("bedrock-runtime")
        return BedrockClientAdapter(runtime, MODEL)
    except ImportError:
        raise SystemExit("boto3 is required for AWS Bedrock. Run: pip install boto3")


def call(client, system, user, max_tokens=900):
    resp = client.messages.create(model=MODEL, max_tokens=max_tokens, temperature=0.3,
                                  system=system, messages=[{"role": "user", "content": user}])
    return "".join(b.text for b in resp.content if b.type == "text").strip()


def load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# --- respond ------------------------------------------------------------------

def validate_query(q: str) -> str | None:
    """A meaningless query makes the model INVENT what the opponent said, and it
    will do so fluently. Better to refuse than to answer an imagined argument."""
    q = (q or "").strip()
    if len(q.split()) < 4:
        return "too short -- give the opponent's actual argument, not a label"
    if re.fullmatch(r"[.\s\-_?\"']+", q):
        return "no content"
    placeholders = {"opponent's argument", "opponents argument", "the argument",
                    "their argument", "query", "question", "test", "attack"}
    if q.lower().strip("\"'.") in placeholders:
        return "that is a placeholder, not an argument"
    return None


def respond(corpus: Path, cfg: dict, attack: str, side: str, fmt: str,
            client, mock: bool, verbose: bool = True, word_limit: int | None = None):
    """`word_limit`, when given, replaces the format's standalone length -- used
    when this answer is one share of a debate speech's word limit (round.py's
    rebuttal defense)."""
    bad = validate_query(attack)
    if bad:
        raise SystemExit(f"Query rejected: {bad}\n"
                         f"  got: {attack!r}\n"
                         f"  try: --query \"criminals will just switch to foreign encryption apps\"")

    r = kb.rebut(corpus, attack, side, k=4, verbose=False)
    strength = r["answer_strength"]

    format_rule = FORMAT_RULES[fmt]
    if word_limit:
        format_rule = re.sub(r"^About \d+ words\.",
                             f"About {word_limit} words. Stay within the "
                             f"{word_limit}-word limit.", format_rule)
    system = BASE.format(side=side.upper(), resolution=cfg["resolution"],
                         format_rule=format_rule)
    system += {"strong": STRONG, "thin": THIN, "none": NONE}[strength]

    user = (f"THEY JUST ARGUED:\n{attack}\n\n"
            f"THEIR LIKELY EVIDENCE BASE (do not repeat it as if it were yours):\n"
            f"{threat_block(r['threat'])}\n\n")
    user += (f"YOUR EVIDENCE:\n{evidence_block(r['answer'])}\n\nAnswer them."
             if r["answer"] else "YOU HAVE NO EVIDENCE ON THIS POINT.\n\nAnswer them.")

    if mock:
        text = {
            "strong": "[mock strong] They say X, but the evidence cuts the other way.",
            "thin": "[mock thin] Reasoning first, with light evidence support.",
            "none": "[mock none] That is not the question that decides this round.",
        }[strength]
        blocked, attempts = [], 1
    else:
        none_tokens = word_limit_tokens(word_limit) if word_limit else 300
        text = call(client, system, user, max_tokens=(
            none_tokens if strength == "none" else
            word_limit_tokens(word_limit) if word_limit else 900))
        attempts, blocked = 1, []
        if strength == "none":
            v = citation_violations(text)
            if v:
                attempts = 2
                text = call(client, system + (
                    f"\n\nYour previous answer contained {', '.join(v)}. You have NO "
                    f"source for that -- it is fabricated. Rewrite with no numbers, "
                    f"years, organisations or study references at all."), user, max_tokens=none_tokens)
                v2 = citation_violations(text)
                if v2:
                    blocked = v2
                    text = ("I don't have a specific source on that point, and I'm not "
                            "going to pretend otherwise. But it doesn't touch my "
                            "contention -- and my opponent hasn't given you a reason to "
                            "think it outweighs what I've already shown you.")

    used = [{"claim_id": c["claim_id"], "cite": c.get("cite")} for c in r["answer"]]
    out = {"attack": attack, "side": side, "format": fmt, "strength": strength,
           "contention_areas": r["contention_areas"], "response": text,
           "evidence_used": used, "citation_guard_attempts": attempts,
           "citation_guard_blocked": blocked}

    if strength == "none":
        gap = {"question": attack, "side": side,
               "contention_areas": r["contention_areas"],
               "logged_at": datetime.now(timezone.utc).isoformat()}
        with (corpus / "gaps.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(gap, ensure_ascii=False) + "\n")

    if verbose:
        print(f"\nTHEY ARGUED: {attack}")
        print(f"EVIDENCE:    {strength.upper()}   areas: {', '.join(r['contention_areas']) or '-'}")
        if blocked:
            print(f"GUARD:       BLOCKED fabrication -> {blocked}")
        elif attempts > 1:
            print("GUARD:       regenerated once (first attempt tried to cite)")
        print(f"\nRESPONSE ({fmt}):\n{text}")
        if used:
            print("\nEVIDENCE USED:")
            for u in used:
                print(f"  - {u['cite']}")
        else:
            print("\nEVIDENCE USED: none (logged to gaps.jsonl)")
    return out


# --- case ---------------------------------------------------------------------

EXTRACT_CONTENTIONS_SYSTEM = """You are segmenting a debater's own constructive speech into its distinct
contentions, arguing {side}. The speech is given below with each line numbered.

Identify EVERY distinct contention actually present in the speech -- however
many there are. Do not force any particular number, do not split one
contention into two, and do not merge two contentions into one.

For each contention, report:
  START_LINE / END_LINE -- the inclusive line number range (from the numbered
    speech below) that contains this contention's argument. Report line
    numbers ONLY -- do not reproduce the speech text itself here. Ranges
    should be given in the order the contentions appear and should not
    overlap.
  TITLE   -- a short (2-5 word) label for this contention
  CLAIM   -- the core one-sentence claim being made
  WARRANT -- the reasoning or mechanism for why the claim is true
  IMPACT  -- why this matters to the round
  QUOTE   -- if this contention quotes or cites a specific source (in
    quotation marks, or clearly attributed to a named source/study/report),
    copy that quote EXACTLY, character for character, from the speech. If
    this contention does not actually contain a quoted or cited source,
    leave this null -- do NOT invent or paraphrase one.
  CITE    -- the source name/attribution given for QUOTE. Leave null if
    QUOTE is null.

Reply as JSON only: {{"contentions": [{{"start_line": int, "end_line": int,
"title": str, "claim": str, "warrant": str, "impact": str, "quote": str|null,
"cite": str|null}}, ...]}}. Use null for any field the speech doesn't
actually support -- never invent content that isn't in the speech."""


def extract_contentions_from_speech(text: str, side: str, client, mock: bool) -> list[dict]:
    """Segment a human's freely-written constructive speech into however many
    contentions it actually contains.

    The model is never asked to reproduce contention text -- only to report a
    (start_line, end_line) range into the line-numbered speech shown to it.
    `text` on each returned dict is then sliced directly from the ORIGINAL
    lines by the code below, so it is verbatim by construction: there is no
    "check if the model copied correctly" step for it, because the model
    never had the chance to reword it in the first place. This is stronger
    than checking a model-reproduced excerpt against the source, which is
    exactly the failure mode this design avoids (a model that paraphrases
    just slightly would otherwise pass or fail the check unpredictably).

    `quote`/`cite` are a different matter -- a short citation the model DOES
    have to reproduce character-for-character -- so those are still returned
    unverified; callers (round.py's _collect_human_contentions_own()) must
    check `quote` against the human's original text before trusting it, same
    anti-fabrication principle as claims.py's locate_quote() guard on the KB
    side.

    A contention with a missing/non-numeric/out-of-range/inverted line range
    is dropped here (with a warning) rather than guessed at -- no fuzzy
    fallback, since that risks silently grabbing the wrong span.
    """
    lines = text.split("\n")
    if mock:
        return [{"title": "Mock Contention", "claim": text[:80] or None,
                 "warrant": None, "impact": None, "quote": None, "cite": None,
                 "text": text}]

    numbered = "\n".join(f"L{i + 1}: {line}" for i, line in enumerate(lines))
    system = EXTRACT_CONTENTIONS_SYSTEM.format(side=side.upper())
    raw = call(client, system, numbered, max_tokens=1200)
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        m = re.search(r"\{.*\}", raw, re.S)
        data = json.loads(m.group()) if m else {}

    out = []
    for c in data.get("contentions", []) or []:
        start, end = c.get("start_line"), c.get("end_line")
        valid = (isinstance(start, int) and isinstance(end, int)
                and 1 <= start <= end <= len(lines))
        if not valid:
            print(f"  [warning] extracted contention {c.get('title', '')!r} had an "
                  f"invalid line range ({start}-{end}) -- dropped")
            continue
        excerpt = "\n".join(lines[start - 1:end]).strip()
        if not excerpt:
            continue
        out.append({
            "title": c.get("title"), "claim": c.get("claim"),
            "warrant": c.get("warrant"), "impact": c.get("impact"),
            "quote": c.get("quote"), "cite": c.get("cite"), "text": excerpt,
        })
    return out


def build_case(corpus: Path, cfg: dict, side: str, client, mock: bool, n_contentions: int):
    """Pick the contention areas where your side is actually strong, then write
    a constructive case from the best claims in each."""
    con = sqlite3.connect(corpus / "kb.sqlite")
    con.row_factory = sqlite3.Row

    tally = {}
    for row in con.execute("SELECT contention_tags, stance FROM claims"):
        for t in json.loads(row["contention_tags"] or "[]"):
            d = tally.setdefault(t, {"own": 0, "opp": 0})
            if row["stance"] == side:
                d["own"] += 1
            elif row["stance"] in ("pro", "con"):
                d["opp"] += 1

    ranked = sorted(tally.items(),
                    key=lambda x: -(x[1]["own"] * x[1]["own"] / max(1, x[1]["own"] + x[1]["opp"])))
    chosen = [t for t, _ in ranked[:n_contentions]]
    print(f"Strongest areas for {side.upper()}:")
    for t in chosen:
        d = tally[t]
        share = 100 * d["own"] / max(1, d["own"] + d["opp"])
        print(f"   {t:32} {d['own']:4} own / {d['opp']:4} opposing  ({share:.0f}% yours)")

    # Each contention is its own call, so each gets an equal share of the
    # constructive's word limit -- the whole speech is what must fit.
    speech_words = SPEECH_WORD_LIMITS["constructive"]
    contention_words = speech_words // max(1, len(chosen))

    contentions = []
    used_claims, used_docs = set(), set()
    for tag in chosen:
        # Pull a wide pool, then diversify in Python. Sorting by "has a number"
        # alone makes the same quantified card win in every tag, which produces
        # three contentions resting on one source -- and one card kills all three.
        pool = con.execute("""
            SELECT * FROM claims WHERE stance = ? AND contention_tags LIKE ?
            ORDER BY (quantification IS NOT NULL) DESC,
                     (published_at IS NOT NULL) DESC, published_at DESC LIMIT 120
        """, (side, f'%"{tag}"%')).fetchall()

        claims, per_doc = [], Counter()
        for pass_no in (1, 2):
            for r in pool:
                if len(claims) >= 6:
                    break
                d = dict(r)
                d.pop("embedding", None)
                if d["claim_id"] in used_claims:
                    continue
                # Pass 1 takes only documents no earlier contention touched.
                if pass_no == 1 and d["doc_id"] in used_docs:
                    continue
                if per_doc[d["doc_id"]] >= 2:
                    continue
                per_doc[d["doc_id"]] += 1
                claims.append(d)
            if len(claims) >= 3:
                break

        if not claims:
            print(f"   (skipping {tag} -- no unused evidence left)")
            continue
        used_claims.update(c["claim_id"] for c in claims)
        used_docs.update(c["doc_id"] for c in claims)

        if mock:
            text = f"[mock contention on {tag}] built from {len(claims)} claims."
        else:
            system = BASE.format(side=side.upper(), resolution=cfg["resolution"],
                                 format_rule=(
                                     f"About {contention_words} words. Stay within the "
                                     f"{contention_words}-word limit. Spoken aloud, no "
                                     f"bullet points."))
            system += (f"\nWrite ONE contention: a claim, its mechanism, its impact, and the "
                       f"best card. Quote only from the evidence given, exactly as written.\n"
                       f"This contention is one of {len(chosen)} in a constructive speech of "
                       f"approximately {speech_words} words. Stay within the "
                       f"{speech_words}-word limit for the whole speech.\n")
            prior = ""
            if contentions:
                prior = ("\n\nEARLIER CONTENTIONS IN THIS CASE ALREADY USE THESE EXAMPLES:\n"
                         + "\n".join(f"- {c['example']}" for c in contentions)
                         + "\nBuild this contention on DIFFERENT ground. Do not reuse those "
                           "examples, statistics or incidents -- a case where every contention "
                           "rests on one story collapses when that story is attacked.")
            text = call(client, system,
                        f"CONTENTION AREA: {tag}\n\nAVAILABLE EVIDENCE:\n"
                        f"{evidence_block(claims)}{prior}",
                        max_tokens=word_limit_tokens(contention_words))
        primary = claims[0]
        contentions.append({
            "area": tag, "text": text,
            "title": tag.replace("_", " ").title(),
            "claim": primary.get("claim"),
            "warrant": primary.get("warrant"),
            "impact": primary.get("impact"),
            "evidence": [{"quote": c.get("quote"), "cite": c.get("cite"),
                         "claim": c.get("claim"), "claim_id": c["claim_id"]}
                        for c in claims],
            "claim_ids": [c["claim_id"] for c in claims],
            "sources": sorted({c.get("host") for c in claims if c.get("host")}),
            "example": (claims[0].get("claim") or "")[:120],
        })

    con.close()
    out = {"resolution": cfg["resolution"], "side": side, "contentions": contentions}
    path = corpus / f"case_{side}.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False))

    print(f"\n{'=' * 70}")
    for i, c in enumerate(contentions, 1):
        print(f"\nCONTENTION {i} -- {c['area']}\n{c['text']}")
    print(f"\n-> {path}")

    # A case is only three arguments if it rests on three different foundations.
    if len(contentions) > 1:
        shared = set.intersection(*(set(c["sources"]) for c in contentions))
        if shared:
            print(f"\nWARNING: every contention draws on {sorted(shared)}.")
            print("If that source is discredited, the whole case falls with it.")
    return out


def show_gaps(corpus: Path):
    path = corpus / "gaps.jsonl"
    if not path.exists():
        print("No gaps logged yet.")
        return
    gaps = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]
    print(f"{len(gaps)} unanswered questions logged\n")
    areas = Counter(a for g in gaps for a in g["contention_areas"])
    print("Weakest areas by gap count:")
    for a, n in areas.most_common():
        print(f"   {a:32} {n}")
    print("\nMost recent:")
    for g in gaps[-10:]:
        print(f"   - {g['question'][:90]}")
    print("\nAdd sources for these areas and rerun the kb/ pipeline to fill them.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["case", "respond", "gaps"])
    ap.add_argument("--corpus", type=Path, default=Path("./corpus"))
    ap.add_argument("--config", type=Path, default=Path("config/resolution.json"))
    ap.add_argument("--side", choices=["pro", "con", "both"], default=None,
                    help="'both' is for case prep; respond needs one side")
    ap.add_argument("--query", default=None)
    ap.add_argument("--format", choices=list(FORMAT_RULES), default="crossfire")
    ap.add_argument("--contentions", type=int, default=3)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if args.command == "gaps":
        show_gaps(args.corpus)
        return

    cfg = load_config(args.config)
    side = args.side or cfg.get("side_tool_defends", "pro")

    client = client_or_die(args.mock)

    if args.command == "case":
        # Which side the tool argues is decided by the app at runtime, so prep
        # both and load whichever is needed for that session.
        for s_ in (["pro", "con"] if side == "both" else [side]):
            print(f"\n{'#' * 70}\n# {s_.upper()} CASE\n{'#' * 70}")
            build_case(args.corpus, cfg, s_, client, args.mock, args.contentions)
    else:
        if not args.query:
            raise SystemExit("respond needs --query")
        if side == "both":
            raise SystemExit("respond needs one side: --side pro or --side con")
        out = respond(args.corpus, cfg, args.query, side, args.format,
                      client, args.mock, verbose=not args.json)
        if args.json:
            print(json.dumps(out, indent=2, ensure_ascii=False))

    if not args.mock and TOKEN_TOTALS["calls"]:
        u = TOKEN_TOTALS
        print(f"\nToken usage: {u['input_tokens']} input + {u['output_tokens']} output "
             f"= {u['input_tokens'] + u['output_tokens']} total, across {u['calls']} call(s).")


if __name__ == "__main__":
    main()