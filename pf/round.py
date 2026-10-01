"""
Step 9: round state -- human vs tool Public Forum debate.

debate.py answers one attack at a time with no memory. A real round needs
memory, because the speeches that win are the ones that reference what came
before:

  - the summary must extend the constructive, not restate it
  - the final focus must weigh clash that actually happened
  - a contention the opponent never answered is CONCEDED, and saying so is
    often how the round is won

A human picks pro or con, the tool argues the other side, and both go
through the fixed 11-step PF order (DEBATE_ORDER). State lives in
debate_state.json; every attack/answer from either side accumulates in one
argument graph (st["attacks"]).

Usage:
    python round.py debate-start  --human-side pro
    python round.py debate-run    --exchanges 4
    python round.py debate-status
"""

import argparse
import json
import os
import re
from types import SimpleNamespace
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from pf import debate as dbt
from pf import opponent as opp

load_dotenv()

MODEL = os.environ.get("BEDROCK_MODEL_ID")
BEDROCK_REGION = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1"

# Multiplier on every call's max tokens. 1.0 = full budget (production).
# Lower it (e.g. 0.5) only to hold cost down while testing -- set the same
# value in debate.py, round.py, claims.py and claims_bedrock_opus.py.
MAX_TOKENS_SCALE = 1.0


def now():
    return datetime.now(timezone.utc).isoformat()


# Real per-call token counts, accumulated across every Bedrock call made
# through THIS adapter in the current process. Bedrock's converse() response
# actually includes a "usage" block (inputTokens/outputTokens/totalTokens)
# that nothing in this codebase used to read -- every prior token estimate
# was a chars/4 guess from generated text length. debate_run() seeds this
# from debate_state.json's persisted "token_usage" at the start of each
# invocation and writes it back after every stage (see debate_run()), so the
# total stays correct across resumed debate-run calls in separate processes,
# not just within one.
TOKEN_TOTALS = {"input_tokens": 0, "output_tokens": 0, "calls": 0}


def _record_usage(usage: dict):
    TOKEN_TOTALS["input_tokens"] += usage.get("inputTokens", 0) or 0
    TOKEN_TOTALS["output_tokens"] += usage.get("outputTokens", 0) or 0
    TOKEN_TOTALS["calls"] += 1


class BedrockMessagesAdapter:
    """Small adapter so debate.py can keep its Anthropic-style client.messages.create call.

    The actual request is sent through Amazon Bedrock using the AWS credential
    chain (AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY, AWS profile, IAM role, etc.).
    """

    def __init__(self, runtime_client, model_id: str):
        self.runtime_client = runtime_client
        self.model_id = model_id

    def create(self, *, model, max_tokens=500, temperature=0.3, system=None, messages=None):
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
        text = response["output"]["message"]["content"]
        blocks = [SimpleNamespace(type="text", text=b["text"])
                  for b in text if "text" in b]
        usage = response.get("usage", {}) or {}
        _record_usage(usage)
        return SimpleNamespace(content=blocks, usage=usage)


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
                "AWS_SECRET_ACCESS_KEY, configure an AWS profile, or use --mock."
            )
        runtime = session.client("bedrock-runtime")
        return BedrockClientAdapter(runtime, MODEL)
    except ImportError:
        raise SystemExit("boto3 is required for AWS Bedrock. Run: pip install boto3")


# ==============================================================================
# SYMMETRIC PF DEBATE MODE -- human vs tool, real fixed PF order
# ==============================================================================
# A human picks pro or con, the tool automatically takes the other side, and
# both go through the actual fixed PF order -- constructive, crossfire,
# rebuttal, summary and final focus -- with a human turn at EVERY stage that
# belongs to the human's side.

DEBATE_INPUT_FN = input  # swappable for scripted/automated testing


def debate_ask(prompt: str) -> str:
    return DEBATE_INPUT_FN(prompt)


def debate_state_path(corpus: Path) -> Path:
    return corpus / "debate_state.json"


def load_debate(corpus: Path) -> dict:
    p = debate_state_path(corpus)
    if not p.exists():
        raise SystemExit("No debate in progress. Run: python round.py debate-start --human-side pro|con")
    return json.loads(p.read_text(encoding="utf-8"))


def save_debate(corpus: Path, st: dict):
    debate_state_path(corpus).write_text(json.dumps(st, indent=2, ensure_ascii=False))


def other_side(side: str) -> str:
    return "con" if side == "pro" else "pro"


def _claim_ids(evidence_used):
    """Normalize evidence_used to a flat list of claim_ids -- opponent.do_attack()
    returns bare ids, debate.respond() returns [{"claim_id":..., "cite":...}, ...].
    The judge just needs the ids either way."""
    if not evidence_used:
        return []
    return [e["claim_id"] if isinstance(e, dict) else e for e in evidence_used]


# Crossfire exchange cap meaning "no cap": the crossfire runs until the human
# chooses to end it at the continue checkpoint (the Streamlit UI always uses this).
UNLIMITED_EXCHANGES = 0

DEBATE_ORDER = [  # (n, stage, side, kind)  kind in {"speech", "crossfire"}
    (1, "constructive", "pro", "speech"),
    (2, "constructive", "con", "speech"),
    (3, "crossfire_1", None, "crossfire"),
    (4, "rebuttal", "pro", "speech"),
    (5, "rebuttal", "con", "speech"),
    (6, "crossfire_2", None, "crossfire"),
    (7, "summary", "pro", "speech"),
    (8, "summary", "con", "speech"),
    (9, "grand_crossfire", None, "crossfire"),
    (10, "final_focus", "pro", "speech"),
    (11, "final_focus", "con", "speech"),
]

# PF rule: crossfire 1 is asked first by whoever spoke first in round 1
# (always pro, since pro constructive is fixed first). crossfire 2 flips to
# the other speaker. grand crossfire returns to the round-1 speaker.
CROSSFIRE_FIRST_ASKER = {"crossfire_1": "pro", "crossfire_2": "con", "grand_crossfire": "pro"}

DEBATE_SUMMARY_SYSTEM = """You are a strong Public Forum debater arguing {my_side} on this resolution:
{resolution}

You are delivering the Summary Speech (Speaker 1). Your primary goal
is to CRYSTALLIZE the round down to the core points of clash.

Review the flow of the debate so far:
- Your Constructive case: {my_constructive}
- Opponent's Constructive case: {opp_constructive}
- Opponent's Rebuttal attacks: {opp_rebuttal}
- Your team's Rebuttal defense/attacks: {my_rebuttal}

Generate a summary of approximately {word_limit} words. Stay within the
{word_limit}-word limit. Structure it by executing these steps:
1. COLLAPSE & SELECT VOTING ISSUES: Narrow the debate down to 2 or 3 central
   key voting issues. Do not try to answer every scattered argument.
2. EXTEND CORE OFFENSE: Re-establish your team's main Constructive arguments
   that survived Rebuttal, proving why your link chain and evidence remain
   intact.
3. DEFEND AGAINST REBUTTAL: Neutralize the opponent's strongest Rebuttal
   attacks on your case.
4. IMPACT WEIGHING: Compare your impacts directly against theirs (explaining
   why your side wins on magnitude, probability, or urgency).

Speak aloud in the first person, clearly organized by key voting issue. Do NOT
introduce completely new arguments that were not presented earlier in the
round. Focus strictly on crystallizing why your team is winning the key
clashes.

{evidence_rule}"""

DEBATE_FINAL_SYSTEM = """You are a strong Public Forum debater arguing {my_side} on this resolution:
{resolution}

You are delivering the Final Focus Speech. This is the absolute final speech
of the debate. Your main objective is to write the judge's ballot by giving
clear, compelling reasons to vote for your team.

Review the flow of the debate coming out of the Summary and Grand Crossfire
rounds:
- Your team's Summary speech: {my_summary}
- Opponent's team Summary speech: {opp_summary}
- Grand Crossfire highlights: {grand_crossfire}

Generate a final focus of approximately {word_limit} words. Stay within the
{word_limit}-word limit. Make it persuasive by executing these steps:
1. FOCUS ON 1 OR 2 CORE VOTING ISSUES ("VOTERS"): Select only the 1 or 2 most
   important winning issues that were extended in your team's Summary speech.
2. STRICT RULE -- NO NEW ARGUMENTS: Do NOT introduce any new arguments,
   evidence, or responses that were not present in your Summary speech.
3. EXPLICIT IMPACT WEIGHING: Clearly explain why your winning impacts
   outweigh the opponent's side (focusing on magnitude, probability, scope,
   or timeframe).
4. BALLOT DIRECTIVE: End with a clear, decisive closing statement telling the
   judge why a vote for {my_side} is the only logical outcome of the round.

Speak aloud, first person, as a polished closing argument. Do NOT use headers
or bullet points.

{evidence_rule}"""


def debate_start(corpus: Path, cfg: dict, human_side: str, client, mock: bool,
                 n_contentions: int = 3, crossfire_exchanges: int | None = None):
    """Human picks a side; the tool automatically takes the other.

    `crossfire_exchanges`, when given, is saved on the debate state so a front
    end that doesn't pass it on every turn (app.py) keeps using the same value.
    The CLI leaves it unset and passes --exchanges to debate-run instead."""
    if human_side not in ("pro", "con"):
        raise SystemExit("--human-side must be pro or con")
    tool_side = other_side(human_side)

    # A long-running process (app.py) can start several debates in a row; each
    # one's token total starts from zero, not from what the last one spent.
    for k in TOKEN_TOTALS:
        TOKEN_TOTALS[k] = 0

    st = {
        "resolution": cfg["resolution"],
        "human_side": human_side, "tool_side": tool_side,
        "stage_index": 0,
        "contentions": {"pro": [], "con": []},
        "speeches": {"pro": {}, "con": {}},
        "attacks": [],           # the argument graph: every attack/answer, either side
        "token_usage": {"input_tokens": 0, "output_tokens": 0, "calls": 0},
        "started_at": now(),
    }
    if crossfire_exchanges is not None:
        st["crossfire_exchanges"] = crossfire_exchanges
    save_debate(corpus, st)
    print(f"Debate started. HUMAN argues {human_side.upper()}, TOOL argues {tool_side.upper()}.")
    print(f"Resolution: {cfg['resolution']}")

    # Only the tool's side needs a KB case: the human always writes their own
    # constructive, with no KB case offered or mixed in.
    case_path = corpus / f"case_{tool_side}.json"
    if not case_path.exists():
        print(f"[building tool's {tool_side.upper()} case]")
        dbt.build_case(corpus, cfg, tool_side, client, mock, n_contentions)

    # Capture whatever build_case() just spent, so it shows in the final
    # token total.
    st["token_usage"] = dict(TOKEN_TOTALS)
    save_debate(corpus, st)
    return st


def _kb_contentions(case_path: Path) -> list:
    """case_{side}.json's KB-built contentions, in debate-state shape, for the
    tool's own constructive. Labeled source: "tool"; nothing downstream
    (opponent.py, judge.py, citation_violations()) reads this field, it
    exists purely so the record stays honest about who wrote it."""
    case = json.loads(case_path.read_text(encoding="utf-8"))
    return [{"n": i, "area": c["area"], "text": c["text"],
             "title": c.get("title"), "claim": c.get("claim"),
             "warrant": c.get("warrant"), "evidence": c.get("evidence", []),
             "impact": c.get("impact"), "claim_ids": c.get("claim_ids", []),
             "source": "tool"}
            for i, c in enumerate(case["contentions"], 1)]


def _print_contentions(contentions: list):
    for c in contentions:
        print(f"\n{c.get('title') or c['area']}:\n{c['text']}")


def _contentions_from_own_speech(text: str, side: str, client, mock: bool) -> list:
    """The human's ENTIRE constructive speech, pasted in one shot -- however
    many contentions it actually contains is however many the case has, no
    fixed count or cap. debate.py's extract_contentions_from_speech() segments
    it into contentions by LINE RANGE rather than asking the model to
    reproduce text, so each contention's `text` is verbatim by construction
    (sliced from the human's own lines, never model-generated) -- see that
    function's docstring for why this is stronger than a
    reproduce-then-verify approach.

    Each contention's `quote` (a short citation, which the model DOES have to
    reproduce character-for-character) is still checked here against the
    FULL original paste before being trusted -- not just its own excerpt, so
    a quote that's fine but landed just outside its assigned line range isn't
    rejected on a technicality. A quote that doesn't appear anywhere in the
    paste is dropped, never kept on the model's word alone.

    An empty paste means no contentions at all, and the paste is asked for
    again (see _finish_own()).
    """
    text = (text or "").strip()
    if not text:
        print("  [empty -- no contentions from your own writing]")
        return []

    extracted = dbt.extract_contentions_from_speech(text, side, client, mock)
    contentions = []
    for j, item in enumerate(extracted, 1):
        quote, cite = item.get("quote"), item.get("cite")
        if quote and quote not in text:
            print("  [warning] extracted quote not found verbatim in your text -- dropped")
            quote, cite = None, None

        title = item.get("title") or f"Contention {j}"
        area = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_") or f"contention_{j}"
        contentions.append({
            "n": j, "area": area, "title": title,
            "claim": item.get("claim"), "warrant": item.get("warrant"),
            "evidence": [{"quote": quote, "cite": cite}] if quote else [],
            "impact": item.get("impact"), "text": item["text"],
            "claim_ids": [], "source": "human",
        })
    return contentions


def _stage_focus_text(st: dict, stage: str, side: str) -> str | None:
    """What a crossfire attack on `side` should actually draw from, given
    which crossfire stage this is -- each one has a different real target in
    PF, and attacking the same constructive text every time (the original
    behavior) made crossfire_2 and grand_crossfire just re-litigate
    crossfire_1 instead of engaging what was said LATER in the round:
      crossfire_1      -- the constructive (returns None; do_attack()'s own
                          default, since that's the only speech given yet)
      crossfire_2      -- `side`'s rebuttal
      grand_crossfire  -- `side`'s summary
    Falls back to None (constructive) if the expected speech isn't there yet
    for some reason -- DEBATE_ORDER guarantees it normally is by this point."""
    key = {"crossfire_2": "rebuttal", "grand_crossfire": "summary"}.get(stage)
    if not key:
        return None
    return st["speeches"][side].get(key, {}).get("text")


def _debate_tool_asks(corpus: Path, cfg: dict, target_side: str, client, mock: bool,
                      prior_attacks: list | None = None, contentions: list | None = None,
                      stage: str = "crossfire_1", focus_text: str | None = None,
                      own_focus_text: str | None = None):
    """Tool asking a question = tool attacking target_side's case. This is
    opponent.py's do_attack(), reused in reverse: pass target_side as the
    'tool_side' argument and it computes the true opponent (the real tool)
    and attacks target_side's contention. Passing this debate's st["attacks"]
    as prior_attacks makes target selection and the generated text aware of
    what's already been argued, so repeated crossfires don't re-hammer the
    same contention with the same point (see opponent.py's pick_target()/
    do_attack() for the fix itself).

    `contentions` should be st["contentions"][target_side] -- it's what
    pick_target() chooses from, so this attacks what target_side
    actually said in THIS debate, not whatever is frozen in
    case_{target_side}.json (which, for a human-argued side, may be an
    untouched KB snapshot the human only used as a reference menu).

    `stage`/`focus_text` (see _stage_focus_text()) tell do_attack() which
    speech to actually attack instead of always the constructive.
    `own_focus_text` is the asker's OWN matching speech (their rebuttal/
    summary), passed through so the attack can compare the two instead of
    reacting to the opponent's speech alone -- call _stage_focus_text() again
    with the asker's own side to build it."""
    return opp.do_attack(corpus, cfg, target_side, contentions, client, mock,
                         prior_attacks=prior_attacks, stage=stage, focus_text=focus_text,
                         own_focus_text=own_focus_text)


def _debate_tool_answers(corpus: Path, cfg: dict, tool_side: str, question: str, fmt: str,
                         client, mock: bool, word_limit: int | None = None):
    # respond() raises SystemExit on a too-short/placeholder query -- the right
    # behavior for a one-off `debate.py respond` CLI call, but fatal here: it
    # would kill the entire multi-stage debate over one throwaway crossfire
    # line. Catch it at the source instead of letting it unwind out of engine.py.
    bad = dbt.validate_query(question)
    if bad:
        return {"attack": question, "side": tool_side, "format": fmt, "strength": "none",
                "contention_areas": [], "response": (
                    "That wasn't a specific enough argument to respond to -- "
                    "no clash there. Let's move on."),
                "evidence_used": [], "citation_guard_attempts": 0,
                "citation_guard_blocked": [], "query_rejected": bad}
    return dbt.respond(corpus, cfg, question, tool_side, fmt, client, mock, verbose=False,
                       word_limit=word_limit)


def _pick_target_contention(st: dict, side: str) -> int | None:
    """Best-effort default target when the human doesn't name one: the
    opposing contention with the fewest attacks against it so far (i.e. the
    one least contested yet). Returns None if that side has no contentions."""
    conts = st["contentions"][side]
    if not conts:
        return None
    counts = {c["n"]: 0 for c in conts}
    for a in st["attacks"]:
        if a["target_side"] == side and a["target_contention_n"] in counts:
            counts[a["target_contention_n"]] += 1
    return min(counts, key=counts.get)


def _collect_multiline(label: str) -> str:
    """Terminal input: collect free text one line at a time until a line
    containing only END.

    A plain single input() prompt silently desyncs everything that follows the
    moment the human pastes more than one line: input() only reads up to the
    first newline, and the rest of the pasted text is left sitting in stdin,
    quietly consumed by whichever debate_ask() calls come next (e.g. the
    following "Continue crossfire?" and the next question/response prompt) as
    if the human had typed those answers -- with no error, just answers that
    don't match what was actually typed. This is exactly what crossfire's
    single-line "Your question:"/"Your response:" prompts used to be exposed
    to."""
    print(f"{label} (multiple lines OK -- end with a line containing only END):")
    lines = []
    while True:
        line = debate_ask("")
        if line.strip() == "END":
            break
        lines.append(line)
    return "\n".join(lines).strip()


# Human-facing version of opponent.py's CROSSFIRE_FOCUS -- same idea (each
# crossfire stage has a different real target), phrased as a short hint
# instead of a debater's system prompt.
HUMAN_CROSSFIRE_HINT = {
    "crossfire_1": "their constructive contentions",
    "crossfire_2": "what was said in the rebuttals",
    "grand_crossfire": "what was said in the summaries",
}

SPEECH_LABELS = {"constructive": "CONSTRUCTIVE", "rebuttal": "REBUTTAL",
                 "summary": "SUMMARY", "final_focus": "FINAL FOCUS"}


def _grand_crossfire_highlights(st: dict) -> str:
    """Condensed Q&A pairs from the grand_crossfire stage, for final focus's
    {grand_crossfire} slot -- truncated per exchange (a highlights reel, not
    the full verbatim exchange)."""
    lines = []
    for a in st["attacks"]:
        if a["round"] != "grand_crossfire":
            continue
        lines.append(f"- {a['by_side'].upper()} asked: \"{a['text'][:200]}\"")
        if a.get("response"):
            lines.append(f"  {a['target_side'].upper()} answered: \"{a['response'][:200]}\"")
    return "\n".join(lines) if lines else "(no grand crossfire exchanges)"


# ==============================================================================
# TURN-BASED ENGINE
# ==============================================================================
# The debate advances one TURN at a time rather than one blocking stage at a
# time, so the same engine drives both the terminal (debate_run() below) and
# the Streamlit UI (app.py), which can't sit inside an input() call:
#
#   debate_next_action()  what happens next: a human prompt, a tool turn, or done
#   debate_submit()       apply the human's answer to that prompt
#   debate_tool_turn()    run the tool's turn
#
# Where the round is inside the current stage (awaiting the constructive
# paste, which crossfire exchange, whose turn it is to ask) lives in st["pending"], saved
# after every turn -- so a crash or a closed browser tab resumes at the exact
# turn, including mid-crossfire. st["pending"] is removed when its stage
# ends, so a finished debate's state and archive have the same shape as
# before.

def _sync_tokens_from(st: dict):
    """Seed (not add to) the module-level accumulator from the saved total --
    each turn can be a different process (a resumed debate-run) or a
    long-lived one that has handled other debates (app.py), so TOKEN_TOTALS
    is only correct if restored from the state before every turn."""
    u = st.get("token_usage") or {}
    for k in TOKEN_TOTALS:
        TOKEN_TOTALS[k] = u.get(k, 0) or 0


def _save_turn(corpus: Path, st: dict):
    # Persisted after every turn so a crash mid-debate doesn't lose track of
    # tokens already spent -- same reasoning as saving the turn itself.
    st["token_usage"] = dict(TOKEN_TOTALS)
    save_debate(corpus, st)


def _begin_stage(corpus: Path, cfg: dict, st: dict, client, mock: bool,
                 crossfire_exchanges: int):
    n, stage, side, kind = DEBATE_ORDER[st.get("stage_index", 0)]
    print(f"\n{'#' * 60}\n# STEP {n}: {stage.upper()}" +
          (f" ({side.upper()})" if side else "") + f"\n{'#' * 60}")
    pending = {"n": n}
    if kind == "crossfire":
        print(f"\n=== {stage.replace('_', ' ').upper()} ===")
        print(f"[Focus: press on {HUMAN_CROSSFIRE_HINT.get(stage, 'the case')}]")
        pending.update(phase="ask", i=0, asker=CROSSFIRE_FIRST_ASKER[stage],
                       exchanges_done=0, max_exchanges=crossfire_exchanges)
    else:
        print(f"\n=== {side.upper()} {SPEECH_LABELS[stage]} ===")
        if side != st["human_side"]:
            pending["phase"] = "tool"
        elif stage == "constructive":
            pending["phase"] = "paste"
        else:
            pending["phase"] = "human"
    st["pending"] = pending
    if kind == "crossfire" and crossfire_exchanges < 0:
        _end_stage(st)


def _end_stage(st: dict):
    n, stage, _, kind = DEBATE_ORDER[st["pending"]["n"] - 1]
    if kind == "crossfire":
        print(f"\n[{stage} ended after {st['pending']['exchanges_done']} exchange(s)]")
    del st["pending"]
    st["stage_index"] = n


def _describe(st: dict) -> dict:
    """The action st["pending"] is waiting on. Human actions carry what a
    front end needs to ask for it: `input` (choice / yesno / text / speech),
    the terminal `prompt`/`label`/`intro` text, and a short `ui_label`."""
    p = st["pending"]
    n, stage, side, kind = DEBATE_ORDER[p["n"] - 1]
    human_side = st["human_side"]
    act = {"n": n, "stage": stage, "phase": p["phase"]}

    if kind == "crossfire":
        asker = p["asker"]
        answerer = other_side(asker)
        act.update(exchange=p["i"] + 1, max_exchanges=p["max_exchanges"],
                   hint=HUMAN_CROSSFIRE_HINT.get(stage, "the case"))
        if p["phase"] == "ask":
            act["side"] = asker
            if asker != human_side:
                return {**act, "kind": "tool", "what": f"{asker.upper()} (tool) asks a question"}
            return {**act, "kind": "human", "input": "text", "required": True,
                    "label": f"\n[{asker.upper()} asks] Your question",
                    "ui_label": "Your crossfire question"}
        if p["phase"] == "answer":
            act.update(side=answerer, question=p["q"])
            if answerer != human_side:
                return {**act, "kind": "tool", "what": f"{answerer.upper()} (tool) answers"}
            return {**act, "kind": "human", "input": "text", "required": False,
                    "label": f"[{answerer.upper()} answers] Your response",
                    "ui_label": "Your answer"}
        return {**act, "kind": "human", "side": human_side, "input": "yesno", "default": "y",
                "prompt": f"\nContinue {stage.replace('_', ' ')}? [Y/n] ",
                "ui_label": "Continue this crossfire?"}

    act["side"] = side
    if p["phase"] == "tool":
        return {**act, "kind": "tool",
                "what": f"{side.upper()} (tool) gives the {SPEECH_LABELS[stage].lower()}"}
    if stage == "constructive":
        return {**act, "kind": "human", "input": "text", "required": False,
                "intro": (f"\nWrite your own {side.upper()} case. Paste your ENTIRE "
                          f"constructive speech in one go, however many contentions it "
                          f"has -- the model will split it into contentions by itself. "
                          f"Limit: {dbt.SPEECH_WORD_LIMITS[stage]} words."),
                "label": f"\nYour {side.upper()} constructive speech",
                "ui_label": (f"Your {side.upper()} constructive speech "
                             f"({dbt.SPEECH_WORD_LIMITS[stage]}-word limit)")}
    limit = dbt.SPEECH_WORD_LIMITS[stage]
    prompt = (f"Attack their case and defend yours ({limit}-word limit). Type your rebuttal, "
              f"one line at a time. End with a line containing only END." if stage == "rebuttal"
              else f"Type your speech ({limit}-word limit), one line at a time. End with a "
                   f"line containing only END.")
    return {**act, "kind": "human", "input": "speech", "prompt": prompt,
            "ui_label": f"Your {SPEECH_LABELS[stage].lower()} ({limit}-word limit)"}


def _migrate_constructive_pending(corpus: Path, st: dict):
    """A debate saved by an older version may be waiting on the removed kb/own
    choice or KB-backfill offer: the choice becomes the paste prompt, and a
    pending backfill keeps the human's own draft, as if it were declined."""
    p = st.get("pending") or {}
    if p.get("phase") == "choose":
        p["phase"] = "paste"
    elif p.get("phase") == "backfill":
        side = DEBATE_ORDER[p["n"] - 1][2]
        draft = p.pop("draft", None) or []
        p.pop("backfill", None)
        _finish_own(st, side, p, draft)
    else:
        return
    save_debate(corpus, st)


def debate_next_action(corpus: Path, cfg: dict, client, mock: bool,
                       crossfire_exchanges: int | None = None) -> dict:
    """What the debate is waiting on: {"kind": "human" | "tool" | "done", ...}.
    Starts the next stage first if the last one just ended (printing its
    header). `crossfire_exchanges` falls back to the value saved by
    debate_start(), then 6."""
    st = load_debate(corpus)
    _sync_tokens_from(st)
    _migrate_constructive_pending(corpus, st)
    if crossfire_exchanges is None:
        crossfire_exchanges = st.get("crossfire_exchanges", 6)
    began = False
    while "pending" not in st and st.get("stage_index", 0) < len(DEBATE_ORDER):
        _begin_stage(corpus, cfg, st, client, mock, crossfire_exchanges)
        began = True
    if began:
        _save_turn(corpus, st)
    if "pending" not in st:
        return {"kind": "done", "n": len(DEBATE_ORDER)}
    return _describe(st)


def _finish_constructive(st: dict, side: str, contentions: list, source: str):
    # Deliberately does NOT write case_{side}.json, for either side or path.
    # That file is the KB-built backstop debate_start() created, and it stays
    # exactly as the KB built it so it's still there to reuse (e.g. a fresh
    # debate where this side is tool-argued instead of human-argued). What
    # was actually argued lives in st["contentions"][side] and
    # st["speeches"][side]["constructive"]; that's what everything downstream
    # (crossfire targeting, rebuttal, the judge) reads, via opponent.py's
    # pick_target()/do_attack() `contentions` argument.
    st["contentions"][side] = contentions
    st["speeches"][side]["constructive"] = {
        "text": "\n\n".join(c["text"] for c in contentions),
        "source": source, "at": now(),
    }
    _end_stage(st)


def _finish_own(st: dict, side: str, p: dict, contentions: list):
    # An empty case is fatal downstream: pick_target() indexes into its
    # scored list unconditionally and crashes with IndexError on an empty
    # one, same for _pick_target_contention()'s min() over an empty dict. So
    # the paste is asked for again until at least one contention comes back,
    # rather than letting an empty paste proceed into an unplayable debate.
    if contentions:
        _finish_constructive(st, side, contentions, "human")
        return
    print(f"\n{side.upper()} needs at least one contention to debate -- "
          f"let's go through this again.")
    p["phase"] = "paste"


def _record_exchange(st: dict, stage: str, p: dict, answer: str, answered: bool,
                     response_evidence: list):
    asker = p["asker"]
    answerer = other_side(asker)
    st["attacks"].append({
        "id": len(st["attacks"]) + 1, "round": stage,
        "by_side": asker, "target_side": answerer, "target_contention_n": p["target_n"],
        "text": p["q"], "response": answer, "answered": answered, "at": now(),
        "attack_evidence_used": p["attack_evidence"],
        "response_evidence_used": response_evidence,
    })
    for k in ("q", "target_n", "attack_evidence"):
        del p[k]
    p["asker"] = answerer
    p["exchanges_done"] += 1
    # Ending the crossfire is its own explicit checkpoint, decoupled from
    # whether it happens to be the human's turn to ask -- a blank at the
    # human's ask-prompt used to end the WHOLE crossfire. No checkpoint after
    # the last possible exchange; the stage ends there on its own.
    if p["max_exchanges"] == UNLIMITED_EXCHANGES or p["i"] < p["max_exchanges"] - 1:
        p["phase"] = "continue"
    else:
        _end_stage(st)


def speech_word_problem(stage: str, text: str) -> str | None:
    """Why a human speech can't be accepted for `stage`, or None if it's
    within its word limit. Crossfire has no limit."""
    limit = dbt.SPEECH_WORD_LIMITS.get(stage)
    n = dbt.count_words(text)
    if not limit or n <= limit:
        return None
    return (f"Your {SPEECH_LABELS[stage].lower()} is {n} words; the limit is {limit}. "
            f"Cut at least {n - limit} word(s) and submit it again.")


def _fit_tool_speech(stage: str, text: str, limit: int | None = None) -> str:
    """The tool's speech (or one part of it) trimmed to its word limit."""
    limit = limit or dbt.SPEECH_WORD_LIMITS[stage]
    fitted = dbt.trim_to_words(text, limit)
    if fitted != text:
        print(f"[tool speech trimmed] {SPEECH_LABELS[stage].lower()}: "
              f"{dbt.count_words(text)} -> {dbt.count_words(fitted)} words "
              f"(limit {limit})")
    return fitted


def debate_submit(corpus: Path, cfg: dict, value: str, client, mock: bool):
    """Apply the human's answer to the action debate_next_action() returned.
    An empty constructive paste or crossfire question, or a speech over its
    word limit, leaves the action unchanged, so the same prompt comes back."""
    st = load_debate(corpus)
    _sync_tokens_from(st)
    if "pending" not in st or _describe(st)["kind"] != "human":
        raise SystemExit("Nothing is waiting on the human -- call debate_next_action() first.")
    p = st["pending"]
    n, stage, side, kind = DEBATE_ORDER[p["n"] - 1]
    value = value or ""
    if kind == "speech":
        problem = speech_word_problem(stage, value)
        if problem:
            print(f"  [over word limit] {problem}")
            return

    if kind == "crossfire":
        answerer = other_side(p["asker"])
        if p["phase"] == "ask":
            if not value.strip():
                return
            # No "which contention are you targeting?" prompt -- a real
            # crossfire question doesn't require declaring a target number
            # first. Auto-picked the same way the tool's own attacks spread
            # across contentions (see opponent.py's pick_target()): whichever
            # of the opponent's contentions has been hit the fewest times so
            # far in this debate.
            p.update(phase="answer", q=value.strip(),
                     target_n=_pick_target_contention(st, answerer),
                     attack_evidence=[])  # human questions aren't tied to KB claim_ids (yet)
        elif p["phase"] == "answer":
            ans = value.strip()
            _record_exchange(st, stage, p, ans, bool(ans), [])
        elif value.strip().lower().startswith("n"):
            _end_stage(st)
        else:
            p["i"] += 1
            p["phase"] = "ask"

    elif stage == "constructive":
        contentions = _contentions_from_own_speech(value, side, client, mock)
        _finish_own(st, side, p, contentions)

    elif stage == "rebuttal":
        tool_side = st["tool_side"]
        st["speeches"][side]["rebuttal"] = {"text": value, "source": "human", "at": now()}
        # A human rebuttal IS an attack on the graph, same as a crossfire
        # question -- it just arrives as one block of text instead of a single
        # question. Without this, debate_status()/the tool's own summary and
        # final focus have no way of knowing your rebuttal ever touched their
        # case. Target auto-picked, same as crossfire.
        st["attacks"].append({
            "id": len(st["attacks"]) + 1, "round": "rebuttal",
            "by_side": side, "target_side": tool_side,
            "target_contention_n": _pick_target_contention(st, tool_side),
            "text": value, "response": None, "answered": None, "at": now(),
            "attack_evidence_used": [],      # human free-text isn't tied to KB claim_ids (yet)
            "response_evidence_used": [],
        })
        _end_stage(st)

    else:
        st["speeches"][side][stage] = {"text": value, "source": "human", "at": now()}
        _end_stage(st)

    _save_turn(corpus, st)


def _rebuttal_word_split(n_defenses: int) -> tuple[int, int | None]:
    """(offense words, words per defense) for the tool's rebuttal, which is an
    offense half plus one defense answer per undefended hit, all inside the
    one rebuttal word limit. With nothing to defend the offense gets it all;
    otherwise 60/40, less a few words for the joining transition line."""
    limit = dbt.SPEECH_WORD_LIMITS["rebuttal"]
    if not n_defenses:
        return limit, None
    offense = limit * 3 // 5
    return offense, max(20, (limit - offense - 10) // n_defenses)


def _tool_rebuttal(corpus: Path, cfg: dict, st: dict, side: str, client, mock: bool):
    human_side, tool_side = st["human_side"], st["tool_side"]
    # Tool's rebuttal has two real jobs, not one:
    #   1. attack the human's WHOLE case, contention by contention, down the
    #      flow -- not just one cherry-picked point. do_rebuttal_speech()
    #      produces one speech that attacks every contention, each with its
    #      own KB-grounded evidence lookup.
    #   2. actually defend its OWN contentions against whatever the human hit
    #      them with in crossfire_1 or in their own rebuttal just before, so
    #      the rebuttal reads like a real PF rebuttal instead of a second
    #      unrelated attack.
    # Which of its own contentions' hits still need a defense. Worked out
    # before the offense half so the two halves can share the rebuttal's one
    # word limit instead of each taking a full-length budget.
    hits_on_tool = [x for x in st["attacks"] if x["target_side"] == tool_side
                    and not (x["response"] and x.get("round") != "rebuttal")]
    offense_words, defense_words = _rebuttal_word_split(len(hits_on_tool))
    a = opp.do_rebuttal_speech(corpus, cfg, tool_side, st["contentions"][human_side],
                               client, mock, word_limit=offense_words)
    a["text"] = _fit_tool_speech("rebuttal", a["text"], offense_words)
    for pc in a["per_contention"]:
        st["attacks"].append({
            "id": len(st["attacks"]) + 1, "round": "rebuttal", "by_side": tool_side,
            "target_side": human_side, "target_contention_n": pc["n"],
            "text": a["text"], "response": None, "answered": None, "at": now(),
            "attack_evidence_used": pc["evidence_used"],
            "response_evidence_used": [],
        })

    defenses = []
    for x in hits_on_tool:  # (hits already answered live in crossfire aren't redone)
        out = _debate_tool_answers(corpus, cfg, tool_side, x["text"], "rebuttal", client, mock,
                                   word_limit=defense_words)
        x["response"] = _fit_tool_speech("rebuttal", out["response"], defense_words)
        x["answered"] = out["strength"] != "none"
        x["response_evidence_used"] = _claim_ids(out.get("evidence_used"))
        defenses.append(x["response"])

    # do_rebuttal_speech()'s output (a["text"]) is only the offense half --
    # it's told not to write its own closing line, specifically so this
    # transition is what joins it to the defense half, instead of a defense
    # paragraph just landing after an already-concluded speech.
    speech_text = a["text"] + (
        "\n\nNow, turning to defend our own case:\n\n" + "\n\n".join(defenses)
        if defenses else "")
    # Backstop for the whole speech: with many defenses, each one's floor
    # of 20 words can add up past the limit.
    speech_text = _fit_tool_speech("rebuttal", speech_text)
    st["speeches"][side]["rebuttal"] = {"text": speech_text, "source": "tool", "at": now()}
    print(f"\n{speech_text}")


def _tool_summary_or_final(st: dict, side: str, stage: str, client, mock: bool):
    opp_side = other_side(side)
    evidence_rule = ("Only reference evidence and arguments that were already "
                     "introduced earlier in this round by either side -- do not "
                     "bring in new sources, statistics, or studies.")
    word_limit = dbt.SPEECH_WORD_LIMITS[stage]
    max_tokens = dbt.word_limit_tokens(word_limit)
    if stage == "summary":
        # All round context (constructive/rebuttal text from both sides)
        # lives directly in the system prompt, so the user message is just a
        # short trigger.
        system = DEBATE_SUMMARY_SYSTEM.format(word_limit=word_limit,
            my_side=side.upper(), resolution=st["resolution"],
            my_constructive=st["speeches"][side].get("constructive", {}).get("text", "(none)"),
            opp_constructive=st["speeches"][opp_side].get("constructive", {}).get("text", "(none)"),
            opp_rebuttal=st["speeches"][opp_side].get("rebuttal", {}).get("text", "(none)"),
            my_rebuttal=st["speeches"][side].get("rebuttal", {}).get("text", "(none)"),
            evidence_rule=evidence_rule)
        ctx = "Give your summary speech now, following the steps above."
    else:
        # All round context (summary speeches + grand crossfire) lives in the
        # system prompt itself, same as summary.
        system = DEBATE_FINAL_SYSTEM.format(word_limit=word_limit,
            my_side=side.upper(), resolution=st["resolution"],
            my_summary=st["speeches"][side].get("summary", {}).get("text", "(none)"),
            opp_summary=st["speeches"][opp_side].get("summary", {}).get("text", "(none)"),
            grand_crossfire=_grand_crossfire_highlights(st),
            evidence_rule=evidence_rule)
        ctx = "Give your final focus speech now, following the steps above."

    if mock:
        text = f"[mock {stage}] weighing based on the argument graph."
    else:
        resp = client.messages.create(model=MODEL, max_tokens=max_tokens,
                                      temperature=0.3, system=system,
                                      messages=[{"role": "user", "content": ctx}])
        text = "".join(b.text for b in resp.content if b.type == "text").strip()
    text = _fit_tool_speech(stage, text)

    st["speeches"][side][stage] = {"text": text, "source": "tool", "at": now()}
    print(f"\n{text}")


def debate_tool_turn(corpus: Path, cfg: dict, client, mock: bool):
    """Run the tool turn debate_next_action() returned."""
    st = load_debate(corpus)
    _sync_tokens_from(st)
    if "pending" not in st or _describe(st)["kind"] != "tool":
        raise SystemExit("It isn't the tool's turn -- call debate_next_action() first.")
    p = st["pending"]
    n, stage, side, kind = DEBATE_ORDER[p["n"] - 1]

    if kind == "crossfire":
        asker = p["asker"]
        answerer = other_side(asker)
        if p["phase"] == "ask":
            a = _debate_tool_asks(corpus, cfg, answerer, client, mock, prior_attacks=st["attacks"],
                                  contentions=st["contentions"][answerer], stage=stage,
                                  focus_text=_stage_focus_text(st, stage, answerer),
                                  own_focus_text=_stage_focus_text(st, stage, asker))
            print(f"\n[{asker.upper()} (tool) asks] {a['text']}")
            p.update(phase="answer", q=a["text"], target_n=a.get("targets_contention"),
                     attack_evidence=_claim_ids(a.get("evidence_used")))
        else:
            out = _debate_tool_answers(corpus, cfg, st["tool_side"], p["q"], "crossfire",
                                       client, mock)
            print(f"[{answerer.upper()} (tool) answers] {out['response']}")
            # "answered" means backed by real evidence strength, not just
            # "said some words back" -- otherwise a zero-evidence answer
            # would make a contention look HELD to the judge when nothing
            # actually defended it.
            _record_exchange(st, stage, p, out["response"], out["strength"] != "none",
                             _claim_ids(out.get("evidence_used")))
    elif stage == "constructive":
        case_path = corpus / f"case_{side}.json"
        if not case_path.exists():
            dbt.build_case(corpus, cfg, side, client, mock, 3)
        contentions = _kb_contentions(case_path)
        # Each contention gets an equal share of the constructive's limit, so
        # the joined speech fits it.
        share = dbt.SPEECH_WORD_LIMITS["constructive"] // max(1, len(contentions))
        for c in contentions:
            c["text"] = _fit_tool_speech("constructive", c["text"], share)
        _print_contentions(contentions)
        _finish_constructive(st, side, contentions, "tool")
    elif stage == "rebuttal":
        _tool_rebuttal(corpus, cfg, st, side, client, mock)
        _end_stage(st)
    else:
        _tool_summary_or_final(st, side, stage, client, mock)
        _end_stage(st)

    _save_turn(corpus, st)


def contention_status(st: dict, side: str, n: int) -> str:
    atk = [a for a in st["attacks"] if a["target_side"] == side and a["target_contention_n"] == n]
    return "CONCEDED" if not atk else ("HELD" if any(a["answered"] for a in atk) else "CONTESTED")


def debate_status(corpus: Path):
    st = load_debate(corpus)
    print(f"HUMAN: {st['human_side'].upper()}   TOOL: {st['tool_side'].upper()}")
    for side in ("pro", "con"):
        print(f"\n{side.upper()} contentions:")
        for c in st["contentions"][side]:
            print(f"  {c['n']}. {c.get('title') or c['area']:24} "
                  f"[{contention_status(st, side, c['n'])}]")
    print(f"\nSpeeches given: pro={list(st['speeches']['pro'])}  con={list(st['speeches']['con'])}")
    print(f"Argument graph edges: {len(st['attacks'])}")
    if "pending" in st:
        a = _describe(st)
        print(f"Next up: step {a['n']} ({a['stage']}) -- "
              f"{a.get('what') or 'waiting on the human: ' + a['ui_label']}")
    u = st.get("token_usage")
    if u:
        print(f"Token usage so far: {u['input_tokens']} input + {u['output_tokens']} output "
             f"= {u['input_tokens'] + u['output_tokens']} total, across {u['calls']} call(s).")


def _cli_answer(action: dict) -> str:
    """Terminal front end for one human action."""
    if action.get("intro"):
        print(action["intro"])
    if action["input"] == "yesno":
        return debate_ask(action["prompt"])
    if action["input"] == "text":
        text = _collect_multiline(action["label"])
        while action.get("required") and not text.strip():
            text = _collect_multiline(action["label"])
        return text
    print(action["prompt"])
    lines = []
    while True:
        line = debate_ask("")
        if line.strip() == "END":
            break
        lines.append(line)
    return "\n".join(lines)


def debate_archive_path(corpus: Path, st: dict) -> Path:
    """Stable per debate: keyed off when it STARTED, so re-finishing re-saves
    the same file rather than making a new one."""
    stamp = re.sub(r"[^0-9]", "", st["started_at"])[:14]
    return corpus / "debates" / f"debate_{stamp}.json"


def debate_finish(corpus: Path) -> Path:
    """Archive a finished debate to corpus/debates/ and return the path."""
    st = load_debate(corpus)
    if st.get("stage_index", 0) < len(DEBATE_ORDER):
        raise SystemExit("The debate isn't finished yet -- nothing to archive.")
    u = st.get("token_usage", TOKEN_TOTALS)
    print(f"\nToken usage (cumulative for this debate): "
         f"{u['input_tokens']} input + {u['output_tokens']} output "
         f"= {u['input_tokens'] + u['output_tokens']} total, across {u['calls']} call(s).")

    path = debate_archive_path(corpus, st)
    path.parent.mkdir(exist_ok=True)
    st["ended_at"] = now()
    path.write_text(json.dumps(st, indent=2, ensure_ascii=False))
    print(f"\nDebate archived -> {path}")
    return path


def debate_run(corpus: Path, cfg: dict, client, mock: bool,
               crossfire_exchanges: int | None = None):
    """Drive the full fixed 11-step PF order in the terminal, resuming from
    exactly where the saved state left off (down to the crossfire exchange)."""
    while True:
        action = debate_next_action(corpus, cfg, client, mock, crossfire_exchanges)
        if action["kind"] == "done":
            break
        if action["kind"] == "tool":
            debate_tool_turn(corpus, cfg, client, mock)
        else:
            debate_submit(corpus, cfg, _cli_answer(action), client, mock)
    return debate_finish(corpus)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["debate-start", "debate-run", "debate-status"])
    ap.add_argument("--corpus", type=Path, default=Path("./corpus"))
    ap.add_argument("--config", type=Path, default=Path("config/resolution.json"))
    ap.add_argument("--human-side", choices=["pro", "con"], default=None,
                    help="debate-start: which side the human argues; tool takes the other")
    ap.add_argument("--exchanges", type=int, default=None,
                    help="debate-run: max exchanges per crossfire stage, 0 = no limit "
                         "(default: the value saved at debate start, else 6)")
    ap.add_argument("--mock", action="store_true")
    args = ap.parse_args()

    if args.command == "debate-status":
        debate_status(args.corpus)
        return

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    client = client_or_die(args.mock)

    if args.command == "debate-start":
        if not args.human_side:
            raise SystemExit("debate-start needs --human-side pro|con")
        debate_start(args.corpus, cfg, args.human_side, client, args.mock)
    else:
        debate_run(args.corpus, cfg, client, args.mock, crossfire_exchanges=args.exchanges)


if __name__ == "__main__":
    main()
