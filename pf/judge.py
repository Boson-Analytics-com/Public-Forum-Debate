"""
Two-sided PF judge -- v2.

v1 of this module fixed the structural problem (reads debate_*.json, the
real two-sided archive, instead of the old one-sided round_*.json) but was
still a single LLM call producing scores nobody could check. Three concrete
weaknesses get fixed here:

1. SELF-CONSISTENCY. One temperature-0 call is one opinion. This version
   samples the judge N times (default 3) and takes the majority verdict,
   averaging the numeric scores across samples. If the samples don't agree
   on a winner, that disagreement is surfaced, not hidden -- a close round
   the judge itself can't call consistently is exactly the round you should
   NOT trust a single RFD from.

2. EVIDENCE-TIED SCORING. The model must now return a "score_basis" for
   each of the four criteria, referencing a specific attack id (attack#N)
   or contention (pro-contention#N / con-contention#N) that the score is
   grounded in. Those references are checked against the actual argument
   graph after the fact -- a score that cites an attack id that doesn't
   exist in this debate is flagged as ungrounded rather than trusted.

3. MECHANISM-SWAP / CLASH-RELEVANCE FLAG. The known blind spot: an attack
   can substitute a different policy mechanism than the one the resolution
   specifies while still passing citation_violations() (which checks
   fabrication, not relevance). The judge is now explicitly instructed to
   check every attack against the resolution's actual mechanism and flag
   swaps in "mechanism_swap_flags" -- this is a post-hoc safety net, not a
   fix at generation time (that belongs in opponent.py).

Evidence tracking depends on debate_state.json actually recording which
claim_ids backed each exchange. If your round.py/pf_engine.py doesn't save
an `evidence_used` (or similarly-named) list per attack yet, this module
still runs -- it just tells you explicitly that evidence-per-exchange isn't
being tracked, instead of silently pretending it has data it doesn't.

Usage:
    python judge_pf.py score  --debate corpus/debates/debate_20260907124513.json
    python judge_pf.py score  --debate corpus/debates/debate_....json --samples 5
    python judge_pf.py latest --corpus ./corpus
    python judge_pf.py trend  --corpus ./corpus
"""

import argparse
import json
import os
import re
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from dotenv import load_dotenv

load_dotenv()

MODEL = os.environ.get("BEDROCK_MODEL_ID")
BEDROCK_REGION = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1"

SIDES = ("pro", "con")
CRITERIA = ("argument_quality", "refutation", "impact", "weighing")

try:
    from pf import debate as dbt
    _citation_violations = dbt.citation_violations
except Exception:
    _citation_violations = lambda text: []


def other(side: str) -> str:
    return "con" if side == "pro" else "pro"


SYSTEM = """You are an experienced Public Forum judge writing a reason for decision (RFD)
for a full debate between PRO and CON. Judge BOTH sides on the same four criteria,
in this order of weight:

1. ARGUMENT QUALITY -- logical reasoning, strength of evidence, credibility of sourcing.
   A confident paragraph with no evidence beats nothing, but loses to a weaker-sounding
   paragraph with a real card behind it.

2. REFUTATION -- did each side actually answer what was thrown at it? An argument that
   went unanswered anywhere in the debate is DROPPED and you must treat it as conceded
   for the rest of the round, even if the side who dropped it later asserts the opposite.

3. IMPACT -- how much does the claimed consequence actually matter (lives, security,
   rights, economic harm, precedent)? Bigger, more certain, more direct impacts outweigh
   vague or speculative ones.

4. WEIGHING -- did either side actually compare impacts, using:
     MAGNITUDE    how big is the harm/benefit
     PROBABILITY  how likely is it to actually happen
     TIMEFRAME    how soon
     SCOPE        how many people are affected
   A side that wins a smaller argument but WEIGHS it explicitly can beat a side that
   wins a bigger argument but never explains why it matters more. Weighing usually
   decides close rounds -- reward it accordingly.

MECHANISM-SWAP CHECK (do this explicitly, it is a common way rounds get won on air):
   The resolution specifies one exact policy mechanism. For EVERY attack in the
   argument graph, check whether it actually argues against that mechanism, or
   quietly substitutes a different one (e.g. attacking "government hacking of
   existing vulnerabilities" when the resolution is about "mandated backdoors" --
   those are different mechanisms with different risk profiles). An attack that
   swaps mechanisms should NOT be credited as refuting the contention it targets,
   even if it sounds persuasive and even if it cites a real, unfabricated statistic.
   List every swap you find in "mechanism_swap_flags".

GROUNDING REQUIREMENT: for each of the four scores, you must cite what it's based
on using the reference tokens "attack#<id>" and/or "pro-contention#<n>" /
"con-contention#<n>" exactly as they appear in the metrics/transcript below. A
score with no real basis in the transcript is not a legitimate score.

Rules you must follow:
- Do NOT reward fluency alone. Judge substance.
- An analytic answer (reasoning, no card) is legitimate and can win a point, but loses
  to carded evidence that directly engages the same point.
- Judge only what was actually said in the transcript. Do not invent arguments neither
  side made, and do not resolve a clash neither side actually argued.
- Treat human-argued and tool-argued turns identically -- score the words on the page,
  not who or what produced them.
- The objective metrics given below are facts about the round (drops, attack/answer
  counts, contention status, evidence-tracking status). Use them; do not contradict
  them without explanation.
- If a side's final focus or summary raises a new argument or new evidence that was
  never established earlier in the round, do not credit it -- note it in
  "dropped_arguments" or your RFD instead, the same way a real judge would refuse to
  vote on a new argument in the last speech.

Return ONLY JSON, no preamble, no markdown fences:
{
  "winner": "pro" | "con",
  "confidence": "clear" | "close" | "very close",
  "scores": {
    "argument_quality": {"pro": 0-10, "con": 0-10},
    "refutation":       {"pro": 0-10, "con": 0-10},
    "impact":           {"pro": 0-10, "con": 0-10},
    "weighing":         {"pro": 0-10, "con": 0-10}
  },
  "score_basis": {
    "argument_quality": "e.g. pro-contention#1, attack#3",
    "refutation": "e.g. attack#2, attack#4",
    "impact": "e.g. con-contention#1",
    "weighing": "e.g. con's final focus weighed magnitude vs pro's summary, which did not weigh"
  },
  "mechanism_swap_flags": [
    {"attack_id": 3, "side": "con", "issue": "attacked lawful hacking of existing bugs, not the mandated-backdoor mechanism the resolution specifies"}
  ],
  "rfd": "4-8 sentences delivered the way a judge would read a ballot aloud, referencing specific clashes",
  "key_clash": "the single exchange that decided the round",
  "dropped_arguments": ["short description of each argument that went completely unanswered, and by whom"],
  "weighing_notes": "who weighed impacts against each other, and whether it was persuasive -- or note if neither side weighed at all",
  "pro_did_well": ["..."],
  "pro_should_fix": ["..."],
  "con_did_well": ["..."],
  "con_should_fix": ["..."]
}"""


# --- objective metrics, per side, from the argument graph ---------------------

_WEIGH_WORDS = {
    "magnitude": [r"\bmillion\b", r"\bbillion\b", r"\bmassive\b", r"\bsignificant\b",
                  r"\bcatastrophic\b", r"\bwidespread\b", r"\bsevere\b"],
    "probability": [r"\blikely\b", r"\bunlikely\b", r"\binevitable\b", r"\bcertain\b",
                    r"\brisk of\b", r"\bprobability\b", r"\bcould\b", r"\bwill\b"],
    "timeframe": [r"\bimmediately\b", r"\balready\b", r"\blong[- ]term\b", r"\bshort[- ]term\b",
                  r"\byears\b", r"\bright now\b", r"\bover time\b"],
    "scope": [r"\beveryone\b", r"\bnationwide\b", r"\bglobal(?:ly)?\b", r"\bmillions of\b",
              r"\bpopulation\b", r"\ball users\b", r"\bevery american\b"],
}

# round.py now records evidence per role: attack_evidence_used (the asker's
# cards) and response_evidence_used (the answerer's cards). Older archives
# from before that fix won't have either -- these fallbacks are only checked
# if the primary field is absent, so older debate_*.json files still parse
# without crashing (just with per_exchange_evidence_tracked=False).
_ATTACK_EVIDENCE_KEYS = ("attack_evidence_used", "evidence_used", "my_evidence_used")
_RESPONSE_EVIDENCE_KEYS = ("response_evidence_used", "defense_evidence_used",
                          "answer_evidence_used")


def _side_text_blob(st: dict, side: str) -> str:
    parts = []
    for c in st.get("contentions", {}).get(side, []):
        parts.append(c.get("text", ""))
    for sp in st.get("speeches", {}).get(side, {}).values():
        parts.append(sp.get("text", "") or "")
    for a in st.get("attacks", []):
        if a.get("by_side") == side:
            parts.append(a.get("text", "") or "")
        if a.get("target_side") == side:
            parts.append(a.get("response", "") or "")
    return "\n".join(p for p in parts if p)


def _weigh_signal_counts(text: str) -> dict:
    return {k: sum(1 for pat in pats if re.search(pat, text, re.I))
            for k, pats in _WEIGH_WORDS.items()}


def _contention_status(st: dict, side: str) -> dict:
    conts = st.get("contentions", {}).get(side, [])
    attacks_on_side = [a for a in st.get("attacks", []) if a.get("target_side") == side]
    conceded, held, contested = [], [], []
    for c in conts:
        n = c.get("n")
        hits = [a for a in attacks_on_side if a.get("target_contention_n") == n]
        if not hits:
            conceded.append(n)
        elif any(a.get("answered") for a in hits):
            held.append(n)
        else:
            contested.append(n)
    return {"run": len(conts), "conceded_to_opponent": conceded,
            "held": held, "contested_unanswered": contested}


def _refutation_stats(st: dict, side: str) -> dict:
    made = [a for a in st.get("attacks", []) if a.get("by_side") == side]
    faced = [a for a in st.get("attacks", []) if a.get("target_side") == side]
    return {
        "attacks_made": len(made),
        "attacks_answered_by_opponent": sum(1 for a in made if a.get("answered")),
        "attacks_faced": len(faced),
        "attacks_answered": sum(1 for a in faced if a.get("answered")),
        "attacks_unanswered": sum(1 for a in faced if not a.get("answered")),
    }


def _first_present(a: dict, keys: tuple) -> list:
    for k in keys:
        if a.get(k):
            return list(a[k])
    return []


def _evidence_stats(st: dict, side: str) -> dict:
    """Evidence actually cited per exchange, if the round file tracks it,
    attributed to whichever side actually cited it -- an attacker's cards
    belong to by_side, an answerer's cards belong to target_side. Falls back
    to just the constructive claim_ids and says so plainly if per-exchange
    evidence isn't tracked anywhere in this file."""
    attacks = st.get("attacks", [])
    any_field_present = any(
        any(k in a for k in _ATTACK_EVIDENCE_KEYS + _RESPONSE_EVIDENCE_KEYS)
        for a in attacks)
    exchange_cards = set()
    for a in attacks:
        if a.get("by_side") == side:
            exchange_cards.update(_first_present(a, _ATTACK_EVIDENCE_KEYS))
        if a.get("target_side") == side:
            exchange_cards.update(_first_present(a, _RESPONSE_EVIDENCE_KEYS))
    constructive_cards = set()
    for c in st.get("contentions", {}).get(side, []):
        constructive_cards.update(c.get("claim_ids", []) or [])
    return {
        "per_exchange_evidence_tracked": any_field_present,
        "constructive_cards": sorted(constructive_cards),
        "exchange_cards": sorted(exchange_cards) if any_field_present else [],
        "total_unique_cards": len(constructive_cards | exchange_cards) if any_field_present
                             else len(constructive_cards),
    }


def _new_material_in_final(st: dict, side: str) -> list:
    flags = []
    earlier = []
    for c in st.get("contentions", {}).get(side, []):
        earlier.append(c.get("text", ""))
    for a in st.get("attacks", []):
        if a.get("by_side") == side:
            earlier.append(a.get("text", "") or "")
        if a.get("target_side") == side:
            earlier.append(a.get("response", "") or "")
    for stage in ("rebuttal", "summary"):
        sp = st.get("speeches", {}).get(side, {}).get(stage)
        if sp and sp.get("text"):
            earlier.append(sp["text"])
    earlier_blob = "\n".join(earlier).lower()

    for stage in ("summary", "final_focus"):
        sp = st.get("speeches", {}).get(side, {}).get(stage)
        text = sp.get("text") if sp else None
        if not text:
            continue
        for hit in _citation_violations(text):
            m = re.search(r"'([^']+)'", hit)
            snippet = (m.group(1) if m else "").lower()
            if snippet and snippet not in earlier_blob:
                flags.append(f"{stage}: {hit} -- not established earlier in {side}'s own case")
    return flags


def metrics(st: dict) -> dict:
    out = {"resolution": st.get("resolution"), "human_side": st.get("human_side"),
           "tool_side": st.get("tool_side")}
    for side in SIDES:
        text = _side_text_blob(st, side)
        out[side] = {
            "argued_by": "human" if side == st.get("human_side") else "tool",
            "contentions": _contention_status(st, side),
            "refutation": _refutation_stats(st, side),
            "evidence": _evidence_stats(st, side),
            "citation_flags_in_speeches": sum(
                len(_citation_violations(sp.get("text", "") or ""))
                for sp in st.get("speeches", {}).get(side, {}).values()),
            "possible_new_material_in_closing": _new_material_in_final(st, side),
            "weighing_language_signals": _weigh_signal_counts(text),
            "speeches_given": sorted(st.get("speeches", {}).get(side, {}).keys()),
        }
    return out


# --- transcript, in the real fixed PF order, with reference ids -----------------

STAGE_LABELS = {"constructive": "CONSTRUCTIVE", "rebuttal": "REBUTTAL",
               "summary": "SUMMARY", "final_focus": "FINAL FOCUS"}
CROSSFIRE_LABELS = {"crossfire_1": "CROSSFIRE", "crossfire_2": "CROSSFIRE",
                    "grand_crossfire": "GRAND CROSSFIRE"}


def _speech_block(st: dict, side: str, stage: str) -> list:
    sp = st.get("speeches", {}).get(side, {}).get(stage)
    if not sp or not sp.get("text"):
        return []
    who = "HUMAN" if side == st.get("human_side") else "TOOL"
    return [f"{side.upper()} {STAGE_LABELS[stage]} ({who}):", sp["text"], ""]


def _crossfire_block(st: dict, stage: str) -> list:
    exchanges = [a for a in st.get("attacks", []) if a.get("round") == stage]
    if not exchanges:
        return []
    out = [f"{CROSSFIRE_LABELS[stage]} ({stage}):"]
    for a in exchanges:
        asker_who = "HUMAN" if a.get("by_side") == st.get("human_side") else "TOOL"
        answerer_who = "HUMAN" if a.get("target_side") == st.get("human_side") else "TOOL"
        out.append(f"  [attack#{a.get('id')}] {a['by_side'].upper()} ({asker_who}) asks: "
                   f"{a.get('text', '')[:300]}")
        out.append(f"  {a['target_side'].upper()} ({answerer_who}) answers "
                   f"[{'answered' if a.get('answered') else 'no clear answer'}]: "
                   f"{(a.get('response') or '')[:300]}")
    out.append("")
    return out


def transcript(st: dict) -> str:
    out = [f"RESOLUTION: {st.get('resolution')}",
           f"HUMAN argues {st.get('human_side', '?').upper()}   "
           f"TOOL argues {st.get('tool_side', '?').upper()}", ""]
    for side in SIDES:
        conts = st.get("contentions", {}).get(side, [])
        if conts:
            out.append(f"{side.upper()} CONTENTIONS:")
            for c in conts:
                out.append(f"  [{side}-contention#{c.get('n')}] [{c.get('area')}] "
                           f"{c.get('text', '')[:500]}")
            out.append("")

    out += _speech_block(st, "pro", "constructive")
    out += _speech_block(st, "con", "constructive")
    out += _crossfire_block(st, "crossfire_1")
    out += _speech_block(st, "pro", "rebuttal")
    out += _speech_block(st, "con", "rebuttal")
    out += _crossfire_block(st, "crossfire_2")
    out += _speech_block(st, "pro", "summary")
    out += _speech_block(st, "con", "summary")
    out += _crossfire_block(st, "grand_crossfire")
    out += _speech_block(st, "pro", "final_focus")
    out += _speech_block(st, "con", "final_focus")

    # Rebuttal-stage attacks and any attack not already shown under a
    # crossfire label (e.g. a straight rebuttal attack with no exchange
    # loop) still need to be visible with their reference ids.
    shown_ids = {a.get("id") for stage in ("crossfire_1", "crossfire_2", "grand_crossfire")
                for a in st.get("attacks", []) if a.get("round") == stage}
    leftover = [a for a in st.get("attacks", []) if a.get("id") not in shown_ids]
    if leftover:
        out.append("OTHER EXCHANGES (rebuttal-stage attacks):")
        for a in leftover:
            out.append(f"  [attack#{a.get('id')}] {a['by_side'].upper()} -> "
                       f"{a['target_side'].upper()}: {a.get('text', '')[:300]}")
            out.append(f"    response [{'answered' if a.get('answered') else 'unanswered'}]: "
                       f"{(a.get('response') or '')[:300]}")
        out.append("")
    return "\n".join(out)


# --- Bedrock adapter -------------------------------------------------------------

# Only accumulates calls made through THIS module's own adapter. When judging
# runs via `engine.py debate` it reuses round.py's client instead (see
# engine.py's run_debate()), so those calls land in round.py's TOKEN_TOTALS;
# this only fills in for a standalone `python judge.py score ...` process.
TOKEN_TOTALS = {"input_tokens": 0, "output_tokens": 0, "calls": 0}


class BedrockMessagesAdapter:
    def __init__(self, runtime_client, model_id: str):
        self.runtime_client = runtime_client
        self.model_id = model_id

    def create(self, *, model=None, max_tokens=2000, temperature=0.2, system=None, messages=None):
        bedrock_messages = []
        for msg in messages or []:
            content = msg.get("content", "")
            if isinstance(content, str):
                content = [{"text": content}]
            bedrock_messages.append({"role": msg["role"], "content": content})
        kwargs = {"modelId": self.model_id, "messages": bedrock_messages,
                 "inferenceConfig": {"maxTokens": max_tokens, "temperature": temperature}}
        if system:
            kwargs["system"] = [{"text": system}]
        try:
            response = self.runtime_client.converse(**kwargs)
        except Exception as e:
            raise SystemExit(
                f"Bedrock call failed (model={self.model_id}): {e}\n"
                f"Common causes: request throttling (wait and retry), a "
                f"context-length error (the debate transcript is too large "
                f"for this model), or an expired/invalid AWS credential. "
                f"Samples already collected in this scoring run are not "
                f"reused -- rerun `judge.py score` once the underlying issue "
                f"is resolved."
            ) from None
        blocks = response.get("output", {}).get("message", {}).get("content", [])
        text_blocks = [SimpleNamespace(type="text", text=b["text"])
                       for b in blocks if isinstance(b, dict) and "text" in b]
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


def _stamp(path: Path) -> str:
    m = re.search(r"(\d{14})", path.stem)
    return m.group(1) if m else "0"


def _find_debates(corpus: Path) -> list:
    d = corpus / "debates"
    if not d.exists():
        return []
    return sorted([p for p in d.glob("debate_*.json") if not p.stem.startswith("judged_")],
                 key=_stamp)


# --- grounding validation ---------------------------------------------------------

def _validate_grounding(decision: dict, st: dict) -> list:
    valid_attacks = {a.get("id") for a in st.get("attacks", [])}
    valid_conts = {f"{side}-{c.get('n')}" for side in SIDES
                  for c in st.get("contentions", {}).get(side, [])}
    issues = []
    for cat, ref in (decision.get("score_basis") or {}).items():
        ref_text = ref if isinstance(ref, str) else json.dumps(ref)
        for m in re.finditer(r"attack#(\d+)", ref_text):
            if int(m.group(1)) not in valid_attacks:
                issues.append(f"{cat}: cites attack#{m.group(1)}, which doesn't exist in this debate")
        for m in re.finditer(r"(pro|con)-contention#(\d+)", ref_text):
            key = f"{m.group(1)}-{m.group(2)}"
            if key not in valid_conts:
                issues.append(f"{cat}: cites {m.group(1)}-contention#{m.group(2)}, "
                              f"which doesn't exist in this debate")
    return issues


def _validate_mechanism_flags(decision: dict, st: dict) -> list:
    valid_attacks = {a.get("id") for a in st.get("attacks", [])}
    issues = []
    for f in decision.get("mechanism_swap_flags") or []:
        aid = f.get("attack_id")
        if aid not in valid_attacks:
            issues.append(f"mechanism_swap_flags cites attack#{aid}, which doesn't exist")
    return issues


# --- self-consistency sampling -----------------------------------------------------

def _call_once(client, system: str, user: str) -> dict | None:
    """Returns None (never raises) if the model's JSON came back truncated or
    unparseable -- this schema (per-side scores, score_basis, mechanism_swap_flags,
    rfd, dropped_arguments, weighing_notes, 4 feedback lists) is large enough on a
    content-rich debate that a single sample occasionally runs out of room. The
    caller drops a None sample and aggregates over whatever did parse, rather than
    letting one bad sample take down the whole judged result."""
    resp = client.messages.create(model=MODEL, max_tokens=4096, temperature=0.3,
                                  system=system, messages=[{"role": "user", "content": user}])
    raw = "".join(b.text for b in resp.content if b.type == "text").strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.S)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
        return None


def _mock_decision(m: dict) -> dict:
    def badness(side):
        c = m[side]["contentions"]
        r = m[side]["refutation"]
        return len(c["conceded_to_opponent"]) + len(c["contested_unanswered"]) + r["attacks_unanswered"]

    winner = "pro" if badness("pro") <= badness("con") else "con"
    return {
        "winner": winner, "confidence": "close",
        "scores": {k: {"pro": 5, "con": 5} for k in CRITERIA},
        "score_basis": {k: "[mock -- no model call made]" for k in CRITERIA},
        "mechanism_swap_flags": [],
        "rfd": "[mock decision -- based only on drop/answer counts, no model call made]",
        "key_clash": "[mock]", "dropped_arguments": [], "weighing_notes": "[mock]",
        "pro_did_well": ["[mock]"], "pro_should_fix": ["[mock]"],
        "con_did_well": ["[mock]"], "con_should_fix": ["[mock]"],
    }


def _aggregate(samples: list) -> dict:
    winners = [s.get("winner") for s in samples]
    counts = Counter(winners)
    majority_winner, majority_count = counts.most_common(1)[0]
    unanimous = majority_count == len(samples)

    avg_scores = {}
    for k in CRITERIA:
        avg_scores[k] = {}
        for side in SIDES:
            vals = [s.get("scores", {}).get(k, {}).get(side) for s in samples]
            vals = [v for v in vals if isinstance(v, (int, float))]
            avg_scores[k][side] = round(statistics.mean(vals), 1) if vals else None

    rep = next((s for s in samples if s.get("winner") == majority_winner), samples[0])

    seen, mech_flags = set(), []
    for s in samples:
        for f in s.get("mechanism_swap_flags") or []:
            key = (f.get("attack_id"), f.get("side"), f.get("issue", "")[:40])
            if key not in seen:
                seen.add(key)
                mech_flags.append(f)

    return {
        "winner": majority_winner,
        "sample_agreement": f"{majority_count}/{len(samples)}",
        "unanimous": unanimous,
        "sample_winners": winners,
        "confidence": rep.get("confidence") if unanimous else "very close (samples disagreed)",
        "scores": avg_scores,
        "score_basis": rep.get("score_basis"),
        "mechanism_swap_flags": mech_flags,
        "rfd": rep.get("rfd"),
        "key_clash": rep.get("key_clash"),
        "dropped_arguments": rep.get("dropped_arguments"),
        "weighing_notes": rep.get("weighing_notes"),
        "pro_did_well": rep.get("pro_did_well"),
        "pro_should_fix": rep.get("pro_should_fix"),
        "con_did_well": rep.get("con_did_well"),
        "con_should_fix": rep.get("con_should_fix"),
    }


def score(debate_path: Path, client, mock: bool, samples: int = 3, out_dir: Path | None = None):
    st = json.loads(debate_path.read_text(encoding="utf-8"))
    m = metrics(st)

    print(f"Debate: {debate_path.name}")
    print(f"HUMAN: {st.get('human_side', '?').upper()}   TOOL: {st.get('tool_side', '?').upper()}")
    for side in SIDES:
        c = m[side]["contentions"]
        r = m[side]["refutation"]
        e = m[side]["evidence"]
        print(f"  {side.upper():4} ({m[side]['argued_by']:5}): "
              f"held {len(c['held'])}, conceded-to-opp {len(c['conceded_to_opponent'])}, "
              f"contested-unanswered {len(c['contested_unanswered'])} | "
              f"attacks answered {r['attacks_answered']}/{r['attacks_faced']} | "
              f"cards {e['total_unique_cards']}"
              f"{'' if e['per_exchange_evidence_tracked'] else ' (constructive only -- per-exchange evidence not tracked in this file)'}")
        for f in m[side]["possible_new_material_in_closing"]:
            print(f"    [flag] {f}")

    if mock:
        decision = _mock_decision(m)
    else:
        user = (f"OBJECTIVE METRICS (facts about this debate, both sides):\n"
                f"{json.dumps(m, indent=2)}\n\nTRANSCRIPT (full order, with reference ids "
                f"for grounding your scores):\n{transcript(st)}")
        wanted = max(1, samples)
        raw_samples = []
        for i in range(wanted):
            s = _call_once(client, SYSTEM, user)
            if s is None:
                print(f"[sample {i + 1}/{wanted} came back truncated/unparseable -- skipping it]")
                continue
            raw_samples.append(s)
        if not raw_samples:
            sys.exit(f"All {wanted} judge samples came back truncated/unparseable. "
                     f"Try again, or raise max_tokens in _call_once() further.")
        if len(raw_samples) < wanted:
            print(f"[judging on {len(raw_samples)}/{wanted} samples -- the rest were dropped above]")
        decision = _aggregate(raw_samples)
        decision["_raw_samples"] = raw_samples
        decision["_samples_used"] = f"{len(raw_samples)}/{wanted}"

    ungrounded = _validate_grounding(decision, st) if not mock else []
    bad_mech_refs = _validate_mechanism_flags(decision, st) if not mock else []

    winner_side = decision.get("winner")
    winner_is_human = winner_side == st.get("human_side")
    result = {
        "debate_file": debate_path.name,
        "judged_at": datetime.now(timezone.utc).isoformat(),
        "samples_taken": 1 if mock else samples,
        "metrics": m, "decision": decision,
        "winner_side": winner_side,
        "winner_argued_by": "human" if winner_is_human else ("tool" if winner_side else "unknown"),
        "ungrounded_score_refs": ungrounded,
        "invalid_mechanism_flag_refs": bad_mech_refs,
        # Only reflects calls made through judge.py's OWN adapter -- accurate
        # for a standalone `python judge.py score ...` process, but reads as
        # zero when judging runs via `engine.py debate`, since that shares
        # round.py's client/adapter instead (see round.py's TOKEN_TOTALS for
        # that path's usage, already folded into the debate's own archive).
        "token_usage": dict(TOKEN_TOTALS),
    }

    target = (out_dir or debate_path.parent) / f"judged_{debate_path.stem}.json"
    target.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    print(f"\n{'=' * 74}")
    who = result["winner_argued_by"].upper()
    agreement = decision.get("sample_agreement", "1/1")
    conf = decision.get("confidence", "?")
    print(f"DECISION: {str(winner_side).upper()} wins ({conf})   agreement {agreement}   "
         f"argued by {who}")
    print(f"{'=' * 74}")
    if not decision.get("unanimous", True):
        print(f"[note] judge samples disagreed on the winner: {decision.get('sample_winners')}")
    sc = decision.get("scores", {})
    if sc:
        print(f"{'':18} {'PRO':>5} {'CON':>5}")
        for k in CRITERIA:
            d = sc.get(k, {})
            print(f"{k:18} {str(d.get('pro', '-')):>5} {str(d.get('con', '-')):>5}")
    if decision.get("score_basis"):
        print("\nSCORE BASIS:")
        for k, v in decision["score_basis"].items():
            print(f"  {k}: {v}")
    if ungrounded:
        print("\n[warning] some scores cite references that don't exist in this debate:")
        for u in ungrounded:
            print(f"  - {u}")
    if decision.get("mechanism_swap_flags"):
        print("\nMECHANISM-SWAP FLAGS (attacked a different policy than the resolution specifies):")
        for f in decision["mechanism_swap_flags"]:
            print(f"  - attack#{f.get('attack_id')} ({f.get('side')}): {f.get('issue')}")
    if bad_mech_refs:
        print("\n[warning] mechanism-swap flags reference invalid attack ids:")
        for u in bad_mech_refs:
            print(f"  - {u}")
    print(f"\nKEY CLASH: {decision.get('key_clash', '-')}")
    if decision.get("dropped_arguments"):
        print("\nDROPPED:")
        for d in decision["dropped_arguments"]:
            print(f"  - {d}")
    print(f"\nWEIGHING: {decision.get('weighing_notes', '-')}")
    print(f"\nRFD:\n{decision.get('rfd', '-')}")
    for side in SIDES:
        did_well = decision.get(f"{side}_did_well") or []
        should_fix = decision.get(f"{side}_should_fix") or []
        if did_well or should_fix:
            print(f"\n{side.upper()} ({m[side]['argued_by']}):")
            for x in did_well:
                print(f"  + {x}")
            for x in should_fix:
                print(f"  - {x}")
    print(f"\n-> {target}")
    return result


def latest(corpus: Path, client, mock: bool, samples: int):
    debates = _find_debates(corpus)
    if not debates:
        sys.exit(f"No debates in {corpus / 'debates'}. Finish one with the two-sided debate flow.")
    return score(debates[-1], client, mock, samples)


def trend(corpus: Path):
    d = corpus / "debates"
    judged = sorted(d.glob("judged_debate_*.json"), key=_stamp) if d.exists() else []
    if not judged:
        sys.exit("No judged debates yet. Run: python judge_pf.py latest")

    human_wins, tool_wins, disagreements = 0, 0, 0
    fixes = {"pro": Counter(), "con": Counter()}
    print(f"{len(judged)} judged debates\n")
    print(f"{'DEBATE':34} {'HUMAN':6} {'WINNER':7} {'WON BY':6} {'AGREEMENT':10}")
    for p in judged:
        r = json.loads(p.read_text(encoding="utf-8"))
        m, dec = r["metrics"], r["decision"]
        winner_by = r.get("winner_argued_by", "?")
        human_wins += winner_by == "human"
        tool_wins += winner_by == "tool"
        disagreements += not dec.get("unanimous", True)
        for side in SIDES:
            for f in dec.get(f"{side}_should_fix", []) or []:
                fixes[side][f[:70]] += 1
        print(f"{r['debate_file'][:34]:34} {m.get('human_side', '?'):6} "
              f"{str(r.get('winner_side', '?')):7} {winner_by:6} "
              f"{dec.get('sample_agreement', '-'):10}")

    print(f"\nRecord: human {human_wins} / tool {tool_wins} / {len(judged)} judged")
    print(f"Judge disagreed with itself on {disagreements}/{len(judged)} debates -- "
         f"treat those RFDs as low-confidence.")
    for side in SIDES:
        if fixes[side]:
            print(f"\nRepeated criticisms of {side.upper()}:")
            for f, n in fixes[side].most_common(6):
                print(f"   [{n}x] {f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["score", "latest", "trend"])
    ap.add_argument("--debate", type=Path, default=None)
    ap.add_argument("--corpus", type=Path, default=Path("./corpus"))
    ap.add_argument("--samples", type=int, default=3,
                    help="Self-consistency samples per judged debate (default 3)")
    ap.add_argument("--mock", action="store_true")
    args = ap.parse_args()

    if args.command == "trend":
        trend(args.corpus)
        return

    client = client_or_die(args.mock)

    if args.command == "score":
        if not args.debate:
            sys.exit("score needs --debate path/to/debate_*.json")
        score(args.debate, client, args.mock, args.samples)
    else:
        latest(args.corpus, client, args.mock, args.samples)

    if not args.mock and TOKEN_TOTALS["calls"]:
        u = TOKEN_TOTALS
        print(f"\nToken usage (judging): {u['input_tokens']} input + {u['output_tokens']} output "
             f"= {u['input_tokens'] + u['output_tokens']} total, across {u['calls']} call(s).")


if __name__ == "__main__":
    main()