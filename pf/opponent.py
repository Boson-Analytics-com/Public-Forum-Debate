"""
Step 11: the AI opponent -- the tool's side of a human-vs-tool debate.

Attacks the human's case in crossfire (do_attack) and delivers the tool's
full down-the-flow rebuttal speech (do_rebuttal_speech). Both are called from
round.py's debate flow, which passes its own Bedrock client in.

The opponent uses the SAME knowledge base, filtered to its own side. That matters:
an opponent inventing arguments from the model's general knowledge is not a useful
sparring partner, because it will attack with claims your evidence cannot engage.
An opponent drawn from your corpus attacks where your corpus is actually weak.

It also gets the same citation guard. An opponent that fabricates statistics
teaches the tool to answer fabrications, which is training on noise.

Mechanism-swap / clash-relevance guard, same one-retry-then-flag shape as the
citation guard. The resolution specifies one exact policy mechanism. An attack
that argues against a DIFFERENT mechanism (e.g. "government hacking of existing
vulnerabilities" when the resolution is about "mandated backdoors") can pass the
citation guard cleanly -- it's not fabricating anything -- while still not actually
engaging the round. The check is generation-time: it runs before the attack is
handed back to round.py, so a swapped attack either gets rewritten to engage the
real mechanism, or is returned with an explicit "mechanism_swap_flag" so it isn't
silently trusted downstream (by round.py's argument graph or by the judge).
"""

import os
import sqlite3
from pathlib import Path

from dotenv import load_dotenv

from pf import debate as dbt
from kb import index as kb

load_dotenv()

# An opponent-specific override, falling back to the shared BEDROCK_MODEL_ID.
# The request itself goes through whichever client round.py passes in.
MODEL = (os.environ.get("OPPONENT_BEDROCK_MODEL_ID")
        or os.environ.get("BEDROCK_MODEL_ID"))

ATTACK_SYSTEM = """You are a strong Public Forum debater arguing {opp_side} on this resolution:
{resolution}

{focus_intro}

A good attack does ONE of these, not all:
- deny the link ("their mechanism does not actually produce that outcome")
- deny the impact ("even if true, it does not matter as much as they claim")
- turn it ("their own evidence proves my side")
- outweigh it ("grant it entirely -- my impact is bigger, likelier or sooner")

Pick the strongest single line of attack and commit to it. Do not list four weak
objections -- that is how rounds are lost.

Your attack MUST engage the resolution's actual mechanism above, not a different
policy that merely resembles it (e.g. do not swap a mandated-backdoor resolution
for an attack on lawful hacking of existing vulnerabilities -- those are different
mechanisms with different risk profiles, and a swap doesn't actually refute anything).

Speak aloud, first person, no headers or bullets. Two to five sentences.

{evidence_rule}"""

# Each crossfire stage has a different real target in PF, and pointing every
# stage's attack at the same constructive contention text (the original
# behavior) made crossfire_2 and grand_crossfire just re-litigate crossfire_1
# instead of actually engaging what got said LATER in the round. Keyed by
# DEBATE_ORDER's stage name; (focus_intro, material_label, own_label) --
# material_label captions the opponent's text in the user message, own_label
# captions the asker's OWN matching speech when one is available (do_attack()
# only includes that block when `own_focus_text` is passed in -- crossfire_1
# has no prior rebuttal/summary to show, so it's just None there).
CROSSFIRE_FOCUS = {
    "crossfire_1": (
        "Your opponent has just run the constructive contention shown below. "
        "Attack its foundational mechanism or the card it leans on.",
        "THEIR CONTENTION", "YOUR CONTENTION"),
    "crossfire_2": (
        "Your opponent just gave the rebuttal shown below, answering attacks "
        "and defending their case. Your own rebuttal is shown too, for "
        "comparison. Find the SINGLE WEAKEST specific claim in THEIR "
        "rebuttal -- an answer that dodged the actual question, a defense "
        "with no real warrant behind it, or a point where they conceded more "
        "than they realize -- and target that exact claim by name. Do not "
        "attack the rebuttal as a vague whole, and do not just re-run your "
        "original attack on their constructive; engage what they actually "
        "said in the rebuttal, using your own rebuttal to show where the "
        "clash actually stands.",
        "THEIR REBUTTAL", "YOUR REBUTTAL"),
    "grand_crossfire": (
        "Your opponent just gave the summary speech shown below, where they "
        "claimed to be winning specific clashes and weighed impacts. Your own "
        "summary is shown too. Directly compare the two and frame this as the "
        "voting issue the judge should decide on: dispute their weighing "
        "(magnitude, probability, timeframe, or scope) against yours, or show "
        "that a clash they claimed to win was never actually established "
        "earlier in the round.",
        "THEIR SUMMARY", "YOUR SUMMARY"),
}

WITH_EVIDENCE = """Use the evidence provided. Quote it exactly as written and name the source."""

NO_EVIDENCE = """You have NO evidence on this point. Argue from logic alone.
Name no study, report, statistic, percentage, year, organisation or author.
Not one. If you find yourself about to cite something, reframe instead."""

# --- full down-the-flow rebuttal ------------------------------------------------
# A real PF rebuttal goes through the ENTIRE opponent case, contention by
# contention, not just one cherry-picked point -- do_attack()/pick_target()
# above are built for a single crossfire-style exchange, which is right for
# crossfire but understates a rebuttal speech. do_rebuttal_speech() below is
# the one place in this codebase that argues against every opposing
# contention in a single call.
REBUTTAL_SPEECH_SYSTEM = """You are an elite Public Forum debater giving the REBUTTAL speech, arguing
{tool_side} on this resolution:
{resolution}

Your opponent just ran the full constructive case below. Go DOWN THE FLOW: take
their contentions in the order given, explicitly naming which one you are on
before you attack it (e.g. "Turning to their first contention on ..."), and
take each one down before moving to the next. Do not skip any of them.

For EACH contention, use CLAIM -> WARRANT -> IMPACT: state what you're arguing,
why it's true, and why it matters to the round. Vary your refutation mechanics
across contentions -- mix mitigation (their impact is smaller than claimed),
logical take-outs (their link/mechanism doesn't hold), and turns (their own
point actually helps your side) rather than using the same move every time.

Your attacks MUST engage the resolution's actual mechanism above, not a
different policy that merely resembles it.

Speak aloud, first person, no headers or bullet points -- signpost with spoken
transitions instead. Assertive, analytical, persuasive. {length_rule}

Do NOT write a closing or summary line ("in conclusion", "therefore the
resolution should be...", etc.) -- this is only the attacking half of your
rebuttal. End on your last contention's attack. A separate part defending
your own case may be appended after this, and a hard conclusion here would
make that read as a disconnected afterthought instead of the same speech.

{evidence_rule}"""

NO_EVIDENCE_PER_CONTENTION = """Some contentions below are marked "YOU HAVE NO EVIDENCE AGAINST THIS
CONTENTION" -- for those specific contentions, argue from logic alone. Name no
study, report, statistic, percentage, year, organisation or author for that
contention. If you find yourself about to cite something there, reframe
instead. This restriction applies ONLY to the no-evidence contentions -- for
the others, use the evidence given exactly as written."""

# --- mechanism-swap / clash-relevance guard ------------------------------------

MECHANISM_CHECK_SYSTEM = """You are a strict debate-relevance checker, not a debater.

RESOLUTION (the exact policy mechanism being debated):
{resolution}

A debater just wrote the TEXT below, aimed at the CONTENTION shown. Decide whether
the text actually engages the resolution's mechanism, or whether it quietly swaps in
a different policy or mechanism that only resembles it (e.g. attacking "government
hacking of existing software vulnerabilities" when the resolution mandates
"backdoors built into encryption by technology companies" -- those are different
mechanisms with different risk profiles, even though both involve government access
to encrypted data).

An analytic argument with no evidence is fine and should pass, AS LONG AS it argues
against the resolution's actual mechanism. Weak reasoning is not a swap. A different
policy is a swap.

Reply with EXACTLY one line, nothing else:
OK
or
SWAP: <one sentence naming the different mechanism it actually argued against>"""


def _check_mechanism_relevance(client, mock: bool, resolution: str,
                               contention_text: str, candidate_text: str):
    """Returns (ok: bool, reason: str | None). Never raises -- a checker that
    can crash the round it's supposed to be guarding is worse than no checker."""
    if mock:
        return True, None
    try:
        user = f"CONTENTION:\n{contention_text[:600]}\n\nTEXT TO CHECK:\n{candidate_text}"
        out = call(client, MECHANISM_CHECK_SYSTEM.format(resolution=resolution), user,
                  max_tokens=80)
        out = out.strip()
        if out.upper().startswith("OK"):
            return True, None
        if out.upper().startswith("SWAP"):
            return False, out.split(":", 1)[-1].strip() if ":" in out else out
        # Unparseable reply -- don't block the round over a formatting miss.
        return True, None
    except Exception:
        return True, None


def _mechanism_guarded(client, mock: bool, resolution: str, contention_text: str,
                       text: str, system_for_retry: str, user_for_retry: str,
                       strength: str):
    """One retry, same shape as the citation guard: if the first pass swaps the
    resolution's mechanism, ask once more with the swap named explicitly. If the
    retry still swaps it, keep the original text but return a flag so round.py /
    the judge know this exchange didn't actually engage the round."""
    ok, reason = _check_mechanism_relevance(client, mock, resolution, contention_text, text)
    if ok:
        return text, None

    if mock:
        return text, reason

    retry_system = system_for_retry + (
        f"\n\nYour previous answer was flagged as arguing against a DIFFERENT "
        f"mechanism than the resolution specifies ({reason}). Rewrite it so it "
        f"engages the resolution's actual mechanism directly -- do not swap "
        f"policies again.")
    text2 = call(client, retry_system, user_for_retry)
    ok2, reason2 = _check_mechanism_relevance(client, mock, resolution, contention_text, text2)
    if ok2:
        return text2, None
    # Still swapped after one retry -- keep the retry's text (it's the more
    # recent attempt) but flag it. Downstream code decides what to do with a
    # flagged exchange; this function's job is only to not hide the problem.
    return text2, reason2


def call(client, system, user, max_tokens=800):
    resp = client.messages.create(model=MODEL, max_tokens=max_tokens, temperature=0.4,
                                  system=system, messages=[{"role": "user", "content": user}])
    return "".join(b.text for b in resp.content if b.type == "text").strip()


def guarded(client, system, user, strength, mock):
    """Same fabrication check the tool gets. An opponent that invents evidence
    trains the tool against arguments it will never actually face."""
    if mock:
        # Long enough to pass the tool's query validator, so mock mode can run a
        # full round end to end without hitting the API.
        return (f"[mock opponent, evidence={strength}] Their mechanism does not "
                f"actually produce the impact they claim, and even granting it "
                f"entirely, the harm on our side arrives sooner and reaches more "
                f"people."), []
    text = call(client, system, user)
    if strength != "none":
        return text, []
    v = dbt.citation_violations(text)
    if not v:
        return text, []
    text = call(client, system + f"\n\nYour previous answer contained {', '.join(v)}. "
                f"You have no source for that. Rewrite with no numbers, years, "
                f"organisations or study references.", user)
    v2 = dbt.citation_violations(text)
    if v2:
        return ("I don't have a specific card on that, but the burden is on them to "
                "show their mechanism actually delivers the impact they claim."), v2
    return text, []


def pick_target(corpus: Path, tool_side: str, contentions: list,
                prior_attacks: list | None = None):
    """Choose which of `tool_side`'s contentions to attack.

    `contentions` is `st["contentions"][tool_side]` from the live debate
    state -- what was actually said in THIS debate, not case_{tool_side}.json
    (a frozen KB snapshot round.py's debate_constructive() never rewrites, so
    for a human-argued side it can be stale). `prior_attacks` is
    `st["attacks"]`: the least-attacked contention wins, with how much
    opposing evidence the corpus holds only breaking ties.
    """
    opp_side = "con" if tool_side == "pro" else "pro"
    already_attacked = {}
    for a in prior_attacks or []:
        if a.get("target_side") == tool_side:
            n = a.get("target_contention_n")
            already_attacked[n] = already_attacked.get(n, 0) + 1

    con = sqlite3.connect(corpus / "kb.sqlite")
    scored = []
    for i, c in enumerate(contentions, 1):
        mine = con.execute("SELECT COUNT(*) FROM claims WHERE stance=? AND contention_tags LIKE ?",
                           (opp_side, f'%"{c["area"]}"%')).fetchone()[0]
        theirs = con.execute("SELECT COUNT(*) FROM claims WHERE stance=? AND contention_tags LIKE ?",
                            (tool_side, f'%"{c["area"]}"%')).fetchone()[0]
        hits = already_attacked.get(i, 0)
        ratio = mine / max(1, theirs)
        scored.append((hits, -ratio, i, c, mine, theirs, ratio))
    con.close()
    # Sort by hits ASCENDING first, ratio only breaking ties among equally-fresh
    # contentions -- before this fix, a single contention with a high enough KB
    # ratio kept winning the discounted score (ratio / (1+hits)) even after
    # being hit 2-3x already, so the tool would attack the same contention
    # repeatedly (seen in practice: contention 3 hit 4 times across one debate,
    # twice within the same grand crossfire, while another contention was only
    # ever hit once) instead of covering the opponent's whole case. Now every
    # contention must be hit an equal number of times before any of them gets
    # hit again.
    scored.sort()
    _, _, i, c, _, _, _ = scored[0]
    return c, i


def do_attack(corpus: Path, cfg: dict, tool_side: str, contentions: list,
              client, mock: bool, prior_attacks: list | None = None,
              stage: str = "crossfire_1", focus_text: str | None = None,
              own_focus_text: str | None = None):
    """`stage` selects which CROSSFIRE_FOCUS entry frames the attack, and
    `focus_text` is the material actually being attacked -- pass the target's
    rebuttal speech text for "crossfire_2", or their summary text for
    "grand_crossfire" (round.py's _stage_focus_text() builds this from the
    live debate state). `own_focus_text` is the ASKER's own matching speech
    (their own rebuttal for crossfire_2, their own summary for
    grand_crossfire) -- included so the attack can explicitly compare the two
    ("your rebuttal already showed X, but their rebuttal claims Y") instead of
    reacting to the opponent's speech in a vacuum. pick_target() still chooses
    WHICH contention this attack counts against for the argument graph either
    way; only the text fed to the model, and the framing around it, changes.
    Falls back to the contention's own constructive text (crossfire_1's
    behavior) when `focus_text` is None.
    """
    opp_side = "con" if tool_side == "pro" else "pro"
    contention, n = pick_target(corpus, tool_side, contentions, prior_attacks)
    focus_intro, material_label, own_label = CROSSFIRE_FOCUS.get(stage, CROSSFIRE_FOCUS["crossfire_1"])
    material_text = focus_text or contention["text"]

    # The opponent searches ITS side against the material it is attacking.
    r = kb.rebut(corpus, material_text[:600], opp_side, k=4, verbose=False)
    strength = r["answer_strength"]

    system = ATTACK_SYSTEM.format(
        opp_side=opp_side.upper(), resolution=cfg["resolution"], focus_intro=focus_intro,
        evidence_rule=WITH_EVIDENCE if r["answer"] else NO_EVIDENCE)
    user = (f"{material_label} ({contention['area']}):\n{material_text}\n\n")
    if own_focus_text:
        user += f"{own_label}:\n{own_focus_text}\n\n"

    # Same-side, same-contention prior attacks in this debate -- without this,
    # do_attack() is a pure function of static contention text + a KB query
    # that doesn't change over the round, so an unprompted call (crossfire,
    # grand crossfire) tends to regenerate the same point nearly verbatim
    # every time it's asked. Telling the model what it already said, and what
    # answer it already got, is what makes later crossfires actually advance
    # the argument instead of re-litigating the first one.
    same_target = [a for a in (prior_attacks or [])
                   if a.get("by_side") == opp_side and a.get("target_contention_n") == n]
    if same_target:
        user += "YOU HAVE ALREADY MADE THESE POINTS AGAINST THIS CONTENTION IN THIS DEBATE:\n"
        for a in same_target:
            user += f"- You said: {a.get('text', '')[:300]}\n"
            if a.get("response"):
                user += f"  They answered: {a['response'][:300]}\n"
        user += ("\nDo NOT repeat any of those points, even reworded. Make a genuinely "
                 "new argument against this contention, or directly counter how they "
                 "answered you above.\n\n")

    user += (f"YOUR EVIDENCE:\n{dbt.evidence_block(r['answer'])}\n\nAttack it."
             if r["answer"] else "YOU HAVE NO EVIDENCE HERE.\n\nAttack it.")

    text, blocked = guarded(client, system, user, strength, mock)
    text, mechanism_flag = _mechanism_guarded(
        client, mock, cfg["resolution"], material_text, text, system, user, strength)

    return {"mode": "attack", "opp_side": opp_side, "targets_contention": n,
            "area": contention["area"], "strength": strength, "text": text,
            "evidence_used": [e["claim_id"] for e in r["answer"]],
            "guard_blocked": blocked,
            "mechanism_swap_flag": mechanism_flag}


def do_rebuttal_speech(corpus: Path, cfg: dict, tool_side: str,
                       opponent_contentions: list, client, mock: bool,
                       word_limit: int | None = None):
    """`word_limit` defaults to the whole rebuttal's limit; round.py passes a
    smaller share when a defense half will be appended to this text.

    One full down-the-flow rebuttal: attack EVERY contention in the
    opponent's constructive case, not just one. Each contention gets its own
    KB evidence lookup (same query shape as do_attack/pick_target) so every
    point stays grounded -- or, when the KB has nothing left for that
    specific contention, the model is told to argue from logic alone and
    cite nothing for that one point.

    Each contention's lookup is independent, though, and doesn't know what
    the OTHER contentions in this same speech already retrieved -- the same
    strong claim can easily be the top match for two different contention
    areas (confirmed in practice: one claim_id came back for both
    law_enforcement_access and organized_crime_cartels against the same
    KB). Without deduping, the model can end up citing the identical quote
    in two different sections of what's supposed to be one speech. So
    claim_ids already used by an earlier contention in this call are
    filtered out of every later contention's evidence before it's ever shown
    to the model -- a contention that only had a duplicate to offer is
    treated as having no evidence for this speech, same as if the KB were
    genuinely empty there.

    NOTE: unlike do_attack(), this does not run the mechanism-swap guard --
    that checker compares one generated text against one contention, and
    doesn't have a natural per-contention meaning against a single speech
    covering several. A mechanism swap inside one paragraph of a long
    rebuttal is not caught here.
    """
    resolution = cfg["resolution"]

    blocks, per_contention, any_bare = [], [], False
    used_claim_ids = set()
    for c in opponent_contentions:
        label = c.get("title") or c.get("area") or f"contention {c.get('n')}"
        r = kb.rebut(corpus, c["text"][:600], tool_side, k=4, verbose=False)
        answer = [e for e in r["answer"] if e["claim_id"] not in used_claim_ids]
        claim_ids = [e["claim_id"] for e in answer]
        used_claim_ids.update(claim_ids)
        per_contention.append({"n": c.get("n"), "strength": r["answer_strength"] if answer else "none",
                               "evidence_used": claim_ids})
        if answer:
            blocks.append(f"CONTENTION {c.get('n')} -- {label}:\n{c['text']}\n\n"
                          f"YOUR EVIDENCE AGAINST THIS CONTENTION:\n"
                          f"{dbt.evidence_block(answer)}")
        else:
            any_bare = True
            blocks.append(f"CONTENTION {c.get('n')} -- {label}:\n{c['text']}\n\n"
                          f"YOU HAVE NO EVIDENCE AGAINST THIS CONTENTION.")

    evidence_rule = WITH_EVIDENCE + (f"\n\n{NO_EVIDENCE_PER_CONTENTION}" if any_bare else "")
    speech_words = dbt.SPEECH_WORD_LIMITS["rebuttal"]
    word_limit = word_limit or speech_words
    if word_limit == speech_words:
        length_rule = (f"Generate a rebuttal of approximately {speech_words} words. "
                       f"Stay within the {speech_words}-word limit.")
    else:
        length_rule = (f"This attacking half gets approximately {word_limit} words of a "
                       f"{speech_words}-word rebuttal. Stay within the {word_limit}-word "
                       f"limit.")
    system = REBUTTAL_SPEECH_SYSTEM.format(tool_side=tool_side.upper(), resolution=resolution,
                                           evidence_rule=evidence_rule, length_rule=length_rule)
    user = "THEIR FULL CONSTRUCTIVE CASE:\n\n" + "\n\n---\n\n".join(blocks)

    if mock:
        text = (f"[mock rebuttal, evidence mixed] Going down their flow: each of "
                f"their {len(opponent_contentions)} contentions fails on link, "
                f"impact, or turns in our favor.")
    else:
        text = call(client, system, user, max_tokens=dbt.word_limit_tokens(word_limit))
        if any_bare:
            v = dbt.citation_violations(text)
            if v:
                text = call(client, system +
                           f"\n\nYour previous answer contained {', '.join(v)} "
                           f"somewhere in a contention marked NO EVIDENCE. Rewrite: "
                           f"keep the evidenced contentions as they were, but for "
                           f"any no-evidence contention use logic only -- no "
                           f"numbers, years, organisations or study references.",
                           user, max_tokens=dbt.word_limit_tokens(word_limit))

    return {"mode": "rebuttal_speech", "text": text, "per_contention": per_contention}
