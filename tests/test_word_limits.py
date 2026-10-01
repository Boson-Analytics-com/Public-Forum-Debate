"""Checks the per-stage speech word limits reach every speech prompt and
max_tokens backstop, identically for pro and con, and that crossfire gets none.

Runs the real (non-mock) generation paths against a fake client that records
each call, in a scratch copy of the KB -- never the live ./corpus.

    python -m tests.test_word_limits     (run from Scripts/)
"""
import json
import re
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

from pf import debate as dbt
from pf import opponent as opp
from pf import round as rnd

ROOT = Path(__file__).resolve().parent.parent   # Scripts/
EXPECTED = {"constructive": 600, "rebuttal": 500, "summary": 400, "final_focus": 400}
CFG = json.loads((ROOT / "config" / "resolution.json").read_text(encoding="utf-8"))


class FakeClient:
    def __init__(self):
        self.calls = []
        self.messages = self

    def create(self, *, model=None, max_tokens, temperature=0.3, system=None, messages=None):
        self.calls.append({"system": " ".join((system or "").split()), "max_tokens": max_tokens})
        text = "A plain spoken answer that names nothing and cites nothing at all here."
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)])


def scratch_corpus() -> Path:
    d = Path(tempfile.mkdtemp(prefix="wordlimits_"))
    for f in ("kb.sqlite", "case_pro.json", "case_con.json"):
        shutil.copy(ROOT / "corpus" / f, d / f)
    return d


def neutral(calls, side):
    """Calls with the side names blanked, so pro and con can be compared."""
    other = rnd.other_side(side)
    return [(re.sub(rf"\b({side}|{other})\b", "SIDE", c["system"], flags=re.I), c["max_tokens"])
            for c in calls]


def test_config():
    assert dbt.SPEECH_WORD_LIMITS == EXPECTED
    assert not any("crossfire" in k for k in dbt.SPEECH_WORD_LIMITS)
    speech_stages = {s for _, s, _, kind in rnd.DEBATE_ORDER if kind == "speech"}
    assert speech_stages == set(EXPECTED)
    # word_limit_tokens() undoes the adapters' scale; that only holds if the
    # debate client (round.py's) and debate.py use the same scale.
    assert rnd.MAX_TOKENS_SCALE == dbt.MAX_TOKENS_SCALE
    for words in EXPECTED.values():
        effective = dbt.word_limit_tokens(words) * rnd.MAX_TOKENS_SCALE
        assert words * 1.4 <= effective <= words * 2, (words, effective)


def test_constructive():
    corpus, per_side = scratch_corpus(), {}
    for side in ("pro", "con"):
        c = FakeClient()
        dbt.build_case(corpus, CFG, side, c, False, 3)
        n = len(json.loads((corpus / f"case_{side}.json").read_text())["contentions"])
        assert c.calls and len(c.calls) == n
        for call in c.calls:
            assert "constructive speech of approximately 600 words" in call["system"]
            assert "Stay within the 600-word limit" in call["system"]
            assert "About 200 words. Stay within the 200-word limit." in call["system"]
            assert call["max_tokens"] == dbt.word_limit_tokens(200)
        per_side[side] = neutral(c.calls, side)
    assert len(per_side["pro"]) == len(per_side["con"])
    assert {m for _, m in per_side["pro"]} == {m for _, m in per_side["con"]}


def _debate_state(corpus, human_side, hits_on_tool):
    tool_side = rnd.other_side(human_side)
    st = {"resolution": CFG["resolution"], "human_side": human_side, "tool_side": tool_side,
          "contentions": {s: rnd._kb_contentions(corpus / f"case_{s}.json")
                          for s in ("pro", "con")},
          "speeches": {s: {k: {"text": f"{s} {k} text."}
                           for k in ("constructive", "rebuttal", "summary")}
                       for s in ("pro", "con")},
          "attacks": []}
    for i in range(hits_on_tool):
        st["attacks"].append({"id": i + 1, "round": "rebuttal", "by_side": human_side,
                              "target_side": tool_side, "target_contention_n": 1,
                              "text": "Their encryption mandate fails because criminals switch apps.",
                              "response": None, "answered": None})
    return st, tool_side


def test_rebuttal():
    corpus = scratch_corpus()
    for hits in (0, 2):
        offense, each = rnd._rebuttal_word_split(hits)
        assert offense + (each or 0) * hits <= EXPECTED["rebuttal"]
        per_side = {}
        for human_side in ("pro", "con"):
            st, tool_side = _debate_state(corpus, human_side, hits)
            c = FakeClient()
            rnd._tool_rebuttal(corpus, CFG, st, tool_side, c, False)
            first = c.calls[0]
            assert first["max_tokens"] == dbt.word_limit_tokens(offense)
            if hits:
                assert f"approximately {offense} words of a 500-word rebuttal" in first["system"]
            else:
                assert ("Generate a rebuttal of approximately 500 words. "
                        "Stay within the 500-word limit.") in first["system"]
            assert "600-750" not in first["system"]
            defenses = [x for x in c.calls[1:] if "Group the argument" in x["system"]]
            assert len(defenses) == hits
            for d in defenses:
                assert f"About {each} words. Stay within the {each}-word limit." in d["system"]
                assert d["max_tokens"] == dbt.word_limit_tokens(each)
            per_side[human_side] = neutral([first] + defenses, tool_side)
        assert [m for _, m in per_side["pro"]] == [m for _, m in per_side["con"]]


def test_summary_and_final():
    corpus = scratch_corpus()
    for stage, noun in (("summary", "summary"), ("final_focus", "final focus")):
        per_side = {}
        for side in ("pro", "con"):
            st, _ = _debate_state(corpus, rnd.other_side(side), 0)
            c = FakeClient()
            rnd._tool_summary_or_final(st, side, stage, c, False)
            (call,) = c.calls
            assert (f"Generate a {noun} of approximately 400 words. "
                    f"Stay within the 400-word limit.") in call["system"]
            assert "minute" not in call["system"]
            assert call["max_tokens"] == dbt.word_limit_tokens(400)
            per_side[side] = neutral(c.calls, side)[0][1]
        assert per_side["pro"] == per_side["con"]


def test_crossfire_untouched():
    corpus = scratch_corpus()
    for side in ("pro", "con"):
        c = FakeClient()
        dbt.respond(corpus, CFG, "Criminals will just switch to foreign encryption apps.",
                    side, "crossfire", c, False, verbose=False)
        assert dbt.FORMAT_RULES["crossfire"] in c.calls[0]["system"]
        assert c.calls[0]["max_tokens"] in (900, 300)   # pre-existing, unscaled-by-words
        c = FakeClient()
        contentions = rnd._kb_contentions(corpus / f"case_{rnd.other_side(side)}.json")
        opp.do_attack(corpus, CFG, side, contentions, c, False)
        for call in c.calls:
            assert "word limit" not in call["system"].lower()
            assert call["max_tokens"] in (80, 800)       # pre-existing values
    assert "crossfire" not in dbt.SPEECH_WORD_LIMITS


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
