"""
Streamlit front end for the human-vs-tool Public Forum debate.

Drives round.py's turn-based engine (debate_next_action / debate_submit /
debate_tool_turn) -- the same engine `python round.py debate-run` uses, so
the terminal and this UI can hand the same debate_state.json back and forth.
Everything shown here is read from that state file (and, at the end, the
archived debate); nothing is kept only in the browser. Judging isn't offered
here: run `python judge.py score --debate <archive>` from the terminal.

Run from Scripts/ (where .env lives):
    streamlit run app.py

Single user: there is one debate_state.json per corpus directory, so two
browser tabs on the same corpus are the same debate.
"""

import contextlib
import io
import json
import os
import re
from pathlib import Path

import streamlit as st

# Streamlit Cloud: copy top-level secrets (Settings -> Secrets) into os.environ
# before the pf modules import, since they read credentials from os.environ.
# Locally there's no secrets.toml, so this is a no-op and .env is used instead.
try:
    for _k, _v in st.secrets.items():
        if isinstance(_v, str):
            os.environ.setdefault(_k, _v)
except Exception:
    pass

from pf import debate as dbt
from pf import round as rnd

st.set_page_config(page_title="PF Debate", layout="wide")
# Streamlit's dialog overlay is a flex box pinned to the top; centre the popup
# vertically ("safe" keeps a tall dialog scrollable instead of clipping its top).
# The sidebar (when open) takes 30% of the screen width; this overrides
# Streamlit's pixel width, so dragging its edge no longer resizes it.
st.markdown("<style>.stDialog { align-items: safe center !important; }"
            'section[data-testid="stSidebar"][aria-expanded="true"] '
            "{ width: 30vw !important; min-width: 30vw !important; "
            "max-width: 30vw !important; }"
            # Round anchors (render_transcript) land below the sticky page
            # header instead of hidden under it when a sidebar link jumps to them.
            ".round-anchor { scroll-margin-top: 14rem; }</style>",
            unsafe_allow_html=True)
ss = st.session_state

# --- login gate --------------------------------------------------------------------
# Single shared demo login; no per-user accounts. Not real auth (the credentials
# live here in plaintext) -- don't rely on it to protect sensitive data.
APP_USERNAME = "BosonDebate"
APP_PASSWORD = "Debate123"
ss.setdefault("authenticated", False)

if not ss.authenticated:
    st.title("Public Forum Debate")
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Log in")
    if submitted:
        if username != APP_USERNAME:
            st.error("Username is incorrect.")
        elif password != APP_PASSWORD:
            st.error("Password is incorrect.")
        else:
            ss.authenticated = True
            st.rerun()
    st.stop()

ss.setdefault("flash", [])
ss.setdefault("halted", None)
# Has this browser session picked a side (or chosen to resume)? Until it has,
# the side-choice popup is shown, even if an older debate is saved on disk.
ss.setdefault("setup_done", False)
# The PF round (1-7) shown on the main page; TOTAL_ROUNDS + 1 is the
# "Debate finished" page. Each round is its own page: when the engine moves on
# to the next round, the page stays on the finished one until "Next" is
# clicked (render_next_gate), and the tool never starts a new round on its own.
ss.setdefault("page_round", 1)

SIDE_COLOR = {"pro": "blue", "con": "orange"}
STAGE_TITLES = {"constructive": "Constructive", "crossfire_1": "Crossfire",
                "rebuttal": "Rebuttal", "crossfire_2": "Second Crossfire",
                "summary": "Summary", "grand_crossfire": "Grand Crossfire",
                "final_focus": "Final Focus"}
# Engine output (otherwise discarded) worth surfacing on the page as a notice.
NOTICE_PATTERNS = ("[warning]", "[empty", "needs at least one", "[over word limit]",
                   "[tool speech trimmed]")


# --- helpers ---------------------------------------------------------------------

def md(text: str) -> str:
    """Show speech text literally: escape markdown/LaTeX characters (a "$5
    billion" would otherwise render as math) and keep line breaks."""
    text = re.sub(r"([\\`*_\[\]#$<>~|])", r"\\\1", text or "")
    return text.replace("\n", "  \n")


def side_tag(side: str) -> str:
    return f"**:{SIDE_COLOR[side]}[{side.upper()}]**"


def who(dstate: dict, side: str) -> str:
    return "You" if side == dstate["human_side"] else "Tool"


def reset_session(setup_done: bool):
    """Forget this browser session's per-debate UI state (round gate, notices,
    typed-but-unsent text) so the next debate starts clean."""
    for k in [k for k in ss if str(k).startswith("ta_")]:
        del ss[k]
    ss.flash, ss.halted, ss.show_record = [], None, False
    ss.setup_done, ss.page_round = setup_done, 1


def run_engine(fn, *args):
    """Call an engine function, capturing what it prints. Returns (result,
    error). Bedrock failures arrive as SystemExit (see the adapters)."""
    buf = io.StringIO()
    result, err = None, None
    try:
        with contextlib.redirect_stdout(buf):
            result = fn(*args)
    except SystemExit as e:
        err = str(e)
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
    out = buf.getvalue()
    ss.flash += [ln.strip() for ln in out.splitlines()
                 if any(p in ln for p in NOTICE_PATTERNS)]
    return result, err


@st.cache_resource(show_spinner=False)
def _client(mock: bool):
    return rnd.client_or_die(mock)


def get_client(mock: bool):
    try:
        return _client(mock)
    except SystemExit as e:
        st.error(str(e))
        st.stop()


def step_title(n: int) -> str:
    _, stage, side, _ = rnd.DEBATE_ORDER[n - 1]
    return STAGE_TITLES[stage] + (f" ({side.upper()})" if side else "")


# The main page counts PF's 7 rounds, not the engine's 11 DEBATE_ORDER steps:
# a speech round is two steps (PRO then CON) under one round number.
ROUND_OF_STAGE = {"constructive": 1, "crossfire_1": 2, "rebuttal": 3, "crossfire_2": 4,
                  "summary": 5, "grand_crossfire": 6, "final_focus": 7}
TOTAL_ROUNDS = max(ROUND_OF_STAGE.values())


def round_no(n: int) -> int:
    """PF round (1-7) that DEBATE_ORDER step `n` belongs to."""
    return ROUND_OF_STAGE[rnd.DEBATE_ORDER[n - 1][1]]


# --- sidebar -----------------------------------------------------------------------

# Fixed paths (run from Scripts/). PF_CORPUS / PF_CONFIG override them, e.g. to
# point the app at a scratch copy of the KB for testing; PF_MOCK=1 runs with
# synthetic output instead of model calls. Tool turns within a round always run
# on their own; each new round still waits for a "Start round" click.
corpus = Path(os.environ.get("PF_CORPUS", "./corpus"))
config_path = Path(os.environ.get("PF_CONFIG", "config/resolution.json"))

env_mock = os.environ.get("PF_MOCK", "") == "1"
# In-app toggle, independent of PF_MOCK: unchecked (default) makes real model
# calls; checking it switches to synthetic output starting with the very next
# call, no LLM calls happen while it's on. Unchecking it switches back to real
# calls on the next turn.
st.sidebar.checkbox("Mock mode (no model calls)", key="mock_mode",
                     help="Skip real LLM calls and use synthetic output instead.")
mock = env_mock or ss.mock_mode
if mock:
    st.sidebar.caption("Mock mode: no model calls, synthetic output.")

# Reserved first so the Rounds list sits at the top of the sidebar; it is
# filled in once the current round is known (render_rounds_bar).
rounds_slot = st.sidebar.container()


# Sidebar "Rounds" navigation. CHANGED: this used to be one dropdown listing
# the engine's 11 steps. It is now a "Rounds" heading over PF's 7 rounds
# (Constructive, Crossfire, Rebuttal, Second Crossfire, Summary, Grand
# Crossfire, Final Focus). Each speech round is a dropdown holding its PRO and
# CON speeches; a crossfire, being one shared step, is a single link with no
# dropdown. Every link jumps to that step's section on the main page (the
# `step-<n>` anchors render_transcript() puts on each heading). Steps not
# reached yet are greyed out and not links, since the page has nothing there.
# The round in progress is highlighted and its dropdown starts open.

def round_steps() -> dict:
    """{PF round number: [DEBATE_ORDER step numbers in it]}."""
    steps = {}
    for n, *_ in rnd.DEBATE_ORDER:
        steps.setdefault(round_no(n), []).append(n)
    return steps


def render_rounds_bar(current: int | None):
    """`current` is the step in progress, 0 before the debate starts, None once
    it's finished."""
    def status(n: int) -> str:
        if current is None or n < current:
            return "done"
        return "current" if n == current else "upcoming"

    with rounds_slot:
        st.markdown("### Rounds")
        for r, steps in round_steps().items():
            _, stage, _, kind = rnd.DEBATE_ORDER[steps[0] - 1]
            title = f"Round {r} · {STAGE_TITLES[stage]}"
            states = [status(n) for n in steps]
            if "current" in states:
                label = f":blue-background[**▶ {title}**]"
            elif all(x == "done" for x in states):
                label = f"✓ {title}"
            else:
                label = f":gray[{title}]"

            if kind == "crossfire":
                with st.container(border=True):
                    # Links stay outside the colour markup, which doesn't nest
                    # them reliably; the highlight goes on a tag beside it.
                    link = f"[{title}](#step-{steps[0]})"
                    if states[0] == "upcoming":
                        st.markdown(label)
                    elif states[0] == "current":
                        st.markdown(f"▶ **{link}** :blue-background[in progress]")
                    else:
                        st.markdown(f"✓ {link}")
                continue

            with st.expander(label, expanded="current" in states):
                for n, state in zip(steps, states):
                    side = rnd.DEBATE_ORDER[n - 1][2]
                    if state == "upcoming":
                        st.markdown(f":gray[{side.upper()} · not yet]")
                    elif state == "current":
                        st.markdown(f"▶ **[{side.upper()}](#step-{n})** "
                                    f":blue-background[in progress]")
                    else:
                        st.markdown(f"✓ [{side.upper()}](#step-{n})")


if not (corpus / "kb.sqlite").exists():
    st.error(f"No knowledge base at {corpus / 'kb.sqlite'}. Build it first (index.py build).")
    st.stop()
if not config_path.exists():
    st.error(f"No resolution config at {config_path}.")
    st.stop()
cfg = json.loads(config_path.read_text(encoding="utf-8"))

# Display label only: resolution.json's text (fed to the model prompts) keeps
# its "Resolved:" wording.
resolution_label = re.sub(r"^\s*Resolved\s*:\s*", "Resolution: ", cfg["resolution"])

# Title, resolution and (right column, filled in by render_status_panel()
# once a debate is loaded) who argues which side + progress. Pinned just under
# Streamlit's 3.75rem toolbar so it stays visible while the transcript
# scrolls; it needs an opaque background of the theme's colour to cover the
# text scrolling under it.
header_bg = "#0e1117" if st.context.theme.type == "dark" else "#ffffff"
# Sticky only works on a direct child of the tall main column: Streamlit can
# wrap a keyed container in a wrapper exactly as tall as itself, which leaves
# it no room to stick. So the rule goes on whichever element sits directly in
# a vertical block and is, or contains, the header.
st.markdown(f"""<style>
[data-testid="stVerticalBlock"] > .st-key-page_header,
[data-testid="stVerticalBlock"] > div:has(.st-key-page_header) {{
    position: sticky; top: 3.75rem; z-index: 99;
    background: {header_bg};
}}
.st-key-page_header {{
    background: {header_bg};
    border-bottom: 1px solid rgba(128, 128, 128, 0.25);
    padding-bottom: 0.5rem;
}}
.st-key-page_header h1 {{ padding-top: 0.25rem; padding-bottom: 0.25rem; }}
</style>""", unsafe_allow_html=True)
page_header = st.container(key="page_header")
with page_header:
    title_col, status_col = st.columns([3, 1], vertical_alignment="center")
    with title_col:
        st.title("Public Forum Debate")
    st.caption(resolution_label)


# --- setup ---------------------------------------------------------------------------

@st.dialog("Choose your side", dismissible=False)
def setup_dialog(unfinished: dict | None):
    if unfinished:
        n_done = unfinished.get("stage_index", 0)
        st.info(f"An unfinished debate is saved: you argue "
                f"{unfinished['human_side'].upper()}, now at round {round_no(n_done + 1)} of "
                f"{TOTAL_ROUNDS} ({step_title(n_done + 1)}).")
        if st.button("Resume that debate", key="resume", use_container_width=True):
            ss.setup_done = True
            ss.page_round = round_no(n_done + 1)   # the round it stopped in carries on
            st.rerun()
        st.divider()
        st.caption("Or start a new one. This discards the unfinished debate.")
    st.markdown(resolution_label)
    human_side = st.radio("Which side will you argue?", ["pro", "con"], key="setup_side",
                          format_func=lambda s: f"{s.upper()}  (the tool argues "
                                                f"{rnd.other_side(s).upper()})")
    if st.button("Start debate", type="primary", key="start", use_container_width=True):
        client = get_client(mock)
        with st.spinner("Starting: building the tool's knowledge-base case if it has none..."):
            _, err = run_engine(rnd.debate_start, corpus, cfg, human_side, client, mock,
                                3, rnd.UNLIMITED_EXCHANGES)
        if err:
            st.error(err)
            return
        reset_session(setup_done=True)
        st.rerun()


if not ss.setup_done:
    saved = rnd.load_debate(corpus) if rnd.debate_state_path(corpus).exists() else None
    unfinished = saved if saved and saved.get("stage_index", 0) < len(rnd.DEBATE_ORDER) else None
    st.info("Pick your side to begin. You argue it in every round; the tool argues the other.")
    render_rounds_bar(0)
    setup_dialog(unfinished)
    st.stop()


# --- the debate ------------------------------------------------------------------------

client = get_client(mock)
action, err = run_engine(rnd.debate_next_action, corpus, cfg, client, mock)
if err:
    st.error(err)
    st.stop()
dstate = rnd.load_debate(corpus)
pending = dstate.get("pending")


def render_status_panel():
    """Top right of the page: which side each party argues, and progress."""
    with status_col:
        st.markdown(f"You: {side_tag(dstate['human_side'])} · "
                    f"Tool: {side_tag(dstate['tool_side'])}")
        done = dstate.get("stage_index", 0)
        total = len(rnd.DEBATE_ORDER)
        if action["kind"] == "done":
            st.progress(1.0, text="Finished")
        else:
            n = action["n"]
            st.progress(done / total, text=f"Round {round_no(n)} of {TOTAL_ROUNDS}: "
                                           f"{step_title(n)}")


def render_sidebar_status():
    with st.sidebar:
        st.divider()
        for side in ("pro", "con"):
            cs = dstate["contentions"][side]
            st.markdown(f"{side_tag(side)} contentions ({len(cs)})")
            if not cs:
                st.caption("not given yet")
            for c in cs:
                st.markdown(f"{c['n']}. {md(c.get('title') or c['area'])} · "
                            f"`{rnd.contention_status(dstate, side, c['n'])}`")
        # Mirrors round.py's contention_status().
        st.caption("**What the tags mean**  \n"
                   "`HELD`: attacked, and at least one attack was answered (a tool answer "
                   "counts only when it's backed by evidence).  \n"
                   "`CONTESTED`: attacked, and no attack on it has been answered yet.  \n"
                   "`CONCEDED`: never attacked, so the other side has let it stand.")
        u = dstate.get("token_usage") or {}
        st.caption(f"Tokens: {u.get('input_tokens', 0)} in + {u.get('output_tokens', 0)} out "
                   f"across {u.get('calls', 0)} call(s)")
        st.divider()
        c1, c2 = st.columns(2)
        if c1.button("Restart", key="restart", use_container_width=True,
                     help="Discard this debate and start it again from round 1, same side."):
            ss.confirm_restart = True
        # Re-opened on every rerun until Yes/Cancel, so the confirm click
        # inside it is handled even on a full rerun.
        if ss.get("confirm_restart"):
            restart_dialog()
        if c2.button("New debate", key="new_debate", type="primary", use_container_width=True,
                     help="Pick a side and start a different debate."):
            reset_session(setup_done=False)
            st.rerun()


@st.dialog("Restart this debate?", dismissible=False)
def restart_dialog():
    side = dstate["human_side"]
    if action["kind"] == "done":
        st.markdown("This debate is finished and already archived in `corpus/debates/`.")
    else:
        st.warning("Everything said so far in this debate is discarded. It can't be undone.")
    st.markdown(f"A fresh debate starts at round 1. You argue {side_tag(side)} again.")
    c1, c2 = st.columns(2)
    if c1.button("Yes, restart", type="primary", key="restart_yes", use_container_width=True):
        with st.spinner("Restarting..."):
            _, err = run_engine(rnd.debate_start, corpus, cfg, side, client, mock,
                                3, rnd.UNLIMITED_EXCHANGES)
        if err:
            st.error(err)
            return
        ss.confirm_restart = False
        reset_session(setup_done=True)
        st.rerun()
    if c2.button("Cancel", key="restart_no", use_container_width=True):
        ss.confirm_restart = False
        st.rerun()


def render_constructive(side: str):
    sp = dstate["speeches"][side].get("constructive")
    if not sp:
        return
    with st.chat_message("user" if side == dstate["human_side"] else "assistant"):
        st.markdown(f"{side_tag(side)} · {who(dstate, side)} · "
                    f"{word_count_label(sp.get('text') or '', 'constructive')}")
        for c in dstate["contentions"][side]:
            st.markdown(f"**Contention {c['n']}: {md(c.get('title') or c['area'])}**")
            st.markdown(md(c["text"]))
            cards = [e for e in c.get("evidence") or [] if e.get("quote")]
            if cards:
                with st.expander(f"Evidence ({len(cards)})"):
                    for e in cards:
                        st.markdown(f"> {md(e['quote'])}  \n— {md(e.get('cite') or 'no cite')}")


def word_count_label(text: str, stage: str) -> str:
    return f"_{dbt.count_words(text)} / {dbt.SPEECH_WORD_LIMITS[stage]} words_"


def render_speech(side: str, stage: str):
    sp = dstate["speeches"][side].get(stage)
    if not sp:
        return
    with st.chat_message("user" if side == dstate["human_side"] else "assistant"):
        st.markdown(f"{side_tag(side)} · {who(dstate, side)} · "
                    f"{word_count_label(sp.get('text') or '', stage)}")
        st.markdown(md(sp.get("text") or "_(empty)_"))


def render_crossfire(stage: str):
    for a in [a for a in dstate["attacks"] if a["round"] == stage]:
        asker, answerer = a["by_side"], a["target_side"]
        with st.chat_message("user" if asker == dstate["human_side"] else "assistant"):
            st.markdown(f"{side_tag(asker)} · {who(dstate, asker)} asks")
            st.markdown(md(a["text"]))
        with st.chat_message("user" if answerer == dstate["human_side"] else "assistant"):
            note = ""
            if answerer == dstate["tool_side"]:
                note = (" · _evidence-backed_" if a["answered"]
                        else " · _no evidence -- reasoning only_")
            st.markdown(f"{side_tag(answerer)} · {who(dstate, answerer)} answers{note}")
            st.markdown(md(a.get("response") or "_(no answer)_"))
    # A question asked but not answered yet (saved in pending).
    if pending and rnd.DEBATE_ORDER[pending["n"] - 1][1] == stage and pending.get("q"):
        asker = pending["asker"]
        with st.chat_message("user" if asker == dstate["human_side"] else "assistant"):
            st.markdown(f"{side_tag(asker)} · {who(dstate, asker)} asks")
            st.markdown(md(pending["q"]))


def render_transcript(page: int):
    """The steps of round `page` played so far (one round per page)."""
    current = action["n"] if action["kind"] != "done" else len(rnd.DEBATE_ORDER)
    for n, stage, side, kind in rnd.DEBATE_ORDER:
        if n > current:
            break
        if round_no(n) != page:
            continue
        # The empty span is the sidebar Rounds links' jump target.
        st.markdown(f'#### <span id="step-{n}" class="round-anchor"></span>'
                    f"Round {round_no(n)} · {STAGE_TITLES[stage]}"
                    + (f" · {side.upper()}" if side else ""), unsafe_allow_html=True)
        if kind == "crossfire":
            render_crossfire(stage)
        elif stage == "constructive":
            render_constructive(side)
        else:
            render_speech(side, stage)


def submit(value: str, spinner: str = "Saving..."):
    with st.spinner(spinner):
        _, err = run_engine(rnd.debate_submit, corpus, cfg, value, client, mock)
    if err:
        ss.halted = err
    st.rerun()


@st.dialog("Over the word limit")
def over_limit_dialog(problem: str):
    st.error(problem)
    st.caption("Your text is still in the box. Shorten it there, then submit again. "
               "The debate won't move on until the speech fits the limit.")
    if st.button("OK, I'll shorten it", type="primary", key="over_limit_ok",
                 use_container_width=True):
        st.rerun()


def text_form(aid: str, label: str, required: bool, button: str, help_text: str = "",
              height: int = 220, spinner: str = "Saving...", stage: str | None = None):
    """`stage`, for a speech, enforces its word limit before submitting."""
    limit = dbt.SPEECH_WORD_LIMITS.get(stage) if stage else None
    if limit:
        st.caption(f"Word limit: **{limit} words**. A longer speech is not accepted.")
    with st.form(f"form_{aid}", clear_on_submit=False):
        text = st.text_area(label, key=f"ta_{aid}", height=height, help=help_text or None)
        sent = st.form_submit_button(button, type="primary", key=f"send_{aid}")
    if sent:
        problem = rnd.speech_word_problem(stage, text) if limit else None
        if limit:
            st.caption(f"{dbt.count_words(text)} / {limit} words")
        if required and not text.strip():
            st.warning("Type something first.")
        elif problem:
            over_limit_dialog(problem)
        else:
            submit(text, spinner)


def render_human_action():
    a = action
    aid = f"{a['n']}_{a['phase']}_{a.get('exchange', 0)}_{len(dstate['attacks'])}"
    side = a["side"]
    st.markdown(f"**Round {round_no(a['n'])} of {TOTAL_ROUNDS}, {step_title(a['n'])}. "
                f"Your turn** ({side.upper()}): {a['ui_label']}")

    if a["phase"] == "paste":
        text_form(aid, a["ui_label"], False, "Submit constructive",
                  "Paste your whole constructive. It is split into contentions by line "
                  "range, so your wording is kept exactly; any quote you cite is checked "
                  "against what you pasted.", height=360,
                  spinner="Splitting your speech into contentions...", stage="constructive")

    elif a["phase"] == "continue":
        st.caption(f"{a['exchange']} exchange(s) done. Keep going or end the crossfire.")
        c1, c2 = st.columns(2)
        if c1.button("Next exchange", type="primary", key=f"yes_{aid}"):
            submit("y")
        if c2.button("End crossfire", key=f"no_{aid}"):
            submit("n")

    elif a["input"] == "text":   # crossfire question / answer
        st.caption(f"Exchange {a['exchange']} · "
                   f"focus: press on {a['hint']}")
        if a["phase"] == "ask":
            text_form(aid, a["ui_label"], True, "Ask", height=120)
        else:
            text_form(aid, a["ui_label"], False, "Answer", "Leaving it empty counts as no answer.",
                      height=120)

    else:   # rebuttal / summary / final focus
        text_form(aid, a["ui_label"], True, "Give speech", height=320, stage=a["stage"])


def render_next_gate(page: int):
    """Round `page` is over: say so and wait for "Next" to open the next
    round's page (or, after the last round, the finished page)."""
    stage = rnd.DEBATE_ORDER[round_steps()[page][0] - 1][1]
    st.success(f"Round {page} of {TOTAL_ROUNDS} complete: {STAGE_TITLES[stage]}.")
    if page < TOTAL_ROUNDS:
        nxt = rnd.DEBATE_ORDER[round_steps()[page + 1][0] - 1][1]
        label = f"Next: Round {page + 1} · {STAGE_TITLES[nxt]} →"
    else:
        label = "Finish →"
    if st.button(label, type="primary", key=f"next_{page}"):
        ss.page_round = page + 1
        ss.scroll_top = True
        st.rerun()


def scroll_to_top():
    """A new round's page starts at the top: Streamlit keeps the scroll
    position across reruns, so reset the main scroll container once."""
    if not ss.pop("scroll_top", False):
        return
    # A fixed script of ours, never user or model text (st.iframe runs it
    # with same-origin access to the app).
    st.iframe("""<script>
      const d = window.parent.document;
      for (const sel of ['[data-testid="stMain"]', '[data-testid="stAppViewContainer"]']) {
        const el = d.querySelector(sel); if (el) el.scrollTo(0, 0);
      }
      window.parent.scrollTo(0, 0);
    </script>""", height=1)


def render_tool_action():
    a = action
    if ss.halted:
        st.error(ss.halted)
        if st.button("Retry", key="retry"):
            ss.halted = None
            st.rerun()
        return
    with st.spinner(f"{a['what']}..."):
        _, err = run_engine(rnd.debate_tool_turn, corpus, cfg, client, mock)
    if err:
        ss.halted = err
    st.rerun()


def render_done():
    st.divider()
    st.subheader("Debate finished")
    path = rnd.debate_archive_path(corpus, dstate)
    if not path.exists():
        _, err = run_engine(rnd.debate_finish, corpus)
        if err:
            st.error(err)
            return
    # The archive's location on disk isn't shown; the record is viewed here
    # or downloaded. (Judging isn't offered in the UI: run judge.py on the
    # archive from the terminal.)
    c1, c2 = st.columns(2)
    showing = ss.get("show_record", False)
    if c1.button("Hide debate record" if showing else "Show debate record",
                 key="show_record_btn", use_container_width=True):
        ss.show_record = not showing
        st.rerun()
    c2.download_button("Download debate JSON", path.read_bytes(), file_name=path.name,
                       mime="application/json", key="dl_debate", type="primary",
                       use_container_width=True)
    if showing:
        st.json(json.loads(path.read_text(encoding="utf-8")), expanded=1)


render_rounds_bar(None if action["kind"] == "done" else action["n"])
render_status_panel()
render_sidebar_status()
scroll_to_top()

# Which page to show: the round in progress, or the one just finished if
# "Next" hasn't been clicked yet.
engine_round = TOTAL_ROUNDS + 1 if action["kind"] == "done" else round_no(action["n"])
page = min(ss.page_round, engine_round)
ss.page_round = page
if page > TOTAL_ROUNDS:
    render_done()
    st.stop()

render_transcript(page)
for note in ss.flash:
    st.warning(note)
ss.flash = []

st.divider()
if engine_round > page:
    render_next_gate(page)
elif action["kind"] == "tool":
    render_tool_action()
else:
    if ss.halted:
        st.error(ss.halted)
        ss.halted = None
    render_human_action()
