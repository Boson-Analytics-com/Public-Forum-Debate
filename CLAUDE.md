# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A pipeline that scrapes web sources into a citation-verified evidence knowledge
base, then uses it to run and judge human-vs-tool Public Forum (PF) debates: a
human argues one side, the tool argues the other from the KB, and an LLM judge
scores the archived round.

- **Not a git repository.** There's no history to consult and no undo. Back up
  files before large edits (`_archive/backup_before_*/` hold earlier copies;
  `_archive/` isn't live code, so exclude it from greps).
- **Python 3.11 venv** at `venv/` (`source venv/bin/activate`). There's no
  `requirements.txt`. Packages in use: `requests beautifulsoup4 lxml trafilatura
  pymupdf` (scraping), `boto3 python-dotenv` (Bedrock), `google-genai`
  (`kb/claims_gemini.py`), `sentence-transformers` (optional, for `index.py`),
  `streamlit` (`app.py`), and `pyflakes` (lint).
- **Run everything from `Scripts/`.** Each module calls `load_dotenv()` at
  import, and `.env` (AWS credentials/region/`BEDROCK_MODEL_ID`, plus
  `GEMINI_API_KEY`/`GEMINI_MODEL` for `claims_gemini.py`) lives here. Never
  print or log `.env`.
- **Layout.** `kb/` = knowledge-base build (steps 1-6). `pf/` = the debate
  engine (`debate`, `opponent`, `round`, `judge`, `engine`). `app.py` = Streamlit
  UI. `config/resolution.json`, `inputs/` (`sources.docx`, `urls.csv`),
  `corpus/` (all generated data), `docs/` (explainer HTML), `tests/`,
  `_archive/` (backups and retired scripts). `pf/` modules import each other
  and `kb.index` as packages (`from pf import debate`, `from kb import index`),
  so **run them as modules** (`python -m pf.round`) from `Scripts/`; running
  `python pf/round.py` fails on imports. `kb/` scripts are standalone and can
  be run either way.

## Pipeline and commands

```
inputs/sources.docx → kb/extract_urls → kb/fetch → kb/parse → documents.jsonl   (steps 1-3)
documents.jsonl → kb/chunk → kb/claims_* → kb/index → kb.sqlite                 (steps 4-6)
kb.sqlite + config/resolution.json → pf/debate, pf/opponent, pf/round, pf/judge
    front ends: python -m pf.round debate-run (terminal) · streamlit run app.py (UI)
```

Everything reads and writes under `./corpus/`. `config/resolution.json` is
hand-written: it holds the resolution text, `side_tool_defends`, and the
`contention_tags` every claim is filed under. Changing the tags after step 5
means re-extracting the whole KB. Every stage is resumable: `fetch` skips
fetched URLs, `claims_*` skip chunks they already have claims for, and
`parse`/`chunk` are offline and safe to re-run.

```
python kb/extract_urls.py inputs/sources.docx -o inputs/urls.csv
python kb/fetch.py inputs/urls.csv --outdir ./corpus --delay 1.5 --workers 4   # --limit 20 for a test batch
python kb/parse.py --corpus ./corpus
python kb/chunk.py --corpus ./corpus                                  # "Offset mismatches" must be 0
python kb/claims_gemini.py --corpus ./corpus --config config/resolution.json [--mock | --limit 30]
python kb/claims_bedrock.py --corpus ./corpus --config config/resolution.json [--mock | --limit 30]
python kb/index.py build|stats --corpus ./corpus
python kb/index.py search --corpus ./corpus --query "..." [--stance pro] [--k 5] [--tags a,b] [--since YYYY-MM-DD] [--json]
python kb/index.py rebut --corpus ./corpus --query "<opponent argument>" --side pro
python _archive/aduit.py --corpus ./corpus [--min-claims 8]          # per-source stance balance (archived; filename typo is real)

python -m pf.debate case --side pro|con|both     # builds corpus/case_<side>.json from the KB
python -m pf.debate respond --side pro --query "..." [--format crossfire|rebuttal|summary|final]
python -m pf.debate gaps                         # summarizes corpus/gaps.jsonl

python -m pf.round debate-start --human-side pro [--mock]
python -m pf.round debate-run [--exchanges 4] [--mock]
python -m pf.round debate-status
python -m pf.engine debate --human-side pro [--exchanges 4] [--mock]   # start/resume, run, then judge
streamlit run app.py                                          # UI; no judging (use pf.judge score)

python -m pf.judge score --debate corpus/debates/debate_<ts>.json [--samples 3] [--mock]
python -m pf.judge latest|trend --corpus ./corpus
```

`pf.*` commands default to `--corpus ./corpus` and
`--config config/resolution.json`.

**Stage 5 backends.** `kb/claims_gemini.py` calls **Gemini** (`GEMINI_API_KEY` and `GEMINI_MODEL`
must both be set). `kb/claims_bedrock.py` produces the same `claims.jsonl` schema via Bedrock,
using whichever `BEDROCK_MODEL_ID` is set. Nothing here uses the Anthropic
API directly (no `ANTHROPIC_API_KEY`). After a run, check `corpus/claims_report.json`:
`rejection_rate_pct` under 10%, `stance_balance` roughly even,
`claims_per_chunk` about 1.5-3. `claims_*.py --mock` deliberately plants a fake
quote per chunk to prove rejection works, so a high rejection rate there is a
pass. Delete `corpus/claims.jsonl` before the real run.

## Anti-fabrication guardrails (don't weaken these)

These layers are the point of the project. When a change makes one of them
fire more often, fix the cause; don't loosen the guard.

1. **Claim quotes are verified** (`claims*.py` `locate_quote()`). Every claim's
   `quote` must appear character-for-character in its source chunk, otherwise
   the claim is rejected and the chunk retried once. Don't lower the match
   threshold to cut rejections.
2. **Retrieval tiers** (`index.py`). Hybrid FTS5/BM25 plus MiniLM embeddings,
   fused with RRF (keyword-only if `sentence-transformers` is missing). Each
   result gets a tier (`direct`/`related`/`none`) from absolute cosine and
   query-coverage signals (`TIER_COS_*`/`TIER_COV_*`, which need calibrating on
   ~20 real questions), not from RRF rank. `none` returns nothing on purpose.
3. **Citation guard** (`debate.py` `citation_violations()`). When an answer has
   no evidence behind it, the output is regex-scanned for years, percentages,
   money, study references, attributions and named organisations. It's
   regenerated once, then replaced with a canned no-source answer. Every
   `none` answer is logged to `corpus/gaps.jsonl`. `fill_gaps.py` is referenced
   in docstrings but doesn't exist. There's deliberately no live web search
   mid-round, since it would bypass quote verification.
4. **Mechanism-swap guard** (`opponent.py`). Checks that a tool attack engages
   the resolution's actual mechanism, not a lookalike policy. Same
   retry-once-then-flag shape; a surviving swap comes back as
   `mechanism_swap_flag`. The judge re-checks swaps independently.
5. **Human contentions stay verbatim.** For the human's constructive,
   `debate.py`'s `extract_contentions_from_speech()` asks the model for line
   ranges only, and the text is sliced from the human's own lines. Any cited
   `quote` must be a substring of the full paste, or it's dropped.
6. **Quotes are referenced, never copied.** `documents.jsonl` `normalized_text`
   is write-once. Chunk and claim offsets point into it; to change cleaning,
   bump `doc_version` and reprocess. Debate state stores `claim_id`s, not
   quote text.

## Debate architecture

**State.** `corpus/debate_state.json` is the single source of truth for a
debate. It holds:
- `human_side` / `tool_side`
- `stage_index`: completed stages of the fixed 11-step `DEBATE_ORDER`
  (constructive ×2, crossfire, rebuttal ×2, crossfire, summary ×2, grand
  crossfire, final focus ×2)
- `contentions` / `speeches` per side
- `attacks`: the argument graph, where every question, answer and rebuttal
  hit from either side is one edge
- `token_usage`
- `pending`: position inside the current stage (awaiting the constructive
  paste, crossfire exchange and asker, a question awaiting its answer). It's deleted when the stage ends.

Contention status is computed from the graph by `contention_status()`:
CONCEDED means never attacked; HELD means at least one hit is marked
`answered` (a tool answer counts only when evidence-backed, a human answer
when it isn't empty); CONTESTED is anything else.

**Turn-based engine (`round.py`).** There are three calls:
- `debate_next_action()` returns `{"kind": "human" | "tool" | "done", ...}`.
  Human actions also carry `input` (choice/yesno/text/speech) and prompt text.
- `debate_submit(value)` applies the human's answer.
- `debate_tool_turn()` runs the tool's turn.

State is saved after every turn, so a crash or a closed tab resumes at the
exact turn. Both front ends are thin loops over these calls and share the state
file, so a debate started in one can be continued in the other.
`debate_run()` (terminal) reads input through `debate_ask()`/`DEBATE_INPUT_FN`.
`app.py` (Streamlit) renders everything from the state file and runs tool turns
synchronously inside a spinner. `debate_finish()` archives to
`corpus/debates/debate_<started_at>.json` (the path is stable per debate).
Single user: one state file per corpus.

**Who does what.**
- `debate.py`: `build_case()` picks the side's strongest contention tags and
  diversifies evidence across documents so one card can't sink every
  contention. `respond()` branches its prompt on evidence strength
  (strong/thin/none). `validate_query()` raises `SystemExit` on junk input;
  inside a debate, `round.py`'s `_debate_tool_answers()` catches that so one
  bad line can't abort the round.
- `opponent.py` is library-only. `do_attack()` handles the tool's crossfire
  questions, where `pick_target()` spreads attacks by hit count first and uses
  the KB ratio only as a tiebreak. `CROSSFIRE_FOCUS` makes `crossfire_1` hit the
  constructive, `crossfire_2` the rebuttal, and `grand_crossfire` the summary.
  `do_rebuttal_speech()` attacks every opposing contention down the flow;
  `round.py` then appends its defense after a transition line.
- `round.py`: the human always writes their own constructive: one paste, no
  KB case offered or mixed in, repeated until at least one contention exists
  (an empty case crashes `pick_target()`). Only the tool's side gets a
  `case_<side>.json` (built by `debate_start` if missing), and a debate never
  overwrites it. `_migrate_constructive_pending()` moves a state saved at the
  old kb/own choice or backfill offer onto the paste flow. What was actually
  argued lives only in state and is passed to `opponent.py` as `contentions`.
  Human crossfire and rebuttal targets are auto-picked by
  `_pick_target_contention()`, which chooses the least-hit contention. Summary
  and final-focus prompts embed the round's speeches directly
  (`DEBATE_SUMMARY_SYSTEM` / `DEBATE_FINAL_SYSTEM`).
- **Word limits** (`debate.py` `SPEECH_WORD_LIMITS`: constructive 600,
  rebuttal 500, summary 400, final focus 400; crossfire has none). A human
  speech over its limit is rejected by `round.py`'s `speech_word_problem()`
  and the same prompt comes back (the UI shows a popup and keeps the text).
  A tool speech is asked to fit, then `_fit_tool_speech()` trims any overrun
  back to the last full sentence inside the limit. It only removes text, so
  quotes stay verbatim. `tests/test_word_limits.py` checks the prompts and backstops.
- `judge.py` reads only `corpus/debates/debate_*.json`; the old
  `corpus/rounds/round_*.json` archives are from a removed self-play harness
  and aren't readable. It computes per-side metrics from the graph *before*
  the LLM call, samples the judge `--samples` times, and takes the majority
  winner with averaged scores. Scores must cite `attack#N` /
  `pro-contention#N` refs, which are validated against the graph. A truncated
  or unparseable sample is dropped; the run fails only if every sample fails.
  Output: `judged_<debate>.json`.
- `engine.py debate` = `debate_start` (if no state) → `debate_run` → `judge.score`,
  with one shared client.

Terminal quirk: multi-line input ends with a literal `END` line. In the rebuttal,
summary and final-focus prompts, a terminal's bracketed-paste escapes can
swallow a pasted `END`; typing it fresh works. The UI's text boxes don't have
this problem.

## Model backend (AWS Bedrock)

`pf/round.py`, `pf/debate.py`, `pf/judge.py` and `kb/claims_bedrock.py` each wrap
Bedrock `converse()` in a `BedrockMessagesAdapter` exposing an
Anthropic-style `client.messages.create()`. Auth uses the boto3 credential
chain. Newer Claude models need a cross-region inference-profile ID in
`BEDROCK_MODEL_ID`.
- **One client per debate.** `round.py`'s `client_or_die()` builds it and
  threads it through `debate.py` and `opponent.py` (and the judge, in
  `engine.py`).
- **The adapter ignores the `model=` argument** and always sends its own
  `model_id`. So `OPPONENT_BEDROCK_MODEL_ID` currently has **no effect**: the
  opponent always runs on `BEDROCK_MODEL_ID`.
- **Errors inside `create()`** (throttling, context length, credentials) are
  re-raised as a readable `SystemExit`. Saved state limits the damage; resume
  the same command.
- **`MAX_TOKENS_SCALE`** in `pf/debate.py`, `pf/round.py`, `kb/claims_gemini.py`
  and `kb/claims_bedrock.py` multiplies every real call's `max_tokens`. It is
  currently `1.0` (production) in all four; set it to `0.5` to halve output
  cost while testing. `opponent.py`'s calls
  are covered by `round.py`'s scale. `judge.py` deliberately has no scale,
  because its JSON schema is large.
- **`TOKEN_TOTALS`** counts real Bedrock usage per module. `round.py`'s counter
  is the debate's total: it's re-seeded from `st["token_usage"]` before each
  turn and saved after it. The counters in `debate.py` and `judge.py` only
  cover their standalone CLIs.
- **`--mock`** (in `pf/debate.py`, `pf/round.py`, `pf/engine.py`, `pf/judge.py`,
  `kb/claims_*.py`, and `PF_MOCK=1` for the UI) skips model calls with synthetic output.

## Verifying changes (one small test, otherwise manual)

- **Lint:** `python -m pyflakes pf/*.py kb/*.py app.py`.
- **Unit test:** `tests/test_word_limits.py` is the only automated check (fake
  client, scratch KB copy). Run `python -m tests.test_word_limits` from
  `Scripts/` (pytest isn't installed in the venv, but also works if added).
- **Never test against the real `./corpus`.** It may hold a live
  `debate_state.json`, and `--mock` runs still write state and archives. Copy
  `corpus/kb.sqlite` and `corpus/case_*.json` into a scratch directory and pass
  `--corpus <scratch>`.
- **End-to-end debate, scripted.** Set `pf.round.DEBATE_INPUT_FN` to a function
  that yields canned answers, then call
  `pf.engine.run_debate(corpus, cfg, side, None, True)` or `pf.round.debate_run(...)`.
  Mock output is deterministic, so stdout and the archive (minus timestamps)
  can be diffed before and after a refactor. Cover an empty paste (it must
  re-ask), both human sides, and ending crossfire early.
- **Resume:** raise an exception from the input function mid-crossfire, then
  run again. The exchange count must continue, not restart.
- **UI:** `streamlit.testing.v1.AppTest.from_file("<abs path>/app.py")`. The app
  has no path inputs: set `PF_CORPUS=<scratch>` and `PF_MOCK=1` (and optionally
  `PF_CONFIG`, default `config/resolution.json`) in the environment before running it, then click through. There
  are no sidebar options; tool turns within a round always run on their own.
  Widget keys: `start`, `yes_*`/`no_*` (crossfire continue), `send_*`,
  `next_<round>` (ends a round page; `next_7` is Finish), `back_<round>` /
  `back_live` (read-only view of an earlier round, and back), `nav_<step>`
  (sidebar Rounds entries, same navigation), `show_record_btn`,
  `dl_debate`, `restart` then `restart_yes`/`restart_no`, `new_debate`.

## Directory notes

- `corpus/raw/`: original fetched bytes (`<sha1-of-url>.html.gz`), untouched.
- `corpus/debates/`: debate archives and `judged_*.json`. `corpus/rounds/`:
  legacy self-play archives.
- `_archive/files not used/`: retired one-off debug scripts. Not examples of
  current usage. `_archive/backup_before_*/`: earlier copies of `app.py`,
  `round.py` etc. (flat, pre-`pf/` layout).
- `.streamlit/config.toml`: disables Streamlit's file watcher (it chokes on
  `sentence-transformers` lazy imports), so **restart the app after editing
  `pf/*.py`**.
- `docs/flowsheet.html`, `docs/human_turn.html`, `docs/architecture_diagram.html`:
  hand-written explainer pages. Parts describe the removed self-play harness.
- `corpus/` also holds `case_pro.json`/`case_con.json`, `gaps.jsonl`,
  `debate_state.json` (live debate) and the per-stage `*_report.json` files.
