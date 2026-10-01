# Public Forum debate tool

Scrapes web sources into a citation-verified evidence knowledge base (KB), then
uses it to run Public Forum (PF) debates: a human argues one side, the tool
argues the other from the KB, and an LLM judge scores the archived round.

The current topic is in `config/resolution.json`: *"The United States federal
government should require technology companies to provide lawful access to
encrypted communications."* The tool defends **pro**, so the human argues
**con**.

```
inputs/sources.docx → kb/extract_urls → kb/fetch → kb/parse → documents.jsonl
documents.jsonl → kb/chunk → kb/claims_* → kb/index → kb.sqlite
kb.sqlite + config/resolution.json → pf/ (debate, opponent, round, judge)
    front ends: python -m pf.round debate-run   (terminal)
                streamlit run app.py            (browser UI)
```

## Layout

```
Scripts/
  app.py                  Streamlit UI
  config/resolution.json  resolution text, side the tool defends, contention tags
  inputs/                 sources.docx, urls.csv
  kb/                     build the knowledge base (steps 1-6)
  pf/                     the debate engine and judge
  tests/                  test_word_limits.py
  docs/                   explainer pages (flowsheet, architecture)
  corpus/                 all generated data (see "Outputs")
  _archive/               backups and retired scripts, not live code
  .env                    credentials (never commit or print)
```

## Setup

Python 3.11 venv at `venv/`:

```bash
source venv/bin/activate
pip install requests beautifulsoup4 lxml trafilatura pymupdf     # scraping
pip install boto3 python-dotenv                                  # AWS Bedrock
pip install google-genai                                         # claims_gemini.py only
pip install sentence-transformers                                # optional: semantic search
pip install streamlit                                            # UI
```

Without `sentence-transformers`, search falls back to keyword-only. It works,
but is worse on rephrased questions.

`.env` (in `Scripts/`) holds:

| Variable | Used by |
|---|---|
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / `AWS_REGION` (or a profile/IAM role) | all Bedrock calls |
| `BEDROCK_MODEL_ID` | debate, judge, `claims_bedrock.py` (newer Claude models need a cross-region inference-profile ID) |
| `GEMINI_API_KEY`, `GEMINI_MODEL` | `claims_gemini.py` |

Nothing uses the Anthropic API directly. Run every command from `Scripts/`.
`pf` modules must be run as modules (`python -m pf.round`); `kb` scripts can be
run as `python kb/<script>.py`.

---

## Part 1 - Scrape sources (steps 1-3, free)

```bash
python kb/extract_urls.py inputs/sources.docx -o inputs/urls.csv
python kb/fetch.py inputs/urls.csv --outdir ./corpus --delay 1.5 --workers 4
python kb/parse.py --corpus ./corpus
```

1. **extract_urls** finds links in a `.docx`, `.pdf`, `.md`, `.txt` or `.html`
   file. Rows with `review=yes` sort to the top; delete any that aren't
   sources. Filters: `--drop-social`, `--drop-assets`, `--drop-homepages`,
   `--only-hosts a.com,b.com`.
2. **fetch** downloads pages. Test with `--limit 20` first. It is resumable and
   never re-downloads. If you see many 403/429 errors, use `--delay 4
   --workers 2`.
3. **parse** cleans text and extracts author, date and headings. Offline and
   safe to re-run.

Check `corpus/parse_report.json` before going on: `parsed` count,
`missing_date`, `bs4_fallback_count`. Many `too_short` drops is normal for
paywalled sites.

## Part 2 - Build the knowledge base (steps 4-6)

### First: the resolution config

`config/resolution.json` is the one file you write by hand:

```json
{
  "resolution": "Resolved: ...",
  "side_tool_defends": "pro",
  "contention_tags": ["law_enforcement_access", "privacy_civil_liberties", "..."]
}
```

Every claim is filed under these tags. Pick 4-8 areas the debate will actually
live in. Changing them after step 5 means re-extracting everything.

### Step 4 - chunk

```bash
python kb/chunk.py --corpus ./corpus
```

`Offset mismatches` must be `0`. Anything else means quotes can't be verified
later, so stop and investigate.

### Step 5 - extract claims (costs money, **this is the KB**)

Two interchangeable backends, same `claims.jsonl` output:

```bash
python kb/claims_gemini.py  --corpus ./corpus --config config/resolution.json
python kb/claims_bedrock.py --corpus ./corpus --config config/resolution.json
```

Test cheaply first:

```bash
... --mock          # no API calls
... --limit 30      # small paid batch; read ten claims in claims.jsonl
```

`--mock` plants one fake quote per chunk to prove the verifier catches it, so
a high rejection rate and a warning there mean the test **passed**. Delete
`corpus/claims.jsonl` before the real run. The real run is resumable.

Every claim's `quote` must appear character-for-character in its source chunk,
or the claim is rejected and the chunk retried once. Don't lower that
threshold to reduce rejections. It's what stops the tool inventing evidence.

Check `corpus/claims_report.json`:

| Field | Healthy |
|---|---|
| `rejection_rate_pct` | under 10% |
| `stance_balance` | pro and con roughly comparable (a lopsided KB is a sourcing problem and makes the tool collapse under attack) |
| `claims_per_chunk` | about 1.5-3 |
| `api_errors` | 0 or near it |

### Step 6 - index

```bash
python kb/index.py build --corpus ./corpus
python kb/index.py stats --corpus ./corpus
python kb/index.py search --corpus ./corpus --query "..." [--stance pro] [--k 5] [--tags a,b] [--since YYYY-MM-DD] [--json]
python kb/index.py rebut  --corpus ./corpus --query "<opponent argument>" --side pro
```

Search is hybrid FTS5/BM25 plus MiniLM embeddings, fused with RRF. Each result
gets a **tier**:

| Tier | Meaning | What the tool does |
|---|---|---|
| `direct` | strong match | argue it with the quote and citation |
| `related` | same area, not exact | use the evidence, extend the logic |
| `none` | nothing useful | reason from its own warrant and cite **nothing** |

`none` returns an empty list on purpose. Tiers come from absolute cosine and
query-coverage signals (`TIER_COS_*`, `TIER_COV_*` at the top of `kb/index.py`),
which should be calibrated on ~20 real questions.

---

## Part 3 - Run a debate

You pick a side, the tool takes the other, and both go through the fixed
11-step PF order: constructive x2, crossfire, rebuttal x2, crossfire, summary
x2, grand crossfire, final focus x2.

**Terminal:**

```bash
python -m pf.round debate-start --human-side con      # add --mock to test without the model
python -m pf.round debate-run --exchanges 4
python -m pf.round debate-status
python -m pf.engine debate --human-side con           # start/resume, run, then judge in one go
```

Multi-line input ends with a literal `END` line.

**Browser UI:**

```bash
streamlit run app.py
```

Restart it after editing `pf/*.py`. The UI doesn't judge; use `pf.judge` below.
Set `PF_MOCK=1` for a no-model run, and `PF_CORPUS` / `PF_CONFIG` to point at a
scratch corpus or config.

Both front ends share `corpus/debate_state.json`, saved after every turn, so a
crash or closed tab resumes at the exact turn, and a debate started in one
front end can be finished in the other. There is one debate per corpus.

Rules the tool enforces:
- The human writes their own constructive (one paste, at least one
  contention). Their contentions are sliced verbatim from their own text.
- Word limits: constructive 600, rebuttal 500, summary 400, final focus 400
  (crossfire has none). An over-limit human speech is rejected and re-asked.
  Tool speeches are trimmed to the last full sentence inside the limit.
- Every contention is tracked as CONCEDED (never attacked), HELD, or CONTESTED
  from the argument graph of all questions, answers and rebuttals.

Anti-fabrication guards on the tool's output:
- **Citation guard:** with no evidence behind an answer, any year, percentage,
  dollar figure, study reference, attribution or named organisation is blocked.
  The answer is regenerated once, then replaced with a canned no-source answer.
  Every `none` answer is logged to `corpus/gaps.jsonl`
  (`python -m pf.debate gaps` summarizes it). There is deliberately no live web
  search mid-round.
- **Mechanism-swap guard:** checks that an attack engages the resolution's real
  mechanism, not a lookalike policy. A surviving swap is flagged and the judge
  re-checks it.

Single-purpose tools:

```bash
python -m pf.debate case --side pro|con|both          # writes corpus/case_<side>.json
python -m pf.debate respond --side pro --query "..." [--format crossfire|rebuttal|summary|final]
```

## Part 4 - Judge

```bash
python -m pf.judge score --debate corpus/debates/debate_<ts>.json [--samples 3] [--mock]
python -m pf.judge latest --corpus ./corpus
python -m pf.judge trend  --corpus ./corpus
```

Per-side metrics are computed from the argument graph *before* the LLM call.
The judge is sampled `--samples` times, and the majority winner is taken with
averaged scores on argument quality, refutation, impact and weighing. Scores
must cite `attack#N` / `pro-contention#N` refs, which are validated against the
graph. Output goes to `corpus/debates/judged_<debate>.json`.

---

## Outputs (`corpus/`)

```
raw/                 original fetched pages, untouched (<sha1-of-url>.html.gz)
manifest.jsonl       what happened to each URL
documents.jsonl      clean text + metadata (write-once; chunk/claim offsets point into it)
chunks.jsonl         passages
claims.jsonl         the knowledge base (claims with verified quotes)
kb.sqlite            searchable index the debate tool queries
case_pro.json        the tool's case, built from the KB (and case_con.json)
gaps.jsonl           questions the KB couldn't answer
debate_state.json    the live debate
debates/             finished debate archives and judged_*.json
rounds/              legacy self-play archives (not readable by the judge)
*_report.json        quality report at each stage
```

Debate state and the argument graph store `claim_id`s, never quote text, so the
KB stays the single source of truth for evidence.

## Tests and lint

```bash
python -m pyflakes pf/*.py kb/*.py app.py
python -m tests.test_word_limits   # word-limit checks, uses a fake client
```

Never test against the real `corpus/`: copy `kb.sqlite` and `case_*.json` to a
scratch directory and pass `--corpus <scratch>`.

## If something breaks

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: pf` / `kb` | Run from `Scripts/` and use `python -m pf.<module>` |
| `BEDROCK_MODEL_ID is not set` | Set it in `.env` |
| Bedrock throttling / context / credentials error | Shown as a readable exit; state is saved, rerun the same command |
| `No documents found` in step 1 | Wrong path, or a scanned PDF with no text layer |
| `No manifest` in step 3 | Step 2 didn't finish |
| High claim rejection rate | Model is paraphrasing; fix the cause, don't loosen the match |
| Debate won't start | Check `config/resolution.json` and that `corpus/kb.sqlite` exists |
