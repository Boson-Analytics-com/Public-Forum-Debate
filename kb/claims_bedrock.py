"""
Step 5 (Bedrock variant): chunks.jsonl -> claims.jsonl
                           *** THIS IS THE KNOWLEDGE BASE ***

Identical to claims.py except the LLM call goes through AWS Bedrock instead of
Anthropic's own API directly. The model itself is not hardcoded to any one
tier -- it's whatever BEDROCK_MODEL_ID is set to in the environment (see
round.py/debate.py/opponent.py/judge.py for the same convention). Everything
else -- the quote-verification guardrail, the retry-once logic, the
reporting -- is untouched, because locate_quote() is a pure string match
against the source text and does not care which model or which API surface
produced the quote.

For each chunk, an LLM extracts the arguable points as structured claims. Every
claim must include a quote copied word-for-word from the chunk. The script then
searches the source document for that quote:

    exact match      -> claim kept, character offsets stored
    whitespace-only  -> claim kept, flagged for review
    no match         -> claim REJECTED and the chunk retried once

This is the guardrail that stops the debate tool inventing evidence. A quote it
cannot locate in the source never enters the knowledge base.

COST NOTE: cost scales with whichever model BEDROCK_MODEL_ID points to -- a
top-tier model costs meaningfully more per token than a mid-tier one, and the
README's $15-40 estimate assumes a mid-tier model, not a top-tier one. Run
--limit 10 or 15 first (smaller than the usual --limit 30) and check actual
spend/cost-per-claim before scaling up, especially since you're spending AWS
credits rather than an unlimited budget.

Setup:
    pip install boto3
    export AWS_ACCESS_KEY_ID=...
    export AWS_SECRET_ACCESS_KEY=...
    export AWS_REGION=us-east-1
    export BEDROCK_MODEL_ID=...   # verify this model is enabled for your account in this region

    One-time AWS-side step, separate from credentials: in the Bedrock console,
    go to Model catalog -> request/enable access to whichever model
    BEDROCK_MODEL_ID names, in your target region. Calls will fail with an
    access-denied error if this hasn't been done, even with fully valid keys.

Usage:
    python claims_bedrock_opus.py --corpus ./corpus --config resolution.json
    python claims_bedrock_opus.py --corpus ./corpus --config resolution.json --limit 15
    python claims_bedrock_opus.py --corpus ./corpus --config resolution.json --mock
"""

import argparse
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# No hardcoded fallback: BEDROCK_MODEL_ID must be set in the actual
# environment. Verify the model ID in the AWS console (Bedrock -> Model
# catalog) before running for real -- Bedrock model IDs (and inference
# profile ARNs) are specific strings assigned at catalog registration.
MODEL = os.environ.get("BEDROCK_MODEL_ID")

# Multiplier on every call's max tokens. 1.0 = full budget (production).
# Lower it (e.g. 0.5) only to hold cost down while testing -- set the same
# value in debate.py, round.py, claims.py and claims_bedrock_opus.py.
MAX_TOKENS_SCALE = 1.0

SYSTEM = """You extract debate evidence from source passages for a Public Forum debate knowledge base.

A claim is an ARGUABLE point, not a fact restatement. "The EU adopted CBAM in 2023" is not a claim. "CBAM raises EU manufacturing costs" is.

For every claim you must give:
- claim: the arguable assertion, one sentence
- warrant: WHY it is true, the mechanism
- impact: WHY IT MATTERS, the consequence if true
- quantification: numbers if the passage has them, else null
- scope: who/where/when it applies, else null
- hedge: the passage's own certainty language ("projected", "may", "found") else null
- stance: "pro", "con", or "neutral" relative to the resolution given
- contention_tags: one or more from the allowed list
- quote: text copied CHARACTER-FOR-CHARACTER from the passage
- quote_role: "statistic", "mechanism", "impact", "concession", or "definition"

CRITICAL RULES FOR quote:
1. Copy verbatim from the passage. Do not paraphrase, correct, shorten or tidy.
2. Do not add or remove punctuation, quotation marks, or ellipses.
3. Do not join text from two different places. One continuous span only.
4. Keep it between 10 and 60 words.
5. If no continuous span supports the claim, DROP THE CLAIM.

Extract stance honestly. A pro-leaning source often contains the best con evidence -- those concessions are valuable, so record them as stance "con" with quote_role "concession".

Extract 0 to 6 claims. Zero is a valid answer for a passage with no arguable content (navigation text, author bios, boilerplate).

Return ONLY a JSON object: {"claims": [...]}. No markdown fences, no commentary."""

USER_TMPL = """RESOLUTION: {resolution}

ALLOWED contention_tags: {tags}

SOURCE: {title} -- {author}, {host}, {published_at}
SECTION: {heading}

PASSAGE:
\"\"\"
{text}
\"\"\"

Extract the claims as JSON."""

RETRY_NOTE = """
Your previous attempt returned quotes that do not appear in the passage.
Copy each quote by locating it in the passage above and reproducing those exact
characters. Do not retype from memory. Drop any claim you cannot quote exactly."""


def normalise_ws(s: str) -> str:
    return " ".join(s.split())


def locate_quote(quote: str, doc_text: str):
    """Return (start, end, how). how is 'exact', 'whitespace', or None."""
    if not quote or len(quote) < 15:
        return None, None, None
    i = doc_text.find(quote)
    if i != -1:
        return i, i + len(quote), "exact"

    # Whitespace-tolerant fallback: map normalised position back to the original
    # by walking both strings together.
    nq = normalise_ws(quote)
    idx_map, buf = [], []
    prev_space = True
    for pos, ch in enumerate(doc_text):
        if ch.isspace():
            if not prev_space:
                buf.append(" ")
                idx_map.append(pos)
            prev_space = True
        else:
            buf.append(ch)
            idx_map.append(pos)
            prev_space = False
    flat = "".join(buf)
    j = flat.find(nq)
    if j == -1:
        return None, None, None
    start = idx_map[j]
    end = idx_map[min(j + len(nq) - 1, len(idx_map) - 1)] + 1
    return start, end, "whitespace"


def parse_json(raw: str):
    """Bedrock's converse API has no forced-JSON response mode (unlike
    Gemini's response_mime_type), so this fence-stripping fallback matters
    more here, not less."""
    txt = raw.strip()
    txt = re.sub(r"^```(?:json)?\s*", "", txt)
    txt = re.sub(r"\s*```$", "", txt)
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", txt, re.S)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                return None
    return None


def mock_extract(chunk, cfg):
    """Offline mode. Takes a real sentence from the chunk as the quote so the
    verification path can be exercised without an API key, plus one deliberately
    fabricated quote to prove rejection works."""
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", chunk["text"]) if len(s.split()) >= 10]
    out = []
    for i, s in enumerate(sents[:2]):
        out.append({
            "claim": f"[mock] point from {chunk['chunk_id']} #{i}",
            "warrant": "[mock] mechanism", "impact": "[mock] consequence",
            "quantification": None, "scope": None, "hedge": None,
            "stance": ["pro", "con"][i % 2],
            "contention_tags": [cfg["contention_tags"][0]],
            "quote": " ".join(s.split()[:30]),
            "quote_role": "mechanism",
        })
    out.append({
        "claim": "[mock] fabricated-quote claim that must be rejected",
        "warrant": "x", "impact": "y", "quantification": None, "scope": None,
        "hedge": None, "stance": "neutral",
        "contention_tags": [cfg["contention_tags"][0]],
        "quote": "this sentence does not appear anywhere in the source document at all",
        "quote_role": "impact",
    })
    return {"claims": out}


def call_llm(client, chunk, cfg, retry=False):
    """--- CHANGED FOR BEDROCK ---
    Same job as the direct-Anthropic version: build the prompt, call the
    model, return raw text. Bedrock's unified converse() API takes the
    system prompt as a list of text blocks and hands back a nested
    output/message/content structure rather than Anthropic's own
    content-block list.
    """
    user = USER_TMPL.format(
        resolution=cfg["resolution"], tags=", ".join(cfg["contention_tags"]),
        title=chunk.get("title") or "unknown", author=chunk.get("author") or "unknown",
        host=chunk.get("host") or "unknown", published_at=chunk.get("published_at") or "undated",
        heading=chunk.get("heading") or "none", text=chunk["text"])
    if retry:
        user += RETRY_NOTE

    resp = client.converse(
        modelId=MODEL,
        system=[{"text": SYSTEM}],
        messages=[{"role": "user", "content": [{"text": user}]}],
        inferenceConfig={"temperature": 0, "maxTokens": max(1, int(2000 * MAX_TOKENS_SCALE))},
    )
    return resp["output"]["message"]["content"][0]["text"]


def process_chunk(chunk, doc_text, cfg, client, mock):
    """Extract, verify, retry once. Returns (kept_claims, stats)."""
    stats = {"api_error": 0, "bad_json": 0, "rejected": 0, "whitespace": 0, "retried": 0}
    kept = []

    for attempt in (0, 1):
        try:
            if mock:
                data = mock_extract(chunk, cfg)
            else:
                raw = call_llm(client, chunk, cfg, retry=(attempt == 1))
                data = parse_json(raw)
                if data is None:
                    stats["bad_json"] += 1
                    continue
        except Exception:
            stats["api_error"] += 1
            time.sleep(2 ** attempt)
            continue

        kept, rejected = [], 0
        for c in data.get("claims", []):
            if not isinstance(c, dict) or not c.get("claim"):
                continue
            start, end, how = locate_quote(c.get("quote", ""), doc_text)
            if how is None:
                rejected += 1
                continue
            if how == "whitespace":
                stats["whitespace"] += 1
            tags = [t for t in (c.get("contention_tags") or []) if t in cfg["contention_tags"]]
            kept.append({
                "claim_id": f"{chunk['chunk_id']}_k{len(kept):02d}",
                "chunk_id": chunk["chunk_id"], "doc_id": chunk["doc_id"],
                "claim": c["claim"], "warrant": c.get("warrant"), "impact": c.get("impact"),
                "quantification": c.get("quantification"), "scope": c.get("scope"),
                "hedge": c.get("hedge"),
                "stance": c.get("stance") if c.get("stance") in ("pro", "con", "neutral") else "neutral",
                "contention_tags": tags or ["unclassified"],
                "quote": doc_text[start:end],
                "quote_role": c.get("quote_role"),
                "quote_start": start, "quote_end": end, "quote_match": how,
                "cite": " -- ".join(x for x in [
                    chunk.get("author"), chunk.get("title"), chunk.get("host"),
                    chunk.get("published_at")] if x),
                "url": chunk.get("url"), "published_at": chunk.get("published_at"),
                "host": chunk.get("host"), "heading": chunk.get("heading"),
                "extracted_at": datetime.now(timezone.utc).isoformat(),
            })

        # Retry only if the model produced claims but could not quote them.
        if rejected and not kept and attempt == 0:
            stats["retried"] += 1
            continue
        stats["rejected"] += rejected
        break

    return kept, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, default=Path("./corpus"))
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--mock", action="store_true", help="No API calls; tests the verify path")
    args = ap.parse_args()

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    for key in ("resolution", "contention_tags"):
        if not cfg.get(key):
            raise SystemExit(f"config is missing '{key}'")

    docs = {d["doc_id"]: d["normalized_text"] for d in
            (json.loads(l) for l in (args.corpus / "documents.jsonl").open(encoding="utf-8") if l.strip())}
    chunks = [json.loads(l) for l in (args.corpus / "chunks.jsonl").open(encoding="utf-8") if l.strip()]

    out_path = args.corpus / "claims.jsonl"
    done = set()
    if out_path.exists():
        done = {json.loads(l)["chunk_id"] for l in out_path.open(encoding="utf-8") if l.strip()}

    todo = [c for c in chunks if c["chunk_id"] not in done and c["doc_id"] in docs]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(chunks)} chunks | {len(done)} already done | {len(todo)} to process")
    if not todo:
        return

    client = None
    if not args.mock:
        # --- CHANGED FOR BEDROCK ---
        missing = [k for k in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "BEDROCK_MODEL_ID")
                   if not os.environ.get(k)]
        if missing:
            raise SystemExit(f"Set {', '.join(missing)}, or use --mock to test the pipeline.")
        import boto3
        region = os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))
        client = boto3.client("bedrock-runtime", region_name=region)

    lock = threading.Lock()
    totals = {"claims": 0, "api_error": 0, "bad_json": 0, "rejected": 0,
              "whitespace": 0, "retried": 0, "empty_chunks": 0}

    with out_path.open("a", encoding="utf-8") as out:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(process_chunk, ch, docs[ch["doc_id"]], cfg, client, args.mock)
                       for ch in todo]
            for i, fut in enumerate(futures, 1):
                kept, stats = fut.result()
                with lock:
                    for c in kept:
                        out.write(json.dumps(c, ensure_ascii=False) + "\n")
                    out.flush()
                    totals["claims"] += len(kept)
                    if not kept:
                        totals["empty_chunks"] += 1
                    for k, v in stats.items():
                        totals[k] += v
                if i % 25 == 0 or i == len(futures):
                    print(f"  {i}/{len(futures)}  claims={totals['claims']} rejected={totals['rejected']}")

    all_claims = [json.loads(l) for l in out_path.open(encoding="utf-8") if l.strip()]
    stance = {s: sum(1 for c in all_claims if c["stance"] == s) for s in ("pro", "con", "neutral")}
    attempted = totals["claims"] + totals["rejected"]
    report = {
        "chunks_processed": len(todo),
        "claims_total": len(all_claims),
        "claims_this_run": totals["claims"],
        "quotes_rejected": totals["rejected"],
        "rejection_rate_pct": round(100 * totals["rejected"] / max(1, attempted), 1),
        "whitespace_matches": totals["whitespace"],
        "retried_chunks": totals["retried"],
        "empty_chunks": totals["empty_chunks"],
        "api_errors": totals["api_error"],
        "bad_json": totals["bad_json"],
        "stance_balance": stance,
        "claims_per_chunk": round(len(all_claims) / max(1, len(chunks)), 2),
    }
    (args.corpus / "claims_report.json").write_text(json.dumps(report, indent=2))

    print(f"\nClaims in KB:     {len(all_claims)}")
    print(f"Quotes rejected:  {totals['rejected']}  ({report['rejection_rate_pct']}%)")
    print(f"Stance balance:   pro {stance['pro']} / con {stance['con']} / neutral {stance['neutral']}")
    print(f"-> {out_path}")
    if report["rejection_rate_pct"] > 10:
        print("\nWARNING: rejection rate above 10%. The model is paraphrasing instead")
        print("of copying. Report this rather than lowering the threshold.")


if __name__ == "__main__":
    main()