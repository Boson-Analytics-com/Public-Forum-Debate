"""
Step 6: claims.jsonl -> kb.sqlite   (build, then search)

Hybrid retrieval over the knowledge base:
  - keyword search via SQLite FTS5/BM25   (good at names, numbers, legislation)
  - semantic search via embeddings        (good at matching an argument to an
                                           argument worded differently)
  - fused with reciprocal rank fusion, filtered by stance and contention

Embeddings are optional. Without sentence-transformers installed it runs
keyword-only, which still works -- just less well on paraphrased questions.

Search returns a RETRIEVAL TIER, which is what the debate tool needs in order to
know how to answer:

    direct   strong match          -> argue with the quote
    related  same area, not exact  -> use the evidence, extend the logic
    none     nothing useful        -> reason from warrant, do NOT cite anything

Usage:
    python index.py build --corpus ./corpus
    python index.py search --corpus ./corpus --query "carbon tax is regressive"
    python index.py search --corpus ./corpus --query "..." --stance con --k 5
    python index.py stats  --corpus ./corpus
"""

import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path

import numpy as np

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Tier thresholds. RRF is used only for RANKING -- it is rank-based, so a
# top-ranked garbage match scores the same as a top-ranked perfect one. Tiering
# therefore uses ABSOLUTE signals: cosine similarity, and what fraction of the
# question's content words the best result actually covers.
TIER_COS_DIRECT = 0.55
TIER_COS_RELATED = 0.32
TIER_COV_DIRECT = 0.55
TIER_COV_RELATED = 0.28

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "of", "to", "in", "on", "for",
    "with", "as", "by", "at", "from", "is", "are", "was", "were", "be", "been",
    "will", "would", "can", "could", "should", "do", "does", "did", "have",
    "has", "had", "that", "this", "these", "those", "it", "its", "their",
    "there", "what", "which", "who", "how", "why", "when", "about", "than",
    "then", "so", "just", "also", "not", "no", "any", "all", "your", "you",
    "my", "we", "they", "them", "he", "she", "his", "her", "actually", "really",
}


def content_words(text: str) -> list:
    words = re.findall(r"[a-z0-9]+", (text or "").lower())
    return [w for w in words if w not in STOPWORDS and len(w) > 2]


def term_coverage(query: str, doc: str) -> float:
    """Fraction of the question's content words present in the result.

    Prefix matching stands in for stemming, so "manufacturers" matches
    "manufacturing" without pulling in a full stemmer dependency.
    """
    qw = set(content_words(query))
    if not qw:
        return 0.0
    dw = content_words(doc)
    hit = 0
    for w in qw:
        stem = w[:max(4, len(w) - 3)]
        if any(d.startswith(stem) or w.startswith(d[:max(4, len(d) - 3)]) for d in dw):
            hit += 1
    return hit / len(qw)

SCHEMA = """
CREATE TABLE IF NOT EXISTS claims (
    claim_id TEXT PRIMARY KEY,
    chunk_id TEXT, doc_id TEXT,
    claim TEXT, warrant TEXT, impact TEXT,
    quantification TEXT, scope TEXT, hedge TEXT,
    stance TEXT, contention_tags TEXT,
    quote TEXT, quote_role TEXT,
    quote_start INTEGER, quote_end INTEGER, quote_match TEXT,
    cite TEXT, url TEXT, published_at TEXT, host TEXT, heading TEXT,
    embedding BLOB
);
CREATE INDEX IF NOT EXISTS idx_stance ON claims(stance);
CREATE INDEX IF NOT EXISTS idx_date   ON claims(published_at);
CREATE VIRTUAL TABLE IF NOT EXISTS claims_fts USING fts5(
    claim_id UNINDEXED, searchtext, tokenize='porter unicode61'
);
"""


_ENCODER = "unset"


def get_encoder():
    """Load the embedding model once. Any failure (not installed, no network,
    no cached model) degrades to keyword-only rather than crashing."""
    global _ENCODER
    if _ENCODER != "unset":
        return _ENCODER
    try:
        from sentence_transformers import SentenceTransformer
        _ENCODER = SentenceTransformer(EMBED_MODEL)
    except ImportError:
        _ENCODER = None
    except Exception as e:
        print(f"[warn] could not load {EMBED_MODEL} ({type(e).__name__}). "
              f"Falling back to keyword-only search.", file=sys.stderr)
        _ENCODER = None
    return _ENCODER


def searchtext(c: dict) -> str:
    """What gets indexed: the argument, not the raw passage. Matching an
    opponent's claim against your claim+warrant beats matching against prose."""
    parts = [c.get("claim"), c.get("warrant"), c.get("impact"),
             " ".join(c.get("contention_tags") or []), c.get("quote")]
    return "\n".join(p for p in parts if p)


def build(corpus: Path):
    claims_path = corpus / "claims.jsonl"
    if not claims_path.exists():
        sys.exit(f"No claims.jsonl at {claims_path}. Run claims.py first.")
    claims = [json.loads(l) for l in claims_path.open(encoding="utf-8") if l.strip()]
    print(f"{len(claims)} claims")

    db_path = corpus / "kb.sqlite"
    if db_path.exists():
        db_path.unlink()
    con = sqlite3.connect(db_path)
    con.executescript(SCHEMA)

    enc = get_encoder()
    if enc is None:
        print("sentence-transformers not installed -> keyword-only index.")
        print("  pip install sentence-transformers   for semantic search")
        vecs = [None] * len(claims)
    else:
        print(f"Embedding with {EMBED_MODEL} ...")
        texts = [f"{c.get('claim','')} {c.get('warrant','') or ''}" for c in claims]
        arr = enc.encode(texts, normalize_embeddings=True, batch_size=64,
                         show_progress_bar=False).astype("float32")
        vecs = [row.tobytes() for row in arr]

    for c, vec in zip(claims, vecs):
        con.execute("""INSERT OR REPLACE INTO claims VALUES
            (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", (
            c["claim_id"], c.get("chunk_id"), c.get("doc_id"),
            c.get("claim"), c.get("warrant"), c.get("impact"),
            json.dumps(c.get("quantification")), c.get("scope"), c.get("hedge"),
            c.get("stance"), json.dumps(c.get("contention_tags") or []),
            c.get("quote"), c.get("quote_role"),
            c.get("quote_start"), c.get("quote_end"), c.get("quote_match"),
            c.get("cite"), c.get("url"), c.get("published_at"),
            c.get("host"), c.get("heading"), vec))
        con.execute("INSERT INTO claims_fts (claim_id, searchtext) VALUES (?,?)",
                    (c["claim_id"], searchtext(c)))
    con.commit()
    con.close()
    print(f"-> {db_path}")


def _fts_query(q: str) -> str:
    """FTS5 OR-query. Bare user text would be parsed as syntax, so quote each
    token and drop anything that is not alphanumeric."""
    toks = [t for t in (w.strip("\"'()[]{}.,;:!?") for w in q.split()) if t.isalnum() and len(t) > 2]
    return " OR ".join(f'"{t}"' for t in toks) if toks else '""'


def search(corpus: Path, query: str, k: int, stance: str | None,
           tags: list | None, since: str | None, verbose: bool = True):
    con = sqlite3.connect(corpus / "kb.sqlite")
    con.row_factory = sqlite3.Row

    where, params = [], []
    if stance:
        where.append("c.stance = ?"); params.append(stance)
    if since:
        where.append("(c.published_at IS NULL OR c.published_at >= ?)"); params.append(since)
    if tags:
        where.append("(" + " OR ".join("c.contention_tags LIKE ?" for _ in tags) + ")")
        params += [f'%"{t}"%' for t in tags]
    filt = (" AND " + " AND ".join(where)) if where else ""

    # --- lexical ---
    lex = [r["claim_id"] for r in con.execute(
        f"""SELECT c.claim_id FROM claims_fts f JOIN claims c ON c.claim_id = f.claim_id
            WHERE claims_fts MATCH ?{filt} ORDER BY bm25(claims_fts) LIMIT 60""",
        [_fts_query(query)] + params)]

    # --- semantic ---
    sem, cos_by_id = [], {}
    enc = get_encoder()
    if enc is not None:
        rows = list(con.execute(
            f"SELECT c.claim_id, c.embedding FROM claims c WHERE c.embedding IS NOT NULL{filt}",
            params))
        if rows:
            mat = np.vstack([np.frombuffer(r["embedding"], dtype="float32") for r in rows])
            qv = enc.encode([query], normalize_embeddings=True).astype("float32")[0]
            sims = mat @ qv
            order = np.argsort(-sims)[:60]
            sem = [rows[i]["claim_id"] for i in order]
            cos_by_id = {rows[i]["claim_id"]: float(sims[i]) for i in order}

    # --- reciprocal rank fusion ---
    fused = {}
    for lst, weight in ((lex, 1.0), (sem, 1.0)):
        for rank, cid in enumerate(lst):
            fused[cid] = fused.get(cid, 0.0) + weight / (60 + rank + 1)

    top = sorted(fused.items(), key=lambda x: -x[1])[:k]

    results = []
    for cid, score in top:
        row = dict(con.execute("SELECT * FROM claims WHERE claim_id = ?", (cid,)).fetchone())
        row["contention_tags"] = json.loads(row["contention_tags"] or "[]")
        row.pop("embedding", None)
        row["score"] = round(score, 5)
        row["cosine"] = round(cos_by_id.get(cid, 0.0), 4)
        row["coverage"] = round(term_coverage(query, searchtext(row)), 3)
        results.append(row)
    con.close()

    # Tier off the best ABSOLUTE match quality, not the fused rank.
    best_cos = max((r["cosine"] for r in results), default=0.0)
    best_cov = max((r["coverage"] for r in results), default=0.0)
    if best_cos >= TIER_COS_DIRECT or best_cov >= TIER_COV_DIRECT:
        tier = "direct"
    elif best_cos >= TIER_COS_RELATED or best_cov >= TIER_COV_RELATED:
        tier = "related"
    else:
        tier = "none"
    if tier == "none":
        results = []

    out = {"query": query, "tier": tier, "best_cosine": round(best_cos, 4),
           "best_coverage": round(best_cov, 3), "embeddings_used": enc is not None,
           "lexical_hits": len(lex), "semantic_hits": len(sem), "results": results}

    if verbose:
        print(f"\nQuery: {query}")
        mode = "hybrid" if enc is not None else "keyword-only"
        print(f"TIER: {tier.upper()}   (cosine {best_cos:.3f}, coverage {best_cov:.2f}, {mode})")
        if tier == "none":
            print("-> No usable evidence. The tool must reason from its own warrant")
            print("   and must NOT cite any source, study or number.")
        for i, r in enumerate(results, 1):
            print(f"\n{i}. [{r['stance'].upper()}] {r['claim']}")
            print(f"   because: {r['warrant']}")
            print(f"   so what: {r['impact']}")
            print(f"   quote:   \"{(r['quote'] or '')[:120]}\"")
            print(f"   cite:    {r['cite']}")
            print(f"   tags:    {r['contention_tags']}  cos {r['cosine']} cov {r['coverage']}")
    return out


def rebut(corpus: Path, attack: str, side: str, k: int, verbose: bool = True):
    """Answer an opponent's attack, rather than matching it.

    Plain search finds claims that AGREE with the attack, because agreement is
    what semantic similarity measures. A debate tool needs the opposite. So:

      1. find the attack's strongest form (opponent side) -> what you are facing
      2. read which contention areas it lives in
      3. search YOUR side within those areas          -> what you answer with
      4. if your side is thin there, say so, so the tool reasons instead of
         bluffing with evidence it does not have
    """
    opp = "con" if side == "pro" else "pro"

    threat = search(corpus, attack, k=3, stance=opp, tags=None, since=None, verbose=False)
    tags = []
    for r in threat["results"]:
        for t in r["contention_tags"]:
            if t not in tags:
                tags.append(t)

    answer = search(corpus, attack, k=k, stance=side, tags=tags or None,
                    since=None, verbose=False)

    # Own-side evidence is judged on ABSOLUTE quality. A weak best match means
    # nothing in the KB actually answers this, however it ranks.
    best = max((r["cosine"] for r in answer["results"]), default=0.0)
    n = len(answer["results"])
    if answer["tier"] == "direct" and n >= 2 and best >= 0.45:
        strength = "strong"
    elif n >= 1 and (best >= 0.30 or answer["tier"] != "none"):
        strength = "thin"
    else:
        strength = "none"

    guidance = {
        "strong": "Answer with the evidence below. Quote it and cite it.",
        "thin": ("Weak coverage. Lead with reasoning from your own warrant; use the "
                 "evidence below only as support, and do not overstate it."),
        "none": ("NO own-side evidence. Reason from your warrant, push the burden back, "
                 "or concede narrowly while holding the contention. Cite NOTHING -- "
                 "name no study, number, organisation or author."),
    }[strength]

    out = {"attack": attack, "side": side, "answer_strength": strength,
           "contention_areas": tags, "guidance": guidance,
           "threat": threat["results"], "answer": answer["results"] if strength != "none" else []}

    if verbose:
        print(f"\nATTACK: {attack}")
        print(f"AREAS:  {', '.join(tags) or '(none identified)'}")
        print(f"\n--- WHAT YOU ARE FACING ({opp.upper()}) ---")
        for r in threat["results"][:2]:
            print(f"  * {r['claim'][:100]}")
        print(f"\n--- YOUR ANSWER ({side.upper()}) --- strength: {strength.upper()}")
        print(f"  {guidance}")
        for i, r in enumerate(out["answer"], 1):
            print(f"\n{i}. {r['claim']}")
            print(f"   because: {r['warrant']}")
            print(f"   quote:   \"{(r['quote'] or '')[:110]}\"")
            print(f"   cite:    {r['cite']}")
            print(f"   cos {r['cosine']}")
    return out


def stats(corpus: Path):
    con = sqlite3.connect(corpus / "kb.sqlite")
    total = con.execute("SELECT COUNT(*) FROM claims").fetchone()[0]
    print(f"Claims:   {total}")
    print("Stance:  ", dict(con.execute("SELECT stance, COUNT(*) FROM claims GROUP BY stance")))
    print("Sources: ", con.execute("SELECT COUNT(DISTINCT doc_id) FROM claims").fetchone()[0])
    print("Hosts:   ", con.execute("SELECT COUNT(DISTINCT host) FROM claims").fetchone()[0])
    print("Quote match:", dict(con.execute("SELECT quote_match, COUNT(*) FROM claims GROUP BY quote_match")))
    tags = {}
    for (raw,) in con.execute("SELECT contention_tags FROM claims"):
        for t in json.loads(raw or "[]"):
            tags[t] = tags.get(t, 0) + 1
    print("Contentions:")
    for t, n in sorted(tags.items(), key=lambda x: -x[1]):
        print(f"   {t:30} {n}")
    con.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["build", "search", "stats", "rebut"])
    ap.add_argument("--corpus", type=Path, default=Path("./corpus"))
    ap.add_argument("--query", default=None)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--stance", choices=["pro", "con", "neutral"], default=None)
    ap.add_argument("--tags", default=None, help="Comma-separated contention tags")
    ap.add_argument("--since", default=None, help="Drop evidence older than YYYY-MM-DD")
    ap.add_argument("--side", choices=["pro", "con"], default="pro",
                    help="Which side the tool argues (rebut mode)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if args.command == "build":
        build(args.corpus)
    elif args.command == "rebut":
        if not args.query:
            sys.exit("rebut needs --query (the opponent's argument)")
        out = rebut(args.corpus, args.query, args.side, args.k, verbose=not args.json)
        if args.json:
            print(json.dumps(out, indent=2, ensure_ascii=False))
    elif args.command == "stats":
        stats(args.corpus)
    else:
        if not args.query:
            sys.exit("search needs --query")
        tags = [t.strip() for t in args.tags.split(",")] if args.tags else None
        out = search(args.corpus, args.query, args.k, args.stance, tags,
                     args.since, verbose=not args.json)
        if args.json:
            print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
