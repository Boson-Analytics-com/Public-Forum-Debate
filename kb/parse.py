"""
Stage 2: archived HTML -> documents.jsonl with real provenance.

This is the step that makes re-scraping worth it. Publication date, author,
section, and headline come from <meta> tags and JSON-LD, which are reliable --
not from regex over prose, which is not. Re-run this as often as you like; it
never touches the network.

Output records carry an immutable `normalized_text`. Every downstream quote is
a (doc_id, start_char, end_char) offset into that exact string, so treat it as
write-once: change the cleaning rules, bump doc_version, reprocess claims.

Usage:
    python parse.py --corpus ./corpus --out ./corpus/documents.jsonl
"""

import argparse
import gzip
import hashlib
import json
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

from bs4 import BeautifulSoup

try:
    import trafilatura
except ImportError:
    trafilatura = None

DATE_META = ["article:published_time", "article:modified_time", "og:published_time",
             "datePublished", "publishdate", "pubdate", "date", "dc.date",
             "dc.date.issued", "citation_publication_date", "sailthru.date"]
AUTHOR_META = ["author", "article:author", "og:article:author", "byl",
               "dc.creator", "citation_author", "sailthru.author",
               "parsely-author", "twitter:creator"]
STRIP_TAGS = ["script", "style", "noscript", "nav", "header", "footer", "aside",
              "form", "iframe", "figure", "figcaption"]


def read_raw(path: Path) -> str:
    data = gzip.open(path, "rb").read() if str(path).endswith(".gz") else path.read_bytes()
    for enc in ("utf-8", "cp1252", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", "replace")


def meta_lookup(soup, keys):
    """Check name=, property=, and itemprop= -- sites use all three."""
    lowered = {k.lower() for k in keys}
    for tag in soup.find_all("meta"):
        for attr in ("name", "property", "itemprop"):
            key = (tag.get(attr) or "").lower()
            if key in lowered:
                val = (tag.get("content") or "").strip()
                if val:
                    return val, f"meta[{attr}={key}]"
    return None, None


def jsonld_records(soup):
    for tag in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(tag.string or "")
        except (json.JSONDecodeError, TypeError):
            continue
        stack = [data]
        while stack:
            node = stack.pop()
            if isinstance(node, list):
                stack.extend(node)
            elif isinstance(node, dict):
                yield node
                stack.extend(v for v in node.values() if isinstance(v, (dict, list)))


def from_jsonld(soup):
    out = {}
    for node in jsonld_records(soup):
        types = node.get("@type", "")
        types = types if isinstance(types, list) else [types]
        if not any(t in {"NewsArticle", "Article", "Report", "ScholarlyArticle",
                         "BlogPosting", "WebPage"} for t in types):
            continue
        out.setdefault("published_at", node.get("datePublished"))
        out.setdefault("modified_at", node.get("dateModified"))
        out.setdefault("headline", node.get("headline"))
        out.setdefault("section", node.get("articleSection"))
        author = node.get("author")
        if author and "author" not in out:
            if isinstance(author, dict):
                out["author"] = author.get("name")
            elif isinstance(author, list) and author:
                names = [a.get("name") if isinstance(a, dict) else str(a) for a in author]
                out["author"] = ", ".join(n for n in names if n)
            elif isinstance(author, str):
                out["author"] = author
        pub = node.get("publisher")
        if isinstance(pub, dict):
            out.setdefault("publisher", pub.get("name"))
    return {k: v for k, v in out.items() if v}


def normalize_date(raw):
    """Return ISO date + a flag, rather than guessing. A wrong date is worse
    than a null one on a current-events resolution."""
    if not raw:
        return None, "missing"
    s = str(raw).strip()
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M%z",
                "%Y-%m-%d", "%Y/%m/%d", "%d %B %Y", "%B %d, %Y", "%b %d, %Y"):
        try:
            return datetime.strptime(s.replace("Z", "+0000") if fmt.endswith("%z") else s,
                                     fmt).date().isoformat(), "parsed"
        except ValueError:
            continue
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", s)
    if m:
        return m.group(0), "regex"
    return None, f"unparsed:{s[:40]}"


def extract_body(html: str) -> tuple:
    """trafilatura when available (much better at boilerplate), soup fallback."""
    if trafilatura:
        text = trafilatura.extract(html, include_comments=False, include_tables=True,
                                  favor_precision=True, no_fallback=False)
        if text and len(text) > 200:
            return text, "trafilatura"
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(STRIP_TAGS):
        tag.decompose()
    node = soup.find("article") or soup.find("main") or soup.body or soup
    return node.get_text("\n"), "bs4_fallback"


def heading_outline(html: str) -> list:
    """Real headings from real tags -- the thing plaintext scraping destroys.

    A sentence under <h2>Limitations</h2> means something different from the
    same sentence under <h2>Findings</h2>. Keep the structure.
    """
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(STRIP_TAGS):
        tag.decompose()
    return [{"level": int(h.name[1]), "text": " ".join(h.get_text().split())}
            for h in soup.find_all(["h1", "h2", "h3"])
            if h.get_text(strip=True)][:40]


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    for a, b in [("\u200b", ""), ("\ufeff", ""), ("\u00a0", " "),
                 ("\u2018", "'"), ("\u2019", "'"), ("\u201c", '"'),
                 ("\u201d", '"'), ("\u2013", "-"), ("\u2014", "--")]:
        text = text.replace(a, b)
    lines = [re.sub(r"[ \t]+", " ", ln.strip()) for ln in text.split("\n")]
    text = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, default=Path("./corpus"))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--min-chars", type=int, default=400)
    args = ap.parse_args()

    manifest = args.corpus / "manifest.jsonl"
    out_path = args.out or (args.corpus / "documents.jsonl")
    if not manifest.exists():
        raise SystemExit(f"No manifest at {manifest}. Run fetch.py first.")

    fetched = [json.loads(l) for l in manifest.open(encoding="utf-8") if l.strip()]
    ok = [r for r in fetched if r.get("status") == "ok" and Path(r["raw_path"]).exists()]
    print(f"{len(fetched)} manifest records | {len(ok)} archived pages to parse")

    seen, records, dropped = {}, [], []
    for rec in ok:
        html = read_raw(Path(rec["raw_path"]))
        soup = BeautifulSoup(html, "lxml")

        ld = from_jsonld(soup)
        date_raw, date_src = ld.get("published_at"), "jsonld"
        if not date_raw:
            date_raw, date_src = meta_lookup(soup, DATE_META)
        published_at, date_status = normalize_date(date_raw)

        author = ld.get("author")
        author_src = "jsonld" if author else None
        if not author:
            author, author_src = meta_lookup(soup, AUTHOR_META)

        body, extractor = extract_body(html)
        normalized = normalize_text(body)

        if len(normalized) < args.min_chars:
            dropped.append({"url": rec["url"], "reason": "too_short", "chars": len(normalized)})
            continue

        digest = hashlib.sha256(normalized.encode()).hexdigest()
        if digest in seen:
            records[seen[digest]]["duplicate_urls"].append(rec["url"])
            dropped.append({"url": rec["url"], "reason": "duplicate_of",
                            "canonical": records[seen[digest]]["doc_id"]})
            continue
        seen[digest] = len(records)

        title = (ld.get("headline")
                 or (soup.title.get_text(strip=True) if soup.title else None))
        canon = soup.find("link", rel="canonical")

        records.append({
            "doc_id": f"doc_{digest[:12]}",
            "doc_version": 1,
            "url": rec["url"],
            "final_url": rec.get("final_url"),
            "canonical_url": canon.get("href") if canon else None,
            "duplicate_urls": [],
            "host": rec.get("host"),
            "title": title,
            "author": author,
            "author_source": author_src,
            "publisher": ld.get("publisher"),
            "section": ld.get("section"),
            "published_at": published_at,
            "date_source": date_src if published_at else None,
            "date_status": date_status,
            "http_last_modified": rec.get("last_modified"),
            "headings": heading_outline(html),
            "extractor": extractor,
            "char_count": len(normalized),
            "content_hash": digest,
            "raw_path": rec["raw_path"],
            "fetched_at": rec.get("fetched_at"),
            "parsed_at": datetime.now(timezone.utc).isoformat(),
            "normalized_text": normalized,
        })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    no_date = [r["doc_id"] for r in records if not r["published_at"]]
    no_author = sum(1 for r in records if not r["author"])
    fallback = sum(1 for r in records if r["extractor"] == "bs4_fallback")
    report = {
        "parsed": len(records),
        "dropped": dropped,
        "missing_date": no_date,
        "missing_author_count": no_author,
        "bs4_fallback_count": fallback,
        "date_status_breakdown": {s: sum(1 for r in records if r["date_status"] == s)
                                  for s in {r["date_status"] for r in records}},
        "median_chars": sorted(r["char_count"] for r in records)[len(records) // 2] if records else 0,
    }
    (args.corpus / "parse_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))

    print(f"Parsed {len(records)}, dropped {len(dropped)}")
    print(f"Missing date: {len(no_date)}  |  missing author: {no_author}  |  bs4 fallback: {fallback}")
    print(f"-> {out_path}")
    print(f"-> {args.corpus / 'parse_report.json'}")


if __name__ == "__main__":
    main()
