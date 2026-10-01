"""
Stage 0: pull EVERY link out of a document that also contains lots of prose.

Capture-everything mode. Nothing is dropped by default -- social links, bare
domains with no scheme, shorteners and homepages all come through. Filtering is
opt-in via flags. Each row is labelled so you can review before fetching.

Finds links in five places:
  1. bare http(s):// URLs in text
  2. scheme-less domains: www.example.com/a  or  example.com/a
  3. markdown [label](url) and html href="url"
  4. docx hyperlink relationships  -- URL hidden in word/_rels/, not in text
  5. pdf link annotations          -- clickable but absent from the text layer

Usage:
    python extract_urls.py sources.docx -o urls.csv
    python extract_urls.py notes.pdf -o urls.csv --drop-assets
"""

import argparse
import csv
import re
import sys
import zipfile
from pathlib import Path
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode

# --- matching ----------------------------------------------------------------

# Scheme-ful URLs. Trailing class excludes sentence punctuation so
# "see https://x.com/a." does not capture the period.
URL_RE = re.compile(r"""(?:https?|ftp)://[^\s<>"'\]\)},;]+[^\s<>"'\]\)}.,;:!?]""", re.I)

# Scheme-less domains. Requires a known TLD so ordinary prose ("version 2.5",
# "notes.txt", "Fig.4") does not match. Case-sensitive on the TLD for the same
# reason -- "trade.In 2026" from a missing space stays out.
TLDS = (
    "com|org|net|edu|gov|mil|int|info|news|press|blog|io|co|ai|dev|app|me|tv|us"
    r"|uk|in|eu|ca|au|de|fr|jp|cn|ru|br|za|nz|ie|nl|se|no|dk|fi|ch|at|be|es|it"
    r"|pl|mx|ar|kr|sg|hk|tw|il|ae|tr|id|ph|vn|th|pk|ng|ke|cl|pe|gr|pt|cz|hu|ro"
)
BARE_DOMAIN_RE = re.compile(
    r"(?<![\w@/.])((?:www\.)?(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+"
    rf"(?:{TLDS})(?:\.[a-z]{{2}})?"
    r"(?:/[^\s<>\"'\]\)},;]*)?)",
)

MD_LINK_RE = re.compile(r"\[[^\]]*\]\(\s*(<?)([^\s\)>]+)")
HTML_HREF_RE = re.compile(r"""(?:href|src)\s*=\s*["']([^"']+)["']""", re.I)

TRACKING_PREFIXES = ("utm_", "fbclid", "gclid", "mc_cid", "mc_eid", "igshid",
                     "ref_src", "ref_url", "spm", "_hsenc", "_hsmi", "yclid",
                     "at_medium", "at_campaign", "cmpid", "ito")

SOCIAL_HOSTS = {"twitter.com", "x.com", "facebook.com", "instagram.com",
                "linkedin.com", "reddit.com", "tiktok.com", "youtube.com",
                "youtu.be", "pinterest.com", "threads.net", "mastodon.social",
                "bsky.app", "t.me"}
SHORTENERS = {"t.co", "bit.ly", "tinyurl.com", "ow.ly", "buff.ly", "lnkd.in",
              "goo.gl", "dlvr.it", "trib.al", "shorturl.at"}
ASSET_EXT = {".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp", ".ico", ".bmp",
             ".css", ".js", ".woff", ".woff2", ".ttf", ".mp3", ".mp4", ".mov",
             ".avi", ".zip", ".gz", ".tar", ".rss", ".atom"}
DOC_EXT = {".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx", ".csv", ".txt"}

# --- readers -----------------------------------------------------------------


def _decode(data: bytes) -> str:
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", "replace")


def from_ooxml(path: Path) -> list:
    """docx/pptx/xlsx. Scans every .rels part plus all visible text.

    A Word link displaying "see this report" stores its target only in
    word/_rels/document.xml.rels. Scanning every .rels part also picks up
    links in footnotes, endnotes, headers, footers and comments.
    """
    found = []
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            body = _decode(z.read(name))
            if name.endswith(".rels"):
                for m in re.finditer(r'Target="([^"]+)"\s+TargetMode="External"', body):
                    found.append((m.group(1), "ooxml_hyperlink"))
                for m in re.finditer(r'Target="((?:https?|ftp)://[^"]+)"', body):
                    found.append((m.group(1), "ooxml_hyperlink"))
            elif name.endswith(".xml") and not name.startswith("_rels"):
                text = re.sub(r"<[^>]+>", " ", body)
                found += [(u, "ooxml_text") for u in URL_RE.findall(text)]
                found += [(m.group(1), "ooxml_bare_domain")
                          for m in BARE_DOMAIN_RE.finditer(text)]
    return found


def from_pdf(path: Path) -> list:
    """Text-layer URLs, link annotations, and line-wrap repair."""
    try:
        import pymupdf
    except ImportError:
        sys.exit("PDF input needs pymupdf:  pip install pymupdf")
    found = []
    with pymupdf.open(path) as doc:
        for page in doc:
            for link in page.get_links():
                uri = link.get("uri", "")
                if uri:
                    found.append((uri, "pdf_annotation"))
            raw = page.get_text()
            # PDFs hard-wrap long URLs; rejoin before matching.
            joined = re.sub(r"([/\-_?=&.])\n\s*", r"\1", raw)
            joined = re.sub(r"\n\s*(?=[a-z0-9\-_%]{2,}[/.])", "", joined)
            found += [(u, "pdf_text") for u in URL_RE.findall(joined)]
            found += [(m.group(1), "pdf_bare_domain")
                      for m in BARE_DOMAIN_RE.finditer(joined)]
    return found


def from_text(path: Path) -> list:
    body = _decode(path.read_bytes())
    found = [(m.group(2), "md_link") for m in MD_LINK_RE.finditer(body)]
    found += [(u, "html_href") for u in HTML_HREF_RE.findall(body)]
    if path.suffix.lower() in {".html", ".htm", ".xml"}:
        body = re.sub(r"<[^>]+>", " ", body)
    found += [(u, "bare_url") for u in URL_RE.findall(body)]
    found += [(m.group(1), "bare_domain") for m in BARE_DOMAIN_RE.finditer(body)]
    return found

# --- normalising -------------------------------------------------------------


def canonicalize(url: str):
    """Return (canonical_url, added_scheme) or (None, False) if unusable."""
    url = url.strip().strip("<>").rstrip(").,;:!?'\"")
    if url.startswith("//"):
        url = "https:" + url
    added = False
    if not re.match(r"^[a-z][a-z0-9+.-]*://", url, re.I):
        if "." not in url.split("/")[0]:
            return None, False
        url = "https://" + url
        added = True
    p = urlparse(url)
    if not p.hostname or "." not in p.hostname:
        return None, False
    host = p.hostname.lower()
    if p.port and not ((p.scheme == "http" and p.port == 80) or
                       (p.scheme == "https" and p.port == 443)):
        host = f"{host}:{p.port}"
    q = [(k, v) for k, v in parse_qsl(p.query, keep_blank_values=True)
         if not k.lower().startswith(TRACKING_PREFIXES)]
    path = p.path.rstrip("/") or "/"
    return urlunparse((p.scheme.lower(), host, path, "", urlencode(q), "")), added


def classify(url: str) -> str:
    p = urlparse(url)
    host = (p.hostname or "").lower().removeprefix("www.")
    ext = Path(p.path).suffix.lower()
    if host in SHORTENERS:
        return "shortener"
    if host in SOCIAL_HOSTS or any(host.endswith("." + s) for s in SOCIAL_HOSTS):
        return "social"
    if ext in ASSET_EXT:
        return "asset"
    if ext in DOC_EXT:
        return "document"
    if p.path in ("", "/"):
        return "homepage"
    return "article"

# --- main --------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path)
    ap.add_argument("-o", "--output", type=Path, default=Path("inputs/urls.csv"))
    ap.add_argument("--drop-social", action="store_true")
    ap.add_argument("--drop-assets", action="store_true")
    ap.add_argument("--drop-homepages", action="store_true")
    ap.add_argument("--only-hosts", default=None, help="Comma-separated host allowlist")
    args = ap.parse_args()

    suffix = args.input.suffix.lower()
    if suffix in {".docx", ".dotx", ".pptx", ".xlsx"}:
        found = from_ooxml(args.input)
    elif suffix == ".pdf":
        found = from_pdf(args.input)
    else:
        found = from_text(args.input)

    merged, rejected = {}, 0
    for raw, how in found:
        canon, added = canonicalize(raw)
        if not canon:
            rejected += 1
            continue
        rec = merged.setdefault(canon, {"url": canon, "sources": set(),
                                        "hits": 0, "added_scheme": False})
        rec["sources"].add(how)
        rec["hits"] += 1
        rec["added_scheme"] |= added

    allow = {h.strip().lower() for h in args.only_hosts.split(",")} if args.only_hosts else None

    rows, skipped = [], []
    for rec in merged.values():
        kind = classify(rec["url"])
        host = urlparse(rec["url"]).hostname
        if allow and host not in allow and host.removeprefix("www.") not in allow:
            skipped.append((rec["url"], "not_in_allowlist")); continue
        if kind == "social" and args.drop_social:
            skipped.append((rec["url"], kind)); continue
        if kind == "asset" and args.drop_assets:
            skipped.append((rec["url"], kind)); continue
        if kind == "homepage" and args.drop_homepages:
            skipped.append((rec["url"], kind)); continue

        # Anything inferred rather than literally written deserves a human look.
        review = "yes" if (rec["added_scheme"] or "bare_domain" in "|".join(rec["sources"])
                           or kind in {"shortener", "homepage"}) else ""
        rows.append({
            "url": rec["url"], "host": host, "kind": kind, "hits": rec["hits"],
            "found_via": "|".join(sorted(rec["sources"])),
            "scheme_added": "yes" if rec["added_scheme"] else "",
            "review": review,
        })

    rows.sort(key=lambda r: (r["review"] == "", r["host"] or "", r["url"]))
    with args.output.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["url", "host", "kind", "hits",
                                           "found_via", "scheme_added", "review"])
        w.writeheader()
        w.writerows(rows)

    kinds = {}
    for r in rows:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
    print(f"Raw matches:        {len(found)}")
    print(f"Unrecognised:       {rejected}")
    print(f"Unique URLs:        {len(merged)}")
    print(f"Kept:               {len(rows)}  across {len({r['host'] for r in rows})} hosts")
    print(f"Skipped by flags:   {len(skipped)}")
    for k, n in sorted(kinds.items(), key=lambda x: -x[1]):
        print(f"   {k:10} {n}")
    flagged = sum(1 for r in rows if r["review"])
    print(f"Flagged for review: {flagged}  (sorted to the top of the CSV)")
    print(f"-> {args.output}")


if __name__ == "__main__":
    main()
