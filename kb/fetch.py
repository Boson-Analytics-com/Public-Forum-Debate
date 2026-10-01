"""
Stage 1: fetch each URL and save the RAW BYTES. No text extraction here.

Fetching is the expensive, rate-limited, irreversible step -- pages change,
paywalls tighten, sites go down. Parsing is cheap and you will redo it many
times as your extraction improves. So this script does nothing but fetch and
archive; parse.py turns the archive into text.

Output:
    raw/<sha1-of-url>.html.gz   original response bytes, untouched
    manifest.jsonl              one append-only record per URL

Resumable: re-running skips URLs already recorded as ok in the manifest.

Usage:
    python fetch.py urls.csv --outdir ./corpus
    python fetch.py urls.csv --outdir ./corpus --delay 2.0 --workers 4
"""

import argparse
import csv
import gzip
import hashlib
import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import requests

UA = ("Mozilla/5.0 (compatible; DebateKB/1.0; research corpus builder; "
      "+contact: you@example.com)")

RETRY_STATUS = {429, 500, 502, 503, 504}
MAX_BYTES = 8 * 1024 * 1024


class HostThrottle:
    """Per-host delay. Concurrency is across hosts, never within one."""

    def __init__(self, delay: float):
        self.delay = delay
        self.last = {}
        self.lock = threading.Lock()

    def wait(self, host: str):
        with self.lock:
            gap = time.monotonic() - self.last.get(host, 0.0)
            sleep = max(0.0, self.delay - gap)
            self.last[host] = time.monotonic() + sleep
        if sleep:
            time.sleep(sleep + random.uniform(0, 0.3))


class RobotsCache:
    def __init__(self, session, respect: bool):
        self.session, self.respect = session, respect
        self.cache, self.lock = {}, threading.Lock()

    def allowed(self, url: str) -> bool:
        if not self.respect:
            return True
        p = urlparse(url)
        base = f"{p.scheme}://{p.netloc}"
        with self.lock:
            rp = self.cache.get(base)
        if rp is None:
            rp = RobotFileParser()
            try:
                r = self.session.get(base + "/robots.txt", timeout=10)
                rp.parse(r.text.splitlines() if r.status_code == 200 else [])
            except Exception:
                rp.parse([])
            with self.lock:
                self.cache[base] = rp
        return rp.can_fetch(UA, url)


def url_key(url: str) -> str:
    return hashlib.sha1(url.encode()).hexdigest()


def load_done(manifest: Path) -> set:
    if not manifest.exists():
        return set()
    done = set()
    with manifest.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("status") == "ok":
                done.add(rec["url"])
    return done


def fetch_one(url, session, throttle, robots, rawdir, attempts):
    host = urlparse(url).hostname or ""
    base = {"url": url, "url_key": url_key(url), "host": host,
            "fetched_at": datetime.now(timezone.utc).isoformat()}

    if not robots.allowed(url):
        return {**base, "status": "blocked_by_robots"}

    last_err = None
    for attempt in range(attempts):
        throttle.wait(host)
        try:
            r = session.get(url, timeout=(10, 30), allow_redirects=True,
                            headers={"Accept": "text/html,application/xhtml+xml"})
            if r.status_code in RETRY_STATUS and attempt < attempts - 1:
                retry_after = r.headers.get("Retry-After")
                wait = float(retry_after) if (retry_after or "").isdigit() else 2 ** attempt * 2
                time.sleep(min(wait, 60))
                last_err = f"http_{r.status_code}"
                continue
            if r.status_code != 200:
                return {**base, "status": "http_error", "http_status": r.status_code}

            body = r.content[:MAX_BYTES]
            path = rawdir / f"{url_key(url)}.html.gz"
            with gzip.open(path, "wb") as fh:
                fh.write(body)

            return {**base,
                    "status": "ok",
                    "http_status": r.status_code,
                    "final_url": r.url,
                    "redirected": r.url.rstrip("/") != url.rstrip("/"),
                    "content_type": r.headers.get("Content-Type", ""),
                    "declared_encoding": r.encoding,
                    "apparent_encoding": r.apparent_encoding,
                    "last_modified": r.headers.get("Last-Modified"),
                    "etag": r.headers.get("ETag"),
                    "bytes": len(body),
                    "truncated": len(r.content) > MAX_BYTES,
                    "body_sha256": hashlib.sha256(body).hexdigest(),
                    "raw_path": str(path),
                    }
        except requests.RequestException as e:
            last_err = type(e).__name__
            if attempt < attempts - 1:
                time.sleep(2 ** attempt * 2)

    return {**base, "status": "failed", "error": last_err}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("urls", type=Path, help="CSV from extract_urls.py, or one URL per line")
    ap.add_argument("--outdir", type=Path, default=Path("./corpus"))
    ap.add_argument("--delay", type=float, default=1.5, help="Seconds between hits on the SAME host")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--attempts", type=int, default=3)
    ap.add_argument("--no-robots", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    rawdir = args.outdir / "raw"
    rawdir.mkdir(parents=True, exist_ok=True)
    manifest = args.outdir / "manifest.jsonl"

    urls = []
    with args.urls.open(newline="", encoding="utf-8") as fh:
        sample = fh.read(2048)
        fh.seek(0)
        if "," in sample and "url" in sample.split("\n")[0].lower():
            for row in csv.DictReader(fh):
                if row.get("url"):
                    urls.append(row["url"].strip())
        else:
            urls = [ln.strip() for ln in fh if ln.strip().startswith("http")]

    urls = list(dict.fromkeys(urls))
    done = load_done(manifest)
    todo = [u for u in urls if u not in done]
    if args.limit:
        todo = todo[:args.limit]

    print(f"{len(urls)} urls | {len(done)} already fetched | {len(todo)} to go")
    if not todo:
        return

    session = requests.Session()
    session.headers["User-Agent"] = UA
    throttle = HostThrottle(args.delay)
    robots = RobotsCache(session, respect=not args.no_robots)

    lock = threading.Lock()
    counts = {}
    with manifest.open("a", encoding="utf-8") as out:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(fetch_one, u, session, throttle, robots, rawdir, args.attempts)
                       for u in todo]
            for i, fut in enumerate(futures, 1):
                rec = fut.result()
                with lock:
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    out.flush()
                    counts[rec["status"]] = counts.get(rec["status"], 0) + 1
                if i % 10 == 0 or i == len(futures):
                    print(f"  {i}/{len(futures)}  {counts}")

    print(f"\n{counts}")
    print(f"-> {rawdir}/  ({len(list(rawdir.glob('*.gz')))} archived pages)")
    print(f"-> {manifest}")
    bad = [s for s in counts if s != "ok"]
    if bad:
        print(f"Review non-ok records in the manifest: {bad}")


if __name__ == "__main__":
    main()
