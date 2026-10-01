"""
Step 4: documents.jsonl -> chunks.jsonl

Cuts each article into passages of roughly 300 words, split on sentence
boundaries so no passage ends mid-sentence.

Every chunk stores start_char/end_char into the document's normalized_text, and
the chunk text is sliced FROM that text rather than rebuilt. That guarantees
offsets are exact -- which is what makes quote verification possible in step 5.

Usage:
    python chunk.py --corpus ./corpus
    python chunk.py --corpus ./corpus --words 250 --overlap 0.15
"""

import argparse
import json
import re
from pathlib import Path

# Sentence boundary: terminal punctuation + whitespace + capital/quote/digit.
# The negative lookbehind protects common abbreviations from being split.
ABBREV = r"(?<!\bMr)(?<!\bMrs)(?<!\bMs)(?<!\bDr)(?<!\bProf)(?<!\bSt)(?<!\bvs)" \
         r"(?<!\be\.g)(?<!\bi\.e)(?<!\betc)(?<!\bFig)(?<!\bNo)(?<!\bpp)" \
         r"(?<!\bJan)(?<!\bFeb)(?<!\bMar)(?<!\bApr)(?<!\bJun)(?<!\bJul)" \
         r"(?<!\bAug)(?<!\bSep)(?<!\bSept)(?<!\bOct)(?<!\bNov)(?<!\bDec)"
SENT_END = re.compile(ABBREV + r'(?<=[.!?])["\')\]]*\s+(?=["\'(\[]?[A-Z0-9])')

# A line that is short, unpunctuated and standalone is a heading in the
# extracted text. Used only as a fallback when the parsed heading list misses.
MIN_CHUNK_CHARS = 120


def split_sentences(text: str):
    """Return [(start, end)] sentence spans. Also breaks on blank lines so a
    paragraph boundary never gets swallowed into the middle of a chunk."""
    spans = []
    for para in re.finditer(r"[^\n]+(?:\n(?!\n)[^\n]+)*", text):
        p_start, p_text = para.start(), para.group()
        cursor = 0
        for m in SENT_END.finditer(p_text):
            end = m.start() + 1
            while end < len(p_text) and p_text[end] in "\"')]":
                end += 1
            if end > cursor:
                spans.append((p_start + cursor, p_start + end))
            cursor = m.end()
        if cursor < len(p_text):
            spans.append((p_start + cursor, p_start + len(p_text)))
    return [(s, e) for s, e in spans if text[s:e].strip()]


def locate_headings(text: str, headings: list):
    """Find where each parsed heading sits in the extracted text.

    parse.py records heading TEXT from the html tags but not its position,
    because trafilatura may reorder or drop lines. So search for it here and
    keep only the ones actually present.
    """
    located = []
    cursor = 0
    for h in headings:
        needle = h.get("text", "").strip()
        if len(needle) < 3:
            continue
        idx = text.find(needle, cursor)
        if idx == -1:
            idx = text.find(needle)
        if idx != -1:
            located.append({"offset": idx, "level": h.get("level", 2), "text": needle})
            cursor = max(cursor, idx + len(needle))
    return sorted(located, key=lambda x: x["offset"])


def heading_for(offset: int, located: list):
    """Nearest heading at or before this offset, plus its parent chain."""
    path, current = [], None
    for h in located:
        if h["offset"] > offset:
            break
        current = h
        path = [x for x in path if x["level"] < h["level"]] + [h]
    return ([p["text"] for p in path], current["text"] if current else None)


def chunk_document(doc: dict, target_words: int, overlap_ratio: float):
    text = doc["normalized_text"]
    spans = split_sentences(text)
    if not spans:
        return []

    located = locate_headings(text, doc.get("headings", []))
    overlap_words = max(0, int(target_words * overlap_ratio))

    chunks, current, current_words = [], [], 0
    idx = 0

    def flush(sent_spans):
        nonlocal idx
        if not sent_spans:
            return
        start, end = sent_spans[0][0], sent_spans[-1][1]
        body = text[start:end]
        if len(body.strip()) < MIN_CHUNK_CHARS and chunks:
            # Too small to stand alone -- extend the previous chunk instead.
            prev = chunks[-1]
            prev["end_char"] = end
            prev["text"] = text[prev["start_char"]:end]
            prev["word_count"] = len(prev["text"].split())
            return
        path, nearest = heading_for(start, located)
        chunks.append({
            "chunk_id": f"{doc['doc_id']}_c{idx:03d}",
            "doc_id": doc["doc_id"],
            "doc_version": doc.get("doc_version", 1),
            "seq": idx,
            "start_char": start,
            "end_char": end,
            "text": body,
            "word_count": len(body.split()),
            "heading": nearest,
            "heading_path": path,
            "title": doc.get("title"),
            "author": doc.get("author"),
            "published_at": doc.get("published_at"),
            "host": doc.get("host"),
            "url": doc.get("url"),
        })
        idx += 1

    for span in spans:
        words = len(text[span[0]:span[1]].split())
        if current_words + words > target_words and current:
            flush(current)
            # Carry the tail forward so a claim spanning a boundary survives.
            tail, tail_words = [], 0
            for s in reversed(current):
                w = len(text[s[0]:s[1]].split())
                if tail_words + w > overlap_words:
                    break
                tail.insert(0, s)
                tail_words += w
            current, current_words = tail, tail_words
        current.append(span)
        current_words += words

    flush(current)
    return chunks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, default=Path("./corpus"))
    ap.add_argument("--words", type=int, default=300, help="Target words per chunk")
    ap.add_argument("--overlap", type=float, default=0.15, help="Overlap as fraction of target")
    args = ap.parse_args()

    docs_path = args.corpus / "documents.jsonl"
    if not docs_path.exists():
        raise SystemExit(f"No documents.jsonl at {docs_path}. Run parse.py first.")

    docs = [json.loads(l) for l in docs_path.open(encoding="utf-8") if l.strip()]
    print(f"{len(docs)} documents")

    out_path = args.corpus / "chunks.jsonl"
    all_chunks, bad_offsets = [], 0

    for doc in docs:
        for ch in chunk_document(doc, args.words, args.overlap):
            # Self-check: the stored text must equal the slice it claims to be.
            if doc["normalized_text"][ch["start_char"]:ch["end_char"]] != ch["text"]:
                bad_offsets += 1
                continue
            all_chunks.append(ch)

    with out_path.open("w", encoding="utf-8") as fh:
        for ch in all_chunks:
            fh.write(json.dumps(ch, ensure_ascii=False) + "\n")

    counts = [c["word_count"] for c in all_chunks]
    with_heading = sum(1 for c in all_chunks if c["heading"])
    report = {
        "documents": len(docs),
        "chunks": len(all_chunks),
        "offset_mismatches": bad_offsets,
        "chunks_per_doc_avg": round(len(all_chunks) / max(1, len(docs)), 1),
        "words_min": min(counts) if counts else 0,
        "words_median": sorted(counts)[len(counts) // 2] if counts else 0,
        "words_max": max(counts) if counts else 0,
        "with_heading_pct": round(100 * with_heading / max(1, len(all_chunks)), 1),
    }
    (args.corpus / "chunk_report.json").write_text(json.dumps(report, indent=2))

    print(f"Chunks:           {len(all_chunks)}  ({report['chunks_per_doc_avg']} per doc)")
    print(f"Words per chunk:  min {report['words_min']} / median {report['words_median']} / max {report['words_max']}")
    print(f"With heading:     {report['with_heading_pct']}%")
    print(f"Offset mismatches:{bad_offsets}   (must be 0)")
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
