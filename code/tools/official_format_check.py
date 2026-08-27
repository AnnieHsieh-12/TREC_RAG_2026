#!/usr/bin/env python3
"""Validate Retrieval and RAG outputs against organizer format rules.

Usage:
  ``python3 tools/official_format_check.py retrieval.tsv rag.jsonl topics.tsv``
  ``python3 tools/official_format_check.py --rag-only rag.jsonl topics.tsv``
"""
import json
from pathlib import Path
import re
import sys

# The 2026 ClimbMix corpus uses exact document ids such as
# ``shard_00459_61697``.  Keep one canonical predicate and apply it to both
# Retrieval rows and RAG references.
CLIMBMIX_RE = re.compile(r"^shard_\d+_\d+$")


def is_climbmix_docid(value):
    return isinstance(value, str) and CLIMBMIX_RE.fullmatch(value) is not None


def check_r(path, topic_ids):
    errs, warns = [], []
    rows_by_topic = {}
    for ln, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        # "exactly six whitespace-separated columns per line"
        parts = line.split()
        if len(parts) != 6:
            errs.append(f"L{ln}: found {len(parts)} columns; expected exactly 6")
            continue
        tid, q0, docid, rank, score, run_id = parts
        # Q0: "fixed string"
        if q0 != "Q0":
            errs.append(f"L{ln}: column 2 is {q0!r}, expected Q0")
        # "Do not emit MS MARCO segment IDs."
        if docid.startswith("msmarco"):
            errs.append(f"L{ln}: MS MARCO id {docid}")
        if not is_climbmix_docid(docid):
            errs.append(f"L{ln}: invalid ClimbMix docid: {docid}")
        try:
            parsed_rank, parsed_score = int(rank), float(score)
        except ValueError:
            errs.append(f"L{ln}: invalid numeric rank/score: {rank!r}/{score!r}")
            continue
        rows_by_topic.setdefault(tid, []).append((parsed_rank, parsed_score, docid, run_id))

    # "topic_id ... preserve exactly"
    missing = topic_ids - set(rows_by_topic)
    extra = set(rows_by_topic) - topic_ids
    if missing:
        errs.append(f"missing {len(missing)} topics: {sorted(missing)[:5]} ...")
    if extra:
        errs.append(f"topic IDs absent from topics file: {sorted(extra)[:5]}")

    ks, run_ids = [], set()
    for tid, rows in rows_by_topic.items():
        rows_sorted = sorted(rows)
        # Retrieval ranks must restart at 1 and remain contiguous.
        ranks = [r[0] for r in rows_sorted]
        if ranks != list(range(1, len(ranks) + 1)):
            errs.append(f"{tid}: ranks are not contiguous from 1 ({ranks[:3]}...)")
        # "Keep scores non-increasing within each narrative."
        scores = [r[1] for r in rows_sorted]
        if any(b > a for a, b in zip(scores, scores[1:])):
            errs.append(f"{tid}: scores are not non-increasing")
        # Document IDs must be unique within a topic.
        docids = [r[2] for r in rows_sorted]
        if len(set(docids)) != len(docids):
            errs.append(f"{tid}: duplicate docid")
        run_ids.update(r[3] for r in rows)
        ks.append(len(rows))

    # "stable identifier for the submitted run"
    if len(run_ids) > 1:
        errs.append(f"inconsistent run_id values: {sorted(run_ids)}")
    # "Do not add documents merely to reach a conventional cutoff such as 10, 100, or 1000."
    if ks:
        from collections import Counter
        kc = Counter(ks)
        distinct = len(kc)
        conv = sum(v for k, v in kc.items() if k in (10, 100, 1000))
        if distinct == 1:
            # The organizer forbids padding merely to reach a conventional
            # cutoff; equal depths alone do not prove that intent.
            warns.append(f"all {len(ks)} topics use k={ks[0]}; confirm this is not cutoff padding")
        elif conv > len(ks) * 0.5:
            warns.append(f"{conv}/{len(ks)} topics use conventional cutoffs 10/100/1000; verify no padding")
        print(f"  k: {min(ks)}-{max(ks)}, {distinct} distinct, median {sorted(ks)[len(ks)//2]}")
    return errs, warns


def check_rag(path, topics):
    errs, warns = [], []
    seen = set()
    for ln, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            d = json.loads(line)          # "File MUST be valid JSONL"
        except json.JSONDecodeError as e:
            errs.append(f"L{ln}: invalid JSON: {e}")
            continue
        # "Every object MUST have metadata, references, answer"
        for k in ("metadata", "references", "answer"):
            if k not in d:
                errs.append(f"L{ln}: missing {k}")
        md = d.get("metadata", {})
        # Organizer-required metadata fields.
        for k in ("team_id", "narrative_id", "narrative", "run_id", "run_desc"):
            if not md.get(k):
                errs.append(f"L{ln}: metadata.{k} is missing or empty")
        nid = md.get("narrative_id")
        if nid in seen:
            errs.append(f"L{ln}: duplicate narrative_id {nid}")
        seen.add(nid)
        # "narrative: exact copy from second column"
        if nid in topics and md.get("narrative") != topics[nid]:
            errs.append(f"L{ln}: narrative is not an exact topics-file copy ({nid})")
        refs = d.get("references", [])
        if not isinstance(refs, list) or any(not isinstance(r, str) for r in refs):
            errs.append(f"L{ln}: references must be a list of docid strings")
            refs = []
        else:
            for ri, ref in enumerate(refs):
                if not is_climbmix_docid(ref):
                    errs.append(f"L{ln}: references[{ri}] is not a ClimbMix docid: {ref!r}")
            if len(refs) != len(set(refs)):
                errs.append(f"L{ln}: references contains duplicate docids")
        words = 0
        answer = d.get("answer", [])
        if not isinstance(answer, list) or not answer:
            errs.append(f"L{ln}: answer must be a non-empty list")
            answer = []
        for si, s in enumerate(answer):
            if not isinstance(s, dict):
                errs.append(f"L{ln}: answer[{si}] must be an object")
                continue
            t = s.get("text", "")
            if not isinstance(t, str) or not t.strip():
                errs.append(f"L{ln}: answer[{si}].text is empty")
            words += len(t.split())
            cits = s.get("citations", [])
            # "array of zero to three citations"
            if len(cits) > 3:
                errs.append(f"L{ln}: answer[{si}] has {len(cits)} citations; maximum is 3")
            for c in cits:
                # "MUST reference valid references entries"
                if isinstance(c, int):
                    if c < 0 or c >= len(refs):
                        errs.append(f"L{ln}: answer[{si}] citation index {c} is out of range")
                elif isinstance(c, str):
                    if c not in refs:
                        errs.append(f"L{ln}: answer[{si}] citation docid is absent from references")
                else:
                    errs.append(f"L{ln}: answer[{si}] has invalid citation type {type(c)}")
        # "sum(len(item['text'].split()) for item in answer) <= 1024"
        if words > 1024:
            errs.append(f"L{ln}: {words} words exceeds 1024 ({nid})")
    missing = set(topics) - seen
    if missing:
        errs.append(f"missing {len(missing)} topics: {sorted(missing)[:5]} ...")
    extra = seen - set(topics)
    if extra:
        errs.append(f"topic IDs absent from topics file: {sorted(extra)[:5]}")
    return errs, warns


def load_topics(topics_path):
    topics = {}
    for line in Path(topics_path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            qid, narrative = line.split("\t", 1)
            topics[qid] = narrative
    return topics


def report(name, fn, path, expected):
    print(f"== {name}: {path}")
    errs, warns = fn(path, expected)
    for error in errs[:15]:
        print("  ✗", error)
    for warning in warns:
        print("  ⚠", warning)
    more = len(errs) - 15
    if more > 0:
        print(f"  ... {more} additional errors")
    status = "compliant" if not errs else f"{len(errs)} violations"
    print(f"  -> {status}; {len(warns)} warnings")
    return len(errs)


def main():
    if len(sys.argv) == 4 and sys.argv[1] == "--rag-only":
        rag_path, topics_path = sys.argv[2:4]
        topics = load_topics(topics_path)
        sys.exit(1 if report("RAG", check_rag, rag_path, topics) else 0)
    if len(sys.argv) != 4:
        raise SystemExit(
            "Usage: official_format_check.py retrieval.tsv rag.jsonl topics.tsv\n"
            "   or: official_format_check.py --rag-only rag.jsonl topics.tsv"
        )

    r_path, rag_path, topics_path = sys.argv[1:4]
    topics = load_topics(topics_path)
    bad = report("R", check_r, r_path, set(topics))
    bad += report("RAG", check_rag, rag_path, topics)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
