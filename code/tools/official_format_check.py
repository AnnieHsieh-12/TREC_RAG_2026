#!/usr/bin/env python3
"""Validate Retrieval and RAG outputs against organizer format rules.

Usage: ``python3 tools/official_format_check.py retrieval.tsv rag.jsonl topics.tsv``
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
            errs.append(f"L{ln}: {len(parts)} 欄（規定恰好 6 欄）")
            continue
        tid, q0, docid, rank, score, run_id = parts
        # Q0: "fixed string"
        if q0 != "Q0":
            errs.append(f"L{ln}: 第 2 欄是 {q0!r} 不是 Q0")
        # "Do not emit MS MARCO segment IDs."
        if docid.startswith("msmarco"):
            errs.append(f"L{ln}: MS MARCO id {docid}")
        if not is_climbmix_docid(docid):
            errs.append(f"L{ln}: 非 ClimbMix docid：{docid}")
        try:
            parsed_rank, parsed_score = int(rank), float(score)
        except ValueError:
            errs.append(f"L{ln}: rank/score 不是合法數字：{rank!r}/{score!r}")
            continue
        rows_by_topic.setdefault(tid, []).append((parsed_rank, parsed_score, docid, run_id))

    # "topic_id ... preserve exactly"
    missing = topic_ids - set(rows_by_topic)
    extra = set(rows_by_topic) - topic_ids
    if missing:
        errs.append(f"缺 {len(missing)} 題：{sorted(missing)[:5]} …")
    if extra:
        errs.append(f"多出不在題目檔的 topic_id：{sorted(extra)[:5]}")

    ks, run_ids = [], set()
    for tid, rows in rows_by_topic.items():
        rows_sorted = sorted(rows)
        # "Retrieval ranks must restart at 1 for each narrative." + 連號
        ranks = [r[0] for r in rows_sorted]
        if ranks != list(range(1, len(ranks) + 1)):
            errs.append(f"{tid}: rank 不是從 1 連號（{ranks[:3]}…）")
        # "Keep scores non-increasing within each narrative."
        scores = [r[1] for r in rows_sorted]
        if any(b > a for a, b in zip(scores, scores[1:])):
            errs.append(f"{tid}: score 有遞增")
        # 同題不得重複 docid
        docids = [r[2] for r in rows_sorted]
        if len(set(docids)) != len(docids):
            errs.append(f"{tid}: docid 重複")
        run_ids.update(r[3] for r in rows)
        ks.append(len(rows))

    # "stable identifier for the submitted run"
    if len(run_ids) > 1:
        errs.append(f"run_id 不一致：{sorted(run_ids)}")
    # "Do not add documents merely to reach a conventional cutoff such as 10, 100, or 1000."
    if ks:
        from collections import Counter
        kc = Counter(ks)
        distinct = len(kc)
        conv = sum(v for k, v in kc.items() if k in (10, 100, 1000))
        if distinct == 1:
            # The organizer forbids padding merely to reach a conventional
            # cutoff; equal depths alone do not prove that intent.
            warns.append(f"全部 {len(ks)} 題同一個 k={ks[0]} —— 請人工確認不是補滿慣用 cutoff")
        elif conv > len(ks) * 0.5:
            warns.append(f"{conv}/{len(ks)} 題的 k 落在慣用 cutoff（10/100/1000）—— 會被質疑補滿")
        print(f"  k：{min(ks)}–{max(ks)}，相異 {distinct} 種，中位 {sorted(ks)[len(ks)//2]}")
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
            errs.append(f"L{ln}: JSON 壞掉 {e}")
            continue
        # "Every object MUST have metadata, references, answer"
        for k in ("metadata", "references", "answer"):
            if k not in d:
                errs.append(f"L{ln}: 缺 {k}")
        md = d.get("metadata", {})
        # 官方必填：team_id, narrative_id, narrative, run_id, run_desc
        for k in ("team_id", "narrative_id", "narrative", "run_id", "run_desc"):
            if not md.get(k):
                errs.append(f"L{ln}: metadata.{k} 缺或空")
        nid = md.get("narrative_id")
        if nid in seen:
            errs.append(f"L{ln}: narrative_id {nid} 重複（規定每題恰好一個物件）")
        seen.add(nid)
        # "narrative: exact copy from second column"
        if nid in topics and md.get("narrative") != topics[nid]:
            errs.append(f"L{ln}: narrative 與題目檔不逐字相同（{nid}）")
        refs = d.get("references", [])
        if not isinstance(refs, list) or any(not isinstance(r, str) for r in refs):
            errs.append(f"L{ln}: references 必須是 docid 字串列表")
            refs = []
        else:
            for ri, ref in enumerate(refs):
                if not is_climbmix_docid(ref):
                    errs.append(f"L{ln}: references[{ri}] 非 ClimbMix docid：{ref!r}")
            if len(refs) != len(set(refs)):
                errs.append(f"L{ln}: references 有重複 docid")
        words = 0
        answer = d.get("answer", [])
        if not isinstance(answer, list) or not answer:
            errs.append(f"L{ln}: answer 必須是非空陣列")
            answer = []
        for si, s in enumerate(answer):
            if not isinstance(s, dict):
                errs.append(f"L{ln}: answer[{si}] 必須是物件")
                continue
            t = s.get("text", "")
            if not isinstance(t, str) or not t.strip():
                errs.append(f"L{ln}: answer[{si}].text 空")
            words += len(t.split())
            cits = s.get("citations", [])
            # "array of zero to three citations"
            if len(cits) > 3:
                errs.append(f"L{ln}: answer[{si}] 有 {len(cits)} 個引用（上限 3）")
            for c in cits:
                # "MUST reference valid references entries"
                if isinstance(c, int):
                    if c < 0 or c >= len(refs):
                        errs.append(f"L{ln}: answer[{si}] 引用索引 {c} 越界")
                elif isinstance(c, str):
                    if c not in refs:
                        errs.append(f"L{ln}: answer[{si}] 引用 docid 不在 references")
                else:
                    errs.append(f"L{ln}: answer[{si}] 引用型別怪異 {type(c)}")
        # "sum(len(item['text'].split()) for item in answer) <= 1024"
        if words > 1024:
            errs.append(f"L{ln}: {words} 詞 > 1024（{nid}）")
    missing = set(topics) - seen
    if missing:
        errs.append(f"缺 {len(missing)} 題：{sorted(missing)[:5]} …")
    return errs, warns


def main():
    r_path, rag_path, topics_path = sys.argv[1:4]
    topics = {}
    for line in Path(topics_path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            qid, narrative = line.split("\t", 1)
            topics[qid] = narrative

    bad = 0
    for name, fn, path in (("R", check_r, r_path), ("RAG", check_rag, rag_path)):
        print(f"== {name}：{path}")
        errs, warns = fn(path, set(topics) if name == "R" else topics)
        for e in errs[:15]:
            print("  ✗", e)
        for w in warns:
            print("  ⚠", w)
        more = len(errs) - 15
        if more > 0:
            print(f"  …還有 {more} 條錯誤")
        print(f"  → {'合規' if not errs else f'{len(errs)} 條違規'}，{len(warns)} 條警告")
        bad += len(errs)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
