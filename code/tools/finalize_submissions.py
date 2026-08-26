#!/usr/bin/env python3
"""收檔：把正式 run 的產物改成 Evalbase 要的身分，再過官方格式驗證。

    python3 tools/finalize_submissions.py --r <r.tsv> --rag <rag.jsonl> --outdir submission_official

改兩件事（都是身分欄位，不動內容）：
  1. team_id: pi-serini（baseline 程式碼繼承來的）→ "2026 cfda rag"（Evalbase 註冊群組，
     見 8/8 螢幕截圖：ir.nist.gov/evalbase/org/trec-2026/2026 cfda rag）
  2. run_id / 第 6 欄 → 與 Evalbase 表單的 Runtag 一致（R: cfda-vfs-deep；RAG: cfda-w5c）

⚠️ metadata.run_id 與 runfile 第 6 欄一定要跟表單 Runtag 完全相同 —— 兩邊對不上
是 TREC 常見的退件原因。
"""
import argparse
import json
import os
import subprocess
import sys

TEAM = "2026 cfda rag"

ap = argparse.ArgumentParser()
ap.add_argument("--r", dest="r_path", help="R runfile（可略）")
ap.add_argument("--rag", dest="rag_path", help="RAG jsonl（可略）")
ap.add_argument("--r-tag", default="cfda-vfs-deep")
ap.add_argument("--rag-tag", default="cfda-w5c")
ap.add_argument("--outdir", default="submission_official")
ap.add_argument(
    "--topics",
    default=os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "..", "data", "topics", "trec_rag_2026_queries.tsv"
    )),
)


def main():
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    outs = {}

    if a.r_path:
        out = os.path.join(a.outdir, "r_output_trec_rag_2026.tsv")
        n = 0
        with open(out, "w") as w:
            for line in open(a.r_path):
                p = line.split()
                if len(p) != 6:
                    sys.exit(f"✗ R 檔第 {n+1} 行不是 6 欄")
                p[5] = a.r_tag
                w.write(" ".join(p) + "\n")
                n += 1
        print(f"R  → {out}（{n} 列，run_id={a.r_tag}）")
        outs["r"] = out

    if a.rag_path:
        out = os.path.join(a.outdir, "rag_output_trec_rag_2026.jsonl")
        n = 0
        with open(out, "w", encoding="utf-8") as w:
            for line in open(a.rag_path, encoding="utf-8"):
                if not line.strip():
                    continue
                d = json.loads(line)
                d["metadata"]["team_id"] = TEAM
                d["metadata"]["run_id"] = a.rag_tag
                w.write(json.dumps(d, ensure_ascii=False) + "\n")
                n += 1
        print(f"RAG → {out}（{n} 題，team_id={TEAM!r}，run_id={a.rag_tag}）")
        outs["rag"] = out

    if "r" in outs and "rag" in outs:
        print("\n── 官方格式驗證 ──")
        rc = subprocess.call([sys.executable, "tools/official_format_check.py",
                              outs["r"], outs["rag"], a.topics])
        sys.exit(rc)


if __name__ == "__main__":
    main()
