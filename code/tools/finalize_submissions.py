#!/usr/bin/env python3
"""Apply the registered team and run identities to submission files."""

import argparse
import json
from pathlib import Path

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval", type=Path)
    parser.add_argument("--rag", type=Path)
    parser.add_argument("--retrieval-tag", default="cfda-vfs-deep")
    parser.add_argument("--team-id", default="cfda")
    parser.add_argument("--rag-tag", default="cfda-w5c")
    parser.add_argument("--outdir", required=True, type=Path)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    if args.retrieval:
        output = args.outdir / "r_output_trec_rag_2026.tsv"
        rows = 0
        with args.retrieval.open(encoding="utf-8") as source, output.open(
            "w", encoding="utf-8"
        ) as destination:
            for rows, line in enumerate(source, 1):
                fields = line.split()
                if len(fields) != 6:
                    raise SystemExit(f"Retrieval row {rows} does not have six fields")
                fields[5] = args.retrieval_tag
                destination.write(" ".join(fields) + "\n")
        print(f"Retrieval: {output} ({rows} rows, tag={args.retrieval_tag})")

    if args.rag:
        output = args.outdir / "rag_output_trec_rag_2026.jsonl"
        topics = 0
        with args.rag.open(encoding="utf-8") as source, output.open(
            "w", encoding="utf-8"
        ) as destination:
            for line in source:
                if not line.strip():
                    continue
                record = json.loads(line)
                record["metadata"]["team_id"] = args.team_id
                record["metadata"]["run_id"] = args.rag_tag
                destination.write(json.dumps(record, ensure_ascii=False) + "\n")
                topics += 1
        print(
            f"RAG: {output} ({topics} topics, team_id={args.team_id!r}, "
            f"run_id={args.rag_tag})"
        )


if __name__ == "__main__":
    main()
