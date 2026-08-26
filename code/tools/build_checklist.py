#!/usr/bin/env python3
"""Build the checklist JSONL consumed by the final RAG controller."""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SIDECAR_ROOT = REPO_ROOT / "sidecar"
sys.path.insert(0, str(SIDECAR_ROOT))

from src.checklist import generate_checklists_file  # noqa: E402
from src.common import load_nchc_key  # noqa: E402
from src.rag_tools import NCHCClient  # noqa: E402


def to_items(record: dict) -> dict:
    order: list[str] = []
    vital: set[str] = set()
    for nugget in record["nuggets"]:
        aspect = str(nugget["mapped_sub_narrative"]).strip().strip('"')
        if aspect not in order:
            order.append(aspect)
        if nugget.get("importance") == "vital":
            vital.add(aspect)
    return {
        "qid": str(record["qid"]),
        "items": [item + (" (vital)" if item in vital else "") for item in order],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topics", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--raw-output", type=Path)
    parser.add_argument("--model", default="gpt-oss-120b")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_output = args.raw_output or args.output.with_suffix(".raw.jsonl")
    topics: list[tuple[str, str]] = []
    with args.topics.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if line.strip():
                topics.append(tuple(line.split("\t", 1)))

    client = NCHCClient(load_nchc_key())
    generate_checklists_file(topics, str(raw_output), client, args.model)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with raw_output.open(encoding="utf-8") as source, args.output.open(
        "w", encoding="utf-8"
    ) as target:
        for line in source:
            if line.strip():
                target.write(json.dumps(to_items(json.loads(line)), ensure_ascii=False) + "\n")

    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
