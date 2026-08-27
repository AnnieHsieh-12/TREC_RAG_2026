#!/usr/bin/env python3
"""Replay the heading repair used before the official W5c submission.

The frozen runtime output contained seven uncited heading-only answer items.
Each was merged into the immediately following cited answer item. This tool
preserves that documented finalization step without storing generated output.
"""

import argparse
import json
from pathlib import Path


def repair(record: dict) -> tuple[dict, int]:
    answer = record.get("answer", [])
    repaired: list[dict] = []
    merges = 0
    index = 0
    while index < len(answer):
        current = answer[index]
        if not current.get("citations") and index + 1 < len(answer):
            following = answer[index + 1]
            repaired.append(
                {
                    "text": f"{current['text']}: {following['text']}",
                    "citations": following["citations"],
                }
            )
            merges += 1
            index += 2
        else:
            repaired.append(current)
            index += 1
    record["answer"] = repaired
    return record, merges


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    topics = 0
    merges = 0
    with args.input.open(encoding="utf-8") as source, args.output.open(
        "w", encoding="utf-8"
    ) as destination:
        for line in source:
            if not line.strip():
                continue
            record, count = repair(json.loads(line))
            destination.write(json.dumps(record, ensure_ascii=False) + "\n")
            topics += 1
            merges += count
    print(f"Replayed {merges} heading merges across {topics} topics")


if __name__ == "__main__":
    main()
