#!/usr/bin/env python3
"""Validate exact checklist coverage before the RAG launcher creates shards."""

import argparse
import json
from pathlib import Path


def load_topic_ids(path: Path) -> list[str]:
    topic_ids = []
    seen = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        columns = line.split("\t")
        if len(columns) != 2 or not columns[0] or not columns[1]:
            raise ValueError(f"Malformed topics row {line_number}")
        qid = columns[0]
        if qid in seen:
            raise ValueError(f"Duplicate topic_id in topics: {qid}")
        seen.add(qid)
        topic_ids.append(qid)
    if not topic_ids:
        raise ValueError("Topics file is empty")
    return topic_ids


def validate(topics_path: Path, checklist_path: Path) -> None:
    topic_ids = load_topic_ids(topics_path)
    checklist_ids = set()
    for line_number, line in enumerate(
        checklist_path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid checklist JSON on line {line_number}: {error}")
        if not isinstance(row, dict):
            raise ValueError(f"Checklist line {line_number} must be an object")
        qid = row.get("qid")
        items = row.get("items")
        if not isinstance(qid, str) or not qid:
            raise ValueError(f"Checklist line {line_number} has an invalid qid")
        if qid in checklist_ids:
            raise ValueError(f"Duplicate checklist qid: {qid}")
        if not isinstance(items, list) or not items or any(
            not isinstance(item, str) or not item.strip() for item in items
        ):
            raise ValueError(
                f"Checklist items for {qid} must be a non-empty array of non-empty strings"
            )
        checklist_ids.add(qid)

    expected = set(topic_ids)
    missing = expected - checklist_ids
    extra = checklist_ids - expected
    if missing:
        raise ValueError(f"Checklist is missing topic IDs: {sorted(missing)[:5]}")
    if extra:
        raise ValueError(f"Checklist has topic IDs absent from topics: {sorted(extra)[:5]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("topics", type=Path)
    parser.add_argument("checklist", type=Path)
    args = parser.parse_args()
    try:
        validate(args.topics, args.checklist)
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
