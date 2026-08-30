#!/usr/bin/env python3
"""Build the checklist JSONL consumed by the final RAG controller."""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SIDECAR_ROOT = REPO_ROOT / "sidecar"
sys.path.insert(0, str(SIDECAR_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(SIDECAR_ROOT / ".env.local")

from src.checklist import generate_checklists_file  # noqa: E402
from src.common import load_nchc_key  # noqa: E402
from src.rag_tools import NCHCClient  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topics", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", default="gpt-oss-120b")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    topics: list[tuple[str, str]] = []
    with args.topics.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if line.strip():
                topics.append(tuple(line.split("\t", 1)))

    client = NCHCClient(load_nchc_key())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    generate_checklists_file(topics, str(args.output), client, args.model)

    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
