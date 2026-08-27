#!/usr/bin/env bash
# Canonical output-format gate. It validates Retrieval and RAG in one process
# and deliberately propagates every non-zero exit status.
set -euo pipefail

CODE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 <retrieval.tsv> <rag.jsonl> <topics.tsv>" >&2
  exit 2
fi

R_OUTPUT="$1"
RAG_OUTPUT="$2"
TOPICS="$3"

for required in "$R_OUTPUT" "$RAG_OUTPUT" "$TOPICS"; do
  if [ ! -s "$required" ]; then
    echo "missing validation input: $required" >&2
    exit 2
  fi
done

python3 "$CODE_ROOT/tools/official_format_check.py" "$R_OUTPUT" "$RAG_OUTPUT" "$TOPICS"
