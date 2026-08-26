#!/usr/bin/env bash
set -euo pipefail

# Final bounded RAG pipeline.
# Required inputs are supplied by environment variables so competition data
# and credentials never need to be committed to the repository.

CODE_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$CODE_ROOT"

if [[ -f .env.local ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env.local
  set +a
fi

: "${OPENAI_API_KEY:?Set OPENAI_API_KEY or create code/.env.local}"
: "${TOPICS:?Set TOPICS to the input topics TSV}"
: "${QRELS_DIR:?Set QRELS_DIR to the evaluation qrels directory}"
: "${CHECKLIST:?Set CHECKLIST to the topic checklist JSONL}"

for required in "$TOPICS" "$CHECKLIST"; do
  [[ -s "$required" ]] || { echo "Missing input: $required" >&2; exit 1; }
done
[[ -d "$QRELS_DIR" ]] || { echo "Missing qrels directory: $QRELS_DIR" >&2; exit 1; }

SHARDS="${SHARDS:-4}"
OUT="${OUT:-$CODE_ROOT/out/final-rag}"
RUN_ID="${RUN_ID:-cfda-final-rag}"
TEAM_ID="${TEAM_ID:-cfda}"
SIDECAR_URLS="${SIDECAR_URLS:-http://127.0.0.1:8765}"

mkdir -p "$OUT/.shards"

IFS=',' read -ra TOKENS <<< "${PYSERINI_TOKENS:-}"
TOKEN_COUNT=${#TOKENS[@]}
IFS=',' read -ra SIDECARS <<< "$SIDECAR_URLS"
SIDECAR_COUNT=${#SIDECARS[@]}

run_entry() {
  local topics_file="$1"
  local log_file="$2"
  local shard_id="${3:-0}"
  local token_args=()

  if [[ "$TOKEN_COUNT" -ge 1 && -n "${TOKENS[0]:-}" ]]; then
    export "CFDA_PYSERINI_TOKEN_$shard_id"="${TOKENS[$((shard_id % TOKEN_COUNT))]}"
    token_args=(--pyserini-token-env "CFDA_PYSERINI_TOKEN_$shard_id")
  fi

  DENSE_DOCS=60 DENSE_CHARS=1200 ANSWER_QUALITY_GATE="${ANSWER_QUALITY_GATE:-1}" \
  npx tsx rag/src/trec-rag-2026/agentic-rag/run_iterative_entry.ts \
    --run-id "$RUN_ID" \
    --output-dir "$OUT" \
    --topics "$topics_file" \
    "${token_args[@]}" \
    --qrels-dir "$QRELS_DIR" \
    --team-id "$TEAM_ID" \
    --resume \
    --sidecar-url "${SIDECARS[$((shard_id % SIDECAR_COUNT))]}" \
    --llm-provider openai_llm \
    --llm-model gpt-5.6-sol \
    --layer-rerank --layer-passages \
    --layer-checklist "$CHECKLIST" \
    --layer-verify --verify-mode weaken \
    --answer-style dense \
    --initial-docs 12 --docs-per-iteration 10 \
    --max-documents-read 60 --max-iterations 6 \
    >"$log_file" 2>&1
}

rm -f "$OUT/.shards"/shard_*.tsv
index=0
while IFS= read -r line; do
  printf '%s\n' "$line" >>"$OUT/.shards/shard_$((index % SHARDS)).tsv"
  index=$((index + 1))
done <"$TOPICS"

pids=()
for shard in $(seq 0 $((SHARDS - 1))); do
  shard_file="$OUT/.shards/shard_$shard.tsv"
  [[ -s "$shard_file" ]] || continue
  echo "shard $shard: $(wc -l <"$shard_file") topics"
  run_entry "$shard_file" "$OUT/.shards/shard_$shard.log" "$shard" &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=$((failed + 1))
done
[[ "$failed" -eq 0 ]] || echo "Warning: $failed shard(s) require resume" >&2

# Complete/resume all topics and assemble one output file. The rescue pass
# disables the operational quality gate so every input topic is represented.
ANSWER_QUALITY_GATE=0 run_entry "$TOPICS" "$OUT/.shards/assemble.log" 0
echo "Final RAG output: $OUT/rag_output_trec_rag_2026.jsonl"
