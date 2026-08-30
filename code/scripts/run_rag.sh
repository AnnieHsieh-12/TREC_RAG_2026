#!/usr/bin/env bash
set -euo pipefail

# Final bounded RAG pipeline.
# Required inputs are supplied by environment variables so competition data
# and credentials never need to be committed to the repository.

CODE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$CODE_ROOT"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  ./node_modules/.bin/tsx src/trec-rag-2026/rag-pipeline/run.ts --help
  exit 0
fi

if [[ -f .env.local ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env.local
  set +a
fi

: "${OPENAI_API_KEY:?Set OPENAI_API_KEY or create code/.env.local}"
: "${TOPICS:?Set TOPICS to the input topics TSV}"
: "${CHECKLIST:?Set CHECKLIST to the topic checklist JSONL}"

for required in "$TOPICS" "$CHECKLIST"; do
  [[ -s "$required" ]] || { echo "Missing input: $required" >&2; exit 1; }
done
PYTHON="${PYTHON:-python3}"
"$PYTHON" tools/validate_checklist.py "$TOPICS" "$CHECKLIST"
if [[ -n "${QRELS_DIR:-}" && ! -d "$QRELS_DIR" ]]; then
  echo "Missing qrels directory: $QRELS_DIR" >&2
  exit 1
fi

SHARDS="${SHARDS:-4}"
OUT="${OUT:-$CODE_ROOT/out/final-rag}"
RUN_ID="${RUN_ID:-cfda-w5c}"
TEAM_ID="${TEAM_ID:-cfda}"
OPENAI_MODEL="${OPENAI_MODEL:-gpt-5.6-sol}"
SIDECAR_URLS="${SIDECAR_URLS:-http://127.0.0.1:8765}"
SUBMISSION_OUT="${SUBMISSION_OUT:-$CODE_ROOT/out/submissions/$RUN_ID}"

if ! [[ "$SHARDS" =~ ^[1-9][0-9]*$ ]]; then
  echo "SHARDS must be a positive integer; got: $SHARDS" >&2
  exit 2
fi
if ! [[ "$RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "RUN_ID must contain only letters, digits, dot, underscore, or hyphen" >&2
  exit 2
fi

mkdir -p "$OUT/.shards"

IFS=',' read -ra TOKENS <<< "${PYSERINI_TOKENS:-}"
TOKEN_COUNT=${#TOKENS[@]}
IFS=',' read -ra SIDECARS <<< "$SIDECAR_URLS"
SIDECAR_COUNT=${#SIDECARS[@]}
if [[ "$SIDECAR_COUNT" -eq 0 ]]; then
  echo "SIDECAR_URLS must contain at least one URL" >&2
  exit 2
fi
for sidecar in "${SIDECARS[@]}"; do
  if [[ -z "$sidecar" ]]; then
    echo "SIDECAR_URLS contains an empty URL" >&2
    exit 2
  fi
done

run_entry() {
  local topics_file="$1"
  local log_file="$2"
  local shard_id="${3:-0}"
  local token_args=()
  local qrels_args=()

  if [[ -n "${QRELS_DIR:-}" ]]; then
    qrels_args=(--qrels-dir "$QRELS_DIR")
  fi

  if [[ "$TOKEN_COUNT" -ge 1 && -n "${TOKENS[0]:-}" ]]; then
    export "CFDA_PYSERINI_TOKEN_$shard_id"="${TOKENS[$((shard_id % TOKEN_COUNT))]}"
    token_args=(--pyserini-token-env "CFDA_PYSERINI_TOKEN_$shard_id")
  fi

  DENSE_DOCS=60 DENSE_CHARS=1200 ANSWER_QUALITY_GATE=1 \
  ./node_modules/.bin/tsx src/trec-rag-2026/rag-pipeline/run.ts \
    --run-id "$RUN_ID" \
    --output-dir "$OUT" \
    --topics "$topics_file" \
    "${token_args[@]}" \
    "${qrels_args[@]}" \
    --team-id "$TEAM_ID" \
    --resume \
    --sidecar-url "${SIDECARS[$((shard_id % SIDECAR_COUNT))]}" \
    --llm-provider openai_llm \
    --llm-model "$OPENAI_MODEL" \
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

# Complete/resume all topics and assemble one output file. Keep the quality
# gate enabled: completeness must not turn a degraded answer into a valid one.
run_entry "$TOPICS" "$OUT/.shards/assemble.log" 0
RAW_RAG="$OUT/rag_output_trec_rag_2026.jsonl"
FINALIZATION_INPUT="$RAW_RAG"
if [[ "${REPLAY_OFFICIAL_W5C_REPAIR:-0}" == "1" ]]; then
  REPAIRED_RAG="$OUT/.w5c-heading-repaired.jsonl"
  "$PYTHON" tools/replay_uncited_heading_repair.py \
    --input "$RAW_RAG" --output "$REPAIRED_RAG"
  FINALIZATION_INPUT="$REPAIRED_RAG"
fi
"$PYTHON" tools/finalize_submissions.py \
  --rag "$FINALIZATION_INPUT" --team-id "$TEAM_ID" --rag-tag "$RUN_ID" \
  --outdir "$SUBMISSION_OUT"
FINAL_RAG="$SUBMISSION_OUT/rag_output_trec_rag_2026.jsonl"
"$PYTHON" tools/official_format_check.py --rag-only "$FINAL_RAG" "$TOPICS"
echo "Final RAG submission: $FINAL_RAG"
