#!/usr/bin/env bash
set -euo pipefail

# Scan an output directory for actual secret values from the current environment.
# Env var names such as NCHC_API_KEY are not considered leaks by themselves.

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <output_dir>" >&2
  exit 2
fi

TARGET_DIR="$1"
if [[ ! -d "$TARGET_DIR" ]]; then
  echo "Not a directory: $TARGET_DIR" >&2
  exit 2
fi

# Load .env.local if present so this can be run from a fresh shell.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "$ROOT_DIR/.env.local" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT_DIR/.env.local"
  set +a
fi

secrets=()
for var in NCHC_API_KEY PYSERINI_API_TOKEN OPENAI_API_KEY; do
  val="${!var:-}"
  if [[ -n "$val" && "$val" != replace_* && "$val" != "replace_if_required" ]]; then
    secrets+=("$var=$val")
  fi
done

if [[ ${#secrets[@]} -eq 0 ]]; then
  echo "No concrete secret values found in environment; nothing to scan."
  exit 0
fi

status=0
while IFS= read -r -d '' file; do
  [[ -f "$file" ]] || continue
  for pair in "${secrets[@]}"; do
    var="${pair%%=*}"
    val="${pair#*=}"
    if grep -F -q -- "$val" "$file"; then
      echo "SECRET LEAK: actual value of $var found in $file" >&2
      status=1
    fi
  done
done < <(find "$TARGET_DIR" -type f -print0)

if [[ $status -ne 0 ]]; then
  exit 1
fi

echo "No configured API key or token values found in $TARGET_DIR"
