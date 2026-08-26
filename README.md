# TREC RAG 2026 — CFDA Final Pipeline

This repository contains the final CFDA Retrieval and Retrieval-Augmented
Generation pipelines for the TREC RAG 2026 track. It intentionally excludes
competition submissions, generated answers, candidate pools, traces,
evaluation results, historical experiments, and the technical report.

## Components

- `code/config/final_pipeline.ts`: one explicit, flattened final Retrieval
  policy with no dependency on development-time experiment wrappers.
- `code/rag/src/`: bounded RAG controller and output validation.
- `code/rag/scripts/run_final_rag.sh`: final 12/10/60/6 RAG launcher with
  per-round top-300 reranking, dense writing, and verify/revise.
- `sidecar/`: local reranking, passage-selection, and sentence-evidence service.
- `code/tools/`: checklist construction, deep-tail reranking, serialization,
  and output-format validation.

## Requirements

- Node.js 20+
- Python 3.11+
- A CUDA-capable GPU is recommended for neural reranking
- Access credentials for the configured LLM and ClimbMix/Pyserini services

Install TypeScript dependencies:

```bash
cd code
npm ci
cd rag && npm ci
```

Install Python dependencies in your own virtual environment:

```bash
pip install -r code/requirements-deepce.txt
pip install -r sidecar/requirements.txt
```

Copy the environment templates and insert credentials locally. Never commit
the resulting `.env.local` files.

```bash
cp code/.env.example code/.env.local
cp sidecar/.env.local.example sidecar/.env.local
```

## Inputs

Competition data is not redistributed. Supply these paths yourself:

- topics TSV: one `topic_id<TAB>narrative` per line;
- qrels directory for local metric computation;
- checklist JSONL: one `{"qid":"...","items":[...]}` object per line.

Non-competition examples are available in `examples/`.

To generate a checklist from topics:

```bash
python code/tools/build_checklist.py \
  --topics /path/to/topics.tsv \
  --output /path/to/checklist.jsonl
```

## Run the local sidecar

```bash
cd sidecar
python -m src.sidecar
```

## Run the final Retrieval policy

```bash
cd code
npm run run:final -- \
  --run-id cfda-final-retrieval \
  --team-id cfda \
  --output-dir out/final-retrieval \
  --topics /path/to/topics.tsv \
  --qrels-dir /path/to/qrels
```

The final policy is defined once in `code/config/final_pipeline.ts`; no
historical version wrapper is required.

Deep-tail reranking and final serialization use:

```bash
cd code
python tools/deep_ce_rerank.py out/final-retrieval --variant 'RRF 1:1'
RUN_DIR="$PWD/out/final-retrieval" bash scripts/run_final_retrieval.sh
```

## Run the final RAG pipeline

```bash
cd code
TOPICS=/path/to/topics.tsv \
QRELS_DIR=/path/to/qrels \
CHECKLIST=/path/to/checklist.jsonl \
SIDECAR_URLS=http://127.0.0.1:8765 \
bash rag/scripts/run_final_rag.sh
```

Optional environment variables include `OUT`, `RUN_ID`, `TEAM_ID`, `SHARDS`,
`PYSERINI_TOKENS`, and comma-separated `SIDECAR_URLS`.

## Validation

```bash
cd code
npm run typecheck
npm run test:ts
npm --prefix rag run check
python -m unittest discover -s tools/tests -p 'test_*.py'
```

To validate generated Retrieval and RAG files together:

```bash
code/scripts/validate_submission.sh \
  /path/to/retrieval.tsv /path/to/rag.jsonl /path/to/topics.tsv
```

## Repository scope

This repository publishes source code only. Generated runs, final submissions,
official test inputs, model caches, document caches, and intermediate outputs
are deliberately excluded.

## License

MIT
