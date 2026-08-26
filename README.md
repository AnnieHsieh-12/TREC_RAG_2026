# TREC RAG 2026 — CFDA Final Pipeline

[![CI](https://github.com/AnnieHsieh-12/TREC_RAG_2026/actions/workflows/ci.yml/badge.svg)](https://github.com/AnnieHsieh-12/TREC_RAG_2026/actions/workflows/ci.yml)

Source-only release of CFDA's final Retrieval and Retrieval-Augmented
Generation pipelines for the TREC RAG 2026 track. Competition inputs,
submissions, generated answers, caches, traces, evaluation results, and the
technical report are intentionally excluded.

## Architecture

The repository has one TypeScript package and one Python sidecar:

- `code/config/final_pipeline.ts` defines the flattened final Retrieval policy.
- `code/src/trec-rag-2026/agentic-rag/` implements the final Retrieval and
  evidence-grounded generation controller.
- `code/src/trec-rag-2026/bounded-rag/` implements the bounded 12/10/60/6 RAG
  controller, including per-round top-300 reranking and verify/revise.
- `code/src/llm/`, `evaluation/`, and `agentic-rag-baseline/` are shared by
  both controllers; there is no duplicated RAG package.
- `sidecar/` provides local reranking, passage selection, sentence evidence,
  and the Codex CLI bridge used by the Retrieval policy.
- `code/tools/` contains checklist generation, deep-tail reranking,
  serialization, and official-format validation.

## Requirements

- Node.js 20.9 or newer
- Python 3.11 or newer
- A CUDA-capable GPU is recommended for neural reranking
- Access to the ClimbMix/Pyserini service
- NCHC credentials for the base Retrieval model
- An OpenAI API key for the bounded RAG launcher
- Codex CLI authentication for the Retrieval policy's query and writer roles

Install the locked TypeScript dependencies:

```bash
cd code
npm ci
cd ..
```

Create a Python environment and install the pinned dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r code/requirements-deepce.txt
pip install -r sidecar/requirements.txt
```

### Codex CLI setup

The Retrieval policy sends query-generation and writer requests through the
local sidecar to `codex exec`. Install and authenticate the official CLI on
the machine running the sidecar:

```bash
npm install --global @openai/codex
codex login
codex login status
```

On a headless server, use `codex login --device-auth`. API-key authentication
is also supported with `printenv OPENAI_API_KEY | codex login --with-api-key`.
See the [official Codex authentication documentation](https://learn.chatgpt.com/docs/auth).

## Configuration

Create local environment files. They are ignored by Git and must never be
committed:

```bash
cp code/.env.example code/.env.local
cp sidecar/.env.local.example sidecar/.env.local
```

Set the following values:

- `code/.env.local`: `NCHC_API_KEY`, `PYSERINI_API_TOKEN`, and
  `OPENAI_API_KEY` when running bounded RAG.
- `sidecar/.env.local`: `PYSERINI_API_TOKEN` and `NCHC_API_KEY`.

The launch scripts load these files automatically. `SIDECAR_URL` defaults to
`http://127.0.0.1:8765`.

## Inputs

Competition data is not redistributed. Provide:

- topics TSV with `topic_id<TAB>narrative` on each line;
- the organizer qrels directory used by the local metric reporter;
- checklist JSONL with one `{"qid":"...","items":[...]}` object per line.

Small fictional examples are available in `examples/`.

Generate a checklist from topics with the NCHC key in
`sidecar/.env.local`:

```bash
python code/tools/build_checklist.py \
  --topics /path/to/topics.tsv \
  --output /path/to/checklist.jsonl
```

## Start the sidecar

Run this in a separate terminal before either final pipeline:

```bash
source .venv/bin/activate
cd sidecar
python -m src.sidecar
```

The service listens on `127.0.0.1:8765`. At startup it downloads the configured
reranking model into the ignored local cache if the model is not already there.

## Run final Retrieval

```bash
cd code
npm run run:retrieval -- \
  --run-id cfda-final-retrieval \
  --team-id cfda \
  --output-dir out/final-retrieval \
  --topics /path/to/topics.tsv \
  --qrels-dir /path/to/qrels
```

The wrapper loads `code/.env.local`. The selected policy is defined once in
`code/config/final_pipeline.ts` and does not import historical version files.

Run deep-tail reranking and serialize the final Retrieval files:

```bash
source ../.venv/bin/activate
python tools/deep_ce_rerank.py out/final-retrieval \
  --variant 'RRF 1:1' \
  --out out/final-retrieval/deepce
RUN_DIR="$PWD/out/final-retrieval" bash scripts/run_final_retrieval.sh
```

## Run final bounded RAG

```bash
cd code
TOPICS=/path/to/topics.tsv \
QRELS_DIR=/path/to/qrels \
CHECKLIST=/path/to/checklist.jsonl \
SIDECAR_URLS=http://127.0.0.1:8765 \
npm run run:rag
```

Optional environment variables are `OUT`, `RUN_ID`, `TEAM_ID`, `SHARDS`,
`PYSERINI_TOKENS`, and comma-separated `SIDECAR_URLS`. The launcher loads
`code/.env.local`, uses the OpenAI API for generation, and uses the sidecar
only for local reranking, passage selection, and evidence verification.

## Validation

Run all type, unit, Python, and offline end-to-end smoke tests:

```bash
cd code
npm run check
```

The smoke test executes one complete bounded-RAG topic against deterministic
mock Pyserini and OpenAI responses; it requires no credentials or network.
GitHub Actions runs the same checks, a production dependency audit, shell
syntax checks, and Python compilation on every push and pull request.

Validate generated Retrieval and RAG files together:

```bash
code/scripts/validate_submission.sh \
  /path/to/retrieval.tsv /path/to/rag.jsonl /path/to/topics.tsv
```

## Reproducibility scope

Node dependencies are locked by `package-lock.json`; Python dependencies are
version-pinned. Model weights are downloaded from their upstream registries
and competition services require separate authorization. Generated runs,
official inputs, model caches, and intermediate outputs remain untracked.

## License

MIT
