# TREC RAG 2026 — CFDA Final Pipeline

[![CI](https://github.com/AnnieHsieh-12/TREC_RAG_2026/actions/workflows/ci.yml/badge.svg)](https://github.com/AnnieHsieh-12/TREC_RAG_2026/actions/workflows/ci.yml)

This repository presents CFDA's final system for the TREC RAG 2026 track. It
combines multi-route retrieval, neural reranking, adaptive evidence
acquisition, and citation-grounded answer generation.

The system is designed around a simple principle: retrieve broadly, spend
additional search only where the evidence is incomplete, and validate every
answer before serialization. The repository focuses on the final competition
pipeline and excludes development experiments and generated artifacts.

## System overview

The system contains two related pipelines:

| Pipeline | Purpose | Output |
| --- | --- | --- |
| Retrieval | Produce a variable-depth document ranking for each topic | Six-column TREC run |
| Bounded RAG | Acquire sufficient evidence and generate a grounded answer | TREC RAG JSONL |

TypeScript coordinates retrieval and generation, while a local Python service
handles neural reranking, passage selection, and sentence-level evidence
matching. The final generation path uses the OpenAI API.

### Key design choices

- **Adaptive retrieval:** an evidence-sufficiency decision determines whether
  the system stops or searches again.
- **Query diversity with drift control:** the original narrative, Query2Doc,
  and validated follow-up queries contribute through weighted RRF.
- **Protected ranking head:** later facet expansion and deep reranking improve
  coverage without destabilizing the highest-ranked documents.
- **Bounded agent loop:** the RAG pipeline has explicit limits on rounds and
  documents read, making its behavior auditable and cost-controlled.
- **Grounded generation:** answer revision and citation ordering operate on
  retrieved evidence before strict output validation.

## Retrieval pipeline

![CFDA final Retrieval pipeline](docs/figures/retrieval_pipeline.png)

For each narrative, the Retrieval pipeline:

1. Runs an anchor BM25 search using the original narrative.
2. Generates one Query2Doc pseudo-document and runs one expanded BM25 search.
   Within this expanded query, the original narrative has a repeat/boost factor
   of 5 to reduce query drift. This is separate from its RRF route weight,
   which remains 1.
3. Fuses the anchor and Query2Doc rankings with weighted reciprocal rank fusion
   (RRF, `k=60`).
4. Reads ranked evidence and asks an LLM whether the evidence is sufficient.
   If not, it validates up to three follow-up BM25 queries, assigns each a
   fusion weight of `0.25`, recomputes RRF, and continues within the configured
   document and iteration budgets.
5. Reranks the top 100 using BM25, cross-encoder, and dense signals with
   `1:1:1` RRF fusion.
6. Retrieves facet queries and splices their pool below a protected top 200.
7. Optionally applies deep cross-encoder reranking to ranks 101–3,000 while
   preserving the top 100.
8. Computes each topic's output depth from the pre-deep scores (`tau=0.20`) and
   writes a six-column TREC run.

The selected Retrieval policy is defined in
[`code/config/final_pipeline.ts`](code/config/final_pipeline.ts).

## Bounded RAG pipeline

![CFDA final bounded RAG pipeline](docs/figures/rag_pipeline.png)

For each narrative, the bounded RAG pipeline:

1. Retrieves BM25 top 1,000 and reranks the top 300.
2. Initially reads 12 documents.
3. Uses the supplied facet checklist and currently read evidence to decide
   whether the evidence is sufficient.
4. When evidence is insufficient, validates up to three follow-up queries. If
   a valid query and budget remain, it performs BM25 retrieval, weighted RRF,
   top-300 reranking, and reads 10 previously unseen documents.
5. Repeats until evidence is sufficient, no valid continuation remains, six
   rounds are reached, or 60 documents have been read.
6. Uses the final evidence and checklist to generate a dense cited answer.
7. Trims to 1,020 words, verifies or weakens unsupported claims, trims again,
   orders citations by support strength, and validates the JSON structure. The
   1,020-word internal cap leaves headroom below the organizer limit of 1,024.
8. Applies deterministic finalization and checks formatting, identifiers, and
   topic completeness before writing the submission file.

## Repository layout

```text
TREC_RAG_2026/
├── README.md
├── examples/                       fictional input examples
├── docs/figures/                   pipeline diagrams used in this README
├── .github/workflows/ci.yml        automated checks
├── code/
│   ├── config/final_pipeline.ts    selected final Retrieval policy
│   ├── scripts/                    public run/build/validation entry points
│   ├── src/llm/                    OpenAI, NCHC, and optional Codex clients
│   ├── src/evaluation/             local qrels discovery and metrics
│   ├── src/trec-rag-2026/
│   │   ├── retrieval-pipeline/     final Retrieval controller
│   │   ├── rag-pipeline/           final bounded RAG controller
│   │   ├── retrieval/              retrieval and reranking components
│   │   └── shared-rag/             shared prompts, contracts, and validation
│   ├── tools/                      checklist, deep rerank, finalization, checks
│   └── tests/                      deterministic offline smoke tests
└── sidecar/
    ├── src/sidecar.py              localhost HTTP service
    ├── requirements.lock           fully resolved Python environment
    └── README.md                   endpoint and configuration details
```

## Requirements

- Node.js 22 or newer
- Python 3.12
- Access to the ClimbMix/Pyserini service
- NCHC credentials for the base Retrieval model
- An OpenAI API key for the final Retrieval query/writer roles and bounded RAG
- A CUDA-capable GPU is recommended for neural reranking

Competition services, model weights, and inputs require separate access.

## Installation

From the repository root, install the locked TypeScript dependencies:

```bash
cd code
npm ci
cd ..
```

Create a Python environment and install the pinned reranking and sidecar
dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r code/requirements-deepce.txt
pip install -r sidecar/requirements.lock
```

## Configuration

Copy the example environment files. The resulting `.env.local` files are
ignored by Git and must not be committed.

```bash
cp code/.env.example code/.env.local
cp sidecar/.env.local.example sidecar/.env.local
```

Configure:

| File | Required values |
| --- | --- |
| `code/.env.local` | `NCHC_API_KEY`, `PYSERINI_API_TOKEN`, `OPENAI_API_KEY` |
| `sidecar/.env.local` | `PYSERINI_API_TOKEN`, `NCHC_API_KEY` |

`SIDECAR_URL` defaults to `http://127.0.0.1:8765`. The server port can be
changed with `SIDECAR_PORT`; use matching values when changing it.

### Optional local Codex backend

The Python service also includes an optional Codex CLI adapter for local
experimentation. It is not used by the final pipeline. To enable it,
authenticate Codex on the machine running the service:

```bash
npm install --global @openai/codex
codex login
codex login status
```

On a headless server, use `codex login --device-auth`. API-key authentication
is also available through `codex login --with-api-key`.

## Input files

Three inputs are expected. Small fictional examples are included, while the
competition data itself is not redistributed.

### Topics TSV

One topic per line, with the exact topic ID and narrative separated by a tab:

```text
topic_id<TAB>narrative
```

See [`examples/topics.example.tsv`](examples/topics.example.tsv).

### Qrels directory

Provide a directory containing one or more `*.qrels` files. Filenames are
discovered automatically and are used only for local metric reporting; qrels
are never used to choose queries or evidence.

### Checklist JSONL

One JSON object per topic:

```json
{"qid":"topic_id","items":["first aspect","second aspect"]}
```

See [`examples/checklist.example.jsonl`](examples/checklist.example.jsonl).
The frozen official 119-topic checklist is not redistributed. A newly generated
checklist follows the same procedure but may not be byte-identical to the
frozen model output.

Generate a checklist from topics with the NCHC key in `sidecar/.env.local`:

```bash
python code/tools/build_checklist.py \
  --topics /path/to/topics.tsv \
  --output /path/to/checklist.jsonl
```

## Start the local neural service

The bounded RAG pipeline calls a small local Python service (the `sidecar`) for
GPU-backed reranking and evidence processing. Start it in a separate terminal
from the repository root:

```bash
source .venv/bin/activate
cd sidecar
python -m src.sidecar
```

It binds only to `127.0.0.1`. At startup it loads or downloads the configured
reranking model into an ignored local cache. Its formal RAG responsibilities
are:

- rerank the current top 300;
- select relevant passages from full documents;
- retrieve sentence-level evidence for verification and citation ordering.

The `/llm` Codex bridge is optional and is not selected by the final scripts.

## Run the final Retrieval pipeline

### 1. Generate the final candidate pool

```bash
cd code
npm run run:retrieval -- \
  --run-id cfda-retrieval \
  --team-id cfda \
  --output-dir out/cfda-retrieval \
  --topics /path/to/topics.tsv \
  --qrels-dir /path/to/qrels
```

The wrapper loads `code/.env.local`. A topic failure makes the command return
non-zero after writing `validation.json` and `failed_topics.json`; an incomplete
run must not proceed to submission building.

The main candidate pool is written to:

```text
code/out/cfda-retrieval/candidate_pool_top5000.trec
```

### 2. Run deep-tail reranking

```bash
source ../.venv/bin/activate
python tools/deep_ce_rerank.py out/cfda-retrieval \
  --head 100 \
  --depth 3000 \
  --variant 'RRF 1:1' \
  --device auto \
  --out out/cfda-retrieval/deepce
```

### 3. Build the two Retrieval outputs

```bash
RUN_DIR="$PWD/out/cfda-retrieval" \
bash scripts/build_retrieval_submissions.sh
```

Default outputs:

```text
code/out/retrieval-submissions/cfda-vfs-unc/r_output_trec_rag_2026.tsv
code/out/retrieval-submissions/cfda-vfs-deep/r_output_trec_rag_2026.tsv
```

## Run the final bounded RAG pipeline

Keep the sidecar running, then use another terminal:

```bash
cd code
TOPICS=/path/to/topics.tsv \
QRELS_DIR=/path/to/qrels \
CHECKLIST=/path/to/checklist.jsonl \
SIDECAR_URLS=http://127.0.0.1:8765 \
RUN_ID=cfda-w5c \
TEAM_ID=cfda \
npm run run:rag
```

The launcher uses four shards by default, resumes completed topics, performs a
final rescue/assembly pass, applies deterministic finalization, and validates
the final JSONL against the complete topics file. Any missing, duplicated, or
extra topic makes the command fail.

Default final output:

```text
code/out/submissions/cfda-w5c/rag_output_trec_rag_2026.jsonl
```

Optional environment variables:

| Variable | Purpose |
| --- | --- |
| `OUT` | Raw RAG run directory |
| `SUBMISSION_OUT` | Finalized submission directory |
| `SHARDS` | Number of parallel topic shards; default `4` |
| `SIDECAR_URLS` | Comma-separated sidecar URLs |
| `PYSERINI_TOKENS` | Comma-separated tokens assigned across shards |
| `RUN_ID`, `TEAM_ID` | Run and team identifiers written to generated records |
| `PYTHON` | Python executable; default `python3` |

## Validation and tests

Run all formatting, type, TypeScript, Python, and offline pipeline tests:

```bash
cd code
npm run check
```

The smoke tests execute one Retrieval topic and one bounded-RAG topic against
deterministic mock services, so they need no credentials or network. GitHub
Actions runs these checks, a production dependency audit, shell syntax checks,
and Python compilation on every push and pull request.

Validate generated Retrieval and RAG files together:

```bash
code/scripts/validate_outputs.sh \
  /path/to/retrieval.tsv \
  /path/to/rag.jsonl \
  /path/to/topics.tsv
```

Scan a generated output directory for configured secret values before sharing
it:

```bash
code/scripts/check_no_secret_leak.sh /path/to/output-directory
```

## Reproducibility

- Node dependencies are locked by `code/package-lock.json`.
- Sidecar dependencies are fully resolved in `sidecar/requirements.lock`.
- Platform-sensitive deep-reranker dependencies are directly version-pinned.
- Model weights come from their upstream registries.
- Competition services and inputs require separate authorization.
- Generated outputs, caches, traces, and intermediate pools remain untracked.

Official submission files and generated outputs are not redistributed. This
repository provides the final pipeline implementation and validation tools.

## License

MIT
