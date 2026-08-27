# TREC RAG 2026 — CFDA Final Pipeline

[![CI](https://github.com/AnnieHsieh-12/TREC_RAG_2026/actions/workflows/ci.yml/badge.svg)](https://github.com/AnnieHsieh-12/TREC_RAG_2026/actions/workflows/ci.yml)

This repository is a source-only release of CFDA's final Retrieval and
Retrieval-Augmented Generation (RAG) pipelines for the TREC RAG 2026 track.
It contains the selected final policy, runtime entry points, deterministic
post-processing, output validation, and offline smoke tests.

Competition topics, qrels, the frozen checklist, generated answers, official
submission files, caches, traces, evaluation results, past experiments, and
the technical report are intentionally not redistributed.

## Final system at a glance

The release contains two related but independently submitted pipelines:

| Pipeline | Purpose | Final public run |
| --- | --- | --- |
| Retrieval | Produce a variable-depth ranked document list for every topic | `cfda-vfs-unc`, `cfda-vfs-deep` |
| Bounded RAG | Retrieve evidence, decide when it is sufficient, and generate a cited answer | `cfda-w5c` |

The TypeScript runners control the pipelines. A local Python sidecar provides
neural reranking, passage selection, and sentence-level evidence retrieval.
The official generation calls use the OpenAI API; the sidecar's Codex CLI
bridge is an optional backend for non-official reruns only.

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

The complete selected Retrieval policy is flattened in
[`code/config/final_pipeline.ts`](code/config/final_pipeline.ts). It does not
import the historical `V0/V1/V2/VF/VFs` experiment wrappers.

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
8. Replays the frozen official heading repair only when the complete raw-file
   hash matches the official W5c runtime, applies the registered team/run IDs,
   and runs a final format and topic-completeness gate.

New RAG runs do not receive heuristic heading rewriting: when the complete
frozen input hash does not match, the repair stage is a pass-through.

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

There is one canonical TypeScript source tree under `code/src`; no duplicated
`code/rag/src` package is required or tracked.

## Requirements

- Node.js 22 or newer
- Python 3.12
- Access to the ClimbMix/Pyserini service
- NCHC credentials for the base Retrieval model
- An OpenAI API key for the final Retrieval query/writer roles and bounded RAG
- A CUDA-capable GPU is recommended for neural reranking

The competition services, model weights, and inputs require separate access
and are not included in this repository.

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

### Optional Codex CLI backend

The official runs used the OpenAI API, not the Codex bridge. To use Codex CLI
as an alternative backend for a non-official rerun, authenticate it on the
machine running the sidecar:

```bash
npm install --global @openai/codex
codex login
codex login status
```

On a headless server, use `codex login --device-auth`. API-key authentication
is also available through `codex login --with-api-key`.

## Input files

Competition data is not redistributed. Three inputs are expected.

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

## Start the sidecar

The sidecar is required by the final bounded RAG pipeline. Start it in a
separate terminal from the repository root:

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
  --run-id VFs-official119 \
  --team-id pi-serini \
  --output-dir out/VFs-official119 \
  --topics /path/to/topics.tsv \
  --qrels-dir /path/to/qrels
```

The wrapper loads `code/.env.local`. A topic failure makes the command return
non-zero after writing `validation.json` and `failed_topics.json`; an incomplete
run must not proceed to submission building.

The main candidate pool is written to:

```text
code/out/VFs-official119/candidate_pool_top5000.trec
```

### 2. Run deep-tail reranking

```bash
source ../.venv/bin/activate
python tools/deep_ce_rerank.py out/VFs-official119 \
  --head 100 \
  --depth 3000 \
  --variant 'RRF 1:1' \
  --device auto \
  --out out/VFs-official119/deepce
```

### 3. Build the two Retrieval outputs

```bash
RUN_DIR="$PWD/out/VFs-official119" \
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
| `RUN_ID`, `TEAM_ID` | Raw runtime identities |
| `PYTHON` | Python executable; default `python3` |

## Runtime and submission identities

The historical raw runtime values are retained to reproduce the frozen
execution artifacts:

```text
Retrieval raw run: VFs-official119
RAG raw run:       W5c-official119
Raw team ID:       pi-serini
```

These are provenance values, not the final public submission identities. The
submission builders emit:

```text
Retrieval run tags: cfda-vfs-unc, cfda-vfs-deep
RAG team ID:       2026 cfda rag
RAG run ID:        cfda-w5c
```

For Retrieval, the six-column TREC output contains the final run tag rather
than a separate team-ID field.

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

## Reproducibility scope

- Node dependencies are locked by `code/package-lock.json`.
- Sidecar dependencies are fully resolved in `sidecar/requirements.lock`.
- Platform-sensitive deep-reranker dependencies are directly version-pinned.
- Model weights come from their upstream registries.
- Competition services and inputs require separate authorization.
- Generated outputs, caches, traces, and intermediate pools remain untracked.

The frozen official submissions are not stored in this repository. Their
SHA-256 checksums are provided for verification:

| Run | SHA-256 |
| --- | --- |
| `cfda-vfs-unc` | `376fb9ef131d9317571933766fc98fdace7223bab6051860b1f332e7d4e56cae` |
| `cfda-vfs-deep` | `4b13a3291c82ae49e9b5212f7f1c1261ce4cee6ee17e3d6a00716313fa795351` |
| `cfda-w5c` | `87dbe373dbb8b95e5a43f7041c998ec7b26cdc34adf9e455f14477ed421d0ec8` |

## License

MIT
