import os

BASE_URL = "http://api.castorini.uwaterloo.ca"
INDEX_NAME = "climbmix-400b"
HITS = 100
RUN_ID = "bm25-narrative-top100"

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
TOPICS_PATH = os.path.join(ROOT_DIR, "data", "topics", "rag25-topics-dev.tsv")
QRELS_DIR = os.path.join(ROOT_DIR, "data", "qrels")
QRELS_FILES = [
    "rag25-climbmix-umbrela-codex-gpt5.5-medium-reasoning-v1.qrels",
    "rag25-climbmix-umbrela-qwen3.5-9b-v2.qrels",
    "rag25-climbmix-umbrela-ministral-3-14b-instruct-2512-v2.qrels",
]
RAW_CACHE_DIR = os.path.join(ROOT_DIR, "cache", "raw")
RUN_OUTPUT_PATH = os.path.join(ROOT_DIR, "runs", "retrieval", f"{RUN_ID}.tsv")
# Official per-topic nugget ground truth (mapped_sub_narrative + importance),
# vendored from https://github.com/TREC-RAG/trec-rag-data - same convention
# as TOPICS_PATH/QRELS_DIR above. DEV-ONLY: the organizers do not release
# nuggets for the official test topics, so as of the M2 test-topic rework
# this file is EVALUATION ground truth only - the live coverage gate runs on
# the generated checklist below instead (which exists for every topic).
NUGGETS_PATH = os.path.join(ROOT_DIR, "data", "nuggets", "rag25-dev-nuggets.jsonl")
# Generated coverage checklists (src/checklist.py): same file format as
# NUGGETS_PATH but produced by LLM narrative decomposition, so it can be
# (re)generated for ANY topic set - including the 119 official test topics,
# which have no official nuggets. This is the coverage gate's default source;
# dev runs use it too, so dev behavior mirrors test-topic conditions.
CHECKLIST_PATH = os.path.join(ROOT_DIR, "data", "checklists", "generated-dev.jsonl")
# Local cache dir for the small (~67MB) sentence-embedding model used by
# src/coverage.py's semantic CoverageItem matcher - kept inside the repo
# (not the default ~/.cache) so it stays inside this project's workspace.
EMBED_CACHE_DIR = os.path.join(ROOT_DIR, ".cache", "fastembed")

NCHC_API_BASE = "https://portal.genai.nchc.org.tw/api/v1"
RAG_RUN_ID = "rag-agentic-v1"
RAG_RUN_DESC = (
    "BM25-seeded iterative agent that searches and reads ClimbMix documents "
    "before writing evidence-grounded sentence-level answers."
)
RAG_OUTPUT_PATH = os.path.join(ROOT_DIR, "runs", "rag", f"{RAG_RUN_ID}.jsonl")
