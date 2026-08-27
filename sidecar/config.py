"""Environment-backed configuration for the local sidecar."""

import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_URL = os.environ.get("PYSERINI_BASE_URL", "http://api.castorini.uwaterloo.ca")
INDEX_NAME = os.environ.get("PYSERINI_INDEX", "climbmix-400b")
NCHC_API_BASE = os.environ.get(
    "NCHC_BASE_URL", "https://portal.genai.nchc.org.tw/api/v1"
)
EMBED_CACHE_DIR = os.environ.get(
    "SIDECAR_MODEL_CACHE", os.path.join(ROOT_DIR, ".cache", "fastembed")
)
