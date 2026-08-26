"""Neural reranking of BM25 candidate pools via the NCHC-hosted models.

Terminology discipline (see docs/PROJECT_STATUS.md "Reporting Discipline"):
this module RERANKS a lexical (BM25) candidate pool with neural scorers. It is
NOT full-corpus dense retrieval - no dense index over the 553M-document corpus
exists, so documents BM25 never retrieved can never appear here.

Two scoring signals, both confirmed live against the NCHC gateway:
- Cross-encoder: POST /rerank with `BGE-Reranker-V2-M3` (Cohere-style API;
  returns a 0-1 calibrated relevance_score per document). Being calibrated
  and query-conditioned, these scores are comparable across topics - which
  raw BM25 magnitudes are not - making them usable for the variable-k
  per-narrative depth cutoff the official submission rules require.
- Bi-encoder: POST /embeddings with `bge-m3` (OpenAI-style; 1024-dim), scored
  by cosine similarity against the narrative embedding.

Both signals score the document's BEST PASSAGE (selected by src/passages.py's
BM25 chunk scorer against the narrative), not the raw document prefix - long
ClimbMix documents would otherwise be truncated by the models' input limits
and scored on whatever happens to sit at the front.
"""

import math
import time

import requests

from config import NCHC_API_BASE
from src.passages import select_passages

RERANK_MODEL = "BGE-Reranker-V2-M3"
EMBED_MODEL = "bge-m3"
# One passage per doc keeps rerank cost linear in pool size; ~512 words is
# within both models' comfortable input range.
PASSAGE_MAX_WORDS = 512


class RerankError(Exception):
    pass


def _post_with_retry(url, headers, payload, max_retries=6, timeout=120):
    attempt = 0
    while True:
        try:
            response = requests.post(url, headers=headers, json=payload,
                                     timeout=timeout)
        except requests.exceptions.RequestException as e:
            attempt += 1
            if attempt > max_retries:
                raise RerankError(f"request failed after {max_retries} retries: {e}")
            time.sleep(2 * attempt)
            continue
        if response.status_code == 429 or response.status_code >= 500:
            attempt += 1
            if attempt > max_retries:
                raise RerankError(
                    f"HTTP {response.status_code} after {max_retries} retries: "
                    f"{response.text[:300]}")
            time.sleep(min(5 * (2 ** attempt), 60))
            continue
        if response.status_code != 200:
            raise RerankError(f"HTTP {response.status_code}: {response.text[:300]}")
        return response.json()


def best_passage(narrative, doc_text):
    """The document's single best narrative-relevant passage (src/passages.py
    scoring), falling back to the document head when selection yields nothing."""
    passages = select_passages(doc_text, narrative, max_passages=1)
    if passages:
        text = passages[0]["text"]
    else:
        text = doc_text or ""
    words = text.split()
    if len(words) > PASSAGE_MAX_WORDS:
        text = " ".join(words[:PASSAGE_MAX_WORDS])
    return text


def cross_encoder_scores(narrative, passages, api_key, batch_size=64):
    """BGE-Reranker-V2-M3 relevance score (0-1) for each passage, in order."""
    url = f"{NCHC_API_BASE}/rerank"
    headers = {"Authorization": f"Bearer {api_key}",
               "Content-Type": "application/json"}
    scores = [None] * len(passages)
    for start in range(0, len(passages), batch_size):
        batch = passages[start:start + batch_size]
        result = _post_with_retry(url, headers, {
            "model": RERANK_MODEL, "query": narrative, "documents": batch,
        })
        for item in result["results"]:
            scores[start + item["index"]] = float(item["relevance_score"])
    if any(s is None for s in scores):
        raise RerankError("rerank response missing scores for some documents")
    return scores


def embedding_cosine_scores(narrative, passages, api_key, batch_size=32):
    """bge-m3 cosine similarity between the narrative and each passage."""
    url = f"{NCHC_API_BASE}/embeddings"
    headers = {"Authorization": f"Bearer {api_key}",
               "Content-Type": "application/json"}

    def embed(texts):
        result = _post_with_retry(url, headers,
                                  {"model": EMBED_MODEL, "input": texts})
        rows = sorted(result["data"], key=lambda d: d["index"])
        return [r["embedding"] for r in rows]

    nvec = embed([narrative])[0]
    nnorm = math.sqrt(sum(x * x for x in nvec))
    scores = []
    for start in range(0, len(passages), batch_size):
        for vec in embed(passages[start:start + batch_size]):
            dot = sum(a * b for a, b in zip(nvec, vec))
            vnorm = math.sqrt(sum(x * x for x in vec))
            scores.append(dot / (nnorm * vnorm) if nnorm and vnorm else 0.0)
    return scores


def rrf_fuse(rankings, k0=60, weights=None):
    """Weighted Reciprocal Rank Fusion. `rankings` is a list of docid lists
    (best first); returns [(docid, fused_score)] sorted best-first."""
    if weights is None:
        weights = [1.0] * len(rankings)
    fused = {}
    for ranking, w in zip(rankings, weights):
        for pos, docid in enumerate(ranking):
            fused[docid] = fused.get(docid, 0.0) + w / (k0 + pos + 1)
    return sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
