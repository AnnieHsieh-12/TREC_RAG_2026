import json
import math

from src.passages import _STOPWORDS, _tokenize

# M2: narrative coverage checklist + retrieval-gap gate.
#
# CoverageItem tracks, per official sub-narrative (from the upstream nugget
# file), whether the agent has: found no evidence yet ("unsearched"), found
# evidence but not written about it ("evidence_found"), written it into the
# answer ("answered"), or given up on it because the round budget ran out
# before evidence was ever found ("blocked").
#
# Matching new evidence/sentences to an item: SEMANTIC (sentence-embedding
# cosine similarity) by default, with a LEXICAL (keyword/IDF) fallback if the
# embedding backend isn't available. This replaces an earlier lexical-only
# version after live 22-topic testing found it under-counted coverage on
# topics whose official sub-narratives are thematically close (e.g. several
# climate-mitigation-themed items, or several "how does X affect child
# development" items): a sentence would score well above the match threshold
# against the *correct* item but still get rejected by the margin-based
# ambiguity check, because a thematically adjacent item scored nearly as high
# on shared vocabulary alone.
#
# This is exactly the distinction the nugget-evaluation literature draws
# between semantic and lexical assignment: AutoNuggetizer (Pradeep et al.,
# "The Great Nugget Recall: Automating Fact Extraction and RAG Evaluation
# with Large Language Models", arXiv 2504.15068 - the paper behind RAGDoll's
# own nuggetizer, already integrated in M3) explicitly assigns nuggets at the
# semantic/conceptual level rather than by keyword match, "crediting
# semantically equivalent answers even when their surface forms differ."
# RAGDoll's own nuggetizer already does this via an LLM judge, but that's a
# separate, offline, per-topic pass - too slow/costly to call on every
# search result and every sentence *during* the live agent loop, which is
# what this module's matcher has to do. Local sentence embeddings
# (`fastembed`, BAAI/bge-small-en-v1.5, ~67MB, ONNX runtime, no GPU/API call)
# are the practical middle ground: no per-call latency or cost, but real
# semantic understanding instead of bag-of-words overlap. Verified against
# the exact failure cases found in live testing (see docs/ROADMAP.md M2)
# before switching the default over.

MIN_MATCH_SCORE = 1.5  # lexical fallback threshold (IDF-weighted score sum)
MIN_SIMILARITY = 0.5  # semantic threshold (cosine similarity, 0-1)
MIN_SIMILARITY_MARGIN = 0.03  # semantic: absolute lead over runner-up required

_embedding_model = None
_embedding_model_load_failed = False


def _get_embedding_model():
    """Lazily load and cache the sentence-embedding model (once per process).
    Returns None (permanently, for the rest of the process) if `fastembed`
    isn't installed or the model can't be loaded, so callers fall back to
    the lexical matcher instead of crashing the agent loop over an optional
    quality upgrade."""
    global _embedding_model, _embedding_model_load_failed
    if _embedding_model is not None:
        return _embedding_model
    if _embedding_model_load_failed:
        return None
    try:
        from fastembed import TextEmbedding

        from config import EMBED_CACHE_DIR

        _embedding_model = TextEmbedding("BAAI/bge-small-en-v1.5", cache_dir=EMBED_CACHE_DIR)
        return _embedding_model
    except Exception as e:
        print(f"  [coverage] semantic matcher unavailable, using lexical fallback: {e}")
        _embedding_model_load_failed = True
        return None


def _cosine(a, b):
    num = 0.0
    da = 0.0
    db = 0.0
    for x, y in zip(a, b):
        num += x * y
        da += x * x
        db += y * y
    if da == 0.0 or db == 0.0:
        return 0.0
    return num / math.sqrt(da * db)


class CoverageItem:
    def __init__(self, item_id, question, importance, nugget_texts):
        self.id = item_id
        self.question = question
        self.importance = importance  # "vital" or "okay"
        self.keywords = _keywords(nugget_texts)  # lexical fallback representation
        self.embedding = None  # semantic representation, set by _embed_items()
        self.status = "unsearched"
        self.supporting_docids = []
        self.blocked_reason = None

    def to_dict(self):
        return {
            "id": self.id,
            "question": self.question,
            "importance": self.importance,
            "status": self.status,
            "supporting_docids": list(self.supporting_docids),
            "blocked_reason": self.blocked_reason,
        }


def _keywords(nugget_texts):
    words = set()
    for t in nugget_texts:
        for w in _tokenize(t):
            if w not in _STOPWORDS and len(w) > 2:
                words.add(w)
    return words


def _embed_items(items):
    """Batch-embed every item's question text once (not per-comparison) and
    store the vector on each item. No-op (leaves .embedding = None on every
    item) if the embedding backend is unavailable - callers detect this and
    use the lexical fallback instead."""
    model = _get_embedding_model()
    if model is None or not items:
        return
    vectors = list(model.embed([i.question for i in items]))
    for item, vec in zip(items, vectors):
        item.embedding = vec


def _clean_subnarrative(raw):
    s = (raw or "").strip()
    if len(s) >= 2 and s.startswith('"') and s.endswith('"'):
        s = s[1:-1]
    return s.strip()


def load_coverage_items(qid, nuggets_path):
    """Parse `nuggets_path` (the official rag25-dev-nuggets.jsonl format) for
    `qid`'s entry into one CoverageItem per distinct mapped_sub_narrative.
    Returns [] if the topic has no nugget entry (not every qid necessarily
    does) or the file doesn't exist - callers should treat that as "coverage
    gating unavailable for this topic," not an error."""
    groups = {}
    order = []
    try:
        with open(nuggets_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if obj.get("qid") != qid:
                    continue
                for n in obj.get("nuggets", []):
                    sub = _clean_subnarrative(n.get("mapped_sub_narrative", ""))
                    if not sub:
                        continue
                    if sub not in groups:
                        groups[sub] = {"importance": "okay", "texts": []}
                        order.append(sub)
                    groups[sub]["texts"].append(n.get("text", ""))
                    if n.get("importance") == "vital":
                        groups[sub]["importance"] = "vital"
                break  # this qid's line is fully processed; stop scanning
    except FileNotFoundError:
        return []

    items = [
        CoverageItem(f"item_{i}", sub, groups[sub]["importance"], groups[sub]["texts"])
        for i, sub in enumerate(order)
    ]
    _embed_items(items)
    return items


def _item_document_frequency(items):
    """Lexical-fallback helper: for each keyword, how many of `items`'
    keyword sets contain it (this topic's own items as the IDF corpus)."""
    df = {}
    for item in items:
        for kw in item.keywords:
            df[kw] = df.get(kw, 0) + 1
    return df


def _best_matching_item_lexical(items, text, min_score=MIN_MATCH_SCORE, min_margin_ratio=1.3):
    """Fallback matcher: IDF-weighted keyword overlap, computed across
    `items` themselves (not corpus-wide) - see module docstring for why this
    exists and its known limitation on topics with thematically-close items."""
    text_tokens = set(_tokenize(text))
    if not text_tokens:
        return None, 0.0
    df = _item_document_frequency(items)
    n = max(len(items), 1)
    scores = []
    for item in items:
        shared = item.keywords & text_tokens
        if not shared:
            continue
        score = sum(math.log(1 + n / df.get(w, n)) for w in shared)
        scores.append((score, item))
    if not scores:
        return None, 0.0
    scores.sort(key=lambda pair: pair[0], reverse=True)
    best_score, best_item = scores[0]
    if best_score < min_score:
        return None, best_score
    if len(scores) > 1:
        second_score = scores[1][0]
        if second_score > 0 and best_score < min_margin_ratio * second_score:
            return None, best_score
    return best_item, best_score


def _best_matching_item_semantic(items, text, model):
    """Primary matcher: cosine similarity between `text`'s embedding and each
    item's precomputed question embedding. Requires both a minimum absolute
    similarity and a minimum absolute lead over the runner-up - same
    "safer to say not-sure than wrongly claim" philosophy as the lexical
    fallback, just calibrated for cosine similarity's bounded [-1, 1] scale
    instead of an open-ended IDF-weighted sum."""
    text_vec = list(model.embed([text]))[0]
    scores = [(_cosine(text_vec, item.embedding), item) for item in items if item.embedding is not None]
    if not scores:
        return None, 0.0
    scores.sort(key=lambda pair: pair[0], reverse=True)
    best_score, best_item = scores[0]
    if best_score < MIN_SIMILARITY:
        return None, best_score
    if len(scores) > 1:
        second_score = scores[1][0]
        if best_score - second_score < MIN_SIMILARITY_MARGIN:
            return None, best_score
    return best_item, best_score


def _best_matching_item(items, text):
    """Try the semantic matcher first; fall back to lexical if the embedding
    backend is unavailable or these items have no embeddings (e.g. loaded
    before the backend became available)."""
    model = _get_embedding_model()
    if model is not None and all(i.embedding is not None for i in items):
        return _best_matching_item_semantic(items, text, model)
    return _best_matching_item_lexical(items, text)


def update_from_evidence(items, docid, text):
    """Call after a search_climbmix/fetch_doc result comes back. Promotes the
    single best-matching 'unsearched' item to 'evidence_found', if any scores
    above the match threshold. Returns the list of item ids that changed
    (0 or 1 elements)."""
    candidates = [i for i in items if i.status == "unsearched"]
    if not candidates:
        return []
    best, score = _best_matching_item(candidates, text)
    if best is None:
        return []
    best.status = "evidence_found"
    if docid and docid not in best.supporting_docids:
        best.supporting_docids.append(docid)
    return [best.id]


def update_from_sentence(items, sentence_text):
    """Call after an add_sentence call. Promotes the single best-matching
    not-yet-answered item to 'answered', if any scores above the match
    threshold. Returns the list of item ids that changed (0 or 1 elements)."""
    candidates = [i for i in items if i.status in ("unsearched", "evidence_found")]
    if not candidates:
        return []
    best, score = _best_matching_item(candidates, sentence_text)
    if best is None:
        return []
    best.status = "answered"
    return [best.id]


def vital_items_needing_write(items):
    return [i for i in items if i.importance == "vital" and i.status == "evidence_found"]


def vital_items_unsearched(items):
    return [i for i in items if i.importance == "vital" and i.status == "unsearched"]


def coverage_summary(items):
    counts = {"unsearched": 0, "evidence_found": 0, "answered": 0, "blocked": 0}
    for item in items:
        counts[item.status] = counts.get(item.status, 0) + 1
    return counts
