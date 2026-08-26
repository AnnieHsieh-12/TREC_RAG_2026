import math
import re

# Select query-relevant chunks within one document using lexical BM25 scores.
# IDF is estimated across that document's chunks, and selected passages are
# returned in descending relevance order.

_WORD_RE = re.compile(r"[A-Za-z0-9']+")

_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "is", "are",
    "was", "were", "with", "as", "by", "at", "from", "this", "that", "it",
    "its", "be", "into", "about", "their", "how", "what", "i", "my", "also",
})

BM25_K1 = 1.5
BM25_B = 0.75


def _tokenize(text):
    return _WORD_RE.findall(text.lower())


def chunk_text(text, chunk_words=512, overlap_words=64):
    """Split text into overlapping word-count chunks, each tagged with an
    approximate character offset into the original text (approximate because
    `text.split()` collapses original whitespace runs to single spaces)."""
    words = text.split()
    if not words:
        return []

    offsets = []
    pos = 0
    for w in words:
        offsets.append(pos)
        pos += len(w) + 1  # +1 for the space " ".join() would insert

    step = max(chunk_words - overlap_words, 1)
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_words, len(words))
        chunks.append({
            "text": " ".join(words[start:end]),
            "offset": offsets[start],
        })
        if end >= len(words):
            break
        start += step
    return chunks


def _bm25_scores(query_terms, chunks_tokens, k1=BM25_K1, b=BM25_B):
    """BM25 score of `query_terms` against each token list in
    `chunks_tokens` (one score per chunk, same order)."""
    n = len(chunks_tokens)
    if n == 0 or not query_terms:
        return [0.0] * n

    doc_freq = {}
    chunk_len = []
    chunk_term_counts = []
    for tokens in chunks_tokens:
        chunk_len.append(len(tokens))
        counts = {}
        for tok in tokens:
            counts[tok] = counts.get(tok, 0) + 1
        chunk_term_counts.append(counts)
        for term in set(tokens):
            doc_freq[term] = doc_freq.get(term, 0) + 1

    avgdl = (sum(chunk_len) / n) if n else 0.0
    idf = {}
    for term in query_terms:
        nt = doc_freq.get(term, 0)
        idf[term] = math.log(1 + (n - nt + 0.5) / (nt + 0.5))

    scores = []
    for counts, dl in zip(chunk_term_counts, chunk_len):
        s = 0.0
        length_norm = (1 - b + b * (dl / avgdl)) if avgdl else 1.0
        for term in query_terms:
            f = counts.get(term, 0)
            if f == 0:
                continue
            denom = f + k1 * length_norm
            s += idf[term] * (f * (k1 + 1)) / denom
        scores.append(s)
    return scores


def select_passages(text, query, max_passages=3, chunk_words=512,
                     overlap_words=64, max_chars_per_passage=1200):
    """Return up to `max_passages` chunks of `text` most relevant to `query`
    (highest-scoring first), each: {text, offset, score, reason}. `reason`
    is 'lexical_match' if the chunk actually shares query terms, or
    'fallback_no_match' if nothing in the document matched (falls back to
    chunks in original document order, so the agent still sees *something*
    rather than nothing)."""
    if not text:
        return []
    query_terms = [t for t in _tokenize(query or "") if t not in _STOPWORDS]
    chunks = chunk_text(text, chunk_words=chunk_words, overlap_words=overlap_words)
    if not chunks:
        return []

    chunks_tokens = [_tokenize(c["text"]) for c in chunks]
    scores = _bm25_scores(query_terms, chunks_tokens)
    scored = sorted(zip(scores, chunks), key=lambda pair: pair[0], reverse=True)
    selected = scored[:max_passages]

    results = []
    for score, c in selected:
        results.append({
            "text": c["text"][:max_chars_per_passage],
            "offset": c["offset"],
            "score": round(score, 4),
            "reason": "lexical_match" if score > 0 else "fallback_no_match",
        })
    return results
