import time

import requests

from config import NCHC_API_BASE
from src.passages import select_passages
from src.retriever import fetch_doc, search_climbmix


class NCHCError(Exception):
    pass


class NCHCClient:
    def __init__(self, api_key, base_url=NCHC_API_BASE):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    def list_models(self):
        url = f"{self.base_url}/models"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        response = requests.get(url, headers=headers, timeout=30)
        if response.status_code != 200:
            raise NCHCError(
                f"list_models failed: {response.status_code} {response.text[:300]}"
            )
        return response.json()

    def chat(
        self,
        model,
        messages,
        tools=None,
        tool_choice=None,
        temperature=0.0,
        max_tokens=None,
        reasoning_effort=None,
        max_retries=8,
        seed=None,
    ):
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {"model": model, "messages": messages}
        # temperature=None: omit the field entirely. Needed for OpenAI
        # reasoning-tier models (e.g. gpt-5-mini) that reject any explicit
        # temperature other than their default and error on temperature=0.0.
        if temperature is not None:
            payload["temperature"] = temperature
        # some reasoning-tier models (e.g. gpt-5.6-terra) reject function
        # tools on /chat/completions unless reasoning_effort is set to
        # "none" (the alternative is the separate /v1/responses endpoint).
        if reasoning_effort is not None:
            payload["reasoning_effort"] = reasoning_effort
        if tools:
            payload["tools"] = tools
        if tool_choice:
            payload["tool_choice"] = tool_choice
        if max_tokens:
            payload["max_tokens"] = max_tokens
        # vLLM-backed endpoints accept a sampling seed; harmless if the
        # backend ignores it. Note (documented in ROADMAP): a seed does NOT
        # guarantee determinism on a shared endpoint - dynamic batching and
        # MoE routing make identical requests diverge regardless.
        if seed is not None:
            payload["seed"] = seed

        attempt = 0
        while True:
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=60)
            except requests.exceptions.RequestException as e:
                attempt += 1
                if attempt > max_retries:
                    raise NCHCError(f"chat connection failed after {max_retries} retries: {e}")
                time.sleep(2 * attempt)
                continue

            if response.status_code == 429:
                attempt += 1
                if attempt > max_retries:
                    raise NCHCError(f"429 rate limited after {max_retries} retries")
                # NCHC's 429s here are "no backend deployment available right
                # now" (LiteLLM cooldown), not a per-caller pace limit, and
                # rarely send a Retry-After header - back off exponentially
                # (capped) instead of hammering a short fixed wait.
                default_wait = min(10 * (2**attempt), 90)
                wait_seconds = int(response.headers.get("Retry-After", default_wait))
                time.sleep(wait_seconds)
                continue

            if response.status_code == 200:
                return response.json()

            if response.status_code >= 500:
                attempt += 1
                if attempt > max_retries:
                    raise NCHCError(
                        f"chat failed after {max_retries} retries: "
                        f"{response.status_code} {response.text[:500]}"
                    )
                time.sleep(2 * attempt)
                continue

            raise NCHCError(
                f"chat failed: {response.status_code} {response.text[:500]}"
            )


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_climbmix",
            "description": (
                "Search the TREC RAG 2026 ClimbMix-400b corpus via the hosted "
                "Pyserini BM25 REST API. Returns up to `hits` ranked candidates "
                "with docid, rank, BM25 score, and a text snippet. Use to find "
                "evidence for the topic narrative or a specific sub-question."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query.",
                    },
                    "hits": {
                        "type": "integer",
                        "description": "Number of ranked results (default 20).",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_doc",
            "description": (
                "Fetch the full text of one ClimbMix document by its exact docid "
                "(e.g. shard_00459_61697), when a search snippet isn't enough to "
                "decide whether to cite it."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "docid": {
                        "type": "string",
                        "description": (
                            "Exact ClimbMix docid returned by a prior "
                            "search_climbmix or fetch_doc call."
                        ),
                    },
                },
                "required": ["docid"],
            },
        },
    },
]


class EvidenceLog:
    """Tracks every docid a topic's tool calls have actually returned, so the
    citation post-processor can reject any docid the model cites but never
    retrieved."""

    def __init__(self):
        self._docs = {}

    def add(self, docid, snippet=None):
        if docid not in self._docs:
            self._docs[docid] = snippet

    def contains(self, docid):
        return docid in self._docs

    def all_docids(self):
        return list(self._docs.keys())


def _snippet(doc, query=None, max_chars=300, max_passages=2):
    """Returns (snippet_text, passages). When `query` is given, the snippet
    is assembled from the most query-relevant chunks of the document (M1,
    src/passages.py) instead of a naive fixed-character prefix; `passages`
    is the raw per-chunk selection detail (offset/score/reason) for tracing,
    or [] when no query was given (falls back to the old prefix behavior) or
    the document had no chunks to score."""
    if doc is None:
        return "", []
    text = doc.get("text", "") if isinstance(doc, dict) else str(doc)
    text = text.strip()
    if query:
        passages = select_passages(text, query, max_passages=max_passages)
        if passages:
            return " ... ".join(p["text"] for p in passages), passages
    return text.replace("\n", " ")[:max_chars], []


def run_search_climbmix(query, token, evidence_log, hits=20, tracer=None, qid=None):
    candidates = search_climbmix(query, hits, token, parse=True)
    results = []
    for c in candidates:
        docid = c["docid"]
        snippet, passages = _snippet(c.get("doc"), query=query)
        evidence_log.add(docid, snippet)
        if tracer is not None and passages:
            tracer.emit(
                "passage_selection", qid, action="search_climbmix_snippet",
                docids=[{"docid": docid}], query=query,
                passages=[
                    {"offset": p["offset"], "score": p["score"], "reason": p["reason"]}
                    for p in passages
                ],
            )
        results.append(
            {
                "docid": docid,
                "rank": c["rank"],
                "score": c["score"],
                "snippet": snippet,
            }
        )
    return results


def run_fetch_doc(docid, token, evidence_log, query=None, tracer=None, qid=None):
    doc = fetch_doc(docid, token, parse=True)
    text = doc.get("text", "") if isinstance(doc, dict) else str(doc)
    evidence_log.add(docid, _snippet(text)[0])
    # The serialized tool result is truncated to 4,000 chars downstream, but
    # cited documents are ~17k chars median - a raw return means the agent
    # "reads the full text" yet only ever sees the head. Select the passages
    # most relevant to the narrative from the FULL text first, so the window
    # the agent actually receives is spent on evidence, not preamble. The
    # full-text fallback is kept for short docs / no-match cases.
    if query:
        passages = select_passages(text, query, max_passages=3)
        if passages:
            if tracer is not None:
                tracer.emit(
                    "passage_selection", qid, action="fetch_doc_passages",
                    docids=[{"docid": docid}],
                    passages=[
                        {"offset": p["offset"], "score": p["score"], "reason": p["reason"]}
                        for p in passages
                    ],
                )
            joined = "\n\n".join(
                f"[excerpt @ char {p['offset']}]\n{p['text']}" for p in passages
            )
            return {"docid": docid, "text": joined[:3800],
                    "note": "most relevant excerpts of the full document"}
    return {"docid": docid, "text": text}


def dispatch_tool_call(name, arguments, token, evidence_log, default_search_hits=20,
                        tracer=None, qid=None, query=None):
    if name == "search_climbmix":
        hits = int(arguments.get("hits", default_search_hits) or default_search_hits)
        return run_search_climbmix(
            arguments["query"], token, evidence_log, hits=hits, tracer=tracer, qid=qid,
        )
    if name == "fetch_doc":
        return run_fetch_doc(arguments["docid"], token, evidence_log,
                             query=query, tracer=tracer, qid=qid)
    raise ValueError(f"unknown tool: {name}")
