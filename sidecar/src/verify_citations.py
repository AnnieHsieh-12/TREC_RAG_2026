import json
import os
import sys
import time

# Windows consoles/redirected-file streams often default to the system
# codepage (e.g. cp950 for Traditional Chinese locales), which cannot
# encode Unicode punctuation LLM output commonly contains (non-breaking
# hyphens, em dashes, curly quotes). Force UTF-8 so verification output
# never crashes on a print() call after doing real work.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import RAG_OUTPUT_PATH
from src.common import load_nchc_key, load_token
from src.rag_tools import NCHCClient
from src.passages import select_passages
from src.retriever import fetch_doc, RetrieverError

VERIFY_MODEL = "gpt-oss-120b"

# Design note: this is a second, separate verification pass, not part of the
# main agent's tool-calling loop. It is inspired by two papers found during
# the project literature review (see docs/agentic-rag-literature-review.md):
#   - FActScore (Min et al., EMNLP 2023): decompose a generation into atomic
#     claims and verify each against a source ("Atomic Fact Validation").
#     Our add_sentence design already keeps sentences close to one claim
#     each, so we verify per-sentence rather than decomposing further.
#   - CiteGuard (ACL 2025): reframes citation checking as "does this citation
#     match what a careful human author would cite for this claim" rather
#     than a binary existence check.
# The main pipeline's EvidenceLog check only confirms a docid was actually
# retrieved. It says nothing about whether the docid's content actually
# supports the specific sentence - that gap is what this script checks.

VERIFY_TOOL = {
    "type": "function",
    "function": {
        "name": "report_verdict",
        "description": (
            "Report whether the source document supports the claim. Call "
            "this exactly once."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "verdict": {
                    "type": "string",
                    "enum": ["supported", "partially_supported", "not_supported"],
                    "description": (
                        "'supported': the document clearly backs the claim. "
                        "'partially_supported': the document is related but "
                        "the claim overstates it, omits important context "
                        "the document itself provides (e.g. a counterargument), "
                        "or is only weakly implied. "
                        "'not_supported': the document does not back the claim."
                    ),
                },
                "explanation": {
                    "type": "string",
                    "description": "One sentence explaining the verdict.",
                },
            },
            "required": ["verdict", "explanation"],
        },
    },
}

VERIFY_SYSTEM_PROMPT = (
    "You are a strict fact-checker. You will be given one claim sentence and "
    "the most relevant excerpts of a document cited as its source. Judge only "
    "whether the document's content actually supports the claim as written "
    "- not whether the claim is true in general, and not using any outside "
    "knowledge. Pay attention to whether the claim ignores context the "
    "document itself provides (for example, presenting one side of a "
    "document that discusses both sides). Call report_verdict exactly once."
)


def _judge_excerpt(sentence_text, doc_text, budget_chars=6000):
    """Build the judge's view of the document: the passages most relevant to
    THE CLAIM SENTENCE, not the document head.

    Why: measured on a real run, 84% of cited documents exceed the judge's
    6,000-char window (median 17k chars), and the agent cites documents
    because of passages found DEEP in them (via the same passage selector) -
    so a head-truncated judge systematically misses the supporting evidence
    and under-counts support. Selecting passages against the sentence aligns
    the judge's window with where the evidence actually is. Falls back to
    the document head when selection yields nothing (short/degenerate docs).
    """
    passages = select_passages(doc_text, sentence_text, max_passages=3)
    if not passages:
        return doc_text[:budget_chars]
    parts = []
    used = 0
    for p in passages:
        text = p["text"]
        if used + len(text) > budget_chars:
            text = text[: budget_chars - used]
        parts.append(f"[excerpt @ char {p['offset']}]\n{text}")
        used += len(text)
        if used >= budget_chars:
            break
    return "\n\n".join(parts)


def verify_one(sentence_text, docid, doc_text, nchc_client):
    user_prompt = (
        f"Claim:\n{sentence_text}\n\n"
        f"Cited document ({docid}), most relevant excerpts:\n"
        f"{_judge_excerpt(sentence_text, doc_text)}"
    )
    messages = [
        {"role": "system", "content": VERIFY_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    result = nchc_client.chat(
        model=VERIFY_MODEL,
        messages=messages,
        tools=[VERIFY_TOOL],
        tool_choice={"type": "function", "function": {"name": "report_verdict"}},
    )
    message = result["choices"][0]["message"]
    tool_calls = message.get("tool_calls") or []
    if not tool_calls:
        return {"verdict": "error", "explanation": "model returned no verdict"}
    try:
        args = json.loads(tool_calls[0]["function"]["arguments"])
    except json.JSONDecodeError:
        return {"verdict": "error", "explanation": "unparseable verdict"}
    return {
        "verdict": args.get("verdict", "error"),
        "explanation": args.get("explanation", ""),
    }


def verify_record(record, pyserini_token, nchc_client, doc_cache, done_keys,
                   results_fh):
    narrative_id = record["metadata"]["narrative_id"]
    references = record["references"]

    for i, sentence in enumerate(record["answer"]):
        for citation_idx in sentence["citations"]:
            docid = references[citation_idx]
            key = (narrative_id, i, docid)
            if key in done_keys:
                continue

            if docid not in doc_cache:
                try:
                    doc = fetch_doc(docid, pyserini_token, parse=True)
                except RetrieverError as e:
                    # The shared Pyserini API is intermittently flapping
                    # (502s that clear up and recur within minutes) - don't
                    # let one stubborn docid kill the whole batch. Record it
                    # as an error verdict and keep going; it is not marked
                    # done, so a later re-run will retry it.
                    result = {
                        "narrative_id": narrative_id,
                        "sentence_index": i,
                        "sentence_text": sentence["text"],
                        "docid": docid,
                        "verdict": "error",
                        "explanation": f"doc fetch failed: {e}",
                    }
                    results_fh.write(json.dumps(result, ensure_ascii=False) + "\n")
                    results_fh.flush()
                    _print_result(result)
                    continue
                doc_cache[docid] = (
                    doc.get("text", "") if isinstance(doc, dict) else str(doc)
                )
            doc_text = doc_cache[docid]

            verdict = verify_one(sentence["text"], docid, doc_text, nchc_client)
            time.sleep(3)  # be gentle on the shared NCHC rate limit
            result = {
                "narrative_id": narrative_id,
                "sentence_index": i,
                "sentence_text": sentence["text"],
                "docid": docid,
                **verdict,
            }
            results_fh.write(json.dumps(result, ensure_ascii=False) + "\n")
            results_fh.flush()
            _print_result(result)


def _print_result(r):
    flag = "" if r["verdict"] == "supported" else "  <-- FLAG"
    print(
        f"[{r['narrative_id']}] sentence {r['sentence_index']} "
        f"cites {r['docid']}: {r['verdict']}{flag}"
    )
    if r["verdict"] != "supported":
        print(f"    claim: {r['sentence_text']}")
        print(f"    why:   {r['explanation']}")


def _results_path_for(path):
    base, _ = os.path.splitext(path)
    return f"{base}.verify_results.jsonl"


def main():
    if len(sys.argv) < 2:
        print(f"usage: python src/verify_citations.py <rag_output_file>")
        print(f"  (default if omitted: {RAG_OUTPUT_PATH})")
        path = RAG_OUTPUT_PATH
    else:
        path = sys.argv[1]

    results_path = _results_path_for(path)

    # Resume support: any (narrative_id, sentence_index, docid) already
    # verified in a prior run is skipped, so an interrupted run (network
    # blip, rate limit) doesn't have to restart the whole ~20+ minute pass.
    # "error" verdicts (e.g. a doc fetch that failed on a flapping API) are
    # deliberately NOT treated as done, so they get retried automatically.
    done_keys = set()
    if os.path.exists(results_path):
        with open(results_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if r["verdict"] == "error":
                    continue
                done_keys.add((r["narrative_id"], r["sentence_index"], r["docid"]))
        if done_keys:
            print(f"Resuming: {len(done_keys)} citation(s) already verified")

    pyserini_token = load_token()
    nchc_key = load_nchc_key()
    nchc_client = NCHCClient(api_key=nchc_key)
    doc_cache = {}

    with open(path, encoding="utf-8") as f, \
            open(results_path, "a", encoding="utf-8") as results_fh:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            verify_record(record, pyserini_token, nchc_client, doc_cache,
                          done_keys, results_fh)

    # Dedupe by (narrative_id, sentence_index, docid), keeping the last
    # occurrence - a retried success overwrites an earlier "error" placeholder
    # from a flaky-API run rather than being double-counted alongside it.
    latest = {}
    with open(results_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            key = (r["narrative_id"], r["sentence_index"], r["docid"])
            latest[key] = r

    counts = {"supported": 0, "partially_supported": 0, "not_supported": 0, "error": 0}
    total = 0
    for r in latest.values():
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
        total += 1

    print(f"\n{total} citation(s) checked:")
    for k, v in counts.items():
        if v:
            print(f"  {k}: {v} ({100 * v / total:.0f}%)")


if __name__ == "__main__":
    main()
