import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    CHECKLIST_PATH,
    RAG_RUN_DESC,
    RAG_RUN_ID,
    RAG_OUTPUT_PATH,
    RAW_CACHE_DIR,
    ROOT_DIR,
    TOPICS_PATH,
)
from src.common import load_nchc_key, load_token, load_topics
from src.coverage import (
    coverage_summary,
    load_coverage_items,
    update_from_evidence,
    update_from_sentence,
    vital_items_needing_write,
    vital_items_unsearched,
)
from src.passages import select_passages
from src.rag_tools import EvidenceLog, NCHCClient, NCHCError, TOOLS, dispatch_tool_call
from src.trace import NullTracer, TraceWriter

TEAM_ID = "cfdalab"
# Selected for the complete development run. Earlier model-selection pilots
# are not treated as reproducible results because their raw logs were not kept.
MODEL = "gpt-oss-120b"
# gpt-oss-120b's research-heavy behavior hit the old cap of 12 before ever
# calling finish_answer (10 rounds of search/fetch before writing a single
# sentence on topic 14) - raised so slower, more thorough models can actually
# finish instead of being cut off mid-research.
MAX_TOOL_ROUNDS = 20
MAX_ANSWER_WORDS = 1024
MAX_CITATIONS_PER_SENTENCE = 3

# Flat, single-purpose tool schemas are used instead of one big nested
# "submit_answer(answer: [{text, cited_docids}])" call: open-model tool-calling
# is noticeably less reliable at filling nested array-of-object parameters than
# at repeated simple flat calls. One add_sentence() call per sentence, then a
# single finish_answer() call, degrades far less often in practice.
ADD_SENTENCE_TOOL = {
    "type": "function",
    "function": {
        "name": "add_sentence",
        "description": (
            "Add one grounded sentence to the final answer. Call this once "
            "per sentence, in order."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "One answer sentence."},
                "cited_docids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "maxItems": MAX_CITATIONS_PER_SENTENCE,
                    "description": (
                        "One to three ClimbMix docids, strongest support first, "
                        "from search_climbmix or fetch_doc results that support "
                        "this sentence."
                    ),
                },
            },
            "required": ["text", "cited_docids"],
        },
    },
}

FINISH_ANSWER_TOOL = {
    "type": "function",
    "function": {
        "name": "finish_answer",
        "description": (
            "Call this exactly once, after you have added every answer "
            "sentence with add_sentence, to signal the answer is complete."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
}

ALL_TOOLS = TOOLS + [ADD_SENTENCE_TOOL, FINISH_ANSWER_TOOL]

SYSTEM_PROMPT = (
    "You are a research assistant answering a user's information need using "
    "only retrieved ClimbMix documents as evidence. Do not use outside "
    "knowledge. If the retrieved candidates are insufficient, call "
    "search_climbmix for more, or fetch_doc for a candidate's full text. "
    "Once you have enough evidence, write the answer sentence by sentence: "
    "call add_sentence(text, cited_docids) once per sentence, where "
    "cited_docids lists one to three docids from actual search_climbmix or "
    "fetch_doc results that support that sentence, ordered strongest first. "
    "A sentence with no supporting docid should not be added. Keep the full "
    "answer at or below 1024 words. When every sentence has been added, call "
    "finish_answer() exactly once with no arguments."
)


def load_cached_candidates(qid, packet_dir=None):
    # packet_dir overrides where the initial evidence packet's candidates come
    # from (e.g. a reranked/fused pool built by the M5 retrieval work, kept in
    # its own directory so the BM25 cache identity is never polluted).
    # Default stays the legacy BM25 cache.
    path = os.path.join(packet_dir or RAW_CACHE_DIR, f"{qid}.json")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _extract_text(doc):
    # cache/raw/*.json was written with parse=false (raw JSON string); handle
    # both that shape and an already-parsed dict defensively.
    if isinstance(doc, dict):
        return doc.get("text", "")
    if isinstance(doc, str):
        try:
            parsed = json.loads(doc)
            return parsed.get("text", doc) if isinstance(parsed, dict) else doc
        except (json.JSONDecodeError, TypeError):
            return doc
    return ""


def initial_evidence_packet(candidates, evidence_log, max_docs=20, snippet_chars=300,
                             query=None, max_passages_per_doc=2, tracer=None, qid=None):
    packet = []
    for c in candidates[:max_docs]:
        docid = c["docid"]
        full_text = _extract_text(c.get("doc")).strip()
        passages = (
            select_passages(full_text, query, max_passages=max_passages_per_doc)
            if query else []
        )
        if passages:
            snippet = " ... ".join(p["text"] for p in passages)
            if tracer is not None:
                tracer.emit(
                    "passage_selection", qid, action="initial_packet_snippet",
                    docids=[{"docid": docid}], query=query,
                    passages=[
                        {"offset": p["offset"], "score": p["score"], "reason": p["reason"]}
                        for p in passages
                    ],
                )
        else:
            snippet = full_text.replace("\n", " ")[:snippet_chars]
        evidence_log.add(docid, snippet)
        packet.append({"docid": docid, "rank": c["rank"], "snippet": snippet})
    return packet


def build_references_and_citations(raw_answer, evidence_log):
    references = []
    seen = {}
    sentences = []
    dropped = 0
    answer_word_count = 0

    if not isinstance(raw_answer, list):
        return references, sentences, 1

    for item_index, item in enumerate(raw_answer):
        if not isinstance(item, dict):
            # model deviated from the {text, cited_docids} schema (e.g. returned
            # a bare string). No citation info survives, so it cannot be grounded.
            dropped += 1
            continue
        text = (item.get("text") or "").strip()
        if not text:
            continue
        sentence_word_count = len(text.split())
        if answer_word_count + sentence_word_count > MAX_ANSWER_WORDS:
            dropped += len(raw_answer) - item_index
            break
        raw_docids = item.get("cited_docids", [])
        if isinstance(raw_docids, str):
            # some models (e.g. Llama-3.1-405B-Instruct-FP8) return array-typed
            # tool arguments as a JSON-encoded string instead of a native JSON
            # array - same class of quirk as the "hits" string/int mismatch in
            # rag_tools.dispatch_tool_call. Recover it rather than dropping an
            # otherwise validly-grounded sentence.
            try:
                raw_docids = json.loads(raw_docids)
            except json.JSONDecodeError:
                raw_docids = []
        if not isinstance(raw_docids, list):
            raw_docids = []
        valid_docids = []
        for docid in raw_docids:
            if (
                isinstance(docid, str)
                and evidence_log.contains(docid)
                and docid not in valid_docids
            ):
                valid_docids.append(docid)
            if len(valid_docids) == MAX_CITATIONS_PER_SENTENCE:
                break
        if not valid_docids:
            dropped += 1
            continue
        indices = []
        for docid in valid_docids:
            if docid not in seen:
                seen[docid] = len(references)
                references.append(docid)
            idx = seen[docid]
            if idx not in indices:
                indices.append(idx)
        sentences.append({"text": text, "citations": indices})
        answer_word_count += sentence_word_count

    return references, sentences, dropped


def _trace_docids(name, args, tool_result):
    # Only docid/rank/score, never full snippet/document text - the trace is
    # for diagnosing agent behavior, not for duplicating retrieved content.
    if name == "search_climbmix" and isinstance(tool_result, list):
        return [
            {"docid": r.get("docid"), "rank": r.get("rank"), "score": r.get("score")}
            for r in tool_result
            if isinstance(r, dict)
        ]
    if name == "fetch_doc" and isinstance(tool_result, dict) and tool_result.get("docid"):
        return [{"docid": tool_result["docid"]}]
    if name == "add_sentence" and isinstance(args, dict):
        cited = args.get("cited_docids", [])
        if isinstance(cited, list):
            return [{"docid": d} for d in cited if isinstance(d, str)]
    return []


def run_topic(qid, narrative, pyserini_token, nchc_client, model=None,
              temperature=0.0, reasoning_effort=None, seed=None, candidate_k=20,
              max_tool_rounds=MAX_TOOL_ROUNDS, default_search_hits=20,
              run_id=None, run_desc=None, tracer=None,
              enable_coverage_gate=True, nuggets_path=CHECKLIST_PATH,
              packet_dir=None):
    model = model or MODEL
    run_id = run_id or RAG_RUN_ID
    run_desc = run_desc or RAG_RUN_DESC
    tracer = tracer or NullTracer()
    evidence_log = EvidenceLog()
    # M2 (test-topic rework): the gate's checklist now defaults to the
    # GENERATED checklist file (works for any topic, test topics included);
    # the official nugget file is evaluation-only. `nuggets_path` still
    # accepts either - both use the same file format.
    # coverage_items is [] whenever the gate is disabled or this topic has no
    # checklist entry - every coverage-related check below is then a no-op and
    # the loop degrades to pre-M2 behavior cleanly. That degradation must
    # never be silent (a submitted run would quietly lose the whole gate), so
    # it is printed AND traced.
    coverage_items = load_coverage_items(qid, nuggets_path) if enable_coverage_gate else []
    if enable_coverage_gate and not coverage_items:
        print(f"  [coverage] WARNING qid {qid}: no checklist entry in "
              f"{nuggets_path} - coverage gate INACTIVE for this topic")
        tracer.emit("coverage_gate_inactive", qid,
                    checklist_path=str(nuggets_path))

    def _coverage_snapshot():
        return coverage_summary(coverage_items) if coverage_items else None

    candidates = load_cached_candidates(qid, packet_dir=packet_dir)
    packet = initial_evidence_packet(
        candidates, evidence_log, max_docs=candidate_k,
        query=narrative, tracer=tracer, qid=qid,
    )

    user_prompt = (
        f"Topic narrative:\n{narrative}\n\n"
        f"Initial candidate evidence (top {len(packet)} BM25 results, "
        "docid + snippet):\n" + json.dumps(packet, ensure_ascii=False, indent=2)
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    collected_sentences = []
    finished = False
    stop_reason = None
    empty_response_retried = False
    stop_error = None
    round_num = -1
    try:
        for round_num in range(max_tool_rounds):
            if round_num > 0:
                time.sleep(2)  # be gentle on the shared NCHC rate limit, within a topic too

            # M2 retrieval-gap gate: once there are too few rounds left to
            # both search AND write up everything already found, stop
            # offering search_climbmix/fetch_doc at all - the API-level tool
            # schema is what the model can literally call, so this is a hard
            # constraint, not a prompt suggestion it can ignore.
            remaining_rounds = max_tool_rounds - round_num
            needing_write = vital_items_needing_write(coverage_items)
            synthesis_only = bool(coverage_items) and remaining_rounds <= len(needing_write) + 1
            tools_for_round = (
                [ADD_SENTENCE_TOOL, FINISH_ANSWER_TOOL] if synthesis_only else ALL_TOOLS
            )
            if synthesis_only and needing_write:
                messages.append({
                    "role": "user",
                    "content": (
                        f"Coverage reminder: you already found evidence for "
                        f"{len(needing_write)} aspect(s) but haven't written "
                        "about them yet: "
                        + "; ".join(i.question for i in needing_write)
                        + f". Only {remaining_rounds} round(s) remain, and "
                        "search/fetch are no longer available. Call "
                        "add_sentence for each of these now, then call "
                        "finish_answer."
                    ),
                })
            tracer.emit(
                "coverage_state", qid, round=round_num,
                coverage_state=_coverage_snapshot(), synthesis_only=synthesis_only,
            )

            t0 = time.time()
            try:
                result = nchc_client.chat(
                    model=model, messages=messages, tools=tools_for_round,
                    temperature=temperature, reasoning_effort=reasoning_effort,
                    seed=seed,
                )
            except NCHCError as e:
                # A single topic's model errors (after the client's own retries
                # are exhausted) must not crash the whole batch - record why we
                # stopped and let main() move on to the next topic.
                stop_error = str(e)
                tracer.emit(
                    "model_error", qid, round=round_num, action="chat",
                    error=stop_error, latency=time.time() - t0, retry_count=None,
                )
                stop_reason = "model_error"
                break
            latency = time.time() - t0

            message = result["choices"][0]["message"]
            tool_calls = message.get("tool_calls")

            if not tool_calls:
                # One retry before treating an empty response as terminal.
                # Motivated by a real M4 observation: Nemotron once found
                # evidence for every checklist item, then returned a response
                # with no tool call at all - the whole topic produced zero
                # sentences. An empty response can be a one-off glitch (this
                # is a nondeterministic shared endpoint); a nudge costs one
                # round and is bounded to a single attempt per run.
                if not empty_response_retried:
                    empty_response_retried = True
                    tracer.emit(
                        "model_response", qid, round=round_num,
                        action="empty_response_retry", latency=latency,
                    )
                    messages.append({
                        "role": "user",
                        "content": (
                            "Your last reply contained no tool call. You must "
                            "respond with a tool call: add_sentence to write "
                            "up evidence you have found, or finish_answer if "
                            "the answer is complete."
                        ),
                    })
                    continue
                tracer.emit(
                    "model_response", qid, round=round_num, action="no_tool_call",
                    latency=latency,
                )
                stop_reason = "no_tool_call"
                break

            messages.append(message)

            for tc in tool_calls:
                name = tc["function"]["name"]
                try:
                    args = json.loads(tc["function"]["arguments"] or "{}")
                except json.JSONDecodeError:
                    args = {}

                t1 = time.time()
                coverage_changed = []
                reason_code = None
                if name == "finish_answer":
                    # M2 gate: refuse to finish while vital coverage items
                    # have evidence sitting unwritten - this is what actually
                    # fixes the "found it but never wrote it" regression,
                    # rather than just hoping the model remembers.
                    still_needing_write = vital_items_needing_write(coverage_items)
                    if still_needing_write:
                        finished = False
                        reason_code = "rejected_finish_vital_unwritten"
                        tool_result = {
                            "status": "rejected",
                            "reason": (
                                "vital aspects have evidence but are not yet "
                                "written: "
                                + "; ".join(i.question for i in still_needing_write)
                                + ". Call add_sentence for these before finishing."
                            ),
                        }
                    else:
                        # Vital items never searched successfully are not
                        # forced to loop forever - marked blocked, transparent
                        # in the trace, and finish is allowed to proceed.
                        for item in vital_items_unsearched(coverage_items):
                            item.status = "blocked"
                            item.blocked_reason = "no_evidence_found_before_finish"
                            coverage_changed.append(item.id)
                        if coverage_changed:
                            reason_code = "blocked_at_finish"
                        finished = True
                        tool_result = {"status": "answer finalized"}
                elif name == "add_sentence":
                    if isinstance(args, dict) and args.get("text"):
                        collected_sentences.append(
                            {
                                "text": args.get("text"),
                                "cited_docids": args.get("cited_docids", []),
                            }
                        )
                        tool_result = {
                            "status": "added",
                            "sentence_index": len(collected_sentences) - 1,
                        }
                        if coverage_items:
                            coverage_changed = update_from_sentence(coverage_items, args["text"])
                            if coverage_changed:
                                reason_code = "sentence_matched"
                    else:
                        tool_result = {"status": "error", "message": "missing 'text'"}
                else:
                    try:
                        tool_result = dispatch_tool_call(
                            name, args, pyserini_token, evidence_log,
                            default_search_hits=default_search_hits,
                            tracer=tracer, qid=qid, query=narrative,
                        )
                    except Exception as e:
                        tool_result = {"error": str(e)}
                    has_error = isinstance(tool_result, dict) and tool_result.get("error")
                    if coverage_items and not has_error:
                        if name == "search_climbmix" and isinstance(tool_result, list):
                            for r in tool_result:
                                if isinstance(r, dict):
                                    coverage_changed.extend(
                                        update_from_evidence(
                                            coverage_items, r.get("docid"),
                                            r.get("snippet", ""),
                                        )
                                    )
                        elif name == "fetch_doc" and isinstance(tool_result, dict):
                            coverage_changed.extend(
                                update_from_evidence(
                                    coverage_items, tool_result.get("docid"),
                                    tool_result.get("text", ""),
                                )
                            )
                        if coverage_changed:
                            reason_code = "evidence_matched"
                tool_latency = time.time() - t1

                tracer.emit(
                    "tool_call", qid, round=round_num, action=name,
                    query=args.get("query") if isinstance(args, dict) else None,
                    docids=_trace_docids(name, args, tool_result),
                    latency=tool_latency,
                    error=(tool_result.get("error") if isinstance(tool_result, dict) else None),
                    coverage_state=_coverage_snapshot(),
                    reason_code=reason_code,
                    # retry_count is a placeholder: retriever.py/NCHCClient
                    # retry internally and don't yet surface an attempt count
                    # to the caller - always None until that's threaded through.
                    retry_count=None,
                )
                if coverage_changed:
                    tracer.emit(
                        "coverage_update", qid, round=round_num,
                        items_changed=coverage_changed, reason_code=reason_code,
                        coverage_state=_coverage_snapshot(),
                    )

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": json.dumps(tool_result, ensure_ascii=False)[:4000],
                    }
                )

            if finished:
                # With the gate above, reaching here now means every vital
                # item is either answered or was explicitly marked blocked -
                # a real guarantee, not just "the model decided to finish".
                stop_reason = "coverage_complete" if collected_sentences else "no_tool_call"
                break
        else:
            # Found live: the agent can fully answer every vital item (0
            # evidence_found, 0 unsearched) and still run out of rounds
            # without ever calling finish_answer - it just kept writing
            # instead of declaring itself done. That's a materially
            # different outcome from genuine unanswered gaps, so it gets its
            # own stop_reason rather than being lumped in as "...with_gaps".
            if (
                coverage_items
                and not vital_items_needing_write(coverage_items)
                and not vital_items_unsearched(coverage_items)
            ):
                stop_reason = "budget_exhausted_all_covered"
            else:
                stop_reason = "budget_exhausted_with_gaps"
    except KeyboardInterrupt:
        tracer.emit("run_end", qid, round=round_num, stop_reason="cancelled",
                    sentences_written=len(collected_sentences),
                    coverage_state=_coverage_snapshot())
        raise
    except Exception as e:
        # Anything unexpected (a malformed API response, a bug) still gets a
        # terminal trace event and doesn't crash the rest of the batch - the
        # same "one stubborn topic shouldn't kill the whole run" principle
        # already used for the shared-API retry logic elsewhere in this repo.
        # Printed (not just traced) so it stays visible in the run's own log.
        print(f"  topic {qid}: unexpected error, stopping this topic: {e}")
        tracer.emit("run_end", qid, round=round_num, stop_reason="model_error",
                    error=str(e), sentences_written=len(collected_sentences),
                    coverage_state=_coverage_snapshot())
        return None

    tracer.emit(
        "run_end", qid, round=round_num, stop_reason=stop_reason,
        error=stop_error, sentences_written=len(collected_sentences),
        coverage_state=_coverage_snapshot(),
        coverage_items=([i.to_dict() for i in coverage_items] if coverage_items else None),
    )

    if not collected_sentences:
        return None
    final_answer = collected_sentences

    references, sentences, dropped = build_references_and_citations(
        final_answer, evidence_log
    )
    if dropped:
        print(f"  topic {qid}: dropped {dropped} sentence(s) with no grounded citation")

    return {
        "metadata": {
            "team_id": TEAM_ID,
            "narrative_id": qid,
            "narrative": narrative,
            "run_id": run_id,
            "run_desc": run_desc,
        },
        "references": references,
        "answer": sentences,
    }


def _paths_for_run_id(run_id):
    # Mirrors config.RAG_OUTPUT_PATH's construction exactly for the default
    # run_id, so passing no run_id at all reproduces today's exact path -
    # behavior-preserving unless a caller explicitly asks for a different one.
    output_path = os.path.join(ROOT_DIR, "runs", "rag", f"{run_id}.jsonl")
    trace_path = os.path.join(ROOT_DIR, "runs", "rag", f"{run_id}.trace.jsonl")
    return output_path, trace_path


def main(topic_ids=None, pause_seconds=8, resume=False, candidate_k=20,
         max_tool_rounds=MAX_TOOL_ROUNDS, default_search_hits=20,
         run_id=None, run_desc=None, output_path=None, trace_path=None,
         enable_coverage_gate=True, nuggets_path=CHECKLIST_PATH, model=None,
         seed=None, packet_dir=None):
    run_id = run_id or RAG_RUN_ID
    run_desc = run_desc or RAG_RUN_DESC
    default_output_path, default_trace_path = _paths_for_run_id(run_id)
    output_path = output_path or default_output_path
    trace_path = trace_path or default_trace_path

    pyserini_token = load_token()
    nchc_key = load_nchc_key()
    nchc_client = NCHCClient(api_key=nchc_key)

    all_topics = dict(load_topics(TOPICS_PATH))
    if topic_ids is None:
        topic_ids = list(all_topics.keys())

    if resume and os.path.exists(output_path):
        with open(output_path, encoding="utf-8") as f:
            already_done = {json.loads(line)["metadata"]["narrative_id"] for line in f if line.strip()}
        topic_ids = [q for q in topic_ids if q not in already_done]
        print(f"Resuming: {len(already_done)} topic(s) already done, {len(topic_ids)} remaining")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tracer = TraceWriter(trace_path, run_id, mode="a" if resume else "w")

    # write incrementally (not all at the end) so a mid-run failure - e.g. a
    # shared-API rate limit - does not lose topics already completed.
    written = 0
    try:
        with open(output_path, "a" if resume else "w", encoding="utf-8") as f:
            for qid in topic_ids:
                if qid not in all_topics:
                    print(f"topic {qid} not found in dev topics, skipping")
                    continue
                print(f"Running topic {qid}...")
                record = run_topic(
                    qid, all_topics[qid], pyserini_token, nchc_client,
                    model=model, seed=seed, packet_dir=packet_dir,
                    candidate_k=candidate_k, max_tool_rounds=max_tool_rounds,
                    default_search_hits=default_search_hits,
                    run_id=run_id, run_desc=run_desc, tracer=tracer,
                    enable_coverage_gate=enable_coverage_gate,
                    nuggets_path=nuggets_path,
                )
                if record is None:
                    print(f"  topic {qid}: no grounded sentences produced, skipping")
                    continue
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                written += 1
                print(
                    f"  topic {qid}: {len(record['answer'])} sentences, "
                    f"{len(record['references'])} references"
                )
                time.sleep(pause_seconds)  # be gentle on the shared NCHC rate limit
    finally:
        tracer.close()

    print(f"Wrote {written} RAG records to {output_path}")
    print(f"Trace written to {trace_path}")


if __name__ == "__main__":
    main()
