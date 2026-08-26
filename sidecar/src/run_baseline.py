import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import HITS, RAW_CACHE_DIR, RUN_ID, RUN_OUTPUT_PATH, TOPICS_PATH
from src.common import format_run_line, load_token, load_topics, write_run_file
from src.retriever import AuthError, RetrieverError, search_climbmix


def cache_path(qid, hits=HITS):
    # The retrieval depth is part of the cache identity: without it, a run at
    # a different `hits` silently reuses (or is silently reused by) another
    # depth's cache - a long-documented collision. Depth-100 keeps the legacy
    # un-suffixed name so the historical cache files remain valid.
    if hits == 100:
        return os.path.join(RAW_CACHE_DIR, f"{qid}.json")
    return os.path.join(RAW_CACHE_DIR, f"{qid}-k{hits}.json")


def get_candidates(qid, query_text, token, hits=HITS):
    path = cache_path(qid, hits)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            cached = json.load(f)
        # A shorter cached list than requested means the cache came from a
        # smaller-k run (pre-fix files have no depth suffix) - refuse to
        # silently under-deliver; refetch at the requested depth instead.
        if len(cached) >= hits:
            return cached[:hits]

    candidates = search_climbmix(query_text, hits, token)

    os.makedirs(RAW_CACHE_DIR, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(candidates, f)

    return candidates


def main(hits=HITS, run_id=RUN_ID, output_path=None):
    if output_path is None:
        output_path = (RUN_OUTPUT_PATH if hits == 100 and run_id == RUN_ID
                       else os.path.join(os.path.dirname(RUN_OUTPUT_PATH),
                                         f"{run_id}.tsv"))
    token = load_token()
    topics = load_topics(TOPICS_PATH)

    if not topics:
        print("No topics loaded, aborting.")
        return

    smoke_qid, smoke_text = topics[0]
    print(f"Smoke test on topic {smoke_qid} (hits={hits})...")
    try:
        candidates = get_candidates(smoke_qid, smoke_text, token, hits=hits)
        print(f"Smoke test OK: got {len(candidates)} candidates for topic {smoke_qid}.")
    except AuthError as e:
        print(f"Smoke test failed with auth error: {e}")
        print("Check PYSERINI_API_TOKEN in .env.local before continuing.")
        return
    except RetrieverError as e:
        print(f"Smoke test failed: {e}")
        return

    all_lines = []
    failed_qids = []

    for qid, text in topics:
        try:
            candidates = get_candidates(qid, text, token, hits=hits)
        except AuthError as e:
            print(f"Aborting: auth error on topic {qid}: {e}")
            return
        except RetrieverError as e:
            print(f"Topic {qid} failed: {e}")
            failed_qids.append(qid)
            continue

        for c in candidates:
            all_lines.append(
                format_run_line(qid, c["docid"], c["rank"], c["score"], run_id)
            )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    write_run_file(output_path, all_lines)

    print(f"Wrote {len(all_lines)} lines to {output_path}")
    print(f"Topics succeeded: {len(topics) - len(failed_qids)} / {len(topics)}")
    if failed_qids:
        print(f"Topics failed: {failed_qids}")


if __name__ == "__main__":
    main()
