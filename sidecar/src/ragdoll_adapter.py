import json
import sys

# M3: RAGDoll's TREC-answer reader (RAGDoll's src/ragdoll/trec_io.py,
# read_trec_answers()) looks for qid/topic_id/topic/query at the row's TOP
# LEVEL, not inside our official schema's nested `metadata.narrative_id`/
# `metadata.narrative`. This adapter copies those two fields up to the top
# level so RAGDoll can read our official output directly - it does not
# rename, remove, or otherwise modify anything in `metadata`, `references`,
# or `answer`, so the row still validates against our own
# src/validate_rag_output.py schema unchanged.


def to_ragdoll_row(record):
    metadata = record.get("metadata", {})
    narrative_id = metadata.get("narrative_id", "")
    narrative = metadata.get("narrative", "")
    return {
        **record,
        "qid": narrative_id,
        "topic_id": narrative_id,
        "query": narrative,
        "topic": narrative,
    }


def convert_file(input_path, output_path):
    written = 0
    with open(input_path, encoding="utf-8") as f_in, \
            open(output_path, "w", encoding="utf-8") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            f_out.write(json.dumps(to_ragdoll_row(record), ensure_ascii=False) + "\n")
            written += 1
    return written


def main():
    if len(sys.argv) != 3:
        print("usage: python src/ragdoll_adapter.py <input.jsonl> <output.jsonl>")
        sys.exit(1)
    written = convert_file(sys.argv[1], sys.argv[2])
    print(f"wrote {written} row(s) to {sys.argv[2]}")


if __name__ == "__main__":
    main()
