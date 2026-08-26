import json
import sys

REQUIRED_TOP_LEVEL_FIELDS = {"metadata", "references", "answer"}
REQUIRED_METADATA_FIELDS = {
    "team_id",
    "narrative_id",
    "narrative",
    "run_id",
    "run_desc",
}
MAX_ANSWER_WORDS = 1024
MAX_CITATIONS_PER_SENTENCE = 3


def validate(path):
    errors = []

    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.rstrip("\n")
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"line {lineno}: invalid JSON ({e})")
                continue

            if not isinstance(obj, dict):
                errors.append(f"line {lineno}: top-level JSON value is not an object")
                continue

            for field in REQUIRED_TOP_LEVEL_FIELDS:
                if field not in obj:
                    errors.append(f"line {lineno}: missing top-level field '{field}'")
            if not REQUIRED_TOP_LEVEL_FIELDS.issubset(obj):
                continue

            metadata = obj["metadata"]
            if not isinstance(metadata, dict):
                errors.append(f"line {lineno}: 'metadata' is not an object")
                continue
            for field in REQUIRED_METADATA_FIELDS:
                if field not in metadata:
                    errors.append(f"line {lineno}: metadata missing '{field}'")
            # skills v0.6.0 (2026-07-25): participant-defined extra metadata
            # fields are explicitly welcome; only the five required keys are
            # validated.
            for field in REQUIRED_METADATA_FIELDS:
                value = metadata.get(field)
                if not isinstance(value, str) or not value.strip():
                    errors.append(
                        f"line {lineno}: metadata.{field} must be a non-empty string"
                    )

            references = obj["references"]
            answer = obj["answer"]
            narrative_id = metadata.get("narrative_id", "?")

            if not isinstance(references, list):
                errors.append(f"line {lineno} (topic {narrative_id}): 'references' is not a list")
                continue
            if not isinstance(answer, list):
                errors.append(f"line {lineno} (topic {narrative_id}): 'answer' is not a list")
                continue

            cited_indices = set()
            answer_word_count = 0
            for i, sentence in enumerate(answer):
                if not isinstance(sentence, dict):
                    errors.append(
                        f"line {lineno} (topic {narrative_id}): answer[{i}] is not an object"
                    )
                    continue
                if "text" not in sentence or "citations" not in sentence:
                    errors.append(
                        f"line {lineno} (topic {narrative_id}): answer[{i}] missing "
                        "'text' or 'citations'"
                    )
                    continue
                if not isinstance(sentence["text"], str) or not sentence["text"].strip():
                    errors.append(
                        f"line {lineno} (topic {narrative_id}): answer[{i}].text is empty"
                    )
                else:
                    answer_word_count += len(sentence["text"].split())
                citations = sentence["citations"]
                if not isinstance(citations, list):
                    errors.append(
                        f"line {lineno} (topic {narrative_id}): answer[{i}].citations "
                        "is not a list"
                    )
                    continue
                # 2026-07-29 rag-task.md update: an empty citations array is
                # structurally valid (support score 0 for weighted recall,
                # excluded from citation precision) - do not reject it.
                if len(citations) > MAX_CITATIONS_PER_SENTENCE:
                    errors.append(
                        f"line {lineno} (topic {narrative_id}): answer[{i}] has "
                        f"{len(citations)} citations; maximum is "
                        f"{MAX_CITATIONS_PER_SENTENCE}"
                    )
                # skills v0.6.0: a citation may be a zero-based integer index
                # OR a docid string that exactly matches a references entry.
                if len(set(map(repr, citations))) != len(citations):
                    errors.append(
                        f"line {lineno} (topic {narrative_id}): answer[{i}] has "
                        "duplicate citations"
                    )
                for c in citations:
                    if type(c) is int and 0 <= c < len(references):
                        cited_indices.add(c)
                    elif isinstance(c, str) and c in references:
                        cited_indices.add(references.index(c))
                    else:
                        errors.append(
                            f"line {lineno} (topic {narrative_id}): answer[{i}] citation "
                            f"{c!r} is neither a valid index nor a docid present in "
                            f"references (len={len(references)})"
                        )

            if answer_word_count > MAX_ANSWER_WORDS:
                errors.append(
                    f"line {lineno} (topic {narrative_id}): answer has "
                    f"{answer_word_count} words; maximum is {MAX_ANSWER_WORDS}"
                )

            # skills v0.6.0: uncited references are allowed and do not hurt
            # the score - report as information, never as an error.
            uncited = set(range(len(references))) - cited_indices
            if uncited:
                print(
                    f"note: line {lineno} (topic {narrative_id}): "
                    f"{len(uncited)} uncited reference(s) (allowed under v0.6.0)"
                )

            string_references = [docid for docid in references if isinstance(docid, str)]
            if len(set(string_references)) != len(string_references):
                errors.append(
                    f"line {lineno} (topic {narrative_id}): 'references' contains duplicates"
                )
            for i, docid in enumerate(references):
                if not isinstance(docid, str) or not docid.strip():
                    errors.append(
                        f"line {lineno} (topic {narrative_id}): references[{i}] "
                        "must be a non-empty string"
                    )

    return errors


def main():
    if len(sys.argv) < 2:
        print("usage: python src/validate_rag_output.py <rag_output_file>")
        sys.exit(1)

    errors = validate(sys.argv[1])

    if errors:
        print(f"FAILED: {len(errors)} issue(s) found")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)

    print("All checks passed.")


if __name__ == "__main__":
    main()
