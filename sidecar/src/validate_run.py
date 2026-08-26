import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import TOPICS_PATH
from src.common import load_topics


def validate(run_path, topics_path=TOPICS_PATH):
    errors = []
    by_qid = defaultdict(list)  # qid -> list of (rank, docid, score)

    with open(run_path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.rstrip("\n")
            if not line:
                continue
            fields = line.split()
            if len(fields) != 6:
                errors.append(f"line {lineno}: expected 6 fields, got {len(fields)}")
                continue

            qid, q0, docid, rank, score, run_id = fields
            if q0 != "Q0":
                errors.append(f"line {lineno}: second field is {q0!r}, expected 'Q0'")

            try:
                rank_int = int(rank)
            except ValueError:
                errors.append(f"line {lineno}: rank {rank!r} is not an integer")
                continue

            try:
                score_val = float(score)
            except ValueError:
                errors.append(f"line {lineno}: score {score!r} is not a number")
                continue

            by_qid[qid].append((rank_int, docid, score_val))

    expected_qids = {qid for qid, _ in load_topics(topics_path)}
    seen_qids = set(by_qid.keys())
    missing_qids = expected_qids - seen_qids
    if missing_qids:
        errors.append(f"missing topics in run file: {sorted(missing_qids)}")

    for qid, rows in by_qid.items():
        ranks = [r[0] for r in rows]
        docids = [r[1] for r in rows]
        scores = [r[2] for r in rows]

        if any(r < 1 or r > 100 for r in ranks):
            errors.append(f"topic {qid}: rank out of 1-100 range")

        if len(set(ranks)) != len(ranks):
            errors.append(f"topic {qid}: duplicate ranks")
        elif sorted(ranks) != list(range(1, len(ranks) + 1)):
            errors.append(f"topic {qid}: ranks are not contiguous starting at 1")

        if len(set(docids)) != len(docids):
            errors.append(f"topic {qid}: duplicate docids")

        rows_by_rank = sorted(rows, key=lambda r: r[0])
        scores_by_rank = [r[2] for r in rows_by_rank]
        if any(
            scores_by_rank[i] < scores_by_rank[i + 1]
            for i in range(len(scores_by_rank) - 1)
        ):
            errors.append(f"topic {qid}: scores are not non-increasing by rank")

    return errors


def main():
    if len(sys.argv) < 2:
        print("usage: python src/validate_run.py <run_file>")
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
