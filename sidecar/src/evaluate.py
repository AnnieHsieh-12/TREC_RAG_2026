import os
import sys

import pytrec_eval

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import QRELS_DIR, QRELS_FILES

MEASURES = {"ndcg_cut_10", "ndcg_cut_20", "ndcg_cut_100", "recall_100", "recall_1000"}


def load_qrels(path):
    qrels = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            fields = line.split()
            if len(fields) != 4:
                continue
            qid, _, docid, rel = fields
            qrels.setdefault(qid, {})[docid] = int(rel)
    return qrels


def load_run(path):
    run = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            fields = line.split()
            if len(fields) != 6:
                continue
            qid, _, docid, _, score, _ = fields
            run.setdefault(qid, {})[docid] = float(score)
    return run


def evaluate_one(qrels_path, run):
    qrels = load_qrels(qrels_path)
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, MEASURES)
    results = evaluator.evaluate(run)

    n = len(results)
    if n == 0:
        return {m: 0.0 for m in MEASURES}

    totals = {m: 0.0 for m in MEASURES}
    for qid_scores in results.values():
        for m in MEASURES:
            totals[m] += qid_scores[m]

    return {m: totals[m] / n for m in MEASURES}


def main():
    if len(sys.argv) < 2:
        print("usage: python src/evaluate.py <run_file> [qrels_file ...]")
        print("  with no qrels args: evaluates against the 3 config.QRELS_FILES")
        sys.exit(1)

    run = load_run(sys.argv[1])

    if len(sys.argv) > 2:
        qrels_paths = sys.argv[2:]
    else:
        qrels_paths = [os.path.join(QRELS_DIR, f) for f in QRELS_FILES]

    for path in qrels_paths:
        scores = evaluate_one(path, run)
        print(f"\n{os.path.basename(path)}")
        for measure in sorted(MEASURES):
            print(f"  {measure}: {scores[measure]:.4f}")


if __name__ == "__main__":
    main()
