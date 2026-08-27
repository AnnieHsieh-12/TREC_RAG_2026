#!/usr/bin/env python3
"""Create a variable-depth Retrieval run from scored and reranked pools.

For each topic, k is the number of original-pool scores at least ``tau`` times
the rank-1 score, clamped to the configured range. Rows follow reranked order;
synthetic monotonically decreasing scores preserve that order in TREC format.
"""
import argparse
import collections
import sys

ap = argparse.ArgumentParser()
ap.add_argument("pool", help="original candidate pool (source of cutoff scores)")
ap.add_argument("reranked", help="reranked pool (source of output ordering)")
ap.add_argument("--out", required=True)
ap.add_argument("--tau", type=float, default=0.2)
ap.add_argument("--min", type=int, default=4)
ap.add_argument("--max", type=int, default=1000)
ap.add_argument("--tag", default="cfda-integrated")


def load(path):
    d = collections.defaultdict(list)
    with open(path, encoding="utf-8") as source:
        for line in source:
            x = line.split()
            if len(x) >= 5:
                d[x[0]].append((int(x[3]), x[2], float(x[4])))
    return {q: sorted(v) for q, v in d.items()}


def main():
    a = ap.parse_args()
    pool, rr = load(a.pool), load(a.reranked)
    if set(pool) != set(rr):
        sys.exit(f"topic sets differ: original only {sorted(set(pool)-set(rr))[:3]}; "
                 f"reranked only {sorted(set(rr)-set(pool))[:3]}")
    out, ks = [], []
    for q in sorted(pool, key=lambda z: (len(z), z)):
        scores = [s for _, _, s in pool[q]]
        top = scores[0]
        k = max(a.min, min(a.max, sum(1 for s in scores if s >= a.tau * top)))
        docs = [d for _, d, _ in rr[q]][:k]
        if len(docs) < k:
            print(f"warning: {q} has only {len(docs)} reranked documents for k={k}")
        ks.append(len(docs))
        for i, d in enumerate(docs):
            out.append(f"{q} Q0 {d} {i + 1} {1.0 - (i + 1) / 1e6:.6f} {a.tag}")
    with open(a.out, "w") as f:
        f.write("\n".join(out) + "\n")
    ks.sort()
    import statistics
    print(f"{a.out}: {len(ks)} topics, {len(out)} rows; k {ks[0]}-{ks[-1]}, "
          f"median {statistics.median(ks):g}, {len(set(ks))} distinct, mean {statistics.mean(ks):.0f}")


if __name__ == "__main__":
    main()
