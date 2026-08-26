#!/usr/bin/env python3
"""Extend a deep-reranked prefix with the untouched candidate-pool tail.

Documents absent from the reranked prefix retain their original relative order.
Synthetic monotonically decreasing scores preserve the merged ordering.
"""
import argparse
import collections

ap = argparse.ArgumentParser()
ap.add_argument("pool", help="original candidate pool")
ap.add_argument("deep", help="deep cross-encoder reranked prefix")
ap.add_argument("--out", required=True)
ap.add_argument("--tag", default="merged")


def load(path):
    d = collections.defaultdict(list)
    for line in open(path):
        x = line.split()
        if len(x) >= 5:
            d[x[0]].append((int(x[3]), x[2]))
    return {q: [docid for _, docid in sorted(v)] for q, v in d.items()}


def main():
    a = ap.parse_args()
    pool, deep = load(a.pool), load(a.deep)
    missing = sorted(set(pool) - set(deep))
    if missing:
        print(f"warning: {len(missing)} topics have no deep reranking; using pool order (for example {missing[:3]})")

    n_rows = n_deep = 0
    with open(a.out, "w") as f:
        for q in pool:
            head = deep.get(q, [])
            seen = set(head)
            order = head + [d for d in pool[q] if d not in seen]
            n_deep += len(head)
            for rank, docid in enumerate(order, 1):
                f.write(f"{q} Q0 {docid} {rank} {1 - rank / 1e6:.6f} {a.tag}\n")
                n_rows += 1
    print(f"{a.out}: {len(pool)} topics, {n_rows} rows; {n_deep} deep-reranked and "
          f"{n_rows - n_deep} original-tail rows")


if __name__ == "__main__":
    main()
