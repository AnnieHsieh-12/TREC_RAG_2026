#!/usr/bin/env python3
"""深化切點：從（原始池、深度 CE 重排後池）產出最終 R 提交檔。

    python3 tools/apply_deep_cut.py <原始池.trec> <重排後池.trec> --out <r_output.tsv>

規則與 dev22 提交檔逐字節一致（已驗證：22 題集合與順序 22/22 重現）：
  k    = clamp( #{ 池分數 s ≥ 0.2 × rank-1 的分數 }, 4, 1000 )
  列   = 重排後池序的前 k 篇
  分數 = 1 − rank/1e6，固定六位小數（合成遞減，滿足官方 score 不遞增規定）

⚠️ τ 的錨點是 **rank-1 那篇的分數，不是全池最大分**。splice 之後池分數
不隨名次單調，兩種錨點在 dev22 差出 7 題的 k —— 錨點寫錯就不是同一個系統。
"""
import argparse
import collections
import sys

ap = argparse.ArgumentParser()
ap.add_argument("pool", help="原始候選池 candidate_pool_top5000.trec（切點的分數來源）")
ap.add_argument("reranked", help="深度 CE 重排後的池（列的順序來源）")
ap.add_argument("--out", required=True)
ap.add_argument("--tau", type=float, default=0.2)
ap.add_argument("--min", type=int, default=4)
ap.add_argument("--max", type=int, default=1000)
ap.add_argument("--tag", default="cfda-integrated")


def load(path):
    d = collections.defaultdict(list)
    for line in open(path):
        x = line.split()
        if len(x) >= 5:
            d[x[0]].append((int(x[3]), x[2], float(x[4])))
    return {q: sorted(v) for q, v in d.items()}


def main():
    a = ap.parse_args()
    pool, rr = load(a.pool), load(a.reranked)
    if set(pool) != set(rr):
        sys.exit(f"✗ 兩份池的題目集不同：只在原始 {sorted(set(pool)-set(rr))[:3]}｜"
                 f"只在重排 {sorted(set(rr)-set(pool))[:3]}")
    out, ks = [], []
    for q in sorted(pool, key=lambda z: (len(z), z)):
        scores = [s for _, _, s in pool[q]]
        top = scores[0]                      # rank-1 的分數（見檔頭警告）
        k = max(a.min, min(a.max, sum(1 for s in scores if s >= a.tau * top)))
        docs = [d for _, d, _ in rr[q]][:k]
        if len(docs) < k:
            print(f"⚠ {q}: 重排池只有 {len(docs)} 篇 < k={k}")
        ks.append(len(docs))
        for i, d in enumerate(docs):
            out.append(f"{q} Q0 {d} {i + 1} {1.0 - (i + 1) / 1e6:.6f} {a.tag}")
    with open(a.out, "w") as f:
        f.write("\n".join(out) + "\n")
    ks.sort()
    import statistics
    print(f"{a.out}｜{len(ks)} 題 {len(out)} 列｜k {ks[0]}–{ks[-1]}，"
          f"中位 {statistics.median(ks):g}，相異 {len(set(ks))} 種，平均 {statistics.mean(ks):.0f}")


if __name__ == "__main__":
    main()
