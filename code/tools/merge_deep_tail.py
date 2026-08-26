#!/usr/bin/env python3
"""把深度 CE 重排的前綴接上原始池的尾巴，產出「全深度」的排序來源。

    python3 tools/merge_deep_tail.py <原始池.trec> <深度CE池.trec> --out <merged.trec>

為什麼需要這一步
----------------
深度 CE 只重排到 rank 3000（每題實際 2900 篇），但候選池有到 5000。
`apply_deep_cut.py` 的列來源是「重排後池序」，直接餵深度 CE 的輸出會在 3000 截斷——
不設限的 `cfda-vfs-deep` 就會少掉 9,211 列（167,013 vs 官方 176,224）。

官方送出的 `cfda-vfs-deep` 是：**深度 CE 排序放前面，池裡沒被重排到的文件按原池序接在後面**。
本工具就是這一步。接完再交給 `apply_deep_cut.py` 決定每題的 k。

驗證：本工具 + `apply_deep_cut.py --max 5000` 可以把
`artifacts/submission/cfda-vfs-deep/r_output_trec_rag_2026.tsv` 逐位元組重現。
（原本這一步是收官當天手打的指令，樹裡沒有腳本——見 MAPPING.md「重現落差」。）

輸出的分數是 1 − rank/1e6 的合成遞減值，只用來保住順序；
真正決定 k 的分數一律取自原始池，不經過這裡。
"""
import argparse
import collections

ap = argparse.ArgumentParser()
ap.add_argument("pool", help="原始候選池 candidate_pool_top5000.trec")
ap.add_argument("deep", help="深度 CE 重排後的池 deepce/candidate_pool_top5000.trec")
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
        print(f"注意：{len(missing)} 題沒有深度 CE 結果，這些題完全照原池序（例：{missing[:3]}）")

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
    print(f"{a.out}｜{len(pool)} 題 {n_rows} 列｜其中 {n_deep} 列來自深度 CE，"
          f"{n_rows - n_deep} 列是原池尾巴")


if __name__ == "__main__":
    main()
