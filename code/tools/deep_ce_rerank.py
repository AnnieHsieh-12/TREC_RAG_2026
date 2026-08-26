#!/usr/bin/env python3
"""Extend cross-encoder reranking below the protected head.

    python tools/deep_ce_rerank.py out/final-retrieval \
        --depth 5000 --head 100 --out out/final-retrieval/deepce

為什麼要做
----------
相關文件的分佈（22 題、三份 qrels、rel>=2）：

    rank 1–1000     27.9%   ← 現在拿得到分的（池 R@1000 = 0.2937）
    rank 1001–5000  19.1%   ← 撈到了，但官方每題上限 1000 篇，交不出去
    從沒撈到        52.9%

而完美檢索在 depth 1000 的天花板是 0.9918 —— 也就是說 1000 這個上限幾乎沒卡到我們，
recall 低是因為**排序**：rank 101 以後完全沒有語意訊號，只有 BM25 三軌的融合分數。
CE 只跑了 head-100。把它延伸到池底，就有機會把那 19.1% 拉進前 1000。

關鍵：重排「1–1000 之內」對 R@1000 毫無作用
-------------------------------------------
Recall@1000 只看哪 1000 篇在集合裡，不看順序。所以 CE 必須看得到 rank 1001+ 的文件，
才可能把它們拉上來 —— depth 一定要遠大於 1000。這是這支跟一般重排腳本最大的差別。

保護 head 的理由
----------------
head-100 已經過三軌重排（BM25 名次 ⊕ MiniLM CE ⊕ bge-m3）+ 校準，nDCG@10 = 0.7456。
單一 CE 重排它只會變差。所以預設 --head 100 原序保留，只重排 101 以後 ——
nDCG@10 構造上完全不動，這支腳本的增益純粹來自 recall/MAP。

已知限制：只看文件開頭
----------------------
快取裡的文件平均 12,202 字元，而 CE 的 max_length=512 token（約 2,000 字元），
所以每篇只讀到前 ~16%。相關段落若在後半就看不到，實際增益會低於 oracle 的 +0.196。
要修得切段落逐段打分再取最大值，成本乘上段數 —— 先量這一版，不夠再說。

輸出的是候選池格式（qid Q0 docid rank score tag），變動 k 的切點另外套。
"""
import argparse
import collections
import glob
import hashlib
import json
import math
import os
import statistics

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
INDEX = os.environ.get("PYSERINI_INDEX", "climbmix-400b")
CACHE = os.environ.get("DOC_CACHE_DIR", os.path.join(".cache", "docs"))
QD = os.environ.get(
    "QRELS_DIR",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "qrels")),
)
RRF_K = 60


def cache_path(docid: str) -> str:
    """與 src/trec-rag-2026/retrieval/doc_cache.ts 同一套雜湊分桶，兩邊共用同一份快取。

    分隔字元是 NUL（\\x00）不是空格 —— doc_cache.ts 第 40 行的模板字串裡是個
    不可見的 0x00。用空格會算出完全不同的雜湊、一篇都讀不到，而且因為 read_doc
    對讀不到的檔是回空字串，錯誤會靜默（CE 收到一堆空文字照樣跑完、給出假結果）。
    """
    h = hashlib.sha1(f"{INDEX}\x00{docid}".encode()).hexdigest()
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in INDEX)
    return os.path.join(CACHE, safe, h[:2], f"{h}.json")


def read_doc(docid: str) -> str:
    try:
        with open(cache_path(docid)) as f:
            v = json.load(f)
        return v.get("text", "") if v.get("found") else ""
    except Exception:
        return ""


def load_pool(run: str):
    by = collections.defaultdict(list)
    with open(os.path.join(run, "candidate_pool_top5000.trec")) as f:
        for line in f:
            x = line.split()
            if len(x) >= 5:
                by[x[0]].append((int(x[3]), x[2], float(x[4])))
    return {q: [(d, s) for _, d, s in sorted(v)] for q, v in by.items()}


def load_narratives(run: str):
    out = {}
    with open(os.path.join(run, "retrieval_trace.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            out[str(r["topic_id"])] = r["anchor_query"]
    return out


def load_qrels():
    out = []
    for p in sorted(glob.glob(os.path.join(QD, "*.qrels"))):
        if "additions" in os.path.basename(p):
            continue
        rel = collections.defaultdict(dict)
        with open(p) as f:
            for line in f:
                x = line.split()
                if len(x) >= 4:
                    rel[x[0]][x[2]] = int(x[3])
        out.append(rel)
    return out


def evaluate(sel, qrels, cut):
    """三份 qrels 各算一次再平均。nDCG 用線性 gain、rel<2 歸零（見 VERSIONS.md §2.2）。"""
    agg = []
    for rel in qrels:
        R, N, M = [], [], []
        for q, docs in sel.items():
            g = rel.get(q, {})
            gold = {d for d, v in g.items() if v >= 2}
            if not gold:
                continue
            top = docs[:cut]
            R.append(len(gold & set(top)) / len(gold))
            dcg = sum((g.get(d, 0) if g.get(d, 0) >= 2 else 0) / math.log2(i + 2)
                      for i, d in enumerate(top[:10]))
            ideal = sorted((v for v in g.values() if v >= 2), reverse=True)[:10]
            idcg = sum(v / math.log2(i + 2) for i, v in enumerate(ideal))
            N.append(dcg / idcg if idcg else 0.0)
            hits, ap = 0, 0.0
            for i, d in enumerate(top):
                if d in gold:
                    hits += 1
                    ap += hits / (i + 1)
            M.append(ap / len(gold))
        agg.append((statistics.mean(N), statistics.mean(R), statistics.mean(M)))
    return tuple(statistics.mean(x[i] for x in agg) for i in range(3))


def per_topic_recall(sel, qrels, cut):
    out = {}
    for q, docs in sel.items():
        vals = []
        for rel in qrels:
            gold = {d for d, v in rel.get(q, {}).items() if v >= 2}
            if gold:
                vals.append(len(gold & set(docs[:cut])) / len(gold))
        if vals:
            out[q] = statistics.mean(vals)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--depth", type=int, default=5000)
    ap.add_argument("--head", type=int, default=100, help="原序保護的前 N 名，不參與重排")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--maxlen", type=int, default=512)
    ap.add_argument("--threads", type=int, default=64)
    ap.add_argument("--out", default="")
    ap.add_argument("--variant", default="", help="直接指定變體（如 'RRF 1:1'），跳過 qrels 評估 —— 官方題必用")
    args = ap.parse_args()

    pool = load_pool(args.run)
    narr = load_narratives(args.run)
    qrels = load_qrels()

    # 這台的 GPU 是 Blackwell（sm_120），機器上每一份 torch 都只編到 sm_90，
    # .to("cuda") 會炸 "no kernel image is available"。CPU 實測 57.5 對/秒
    # （64 執行緒），全池 108,622 對約 31 分鐘，夠用，所以不折騰裝 cu128。
    dev = "cpu"
    torch.set_num_threads(args.threads)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL).to(dev).eval()

    ce_order, missing_total, scored_total = {}, 0, 0
    for qid in sorted(pool, key=lambda x: (len(x), x)):
        docs = [d for d, _ in pool[qid][:args.depth]]
        tail = docs[args.head:]
        texts = [read_doc(d) for d in tail]
        have = [(d, t) for d, t in zip(tail, texts) if t]
        missing_total += len(tail) - len(have)
        scores = []
        with torch.inference_mode():
            for i in range(0, len(have), args.batch):
                chunk = have[i:i + args.batch]
                enc = tok([narr[qid]] * len(chunk), [t for _, t in chunk],
                          padding=True, truncation=True, max_length=args.maxlen,
                          return_tensors="pt").to(dev)
                scores.extend(model(**enc).logits.squeeze(-1).float().cpu().tolist())
        scored_total += len(have)
        # 沒有全文的文件不能給 CE 分數，也不該被踢掉 —— 讓它們保持原相對順序排在後面。
        have_ids = {d for d, _ in have}
        ranked = [d for d, _ in sorted(zip([d for d, _ in have], scores), key=lambda x: -x[1])]
        ranked += [d for d in tail if d not in have_ids]
        ce_order[qid] = ranked
        print(f"  {qid}: 重排 {len(have)} 篇（缺全文 {len(tail) - len(have)}）", flush=True)

    print(f"\n重排 {scored_total:,} 篇，缺全文 {missing_total:,} 篇 "
          f"（{missing_total / max(1, scored_total + missing_total):.1%}）\n")

    variants = {}
    for label, w_pool, w_ce in [("原始（BM25 融合）", 1.0, 0.0),
                                ("RRF 1:1", 1.0, 1.0),
                                ("RRF 1:2 偏 CE", 1.0, 2.0),
                                ("純 CE", 0.0, 1.0)]:
        sel = {}
        for qid in pool:
            docs = [d for d, _ in pool[qid][:args.depth]]
            head, tail = docs[:args.head], docs[args.head:]
            pr = {d: i for i, d in enumerate(tail)}
            cr = {d: i for i, d in enumerate(ce_order[qid])}
            fused = sorted(tail, key=lambda d: -(w_pool / (RRF_K + pr[d] + 1)
                                                 + w_ce / (RRF_K + cr.get(d, len(tail)) + 1)))
            sel[qid] = head + fused
        variants[label] = sel

    # 官方 119 題沒有 qrels：--variant 直接指定（dev 已驗證 RRF 1:1 最優），
    # 跳過整段評估與配對檢定 —— 那些都要 qrels 才有意義。
    if args.variant:
        assert args.variant in variants, f"未知 variant：{list(variants)}"
        best = args.variant
        if args.out:
            os.makedirs(args.out, exist_ok=True)
            p = os.path.join(args.out, "candidate_pool_top5000.trec")
            with open(p, "w") as f:
                for qid in sorted(variants[best]):
                    for i, docid in enumerate(variants[best][qid], 1):
                        f.write(f"{qid} Q0 {docid} {i} {1.0 - i * 1e-6:.6f} cfda-deepce\n")
            print(f"寫出（{best}，未經 qrels 評估）：{p}")
        return

    print(f"{'設定':<20}{'nDCG@10':>10}{'R@1000':>10}{'MAP@1000':>11}{'Δ Recall':>11}")
    print("─" * 62)
    base = None
    for label, sel in variants.items():
        n, r, m = evaluate(sel, qrels, 1000)
        if base is None:
            base = r
        print(f"{label:<20}{n:>10.4f}{r:>10.4f}{m:>11.4f}{r - base:>+11.4f}")
    print(f"\n對照：完美重排的上限 R@1000 = 0.4895（池 R@5000），理論天花板 0.9918。")

    best = max(variants, key=lambda k: evaluate(variants[k], qrels, 1000)[1])
    print(f"\n逐題配對：{best} vs 原始（R@1000）")
    a = per_topic_recall(variants["原始（BM25 融合）"], qrels, 1000)
    b = per_topic_recall(variants[best], qrels, 1000)
    d = [b[q] - a[q] for q in a]
    w = sum(1 for x in d if x > 1e-9)
    l = sum(1 for x in d if x < -1e-9)
    se = statistics.stdev(d) / math.sqrt(len(d)) if len(d) > 1 else 0.0
    print(f"  平均 {statistics.mean(d):+.4f}  95%CI [{statistics.mean(d) - 1.96 * se:+.4f}, "
          f"{statistics.mean(d) + 1.96 * se:+.4f}]  {w} 勝 {l} 敗")
    top = sorted(d, reverse=True)[:1]
    print(f"  最大單題增益 {top[0]:+.4f}（佔總平均的 {top[0] / len(d) / max(1e-9, statistics.mean(d)):.0%}）")

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        p = os.path.join(args.out, "candidate_pool_top5000.trec")
        with open(p, "w") as f:
            for qid in sorted(variants[best]):
                for i, docid in enumerate(variants[best][qid], 1):
                    f.write(f"{qid} Q0 {docid} {i} {1.0 - i * 1e-6:.6f} cfda-deepce\n")
        print(f"\n寫出（{best}）：{p}")


if __name__ == "__main__":
    main()
