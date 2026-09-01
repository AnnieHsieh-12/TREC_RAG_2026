#!/usr/bin/env python3
"""Extend cross-encoder reranking below a protected ranking head.

    python tools/deep_ce_rerank.py out/cfda-retrieval \
        --depth 3000 --head 100 --out out/cfda-retrieval/deepce

The first ``--head`` documents retain their input order. The remaining
documents are cross-encoder scored and RRF-fused with their pool order. Output
uses the standard ``qid Q0 docid rank score tag`` candidate-pool format.

The model reads a truncated document prefix (``--maxlen`` tokens), so relevant
evidence outside that prefix cannot affect its score.
"""
import argparse
import collections
import concurrent.futures
import glob
import hashlib
import json
import math
import os
import statistics
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
CODE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def load_local_env(path):
    """Load simple KEY=VALUE entries without printing or overwriting env."""
    try:
        with open(path, encoding="utf-8") as env_file:
            for raw in env_file:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key, value = key.strip(), value.strip()
                if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                    value = value[1:-1]
                if key and value:
                    os.environ.setdefault(key, value)
    except FileNotFoundError:
        pass


load_local_env(os.path.join(CODE_ROOT, ".env.local"))
INDEX = os.environ.get("PYSERINI_INDEX", "climbmix-400b")
CACHE = os.environ.get("DOC_CACHE_DIR", os.path.join(".cache", "docs"))
BASE_URL = os.environ.get("PYSERINI_BASE_URL", "http://api.castorini.uwaterloo.ca")
QD = os.environ.get(
    "QRELS_DIR",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "qrels")),
)
RRF_K = 60
_fetch_lock = threading.Lock()
_last_fetch_at = 0.0


def cache_path(docid: str) -> str:
    """Return the path used by the shared TypeScript document cache."""
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


def write_doc(docid, found, text):
    path = cache_path(docid)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=os.path.dirname(path), delete=False
        ) as output:
            temporary = output.name
            json.dump({"found": bool(found), "text": text if found else ""}, output)
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def extract_text(doc):
    if isinstance(doc, str):
        return doc
    if isinstance(doc, dict) and isinstance(doc.get("text"), str):
        return doc["text"]
    return json.dumps(doc if doc is not None else "", ensure_ascii=False)


def pace_fetch(seconds):
    global _last_fetch_at
    if seconds <= 0:
        return
    with _fetch_lock:
        now = time.monotonic()
        wait = _last_fetch_at + seconds - now
        if wait > 0:
            time.sleep(wait)
            now = time.monotonic()
        _last_fetch_at = now


def fetch_doc(docid, token, pace_seconds=0.1, max_retries=8):
    """Fetch one missing document and populate the shared TypeScript cache."""
    if read_doc(docid):
        return True
    url = (
        f"{BASE_URL.rstrip('/')}/v1/{urllib.parse.quote(INDEX, safe='')}/doc/"
        f"{urllib.parse.quote(docid, safe='')}?parse=true"
    )
    for attempt in range(1, max_retries + 1):
        pace_fetch(pace_seconds)
        request = urllib.request.Request(
            url, headers={"Authorization": f"Bearer {token}"}
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = json.loads(response.read().decode("utf-8"))
            text = extract_text(payload.get("doc"))
            if not text:
                return False
            write_doc(docid, True, text)
            return True
        except urllib.error.HTTPError as error:
            if error.code == 404:
                write_doc(docid, False, "")
                return False
            if error.code not in (429, 500, 502, 503, 504) or attempt == max_retries:
                return False
            retry_after = error.headers.get("Retry-After")
            delay = float(retry_after) if retry_after and retry_after.isdigit() else 0
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            if attempt == max_retries:
                return False
            delay = 0
        time.sleep(max(delay, min(60, 2 ** (attempt - 1))))
    return False


def prefetch_missing(pool, depth, head, token, workers, pace_seconds):
    needed = []
    seen = set()
    for qid in pool:
        for docid, _ in pool[qid][head:depth]:
            if docid not in seen and not read_doc(docid):
                seen.add(docid)
                needed.append(docid)
    if not needed:
        return
    if not token:
        raise RuntimeError(
            "PYSERINI_API_TOKEN is required to fetch uncached deep-tail documents"
        )
    print(f"Prefetching {len(needed):,} uncached deep-tail documents...", flush=True)
    completed = 0
    fetched = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(fetch_doc, docid, token, pace_seconds): docid
            for docid in needed
        }
        for future in concurrent.futures.as_completed(futures):
            completed += 1
            try:
                fetched += int(future.result())
            except Exception:
                pass
            if completed % 250 == 0 or completed == len(needed):
                print(
                    f"  fetched {fetched:,}/{completed:,}; "
                    f"remaining {len(needed) - completed:,}",
                    flush=True,
                )


def require_coverage(scored, missing, max_missing_fraction):
    total = scored + missing
    fraction = missing / max(1, total)
    if fraction > max_missing_fraction:
        raise RuntimeError(
            f"deep rerank aborted: missing text {missing:,}/{total:,} "
            f"({fraction:.1%}) exceeds --max-missing-fraction "
            f"{max_missing_fraction:.1%}"
        )


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
    """Average metrics across qrels using linear nDCG gain and rel >= 2."""
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
    ap.add_argument("--head", type=int, default=100, help="input-order prefix excluded from reranking")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--maxlen", type=int, default=512)
    ap.add_argument("--threads", type=int, default=64)
    ap.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="inference device; auto prefers CUDA, then MPS, then CPU",
    )
    ap.add_argument("--out", default="")
    ap.add_argument("--variant", default="", help="select a variant such as 'RRF 1:1' without qrels evaluation")
    ap.add_argument("--fetch-workers", type=int, default=4)
    ap.add_argument("--fetch-pace", type=float, default=0.1)
    ap.add_argument("--max-missing-fraction", type=float, default=0.01)
    ap.add_argument("--no-fetch-missing", action="store_true")
    args = ap.parse_args()

    if args.fetch_workers < 1:
        ap.error("--fetch-workers must be positive")
    if args.fetch_pace < 0:
        ap.error("--fetch-pace must be non-negative")
    if not 0 <= args.max_missing_fraction <= 1:
        ap.error("--max-missing-fraction must be between 0 and 1")

    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as error:
        ap.error(
            "deep reranking dependencies are missing; install requirements.lock"
        )

    pool = load_pool(args.run)
    narr = load_narratives(args.run)
    qrels = load_qrels()
    if not args.no_fetch_missing:
        prefetch_missing(
            pool,
            args.depth,
            args.head,
            os.environ.get("PYSERINI_API_TOKEN", "").strip(),
            args.fetch_workers,
            args.fetch_pace,
        )

    if args.device == "auto":
        if torch.cuda.is_available():
            dev = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            dev = "mps"
        else:
            dev = "cpu"
    else:
        dev = args.device
    if dev == "cuda" and not torch.cuda.is_available():
        ap.error("--device cuda requested, but CUDA is unavailable")
    if dev == "mps" and not (
        getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    ):
        ap.error("--device mps requested, but MPS is unavailable")
    print(f"Using inference device: {dev}", flush=True)
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
        # Documents missing cached text retain their relative order at the end.
        have_ids = {d for d, _ in have}
        ranked = [d for d, _ in sorted(zip([d for d, _ in have], scores), key=lambda x: -x[1])]
        ranked += [d for d in tail if d not in have_ids]
        ce_order[qid] = ranked
        print(f"  {qid}: reranked {len(have)} documents; missing text {len(tail) - len(have)}", flush=True)

    print(f"\nReranked {scored_total:,} documents; missing text {missing_total:,} "
          f"({missing_total / max(1, scored_total + missing_total):.1%})\n")
    require_coverage(scored_total, missing_total, args.max_missing_fraction)

    variants = {}
    for label, w_pool, w_ce in [("Original pool", 1.0, 0.0),
                                ("RRF 1:1", 1.0, 1.0),
                                ("RRF 1:2", 1.0, 2.0),
                                ("CE only", 0.0, 1.0)]:
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

    # Without qrels, use the explicitly selected variant and skip evaluation.
    if args.variant:
        assert args.variant in variants, f"unknown variant: {list(variants)}"
        best = args.variant
        if args.out:
            os.makedirs(args.out, exist_ok=True)
            p = os.path.join(args.out, "candidate_pool_top5000.trec")
            with open(p, "w") as f:
                for qid in sorted(variants[best]):
                    for i, docid in enumerate(variants[best][qid], 1):
                        f.write(f"{qid} Q0 {docid} {i} {1.0 - i * 1e-6:.6f} cfda-deepce\n")
            print(f"Wrote {best} without qrels evaluation: {p}")
        return

    print(f"{'Variant':<20}{'nDCG@10':>10}{'R@1000':>10}{'MAP@1000':>11}{'Delta R':>11}")
    print("─" * 62)
    base = None
    for label, sel in variants.items():
        n, r, m = evaluate(sel, qrels, 1000)
        if base is None:
            base = r
        print(f"{label:<20}{n:>10.4f}{r:>10.4f}{m:>11.4f}{r - base:>+11.4f}")
    best = max(variants, key=lambda k: evaluate(variants[k], qrels, 1000)[1])
    print(f"\nPer-topic paired comparison: {best} vs original (R@1000)")
    a = per_topic_recall(variants["Original pool"], qrels, 1000)
    b = per_topic_recall(variants[best], qrels, 1000)
    d = [b[q] - a[q] for q in a]
    w = sum(1 for x in d if x > 1e-9)
    l = sum(1 for x in d if x < -1e-9)
    se = statistics.stdev(d) / math.sqrt(len(d)) if len(d) > 1 else 0.0
    print(f"  mean {statistics.mean(d):+.4f}  95% CI [{statistics.mean(d) - 1.96 * se:+.4f}, "
          f"{statistics.mean(d) + 1.96 * se:+.4f}]  {w} wins, {l} losses")
    top = sorted(d, reverse=True)[:1]
    print(f"  largest topic gain {top[0]:+.4f} "
          f"({top[0] / len(d) / max(1e-9, statistics.mean(d)):.0%} of the mean)")

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        p = os.path.join(args.out, "candidate_pool_top5000.trec")
        with open(p, "w") as f:
            for qid in sorted(variants[best]):
                for i, docid in enumerate(variants[best][qid], 1):
                    f.write(f"{qid} Q0 {docid} {i} {1.0 - i * 1e-6:.6f} cfda-deepce\n")
        print(f"\nWrote {best}: {p}")


if __name__ == "__main__":
    main()
