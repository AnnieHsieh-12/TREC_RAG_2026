import { findInDocument } from "../retrieval/find_in_document";

export type LedgerStats = {
  pairs: number;
  kept: number;
  rewired: number;
  dropped: number;
  orphan: number;
};

type Draft = {
  references: string[];
  answer: { text: string; citations: number[] }[];
};

export function supportScore(sentence: string, docText: string): number {
  const ps = findInDocument(docText, [sentence], {
    mode: "hybrid",
    maxPassages: 1,
    windowChars: 1500,
  });
  if (ps.length === 0) return 0;
  const st = new Set(
    (sentence.toLowerCase().match(/[a-z0-9][a-z0-9'-]*/g) ?? []).filter(
      (t) => t.length > 3,
    ),
  );
  if (st.size === 0) return 0;
  const pt = new Set(
    ps[0].text.toLowerCase().match(/[a-z0-9][a-z0-9'-]*/g) ?? [],
  );
  let hit = 0;
  for (const t of st) if (pt.has(t)) hit++;
  return hit / st.size;
}

export function enforceLedger(
  draft: Draft,
  docText: Map<string, string>,
  opts: { minSupport?: number } = {},
): { draft: Draft; stats: LedgerStats } {
  const minSupport = opts.minSupport ?? 0.45;
  const stats: LedgerStats = {
    pairs: 0,
    kept: 0,
    rewired: 0,
    dropped: 0,
    orphan: 0,
  };
  const refs = draft.references;
  if (refs.length === 0 || draft.answer.length === 0) return { draft, stats };

  const readable = refs.filter((d) => (docText.get(d) ?? "").length > 0);

  const outSentences = draft.answer.map((s) => {
    const keptDocids: string[] = [];
    for (const ci of s.citations) {
      const docid = refs[ci];
      if (!docid) continue;
      stats.pairs++;
      const text = docText.get(docid) ?? "";
      if (text && supportScore(s.text, text) >= minSupport) {
        keptDocids.push(docid);
        stats.kept++;
        continue;
      }
      let best = "",
        bestScore = 0;
      for (const cand of readable) {
        if (cand === docid || keptDocids.includes(cand)) continue;
        const sc = supportScore(s.text, docText.get(cand) ?? "");
        if (sc > bestScore) {
          bestScore = sc;
          best = cand;
        }
      }
      if (best && bestScore >= minSupport) {
        keptDocids.push(best);
        stats.rewired++;
      } else stats.dropped++;
    }
    if (keptDocids.length === 0) stats.orphan++;
    return { text: s.text, docids: [...new Set(keptDocids)].slice(0, 3) };
  });

  const newRefs: string[] = [];
  const idx = new Map<string, number>();
  for (const s of outSentences) {
    for (const d of s.docids)
      if (!idx.has(d)) {
        idx.set(d, newRefs.length);
        newRefs.push(d);
      }
  }
  const answer = outSentences
    .filter((s) => s.docids.length > 0)
    .map((s) => ({
      text: s.text,
      citations: s.docids.map((d) => idx.get(d)!),
    }));

  if (answer.length === 0) return { draft, stats };
  return { draft: { references: newRefs, answer }, stats };
}
