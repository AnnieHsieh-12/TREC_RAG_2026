export type AnswerSentence = { text: string; citations: number[] };
export type AnswerDraft = { references: string[]; answer: AnswerSentence[] };
export type ReattributeStats = {
  sentences: number;
  reattributed: number;
  avg_top_score: number;
};

type Env = NodeJS.ProcessEnv;
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

async function bgeRerank(
  query: string,
  documents: string[],
  env: Env,
): Promise<number[]> {
  const base = (
    env.NCHC_BASE_URL || "https://portal.genai.nchc.org.tw/api/v1"
  ).replace(/\/+$/, "");
  const key = (env.NCHC_API_KEY || env.NCHC_GENAI_API_KEY || "").trim();
  for (let attempt = 1; attempt <= 5; attempt++) {
    let r: Response;
    try {
      r = await fetch(`${base}/rerank`, {
        method: "POST",
        headers: { "x-api-key": key, "content-type": "application/json" },
        body: JSON.stringify({ model: "BGE-Reranker-V2-M3", query, documents }),
      });
    } catch (e) {
      if (attempt === 5)
        throw new Error(
          `rerank fetch failed: ${e instanceof Error ? e.message : String(e)}`,
        );
      await sleep(Math.max(800, 500 * 2 ** attempt));
      continue;
    }
    if (r.ok) {
      const j = (await r.json()) as any;
      const out = new Array(documents.length).fill(0);
      for (const res of j.results ?? [])
        if (typeof res.index === "number")
          out[res.index] = Number(res.relevance_score) || 0;
      return out;
    }
    if (![429, 500, 502, 503, 504].includes(r.status) || attempt === 5)
      throw new Error(`rerank HTTP ${r.status}`);
    await sleep(
      r.status === 429
        ? Math.min(30000, 2000 * 2 ** attempt)
        : Math.max(800, 500 * 2 ** attempt),
    );
  }
  throw new Error("rerank failed");
}

export async function reattributeCitations(
  draft: AnswerDraft,
  docTextByDocid: Map<string, string>,
  env: Env,
  opts: { threshold: number; maxCites: number; snippetChars: number } = {
    threshold: 0.3,
    maxCites: 2,
    snippetChars: 1500,
  },
): Promise<{ draft: AnswerDraft; stats: ReattributeStats }> {
  const refs = draft.references;
  if (refs.length === 0 || draft.answer.length === 0)
    return {
      draft,
      stats: {
        sentences: draft.answer.length,
        reattributed: 0,
        avg_top_score: 0,
      },
    };
  const refTexts = refs.map((docid) =>
    (docTextByDocid.get(docid) ?? "").slice(0, opts.snippetChars),
  );
  const scorable = refs
    .map((_, i) => (refTexts[i].length > 0 ? i : -1))
    .filter((i) => i >= 0);

  const newAnswer: AnswerSentence[] = [];
  const usedDocids = new Set<string>();
  let reattributed = 0,
    topSum = 0,
    scored = 0;

  for (const sent of draft.answer) {
    let chosen: number[];
    if (scorable.length === 0) {
      chosen = sent.citations;
    } else {
      let scores: number[];
      try {
        scores = await bgeRerank(
          sent.text,
          scorable.map((i) => refTexts[i]),
          env,
        );
      } catch {
        newAnswer.push(sent);
        sent.citations.forEach((c) => refs[c] && usedDocids.add(refs[c]));
        continue;
      }
      await sleep(120);
      const ranked = scorable
        .map((refIdx, k) => ({ refIdx, score: scores[k] ?? 0 }))
        .sort((a, b) => b.score - a.score);
      topSum += ranked[0].score;
      scored++;
      const good = ranked
        .filter((r) => r.score >= opts.threshold)
        .slice(0, opts.maxCites);
      chosen = (good.length > 0 ? good : [ranked[0]]).map((r) => r.refIdx);
      const before = [...sent.citations].sort().join(",");
      if (chosen.slice().sort().join(",") !== before) reattributed++;
    }
    const docids = [...new Set(chosen.map((i) => refs[i]).filter(Boolean))];
    if (docids.length === 0) continue;
    docids.forEach((d) => usedDocids.add(d));
    newAnswer.push({
      text: sent.text,
      citations: docids as unknown as number[],
    });
  }

  const newRefs: string[] = [];
  const idxOf = new Map<string, number>();
  for (const s of newAnswer)
    for (const d of s.citations as unknown as string[])
      if (!idxOf.has(d)) {
        idxOf.set(d, newRefs.length);
        newRefs.push(d);
      }
  const remapped = newAnswer.map((s) => ({
    text: s.text,
    citations: (s.citations as unknown as string[])
      .map((d) => idxOf.get(d)!)
      .slice(0, 3),
  }));

  return {
    draft: { references: newRefs, answer: remapped },
    stats: {
      sentences: draft.answer.length,
      reattributed,
      avg_top_score: scored ? topSum / scored : 0,
    },
  };
}
