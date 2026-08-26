// Dense reranking scores via NCHC bge-m3 embeddings.
// Embeds the query and each candidate document, returns cosine similarity per docid.
// Used as one signal in the three-way (BM25 + CE + dense) controlled fusion.

type Env = NodeJS.ProcessEnv;

function cosine(a: number[], b: number[]): number {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
  return na && nb ? dot / (Math.sqrt(na) * Math.sqrt(nb)) : 0;
}

async function embed(inputs: string[], env: Env): Promise<number[][]> {
  const base = (env.NCHC_BASE_URL || "https://portal.genai.nchc.org.tw/api/v1").replace(/\/+$/, "");
  const key = (env.NCHC_API_KEY || env.NCHC_GENAI_API_KEY || "").trim();
  for (let attempt = 1; attempt <= 6; attempt++) {
    let r: Response;
    try {
      r = await fetch(`${base}/embeddings`, {
        method: "POST",
        headers: { "x-api-key": key, "content-type": "application/json" },
        body: JSON.stringify({ model: "bge-m3", input: inputs }),
      });
    } catch (e) { if (attempt === 6) throw new Error(`embeddings fetch failed: ${e instanceof Error ? e.message : String(e)}`); await sleep(Math.max(1000, 600 * 2 ** attempt)); continue; }
    if (r.ok) { const j = await r.json() as any; return (j.data ?? []).map((d: any) => d.embedding as number[]); }
    if (![429, 500, 502, 503, 504].includes(r.status) || attempt === 6) throw new Error(`embeddings HTTP ${r.status}`);
    await sleep(r.status === 429 ? Math.min(30000, 2000 * 2 ** attempt) : Math.max(1000, 600 * 2 ** attempt));
  }
  throw new Error("embeddings failed");
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

// Returns docid -> cosine(query, doc). Docs are embedded in batches to stay within request limits.
export async function denseScores(
  query: string,
  docs: { docid: string; text: string }[],
  env: Env,
  opts: { maxChars: number; batchSize: number } = { maxChars: 2000, batchSize: 24 },
): Promise<Map<string, number>> {
  const scores = new Map<string, number>();
  if (docs.length === 0) return scores;
  const [qVec] = await embed([query.slice(0, opts.maxChars)], env);
  for (let i = 0; i < docs.length; i += opts.batchSize) {
    const batch = docs.slice(i, i + opts.batchSize);
    const vecs = await embed(batch.map((d) => d.text.slice(0, opts.maxChars)), env);
    batch.forEach((d, j) => { if (vecs[j]) scores.set(d.docid, cosine(qVec, vecs[j])); });
    await sleep(150);
  }
  return scores;
}
