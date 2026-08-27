// Local CrossEncoder reranker, no external API calls, no Python required.
// Model: Xenova/ms-marco-MiniLM-L-6-v2 (ONNX port of
// cross-encoder/ms-marco-MiniLM-L-6-v2).
//
// Uses the low-level tokenizer + model API (not the high-level `pipeline()` helper) because
// transformers.js's TextClassificationPipeline does not forward `text_pair` to the tokenizer,
// so sentence-pair scoring (query, document) is not reachable through the pipeline() shortcut.
//
// This reranker only changes the order of a fixed candidate set.

const MODEL_ID = "Xenova/ms-marco-MiniLM-L-6-v2";

type LoadedModel = { tokenizer: any; model: any };
let loadedPromise: Promise<LoadedModel> | null = null;

async function getModel(): Promise<LoadedModel> {
  if (!loadedPromise) {
    loadedPromise = (async () => {
      const { AutoTokenizer, AutoModelForSequenceClassification, env } =
        await import("@huggingface/transformers");
      env.cacheDir = process.env.TRANSFORMERS_CACHE ?? ".cache/transformers";
      const tokenizer = await AutoTokenizer.from_pretrained(MODEL_ID);
      const model =
        await AutoModelForSequenceClassification.from_pretrained(MODEL_ID);
      return { tokenizer, model };
    })();
  }
  return loadedPromise;
}

export type RerankCandidate = { docid: string; text: string };
// `score` is the raw cross-encoder logit (unbounded); `calibratedScore` is sigmoid(score) in (0,1),
// a pseudo-confidence usable as a cross-topic threshold per the "variable submission depth,
// no padding to a conventional cutoff" rule.
export type RerankedEntry = {
  docid: string;
  score: number;
  calibratedScore: number;
};

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

/**
 * Rerank a fixed set of candidates against a single query using a local CrossEncoder.
 * Candidate membership is never changed here -- only relative order.
 */
export async function rerankWithCrossEncoder(
  query: string,
  candidates: RerankCandidate[],
  options: { maxCharsPerDoc?: number } = {},
): Promise<RerankedEntry[]> {
  if (candidates.length === 0) return [];
  const { tokenizer, model } = await getModel();
  const maxChars = options.maxCharsPerDoc ?? 2000;

  const scores: RerankedEntry[] = [];
  for (const candidate of candidates) {
    const docText = candidate.text.slice(0, maxChars);
    const inputs = tokenizer(query, {
      text_pair: docText,
      padding: true,
      truncation: true,
    });
    const output = await model(inputs);
    // ms-marco-MiniLM cross-encoders are trained as single-logit relevance regressors.
    const logits: number[] = Array.from(
      output.logits.data as ArrayLike<number>,
    );
    const score = logits[0] ?? 0;
    scores.push({
      docid: candidate.docid,
      score,
      calibratedScore: sigmoid(score),
    });
  }
  return scores.sort((a, b) => b.score - a.score);
}
