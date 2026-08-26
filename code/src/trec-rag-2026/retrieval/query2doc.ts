// Query2Doc-style pseudo-document query expansion.
//
// Reference: Wang, Yang, Wei (EMNLP 2023) "Query2doc: Query Expansion with
// Large Language Models" (arXiv:2303.07678). The LLM writes a short
// hypothetical answer passage; for sparse (BM25) retrieval the expanded query
// is the original query repeated n times concatenated with the pseudo-doc, so
// the original query terms keep dominant term frequency (drift protection).
//
// This module ONLY builds the expanded query string + one LLM call. Retrieval
// (search) and fusion (weightedRrf) are done by the existing runner, so this is
// a single additive signal on top of the raw-narrative anchor run.

import type { LlmClient } from "../../llm/types";

export type Query2DocResult = {
  pseudoDoc: string;      // the raw generated passage (for tracing)
  expandedQuery: string;  // narrative*repeat + pseudoDoc, ready for BM25
  ok: boolean;            // false => generation failed, caller should fall back to raw query only
};

function buildPrompt(narrative: string): string {
  return [
    "Write a short, factual passage (about 100-150 words) that directly answers the question below.",
    "Write as if it were an excerpt from a relevant reference document.",
    "Be concrete: include the specific named entities, technical terms, organizations, places, dates, and facts that a truly relevant document would mention.",
    "Do not hedge, do not add meta commentary, do not restate the question. Output only the passage text.",
    "",
    `Question: ${narrative}`,
  ].join("\n");
}

// Keep a pseudo-doc from ballooning the BM25 query. BM25 treats the query as a
// bag of words, so trimming to a word budget bounds the request size while
// preserving the useful expansion vocabulary.
function trimWords(text: string, maxWords: number): string {
  const words = text.replace(/\s+/g, " ").trim().split(" ");
  return words.length <= maxWords ? words.join(" ") : words.slice(0, maxWords).join(" ");
}

// Strip characters that would break the Pyserini query or add noise; BM25 only
// needs alphanumeric tokens.
function sanitizeForQuery(text: string): string {
  return text.replace(/[\r\n]+/g, " ").replace(/["'`{}\[\]]/g, " ").replace(/\s+/g, " ").trim();
}

export async function generateQuery2Doc(
  llm: LlmClient,
  narrative: string,
  opts: { queryRepeat: number; maxPseudoDocWords: number; maxTokens: number } = { queryRepeat: 5, maxPseudoDocWords: 180, maxTokens: 400 },
): Promise<Query2DocResult> {
  const rawQuery = sanitizeForQuery(narrative);
  try {
    const result = await llm.generate({
      messages: [{ role: "user", content: buildPrompt(narrative) }],
      temperature: 0,
      maxTokens: opts.maxTokens,
    });
    const pseudoDoc = trimWords(sanitizeForQuery(result.text ?? ""), opts.maxPseudoDocWords);
    if (pseudoDoc.length < 20) {
      // empty / degenerate generation: no usable expansion, signal fallback
      return { pseudoDoc, expandedQuery: rawQuery, ok: false };
    }
    // Query2Doc sparse form: repeat the original query so its terms keep
    // dominant TF, then append the pseudo-doc vocabulary.
    const expandedQuery = sanitizeForQuery(
      Array.from({ length: Math.max(1, opts.queryRepeat) }, () => rawQuery).join(" ") + " " + pseudoDoc,
    );
    return { pseudoDoc, expandedQuery, ok: true };
  } catch {
    return { pseudoDoc: "", expandedQuery: rawQuery, ok: false };
  }
}
