export type AgenticRagBaselineReadDoc = { docid: string; text: string; truncated: boolean; rankHint: number; query: string };

export type ExtractiveFallbackAnswerDraft = {
  references: string[];
  answer: Array<{ text: string; citations: number[] }>;
};

export function buildExtractiveFallbackAnswerDraft(readDocs: Map<string, AgenticRagBaselineReadDoc>): ExtractiveFallbackAnswerDraft {
  const selected = [...readDocs.values()].filter((doc) => doc.text.trim().length > 0).slice(0, 3);
  if (selected.length === 0) throw new Error("Extractive answer fallback requires at least one read document with text.");
  return {
    references: selected.map((doc) => doc.docid),
    answer: selected.map((doc, index) => ({
      text: extractFallbackExcerpt(doc.text),
      citations: [index],
    })),
  };
}

function extractFallbackExcerpt(text: string): string {
  const normalized = text.replace(/\s+/g, " ").trim();
  const sentences = normalized.match(/[^.!?。！？]+[.!?。！？]+|[^.!?。！？]+$/g) ?? [normalized];
  const candidate = sentences.map((sentence) => sentence.trim()).find((sentence) => sentence.length >= 30) ?? normalized;
  const clipped = candidate.length <= 260 ? candidate : `${candidate.slice(0, 257).trimEnd()}...`;
  return clipped.trim();
}
