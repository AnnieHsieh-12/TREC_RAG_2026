// Grounded citation verification (support-oriented).
//
// The RAG "support" metric asks: is each cited sentence actually supported by
// the document it cites? LLMs frequently cite a topically-related doc that does
// not actually state the sentence's claim. This module does a cheap, LLM-free
// check: for each (sentence, cited docid) pair, locate the best-matching passage
// in the doc and measure how much of the sentence's content vocabulary appears
// there. Citations whose support is too weak are dropped; sentences left with no
// citation are removed; references are rebuilt to only still-cited docids.
//
// This targets support precision without changing retrieval or coverage.

export type AnswerSentence = { text: string; citations: number[] };
export type AnswerDraft = { references: string[]; answer: AnswerSentence[] };
export type VerifyStats = {
  sentences_before: number;
  sentences_after: number;
  citations_before: number;
  citations_after: number;
  citations_dropped: number;
  sentences_dropped: number;
};

const STOPWORDS = new Set(
  "a an the and or but of to in on for with as at by from is are was were be been being this that these those it its into about how what why when where who which does do did can could should would will shall not no also has have had their his her they them we you your our more most than then over under between during such other some any many each also may might must these those".split(
    " ",
  ),
);

function contentTokens(text: string): string[] {
  const toks = text.toLowerCase().match(/[a-z0-9][a-z0-9'-]*/g) ?? [];
  return toks.filter((t) => t.length >= 3 && !STOPWORDS.has(t));
}

// Best-window content-word overlap: fraction of the sentence's distinct content
// tokens that appear anywhere in the document text. (Doc-level presence is a
// deliberately lenient proxy — it drops citations to clearly off-claim docs
// while not over-pruning paraphrases.)
export function supportScore(sentence: string, docText: string): number {
  const sTokens = [...new Set(contentTokens(sentence))];
  if (sTokens.length === 0) return 1; // nothing to check → don't penalize
  const docSet = new Set(contentTokens(docText));
  const hit = sTokens.filter((t) => docSet.has(t)).length;
  return hit / sTokens.length;
}

export function verifyCitations(
  draft: AnswerDraft,
  docTextByDocid: Map<string, string>,
  opts: { supportThreshold: number } = { supportThreshold: 0.4 },
): { draft: AnswerDraft; stats: VerifyStats } {
  const citations_before = draft.answer.reduce(
    (n, s) => n + s.citations.length,
    0,
  );
  const sentences_before = draft.answer.length;

  const keptSentences: AnswerSentence[] = [];
  const usedDocids = new Set<string>();

  for (const sent of draft.answer) {
    const goodDocids: string[] = [];
    for (const c of sent.citations) {
      const docid = draft.references[c];
      if (!docid) continue;
      const text = docTextByDocid.get(docid);
      if (text === undefined) {
        goodDocids.push(docid);
        continue;
      } // no text to check → keep as-is
      if (supportScore(sent.text, text) >= opts.supportThreshold)
        goodDocids.push(docid);
    }
    const uniqGood = [...new Set(goodDocids)];
    if (uniqGood.length === 0) continue; // sentence lost all support → drop it
    uniqGood.forEach((d) => usedDocids.add(d));
    // temporarily store docids; remap to reference indices after references are finalized
    keptSentences.push({
      text: sent.text,
      citations: uniqGood as unknown as number[],
    });
  }

  // Rebuild references from only still-cited docids, preserving first-seen order.
  const newRefs: string[] = [];
  const idxOf = new Map<string, number>();
  for (const s of keptSentences) {
    for (const docid of s.citations as unknown as string[]) {
      if (!idxOf.has(docid)) {
        idxOf.set(docid, newRefs.length);
        newRefs.push(docid);
      }
    }
  }
  const remapped: AnswerSentence[] = keptSentences.map((s) => ({
    text: s.text,
    citations: (s.citations as unknown as string[])
      .map((docid) => idxOf.get(docid)!)
      .slice(0, 3),
  }));

  const citations_after = remapped.reduce((n, s) => n + s.citations.length, 0);
  return {
    draft: { references: newRefs, answer: remapped },
    stats: {
      sentences_before,
      sentences_after: remapped.length,
      citations_before,
      citations_after,
      citations_dropped: citations_before - citations_after,
      sentences_dropped: sentences_before - remapped.length,
    },
  };
}
