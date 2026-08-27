import {
  RAG_PROMPT_PROFILE,
  type TopicIdentity,
} from "../shared-rag/contracts";

export type EvidenceDocumentForPrompt = {
  docid: string;
  text: string;
};

export type JudgePromptInput = {
  topic: TopicIdentity;
  iteration: number;
  maxIterations: number;
  previousQueries: string[];
  documents: EvidenceDocumentForPrompt[];
};

export type FollowupQueryPromptInput = {
  topic: TopicIdentity;
  previousQueries: string[];
  missingAspects: string[];
  recommendedFollowupFocus?: string;
};

export type AnswerGenerationPromptInput = {
  topic: TopicIdentity;
  documents: EvidenceDocumentForPrompt[];
};

export function buildJudgePrompt(input: JudgePromptInput): string {
  return [
    `Prompt version: ${RAG_PROMPT_PROFILE}`,
    "You are an evidence sufficiency judge for a TREC RAG baseline.",
    "Use only the provided topic and evidence documents. Do not use outside knowledge.",
    "Decide whether the current evidence is sufficient to answer all important aspects of the narrative.",
    "Do not write the final answer.",
    "Return only strict JSON. Do not use Markdown fences, comments, or explanatory prose.",
    "",
    "Required JSON shape:",
    '{"enough":false,"confidence":0.0,"covered_aspects":["..."],"missing_aspects":["..."],"needs_followup":true,"recommended_followup_focus":"...","rationale":"..."}',
    "",
    formatTopic(input.topic),
    `Iteration: ${input.iteration} of ${input.maxIterations}`,
    `Previous queries: ${JSON.stringify(input.previousQueries)}`,
    "Evidence documents:",
    formatDocuments(input.documents),
  ].join("\n");
}

export function buildFollowupQueryPrompt(
  input: FollowupQueryPromptInput,
): string {
  return [
    "Return exactly one strict JSON object and nothing else:",
    '{"subquery":"..."}',
    "The subquery must be a concise BM25 keyword query, 3-160 characters, no Markdown, no code, no braces in the value.",
    "Use only the missing aspects and recommended focus. Do not explain.",
    `Previous queries: ${JSON.stringify(input.previousQueries)}`,
    `Missing aspects: ${JSON.stringify(input.missingAspects)}`,
    `Recommended focus: ${input.recommendedFollowupFocus ?? ""}`,
  ].join("\n");
}

export function buildAnswerGenerationPrompt(
  input: AnswerGenerationPromptInput,
): string {
  return [
    `Prompt version: ${RAG_PROMPT_PROFILE}`,
    "You generate an evidence-grounded TREC RAG answer.",
    "Use only the provided evidence documents. Do not use outside knowledge.",
    "Break the answer into individual factual sentences.",
    "Every answer sentence must have citations.",
    "Citations are zero-indexed positions into the references array, not docids.",
    "Each sentence may cite at most three references.",
    "References must be unique ClimbMix docids and every reference must be cited at least once.",
    "Do not include uncited retrieved documents in references.",
    "Keep the complete answer under 1024 words.",
    "Return only strict JSON. Do not use Markdown fences, comments, or explanatory prose.",
    "",
    "Required JSON shape:",
    '{"references":["shard_00000_00000"],"answer":[{"text":"Supported sentence.","citations":[0]}]}',
    "",
    formatTopic(input.topic),
    "Evidence documents:",
    formatDocuments(input.documents),
  ].join("\n");
}

export type DenseAnswerPromptInput = AnswerGenerationPromptInput & {
  checklist?: string[];
};

// Fill the 1024-word budget with dense, atomic, entity-rich sentences.
// Motivated by measurement (our answers averaged 213-245 words = 21% of the
// cap) and literature: nugget recall rises monotonically with length
// (Crucible 0.599->0.717->0.812) and the strict-vital metric has no verbosity
// penalty; GINGER's winning shape is short atomic entity-dense sentences
// organized by facet.
export function buildDenseAnswerGenerationPrompt(
  input: DenseAnswerPromptInput,
): string {
  const checklistBlock =
    input.checklist && input.checklist.length > 0
      ? [
          "Organize the answer by these aspects of the topic; write roughly 80-120 words for EACH aspect (a few sentences each), in this order:",
          ...input.checklist.map(
            (c, i) => `${i + 1}. ${c.replace(/ \(vital\)$/, "")}`,
          ),
          "",
        ]
      : [];
  return [
    `Prompt version: ${RAG_PROMPT_PROFILE}-dense`,
    "You generate an evidence-grounded TREC RAG answer.",
    "Use only the provided evidence documents. Do not use outside knowledge.",
    "",
    "LENGTH: the complete answer MUST total between 900 and 1010 words. Short answers waste the official budget and lose coverage. Keep adding supported factual sentences until you approach 1010 words.",
    "",
    "SENTENCE STYLE - every sentence must be:",
    "- atomic: exactly one factual claim per sentence, at most ~35 words;",
    "- concrete: preserve named entities, numbers, dates, organizations, places verbatim from the evidence;",
    "- grounded: cited to the evidence documents that state the fact.",
    "- maximally SPECIFIC: prefer the most specific fact available - named people, organizations, places, causes and their outcomes, findings and their evidence. A precise claim (who did what, why, with what result) scores; a general statement about the same aspect does not.",
    "Do NOT write generic overview sentences that only describe the topic in the abstract (e.g. 'This is an important and dynamic field', 'X is part of culture'). Such sentences are worthless and must be omitted.",
    "Good: 'Persistent gender and racial pay gaps affect athlete compensation, and business interests shape both media-rights deals and team management.'",
    "Bad: 'Athlete compensation is a controversial and important topic.'",
    "No introduction, no conclusion, no hedging, no restating the question - only supported facts.",
    ...(process.env.DENSE_DRAFT_HINT
      ? ["", `FACT SELECTION EMPHASIS: ${process.env.DENSE_DRAFT_HINT}`]
      : []),
    "",
    ...checklistBlock,
    "Citations are zero-indexed positions into the references array, not docids.",
    "Each sentence may cite at most three references.",
    "References must be unique ClimbMix docids and every reference must be cited at least once.",
    "Do not include uncited retrieved documents in references.",
    "Return only strict JSON. Do not use Markdown fences, comments, or explanatory prose.",
    "",
    "Required JSON shape:",
    '{"references":["shard_00000_00000"],"answer":[{"text":"Supported sentence.","citations":[0]}]}',
    "",
    formatTopic(input.topic),
    "Evidence documents:",
    formatDocuments(input.documents),
  ].join("\n");
}

export function buildCompactAnswerGenerationPrompt(
  input: AnswerGenerationPromptInput,
): string {
  return [
    "Return only this JSON shape, with no Markdown or explanation:",
    '{"references":["shard_00000_00000"],"answer":[{"text":"Supported sentence.","citations":[0]}]}',
    "Use only the provided docs. Every sentence needs citations. Citation numbers are references array indexes. Use only cited docids in references.",
    formatTopic(input.topic),
    "Docs:",
    formatDocuments(
      input.documents
        .slice(0, 5)
        .map((doc) => ({ ...doc, text: doc.text.slice(0, 800) })),
    ),
  ].join("\n");
}

export type VerifyReviseSentence = {
  text: string;
  citations: number[];
  evidence: Array<{ docid: string; excerpt: string }>;
};

export type VerifyRevisePromptInput = {
  topic: TopicIdentity;
  references: string[];
  sentences: VerifyReviseSentence[];
  mode?: "drop" | "weaken";
};

// Post-answer self-verification pass: re-read the cited evidence and repair
// over-extended sentences. Motivated by measured citation quality: only
// 24-35% of sentences are fully supported; most are partially supported
// (over-extended relative to what the citation actually says).
export function buildVerifyRevisePrompt(
  input: VerifyRevisePromptInput,
): string {
  const sentenceBlocks = input.sentences
    .map((sentence, index) =>
      [
        `Sentence ${index}: ${sentence.text}`,
        `Citations: ${JSON.stringify(sentence.citations)}`,
        "Cited evidence excerpts:",
        sentence.evidence.length === 0
          ? "(none available)"
          : sentence.evidence
              .map((e) => `- [${e.docid}] ${e.excerpt}`)
              .join("\n"),
      ].join("\n"),
    )
    .join("\n\n");
  return [
    "You are revising a cited RAG answer so that every sentence is fully supported by its cited evidence.",
    "For each sentence below, compare the sentence against its cited evidence excerpts and act:",
    "- Fully supported: keep the sentence unchanged with the same citations.",
    "- Over-extended (partially supported): rewrite it so it claims ONLY what the excerpts support. Do not add new claims.",
    input.mode === "weaken"
      ? "- Not supported by its citations: rewrite it into a weaker related claim that the excerpts DO support (hedge, narrow, or generalize it); if other cited excerpts support the point, switch its citations to those; remove the sentence only when nothing in any excerpt relates to it."
      : "- Not supported at all: remove the sentence.",
    "Rules: do not invent new references or citations; keep citation indices pointing into the references array below;",
    "each sentence at most three citations; keep the overall answer coherent and under 1024 words;",
    "keep the original sentence order for kept/rewritten sentences.",
    "Return only strict JSON, no Markdown fences, exactly this shape:",
    '{"references":["shard_00000_00000"],"answer":[{"text":"Supported sentence.","citations":[0]}]}',
    "The references array must stay the same list (unused entries are allowed and will be pruned automatically).",
    "",
    formatTopic(input.topic),
    `References array (index -> docid): ${JSON.stringify(input.references)}`,
    "",
    "Sentences to verify:",
    sentenceBlocks,
  ].join("\n");
}

function formatTopic(topic: TopicIdentity): string {
  return [
    "Topic:",
    `narrative_id: ${topic.qid}`,
    `title: ${topic.title}`,
    "narrative:",
    topic.narrative,
  ].join("\n");
}

function formatDocuments(documents: EvidenceDocumentForPrompt[]): string {
  if (documents.length === 0) return "(none)";
  return documents
    .map((document, index) =>
      [
        `Document ${index}:`,
        `docid: ${document.docid}`,
        "text:",
        document.text,
      ].join("\n"),
    )
    .join("\n\n");
}
