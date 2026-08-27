import { RAG_PROMPT_PROFILE, type TopicIdentity } from "./contracts";

export type EvidenceDocumentForPrompt = {
  docid: string;
  text: string;
};

export function buildGroundedRevisionPrompt(
  topic: TopicIdentity,
  sentences: { text: string; docids: string[] }[],
  citedDocs: { docid: string; text: string }[],
): string {
  return [
    "You revise a draft answer so every sentence is FULLY supported by the documents it cites.",
    "For each sentence: check whether the cited documents actually state its claim.",
    "  - If the sentence over-claims (says more than the evidence), rewrite it to state only what the cited documents support.",
    "  - If a cited document does not support the sentence, drop that citation. Prefer citing a document that genuinely supports the claim.",
    "  - If no cited document supports the sentence at all, weaken it to the strongest statement the cited documents DO support.",
    "NEVER delete a sentence. Return exactly as many sentences as you were given, in the same order. A weakened sentence is always better than a missing one.",
    "Keep the answer informative and specific; do not add facts that are not in the cited documents.",
    "Citations are ClimbMix docid strings. Each sentence may cite at most three documents.",
    "Return ONLY strict JSON, no prose:",
    '{"answer":[{"text":"...","citations":["shard_00000_00000"]}]}',
    "",
    `Narrative: ${topic.narrative}`,
    "",
    "Cited documents (docid → text):",
    citedDocs.map((d) => `[${d.docid}]\n${d.text}`).join("\n\n"),
    "",
    "Draft answer (each sentence with the docids it currently cites):",
    sentences
      .map((s, i) => `${i + 1}. ${s.text}  — cites: ${s.docids.join(", ")}`)
      .join("\n"),
  ].join("\n");
}

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
    "You are an evidence sufficiency judge for a TREC RAG pipeline.",
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

export function buildComprehensiveAnswerPrompt(
  input: AnswerGenerationPromptInput & { aspects?: string[]; atomic?: boolean },
): string {
  const aspectBlock =
    input.aspects && input.aspects.length > 0
      ? [
          "Aspects to cover (write at least one grounded sentence for EACH; do not drop any):",
          ...input.aspects.map((a, i) => `  ${i + 1}. ${a}`),
        ].join("\n")
      : "First identify every distinct aspect the narrative asks about, then cover each of them.";
  return [
    `Prompt version: ${RAG_PROMPT_PROFILE}`,
    "You generate an evidence-grounded TREC RAG answer that comprehensively covers the narrative.",
    "Use only the provided evidence documents. Do not use outside knowledge.",
    aspectBlock,
    "Write a thorough answer: aim to cover every aspect above with specific facts, figures, named entities, and details found in the evidence.",
    "Break the answer into many individual factual sentences (typically 12-25 sentences for a multi-aspect narrative).",
    "Prefer more specific, information-dense sentences over few vague ones. Each sentence should add a distinct fact.",
    "Every answer sentence must have citations. Citations are zero-indexed positions into the references array, not docids.",
    input.atomic
      ? "Cite EXACTLY ONE reference per sentence - the single document that fully supports it. If no single document supports the sentence on its own, split or rewrite the sentence until one does."
      : "Each sentence may cite at most three references. Only cite a document that genuinely supports that exact sentence.",
    "References must be unique ClimbMix docids and every reference must be cited at least once. Do not include uncited documents.",
    "Keep the complete answer at or below 1024 words.",
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

export function buildAspectAnswerPrompt(input: {
  topic: TopicIdentity;
  aspect: string;
  documents: EvidenceDocumentForPrompt[];
  alreadyWritten?: string[];
  atomic?: { maxWords: number; sentences: number };
  expectedFacts?: string[];
}): string {
  const already =
    input.alreadyWritten && input.alreadyWritten.length > 0
      ? [
          "Already written for this aspect (do NOT repeat these; only add NEW distinct facts not covered here):",
          ...input.alreadyWritten.map((s) => `  - ${s}`),
          "",
        ].join("\n")
      : "";
  const wanted =
    input.expectedFacts && input.expectedFacts.length > 0
      ? [
          "What a complete answer is expected to state about this aspect (use as a checklist of what to LOOK FOR in the evidence):",
          ...input.expectedFacts.map((f) => `  - ${f}`),
          "Only write those of the above that the evidence actually supports, and write them with the evidence's own specifics. NEVER state one of them just because it is listed here — an unsupported sentence costs more than a missing one.",
          "",
        ].join("\n")
      : "";
  return [
    "You write a few evidence-grounded sentences answering ONE specific aspect of a TREC RAG narrative.",
    "Use only the provided evidence documents. Do not use outside knowledge.",
    `Aspect to answer: ${input.aspect}`,
    wanted,
    already,
    input.atomic
      ? `Write ${input.atomic.sentences} factual sentences that specifically answer this aspect.`
      : "Write 2-5 factual sentences that specifically answer this aspect.",
    "CRITICAL — each sentence must state a SPECIFIC, checkable claim: a concrete fact, statistic, date, named entity, mechanism, or an explicit enumerated list taken from the evidence.",
    "Do NOT write generic overview sentences that only describe the topic in the abstract (e.g. 'This is an important and dynamic field', 'Sports are part of culture', 'This raises many questions'). Such sentences are worthless and must be omitted.",
    ...(input.atomic
      ? [
          `SENTENCE STYLE - write ATOMIC sentences: ONE fact per sentence, at most ${input.atomic.maxWords} words each.`,
          "Do NOT merge several claims into one long sentence. If the evidence names a set of five groups, write five short sentences, one per group - not one sentence listing all five.",
          "Each sentence must be fully supported by a SINGLE document on its own, because it will be checked against exactly one cited source.",
          "Good: 'Racial inclusion remains a challenge in professional sports.' then 'Gender inclusion remains a challenge in professional sports.'",
          "Bad (merges five claims, no single source supports it): 'Inclusion remains a challenge across race, gender, social class, sexuality, and disability.'",
          "Write MORE sentences rather than longer ones - breadth comes from sentence count, not sentence length.",
        ]
      : [
          "Write DENSE, ENUMERATIVE sentences that state several related claims at once. 25-40 words per sentence is normal and good here.",
        ]),
    "This is scored by how many distinct required claims the answer states. Those claims are short thematic statements, and they come in FAMILIES that share a stem and vary one dimension — for example 'racial inclusion is a challenge', 'gender inclusion is a challenge', 'inclusion of disabled athletes is a challenge'.",
    input.atomic
      ? "So whenever the evidence names a SET of groups, causes, effects, dimensions, stakeholders, or examples, name EVERY member of that set explicitly. Each named member earns credit; a set left as 'various factors' or 'several groups' earns none."
      : "So whenever the evidence names a SET of groups, causes, effects, dimensions, stakeholders, or examples, name EVERY member of that set explicitly in one sentence. Each named member earns credit; a set left as 'various factors' or 'several groups' earns none.",
    ...(input.atomic
      ? [
          "Cover a set by writing one sentence PER MEMBER, back to back - never by dropping members.",
          "Good (five members, five sentences): 'Racial inclusion remains a challenge in professional sports.' 'Gender inclusion remains a challenge in professional sports.' 'Social class remains a barrier to inclusion in professional sports.' ... (one per member)",
          "Bad (drops members): 'Inclusion remains a challenge for several groups.'",
        ]
      : [
          "Good (one sentence covering five distinct claims): 'Persistent gender and racial pay gaps affect athlete compensation, and business interests shape athlete pay, media-rights deals, and team management alike.'",
          "Good (one sentence covering five distinct claims): 'Inclusion remains a challenge across race, gender, social class, sexuality, and disability.'",
          "Bad (same ground, only one claim stated): 'There are pay gaps in sports, and business interests are influential.'",
        ]),
    "Bad (vague, names nothing): 'Athlete compensation is a controversial and important topic.'",
    "Do NOT chase numbers, dates, or proper nouns for their own sake — the claims being scored are thematic, so breadth of named dimensions matters far more than statistics.",
    "Every sentence must have citations. Citations are zero-indexed positions into the references array, not docids.",
    input.atomic
      ? "Cite EXACTLY ONE reference per sentence - the single document that fully supports it. If no single document supports the sentence on its own, split or rewrite the sentence until one does."
      : "Each sentence may cite at most three references. Only cite a document that genuinely supports that exact sentence.",
    "References must be unique ClimbMix docids; every reference must be cited at least once; do not include uncited documents.",
    "If the evidence does not cover this aspect at all, return an empty answer array.",
    "Return only strict JSON. No Markdown fences, no prose.",
    "",
    "Required JSON shape:",
    '{"references":["shard_00000_00000"],"answer":[{"text":"Supported sentence.","citations":[0]}]}',
    "",
    formatTopic(input.topic),
    "Evidence documents:",
    formatDocuments(input.documents),
  ].join("\n");
}

export function buildReflectionPrompt(
  topic: TopicIdentity,
  answerText: string,
): string {
  return [
    "You review a draft answer against a narrative and identify what is still MISSING or only thinly covered.",
    "Return ONLY strict JSON, no prose:",
    '{"gaps":["missing aspect one","missing aspect two"]}',
    "Rules: 0-2 gaps; each a short noun phrase (3-10 words) naming a distinct aspect the narrative asks about but the draft does not adequately cover; if the draft already covers everything well, return an empty array.",
    "",
    formatTopic(topic),
    "Draft answer:",
    answerText,
  ].join("\n");
}

export function buildAspectDecompositionPrompt(
  topic: TopicIdentity,
  expectedFacts = false,
): string {
  return [
    "List every distinct aspect the following narrative asks about, so that together they cover the whole question.",
    "Return ONLY a strict JSON object, no prose:",
    expectedFacts
      ? '{"aspects":[{"title":"aspect one","expected_facts":["a fact a good answer would state","another fact"]}]}'
      : '{"aspects":["aspect one","aspect two"]}',
    "Rules: 8-12 aspects; each a short noun phrase (3-10 words); split the narrative FINELY so each aspect maps to a distinct sub-question a good answer must address; cover every distinct thing asked; do not overlap; do not add aspects the narrative does not ask about.",
    "Prefer MORE, NARROWER aspects over few broad ones: the answer is scored on how many distinct required facts it covers, so a fine split gives each fact its own targeted evidence.",
    ...(expectedFacts
      ? [
          "For each aspect also give 2-4 expected_facts: each a single short declarative sentence stating a fact a complete answer would state about that aspect.",
          "expected_facts are what a good answer SHOULD say, written from the narrative alone - you have no evidence documents here, so state them as the kind of claim to look for, not as verified fact.",
          "Make them specific and checkable (who did what, with what effect), not restatements of the aspect title.",
        ]
      : []),
    "",
    formatTopic(topic),
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

export type DenseAnswerPromptInput = AnswerGenerationPromptInput & {
  checklist?: string[];
  enumerative?: { minWords: number; maxWords: number };
};

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
    ...(input.enumerative
      ? [
          `SENTENCE STYLE - write ENUMERATIVE sentences of ${input.enumerative.minWords}-${input.enumerative.maxWords} words:`,
          "- each sentence must name EVERY member of one dimension that the evidence supports, not just one member;",
          "- Bad: 'The company operates sites in North America.' Good: 'The company operates sites in the United States, Canada and Mexico, opened in 2015, 2017 and 2019 respectively.'",
          "- list ONLY members that actually appear in the evidence. NEVER add a member to round out a list: one unsupported member turns the whole sentence unsupported, which costs more than a shorter list gains.",
        ]
      : [
          "SENTENCE STYLE - every sentence must be:",
          "- atomic: exactly one factual claim per sentence, at most ~35 words;",
        ]),
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

export type LedgerAnswerPromptInput = DenseAnswerPromptInput & {
  subquestions: { id: string; text: string }[];
  /** Previous validation failure, included when regenerating the answer. */
  violation?: string;
};

export function buildLedgerAnswerPrompt(
  input: LedgerAnswerPromptInput,
): string {
  return [
    `Prompt profile: ${RAG_PROMPT_PROFILE}-evidence-plan`,
    "You generate an evidence-grounded TREC RAG answer under an EVIDENCE LEDGER policy.",
    "Use only the provided evidence documents. Do not use outside knowledge.",
    "",
    "LENGTH: the complete answer MUST total between 900 and 1010 words. Short answers waste the official budget and lose coverage.",
    "",
    ...(input.enumerative
      ? [
          `SENTENCE STYLE - write ENUMERATIVE sentences of ${input.enumerative.minWords}-${input.enumerative.maxWords} words:`,
          "- each sentence must name EVERY member of one dimension that the evidence supports, not just one member;",
          "- list ONLY members that actually appear in the evidence. A member you cannot quote verbatim will fail the ledger check and reject the whole answer.",
        ]
      : [
          "SENTENCE STYLE - every sentence must be:",
          "- atomic: exactly one factual claim per sentence, at most ~35 words;",
        ]),
    "- concrete: preserve named entities, numbers, dates, organizations, places verbatim from the evidence;",
    "- maximally SPECIFIC: a precise claim (who did what, why, with what result) scores; a general statement about the same aspect does not.",
    "No introduction, no conclusion, no hedging, no restating the question - only supported facts.",
    "",
    "EVIDENCE LEDGER RULES - these are enforced by a checker; violating any of them rejects the whole answer:",
    "1. Every sentence must carry at least one evidence record.",
    "2. Each evidence record's `exact_quote` must be copied VERBATIM, character for character, from the evidence document it names. Do not paraphrase, do not fix typos, do not join separated lines. Minimum 20 characters.",
    "3. Each evidence record's `claim` must be EXACTLY EQUAL to the sentence's own `text`. Write the sentence first, then repeat it as the claim - never write a sentence the quote does not support.",
    "4. `citations` lists the docids you cite. The FIRST citation must be the docid of the FIRST evidence record. Every citation needs a matching evidence record, and every evidence record needs a matching citation.",
    "5. You may only cite documents that appear in the evidence below. A search snippet is not evidence.",
    "",
    "SUBQUESTIONS - every required subquestion must be represented by evidence-backed sentences.",
    "Explanatory subquestions (how/why/effects) need at least TWO distinct sentences; atomic-fact subquestions (who/where/when/how many) need one.",
    ...input.subquestions.map((q) => `  ${q.id}: ${q.text}`),
    "",
    ...(input.violation
      ? [
          "YOUR PREVIOUS ANSWER WAS REJECTED BY THE LEDGER CHECKER:",
          `  ${input.violation}`,
          "Fix that specific problem. Do not shorten the answer to avoid it.",
          "",
        ]
      : []),
    "Return only strict JSON. Do not use Markdown fences, comments, or explanatory prose.",
    "",
    "Required JSON shape:",
    '{"answer_plan":[{"text":"One factual sentence.","citations":["shard_00000_00000"],' +
      '"evidence":[{"docid":"shard_00000_00000","exact_quote":"verbatim span of at least twenty characters",' +
      '"claim":"One factual sentence.","subquestion_ids":["Q1"]}]}]}',
    "",
    formatTopic(input.topic),
    "Evidence documents:",
    formatDocuments(input.documents),
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
  mode?: "drop" | "weaken" | "reattribute";
};

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
  const ACTION = {
    weaken:
      "- Not supported by its citations: rewrite it into a weaker related claim that the excerpts DO support (hedge, narrow, or generalize it); if other cited excerpts support the point, switch its citations to those; remove the sentence only when nothing in any excerpt relates to it.",
    drop: "- Not supported at all: remove the sentence.",
    reattribute:
      "- Not fully supported by its CURRENT citations: do NOT touch the wording. Instead search ALL the references listed below and re-cite the sentence to whichever reference(s) actually state it. Remove the sentence ONLY if no reference in the entire list supports any part of it.",
  } as const;
  const mode = (
    input.mode && input.mode in ACTION ? input.mode : "drop"
  ) as keyof typeof ACTION;
  return [
    "You are revising a cited RAG answer so that every sentence is fully supported by its cited evidence.",
    "For each sentence below, compare the sentence against its cited evidence excerpts and act:",
    "- Fully supported: keep the sentence unchanged with the same citations.",
    ...(mode === "reattribute"
      ? [
          "- Over-extended (partially supported): keep the wording EXACTLY as written. Only fix its citations.",
        ]
      : [
          "- Over-extended (partially supported): rewrite it so it claims ONLY what the excerpts support. Do not add new claims.",
        ]),
    ACTION[mode],
    ...(mode === "reattribute"
      ? [
          "HARD RULE: you may not add, delete, reorder or reword a single word of any sentence you keep. Your only edit is the citations array. Answers that change wording will be rejected.",
        ]
      : []),
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

export function buildIntegrationPrompt(input: {
  topic: TopicIdentity;
  groups: { aspect: string; sentences: string[] }[];
  minWords: number;
  maxWords: number;
}): string {
  const blocks = input.groups
    .map((g, i) =>
      [`[${i}] ${g.aspect}`, ...g.sentences.map((s) => `    - ${s}`)].join(
        "\n",
      ),
    )
    .join("\n\n");
  return [
    "You are integrating per-aspect draft answers into ONE coherent answer for the topic below.",
    "",
    "WHAT YOU MUST PRESERVE - this is the top priority:",
    "- EVERY distinct factual claim in the drafts must survive into your output. Do not summarize, do not compress, do not drop details.",
    "- Keep named entities, numbers, dates, organizations and places EXACTLY as written in the drafts.",
    "- If two drafts state the same fact, merge them into one sentence. That is the ONLY kind of removal allowed.",
    "",
    "WHAT YOU MAY CHANGE:",
    "- sentence order, so related facts sit together and the answer reads as one piece;",
    "- wording at sentence boundaries, so the text flows (transitions, pronouns, joining two short sentences about the same thing).",
    "",
    "SOURCE MARKING - every output sentence must carry it:",
    `- each sentence must list the draft block indices [0..${input.groups.length - 1}] it draws on, in a "sources" array;`,
    "- list ONLY blocks that actually contain the claim you wrote. Prefer ONE block per sentence.",
    "- a sentence that merges facts from two blocks must list both.",
    "",
    `LENGTH: the complete answer MUST total between ${input.minWords} and ${input.maxWords} words.`,
    "Do not write an introduction or a conclusion. No hedging. Only supported facts.",
    "",
    "Return only strict JSON, no Markdown fences:",
    '{"answer":[{"text":"A factual sentence.","sources":[0]}]}',
    "",
    formatTopic(input.topic),
    "",
    "Per-aspect draft answers:",
    blocks,
  ].join("\n");
}
