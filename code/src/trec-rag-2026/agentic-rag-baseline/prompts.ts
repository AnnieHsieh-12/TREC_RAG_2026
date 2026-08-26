import { AGENTIC_RAG_BASELINE_PROMPT_VERSION, type TopicIdentity } from "./contracts";

export type EvidenceDocumentForPrompt = {
  docid: string;
  text: string;
};

// ④ LLM grounded revision (support upgrade): the model reads each answer sentence together with the
// exact documents it cites, verifies support, rewrites over-claims to match the evidence, weakens or
// drops unsupported claims. Citations are docid strings (v0.6.0 allows this). One call per topic.
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
    sentences.map((s, i) => `${i + 1}. ${s.text}  — cites: ${s.docids.join(", ")}`).join("\n"),
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
    `Prompt version: ${AGENTIC_RAG_BASELINE_PROMPT_VERSION}`,
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

export function buildFollowupQueryPrompt(input: FollowupQueryPromptInput): string {
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

export function buildAnswerGenerationPrompt(input: AnswerGenerationPromptInput): string {
  return [
    `Prompt version: ${AGENTIC_RAG_BASELINE_PROMPT_VERSION}`,
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

// v4-style comprehensive generation: decompose the narrative into aspects and
// require the answer to cover EVERY aspect, one factual sentence at a time.
// This drives nugget coverage up (short single-pass answers cover ~16%; aspect-
// driven multi-sentence answers cover far more) while staying within the
// official 1024-word / <=3-citations-per-sentence limits.
export function buildComprehensiveAnswerPrompt(input: AnswerGenerationPromptInput & { aspects?: string[]; atomic?: boolean }): string {
  const aspectBlock = input.aspects && input.aspects.length > 0
    ? ["Aspects to cover (write at least one grounded sentence for EACH; do not drop any):", ...input.aspects.map((a, i) => `  ${i + 1}. ${a}`)].join("\n")
    : "First identify every distinct aspect the narrative asks about, then cover each of them.";
  return [
    `Prompt version: ${AGENTIC_RAG_BASELINE_PROMPT_VERSION}`,
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

// Per-aspect (v4-style) sub-answer: given ONE aspect and the documents retrieved
// specifically for it, write a few grounded sentences answering just that aspect.
// Merging these per-aspect sub-answers is what drives high nugget coverage
// (each aspect gets its own targeted evidence instead of sharing one pool).
export function buildAspectAnswerPrompt(input: { topic: TopicIdentity; aspect: string; documents: EvidenceDocumentForPrompt[]; alreadyWritten?: string[]; atomic?: { maxWords: number; sentences: number }; expectedFacts?: string[] }): string {
  const already = input.alreadyWritten && input.alreadyWritten.length > 0
    ? ["Already written for this aspect (do NOT repeat these; only add NEW distinct facts not covered here):", ...input.alreadyWritten.map((s) => `  - ${s}`), ""].join("\n")
    : "";
  // 面向標題只說「寫這個主題」，expected_facts 說「要講到什麼」—— 後者才是 nugget 在量的。
  // 措辭刻意保守：這些是分解階段從敘述猜的，沒有證據支撐，所以只能當「該找什麼」的指引，
  // 絕不可以讓模型把它們當成事實直接寫進答案（那會製造無憑據宣稱，NS 會爆掉）。
  const wanted = input.expectedFacts && input.expectedFacts.length > 0
    ? ["What a complete answer is expected to state about this aspect (use as a checklist of what to LOOK FOR in the evidence):",
       ...input.expectedFacts.map((f) => `  - ${f}`),
       "Only write those of the above that the evidence actually supports, and write them with the evidence's own specifics. NEVER state one of them just because it is listed here — an unsupported sentence costs more than a missing one.",
       ""].join("\n")
    : "";
  return [
    "You write a few evidence-grounded sentences answering ONE specific aspect of a TREC RAG narrative.",
    "Use only the provided evidence documents. Do not use outside knowledge.",
    `Aspect to answer: ${input.aspect}`,
    wanted,
    already,
    // ⑧ 句型。預設是「列舉式長句」—— 刻意往涵蓋率調的（一句塞多個 claim）。
    // atomic 模式（V2b）反過來：一句一事實、一句一來源。理由見 versions/V2b.ts。
    // 這是同一個階段的兩個候選，不是誰對誰錯 —— 團隊在別的線上量過，
    // 列舉式涵蓋較高但支持度崩，原子句相反。本線要用統一評分器重新量一次。
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
    // 得分的 claim 是「X 是個挑戰」這種同族短陳述，所以「把集合的每個成員都點名」是涵蓋率的主要來源。
    // ⚠️ 但這條規則的實作方式必須跟句型模式一致，否則 prompt 會自相矛盾：
    // atomic 模式叫模型「五個群體寫五句」，而這裡原本要求「一句話點完五個」，
    // 連 Good/Bad 例句都用到同一個句子（一邊標 Good、一邊標 Bad）。
    // V2b 就是這樣跑的 —— 原子指令壓過了集合列舉，涵蓋率從 0.4729 崩到 0.3311（p=0.000）。
    // 修法：保留「每個成員都要點名」這個目標，但**點名的載體隨模式改變**——
    // 列舉式放同一句，原子式放連續數句。目標不變，形式不打架。
    "This is scored by how many distinct required claims the answer states. Those claims are short thematic statements, and they come in FAMILIES that share a stem and vary one dimension — for example 'racial inclusion is a challenge', 'gender inclusion is a challenge', 'inclusion of disabled athletes is a challenge'.",
    // ⚠️ 非原子分支的字句必須與 V2 逐字相同，否則 V2 的既有分數不可比（p=0.000 的對照全部作廢）。
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

// v4-style reflection: after a draft answer is assembled, ask the model which
// aspects of the narrative are still missing or thinly covered, so we can run
// extra targeted retrieval+generation passes for them.
export function buildReflectionPrompt(topic: TopicIdentity, answerText: string): string {
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

// One cheap LLM call to list the distinct aspects a narrative asks about.
// ⑦ 面向分解。`expectedFacts` 打開時，每個面向additionally要吐 2-4 條「一個好答案
// 該講到的事實」—— 這是照原作者 checklist 產生器的格式補上的。
//
// 為什麼補這個：他的 checklist 每個面向帶 expected_facts，我們只有標題。
// 面向標題（"Athlete Compensation Structures"）只告訴寫手「要寫這個主題」，
// expected_facts（"NIL 讓大學運動員可以從自己的姓名肖像獲利"）告訴它「要講到什麼」——
// 而 nugget 涵蓋量的正是後者。他的文件也說 gap-fill 之所以變成 no-op，
// 是「once the writer organizes by checklist」，也就是 checklist 本身就把洞補掉了。
//
// ⚠️ 只從敘述文字產生，絕不餵 nugget 或 qrels —— 119 題正式測資沒有 nugget，
//    靠 nugget 產出來的 checklist 在正式跑上無法使用，dev 上的分數也會是假的。
export function buildAspectDecompositionPrompt(topic: TopicIdentity, expectedFacts = false): string {
  return [
    "List every distinct aspect the following narrative asks about, so that together they cover the whole question.",
    "Return ONLY a strict JSON object, no prose:",
    expectedFacts
      ? '{"aspects":[{"title":"aspect one","expected_facts":["a fact a good answer would state","another fact"]}]}'
      : '{"aspects":["aspect one","aspect two"]}',
    "Rules: 8-12 aspects; each a short noun phrase (3-10 words); split the narrative FINELY so each aspect maps to a distinct sub-question a good answer must address; cover every distinct thing asked; do not overlap; do not add aspects the narrative does not ask about.",
    "Prefer MORE, NARROWER aspects over few broad ones: the answer is scored on how many distinct required facts it covers, so a fine split gives each fact its own targeted evidence.",
    ...(expectedFacts
      ? ["For each aspect also give 2-4 expected_facts: each a single short declarative sentence stating a fact a complete answer would state about that aspect.",
         "expected_facts are what a good answer SHOULD say, written from the narrative alone - you have no evidence documents here, so state them as the kind of claim to look for, not as verified fact.",
         "Make them specific and checkable (who did what, with what effect), not restatements of the aspect title."]
      : []),
    "",
    formatTopic(topic),
  ].join("\n");
}

export function buildCompactAnswerGenerationPrompt(input: AnswerGenerationPromptInput): string {
  return [
    "Return only this JSON shape, with no Markdown or explanation:",
    '{"references":["shard_00000_00000"],"answer":[{"text":"Supported sentence.","citations":[0]}]}',
    "Use only the provided docs. Every sentence needs citations. Citation numbers are references array indexes. Use only cited docids in references.",
    formatTopic(input.topic),
    "Docs:",
    formatDocuments(input.documents.slice(0, 5).map((doc) => ({ ...doc, text: doc.text.slice(0, 800) }))),
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
    .map((document, index) => [
      `Document ${index}:`,
      `docid: ${document.docid}`,
      "text:",
      document.text,
    ].join("\n"))
    .join("\n\n");
}

// ---- 以下自 的 team-stack-w4 移植（V3 密集寫作）----------------------
// 診斷：答案原本只寫 217-245 詞，官方上限 1024（只用 21%；官方範例 Piika 寫 665 詞）。
// 文獻確認寫越長涵蓋越高且無字數懲罰。實測 V_strict 0.262 -> 0.408（+17.6pp），
// 換 gpt-5.6-sol 寫手後 0.414 / FS 96.4%。見 specs/V3.md。

export type DenseAnswerPromptInput = AnswerGenerationPromptInput & {
  checklist?: string[];
  // S4 列舉式句型（側翼）。給了就把「一句一事實」換成「一句點名一個維度裡的每個成員」。
  // 只能在 evidence_ledger 打開時使用 —— 長句更容易夾帶沒被支持的成員，ledger 是唯一的安全網。
  enumerative?: { minWords: number; maxWords: number };
};

// W5-1: fill the 1024-word budget with dense, atomic, entity-rich sentences.
// Motivated by measurement (our answers averaged 213-245 words = 21% of the
// cap) and literature: nugget recall rises monotonically with length
// (Crucible 0.599->0.717->0.812) and the strict-vital metric has no verbosity
// penalty; GINGER's winning shape is short atomic entity-dense sentences
// organized by facet.
export function buildDenseAnswerGenerationPrompt(input: DenseAnswerPromptInput): string {
  const checklistBlock = input.checklist && input.checklist.length > 0
    ? [
        "Organize the answer by these aspects of the topic; write roughly 80-120 words for EACH aspect (a few sentences each), in this order:",
        ...input.checklist.map((c, i) => `${i + 1}. ${c.replace(/ \(vital\)$/, "")}`),
        "",
      ]
    : [];
  return [
    `Prompt version: ${AGENTIC_RAG_BASELINE_PROMPT_VERSION}-dense`,
    "You generate an evidence-grounded TREC RAG answer.",
    "Use only the provided evidence documents. Do not use outside knowledge.",
    "",
    "LENGTH: the complete answer MUST total between 900 and 1010 words. Short answers waste the official budget and lose coverage. Keep adding supported factual sentences until you approach 1010 words.",
    "",
    ...(input.enumerative
      ? [
          // S4：gold nugget 本身是無數字的主題性短句，而且會成家族出現 ——
          // 一句點名整個維度的所有成員，比一句一個事實更容易同時命中一整組 nugget。
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
    ...(process.env.DENSE_DRAFT_HINT ? ["", `FACT SELECTION EMPHASIS: ${process.env.DENSE_DRAFT_HINT}`] : []),
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

// ---- Evidence Ledger v2 的寫作 prompt（V6）-----------------------------
// 對應 annie/peiju 的 record_evidence + finalize_research 協定：她的 agent 是「先呼叫
// record_evidence 把逐字引文釘死，才能在 finalize_research 交出 answer_plan」。
// 我們是單次生成，所以把兩步併成一次 JSON —— 模型必須在同一份輸出裡交出
// 句子與它的逐字引文，規則由 evidence_ledger_v2.ts 逐字執法，違規就重生成。

export type LedgerAnswerPromptInput = DenseAnswerPromptInput & {
  subquestions: { id: string; text: string }[];
  /** 上一次違規的原因；重試時附上，對應她的 fail() 把錯誤丟回給 agent */
  violation?: string;
};

export function buildLedgerAnswerPrompt(input: LedgerAnswerPromptInput): string {
  return [
    `Prompt version: ${AGENTIC_RAG_BASELINE_PROMPT_VERSION}-ledger-v2`,
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
    '{"answer_plan":[{"text":"One factual sentence.","citations":["shard_00000_00000"],'
      + '"evidence":[{"docid":"shard_00000_00000","exact_quote":"verbatim span of at least twenty characters",'
      + '"claim":"One factual sentence.","subquestion_ids":["Q1"]}]}]}',
    "",
    formatTopic(input.topic),
    "Evidence documents:",
    formatDocuments(input.documents),
  ].join("\n");
}

// ---- 逐句驗證改寫（V3）------------------------------------------------
// 對每一句比對它的引用證據：完全支持就原封保留；過度宣稱就改寫成證據支持的範圍；
// 完全沒支持時依 mode 處理 —— drop 直接刪句，weaken 改寫成較弱但有證據的說法。
// weaken 是為了保涵蓋：在 recall 型指標下刪句 = 刪掉它帶的 nugget。

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

export function buildVerifyRevisePrompt(input: VerifyRevisePromptInput): string {
  const sentenceBlocks = input.sentences.map((sentence, index) => [
    `Sentence ${index}: ${sentence.text}`,
    `Citations: ${JSON.stringify(sentence.citations)}`,
    "Cited evidence excerpts:",
    sentence.evidence.length === 0
      ? "(none available)"
      : sentence.evidence.map((e) => `- [${e.docid}] ${e.excerpt}`).join("\n"),
  ].join("\n")).join("\n\n");
  // ⑨ 三種模式，差別在「句子撐不住的時候動什麼」—— 這一個選擇就決定了涵蓋要不要付代價。
  //
  //   weaken       改寫成更弱的說法。動的是「過度延伸但有部分支持」的句子，這類很多，
  //                **而且它們正在涵蓋 nugget** —— 寫弱了引用合格、nugget 卻配不上。實測扣涵蓋 8–10pp。
  //   drop         只刪完全沒支持的句子。那些是模型憑空生的，本來就配不到 nugget，
  //                所以傷得比 weaken 小。原作者用這個模式，FS 96.4% / V_strict 0.4137。
  //   reattribute  **一個字都不改**，只改引用：把句子標到 references 裡真正支持它的那一篇；
  //                真的整份證據都不支持才刪。
  //
  // reattribute 的依據是量出來的：純後處理版（tools/split_citations.py --reattribute）
  // 讓 V2 的 FS 18.3%→37.6%，而涵蓋 0.4729→0.4867 **不減反增** —— 因為 nugget 是拿
  // 整篇文字比對的，文字不動涵蓋就不可能掉。這裡把同一件事交給模型做，
  // 它看得到完整證據，判斷會比後處理的詞彙重疊法準得多。
  const ACTION = {
    weaken: "- Not supported by its citations: rewrite it into a weaker related claim that the excerpts DO support (hedge, narrow, or generalize it); if other cited excerpts support the point, switch its citations to those; remove the sentence only when nothing in any excerpt relates to it.",
    drop: "- Not supported at all: remove the sentence.",
    reattribute: "- Not fully supported by its CURRENT citations: do NOT touch the wording. Instead search ALL the references listed below and re-cite the sentence to whichever reference(s) actually state it. Remove the sentence ONLY if no reference in the entire list supports any part of it.",
  } as const;
  const mode = (input.mode && input.mode in ACTION ? input.mode : "drop") as keyof typeof ACTION;
  return [
    "You are revising a cited RAG answer so that every sentence is fully supported by its cited evidence.",
    "For each sentence below, compare the sentence against its cited evidence excerpts and act:",
    "- Fully supported: keep the sentence unchanged with the same citations.",
    ...(mode === "reattribute"
      // reattribute 模式下連「過度延伸」也不准改寫 —— 改寫就是動文字，涵蓋就會開始流失。
      ? ["- Over-extended (partially supported): keep the wording EXACTLY as written. Only fix its citations."]
      : ["- Over-extended (partially supported): rewrite it so it claims ONLY what the excerpts support. Do not add new claims."]),
    ACTION[mode],
    ...(mode === "reattribute"
      ? ["HARD RULE: you may not add, delete, reorder or reword a single word of any sentence you keep. Your only edit is the citations array. Answers that change wording will be rejected."]
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

// ── 第二層引用：把各面向的子答案整合成一篇（移植 CFDA 2025）──────────────
//
// 去年 CFDA 的 AG pipeline 是兩層引用：
//   第一層  每個 sub-query s_i 只用它自己的證據池 T_i 寫出 a_i
//   第二層  整合 LLM 產出 y = F(q, {[i]: a_i})，**每句標記它取自哪些 a_i**
//   然後    每句的候選證據 = 它標記到的那些 T_i 的聯集，再逐句驗證支持
//
// 我們只有第一層：各面向的句子直接接起來。這造成兩個問題：
//   1. 讀起來是清單不是文章（面向之間沒有銜接）
//   2. **每句的出處範圍是全題的證據**，而不是「它真正取材的那個面向」——
//      我們量到「只引一篇的句子 FS 也只有 39%」，就是因為句子把多篇揉在一起。
//      標記機制把出處收斂回單一面向的池子，是 FS 天花板的直接解方。
//
// ⚠️ 這一層會**重寫文字**，所以有跟 verify_revise(weaken) 同類的風險：
//    改寫時把具體事實抹平 → nugget 配不上 → 涵蓋掉。prompt 裡因此把
//    「保留每一個具體事實」列為最高優先，並明確禁止摘要與精簡。
export function buildIntegrationPrompt(input: {
  topic: TopicIdentity;
  groups: { aspect: string; sentences: string[] }[];
  minWords: number;
  maxWords: number;
}): string {
  const blocks = input.groups
    .map((g, i) => [`[${i}] ${g.aspect}`, ...g.sentences.map((s) => `    - ${s}`)].join("\n"))
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
