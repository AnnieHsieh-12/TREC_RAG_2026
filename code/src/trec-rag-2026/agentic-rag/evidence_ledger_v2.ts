// Evidence Ledger v2 —— 自 annie/peiju 的 agentic-research 忠實移植（V6，階段 ⑨）。
//
// 來源：
//   annie/peiju/src/trec-rag-2026/agentic-research/finalize_research.ts   （執法規則）
//   annie/peiju/src/trec-rag-2026/agentic-research/session_state.ts       （EvidenceRecord、minimumCoveredAnswerSentences）
//   annie/peiju/src/trec-rag-2026/agentic-research/schemas.ts             （RecordEvidenceParamsSchema）
//   annie/peiju/tests/trec-rag-2026/agentic_rag_v2.test.ts                （record_evidence 的行為）
//
// ⚠️ `record_evidence.ts` 本身在她的 repo 裡是 untracked、沒進版控，複本裡也沒有。
//    這裡照 schema + 測試斷言重建：exact_quote 必須逐字出現在該文件的曝光文字裡，
//    否則 throw；evidence_id 是 `evidence-N`（1 起算）。
//
// 與先前 `evidence_ledger.ts`（近似版）的根本差別：
//   近似版 = 事後檢查。句子先寫完，再回頭用 supportScore >= 0.45 判斷要不要改指/拿掉引用。
//   本版   = 生成時強制。模型必須先交出 evidence record（逐字引文 + 該引文支持的 claim），
//            claim 必須「等於」最終句子；違規直接拋錯，由呼叫端重新生成。
//            —— 這才是原作者拿到 weighted support 0.9402 / FS 89.6% / NS 2.4% 的機制。
//
// 我們的 pipeline 是程式控制迴圈、不是 tool-calling agent，所以對應關係是：
//   她的 read_document / find_in_document 曝光  →  我們的 readDocs（docid → 已讀文字）
//   她的 research plan subquestions            →  我們的 aspects（decomposeAspects 的產物）
//   她的 fail() 讓 agent 重呼叫 finalize        →  我們在生成端重試（帶著錯誤訊息）

export type EvidenceRecord = {
  evidence_id: string;
  event_seq: number;
  docid: string;
  subquestion_ids: string[];
  exact_quote: string;
  claim: string;
};

export type Subquestion = { id: string; original_text: string; priority: "required" | "optional" };

export type AnswerPlanSentence = { text: string; citations: string[]; evidence_ids: string[] };

/** 她的 fail()：拋出可讀的原因，由呼叫端決定重試或退回。 */
export class LedgerViolation extends Error {
  constructor(message: string) {
    super(message);
    this.name = "LedgerViolation";
  }
}

// ── 自 session_state.ts 逐字移植 ──────────────────────────────────────────
// evidence-ledger-v2：「解釋型」的子問題至少要兩句話撐；原子事實一句就夠。
export function minimumCoveredAnswerSentences(subquestion: Subquestion): number {
  const text = subquestion.original_text.trim().toLowerCase();
  const atomicFactPatterns = [
    /\bhow many\b/,
    /\blocation\b/,
    /\bwhere (?:is|are|was|were)\b/,
    /\bwho (?:is|are|was|were)\b/,
    /\bwhen (?:is|are|was|were|did)\b/,
    /\bwhat year\b/,
    /\bwhich year\b/,
    /\bmost important resource\b/,
  ];
  return atomicFactPatterns.some((pattern) => pattern.test(text)) ? 1 : 2;
}

/** 曝光紀錄：docid → 我們真的讀進 prompt 的文字。她的 readExposures/findExposures 的等價物。 */
export class ExposureLedger {
  private exposed = new Map<string, string>();
  readonly records: EvidenceRecord[] = [];
  private seq = 0;

  expose(docid: string, text: string) {
    const prev = this.exposed.get(docid);
    this.exposed.set(docid, prev ? `${prev}\n${text}` : text);
  }

  hasExposedDoc(docid: string): boolean {
    return this.exposed.has(docid);
  }

  exposedText(docid: string): string {
    return this.exposed.get(docid) ?? "";
  }

  /** 自 schemas.ts 的 RecordEvidenceParamsSchema + 測試斷言重建。 */
  recordEvidence(params: {
    docid: string;
    subquestion_ids: string[];
    exact_quote: string;
    claim: string;
  }): EvidenceRecord {
    const { docid, subquestion_ids, exact_quote, claim } = params;
    if (!subquestion_ids.length) throw new LedgerViolation("record_evidence requires at least one subquestion_id.");
    if (exact_quote.trim().length < 20)
      throw new LedgerViolation(`exact_quote must be at least 20 characters: ${JSON.stringify(exact_quote.slice(0, 40))}`);
    if (!claim.trim()) throw new LedgerViolation("record_evidence requires a non-empty claim.");
    if (!this.hasExposedDoc(docid))
      throw new LedgerViolation(`evidence docid was not exposed through read_document/find_in_document: ${docid}`);
    if (!normalize(this.exposedText(docid)).includes(normalize(exact_quote)))
      throw new LedgerViolation(
        `exact_quote does not appear verbatim in the exposed text of ${docid}: ${JSON.stringify(exact_quote.slice(0, 60))}`,
      );
    const record: EvidenceRecord = {
      evidence_id: `evidence-${this.records.length + 1}`,
      event_seq: ++this.seq,
      docid,
      subquestion_ids: [...subquestion_ids],
      exact_quote,
      claim,
    };
    this.records.push(record);
    return record;
  }
}

// 逐字比對前只把空白正規化 —— 換行/縮排是我們送進 prompt 時加的，不該因此判定造假。
function normalize(s: string): string {
  return s.replace(/\s+/g, " ").trim();
}

// ── 自 finalize_research.ts 逐字移植（answer_plan 那一段的四條規則）──────────
/**
 * 檢查 answer_plan 是否合規。任何一條不過就 throw LedgerViolation —— 對應她的 fail()。
 * 回傳通過驗證的句子（只留 text + citations，evidence 已完成它的任務）。
 */
export function enforceAnswerPlan(args: {
  ledger: ExposureLedger;
  answerPlan: AnswerPlanSentence[];
  subquestions: Subquestion[];
  /** "evidence-ledger-v2" 才套用「解釋型子問題至少兩句」的深度要求 */
  policy: "evidence-ledger-v1" | "evidence-ledger-v2";
}): { text: string; citations: string[] }[] {
  const { ledger, answerPlan, subquestions, policy } = args;
  const fail = (msg: string) => {
    throw new LedgerViolation(msg);
  };
  if (!answerPlan.length) fail(`${policy} requires a non-empty answer_plan.`);
  const evidenceById = new Map(ledger.records.map((e) => [e.evidence_id, e]));

  const verified = answerPlan.map((sentence, index) => {
    const text = sentence.text.trim();
    if (!text || text.includes("\n")) fail(`answer_plan[${index}] must be one non-empty line.`);
    if (new Set(sentence.citations).size !== sentence.citations.length)
      fail(`answer_plan[${index}] contains duplicate citations.`);
    if (new Set(sentence.evidence_ids).size !== sentence.evidence_ids.length)
      fail(`answer_plan[${index}] contains duplicate evidence_ids.`);
    const evidence = sentence.evidence_ids.map((id) => {
      const record = evidenceById.get(id);
      if (!record) fail(`answer_plan[${index}] references unknown evidence_id: ${id}`);
      return record!;
    });
    // 規則 2：evidence 的 claim 必須「等於」最終句子 —— 不能先寫句子再補引文。
    if (evidence.some((record) => normalize(record.claim) !== normalize(text)))
      fail(
        `every evidence record for answer_plan[${index}] must use a claim exactly equal to the final sentence text.`,
      );
    // 規則 3：第一個 citation 要對應第一筆 evidence record。
    if (evidence[0]?.docid !== sentence.citations[0])
      fail(`answer_plan[${index}] first citation must be the docid of its first evidence record.`);
    for (const citation of sentence.citations) {
      // 規則 4：沒被真的讀過的文件不能引用（搜尋 snippet 不算）。
      if (!ledger.hasExposedDoc(citation))
        fail(`answer_plan[${index}] citation was not exposed: ${citation}`);
      if (!evidence.some((record) => record.docid === citation))
        fail(`answer_plan[${index}] citation lacks a matching evidence record: ${citation}`);
    }
    for (const record of evidence)
      if (!sentence.citations.includes(record.docid))
        fail(`answer_plan[${index}] evidence ${record.evidence_id} lacks a matching citation.`);
    return { text, citations: [...sentence.citations] };
  });

  // 覆蓋深度：每個 required 子問題要有足夠多的、有證據撐的句子代表它。
  for (const subquestion of subquestions.filter((q) => q.priority === "required")) {
    const representedSentenceCount = answerPlan.filter((sentence) =>
      sentence.evidence_ids.some((evidenceId) =>
        evidenceById.get(evidenceId)?.subquestion_ids.includes(subquestion.id),
      ),
    ).length;
    const minimum = policy === "evidence-ledger-v2" ? minimumCoveredAnswerSentences(subquestion) : 1;
    if (representedSentenceCount < minimum)
      fail(
        `required subquestion ${subquestion.id} must be represented by at least ${minimum} distinct evidence-backed answer sentence${minimum === 1 ? "" : "s"}; received ${representedSentenceCount}.`,
      );
  }
  return verified;
}
