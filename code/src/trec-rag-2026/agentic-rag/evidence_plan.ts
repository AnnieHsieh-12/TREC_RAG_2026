
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

/** A readable validation failure that lets the caller retry or fall back. */
export class LedgerViolation extends Error {
  constructor(message: string) {
    super(message);
    this.name = "LedgerViolation";
  }
}

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

/** Maps each document ID to the exact text exposed to the model. */
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

  /** Record one evidence span after checking it against exposed text. */
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

function normalize(s: string): string {
  return s.replace(/\s+/g, " ").trim();
}

/**
 * Validate an answer plan and return the supported text/citation pairs.
 */
export function enforceAnswerPlan(args: {
  ledger: ExposureLedger;
  answerPlan: AnswerPlanSentence[];
  subquestions: Subquestion[];
  /** The strict policy requires at least two sentences for explanatory items. */
  policy: "basic" | "strict";
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
    if (evidence.some((record) => normalize(record.claim) !== normalize(text)))
      fail(
        `every evidence record for answer_plan[${index}] must use a claim exactly equal to the final sentence text.`,
      );
    if (evidence[0]?.docid !== sentence.citations[0])
      fail(`answer_plan[${index}] first citation must be the docid of its first evidence record.`);
    for (const citation of sentence.citations) {
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

  for (const subquestion of subquestions.filter((q) => q.priority === "required")) {
    const representedSentenceCount = answerPlan.filter((sentence) =>
      sentence.evidence_ids.some((evidenceId) =>
        evidenceById.get(evidenceId)?.subquestion_ids.includes(subquestion.id),
      ),
    ).length;
    const minimum = policy === "strict" ? minimumCoveredAnswerSentences(subquestion) : 1;
    if (representedSentenceCount < minimum)
      fail(
        `required subquestion ${subquestion.id} must be represented by at least ${minimum} distinct evidence-backed answer sentence${minimum === 1 ? "" : "s"}; received ${representedSentenceCount}.`,
      );
  }
  return verified;
}
