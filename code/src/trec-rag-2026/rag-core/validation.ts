import {
  CLIMBMIX_DOCID_RE,
  type RagRunConfig,
  type AgenticRagOutputObject,
  type RagReferenceNormalizationResult,
  type TopicIdentity,
  type ValidationIssue,
  type ValidationResult,
} from "./contracts";

/**
 * Post-submission validator hardened against the 2026-07-29 track rules
 * (trec-rag-skills commit f281e88).  The official shape rules and our
 * producer-only invariants are deliberately kept in separate functions.
 */
export class AgenticRagValidationError extends Error {
  readonly issues: ValidationIssue[];

  constructor(message: string, issues: ValidationIssue[]) {
    super(message);
    this.name = "AgenticRagValidationError";
    this.issues = issues;
  }
}

type ValidationArgs = {
  config: RagRunConfig;
  topic: TopicIdentity;
  readDocids: Set<string>;
};

export function parseRagOutputObjectJson(text: string): unknown {
  try {
    return JSON.parse(text);
  } catch (error) {
    throw new AgenticRagValidationError("RAG output is not valid JSON.", [
      {
        code: "INVALID_JSON",
        message: error instanceof Error ? error.message : String(error),
      },
    ]);
  }
}

/** Validate only organizer-defined format rules. Extra metadata is allowed. */
export function validateOfficialRagOutputObject(
  value: unknown,
): ValidationResult {
  const issues = collectOfficialIssues(value);
  return { ok: issues.length === 0, issues };
}

/**
 * Backward-compatible producer validator.  It returns the canonical internal
 * representation, where direct-docid citations have been resolved to indices.
 */
export function validateRagOutputObject(
  value: unknown,
  args: ValidationArgs,
  _options: { allowUncitedReferences?: boolean } = {},
): AgenticRagOutputObject {
  const canonical = resolveDirectDocidCitations(value);
  const issues = [
    ...collectOfficialIssues(canonical),
    ...collectProducerPolicyIssues(canonical, args),
  ];
  if (issues.length > 0) {
    throw new AgenticRagValidationError(
      `Invalid RAG output object: ${issues
        .map((issue) => issue.code)
        .join(",")
        .slice(0, 300)}`,
      issues,
    );
  }
  return canonical as AgenticRagOutputObject;
}

export function validateRagOutputObjectStrict(
  value: unknown,
  args: ValidationArgs,
): ValidationResult {
  const canonical = resolveDirectDocidCitations(value);
  const issues = [
    ...collectOfficialIssues(canonical),
    ...collectProducerPolicyIssues(canonical, args),
  ];
  return { ok: issues.length === 0, issues };
}

/**
 * Sanitize a model draft before final validation: resolve direct docids,
 * discard unread/duplicate references, remove invalid citations, then prune
 * references that the model did not cite.  These are producer choices, not
 * organizer requirements.
 */
export function normalizeRagOutputObjectReferences(
  value: unknown,
  args: ValidationArgs,
): RagReferenceNormalizationResult {
  const draft = sanitizeModelDraft(value, args.readDocids);
  const validated = validateRagOutputObject(draft, args);
  const cited = new Set<number>();
  for (const sentence of validated.answer) {
    for (const citation of sentence.citations) cited.add(citation);
  }

  const oldToNew = new Map<number, number>();
  const references: string[] = [];
  validated.references.forEach((reference, oldIndex) => {
    if (!cited.has(oldIndex)) return;
    oldToNew.set(oldIndex, references.length);
    references.push(reference);
  });

  const normalized: AgenticRagOutputObject = {
    metadata: validated.metadata,
    references,
    answer: validated.answer.map((sentence) => ({
      text: sentence.text,
      citations: [
        ...new Set(
          sentence.citations.map((citation) => {
            const remapped = oldToNew.get(citation);
            if (remapped === undefined) {
              throw new AgenticRagValidationError(
                "Citation could not be remapped.",
                [
                  {
                    code: "CITATION_REMAP_FAILED",
                    message: `citation ${citation} has no retained reference`,
                  },
                ],
              );
            }
            return remapped;
          }),
        ),
      ],
    })),
  };

  const finalObject = validateRagOutputObject(normalized, args);
  return {
    ragObject: finalObject,
    removedReferenceCount: validated.references.length - references.length,
  };
}

function collectOfficialIssues(value: unknown): ValidationIssue[] {
  const issues: ValidationIssue[] = [];
  if (!isRecord(value))
    return [
      { code: "INVALID_SHAPE", message: "RAG output must be an object." },
    ];
  requireExactKeys(
    value,
    ["metadata", "references", "answer"],
    "top-level",
    issues,
  );

  const metadata = value.metadata;
  if (!isRecord(metadata)) {
    issues.push({
      code: "INVALID_METADATA",
      message: "metadata must be an object.",
    });
  } else {
    requireKeys(
      metadata,
      ["team_id", "run_id", "narrative_id", "narrative", "run_desc"],
      "metadata",
      issues,
    );
    for (const key of [
      "team_id",
      "run_id",
      "narrative_id",
      "narrative",
      "run_desc",
    ]) {
      if (typeof metadata[key] !== "string" || metadata[key].trim() === "") {
        issues.push({
          code: "INVALID_METADATA_VALUE",
          message: `metadata.${key} must be non-empty.`,
        });
      }
    }
  }

  const references = value.references;
  if (!Array.isArray(references)) {
    issues.push({
      code: "INVALID_REFERENCES",
      message: "references must be an array.",
    });
  }
  const referenceStrings = Array.isArray(references)
    ? validateReferences(references, issues)
    : [];

  const answer = value.answer;
  if (!Array.isArray(answer) || answer.length === 0) {
    issues.push({
      code: "INVALID_ANSWER",
      message: "answer must be a non-empty array.",
    });
    return issues;
  }

  let words = 0;
  answer.forEach((sentence, sentenceIndex) => {
    if (!isRecord(sentence)) {
      issues.push({
        code: "INVALID_ANSWER_SENTENCE",
        message: `answer[${sentenceIndex}] must be an object.`,
      });
      return;
    }
    requireExactKeys(
      sentence,
      ["text", "citations"],
      `answer[${sentenceIndex}]`,
      issues,
    );
    if (typeof sentence.text !== "string" || sentence.text.trim() === "") {
      issues.push({
        code: "INVALID_SENTENCE_TEXT",
        message: `answer[${sentenceIndex}].text must be non-empty.`,
      });
    } else {
      words += sentence.text.trim().split(/\s+/).filter(Boolean).length;
    }

    if (!Array.isArray(sentence.citations)) {
      issues.push({
        code: "INVALID_CITATIONS",
        message: `answer[${sentenceIndex}].citations must be an array.`,
      });
      return;
    }
    if (sentence.citations.length > 3) {
      issues.push({
        code: "TOO_MANY_CITATIONS",
        message: `answer[${sentenceIndex}] has more than three citations.`,
      });
    }

    const seen = new Set<string>();
    sentence.citations.forEach((citation, citationIndex) => {
      const identity = citationIdentity(citation, referenceStrings);
      if (identity === undefined) {
        issues.push({
          code: Number.isInteger(citation)
            ? "CITATION_OUT_OF_RANGE"
            : "INVALID_CITATION",
          message: `answer[${sentenceIndex}].citations[${citationIndex}] must be a valid reference index or exact docid.`,
        });
        return;
      }
      if (seen.has(identity)) {
        issues.push({
          code: "DUPLICATE_SENTENCE_CITATION",
          message: `answer[${sentenceIndex}] repeats citation ${identity}.`,
        });
      }
      seen.add(identity);
    });
  });

  if (words > 1024) {
    issues.push({
      code: "ANSWER_TOO_LONG",
      message: `answer is ${words} words; the limit is 1024.`,
    });
  }
  return issues;
}

function collectProducerPolicyIssues(
  value: unknown,
  args: ValidationArgs,
): ValidationIssue[] {
  if (
    !isRecord(value) ||
    !isRecord(value.metadata) ||
    !Array.isArray(value.references)
  )
    return [];
  const issues: ValidationIssue[] = [];
  const metadata = value.metadata;

  // These fields/invariants describe our producer, not the organizer's minimum schema.
  requireKeys(
    metadata,
    ["type", "title", "prompt"],
    "producer metadata",
    issues,
  );
  if (metadata.team_id !== args.config.teamId) {
    issues.push({
      code: "TEAM_ID_MISMATCH",
      message: "metadata.team_id must match producer config.",
    });
  }
  if (metadata.run_id !== args.config.runId) {
    issues.push({
      code: "RUN_ID_MISMATCH",
      message: "metadata.run_id must match producer config.",
    });
  }
  if (metadata.type !== "automatic") {
    issues.push({
      code: "INVALID_TYPE",
      message: 'producer metadata.type must equal "automatic".',
    });
  }
  if (metadata.narrative_id !== args.topic.qid) {
    issues.push({
      code: "NARRATIVE_ID_MISMATCH",
      message: "metadata.narrative_id must preserve qid.",
    });
  }
  if (metadata.title !== "") {
    issues.push({
      code: "INVALID_DEV_TITLE",
      message: 'producer metadata.title must equal "" in dev mode.',
    });
  }
  if (metadata.narrative !== args.topic.narrative) {
    issues.push({
      code: "NARRATIVE_MISMATCH",
      message: "metadata.narrative must preserve the topic narrative.",
    });
  }
  if (metadata.prompt !== args.config.promptVersion) {
    issues.push({
      code: "PROMPT_VERSION_MISMATCH",
      message: "metadata.prompt must match producer config.",
    });
  }
  value.references.forEach((reference, index) => {
    if (typeof reference === "string" && !args.readDocids.has(reference)) {
      issues.push({
        code: "REFERENCE_NOT_READ",
        message: `references[${index}] was not read by the producer.`,
      });
    }
  });
  return issues;
}

function resolveDirectDocidCitations(value: unknown): unknown {
  if (
    !isRecord(value) ||
    !Array.isArray(value.references) ||
    !Array.isArray(value.answer)
  )
    return value;
  const references = value.references;
  if (!references.every((reference) => typeof reference === "string"))
    return value;
  const indexByDocid = new Map<string, number>();
  references.forEach((reference, index) =>
    indexByDocid.set(reference as string, index),
  );
  return {
    ...value,
    answer: value.answer.map((sentence) => {
      if (!isRecord(sentence) || !Array.isArray(sentence.citations))
        return sentence;
      return {
        ...sentence,
        citations: sentence.citations.map((citation) =>
          typeof citation === "string" && indexByDocid.has(citation)
            ? indexByDocid.get(citation)
            : citation,
        ),
      };
    }),
  };
}

function sanitizeModelDraft(value: unknown, readDocids: Set<string>): unknown {
  const resolved = resolveDirectDocidCitations(value);
  if (
    !isRecord(resolved) ||
    !Array.isArray(resolved.references) ||
    !Array.isArray(resolved.answer)
  )
    return resolved;
  if (!resolved.references.every((reference) => typeof reference === "string"))
    return resolved;

  const references: string[] = [];
  const oldToNew = new Map<number, number>();
  const indexByDocid = new Map<string, number>();
  resolved.references.forEach((reference, oldIndex) => {
    const docid = reference as string;
    if (!readDocids.has(docid)) return;
    const existing = indexByDocid.get(docid);
    if (existing !== undefined) {
      oldToNew.set(oldIndex, existing);
      return;
    }
    oldToNew.set(oldIndex, references.length);
    indexByDocid.set(docid, references.length);
    references.push(docid);
  });

  return {
    ...resolved,
    references,
    answer: resolved.answer.map((sentence) => {
      if (!isRecord(sentence) || !Array.isArray(sentence.citations))
        return sentence;
      const citations: number[] = [];
      for (const citation of sentence.citations) {
        if (!Number.isInteger(citation)) continue;
        const mapped = oldToNew.get(citation as number);
        if (mapped !== undefined && !citations.includes(mapped))
          citations.push(mapped);
      }
      return { ...sentence, citations };
    }),
  };
}

function validateReferences(
  references: unknown[],
  issues: ValidationIssue[],
): string[] {
  const result: string[] = [];
  const seen = new Set<string>();
  references.forEach((reference, index) => {
    if (typeof reference !== "string" || reference.trim() === "") {
      issues.push({
        code: "INVALID_REFERENCE",
        message: `references[${index}] must be a non-empty string.`,
      });
      return;
    }
    if (reference !== reference.trim() || !CLIMBMIX_DOCID_RE.test(reference)) {
      issues.push({
        code: "INVALID_CLIMBMIX_DOCID",
        message: `references[${index}] is not an exact ClimbMix docid.`,
      });
    }
    if (seen.has(reference)) {
      issues.push({
        code: "DUPLICATE_REFERENCE",
        message: `Duplicate reference: ${reference}`,
      });
    }
    seen.add(reference);
    result.push(reference);
  });
  return result;
}

function citationIdentity(
  citation: unknown,
  references: string[],
): string | undefined {
  if (Number.isInteger(citation)) {
    const index = citation as number;
    return index >= 0 && index < references.length
      ? references[index]
      : undefined;
  }
  if (typeof citation === "string" && references.includes(citation))
    return citation;
  return undefined;
}

function requireKeys(
  value: Record<string, unknown>,
  expectedKeys: string[],
  label: string,
  issues: ValidationIssue[],
): void {
  for (const key of expectedKeys) {
    if (!Object.hasOwn(value, key)) {
      issues.push({
        code: "MISSING_FIELD",
        message: `Missing field in ${label}: ${key}`,
      });
    }
  }
}

function requireExactKeys(
  value: Record<string, unknown>,
  expectedKeys: string[],
  label: string,
  issues: ValidationIssue[],
): void {
  const expected = new Set(expectedKeys);
  for (const key of Object.keys(value)) {
    if (!expected.has(key))
      issues.push({
        code: "EXTRA_FIELD",
        message: `Unexpected field in ${label}: ${key}`,
      });
  }
  requireKeys(value, expectedKeys, label, issues);
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
