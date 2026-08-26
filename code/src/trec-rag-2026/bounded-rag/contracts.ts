export const AGENTIC_RAG_BASELINE_PROMPT_VERSION = "agentic_rag_baseline_v1";
export const CLIMBMIX_DOCID_RE = /^shard_\d+_\d+$/;

export type AgenticRagBaselineMode = "dev";

// The five official required keys are
// team_id / narrative_id / narrative / run_id / run_desc; any additional
// participant-defined fields are explicitly allowed.
export type AgenticRagOutputMetadata = {
  team_id: string;
  run_id: string;
  run_desc: string;
  type: "automatic";
  narrative_id: string;
  title: string;
  narrative: string;
  prompt: string;
} & Record<string, unknown>;

/** Organizer-facing input may cite by zero-based index or exact ClimbMix docid. */
export type RawRagCitation = number | string;

/** Canonical producer representation after direct docids are resolved. */
export type AgenticRagAnswerSentence = {
  text: string;
  citations: number[];
};

export type AgenticRagOutputObject = {
  metadata: AgenticRagOutputMetadata;
  references: string[];
  answer: AgenticRagAnswerSentence[];
};

export type AgenticRagBaselineConfig = {
  runId: string;
  teamId: string;
  mode: AgenticRagBaselineMode;
  promptVersion: string;
  runDesc: string;
  layersDesc: string;
};

export type TopicIdentity = {
  qid: string;
  title: string;
  narrative: string;
};

export type ValidationIssue = {
  code: string;
  message: string;
  details?: Record<string, unknown>;
};

export type ValidationResult = {
  ok: boolean;
  issues: ValidationIssue[];
};

export type RagReferenceNormalizationResult = {
  ragObject: AgenticRagOutputObject;
  removedReferenceCount: number;
};
