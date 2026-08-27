export const RAG_PROMPT_PROFILE = "cfda_rag_final";
export const CLIMBMIX_DOCID_RE = /^shard_\d+_\d+$/;

export type RagRunMode = "automatic";

export type AgenticRagOutputMetadata = {
  team_id: string;
  run_id: string;
  type: "automatic";
  narrative_id: string;
  title: string;
  narrative: string;
  prompt: string;
  // v0.6.0 required: short description of the submitted system/run.
  run_desc: string;
  // v0.6.0: participant-defined metadata fields are allowed.
  generator?: string;
  retrieval_depth?: number;
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

export type RagRunConfig = {
  runId: string;
  teamId: string;
  mode: RagRunMode;
  promptVersion: string;
  runDesc: string;
  layersDesc?: string;
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
