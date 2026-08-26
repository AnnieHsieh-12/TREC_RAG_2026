import assert from "node:assert/strict";
import { FINAL_MODELS, FINAL_POLICY } from "../config/final_pipeline";

const expected = {
  retrieval_policy: "cfda-trec-rag-2026-final",
  output_depth: 5000,
  q2d_enabled: true,
  rerank_depth: 100,
  fusion_dense: true,
  facet_queries: true,
  facet_depth: 1000,
  splice_head_keep: 200,
  per_aspect_generation: true,
  comprehensive_answer: true,
  reflection: true,
  breadth_first: true,
  answer_doc_chars: 1600,
  aspect_max_tokens: 4096,
  answer_max_tokens: 8192,
  aspect_writer: true,
  citation_verify: true,
  llm_revise: true,
  reattribute: true,
  revise_max_tokens: 16384,
  verify_revise: false,
  nugget_loop: false,
};

assert.deepEqual(FINAL_POLICY, expected);
assert.deepEqual(FINAL_MODELS, {
  base: "gpt-oss-120b",
  writer: "codex:gpt-5.6-sol",
  query: "codex:gpt-5.6-sol",
});

console.log("final policy: ok");
