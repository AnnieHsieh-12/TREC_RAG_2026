import { pathToFileURL } from "node:url";
import {
  runFinalRetrievalPipeline,
  type PolicyOverride,
} from "../src/trec-rag-2026/retrieval-pipeline/runner";
import {
  applyFinalModels,
  parse,
} from "../src/trec-rag-2026/retrieval-pipeline/cli";

/**
 * Final competition policy, flattened from the development-time experiment
 * wrappers.
 *
 * Keep every override explicit here so the public entry point does not depend
 * on historical experiment wrappers.
 */
export const FINAL_POLICY: PolicyOverride = {
  retrieval_policy: "VFs-vf-with-sol-writing",

  // Retrieval and candidate construction.
  output_depth: 5000,
  bm25_anchor_weight: 1,
  followup_query_weight: 0.25,
  rrf_k: 60,
  q2d_enabled: true,
  q2d_weight: 1,
  q2d_query_repeat: 5,
  rerank_depth: 100,
  fusion_dense: true,
  fusion_bm25_weight: 1,
  fusion_ce_weight: 1,
  fusion_dense_weight: 1,
  fusion_rrf_k: 60,
  facet_queries: true,
  facet_depth: 1000,
  splice_head_keep: 200,
  ce_dead_threshold: 0.5,
  vark_threshold: 0.5,
  vark_min: 4,
  vark_max: 15,
  vark_relative_frac: 0.5,

  // Evidence acquisition and answer generation.
  per_aspect_generation: true,
  comprehensive_answer: true,
  reflection: true,
  breadth_first: true,
  answer_doc_chars: 1600,
  aspect_max_tokens: 4096,
  answer_max_tokens: 8192,
  aspect_writer: true,

  // Citation handling.
  citation_verify: true,
  llm_revise: true,
  reattribute: true,
  revise_max_tokens: 16384,
  verify_revise: false,

  // This experimental loop was not part of the selected policy.
  nugget_loop: false,
};

export const FINAL_MODELS = {
  base: "gpt-oss-120b",
  writer: "openai:gpt-5.6-sol",
  query: "openai:gpt-5.6-sol",
};

const USAGE = `Usage: npm run run:retrieval -- [options]

Required:
  --run-id ID
  --output-dir PATH
  --topics PATH
  --qrels-dir PATH

Optional:
  --team-id ID                 default: pi-serini
  --limit-topics N
  --force
  --resume`;

async function main() {
  if (process.argv.includes("--help") || process.argv.includes("-h")) {
    console.log(USAGE);
    return;
  }
  const options = applyFinalModels(parse(process.argv.slice(2)), FINAL_MODELS);
  const result = await runFinalRetrievalPipeline({
    ...options,
    policy: FINAL_POLICY,
  });
  console.log(JSON.stringify(result, null, 2));
}

if (
  process.argv[1] &&
  import.meta.url === pathToFileURL(process.argv[1]).href
) {
  main().catch((error) => {
    console.error(error instanceof Error ? error.message : String(error));
    process.exitCode = 1;
  });
}
