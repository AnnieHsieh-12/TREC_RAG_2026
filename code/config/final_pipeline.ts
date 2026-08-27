import { pathToFileURL } from "node:url";
import {
  runIterativeAgenticRag,
  type PolicyOverride,
} from "../src/trec-rag-2026/agentic-rag/iterative_runner";
import {
  applyFinalModels,
  parse,
} from "../src/trec-rag-2026/agentic-rag/run_iterative_entry";

/**
 * Final competition policy, flattened from the development-time experiment
 * wrappers.
 *
 * Keep every override explicit here so the public entry point does not depend
 * on historical experiment wrappers.
 */
export const FINAL_POLICY: PolicyOverride = {
  retrieval_policy: "cfda-trec-rag-2026-final",

  // Retrieval and candidate construction.
  output_depth: 5000,
  q2d_enabled: true,
  rerank_depth: 100,
  fusion_dense: true,
  facet_queries: true,
  facet_depth: 1000,
  splice_head_keep: 200,

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
  writer: "codex:gpt-5.6-sol",
  query: "codex:gpt-5.6-sol",
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
  const result = await runIterativeAgenticRag({
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
