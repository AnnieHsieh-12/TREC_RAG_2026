import { pathToFileURL } from "node:url";
import { runFinalRagPipeline, type IterativeOptions } from "./runner";
const USAGE = `Usage: npm run run:rag

Required environment variables:
  TOPICS, CHECKLIST, OPENAI_API_KEY

Optional environment variables:
  QRELS_DIR, OUT, RUN_ID, TEAM_ID, SHARDS, SIDECAR_URLS, PYSERINI_TOKENS,
  SUBMISSION_OUT, PYTHON`;
export function parse(
  argv: string[],
  env: NodeJS.ProcessEnv = process.env,
): IterativeOptions {
  const r: Record<string, string | boolean | undefined> = {};
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (
      a === "--force" ||
      a === "--resume" ||
      a === "--layer-rerank" ||
      a === "--layer-passages" ||
      a === "--layer-verify" ||
      a === "--layer-gapfill"
    ) {
      r[a.slice(2)] = true;
      continue;
    }
    if (!a.startsWith("--")) throw new Error(`Unexpected ${a}`);
    const v = argv[++i];
    if (!v || v.startsWith("--")) throw new Error(`Missing ${a}`);
    r[a.slice(2)] = v;
  }
  const provider = str(r["llm-provider"]) || "nchc_llm";
  const llm =
    provider === "codex_llm"
      ? {
          provider: "codex_llm",
          model: str(r["llm-model"]) || "gpt-5.6-sol",
          baseUrl:
            str(r["sidecar-url"]) || env.SIDECAR_URL || "http://127.0.0.1:8765",
          temperature: 0,
          maxTokens: 2048,
        }
      : provider === "openai_llm"
        ? {
            provider: "openai_llm",
            model: str(r["llm-model"]) || "gpt-5.6-terra",
            apiKeyEnv: str(r["llm-api-key-env"]) || "OPENAI_API_KEY",
            temperature: 0,
            maxTokens: 2048,
          }
        : {
            provider: "nchc_llm",
            model: str(r["llm-model"]) || "gpt-oss-120b",
            apiKeyEnv: str(r["llm-api-key-env"]) || "NCHC_API_KEY",
            baseUrl:
              env.NCHC_BASE_URL || "https://portal.genai.nchc.org.tw/api/v1",
            temperature: 0,
            maxTokens: 2048,
          };
  return {
    runId: req(r["run-id"], "--run-id"),
    teamId: str(r["team-id"]) || "cfda",
    outputDir: req(r["output-dir"], "--output-dir"),
    topicsPath: req(r.topics, "--topics"),
    qrelsDir: str(r["qrels-dir"]) || undefined,
    pyseriniBaseUrl:
      str(r["pyserini-base-url"]) ||
      env.PYSERINI_BASE_URL ||
      "http://api.castorini.uwaterloo.ca",
    pyseriniIndex:
      str(r["pyserini-index"]) || env.PYSERINI_INDEX || "climbmix-400b",
    pyseriniTokenEnv: str(r["pyserini-token-env"]) || "PYSERINI_API_TOKEN",
    limitTopics: r["limit-topics"] ? Number(r["limit-topics"]) : undefined,
    initialDocs: r["initial-docs"] ? Number(r["initial-docs"]) : 5,
    docsPerIteration: r["docs-per-iteration"]
      ? Number(r["docs-per-iteration"])
      : 4,
    maxDocumentsRead: r["max-documents-read"]
      ? Number(r["max-documents-read"])
      : 12,
    maxIterations: r["max-iterations"] ? Number(r["max-iterations"]) : 3,
    documentReadLimit: r["document-read-limit"]
      ? Number(r["document-read-limit"])
      : 200,
    llm,
    force: r.force === true,
    resume: r.resume === true,
    env,
    sidecarUrl: str(r["sidecar-url"]) || env.SIDECAR_URL || "",
    layerRerank: r["layer-rerank"] === true,
    layerPassages: r["layer-passages"] === true,
    layerChecklist: str(r["layer-checklist"]) || undefined,
    layerVerify: r["layer-verify"] === true,
    verifyMode: (str(r["verify-mode"]) === "weaken" ? "weaken" : "drop") as
      | "drop"
      | "weaken",
    runDesc: str(r["run-desc"]) || undefined,
    answerStyle: (str(r["answer-style"]) === "dense" ? "dense" : "standard") as
      | "standard"
      | "dense",
    layerGapfill: r["layer-gapfill"] === true,
    evidenceSelect: (str(r["evidence-select"]) === "setr" ? "setr" : "topk") as
      | "topk"
      | "setr",
  };
}
function str(v: unknown) {
  return typeof v === "string" ? v.trim() : "";
}
function req(v: unknown, n: string) {
  const s = str(v);
  if (!s) throw new Error(`Missing ${n}`);
  return s;
}
async function main() {
  if (process.argv.includes("--help") || process.argv.includes("-h")) {
    console.log(USAGE);
    return;
  }
  console.log(
    JSON.stringify(
      await runFinalRagPipeline(parse(process.argv.slice(2))),
      null,
      2,
    ),
  );
}
if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href)
  main().catch((e) => {
    console.error(e instanceof Error ? e.message : String(e));
    process.exitCode = 1;
  });
