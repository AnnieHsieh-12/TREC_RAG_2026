import { pathToFileURL } from "node:url";
import { runFinalRetrievalPipeline, type IterativeOptions } from "./runner";
export function parse(
  argv: string[],
  env: NodeJS.ProcessEnv = process.env,
): IterativeOptions {
  const r: Record<string, string | boolean | undefined> = {};
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--force" || a === "--resume") {
      r[a.slice(2)] = true;
      continue;
    }
    if (!a.startsWith("--")) throw new Error(`Unexpected ${a}`);
    const v = argv[++i];
    if (!v || v.startsWith("--")) throw new Error(`Missing ${a}`);
    r[a.slice(2)] = v;
  }
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
    llm: llmCfg(str(r["llm-model"]) || "gpt-oss-120b", env),
    ...(str(r["llm-writer-model"])
      ? { llmWriter: llmCfg(str(r["llm-writer-model"]), env) }
      : {}),
    ...(str(r["llm-query-model"])
      ? { llmQuery: llmCfg(str(r["llm-query-model"]), env) }
      : {}),
    force: r.force === true,
    resume: r.resume === true,
    env,
  };
}

export function llmCfg(spec: string, env: NodeJS.ProcessEnv) {
  const i = spec.indexOf(":");
  const provider = i > 0 ? spec.slice(0, i) : "nchc";
  const model = i > 0 ? spec.slice(i + 1) : spec;
  if (provider === "codex")
    return {
      provider: "codex_llm",
      model,
      baseUrl: env.SIDECAR_URL || "http://127.0.0.1:8765",
      temperature: 0,
      maxTokens: 8192,
    };
  if (provider === "openai")
    return {
      provider: "openai_llm",
      model,
      apiKeyEnv: "OPENAI_API_KEY",
      baseUrl: env.OPENAI_BASE_URL || "https://api.openai.com/v1",
      temperature: 0,
      maxTokens: 8192,
    };
  return {
    provider: "nchc_llm",
    model,
    apiKeyEnv: "NCHC_API_KEY",
    baseUrl: env.NCHC_BASE_URL || "https://portal.genai.nchc.org.tw/api/v1",
    temperature: 0,
    maxTokens: 2048,
  };
}

export function applyFinalModels(
  opts: IterativeOptions,
  models: { base?: string; writer?: string; query?: string },
  env: NodeJS.ProcessEnv = process.env,
): IterativeOptions {
  const out = { ...opts };
  if (models.base && !process.argv.includes("--llm-model"))
    out.llm = llmCfg(models.base, env) as any;
  if (models.writer && !opts.llmWriter)
    out.llmWriter = llmCfg(models.writer, env) as any;
  if (models.query && !opts.llmQuery)
    out.llmQuery = llmCfg(models.query, env) as any;
  return out;
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
  console.log(
    JSON.stringify(
      await runFinalRetrievalPipeline(parse(process.argv.slice(2))),
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
