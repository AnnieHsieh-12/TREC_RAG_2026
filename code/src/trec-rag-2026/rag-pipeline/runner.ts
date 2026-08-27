import {
  existsSync,
  mkdirSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { basename, dirname, join, resolve } from "node:path";
import { parseTrecRag2026TopicsTsv } from "../retrieval/topics";
import { createLlmClient, generateJsonWithRetry } from "../../llm/create";
import {
  normalizeLlmClientConfig,
  safeLlmConfigForArtifacts,
  type RawLlmClientConfig,
} from "../../llm/config";
import type {
  LlmAttemptTrace,
  LlmClient,
  LlmJsonValidationResult,
} from "../../llm/types";
import {
  buildAnswerGenerationPrompt,
  buildCompactAnswerGenerationPrompt,
  buildDenseAnswerGenerationPrompt,
  buildVerifyRevisePrompt,
} from "./prompts";
import {
  RAG_PROMPT_PROFILE,
  type RagRunConfig,
  type AgenticRagOutputObject,
  type TopicIdentity,
} from "../shared-rag/contracts";
import {
  normalizeRagOutputObjectReferences,
  validateRagOutputObjectStrict,
} from "../shared-rag/validation";
import {
  buildExtractiveFallbackAnswerDraft,
  type ReadDocument,
} from "../shared-rag/fallback";
import {
  evaluateRankings,
  type Qrels,
  type Rankings,
} from "../../evaluation/retrieval_metrics";
import { discoverQrelsFiles } from "../../evaluation/qrels_files";

type Topic = { qid: string; title: string; narrative: string };
type Hit = { docid: string; score: number };
type ReadDoc = ReadDocument;
type AnswerDraft = {
  references: string[];
  answer: Array<{ text: string; citations: number[] }>;
};
type Judge = {
  enough: boolean;
  missing_aspects: string[];
  followup_queries: string[];
};
export type IterativeOptions = {
  runId: string;
  teamId: string;
  outputDir: string;
  topicsPath: string;
  qrelsDir: string;
  pyseriniBaseUrl: string;
  pyseriniIndex: string;
  pyseriniTokenEnv: string;
  limitTopics?: number;
  initialDocs: number;
  docsPerIteration: number;
  maxDocumentsRead: number;
  maxIterations: number;
  documentReadLimit: number;
  llm: RawLlmClientConfig;
  force?: boolean;
  resume?: boolean;
  env?: NodeJS.ProcessEnv;
  sidecarUrl?: string;
  layerRerank?: boolean;
  layerPassages?: boolean;
  layerChecklist?: string;
  layerVerify?: boolean;
  verifyMode?: "drop" | "weaken";
  runDesc?: string;
  answerStyle?: "standard" | "dense";
  layerGapfill?: boolean;
  evidenceSelect?: "topk" | "setr";
};
const POLICY = {
  retrieval_policy: "iterative-agentic-anchor-bm25-weighted-rrf-w025",
  output_depth: 1000,
  bm25_anchor_weight: 1,
  followup_query_weight: 0.25,
  rrf_k: 60,
} as const;
const CUTS = [10, 20, 50, 100, 500, 1000],
  NDCG = [10, 20, 100, 1000];
export async function runFinalRagPipeline(o: IterativeOptions) {
  const env = o.env ?? process.env,
    out = resolve(o.outputDir);
  if (o.force) rmSync(out, { recursive: true, force: true });
  mkdirSync(join(out, "topics"), { recursive: true });
  const llmCfg = normalizeLlmClientConfig(o.llm),
    llm = createLlmClient(llmCfg, env);
  const layersDesc =
    ["rerank", "passages", "checklist", "verify"]
      .filter((l) => (o as any)[`layer${l[0].toUpperCase()}${l.slice(1)}`])
      .join("+") || "standard";
  const cfg: RagRunConfig = {
    runId: o.runId,
    teamId: o.teamId,
    mode: "automatic",
    promptVersion: RAG_PROMPT_PROFILE,
    runDesc:
      o.runDesc ||
      `iterative agentic RAG (${layersDesc}); BM25 top-1000 + weighted RRF; generator ${String(o.llm.model ?? "gpt-oss-120b")}`,
    layersDesc,
  };
  const topics = parseTrecRag2026TopicsTsv(
    readFileSync(resolve(o.topicsPath), "utf8"),
  )
    .map((t) => ({ qid: t.topicId, title: "", narrative: t.narrative }))
    .slice(0, o.limitTopics ?? Infinity);
  const checklistByQid = new Map<string, string[]>();
  if (o.layerChecklist) {
    for (const line of readFileSync(resolve(o.layerChecklist), "utf8").split(
      /\r?\n/,
    )) {
      if (!line.trim()) continue;
      const row = JSON.parse(line);
      checklistByQid.set(
        String(row.qid),
        (Array.isArray(row.items) ? row.items : []).map(String),
      );
    }
  }
  writeJson(join(out, "config.json"), {
    run_id: o.runId,
    team_id: o.teamId,
    policy: POLICY,
    initial_docs: o.initialDocs,
    docs_per_iteration: o.docsPerIteration,
    max_documents_read: o.maxDocumentsRead,
    max_iterations: o.maxIterations,
    document_read_limit: o.documentReadLimit,
    llm: safeLlmConfigForArtifacts(llmCfg),
    layers: {
      sidecar_url: o.sidecarUrl ?? null,
      rerank: Boolean(o.layerRerank),
      passages: Boolean(o.layerPassages),
      checklist: o.layerChecklist ?? null,
      verify: Boolean(o.layerVerify),
      rerank_score_note: o.layerRerank
        ? "post-rerank scores are rank-derived (1000-i), not RRF values"
        : null,
    },
  });
  const summary = {
    run_id: o.runId,
    selected_topics: topics.length,
    processed_count: 0,
    failed_count: 0,
    iterations_by_topic: {} as Record<string, number>,
    stop_reason_by_topic: {} as Record<string, string>,
    llm_call_count: 0,
    llm_failed_call_count: 0,
    llm_retry_count: 0,
    judge_fallback_count: 0,
    average_documents_read_successful: 0,
  };
  for (const [idx, t] of topics.entries()) {
    const sp = join(out, "topics", `${t.qid}.status.json`);
    if (o.resume && existsSync(sp) && readIf(sp)?.status === "completed") {
      console.error(`skip ${t.qid}`);
      continue;
    }
    try {
      const r = await processTopic({
        topic: t,
        o,
        llm,
        cfg,
        out,
        env,
        summary,
        checklist: checklistByQid.get(t.qid) ?? [],
      });
      writeJson(
        join(out, "topics", `${t.qid}.iteration_trace.json`),
        r.iterationTrace,
      );
      writeJson(join(out, "topics", `${t.qid}.judge_trace.json`), r.judgeTrace);
      writeJson(
        join(out, "topics", `${t.qid}.retrieval-trace.json`),
        r.retrievalTrace,
      );
      writeJson(join(out, "topics", `${t.qid}.rag-draft.json`), r.ragObject);
      writeJson(join(out, "topics", `${t.qid}.validation.json`), r.validation);
      writeFileSync(
        join(out, "topics", `${t.qid}.runfile.trec`),
        topicRun(t.qid, r.ranking, o.runId),
      );
      writeJson(
        join(out, "topics", `${t.qid}.final_read_docs_trace.json`),
        r.readTrace,
      );
      writeJson(sp, {
        topic_id: t.qid,
        status: "completed",
        stop_reason: r.stopReason,
        iterations: r.iterations,
      });
      summary.processed_count++;
      summary.iterations_by_topic[t.qid] = r.iterations;
      summary.stop_reason_by_topic[t.qid] = r.stopReason;
      console.error(
        `${idx + 1}/${topics.length} ${t.qid} iter=${r.iterations} stop=${r.stopReason} read=${r.readTrace.documents_read_successful}`,
      );
    } catch (e) {
      summary.failed_count++;
      writeJson(sp, {
        topic_id: t.qid,
        status: "failed",
        error: redact(e instanceof Error ? e.message : String(e), env),
      });
      console.error(`${idx + 1}/${topics.length} ${t.qid} FAILED`);
    }
    await sleep(500);
  }
  const rankings = assemble(out, topics),
    runfile = render(
      rankings,
      topics.map((t) => t.qid),
      o.runId,
    );
  writeFileSync(join(out, "candidate_pool_top1000.trec"), runfile);
  writeFileSync(join(out, "retrieval.internal.trec-run.tsv"), runfile);
  const completed = topics.filter(
    (t) =>
      readIf(join(out, "topics", `${t.qid}.status.json`))?.status ===
      "completed",
  );
  const rags = completed.map((t) =>
    readIf(join(out, "topics", `${t.qid}.rag-draft.json`)),
  );
  writeFileSync(
    join(out, "rag_output_trec_rag_2026.jsonl"),
    rags.map((x) => JSON.stringify(x)).join("\n") + (rags.length ? "\n" : ""),
  );
  writeJsonl(
    join(out, "iteration_trace.jsonl"),
    completed
      .map((t) => readIf(join(out, "topics", `${t.qid}.iteration_trace.json`)))
      .flat(),
  );
  const readTraces = completed.map((t) =>
    readIf(join(out, "topics", `${t.qid}.final_read_docs_trace.json`)),
  );
  writeJsonl(join(out, "final_read_docs_trace.jsonl"), readTraces);
  writeJsonl(
    join(out, "retrieval_trace.jsonl"),
    completed.map((t) =>
      readIf(join(out, "topics", `${t.qid}.retrieval-trace.json`)),
    ),
  );
  const failed = topics
    .map((t) => readIf(join(out, "topics", `${t.qid}.status.json`)))
    .filter((s: any) => s?.status === "failed")
    .map((s: any) => ({ topic_id: s.topic_id, error: s.error }));
  writeJson(join(out, "failed_topics.json"), failed);
  const metrics = evalAll(
    qrelsPaths(resolve(o.qrelsDir)),
    topics.map((t) => t.qid),
    rankings,
  );
  writeJson(join(out, "metrics.json"), metrics.summary);
  writeJson(join(out, "per_topic_metrics.json"), metrics.perTopic);
  const validation = {
    ok: failed.length === 0 && rags.length === topics.length,
    output_count: rags.length,
    expected_count: topics.length,
  };
  writeJson(join(out, "validation.json"), validation);
  summary.failed_count = failed.length;
  summary.average_documents_read_successful =
    readTraces.reduce(
      (s: any, x: any) => s + (x?.documents_read_successful ?? 0),
      0,
    ) / (readTraces.length || 1);
  writeJson(join(out, "run-summary.internal.json"), summary);
  writeFileSync(
    join(out, "provenance.md"),
    `# Final RAG pipeline\n\nNo RAGDoll. No qrels query selection. No source_run_dir.\n`,
  );
  if (!validation.ok)
    throw new Error(
      `RAG run incomplete: produced ${validation.output_count}/${validation.expected_count} topics; see validation.json and failed_topics.json`,
    );
  return { outputDir: out, validation };
}
async function processTopic(a: {
  topic: Topic;
  o: IterativeOptions;
  llm: LlmClient;
  cfg: RagRunConfig;
  out: string;
  env: NodeJS.ProcessEnv;
  summary: any;
  checklist: string[];
}) {
  const queries = [a.topic.narrative],
    runs: Hit[][] = [];
  const anchor = await search(a.o, a.topic.narrative, 1000, a.env);
  runs.push(anchor);
  const readDocs = new Map<string, ReadDoc>(),
    failedRead: string[] = [];
  const iterationTrace: any[] = [],
    judgeTrace: any[] = [];
  let ranking = weightedRrf(
      runs,
      [POLICY.bm25_anchor_weight],
      POLICY.output_depth,
      POLICY.rrf_k,
    ),
    stopReason = "max_iterations",
    iterations = 0;
  ranking = await maybeRerank(a.o, a.topic, ranking);
  await readNew({
    o: a.o,
    hits: ranking,
    readDocs,
    failedRead,
    count: a.o.initialDocs,
    query: a.topic.narrative,
    narrative: a.topic.narrative,
    qid: a.topic.qid,
    env: a.env,
  });
  const writePartial = () =>
    writeTopicPartial({
      out: a.out,
      topic: a.topic,
      iterationTrace,
      judgeTrace,
      queries,
      runs,
      ranking,
      readDocs,
      failedRead,
    });
  for (let it = 0; it < a.o.maxIterations; it++) {
    iterations = it + 1;
    let judge: Judge;
    try {
      judge = await judgeEvidence({
        topic: a.topic,
        llm: a.llm,
        readDocs,
        queries,
        out: a.out,
        env: a.env,
        summary: a.summary,
        iteration: it,
        checklist: a.checklist,
      });
      judgeTrace.push({ iteration: it, judge });
    } catch (e) {
      a.summary.judge_fallback_count++;
      const err = redact(e instanceof Error ? e.message : String(e), a.env);
      const code = classifyJudgeStopReason(err);
      stopReason = code;
      judgeTrace.push({ iteration: it, error_code: code, error: err });
      iterationTrace.push({
        topic_id: a.topic.qid,
        iteration: it,
        queries: [...queries],
        documents_read_successful: readDocs.size,
        judge_error: { error_code: code, message: err },
        stop_reason: stopReason,
        ranking_top10: ranking.slice(0, 10),
      });
      writePartial();
      break;
    }
    let iterationStopReason = "continue";
    if (judge.enough) {
      stopReason = "enough";
      iterationStopReason = stopReason;
      iterationTrace.push({
        topic_id: a.topic.qid,
        iteration: it,
        queries: [...queries],
        documents_read_successful: readDocs.size,
        judge,
        stop_reason: iterationStopReason,
        ranking_top10: ranking.slice(0, 10),
      });
      writePartial();
      break;
    }
    if (readDocs.size >= a.o.maxDocumentsRead) {
      stopReason = "max_documents_read";
      iterationStopReason = stopReason;
      iterationTrace.push({
        topic_id: a.topic.qid,
        iteration: it,
        queries: [...queries],
        documents_read_successful: readDocs.size,
        judge,
        stop_reason: iterationStopReason,
        ranking_top10: ranking.slice(0, 10),
      });
      writePartial();
      break;
    }
    if (it === a.o.maxIterations - 1) {
      stopReason = "max_iterations";
      iterationStopReason = stopReason;
      iterationTrace.push({
        topic_id: a.topic.qid,
        iteration: it,
        queries: [...queries],
        documents_read_successful: readDocs.size,
        judge,
        stop_reason: iterationStopReason,
        ranking_top10: ranking.slice(0, 10),
      });
      writePartial();
      break;
    }
    const fqs = validateFollowups(
      judge.followup_queries,
      queries,
      a.topic.narrative,
    );
    if (fqs.length === 0) {
      stopReason = "no_valid_followup_query";
      iterationStopReason = stopReason;
      iterationTrace.push({
        topic_id: a.topic.qid,
        iteration: it,
        queries: [...queries],
        documents_read_successful: readDocs.size,
        judge,
        stop_reason: iterationStopReason,
        ranking_top10: ranking.slice(0, 10),
      });
      writePartial();
      break;
    }
    iterationTrace.push({
      topic_id: a.topic.qid,
      iteration: it,
      queries: [...queries],
      documents_read_successful: readDocs.size,
      judge,
      followup_queries: fqs,
      stop_reason: iterationStopReason,
      ranking_top10: ranking.slice(0, 10),
    });
    for (const q of fqs) {
      queries.push(q);
      const h = await search(a.o, q, 1000, a.env);
      runs.push(h);
      await sleep(250);
    }
    ranking = weightedRrf(
      runs,
      [
        POLICY.bm25_anchor_weight,
        ...runs.slice(1).map(() => POLICY.followup_query_weight),
      ],
      POLICY.output_depth,
      POLICY.rrf_k,
    );
    ranking = await maybeRerank(a.o, a.topic, ranking);
    await readNew({
      o: a.o,
      hits: ranking,
      readDocs,
      failedRead,
      count: a.o.docsPerIteration,
      query: "weighted_rrf_fused",
      narrative: a.topic.narrative,
      qid: a.topic.qid,
      env: a.env,
    });
    writePartial();
    if (readDocs.size >= a.o.maxDocumentsRead && it + 1 >= a.o.maxIterations)
      stopReason = "max_iterations";
  }
  if (stopReason === "max_iterations" && readDocs.size >= a.o.maxDocumentsRead)
    stopReason = "max_documents_read";
  let ragObject = await answer({
    topic: a.topic,
    cfg: a.cfg,
    llm: a.llm,
    readDocs,
    out: a.out,
    env: a.env,
    summary: a.summary,
    checklist: a.checklist,
    style: a.o.answerStyle ?? "standard",
    evidenceSelect: a.o.evidenceSelect ?? "topk",
  });
  if (a.o.layerGapfill && a.o.sidecarUrl && a.checklist.length > 0)
    ragObject = await gapFill({
      topic: a.topic,
      cfg: a.cfg,
      llm: a.llm,
      ragObject,
      readDocs,
      checklist: a.checklist,
      o: a.o,
      out: a.out,
      env: a.env,
      summary: a.summary,
    });
  ragObject = trimToWordCap(
    ragObject,
    a.cfg,
    a.topic,
    new Set(readDocs.keys()),
    1020,
  );
  if (a.o.layerVerify)
    ragObject = await verifyRevise({
      topic: a.topic,
      cfg: a.cfg,
      llm: a.llm,
      ragObject,
      out: a.out,
      env: a.env,
      summary: a.summary,
      sidecarUrl: a.o.sidecarUrl,
      mode: a.o.verifyMode ?? "drop",
    });
  ragObject = trimToWordCap(
    ragObject,
    a.cfg,
    a.topic,
    new Set(readDocs.keys()),
    1020,
  );
  if (a.o.sidecarUrl)
    ragObject = await orderCitationsByStrength(ragObject, a.o.sidecarUrl);
  const validation = validateRagOutputObjectStrict(ragObject, {
    config: a.cfg,
    topic: a.topic,
    readDocids: new Set(readDocs.keys()),
  });
  if (!validation.ok)
    throw new Error(
      `RAG validation failed: ${validation.issues.map((i) => i.code).join(",")}`,
    );
  if (process.env.ANSWER_QUALITY_GATE === "1") {
    const gw = ragObject.answer.reduce(
      (n, s) => n + s.text.split(/\s+/).filter(Boolean).length,
      0,
    );
    const cited = ragObject.answer.filter((s) => s.citations.length > 0).length;
    const ratio = ragObject.answer.length ? cited / ragObject.answer.length : 0;
    if (gw < 600 || ragObject.references.length === 0 || ratio < 0.8)
      throw new Error(
        `DEGRADED_ANSWER words=${gw} refs=${ragObject.references.length} cited_ratio=${ratio.toFixed(2)}`,
      );
  }
  const cited = [
    ...new Set(
      ragObject.answer
        .flatMap((s) => s.citations)
        .map((i) => ragObject.references[i])
        .filter(Boolean),
    ),
  ];
  return {
    ranking,
    ragObject,
    validation,
    iterationTrace,
    judgeTrace,
    iterations,
    stopReason,
    readTrace: {
      topic_id: a.topic.qid,
      candidate_pool_size: ranking.length,
      documents_read_attempted: readDocs.size + failedRead.length,
      documents_read_successful: readDocs.size,
      read_docids: [...readDocs.keys()],
      failed_read_docids: failedRead,
      final_answer_cited_docids: cited,
    },
    retrievalTrace: buildRetrievalTrace(a.topic, queries, runs, ranking),
  };
}
async function judgeEvidence(a: {
  topic: TopicIdentity;
  llm: LlmClient;
  readDocs: Map<string, ReadDoc>;
  queries: string[];
  out: string;
  env: NodeJS.ProcessEnv;
  summary: any;
  iteration: number;
  checklist?: string[];
}) {
  // Wording matters on the NCHC gpt-oss endpoint: the original demanding
  const checklistLines =
    a.checklist && a.checklist.length > 0
      ? [
          "Aspect checklist (use when judging sufficiency and choosing follow-up queries; prefer queries that target aspects not yet covered by the evidence):",
          ...a.checklist.map((c) => `- ${c.replace(/ \(vital\)$/, "")}`),
        ]
      : [];
  const prompt = [
    "Return ONLY one JSON object. No markdown. No explanation.",
    'Required schema: {"enough":boolean,"queries":string[]}',
    'If the evidence already supports the topic, return {"enough":true,"queries":[]}.',
    'If evidence is insufficient, return {"enough":false,"queries":[...]} with 1-3 BM25 keyword queries.',
    "Query rules: 4-12 English tokens; keyword phrase, not sentence/question; one aspect only; no duplicates.",
    "Forbidden words: obtain, find, source, sources, detail, detailed, comprehensive, concrete, examples, overview, history, impact, provide, explain.",
    `Previous queries: ${JSON.stringify(a.queries)}`,
    `Topic: ${a.topic.narrative}`,
    ...checklistLines,
    "Evidence:",
    [...a.readDocs.values()]
      .map((d, i) => `[${i}] ${d.docid}\n${d.text.slice(0, 1200)}`)
      .join("\n\n"),
  ].join("\n");
  let lastError = "LLM_JSON_PARSE_FAILED";
  for (let attempt = 1; attempt <= 5; attempt++) {
    const started = Date.now();
    try {
      const attemptPrompt =
        attempt === 1
          ? prompt
          : `${prompt}\n[retry ${attempt}: previous attempt returned an empty message; answer the JSON now]`;
      const result = await a.llm.generate({
        messages: [{ role: "user", content: attemptPrompt }],
        temperature: 0,
        maxTokens: 512,
        responseFormat: "json_object",
      });
      const parsed = parseJudgeResponse(
        result.text,
        a.queries,
        a.topic.narrative,
      );
      recordAttempt({
        attempt: {
          attempt,
          provider: result.provider,
          model: result.model,
          latencyMs: result.latencyMs,
          success: parsed.ok,
          outputChars: result.text.length,
          ...(parsed.ok ? {} : { errorCode: "LLM_JSON_PARSE_FAILED" }),
          ...(result.requestId ? { requestId: result.requestId } : {}),
          ...(result.usage ? { usage: result.usage } : {}),
        },
        stage: "judge",
        qid: a.topic.qid,
        out: a.out,
        env: a.env,
        summary: a.summary,
      });
      if (parsed.ok) return parsed.value;
      lastError = parsed.message;
    } catch (e) {
      lastError = redact(e instanceof Error ? e.message : String(e), a.env);
      recordAttempt({
        attempt: {
          attempt,
          provider: a.llm.provider,
          model: a.llm.model,
          latencyMs: Date.now() - started,
          success: false,
          errorCode: /empty assistant message/i.test(lastError)
            ? "LLM_EMPTY_ASSISTANT_MESSAGE"
            : /429/.test(lastError)
              ? "LLM_RATE_LIMIT"
              : /5\\d\\d/.test(lastError)
                ? "LLM_SERVER_ERROR"
                : "LLM_PROVIDER_FAILED",
          outputChars: 0,
        },
        stage: "judge",
        qid: a.topic.qid,
        out: a.out,
        env: a.env,
        summary: a.summary,
      });
    }
    if (attempt < 5) await sleep(300 * 2 ** attempt);
  }
  throw new Error(lastError || "LLM_JSON_PARSE_FAILED");
}
function validateJudge(
  v: unknown,
  prev: string[],
  anchor: string,
): LlmJsonValidationResult<Judge> {
  if (!isRecord(v) || typeof v.enough !== "boolean")
    return { ok: false, message: "judge shape" };
  const rawQueries = Array.isArray((v as any).queries)
    ? (v as any).queries
    : Array.isArray((v as any).followup_queries)
      ? (v as any).followup_queries
      : [];
  const missing = Array.isArray((v as any).missing_aspects)
    ? (v as any).missing_aspects.map(String)
    : [];
  const legal = validateFollowups(rawQueries, prev, anchor);
  return {
    ok: true,
    value: {
      enough: Boolean((v as any).enough),
      missing_aspects: missing,
      followup_queries: (v as any).enough ? [] : legal,
    },
  };
}
function parseJudgeResponse(
  text: string,
  prev: string[],
  anchor: string,
): LlmJsonValidationResult<Judge> {
  const json = extractObjectCandidate(text);
  if (json) {
    try {
      const parsed = JSON.parse(json);
      return validateJudge(parsed, prev, anchor);
    } catch {}
  }
  const lines = text
    .split(/\r?\n|,/)
    .map((x) =>
      x
        .replace(/^[-*\d.\s\"']+/, "")
        .replace(/[\"']+$/, "")
        .trim(),
    )
    .filter(Boolean);
  const legal = validateFollowups(lines, prev, anchor);
  if (legal.length > 0)
    return {
      ok: true,
      value: { enough: false, missing_aspects: [], followup_queries: legal },
    };
  if (/\benough\b\s*[:=]\s*true|\bsufficient\b/i.test(text))
    return {
      ok: true,
      value: { enough: true, missing_aspects: [], followup_queries: [] },
    };
  return { ok: false, message: "LLM_JSON_PARSE_FAILED" };
}
function extractObjectCandidate(text: string) {
  const fenced = /```(?:json)?\s*([\s\S]*?)```/i.exec(text);
  const t = (fenced ? fenced[1] : text).trim();
  if (t.startsWith("{") && t.endsWith("}")) return t;
  const a = t.indexOf("{"),
    b = t.lastIndexOf("}");
  return a >= 0 && b > a ? t.slice(a, b + 1) : null;
}
const BAD = new Set(
  "obtain find source sources detail detailed comprehensive concrete examples evidence information overview history impact provide directly addresses address account explain and".split(
    " ",
  ),
);
function toks(q: string) {
  return q.toLowerCase().match(/[a-z0-9-]+/g) ?? [];
}
function validateFollowups(qs: unknown[], prev: string[], anchor: string) {
  const prevNorm = new Set(prev.map((q) => toks(q).join(" ")));
  const anchorSet = new Set(toks(anchor));
  const out: string[] = [];
  for (const raw of qs.map(String)) {
    const q = raw.trim().replace(/\s+/g, " ");
    const t = toks(q),
      norm = t.join(" ");
    if (
      t.length < 4 ||
      t.length > 12 ||
      /[?{}.;:]|```/.test(q) ||
      t.some((x) => BAD.has(x)) ||
      prevNorm.has(norm) ||
      out.some((x) => toks(x).join(" ") === norm)
    )
      continue;
    const overlap =
      t.filter((x) => anchorSet.has(x)).length /
      Math.max(1, new Set([...t, ...anchorSet]).size);
    if (overlap > 0.75) continue;
    out.push(q);
    if (out.length >= 3) break;
  }
  return out;
}
async function answer(a: {
  topic: TopicIdentity;
  cfg: RagRunConfig;
  llm: LlmClient;
  readDocs: Map<string, ReadDoc>;
  out: string;
  env: NodeJS.ProcessEnv;
  summary: any;
  checklist?: string[];
  style?: "standard" | "dense";
  evidenceSelect?: "topk" | "setr";
}) {
  let draft: any;
  let docs = [...a.readDocs.values()];
  const dense = a.style === "dense";
  const DENSE_DOCS = Number(process.env.DENSE_DOCS || 12),
    DENSE_CHARS = Number(process.env.DENSE_CHARS || 2500);
  if (dense && a.evidenceSelect === "setr" && docs.length > DENSE_DOCS) {
    docs = await setrSelect(a, docs, DENSE_DOCS);
  }
  try {
    draft = await generateJsonWithRetry({
      client: a.llm,
      messages: [
        {
          role: "user",
          content: dense
            ? buildDenseAnswerGenerationPrompt({
                topic: a.topic,
                checklist: a.checklist ?? [],
                documents: docs.slice(0, DENSE_DOCS).map((d) => ({
                  docid: d.docid,
                  text: d.text.slice(0, DENSE_CHARS),
                })),
              })
            : buildAnswerGenerationPrompt({
                topic: a.topic,
                documents: docs.slice(0, 6).map((d) => ({
                  docid: d.docid,
                  text: d.text.slice(0, 1000),
                })),
              }),
        },
      ],
      temperature: 0,
      maxTokens: dense ? 8192 : 2048,
      validate: validateAnswer,
      stage: "answer_generation",
      maxRequestRetries: 4,
      onAttempt: (attempt) =>
        recordAttempt({
          attempt,
          stage: "answer_generation",
          qid: a.topic.qid,
          out: a.out,
          env: a.env,
          summary: a.summary,
        }),
    });
  } catch {
    try {
      draft = await generateJsonWithRetry({
        client: a.llm,
        messages: [
          {
            role: "user",
            content: buildCompactAnswerGenerationPrompt({
              topic: a.topic,
              documents: docs
                .slice(0, 5)
                .map((d) => ({ docid: d.docid, text: d.text.slice(0, 800) })),
            }),
          },
        ],
        temperature: 0,
        maxTokens: 1024,
        validate: validateAnswer,
        stage: "answer_generation",
        maxRequestRetries: 4,
        onAttempt: (attempt) =>
          recordAttempt({
            attempt,
            stage: "answer_generation",
            qid: a.topic.qid,
            out: a.out,
            env: a.env,
            summary: a.summary,
          }),
      });
    } catch {
      draft = { value: buildExtractiveFallbackAnswerDraft(a.readDocs) };
    }
  }
  const full: AgenticRagOutputObject = {
    metadata: {
      team_id: a.cfg.teamId,
      run_id: a.cfg.runId,
      run_desc: a.cfg.runDesc,
      type: "automatic",
      narrative_id: a.topic.qid,
      title: "",
      narrative: a.topic.narrative,
      prompt: a.cfg.promptVersion,
      generator: a.llm.model,
      layers: a.cfg.layersDesc,
    },
    references: draft.value.references,
    answer: draft.value.answer,
  };
  return normalizeRagOutputObjectReferences(full, {
    config: a.cfg,
    topic: a.topic,
    readDocids: new Set(a.readDocs.keys()),
  }).ragObject;
}
function validateAnswer(v: unknown): LlmJsonValidationResult<AnswerDraft> {
  if (!(isRecord(v) && Array.isArray(v.references) && Array.isArray(v.answer)))
    return { ok: false, message: "answer shape" };
  if (v.answer.length === 0 || v.references.length === 0)
    return { ok: false, message: "answer/references must be non-empty" };
  return { ok: true, value: v as AnswerDraft };
}
async function readNew(a: {
  o: IterativeOptions;
  hits: Hit[];
  readDocs: Map<string, ReadDoc>;
  failedRead: string[];
  count: number;
  query: string;
  narrative?: string;
  qid?: string;
  env: NodeJS.ProcessEnv;
}) {
  let added = 0;
  for (const [i, h] of a.hits.entries()) {
    if (added >= a.count || a.readDocs.size >= a.o.maxDocumentsRead) return;
    if (a.readDocs.has(h.docid)) continue;
    try {
      const d = await readDocMaybePassages(
        a.o,
        h.docid,
        a.narrative,
        a.qid,
        a.env,
      );
      if (d.found) {
        a.readDocs.set(h.docid, {
          docid: h.docid,
          text: d.text,
          truncated: d.truncated,
          rankHint: i + 1,
          query: a.query,
        });
        added++;
      } else a.failedRead.push(h.docid);
    } catch {
      a.failedRead.push(h.docid);
    }
    await sleep(200);
  }
}
async function readDocMaybePassages(
  o: IterativeOptions,
  docid: string,
  narrative: string | undefined,
  qid: string | undefined,
  env: NodeJS.ProcessEnv,
) {
  if (o.layerPassages && o.sidecarUrl && narrative) {
    const res = await sidecarPost(
      o.sidecarUrl,
      "/passages",
      { docid, query: narrative, qid, max_passages: 3 },
      120000,
    );
    if (
      res &&
      res.found &&
      typeof res.text === "string" &&
      res.text.trim().length > 0
    )
      return { found: true, text: res.text, truncated: false };
  }
  return readDoc(o, docid, o.documentReadLimit, env);
}
async function sidecarPost(
  base: string,
  path: string,
  body: unknown,
  timeoutMs = 60000,
): Promise<any | null> {
  try {
    const ac = new AbortController();
    const timer = setTimeout(() => ac.abort(), timeoutMs);
    try {
      const r = await fetch(`${base.replace(/\/+$/, "")}${path}`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(body),
        signal: ac.signal,
      });
      if (!r.ok) return null;
      return await r.json();
    } finally {
      clearTimeout(timer);
    }
  } catch {
    return null;
  }
}
async function maybeRerank(
  o: IterativeOptions,
  topic: Topic,
  ranking: ReturnType<typeof weightedRrf>,
) {
  if (!o.layerRerank || !o.sidecarUrl) return ranking;
  const res = await sidecarPost(
    o.sidecarUrl,
    "/rerank",
    {
      qid: topic.qid,
      query: topic.narrative,
      docids: ranking.map((r) => r.docid),
      depth: 300,
    },
    300000,
  );
  if (!res || !Array.isArray(res.order)) return ranking;
  const byId = new Map(ranking.map((r) => [r.docid, r]));
  const seen = new Set<string>();
  const ordered: typeof ranking = [];
  for (const d of res.order as string[]) {
    const e = byId.get(d);
    if (e && !seen.has(d)) {
      ordered.push(e);
      seen.add(d);
    }
  }
  for (const r of ranking) {
    if (!seen.has(r.docid)) ordered.push(r);
  }
  return ordered.map((e, i) => ({
    docid: e.docid,
    rank: i + 1,
    score: 1000 - i,
  }));
}
async function verifyRevise(a: {
  topic: TopicIdentity;
  cfg: RagRunConfig;
  llm: LlmClient;
  ragObject: AgenticRagOutputObject;
  out: string;
  env: NodeJS.ProcessEnv;
  summary: any;
  sidecarUrl?: string;
  mode?: "drop" | "weaken";
}): Promise<AgenticRagOutputObject> {
  const orig = a.ragObject;
  const tracePath = join(a.out, "topics", `${a.topic.qid}.verify_trace.json`);
  const sentences = orig.answer.map((s) => ({
    text: s.text,
    citations: s.citations,
    docids: [
      ...new Set(s.citations.map((ci) => orig.references[ci]).filter(Boolean)),
    ].slice(0, 2),
  }));
  let evidence: any = null;
  if (a.sidecarUrl) {
    for (let att = 0; att < 3 && !evidence; att++) {
      evidence = await sidecarPost(
        a.sidecarUrl,
        "/sentence_evidence",
        {
          sentences: sentences.map((s) => ({ text: s.text, docids: s.docids })),
          budget_chars: 1200,
        },
        300000,
      );
      if (!evidence) await sleep(2000 * (att + 1));
    }
    if (!evidence) {
      writeJson(tracePath, {
        adopted: false,
        reason: "sentence_evidence_unavailable",
        evidence_available: false,
      });
      return orig;
    }
  }
  const evs: any[] =
    evidence?.sentences ?? sentences.map(() => ({ excerpts: [] }));
  const prompt = buildVerifyRevisePrompt({
    topic: a.topic,
    references: orig.references,
    mode: a.mode ?? "drop",
    sentences: sentences.map((s, i) => ({
      text: s.text,
      citations: s.citations,
      evidence: (evs[i]?.excerpts ?? []).map((e: any) => ({
        docid: String(e.docid),
        excerpt: String(e.text).slice(0, 1200),
      })),
    })),
  });
  try {
    const BATCH = 20;
    let revisedAnswer: Array<{ text: string; citations: number[] }> = [];
    if (sentences.length <= BATCH + 5) {
      const draft = await generateJsonWithRetry({
        client: a.llm,
        messages: [{ role: "user", content: prompt }],
        temperature: 0,
        maxTokens: 8192,
        validate: validateAnswer,
        stage: "verify_revise",
        maxRequestRetries: 3,
        onAttempt: (attempt) =>
          recordAttempt({
            attempt,
            stage: "verify_revise",
            qid: a.topic.qid,
            out: a.out,
            env: a.env,
            summary: a.summary,
          }),
      });
      revisedAnswer = draft.value.answer;
    } else {
      for (let b = 0; b < sentences.length; b += BATCH) {
        const batch = sentences.slice(b, b + BATCH);
        const bp = buildVerifyRevisePrompt({
          topic: a.topic,
          references: orig.references,
          mode: a.mode ?? "drop",
          sentences: batch.map((s2, j) => ({
            text: s2.text,
            citations: s2.citations,
            evidence: (evs[b + j]?.excerpts ?? []).map((e: any) => ({
              docid: String(e.docid),
              excerpt: String(e.text).slice(0, 1200),
            })),
          })),
        });
        const draft = await generateJsonWithRetry({
          client: a.llm,
          messages: [{ role: "user", content: bp }],
          temperature: 0,
          maxTokens: 8192,
          validate: validateAnswer,
          stage: "verify_revise",
          maxRequestRetries: 3,
          onAttempt: (attempt) =>
            recordAttempt({
              attempt,
              stage: "verify_revise",
              qid: a.topic.qid,
              out: a.out,
              env: a.env,
              summary: a.summary,
            }),
        });
        revisedAnswer.push(...draft.value.answer);
      }
    }
    const full: AgenticRagOutputObject = {
      metadata: orig.metadata,
      references: orig.references,
      answer: revisedAnswer,
    };
    const normalized = normalizeRagOutputObjectReferences(full, {
      config: a.cfg,
      topic: a.topic,
      readDocids: new Set(orig.references),
    }).ragObject;
    const check = validateRagOutputObjectStrict(normalized, {
      config: a.cfg,
      topic: a.topic,
      readDocids: new Set(orig.references),
    });
    const stats = verifyDiffStats(orig, normalized);
    if (check.ok && normalized.answer.length > 0) {
      writeJson(tracePath, {
        adopted: true,
        evidence_available: Boolean(evidence),
        ...stats,
      });
      return normalized;
    }
    writeJson(tracePath, {
      adopted: false,
      reason: check.ok
        ? "empty_answer"
        : check.issues.map((i) => i.code).join(","),
      evidence_available: Boolean(evidence),
      ...stats,
    });
    return orig;
  } catch (e) {
    writeJson(tracePath, {
      adopted: false,
      reason: redact(e instanceof Error ? e.message : String(e), a.env).slice(
        0,
        300,
      ),
      evidence_available: Boolean(evidence),
    });
    return orig;
  }
}
async function orderCitationsByStrength(
  rag: AgenticRagOutputObject,
  sidecarUrl: string,
): Promise<AgenticRagOutputObject> {
  const multi = rag.answer.filter((s) => s.citations.length > 1);
  if (multi.length === 0) return rag;
  const res = await sidecarPost(
    sidecarUrl,
    "/sentence_evidence",
    {
      sentences: rag.answer.map((s) => ({
        text: s.text,
        docids: s.citations.map((ci) => rag.references[ci]).filter(Boolean),
      })),
      budget_chars: 200,
    },
    180000,
  );
  const evs: any[] = res?.sentences;
  if (!Array.isArray(evs)) return rag;
  const answer = rag.answer.map((s, i) => {
    if (s.citations.length < 2) return s;
    const scores = new Map<string, number>();
    for (const e of evs[i]?.excerpts ?? [])
      scores.set(String(e.docid), Number(e.score) || 0);
    const ordered = [...s.citations].sort(
      (a, b) =>
        (scores.get(rag.references[b]) ?? 0) -
        (scores.get(rag.references[a]) ?? 0),
    );
    return { ...s, citations: ordered };
  });
  return { ...rag, answer };
}
function trimToWordCap(
  rag: AgenticRagOutputObject,
  cfg: RagRunConfig,
  topic: TopicIdentity,
  readDocids: Set<string>,
  cap: number,
): AgenticRagOutputObject {
  let total = rag.answer.reduce(
    (n, s) => n + s.text.split(/\s+/).filter(Boolean).length,
    0,
  );
  if (total <= cap) return rag;
  const answer = [...rag.answer];
  while (answer.length > 1 && total > cap) {
    const last = answer.pop()!;
    total -= last.text.split(/\s+/).filter(Boolean).length;
  }
  const full: AgenticRagOutputObject = {
    metadata: rag.metadata,
    references: rag.references,
    answer,
  };
  return normalizeRagOutputObjectReferences(full, {
    config: cfg,
    topic,
    readDocids,
  }).ragObject;
}
async function gapFill(a: {
  topic: Topic;
  cfg: RagRunConfig;
  llm: LlmClient;
  ragObject: AgenticRagOutputObject;
  readDocs: Map<string, ReadDoc>;
  checklist: string[];
  o: IterativeOptions;
  out: string;
  env: NodeJS.ProcessEnv;
  summary: any;
}): Promise<AgenticRagOutputObject> {
  const rag = a.ragObject;
  const tracePath = join(a.out, "topics", `${a.topic.qid}.gapfill_trace.json`);
  try {
    const aspects = a.checklist.map((c) => c.replace(/ \(vital\)$/, ""));
    const cov = await sidecarPost(
      a.o.sidecarUrl!,
      "/aspect_coverage",
      { aspects, sentences: rag.answer.map((s) => s.text) },
      120000,
    );
    const rows: any[] = cov?.aspects;
    if (!Array.isArray(rows)) {
      writeJson(tracePath, { ok: false, reason: "sidecar_unavailable" });
      return rag;
    }
    const uncovered = rows
      .filter((r) => !r.covered)
      .map((r) => String(r.aspect))
      .slice(0, 3);
    let totalWords = rag.answer.reduce(
      (n, s) => n + s.text.split(/\s+/).filter(Boolean).length,
      0,
    );
    const appended: any[] = [];
    let references = [...rag.references];
    let answer = [...rag.answer];
    for (const aspect of uncovered) {
      if (totalWords >= 950) break;
      const hits = await search(
        a.o,
        `${aspect} ${a.topic.narrative.split(/\s+/).slice(0, 8).join(" ")}`,
        50,
        a.env,
      );
      const fresh = hits
        .filter((h: Hit) => !a.readDocs.has(h.docid))
        .slice(0, 3);
      const docsForAspect: Array<{ docid: string; text: string }> = [];
      for (const h of fresh) {
        try {
          const d = await readDocMaybePassages(
            a.o,
            h.docid,
            aspect,
            a.topic.qid,
            a.env,
          );
          if (d.found) {
            a.readDocs.set(h.docid, {
              docid: h.docid,
              text: d.text,
              truncated: false,
              rankHint: 0,
              query: aspect,
            });
            docsForAspect.push({ docid: h.docid, text: d.text.slice(0, 2200) });
          }
        } catch {
          /* skip unreadable */
        }
        await sleep(150);
      }
      if (docsForAspect.length === 0) continue;
      try {
        const draft = await generateJsonWithRetry({
          client: a.llm,
          messages: [
            {
              role: "user",
              content: [
                "You are extending an evidence-grounded answer with ONE missing aspect.",
                `Aspect to cover: ${aspect}`,
                "Write 3-6 atomic factual sentences (max ~35 words each) strictly from the documents below. Preserve entities, numbers, dates verbatim. No overview or filler sentences.",
                "Citations are zero-indexed positions into YOUR returned references array.",
                'Return only strict JSON: {"references":["docid"],"answer":[{"text":"...","citations":[0]}]}',
                "",
                `Topic:\nnarrative_id: ${a.topic.qid}\nnarrative:\n${a.topic.narrative}`,
                "Documents:",
                docsForAspect
                  .map(
                    (d, i) =>
                      `Document ${i}:\ndocid: ${d.docid}\ntext:\n${d.text}`,
                  )
                  .join("\n\n"),
              ].join("\n"),
            },
          ],
          temperature: 0,
          maxTokens: 2048,
          validate: validateAnswer,
          stage: "gapfill",
          maxRequestRetries: 3,
          onAttempt: (attempt) =>
            recordAttempt({
              attempt,
              stage: "gapfill",
              qid: a.topic.qid,
              out: a.out,
              env: a.env,
              summary: a.summary,
            }),
        });
        for (const s2 of draft.value.answer) {
          if (totalWords >= 980) break;
          const mapped: number[] = [];
          for (const ci of s2.citations.slice(0, 3)) {
            const docid = draft.value.references[ci];
            if (typeof docid !== "string" || !a.readDocs.has(docid)) continue;
            let gi = references.indexOf(docid);
            if (gi === -1) {
              references.push(docid);
              gi = references.length - 1;
            }
            if (!mapped.includes(gi)) mapped.push(gi);
          }
          if (mapped.length === 0) continue;
          answer.push({ text: s2.text, citations: mapped });
          totalWords += s2.text.split(/\s+/).filter(Boolean).length;
          appended.push({ aspect, text: s2.text.slice(0, 80) });
        }
      } catch {
        /* aspect fill failed - skip */
      }
    }
    writeJson(tracePath, {
      ok: true,
      uncovered,
      appended_count: appended.length,
      appended,
    });
    if (appended.length === 0) return rag;
    const full: AgenticRagOutputObject = {
      metadata: rag.metadata,
      references,
      answer,
    };
    return normalizeRagOutputObjectReferences(full, {
      config: a.cfg,
      topic: a.topic,
      readDocids: new Set(a.readDocs.keys()),
    }).ragObject;
  } catch (e) {
    writeJson(tracePath, {
      ok: false,
      reason: redact(e instanceof Error ? e.message : String(e), a.env).slice(
        0,
        200,
      ),
    });
    return rag;
  }
}
async function setrSelect(
  a: {
    topic: TopicIdentity;
    llm: LlmClient;
    out: string;
    env: NodeJS.ProcessEnv;
    summary: any;
    checklist?: string[];
  },
  docs: ReadDoc[],
  k: number,
): Promise<ReadDoc[]> {
  try {
    const listing = docs
      .map(
        (d, i) =>
          `[${i}] ${d.docid}: ${d.text.replace(/\s+/g, " ").slice(0, 300)}`,
      )
      .join("\n");
    const aspects = (a.checklist ?? [])
      .map((c) => c.replace(/ \(vital\)$/, ""))
      .join("; ");
    const draft = await generateJsonWithRetry({
      client: a.llm,
      messages: [
        {
          role: "user",
          content: [
            "You select an evidence SET for a multi-aspect report.",
            `Information requirements (aspects to cover): ${aspects || "all aspects of the topic narrative"}`,
            "First think about which requirement each candidate can answer; then choose the set of at most " +
              k +
              " documents that TOGETHER cover the most requirements. Prefer complementary documents over redundant ones; a document that covers an otherwise-uncovered requirement beats a fifth document about an already-covered one.",
            'Return only strict JSON: {"selected":[indices in priority order]}',
            "",
            `Topic narrative: ${a.topic.narrative}`,
            "Candidate documents:",
            listing,
          ].join("\n"),
        },
      ],
      temperature: 0,
      maxTokens: 1024,
      validate: (
        v: unknown,
      ): LlmJsonValidationResult<{ selected: number[] }> => {
        if (
          !(
            typeof v === "object" &&
            v !== null &&
            Array.isArray((v as any).selected)
          )
        )
          return { ok: false, message: "selected shape" };
        return { ok: true, value: v as { selected: number[] } };
      },
      stage: "evidence_select",
      maxRequestRetries: 3,
      onAttempt: (attempt) =>
        recordAttempt({
          attempt,
          stage: "evidence_select",
          qid: a.topic.qid,
          out: a.out,
          env: a.env,
          summary: a.summary,
        }),
    });
    const seen = new Set<number>();
    const picked: ReadDoc[] = [];
    for (const idx of draft.value.selected) {
      if (
        typeof idx === "number" &&
        idx >= 0 &&
        idx < docs.length &&
        !seen.has(idx)
      ) {
        seen.add(idx);
        picked.push(docs[idx]);
        if (picked.length >= k) break;
      }
    }
    if (picked.length === 0) return docs.slice(0, k);
    for (const d of docs) {
      if (picked.length >= k) break;
      if (!picked.includes(d)) picked.push(d);
    }
    writeJson(
      join(a.out, "topics", `${a.topic.qid}.evidence_select_trace.json`),
      { candidates: docs.length, selected: picked.map((d) => d.docid) },
    );
    return picked;
  } catch {
    return docs.slice(0, k);
  }
}
function verifyDiffStats(
  before: AgenticRagOutputObject,
  after: AgenticRagOutputObject,
) {
  const beforeTexts = new Set(before.answer.map((s) => s.text.trim()));
  const kept = after.answer.filter((s) =>
    beforeTexts.has(s.text.trim()),
  ).length;
  return {
    sentences_before: before.answer.length,
    sentences_after: after.answer.length,
    kept_unchanged: kept,
    rewritten: Math.max(0, after.answer.length - kept),
    dropped: Math.max(0, before.answer.length - after.answer.length),
  };
}
async function searchOnce(
  o: IterativeOptions,
  query: string,
  depth: number,
  env: NodeJS.ProcessEnv,
) {
  const token = env[o.pyseriniTokenEnv]?.trim();
  const url = `${o.pyseriniBaseUrl.replace(/\/+$/, "")}/v1/${o.pyseriniIndex}/search?${new URLSearchParams({ query, hits: String(depth) })}`;
  for (let a = 1; a <= 6; a++) {
    const r = await fetch(url, {
      headers: token ? { authorization: `Bearer ${token}` } : {},
    });
    if (r.ok) {
      const v = (await r.json()) as any;
      return (v.candidates ?? [])
        .filter((c: any) => typeof c.docid === "string")
        .map((c: any) => ({ docid: c.docid, score: Number(c.score) || 0 }))
        .slice(0, depth);
    }
    if (![429, 500, 502, 503, 504].includes(r.status) || a === 6)
      throw new Error(`Pyserini HTTP ${r.status}`);
    await sleep(Math.max(1000, 600 * 2 ** a));
  }
  throw new Error("search failed");
}
async function search(
  o: IterativeOptions,
  query: string,
  depth: number,
  env: NodeJS.ProcessEnv,
) {
  const giant = (e: any) => {
    const s =
      String(e?.message ?? e) + " " + String((e as any)?.cause?.message ?? "");
    return /terminated|ECONNRESET|fetch failed|aborted|Cannot create a string/i.test(
      s,
    );
  };
  const queries = query.length > 200 ? [query, query.slice(0, 200)] : [query];
  let last: any;
  for (const q of queries) {
    let d = depth;
    while (true) {
      try {
        return await searchOnce(o, q, d, env);
      } catch (e) {
        if (!giant(e)) throw e;
        last = e;
      }
      if (d <= 313) break;
      d = Math.max(313, Math.floor(d / 2));
    }
  }
  throw last;
}
async function readDoc(
  o: IterativeOptions,
  docid: string,
  limit: number,
  env: NodeJS.ProcessEnv,
) {
  const token = env[o.pyseriniTokenEnv]?.trim();
  const url = `${o.pyseriniBaseUrl.replace(/\/+$/, "")}/v1/${o.pyseriniIndex}/doc/${encodeURIComponent(docid)}`;
  for (let a = 1; a <= 6; a++) {
    const r = await fetch(url, {
      headers: token ? { authorization: `Bearer ${token}` } : {},
    });
    if (r.status === 404) return { found: false, text: "", truncated: false };
    if (r.ok) {
      const v = (await r.json()) as any;
      const text = extractText(v.doc),
        lines = text.split(/\r?\n/);
      return {
        found: true,
        text: lines.slice(0, limit).join("\n"),
        truncated: lines.length > limit,
      };
    }
    if (![429, 500, 502, 503, 504].includes(r.status) || a === 6)
      throw new Error(`Pyserini doc HTTP ${r.status}`);
    await sleep(Math.max(1000, 600 * 2 ** a));
  }
  throw new Error("read failed");
}
function weightedRrf(rs: Hit[][], ws: number[], depth: number, k: number) {
  const m = new Map<string, { score: number; best: number }>();
  rs.forEach((hits, ri) =>
    hits.forEach((h, i) => {
      const p = m.get(h.docid) ?? { score: 0, best: Infinity };
      p.score += (ws[ri] ?? 1) / (k + i + 1);
      p.best = Math.min(p.best, i + 1);
      m.set(h.docid, p);
    }),
  );
  return [...m.entries()]
    .sort(
      (a, b) =>
        b[1].score - a[1].score ||
        a[1].best - b[1].best ||
        a[0].localeCompare(b[0]),
    )
    .slice(0, depth)
    .map(([docid, v], i) => ({ docid, rank: i + 1, score: v.score }));
}
function buildRetrievalTrace(
  topic: TopicIdentity,
  queries: string[],
  runs: Hit[][],
  ranking: ReturnType<typeof weightedRrf>,
) {
  return {
    topic_id: topic.qid,
    policy: POLICY,
    anchor_query: topic.narrative,
    followup_queries: queries.slice(1),
    query_count: queries.length,
    per_query_top10: runs.map((h, i) => ({
      query: queries[i],
      weight:
        i === 0 ? POLICY.bm25_anchor_weight : POLICY.followup_query_weight,
      top10: h.slice(0, 10),
    })),
    fused_top10: ranking.slice(0, 10),
    candidate_count: ranking.length,
  };
}
function writeTopicPartial(a: {
  out: string;
  topic: TopicIdentity;
  iterationTrace: any[];
  judgeTrace: any[];
  queries: string[];
  runs: Hit[][];
  ranking: ReturnType<typeof weightedRrf>;
  readDocs: Map<string, ReadDoc>;
  failedRead: string[];
}) {
  const base = join(a.out, "topics");
  writeJson(
    join(base, `${a.topic.qid}.iteration_trace.json`),
    a.iterationTrace,
  );
  writeJson(join(base, `${a.topic.qid}.judge_trace.json`), a.judgeTrace);
  writeJson(
    join(base, `${a.topic.qid}.retrieval-trace.json`),
    buildRetrievalTrace(a.topic, a.queries, a.runs, a.ranking),
  );
  writeJson(join(base, `${a.topic.qid}.final_read_docs_trace.json`), {
    topic_id: a.topic.qid,
    candidate_pool_size: a.ranking.length,
    documents_read_attempted: a.readDocs.size + a.failedRead.length,
    documents_read_successful: a.readDocs.size,
    read_docids: [...a.readDocs.keys()],
    failed_read_docids: a.failedRead,
    final_answer_cited_docids: [],
  });
}
function classifyJudgeStopReason(message: string) {
  if (/empty assistant message/i.test(message))
    return "judge_empty_assistant_message";
  if (/429|rate limit/i.test(message)) return "judge_rate_limit";
  if (/5\\d\\d|server/i.test(message)) return "judge_server_error";
  if (
    /fetch failed|network|timeout|timed out|ECONNRESET|ETIMEDOUT/i.test(message)
  )
    return "judge_transient_request_failed";
  return "judge_json_parse_failed";
}
function recordAttempt(a: {
  attempt: LlmAttemptTrace;
  stage: string;
  qid: string;
  out: string;
  env: NodeJS.ProcessEnv;
  summary: any;
}) {
  a.summary.llm_call_count++;
  if (!a.attempt.success) a.summary.llm_failed_call_count++;
  if (a.attempt.attempt > 1) a.summary.llm_retry_count++;
  appendJsonl(
    join(a.out, "llm_trace.jsonl"),
    {
      qid: a.qid,
      stage: a.stage,
      attempt: a.attempt.attempt,
      success: a.attempt.success,
      error_code: a.attempt.errorCode,
      provider: a.attempt.provider,
      model: a.attempt.model,
      latency_ms: a.attempt.latencyMs,
      output_chars: a.attempt.outputChars,
    },
    a.env,
  );
}
function extractText(doc: any) {
  return typeof doc === "string"
    ? doc
    : doc && typeof doc.text === "string"
      ? doc.text
      : JSON.stringify(doc ?? "");
}
function topicRun(qid: string, r: any[], runId: string) {
  return (
    r
      .map(
        (e, i) =>
          `${qid} Q0 ${e.docid} ${i + 1} ${e.score.toFixed(8)} ${runId}`,
      )
      .join("\n") + "\n"
  );
}
function render(r: Rankings, qids: string[], runId: string) {
  return (
    qids
      .flatMap((q) =>
        (r.get(q) ?? []).map(
          (e, i) =>
            `${q} Q0 ${e.docid} ${i + 1} ${e.score.toFixed(8)} ${runId}`,
        ),
      )
      .join("\n") + "\n"
  );
}
function assemble(out: string, topics: Topic[]): Rankings {
  const r: Rankings = new Map();
  for (const t of topics) {
    const p = join(out, "topics", `${t.qid}.runfile.trec`);
    if (existsSync(p))
      r.set(
        t.qid,
        readFileSync(p, "utf8")
          .split(/\r?\n/)
          .filter(Boolean)
          .map((line) => {
            const [, , docid, rank, score] = line.split(/\s+/);
            return { docid, rank: Number(rank), score: Number(score) };
          }),
      );
  }
  return r;
}
function qrelsPaths(dir: string) {
  return discoverQrelsFiles(dir);
}
function parseQrels(path: string, qids: string[]): Qrels {
  const w = new Set(qids),
    q: Qrels = new Map();
  for (const line of readFileSync(path, "utf8").split(/\r?\n/)) {
    if (!line.trim()) continue;
    const [qid, , docid, rel] = line.split(/\s+/);
    if (!w.has(qid)) continue;
    const m = q.get(qid) ?? new Map<string, number>();
    m.set(docid, Number(rel) || 0);
    q.set(qid, m);
  }
  return q;
}
function evalAll(paths: string[], qids: string[], rankings: Rankings) {
  const rows = paths.map((p) => {
    const q = parseQrels(p, qids),
      res = evaluateRankings(
        q,
        rankings,
        qids,
        { recallCutoffs: CUTS, ndcgCutoffs: NDCG, mrrCutoffs: [1000] },
        {
          recallRelevantThreshold: 2,
          binaryRelevantThreshold: 2,
          ndcgGainMode: "linear",
        },
      );
    const metrics = {
      ndcg_10: res.ndcgByCutoff.get(10) ?? 0,
      ndcg_20: res.ndcgByCutoff.get(20) ?? 0,
      ndcg_100: res.ndcgByCutoff.get(100) ?? 0,
      ndcg_1000: res.ndcgByCutoff.get(1000) ?? 0,
      recall_20: res.macroRecallByCutoff.get(20) ?? 0,
      recall_100: res.macroRecallByCutoff.get(100) ?? 0,
      recall_500: res.macroRecallByCutoff.get(500) ?? 0,
      recall_1000: res.macroRecallByCutoff.get(1000) ?? 0,
      map: res.map,
      mrr: res.mrrByCutoff.get(1000) ?? 0,
    };
    return {
      qrels_path: p,
      qrels_filename: basename(p),
      metrics,
      per_topic: {},
    };
  });
  const keys = Object.keys(rows[0].metrics);
  return {
    summary: {
      qrels: rows.map(({ per_topic: _p, ...x }) => x),
      arithmetic_mean_across_qrels: Object.fromEntries(
        keys.map((k) => [
          k,
          rows.reduce((s: any, r: any) => s + (r.metrics as any)[k], 0) /
            rows.length,
        ]),
      ),
    },
    perTopic: {},
  };
}
function writeJson(p: string, v: any) {
  mkdirSync(dirname(p), { recursive: true });
  writeFileSync(p, JSON.stringify(v, null, 2) + "\n");
}
function writeJsonl(p: string, rows: any[]) {
  writeFileSync(
    p,
    rows.map((r) => JSON.stringify(r)).join("\n") + (rows.length ? "\n" : ""),
  );
}
function appendJsonl(p: string, v: any, _env: NodeJS.ProcessEnv) {
  mkdirSync(dirname(p), { recursive: true });
  writeFileSync(p, JSON.stringify(v) + "\n", { flag: "a" });
}
function readIf(p: string): any {
  return existsSync(p) ? JSON.parse(readFileSync(p, "utf8")) : null;
}
function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}
function sleep(ms: number) {
  return new Promise((r) => setTimeout(r, ms));
}
function redact(s: string, env: NodeJS.ProcessEnv) {
  for (const [k, v] of Object.entries(env)) {
    if (v && /(?:KEY|TOKEN|SECRET|PASSWORD)$/i.test(k))
      s = s.split(v).join(`[redacted ${k}]`);
  }
  return s;
}
