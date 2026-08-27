import {
  existsSync,
  mkdirSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { basename, dirname, join, resolve } from "node:path";
import { parseTrecRag2026TopicsTsv } from "../retrieval/topics";
import { generateQuery2Doc } from "../retrieval/query2doc";
import { generateFacetQueries } from "../retrieval/facet_queries";
import { findInDocument, adaptiveBudget } from "../retrieval/find_in_document";
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
  buildComprehensiveAnswerPrompt,
  buildDenseAnswerGenerationPrompt,
  buildLedgerAnswerPrompt,
  buildVerifyRevisePrompt,
  buildAspectDecompositionPrompt,
  buildAspectAnswerPrompt,
  buildIntegrationPrompt,
  buildReflectionPrompt,
  buildGroundedRevisionPrompt,
} from "../shared-rag/prompts";
import {
  ExposureLedger,
  enforceAnswerPlan,
  LedgerViolation,
} from "./evidence_plan";
import { verifyCitations } from "./citation_verify";
import { denseScores } from "../retrieval/dense_rerank";
import { reattributeCitations } from "./citation_reattribute";
import { predictNuggets, findNuggetGaps } from "./nugget_loop";
import { enforceLedger, supportScore } from "./evidence_ledger";
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
import { AgenticRagValidationError } from "../shared-rag/validation";
import {
  evaluateRankings,
  type Qrels,
  type Rankings,
} from "../../evaluation/retrieval_metrics";
import { discoverQrelsFiles } from "../../evaluation/qrels_files";
import {
  rerankWithCrossEncoder,
  type RerankCandidate,
} from "../retrieval/cross_encoder_rerank";
import { readCache, writeCache } from "../retrieval/doc_cache";

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
  /** Explicit policy overrides applied by the public final entry point. */
  policy?: PolicyOverride;
  /** Optional role-specific clients; unspecified roles use the base client. */
  llmWriter?: RawLlmClientConfig;
  llmQuery?: RawLlmClientConfig;
};
/** Base, writer, and query-generation clients. */
type LlmSet = { base: LlmClient; writer: LlmClient; query: LlmClient };
const BASE_POLICY = {
  retrieval_policy:
    "iterative-agentic-anchor-bm25-weighted-rrf-w025-query2doc-fusion-peraspect-official-top5000-varK-breadthfirst",
  output_depth: 5000,
  bm25_anchor_weight: 1,
  followup_query_weight: 0.25,
  rrf_k: 60,
  rerank_depth: 100,
  q2d_enabled: true,
  q2d_weight: 1.0,
  q2d_query_repeat: 5,
  fusion_dense: true,
  fusion_bm25_weight: 1.0,
  fusion_ce_weight: 1.0,
  fusion_dense_weight: 1.0,
  fusion_rrf_k: 60,
  comprehensive_answer: true,
  citation_verify: true,
  support_threshold: 0.4,
  answer_doc_chars: 1600,
  answer_max_tokens: 4096,
  per_aspect_generation: true,
  aspect_docs: 4,
  aspect_search_depth: 100,
  aspect_max_tokens: 1500,
  aspect_rounds: 3,
  aspect_target_sentences: 5,
  reflection: true,
  reflection_max_gaps: 2,
  reattribute: true,
  reattribute_threshold: 0.3,
  reattribute_max_cites: 2,
  vark_threshold: 0.5,
  vark_min: 4,
  vark_max: 15,
  llm_revise: true,
  revise_snippet_chars: 1200,
  revise_max_tokens: 3000,
  breadth_first: true,
  max_answer_words: 1000,
  aspect_max: 12,
  aspect_reserve_frac: 0.75,
  aspect_ce_rerank: false,
  aspect_ce_pool: 20,
  integrate_answer: false,
  integrate_max_tokens: 24576,
  revise_never_drop: true,
  nugget_loop: true,
  nugget_max: 50,
  nugget_per_aspect: 6,
  nugget_max_gaps: 15,
  nugget_context_docs: 10,
  nugget_context_chars: 1200,
  ce_dead_threshold: 0.5,
  vark_relative_frac: 0.5,
  dense_writing: false,
  dense_evidence_docs: 12,
  dense_evidence_chars: 2500,
  dense_after_aspect: false,
  facet_queries: false,
  facet_span_autofix: false,
  facet_depth: 1000,
  splice_head_keep: 200,
  aspect_decompose_max_tokens: 1500,
  aspect_expected_facts: false,
  find_in_document: false,
  find_mode: "hybrid",
  find_max_passages: 6,
  find_window_chars: 1200,
  find_scan_limit: 200000,
  find_read_lines: 2000,
  adaptive_budget: false,
  find_impl: "lines",
  find_context_before: 5,
  find_context_after: 8,
  evidence_ledger: false,
  ledger_min_support: 0.45,
  fixed_topk_evidence: 0,
  verify_revise: false,
  verify_mode: "weaken",
  verify_batch: 20,
  verify_max_tokens: 8192,
  order_citations_by_strength: true,
  setr_select: false,
  setr_k: 12,
  tail_reselect: false,
  tail_reselect_head_keep: 200,
  tail_reselect_pool_depth: 5000,
  tail_reselect_fill: 800,
  tail_reselect_text_chars: 2000,
  enumerative_style: false,
  enumerative_words_min: 25,
  enumerative_words_max: 40,
  coverage_gate: false,
  coverage_gate_min_sentences: 3,
  coverage_gate_max_passes: 2,
  single_source: false,
  single_source_sentences: 2,
  atomic_sentences: false,
  atomic_max_words: 18,
  atomic_per_aspect_sentences: 6,
  aspect_writer: false,
  evidence_plan: false,
  evidence_plan_policy: "strict",
  evidence_plan_max_retries: 3,
  evidence_plan_max_tokens: 16384,
};
export type Policy = typeof BASE_POLICY;
export type PolicyOverride = Partial<Policy>;
export { BASE_POLICY };
let POLICY: Policy = BASE_POLICY;
const CUTS = [10, 20, 50, 100, 500, 1000],
  NDCG = [10, 20, 100, 1000];
function writeVariableKSubmission(
  out: string,
  topics: { qid: string }[],
  runId: string,
  tau: number,
  minK: number,
  maxK: number,
) {
  const lines: string[] = [];
  const counts: Record<string, number> = {};
  const modes: Record<string, string> = {};
  for (const t of topics) {
    const p = join(out, "topics", `${t.qid}.fusion_scores.json`);
    const fsj = existsSync(p) ? (readIf(p) as any[]) : null;
    const hasCe =
      Array.isArray(fsj) &&
      fsj.length > 0 &&
      fsj.some((x) => typeof x.ce_calibrated === "number");
    let ranked: { docid: string; score: number }[] = [];
    let confident = 0;
    let mode = "";
    if (hasCe) {
      confident = (fsj as any[]).filter(
        (x) => typeof x.ce_calibrated === "number" && x.ce_calibrated >= tau,
      ).length;
      ranked = (fsj as any[]).map((x) => ({
        docid: String(x.docid),
        score: Number(x.fused) || 0,
      }));
      mode = "ce_calibrated";
    } else {
      const rp = join(out, "topics", `${t.qid}.runfile.trec`);
      if (!existsSync(rp)) continue;
      ranked = readFileSync(rp, "utf8")
        .split(/\r?\n/)
        .filter(Boolean)
        .map((line) => {
          const [, , docid, , score] = line.split(/\s+/);
          return { docid, score: Number(score) || 0 };
        });
      if (ranked.length === 0) continue;
      const top = ranked[0].score;
      const cut = top * POLICY.vark_relative_frac;
      confident = ranked.filter((x) => x.score >= cut).length;
      mode = "relative_score";
    }
    let k = confident;
    if (k < minK) k = minK;
    if (k > maxK) k = maxK;
    if (k > ranked.length) k = ranked.length;
    counts[t.qid] = k;
    modes[t.qid] = mode;
    ranked
      .slice(0, k)
      .forEach((x, i) =>
        lines.push(
          `${t.qid} Q0 ${x.docid} ${i + 1} ${x.score.toFixed(8)} ${runId}`,
        ),
      );
  }
  writeFileSync(
    join(out, "submission_variable_k.trec"),
    lines.join("\n") + (lines.length ? "\n" : ""),
  );
  writeJson(join(out, "variable_k_counts.json"), {
    k_by_topic: counts,
    mode_by_topic: modes,
  });
}
async function rerankTopOfRanking(
  o: IterativeOptions,
  topic: Topic,
  ranking: { docid: string; rank: number; score: number }[],
  readDocs: Map<string, ReadDoc>,
  env: NodeJS.ProcessEnv,
  depth: number,
  out: string,
) {
  const head = ranking.slice(0, depth),
    tail = ranking.slice(depth);
  const candidates: RerankCandidate[] = [];
  const unfetched: { docid: string; score: number }[] = [];
  for (const entry of head) {
    const cached = readDocs.get(entry.docid);
    if (cached) {
      candidates.push({ docid: entry.docid, text: cached.text });
      continue;
    }
    try {
      const d = await readDoc(o, entry.docid, o.documentReadLimit, env);
      if (d.found) candidates.push({ docid: entry.docid, text: d.text });
      else unfetched.push(entry);
      await sleep(150);
    } catch {
      unfetched.push(entry);
    }
  }
  if (candidates.length === 0) {
    writeJson(join(out, "topics", `${topic.qid}.fusion_scores.json`), []);
    return ranking;
  }
  const bm25Rank = new Map<string, number>();
  candidates.forEach((c, i) => bm25Rank.set(c.docid, i));
  const reranked = await rerankWithCrossEncoder(topic.narrative, candidates);
  const ceRank = new Map<string, number>();
  reranked.forEach((r, i) => ceRank.set(r.docid, i));
  const ceCal = new Map<string, number>();
  reranked.forEach((r) => ceCal.set(r.docid, r.calibratedScore));
  let denseRank = new Map<string, number>();
  let denseCos = new Map<string, number>();
  if (POLICY.fusion_dense) {
    try {
      denseCos = await denseScores(topic.narrative, candidates, env);
      const sorted = [...candidates]
        .map((c) => c.docid)
        .sort((a, b) => (denseCos.get(b) ?? 0) - (denseCos.get(a) ?? 0));
      sorted.forEach((d, i) => denseRank.set(d, i));
    } catch {
      denseRank = new Map();
    }
  }
  const K = POLICY.fusion_rrf_k,
    useDense = denseRank.size > 0;
  const maxCe = ceCal.size > 0 ? Math.max(...ceCal.values()) : Infinity;
  const ceDead =
    POLICY.ce_dead_threshold > 0 &&
    ceCal.size > 0 &&
    maxCe < POLICY.ce_dead_threshold;
  const fused = candidates
    .map((c) => {
      const rb = bm25Rank.get(c.docid) ?? candidates.length,
        rc = ceRank.get(c.docid) ?? candidates.length,
        rd = denseRank.get(c.docid) ?? candidates.length;
      if (ceDead) return { docid: c.docid, fused: 1 / (K + rb + 1) };
      let s =
        POLICY.fusion_bm25_weight / (K + rb + 1) +
        POLICY.fusion_ce_weight / (K + rc + 1);
      if (useDense) s += POLICY.fusion_dense_weight / (K + rd + 1);
      return { docid: c.docid, fused: s };
    })
    .sort((a, b) => b.fused - a.fused);
  if (ceDead)
    console.error(
      `  ${topic.qid}: CE dead (max ce_calibrated ${maxCe.toFixed(3)} < ${POLICY.ce_dead_threshold}) -> BM25 fallback`,
    );
  writeJson(
    join(out, "topics", `${topic.qid}.fusion_scores.json`),
    fused.map((f) => ({
      docid: f.docid,
      fused: f.fused,
      bm25_rank: bm25Rank.get(f.docid),
      ce_rank: ceRank.get(f.docid),
      dense_rank: useDense ? denseRank.get(f.docid) : null,
      ce_calibrated: ceCal.get(f.docid) ?? null,
      dense_cosine: useDense ? (denseCos.get(f.docid) ?? null) : null,
      ce_dead: ceDead,
    })),
  );
  const newHead = [
    ...fused.map((f) => ({ docid: f.docid, score: f.fused })),
    ...unfetched,
  ];
  return [...newHead, ...tail].map((e, i) => ({
    docid: e.docid,
    rank: i + 1,
    score: e.score,
  }));
}
export async function runFinalRetrievalPipeline(o: IterativeOptions) {
  POLICY = { ...BASE_POLICY, ...(o.policy ?? {}) };
  if (
    POLICY.enumerative_style &&
    !POLICY.evidence_ledger &&
    !POLICY.evidence_plan
  )
    throw new Error(
      "enumerative_style requires evidence_ledger or evidence_plan",
    );
  if (POLICY.evidence_plan && POLICY.evidence_ledger)
    throw new Error("evidence_plan and evidence_ledger are mutually exclusive");
  if (POLICY.atomic_sentences && POLICY.enumerative_style)
    throw new Error(
      "atomic_sentences and enumerative_style are mutually exclusive",
    );
  if (POLICY.dense_after_aspect && !POLICY.dense_writing)
    throw new Error("dense_after_aspect requires dense_writing");
  if (POLICY.dense_after_aspect && POLICY.per_aspect_generation)
    throw new Error(
      "dense_after_aspect and per_aspect_generation are mutually exclusive",
    );
  const env = o.env ?? process.env,
    out = resolve(o.outputDir);
  if (o.force) rmSync(out, { recursive: true, force: true });
  mkdirSync(join(out, "topics"), { recursive: true });
  const llmCfg = normalizeLlmClientConfig(o.llm),
    llm = createLlmClient(llmCfg, env);
  const writerCfg = o.llmWriter
      ? normalizeLlmClientConfig(o.llmWriter)
      : llmCfg,
    queryCfg = o.llmQuery ? normalizeLlmClientConfig(o.llmQuery) : llmCfg;
  const llms: LlmSet = {
    base: llm,
    writer: o.llmWriter ? createLlmClient(writerCfg, env) : llm,
    query: o.llmQuery ? createLlmClient(queryCfg, env) : llm,
  };
  const cfg: RagRunConfig = {
    runId: o.runId,
    teamId: o.teamId,
    mode: "automatic",
    promptVersion: RAG_PROMPT_PROFILE,
    runDesc: POLICY.retrieval_policy,
  };
  const topics = parseTrecRag2026TopicsTsv(
    readFileSync(resolve(o.topicsPath), "utf8"),
  )
    .map((t) => ({ qid: t.topicId, title: "", narrative: t.narrative }))
    .slice(0, o.limitTopics ?? Infinity);
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
    llm_writer: safeLlmConfigForArtifacts(writerCfg),
    llm_query: safeLlmConfigForArtifacts(queryCfg),
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
    aspect_decomposition_failures: 0,
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
        llms,
        cfg,
        out,
        env,
        summary,
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
      if (
        process.env.POOL_QUALITY_GATE === "1" &&
        r.ranking.length < POLICY.output_depth
      )
        throw new Error(
          `TRUNCATED_POOL ${r.ranking.length}/${POLICY.output_depth}`,
        );
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
      const issues =
        e instanceof AgenticRagValidationError ? e.issues : undefined;
      writeJson(sp, {
        topic_id: t.qid,
        status: "failed",
        error: redact(e instanceof Error ? e.message : String(e), env),
        ...(issues ? { issues } : {}),
      });
      console.error(
        `${idx + 1}/${topics.length} ${t.qid} FAILED${issues ? " " + issues.map((i) => i.code).join(",") : ""}`,
      );
    }
    await sleep(500);
  }
  const rankings = assemble(out, topics),
    runfile = render(
      rankings,
      topics.map((t) => t.qid),
      o.runId,
    );
  writeFileSync(join(out, "candidate_pool_top5000.trec"), runfile);
  writeFileSync(join(out, "retrieval.internal.trec-run.tsv"), runfile);
  writeVariableKSubmission(
    out,
    topics,
    o.runId,
    POLICY.vark_threshold,
    POLICY.vark_min,
    POLICY.vark_max,
  );
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
    `# Final Retrieval pipeline\n\nNo RAGDoll. No qrels query selection. No source_run_dir.\n`,
  );
  return { outputDir: out, validation };
}
async function spliceFacetPool(
  a: {
    topic: Topic;
    o: IterativeOptions;
    llms: LlmSet;
    out: string;
    env: NodeJS.ProcessEnv;
  },
  ranking: { docid: string; rank: number; score: number }[],
  poolText?: Map<string, string>,
) {
  const keep = Math.min(POLICY.splice_head_keep, ranking.length);
  try {
    const fq = await generateFacetQueries(a.llms.query, a.topic.narrative, {
      spanAutofix:
        process.env.FACET_SPAN_AUTOFIX === "1" || POLICY.facet_span_autofix,
    });
    writeJson(join(a.out, "topics", `${a.topic.qid}.facet_queries.json`), {
      topic_id: a.topic.qid,
      ok: fq.ok,
      rewrites: fq.rewrites,
      facets: fq.facets,
      cores: fq.cores,
      background_hints: fq.backgroundHints,
      repairs: fq.repairs,
      failure_reasons: fq.failureReasons,
    });
    if (!fq.ok) return ranking;
    const queries = [...fq.rewrites, ...fq.facets];
    const runs: Hit[][] = [];
    for (const q of queries) {
      try {
        runs.push(await search(a.o, q, POLICY.facet_depth, a.env, poolText));
      } catch {}
      await sleep(250);
    }
    if (runs.length === 0) return ranking;
    const facetPool = weightedRrf(
      runs,
      runs.map(() => 1),
      POLICY.output_depth,
      POLICY.rrf_k,
    );
    const head = ranking.slice(0, keep),
      headSet = new Set(head.map((e) => e.docid));
    const tail = facetPool
      .filter((e) => !headSet.has(e.docid))
      .slice(0, Math.max(0, POLICY.output_depth - keep));
    const tailSet = new Set(tail.map((e) => e.docid));
    const backfill = ranking
      .slice(keep)
      .filter((e) => !headSet.has(e.docid) && !tailSet.has(e.docid))
      .slice(0, Math.max(0, POLICY.output_depth - keep - tail.length));
    const spliced = [
      ...head,
      ...tail.map((e) => ({ docid: e.docid, rank: 0, score: e.score })),
      ...backfill,
    ].map((e, i) => ({ docid: e.docid, rank: i + 1, score: e.score }));
    writeJson(join(a.out, "topics", `${a.topic.qid}.splice.json`), {
      topic_id: a.topic.qid,
      head_keep: keep,
      facet_queries: queries.length,
      facet_pool_size: facetPool.length,
      before_size: ranking.length,
      after_size: spliced.length,
      new_in_tail: tail.length,
    });
    return spliced;
  } catch {
    return ranking;
  }
}
function sentenceStyleStats(d: { answer: { text: string }[] }) {
  const w = d.answer.map(
    (s) =>
      String(s.text ?? "")
        .trim()
        .split(/\s+/)
        .filter(Boolean).length,
  );
  if (w.length === 0) return { sentences: 0 };
  const sorted = [...w].sort((x, y) => x - y),
    total = w.reduce((s, x) => s + x, 0);
  const lo = POLICY.enumerative_words_min,
    hi = POLICY.enumerative_words_max,
    inRange = w.filter((x) => x >= lo && x <= hi).length;
  return {
    sentences: w.length,
    words_total: total,
    words_mean: Number((total / w.length).toFixed(1)),
    words_median: sorted[Math.floor(sorted.length / 2)],
    words_min: sorted[0],
    words_max: sorted[sorted.length - 1],
    in_target_range: inRange,
    in_target_range_pct: Number(((100 * inRange) / w.length).toFixed(1)),
    target_range: [lo, hi],
    enumerative: POLICY.enumerative_style,
  };
}
async function reselectTail(
  a: { topic: Topic; out: string },
  ranking: { docid: string; rank: number; score: number }[],
  poolText?: Map<string, string>,
) {
  const keep = Math.min(POLICY.tail_reselect_head_keep, ranking.length);
  if (!poolText || poolText.size === 0 || ranking.length <= keep)
    return ranking;
  try {
    const head = ranking.slice(0, keep),
      tail = ranking.slice(keep, POLICY.tail_reselect_pool_depth),
      rest = ranking.slice(POLICY.tail_reselect_pool_depth);
    const cands: RerankCandidate[] = [];
    let noText = 0;
    for (const e of tail) {
      const t = poolText.get(e.docid);
      if (t) cands.push({ docid: e.docid, text: t });
      else noText++;
    }
    if (cands.length === 0) return ranking;
    const scored = await rerankWithCrossEncoder(a.topic.narrative, cands);
    const byDoc = new Map(tail.map((e) => [e.docid, e]));
    const picked = scored
      .slice(0, POLICY.tail_reselect_fill)
      .map((s) => byDoc.get(s.docid))
      .filter((e): e is (typeof tail)[number] => Boolean(e));
    const pickedSet = new Set(picked.map((e) => e.docid));
    const merged = [
      ...head,
      ...picked,
      ...tail.filter((e) => !pickedSet.has(e.docid)),
      ...rest,
    ];
    const base = head.length ? head[head.length - 1].score : 1;
    const out = merged.map((e, i) => ({
      docid: e.docid,
      rank: i + 1,
      score: i < keep ? e.score : base - (i - keep + 1) * 1e-6,
    }));
    writeJson(join(a.out, "topics", `${a.topic.qid}.tail_reselect.json`), {
      topic_id: a.topic.qid,
      head_keep: keep,
      tail_size: tail.length,
      ce_scored: cands.length,
      no_text: noText,
      filled: picked.length,
      pool_text_size: poolText.size,
      head_unchanged: out
        .slice(0, keep)
        .every((e, i) => e.docid === head[i].docid),
      after_size: out.length,
    });
    return out;
  } catch {
    return ranking;
  }
}
async function processTopic(a: {
  topic: Topic;
  o: IterativeOptions;
  llm: LlmClient;
  llms: LlmSet;
  cfg: RagRunConfig;
  out: string;
  env: NodeJS.ProcessEnv;
  summary: any;
}) {
  const queries = [a.topic.narrative],
    runs: Hit[][] = [],
    weights: number[] = [];
  const poolText = POLICY.tail_reselect ? new Map<string, string>() : undefined;
  const anchor = await search(a.o, a.topic.narrative, 5000, a.env, poolText);
  runs.push(anchor);
  weights.push(POLICY.bm25_anchor_weight);
  const readDocs = new Map<string, ReadDoc>(),
    failedRead: string[] = [];
  const iterationTrace: any[] = [],
    judgeTrace: any[] = [];
  if (POLICY.q2d_enabled) {
    const q2d = await generateQuery2Doc(a.llms.query, a.topic.narrative, {
      queryRepeat: POLICY.q2d_query_repeat,
      maxPseudoDocWords: 180,
      maxTokens: 400,
    });
    writeJson(join(a.out, "topics", `${a.topic.qid}.query2doc.json`), {
      topic_id: a.topic.qid,
      ok: q2d.ok,
      query_repeat: POLICY.q2d_query_repeat,
      q2d_weight: POLICY.q2d_weight,
      pseudo_doc: q2d.pseudoDoc,
      expanded_query: q2d.expandedQuery,
    });
    if (q2d.ok) {
      const q2dRun = await search(
        a.o,
        q2d.expandedQuery,
        5000,
        a.env,
        poolText,
      );
      runs.push(q2dRun);
      weights.push(POLICY.q2d_weight);
      queries.push(`[query2doc] ${q2d.expandedQuery.slice(0, 200)}`);
      await sleep(250);
    }
  }
  let ranking = weightedRrf(runs, weights, POLICY.output_depth, POLICY.rrf_k),
    stopReason = "max_iterations",
    iterations = 0;
  let earlyAspects: string[] = [];
  let budget = {
    maxDocs: a.o.maxDocumentsRead,
    maxIterations: a.o.maxIterations,
  };
  if (POLICY.find_in_document || POLICY.adaptive_budget) {
    earlyAspects = await decomposeAspects({
      topic: a.topic,
      llm: a.llms.base,
      out: a.out,
      env: a.env,
      summary: a.summary,
    });
    if (POLICY.adaptive_budget && earlyAspects.length > 0)
      budget = adaptiveBudget(earlyAspects.length);
  }
  const findQueries = POLICY.find_in_document
    ? [a.topic.narrative, ...earlyAspects]
    : undefined;
  await readNew({
    o: a.o,
    hits: ranking,
    readDocs,
    failedRead,
    count: a.o.initialDocs,
    query: a.topic.narrative,
    env: a.env,
    maxDocs: budget.maxDocs,
    findQueries,
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
  for (let it = 0; it < budget.maxIterations; it++) {
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
    if (readDocs.size >= budget.maxDocs) {
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
    if (it === budget.maxIterations - 1) {
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
      const h = await search(a.o, q, 5000, a.env, poolText);
      runs.push(h);
      weights.push(POLICY.followup_query_weight);
      await sleep(250);
    }
    ranking = weightedRrf(runs, weights, POLICY.output_depth, POLICY.rrf_k);
    await readNew({
      o: a.o,
      hits: ranking,
      readDocs,
      failedRead,
      count: a.o.docsPerIteration,
      query: "weighted_rrf_fused",
      env: a.env,
      maxDocs: budget.maxDocs,
      findQueries,
    });
    writePartial();
    if (readDocs.size >= budget.maxDocs && it + 1 >= budget.maxIterations)
      stopReason = "max_iterations";
  }
  if (stopReason === "max_iterations" && readDocs.size >= budget.maxDocs)
    stopReason = "max_documents_read";
  if (POLICY.rerank_depth > 0)
    ranking = await rerankTopOfRanking(
      a.o,
      a.topic,
      ranking,
      readDocs,
      a.env,
      POLICY.rerank_depth,
      a.out,
    );
  if (POLICY.facet_queries) {
    ranking = await spliceFacetPool(a, ranking, poolText);
  }
  if (POLICY.tail_reselect) {
    ranking = await reselectTail(a, ranking, poolText);
  }
  if (POLICY.fixed_topk_evidence > 0) {
    const want = ranking.slice(0, POLICY.fixed_topk_evidence);
    readDocs.clear();
    await readNew({
      o: a.o,
      hits: want.map((e) => ({ docid: e.docid, score: e.score })),
      readDocs,
      failedRead,
      count: POLICY.fixed_topk_evidence,
      query: "fixed_topk_evidence",
      env: a.env,
      maxDocs: POLICY.fixed_topk_evidence,
      findQueries,
    });
  }
  const ragObject = await answer({
    topic: a.topic,
    o: a.o,
    cfg: a.cfg,
    llm: a.llm,
    llms: a.llms,
    readDocs,
    out: a.out,
    env: a.env,
    summary: a.summary,
    presetAspects: earlyAspects.length > 0 ? earlyAspects : undefined,
    budget,
  });
  const validation = validateRagOutputObjectStrict(ragObject, {
    config: a.cfg,
    topic: a.topic,
    readDocids: new Set(readDocs.keys()),
  });
  if (!validation.ok)
    throw new Error(
      `RAG validation failed: ${validation.issues.map((i) => i.code).join(",")}`,
    );
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
}) {
  const prompt = [
    "Return ONLY one JSON object. No markdown. No explanation.",
    'Required schema: {"enough":boolean,"queries":string[]}',
    'If the evidence already supports the topic, return {"enough":true,"queries":[]}.',
    'If evidence is insufficient, return {"enough":false,"queries":[...]} with 1-3 BM25 keyword queries.',
    "Query rules: 4-12 English tokens; keyword phrase, not sentence/question; one aspect only; no duplicates.",
    "Forbidden words: obtain, find, source, sources, detail, detailed, comprehensive, concrete, examples, overview, history, impact, provide, explain.",
    `Previous queries: ${JSON.stringify(a.queries)}`,
    `Topic: ${a.topic.narrative}`,
    "Evidence:",
    [...a.readDocs.values()]
      .map((d, i) => `[${i}] ${d.docid}\n${d.text.slice(0, 1200)}`)
      .join("\n\n"),
  ].join("\n");
  let lastError = "LLM_JSON_PARSE_FAILED";
  for (let attempt = 1; attempt <= 5; attempt++) {
    const started = Date.now();
    try {
      const result = await a.llm.generate({
        messages: [{ role: "user", content: prompt }],
        temperature: 0,
        maxTokens: 2000,
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
function stripInlineCitations(text: string): string {
  return text
    .replace(/\s*\[\d+(?:\s*,\s*\d+)*\]/g, "")
    .replace(/\s+([.,;:])/g, "$1")
    .replace(/\s{2,}/g, " ")
    .trim();
}
async function answerOneAspect(
  a: {
    topic: TopicIdentity;
    o: IterativeOptions;
    llm: LlmClient;
    out: string;
    env: NodeJS.ProcessEnv;
    summary: any;
  },
  aspect: string,
  docidText: Map<string, string>,
  shared: Map<string, ReadDoc>,
): Promise<{
  sentences: { text: string; docids: string[] }[];
  docsUsed: number;
}> {
  let hits: Hit[] = [];
  try {
    hits = await search(a.o, aspect, POLICY.aspect_search_depth, a.env);
  } catch {
    hits = [];
  }
  if (POLICY.aspect_ce_rerank && hits.length > 1) {
    const pool = hits.slice(0, POLICY.aspect_ce_pool);
    const cands: { docid: string; text: string }[] = [];
    for (const h of pool) {
      let t: string | undefined = docidText.get(h.docid);
      if (t === undefined) {
        try {
          const d = await readDoc(a.o, h.docid, a.o.documentReadLimit, a.env);
          await sleep(120);
          if (!d.found) continue;
          const ft: string = d.text;
          t = ft;
          docidText.set(h.docid, ft);
          if (!shared.has(h.docid))
            shared.set(h.docid, {
              docid: h.docid,
              text: ft,
              truncated: d.truncated,
              rankHint: 0,
              query: `per_aspect_ce:${aspect.slice(0, 40)}`,
            });
        } catch {
          continue;
        }
      }
      if (typeof t === "string" && t.length > 0)
        cands.push({ docid: h.docid, text: t });
    }
    if (cands.length > 1) {
      try {
        const ranked = await rerankWithCrossEncoder(aspect, cands, {
          maxCharsPerDoc: 2000,
        });
        const order = new Map(ranked.map((r, i) => [r.docid, i]));
        hits = [...hits].sort(
          (x, y) => (order.get(x.docid) ?? 1e9) - (order.get(y.docid) ?? 1e9),
        );
      } catch {
        /* Keep BM25 order if cross-encoder scoring is unavailable. */
      }
    }
  }
  const out: { text: string; docids: string[] }[] = [];
  let cursor = 0,
    docsUsed = 0;
  for (let round = 0; round < POLICY.aspect_rounds; round++) {
    const docs: { docid: string; text: string }[] = [];
    while (docs.length < POLICY.aspect_docs && cursor < hits.length) {
      const h = hits[cursor++];
      let text: string | undefined = docidText.get(h.docid);
      if (text === undefined) {
        try {
          const d = await readDoc(a.o, h.docid, a.o.documentReadLimit, a.env);
          await sleep(200);
          if (!d.found) continue;
          const ft = d.text;
          text = ft;
          docidText.set(h.docid, ft);
          if (!shared.has(h.docid))
            shared.set(h.docid, {
              docid: h.docid,
              text: ft,
              truncated: d.truncated,
              rankHint: 0,
              query: `per_aspect:${aspect.slice(0, 40)}`,
            });
        } catch {
          continue;
        }
      }
      if (text === undefined) continue;
      docs.push({
        docid: h.docid,
        text: text.slice(0, POLICY.answer_doc_chars),
      });
    }
    if (docs.length === 0) break;
    docsUsed += docs.length;
    if (POLICY.single_source) {
      let addedThisRound = 0;
      for (const d of docs) {
        try {
          const draft = await generateJsonWithRetry({
            client: a.llm,
            messages: [
              {
                role: "user",
                content: buildAspectAnswerPrompt({
                  topic: a.topic,
                  aspect,
                  documents: [d],
                  alreadyWritten: out.map((s) => s.text),
                  ...(expectedFacts.get(aspect)
                    ? { expectedFacts: expectedFacts.get(aspect) }
                    : {}),
                  ...(POLICY.atomic_sentences
                    ? {
                        atomic: {
                          maxWords: POLICY.atomic_max_words,
                          sentences: POLICY.single_source_sentences,
                        },
                      }
                    : {}),
                }),
              },
            ],
            temperature: 0,
            maxTokens: POLICY.aspect_max_tokens,
            validate: validateAnswer,
            stage: "aspect_single_source",
            maxRequestRetries: 2,
            onAttempt: (attempt) =>
              recordAttempt({
                attempt,
                stage: "aspect_single_source",
                qid: a.topic.qid,
                out: a.out,
                env: a.env,
                summary: a.summary,
              }),
          });
          const arr: any[] = Array.isArray((draft.value as any).answer)
            ? (draft.value as any).answer
            : [];
          for (const s of arr) {
            if (typeof s?.text !== "string" || !s.text.trim()) continue;
            out.push({ text: stripInlineCitations(s.text), docids: [d.docid] });
            addedThisRound++;
            if (out.length >= POLICY.aspect_target_sentences) break;
          }
        } catch {
          continue;
        }
        if (out.length >= POLICY.aspect_target_sentences) break;
      }
      if (addedThisRound === 0) break;
      if (out.length >= POLICY.aspect_target_sentences) break;
      continue;
    }
    try {
      const draft = await generateJsonWithRetry({
        client: a.llm,
        messages: [
          {
            role: "user",
            content: buildAspectAnswerPrompt({
              topic: a.topic,
              aspect,
              documents: docs,
              alreadyWritten: out.map((s) => s.text),
              ...(expectedFacts.get(aspect)
                ? { expectedFacts: expectedFacts.get(aspect) }
                : {}),
              ...(POLICY.atomic_sentences
                ? {
                    atomic: {
                      maxWords: POLICY.atomic_max_words,
                      sentences: POLICY.atomic_per_aspect_sentences,
                    },
                  }
                : {}),
            }),
          },
        ],
        temperature: 0,
        maxTokens: POLICY.aspect_max_tokens,
        validate: validateAnswer,
        stage: "answer_generation",
        maxRequestRetries: 3,
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
      const val: any = draft.value;
      const subRefs: string[] = Array.isArray(val.references)
        ? val.references.map(String)
        : [];
      const arr: any[] = Array.isArray(val.answer) ? val.answer : [];
      let addedThisRound = 0;
      for (const s of arr) {
        if (typeof s?.text !== "string" || !s.text.trim()) continue;
        const idxs: number[] = Array.isArray(s.citations) ? s.citations : [];
        const docids = [
          ...new Set(
            idxs
              .map((i) => subRefs[i])
              .filter(
                (d): d is string =>
                  typeof d === "string" && docs.some((x) => x.docid === d),
              ),
          ),
        ];
        if (docids.length === 0) continue;
        out.push({ text: stripInlineCitations(s.text), docids });
        addedThisRound++;
      }
      if (addedThisRound === 0) break;
    } catch {
      break;
    }
    if (out.length >= POLICY.aspect_target_sentences) break;
  }
  return { sentences: out, docsUsed };
}
async function generatePerAspectAnswer(
  a: {
    topic: TopicIdentity;
    o: IterativeOptions;
    llm: LlmClient;
    out: string;
    env: NodeJS.ProcessEnv;
    summary: any;
  },
  aspects: string[],
  shared: Map<string, ReadDoc>,
): Promise<{
  references: string[];
  answer: { text: string; citations: number[] }[];
  perAspect: any[];
  docText: Map<string, string>;
  nuggets: any;
}> {
  const docidText = new Map<string, string>(
    [...shared.values()].map((d) => [d.docid, d.text]),
  );
  const sentences: { text: string; docids: string[] }[] = [];
  const perAspect: any[] = [];
  const groups: { text: string; docids: string[] }[][] = [];
  let nuggetTrace: any = null;
  for (const aspect of aspects) {
    const r = await answerOneAspect(a, aspect, docidText, shared);
    sentences.push(...r.sentences);
    groups.push(r.sentences);
    perAspect.push({ aspect, docs: r.docsUsed, sentences: r.sentences.length });
  }
  if (POLICY.coverage_gate && aspects.length > 0) {
    const weak = () =>
      aspects
        .map((asp, i) => ({ asp, i, n: groups[i]?.length ?? 0 }))
        .filter((x) => x.n < POLICY.coverage_gate_min_sentences);
    for (let pass = 0; pass < POLICY.coverage_gate_max_passes; pass++) {
      const todo = weak();
      if (todo.length === 0) break;
      for (const { asp, i } of todo) {
        const r = await answerOneAspect(a, asp, docidText, shared);
        const seen = new Set((groups[i] ?? []).map((s) => s.text));
        const fresh = r.sentences.filter((s) => !seen.has(s.text));
        if (fresh.length === 0) continue;
        sentences.push(...fresh);
        groups[i] = [...(groups[i] ?? []), ...fresh];
        const t = perAspect[i];
        if (t) {
          t.sentences = groups[i].length;
          t.gate_passes = (t.gate_passes ?? 0) + 1;
        }
      }
    }
    for (const [i, t] of perAspect.entries())
      t.coverage_status =
        (groups[i]?.length ?? 0) >= POLICY.coverage_gate_min_sentences
          ? "covered"
          : "weak";
  }
  if (POLICY.reflection && sentences.length > 0) {
    try {
      const answerText = sentences.map((s) => s.text).join(" ");
      const rr = await a.llm.generate({
        messages: [
          { role: "user", content: buildReflectionPrompt(a.topic, answerText) },
        ],
        temperature: 0,
        maxTokens: 400,
        responseFormat: "json_object",
      });
      const j = extractObjectCandidate(rr.text);
      const gaps: string[] = j
        ? (JSON.parse(j)?.gaps || [])
            .map((x: any) => String(x).trim())
            .filter((x: string) => x.length > 0)
            .slice(0, POLICY.reflection_max_gaps)
        : [];
      for (const gap of gaps) {
        const r = await answerOneAspect(a, gap, docidText, shared);
        sentences.push(...r.sentences);
        groups.push(r.sentences);
        perAspect.push({
          aspect: `[reflect] ${gap}`,
          docs: r.docsUsed,
          sentences: r.sentences.length,
        });
      }
    } catch {}
  }
  if (POLICY.nugget_loop && sentences.length > 0) {
    try {
      const evidence = [...shared.values()]
        .slice(0, POLICY.nugget_context_docs)
        .map((d) => ({ docid: d.docid, text: d.text }));
      const predicted = await predictNuggets(
        a.llm,
        a.topic,
        evidence,
        aspects,
        {
          maxNuggets: POLICY.nugget_max,
          perAspect: POLICY.nugget_per_aspect,
          contextDocs: POLICY.nugget_context_docs,
          contextChars: POLICY.nugget_context_chars,
        },
      );
      const { gaps, assignments } = await findNuggetGaps(
        a.llm,
        a.topic,
        sentences.map((s) => s.text).join(" "),
        predicted,
        POLICY.nugget_max_gaps,
      );
      nuggetTrace = {
        predicted: predicted.length,
        gap_count: gaps.length,
        gaps,
        assignments,
      };
      for (const gap of gaps) {
        const r = await answerOneAspect(a, gap, docidText, shared);
        if (r.sentences.length === 0) continue;
        sentences.push(...r.sentences);
        groups.push(r.sentences);
        perAspect.push({
          aspect: `[nugget] ${gap}`,
          docs: r.docsUsed,
          sentences: r.sentences.length,
        });
      }
    } catch {}
  }
  const wc = (t: string) => t.split(/\s+/).filter(Boolean).length;
  const capped: { text: string; docids: string[] }[] = [];
  let words = 0;
  if (POLICY.breadth_first && groups.length > 0) {
    const depth = Math.max(...groups.map((g) => g.length));
    for (let round = 0; round < depth; round++)
      for (const g of groups) {
        const s = g[round];
        if (!s) continue;
        const w = wc(s.text);
        if (words + w > POLICY.max_answer_words && capped.length > 0) continue;
        capped.push(s);
        words += w;
      }
  } else {
    for (const s of sentences) {
      const w = wc(s.text);
      if (words + w > POLICY.max_answer_words && capped.length > 0) break;
      capped.push(s);
      words += w;
    }
  }
  let final = capped;
  if (POLICY.integrate_answer && groups.length > 0 && capped.length > 0) {
    try {
      const keep = new Set(capped.map((s) => s.text));
      const blocks = groups
        .map((g, i) => ({
          aspect: perAspect[i]?.aspect ?? `aspect ${i}`,
          sents: g.filter((s) => keep.has(s.text)),
        }))
        .filter((b) => b.sents.length > 0);
      if (blocks.length > 1) {
        const r = await a.llm.generate({
          messages: [
            {
              role: "user",
              content: buildIntegrationPrompt({
                topic: a.topic,
                groups: blocks.map((b) => ({
                  aspect: b.aspect,
                  sentences: b.sents.map((s) => s.text),
                })),
                minWords: Math.round(POLICY.max_answer_words * 0.88),
                maxWords: POLICY.max_answer_words,
              }),
            },
          ],
          temperature: 0,
          maxTokens: POLICY.integrate_max_tokens,
          responseFormat: "json_object",
        });
        recordAttempt({
          attempt: {
            attempt: 1,
            provider: a.llm.provider,
            model: a.llm.model,
            latencyMs: r.latencyMs,
            success: true,
            outputChars: r.text.length,
          },
          stage: "integrate",
          qid: a.topic.qid,
          out: a.out,
          env: a.env,
          summary: a.summary,
        });
        const j = extractObjectCandidate(r.text);
        const arr: any[] = j ? (JSON.parse(j)?.answer ?? []) : [];
        const merged: { text: string; docids: string[] }[] = [];
        for (const s of arr) {
          if (typeof s?.text !== "string" || !s.text.trim()) continue;
          const src: number[] = Array.isArray(s.sources)
            ? s.sources
                .map((x: any) => Number(x))
                .filter(
                  (x: number) =>
                    Number.isInteger(x) && x >= 0 && x < blocks.length,
                )
            : [];
          const pool = (src.length ? src : blocks.map((_, i) => i)).flatMap(
            (i) => blocks[i].sents.flatMap((x) => x.docids),
          );
          const docids = [...new Set(pool)];
          if (docids.length === 0) continue;
          merged.push({ text: s.text.trim(), docids });
        }
        const before = capped.reduce((n, s) => n + wc(s.text), 0),
          after = merged.reduce((n, s) => n + wc(s.text), 0);
        if (
          merged.length >= Math.min(8, capped.length * 0.5) &&
          after >= before * 0.75
        )
          final = merged;
        else
          console.error(
            `  !! ${a.topic.qid}: integration reduced coverage (${capped.length} sentences/${before} words -> ${merged.length} sentences/${after} words); keeping the unintegrated draft`,
          );
      }
    } catch (e) {
      console.error(
        `  !! ${a.topic.qid}: integration failed (${e instanceof Error ? e.message.slice(0, 80) : String(e)}); keeping the unintegrated draft`,
      );
    }
  }
  const refs: string[] = [];
  const idxOf = new Map<string, number>();
  for (const s of final)
    for (const d of s.docids)
      if (!idxOf.has(d)) {
        idxOf.set(d, refs.length);
        refs.push(d);
      }
  const answer = final.map((s) => ({
    text: s.text,
    citations: s.docids.map((d) => idxOf.get(d)!).slice(0, 3),
  }));
  return {
    references: refs,
    answer,
    perAspect,
    docText: docidText,
    nuggets: nuggetTrace,
  };
}
async function groundedReviseAnswer(
  a: {
    topic: TopicIdentity;
    llm: LlmClient;
    out?: string;
    env?: NodeJS.ProcessEnv;
    summary?: any;
  },
  draft: {
    references: string[];
    answer: { text: string; citations: number[] }[];
  },
  docText: Map<string, string>,
): Promise<{
  references: string[];
  answer: { text: string; citations: number[] }[];
} | null> {
  const refs = draft.references;
  if (refs.length === 0 || draft.answer.length === 0) return null;
  const sents = draft.answer.map((s) => ({
    text: s.text,
    docids: [...new Set(s.citations.map((c) => refs[c]).filter(Boolean))],
  }));
  const citedDocids = [...new Set(sents.flatMap((s) => s.docids))];
  if (citedDocids.length === 0) return null;
  const citedDocs = citedDocids.map((d) => ({
    docid: d,
    text: (docText.get(d) ?? "").slice(0, POLICY.revise_snippet_chars),
  }));
  try {
    let arr: any[] | null = null;
    for (let attempt = 1; attempt <= 3 && arr === null; attempt++) {
      const t0 = Date.now();
      let text = "";
      let errorCode: string | undefined;
      try {
        const r = await a.llm.generate({
          messages: [
            {
              role: "user",
              content: buildGroundedRevisionPrompt(a.topic, sents, citedDocs),
            },
          ],
          temperature: 0,
          maxTokens: POLICY.revise_max_tokens,
          responseFormat: "json_object",
        });
        text = r.text ?? "";
        const j = extractObjectCandidate(text);
        if (j) {
          const parsed = JSON.parse(j)?.answer;
          if (Array.isArray(parsed)) arr = parsed;
          else errorCode = "LLM_JSON_SHAPE";
        } else
          errorCode = text
            ? "LLM_JSON_PARSE_FAILED"
            : "LLM_EMPTY_ASSISTANT_MESSAGE";
      } catch (e) {
        errorCode = e instanceof Error ? e.name : "LLM_ERROR";
      }
      if (a.out && a.env && a.summary)
        recordAttempt({
          attempt: {
            attempt,
            success: arr !== null,
            errorCode,
            provider: a.llm.provider,
            model: a.llm.model,
            latencyMs: Date.now() - t0,
            outputChars: text.length,
          } as any,
          stage: "grounded_revise",
          qid: a.topic.qid,
          out: a.out,
          env: a.env,
          summary: a.summary,
        });
    }
    if (arr === null) return null;
    const out: { text: string; docids: string[] }[] = [];
    for (const s of arr) {
      if (typeof s?.text !== "string" || !s.text.trim()) continue;
      const rc: any[] = Array.isArray(s.citations) ? s.citations : [];
      const dc: string[] = [
        ...new Set(
          rc.map((c: any) => String(c)).filter((d: string) => docText.has(d)),
        ),
      ].slice(0, 3);
      if (dc.length === 0) continue;
      out.push({ text: s.text.trim(), docids: dc });
    }
    if (out.length === 0) return null;
    const newRefs: string[] = [];
    const idx = new Map<string, number>();
    for (const s of out)
      for (const d of s.docids)
        if (!idx.has(d)) {
          idx.set(d, newRefs.length);
          newRefs.push(d);
        }
    return {
      references: newRefs,
      answer: out.map((s) => ({
        text: s.text,
        citations: s.docids.map((d) => idx.get(d)!),
      })),
    };
  } catch {
    return null;
  }
}
const expectedFacts = new Map<string, string[]>();
async function decomposeAspects(a: {
  topic: TopicIdentity;
  llm: LlmClient;
  out: string;
  env: NodeJS.ProcessEnv;
  summary: any;
}): Promise<string[]> {
  let lastErr = "";
  for (let attempt = 1; attempt <= 3; attempt++) {
    try {
      const r = await a.llm.generate({
        messages: [
          {
            role: "user",
            content: buildAspectDecompositionPrompt(
              a.topic,
              POLICY.aspect_expected_facts,
            ),
          },
        ],
        temperature: 0,
        maxTokens: POLICY.aspect_decompose_max_tokens,
        responseFormat: "json_object",
      });
      const j = extractObjectCandidate(r.text);
      if (!j) {
        lastErr = `no JSON object in reply (${r.text.slice(0, 80)})`;
      } else {
        const parsed = JSON.parse(j);
        const arr = Array.isArray(parsed?.aspects) ? parsed.aspects : [];
        const out: string[] = [];
        expectedFacts.clear();
        for (const x of arr) {
          const title = (
            typeof x === "string" ? x : String(x?.title ?? "")
          ).trim();
          if (!title) continue;
          out.push(title);
          const ef = Array.isArray(x?.expected_facts)
            ? x.expected_facts.map((f: any) => String(f).trim()).filter(Boolean)
            : [];
          if (ef.length > 0) expectedFacts.set(title, ef);
          if (out.length >= POLICY.aspect_max) break;
        }
        if (out.length > 0) return out;
        lastErr = "parsed but aspects[] empty";
      }
    } catch (e) {
      lastErr = redact(e instanceof Error ? e.message : String(e), a.env);
    }
    if (attempt < 3) await sleep(500 * 2 ** attempt);
  }
  console.error(
    `  !! ${a.topic.qid}: aspect decomposition FAILED after 3 attempts (${lastErr}) -> per-aspect disabled for this topic`,
  );
  a.summary.aspect_decomposition_failures =
    (a.summary.aspect_decomposition_failures ?? 0) + 1;
  return [];
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
            `First think about which requirement each candidate can answer; then choose the set of at most ${k} documents that TOGETHER cover the most requirements. Prefer complementary documents over redundant ones; a document that covers an otherwise-uncovered requirement beats a fifth document about an already-covered one.`,
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
      ): LlmJsonValidationResult<{ selected: number[] }> =>
        typeof v === "object" &&
        v !== null &&
        Array.isArray((v as any).selected)
          ? { ok: true, value: v as { selected: number[] } }
          : { ok: false, message: "selected shape" },
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
      {
        candidates: docs.length,
        selected: picked.map((d) => d.docid),
        aspects: a.checklist ?? [],
      },
    );
    return picked;
  } catch {
    return docs.slice(0, k);
  }
}
async function answer(a: {
  topic: TopicIdentity;
  o: IterativeOptions;
  cfg: RagRunConfig;
  llm: LlmClient;
  llms: LlmSet;
  readDocs: Map<string, ReadDoc>;
  out: string;
  env: NodeJS.ProcessEnv;
  summary: any;
  presetAspects?: string[];
  budget?: { maxDocs: number; maxIterations: number };
}) {
  if (a.env.R_ONLY === "1") {
    const s = sanitizeAnswerDraft(
      buildExtractiveFallbackAnswerDraft(a.readDocs),
    );
    const full: AgenticRagOutputObject = {
      metadata: {
        team_id: a.cfg.teamId,
        run_id: a.cfg.runId,
        type: "automatic",
        narrative_id: a.topic.qid,
        title: "",
        narrative: a.topic.narrative,
        prompt: a.cfg.promptVersion,
        run_desc: POLICY.retrieval_policy,
        generator: a.llm.model,
        retrieval_depth: POLICY.output_depth,
      },
      references: s.references,
      answer: s.answer,
    };
    return normalizeRagOutputObjectReferences(full, {
      config: a.cfg,
      topic: a.topic,
      readDocids: new Set(a.readDocs.keys()),
    }).ragObject;
  }
  let draft: any;
  let docs = [...a.readDocs.values()];
  if (POLICY.setr_select && docs.length > POLICY.setr_k) {
    docs = await setrSelect(
      {
        topic: a.topic,
        llm: a.llms.base,
        out: a.out,
        env: a.env,
        summary: a.summary,
        checklist: a.presetAspects,
      },
      docs,
      POLICY.setr_k,
    );
  }
  const aspects =
    a.presetAspects ??
    (POLICY.comprehensive_answer ||
    POLICY.per_aspect_generation ||
    POLICY.dense_writing
      ? await decomposeAspects(a)
      : []);
  let perAspectTrace: any = null;
  let nuggetTrace: any = null;
  let ledgerTrace: any = null;
  const extraDocText = new Map<string, string>();
  if (POLICY.evidence_plan) {
    const denseDocs = docs.slice(0, POLICY.dense_evidence_docs).map((d) => ({
      docid: d.docid,
      text: d.text.slice(0, POLICY.dense_evidence_chars),
    }));
    const subs = (aspects.length > 0 ? aspects : [a.topic.narrative]).map(
      (t, i) => ({ id: `Q${i + 1}`, text: t.replace(/ \(vital\)$/, "") }),
    );
    const subqs = subs.map((s) => ({
      id: s.id,
      original_text: s.text,
      priority: "required" as const,
    }));
    let violation: string | undefined;
    for (
      let attempt = 1;
      attempt <= POLICY.evidence_plan_max_retries && !draft;
      attempt++
    ) {
      try {
        const raw = await generateJsonWithRetry({
          client: a.llms.writer,
          messages: [
            {
              role: "user",
              content: buildLedgerAnswerPrompt({
                topic: a.topic,
                documents: denseDocs,
                subquestions: subs,
                ...(violation ? { violation } : {}),
                ...(POLICY.enumerative_style
                  ? {
                      enumerative: {
                        minWords: POLICY.enumerative_words_min,
                        maxWords: POLICY.enumerative_words_max,
                      },
                    }
                  : {}),
              }),
            },
          ],
          temperature: 0,
          maxTokens: POLICY.evidence_plan_max_tokens,
          validate: validateLedgerPlan,
          stage: "answer_generation",
          maxRequestRetries: 3,
          onAttempt: (at) =>
            recordAttempt({
              attempt: at,
              stage: "answer_generation",
              qid: a.topic.qid,
              out: a.out,
              env: a.env,
              summary: a.summary,
            }),
        });
        const ledger = new ExposureLedger();
        for (const d of denseDocs) ledger.expose(d.docid, d.text);
        const plan = raw.value.answer_plan.map((s) => ({
          text: s.text,
          citations: s.citations,
          evidence_ids: s.evidence.map(
            (e) =>
              ledger.recordEvidence({
                docid: e.docid,
                subquestion_ids:
                  Array.isArray(e.subquestion_ids) && e.subquestion_ids.length
                    ? e.subquestion_ids
                    : [subs[0].id],
                exact_quote: e.exact_quote,
                claim: e.claim,
              }).evidence_id,
          ),
        }));
        const verified = enforceAnswerPlan({
          ledger,
          answerPlan: plan,
          subquestions: subqs,
          policy: POLICY.evidence_plan_policy as "basic" | "strict",
        });
        const refs = [...new Set(verified.flatMap((s) => s.citations))];
        draft = {
          value: {
            references: refs,
            answer: verified.map((s) => ({
              text: s.text,
              citations: s.citations
                .map((d) => refs.indexOf(d))
                .filter((i) => i >= 0),
            })),
          },
        };
        ledgerTrace = {
          mode: "evidence_plan",
          policy: POLICY.evidence_plan_policy,
          attempts: attempt,
          sentences: verified.length,
          evidence_records: ledger.records.length,
        };
      } catch (e) {
        if (e instanceof LedgerViolation) {
          violation = e.message;
          ledgerTrace = {
            mode: "evidence_plan_rejected",
            attempts: attempt,
            violation,
          };
        } else {
          ledgerTrace = {
            mode: "evidence_plan_error",
            attempts: attempt,
            error: redact(e instanceof Error ? e.message : String(e), a.env),
          };
        }
      }
    }
  }
  if (POLICY.dense_after_aspect && aspects.length > 0) {
    try {
      const before = a.readDocs.size;
      const pa = await generatePerAspectAnswer(
        {
          topic: a.topic,
          o: a.o,
          llm: POLICY.aspect_writer ? a.llms.writer : a.llm,
          out: a.out,
          env: a.env,
          summary: a.summary,
        },
        aspects,
        a.readDocs,
      );
      for (const [k, v] of pa.docText) extraDocText.set(k, v);
      perAspectTrace = {
        ...(pa.perAspect ?? {}),
        mode: "evidence_only",
        docs_before: before,
        docs_after: a.readDocs.size,
      };
      docs = [...a.readDocs.values()];
    } catch {}
  }
  if (!draft && POLICY.dense_writing) {
    try {
      const denseDocs = docs.slice(0, POLICY.dense_evidence_docs).map((d) => ({
        docid: d.docid,
        text: d.text.slice(0, POLICY.dense_evidence_chars),
      }));
      draft = await generateJsonWithRetry({
        client: a.llms.writer,
        messages: [
          {
            role: "user",
            content: buildDenseAnswerGenerationPrompt({
              topic: a.topic,
              documents: denseDocs,
              ...(aspects.length > 0 ? { checklist: aspects } : {}),
              ...(POLICY.enumerative_style
                ? {
                    enumerative: {
                      minWords: POLICY.enumerative_words_min,
                      maxWords: POLICY.enumerative_words_max,
                    },
                  }
                : {}),
            }),
          },
        ],
        temperature: 0,
        maxTokens: POLICY.answer_max_tokens,
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
    } catch {}
  }
  if (!draft && POLICY.per_aspect_generation && aspects.length > 0) {
    try {
      const pa = await generatePerAspectAnswer(
        {
          topic: a.topic,
          o: a.o,
          llm: POLICY.aspect_writer ? a.llms.writer : a.llm,
          out: a.out,
          env: a.env,
          summary: a.summary,
        },
        aspects,
        a.readDocs,
      );
      perAspectTrace = pa.perAspect;
      nuggetTrace = pa.nuggets;
      for (const [k, v] of pa.docText) extraDocText.set(k, v);
      if (pa.answer.length >= Math.min(3, aspects.length))
        draft = { value: { references: pa.references, answer: pa.answer } };
    } catch {}
  }
  if (!draft) {
    try {
      const promptDocs = docs.map((d) => ({
        docid: d.docid,
        text: d.text.slice(0, POLICY.answer_doc_chars),
      }));
      draft = await generateJsonWithRetry({
        client: a.llm,
        messages: [
          {
            role: "user",
            content: POLICY.comprehensive_answer
              ? buildComprehensiveAnswerPrompt({
                  topic: a.topic,
                  documents: promptDocs,
                  aspects,
                  atomic: POLICY.atomic_sentences,
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
        maxTokens: POLICY.answer_max_tokens,
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
          maxTokens: 2000,
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
  }
  let sanitized = sanitizeAnswerDraft(draft.value);
  if (sanitized.answer.length === 0)
    sanitized = sanitizeAnswerDraft(
      buildExtractiveFallbackAnswerDraft(a.readDocs),
    );
  {
    const docText = new Map<string, string>(
      [...a.readDocs.values()].map((d) => [d.docid, d.text]),
    );
    for (const [k, v] of extraDocText) if (!docText.has(k)) docText.set(k, v);
    let verifyStats: any = null;
    if (POLICY.evidence_ledger) {
      try {
        const lg = enforceLedger(sanitized as any, docText, {
          minSupport: POLICY.ledger_min_support,
        });
        sanitized = lg.draft as any;
        verifyStats = { mode: "evidence_ledger", ...lg.stats };
      } catch {}
    }
    if (POLICY.llm_revise) {
      try {
        const before = sanitized.answer.length;
        const rev = await groundedReviseAnswer(
          {
            topic: a.topic,
            llm: a.llms.writer,
            out: a.out,
            env: a.env,
            summary: a.summary,
          },
          sanitized as any,
          docText,
        );
        if (rev && rev.answer.length > 0) {
          if (POLICY.revise_never_drop && rev.answer.length < before) {
            verifyStats = {
              mode: "llm_revise_rejected",
              before,
              after: rev.answer.length,
            };
          } else {
            sanitized = rev as any;
            verifyStats = { mode: "llm_revise", sentences: rev.answer.length };
          }
        }
      } catch {}
    }
    if (verifyStats && verifyStats.mode === "llm_revise_rejected")
      verifyStats = null;
    if (verifyStats && verifyStats.mode === "evidence_ledger")
      verifyStats = { ...verifyStats, ledger_then: "reattribute_or_verify" };
    if (!verifyStats && POLICY.reattribute) {
      try {
        const r = await reattributeCitations(sanitized as any, docText, a.env, {
          threshold: POLICY.reattribute_threshold,
          maxCites: POLICY.reattribute_max_cites,
          snippetChars: 1500,
        });
        if (r.draft.answer.length > 0) {
          sanitized = r.draft as any;
          verifyStats = { mode: "reattribute", ...r.stats };
        }
      } catch {}
    }
    if (!verifyStats && POLICY.citation_verify) {
      const v = verifyCitations(sanitized as any, docText, {
        supportThreshold: POLICY.support_threshold,
      });
      if (v.draft.answer.length > 0) sanitized = v.draft as any;
      else
        sanitized = sanitizeAnswerDraft(
          buildExtractiveFallbackAnswerDraft(a.readDocs),
        );
      verifyStats = { mode: "keyword", ...v.stats };
    }
    if (POLICY.verify_revise && sanitized.answer.length > 0) {
      try {
        const B = POLICY.verify_batch,
          refs = sanitized.references,
          sents = sanitized.answer;
        const revised: { text: string; citations: number[] }[] = [];
        for (let b = 0; b < sents.length; b += B) {
          const batch = sents.slice(b, b + B);
          const prompt = buildVerifyRevisePrompt({
            topic: a.topic,
            references: refs,
            mode: POLICY.verify_mode as any,
            sentences: batch.map((s2) => ({
              text: s2.text,
              citations: s2.citations,
              evidence: [
                ...new Set(s2.citations.map((ci) => refs[ci]).filter(Boolean)),
              ]
                .slice(0, 2)
                .map((d) => ({
                  docid: d,
                  excerpt: (docText.get(d) ?? "").slice(0, 1200),
                })),
            })),
          });
          const draft = await generateJsonWithRetry({
            client: a.llms.writer,
            messages: [{ role: "user", content: prompt }],
            temperature: 0,
            maxTokens: POLICY.verify_max_tokens,
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
          revised.push(...(draft.value.answer as any[]));
        }
        if (revised.length > 0) {
          const r = sanitizeAnswerDraft({
            references: refs,
            answer: revised,
          } as any);
          if (r.answer.length > 0) sanitized = r;
        }
      } catch {}
    }
    writeJson(join(a.out, "topics", `${a.topic.qid}.gen_trace.json`), {
      topic_id: a.topic.qid,
      aspects,
      aspect_count: aspects.length,
      aspect_decomposition_ok: aspects.length > 0,
      per_aspect: perAspectTrace,
      nuggets: nuggetTrace,
      evidence_plan: ledgerTrace,
      verify: verifyStats,
      style: sentenceStyleStats(sanitized),
    });
  }
  {
    const lim = POLICY.max_answer_words;
    let words = 0;
    const kept: typeof sanitized.answer = [];
    for (const s of sanitized.answer) {
      const w = s.text.split(/\s+/).filter(Boolean).length;
      if (words + w > lim && kept.length > 0) break;
      kept.push(s);
      words += w;
    }
    if (kept.length < sanitized.answer.length)
      sanitized = { references: sanitized.references, answer: kept };
  }
  if (POLICY.order_citations_by_strength) {
    const dt = new Map<string, string>(
      [...a.readDocs.values()].map((d) => [d.docid, d.text]),
    );
    for (const [k, v] of extraDocText) if (!dt.has(k)) dt.set(k, v);
    sanitized = {
      references: sanitized.references,
      answer: sanitized.answer.map((s) =>
        s.citations.length < 2
          ? s
          : {
              ...s,
              citations: [...s.citations].sort(
                (x, y) =>
                  supportScore(s.text, dt.get(sanitized.references[y]) ?? "") -
                  supportScore(s.text, dt.get(sanitized.references[x]) ?? ""),
              ),
            },
      ),
    };
  }
  const full: AgenticRagOutputObject = {
    metadata: {
      team_id: a.cfg.teamId,
      run_id: a.cfg.runId,
      type: "automatic",
      narrative_id: a.topic.qid,
      title: "",
      narrative: a.topic.narrative,
      prompt: a.cfg.promptVersion,
      run_desc: POLICY.retrieval_policy,
      generator: a.llm.model,
      retrieval_depth: POLICY.output_depth,
    },
    references: sanitized.references,
    answer: sanitized.answer,
  };
  return normalizeRagOutputObjectReferences(full, {
    config: a.cfg,
    topic: a.topic,
    readDocids: new Set(a.readDocs.keys()),
  }).ragObject;
}
function sanitizeAnswerDraft(draft: AnswerDraft): AnswerDraft {
  const refCount = Array.isArray(draft.references)
    ? draft.references.length
    : 0;
  const answer = (Array.isArray(draft.answer) ? draft.answer : [])
    .map((sentence) => ({
      text: typeof sentence?.text === "string" ? sentence.text : "",
      citations: [
        ...new Set(
          (Array.isArray(sentence?.citations) ? sentence.citations : []).filter(
            (c) => Number.isInteger(c) && c >= 0 && c < refCount,
          ),
        ),
      ],
    }))
    .filter(
      (sentence) =>
        sentence.text.trim() !== "" && sentence.citations.length > 0,
    );
  return {
    references: Array.isArray(draft.references) ? draft.references : [],
    answer,
  };
}
function validateAnswer(v: unknown): LlmJsonValidationResult<AnswerDraft> {
  return isRecord(v) && Array.isArray(v.references) && Array.isArray(v.answer)
    ? { ok: true, value: v as AnswerDraft }
    : { ok: false, message: "answer shape" };
}
type LedgerPlan = {
  answer_plan: {
    text: string;
    citations: string[];
    evidence: {
      docid: string;
      exact_quote: string;
      claim: string;
      subquestion_ids: string[];
    }[];
  }[];
};
function validateLedgerPlan(v: unknown): LlmJsonValidationResult<LedgerPlan> {
  if (
    !isRecord(v) ||
    !Array.isArray(v.answer_plan) ||
    v.answer_plan.length === 0
  )
    return { ok: false, message: "answer_plan shape" };
  for (const s of v.answer_plan as any[]) {
    if (
      !isRecord(s) ||
      typeof s.text !== "string" ||
      !Array.isArray(s.citations) ||
      !Array.isArray(s.evidence) ||
      s.evidence.length === 0
    )
      return { ok: false, message: "answer_plan sentence shape" };
    for (const e of s.evidence)
      if (
        !isRecord(e) ||
        typeof e.docid !== "string" ||
        typeof e.exact_quote !== "string" ||
        typeof e.claim !== "string"
      )
        return { ok: false, message: "evidence record shape" };
  }
  return { ok: true, value: v as unknown as LedgerPlan };
}
async function readNew(a: {
  o: IterativeOptions;
  hits: Hit[];
  readDocs: Map<string, ReadDoc>;
  failedRead: string[];
  count: number;
  query: string;
  env: NodeJS.ProcessEnv;
  maxDocs?: number;
  findQueries?: string[];
}) {
  const cap = a.maxDocs ?? a.o.maxDocumentsRead;
  let added = 0;
  for (const [i, h] of a.hits.entries()) {
    if (added >= a.count || a.readDocs.size >= cap) return;
    if (a.readDocs.has(h.docid)) continue;
    try {
      const lines = POLICY.find_in_document
        ? POLICY.find_read_lines
        : a.o.documentReadLimit;
      const d = await readDoc(a.o, h.docid, lines, a.env);
      if (d.found) {
        let text = d.text;
        if (POLICY.find_in_document && (a.findQueries?.length ?? 0) > 0) {
          const ps = findInDocument(text, a.findQueries as string[], {
            mode: POLICY.find_mode as any,
            maxPassages: POLICY.find_max_passages,
            windowChars: POLICY.find_window_chars,
            scanLimit: POLICY.find_scan_limit,
            impl: POLICY.find_impl as "lines" | "windows",
            contextBefore: POLICY.find_context_before,
            contextAfter: POLICY.find_context_after,
          });
          if (ps.length > 0)
            text = ps.map((p) => `[offset ${p.offset}] ${p.text}`).join("\n\n");
        }
        a.readDocs.set(h.docid, {
          docid: h.docid,
          text,
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
const RETRY_MAX = 16;
function retryDelay(r: Response | null, a: number) {
  const ra = Number(r?.headers.get("retry-after"));
  const server = Number.isFinite(ra) && ra > 0 ? Math.min(30000, ra * 1000) : 0;
  const growth = Math.max(1000, 600 * 2 ** Math.min(a, 7));
  const base = Math.max(server, growth);
  return base + Math.floor(Math.random() * Math.max(500, base * 0.5));
}
async function search(
  o: IterativeOptions,
  query: string,
  depth: number,
  env: NodeJS.ProcessEnv,
  sink?: Map<string, string>,
) {
  const giant = (e: unknown) => {
    const m =
      String((e as any)?.message ?? e) + String((e as any)?.cause ?? "");
    return (
      m.includes("Cannot create a string") ||
      m.includes("terminated") ||
      m.includes("ECONNRESET") ||
      m.includes("fetch failed") ||
      m.includes("aborted")
    );
  };
  let lastErr: unknown;
  const tries = query.length > 200 ? [query, query.slice(0, 200)] : [query];
  for (const q of tries) {
    if (q !== query)
      console.error("search: retrying with the first 200 query characters");
    for (let d = depth; d >= Math.min(depth, 313); d = Math.floor(d / 2)) {
      try {
        return await searchAtDepth(o, q, d, env, sink);
      } catch (e) {
        lastErr = e;
        if (!giant(e)) throw e;
        console.error(
          `search: oversized response (${String((e as any)?.message ?? e).slice(0, 40)}); halving depth ${d}`,
        );
      }
    }
  }
  throw lastErr;
}
async function searchAtDepth(
  o: IterativeOptions,
  query: string,
  depth: number,
  env: NodeJS.ProcessEnv,
  sink?: Map<string, string>,
) {
  const token = env[o.pyseriniTokenEnv]?.trim();
  const url = `${o.pyseriniBaseUrl.replace(/\/+$/, "")}/v1/${o.pyseriniIndex}/search?${new URLSearchParams({ query, hits: String(depth) })}`;
  for (let a = 1; a <= RETRY_MAX; a++) {
    let r: Response;
    try {
      r = await fetch(url, {
        headers: token ? { authorization: `Bearer ${token}` } : {},
      });
    } catch (e) {
      if (a === RETRY_MAX)
        throw new Error(
          `Pyserini fetch failed: ${e instanceof Error ? e.message : String(e)}`,
        );
      await sleep(retryDelay(null, a));
      continue;
    }
    if (r.ok) {
      const v = (await r.json()) as any;
      const cs = (v.candidates ?? [])
        .filter((c: any) => typeof c.docid === "string")
        .slice(0, depth);
      if (sink)
        for (const c of cs) {
          if (typeof c.doc === "string" && !sink.has(c.docid))
            sink.set(c.docid, c.doc.slice(0, POLICY.tail_reselect_text_chars));
        }
      return cs.map((c: any) => ({
        docid: c.docid,
        score: Number(c.score) || 0,
      }));
    }
    if (![429, 500, 502, 503, 504].includes(r.status) || a === RETRY_MAX)
      throw new Error(`Pyserini HTTP ${r.status}`);
    await sleep(retryDelay(r, a));
  }
  throw new Error("search failed");
}
function cutDoc(text: string, limit: number) {
  const lines = text.split(/\r?\n/);
  return {
    found: true,
    text: lines.slice(0, limit).join("\n"),
    truncated: lines.length > limit,
  };
}
async function readDoc(
  o: IterativeOptions,
  docid: string,
  limit: number,
  env: NodeJS.ProcessEnv,
) {
  const hit = readCache(o.pyseriniIndex, docid);
  if (hit)
    return hit.found
      ? cutDoc(hit.text, limit)
      : { found: false, text: "", truncated: false };
  const token = env[o.pyseriniTokenEnv]?.trim();
  const url = `${o.pyseriniBaseUrl.replace(/\/+$/, "")}/v1/${o.pyseriniIndex}/doc/${encodeURIComponent(docid)}`;
  const maxAttempts = RETRY_MAX;
  for (let a = 1; a <= maxAttempts; a++) {
    let r: Response;
    try {
      r = await fetch(url, {
        headers: token ? { authorization: `Bearer ${token}` } : {},
      });
    } catch (e) {
      if (a === maxAttempts)
        throw new Error(
          `Pyserini doc fetch failed: ${e instanceof Error ? e.message : String(e)}`,
        );
      await sleep(retryDelay(null, a));
      continue;
    }
    if (r.status === 404) {
      writeCache(o.pyseriniIndex, docid, { found: false, text: "" });
      return { found: false, text: "", truncated: false };
    }
    if (r.ok) {
      const v = (await r.json()) as any;
      const text = extractText(v.doc);
      writeCache(o.pyseriniIndex, docid, { found: true, text });
      return cutDoc(text, limit);
    }
    if (![429, 500, 502, 503, 504].includes(r.status) || a === maxAttempts)
      throw new Error(`Pyserini doc HTTP ${r.status}`);
    await sleep(retryDelay(r, a));
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
        i === 0
          ? POLICY.bm25_anchor_weight
          : typeof queries[i] === "string" &&
              queries[i].startsWith("[query2doc]")
            ? POLICY.q2d_weight
            : POLICY.followup_query_weight,
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
