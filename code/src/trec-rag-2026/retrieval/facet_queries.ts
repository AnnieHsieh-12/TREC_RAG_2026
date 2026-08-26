// Core Facet 多查詢建池（V4，① 查詢建構 / ⑤ 尾段召回的上游）。
//
// **忠實移植** 自 andy/shaohuawu 的 Dynamic Core Retrieval Stage 1：
//   worktrees/pi-serini-dynamic-core/src/orchestration/dynamic_core_retrieval/qualification.ts
//   worktrees/pi-serini-dynamic-core/resources/dynamic-core-retrieval/prompts/*.txt
//   worktrees/pi-serini-dynamic-core/resources/dynamic-core-retrieval/schemas/*.json
// 那 9 支 prompt 與 15 支 schema 已原封複製到本專案的 resources/dynamic-core-retrieval/
// （md5 相同）。這裡實作的是他的四段流程，去掉可重現性機具
// （DurableArtifactStore / JCS 正規化 / preflight / planned calls）—— 那些與方法無關。
//
// 他的流程與實測（22 題，三份 qrels）：
//   original_bm25          R@1000 0.2506   ← 只用原題
//   core_one_shot          R@1000 0.2974   ← 只做第一輪 Core 查詢
//   observation_blind      R@1000 0.3052   ← **最佳**：再加一輪「不看檢索結果」的重寫
//   retrieval_conditioned  R@1000 0.2928   ← 讀了再決定第二輪，反而更差
//
// 所以第二輪一定要 Blind：conditioned 版本實測 −0.0038、8–14 輸，沒有淨增益。
//
// 面向分兩類，只有前者建立檢索分支：
//   Core Facet       = 原題明確必答「且可搜尋」-> 各自發一條 Initial Query
//   Background hints = 只提供查詢提示 -> 不單獨搜尋
//
// 共同代價是「前段崩壞」（facet RRF 讓 MAP@100 0.078 -> 0.048）。所以這個模組產生的池子
// **只用來補後段**，前段由 runner 的 splice 保護（POLICY.splice_head_keep），不參與頭部排序。

import { readFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import type { LlmClient } from "../../llm/types";
// 他的驗證器,原檔逐字搬（md5 相同）:andy/.../pi-serini-dynamic-core/src/lib/json_validation.ts
// 相依只有 typebox 這個 npm 套件(已加進 package.json),沒有牽到 pi-serini 的其他部分。
import { compileJsonValidator, formatJsonValidationError } from "../../lib/json_validation";
// 他的查詢正規化與碼點排序,原檔逐字搬（md5 相同,含 Unicode 17.0 的 case folding 對照表）:
//   andy/.../dynamic_core_retrieval/{surface_query,unicode_order,unicode_case_folding}.ts
//   + generated/unicode_case_folding_17_0.ts
import { normalizeSurfaceQuery } from "./dynamic_core/surface_query";
import { compareUnicodeCodePoints } from "./dynamic_core/unicode_order";

const RES = resolve(dirname(fileURLToPath(import.meta.url)), "../../../resources/dynamic-core-retrieval");
const promptCache = new Map<string, string>();
function frozenPrompt(name: string): string {
  const cached = promptCache.get(name);
  if (cached !== undefined) return cached;
  const text = readFileSync(join(RES, "prompts", `${name}.txt`), "utf8").trim();
  promptCache.set(name, text);
  return text;
}

export type CoreFacet = {
  core_id: string;
  subquestion: string;
  source_spans: { start: number; end: number; text: string }[];
  initial_query?: string;
};

export type FacetQuerySet = {
  /** 第二輪 observation-blind 重寫（每個 Core 一條）—— 他實測最佳的那一組 */
  rewrites: string[];
  /** 每個 Core 的 initial query */
  facets: string[];
  ok: boolean;
  /** 稽核用：拆出來的 Core、背景提示、是否用到 repair */
  cores?: CoreFacet[];
  backgroundHints?: string[];
  repairs?: { decomposition: boolean; initialQuery: boolean };
  failureReasons?: string[];
};

// ── 逐字移植：他的確定性檢查（qualification.ts:111）───────────────────────
function deterministicDecompositionErrors(narrative: string, cores: CoreFacet[]): string[] {
  const errors: string[] = [];
  if (cores.length < 1) errors.push("cores must be non-empty");
  if (cores.length > 20) errors.push("cores must contain at most 20 facets; merge over-granular facets");
  const coreIds = new Set<string>();
  const subquestions = new Set<string>();
  for (const core of cores) {
    if (coreIds.has(core.core_id)) errors.push(`duplicate core_id: ${core.core_id}`);
    coreIds.add(core.core_id);
    const normalized = normalizeSurfaceQuery(core.subquestion);
    if (normalized.length === 0) errors.push(`empty normalized subquestion: ${core.core_id}`);
    else if (subquestions.has(normalized)) errors.push(`duplicate normalized subquestion: ${core.core_id}`);
    subquestions.add(normalized);
    if (!core.source_spans?.length) errors.push(`source_spans must be non-empty: ${core.core_id}`);
    for (const [index, span] of (core.source_spans ?? []).entries()) {
      if (
        !Number.isSafeInteger(span.start) || !Number.isSafeInteger(span.end) ||
        span.start < 0 || span.end <= span.start || span.end > narrative.length ||
        narrative.slice(span.start, span.end) !== span.text
      ) errors.push(`source span ${index} is not an exact narrative slice: ${core.core_id}`);
    }
  }
  return errors;
}

// role → 他的 output schema 檔名。
// 他的 frozen model client 是用 structured output 強制輸出形狀的（compileJsonValidator + outputSchema），
// prompt 裡只寫「Return only JSON matching <schema>」。我們的 provider 沒有那層強制
// （NCHC 不送 response_format、Codex 橋連 responseFormat 都會忽略），
// 所以必須把 schema 本文附進 prompt —— 不然模型只會自己編欄位名。
// 實測：不附 schema 時模型回 {"facet","spans"}，附了才回 {"core_id","subquestion","source_spans"}。
const OUTPUT_SCHEMA: Record<string, string> = {
  "core-decomposition-v1": "core-decomposition-output-v1",
  "decomposition-repair-v1": "core-decomposition-output-v1",
  "decomposition-validation-v1": "decomposition-validation-output-v1",
  "initial-query-generation-v1": "initial-query-output-v1",
  "initial-query-repair-v1": "initial-query-output-v1",
  "blind-reformulation-v1": "refinement-output-v1",
};
const schemaCache = new Map<string, { text: string; validator: ReturnType<typeof compileJsonValidator<any>> }>();
function frozenSchema(role: string) {
  const name = OUTPUT_SCHEMA[role];
  if (!name) return null;
  const cached = schemaCache.get(name);
  if (cached !== undefined) return cached;
  const text = readFileSync(join(RES, "schemas", `${name}.schema.json`), "utf8").trim();
  // 他的 asSchema()：把 JSON Schema 檔直接當 TSchema 餵給 compileJsonValidator
  const entry = { text, validator: compileJsonValidator(JSON.parse(text)) };
  schemaCache.set(name, entry);
  return entry;
}

// maxTokens 一律開大：gpt-oss-120b 是推理模型,推理 token 也吃這個額度,
// 而附了 schema 之後 prompt 更長 —— 今天已經被這個坑咬過兩次(面向分解截斷、per-aspect 75% 回空)。
// 他的原版有 maxAcceptedOutputTokens 逐角色設定;我們沒有那份設定檔,所以一律給寬。
const FROZEN_ATTEMPTS = 4;
// ── 逐字移植：他的 initial query 確定性檢查（qualification.ts:165）─────────
// 我先前只檢查「有沒有漏掉某個 Core」，其餘七條都沒做 —— 那是簡化，現在補齊。
function deterministicInitialQueryErrors(
  narrative: string,
  cores: CoreFacet[],
  byCore: Map<string, string>,
): string[] {
  const errors: string[] = [];
  const expectedIds = cores.map((c) => c.core_id);
  const expectedSet = new Set(expectedIds);
  const seenCoreIds = new Set<string>();
  const seenQueries = new Map<string, string>();
  const normalizedOriginal = normalizeSurfaceQuery(narrative);
  for (const [coreId, query] of byCore) {
    if (!expectedSet.has(coreId)) errors.push(`unknown core_id: ${coreId}`);
    if (seenCoreIds.has(coreId)) errors.push(`duplicate query core_id: ${coreId}`);
    seenCoreIds.add(coreId);
    const normalized = normalizeSurfaceQuery(query);
    const tokenCount = normalized.split(" ").filter(Boolean).length;
    if (normalized.length === 0) errors.push(`empty normalized query: ${coreId}`);
    if (tokenCount > 32) errors.push(`query exceeds 32 surface tokens: ${coreId}`);
    if ([...normalized].length > 256) errors.push(`query exceeds 256 Unicode code points: ${coreId}`);
    if (normalized === normalizedOriginal) errors.push(`query duplicates the original narrative: ${coreId}`);
    const prior = seenQueries.get(normalized);
    if (prior !== undefined) errors.push(`query duplicates ${prior}: ${coreId}`);
    seenQueries.set(normalized, coreId);
  }
  for (const coreId of expectedIds)
    if (!seenCoreIds.has(coreId)) errors.push(`missing query for ${coreId}`);
  if (byCore.size !== expectedIds.length)
    errors.push("initial query output count does not match the frozen Core count");
  return errors;
}

async function callFrozen(llm: LlmClient, role: string, input: Record<string, unknown>, maxTokens = 4000): Promise<any> {
  const schema = frozenSchema(role);
  const base = [
    frozenPrompt(role),
    ...(schema ? ["", "Return JSON conforming exactly to this JSON Schema. Use these property names verbatim:", schema.text] : []),
    "",
    "Input:",
    JSON.stringify(input),
  ].join("\n");
  let last = "";
  let schemaFeedback = "";
  for (let attempt = 1; attempt <= FROZEN_ATTEMPTS; attempt++) {
    try {
      const r = await llm.generate({
        messages: [{ role: "user", content: base + schemaFeedback }],
        temperature: 0, maxTokens, responseFormat: "json_object",
      });
      const t = r.text.trim();
      if (!t) { last = `${role}: empty assistant message`; }
      else {
        const fenced = /```(?:json)?\s*([\s\S]*?)```/i.exec(t);
        const body = (fenced ? fenced[1] : t).trim();
        const a = body.indexOf("{"), b = body.lastIndexOf("}");
        const parsed = JSON.parse(a >= 0 && b > a ? body.slice(a, b + 1) : body);
        // 他的 frozen_model_client 對每次輸出跑 compileJsonValidator(outputSchema).validate()，
        // 形狀不對就拋錯。我們照做,但把錯誤塞回 prompt 再試 —— 他有 structured output 可以
        // 保證形狀,我們沒有,所以用重試補。
        const errors = schema ? schema.validator.errors(parsed) : [];
        if (errors.length === 0) return parsed;
        const detail = formatJsonValidationError(errors);
        last = `${role}: output failed schema validation: ${detail}`;
        schemaFeedback = `\n\nYour previous output was REJECTED by the schema validator:\n  ${detail}\nFix exactly those problems. Use the property names from the schema verbatim.`;
      }
    } catch (e) {
      last = `${role}: ${e instanceof Error ? e.message : String(e)}`;
    }
    if (attempt < FROZEN_ATTEMPTS) await new Promise((r2) => setTimeout(r2, 400 * 2 ** attempt));
  }
  throw new Error(last || `${role}: failed after ${FROZEN_ATTEMPTS} attempts`);
}

// ⚠️ 對原版的一處偏離：span 位移的確定性修補。**預設關閉（facet_span_autofix）。**
//
// 他的檢查要求 narrative.slice(start,end) === span.text，模型算錯位移就整題判失敗
// （他的 qualifyTopics 也會有 failureCount，所以「失敗」本身是忠實行為）。
// 但實測 gpt-oss-120b 當查詢模型時，21 題裡有一半因為這條掛掉 —— 而模型抄的 text
// 通常是對的，錯的只是它自己算的字元位置。
//
// 這裡在檢查之前先修：若 narrative 裡找得到逐字相同的 span.text，就用 indexOf 把
// start/end 算對。規則的本意（span 必須是題目原文的精確片段）完全保留 ——
// text 仍然必須逐字存在，找不到的一樣會被下游的確定性檢查擋掉。
//
// **但它有一個不明顯的連帶效應**：span 錯誤本來會觸發 decomposition_repair，
// 而修復後的拆解往往比第一次細（實測 2 個 Core -> 5 個）。開了 autofix 就不再觸發，
// 於是「一題拆幾個 Core」這個核心產物被改變了 —— 這已經不只是錯誤處理的差異。
// 所以預設關閉，照他的行為走；只有在確認強模型也大量失敗時才考慮打開。
function repairSpanOffsets(narrative: string, spans: any[], enabled: boolean): { start: number; end: number; text: string }[] {
  const out: { start: number; end: number; text: string }[] = [];
  for (const raw of spans ?? []) {
    const text = typeof raw?.text === "string" ? raw.text : "";
    if (!text) continue;
    const start = Number(raw?.start), end = Number(raw?.end);
    if (Number.isSafeInteger(start) && Number.isSafeInteger(end) && narrative.slice(start, end) === text) {
      out.push({ start, end, text });   // 位移本來就對，原封不動
      continue;
    }
    const found = enabled ? narrative.indexOf(text) : -1;
    if (found >= 0) out.push({ start: found, end: found + text.length, text });
    else out.push({ start, end, text });   // 關閉時或找不到 -> 保留原值，讓確定性檢查去擋
  }
  return out;
}

// autofix 開啟時，把修好的位移寫回 decomposition 物件 —— 語意驗證 LLM 看的是這個
// 物件，不是 asCores 的產物。不寫回的話驗證員永遠看到錯的位移，修復模型又算不對
// 字元位置，於是二次驗證必掛（API 版 sol 的位移系統性全錯，rag2026-0/1 實測）。
function syncSpansIntoDecomposition(decomposition: any, cores: CoreFacet[]): void {
  if (!Array.isArray(decomposition?.cores)) return;
  const byId = new Map(cores.map((c) => [c.core_id, c.source_spans]));
  for (const raw of decomposition.cores) {
    const fixed = raw && typeof raw.core_id === "string" ? byId.get(raw.core_id) : undefined;
    if (fixed) raw.source_spans = fixed.map((x) => ({ start: x.start, end: x.end, text: x.text }));
  }
}

function asCores(v: any, narrative: string, spanAutofix: boolean): CoreFacet[] {
  return Array.isArray(v?.cores)
    ? v.cores.filter((c: any) => c && typeof c.core_id === "string" && typeof c.subquestion === "string")
        .map((c: any) => ({
          core_id: String(c.core_id),
          subquestion: String(c.subquestion),
          source_spans: repairSpanOffsets(narrative, Array.isArray(c.source_spans) ? c.source_spans : [], spanAutofix),
          initial_query: typeof c.initial_query === "string" ? c.initial_query : undefined,
        }))
    : [];
}

/**
 * 產生 Core Facet 查詢集合。
 *
 * 流程（對應他的 qualifyOneTopic + stage1 的 blind 第二輪）：
 *   1. core_decomposition          拆出 Core facets，每個都要標題目原文的字元區間
 *   2. 確定性檢查 + decomposition_validation   驗證覆蓋、忠實度、重複、可搜尋性
 *   3. decomposition_repair        只在有錯時跑一次
 *   4. initial_query_generation    每個 Core 一條 BM25 查詢（+ 一次 repair）
 *   5. blind_reformulation         每個 Core 再產一條「不看檢索結果」的第二輪查詢
 *
 * 任何一步失敗都回 ok:false，呼叫端退回只用原題 + Q2D（fail-open，不影響 V3 的行為）。
 */
export async function generateFacetQueries(
  llm: LlmClient,
  narrative: string,
  opts: { spanAutofix?: boolean } = {},
): Promise<FacetQuerySet> {
  const fail = (reasons: string[]): FacetQuerySet => ({ rewrites: [], facets: [], ok: false, failureReasons: reasons });
  const repairs = { decomposition: false, initialQuery: false };
  try {
    // 1. 拆解
    let decomposition = await callFrozen(llm, "core-decomposition-v1", {
      schema_version: "core-decomposition-input-v1", narrative,
    }, 6000);
    let cores: CoreFacet[] = asCores(decomposition, narrative, opts.spanAutofix ?? false);
    if (opts.spanAutofix) syncSpansIntoDecomposition(decomposition, cores);
    let backgroundHints: string[] = Array.isArray(decomposition?.background_hints) ? decomposition.background_hints.map(String) : [];

    // 2. 確定性檢查 + 語意驗證
    let errors = deterministicDecompositionErrors(narrative, cores);
    const validation = await callFrozen(llm, "decomposition-validation-v1", {
      schema_version: "decomposition-validation-input-v1", narrative, decomposition,
    }, 8000);
    const verdict = String(validation?.verdict ?? "pass");
    const repairInstructions: string[] = Array.isArray(validation?.repair_instructions)
      ? validation.repair_instructions.map(String) : [];
    if (verdict === "fail") return fail([...errors, ...repairInstructions, "semantic validator returned fail"]);

    // 3. 修復（只跑一次，與他的流程一致）
    if (errors.length > 0 || verdict === "repair") {
      const instructions = [...errors, ...repairInstructions].filter((x, i, all) => all.indexOf(x) === i);
      if (instructions.length === 0) return fail(["validator asked for repair without instructions"]);
      decomposition = await callFrozen(llm, "decomposition-repair-v1", {
        schema_version: "decomposition-repair-input-v1", narrative, decomposition, repair_instructions: instructions,
      }, 6000);
      cores = asCores(decomposition, narrative, opts.spanAutofix ?? false);
      if (opts.spanAutofix) syncSpansIntoDecomposition(decomposition, cores);
      backgroundHints = Array.isArray(decomposition?.background_hints) ? decomposition.background_hints.map(String) : backgroundHints;
      repairs.decomposition = true;
      errors = deterministicDecompositionErrors(narrative, cores);
      // 他修復後會**再跑一次語意驗證**，兩邊都要過才算數（qualification.ts:296-318）。
      // 我先前只重跑確定性檢查 —— 那是簡化，現在補上。
      const revalidation = await callFrozen(llm, "decomposition-validation-v1", {
        schema_version: "decomposition-validation-input-v1", narrative, decomposition,
      }, 8000);
      const secondVerdict = String(revalidation?.verdict ?? "pass");
      const secondInstructions: string[] = Array.isArray(revalidation?.repair_instructions)
        ? revalidation.repair_instructions.map(String) : [];
      if (errors.length > 0 || secondVerdict !== "pass")
        return fail([...errors, ...secondInstructions, `second semantic verdict: ${secondVerdict}`]);
    }
    if (cores.length === 0) return fail(["no cores after decomposition"]);
    // 他在產 initial query 之前會把 Core 依「最小 span 起點、同起點再比 core_id」排序
    // （qualification.ts:320），順序會影響模型看到的上下文，所以照做。
    cores = [...cores].sort((l, r) => {
      const ls = Math.min(...(l.source_spans.length ? l.source_spans.map((x) => x.start) : [0]));
      const rs = Math.min(...(r.source_spans.length ? r.source_spans.map((x) => x.start) : [0]));
      return ls - rs || compareUnicodeCodePoints(l.core_id, r.core_id);
    });

    // 4. 每個 Core 一條 initial query（+ 一次 repair）
    const queryInput = {
      schema_version: "initial-query-input-v1", narrative,
      cores: cores.map((c) => ({ core_id: c.core_id, subquestion: c.subquestion, source_spans: c.source_spans })),
      background_hints: backgroundHints,
    };
    let queries = await callFrozen(llm, "initial-query-generation-v1", queryInput, 8000);
    let byCore = new Map<string, string>(
      (Array.isArray(queries?.queries) ? queries.queries : [])
        .filter((q: any) => q && typeof q.core_id === "string" && typeof q.query === "string" && q.query.trim())
        .map((q: any) => [String(q.core_id), String(q.query).trim()]),
    );
    let missing = deterministicInitialQueryErrors(narrative, cores, byCore);
    if (missing.length > 0) {
      queries = await callFrozen(llm, "initial-query-repair-v1", { ...queryInput, previous: queries, failures: missing }, 8000);
      byCore = new Map<string, string>(
        (Array.isArray(queries?.queries) ? queries.queries : [])
          .filter((q: any) => q && typeof q.core_id === "string" && typeof q.query === "string" && q.query.trim())
          .map((q: any) => [String(q.core_id), String(q.query).trim()]),
      );
      repairs.initialQuery = true;
      missing = deterministicInitialQueryErrors(narrative, cores, byCore);
      if (missing.length > 0) return fail(missing);
    }
    const facets = cores.map((c) => byCore.get(c.core_id)!).filter(Boolean);

    // 5. observation-blind 第二輪（他實測最佳的一組：R@1000 0.3052）
    const rewrites: string[] = [];
    for (const c of cores) {
      try {
        const second = await callFrozen(llm, "blind-reformulation-v1", {
          schema_version: "blind-reformulation-input-v1", narrative,
          core: { core_id: c.core_id, subquestion: c.subquestion, source_spans: c.source_spans },
          background_hints: backgroundHints,
          initial_query: byCore.get(c.core_id),
        }, 2500);
        // refinement-output-v1 的欄位是 refinement_query，不是 query
        const q = typeof second?.refinement_query === "string" ? second.refinement_query.trim() : "";
        if (q) rewrites.push(q);
      } catch { /* 單一 Core 的第二輪失敗不影響其他 —— 他的 fallback 也是這樣 */ }
    }

    // 去重（同一條查詢重複發沒有意義）
    const seen = new Set<string>();
    const dedup = (list: string[]) => list.filter((q) => {
      const k = normalizeSurfaceQuery(q);
      if (!k || seen.has(k)) return false;
      seen.add(k); return true;
    });
    const facetQueries = dedup(facets);
    const secondRound = dedup(rewrites);
    if (facetQueries.length === 0) return fail(["all initial queries were empty or duplicates"]);

    return { rewrites: secondRound, facets: facetQueries, ok: true, cores, backgroundHints, repairs };
  } catch (e) {
    return fail([e instanceof Error ? e.message : String(e)]);
  }
}
