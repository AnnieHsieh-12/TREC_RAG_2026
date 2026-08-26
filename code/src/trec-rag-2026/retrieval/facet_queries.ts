
import { readFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import type { LlmClient } from "../../llm/types";
import { compileJsonValidator, formatJsonValidationError } from "../../lib/json_validation";
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
  /** One observation-blind second-pass rewrite per core facet. */
  rewrites: string[];
  /** Initial query for each core facet. */
  facets: string[];
  ok: boolean;
  /** Audit details for decomposition, background hints, and repairs. */
  cores?: CoreFacet[];
  backgroundHints?: string[];
  repairs?: { decomposition: boolean; initialQuery: boolean };
  failureReasons?: string[];
};

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
  const entry = { text, validator: compileJsonValidator(JSON.parse(text)) };
  schemaCache.set(name, entry);
  return entry;
}

const FROZEN_ATTEMPTS = 4;
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

function repairSpanOffsets(narrative: string, spans: any[], enabled: boolean): { start: number; end: number; text: string }[] {
  const out: { start: number; end: number; text: string }[] = [];
  for (const raw of spans ?? []) {
    const text = typeof raw?.text === "string" ? raw.text : "";
    if (!text) continue;
    const start = Number(raw?.start), end = Number(raw?.end);
    if (Number.isSafeInteger(start) && Number.isSafeInteger(end) && narrative.slice(start, end) === text) {
      out.push({ start, end, text });
      continue;
    }
    const found = enabled ? narrative.indexOf(text) : -1;
    if (found >= 0) out.push({ start: found, end: found + text.length, text });
    else out.push({ start, end, text });
  }
  return out;
}

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
 * Generate core-facet queries through decomposition, deterministic and LLM
 * validation, one repair attempt, initial BM25 query generation, and an
 * observation-blind reformulation pass.
 *
 * Any failed stage returns `ok:false`; the caller falls back to the anchor and
 * Query2Doc results.
 */
export async function generateFacetQueries(
  llm: LlmClient,
  narrative: string,
  opts: { spanAutofix?: boolean } = {},
): Promise<FacetQuerySet> {
  const fail = (reasons: string[]): FacetQuerySet => ({ rewrites: [], facets: [], ok: false, failureReasons: reasons });
  const repairs = { decomposition: false, initialQuery: false };
  try {
    let decomposition = await callFrozen(llm, "core-decomposition-v1", {
      schema_version: "core-decomposition-input-v1", narrative,
    }, 6000);
    let cores: CoreFacet[] = asCores(decomposition, narrative, opts.spanAutofix ?? false);
    if (opts.spanAutofix) syncSpansIntoDecomposition(decomposition, cores);
    let backgroundHints: string[] = Array.isArray(decomposition?.background_hints) ? decomposition.background_hints.map(String) : [];

    let errors = deterministicDecompositionErrors(narrative, cores);
    const validation = await callFrozen(llm, "decomposition-validation-v1", {
      schema_version: "decomposition-validation-input-v1", narrative, decomposition,
    }, 8000);
    const verdict = String(validation?.verdict ?? "pass");
    const repairInstructions: string[] = Array.isArray(validation?.repair_instructions)
      ? validation.repair_instructions.map(String) : [];
    if (verdict === "fail") return fail([...errors, ...repairInstructions, "semantic validator returned fail"]);

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
    cores = [...cores].sort((l, r) => {
      const ls = Math.min(...(l.source_spans.length ? l.source_spans.map((x) => x.start) : [0]));
      const rs = Math.min(...(r.source_spans.length ? r.source_spans.map((x) => x.start) : [0]));
      return ls - rs || compareUnicodeCodePoints(l.core_id, r.core_id);
    });

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

    const rewrites: string[] = [];
    for (const c of cores) {
      try {
        const second = await callFrozen(llm, "blind-reformulation-v1", {
          schema_version: "blind-reformulation-input-v1", narrative,
          core: { core_id: c.core_id, subquestion: c.subquestion, source_spans: c.source_spans },
          background_hints: backgroundHints,
          initial_query: byCore.get(c.core_id),
        }, 2500);
        const q = typeof second?.refinement_query === "string" ? second.refinement_query.trim() : "";
        if (q) rewrites.push(q);
      } catch { /* A failed rewrite does not invalidate the other core facets. */ }
    }

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
