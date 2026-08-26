// ⑤ Self-assign gap-fill loop.
//
// The RAG task is scored as recall over vital nuggets, so the generator is optimising a target it never
// sees. This module closes that loop at inference time: it predicts the nugget list for a narrative with
// the SAME prompts the organizers' AutoNuggetizer uses (ported from RAGDoll's
// `ragdoll/nuggetizer/prompts.py`), assigns the draft answer against them, and reports which vital
// nuggets the draft misses so the caller can run targeted retrieval for exactly those.
//
// The predicted list is built only from the narrative and retrieved evidence.

import type { LlmClient } from "../../llm/types";
import type { TopicIdentity } from "../agentic-rag-baseline/contracts";

export type NuggetAssignment = "support" | "partial_support" | "not_support";

const CREATOR_SYSTEM =
  "You are NuggetizeLLM, an intelligent assistant that can update a list of atomic nuggets to best provide all the information required for the query.";
const SCORER_SYSTEM =
  "You are NuggetizeScoreLLM, an intelligent assistant that can label a list of atomic nuggets based on their importance for a given search query.";
const ASSIGNER_SYSTEM =
  "You are NuggetizeAssignerLLM, an intelligent assistant that can label a list of atomic nuggets based on if they are captured by a given passage.";

export function buildNuggetCreatePrompt(query: string, context: string, maxNuggets: number, initial: string[] = []): string {
  return [
    `Update the list of atomic nuggets of information (1-12 words), if needed, so they best provide the information required for the query. Leverage only the initial list of nuggets (if exists) and the provided context (this is an iterative process). Return only the final list of all nuggets in a Pythonic list format (even if no updates). Make sure there is no redundant information. Ensure the updated nugget list has at most ${maxNuggets} nuggets (can be less), keeping only the most vital ones. Order them in decreasing order of importance. Prefer nuggets that provide more interesting information.`,
    "",
    `Search Query: ${query}`,
    "Context:",
    context,
    `Search Query: ${query}`,
    `Initial Nugget List: ${JSON.stringify(initial)}`,
    `Initial Nugget List Length: ${initial.length}`,
    "",
    'Only update the list of atomic nuggets (if needed, else return as is). Do not explain. Always answer in short nuggets (not questions). List in the form ["a", "b", ...].',
    "Updated Nugget List:",
  ].join("\n");
}

export function buildNuggetScorePrompt(query: string, nuggets: string[]): string {
  return [
    `Based on the query, label each of the ${nuggets.length} nuggets either a vital or okay based on the following criteria. Vital nuggets represent concepts that must be present in a "good" answer; on the other hand, okay nuggets contribute worthwhile information about the target but are not essential. Return the list of labels in a Pythonic list format (type: List[str]). The list should be in the same order as the input nuggets. Make sure to provide a label for each nugget.`,
    "",
    `Search Query: ${query}`,
    `Nugget List: ${JSON.stringify(nuggets)}`,
    "",
    "Only return the list of labels (List[str]). Do not explain.",
    "Labels:",
  ].join("\n");
}

export function buildNuggetAssignPrompt(query: string, context: string, nuggets: string[]): string {
  return [
    `Based on the query and passage, label each of the ${nuggets.length} nuggets either as support, partial_support, or not_support using the following criteria. A nugget that is fully captured in the passage should be labeled as support. A nugget that is partially captured in the passage should be labeled as partial_support. If the nugget is not captured at all, label it as not_support. Return the list of labels in a Pythonic list format (type: List[str]). The list should be in the same order as the input nuggets. Make sure to provide a label for each nugget.`,
    "",
    `Search Query: ${query}`,
    `Passage: ${context}`,
    `Nugget List: ${JSON.stringify(nuggets)}`,
    "",
    "Only return the list of labels (List[str]). Do not explain.",
    "Labels:",
  ].join("\n");
}

/** Tolerant port of RAGDoll's `parse_label_list`: pull the outermost [...] and read quoted or bare items. */
export function parseStringList(text: string): string[] | null {
  const cleaned = text.replace(/```python/g, "").replace(/```/g, "").trim();
  const open = cleaned.indexOf("[");
  const close = cleaned.lastIndexOf("]");
  if (open === -1 || close === -1 || close <= open) return null;
  const body = cleaned.slice(open + 1, close);
  const items: string[] = [];
  let i = 0;
  while (i < body.length) {
    while (i < body.length && (body[i] === "," || /\s/.test(body[i]))) i++;
    if (i >= body.length) break;
    const ch = body[i];
    if (ch === "'" || ch === '"') {
      const quote = ch;
      i++;
      let buf = "";
      let closed = false;
      while (i < body.length) {
        if (body[i] === "\\" && i + 1 < body.length) { buf += body[i + 1]; i += 2; continue; }
        if (body[i] === quote) { i++; closed = true; break; }
        buf += body[i]; i++;
      }
      if (!closed) return null;
      items.push(buf);
    } else {
      let buf = "";
      while (i < body.length && body[i] !== ",") { buf += body[i]; i++; }
      const trimmed = buf.trim();
      if (trimmed) items.push(trimmed);
    }
  }
  return items;
}

async function askForList(llm: LlmClient, system: string, user: string, maxTokens: number): Promise<string[] | null> {
  try {
    const r = await llm.generate({ messages: [{ role: "system", content: system }, { role: "user", content: user }], temperature: 0, maxTokens });
    return parseStringList(r.text);
  } catch { return null; }
}

/**
 * Predict the nugget list for this narrative from our own retrieved evidence, one aspect at a time.
 *
 * Generate nuggets per aspect and take their union so consolidation occurs
 * within a focused sub-question rather than across the entire narrative.
 */
export async function predictNuggets(
  llm: LlmClient,
  topic: TopicIdentity,
  docs: { docid: string; text: string }[],
  aspects: string[],
  opts: { maxNuggets: number; perAspect: number; contextDocs: number; contextChars: number },
): Promise<string[]> {
  const focuses = aspects.length > 0 ? aspects : [""];
  const seen = new Map<string, string>(); // normalised text -> first spelling kept
  for (const focus of focuses) {
    const context = docs.slice(0, opts.contextDocs).map((d, i) => `[${i}] ${d.text.slice(0, opts.contextChars)}`).join("\n\n");
    const query = focus ? `${topic.narrative}\n\nFocus only on this sub-question: ${focus}` : topic.narrative;
    const list = await askForList(llm, CREATOR_SYSTEM, buildNuggetCreatePrompt(query, context, opts.perAspect), 2000);
    if (!list) continue; // a failed aspect just contributes nothing
    for (const raw of list) {
      const text = raw.trim();
      if (!text) continue;
      const key = text.toLowerCase().replace(/[^a-z0-9 ]/g, "").replace(/\s+/g, " ").trim();
      if (key && !seen.has(key)) seen.set(key, text);
    }
    if (seen.size >= opts.maxNuggets) break;
  }
  return [...seen.values()].slice(0, opts.maxNuggets);
}

/** Keep only the nuggets the scorer calls vital — the V/V_strict metrics ignore the rest. */
export async function selectVitalNuggets(llm: LlmClient, topic: TopicIdentity, nuggets: string[]): Promise<string[]> {
  if (nuggets.length === 0) return [];
  const labels = await askForList(llm, SCORER_SYSTEM, buildNuggetScorePrompt(topic.narrative, nuggets), 1200);
  if (!labels || labels.length !== nuggets.length) return nuggets; // scorer unusable: treat all as vital
  const vital = nuggets.filter((_, i) => labels[i].trim().toLowerCase().startsWith("vital"));
  return vital.length > 0 ? vital : nuggets;
}

const CHUNK = 15; // long nugget lists make the assigner miscount; keep each prompt short

/** Assign the draft answer against the nugget list, in chunks. Returns null if any chunk fails. */
export async function assignNuggets(
  llm: LlmClient,
  topic: TopicIdentity,
  answerText: string,
  nuggets: string[],
): Promise<NuggetAssignment[] | null> {
  if (nuggets.length === 0) return [];
  const out: NuggetAssignment[] = [];
  for (let start = 0; start < nuggets.length; start += CHUNK) {
    const chunk = nuggets.slice(start, start + CHUNK);
    const labels = await askForList(llm, ASSIGNER_SYSTEM, buildNuggetAssignPrompt(topic.narrative, answerText, chunk), 1200);
    if (!labels || labels.length !== chunk.length) return null;
    for (const label of labels) {
      const low = label.trim().toLowerCase();
      if (low.includes("partial")) out.push("partial_support");
      else if (low.includes("not") || low.includes("no_support")) out.push("not_support");
      else if (low.includes("support")) out.push("support");
      else out.push("not_support");
    }
  }
  return out;
}

/**
 * Which vital nuggets the draft fails to state completely. Partially-supported nuggets are included:
 * under V_strict a partial nugget scores zero, so it is just as much a gap as a missing one.
 */
export async function findNuggetGaps(
  llm: LlmClient,
  topic: TopicIdentity,
  answerText: string,
  vitalNuggets: string[],
  maxGaps: number,
): Promise<{ gaps: string[]; assignments: NuggetAssignment[] | null }> {
  const assignments = await assignNuggets(llm, topic, answerText, vitalNuggets);
  if (!assignments) return { gaps: [], assignments: null };
  const gaps = vitalNuggets.filter((_, i) => assignments[i] !== "support").slice(0, maxGaps);
  return { gaps, assignments };
}
