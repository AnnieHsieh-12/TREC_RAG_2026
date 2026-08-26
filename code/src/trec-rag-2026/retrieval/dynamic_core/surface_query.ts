import { UNICODE_CASE_FOLDING_17_0 } from "./generated/unicode_case_folding_17_0";
import { defaultCaseFold } from "./unicode_case_folding";

const TERMINAL_PUNCTUATION = new Set(["?", "!", "。"]);
const UNICODE_WHITE_SPACE = /\p{White_Space}+/gu;

function normalizeWhiteSpace(input: string): string {
  return input.replace(UNICODE_WHITE_SPACE, " ").replace(/^ +| +$/gu, "");
}

export function normalizeSurfaceQuery(input: string): string {
  let normalized = normalizeWhiteSpace(
    defaultCaseFold(input.normalize("NFKC"), UNICODE_CASE_FOLDING_17_0),
  );
  while (TERMINAL_PUNCTUATION.has(normalized.at(-1) ?? "")) {
    normalized = normalized.slice(0, -1).replace(/ +$/u, "");
  }
  return normalized;
}
