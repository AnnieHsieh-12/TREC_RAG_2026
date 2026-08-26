
export type Passage = {
  offset: number;
  text: string;
  score: number;
  matched: string[];
  lineStart?: number;
  lineEnd?: number;
};

export type FindMode = "exact" | "lexical" | "hybrid";

export function findInDocumentLines(
  text: string,
  queries: string[],
  opts: { mode?: FindMode; maxPassages?: number; scanLimit?: number; contextBefore?: number; contextAfter?: number; caseSensitive?: boolean } = {},
): Passage[] {
  const mode = opts.mode ?? "hybrid";
  const maxPassages = opts.maxPassages ?? 6;
  const scanLimit = opts.scanLimit ?? 200_000;
  const before = opts.contextBefore ?? 5;
  const after = opts.contextAfter ?? 8;
  const caseSensitive = opts.caseSensitive ?? false;

  const body = text.length > scanLimit ? text.slice(0, scanLimit) : text;
  if (!body.trim() || queries.length === 0) return [];

  const patterns = queries.map((p) => p.trim()).filter(Boolean);
  const queryTerms = new Set(terms(queries.join(" ")));
  const lines = body.split(/\r?\n/);
  const lineOffsets: number[] = [];
  { let acc = 0; for (const l of lines) { lineOffsets.push(acc); acc += l.length + 1; } }

  const flags = caseSensitive ? "g" : "gi";
  const candidates: Array<{ start: number; end: number; matched: string[]; score: number }> = [];
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i] ?? "";
    const hay = caseSensitive ? line : line.toLowerCase();
    const matched = patterns.filter((p) => hay.includes(caseSensitive ? p : p.toLowerCase()));
    let score = 0;
    if (mode === "exact" || mode === "hybrid") score += matched.length * 10;
    if (mode === "lexical" || mode === "hybrid") {
      const lineTerms = terms(line);
      score += lineTerms.filter((t) => queryTerms.has(t)).length;
    }
    if (score > 0) {
      for (const p of patterns)
        score += ((line.match(new RegExp(escapeRegExp(p), flags)) ?? []).length - (matched.includes(p) ? 1 : 0)) * 2;
      candidates.push({ start: Math.max(1, i + 1 - before), end: Math.min(lines.length, i + 1 + after), matched, score });
    }
  }
  candidates.sort((a, b) => b.score - a.score || a.start - b.start);

  const merged: typeof candidates = [];
  for (const c of candidates) {
    if (merged.length >= maxPassages) break;
    const existing = merged.find((m) => !(c.end < m.start || c.start > m.end));
    if (existing) {
      existing.start = Math.min(existing.start, c.start);
      existing.end = Math.max(existing.end, c.end);
      existing.score += c.score;
      existing.matched = [...new Set([...existing.matched, ...c.matched])].sort();
    } else merged.push({ ...c, matched: [...new Set(c.matched)].sort() });
  }
  return merged.map((p) => ({
    offset: lineOffsets[p.start - 1] ?? 0,
    text: lines.slice(p.start - 1, p.end).join("\n"),
    score: Number(p.score.toFixed(4)),
    matched: p.matched,
    lineStart: p.start,
    lineEnd: p.end,
  }));
}

function terms(text: string): string[] {
  return text.toLowerCase().match(/[\p{L}\p{N}]+/gu) ?? [];
}
function escapeRegExp(s: string) {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

const STOP = new Set(
  ("the a an and or of to in for on with is are was were be been by at from as that this these those it its" +
   " what which who whom how why when where does do did can could should would will may might must not no")
    .split(" "),
);

function toks(s: string): string[] {
  return (s.toLowerCase().match(/[a-z0-9][a-z0-9'-]*/g) ?? []).filter((t) => !STOP.has(t) && t.length > 1);
}

function windows(text: string, size: number, stride: number): { offset: number; text: string }[] {
  const out: { offset: number; text: string }[] = [];
  if (text.length <= size) return [{ offset: 0, text }];
  for (let i = 0; i < text.length; i += stride) {
    out.push({ offset: i, text: text.slice(i, i + size) });
    if (i + size >= text.length) break;
  }
  return out;
}

/** Select the most relevant passages from a complete document. */
export function findInDocument(
  text: string,
  queries: string[],
  opts: { mode?: FindMode; maxPassages?: number; windowChars?: number; scanLimit?: number; impl?: "lines" | "windows"; contextBefore?: number; contextAfter?: number } = {},
): Passage[] {
  if ((opts.impl ?? "lines") === "lines") return findInDocumentLines(text, queries, opts);
  const mode = opts.mode ?? "hybrid";
  const maxPassages = opts.maxPassages ?? 6;
  const windowChars = opts.windowChars ?? 1200;
  const scanLimit = opts.scanLimit ?? 200_000;

  const body = text.length > scanLimit ? text.slice(0, scanLimit) : text;
  if (!body.trim() || queries.length === 0) return [];

  const phrases = queries.map((q) => q.trim().toLowerCase()).filter((q) => q.length >= 4);
  const qTokens = new Set(queries.flatMap(toks));

  const scored = windows(body, windowChars, Math.floor(windowChars / 2)).map((w) => {
    const lower = w.text.toLowerCase();
    let score = 0;
    const matched: string[] = [];

    if (mode === "exact" || mode === "hybrid") {
      for (const p of phrases) {
        let idx = lower.indexOf(p), hits = 0;
        while (idx !== -1) { hits++; idx = lower.indexOf(p, idx + p.length); }
        if (hits > 0) { score += hits * 10; matched.push(p); }
      }
    }
    if (mode === "lexical" || mode === "hybrid") {
      const wt = toks(w.text);
      if (wt.length > 0) {
        const hit = wt.filter((t) => qTokens.has(t)).length;
        score += (hit / Math.sqrt(wt.length)) * 5;
      }
    }
    return { offset: w.offset, text: w.text, score, matched };
  });

  return scored
    .filter((p) => p.score > 0)
    .sort((a, b) => b.score - a.score)
    .reduce<Passage[]>((keep, p) => {
      if (keep.length >= maxPassages) return keep;
      if (keep.some((k) => Math.abs(k.offset - p.offset) < windowChars / 2)) return keep;
      keep.push(p);
      return keep;
    }, []);
}

/** Scale the evidence budget with the number of requested aspects. */
export function adaptiveBudget(aspectCount: number): { maxDocs: number; maxIterations: number } {
  if (aspectCount >= 7) return { maxDocs: 30, maxIterations: 4 };
  if (aspectCount >= 4) return { maxDocs: 20, maxIterations: 3 };
  return { maxDocs: 12, maxIterations: 2 };
}
