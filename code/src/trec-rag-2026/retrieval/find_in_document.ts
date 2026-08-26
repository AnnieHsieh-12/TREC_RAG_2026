// 文件內段落定位（V5，階段 ⑦ 讀證據）。
//
// 問題：現行 readDoc 只取每篇文件的「前 N 行」（POLICY.document_read_limit=200）。
// ClimbMix 的文件很長，關鍵證據若出現在第 5000 行就永遠拿不到 —— 這不是排序問題，
// 是「證據根本沒進到 prompt 裡」。另一條線也獨立踩過同一個坑
// （M1：把固定 300 字前綴換成 BM25 打分的 512 詞 chunk，實測選到 offset 33908、87134 的段落）。
//
// 做法沿用既有的 find_in_document 設計（參考實作 79 行，但綁在 Pi-search 框架上：
// PiSearchBackend / ResearchSessionState / schemas）。
// 這裡只重實作演算法本身，不帶框架耦合：
//   exact   字面片語命中，每次 +10
//   lexical 詞彙重疊（不管詞序）
//   hybrid  兩者相加（預設）
//
// 與原始設計的差異：原始設計是 agent 主動呼叫（每篇最多 3 次、每次最多 10 段），
// 這裡是 runner 在讀文件時自動套用 —— 因為本專案的迴圈是程式控制的，沒有 agent 決定何時呼叫。

export type Passage = {
  offset: number;   // 在原文中的字元位置，方便追溯
  text: string;
  score: number;
  matched: string[]; // 命中的片語（exact 部分），供 trace 稽核
  lineStart?: number; // 逐行版才有（1 起算）
  lineEnd?: number;
};

export type FindMode = "exact" | "lexical" | "hybrid";

// ── 忠實移植版（預設）────────────────────────────────────────────────────
// 逐字移植 annie/peiju 的 agentic-research/find_in_document.ts 的演算法本體，
// 只拿掉框架耦合（PiSearchBackend / ResearchSessionState / budgets / abort signal）——
// 她那支是 agent 主動呼叫的工具，我們是 runner 讀文件時自動套用。
//
// 演算法（與原版逐條對應）：
//   • 以「行」為單位掃描，不是字元視窗
//   • exact   ：命中的 pattern 數 × 10
//   • lexical ：該行與 query 詞彙的重疊「個數」（絕對數，不是比例）
//   • 額外加分：同一 pattern 在該行重複出現，每多一次 +2
//   • 段落 = 命中行的前 5 行 ~ 後 8 行（context_before / context_after 預設值）
//   • 依 score 由高到低、同分則 start 由小到大排序
//   • 合併重疊段落：範圍取聯集、分數相加、matched 取聯集後排序
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

  // 她的 params 有 query（做 lexical）與 patterns（做 exact）兩個欄位。
  // 我們只有面向清單，所以同一份清單同時當 query 與 patterns —— 面向本身就是短名詞片語。
  const patterns = queries.map((p) => p.trim()).filter(Boolean);
  const queryTerms = new Set(terms(queries.join(" ")));
  const lines = body.split(/\r?\n/);
  // 行號 → 字元位移，讓回傳值跟既有呼叫端相容
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

// 她的 terms() / escapeRegExp()，逐字移植。
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

// 把長文件切成有重疊的視窗。重疊是必要的：證據句常跨越切點，
// 沒有重疊就會被切成兩半、兩邊都拿不到完整句子。
function windows(text: string, size: number, stride: number): { offset: number; text: string }[] {
  const out: { offset: number; text: string }[] = [];
  if (text.length <= size) return [{ offset: 0, text }];
  for (let i = 0; i < text.length; i += stride) {
    out.push({ offset: i, text: text.slice(i, i + size) });
    if (i + size >= text.length) break;
  }
  return out;
}

/**
 * 在一篇文件裡找出最相關的幾段。
 *
 * @param text     文件全文
 * @param queries  要找的東西：題目敘述 + 各面向。面向越具體、定位越準。
 * @param opts.mode          exact / lexical / hybrid（預設 hybrid，與原始設計相同）
 * @param opts.maxPassages   回傳幾段（原始設計每次呼叫最多 10 段）
 * @param opts.windowChars   視窗大小
 * @param opts.scanLimit     最多掃多少字元（上限參考 50,000 text units）
 */
export function findInDocument(
  text: string,
  queries: string[],
  opts: { mode?: FindMode; maxPassages?: number; windowChars?: number; scanLimit?: number; impl?: "lines" | "windows"; contextBefore?: number; contextAfter?: number } = {},
): Passage[] {
  // 預設走忠實移植的逐行版；"windows" 是本專案自建的字元視窗版，留著當對照。
  if ((opts.impl ?? "lines") === "lines") return findInDocumentLines(text, queries, opts);
  const mode = opts.mode ?? "hybrid";
  const maxPassages = opts.maxPassages ?? 6;
  const windowChars = opts.windowChars ?? 1200;
  const scanLimit = opts.scanLimit ?? 200_000;

  const body = text.length > scanLimit ? text.slice(0, scanLimit) : text;
  if (!body.trim() || queries.length === 0) return [];

  // exact 用的片語：面向本身就是短名詞片語，直接當片語找。
  const phrases = queries.map((q) => q.trim().toLowerCase()).filter((q) => q.length >= 4);
  // lexical 用的詞袋
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
        // 用比例而非絕對數，否則長視窗只因為字多就贏
        score += (hit / Math.sqrt(wt.length)) * 5;
      }
    }
    return { offset: w.offset, text: w.text, score, matched };
  });

  return scored
    .filter((p) => p.score > 0)
    .sort((a, b) => b.score - a.score)
    // 去掉重疊過多的相鄰視窗，避免回傳的都是同一段
    .reduce<Passage[]>((keep, p) => {
      if (keep.length >= maxPassages) return keep;
      if (keep.some((k) => Math.abs(k.offset - p.offset) < windowChars / 2)) return keep;
      keep.push(p);
      return keep;
    }, []);
}

/**
 * 依「題目有幾個面向」決定讀多少文件（自適應預算）。
 * 實測分佈：≤3 個必答子問題 -> 2 輪 / 12 篇；4–6 -> 3 輪 / 20 篇；≥7 -> 4 輪 / 30 篇
 * （dev22 的分佈是 8 / 13 / 1 題）。
 * 現行 runner 是固定 3 輪 / 12 篇，複雜題目讀得太少。
 */
export function adaptiveBudget(aspectCount: number): { maxDocs: number; maxIterations: number } {
  if (aspectCount >= 7) return { maxDocs: 30, maxIterations: 4 };
  if (aspectCount >= 4) return { maxDocs: 20, maxIterations: 3 };
  return { maxDocs: 12, maxIterations: 2 };
}
