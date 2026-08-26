// Evidence ledger（V6，階段 ⑨ 引用驗證）。結構性引用驗證（evidence-ledger 型）。
//
// 四條硬規則：
//   1. 引文必須逐字出現在 read_document / find_in_document 曝光的文字裡
//   2. evidence claim 必須「等於」最終句子（不能寫完再改）
//   3. 每個 citation 都要有對應的 evidence record，第一個 citation 對應第一筆 record
//   4. 搜尋 snippet 本身不算證據，必須真的讀過
// 實測：weighted support 0.9402、FS 89.6%、NS 2.4% —— 全隊引用最強。
//
// ⚠️ 這裡是「近似」而非移植。原始產物與實作都不在這份複本裡
// （runs/.../agentic-rag-integrated-v1-gpt54mini-dev22/ 底下沒有答案檔，src/ 也搜不到 ledger），
// 而且原始流程是「先 record_evidence 再寫句子」——由 agent 在生成前就把引文釘死。
// 本專案是程式控制迴圈、句子先產出，所以只能做**事後檢查**：
// 對每個 (句子, 被引用文件) 對，回文件裡找最支持它的段落；找不到就改指或拿掉那個引用。
//
// 關鍵設計：**永不刪句**。在 recall 型的涵蓋指標下刪句 = 刪掉它帶的 nugget，
// 這是 runner 既有的 revise_never_drop 精神，ledger 也必須遵守。
// 句子失去所有引用時保留句子、標記為未支持，交由後續的 re-attribution 處理。

import { findInDocument } from "../retrieval/find_in_document";

export type LedgerStats = {
  pairs: number;      // 檢查過的 (句子, 文件) 對
  kept: number;       // 通過
  rewired: number;    // 改指到別的 reference
  dropped: number;    // 找不到支持，拿掉該引用
  orphan: number;     // 拿掉後失去所有引用的句子（保留句子，不刪）
};

type Draft = { references: string[]; answer: { text: string; citations: number[] }[] };

// 句子與某段文字的支持度：用實詞重疊比例。
// 不用「逐字片語」是因為生成的句子本來就是改寫過的，逐字比對會全軍覆沒；
// 用 findInDocument 找出文件裡最相關的那一段，再看該段涵蓋了句子多少實詞。
export function supportScore(sentence: string, docText: string): number {
  const ps = findInDocument(docText, [sentence], { mode: "hybrid", maxPassages: 1, windowChars: 1500 });
  if (ps.length === 0) return 0;
  const st = new Set((sentence.toLowerCase().match(/[a-z0-9][a-z0-9'-]*/g) ?? []).filter((t) => t.length > 3));
  if (st.size === 0) return 0;
  const pt = new Set((ps[0].text.toLowerCase().match(/[a-z0-9][a-z0-9'-]*/g) ?? []));
  let hit = 0;
  for (const t of st) if (pt.has(t)) hit++;
  return hit / st.size;
}

export function enforceLedger(
  draft: Draft,
  docText: Map<string, string>,
  opts: { minSupport?: number } = {},
): { draft: Draft; stats: LedgerStats } {
  const minSupport = opts.minSupport ?? 0.45;
  const stats: LedgerStats = { pairs: 0, kept: 0, rewired: 0, dropped: 0, orphan: 0 };
  const refs = draft.references;
  if (refs.length === 0 || draft.answer.length === 0) return { draft, stats };

  // 只考慮真的讀過的文件（規則 4：snippet 不算證據）
  const readable = refs.filter((d) => (docText.get(d) ?? "").length > 0);

  const outSentences = draft.answer.map((s) => {
    const keptDocids: string[] = [];
    for (const ci of s.citations) {
      const docid = refs[ci];
      if (!docid) continue;
      stats.pairs++;
      const text = docText.get(docid) ?? "";
      if (text && supportScore(s.text, text) >= minSupport) {
        keptDocids.push(docid);
        stats.kept++;
        continue;
      }
      // 規則 1 沒過 -> 先試著改指到別的、真的支持它的 reference
      let best = "", bestScore = 0;
      for (const cand of readable) {
        if (cand === docid || keptDocids.includes(cand)) continue;
        const sc = supportScore(s.text, docText.get(cand) ?? "");
        if (sc > bestScore) { bestScore = sc; best = cand; }
      }
      if (best && bestScore >= minSupport) { keptDocids.push(best); stats.rewired++; }
      else stats.dropped++;
    }
    if (keptDocids.length === 0) stats.orphan++;   // 保留句子，不刪（never-drop）
    return { text: s.text, docids: [...new Set(keptDocids)].slice(0, 3) };
  });

  // 重建 references（規則 3：順序對齊，第一個 citation 對應第一筆）
  const newRefs: string[] = [];
  const idx = new Map<string, number>();
  for (const s of outSentences) {
    for (const d of s.docids) if (!idx.has(d)) { idx.set(d, newRefs.length); newRefs.push(d); }
  }
  const answer = outSentences
    .filter((s) => s.docids.length > 0)   // 官方 schema 要求每句都要有引用
    .map((s) => ({ text: s.text, citations: s.docids.map((d) => idx.get(d)!) }));

  // 全軍覆沒時退回原稿，不要交出空答案
  if (answer.length === 0) return { draft, stats };
  return { draft: { references: newRefs, answer }, stats };
}
