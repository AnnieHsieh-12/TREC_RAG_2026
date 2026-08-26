import { existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { basename, dirname, join, resolve } from "node:path";
import { parseTrecRag2026TopicsTsv } from "../retrieval/topics";
import { generateQuery2Doc } from "../retrieval/query2doc";
import { generateFacetQueries } from "../retrieval/facet_queries";
import { findInDocument, adaptiveBudget } from "../retrieval/find_in_document";
import { createLlmClient, generateJsonWithRetry } from "../../llm/create";
import { normalizeLlmClientConfig, safeLlmConfigForArtifacts, type RawLlmClientConfig } from "../../llm/config";
import type { LlmAttemptTrace, LlmClient, LlmJsonValidationResult } from "../../llm/types";
import { buildAnswerGenerationPrompt, buildCompactAnswerGenerationPrompt, buildComprehensiveAnswerPrompt, buildDenseAnswerGenerationPrompt, buildLedgerAnswerPrompt, buildVerifyRevisePrompt, buildAspectDecompositionPrompt, buildAspectAnswerPrompt, buildIntegrationPrompt, buildReflectionPrompt, buildGroundedRevisionPrompt } from "../agentic-rag-baseline/prompts";
import { ExposureLedger, enforceAnswerPlan, LedgerViolation } from "./evidence_ledger_v2";
import { verifyCitations } from "./citation_verify";
import { denseScores } from "../retrieval/dense_rerank";
import { reattributeCitations } from "./citation_reattribute";
import { predictNuggets, findNuggetGaps } from "./nugget_loop";
import { enforceLedger, supportScore } from "./evidence_ledger";
import { AGENTIC_RAG_BASELINE_PROMPT_VERSION, type AgenticRagBaselineConfig, type AgenticRagOutputObject, type TopicIdentity } from "../agentic-rag-baseline/contracts";
import { normalizeRagOutputObjectReferences, validateRagOutputObjectStrict } from "../agentic-rag-baseline/validation";
import { buildExtractiveFallbackAnswerDraft, type AgenticRagBaselineReadDoc } from "../agentic-rag-baseline/fallback";
import { AgenticRagValidationError } from "../agentic-rag-baseline/validation";
import { evaluateRankings, type Qrels, type Rankings } from "../../evaluation/retrieval_metrics";
import { rerankWithCrossEncoder, type RerankCandidate } from "../retrieval/cross_encoder_rerank";
import { readCache, writeCache } from "../retrieval/doc_cache";

type Topic={qid:string;title:string;narrative:string}; type Hit={docid:string;score:number}; type ReadDoc=AgenticRagBaselineReadDoc; type AnswerDraft={references:string[];answer:Array<{text:string;citations:number[]}>}; type Judge={enough:boolean;missing_aspects:string[];followup_queries:string[]};
export type IterativeOptions={runId:string;teamId:string;outputDir:string;topicsPath:string;qrelsDir:string;pyseriniBaseUrl:string;pyseriniIndex:string;pyseriniTokenEnv:string;limitTopics?:number;initialDocs:number;docsPerIteration:number;maxDocumentsRead:number;maxIterations:number;documentReadLimit:number;llm:RawLlmClientConfig;force?:boolean;resume?:boolean;env?:NodeJS.ProcessEnv;
/** 階梯版本的 POLICY 覆寫。由 versions/V*.ts 傳入;不傳就用 BASE_POLICY(最新全開狀態)。 */
policy?:PolicyOverride;
/** 依角色分配模型。不傳就沿用 llm。
 *  寫手(答案生成/改寫)與查詢生成建議走 codex:gpt-5.6-sol(訂閱額度,API $0);
 *  judge/面向分解/nugget 這種高頻角色留給 nchc:gpt-oss-120b(免費且量大)。
 *  理由見 specs/README.md「模型角色分配」:V2 全開時每題約 95 次呼叫,全用 Sol 要 8.7 小時。 */
llmWriter?:RawLlmClientConfig;llmQuery?:RawLlmClientConfig};
/** 三個角色的 client。不設定的角色會 fallback 到 base。 */
type LlmSet={base:LlmClient;writer:LlmClient;query:LlmClient};
const BASE_POLICY={retrieval_policy:"iterative-agentic-anchor-bm25-weighted-rrf-w025-query2doc-fusion-peraspect-official-top5000-varK-breadthfirst",output_depth:5000,bm25_anchor_weight:1,followup_query_weight:0.25,rrf_k:60,rerank_depth:100,q2d_enabled:true,q2d_weight:1.0,q2d_query_repeat:5,fusion_dense:true,fusion_bm25_weight:1.0,fusion_ce_weight:1.0,fusion_dense_weight:1.0,fusion_rrf_k:60,comprehensive_answer:true,citation_verify:true,support_threshold:0.4,answer_doc_chars:1600,answer_max_tokens:4096,per_aspect_generation:true,aspect_docs:4,aspect_search_depth:100,aspect_max_tokens:1500,aspect_rounds:3,aspect_target_sentences:5,reflection:true,reflection_max_gaps:2,reattribute:true,reattribute_threshold:0.3,reattribute_max_cites:2,vark_threshold:0.5,vark_min:4,vark_max:15,llm_revise:true,revise_snippet_chars:1200,revise_max_tokens:3000,
// Breadth-first budgeting (coverage rebuild). The metric is recall over vital nuggets, so the 1024-word
// budget must be SPREAD across every aspect rather than consumed first-come-first-served: the old code
// appended aspects in order and truncated at 1000 words, silently dropping the last aspects entirely.
breadth_first:true,max_answer_words:1000,aspect_max:12,aspect_reserve_frac:0.75,aspect_ce_rerank:false,aspect_ce_pool:20,integrate_answer:false,integrate_max_tokens:24576,
// Never drop a sentence during grounded revision: under a recall metric, deleting a sentence deletes the
// nuggets it carried. Revision may reword or re-cite, but the sentence count must not shrink.
revise_never_drop:true,
// ⑤ Self-assign gap-fill: predict this narrative's nuggets with the organizers' own AutoNuggetizer
// prompts, assign the draft against them, and run targeted retrieval for the vital ones still missing.
// Sized from the dev gold nuggets: 41 vital nuggets per topic at the median (range 23-72), grouped into
// a median of 8 sub-narratives. Predicting only 30 would under-generate the checklist from the start.
nugget_loop:true,nugget_max:50,nugget_per_aspect:6,nugget_max_gaps:15,nugget_context_docs:10,nugget_context_chars:1200,
// CE 失效退回：整題最高 ce_calibrated 都達不到門檻 => CrossEncoder 對這題沒有鑑別力,
// 讓它和 dense 一起否決 BM25 只會製造雜訊 -> 該題退回純 BM25 名次。
// dev22 只有 topic 515 觸發(max_ce=0.362;其餘 21 題 0.811-0.998,它的 k=4 是被 vark_min 夾上來的)。
// 離線實測 A/B1/B2/B3 四個 run:提交檔 nDCG@k +0.0008/+0.0052/+0.0013/+0.0002(全正),只動 1 題。
// 對照:改用「每軌保底」會動到 21/22 題且 nDCG@k -0.0113,降低 rrf_k 則單調變差 -- 兩者皆已否決。
// 設 0 可停用。掃描工具:tools/sweep_fusion_fix.py
ce_dead_threshold:0.5,
// 沒有 reranker 時(rerank_depth=0)變深度 k 的替代信心指標:分數 >= top1 * 此比例的文件數。
// 官方禁止補滿固定深度,所以即使沒有 CE 也必須產生每題不同的 k。
vark_relative_frac:0.5,
// ⑧ 密集寫作(V3,自 的 team-stack-w4 移植)。單次生成、寫滿 900-1010 詞。
// 診斷:答案原本只寫 217-245 詞,官方上限 1024(只用 21%)。實測 V_strict 0.262 -> 0.408,
// 再換 gpt-5.6-sol 寫手 -> 0.414 / FS 96.4%,解掉「涵蓋高就引用差」的兩難。
// 注意:打開這個要同時把 per_aspect_generation 關掉 -- 兩者是⑧的不同候選,不疊加。
// 副作用:每題 LLM 呼叫從 ~95 掉到 ~5,這才是 Sol(每次呼叫都 shell 出 CLI,很慢)用得起的原因。
dense_writing:false,dense_evidence_docs:12,dense_evidence_chars:2500,
// ⑦⑧ 混合:先用 per-aspect 去「收集證據」,再讓寫手一次讀完全部寫一篇。
// 為什麼:V2 的涵蓋 0.4729 贏在證據量(per-aspect 每個面向各自 BM25,實測每題讀 88 篇),
// 不是贏在寫手;而密集寫作/V3 的引用品質贏在「一次讀完統一寫」。兩邊的優勢來源不同,
// 理論上可以疊 —— 這一格從來沒人測過(我們的檢索 × 單次寫完 × Sol)。
// per-aspect 那輪的答案直接丟掉,只留它讀進 readDocs 的文件(那輪走免費模型,不花 Sol)。
dense_after_aspect:false,
// ①⑤ Core Facet 多查詢建池 + splice 前段保護(V4)。
// facet 多查詢讓 R@5000 0.441 -> 0.532(topic-paired 22-0 全勝),但前段崩壞
//       (MAP@100 0.078 -> 0.048、nDCG@100 0.633 -> 0.47)。
// splice-200(前 200 照舊 + 後段接新池)讓 R@1000 +25%,nDCG@10 完全不變。
// 兩者互補:facet 負責「找得到更多」,splice 負責「不讓它弄壞已經排好的前段」。
// 多條線獨立收斂到同一原則(splice / tail-901 / quality gate / 深度實測資料)。
facet_queries:false,facet_span_autofix:false, /* ⚠️ 偏離原版:自動修正 source span 的字元位移。預設關閉(照 andy 原版:位移錯就整題失敗)。
// 開啟會連帶讓 decomposition_repair 不再被觸發,改變「一題拆幾個 Core」——見 facet_queries.ts 的註解。 */
facet_rewrites:4,facet_subaspects:10 /* 已停用:Core 數量由 andy 原版的 decomposition 自己決定(4-7 個/題,上限 20),不再由參數指定 */,facet_depth:1000,splice_head_keep:200,
// 面向分解的輸出上限。原本寫死 600,而 prompt 要 8-12 個面向、每個 3-10 個字 --
// 面向詞一長就會把 JSON 截斷在中間(實測:12 個面向 = 504 字元剛好爆掉,結尾的 ] 和 } 被切掉),
// extractObjectCandidate 找不到收尾的 } 就回 null,而 temperature=0 讓每次重試都截在同一處 -> 三次全掛。
// 這是一直存在的隱性 bug,只是面向詞短的時候剛好沒事(B2 那次拿到 9 個短面向就過了)。
aspect_decompose_max_tokens:1500,aspect_expected_facts:false,
// ⑦ 讀證據升級(V5)。
// 現行是每篇只取「前 document_read_limit 行」-- 關鍵證據若在文件深處就永遠拿不到。
// 開啟後改成:抓較長的全文 -> 用 find_in_document 依題目與面向定位最相關的幾段 -> 只把那幾段送進 prompt。
// adaptive_budget 則是依面向數決定讀幾篇(實測:<=3 面向 12 篇 / 4-6 面向 20 篇 / >=7 面向 30 篇),
// 取代現行固定 12 篇 -- 複雜題目本來就該多讀。
find_in_document:false,find_mode:"hybrid",find_max_passages:6,find_window_chars:1200,find_scan_limit:200000,
find_read_lines:2000,adaptive_budget:false,
// ⑦ 篇內定位有兩種實作:"lines" = annie/peiju 原版的逐行演算法(忠實移植,預設);
// "windows" = 本專案自建的字元視窗版。切旗標就能比,不必另開版本檔。
find_impl:"lines",find_context_before:5,find_context_after:8,
// ⑨ Evidence ledger(V6)。對每個 (句子, 被引用文件) 對回文件裡驗證支持度,
// 不過就改指到別的 reference,都找不到才拿掉該引用 -- 但**永不刪句**(刪句 = 刪掉它帶的 nugget)。
// 實測:weighted support 0.9402 / FS 89.6% / NS 2.4%,全隊引用最強。
// 排在 llm_revise 與 reattribute 之前,因為它是結構性約束,應該先過硬規則再讓 LLM 潤飾。
evidence_ledger:false,ledger_min_support:0.45,
// ⑦ 固定深度證據(Official Fixed-Depth Top-20)。>0 時,生成用的證據直接取
// 最終排序的前 K 篇,而不是 agentic 迴圈「剛好讀到」的那些 -- 讓證據與排序品質直接掛鉤。
// 實測:相對 capped-citation legacy,Citation Support +0.189、Unsupported -0.196。
// 這是 ⑦ 的第三種做法(另外兩種:的 12 篇 x 2500 字、find_in_document)。
fixed_topk_evidence:0,
// ⑨ 逐句驗證改寫。對每句比對它的引用證據:完全支持保留、過度宣稱改寫成證據支持的範圍、
// 完全沒支持時依 verify_mode 處理:
//   drop        刪句
//   weaken      改寫成較弱但有證據的說法(實測扣涵蓋 8-10pp,因為被改寫的句子正在涵蓋 nugget)
//   reattribute 一個字不改,只把引用改標到 references 裡真正支持它的那篇;整份都不支持才刪
// weaken 是為了保涵蓋 -- 在 recall 型指標下刪句 = 刪掉它帶的 nugget。
// 分批的理由:一次送 70 句會失敗,實測 20 句/批可行。
verify_revise:false,verify_mode:"weaken",verify_batch:20,verify_max_tokens:8192,
// 官方 v0.6.0 新規:一句有多個引用時,要按支持強度由高到低排序。
// 這裡用本地的 supportScore 算(不依賴外部服務),所以任何環境都能符合規定。
order_citations_by_strength:true,
// ⑦ SETR 式證據「集合」選擇。挑出 k 篇「合起來」涵蓋最多檢查表面向的文件,
// 而不是照排名取前 k 篇 -- 一篇補到還沒被涵蓋的面向,勝過第五篇講同一件事的。
// 這是 ⑦ 的第四種做法(另外三種:照排名取 k 篇、find_in_document 篇內選段、固定 Top-20)。
// 一次 LLM 呼叫;失敗就退回排名順序。
setr_select:false,setr_k:12,
// ⑤ S1 尾段重選。前 tail_reselect_head_keep 名逐字凍結,對 head+1..pool_depth 全部跑 CrossEncoder,
// 取 CE 前 fill 篇填進 head+1..head+fill。前段沒動 -> nDCG@10/@20/@100 數學上不可能變。
// 文字來源是 search() 回應裡本來就有、但先前被丟掉的 doc 欄位(平均 12k 字元的全文),
// 所以額外的文件抓取是 0 次 -- specs/S1.md 擔心的 10.5 萬次抓取不存在。
// 只在 tail_reselect 打開時才留文字,避免平白吃記憶體。
tail_reselect:false,tail_reselect_head_keep:200,tail_reselect_pool_depth:5000,tail_reselect_fill:800,tail_reselect_text_chars:2000,
// ⑧ S4 列舉式句型。把「一句一事實」換成「一句點名一個維度裡的每個成員」。
// 實測(另一線)涵蓋 25.6% -> 32.5%,但 hard 支持 89.5% -> 56.3% —— 階梯裡最大的單一退步。
// 賭注:evidence ledger 是逐字比對的結構性約束,理論上與句型無關,能把沒證據的成員的
// citation 拿掉而「不刪句」,於是拿到列舉式的涵蓋、不付它的支持度代價。
// 這兩個模組從未組合過。所以 enumerative_style 沒有 evidence_ledger 就直接拋錯,不准單獨開。
enumerative_style:false,enumerative_words_min:25,enumerative_words_max:40,
// ⑧ 句型的第三個候選(V2b):原子句 —— 一句一事實、一句一來源。
// 現行 per-aspect 的 prompt 是刻意調成「列舉式長句」的(一句塞多個 claim、最多引 3 篇),
// 實測結果:30 句 × 33 詞、每句 1.7 篇 → 涵蓋 0.4729 但 FS 只有 18.4%
// (沒有任何單一文件能支持一個混了兩篇內容的句子)。
// atomic 模式反過來:句子拆短、每句只引 1 篇,用句數而不是句長換廣度。
// ⚠️ 團隊在別的線上量過同一個取捨,方向是「原子句涵蓋較低但支持度高很多」,
//    所以這不是穩贏的改動,要用統一評分器實際量。互斥於 enumerative_style。
// ⑪ 強制覆蓋閘（integrated-v1 的 finalize_research 移植）。實作與理由見 generatePerAspectAnswer。
coverage_gate:false,coverage_gate_min_sentences:3,coverage_gate_max_passes:2,
// ⑫ 單一來源生成：一次只餵一篇文件，引用正確是 by construction。實作與理由見 answerOneAspect。
single_source:false,single_source_sentences:2,
atomic_sentences:false,atomic_max_words:18,atomic_per_aspect_sentences:6,
// ⑩ per-aspect 的句子生成預設走 base client(免費),因為它每題約 25 次呼叫 ——
// 全部走 Sol 的話 22 題要 550 次、好幾個小時。開這個旗標就改走 writer。
// ⚠️ 不是穩贏:已知證據顯示 Sol 寫得比較保守 —— 同一套密集寫作 prompt,
//    gpt-oss 涵蓋 0.408 vs Sol 0.395(原作者實測),而 Sol 的 FS 是 67% vs 97%。
//    也就是說換 Sol 大概率是「涵蓋略降、引用大升」,要量才知道淨值。
aspect_writer:false,
// ⑨ V6 Evidence Ledger v2 —— 自 annie/peiju 的 agentic-research 忠實移植(見 evidence_ledger_v2.ts)。
// 與 evidence_ledger(近似版,事後檢查)互斥:v2 是生成時強制 —— 模型必須交出逐字引文,
// claim 必須等於句子,違規就整份重生成(對應她的 fail() 讓 agent 重呼叫 finalize_research)。
// ledger_v2_policy: v1 = 每個 required 子問題至少 1 句;v2 = 解釋型子問題至少 2 句。
ledger_v2:false,ledger_v2_policy:"evidence-ledger-v2",ledger_v2_max_retries:3,ledger_v2_max_tokens:16384};
export type Policy=typeof BASE_POLICY;
export type PolicyOverride=Partial<Policy>;
export {BASE_POLICY};
// 每個階梯版本是 versions/ 底下自己的一支程式,啟動時把自己的 POLICY 傳進來(見 IterativeOptions.policy)。
// 這裡的 let 只在 runIterativeAgenticRag() 開頭被指定一次 -- 一個 process 只跑一個版本。
let POLICY:Policy=BASE_POLICY;
const CUTS=[10,20,50,100,500,1000],NDCG=[10,20,100,1000];
// Variable-k submission (Piika-aligned): per topic keep only the confident docs (ce_calibrated >= tau),
// clamped to [min,max], instead of padding to a fixed cutoff. Reads the persisted fusion_scores.json.
function writeVariableKSubmission(out:string,topics:{qid:string}[],runId:string,tau:number,minK:number,maxK:number){
  const lines:string[]=[]; const counts:Record<string,number>={}; const modes:Record<string,string>={};
  for(const t of topics){
    const p=join(out,"topics",`${t.qid}.fusion_scores.json`);
    const fsj=existsSync(p)?(readIf(p) as any[]):null;
    const hasCe=Array.isArray(fsj)&&fsj.length>0&&fsj.some(x=>typeof x.ce_calibrated==="number");
    let ranked:{docid:string;score:number}[]=[]; let confident=0; let mode="";
    if(hasCe){
      // 現行路徑:CE 校準分數當信心指標
      confident=(fsj as any[]).filter(x=>typeof x.ce_calibrated==="number"&&x.ce_calibrated>=tau).length;
      ranked=(fsj as any[]).map(x=>({docid:String(x.docid),score:Number(x.fused)||0})); mode="ce_calibrated";
    } else {
      // 無 reranker 時(rerank_depth=0,例如 V0 基準版)沒有 ce_calibrated 可用。
      // 官方仍要求變深度 per-narrative k,不得補滿固定深度,所以改用相對分數門檻:
      // 取分數 >= top1 * vark_relative_frac 的文件數當 k。
      const rp=join(out,"topics",`${t.qid}.runfile.trec`); if(!existsSync(rp))continue;
      ranked=readFileSync(rp,"utf8").split(/\r?\n/).filter(Boolean).map(line=>{const[,,docid,,score]=line.split(/\s+/);return{docid,score:Number(score)||0}});
      if(ranked.length===0)continue;
      const top=ranked[0].score; const cut=top*POLICY.vark_relative_frac;
      confident=ranked.filter(x=>x.score>=cut).length; mode="relative_score";
    }
    let k=confident; if(k<minK)k=minK; if(k>maxK)k=maxK; if(k>ranked.length)k=ranked.length;
    counts[t.qid]=k; modes[t.qid]=mode;
    ranked.slice(0,k).forEach((x,i)=>lines.push(`${t.qid} Q0 ${x.docid} ${i+1} ${x.score.toFixed(8)} ${runId}`));
  }
  writeFileSync(join(out,"submission_variable_k.trec"),lines.join("\n")+(lines.length?"\n":""));
  writeJson(join(out,"variable_k_counts.json"),{k_by_topic:counts,mode_by_topic:modes});
}
async function rerankTopOfRanking(o:IterativeOptions,topic:Topic,ranking:{docid:string;rank:number;score:number}[],readDocs:Map<string,ReadDoc>,env:NodeJS.ProcessEnv,depth:number,out:string){
  const head=ranking.slice(0,depth), tail=ranking.slice(depth);
  const candidates:RerankCandidate[]=[]; const unfetched:{docid:string;score:number}[]=[];
  for(const entry of head){
    const cached=readDocs.get(entry.docid);
    if(cached){candidates.push({docid:entry.docid,text:cached.text});continue;}
    try{const d=await readDoc(o,entry.docid,o.documentReadLimit,env); if(d.found)candidates.push({docid:entry.docid,text:d.text}); else unfetched.push(entry); await sleep(150);}
    catch{unfetched.push(entry);}
  }
  if(candidates.length===0){writeJson(join(out,"topics",`${topic.qid}.fusion_scores.json`),[]);return ranking;}
  // BM25 rank = order candidates were taken from the fused ranking.
  const bm25Rank=new Map<string,number>(); candidates.forEach((c,i)=>bm25Rank.set(c.docid,i));
  // CrossEncoder signal.
  const reranked=await rerankWithCrossEncoder(topic.narrative,candidates);
  const ceRank=new Map<string,number>(); reranked.forEach((r,i)=>ceRank.set(r.docid,i));
  const ceCal=new Map<string,number>(); reranked.forEach(r=>ceCal.set(r.docid,r.calibratedScore));
  // Dense signal (optional, best-effort — if the embeddings API fails, fall back to BM25+CE only).
  let denseRank=new Map<string,number>(); let denseCos=new Map<string,number>();
  if(POLICY.fusion_dense){ try{ denseCos=await denseScores(topic.narrative,candidates,env); const sorted=[...candidates].map(c=>c.docid).sort((a,b)=>(denseCos.get(b)??0)-(denseCos.get(a)??0)); sorted.forEach((d,i)=>denseRank.set(d,i)); }catch{ denseRank=new Map(); } }
  const K=POLICY.fusion_rrf_k, useDense=denseRank.size>0;
  // CE 失效偵測(見 POLICY.ce_dead_threshold 的說明)。RRF 的設計會懲罰「只有一軌看好」的文件,
  // 那在 CE 有鑑別力時是對的;但 CE 整題都給不出高分時,它的名次只是雜訊,不該拿來否決 BM25。
  const maxCe=ceCal.size>0?Math.max(...ceCal.values()):Infinity;
  const ceDead=POLICY.ce_dead_threshold>0&&ceCal.size>0&&maxCe<POLICY.ce_dead_threshold;
  const fused=candidates.map(c=>{ const rb=bm25Rank.get(c.docid)??candidates.length, rc=ceRank.get(c.docid)??candidates.length, rd=denseRank.get(c.docid)??candidates.length;
    // 退回模式:分數單調對應 BM25 名次,讓下游(含 variable-k)看到的順序就是 BM25 順序。
    if(ceDead)return {docid:c.docid,fused:1/(K+rb+1)};
    let s=POLICY.fusion_bm25_weight/(K+rb+1)+POLICY.fusion_ce_weight/(K+rc+1); if(useDense)s+=POLICY.fusion_dense_weight/(K+rd+1);
    return {docid:c.docid,fused:s}; }).sort((a,b)=>b.fused-a.fused);
  if(ceDead)console.error(`  ${topic.qid}: CE dead (max ce_calibrated ${maxCe.toFixed(3)} < ${POLICY.ce_dead_threshold}) -> BM25 fallback`);
  writeJson(join(out,"topics",`${topic.qid}.fusion_scores.json`),fused.map(f=>({docid:f.docid,fused:f.fused,bm25_rank:bm25Rank.get(f.docid),ce_rank:ceRank.get(f.docid),dense_rank:useDense?denseRank.get(f.docid):null,ce_calibrated:ceCal.get(f.docid)??null,dense_cosine:useDense?(denseCos.get(f.docid)??null):null,ce_dead:ceDead})));
  const newHead=[...fused.map(f=>({docid:f.docid,score:f.fused})),...unfetched];
  return[...newHead,...tail].map((e,i)=>({docid:e.docid,rank:i+1,score:e.score}));
}
export async function runIterativeAgenticRag(o:IterativeOptions){POLICY={...BASE_POLICY,...(o.policy??{})};
// S4 的硬性前置:列舉式長句沒有 ledger 保護就是重演 hard 支持 −33.2pp 的那次失敗。寧可開不起來也不要跑出沒用的結果。
if(POLICY.enumerative_style&&!POLICY.evidence_ledger&&!POLICY.ledger_v2)throw new Error("enumerative_style 需要 evidence_ledger 或 ledger_v2 一起開(見 docs/specs/S4.md)");
// ledger 的兩種實作互斥:v2 產出的草稿已經逐字驗證過,再讓近似版事後改指只會把它弄髒。
if(POLICY.ledger_v2&&POLICY.evidence_ledger)throw new Error("ledger_v2 與 evidence_ledger(近似版)不可同時開啟");
if(POLICY.atomic_sentences&&POLICY.enumerative_style)throw new Error("atomic_sentences 與 enumerative_style 是 ⑧ 的兩個相反候選,不可同時開啟");
if(POLICY.dense_after_aspect&&!POLICY.dense_writing)throw new Error("dense_after_aspect 需要 dense_writing 一起開 —— 它只負責收證據,寫作是密集寫作那條路徑做的");
if(POLICY.dense_after_aspect&&POLICY.per_aspect_generation)throw new Error("dense_after_aspect 與 per_aspect_generation 不可同時開啟 —— 前者已經會跑一次 per-aspect 收證據,再開會跑兩次"); const env=o.env??process.env,out=resolve(o.outputDir); if(o.force)rmSync(out,{recursive:true,force:true}); mkdirSync(join(out,"topics"),{recursive:true}); const llmCfg=normalizeLlmClientConfig(o.llm),llm=createLlmClient(llmCfg,env);
const writerCfg=o.llmWriter?normalizeLlmClientConfig(o.llmWriter):llmCfg, queryCfg=o.llmQuery?normalizeLlmClientConfig(o.llmQuery):llmCfg;
const llms:LlmSet={base:llm,writer:o.llmWriter?createLlmClient(writerCfg,env):llm,query:o.llmQuery?createLlmClient(queryCfg,env):llm}; const cfg:AgenticRagBaselineConfig={runId:o.runId,teamId:o.teamId,mode:"dev",promptVersion:AGENTIC_RAG_BASELINE_PROMPT_VERSION}; const topics=parseTrecRag2026TopicsTsv(readFileSync(resolve(o.topicsPath),"utf8")).map(t=>({qid:t.topicId,title:"",narrative:t.narrative})).slice(0,o.limitTopics??Infinity);
writeJson(join(out,"config.json"),{run_id:o.runId,team_id:o.teamId,policy:POLICY,initial_docs:o.initialDocs,docs_per_iteration:o.docsPerIteration,max_documents_read:o.maxDocumentsRead,max_iterations:o.maxIterations,document_read_limit:o.documentReadLimit,llm:safeLlmConfigForArtifacts(llmCfg),llm_writer:safeLlmConfigForArtifacts(writerCfg),llm_query:safeLlmConfigForArtifacts(queryCfg)});
const summary={run_id:o.runId,selected_topics:topics.length,processed_count:0,failed_count:0,iterations_by_topic:{} as Record<string,number>,stop_reason_by_topic:{} as Record<string,string>,llm_call_count:0,llm_failed_call_count:0,llm_retry_count:0,judge_fallback_count:0,aspect_decomposition_failures:0,average_documents_read_successful:0};
for(const [idx,t] of topics.entries()){const sp=join(out,"topics",`${t.qid}.status.json`); if(o.resume&&existsSync(sp)&&readIf(sp)?.status==="completed"){console.error(`skip ${t.qid}`);continue;} try{const r=await processTopic({topic:t,o,llm,llms,cfg,out,env,summary}); writeJson(join(out,"topics",`${t.qid}.iteration_trace.json`),r.iterationTrace); writeJson(join(out,"topics",`${t.qid}.judge_trace.json`),r.judgeTrace); writeJson(join(out,"topics",`${t.qid}.retrieval-trace.json`),r.retrievalTrace); writeJson(join(out,"topics",`${t.qid}.rag-draft.json`),r.ragObject); writeJson(join(out,"topics",`${t.qid}.validation.json`),r.validation); // 池完整性閘門（POOL_QUALITY_GATE=1 才啟用）：搜尋靜默失敗會讓池縮水但題目照樣
 // 標 completed（官方跑 rag2026-19/27 實例：3784/3398 列）。缺池就標 failed 讓 resume 重做。
 if(process.env.POOL_QUALITY_GATE==="1"&&r.ranking.length<POLICY.output_depth)
   throw new Error(`TRUNCATED_POOL ${r.ranking.length}/${POLICY.output_depth}`);
 writeFileSync(join(out,"topics",`${t.qid}.runfile.trec`),topicRun(t.qid,r.ranking,o.runId)); writeJson(join(out,"topics",`${t.qid}.final_read_docs_trace.json`),r.readTrace); writeJson(sp,{topic_id:t.qid,status:"completed",stop_reason:r.stopReason,iterations:r.iterations}); summary.processed_count++; summary.iterations_by_topic[t.qid]=r.iterations; summary.stop_reason_by_topic[t.qid]=r.stopReason; console.error(`${idx+1}/${topics.length} ${t.qid} iter=${r.iterations} stop=${r.stopReason} read=${r.readTrace.documents_read_successful}`);}catch(e){summary.failed_count++; const issues=e instanceof AgenticRagValidationError?e.issues:undefined; writeJson(sp,{topic_id:t.qid,status:"failed",error:redact(e instanceof Error?e.message:String(e),env),...(issues?{issues}:{})}); console.error(`${idx+1}/${topics.length} ${t.qid} FAILED${issues?" "+issues.map(i=>i.code).join(","):""}`);} await sleep(500);}
const rankings=assemble(out,topics), runfile=render(rankings,topics.map(t=>t.qid),o.runId); writeFileSync(join(out,"candidate_pool_top5000.trec"),runfile); writeFileSync(join(out,"retrieval.internal.trec-run.tsv"),runfile); writeVariableKSubmission(out,topics,o.runId,POLICY.vark_threshold,POLICY.vark_min,POLICY.vark_max); const completed=topics.filter(t=>readIf(join(out,"topics",`${t.qid}.status.json`))?.status==="completed"); const rags=completed.map(t=>readIf(join(out,"topics",`${t.qid}.rag-draft.json`))); writeFileSync(join(out,"rag_output_trec_rag_2026.jsonl"),rags.map(x=>JSON.stringify(x)).join("\n")+(rags.length?"\n":"")); writeJsonl(join(out,"iteration_trace.jsonl"),completed.map(t=>readIf(join(out,"topics",`${t.qid}.iteration_trace.json`))).flat()); const readTraces=completed.map(t=>readIf(join(out,"topics",`${t.qid}.final_read_docs_trace.json`))); writeJsonl(join(out,"final_read_docs_trace.jsonl"),readTraces); writeJsonl(join(out,"retrieval_trace.jsonl"),completed.map(t=>readIf(join(out,"topics",`${t.qid}.retrieval-trace.json`)))); const failed=topics.map(t=>readIf(join(out,"topics",`${t.qid}.status.json`))).filter((s:any)=>s?.status==="failed").map((s:any)=>({topic_id:s.topic_id,error:s.error})); writeJson(join(out,"failed_topics.json"),failed); const metrics=evalAll(qrelsPaths(resolve(o.qrelsDir)),topics.map(t=>t.qid),rankings); writeJson(join(out,"metrics.json"),metrics.summary); writeJson(join(out,"per_topic_metrics.json"),metrics.perTopic); const validation={ok:failed.length===0&&rags.length===topics.length,output_count:rags.length,expected_count:topics.length}; writeJson(join(out,"validation.json"),validation); summary.failed_count=failed.length; summary.average_documents_read_successful=readTraces.reduce((s:any,x:any)=>s+(x?.documents_read_successful??0),0)/(readTraces.length||1); writeJson(join(out,"run-summary.internal.json"),summary); writeFileSync(join(out,"provenance.md"),`# Iterative agentic RAG\n\nNo RAGDoll. No qrels query selection. No source_run_dir.\n`); return{outputDir:out,validation};}
// V4:Core Facet 多查詢建池 + splice。回傳「前段凍結、後段換成新池」的排序。
// 失敗(查詢生成失敗/全被 checker 濾掉/搜尋失敗)一律原封退回,不影響 V3 的行為。
async function spliceFacetPool(a:{topic:Topic;o:IterativeOptions;llms:LlmSet;out:string;env:NodeJS.ProcessEnv},ranking:{docid:string;rank:number;score:number}[],poolText?:Map<string,string>){
  const keep=Math.min(POLICY.splice_head_keep,ranking.length);
  try{
    const fq=await generateFacetQueries(a.llms.query,a.topic.narrative,{spanAutofix:process.env.FACET_SPAN_AUTOFIX==="1"||POLICY.facet_span_autofix});
    writeJson(join(a.out,"topics",`${a.topic.qid}.facet_queries.json`),{topic_id:a.topic.qid,ok:fq.ok,rewrites:fq.rewrites,facets:fq.facets,cores:fq.cores,background_hints:fq.backgroundHints,repairs:fq.repairs,failure_reasons:fq.failureReasons});
    if(!fq.ok)return ranking;
    const queries=[...fq.rewrites,...fq.facets];
    const runs:Hit[][]=[];
    for(const q of queries){ try{runs.push(await search(a.o,q,POLICY.facet_depth,a.env,poolText));}catch{} await sleep(250); }
    if(runs.length===0)return ranking;
    // plain RRF(不加權,因為這些查詢彼此地位相同;原題的優勢已經由前段凍結保障)
    const facetPool=weightedRrf(runs,runs.map(()=>1),POLICY.output_depth,POLICY.rrf_k);
    const head=ranking.slice(0,keep), headSet=new Set(head.map(e=>e.docid));
    const tail=facetPool.filter(e=>!headSet.has(e.docid)).slice(0,Math.max(0,POLICY.output_depth-keep));
    // facet 池比要替換的尾段小（core 少的題，如官方 rag2026-25 只拆出 2 core
    // → facet 池 2212 → 池縮成 2234）時，用原始尾段回填補滿 —— facet 文件仍
    // 排前面，語意是「facet 優先，不足才回退原排序」。dev22 每題都有 7 core，
    // 池夠大，從未觸發；官方題觸發了 5+ 題（全被池閘門攔下）。
    const tailSet=new Set(tail.map(e=>e.docid));
    const backfill=ranking.slice(keep)
      .filter(e=>!headSet.has(e.docid)&&!tailSet.has(e.docid))
      .slice(0,Math.max(0,POLICY.output_depth-keep-tail.length));
    const spliced=[...head,...tail.map(e=>({docid:e.docid,rank:0,score:e.score})),...backfill]
      .map((e,i)=>({docid:e.docid,rank:i+1,score:e.score}));
    writeJson(join(a.out,"topics",`${a.topic.qid}.splice.json`),{topic_id:a.topic.qid,head_keep:keep,facet_queries:queries.length,facet_pool_size:facetPool.length,before_size:ranking.length,after_size:spliced.length,new_in_tail:tail.length});
    return spliced;
  }catch{return ranking;}
}
// S4 驗收條件之三是「平均句長落在 25–40 詞」。順手記進 gen_trace,跑完不用再另外算一次。
// 非 S4 的版本也會記 —— 這樣才有 baseline 可比(V3 的密集寫作是 ≤35 詞的原子句)。
function sentenceStyleStats(d:{answer:{text:string}[]}){
  const w=d.answer.map(s=>String(s.text??"").trim().split(/\s+/).filter(Boolean).length);
  if(w.length===0)return{sentences:0};
  const sorted=[...w].sort((x,y)=>x-y), total=w.reduce((s,x)=>s+x,0);
  const lo=POLICY.enumerative_words_min, hi=POLICY.enumerative_words_max, inRange=w.filter(x=>x>=lo&&x<=hi).length;
  return{sentences:w.length,words_total:total,words_mean:Number((total/w.length).toFixed(1)),words_median:sorted[Math.floor(sorted.length/2)],words_min:sorted[0],words_max:sorted[sorted.length-1],in_target_range:inRange,in_target_range_pct:Number((100*inRange/w.length).toFixed(1)),target_range:[lo,hi],enumerative:POLICY.enumerative_style};
}
// ⑤ S1 尾段重選(側翼)。要攻擊的是 recall funnel 上最大的單點損失:
// internal pool R@5000 = 0.4752,砍到 1000 只剩 0.2972(−0.178)。
// rank 201–1000 目前只是 RRF 分數排序 -- 沒有任何模型真的看過 1001–5000 那 4000 篇。
//
// 前 head_keep 名逐字凍結(與 V4 的 splice 保護一致),所以 nDCG@10/@20/@100 數學上不可能變 --
// 這是驗收條件,不是希望。團隊 W2/W3 那次「深池重排回 top-100 位移 74.73% 相關候選」的失敗
// 動的是頭部;S1 動不到頭部。
//
// 分數欄位:重選後 rank 順序與 score 順序必須一致,否則 trec_eval 會照 score 重排、把重選結果洗掉。
// 前段保留原分數,後段改成從前段最後一名往下遞減的合成分數。
// 任何一步失敗都原封退回上一版的排序。
async function reselectTail(a:{topic:Topic;out:string},ranking:{docid:string;rank:number;score:number}[],poolText?:Map<string,string>){
  const keep=Math.min(POLICY.tail_reselect_head_keep,ranking.length);
  if(!poolText||poolText.size===0||ranking.length<=keep)return ranking;
  try{
    const head=ranking.slice(0,keep), tail=ranking.slice(keep,POLICY.tail_reselect_pool_depth), rest=ranking.slice(POLICY.tail_reselect_pool_depth);
    const cands:RerankCandidate[]=[]; let noText=0;
    for(const e of tail){const t=poolText.get(e.docid); if(t)cands.push({docid:e.docid,text:t}); else noText++;}
    if(cands.length===0)return ranking;
    const scored=await rerankWithCrossEncoder(a.topic.narrative,cands);
    const byDoc=new Map(tail.map(e=>[e.docid,e]));
    const picked=scored.slice(0,POLICY.tail_reselect_fill).map(s=>byDoc.get(s.docid)).filter((e):e is typeof tail[number]=>Boolean(e));
    const pickedSet=new Set(picked.map(e=>e.docid));
    const merged=[...head,...picked,...tail.filter(e=>!pickedSet.has(e.docid)),...rest];
    const base=head.length?head[head.length-1].score:1;
    const out=merged.map((e,i)=>({docid:e.docid,rank:i+1,score:i<keep?e.score:base-(i-keep+1)*1e-6}));
    writeJson(join(a.out,"topics",`${a.topic.qid}.tail_reselect.json`),{topic_id:a.topic.qid,head_keep:keep,tail_size:tail.length,ce_scored:cands.length,no_text:noText,filled:picked.length,pool_text_size:poolText.size,head_unchanged:out.slice(0,keep).every((e,i)=>e.docid===head[i].docid),after_size:out.length});
    return out;
  }catch{return ranking;}
}
async function processTopic(a:{topic:Topic;o:IterativeOptions;llm:LlmClient;llms:LlmSet;cfg:AgenticRagBaselineConfig;out:string;env:NodeJS.ProcessEnv;summary:any}){const queries=[a.topic.narrative], runs:Hit[][]=[], weights:number[]=[]; const poolText=POLICY.tail_reselect?new Map<string,string>():undefined; const anchor=await search(a.o,a.topic.narrative,5000,a.env,poolText); runs.push(anchor); weights.push(POLICY.bm25_anchor_weight); const readDocs=new Map<string,ReadDoc>(), failedRead:string[]=[]; const iterationTrace:any[]=[], judgeTrace:any[]=[];
// Query2Doc: one LLM call generates a pseudo-answer passage; its BM25 run is fused (weighted RRF) with the protected raw-narrative anchor. Additive recall signal, everything downstream unchanged.
if(POLICY.q2d_enabled){const q2d=await generateQuery2Doc(a.llms.query,a.topic.narrative,{queryRepeat:POLICY.q2d_query_repeat,maxPseudoDocWords:180,maxTokens:400}); writeJson(join(a.out,"topics",`${a.topic.qid}.query2doc.json`),{topic_id:a.topic.qid,ok:q2d.ok,query_repeat:POLICY.q2d_query_repeat,q2d_weight:POLICY.q2d_weight,pseudo_doc:q2d.pseudoDoc,expanded_query:q2d.expandedQuery}); if(q2d.ok){const q2dRun=await search(a.o,q2d.expandedQuery,5000,a.env,poolText); runs.push(q2dRun); weights.push(POLICY.q2d_weight); queries.push(`[query2doc] ${q2d.expandedQuery.slice(0,200)}`); await sleep(250);}}
let ranking=weightedRrf(runs,weights,POLICY.output_depth,POLICY.rrf_k), stopReason="max_iterations", iterations=0;
// ⑦ V5:面向要在「讀文件之前」就拿到 -- 它同時是段落定位的查詢、也是自適應預算的依據。
// (原本 decomposeAspects 是在 answer() 裡才呼叫,那時文件早就讀完了。)
let earlyAspects:string[]=[]; let budget={maxDocs:a.o.maxDocumentsRead,maxIterations:a.o.maxIterations};
if(POLICY.find_in_document||POLICY.adaptive_budget){
  earlyAspects=await decomposeAspects({topic:a.topic,llm:a.llms.base,out:a.out,env:a.env,summary:a.summary});
  if(POLICY.adaptive_budget&&earlyAspects.length>0)budget=adaptiveBudget(earlyAspects.length);
}
const findQueries=POLICY.find_in_document?[a.topic.narrative,...earlyAspects]:undefined;
await readNew({o:a.o,hits:ranking,readDocs,failedRead,count:a.o.initialDocs,query:a.topic.narrative,env:a.env,maxDocs:budget.maxDocs,findQueries});
const writePartial=()=>writeTopicPartial({out:a.out,topic:a.topic,iterationTrace,judgeTrace,queries,runs,ranking,readDocs,failedRead});
for(let it=0;it<budget.maxIterations;it++){iterations=it+1; let judge:Judge; try{judge=await judgeEvidence({topic:a.topic,llm:a.llm,readDocs,queries,out:a.out,env:a.env,summary:a.summary,iteration:it}); judgeTrace.push({iteration:it,judge});}catch(e){a.summary.judge_fallback_count++; const err=redact(e instanceof Error?e.message:String(e),a.env); const code=classifyJudgeStopReason(err); stopReason=code; judgeTrace.push({iteration:it,error_code:code,error:err}); iterationTrace.push({topic_id:a.topic.qid,iteration:it,queries:[...queries],documents_read_successful:readDocs.size,judge_error:{error_code:code,message:err},stop_reason:stopReason,ranking_top10:ranking.slice(0,10)}); writePartial(); break;} let iterationStopReason="continue"; if(judge.enough){stopReason="enough"; iterationStopReason=stopReason; iterationTrace.push({topic_id:a.topic.qid,iteration:it,queries:[...queries],documents_read_successful:readDocs.size,judge,stop_reason:iterationStopReason,ranking_top10:ranking.slice(0,10)}); writePartial(); break;} if(readDocs.size>=budget.maxDocs){stopReason="max_documents_read"; iterationStopReason=stopReason; iterationTrace.push({topic_id:a.topic.qid,iteration:it,queries:[...queries],documents_read_successful:readDocs.size,judge,stop_reason:iterationStopReason,ranking_top10:ranking.slice(0,10)}); writePartial(); break;} if(it===budget.maxIterations-1){stopReason="max_iterations"; iterationStopReason=stopReason; iterationTrace.push({topic_id:a.topic.qid,iteration:it,queries:[...queries],documents_read_successful:readDocs.size,judge,stop_reason:iterationStopReason,ranking_top10:ranking.slice(0,10)}); writePartial(); break;} const fqs=validateFollowups(judge.followup_queries,queries,a.topic.narrative); if(fqs.length===0){stopReason="no_valid_followup_query"; iterationStopReason=stopReason; iterationTrace.push({topic_id:a.topic.qid,iteration:it,queries:[...queries],documents_read_successful:readDocs.size,judge,stop_reason:iterationStopReason,ranking_top10:ranking.slice(0,10)}); writePartial(); break;} iterationTrace.push({topic_id:a.topic.qid,iteration:it,queries:[...queries],documents_read_successful:readDocs.size,judge,followup_queries:fqs,stop_reason:iterationStopReason,ranking_top10:ranking.slice(0,10)}); for(const q of fqs){queries.push(q); const h=await search(a.o,q,5000,a.env,poolText); runs.push(h); weights.push(POLICY.followup_query_weight); await sleep(250);} ranking=weightedRrf(runs,weights,POLICY.output_depth,POLICY.rrf_k); await readNew({o:a.o,hits:ranking,readDocs,failedRead,count:a.o.docsPerIteration,query:"weighted_rrf_fused",env:a.env,maxDocs:budget.maxDocs,findQueries}); writePartial(); if(readDocs.size>=budget.maxDocs&&it+1>=budget.maxIterations)stopReason="max_iterations";}
if(stopReason==="max_iterations"&&readDocs.size>=budget.maxDocs)stopReason="max_documents_read"; if(POLICY.rerank_depth>0)ranking=await rerankTopOfRanking(a.o,a.topic,ranking,readDocs,a.env,POLICY.rerank_depth,a.out);
// ①⑤ V4:用 Core Facet 多查詢另建一個池,只拿來補後段。前 splice_head_keep 名逐字凍結,
// 所以 nDCG@10/@20/@100 依設計完全不變 -- 這是「保護」的定義,也是驗收條件。
if(POLICY.facet_queries){ranking=await spliceFacetPool(a,ranking,poolText);}
// ⑤ S1 尾段重選:CE 真的看過 201–5000 之後才決定誰進 201–1000(原本那段只是 RRF 分數排序)。
if(POLICY.tail_reselect){ranking=await reselectTail(a,ranking,poolText);}
// ⑦ 固定深度證據:用最終排序的前 K 篇當證據,取代 agentic 迴圈讀到的那些。
if(POLICY.fixed_topk_evidence>0){
  const want=ranking.slice(0,POLICY.fixed_topk_evidence);
  readDocs.clear();
  await readNew({o:a.o,hits:want.map(e=>({docid:e.docid,score:e.score})),readDocs,failedRead,
                 count:POLICY.fixed_topk_evidence,query:"fixed_topk_evidence",env:a.env,
                 maxDocs:POLICY.fixed_topk_evidence,findQueries});
}
const ragObject=await answer({topic:a.topic,o:a.o,cfg:a.cfg,llm:a.llm,llms:a.llms,readDocs,out:a.out,env:a.env,summary:a.summary,presetAspects:earlyAspects.length>0?earlyAspects:undefined,budget}); const validation=validateRagOutputObjectStrict(ragObject,{config:a.cfg,topic:a.topic,readDocids:new Set(readDocs.keys())}); if(!validation.ok)throw new Error(`RAG validation failed: ${validation.issues.map(i=>i.code).join(",")}`); const cited=[...new Set(ragObject.answer.flatMap(s=>s.citations).map(i=>ragObject.references[i]).filter(Boolean))]; return{ranking,ragObject,validation,iterationTrace,judgeTrace,iterations,stopReason,readTrace:{topic_id:a.topic.qid,candidate_pool_size:ranking.length,documents_read_attempted:readDocs.size+failedRead.length,documents_read_successful:readDocs.size,read_docids:[...readDocs.keys()],failed_read_docids:failedRead,final_answer_cited_docids:cited},retrievalTrace:buildRetrievalTrace(a.topic,queries,runs,ranking)};}
async function judgeEvidence(a:{topic:TopicIdentity;llm:LlmClient;readDocs:Map<string,ReadDoc>;queries:string[];out:string;env:NodeJS.ProcessEnv;summary:any;iteration:number}){const prompt=["Return ONLY one JSON object. No markdown. No explanation.",'Required schema: {"enough":boolean,"queries":string[]}',"If the evidence already supports the topic, return {\"enough\":true,\"queries\":[]}.","If evidence is insufficient, return {\"enough\":false,\"queries\":[...]} with 1-3 BM25 keyword queries.","Query rules: 4-12 English tokens; keyword phrase, not sentence/question; one aspect only; no duplicates.","Forbidden words: obtain, find, source, sources, detail, detailed, comprehensive, concrete, examples, overview, history, impact, provide, explain.",`Previous queries: ${JSON.stringify(a.queries)}`,`Topic: ${a.topic.narrative}`,"Evidence:",[...a.readDocs.values()].map((d,i)=>`[${i}] ${d.docid}\n${d.text.slice(0,1200)}`).join("\n\n")].join("\n"); let lastError="LLM_JSON_PARSE_FAILED"; for(let attempt=1;attempt<=5;attempt++){const started=Date.now(); try{const result=await a.llm.generate({messages:[{role:"user",content:prompt}],temperature:0,maxTokens:2000,responseFormat:"json_object"}); const parsed=parseJudgeResponse(result.text,a.queries,a.topic.narrative); recordAttempt({attempt:{attempt,provider:result.provider,model:result.model,latencyMs:result.latencyMs,success:parsed.ok,outputChars:result.text.length,...(parsed.ok?{}:{errorCode:"LLM_JSON_PARSE_FAILED"}),...(result.requestId?{requestId:result.requestId}:{}),...(result.usage?{usage:result.usage}:{})},stage:"judge",qid:a.topic.qid,out:a.out,env:a.env,summary:a.summary}); if(parsed.ok)return parsed.value; lastError=parsed.message; }catch(e){lastError=redact(e instanceof Error?e.message:String(e),a.env); recordAttempt({attempt:{attempt,provider:a.llm.provider,model:a.llm.model,latencyMs:Date.now()-started,success:false,errorCode:/empty assistant message/i.test(lastError)?"LLM_EMPTY_ASSISTANT_MESSAGE":/429/.test(lastError)?"LLM_RATE_LIMIT":/5\\d\\d/.test(lastError)?"LLM_SERVER_ERROR":"LLM_PROVIDER_FAILED",outputChars:0},stage:"judge",qid:a.topic.qid,out:a.out,env:a.env,summary:a.summary});} if(attempt<5)await sleep(300*2**attempt);} throw new Error(lastError||"LLM_JSON_PARSE_FAILED");}
function validateJudge(v:unknown,prev:string[],anchor:string):LlmJsonValidationResult<Judge>{if(!isRecord(v)||typeof v.enough!=="boolean")return{ok:false,message:"judge shape"}; const rawQueries=Array.isArray((v as any).queries)?(v as any).queries:Array.isArray((v as any).followup_queries)?(v as any).followup_queries:[]; const missing=Array.isArray((v as any).missing_aspects)?(v as any).missing_aspects.map(String):[]; const legal=validateFollowups(rawQueries,prev,anchor); return{ok:true,value:{enough:Boolean((v as any).enough),missing_aspects:missing,followup_queries:(v as any).enough?[]:legal}};}
function parseJudgeResponse(text:string,prev:string[],anchor:string):LlmJsonValidationResult<Judge>{const json=extractObjectCandidate(text); if(json){try{const parsed=JSON.parse(json); return validateJudge(parsed,prev,anchor);}catch{}} const lines=text.split(/\r?\n|,/).map(x=>x.replace(/^[-*\d.\s\"']+/,"").replace(/[\"']+$/,"").trim()).filter(Boolean); const legal=validateFollowups(lines,prev,anchor); if(legal.length>0)return{ok:true,value:{enough:false,missing_aspects:[],followup_queries:legal}}; if(/\benough\b\s*[:=]\s*true|\bsufficient\b/i.test(text))return{ok:true,value:{enough:true,missing_aspects:[],followup_queries:[]}}; return{ok:false,message:"LLM_JSON_PARSE_FAILED"};}
function extractObjectCandidate(text:string){const fenced=/```(?:json)?\s*([\s\S]*?)```/i.exec(text); const t=(fenced?fenced[1]:text).trim(); if(t.startsWith("{")&&t.endsWith("}"))return t; const a=t.indexOf("{"), b=t.lastIndexOf("}"); return a>=0&&b>a?t.slice(a,b+1):null;}
const BAD=new Set("obtain find source sources detail detailed comprehensive concrete examples evidence information overview history impact provide directly addresses address account explain and".split(" ")); function toks(q:string){return q.toLowerCase().match(/[a-z0-9-]+/g)??[]} function validateFollowups(qs:unknown[],prev:string[],anchor:string){const prevNorm=new Set(prev.map(q=>toks(q).join(" "))); const anchorSet=new Set(toks(anchor)); const out:string[]=[]; for(const raw of qs.map(String)){const q=raw.trim().replace(/\s+/g," "); const t=toks(q), norm=t.join(" "); if(t.length<4||t.length>12||/[?{}.;:]|```/.test(q)||t.some(x=>BAD.has(x))||prevNorm.has(norm)||out.some(x=>toks(x).join(" ")===norm))continue; const overlap=t.filter(x=>anchorSet.has(x)).length/Math.max(1,new Set([...t,...anchorSet]).size); if(overlap>0.75)continue; out.push(q); if(out.length>=3)break;} return out;}
// v4-port per-aspect answer: for each aspect run its OWN BM25 search + reads, generate a few grounded
// sentences from that aspect's targeted evidence, then merge all aspects into one cited answer. This is
// the mechanism (targeted per-aspect evidence) that lets v4 reach high nugget coverage. It only feeds
// generation — the R-task ranking is unchanged (approach ①).
// LLMs often echo their local citation index inline (e.g. "... revenue [2]."). After merging per-aspect
// sub-answers those inline markers are stale (the structured `citations` field is remapped, the text is not),
// so strip them — the citations array is the single source of truth.
function stripInlineCitations(text:string):string{return text.replace(/\s*\[\d+(?:\s*,\s*\d+)*\]/g,"").replace(/\s+([.,;:])/g,"$1").replace(/\s{2,}/g," ").trim();}
// Answer ONE aspect with up to POLICY.aspect_rounds rounds: each round reads a deeper batch of that
// aspect's ranking and writes NEW sentences (told not to repeat prior ones), accumulating evidence —
// this is v4's "dig deeper per sub-question" behaviour. Mutates docidText/shared with docs it reads.
async function answerOneAspect(a:{topic:TopicIdentity;o:IterativeOptions;llm:LlmClient;out:string;env:NodeJS.ProcessEnv;summary:any},aspect:string,docidText:Map<string,string>,shared:Map<string,ReadDoc>):Promise<{sentences:{text:string;docids:string[]}[];docsUsed:number}>{
  let hits:Hit[]=[]; try{hits=await search(a.o,aspect,POLICY.aspect_search_depth,a.env);}catch{hits=[];}
  // A2 每個面向的證據先用 CrossEncoder 重排再讀（CFDA 2025 的做法:每個 sub-query 先跑
  // MiniLM cross-encoder 選出 top-K 當該 sub-query 的證據池 T_i,而不是直接吃 BM25 排序）。
  // 我們用的是同一顆模型 Xenova/ms-marco-MiniLM-L-6-v2,所以是忠實移植。
  //
  // 為什麼值得:V2-d 已經證明 A2 的證據量會影響分數(每面向 4→6 篇,+0.0067)。
  // 這一步不加量、只換品質 —— 同樣讀 aspect_docs 篇,但讀的是 CE 認為最相關的那幾篇。
  // BM25 是詞彙匹配,面向敘述又通常是自然語言短句,兩者不匹配時前 4 名可能都不相關。
  //
  // 成本:每個面向要先把候選抓回來才能餵 CE,所以會多讀 aspect_ce_pool 篇文件。
  // 這是這個做法唯一的代價 —— Pyserini 取文件量上升,不是 LLM 呼叫。
  if(POLICY.aspect_ce_rerank&&hits.length>1){
    const pool=hits.slice(0,POLICY.aspect_ce_pool);
    const cands:{docid:string;text:string}[]=[];
    for(const h of pool){ let t:string|undefined=docidText.get(h.docid);
      // ⚠️ 這裡**一定要同時登記 shared**。只寫 docidText 的話，主迴圈看到文字已存在就
      // 跳過抓取，連帶跳過 shared 登記；模型引用了這篇，驗證器卻說「沒讀過這篇」
      // （REFERENCE_NOT_READ），整題作廢。V2-ce 第一次跑 21/22 題就是這樣掛的。
      if(t===undefined){ try{const d=await readDoc(a.o,h.docid,a.o.documentReadLimit,a.env); await sleep(120); if(!d.found)continue; const ft:string=d.text; t=ft; docidText.set(h.docid,ft); if(!shared.has(h.docid))shared.set(h.docid,{docid:h.docid,text:ft,truncated:d.truncated,rankHint:0,query:`per_aspect_ce:${aspect.slice(0,40)}`});}catch{continue;} }
      if(typeof t==="string"&&t.length>0)cands.push({docid:h.docid,text:t}); }
    if(cands.length>1){ try{
      const ranked=await rerankWithCrossEncoder(aspect,cands,{maxCharsPerDoc:2000});
      const order=new Map(ranked.map((r,i)=>[r.docid,i]));
      // CE 排到的放前面(照 CE 順序),沒被 CE 看到的維持原 BM25 順序接在後面
      hits=[...hits].sort((x,y)=>(order.get(x.docid)??1e9)-(order.get(y.docid)??1e9));
    }catch{/* CE 失效就維持 BM25 順序,不要讓整題掛掉 */} }
  }
  const out:{text:string;docids:string[]}[]=[]; let cursor=0, docsUsed=0;
  for(let round=0;round<POLICY.aspect_rounds;round++){
    const docs:{docid:string;text:string}[]=[];
    while(docs.length<POLICY.aspect_docs&&cursor<hits.length){ const h=hits[cursor++]; let text:string|undefined=docidText.get(h.docid);
      if(text===undefined){ try{const d=await readDoc(a.o,h.docid,a.o.documentReadLimit,a.env); await sleep(200); if(!d.found)continue; const ft=d.text; text=ft; docidText.set(h.docid,ft); if(!shared.has(h.docid))shared.set(h.docid,{docid:h.docid,text:ft,truncated:d.truncated,rankHint:0,query:`per_aspect:${aspect.slice(0,40)}`});}catch{continue;}}
      if(text===undefined)continue;
      docs.push({docid:h.docid,text:text.slice(0,POLICY.answer_doc_chars)}); }
    if(docs.length===0)break;
    docsUsed+=docs.length;
    // ⑫ 單一來源生成：**一次只餵一篇文件**，模型物理上沒有別的來源可以引。
    //
    // 為什麼要這樣做 —— FS 的天花板是結構性的，不是 prompt 沒寫清楚：
    //   V2d-split 拆句      FS 37.1%    V2-r2 重新歸屬  38.0%
    //   evidence ledger     FS 27.7%    drop 刪句       55.3%（涵蓋 −0.086）
    //   V2b 原子句 prompt   FS 76.8%（涵蓋 −0.14）
    // 前四個都是事後手術，全部失敗，因為句子是讀 4-6 篇綜合出來的，
    // **沒有任何單篇文件能獨力支持它**。V2b 已經在 prompt 裡要求「一句一來源」，
    // 但六篇一起餵進去，模型腦子裡早就混了，做不到。
    //
    // 這一版不靠 prompt 拜託，靠**輸入端物理隔離**：每篇文件各自呼叫一次。
    // 引用正確是 by construction，不是靠模型自律。這是 integrated-v1 的
    // record_evidence 精神接到我們的架構上 —— 但保住 per-aspect 的證據廣度
    // （仍然每題讀 86 篇，不像 integrated-v1 只讀 5.5 篇）。
    //
    // 代價：LLM 呼叫從「每輪 1 次」變成「每輪 aspect_docs 次」（4-6 倍）。
    // 用免費模型所以只是時間成本。alreadyWritten 累積傳下去，避免各篇寫出重複內容。
    if(POLICY.single_source){
      let addedThisRound=0;
      for(const d of docs){
        try{ const draft=await generateJsonWithRetry({client:a.llm,messages:[{role:"user",content:buildAspectAnswerPrompt({topic:a.topic,aspect,documents:[d],alreadyWritten:out.map(s=>s.text),...(expectedFacts.get(aspect)?{expectedFacts:expectedFacts.get(aspect)}:{}),...(POLICY.atomic_sentences?{atomic:{maxWords:POLICY.atomic_max_words,sentences:POLICY.single_source_sentences}}:{})})}],temperature:0,maxTokens:POLICY.aspect_max_tokens,validate:validateAnswer,stage:"aspect_single_source",maxRequestRetries:2,onAttempt:(attempt)=>recordAttempt({attempt,stage:"aspect_single_source",qid:a.topic.qid,out:a.out,env:a.env,summary:a.summary})});
          const arr:any[]=Array.isArray((draft.value as any).answer)?(draft.value as any).answer:[];
          for(const s of arr){ if(typeof s?.text!=="string"||!s.text.trim())continue;
            // 引用一律強制指向這一篇 —— 模型只看得到它，任何別的 docid 都是幻覺
            out.push({text:stripInlineCitations(s.text),docids:[d.docid]}); addedThisRound++;
            if(out.length>=POLICY.aspect_target_sentences)break; }
        }catch{continue;}   // 單篇失敗就跳過那篇，不要整個面向掛掉
        if(out.length>=POLICY.aspect_target_sentences)break;
      }
      if(addedThisRound===0)break;
      if(out.length>=POLICY.aspect_target_sentences)break;
      continue;
    }
    try{ const draft=await generateJsonWithRetry({client:a.llm,messages:[{role:"user",content:buildAspectAnswerPrompt({topic:a.topic,aspect,documents:docs,alreadyWritten:out.map(s=>s.text),...(expectedFacts.get(aspect)?{expectedFacts:expectedFacts.get(aspect)}:{}),...(POLICY.atomic_sentences?{atomic:{maxWords:POLICY.atomic_max_words,sentences:POLICY.atomic_per_aspect_sentences}}:{})})}],temperature:0,maxTokens:POLICY.aspect_max_tokens,validate:validateAnswer,stage:"answer_generation",maxRequestRetries:3,onAttempt:(attempt)=>recordAttempt({attempt,stage:"answer_generation",qid:a.topic.qid,out:a.out,env:a.env,summary:a.summary})});
      const val:any=draft.value; const subRefs:string[]=Array.isArray(val.references)?val.references.map(String):[]; const arr:any[]=Array.isArray(val.answer)?val.answer:[]; let addedThisRound=0;
      for(const s of arr){ if(typeof s?.text!=="string"||!s.text.trim())continue; const idxs:number[]=Array.isArray(s.citations)?s.citations:[]; const docids=[...new Set(idxs.map(i=>subRefs[i]).filter((d):d is string=>typeof d==="string"&&docs.some(x=>x.docid===d)))]; if(docids.length===0)continue; out.push({text:stripInlineCitations(s.text),docids}); addedThisRound++; }
      if(addedThisRound===0)break; // model had nothing new to add for this aspect
    }catch{break;}
    if(out.length>=POLICY.aspect_target_sentences)break; // gathered enough for this aspect
  }
  return{sentences:out,docsUsed};
}
async function generatePerAspectAnswer(a:{topic:TopicIdentity;o:IterativeOptions;llm:LlmClient;out:string;env:NodeJS.ProcessEnv;summary:any},aspects:string[],shared:Map<string,ReadDoc>):Promise<{references:string[];answer:{text:string;citations:number[]}[];perAspect:any[];docText:Map<string,string>;nuggets:any}>{
  const docidText=new Map<string,string>([...shared.values()].map(d=>[d.docid,d.text]));
  const sentences:{text:string;docids:string[]}[]=[]; const perAspect:any[]=[]; const groups:{text:string;docids:string[]}[][]=[]; let nuggetTrace:any=null;
  for(const aspect of aspects){ const r=await answerOneAspect(a,aspect,docidText,shared); sentences.push(...r.sentences); groups.push(r.sentences); perAspect.push({aspect,docs:r.docsUsed,sentences:r.sentences.length}); }
  // ⑪ 強制覆蓋閘（移植自 integrated-v1 的 finalize_research）。
  // 差別在「強制 vs 建議」：下面的 reflection 是問模型「還缺什麼」，回 0-2 個 gap，
  // 模型說夠了就結束 —— 是建議性的。integrated-v1 的閘是結構性的：每個子問題都必須
  // 被標記 covered/weak/…，沒覆蓋到就 finalize 失敗，逼它繼續找。
  // 這裡用**確定性規則**判定（句數不足即 weak），不再多花一次模型呼叫去問「夠了嗎」——
  // 模型自評「夠了」正是 reflection 會提早收工的原因。
  // ⚠️ limitation 那條沒有移植：查過 integrated-v1 的 schema，limitation 只存在研究狀態裡
  // 供 finalize 驗證，**不會變成答案句子**。寫「證據不足」進答案既沒有可引用的文件
  // （官方 schema 要求每句都要有引用），也不會涵蓋到任何 nugget。
  if(POLICY.coverage_gate&&aspects.length>0){
    const weak=()=>aspects.map((asp,i)=>({asp,i,n:groups[i]?.length??0})).filter(x=>x.n<POLICY.coverage_gate_min_sentences);
    for(let pass=0;pass<POLICY.coverage_gate_max_passes;pass++){
      const todo=weak(); if(todo.length===0)break;
      for(const {asp,i} of todo){
        // 重跑同一個面向：answerOneAspect 內部每輪讀更深一批文件，所以這一次會看到新證據
        const r=await answerOneAspect(a,asp,docidText,shared);
        const seen=new Set((groups[i]??[]).map(s=>s.text));
        const fresh=r.sentences.filter(s=>!seen.has(s.text));   // 不重複計入同一句
        if(fresh.length===0)continue;
        sentences.push(...fresh); groups[i]=[...(groups[i]??[]),...fresh];
        const t=perAspect[i]; if(t){t.sentences=groups[i].length; t.gate_passes=(t.gate_passes??0)+1;}
      }
    }
    for(const [i,t] of perAspect.entries())
      t.coverage_status=(groups[i]?.length??0)>=POLICY.coverage_gate_min_sentences?"covered":"weak";
  }
  // v4-style reflection: find aspects still missing/thin, run extra passes for them.
  if(POLICY.reflection&&sentences.length>0){ try{
    const answerText=sentences.map(s=>s.text).join(" ");
    const rr=await a.llm.generate({messages:[{role:"user",content:buildReflectionPrompt(a.topic,answerText)}],temperature:0,maxTokens:400,responseFormat:"json_object"});
    const j=extractObjectCandidate(rr.text); const gaps:string[]=j?((JSON.parse(j)?.gaps)||[]).map((x:any)=>String(x).trim()).filter((x:string)=>x.length>0).slice(0,POLICY.reflection_max_gaps):[];
    for(const gap of gaps){ const r=await answerOneAspect(a,gap,docidText,shared); sentences.push(...r.sentences); groups.push(r.sentences); perAspect.push({aspect:`[reflect] ${gap}`,docs:r.docsUsed,sentences:r.sentences.length}); }
  }catch{} }
  // ⑤ Self-assign gap-fill: the draft is scored on vital-nugget recall, so assign it against a predicted
  // nugget list and run one targeted retrieval+generation pass per vital nugget it does not yet state
  // completely. Partially-supported nuggets count as gaps because V_strict scores them zero.
  if(POLICY.nugget_loop&&sentences.length>0){ try{
    const evidence=[...shared.values()].slice(0,POLICY.nugget_context_docs).map(d=>({docid:d.docid,text:d.text}));
    const predicted=await predictNuggets(a.llm,a.topic,evidence,aspects,{maxNuggets:POLICY.nugget_max,perAspect:POLICY.nugget_per_aspect,contextDocs:POLICY.nugget_context_docs,contextChars:POLICY.nugget_context_chars});
    // Target every predicted nugget rather than only the ones our scorer calls vital: the dev gold is 74%
    // vital (925 of 1255), so a filter that discarded ~46% would mostly be discarding real targets, and
    // the gap budget already caps how much work this can cause.
    const {gaps,assignments}=await findNuggetGaps(a.llm,a.topic,sentences.map(s=>s.text).join(" "),predicted,POLICY.nugget_max_gaps);
    nuggetTrace={predicted:predicted.length,gap_count:gaps.length,gaps,assignments};
    for(const gap of gaps){ const r=await answerOneAspect(a,gap,docidText,shared); if(r.sentences.length===0)continue; sentences.push(...r.sentences); groups.push(r.sentences); perAspect.push({aspect:`[nugget] ${gap}`,docs:r.docsUsed,sentences:r.sentences.length}); }
  }catch{} }
  // Budget allocation under the official word limit. The old policy appended aspects in order and cut at
  // the limit, so the last aspects were dropped whole — a direct loss of breadth under a recall metric.
  // Breadth-first instead takes sentence 1 of every aspect, then sentence 2 of every aspect, and so on,
  // so each aspect is represented before any aspect gets a second sentence.
  const wc=(t:string)=>t.split(/\s+/).filter(Boolean).length;
  const capped:{text:string;docids:string[]}[]=[]; let words=0;
  if(POLICY.breadth_first&&groups.length>0){
    const depth=Math.max(...groups.map(g=>g.length));
    for(let round=0;round<depth;round++)for(const g of groups){ const s=g[round]; if(!s)continue; const w=wc(s.text); if(words+w>POLICY.max_answer_words&&capped.length>0)continue; capped.push(s); words+=w; }
  } else {
    for(const s of sentences){ const w=wc(s.text); if(words+w>POLICY.max_answer_words&&capped.length>0)break; capped.push(s); words+=w; }
  }
  // ── 第二層引用（CFDA 2025 的 answer integration）────────────────────────
  // 各面向的子答案 → 整合 LLM 合成一篇，**每句標記它取自哪幾個面向**，
  // 該句的候選證據就收斂成那些面向的證據池聯集，而不是全題的證據。
  // 這是針對 FS 天花板的直接解方：實測「只引一篇的句子 FS 也只有 39%」，
  // 因為 per-aspect 是讀 4–6 篇再寫一句，模型把多篇揉在一起，怎麼標都拿不到 FS。
  //
  // ⚠️ 這一層會重寫文字，跟 verify_revise(weaken) 同類的風險：改寫時把具體事實抹平
  //    就會掉涵蓋。所以整合失敗或明顯縮水時**退回原本的 capped**，不冒險。
  let final=capped;
  if(POLICY.integrate_answer&&groups.length>0&&capped.length>0){
    try{
      // 只把進了預算的句子送去整合，並記住每句屬於哪一組
      const keep=new Set(capped.map(s=>s.text));
      const blocks=groups.map((g,i)=>({aspect:perAspect[i]?.aspect??`aspect ${i}`,
        sents:g.filter(s=>keep.has(s.text))})).filter(b=>b.sents.length>0);
      if(blocks.length>1){
        const r=await a.llm.generate({messages:[{role:"user",content:buildIntegrationPrompt({topic:a.topic,
          groups:blocks.map(b=>({aspect:b.aspect,sentences:b.sents.map(s=>s.text)})),
          minWords:Math.round(POLICY.max_answer_words*0.88),maxWords:POLICY.max_answer_words})}],
          temperature:0,maxTokens:POLICY.integrate_max_tokens,responseFormat:"json_object"});
        recordAttempt({attempt:{attempt:1,provider:a.llm.provider,model:a.llm.model,latencyMs:r.latencyMs,success:true,outputChars:r.text.length},stage:"integrate",qid:a.topic.qid,out:a.out,env:a.env,summary:a.summary});
        const j=extractObjectCandidate(r.text);
        const arr:any[]=j?(JSON.parse(j)?.answer??[]):[];
        const merged:{text:string;docids:string[]}[]=[];
        for(const s of arr){
          if(typeof s?.text!=="string"||!s.text.trim())continue;
          const src:number[]=Array.isArray(s.sources)?s.sources.map((x:any)=>Number(x)).filter((x:number)=>Number.isInteger(x)&&x>=0&&x<blocks.length):[];
          // 該句的候選證據 = 它標記到的那些面向的證據聯集；沒標記就退回全部（保守）
          const pool=(src.length?src:blocks.map((_,i)=>i)).flatMap(i=>blocks[i].sents.flatMap(x=>x.docids));
          const docids=[...new Set(pool)];
          if(docids.length===0)continue;
          merged.push({text:s.text.trim(),docids});
        }
        // 退回條件：整合後內容明顯縮水就不要 —— 那正是 weaken 那類的失敗模式
        const before=capped.reduce((n,s)=>n+wc(s.text),0), after=merged.reduce((n,s)=>n+wc(s.text),0);
        if(merged.length>=Math.min(8,capped.length*0.5)&&after>=before*0.75)final=merged;
        else console.error(`  !! ${a.topic.qid}: integration 縮水（${capped.length} 句 ${before} 詞 → ${merged.length} 句 ${after} 詞），退回未整合版`);
      }
    }catch(e){console.error(`  !! ${a.topic.qid}: integration 失敗（${e instanceof Error?e.message.slice(0,80):String(e)}），退回未整合版`);}
  }
  const refs:string[]=[]; const idxOf=new Map<string,number>();
  for(const s of final)for(const d of s.docids)if(!idxOf.has(d)){idxOf.set(d,refs.length);refs.push(d);}
  const answer=final.map(s=>({text:s.text,citations:s.docids.map(d=>idxOf.get(d)!).slice(0,3)}));
  return{references:refs,answer,perAspect,docText:docidText,nuggets:nuggetTrace};
}
// ④ LLM grounded revision: one LLM call reads each sentence + its cited docs, rewrites over-claims,
// weakens/drops unsupported claims, fixes citations. Stronger than BGE relevance for the support metric.
async function groundedReviseAnswer(a:{topic:TopicIdentity;llm:LlmClient;out?:string;env?:NodeJS.ProcessEnv;summary?:any},draft:{references:string[];answer:{text:string;citations:number[]}[]},docText:Map<string,string>):Promise<{references:string[];answer:{text:string;citations:number[]}[]}|null>{
  const refs=draft.references; if(refs.length===0||draft.answer.length===0)return null;
  const sents=draft.answer.map(s=>({text:s.text,docids:[...new Set(s.citations.map(c=>refs[c]).filter(Boolean))]}));
  const citedDocids=[...new Set(sents.flatMap(s=>s.docids))]; if(citedDocids.length===0)return null;
  const citedDocs=citedDocids.map(d=>({docid:d,text:(docText.get(d)??"").slice(0,POLICY.revise_snippet_chars)}));
  // ⚠️ 這裡原本是單發、解不開 JSON 就 return null,那題永久退到較弱的 reattribute。
  // 其他每個要 JSON 的階段都有 maxRequestRetries:3–4,只有這條沒有,而且沒有 recordAttempt
  // ——失敗在 llm_trace 上是隱形的,這正是 V2 的 revise 失效能潛伏 22 題的原因之一。
  try{
    let arr:any[]|null=null;
    for(let attempt=1;attempt<=3&&arr===null;attempt++){
      const t0=Date.now(); let text=""; let errorCode:string|undefined;
      try{
        const r=await a.llm.generate({messages:[{role:"user",content:buildGroundedRevisionPrompt(a.topic,sents,citedDocs)}],temperature:0,maxTokens:POLICY.revise_max_tokens,responseFormat:"json_object"});
        text=r.text??"";
        const j=extractObjectCandidate(text);
        if(j){const parsed=JSON.parse(j)?.answer; if(Array.isArray(parsed))arr=parsed; else errorCode="LLM_JSON_SHAPE";}
        else errorCode=text?"LLM_JSON_PARSE_FAILED":"LLM_EMPTY_ASSISTANT_MESSAGE";
      }catch(e){errorCode=e instanceof Error?e.name:"LLM_ERROR";}
      if(a.out&&a.env&&a.summary)recordAttempt({attempt:{attempt,success:arr!==null,errorCode,provider:a.llm.provider,model:a.llm.model,latencyMs:Date.now()-t0,outputChars:text.length} as any,stage:"grounded_revise",qid:a.topic.qid,out:a.out,env:a.env,summary:a.summary});
    }
    if(arr===null)return null;
    const out:{text:string;docids:string[]}[]=[];
    for(const s of arr){ if(typeof s?.text!=="string"||!s.text.trim())continue; const rc:any[]=Array.isArray(s.citations)?s.citations:[]; const dc:string[]=[...new Set(rc.map((c:any)=>String(c)).filter((d:string)=>docText.has(d)))].slice(0,3); if(dc.length===0)continue; out.push({text:s.text.trim(),docids:dc}); }
    if(out.length===0)return null;
    const newRefs:string[]=[]; const idx=new Map<string,number>();
    for(const s of out)for(const d of s.docids)if(!idx.has(d)){idx.set(d,newRefs.length);newRefs.push(d);}
    return{references:newRefs,answer:out.map(s=>({text:s.text,citations:s.docids.map(d=>idx.get(d)!)}))};
  }catch{return null;}
}
// 面向分解。原本只試一次、失敗就靜默回空陣列 —— 那會讓整個 per-aspect 機制被無聲關閉,
// 生成掉到完全不同的後備路徑,而且任何 log 都看不出來(V2 smoke 就踩到:0 個面向,
// 答案變成 comprehensive 單次生成的 21 句,而同樣的程式在 B2 的 22 題是 0 失敗)。
// 這是整個 V2/V3 的前置條件(V3 的密集寫作也拿 aspects 當寫作大綱),所以加重試 + 明確警告。
// 面向 → 該面向的 expected_facts。decomposeAspects() 填、generatePerAspectAnswer() 讀。
// 用模組層的 Map 是因為面向在管線裡是以字串陣列傳遞的，改成物件會動到七八個呼叫點與
// 既有的 preset aspects 介面；一個 process 只跑一個版本、逐題循序處理，所以這是安全的。
const expectedFacts = new Map<string,string[]>();
async function decomposeAspects(a:{topic:TopicIdentity;llm:LlmClient;out:string;env:NodeJS.ProcessEnv;summary:any}):Promise<string[]>{
  let lastErr="";
  for(let attempt=1;attempt<=3;attempt++){
    try{
      const r=await a.llm.generate({messages:[{role:"user",content:buildAspectDecompositionPrompt(a.topic,POLICY.aspect_expected_facts)}],temperature:0,maxTokens:POLICY.aspect_decompose_max_tokens,responseFormat:"json_object"});
      const j=extractObjectCandidate(r.text);
      if(!j){lastErr=`no JSON object in reply (${r.text.slice(0,80)})`;}
      else{
        const parsed=JSON.parse(j); const arr=Array.isArray(parsed?.aspects)?parsed.aspects:[];
        // aspect_expected_facts 打開時每個元素是 {title, expected_facts[]}；關閉時是純字串。
        // 兩種格式都要吃，因為舊版本的既有跑用的是字串陣列，格式一變就不可比。
        const out:string[]=[]; expectedFacts.clear();
        for(const x of arr){
          const title=(typeof x==="string"?x:String(x?.title??"")).trim();
          if(!title)continue;
          out.push(title);
          const ef=Array.isArray(x?.expected_facts)?x.expected_facts.map((f:any)=>String(f).trim()).filter(Boolean):[];
          if(ef.length>0)expectedFacts.set(title,ef);
          if(out.length>=POLICY.aspect_max)break;
        }
        if(out.length>0)return out;
        lastErr="parsed but aspects[] empty";
      }
    }catch(e){lastErr=redact(e instanceof Error?e.message:String(e),a.env);}
    if(attempt<3)await sleep(500*2**attempt);
  }
  // 走到這裡代表三次都失敗。這一題的 per-aspect / 密集寫作大綱都會失效,必須看得見。
  console.error(`  !! ${a.topic.qid}: aspect decomposition FAILED after 3 attempts (${lastErr}) -> per-aspect disabled for this topic`);
  a.summary.aspect_decomposition_failures=(a.summary.aspect_decomposition_failures??0)+1;
  return [];
}

// ⑦ SETR 式證據集合選擇:挑「合起來」涵蓋最多面向的 k 篇,而不是照排名取前 k 篇。
// 互補性優先 -- 一篇補到還沒被涵蓋的面向,勝過第五篇講同一件事的。
// 一次 LLM 呼叫;任何失敗都退回排名順序(fail-open)。
async function setrSelect(a:{topic:TopicIdentity;llm:LlmClient;out:string;env:NodeJS.ProcessEnv;summary:any;checklist?:string[]},docs:ReadDoc[],k:number):Promise<ReadDoc[]>{
  try{
    const listing=docs.map((d,i)=>`[${i}] ${d.docid}: ${d.text.replace(/\s+/g," ").slice(0,300)}`).join("\n");
    const aspects=(a.checklist??[]).map(c=>c.replace(/ \(vital\)$/,"")).join("; ");
    const draft=await generateJsonWithRetry({client:a.llm,messages:[{role:"user",content:[
      "You select an evidence SET for a multi-aspect report.",
      `Information requirements (aspects to cover): ${aspects||"all aspects of the topic narrative"}`,
      `First think about which requirement each candidate can answer; then choose the set of at most ${k} documents that TOGETHER cover the most requirements. Prefer complementary documents over redundant ones; a document that covers an otherwise-uncovered requirement beats a fifth document about an already-covered one.`,
      'Return only strict JSON: {"selected":[indices in priority order]}',"",
      `Topic narrative: ${a.topic.narrative}`,"Candidate documents:",listing].join("\n")}],
      temperature:0,maxTokens:1024,
      validate:(v:unknown):LlmJsonValidationResult<{selected:number[]}>=>(typeof v==="object"&&v!==null&&Array.isArray((v as any).selected))?{ok:true,value:v as {selected:number[]}}:{ok:false,message:"selected shape"},
      stage:"evidence_select",maxRequestRetries:3,
      onAttempt:(attempt)=>recordAttempt({attempt,stage:"evidence_select",qid:a.topic.qid,out:a.out,env:a.env,summary:a.summary})});
    const seen=new Set<number>(); const picked:ReadDoc[]=[];
    for(const idx of draft.value.selected){if(typeof idx==="number"&&idx>=0&&idx<docs.length&&!seen.has(idx)){seen.add(idx);picked.push(docs[idx]);if(picked.length>=k)break;}}
    if(picked.length===0)return docs.slice(0,k);
    for(const d of docs){if(picked.length>=k)break; if(!picked.includes(d))picked.push(d);}
    writeJson(join(a.out,"topics",`${a.topic.qid}.evidence_select_trace.json`),{candidates:docs.length,selected:picked.map(d=>d.docid),aspects:a.checklist??[]});
    return picked;
  }catch{return docs.slice(0,k);}
}
async function answer(a:{topic:TopicIdentity;o:IterativeOptions;cfg:AgenticRagBaselineConfig;llm:LlmClient;llms:LlmSet;readDocs:Map<string,ReadDoc>;out:string;env:NodeJS.ProcessEnv;summary:any;presetAspects?:string[];budget?:{maxDocs:number;maxIterations:number}}){
if(a.env.R_ONLY==="1"){const s=sanitizeAnswerDraft(buildExtractiveFallbackAnswerDraft(a.readDocs)); const full:AgenticRagOutputObject={metadata:{team_id:a.cfg.teamId,run_id:a.cfg.runId,type:"automatic",narrative_id:a.topic.qid,title:"",narrative:a.topic.narrative,prompt:a.cfg.promptVersion,run_desc:POLICY.retrieval_policy,generator:a.llm.model,retrieval_depth:POLICY.output_depth},references:s.references,answer:s.answer}; return normalizeRagOutputObjectReferences(full,{config:a.cfg,topic:a.topic,readDocids:new Set(a.readDocs.keys())}).ragObject;}
let draft:any; let docs=[...a.readDocs.values()];
// ⑦ SETR:選出「合起來」涵蓋最多面向的證據集合,取代照排名取前 k 篇。
if(POLICY.setr_select&&docs.length>POLICY.setr_k){docs=await setrSelect({topic:a.topic,llm:a.llms.base,out:a.out,env:a.env,summary:a.summary,checklist:a.presetAspects},docs,POLICY.setr_k);}
// v4-style comprehensive generation: decompose narrative into aspects, then require the answer to cover each. Feed ALL read docs (not just 6). Drives coverage.
const aspects=a.presetAspects??((POLICY.comprehensive_answer||POLICY.per_aspect_generation||POLICY.dense_writing)?await decomposeAspects(a):[]);
let perAspectTrace:any=null; let nuggetTrace:any=null; let ledgerTrace:any=null; const extraDocText=new Map<string,string>();
// v4-port: per-aspect generation first. If it yields a usable answer, use it; otherwise fall back to comprehensive single-call, then compact, then extractive.
// ⑧ 密集寫作(V3):單次生成、寫滿 900-1010 詞,證據包 12 篇 x 2500 字,面向清單當寫作大綱。
// 走 writer client(V3 起是 codex:gpt-5.6-sol)。失敗就落到下面的 comprehensive/compact/extractive 級聯。
// ⑨ V6 Evidence Ledger v2:生成時強制。模型必須在同一份 JSON 裡交出句子與逐字引文,
// 規則由 evidence_ledger_v2 逐字執法(移植自 annie/peiju),違規就帶著錯誤訊息整份重生成 ——
// 對應她的 fail() 讓 agent 重新呼叫 finalize_research。重試用完就落到下面的 dense_writing。
if(POLICY.ledger_v2){
  const denseDocs=docs.slice(0,POLICY.dense_evidence_docs).map(d=>({docid:d.docid,text:d.text.slice(0,POLICY.dense_evidence_chars)}));
  const subs=(aspects.length>0?aspects:[a.topic.narrative]).map((t,i)=>({id:`Q${i+1}`,text:t.replace(/ \(vital\)$/,"")}));
  const subqs=subs.map(s=>({id:s.id,original_text:s.text,priority:"required" as const}));
  let violation:string|undefined;
  for(let attempt=1;attempt<=POLICY.ledger_v2_max_retries&&!draft;attempt++){
    try{
      const raw=await generateJsonWithRetry({client:a.llms.writer,messages:[{role:"user",content:buildLedgerAnswerPrompt({topic:a.topic,documents:denseDocs,subquestions:subs,...(violation?{violation}:{}),...(POLICY.enumerative_style?{enumerative:{minWords:POLICY.enumerative_words_min,maxWords:POLICY.enumerative_words_max}}:{})})}],temperature:0,maxTokens:POLICY.ledger_v2_max_tokens,validate:validateLedgerPlan,stage:"answer_generation",maxRequestRetries:3,onAttempt:(at)=>recordAttempt({attempt:at,stage:"answer_generation",qid:a.topic.qid,out:a.out,env:a.env,summary:a.summary})});
      // 每次重試都用全新的 ledger —— 上一輪被拒的 evidence record 不該留下來。
      const ledger=new ExposureLedger(); for(const d of denseDocs)ledger.expose(d.docid,d.text);
      const plan=raw.value.answer_plan.map(s=>({text:s.text,citations:s.citations,
        evidence_ids:s.evidence.map(e=>ledger.recordEvidence({docid:e.docid,subquestion_ids:Array.isArray(e.subquestion_ids)&&e.subquestion_ids.length?e.subquestion_ids:[subs[0].id],exact_quote:e.exact_quote,claim:e.claim}).evidence_id)}));
      const verified=enforceAnswerPlan({ledger,answerPlan:plan,subquestions:subqs,policy:POLICY.ledger_v2_policy as "evidence-ledger-v1"|"evidence-ledger-v2"});
      const refs=[...new Set(verified.flatMap(s=>s.citations))];
      draft={value:{references:refs,answer:verified.map(s=>({text:s.text,citations:s.citations.map(d=>refs.indexOf(d)).filter(i=>i>=0)}))}};
      ledgerTrace={mode:"evidence_ledger_v2",policy:POLICY.ledger_v2_policy,attempts:attempt,sentences:verified.length,evidence_records:ledger.records.length};
    }catch(e){
      if(e instanceof LedgerViolation){violation=e.message; ledgerTrace={mode:"evidence_ledger_v2_rejected",attempts:attempt,violation};}
      else {ledgerTrace={mode:"evidence_ledger_v2_error",attempts:attempt,error:redact(e instanceof Error?e.message:String(e),a.env)};}
    }
  }
}
// ⑦ 先讓 per-aspect 把證據收齊(每個面向各自搜尋、各自讀),再交給下面的密集寫作。
// generatePerAspectAnswer 讀到的文件會寫進 a.readDocs(它就是傳進去的 shared),
// 所以這裡跑完 docs 會從迴圈的 ~12 篇長到 ~88 篇。它的答案刻意不採用。
if(POLICY.dense_after_aspect&&aspects.length>0){try{
  const before=a.readDocs.size;
  const pa=await generatePerAspectAnswer({topic:a.topic,o:a.o,llm:POLICY.aspect_writer?a.llms.writer:a.llm,out:a.out,env:a.env,summary:a.summary},aspects,a.readDocs);
  for(const [k,v] of pa.docText)extraDocText.set(k,v);
  perAspectTrace={...(pa.perAspect??{}),mode:"evidence_only",docs_before:before,docs_after:a.readDocs.size};
  docs=[...a.readDocs.values()];   // ← 密集寫作接下來吃的就是這一份
}catch{}}
if(!draft&&POLICY.dense_writing){try{
  const denseDocs=docs.slice(0,POLICY.dense_evidence_docs).map(d=>({docid:d.docid,text:d.text.slice(0,POLICY.dense_evidence_chars)}));
  draft=await generateJsonWithRetry({client:a.llms.writer,messages:[{role:"user",content:buildDenseAnswerGenerationPrompt({topic:a.topic,documents:denseDocs,...(aspects.length>0?{checklist:aspects}:{}),...(POLICY.enumerative_style?{enumerative:{minWords:POLICY.enumerative_words_min,maxWords:POLICY.enumerative_words_max}}:{})})}],temperature:0,maxTokens:POLICY.answer_max_tokens,validate:validateAnswer,stage:"answer_generation",maxRequestRetries:4,onAttempt:(attempt)=>recordAttempt({attempt,stage:"answer_generation",qid:a.topic.qid,out:a.out,env:a.env,summary:a.summary})});
}catch{}}
if(!draft&&POLICY.per_aspect_generation&&aspects.length>0){try{const pa=await generatePerAspectAnswer({topic:a.topic,o:a.o,llm:POLICY.aspect_writer?a.llms.writer:a.llm,out:a.out,env:a.env,summary:a.summary},aspects,a.readDocs); perAspectTrace=pa.perAspect; nuggetTrace=pa.nuggets; for(const [k,v] of pa.docText)extraDocText.set(k,v); if(pa.answer.length>=Math.min(3,aspects.length))draft={value:{references:pa.references,answer:pa.answer}};}catch{}}
if(!draft){try{const promptDocs=docs.map(d=>({docid:d.docid,text:d.text.slice(0,POLICY.answer_doc_chars)})); draft=await generateJsonWithRetry({client:a.llm,messages:[{role:"user",content:POLICY.comprehensive_answer?buildComprehensiveAnswerPrompt({topic:a.topic,documents:promptDocs,aspects,atomic:POLICY.atomic_sentences}):buildAnswerGenerationPrompt({topic:a.topic,documents:docs.slice(0,6).map(d=>({docid:d.docid,text:d.text.slice(0,1000)}))})}],temperature:0,maxTokens:POLICY.answer_max_tokens,validate:validateAnswer,stage:"answer_generation",maxRequestRetries:4,onAttempt:(attempt)=>recordAttempt({attempt,stage:"answer_generation",qid:a.topic.qid,out:a.out,env:a.env,summary:a.summary})});}catch{try{draft=await generateJsonWithRetry({client:a.llm,messages:[{role:"user",content:buildCompactAnswerGenerationPrompt({topic:a.topic,documents:docs.slice(0,5).map(d=>({docid:d.docid,text:d.text.slice(0,800)}))})}],temperature:0,maxTokens:2000,validate:validateAnswer,stage:"answer_generation",maxRequestRetries:4,onAttempt:(attempt)=>recordAttempt({attempt,stage:"answer_generation",qid:a.topic.qid,out:a.out,env:a.env,summary:a.summary})});}catch{draft={value:buildExtractiveFallbackAnswerDraft(a.readDocs)}}}} let sanitized=sanitizeAnswerDraft(draft.value); if(sanitized.answer.length===0)sanitized=sanitizeAnswerDraft(buildExtractiveFallbackAnswerDraft(a.readDocs));
// Grounded citation verification: drop citations whose cited doc does not actually support the sentence. Logs before/after so support gain is attributable.
{const docText=new Map<string,string>([...a.readDocs.values()].map(d=>[d.docid,d.text])); for(const [k,v] of extraDocText)if(!docText.has(k))docText.set(k,v);
let verifyStats:any=null;
// Re-attribution (CiteFix/VeriCite style): re-point each sentence's citation to the best-supporting
// reference via BGE-Reranker. Preserves coverage (sentence stays) while fixing support. Falls back to
// the keyword-overlap verify if re-attribution fails/disabled.
// Never-drop guard: the model is told to return every sentence, but if it still shrinks the answer we
// reject the whole revision rather than lose the nuggets those sentences carried. Falls through to
// re-attribution, which fixes citations without touching sentence count.
// ⑨ V6:先過 evidence ledger 的硬規則,再讓 LLM 潤飾。
if(POLICY.evidence_ledger){try{const lg=enforceLedger(sanitized as any,docText,{minSupport:POLICY.ledger_min_support}); sanitized=lg.draft as any; verifyStats={mode:"evidence_ledger",...lg.stats};}catch{}}
if(POLICY.llm_revise){try{const before=sanitized.answer.length; const rev=await groundedReviseAnswer({topic:a.topic,llm:a.llms.writer,out:a.out,env:a.env,summary:a.summary},sanitized as any,docText);
  if(rev&&rev.answer.length>0){ if(POLICY.revise_never_drop&&rev.answer.length<before){ verifyStats={mode:"llm_revise_rejected",before,after:rev.answer.length}; } else { sanitized=rev as any; verifyStats={mode:"llm_revise",sentences:rev.answer.length}; } }}catch{}}
if(verifyStats&&verifyStats.mode==="llm_revise_rejected")verifyStats=null;
if(verifyStats&&verifyStats.mode==="evidence_ledger")verifyStats={...verifyStats,ledger_then:"reattribute_or_verify"}; // let re-attribution handle it
if(!verifyStats&&POLICY.reattribute){try{const r=await reattributeCitations(sanitized as any,docText,a.env,{threshold:POLICY.reattribute_threshold,maxCites:POLICY.reattribute_max_cites,snippetChars:1500}); if(r.draft.answer.length>0){sanitized=r.draft as any; verifyStats={mode:"reattribute",...r.stats};}}catch{}}
if(!verifyStats&&POLICY.citation_verify){const v=verifyCitations(sanitized as any,docText,{supportThreshold:POLICY.support_threshold}); if(v.draft.answer.length>0)sanitized=v.draft as any; else sanitized=sanitizeAnswerDraft(buildExtractiveFallbackAnswerDraft(a.readDocs)); verifyStats={mode:"keyword",...v.stats};}
// ⑨ 逐句驗證改寫(V3):比對每句與其引用證據,過度宣稱就縮回證據範圍。
if(POLICY.verify_revise&&sanitized.answer.length>0){try{
  const B=POLICY.verify_batch, refs=sanitized.references, sents=sanitized.answer;
  const revised:{text:string;citations:number[]}[]=[];
  for(let b=0;b<sents.length;b+=B){
    const batch=sents.slice(b,b+B);
    const prompt=buildVerifyRevisePrompt({topic:a.topic,references:refs,mode:POLICY.verify_mode as any,
      sentences:batch.map(s2=>({text:s2.text,citations:s2.citations,
        evidence:[...new Set(s2.citations.map(ci=>refs[ci]).filter(Boolean))].slice(0,2)
          .map(d=>({docid:d,excerpt:(docText.get(d)??"").slice(0,1200)}))}))});
    const draft=await generateJsonWithRetry({client:a.llms.writer,messages:[{role:"user",content:prompt}],temperature:0,maxTokens:POLICY.verify_max_tokens,validate:validateAnswer,stage:"verify_revise",maxRequestRetries:3,onAttempt:(attempt)=>recordAttempt({attempt,stage:"verify_revise",qid:a.topic.qid,out:a.out,env:a.env,summary:a.summary})});
    revised.push(...(draft.value.answer as any[]));
  }
  // 只在驗證後的結果非空時採用,否則保留原稿(不要因為一次解析失敗就交出空答案)。
  if(revised.length>0){const r=sanitizeAnswerDraft({references:refs,answer:revised} as any); if(r.answer.length>0)sanitized=r;}
}catch{}}
writeJson(join(a.out,"topics",`${a.topic.qid}.gen_trace.json`),{topic_id:a.topic.qid,aspects,aspect_count:aspects.length,aspect_decomposition_ok:aspects.length>0,per_aspect:perAspectTrace,nuggets:nuggetTrace,ledger_v2:ledgerTrace,verify:verifyStats,style:sentenceStyleStats(sanitized)});}
// Final word-limit enforcement. The allocator caps the draft at max_answer_words, but the grounded
// revision that runs afterwards rewrites sentences to state facts completely and can make them longer --
// dev22 produced one answer of 1237 words against the official 1024-word ceiling. Re-apply the cap on
// whatever the verification cascade returned, dropping only trailing sentences.
{const lim=POLICY.max_answer_words; let words=0; const kept:typeof sanitized.answer=[];
 for(const s of sanitized.answer){const w=s.text.split(/\s+/).filter(Boolean).length; if(words+w>lim&&kept.length>0)break; kept.push(s); words+=w;}
 if(kept.length<sanitized.answer.length)sanitized={references:sanitized.references,answer:kept};}
// 官方 v0.6.0:一句多引用要按支持強度排序。用本地 supportScore,不依賴外部服務。
if(POLICY.order_citations_by_strength){const dt=new Map<string,string>([...a.readDocs.values()].map(d=>[d.docid,d.text])); for(const [k,v] of extraDocText)if(!dt.has(k))dt.set(k,v);
  sanitized={references:sanitized.references,answer:sanitized.answer.map(s=>s.citations.length<2?s:{...s,citations:[...s.citations].sort((x,y)=>supportScore(s.text,dt.get(sanitized.references[y])??"")-supportScore(s.text,dt.get(sanitized.references[x])??""))})};}
const full:AgenticRagOutputObject={metadata:{team_id:a.cfg.teamId,run_id:a.cfg.runId,type:"automatic",narrative_id:a.topic.qid,title:"",narrative:a.topic.narrative,prompt:a.cfg.promptVersion,run_desc:POLICY.retrieval_policy,generator:a.llm.model,retrieval_depth:POLICY.output_depth},references:sanitized.references,answer:sanitized.answer}; return normalizeRagOutputObjectReferences(full,{config:a.cfg,topic:a.topic,readDocids:new Set(a.readDocs.keys())}).ragObject;}
// The LLM occasionally miscounts and cites a references[] index that doesn't exist
// (CITATION_OUT_OF_RANGE). Rather than failing the whole topic on a single formatting
// slip, drop the invalid citation indices and drop any sentence left with none.
function sanitizeAnswerDraft(draft:AnswerDraft):AnswerDraft{
  const refCount=Array.isArray(draft.references)?draft.references.length:0;
  const answer=(Array.isArray(draft.answer)?draft.answer:[])
    .map(sentence=>({
      text:typeof sentence?.text==="string"?sentence.text:"",
      citations:[...new Set((Array.isArray(sentence?.citations)?sentence.citations:[]).filter(c=>Number.isInteger(c)&&c>=0&&c<refCount))],
    }))
    .filter(sentence=>sentence.text.trim()!==""&&sentence.citations.length>0);
  return{references:Array.isArray(draft.references)?draft.references:[],answer};
}
function validateAnswer(v:unknown):LlmJsonValidationResult<AnswerDraft>{return isRecord(v)&&Array.isArray(v.references)&&Array.isArray(v.answer)?{ok:true,value:v as AnswerDraft}:{ok:false,message:"answer shape"};}
type LedgerPlan={answer_plan:{text:string;citations:string[];evidence:{docid:string;exact_quote:string;claim:string;subquestion_ids:string[]}[]}[]};
// ⑨ V6:evidence ledger 的輸出形狀。只檢查結構,內容規則交給 evidence_ledger_v2 逐字執法。
function validateLedgerPlan(v:unknown):LlmJsonValidationResult<LedgerPlan>{
  if(!isRecord(v)||!Array.isArray(v.answer_plan)||v.answer_plan.length===0)return{ok:false,message:"answer_plan shape"};
  for(const s of v.answer_plan as any[]){
    if(!isRecord(s)||typeof s.text!=="string"||!Array.isArray(s.citations)||!Array.isArray(s.evidence)||s.evidence.length===0)return{ok:false,message:"answer_plan sentence shape"};
    for(const e of s.evidence)if(!isRecord(e)||typeof e.docid!=="string"||typeof e.exact_quote!=="string"||typeof e.claim!=="string")return{ok:false,message:"evidence record shape"};
  }
  return{ok:true,value:v as unknown as LedgerPlan};
}
async function readNew(a:{o:IterativeOptions;hits:Hit[];readDocs:Map<string,ReadDoc>;failedRead:string[];count:number;query:string;env:NodeJS.ProcessEnv;maxDocs?:number;findQueries?:string[]}){const cap=a.maxDocs??a.o.maxDocumentsRead; let added=0; for(const [i,h] of a.hits.entries()){if(added>=a.count||a.readDocs.size>=cap)return; if(a.readDocs.has(h.docid))continue; try{
  // find_in_document 開啟時讀更多行(才有東西可定位),然後只留最相關的幾段;關閉時維持原本的前 N 行。
  const lines=POLICY.find_in_document?POLICY.find_read_lines:a.o.documentReadLimit;
  const d=await readDoc(a.o,h.docid,lines,a.env);
  if(d.found){
    let text=d.text;
    if(POLICY.find_in_document&&(a.findQueries?.length??0)>0){
      const ps=findInDocument(text,a.findQueries as string[],{mode:POLICY.find_mode as any,maxPassages:POLICY.find_max_passages,windowChars:POLICY.find_window_chars,scanLimit:POLICY.find_scan_limit,impl:POLICY.find_impl as "lines"|"windows",contextBefore:POLICY.find_context_before,contextAfter:POLICY.find_context_after});
      // 找不到任何相關段落時退回原本的前 N 行,不要讓這篇變成空的。
      if(ps.length>0)text=ps.map(p=>`[offset ${p.offset}] ${p.text}`).join("\n\n");
    }
    a.readDocs.set(h.docid,{docid:h.docid,text,truncated:d.truncated,rankHint:i+1,query:a.query}); added++;
  } else a.failedRead.push(h.docid);
}catch{a.failedRead.push(h.docid)} await sleep(200);}}
// Pyserini 的限流:視窗只有 1 秒(伺服器回 retry-after: 1),而且是**綁 token 的**
// (實測:同一瞬間 token A 被擋、token B 仍回 200;亂填 token 是 401)。
//
// 但 429 有兩種,要分開處理 —— 這是實測踩出來的:
//   偶發撞車  兩片剛好同時發請求。等 retry-after 那 1 秒就過了,
//             但**退避一定要加隨機抖動**:舊版退避是決定性的,兩片會同步睡、
//             同步醒、再撞一次,八次撞完就宣告失敗。
//   持續飽和  S1 這種每題搜幾十次的版本,單一 token 的配額根本不夠兩片用。
//             這時候等 1 秒毫無意義,只有等久一點讓對面先跑完才有機會。
//
// 所以退避取「retry-after」與「指數成長」的**較大值**,再加抖動:
// 偶發時抖動負責錯開,持續飽和時指數負責把總容忍時間撐到約 150 秒。
//
// ⚠️ 別只照 retry-after 等。實測過:那樣把總容忍從 152 秒縮到 24 秒,
//    S1 的失敗率反而從 8 題掛 2 題惡化到 8 題掛 4 題。抖動是修正,縮短容忍是退步。
const RETRY_MAX = 16;
function retryDelay(r: Response | null, a: number) {
  const ra = Number(r?.headers.get("retry-after"));
  const server = Number.isFinite(ra) && ra > 0 ? Math.min(30000, ra * 1000) : 0;
  const growth = Math.max(1000, 600 * 2 ** Math.min(a, 7));            // 上限 76.8 秒
  const base = Math.max(server, growth);
  return base + Math.floor(Math.random() * Math.max(500, base * 0.5)); // 抖動:讓共用 token 的行程錯開
}
// sink:S1 用。Pyserini 的 search 回應每個候選都附 doc 全文,原本被直接丟掉;
// 打開 tail_reselect 時順手接住(截到 tail_reselect_text_chars,CE 反正只吃前 512 token)。
// 巨型回應防護：某些查詢命中一批超大文件，5000 篇全文的 JSON 回應可超過
// Node 的 512MB 字串上限（官方 rag2026-36 實測 532MB → r.json() 直接丟
// "Cannot create a string longer than…"，重試多少次都一樣）。遇到這個
// 特定錯誤就把深度折半再試 —— 少拉的部分由融合端其他查詢與池閘門把關。
async function search(o:IterativeOptions,query:string,depth:number,env:NodeJS.ProcessEnv,sink?:Map<string,string>){
  // 巨型回應家族：本地字串上限（rag2026-36 實測 532MB）、下載中途被切
  //（rag2026-70：600 字元查詢在 hits>=100 就 connection reset，截 200 字元
  // 後 hits=5000 正常 96MB）。兩段降級：先深度折半，再查詢截斷 —— 全部
  // 失敗才把最後的錯誤丟回去，不吞其他類型的錯。
  const giant=(e:unknown)=>{const m=String((e as any)?.message??e)+String((e as any)?.cause??"");
    return m.includes("Cannot create a string")||m.includes("terminated")||m.includes("ECONNRESET")||m.includes("fetch failed")||m.includes("aborted");};
  let lastErr:unknown;
  const tries=query.length>200?[query,query.slice(0,200)]:[query];
  for(const q of tries){
    if(q!==query) console.error("search: 查詢降級為前 200 字元");
    for(let d=depth; d>=Math.min(depth,313); d=Math.floor(d/2)){
      try{ return await searchAtDepth(o,q,d,env,sink); }
      catch(e){ lastErr=e;
        if(!giant(e)) throw e;
        console.error(`search: 巨型回應（${String((e as any)?.message??e).slice(0,40)}），深度 ${d} 折半`);
      }
    }
  }
  throw lastErr;
}
async function searchAtDepth(o:IterativeOptions,query:string,depth:number,env:NodeJS.ProcessEnv,sink?:Map<string,string>){const token=env[o.pyseriniTokenEnv]?.trim(); const url=`${o.pyseriniBaseUrl.replace(/\/+$/,'')}/v1/${o.pyseriniIndex}/search?${new URLSearchParams({query,hits:String(depth)})}`; for(let a=1;a<=RETRY_MAX;a++){let r:Response; try{r=await fetch(url,{headers:token?{authorization:`Bearer ${token}`}:{}});}catch(e){if(a===RETRY_MAX)throw new Error(`Pyserini fetch failed: ${e instanceof Error?e.message:String(e)}`); await sleep(retryDelay(null,a)); continue;} if(r.ok){const v=await r.json() as any; const cs=(v.candidates??[]).filter((c:any)=>typeof c.docid==='string').slice(0,depth); if(sink)for(const c of cs){if(typeof c.doc==='string'&&!sink.has(c.docid))sink.set(c.docid,c.doc.slice(0,POLICY.tail_reselect_text_chars));} return cs.map((c:any)=>({docid:c.docid,score:Number(c.score)||0}))} if(![429,500,502,503,504].includes(r.status)||a===RETRY_MAX)throw new Error(`Pyserini HTTP ${r.status}`); await sleep(retryDelay(r,a));} throw new Error('search failed')}
// 截斷永遠在最後做,快取存的是未截斷全文 —— 這樣改 document-read-limit 不會讓快取失效。
function cutDoc(text:string,limit:number){const lines=text.split(/\r?\n/); return{found:true,text:lines.slice(0,limit).join('\n'),truncated:lines.length>limit}}
async function readDoc(o:IterativeOptions,docid:string,limit:number,env:NodeJS.ProcessEnv){const hit=readCache(o.pyseriniIndex,docid); if(hit)return hit.found?cutDoc(hit.text,limit):{found:false,text:"",truncated:false}; const token=env[o.pyseriniTokenEnv]?.trim(); const url=`${o.pyseriniBaseUrl.replace(/\/+$/,'')}/v1/${o.pyseriniIndex}/doc/${encodeURIComponent(docid)}`; const maxAttempts=RETRY_MAX; for(let a=1;a<=maxAttempts;a++){let r:Response; try{r=await fetch(url,{headers:token?{authorization:`Bearer ${token}`}:{}});}catch(e){if(a===maxAttempts)throw new Error(`Pyserini doc fetch failed: ${e instanceof Error?e.message:String(e)}`); await sleep(retryDelay(null,a)); continue;} if(r.status===404){writeCache(o.pyseriniIndex,docid,{found:false,text:""}); return{found:false,text:"",truncated:false}} if(r.ok){const v=await r.json() as any; const text=extractText(v.doc); writeCache(o.pyseriniIndex,docid,{found:true,text}); return cutDoc(text,limit)} if(![429,500,502,503,504].includes(r.status)||a===maxAttempts)throw new Error(`Pyserini doc HTTP ${r.status}`); await sleep(retryDelay(r,a));} throw new Error('read failed')}
function weightedRrf(rs:Hit[][],ws:number[],depth:number,k:number){const m=new Map<string,{score:number;best:number}>();rs.forEach((hits,ri)=>hits.forEach((h,i)=>{const p=m.get(h.docid)??{score:0,best:Infinity};p.score+=(ws[ri]??1)/(k+i+1);p.best=Math.min(p.best,i+1);m.set(h.docid,p)}));return[...m.entries()].sort((a,b)=>b[1].score-a[1].score||a[1].best-b[1].best||a[0].localeCompare(b[0])).slice(0,depth).map(([docid,v],i)=>({docid,rank:i+1,score:v.score}))}
function buildRetrievalTrace(topic:TopicIdentity,queries:string[],runs:Hit[][],ranking:ReturnType<typeof weightedRrf>){return{topic_id:topic.qid,policy:POLICY,anchor_query:topic.narrative,followup_queries:queries.slice(1),query_count:queries.length,per_query_top10:runs.map((h,i)=>({query:queries[i],weight:i===0?POLICY.bm25_anchor_weight:(typeof queries[i]==="string"&&queries[i].startsWith("[query2doc]"))?POLICY.q2d_weight:POLICY.followup_query_weight,top10:h.slice(0,10)})),fused_top10:ranking.slice(0,10),candidate_count:ranking.length}}
function writeTopicPartial(a:{out:string;topic:TopicIdentity;iterationTrace:any[];judgeTrace:any[];queries:string[];runs:Hit[][];ranking:ReturnType<typeof weightedRrf>;readDocs:Map<string,ReadDoc>;failedRead:string[]}){const base=join(a.out,"topics"); writeJson(join(base,`${a.topic.qid}.iteration_trace.json`),a.iterationTrace); writeJson(join(base,`${a.topic.qid}.judge_trace.json`),a.judgeTrace); writeJson(join(base,`${a.topic.qid}.retrieval-trace.json`),buildRetrievalTrace(a.topic,a.queries,a.runs,a.ranking)); writeJson(join(base,`${a.topic.qid}.final_read_docs_trace.json`),{topic_id:a.topic.qid,candidate_pool_size:a.ranking.length,documents_read_attempted:a.readDocs.size+a.failedRead.length,documents_read_successful:a.readDocs.size,read_docids:[...a.readDocs.keys()],failed_read_docids:a.failedRead,final_answer_cited_docids:[]});}
function classifyJudgeStopReason(message:string){if(/empty assistant message/i.test(message))return"judge_empty_assistant_message"; if(/429|rate limit/i.test(message))return"judge_rate_limit"; if(/5\\d\\d|server/i.test(message))return"judge_server_error"; if(/fetch failed|network|timeout|timed out|ECONNRESET|ETIMEDOUT/i.test(message))return"judge_transient_request_failed"; return"judge_json_parse_failed";}
function recordAttempt(a:{attempt:LlmAttemptTrace;stage:string;qid:string;out:string;env:NodeJS.ProcessEnv;summary:any}){a.summary.llm_call_count++; if(!a.attempt.success)a.summary.llm_failed_call_count++; if(a.attempt.attempt>1)a.summary.llm_retry_count++; appendJsonl(join(a.out,"llm_trace.jsonl"),{qid:a.qid,stage:a.stage,attempt:a.attempt.attempt,success:a.attempt.success,error_code:a.attempt.errorCode,provider:a.attempt.provider,model:a.attempt.model,latency_ms:a.attempt.latencyMs,output_chars:a.attempt.outputChars},a.env)}
function extractText(doc:any){return typeof doc==='string'?doc:(doc&&typeof doc.text==='string'?doc.text:JSON.stringify(doc??''))} function topicRun(qid:string,r:any[],runId:string){return r.map((e,i)=>`${qid} Q0 ${e.docid} ${i+1} ${e.score.toFixed(8)} ${runId}`).join('\n')+'\n'} function render(r:Rankings,qids:string[],runId:string){return qids.flatMap(q=>(r.get(q)??[]).map((e,i)=>`${q} Q0 ${e.docid} ${i+1} ${e.score.toFixed(8)} ${runId}`)).join('\n')+'\n'} function assemble(out:string,topics:Topic[]):Rankings{const r:Rankings=new Map(); for(const t of topics){const p=join(out,"topics",`${t.qid}.runfile.trec`); if(existsSync(p))r.set(t.qid,readFileSync(p,'utf8').split(/\r?\n/).filter(Boolean).map(line=>{const[,,docid,rank,score]=line.split(/\s+/);return{docid,rank:Number(rank),score:Number(score)}}))} return r}
function qrelsPaths(dir:string){return['rag25-climbmix-umbrela-codex-gpt5.5-medium-reasoning-v1.qrels','rag25-climbmix-umbrela-ministral-3-14b-instruct-2512-v2.qrels','rag25-climbmix-umbrela-qwen3.5-9b-v2.qrels'].map(n=>join(dir,n))} function parseQrels(path:string,qids:string[]):Qrels{const w=new Set(qids),q:Qrels=new Map(); for(const line of readFileSync(path,'utf8').split(/\r?\n/)){if(!line.trim())continue; const[qid,,docid,rel]=line.split(/\s+/); if(!w.has(qid))continue; const m=q.get(qid)??new Map<string,number>(); m.set(docid,Number(rel)||0); q.set(qid,m)} return q} function evalAll(paths:string[],qids:string[],rankings:Rankings){const rows=paths.map(p=>{const q=parseQrels(p,qids),res=evaluateRankings(q,rankings,qids,{recallCutoffs:CUTS,ndcgCutoffs:NDCG,mrrCutoffs:[1000]},{recallRelevantThreshold:2,binaryRelevantThreshold:2,ndcgGainMode:'linear'}); const metrics={ndcg_10:res.ndcgByCutoff.get(10)??0,ndcg_20:res.ndcgByCutoff.get(20)??0,ndcg_100:res.ndcgByCutoff.get(100)??0,ndcg_1000:res.ndcgByCutoff.get(1000)??0,recall_20:res.macroRecallByCutoff.get(20)??0,recall_100:res.macroRecallByCutoff.get(100)??0,recall_500:res.macroRecallByCutoff.get(500)??0,recall_1000:res.macroRecallByCutoff.get(1000)??0,map:res.map,mrr:res.mrrByCutoff.get(1000)??0}; return{qrels_path:p,qrels_filename:basename(p),metrics,per_topic:{}}}); const keys=Object.keys(rows[0].metrics); return{summary:{qrels:rows.map(({per_topic:_p,...x})=>x),arithmetic_mean_across_qrels:Object.fromEntries(keys.map(k=>[k,rows.reduce((s:any,r:any)=>s+(r.metrics as any)[k],0)/rows.length]))},perTopic:{}}}
function writeJson(p:string,v:any){mkdirSync(dirname(p),{recursive:true});writeFileSync(p,JSON.stringify(v,null,2)+'\n')} function writeJsonl(p:string,rows:any[]){writeFileSync(p,rows.map(r=>JSON.stringify(r)).join('\n')+(rows.length?'\n':''))} function appendJsonl(p:string,v:any,_env:NodeJS.ProcessEnv){mkdirSync(dirname(p),{recursive:true});writeFileSync(p,JSON.stringify(v)+'\n',{flag:'a'})} function readIf(p:string):any{return existsSync(p)?JSON.parse(readFileSync(p,'utf8')):null} function isRecord(v:unknown):v is Record<string,unknown>{return typeof v==='object'&&v!==null&&!Array.isArray(v)} function sleep(ms:number){return new Promise(r=>setTimeout(r,ms))} function redact(s:string,env:NodeJS.ProcessEnv){for(const k of ['NCHC_API_KEY','PYSERINI_API_TOKEN']){const v=env[k]; if(v)s=s.split(v).join(`[redacted ${k}]`)} return s}
