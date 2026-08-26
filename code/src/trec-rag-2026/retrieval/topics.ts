import { readFile } from "node:fs/promises";
import { resolve } from "node:path";

export const TREC_RAG_2026_TOPICS_ENV = "TREC_RAG_2026_TOPICS_PATH";
export const DEFAULT_TREC_RAG_2026_TOPICS_SIBLING_PATH =
  "../trec-rag-data/trec-rag-2026/test-data/trec_rag_2026_queries.tsv";

export type ResolveTrecRag2026TopicsPathOptions = {
  explicitTopicsPath?: string;
  env?: NodeJS.ProcessEnv;
  repoRoot?: string;
};

export function resolveTrecRag2026TopicsPath(options: ResolveTrecRag2026TopicsPathOptions = {}): string {
  const repoRoot = options.repoRoot ?? process.cwd();
  if (options.explicitTopicsPath) {
    return resolve(repoRoot, options.explicitTopicsPath);
  }
  const envTopicsPath = (options.env ?? process.env)[TREC_RAG_2026_TOPICS_ENV]?.trim();
  if (envTopicsPath) {
    return resolve(repoRoot, envTopicsPath);
  }
  return resolve(repoRoot, DEFAULT_TREC_RAG_2026_TOPICS_SIBLING_PATH);
}

export type TrecRagTopic = {
  topicId: string;
  narrative: string;
};

export function parseTrecRag2026TopicsTsv(text: string): TrecRagTopic[] {
  const normalized = text.replace(/^\uFEFF/, "");
  const lines = normalized.split(/\r?\n/);
  if (lines.length > 0 && lines[lines.length - 1] === "") {
    lines.pop();
  }
  if (lines.length === 0) {
    throw new Error("Topics file is empty.");
  }

  const topics: TrecRagTopic[] = [];
  const seen = new Set<string>();
  for (const [index, line] of lines.entries()) {
    const lineNumber = index + 1;
    const columns = line.split("\t");
    if (columns.length !== 2) {
      throw new Error(
        `Malformed topics row ${lineNumber}: expected exactly two TSV columns (topic_id and narrative).`,
      );
    }
    const [topicId, narrative] = columns;
    if (!topicId) {
      throw new Error(`Malformed topics row ${lineNumber}: topic_id must be non-empty.`);
    }
    if (/\s/.test(topicId)) {
      throw new Error(`Malformed topics row ${lineNumber}: topic_id must not contain whitespace.`);
    }
    if (!narrative) {
      throw new Error(`Malformed topics row ${lineNumber}: narrative must be non-empty.`);
    }
    if (seen.has(topicId)) {
      throw new Error(`Duplicate topic_id in topics file: ${topicId}`);
    }
    seen.add(topicId);
    topics.push({ topicId, narrative });
  }
  return topics;
}

export async function readTrecRag2026Topics(path: string): Promise<TrecRagTopic[]> {
  const text = await readFile(path, "utf8");
  return parseTrecRag2026TopicsTsv(text);
}
