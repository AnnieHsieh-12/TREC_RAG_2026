import assert from "node:assert/strict";
import { mkdtempSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

import { runIterativeAgenticRag } from "../src/trec-rag-2026/bounded-rag/runner";

const DOCID = "shard_00001_1";
const QREL_FILES = [
  "rag25-climbmix-umbrela-codex-gpt5.5-medium-reasoning-v1.qrels",
  "rag25-climbmix-umbrela-ministral-3-14b-instruct-2512-v2.qrels",
  "rag25-climbmix-umbrela-qwen3.5-9b-v2.qrels",
];

test("bounded RAG completes one topic with mocked retrieval and LLM services", async () => {
  const root = mkdtempSync(join(tmpdir(), "cfda-rag-smoke-"));
  const topics = join(root, "topics.tsv");
  const qrels = join(root, "qrels");
  const output = join(root, "output");
  writeFileSync(topics, "1\tExplain the documented fact.\n");
  mkdirSync(qrels);
  for (const filename of QREL_FILES) {
    writeFileSync(join(qrels, filename), `1 0 ${DOCID} 2\n`);
  }

  const originalFetch = globalThis.fetch;
  let llmCalls = 0;
  globalThis.fetch = async (input, init) => {
    const url = String(input);
    if (url.includes("/search?")) {
      return jsonResponse({ candidates: [{ docid: DOCID, score: 9.5 }] });
    }
    if (url.includes("/doc/")) {
      return jsonResponse({
        doc: { text: "The documented fact is supported by this passage." },
      });
    }
    if (url === "http://mock.openai/v1/chat/completions") {
      llmCalls += 1;
      const content =
        llmCalls === 1
          ? JSON.stringify({ enough: true, queries: [] })
          : JSON.stringify({
              references: [DOCID],
              answer: [
                { text: "The documented fact is supported.", citations: [0] },
              ],
            });
      return jsonResponse({ choices: [{ message: { content } }] });
    }
    throw new Error(
      `Unexpected smoke-test request: ${url} ${init?.method ?? "GET"}`,
    );
  };

  try {
    const result = await runIterativeAgenticRag({
      runId: "offline-smoke",
      teamId: "cfda",
      outputDir: output,
      topicsPath: topics,
      qrelsDir: qrels,
      pyseriniBaseUrl: "http://mock.pyserini",
      pyseriniIndex: "climbmix-400b",
      pyseriniTokenEnv: "PYSERINI_API_TOKEN",
      initialDocs: 1,
      docsPerIteration: 1,
      maxDocumentsRead: 1,
      maxIterations: 1,
      documentReadLimit: 20,
      llm: {
        provider: "openai_llm",
        model: "mock-model",
        apiKeyEnv: "OPENAI_API_KEY",
        baseUrl: "http://mock.openai/v1",
      },
      env: { OPENAI_API_KEY: "test-only", PYSERINI_API_TOKEN: "test-only" },
      force: true,
    });

    assert.deepEqual(result.validation, {
      ok: true,
      output_count: 1,
      expected_count: 1,
    });
    assert.equal(llmCalls, 2);
    const rows = readFileSync(
      join(output, "rag_output_trec_rag_2026.jsonl"),
      "utf8",
    )
      .trim()
      .split("\n")
      .map((line) => JSON.parse(line));
    assert.equal(rows.length, 1);
    assert.deepEqual(rows[0].references, [DOCID]);
    assert.deepEqual(rows[0].answer[0].citations, [0]);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

function jsonResponse(value: unknown): Response {
  return new Response(JSON.stringify(value), {
    status: 200,
    headers: { "content-type": "application/json" },
  });
}
