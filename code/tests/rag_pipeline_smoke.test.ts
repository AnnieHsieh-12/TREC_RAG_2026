import assert from "node:assert/strict";
import {
  existsSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

import { runFinalRagPipeline } from "../src/trec-rag-2026/rag-pipeline/runner";

const DOCID = "shard_00001_1";
test("final RAG pipeline completes without qrels", async () => {
  const root = mkdtempSync(join(tmpdir(), "cfda-rag-smoke-"));
  const topics = join(root, "topics.tsv");
  const output = join(root, "output");
  writeFileSync(topics, "1\tExplain the documented fact.\n");

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
      const body = JSON.parse(String(init?.body ?? "{}"));
      assert.equal("reasoning_effort" in body, false);
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
    const result = await runFinalRagPipeline({
      runId: "offline-smoke",
      teamId: "cfda",
      outputDir: output,
      topicsPath: topics,
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
    assert.equal(existsSync(join(output, "metrics.json")), false);
    assert.equal(existsSync(join(output, "per_topic_metrics.json")), false);

    // A resume-only assembly must describe the durable completed run, not
    // report zero work simply because the topic was skipped in this process.
    await runFinalRagPipeline({
      runId: "offline-smoke",
      teamId: "cfda",
      outputDir: output,
      topicsPath: topics,
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
      resume: true,
    });
    const summary = JSON.parse(
      readFileSync(join(output, "run-summary.internal.json"), "utf8"),
    );
    assert.equal(summary.processed_count, 1);
    assert.equal(summary.llm_call_count, 2);
    assert.equal(summary.iterations_by_topic["1"], 1);
    assert.notEqual(summary.stop_reason_by_topic["1"], undefined);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("RAG rejects unrelated qrels before replacing output", async () => {
  const root = mkdtempSync(join(tmpdir(), "cfda-rag-qrels-preflight-"));
  const topics = join(root, "topics.tsv");
  const qrels = join(root, "qrels");
  const output = join(root, "output");
  writeFileSync(topics, "1\tExample narrative.\n");
  mkdirSync(qrels);
  mkdirSync(output);
  writeFileSync(join(qrels, "unrelated.qrels"), "other 0 shard_00001_1 2\n");
  writeFileSync(join(output, "sentinel"), "keep");

  await assert.rejects(
    runFinalRagPipeline({
      runId: "preflight",
      teamId: "cfda",
      outputDir: output,
      topicsPath: topics,
      qrelsDir: qrels,
      pyseriniBaseUrl: "http://unused",
      pyseriniIndex: "climbmix-400b",
      pyseriniTokenEnv: "PYSERINI_API_TOKEN",
      initialDocs: 1,
      docsPerIteration: 1,
      maxDocumentsRead: 1,
      maxIterations: 1,
      documentReadLimit: 1,
      llm: { provider: "openai_llm", model: "unused" },
      force: true,
    }),
    /no topic IDs in common/,
  );
  assert.equal(readFileSync(join(output, "sentinel"), "utf8"), "keep");
});

function jsonResponse(value: unknown): Response {
  return new Response(JSON.stringify(value), {
    status: 200,
    headers: { "content-type": "application/json" },
  });
}
