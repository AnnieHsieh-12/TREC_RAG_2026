import assert from "node:assert/strict";
import { mkdtempSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

import { FINAL_POLICY } from "../config/final_pipeline";
import { runIterativeAgenticRag } from "../src/trec-rag-2026/agentic-rag/iterative_runner";

const DOCID = "shard_00999_42";

test("final Retrieval orchestration completes one mocked topic", async () => {
  const root = mkdtempSync(join(tmpdir(), "cfda-retrieval-smoke-"));
  const topics = join(root, "topics.tsv");
  const qrels = join(root, "qrels");
  const output = join(root, "output");
  writeFileSync(topics, "1\tExplain the verified retrieval fact.\n");
  mkdirSync(qrels);
  writeFileSync(join(qrels, "fixture.qrels"), `1 0 ${DOCID} 2\n`);

  const originalFetch = globalThis.fetch;
  globalThis.fetch = async (input) => {
    const url = String(input);
    if (url.includes("/search?")) {
      return jsonResponse({ candidates: [{ docid: DOCID, score: 10 }] });
    }
    if (url.includes("/doc/")) {
      return jsonResponse({
        doc: { text: "The verified retrieval fact appears here." },
      });
    }
    if (url === "http://mock.nchc/chat/completions") {
      return jsonResponse({
        choices: [{ message: { content: '{"enough":true,"queries":[]}' } }],
      });
    }
    throw new Error(`Unexpected Retrieval smoke-test request: ${url}`);
  };

  try {
    const result = await runIterativeAgenticRag({
      runId: "retrieval-smoke",
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
        provider: "nchc_llm",
        model: "mock-model",
        apiKeyEnv: "NCHC_API_KEY",
        baseUrl: "http://mock.nchc",
      },
      env: {
        NCHC_API_KEY: "test-only",
        PYSERINI_API_TOKEN: "test-only",
        R_ONLY: "1",
      },
      force: true,
      policy: {
        ...FINAL_POLICY,
        q2d_enabled: false,
        rerank_depth: 0,
        fusion_dense: false,
        facet_queries: false,
        per_aspect_generation: false,
        comprehensive_answer: false,
        reflection: false,
        citation_verify: false,
        llm_revise: false,
        reattribute: false,
      },
    });

    assert.deepEqual(result.validation, {
      ok: true,
      output_count: 1,
      expected_count: 1,
    });
    const pool = readFileSync(
      join(output, "candidate_pool_top5000.trec"),
      "utf8",
    );
    assert.match(pool, new RegExp(`^1 Q0 ${DOCID} 1 `));
    const metrics = JSON.parse(
      readFileSync(join(output, "metrics.json"), "utf8"),
    );
    assert.equal(metrics.qrels[0].qrels_filename, "fixture.qrels");
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
