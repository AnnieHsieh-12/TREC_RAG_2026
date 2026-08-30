import assert from "node:assert/strict";
import test from "node:test";

import { trimToWordCap } from "../src/trec-rag-2026/rag-pipeline/runner";
import type {
  AgenticRagOutputObject,
  RagRunConfig,
  TopicIdentity,
} from "../src/trec-rag-2026/shared-rag/contracts";
import { validateOfficialRagOutputObject } from "../src/trec-rag-2026/shared-rag/validation";

const config: RagRunConfig = {
  runId: "test-run",
  teamId: "test-team",
  mode: "automatic",
  promptVersion: "agentic_rag_baseline_v1",
  runDesc: "trim test",
};
const topic: TopicIdentity = {
  qid: "rag2026-test",
  title: "",
  narrative: "Test narrative",
};
const docid = "shard_00001_1";

test("a single oversized answer item is trimmed to the word budget", () => {
  const rag: AgenticRagOutputObject = {
    metadata: {
      type: "automatic",
      run_id: config.runId,
      team_id: config.teamId,
      narrative_id: topic.qid,
      narrative: topic.narrative,
      title: topic.title,
      prompt: config.promptVersion,
      run_desc: config.runDesc,
    },
    references: [docid],
    answer: [
      {
        text: Array(1_200).fill("evidence").join(" "),
        citations: [0],
      },
    ],
  };

  const trimmed = trimToWordCap(rag, config, topic, new Set([docid]), 1_020);
  const words = trimmed.answer.reduce(
    (sum, item) => sum + item.text.split(/\s+/).filter(Boolean).length,
    0,
  );

  assert.equal(words, 1_020);
  assert.match(trimmed.answer[0].text, /\.$/);
  assert.deepEqual(trimmed.answer[0].citations, [0]);
  assert.equal(validateOfficialRagOutputObject(trimmed).ok, true);
});
