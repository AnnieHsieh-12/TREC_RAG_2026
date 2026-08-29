import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  validateOfficialRagOutputObject,
  validateRagOutputObject,
  validateRagOutputObjectStrict,
} from "../src/trec-rag-2026/shared-rag/validation";

const fixture = JSON.parse(
  readFileSync(
    new URL("../test-fixtures/rag_validation_cases.json", import.meta.url),
    "utf8",
  ),
);
const args = {
  config: {
    runId: "test-run",
    teamId: "pi-serini",
    mode: "automatic" as const,
    promptVersion: "agentic_rag_baseline_v1",
    runDesc: "test fixture",
  },
  topic: { qid: "1", title: "", narrative: "Example narrative" },
  readDocids: new Set(fixture.references as string[]),
};

function clone<T>(value: T): T {
  return JSON.parse(JSON.stringify(value));
}

test("official rules accept empty citations, uncited references, and extra metadata", () => {
  assert.deepEqual(validateOfficialRagOutputObject(fixture), {
    ok: true,
    issues: [],
  });
});

test("direct-docid citations are accepted and canonicalized to indices", () => {
  const value = clone(fixture);
  value.answer[0].citations = ["shard_00002_2"];
  assert.equal(validateOfficialRagOutputObject(value).ok, true);
  assert.deepEqual(validateRagOutputObject(value, args).answer[0].citations, [
    1,
  ]);
});

test("1024 words pass and 1025 words fail", () => {
  const pass = clone(fixture);
  pass.answer = [{ text: Array(1024).fill("word").join(" "), citations: [] }];
  assert.equal(validateOfficialRagOutputObject(pass).ok, true);
  const fail = clone(pass);
  fail.answer[0].text += " word";
  assert.ok(
    validateOfficialRagOutputObject(fail).issues.some(
      (issue) => issue.code === "ANSWER_TOO_LONG",
    ),
  );
});

test("invalid docids and out-of-range citations fail", () => {
  const badDocid = clone(fixture);
  badDocid.references[0] = "climbmix-1";
  assert.ok(
    validateOfficialRagOutputObject(badDocid).issues.some(
      (issue) => issue.code === "INVALID_CLIMBMIX_DOCID",
    ),
  );
  const badCitation = clone(fixture);
  badCitation.answer[0].citations = [99];
  assert.ok(
    validateOfficialRagOutputObject(badCitation).issues.some(
      (issue) => issue.code === "CITATION_OUT_OF_RANGE",
    ),
  );
});

test("producer identity checks are separate from official shape checks", () => {
  const wrong = clone(fixture);
  wrong.metadata.run_id = "other-run";
  assert.equal(validateOfficialRagOutputObject(wrong).ok, true);
  const result = validateRagOutputObjectStrict(wrong, args);
  assert.ok(result.issues.some((issue) => issue.code === "RUN_ID_MISMATCH"));
});

test("the organizer limit is at most three citations per answer item", () => {
  const value = clone(fixture);
  value.answer[0].citations = [0, 1, 2, "shard_00001_1"];
  assert.ok(
    validateOfficialRagOutputObject(value).issues.some(
      (issue) => issue.code === "TOO_MANY_CITATIONS",
    ),
  );
});

test("each answer item contains exactly one sentence", () => {
  const value = clone(fixture);
  value.answer[0].text = "First factual sentence. Second factual sentence.";
  assert.ok(
    validateOfficialRagOutputObject(value).issues.some(
      (issue) => issue.code === "MULTIPLE_SENTENCES_IN_ITEM",
    ),
  );
});
