import assert from "node:assert/strict";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

import { loadChecklist } from "../src/trec-rag-2026/rag-pipeline/runner";

function fixture(contents: string) {
  const root = mkdtempSync(join(tmpdir(), "cfda-checklist-"));
  const path = join(root, "checklist.jsonl");
  writeFileSync(path, contents);
  return path;
}

test("checklist covers every selected topic exactly once", () => {
  const valid = fixture(
    '{"qid":"1","items":[" first aspect "]}\n' +
      '{"qid":"2","items":["second aspect"]}\n',
  );
  assert.deepEqual(loadChecklist(valid, ["1"]).get("1"), ["first aspect"]);

  assert.throws(
    () =>
      loadChecklist(fixture('{"qid":"1","items":["aspect"]}\n'), ["1", "2"]),
    /missing 1 selected topic/,
  );
  assert.throws(
    () =>
      loadChecklist(
        fixture('{"qid":"1","items":["a"]}\n' + '{"qid":"1","items":["b"]}\n'),
        ["1"],
      ),
    /Duplicate checklist qid/,
  );
  assert.deepEqual(
    loadChecklist(
      fixture(
        '{"qid":"1","items":["a"]}\n' + '{"qid":"other-shard","items":["b"]}\n',
      ),
      ["1"],
    ).get("1"),
    ["a"],
  );
  assert.throws(
    () => loadChecklist(fixture('{"qid":"1","items":[]}\n'), ["1"]),
    /non-empty array/,
  );
});
