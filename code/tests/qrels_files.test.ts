import assert from "node:assert/strict";
import test from "node:test";

import { requireQrelsTopicOverlap } from "../src/evaluation/qrels_files";

test("qrels must overlap the input topics", () => {
  assert.deepEqual(
    requireQrelsTopicOverlap(new Map([["2", {}]]), ["1", "2"], "dev.qrels"),
    ["2"],
  );
  assert.throws(
    () =>
      requireQrelsTopicOverlap(
        new Map([["unrelated", {}]]),
        ["1", "2"],
        "wrong.qrels",
      ),
    /no topic IDs in common/,
  );
});
