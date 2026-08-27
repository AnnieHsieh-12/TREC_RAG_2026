import assert from "node:assert/strict";
import { mkdtempSync, mkdirSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

import { discoverQrelsFiles } from "../src/evaluation/qrels_files";

test("qrels discovery accepts arbitrary .qrels filenames", () => {
  const root = mkdtempSync(join(tmpdir(), "cfda-qrels-"));
  writeFileSync(join(root, "z.qrels"), "");
  writeFileSync(join(root, "a.qrels"), "");
  writeFileSync(join(root, "notes.txt"), "");
  assert.deepEqual(
    discoverQrelsFiles(root).map((path) => path.split("/").at(-1)),
    ["a.qrels", "z.qrels"],
  );
});

test("qrels discovery fails clearly when the directory has no qrels", () => {
  const root = mkdtempSync(join(tmpdir(), "cfda-qrels-empty-"));
  mkdirSync(join(root, "nested"));
  assert.throws(() => discoverQrelsFiles(root), /No \.qrels files found/);
});
