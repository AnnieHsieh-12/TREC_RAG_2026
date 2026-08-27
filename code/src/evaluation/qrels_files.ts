import { createHash } from "node:crypto";
import { readFileSync, readdirSync } from "node:fs";
import { basename, join, resolve } from "node:path";

/** Return every qrels file in a directory, in deterministic filename order. */
export function discoverQrelsFiles(directory: string): string[] {
  const root = resolve(directory);
  const files = readdirSync(root, { withFileTypes: true })
    .filter((entry) => entry.isFile() && entry.name.endsWith(".qrels"))
    .map((entry) => join(root, entry.name))
    .sort();

  if (files.length === 0) {
    throw new Error(`No .qrels files found in ${root}`);
  }
  return files;
}

/** Record which optional diagnostic judgments were used without copying them. */
export function describeQrelsFiles(paths: string[]) {
  return paths.map((path) => ({
    filename: basename(path),
    sha256: createHash("sha256").update(readFileSync(path)).digest("hex"),
  }));
}
