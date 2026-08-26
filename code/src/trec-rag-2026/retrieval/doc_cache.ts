// Read-through disk cache for complete documents returned by Pyserini.
// Store full text here and apply per-run truncation in the caller so runs with
// different document-read limits can share the same cached response.
import { createHash } from "node:crypto";
import {
  existsSync,
  mkdirSync,
  readFileSync,
  renameSync,
  writeFileSync,
} from "node:fs";
import { join } from "node:path";

export type CachedDoc = { found: boolean; text: string };

const ROOT = process.env.DOC_CACHE_DIR || join(".cache", "docs");
const ENABLED = process.env.DOC_CACHE !== "0";

function entryPath(index: string, docid: string): string {
  const hash = createHash("sha1")
    .update(`${index}\0${docid}`)
    .digest("hex");
  const safeIndex = index.replace(/[^A-Za-z0-9._-]/g, "_");
  return join(ROOT, safeIndex, hash.slice(0, 2), `${hash}.json`);
}

export function readCache(index: string, docid: string): CachedDoc | null {
  if (!ENABLED) return null;
  const path = entryPath(index, docid);
  if (!existsSync(path)) return null;
  try {
    const value = JSON.parse(readFileSync(path, "utf8")) as CachedDoc;
    if (typeof value?.found !== "boolean" || typeof value?.text !== "string") {
      return null;
    }
    return value;
  } catch {
    return null;
  }
}

export function writeCache(
  index: string,
  docid: string,
  doc: CachedDoc,
): void {
  if (!ENABLED) return;
  const path = entryPath(index, docid);
  try {
    mkdirSync(join(path, ".."), { recursive: true });
    const temporaryPath = `${path}.${process.pid}.tmp`;
    writeFileSync(temporaryPath, JSON.stringify(doc), "utf8");
    renameSync(temporaryPath, path);
  } catch {
    // The cache is an optimization; a write failure must not stop the run.
  }
}
