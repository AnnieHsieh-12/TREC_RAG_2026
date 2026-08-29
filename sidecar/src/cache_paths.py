"""Safe cache paths for untrusted HTTP identifiers."""

import hashlib
import os
import re


SAFE_QID = re.compile(r"[A-Za-z0-9._-]+")


def doc_cache_path(root: str, docid: object) -> str:
    digest = hashlib.sha256(str(docid).encode("utf-8")).hexdigest()
    return os.path.join(root, digest[:2], f"{digest}.txt")


def raw_pool_path(root: str, qid: object, suffix: str) -> str | None:
    value = str(qid)
    if SAFE_QID.fullmatch(value) is None:
        return None
    return os.path.join(root, f"{value}{suffix}.json")
