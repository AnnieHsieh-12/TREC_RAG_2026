import os

from dotenv import load_dotenv


def load_token():
    load_dotenv(".env.local")
    token = os.environ.get("PYSERINI_API_TOKEN")
    if not token:
        raise RuntimeError(
            "PYSERINI_API_TOKEN not found. Add it to .env.local (see .env.local.example)."
        )
    return token


def load_nchc_key():
    load_dotenv(".env.local")
    key = os.environ.get("NCHC_API_KEY")
    if not key:
        raise RuntimeError(
            "NCHC_API_KEY not found. Add it to .env.local as NCHC_API_KEY=<key>."
        )
    return key


def load_topics(path):
    topics = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            qid, text = line.split("\t", 1)
            topics.append((qid, text))
    return topics


def format_run_line(qid, docid, rank, score, run_id):
    return f"{qid} Q0 {docid} {rank} {score} {run_id}"


def write_run_file(path, lines):
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")
