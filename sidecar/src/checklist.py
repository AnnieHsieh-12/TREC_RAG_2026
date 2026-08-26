"""Generate a coverage checklist for ANY topic from its narrative alone.

Why this exists: the official nugget file (data/nuggets/rag25-dev-nuggets.jsonl)
only covers the 22 development topics - the organizers do not release nuggets
for the 119 official test topics (by design, so participants cannot tune
against the answer key). The M2 coverage gate therefore needs a checklist
source that works from nothing but the narrative text, which is the one thing
every topic (dev or test) always has.

Methodology note: this mirrors how the official dev nuggets were themselves
produced - the organizers used LLM-automated nugget creation (AutoNuggetizer;
arXiv 2411.09607, arXiv 2504.15068 report that fully-automatic nugget
creation/assignment correlates strongly with human evaluation). We decompose
at the sub-narrative level (what CoverageItem actually gates on) rather than
the individual-nugget level, since the gate needs "which aspects must be
covered", not an exhaustive fact list.

Output format is IDENTICAL to the official nugget file (one JSON object per
topic: {"qid": ..., "nuggets": [{"text", "mapped_sub_narrative",
"importance"}]}), so src/coverage.py's load_coverage_items() reads generated
checklists and official nuggets interchangeably - the gate does not know or
care which source it got.

Official nuggets remain the EVALUATION ground truth on dev topics; generated
checklists are what the gate runs on everywhere (dev included, so dev runs
mirror test-topic conditions).
"""

import json
import re

DECOMPOSE_SYSTEM = (
    "You decompose a research narrative into the distinct sub-aspects a "
    "complete answer must cover. You return strict JSON only - no prose, no "
    "markdown fences."
)

DECOMPOSE_USER_TEMPLATE = """Narrative:
{narrative}

Decompose this narrative into the distinct sub-aspects a complete, well-rounded
answer must cover. Return strict JSON with exactly this shape:

{{"sub_aspects": [
  {{"title": "<short noun-phrase name of the aspect, 3-8 words>",
    "importance": "vital" | "okay",
    "expected_facts": ["<one-sentence fact a good answer would state>", ...]}}
]}}

Rules:
- 5 to 12 sub_aspects, mutually distinct (no overlapping or duplicate aspects).
- "vital" = the narrative explicitly asks about it or an answer omitting it
  would clearly be incomplete; "okay" = secondary/supporting aspect.
- 2 to 4 expected_facts per aspect, each a single short declarative sentence.
- Titles must be specific to this narrative, not generic ("Economic impact of
  X on Y", not "Impacts").
- JSON only. No text before or after."""


def _extract_json(text):
    """Parse the model's reply as JSON, tolerating markdown fences or stray
    prose around the object. Raises ValueError if no parseable object found."""
    if not text:
        raise ValueError("empty model reply")
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        return json.loads(fenced.group(1))
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        return json.loads(text[start:end + 1])
    raise ValueError(f"no JSON object found in reply: {text[:200]}")


def _validate_sub_aspects(obj):
    """Check the decomposition JSON has the required shape; returns the
    cleaned sub_aspects list or raises ValueError."""
    aspects = obj.get("sub_aspects")
    if not isinstance(aspects, list) or not (3 <= len(aspects) <= 15):
        raise ValueError(f"sub_aspects missing or bad count: {type(aspects)} "
                         f"{len(aspects) if isinstance(aspects, list) else ''}")
    cleaned = []
    for a in aspects:
        title = (a.get("title") or "").strip()
        importance = a.get("importance")
        facts = a.get("expected_facts")
        if not title:
            raise ValueError(f"aspect missing title: {a}")
        if importance not in ("vital", "okay"):
            importance = "okay"
        if not isinstance(facts, list) or not facts:
            facts = [title]
        cleaned.append({"title": title, "importance": importance,
                        "expected_facts": [str(f).strip() for f in facts if str(f).strip()]})
    return cleaned


def generate_checklist(qid, narrative, nchc_client, model, max_attempts=3):
    """One LLM call (with parse-failure retries): narrative -> sub-aspect
    checklist, returned as an official-nugget-format dict:
    {"qid": qid, "nuggets": [{"text", "mapped_sub_narrative", "importance"}]}.
    Raises the last error if all attempts fail - callers decide whether a
    missing checklist is fatal for their run."""
    last_error = None
    for attempt in range(max_attempts):
        try:
            result = nchc_client.chat(
                model=model,
                messages=[
                    {"role": "system", "content": DECOMPOSE_SYSTEM},
                    {"role": "user",
                     "content": DECOMPOSE_USER_TEMPLATE.format(narrative=narrative)},
                ],
                temperature=0.0,
            )
            reply = result["choices"][0]["message"].get("content") or ""
            aspects = _validate_sub_aspects(_extract_json(reply))
            nuggets = []
            for a in aspects:
                for fact in a["expected_facts"]:
                    nuggets.append({
                        "text": fact,
                        # quoted to match the official file's convention
                        # (load_coverage_items strips the quotes either way)
                        "mapped_sub_narrative": f"\"{a['title']}\"",
                        "importance": a["importance"],
                    })
            return {"qid": qid, "nuggets": nuggets}
        except Exception as e:  # parse errors and malformed replies retried
            last_error = e
            print(f"  [checklist] qid {qid} attempt {attempt + 1} failed: {e}")
    raise RuntimeError(
        f"checklist generation failed for qid {qid} after {max_attempts} attempts"
    ) from last_error


def generate_checklists_file(topics, output_path, nchc_client, model,
                             pause_seconds=2):
    """Generate checklists for [(qid, narrative), ...] into one JSONL file
    (official nugget file format, one line per topic). Resumable: topics whose
    qid already appears in the output file are skipped."""
    import os
    import time

    done = set()
    if os.path.exists(output_path):
        with open(output_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    done.add(json.loads(line)["qid"])
    if done:
        print(f"Resuming: {len(done)} checklist(s) already generated")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    written = 0
    with open(output_path, "a", encoding="utf-8") as out:
        for qid, narrative in topics:
            if qid in done:
                continue
            record = generate_checklist(qid, narrative, nchc_client, model)
            subs = {n["mapped_sub_narrative"] for n in record["nuggets"]}
            vitals = {n["mapped_sub_narrative"] for n in record["nuggets"]
                      if n["importance"] == "vital"}
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            written += 1
            print(f"  qid {qid}: {len(subs)} sub-aspects "
                  f"({len(vitals)} vital), {len(record['nuggets'])} facts")
            time.sleep(pause_seconds)
    print(f"Wrote {written} new checklist(s) to {output_path}")
