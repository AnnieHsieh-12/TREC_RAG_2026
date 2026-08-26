import json
import time

# TraceEvent shape and stop_reason vocabulary per docs/ROADMAP.md M0. Tracing
# is diagnostic only - it must never be the reason an actual run fails, so
# every write is best-effort and swallows its own errors.

STOP_REASONS = {
    "coverage_complete",
    "budget_exhausted_with_gaps",
    # M2: the round budget ran out, but every vital coverage item was already
    # answered/blocked (0 evidence_found, 0 unsearched) - the agent just
    # never got around to calling finish_answer. Distinct from genuine
    # unanswered gaps, found live during M2 testing on topic 14 (all 9
    # official sub-narratives were answered by round 19 of 20, but the model
    # spent its last round writing one more sentence instead of finishing).
    "budget_exhausted_all_covered",
    "no_tool_call",
    "model_error",
    "cancelled",
}


class TraceWriter:
    def __init__(self, path, run_id, mode="a"):
        self.run_id = run_id
        self._seq = 0
        self._fh = open(path, mode, encoding="utf-8")

    def emit(self, event_type, narrative_id, round=None, **fields):
        self._seq += 1
        event = {
            "run_id": self.run_id,
            "narrative_id": narrative_id,
            "seq": self._seq,
            "round": round,
            "event_type": event_type,
            "timestamp": time.time(),
        }
        event.update(fields)
        try:
            self._fh.write(json.dumps(event, ensure_ascii=False) + "\n")
            self._fh.flush()
        except Exception as e:
            print(f"  [trace] warning: failed to write trace event: {e}")

    def close(self):
        try:
            self._fh.close()
        except Exception:
            pass


class NullTracer:
    """No-op tracer, used when a caller doesn't pass a TraceWriter."""

    def emit(self, *args, **kwargs):
        pass

    def close(self):
        pass
