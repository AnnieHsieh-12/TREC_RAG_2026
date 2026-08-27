# Local ML sidecar

The TypeScript controller calls this local Python service for:

- `POST /rerank`: MiniLM reranking with incoming-rank RRF;
- `POST /passages`: passage selection from fetched documents;
- `POST /sentence_evidence`: evidence excerpts for citation verification;
- `POST /aspect_coverage`: semantic checklist coverage for gap filling;
- `POST /llm`: an optional guarded, read-only Codex CLI bridge for non-official
  reruns.

Create a Python environment, install `requirements.lock`, copy
`.env.local.example` to `.env.local`, and start the service from this directory:

```bash
python -m src.sidecar
```

The default address is `http://127.0.0.1:8765`. Model and document caches are
downloaded or created locally and are excluded from Git.

Only the optional `/llm` endpoint requires an authenticated `codex`
executable. Confirm the session with `codex login status` before selecting
that alternative backend.
