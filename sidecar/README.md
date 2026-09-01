# Local ML sidecar

The TypeScript controller calls this local Python service for:

- `POST /rerank`: MiniLM reranking with incoming-rank RRF;
- `POST /passages`: passage selection from fetched documents;
- `POST /sentence_evidence`: evidence excerpts for citation verification;
- `POST /aspect_coverage`: semantic checklist coverage for gap filling;
- `POST /llm`: an isolated, tool-disabled Codex CLI bridge used by the RAG
  launcher.

Create a Python environment, install `../requirements.lock`, copy
`.env.local.example` to `.env.local`, and start the service from this directory:

```bash
python -m src.sidecar
```

The default address is `http://127.0.0.1:8765`. Model and document caches are
downloaded or created locally and are excluded from Git.

The `/llm` endpoint requires an authenticated `codex` executable. Confirm the
session with `codex login status` before running the final RAG pipeline.

Retrieved documents are untrusted. Each Codex request runs in a new empty
workspace with pipeline credentials removed from its environment. The bridge
disables shell, browser, app, MCP, plugin, hook, skill, and related tool
surfaces; ignores user configuration and rules; uses an ephemeral read-only
session; and permits no approval escalation. Unsupported CLI versions fail
closed. Keep this service on `127.0.0.1`; do not expose it as a shared network
endpoint.
