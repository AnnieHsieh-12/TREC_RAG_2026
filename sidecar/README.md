# Local ML sidecar

The TypeScript controller calls this local Python service for:

- `POST /rerank`: MiniLM reranking with incoming-rank RRF;
- `POST /passages`: passage selection from fetched documents;
- `POST /sentence_evidence`: evidence excerpts for citation verification.

Create a Python environment, install `requirements.txt`, copy
`.env.local.example` to `.env.local`, and start the service from this directory:

```bash
python -m src.sidecar
```

The default address is `http://127.0.0.1:8765`. Model and document caches are
downloaded or created locally and are excluded from Git.
