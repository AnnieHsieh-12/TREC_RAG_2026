import type { CodexLlmConfig } from "./config";
import { LlmProviderError } from "./errors";
import type { LlmClient, LlmGenerateOptions, LlmGenerateResult, LlmMessage } from "./types";

// Controller on ChatGPT Plus subscription quota via the local sidecar's
// /llm endpoint (which shells out to the Codex CLI). W5 user directive:
// dev-loop generation must not spend the OpenAI API key.
export class CodexLlmClient implements LlmClient {
  readonly provider = "codex_llm" as const;
  readonly model: string;
  private readonly baseUrl: string;

  constructor(config: CodexLlmConfig) {
    this.model = config.model;
    this.baseUrl = config.baseUrl.replace(/\/+$/, "");
  }

  async generate(options: LlmGenerateOptions): Promise<LlmGenerateResult> {
    const started = Date.now();
    // codex exec takes a single prompt; fold the message list into one text.
    const prompt = options.messages
      .map((m: LlmMessage) => (m.role === "user" ? m.content : `[${m.role}]\n${m.content}`))
      .join("\n\n");
    let response: Response;
    try {
      const ac = new AbortController();
      const timer = setTimeout(() => ac.abort(), 320000);
      try {
        response = await fetch(`${this.baseUrl}/llm`, {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ prompt, model: this.model }),
          signal: options.signal ?? ac.signal,
        });
      } finally {
        clearTimeout(timer);
      }
    } catch (error) {
      throw new LlmProviderError(
        `codex_llm sidecar request failed: ${error instanceof Error ? error.message : String(error)}`,
      );
    }
    const responseText = await response.text();
    if (!response.ok) {
      throw new LlmProviderError(`codex_llm sidecar HTTP ${response.status}: ${responseText.slice(0, 300)}`);
    }
    let parsed: { text?: unknown; calls_total?: unknown };
    try {
      parsed = JSON.parse(responseText) as { text?: unknown; calls_total?: unknown };
    } catch {
      throw new LlmProviderError("codex_llm sidecar returned malformed JSON.");
    }
    const text = typeof parsed.text === "string" ? parsed.text : "";
    if (text.trim().length === 0) throw new LlmProviderError("codex_llm returned an empty assistant message.");
    return {
      provider: this.provider,
      model: this.model,
      text,
      latencyMs: Date.now() - started,
      ...(typeof parsed.calls_total === "number" ? { requestId: `codex-call-${parsed.calls_total}` } : {}),
    };
  }
}
