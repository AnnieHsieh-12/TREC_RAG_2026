import type { NchcLlmConfig } from "./config";
import { errorMessageWithRedactedSecrets, LlmConfigError, LlmProviderError } from "./errors";
import type { LlmClient, LlmGenerateOptions, LlmGenerateResult, LlmMessage, LlmUsage } from "./types";

export const NCHC_CHAT_COMPLETIONS_PATH = "/chat/completions";

type NchcChatCompletionsResponse = {
  id?: unknown;
  choices?: unknown;
  usage?: unknown;
};

export class NchcLlmClient implements LlmClient {
  readonly provider = "nchc_llm" as const;
  readonly model: string;
  private readonly baseUrl: string;
  private readonly apiKeyEnv: string;
  private readonly env: NodeJS.ProcessEnv;

  constructor(config: NchcLlmConfig, env: NodeJS.ProcessEnv = process.env) {
    this.model = config.model;
    this.baseUrl = trimTrailingSlash(config.baseUrl);
    this.apiKeyEnv = config.apiKeyEnv;
    this.env = env;
  }

  async generate(options: LlmGenerateOptions): Promise<LlmGenerateResult> {
    const apiKey = this.env[this.apiKeyEnv]?.trim();
    if (!apiKey) throw new LlmConfigError(`${this.apiKeyEnv} must be set for nchc_llm.`);

    const started = Date.now();
    const url = `${this.baseUrl}${NCHC_CHAT_COMPLETIONS_PATH}`;
    let response: Response;
    try {
      response = await fetch(url, {
        method: "POST",
        headers: {
          "content-type": "application/json",
          "x-api-key": apiKey,
        },
        body: JSON.stringify({
          model: this.model,
          messages: options.messages.map(toOpenAiMessage),
          temperature: options.temperature,
          max_tokens: options.maxTokens,
          // NCHC gpt-oss currently corrupts some outputs when OpenAI-style response_format is sent.
          // Keep JSON enforcement in prompts and validators instead of passing response_format.

        }),
        signal: options.signal,
      });
    } catch (error) {
      throw new LlmProviderError(
        `nchc_llm request failed: ${errorMessageWithRedactedSecrets(error, this.env)}`,
      );
    }

    const responseText = await response.text();
    if (!response.ok) {
      throw new LlmProviderError(
        `nchc_llm HTTP ${response.status}: ${errorMessageWithRedactedSecrets(responseText, this.env)}`,
      );
    }

    let parsed: NchcChatCompletionsResponse;
    try {
      parsed = JSON.parse(responseText) as NchcChatCompletionsResponse;
    } catch (error) {
      throw new LlmProviderError(
        `nchc_llm returned malformed JSON: ${errorMessageWithRedactedSecrets(error, this.env)}`,
      );
    }

    const text = extractAssistantText(parsed);
    if (text.trim().length === 0) throw new LlmProviderError("nchc_llm returned an empty assistant message.");

    return {
      provider: this.provider,
      model: this.model,
      text,
      latencyMs: Date.now() - started,
      ...(typeof parsed.id === "string" ? { requestId: parsed.id } : {}),
      ...(normalizeUsage(parsed.usage) ? { usage: normalizeUsage(parsed.usage) } : {}),
    };
  }
}

function toOpenAiMessage(message: LlmMessage): { role: string; content: string } {
  return { role: message.role, content: message.content };
}

function trimTrailingSlash(value: string): string {
  return value.replace(/\/+$/, "");
}

function extractAssistantText(value: NchcChatCompletionsResponse): string {
  if (!Array.isArray(value.choices) || value.choices.length === 0) return "";
  const first = value.choices[0];
  if (!isRecord(first)) return "";
  const message = first.message;
  if (isRecord(message) && typeof message.content === "string") return message.content;
  if (typeof first.text === "string") return first.text;
  return "";
}

function normalizeUsage(value: unknown): LlmUsage | undefined {
  if (!isRecord(value)) return undefined;
  const usage: LlmUsage = {};
  if (typeof value.prompt_tokens === "number") usage.inputTokens = value.prompt_tokens;
  if (typeof value.completion_tokens === "number") usage.outputTokens = value.completion_tokens;
  if (typeof value.total_tokens === "number") usage.totalTokens = value.total_tokens;
  return Object.keys(usage).length > 0 ? usage : undefined;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
