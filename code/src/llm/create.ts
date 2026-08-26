import type { LlmClientConfig } from "./config";
import { LlmConfigError, LlmJsonParseError, redactSecrets } from "./errors";
import { NchcLlmClient } from "./nchc_llm";
import { OpenAiLlmClient } from "./openai_llm";
import { CodexLlmClient } from "./codex_llm";
import type {
  GenerateJsonWithRetryOptions,
  GenerateJsonWithRetryResult,
  LlmClient,
  LlmGenerateResult,
  LlmMessage,
  LlmUsage,
  LlmAttemptTrace,
} from "./types";

export function createLlmClient(config: LlmClientConfig, env: NodeJS.ProcessEnv = process.env): LlmClient {
  if (config.provider === "nchc_llm") return new NchcLlmClient(config, env);
  if (config.provider === "openai_llm") return new OpenAiLlmClient(config, env);
  if (config.provider === "codex_llm") return new CodexLlmClient(config);
  throw new LlmConfigError(`Unsupported LLM provider: ${(config as { provider?: unknown }).provider}`);
}

export async function generateJsonWithRetry<T>(
  options: GenerateJsonWithRetryOptions<T>,
): Promise<GenerateJsonWithRetryResult<T>> {
  const requestIds: string[] = [];
  let totalLatencyMs = 0;
  let combinedUsage: LlmUsage | undefined;
  const attemptTrace: LlmAttemptTrace[] = [];
  let firstFailure = "";
  let lastRawText = "";
  let globalAttempt = 0;
  const maxJsonAttempts = 1 + (options.maxJsonRepairAttempts ?? 2);
  const maxRequestAttempts = 1 + (options.maxRequestRetries ?? 2);

  for (let jsonAttempt = 1; jsonAttempt <= maxJsonAttempts; jsonAttempt += 1) {
    const messages = jsonAttempt === 1 ? options.messages : buildCorrectionMessages(options.messages, firstFailure);
    let result: LlmGenerateResult | null = null;

    for (let requestAttempt = 1; requestAttempt <= maxRequestAttempts; requestAttempt += 1) {
      globalAttempt += 1;
      const started = Date.now();
      try {
        result = await options.client.generate({
          messages,
          temperature: options.temperature,
          maxTokens: options.maxTokens,
          responseFormat: "json_object",
          signal: options.signal,
        });
        totalLatencyMs += result.latencyMs;
        if (result.requestId) requestIds.push(result.requestId);
        combinedUsage = mergeUsage(combinedUsage, result.usage);
        break;
      } catch (error) {
        const trace = buildFailedProviderAttemptTrace(globalAttempt, options.client, Date.now() - started, error);
        attemptTrace.push(trace);
        options.onAttempt?.(trace);
        if (requestAttempt >= maxRequestAttempts || !isRetryableProviderError(error)) {
          throw new LlmJsonParseError(trace.errorCode ?? "LLM_PROVIDER_FAILED", {
            stage: options.stage,
            attempts: globalAttempt,
            reason: redactSecrets(error instanceof Error ? error.message : String(error)),
            attempt_trace: attemptTrace,
          });
        }
        await sleep(backoffMs(requestAttempt));
      }
    }

    if (result === null) continue;
    lastRawText = result.text;
    const parsed = parseStrictJsonObject(result.text);
    if (!parsed.ok) {
      firstFailure = redactSecrets(parsed.message);
      const trace = buildAttemptTrace(globalAttempt, result, false, "LLM_JSON_PARSE_FAILED");
      attemptTrace.push(trace);
      options.onAttempt?.(trace);
      if (jsonAttempt === maxJsonAttempts) throw jsonFailure(options.stage, firstFailure, result.text, globalAttempt, attemptTrace);
      continue;
    }
    const validation = options.validate(parsed.value);
    if (!validation.ok) {
      firstFailure = redactSecrets(validation.message);
      const trace = buildAttemptTrace(globalAttempt, result, false, "LLM_JSON_PARSE_FAILED");
      attemptTrace.push(trace);
      options.onAttempt?.(trace);
      if (jsonAttempt === maxJsonAttempts) {
        throw new LlmJsonParseError("LLM_JSON_PARSE_FAILED", {
          stage: options.stage,
          attempts: globalAttempt,
          reason: redactSecrets(validation.message),
          attempt_trace: attemptTrace,
          ...validation.details,
        });
      }
      continue;
    }
    const trace = buildAttemptTrace(globalAttempt, result, true);
    attemptTrace.push(trace);
    options.onAttempt?.(trace);
    return {
      value: validation.value,
      attempts: globalAttempt,
      rawText: result.text,
      provider: result.provider,
      model: result.model,
      latencyMs: totalLatencyMs,
      requestIds,
      ...(combinedUsage ? { usage: combinedUsage } : {}),
      attemptTrace,
    };
  }

  throw jsonFailure(options.stage, firstFailure || "No valid JSON object produced.", lastRawText, globalAttempt, attemptTrace);
}

function buildCorrectionMessages(originalMessages: LlmMessage[], failure: string): LlmMessage[] {
  return [
    ...originalMessages,
    {
      role: "assistant",
      content: "The previous response was invalid JSON or failed schema validation.",
    },
    {
      role: "user",
      content: [
        "Correct the previous response.",
        `Validation/parsing problem: ${failure}`,
        "Return only one strict JSON object. Do not include Markdown fences, comments, prose, or explanations.",
      ].join("\n"),
    },
  ];
}

function parseStrictJsonObject(text: string): { ok: true; value: unknown } | { ok: false; message: string } {
  const candidate = extractJsonCandidate(text);
  try {
    const parsed = parsePossiblyStringEncodedJson(candidate);
    if (!isRecord(parsed)) return { ok: false, message: "Parsed JSON value is not an object." };
    return { ok: true, value: parsed };
  } catch (firstError) {
    try {
      const repaired = repairMalformedQuotedObject(candidate);
      if (repaired !== null) {
        const parsed = parsePossiblyStringEncodedJson(repaired);
        if (isRecord(parsed)) return { ok: true, value: parsed };
      }
    } catch {
    }
    return { ok: false, message: firstError instanceof Error ? firstError.message : String(firstError) };
  }
}

function extractJsonCandidate(text: string): string {
  const fenced = /```(?:json)?\s*([\s\S]*?)```/i.exec(text);
  if (fenced) return fenced[1].trim();
  return text.trim();
}

function parsePossiblyStringEncodedJson(candidate: string): unknown {
  let parsed: unknown = JSON.parse(candidate);
  for (let depth = 0; depth < 2 && typeof parsed === "string"; depth += 1) {
    const nested = extractJsonCandidate(parsed.trim());
    if (!nested.startsWith("{") && !nested.startsWith("[")) break;
    parsed = JSON.parse(nested);
  }
  return parsed;
}

function repairMalformedQuotedObject(candidate: string): string | null {
  const trimmed = candidate.trim();
  if (!trimmed.startsWith('{"')) return null;
  const withoutPrefix = trimmed.slice(2);
  const withoutTrailingQuote = withoutPrefix.endsWith('"') ? withoutPrefix.slice(0, -1) : withoutPrefix;
  const inner = withoutTrailingQuote.replace(/\\"/g, '"');
  if (inner.startsWith("{") && inner.endsWith("}")) return inner;
  if (inner.startsWith('"') && inner.endsWith("}")) return `{${inner}`;
  if (inner.startsWith('"')) return `{${inner}}`;
  return null;
}

function jsonFailure(
  stage: string | undefined,
  reason: string,
  rawText: string,
  attempts: number,
  attemptTrace: LlmAttemptTrace[],
): LlmJsonParseError {
  return new LlmJsonParseError("LLM_JSON_PARSE_FAILED", {
    stage,
    attempts,
    reason,
    raw_output_chars: rawText.length,
    attempt_trace: attemptTrace,
  });
}

function buildAttemptTrace(
  attempt: number,
  result: LlmGenerateResult,
  success: boolean,
  errorCode?: string,
): LlmAttemptTrace {
  return {
    attempt,
    provider: result.provider,
    model: result.model,
    latencyMs: result.latencyMs,
    success,
    ...(errorCode ? { errorCode } : {}),
    outputChars: result.text.length,
    ...(result.requestId ? { requestId: result.requestId } : {}),
    ...(result.usage ? { usage: result.usage } : {}),
  };
}

function buildFailedProviderAttemptTrace(
  attempt: number,
  client: LlmClient,
  latencyMs: number,
  error: unknown,
): LlmAttemptTrace {
  return {
    attempt,
    provider: client.provider,
    model: client.model,
    latencyMs,
    success: false,
    errorCode: classifyProviderError(error),
    outputChars: 0,
  };
}

function classifyProviderError(error: unknown): string {
  const message = error instanceof Error ? error.message : String(error);
  if (/empty assistant message/i.test(message)) return "LLM_EMPTY_ASSISTANT_MESSAGE";
  if (/HTTP\s+429/.test(message)) return "LLM_RATE_LIMIT";
  if (/HTTP\s+5\d\d/.test(message)) return "LLM_SERVER_ERROR";
  if (/fetch failed|network|timeout|timed out|ECONNRESET|ETIMEDOUT|terminated|socket hang up|other side closed|EPIPE|ECONNREFUSED/i.test(message)) return "LLM_TRANSIENT_REQUEST_FAILED";
  return "LLM_PROVIDER_FAILED";
}

function isRetryableProviderError(error: unknown): boolean {
  return ["LLM_EMPTY_ASSISTANT_MESSAGE", "LLM_RATE_LIMIT", "LLM_SERVER_ERROR", "LLM_TRANSIENT_REQUEST_FAILED"].includes(
    classifyProviderError(error),
  );
}

function backoffMs(requestAttempt: number): number {
  const base = Math.min(250 * 2 ** Math.max(0, requestAttempt - 1), 2000);
  const jitter = Math.floor(Math.random() * Math.max(50, Math.floor(base * 0.25)));
  return base + jitter;
}

async function sleep(ms: number): Promise<void> {
  await new Promise((resolve) => setTimeout(resolve, ms));
}

function mergeUsage(left: LlmUsage | undefined, right: LlmGenerateResult["usage"]): LlmUsage | undefined {
  if (!right) return left;
  return {
    inputTokens: addOptional(left?.inputTokens, right.inputTokens),
    outputTokens: addOptional(left?.outputTokens, right.outputTokens),
    totalTokens: addOptional(left?.totalTokens, right.totalTokens),
  };
}

function addOptional(left: number | undefined, right: number | undefined): number | undefined {
  if (left === undefined) return right;
  if (right === undefined) return left;
  return left + right;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
