import { LlmConfigError } from "./errors";
import type { LlmProviderName } from "./types";

export type NchcLlmConfig = {
  provider: "nchc_llm";
  model: string;
  baseUrl: string;
  apiKeyEnv: string;
  temperature: number;
  maxTokens: number;
};

export type OpenAiLlmConfig = {
  provider: "openai_llm";
  model: string;
  baseUrl: string;
  apiKeyEnv: string;
  temperature: number; // kept for interface parity; the client omits it on the wire
  maxTokens: number;
};

export type CodexLlmConfig = {
  provider: "codex_llm";
  model: string;
  baseUrl: string; // sidecar base URL
  apiKeyEnv: string; // unused; kept for interface parity
  temperature: number; // unused on the wire
  maxTokens: number;
};

export type LlmClientConfig = NchcLlmConfig | OpenAiLlmConfig | CodexLlmConfig;

export type RawLlmClientConfig = {
  provider?: unknown;
  model?: unknown;
  base_url?: unknown;
  baseUrl?: unknown;
  api_key_env?: unknown;
  apiKeyEnv?: unknown;
  temperature?: unknown;
  max_tokens?: unknown;
  maxTokens?: unknown;
};

export const DEFAULT_NCHC_BASE_URL = "https://portal.genai.nchc.org.tw/api/v1";
export const DEFAULT_NCHC_MODEL = "gpt-oss-120b";
export const DEFAULT_NCHC_API_KEY_ENV = "NCHC_API_KEY";
export const DEFAULT_LLM_TEMPERATURE = 0;
export const DEFAULT_LLM_MAX_TOKENS = 2048;

export const DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1";
export const DEFAULT_OPENAI_API_KEY_ENV = "OPENAI_API_KEY";

export const DEFAULT_SIDECAR_BASE_URL = "http://127.0.0.1:8765";

export function normalizeLlmClientConfig(raw: RawLlmClientConfig): LlmClientConfig {
  const provider = normalizeProvider(raw.provider ?? "nchc_llm");
  if (provider === "codex_llm") {
    return {
      provider,
      model: normalizeNonEmptyString(raw.model ?? "gpt-5.6-sol", "llm.model"),
      baseUrl: normalizeNonEmptyString(raw.baseUrl ?? raw.base_url ?? DEFAULT_SIDECAR_BASE_URL, "llm.base_url"),
      apiKeyEnv: "UNUSED",
      temperature: normalizeFiniteNumber(raw.temperature ?? DEFAULT_LLM_TEMPERATURE, "llm.temperature"),
      maxTokens: normalizePositiveInteger(raw.maxTokens ?? raw.max_tokens ?? DEFAULT_LLM_MAX_TOKENS, "llm.max_tokens"),
    };
  }
  if (provider === "openai_llm") {
    return {
      provider,
      model: normalizeNonEmptyString(raw.model, "llm.model"),
      baseUrl: normalizeNonEmptyString(raw.baseUrl ?? raw.base_url ?? DEFAULT_OPENAI_BASE_URL, "llm.base_url"),
      apiKeyEnv: normalizeNonEmptyString(raw.apiKeyEnv ?? raw.api_key_env ?? DEFAULT_OPENAI_API_KEY_ENV, "llm.api_key_env"),
      temperature: normalizeFiniteNumber(raw.temperature ?? DEFAULT_LLM_TEMPERATURE, "llm.temperature"),
      maxTokens: normalizePositiveInteger(raw.maxTokens ?? raw.max_tokens ?? DEFAULT_LLM_MAX_TOKENS, "llm.max_tokens"),
    };
  }
  return {
    provider,
    model: normalizeNonEmptyString(raw.model ?? DEFAULT_NCHC_MODEL, "llm.model"),
    baseUrl: normalizeNonEmptyString(raw.baseUrl ?? raw.base_url ?? DEFAULT_NCHC_BASE_URL, "llm.base_url"),
    apiKeyEnv: normalizeNonEmptyString(raw.apiKeyEnv ?? raw.api_key_env ?? DEFAULT_NCHC_API_KEY_ENV, "llm.api_key_env"),
    temperature: normalizeFiniteNumber(raw.temperature ?? DEFAULT_LLM_TEMPERATURE, "llm.temperature"),
    maxTokens: normalizePositiveInteger(raw.maxTokens ?? raw.max_tokens ?? DEFAULT_LLM_MAX_TOKENS, "llm.max_tokens"),
  };
}

export function safeLlmConfigForArtifacts(config: LlmClientConfig): Record<string, unknown> {
  return {
    provider: config.provider,
    model: config.model,
    base_url: config.baseUrl,
    api_key_env: config.apiKeyEnv,
    temperature: config.temperature,
    max_tokens: config.maxTokens,
  };
}

function normalizeProvider(value: unknown): LlmProviderName {
  if (value === "nchc_llm" || value === "openai_llm" || value === "codex_llm") return value;
  throw new LlmConfigError("llm.provider must be nchc_llm, openai_llm, or codex_llm.");
}

function normalizeNonEmptyString(value: unknown, fieldName: string): string {
  if (typeof value !== "string" || value.trim().length === 0) {
    throw new LlmConfigError(`${fieldName} must be a non-empty string.`);
  }
  return value.trim();
}

function normalizeFiniteNumber(value: unknown, fieldName: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new LlmConfigError(`${fieldName} must be a finite number.`);
  }
  return value;
}

function normalizePositiveInteger(value: unknown, fieldName: string): number {
  if (typeof value !== "number" || !Number.isInteger(value) || value < 1) {
    throw new LlmConfigError(`${fieldName} must be a positive integer.`);
  }
  return value;
}
