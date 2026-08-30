export type LlmProviderName = "nchc_llm" | "openai_llm" | "codex_llm";

export type LlmMessage = {
  role: "system" | "user" | "assistant";
  content: string;
};

export type LlmGenerateOptions = {
  messages: LlmMessage[];
  temperature: number;
  maxTokens: number;
  responseFormat?: "json_object";
  signal?: AbortSignal;
};

export type LlmUsage = {
  inputTokens?: number;
  outputTokens?: number;
  totalTokens?: number;
};

export type LlmGenerateResult = {
  provider: LlmProviderName;
  model: string;
  text: string;
  requestId?: string;
  latencyMs: number;
  usage?: LlmUsage;
};

export interface LlmClient {
  readonly provider: LlmProviderName;
  readonly model: string;
  generate(options: LlmGenerateOptions): Promise<LlmGenerateResult>;
}

export type LlmJsonValidationResult<T> =
  | { ok: true; value: T }
  | { ok: false; message: string; details?: Record<string, unknown> };

export type GenerateJsonWithRetryOptions<T> = {
  client: LlmClient;
  messages: LlmMessage[];
  temperature: number;
  maxTokens: number;
  validate: (value: unknown) => LlmJsonValidationResult<T>;
  stage?: string;
  signal?: AbortSignal;
  maxJsonRepairAttempts?: number;
  maxRequestRetries?: number;
  onAttempt?: (attempt: LlmAttemptTrace) => void;
};

export type LlmAttemptTrace = {
  attempt: number;
  provider: LlmProviderName;
  model: string;
  latencyMs: number;
  success: boolean;
  errorCode?: string;
  outputChars: number;
  requestId?: string;
  usage?: LlmUsage;
};

export type GenerateJsonWithRetryResult<T> = {
  value: T;
  attempts: number;
  rawText: string;
  provider: LlmProviderName;
  model: string;
  latencyMs: number;
  requestIds: string[];
  usage?: LlmUsage;
  attemptTrace: LlmAttemptTrace[];
};
