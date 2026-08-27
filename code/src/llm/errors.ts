export class LlmConfigError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "LlmConfigError";
  }
}

export class LlmProviderError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "LlmProviderError";
  }
}

export class LlmJsonParseError extends Error {
  readonly code = "LLM_JSON_PARSE_FAILED" as const;
  readonly details?: Record<string, unknown>;

  constructor(message: string, details?: Record<string, unknown>) {
    super(message);
    this.name = "LlmJsonParseError";
    this.details = details;
  }

  toStructuredError(): {
    error: {
      code: "LLM_JSON_PARSE_FAILED";
      message: string;
      details?: Record<string, unknown>;
    };
  } {
    return {
      error: {
        code: this.code,
        message: this.message,
        ...(this.details ? { details: this.details } : {}),
      },
    };
  }
}

const SECRET_KEY_RE = /TOKEN|SECRET|PASSWORD|AUTH|API[_-]?KEY/i;

export function redactSecrets(
  text: string,
  env: NodeJS.ProcessEnv = process.env,
): string {
  let redacted = text;
  redacted = redacted.replace(
    /authorization\s*:\s*bearer\s+\S+/gi,
    "authorization: bearer [redacted]",
  );
  redacted = redacted.replace(/x-api-key\s*:\s*\S+/gi, "x-api-key: [redacted]");
  redacted = redacted.replace(
    /(api[_-]?key|api[_-]?token|token|secret|password)=\S+/gi,
    "$1=[redacted]",
  );
  for (const [key, value] of Object.entries(env)) {
    if (!value || value.length === 0) continue;
    if (SECRET_KEY_RE.test(key)) {
      redacted = redacted.split(value).join(`[redacted ${key}]`);
    }
  }
  return redacted;
}

export function errorMessageWithRedactedSecrets(
  error: unknown,
  env: NodeJS.ProcessEnv = process.env,
): string {
  const message = error instanceof Error ? error.message : String(error);
  return redactSecrets(message, env);
}
