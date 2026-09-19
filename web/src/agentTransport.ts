export async function latestActionFetch(
  url: string,
  init: RequestInit,
  attachmentIds: readonly string[] = [],
  onRunId?: (runId: string) => void,
): Promise<Response> {
  if (typeof init.body !== "string") {
    throw new Error("Audrey's agent request body was not JSON text.");
  }
  const payload: unknown = JSON.parse(init.body);
  const envelope = recordOf(payload);
  const messages = Array.isArray(envelope.messages) ? envelope.messages : [];
  const latest = messages.at(-1);
  const latestMessage = recordOf(latest);
  if (latestMessage.role !== "user") {
    throw new Error("Audrey's latest browser action was not a user message.");
  }
  const body = JSON.stringify({
    threadId: envelope.threadId,
    runId: envelope.runId,
    messages: [latestMessage],
    ...(attachmentIds.length > 0 ? { attachmentIds } : {}),
  });
  const response = await fetch(url, {
    ...init,
    body,
    credentials: "same-origin",
  });
  const runId = response.headers.get("X-Audrey-Run-ID")?.trim();
  if (runId) onRunId?.(runId);
  return response;
}

function recordOf(value: unknown): Record<string, unknown> {
  return value !== null && typeof value === "object"
    ? (value as Record<string, unknown>)
    : {};
}
