import { apiJson } from "./api";

export type PrivateTaskStatus = "queued" | "running" | "waiting_for_user" | "paused" | "completed" | "cancelled" | "failed";

export interface PrivateTask {
  id: string;
  conversation_id: string;
  revision: number;
  status: PrivateTaskStatus;
  stop_reason: string;
  progress: number;
  updated_at: string;
  native_run_id: string;
  usage: { steps: number; wall_seconds: number; model_tokens: number; tool_calls: number };
  limits: { max_steps: number; max_wall_seconds: number; max_model_tokens: number; max_tool_calls: number };
}

export function getPrivateTask(conversationId: string, signal?: AbortSignal): Promise<{ task: PrivateTask | null }> {
  return apiJson(`/api/conversations/${encodeURIComponent(conversationId)}/private-task`, {
    signal, cache: "no-store",
  });
}

export function actOnPrivateTask(
  taskId: string,
  expectedRevision: number,
  action: "pause" | "cancel",
  signal?: AbortSignal,
): Promise<{ task: PrivateTask }> {
  return apiJson(`/api/private-tasks/${encodeURIComponent(taskId)}/actions`, {
    method: "POST", signal,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ expected_revision: expectedRevision, action }),
  });
}
