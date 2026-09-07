export interface CurrentUser {
  id: string;
  email: string;
  display_name: string;
  role: "admin" | "user";
  status: "active" | string;
  auth_provider: string;
}

export type AudreyMode =
  | "auto"
  | "fast"
  | "deep"
  | "research"
  | "local"
  | "cloud"
  | "video";

export interface Conversation {
  id: string;
  title: string;
  default_mode: AudreyMode;
  created_at: string;
  updated_at: string;
  last_message_at: string | null;
  archived_at: string | null;
}

export interface ConversationMessage {
  id: string;
  run_id: string | null;
  sequence: number;
  role: "user" | "assistant" | "tool";
  status: "in_progress" | "completed" | "incomplete";
  content: string;
  created_at: string;
  updated_at: string;
}

export interface ListResponse<T> {
  items: T[];
  next_cursor: string | null;
}

export interface ConversationListOptions {
  archived?: boolean;
  cursor?: string | null;
  search?: string;
}

export interface ConversationPatch {
  title?: string;
  default_mode?: AudreyMode;
  archived?: boolean;
}

export class ApiError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

export async function apiJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await apiResponse(path, init);
  return (await response.json()) as T;
}

async function apiResponse(path: string, init?: RequestInit): Promise<Response> {
  const { headers, ...options } = init ?? {};
  const response = await fetch(path, {
    ...options,
    credentials: "same-origin",
    headers: {
      Accept: "application/json",
      ...headers,
    },
  });

  if (!response.ok) {
    let detail = `Request failed with HTTP ${response.status}.`;
    try {
      const payload = (await response.json()) as { detail?: string };
      if (typeof payload.detail === "string" && payload.detail.trim()) {
        detail = payload.detail;
      }
    } catch {
      // Preserve the status-based message when the server did not return JSON.
    }
    throw new ApiError(response.status, detail);
  }
  return response;
}

export function getCurrentUser(): Promise<CurrentUser> {
  return apiJson<CurrentUser>("/api/me");
}

export function updateCurrentUserDisplayName(
  displayName: string,
): Promise<CurrentUser> {
  return apiJson<CurrentUser>("/api/me", {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ display_name: displayName }),
  });
}

export function listConversations(
  options: ConversationListOptions = {},
): Promise<ListResponse<Conversation>> {
  const params = new URLSearchParams({
    archived: String(Boolean(options.archived)),
    limit: "100",
  });
  const search = options.search?.trim();
  if (search) params.set("q", search);
  if (options.cursor) params.set("cursor", options.cursor);
  return apiJson<ListResponse<Conversation>>(`/api/conversations?${params}`);
}

export function createConversation(mode: AudreyMode): Promise<Conversation> {
  return apiJson<Conversation>("/api/conversations", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title: "New conversation", default_mode: mode }),
  });
}

export function listMessages(
  conversationId: string,
): Promise<ListResponse<ConversationMessage>> {
  return apiJson<ListResponse<ConversationMessage>>(
    `/api/conversations/${encodeURIComponent(conversationId)}/messages?limit=100`,
  );
}

export function updateConversationMode(
  conversationId: string,
  mode: AudreyMode,
): Promise<Conversation> {
  return updateConversation(conversationId, { default_mode: mode });
}

export function updateConversation(
  conversationId: string,
  patch: ConversationPatch,
): Promise<Conversation> {
  return apiJson<Conversation>(
    `/api/conversations/${encodeURIComponent(conversationId)}`,
    {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(patch),
    },
  );
}

export async function deleteConversation(conversationId: string): Promise<void> {
  await apiResponse(`/api/conversations/${encodeURIComponent(conversationId)}`, {
    method: "DELETE",
  });
}
