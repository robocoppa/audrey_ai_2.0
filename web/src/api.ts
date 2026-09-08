export interface CurrentUser {
  id: string;
  email: string;
  display_name: string;
  role: "admin" | "user";
  status: "active" | string;
  auth_provider: string;
}

export interface UserPreferences {
  timezone: string;
  persona: string;
  detail: "concise" | "balanced" | "detailed";
  tone: "natural" | "professional" | "casual";
  show_progress: boolean;
  created_at: string;
  updated_at: string;
}

export type UserPreferencesUpdate = Pick<
  UserPreferences,
  "timezone" | "persona" | "detail" | "tone" | "show_progress"
>;

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
  attachments: MessageAttachment[];
}

export interface MessageAttachment {
  id: string;
  filename: string;
  mime: string;
  kind: "text" | "image" | "video";
  bytes: number;
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

export interface AudreyFile {
  id: string;
  filename: string;
  mime: string;
  bytes: number;
  uploaded_at: string;
  kind: "text" | "image" | "video";
  chunks: number;
  status: string;
  failure_reason: string;
  duration_s: number;
  summary: string;
  source_freed_at: string;
  leased_at: string;
  source_url: string;
  transcript_source: string;
  fetch_downloaded_bytes: number;
  fetch_total_bytes: number;
}

export interface AudreyFileLimits {
  max_upload_bytes: number;
  max_user_bytes: number;
  allowed_extensions: string[];
  chunked_max_bytes: number;
  part_size: number;
}

export interface AudreyFileList {
  items: AudreyFile[];
  total_bytes: number;
  server_time: string;
  limits: AudreyFileLimits;
}

export interface AudreyFileUpload {
  id: string;
  filename: string;
  mime: string;
  bytes: number;
  kind: "text" | "image" | "video";
  chunks: number;
  status: string;
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

export async function apiResponse(path: string, init?: RequestInit): Promise<Response> {
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

export function getCurrentUserPreferences(): Promise<UserPreferences> {
  return apiJson<UserPreferences>("/api/me/preferences");
}

export function updateCurrentUserPreferences(
  preferences: UserPreferencesUpdate,
): Promise<UserPreferences> {
  return apiJson<UserPreferences>("/api/me/preferences", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(preferences),
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
    body: JSON.stringify({ default_mode: mode }),
  });
}

export function getConversation(conversationId: string): Promise<Conversation> {
  return apiJson<Conversation>(
    `/api/conversations/${encodeURIComponent(conversationId)}`,
  );
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

export function listFiles(): Promise<AudreyFileList> {
  return apiJson<AudreyFileList>("/api/files");
}

export function getFile(fileId: string): Promise<AudreyFile> {
  return apiJson<AudreyFile>(`/api/files/${encodeURIComponent(fileId)}`);
}

export async function uploadFile(
  file: File,
  limits: AudreyFileLimits,
  onProgress: (fraction: number) => void = () => undefined,
): Promise<AudreyFileUpload> {
  if (file.size <= limits.max_upload_bytes) {
    const body = new FormData();
    body.append("file", file, file.name);
    onProgress(0);
    const result = await apiJson<AudreyFileUpload>("/api/files", {
      method: "POST",
      body,
    });
    onProgress(1);
    return result;
  }
  if (file.size > limits.chunked_max_bytes) {
    throw new Error(
      `This file exceeds Audrey's ${formatByteLimit(limits.chunked_max_bytes)} upload limit.`,
    );
  }

  const session = await apiJson<{
    upload_id: string;
    part_size: number;
    parts_total: number;
    received_parts: number[];
  }>("/api/files/upload-sessions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ filename: file.name, total_bytes: file.size }),
  });
  const received = new Set(session.received_parts);
  let completed = received.size;
  onProgress(completed / session.parts_total);
  for (let part = 0; part < session.parts_total; part += 1) {
    if (received.has(part)) continue;
    const start = part * session.part_size;
    const end = Math.min(file.size, start + session.part_size);
    await apiJson(`/api/files/upload-sessions/${encodeURIComponent(session.upload_id)}/parts/${part}`, {
      method: "PUT",
      headers: { "Content-Type": "application/octet-stream" },
      body: file.slice(start, end),
    });
    completed += 1;
    onProgress(completed / session.parts_total);
  }
  return apiJson<AudreyFileUpload>(
    `/api/files/upload-sessions/${encodeURIComponent(session.upload_id)}/complete`,
    { method: "POST" },
  );
}

export function deleteFile(fileId: string): Promise<{
  id: string;
  deleted: boolean;
  pending_cleanup: boolean;
}> {
  return apiJson(`/api/files/${encodeURIComponent(fileId)}`, {
    method: "DELETE",
  });
}

function formatByteLimit(bytes: number): string {
  const mebibytes = bytes / (1024 * 1024);
  if (mebibytes < 1024) return `${Math.round(mebibytes)} MB`;
  return `${(mebibytes / 1024).toFixed(1)} GB`;
}
