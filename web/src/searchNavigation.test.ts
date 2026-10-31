import { afterEach, describe, expect, it, vi } from "vitest";
import * as api from "./api";
import { resolveSearchWorkspace } from "./searchNavigation";

const conversation: api.Conversation = {
  id: "conversation", title: "Saved conversation", project_id: null, private_required: false,
  default_mode: "auto", default_model_id: "auto", created_at: "2026-10-31T10:00:00Z",
  updated_at: "2026-10-31T10:00:00Z", last_message_at: "2026-10-31T10:00:00Z", archived_at: null,
};
const file: api.AudreyFile = {
  id: "file", filename: "Retained document.txt", kind: "text", mime: "text/plain", bytes: 12,
  uploaded_at: conversation.created_at, status: "ready", chunks: 1, failure_reason: "", duration_s: 0,
  summary: "", source_freed_at: conversation.created_at, leased_at: "", source_url: "",
  transcript_source: "", fetch_downloaded_bytes: 0, fetch_total_bytes: 0,
};
function item(target: api.SearchTarget = { kind: "message", conversation_id: conversation.id, message_id: "message", sequence: 2 }): api.SearchResult {
  return { id: "match", source_id: "source", kind: target.kind, revision: "revision", title: "Saved work", excerpt: "local",
    highlights: [], offset: 0, updated_at: conversation.updated_at, project_id: null, private_required: false,
    availability: "text_only", target };
}
function proof(result: api.SearchResult): api.SearchSource {
  return { ...result, text: "Saved text", total_chars: 10, next_offset: null };
}
afterEach(() => vi.restoreAllMocks());

describe("Search workspace navigation", () => {
  it("opens the owned archived Private conversation without changing its model or creating anything", async () => {
    const result = { ...item(), private_required: true };
    const saved = { ...conversation, default_model_id: "private", private_required: true, archived_at: conversation.created_at };
    const read = vi.spyOn(api, "getConversation").mockResolvedValue(saved);
    vi.spyOn(api, "getSearchSource").mockResolvedValue(proof(result));
    const signal = new AbortController().signal;
    expect(await resolveSearchWorkspace(result, signal)).toEqual({ kind: "message", conversation: saved, messageId: "message" });
    expect(read).toHaveBeenCalledWith(conversation.id, signal);
    expect(api.getSearchSource).toHaveBeenCalledWith(result, 0, signal);
    expect(read.mock.invocationCallOrder[0]).toBeLessThan(vi.mocked(api.getSearchSource).mock.invocationCallOrder[0]);
  });

  it("opens the canonical file even when only retained text exists", async () => {
    const result = item({ kind: "file", file_id: file.id, artifact: "text" });
    vi.spyOn(api, "getFile").mockResolvedValue(file);
    vi.spyOn(api, "getSearchSource").mockResolvedValue(proof(result));
    expect(await resolveSearchWorkspace(result, new AbortController().signal)).toEqual({ kind: "file", file, artifact: "text" });
  });

  it("opens the existing report task without requiring its former Project to exist", async () => {
    const result = item({ kind: "agent", task_id: "agent", step_id: "review" });
    const task = { id: "agent", project_context: { available: false } } as api.AgentTask;
    vi.spyOn(api, "getAgentTask").mockResolvedValue(task);
    vi.spyOn(api, "getSearchSource").mockResolvedValue(proof(result));
    expect(await resolveSearchWorkspace(result, new AbortController().signal)).toEqual({ kind: "agent", task });
  });

  it("opens a Private note's canonical Project", async () => {
    const result = { ...item({ kind: "knowledge", project_id: "project", entry_id: "note" }), private_required: true };
    const project = { id: "project", name: "Private work", instructions: "", private_required: true,
      created_at: conversation.created_at, updated_at: conversation.updated_at };
    vi.spyOn(api, "getProject").mockResolvedValue(project);
    vi.spyOn(api, "getSearchSource").mockResolvedValue(proof(result));
    expect(await resolveSearchWorkspace(result, new AbortController().signal)).toEqual({ kind: "knowledge", project, entryId: "note" });
  });

  it.each(["revision", "source_id", "kind", "target", "private_required"])("refuses a changed canonical source %s", async (field) => {
    const result = item();
    vi.spyOn(api, "getConversation").mockResolvedValue(conversation);
    const source = proof(result);
    const changed = { ...source, [field]: field === "target" ? { ...result.target, conversation_id: "different" } : field === "private_required" ? true : "different" };
    vi.spyOn(api, "getSearchSource").mockResolvedValue(changed as api.SearchSource);
    await expect(resolveSearchWorkspace(result, new AbortController().signal)).rejects.toMatchObject({ status: 409 });
  });

  it.each([403, 404, 409])("keeps the viewer when a source crosses a %i boundary during the target read", async (status) => {
    vi.spyOn(api, "getConversation").mockResolvedValue(conversation);
    vi.spyOn(api, "getSearchSource").mockRejectedValue(new api.ApiError(status, "Unavailable"));
    await expect(resolveSearchWorkspace(item(), new AbortController().signal)).rejects.toMatchObject({ status });
  });

  it.each([
    { id: "another" }, { project_id: "different-project" }, { private_required: true },
  ])("refuses mismatched canonical conversation metadata %j", async (patch) => {
    vi.spyOn(api, "getConversation").mockResolvedValue({ ...conversation, ...patch });
    vi.spyOn(api, "getSearchSource");
    await expect(resolveSearchWorkspace(item(), new AbortController().signal)).rejects.toMatchObject({ status: 409 });
    expect(api.getSearchSource).not.toHaveBeenCalled();
  });

  it("ignores completion after cancellation even if the transport ignores AbortSignal", async () => {
    const controller = new AbortController();
    vi.spyOn(api, "getConversation").mockResolvedValue(conversation);
    vi.spyOn(api, "getSearchSource").mockImplementation(async () => {
      controller.abort(); return proof(item());
    });
    await expect(resolveSearchWorkspace(item(), controller.signal)).rejects.toMatchObject({ name: "AbortError" });
  });
});
