import { afterEach, describe, expect, it, vi } from "vitest";

import {
  addProjectFile,
  cancelDocumentJob,
  createConversation,
  createProjectBrief,
  createProject,
  createProjectConversation,
  decideDocumentJob,
  deleteProject,
  fetchVideoFromUrl,
  getFileArtifactDownloadUrl,
  getDocumentJob,
  getFileDownloadUrl,
  listDocumentJobs,
  listProjects,
  removeProjectFile,
  resetAdminModelPolicy,
  updateAdminModel,
  updateConversation,
  updateConversationModel,
  uploadFile,
  type AudreyFileLimits,
} from "./api";

const LIMITS: AudreyFileLimits = {
  max_upload_bytes: 4,
  max_user_bytes: 1_000,
  allowed_extensions: [".txt"],
  chunked_max_bytes: 20,
  part_size: 4,
  fetch_hosts: ["www.youtube.com"],
  max_images_per_turn: 4,
};

describe("native file uploads", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("builds an encoded same-origin original download URL", () => {
    expect(getFileDownloadUrl("file / 123")).toBe("/api/files/file%20%2F%20123/download");
  });

  it("builds an encoded same-origin artifact download URL", () => {
    expect(getFileArtifactDownloadUrl("file / 123", "visual")).toBe(
      "/api/files/file%20%2F%20123/artifacts/visual/download",
    );
  });

  it("queues a trimmed video URL through the same-origin native API", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({
      id: "file_video",
      filename: "video123",
      mime: "",
      bytes: 0,
      kind: "video",
      chunks: 0,
      status: "fetch_pending",
    }));
    vi.stubGlobal("fetch", fetchMock);

    const result = await fetchVideoFromUrl("  https://www.youtube.com/watch?v=video123  ");

    expect(result.status).toBe("fetch_pending");
    expect(fetchMock).toHaveBeenCalledWith(
      "/api/files/from-url",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({ url: "https://www.youtube.com/watch?v=video123" }),
        credentials: "same-origin",
      }),
    );
    expect((fetchMock.mock.calls[0][1] as RequestInit).headers).not.toHaveProperty("Authorization");
  });

  it("reports byte progress for multipart files within the single-request limit", async () => {
    let progressListener: ((event: ProgressEvent) => void) | undefined;
    let loadListener: (() => void) | undefined;
    const request = {
      status: 200,
      responseText: JSON.stringify({
        id: "file_small",
        filename: "small.txt",
        mime: "text/plain",
        bytes: 4,
        kind: "text",
        chunks: 1,
        status: "ready",
      }),
      upload: {
        addEventListener: vi.fn((name: string, listener: (event: ProgressEvent) => void) => {
          if (name === "progress") progressListener = listener;
        }),
      },
      open: vi.fn(),
      setRequestHeader: vi.fn(),
      addEventListener: vi.fn((name: string, listener: () => void) => {
        if (name === "load") loadListener = listener;
      }),
      send: vi.fn((body: FormData) => {
        expect(body).toBeInstanceOf(FormData);
        progressListener?.({ lengthComputable: true, loaded: 2, total: 4 } as ProgressEvent);
        loadListener?.();
      }),
      withCredentials: false,
    };
    const xhr = vi.fn(() => request);
    vi.stubGlobal("XMLHttpRequest", xhr);
    const progress: number[] = [];

    const result = await uploadFile(
      new File(["tiny"], "small.txt", { type: "text/plain" }),
      LIMITS,
      (value) => progress.push(value),
    );

    expect(result.id).toBe("file_small");
    expect(xhr).toHaveBeenCalledOnce();
    expect(request.open).toHaveBeenCalledWith("POST", "/api/files");
    expect(request.withCredentials).toBe(true);
    expect(request.setRequestHeader).toHaveBeenCalledWith("Accept", "application/json");
    expect(progress).toEqual([0, 0.5, 1]);
  });

  it("uploads larger files as bounded sequential parts", async () => {
    const partSizes: number[] = [];
    const fetchMock = vi.fn().mockImplementation(
      async (path: string, request?: RequestInit) => {
        if (path === "/api/files/upload-sessions") {
          return jsonResponse({
            upload_id: "upload_123",
            part_size: 4,
            parts_total: 3,
            received_parts: [],
          });
        }
        if (path.includes("/parts/")) {
          partSizes.push((request?.body as Blob).size);
          return jsonResponse({ ok: true });
        }
        return jsonResponse({
          id: "file_large",
          filename: "large.txt",
          mime: "text/plain",
          bytes: 10,
          kind: "text",
          chunks: 1,
          status: "ready",
        });
      },
    );
    vi.stubGlobal("fetch", fetchMock);
    const progress: number[] = [];

    const result = await uploadFile(
      new File(["0123456789"], "large.txt", { type: "text/plain" }),
      LIMITS,
      (value) => progress.push(value),
    );

    expect(result.id).toBe("file_large");
    expect(partSizes).toEqual([4, 4, 2]);
    expect(fetchMock.mock.calls.map(([path]) => path)).toEqual([
      "/api/files/upload-sessions",
      "/api/files/upload-sessions/upload_123/parts/0",
      "/api/files/upload-sessions/upload_123/parts/1",
      "/api/files/upload-sessions/upload_123/parts/2",
      "/api/files/upload-sessions/upload_123/complete",
    ]);
    expect(progress).toEqual([0, 1 / 3, 2 / 3, 1]);
  });

  it("rejects a file above the published limit before sending bytes", async () => {
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);

    await expect(uploadFile(
      new File(["012345678901234567890"], "too-large.txt"),
      LIMITS,
    )).rejects.toThrow("exceeds Audrey's");
    expect(fetchMock).not.toHaveBeenCalled();
  });
});

describe("owner-scoped project requests", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("encodes project resources and sends explicit membership changes", async () => {
    const fetchMock = vi.fn().mockImplementation(() => Promise.resolve(jsonResponse({})));
    vi.stubGlobal("fetch", fetchMock);

    await listProjects("next page");
    await createProject("Launch", "Keep answers concise.");
    await createProjectConversation("proj / 1", "research");
    await addProjectFile("proj / 1", "file / 1");
    await removeProjectFile("proj / 1", "file / 1");
    await updateConversation("con / 1", { project_id: null });
    await deleteProject("proj / 1");

    expect(fetchMock).toHaveBeenNthCalledWith(
      1,
      "/api/projects?limit=100&cursor=next+page",
      expect.objectContaining({ credentials: "same-origin" }),
    );
    expect(fetchMock).toHaveBeenNthCalledWith(
      2,
      "/api/projects",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({ name: "Launch", instructions: "Keep answers concise." }),
      }),
    );
    expect(fetchMock).toHaveBeenNthCalledWith(
      3,
      "/api/projects/proj%20%2F%201/conversations",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({ model_id: "research" }),
      }),
    );
    expect(fetchMock).toHaveBeenNthCalledWith(
      4,
      "/api/projects/proj%20%2F%201/files",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({ file_id: "file / 1" }),
      }),
    );
    expect(fetchMock).toHaveBeenNthCalledWith(
      5,
      "/api/projects/proj%20%2F%201/files/file%20%2F%201",
      expect.objectContaining({ method: "DELETE" }),
    );
    expect(fetchMock).toHaveBeenNthCalledWith(
      6,
      "/api/conversations/con%20%2F%201",
      expect.objectContaining({
        method: "PATCH",
        body: JSON.stringify({ project_id: null }),
      }),
    );
    expect(fetchMock).toHaveBeenNthCalledWith(
      7,
      "/api/projects/proj%20%2F%201",
      expect.objectContaining({ method: "DELETE" }),
    );
  });
});

describe("owner-scoped document requests", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("creates, reviews, polls, and cancels through encoded native routes", async () => {
    const fetchMock = vi.fn().mockImplementation(() => Promise.resolve(jsonResponse({ items: [] })));
    vi.stubGlobal("fetch", fetchMock);
    const request = {
      template_id: "project-brief-v1" as const,
      filename: "Launch brief.docx",
      fields: {
        title: "Launch",
        prepared_for: "Build Ryte",
        prepared_on: "2026-10-04",
        summary: "A reviewed launch brief.",
        objectives: ["Ship safely"],
        next_steps: ["Approve the request"],
      },
      idempotency_key: "brief-one",
    };

    await listDocumentJobs();
    await createProjectBrief(request);
    await getDocumentJob("job / one");
    await decideDocumentJob("job / one", "a".repeat(64), "approved");
    await cancelDocumentJob("job / one");

    expect(fetchMock).toHaveBeenNthCalledWith(
      1,
      "/api/document-jobs?limit=50",
      expect.objectContaining({ credentials: "same-origin" }),
    );
    expect(fetchMock).toHaveBeenNthCalledWith(
      2,
      "/api/document-jobs/template-to-docx",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify(request),
      }),
    );
    expect(fetchMock).toHaveBeenNthCalledWith(
      3,
      "/api/document-jobs/job%20%2F%20one",
      expect.objectContaining({ credentials: "same-origin" }),
    );
    expect(fetchMock).toHaveBeenNthCalledWith(
      4,
      "/api/document-jobs/job%20%2F%20one/decision",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({
          decision: "approved",
          operation_digest: "a".repeat(64),
        }),
      }),
    );
    expect(fetchMock).toHaveBeenNthCalledWith(
      5,
      "/api/document-jobs/job%20%2F%20one/cancel",
      expect.objectContaining({ method: "POST" }),
    );
  });
});

describe("server-owned model selections", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("sends stable model IDs when conversations are created and changed", async () => {
    const fetchMock = vi.fn().mockImplementation(() => Promise.resolve(jsonResponse({})));
    vi.stubGlobal("fetch", fetchMock);

    await createConversation("direct/qwen3.8-27b");
    await updateConversationModel("con_example", "research");

    expect(fetchMock).toHaveBeenNthCalledWith(
      1,
      "/api/conversations",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({ model_id: "direct/qwen3.8-27b" }),
      }),
    );
    expect(fetchMock).toHaveBeenNthCalledWith(
      2,
      "/api/conversations/con_example",
      expect.objectContaining({
        method: "PATCH",
        body: JSON.stringify({ model_id: "research" }),
      }),
    );
  });

  it("encodes direct-model IDs and sends the complete policy mutation", async () => {
    const fetchMock = vi.fn().mockImplementation(() => Promise.resolve(jsonResponse({})));
    vi.stubGlobal("fetch", fetchMock);

    await updateAdminModel("direct/qwen3.8-27b", {
      enabled: false,
      audience: "testers",
    });
    await resetAdminModelPolicy("direct/qwen3.8-27b");

    expect(fetchMock).toHaveBeenCalledWith(
      "/api/admin/models/direct%2Fqwen3.8-27b",
      expect.objectContaining({
        method: "PATCH",
        body: JSON.stringify({ enabled: false, audience: "testers" }),
      }),
    );
    expect(fetchMock).toHaveBeenCalledWith(
      "/api/admin/model-policies/direct%2Fqwen3.8-27b",
      expect.objectContaining({ method: "DELETE" }),
    );
  });
});

function jsonResponse(payload: unknown): Response {
  return new Response(JSON.stringify(payload), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });
}
