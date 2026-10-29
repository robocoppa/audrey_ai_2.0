import { afterEach, describe, expect, it, vi } from "vitest";

import {
  actOnAgentTask,
  addProjectFile,
  createConversation,
  createAgentTask,
  createProject,
  createProjectConversation,
  deleteProject,
  deleteAgentTask,
  fetchVideoFromUrl,
  getFileArtifactDownloadUrl,
  getFileDownloadUrl,
  getAgentTask,
  getSearchSource,
  listAgentTasks,
  listProjects,
  removeProjectFile,
  resetAdminModelPolicy,
  searchContent,
  updateAdminModel,
  updateConversation,
  updateConversationModel,
  uploadFile,
  uploadPrivateProjectFile,
  type AudreyFileLimits,
  type SearchResult,
} from "./api";

describe("native content search requests", () => {
  afterEach(() => { vi.unstubAllGlobals(); });

  it("encodes the query and filters while preserving cancellation and local authentication", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({ items: [], next_cursor: null, partial: false, indexing: false, warnings: [] }));
    vi.stubGlobal("fetch", fetchMock);
    const signal = new AbortController().signal;
    await searchContent("plans & <notes> / 🗂", { kind: "file", project_id: "prj / private", archived: "archived", since: "2026-10-01", until: "2026-10-29", cursor: "cursor+/?=" }, signal);

    const [path, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(path.startsWith("/api/search?")).toBe(true);
    const params = new URL(path, "https://audrey.test").searchParams;
    expect(Object.fromEntries(params)).toEqual({ q: "plans & <notes> / 🗂", kind: "file", project_id: "prj / private", archived: "archived", since: "2026-10-01", until: "2026-10-29", cursor: "cursor+/?=" });
    expect(init).toEqual(expect.objectContaining({ credentials: "same-origin", cache: "no-store", signal }));
    expect(init.headers).not.toHaveProperty("Authorization");
  });

  it("searches all source kinds and archive states by default", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({ items: [], next_cursor: null, partial: false, indexing: false, warnings: [] }));
    vi.stubGlobal("fetch", fetchMock);
    await searchContent("a saved decision");
    expect(fetchMock).toHaveBeenCalledWith("/api/search?q=a+saved+decision&kind=all&archived=all", expect.objectContaining({ cache: "no-store" }));
  });

  it("pins canonical source reads to the returned source ID, revision, and real text offset", async () => {
    const fetchMock = vi.fn().mockImplementation(async () => jsonResponse({ text: "Saved source" }));
    vi.stubGlobal("fetch", fetchMock);
    const signal = new AbortController().signal;
    const item = { source_id: "file / transcript", revision: "sha/+=", offset: 150 } as SearchResult;
    await getSearchSource(item, undefined, signal);
    await getSearchSource(item, 4150, signal);
    const params = new URL(fetchMock.mock.calls[0][0] as string, "https://audrey.test").searchParams;
    expect(Object.fromEntries(params)).toEqual({ source_id: "file / transcript", revision: "sha/+=", offset: "150" });
    expect(fetchMock).toHaveBeenLastCalledWith("/api/search/source?source_id=file+%2F+transcript&revision=sha%2F%2B%3D&offset=4150", expect.objectContaining({ credentials: "same-origin", cache: "no-store", signal }));
  });
});

describe("saved agent task requests", () => {
  afterEach(() => { vi.unstubAllGlobals(); });

  it("uses the authenticated native API, encoded task IDs, idempotency and revisions", async () => {
    const fetchMock = vi.fn().mockImplementation(async () => jsonResponse({ items: [] }));
    vi.stubGlobal("fetch", fetchMock);
    const signal = new AbortController().signal;
    await listAgentTasks(signal);
    await createAgentTask("Compare official sources", "request-unique");
    await getAgentTask("agent / owned", signal);
    await actOnAgentTask("agent / owned", "pause", 7);
    await actOnAgentTask("agent / owned", "rerun", 8);
    await deleteAgentTask("agent / owned", 8);
    expect(fetchMock).toHaveBeenCalledWith("/api/agents", expect.objectContaining({ credentials: "same-origin", signal, cache: "no-store" }));
    expect(fetchMock).toHaveBeenCalledWith("/api/agents", expect.objectContaining({ method: "POST", credentials: "same-origin", body: JSON.stringify({ objective: "Compare official sources", request_id: "request-unique" }) }));
    expect(fetchMock).toHaveBeenCalledWith("/api/agents/agent%20%2F%20owned", expect.objectContaining({ credentials: "same-origin", signal }));
    expect(fetchMock).toHaveBeenCalledWith("/api/agents/agent%20%2F%20owned/actions", expect.objectContaining({ method: "POST", credentials: "same-origin", body: JSON.stringify({ action: "pause", expected_revision: 7 }) }));
    expect(fetchMock).toHaveBeenCalledWith("/api/agents/agent%20%2F%20owned/actions", expect.objectContaining({ method: "POST", credentials: "same-origin", body: JSON.stringify({ action: "rerun", expected_revision: 8 }) }));
    expect(fetchMock).toHaveBeenCalledWith("/api/agents/agent%20%2F%20owned?expected_revision=8", expect.objectContaining({ method: "DELETE", credentials: "same-origin" }));
    for (const [, init] of fetchMock.mock.calls as unknown as Array<[string, RequestInit]>) {
      expect(init.headers).not.toHaveProperty("Authorization");
    }
  });
});

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

  it("reports transfer completion while a multipart file is still processing", async () => {
    let progressListener: ((event: ProgressEvent) => void) | undefined;
    let uploadLoadListener: (() => void) | undefined;
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
          if (name === "load") uploadLoadListener = () => listener({} as ProgressEvent);
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
      }),
      withCredentials: false,
    };
    const xhr = vi.fn(function () { return request; });
    vi.stubGlobal("XMLHttpRequest", xhr);
    const progress: number[] = [];

    let settled = false;
    const pending = uploadFile(
      new File(["tiny"], "small.txt", { type: "text/plain" }),
      LIMITS,
      (value) => progress.push(value),
    ).then((result) => {
      settled = true;
      return result;
    });

    expect(progress).toEqual([0, 0.5]);
    // No final byte-progress event: upload load must still expose processing.
    uploadLoadListener?.();
    await Promise.resolve();
    expect(progress).toEqual([0, 0.5, 1]);
    expect(settled).toBe(false);

    loadListener?.();
    const result = await pending;
    expect(result.id).toBe("file_small");
    expect(xhr).toHaveBeenCalledOnce();
    expect(request.open).toHaveBeenCalledWith("POST", "/api/files");
    expect(request.withCredentials).toBe(true);
    expect(request.setRequestHeader).toHaveBeenCalledWith("Accept", "application/json");
    expect(progress).toEqual([0, 0.5, 1, 1]);
  });

  it("reports all chunked bytes transferred before server processing finishes", async () => {
    const partSizes: number[] = [];
    let finishProcessing: ((response: Response) => void) | undefined;
    const processingResponse = new Promise<Response>((resolve) => { finishProcessing = resolve; });
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
        return processingResponse;
      },
    );
    vi.stubGlobal("fetch", fetchMock);
    const progress: number[] = [];

    let settled = false;
    const pending = uploadFile(
      new File(["0123456789"], "large.txt", { type: "text/plain" }),
      LIMITS,
      (value) => progress.push(value),
    ).then((result) => {
      settled = true;
      return result;
    });

    await vi.waitFor(() => expect(fetchMock).toHaveBeenCalledWith(
      "/api/files/upload-sessions/upload_123/complete", expect.anything(),
    ));
    expect(progress).toEqual([0, 1 / 3, 2 / 3, 1]);
    expect(settled).toBe(false);
    finishProcessing?.(jsonResponse({
      id: "file_large", filename: "large.txt", mime: "text/plain",
      bytes: 10, kind: "text", chunks: 1, status: "ready",
    }));
    const result = await pending;
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

  it("sends Private documents only to their encoded conversation endpoint", async () => {
    let onLoad: (() => void) | undefined;
    const request = {
      status: 200, responseText: JSON.stringify({ id: "private_file", status: "ready" }),
      upload: { addEventListener: vi.fn() }, open: vi.fn(), setRequestHeader: vi.fn(),
      addEventListener: vi.fn((name: string, listener: () => void) => { if (name === "load") onLoad = listener; }),
      send: vi.fn(() => queueMicrotask(() => onLoad?.())), withCredentials: false,
    };
    vi.stubGlobal("XMLHttpRequest", vi.fn(function () { return request; }));
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);
    await uploadFile(new File(["one"], "private.txt"), LIMITS, undefined, "con / private");
    expect(request.open).toHaveBeenCalledWith("POST", "/api/conversations/con%20%2F%20private/files");
    expect(request.withCredentials).toBe(true);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("uploads a protected project document through its encoded project endpoint", async () => {
    let onLoad: (() => void) | undefined;
    const request = {
      status: 200, responseText: JSON.stringify({ id: "private_file", status: "ready" }),
      upload: { addEventListener: vi.fn() }, open: vi.fn(), setRequestHeader: vi.fn(),
      addEventListener: vi.fn((name: string, listener: () => void) => { if (name === "load") onLoad = listener; }),
      send: vi.fn(() => queueMicrotask(() => onLoad?.())), withCredentials: false,
    };
    vi.stubGlobal("XMLHttpRequest", vi.fn(function () { return request; }));
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);
    await uploadPrivateProjectFile("proj / private", new File(["one"], "private.txt"), LIMITS);
    expect(request.open).toHaveBeenCalledWith("POST", "/api/projects/proj%20%2F%20private/uploads");
    expect(fetchMock).not.toHaveBeenCalled();
    await expect(uploadPrivateProjectFile("proj_private", new File(["one"], "bad.png"), LIMITS)).rejects.toThrow("Private accepts documents only");
  });

  it.each([
    ["large.txt", "too large"], ["image.png", "one"], ["empty.txt", ""],
  ])("refuses unsupported Private upload %s without using any transport", async (name, content) => {
    const fetchMock = vi.fn();
    const xhrMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);
    vi.stubGlobal("XMLHttpRequest", xhrMock);
    await expect(uploadFile(new File([content], name), LIMITS, undefined, "con_private")).rejects.toThrow();
    expect(fetchMock).not.toHaveBeenCalled();
    expect(xhrMock).not.toHaveBeenCalled();
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

  it("creates a Private project with its immutable processing choice", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({ id: "proj_private", private_required: true }));
    vi.stubGlobal("fetch", fetchMock);
    await createProject("Private taxes", "Use original documents.", "private");
    expect(fetchMock).toHaveBeenCalledWith("/api/projects", expect.objectContaining({
      body: JSON.stringify({ name: "Private taxes", instructions: "Use original documents.", model_id: "private" }),
    }));
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
        body: JSON.stringify({ name: "Launch", instructions: "Keep answers concise.", model_id: "auto" }),
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
