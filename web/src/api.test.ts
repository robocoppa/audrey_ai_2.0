import { afterEach, describe, expect, it, vi } from "vitest";

import { uploadFile, type AudreyFileLimits } from "./api";

const LIMITS: AudreyFileLimits = {
  max_upload_bytes: 4,
  max_user_bytes: 1_000,
  allowed_extensions: [".txt"],
  chunked_max_bytes: 20,
  part_size: 4,
};

describe("native file uploads", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("uses multipart for a file within the single-request limit", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({
      id: "file_small",
      filename: "small.txt",
      mime: "text/plain",
      bytes: 4,
      kind: "text",
      chunks: 1,
      status: "ready",
    }));
    vi.stubGlobal("fetch", fetchMock);
    const progress: number[] = [];

    const result = await uploadFile(
      new File(["tiny"], "small.txt", { type: "text/plain" }),
      LIMITS,
      (value) => progress.push(value),
    );

    expect(result.id).toBe("file_small");
    expect(fetchMock).toHaveBeenCalledOnce();
    const [path, request] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(path).toBe("/api/files");
    expect(request.method).toBe("POST");
    expect(request.body).toBeInstanceOf(FormData);
    expect(progress).toEqual([0, 1]);
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

function jsonResponse(payload: unknown): Response {
  return new Response(JSON.stringify(payload), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });
}
