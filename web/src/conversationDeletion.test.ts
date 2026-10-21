import { afterEach, describe, expect, test, vi } from "vitest";
import { deleteConversation } from "./api";

const ID = "con / private";
const PATH = "/api/conversations/con%20%2F%20private";

function errorResponse(status: number, detail: string): Response {
  return new Response(JSON.stringify({ detail }), {
    status, headers: { "Content-Type": "application/json" },
  });
}

function requests(fetchMock: ReturnType<typeof vi.fn>) {
  return fetchMock.mock.calls.map(([path, init]) => ({ path, method: init?.method ?? "GET" }));
}

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe("conversation deletion acknowledgements", () => {
  test("a successful delete does not make a verification request", async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(null, { status: 204 }));
    vi.stubGlobal("fetch", fetchMock);
    await expect(deleteConversation(ID)).resolves.toBeUndefined();
    expect(requests(fetchMock)).toEqual([{ path: PATH, method: "DELETE" }]);
  });

  test("an upstream failure is resolved only by canonical absence of the same conversation", async () => {
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(errorResponse(520, "The proxy lost the acknowledgement."))
      .mockResolvedValueOnce(errorResponse(404, "Conversation not found."));
    vi.stubGlobal("fetch", fetchMock);
    await expect(deleteConversation(ID)).resolves.toBeUndefined();
    expect(requests(fetchMock)).toEqual([
      { path: PATH, method: "DELETE" }, { path: PATH, method: "GET" },
    ]);
  });

  test("a lost network response can be verified without repeating the delete", async () => {
    const fetchMock = vi.fn()
      .mockRejectedValueOnce(new TypeError("Failed to fetch"))
      .mockResolvedValueOnce(errorResponse(404, "Conversation not found."));
    vi.stubGlobal("fetch", fetchMock);
    await expect(deleteConversation(ID)).resolves.toBeUndefined();
    expect(requests(fetchMock)).toEqual([
      { path: PATH, method: "DELETE" }, { path: PATH, method: "GET" },
    ]);
  });

  test.each(["still exists", "verification unavailable", "proxy not found"])(
    "preserves the original deletion error when %s",
    async (outcome) => {
      const verification = outcome === "still exists"
        ? new Response(JSON.stringify({ id: ID }), { status: 200 })
        : outcome === "verification unavailable"
          ? errorResponse(503, "The verification service is unavailable.")
          : new Response("<html>Proxy route not found</html>", { status: 404 });
      const fetchMock = vi.fn()
        .mockResolvedValueOnce(errorResponse(520, "Original deletion failure."))
        .mockResolvedValueOnce(verification);
      vi.stubGlobal("fetch", fetchMock);
      await expect(deleteConversation(ID)).rejects.toMatchObject({
        name: "ApiError", status: 520, message: "Original deletion failure.",
      });
      expect(requests(fetchMock)).toEqual([
        { path: PATH, method: "DELETE" }, { path: PATH, method: "GET" },
      ]);
    },
  );

  test.each([401, 403, 409])("does not verify a refused deletion with HTTP %i", async (status) => {
    const fetchMock = vi.fn().mockResolvedValue(errorResponse(status, "Deletion refused."));
    vi.stubGlobal("fetch", fetchMock);
    await expect(deleteConversation(ID)).rejects.toMatchObject({ status, message: "Deletion refused." });
    expect(requests(fetchMock)).toEqual([{ path: PATH, method: "DELETE" }]);
  });

  test("does not turn an aborted delete into a verification request", async () => {
    const aborted = new DOMException("The request was aborted.", "AbortError");
    const fetchMock = vi.fn().mockRejectedValue(aborted);
    vi.stubGlobal("fetch", fetchMock);
    await expect(deleteConversation(ID)).rejects.toBe(aborted);
    expect(requests(fetchMock)).toEqual([{ path: PATH, method: "DELETE" }]);
  });

  test("bounds a stalled verification and restores the original error", async () => {
    vi.useFakeTimers();
    let verificationSignal: AbortSignal | undefined;
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(errorResponse(520, "Original deletion failure."))
      .mockImplementationOnce((_path: string, init: RequestInit) => new Promise<Response>((_resolve, reject) => {
        verificationSignal = init.signal ?? undefined;
        verificationSignal?.addEventListener("abort", () => {
          reject(new DOMException("Verification timed out.", "AbortError"));
        }, { once: true });
      }));
    vi.stubGlobal("fetch", fetchMock);
    const pending = deleteConversation(ID);
    const rejected = expect(pending).rejects.toMatchObject({ status: 520, message: "Original deletion failure." });
    await vi.waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2));
    expect(verificationSignal?.aborted).toBe(false);
    await vi.advanceTimersByTimeAsync(5000);
    await rejected;
    expect(verificationSignal?.aborted).toBe(true);
    expect(vi.getTimerCount()).toBe(0);
    expect(requests(fetchMock)).toEqual([
      { path: PATH, method: "DELETE" }, { path: PATH, method: "GET" },
    ]);
  });
});
