import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { App } from "./App";
import { latestActionFetch } from "./agentTransport";

describe("App", () => {
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  it("loads the current same-origin Audrey identity", async () => {
    const fetchMock = vi.fn().mockImplementation((path: string) => {
      const payload = path === "/api/me"
        ? {
            id: "usr_example",
            email: "alice@example.com",
            display_name: "Alice",
            role: "user",
            status: "active",
            auth_provider: "cloudflare_access",
          }
        : { items: [], next_cursor: null };
      return Promise.resolve(
        new Response(JSON.stringify(payload), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        }),
      );
    });
    vi.stubGlobal("fetch", fetchMock);

    render(<App />);

    expect(await screen.findByText("Alice")).toBeInTheDocument();
    expect(screen.getByText("Authenticated")).toBeInTheDocument();
    expect(fetchMock).toHaveBeenCalledWith(
      "/api/me",
      expect.objectContaining({ credentials: "same-origin" }),
    );
  });

  it("shows a sign-in message without storing a browser token", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ detail: "Not authenticated." }), {
          status: 401,
          headers: { "Content-Type": "application/json" },
        }),
      ),
    );

    render(<App />);

    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Sign in through the Audrey",
    );
    expect(window.localStorage).toHaveLength(0);
    expect(window.sessionStorage).toHaveLength(0);
  });

  it("sends only the latest user action through the same-origin transport", async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(null, { status: 200 }));
    vi.stubGlobal("fetch", fetchMock);

    await latestActionFetch("/api/agent?mode=fast", {
      method: "POST",
      body: JSON.stringify({
        threadId: "con_example",
        runId: "run_example",
        messages: [
          { id: "prior-user", role: "user", content: "Prior question" },
          { id: "prior-assistant", role: "assistant", content: "Prior answer" },
          { id: "latest-user", role: "user", content: "Hello natively" },
        ],
        state: { browserOwned: false },
        tools: [{ name: "browser_tool" }],
      }),
    });

    expect(fetchMock).toHaveBeenCalledOnce();
    const [url, request] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toBe("/api/agent?mode=fast");
    expect(request.credentials).toBe("same-origin");
    const body = JSON.parse(String(request.body)) as {
      threadId: string;
      messages: Array<{ role: string; content: string }>;
    };
    expect(body.threadId).toBe("con_example");
    expect(body.messages).toHaveLength(1);
    expect(body.messages[0]).toEqual(
      expect.objectContaining({ role: "user", content: "Hello natively" }),
    );
    expect(body).not.toHaveProperty("state");
    expect(body).not.toHaveProperty("tools");
  });
});
