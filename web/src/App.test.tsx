import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
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
            display_name: "Alice Example",
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

    const identity = await screen.findByLabelText("Signed in user");
    expect(screen.getByRole("link", { name: "Audrey home" })).toContainElement(
      document.querySelector(".brand-wordmark img"),
    );
    expect(document.querySelector(".brand-wordmark img")).toHaveAttribute(
      "src",
      expect.stringContaining("builtryte-wordmark.png"),
    );
    expect(identity).toHaveTextContent("Alice");
    expect(identity).not.toHaveTextContent("Example");
    expect(screen.queryByText("Authenticated")).not.toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Log out" })).toHaveAttribute(
      "href",
      "/cdn-cgi/access/logout",
    );
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

  it("uses the account handle when the provider has no display name", async () => {
    const fetchMock = vi.fn().mockImplementation((path: string) => {
      const payload = path === "/api/me"
        ? {
            id: "usr_example",
            email: "alice@example.com",
            display_name: "",
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

    expect(await screen.findByLabelText("Signed in user")).toHaveTextContent("alice");
  });

  it("updates the current user's profile name and refreshes the header", async () => {
    let currentName = "";
    const fetchMock = vi.fn().mockImplementation(
      (path: string, request?: RequestInit) => {
        if (path === "/api/me" && request?.method === "PATCH") {
          currentName = "Alice Example";
          return Promise.resolve(new Response(JSON.stringify({
            id: "usr_example",
            email: "alice@example.com",
            display_name: currentName,
            role: "user",
            status: "active",
            auth_provider: "cloudflare_access",
          }), { status: 200, headers: { "Content-Type": "application/json" } }));
        }
        const payload = path === "/api/me"
          ? {
              id: "usr_example",
              email: "alice@example.com",
              display_name: currentName,
              role: "user",
              status: "active",
              auth_provider: "cloudflare_access",
            }
          : { items: [], next_cursor: null };
        return Promise.resolve(new Response(JSON.stringify(payload), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        }));
      },
    );
    vi.stubGlobal("fetch", fetchMock);

    render(<App />);

    fireEvent.click(await screen.findByRole("button", { name: "Edit profile name" }));
    fireEvent.change(screen.getByRole("textbox", { name: "Profile name" }), {
      target: { value: "Alice Example" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Save" }));

    await waitFor(() => {
      expect(screen.getByLabelText("Signed in user")).toHaveTextContent("Alice");
    });
    const patch = fetchMock.mock.calls.find(
      ([path, request]) => path === "/api/me" && request?.method === "PATCH",
    ) as [string, RequestInit] | undefined;
    expect(patch).toBeDefined();
    expect(JSON.parse(String(patch?.[1].body))).toEqual({
      display_name: "Alice Example",
    });
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
