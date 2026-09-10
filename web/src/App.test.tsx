import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { App } from "./App";
import { latestActionFetch } from "./agentTransport";

const DEFAULT_PREFERENCES = {
  timezone: "UTC",
  persona: "",
  detail: "balanced",
  tone: "natural",
  show_progress: true,
  created_at: "2026-09-01T00:00:00+00:00",
  updated_at: "2026-09-01T00:00:00+00:00",
} as const;

describe("App", () => {
  afterEach(() => {
    cleanup();
    vi.useRealTimers();
    vi.unstubAllGlobals();
    window.history.replaceState({}, "", "/");
  });

  it("shows a centered, text-free Audrey Auto loader while the session resolves", () => {
    vi.stubGlobal("fetch", vi.fn(() => new Promise<Response>(() => undefined)));

    render(<App />);

    const loader = screen.getByRole("status", { name: "Loading Audrey" });
    expect(loader).toHaveClass("audrey-loader-fullscreen");
    expect(loader).toHaveTextContent("");
    expect(loader.querySelector(".audrey-loading-orbit")).toBeInTheDocument();
    expect(loader.querySelector("img")).toHaveAttribute(
      "src",
      expect.stringContaining("audrey2.png"),
    );
    expect(screen.queryByText("Checking session…")).not.toBeInTheDocument();
    expect(screen.queryByText("A quieter place to think.")).not.toBeInTheDocument();
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
        : path === "/api/me/preferences"
          ? DEFAULT_PREFERENCES
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
    expect(document.querySelector("#light-wordmark-on-dark feColorMatrix")).toHaveAttribute(
      "values",
      expect.stringContaining("0.843"),
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

  it("holds through multiple transient Access bootstrap rejections", async () => {
    let identityReads = 0;
    window.history.replaceState({}, "", "/?__cf_access_message=logged_out&kept=yes#chat");
    const fetchMock = vi.fn().mockImplementation((path: string) => {
      if (path === "/api/me") {
        identityReads += 1;
        if (identityReads <= 3) {
          return Promise.resolve(
            new Response(JSON.stringify({ detail: "Missing bearer token." }), {
              status: 401,
              headers: { "Content-Type": "application/json" },
            }),
          );
        }
        return Promise.resolve(
          new Response(JSON.stringify({
            id: "usr_example",
            email: "alice@example.com",
            display_name: "Alice Example",
            role: "user",
            status: "active",
            auth_provider: "cloudflare_access",
          }), {
            status: 200,
            headers: { "Content-Type": "application/json" },
          }),
        );
      }
      if (path === "/api/me/preferences") {
        return Promise.resolve(
          new Response(JSON.stringify(DEFAULT_PREFERENCES), {
            status: 200,
            headers: { "Content-Type": "application/json" },
          }),
        );
      }
      return Promise.resolve(
        new Response(JSON.stringify({ items: [], next_cursor: null }), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        }),
      );
    });
    vi.stubGlobal("fetch", fetchMock);

    render(<App />);

    expect(screen.getByRole("status", { name: "Finishing secure sign-in" })).toHaveTextContent(
      "Finishing your secure sign-in",
    );
    expect(screen.getByText(/Cloudflare Access is confirming this browser\./u)).toBeInTheDocument();
    expect(screen.queryByRole("banner")).not.toBeInTheDocument();

    const identity = await screen.findByLabelText(
      "Signed in user",
      undefined,
      { timeout: 7000 },
    );
    expect(identity).toHaveTextContent("Alice");
    expect(identityReads).toBe(4);
    expect(screen.queryByText("A quieter place to think.")).not.toBeInTheDocument();
    expect(window.location.search).toBe("?kept=yes");
    expect(window.location.hash).toBe("#chat");
    expect(window.location.href).not.toContain("__cf_access_message");
  });

  it("offers retry and logout after Access authentication remains unavailable", async () => {
    vi.useFakeTimers();
    let sessionReady = false;
    const fetchMock = vi.fn().mockImplementation((path: string) => {
      if (!sessionReady) {
        return Promise.resolve(new Response(JSON.stringify({ detail: "Not authenticated." }), {
          status: 401,
          headers: { "Content-Type": "application/json" },
        }));
      }
      const payload = path === "/api/me"
        ? {
            id: "usr_example",
            email: "alice@example.com",
            display_name: "Alice Example",
            role: "user",
            status: "active",
            auth_provider: "cloudflare_access",
          }
        : path === "/api/me/preferences"
          ? DEFAULT_PREFERENCES
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

    await act(async () => {
      await vi.runAllTimersAsync();
    });
    expect(screen.getByRole("alert")).toHaveTextContent(
      "did not establish this browser session within 30 seconds",
    );
    expect(screen.getByRole("heading", {
      name: "Sign-in is taking longer than expected",
    })).toBeInTheDocument();
    expect(screen.queryByRole("banner")).not.toBeInTheDocument();
    expect(screen.queryByText("A quieter place to think.")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Retry session" })).toBeEnabled();
    expect(screen.getByRole("link", { name: "Log out" })).toHaveAttribute(
      "href",
      "/cdn-cgi/access/logout",
    );
    expect(window.localStorage).toHaveLength(0);
    expect(window.sessionStorage).toHaveLength(0);

    vi.useRealTimers();
    sessionReady = true;
    fireEvent.click(screen.getByRole("button", { name: "Retry session" }));
    expect(await screen.findByLabelText("Signed in user")).toHaveTextContent(
      "Alice",
    );
    expect(fetchMock).toHaveBeenCalled();
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
        : path === "/api/me/preferences"
          ? DEFAULT_PREFERENCES
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
          : path === "/api/me/preferences"
            ? DEFAULT_PREFERENCES
            : { items: [], next_cursor: null };
        return Promise.resolve(new Response(JSON.stringify(payload), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        }));
      },
    );
    vi.stubGlobal("fetch", fetchMock);

    render(<App />);

    fireEvent.click(await screen.findByRole("button", { name: "Open account settings" }));
    fireEvent.change(screen.getByRole("textbox", { name: "Profile name" }), {
      target: { value: "Alice Example" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Save profile" }));

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

  it("updates validated Audrey preferences from account settings", async () => {
    const savedPreferences = {
      ...DEFAULT_PREFERENCES,
      timezone: "America/Denver",
      persona: "Be direct and practical.",
      detail: "concise" as const,
      tone: "professional" as const,
      show_progress: false,
    };
    const fetchMock = vi.fn().mockImplementation(
      (path: string, request?: RequestInit) => {
        if (path === "/api/me/preferences" && request?.method === "PUT") {
          return Promise.resolve(new Response(JSON.stringify(savedPreferences), {
            status: 200,
            headers: { "Content-Type": "application/json" },
          }));
        }
        const payload = path === "/api/me"
          ? {
              id: "usr_example",
              email: "alice@example.com",
              display_name: "Alice Example",
              role: "user",
              status: "active",
              auth_provider: "cloudflare_access",
            }
          : path === "/api/me/preferences"
            ? DEFAULT_PREFERENCES
            : { items: [], next_cursor: null };
        return Promise.resolve(new Response(JSON.stringify(payload), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        }));
      },
    );
    vi.stubGlobal("fetch", fetchMock);

    render(<App />);

    fireEvent.click(await screen.findByRole("button", { name: "Open account settings" }));
    fireEvent.change(screen.getByRole("textbox", { name: "Timezone" }), {
      target: { value: "America/Denver" },
    });
    fireEvent.change(screen.getByRole("textbox", { name: "Persona and style" }), {
      target: { value: "Be direct and practical." },
    });
    fireEvent.change(screen.getByRole("combobox", { name: "Response detail" }), {
      target: { value: "concise" },
    });
    fireEvent.change(screen.getByRole("combobox", { name: "Tone" }), {
      target: { value: "professional" },
    });
    fireEvent.click(screen.getByRole("checkbox", {
      name: "Show the live stage and source summary above the composer",
    }));
    fireEvent.click(screen.getByRole("button", { name: "Save preferences" }));

    await waitFor(() => {
      expect(screen.getByRole("button", { name: "Save preferences" })).not.toBeDisabled();
    });
    const update = fetchMock.mock.calls.find(
      ([path, request]) => path === "/api/me/preferences" && request?.method === "PUT",
    ) as [string, RequestInit] | undefined;
    expect(update).toBeDefined();
    expect(JSON.parse(String(update?.[1].body))).toEqual({
      timezone: "America/Denver",
      persona: "Be direct and practical.",
      detail: "concise",
      tone: "professional",
      show_progress: false,
    });
  });
  it("creates, reveals once, and revokes personal tokens without browser storage", async () => {
    const existingToken = {
      id: "pat_existing",
      name: "Laptop token",
      scopes: ["compat:full"],
      created_at: "2026-09-01T00:00:00+00:00",
      expires_at: "2099-10-01T00:00:00+00:00",
      last_used_at: null,
      revoked_at: null,
    };
    const createdToken = {
      ...existingToken,
      id: "pat_created",
      name: "CLI token",
      scopes: ["account:read", "compat:full"],
      token: "aud_pat_created.one-time-secret",
    };
    const fetchMock = vi.fn().mockImplementation(
      (path: string, request?: RequestInit) => {
        if (path === "/api/tokens" && request?.method === "POST") {
          return Promise.resolve(new Response(JSON.stringify(createdToken), {
            status: 201,
            headers: { "Content-Type": "application/json" },
          }));
        }
        if (path === "/api/tokens") {
          return Promise.resolve(new Response(JSON.stringify({ items: [existingToken] }), {
            status: 200,
            headers: { "Content-Type": "application/json" },
          }));
        }
        if (path === "/api/tokens/pat_existing" && request?.method === "DELETE") {
          return Promise.resolve(new Response(JSON.stringify({
            id: "pat_existing",
            revoked: true,
          }), {
            status: 200,
            headers: { "Content-Type": "application/json" },
          }));
        }
        const payload = path === "/api/me"
          ? {
              id: "usr_example",
              email: "alice@example.com",
              display_name: "Alice Example",
              role: "user",
              status: "active",
              auth_provider: "cloudflare_access",
            }
          : path === "/api/me/preferences"
            ? DEFAULT_PREFERENCES
            : { items: [], next_cursor: null };
        return Promise.resolve(new Response(JSON.stringify(payload), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        }));
      },
    );
    vi.stubGlobal("fetch", fetchMock);

    render(<App />);

    fireEvent.click(await screen.findByRole("button", { name: "Open account settings" }));
    fireEvent.click(screen.getByRole("button", { name: "Manage personal tokens" }));
    expect(await screen.findByText("Laptop token")).toBeInTheDocument();

    fireEvent.change(screen.getByRole("textbox", { name: "Token name" }), {
      target: { value: "CLI token" },
    });
    fireEvent.change(screen.getByRole("spinbutton", { name: "Token lifetime in days" }), {
      target: { value: "30" },
    });
    fireEvent.click(screen.getByRole("checkbox", {
      name: "Read Audrey account details and preferences",
    }));
    fireEvent.click(screen.getByRole("button", { name: "Create token" }));

    const secret = await screen.findByRole("textbox", { name: "New personal token" });
    expect(secret).toHaveValue("aud_pat_created.one-time-secret");
    expect(screen.getByRole("button", { name: "Close settings" })).toBeDisabled();
    const create = fetchMock.mock.calls.find(
      ([path, request]) => path === "/api/tokens" && request?.method === "POST",
    ) as [string, RequestInit] | undefined;
    expect(JSON.parse(String(create?.[1].body))).toEqual({
      name: "CLI token",
      scopes: ["compat:full", "account:read"],
      expires_in_days: 30,
    });
    expect(window.localStorage).toHaveLength(0);
    expect(window.sessionStorage).toHaveLength(0);

    fireEvent.click(screen.getByRole("button", { name: "I saved it" }));
    expect(screen.getByRole("button", { name: "Close settings" })).not.toBeDisabled();
    expect(screen.queryByRole("textbox", { name: "New personal token" })).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Revoke Laptop token" }));
    fireEvent.click(screen.getByRole("button", { name: "Confirm revoke" }));
    await waitFor(() => {
      expect(screen.queryByText("Laptop token")).not.toBeInTheDocument();
    });
    expect(fetchMock).toHaveBeenCalledWith(
      "/api/tokens/pat_existing",
      expect.objectContaining({ method: "DELETE", credentials: "same-origin" }),
    );
  });

  it("downloads every chat-export page and confirms durable account deletion", async () => {
    const firstMessage = {
      message_id: "msg_export_1",
      conversation_id: "con_export",
      conversation_title: "Exported conversation",
      conversation_created_at: "2026-09-01T00:00:00+00:00",
      conversation_updated_at: "2026-09-01T00:01:00+00:00",
      role: "user",
      content: "First export page",
      created_at: "2026-09-01T00:00:00+00:00",
      archived_at: "2026-09-01T00:02:00+00:00",
      partial: false,
      virtual_model: "audrey_fast",
      concrete_model: "test-model",
      prompt_tokens: 12,
      completion_tokens: 0,
    };
    const completedPurge = {
      schema_version: 1,
      purge_id: "purge_browser",
      cutoff_at: "2026-09-09T00:00:00+00:00",
      requested_at: "2026-09-09T00:00:00+00:00",
      status: "completed",
      completed_at: "2026-09-09T00:00:01+00:00",
      files: { pending: 0, attempts: 1, with_error: 0, completed: 1 },
      paths: { pending: 0, attempts: 1, with_error: 0, completed: 1 },
      local_delivery: { completed: true, attempts: 1, with_error: false },
      sidecar: {
        acknowledged: true,
        completed: true,
        status: "completed",
        attempts: 1,
        with_error: false,
      },
    };
    const fetchMock = vi.fn().mockImplementation(
      (path: string, request?: RequestInit) => {
        if (path === "/v1/me/chat-history/export?limit=200") {
          return Promise.resolve(new Response(JSON.stringify({
            schema_version: 1,
            items: [firstMessage],
            next_cursor: "next page",
          }), { status: 200, headers: { "Content-Type": "application/json" } }));
        }
        if (path === "/v1/me/chat-history/export?limit=200&cursor=next+page") {
          return Promise.resolve(new Response(JSON.stringify({
            schema_version: 1,
            items: [{ ...firstMessage, message_id: "msg_export_2", content: "Second page" }],
            next_cursor: null,
          }), { status: 200, headers: { "Content-Type": "application/json" } }));
        }
        if (path === "/v1/me/data-purge" && request?.method === "POST") {
          return Promise.resolve(new Response(JSON.stringify(completedPurge), {
            status: 202,
            headers: { "Content-Type": "application/json" },
          }));
        }
        const payload = path === "/api/me"
          ? {
              id: "usr_example",
              email: "alice@example.com",
              display_name: "Alice Example",
              role: "user",
              status: "active",
              auth_provider: "cloudflare_access",
            }
          : path === "/api/me/preferences"
            ? DEFAULT_PREFERENCES
            : { items: [], next_cursor: null };
        return Promise.resolve(new Response(JSON.stringify(payload), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        }));
      },
    );
    const createObjectURL = vi.fn().mockReturnValue("blob:audrey-export");
    const revokeObjectURL = vi.fn();
    vi.stubGlobal("fetch", fetchMock);
    vi.stubGlobal("URL", { createObjectURL, revokeObjectURL });
    let downloadedName = "";
    const click = vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(
      function captureDownload(this: HTMLAnchorElement) {
        downloadedName = this.download;
      },
    );

    render(<App />);

    fireEvent.click(await screen.findByRole("button", { name: "Open account settings" }));
    fireEvent.click(screen.getByRole("button", { name: "Download chat history" }));
    expect(await screen.findByText("Downloaded 2 archived messages.")).toBeInTheDocument();
    expect(createObjectURL).toHaveBeenCalledOnce();
    expect(downloadedName).toMatch(/^audrey-chat-history-\d{4}-\d{2}-\d{2}\.json$/u);
    const artifact = JSON.parse(
      await (createObjectURL.mock.calls[0][0] as Blob).text(),
    ) as { schema_version: number; items: Array<{ message_id: string }> };
    expect(artifact.schema_version).toBe(1);
    expect(artifact.items.map(({ message_id }) => message_id)).toEqual([
      "msg_export_1",
      "msg_export_2",
    ]);
    await waitFor(() => expect(revokeObjectURL).toHaveBeenCalledWith("blob:audrey-export"));

    fireEvent.click(screen.getByRole("button", { name: "Delete Audrey data" }));
    const destructiveButton = screen.getByRole("button", { name: "Delete all Audrey data" });
    expect(destructiveButton).toBeDisabled();
    fireEvent.change(screen.getByRole("textbox", { name: "Deletion confirmation" }), {
      target: { value: "DELETE ALL MY AUDREY DATA" },
    });
    expect(destructiveButton).toBeEnabled();
    fireEvent.click(destructiveButton);

    expect(await screen.findByRole("heading", { name: "Deletion complete" })).toBeVisible();
    const purgeRequest = fetchMock.mock.calls.find(
      ([path, request]) => path === "/v1/me/data-purge" && request?.method === "POST",
    ) as [string, RequestInit] | undefined;
    expect(JSON.parse(String(purgeRequest?.[1].body))).toEqual({
      confirmation: "DELETE ALL MY AUDREY DATA",
    });
    expect(new Headers(purgeRequest?.[1].headers).get("Idempotency-Key")).toMatch(
      /^native-ui-/u,
    );
    await waitFor(() => {
      expect(fetchMock.mock.calls.filter(([path]) => path === "/api/me")).toHaveLength(2);
    });
    expect(window.localStorage).toHaveLength(0);
    expect(window.sessionStorage).toHaveLength(0);
    click.mockRestore();
  });



  it("sends only the latest user action through the same-origin transport", async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(null, { status: 200 }));
    vi.stubGlobal("fetch", fetchMock);

    await latestActionFetch(
      "/api/agent?mode=fast",
      {
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
      },
      ["file_notes"],
    );

    expect(fetchMock).toHaveBeenCalledOnce();
    const [url, request] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toBe("/api/agent?mode=fast");
    expect(request.credentials).toBe("same-origin");
    const body = JSON.parse(String(request.body)) as {
      threadId: string;
      messages: Array<{ role: string; content: string }>;
      attachmentIds: string[];
    };
    expect(body.threadId).toBe("con_example");
    expect(body.messages).toHaveLength(1);
    expect(body.messages[0]).toEqual(
      expect.objectContaining({ role: "user", content: "Hello natively" }),
    );
    expect(body.attachmentIds).toEqual(["file_notes"]);
    expect(body).not.toHaveProperty("state");
    expect(body).not.toHaveProperty("tools");
  });
});
