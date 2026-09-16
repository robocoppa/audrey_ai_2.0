import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { AdminPanel } from "./AdminPanel";

const PENDING_USER = {
  id: "usr_pending",
  email: "pending@example.com",
  display_name: "Pending Person",
  role: "user",
  status: "pending",
  groups: [],
  auth_provider: "cloudflare_access",
  created_at: "2026-09-15T00:00:00+00:00",
  updated_at: "2026-09-15T00:00:00+00:00",
  last_seen_at: "2026-09-15T00:00:00+00:00",
} as const;

const DIRECT_MODEL = {
  id: "direct/qwen3.8-27b",
  label: "Qwen 3.8 27B",
  description: "Direct local model access.",
  kind: "direct",
  mode: "direct",
  presentation: "local",
  capabilities: ["text"],
  enabled: true,
  audience: "admins",
  concrete_model: "qwen3.8:27b",
  policy_overridden: false,
} as const;

describe("AdminPanel", () => {
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  it("approves pending accounts and updates server model policy", async () => {
    const changed = vi.fn();
    const fetchMock = vi.fn().mockImplementation(
      async (path: string, request?: RequestInit) => {
        if (path === "/api/admin/users") return jsonResponse({ items: [PENDING_USER] });
        if (path === "/api/admin/models") {
          return jsonResponse({ items: [DIRECT_MODEL], source: "ollama", warning: "" });
        }
        if (path === "/api/admin/users/usr_pending/approve") {
          return jsonResponse({
            ...PENDING_USER,
            status: "active",
            groups: ["users"],
          });
        }
        if (path === "/api/admin/users/usr_pending") {
          const patch = JSON.parse(String(request?.body)) as { groups: string[] };
          return jsonResponse({
            ...PENDING_USER,
            status: "active",
            groups: patch.groups,
          });
        }
        if (path === "/api/admin/models/direct%2Fqwen3.8-27b") {
          const patch = JSON.parse(String(request?.body)) as {
            enabled: boolean;
            audience: string;
          };
          return jsonResponse({ ...DIRECT_MODEL, ...patch, policy_overridden: true });
        }
        if (path === "/api/admin/model-policies/direct%2Fqwen3.8-27b") {
          return jsonResponse(DIRECT_MODEL);
        }
        throw new Error(`Unexpected request: ${path}`);
      },
    );
    vi.stubGlobal("fetch", fetchMock);

    render(
      <AdminPanel
        currentUserId="usr_admin"
        onChanged={changed}
        onClose={vi.fn()}
      />,
    );

    fireEvent.click(await screen.findByRole("button", { name: "Approve as user" }));
    await waitFor(() => expect(screen.getByText("active")).toBeVisible());
    expect(fetchMock).toHaveBeenCalledWith(
      "/api/admin/users/usr_pending/approve",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({ tester: false }),
      }),
    );

    fireEvent.change(screen.getByRole("combobox", { name: "Role for Pending Person" }), {
      target: { value: "tester" },
    });
    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith(
      "/api/admin/users/usr_pending",
      expect.objectContaining({
        method: "PATCH",
        body: JSON.stringify({ groups: ["testers", "users"] }),
      }),
    ));

    fireEvent.click(screen.getByRole("tab", { name: /Models/u }));
    expect(screen.getByText("Live Ollama inventory")).toBeVisible();
    fireEvent.click(screen.getByRole("button", { name: "Enabled" }));
    await waitFor(() => expect(screen.getByRole("button", { name: "Disabled" })).toBeVisible());
    expect(fetchMock).toHaveBeenCalledWith(
      "/api/admin/models/direct%2Fqwen3.8-27b",
      expect.objectContaining({
        method: "PATCH",
        body: JSON.stringify({ enabled: false, audience: "admins" }),
      }),
    );
    fireEvent.click(screen.getByRole("button", { name: "Reset default" }));
    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith(
      "/api/admin/model-policies/direct%2Fqwen3.8-27b",
      expect.objectContaining({ method: "DELETE" }),
    ));
    expect(changed).toHaveBeenCalledTimes(4);
  });

  it("confirms permanent account deletion and shows durable progress", async () => {
    const fetchMock = vi.fn().mockImplementation(async (path: string, request?: RequestInit) => {
      if (path === "/api/admin/users" && !request?.method) {
        return jsonResponse({ items: [PENDING_USER] });
      }
      if (path === "/api/admin/models") {
        return jsonResponse({ items: [], source: "ollama", warning: "" });
      }
      if (path === "/api/admin/users/usr_pending" && request?.method === "DELETE") {
        return jsonResponse({ id: "usr_pending", status: "deleting", purge_id: "purge_test" });
      }
      throw new Error(`Unexpected request: ${path}`);
    });
    vi.stubGlobal("fetch", fetchMock);
    render(<AdminPanel currentUserId="usr_admin" onChanged={vi.fn()} onClose={vi.fn()} />);

    fireEvent.click(await screen.findByRole("button", { name: "Delete account Pending Person" }));
    expect(fetchMock).not.toHaveBeenCalledWith(
      "/api/admin/users/usr_pending", expect.objectContaining({ method: "DELETE" }),
    );
    fireEvent.click(screen.getByRole("button", { name: "Yes, delete account" }));
    await waitFor(() => expect(screen.getByText("deleting")).toBeVisible());
    expect(fetchMock).toHaveBeenCalledWith(
      "/api/admin/users/usr_pending", expect.objectContaining({ method: "DELETE" }),
    );
    expect(screen.getByText("Purging data…")).toBeVisible();
  });
});

function jsonResponse(payload: unknown): Response {
  return new Response(JSON.stringify(payload), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });
}
