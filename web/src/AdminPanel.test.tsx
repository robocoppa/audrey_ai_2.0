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
  audience: "testers",
  concrete_model: "qwen3.8:27b",
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
        if (path === "/api/admin/models") return jsonResponse({ items: [DIRECT_MODEL] });
        if (path === "/api/admin/users/usr_pending/approve") {
          return jsonResponse({
            ...PENDING_USER,
            status: "active",
            groups: ["users"],
          });
        }
        if (path === "/api/admin/models/direct%2Fqwen3.8-27b") {
          const patch = JSON.parse(String(request?.body)) as {
            enabled: boolean;
            audience: string;
          };
          return jsonResponse({ ...DIRECT_MODEL, ...patch });
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

    fireEvent.click(await screen.findByRole("button", { name: "Approve" }));
    await waitFor(() => expect(screen.getByText("active")).toBeVisible());
    expect(fetchMock).toHaveBeenCalledWith(
      "/api/admin/users/usr_pending/approve",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({ tester: false }),
      }),
    );

    fireEvent.click(screen.getByRole("button", { name: "Enabled" }));
    await waitFor(() => expect(screen.getByRole("button", { name: "Disabled" })).toBeVisible());
    expect(fetchMock).toHaveBeenCalledWith(
      "/api/admin/models/direct%2Fqwen3.8-27b",
      expect.objectContaining({
        method: "PATCH",
        body: JSON.stringify({ enabled: false, audience: "testers" }),
      }),
    );
    expect(changed).toHaveBeenCalledTimes(2);
  });
});

function jsonResponse(payload: unknown): Response {
  return new Response(JSON.stringify(payload), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });
}
