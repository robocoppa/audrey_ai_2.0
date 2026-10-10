import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { AccountSettings } from "./AccountSettings";
import type { CurrentUser, UserPreferences } from "./api";

const USER: CurrentUser = {
  id: "usr_example",
  email: "alice@example.com",
  display_name: "Alice",
  role: "user",
  status: "active",
  groups: [],
  auth_provider: "cloudflare_access",
};

const PREFERENCES: UserPreferences = {
  timezone: "UTC",
  persona: "",
  detail: "balanced",
  tone: "natural",
  show_progress: true,
  created_at: "2026-10-09T00:00:00+00:00",
  updated_at: "2026-10-09T00:00:00+00:00",
};

function openSettings(onClose = vi.fn()) {
  return render(
    <AccountSettings
      user={USER}
      preferences={PREFERENCES}
      onUserChange={vi.fn()}
      onPreferencesChange={vi.fn()}
      onDataPurgeAttempted={vi.fn()}
      onClose={onClose}
    />,
  );
}

describe("AccountSettings keyboard focus", () => {
  afterEach(cleanup);

  it("opens on a button instead of focusing a field", () => {
    openSettings();

    expect(screen.getByRole("button", { name: "Close settings" })).toHaveFocus();
    expect(screen.getByRole("textbox", { name: "Profile name" })).not.toHaveFocus();
  });

  it("still dismisses with Escape after moving focus into the dialog", () => {
    const onClose = vi.fn();
    openSettings(onClose);

    fireEvent.keyDown(screen.getByRole("button", { name: "Close settings" }), {
      key: "Escape",
    });

    expect(onClose).toHaveBeenCalledOnce();
  });
});
