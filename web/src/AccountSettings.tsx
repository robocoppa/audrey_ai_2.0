import { useEffect, useState, type FormEvent } from "react";

import {
  updateCurrentUserDisplayName,
  updateCurrentUserPreferences,
  type CurrentUser,
  type UserPreferences,
  type UserPreferencesUpdate,
} from "./api";

export function AccountSettings({
  user,
  preferences,
  onUserChange,
  onPreferencesChange,
  onClose,
}: {
  user: CurrentUser;
  preferences: UserPreferences;
  onUserChange: (user: CurrentUser) => void;
  onPreferencesChange: (preferences: UserPreferences) => void;
  onClose: () => void;
}) {
  const [displayName, setDisplayName] = useState(user.display_name);
  const [draft, setDraft] = useState<UserPreferencesUpdate>({
    timezone: preferences.timezone,
    persona: preferences.persona,
    detail: preferences.detail,
    tone: preferences.tone,
    show_progress: preferences.show_progress,
  });
  const [profileSaving, setProfileSaving] = useState(false);
  const [preferencesSaving, setPreferencesSaving] = useState(false);
  const [profileError, setProfileError] = useState("");
  const [preferencesError, setPreferencesError] = useState("");
  const busy = profileSaving || preferencesSaving;

  useEffect(() => {
    function closeOnEscape(event: KeyboardEvent) {
      if (event.key === "Escape" && !busy) onClose();
    }
    window.addEventListener("keydown", closeOnEscape);
    return () => window.removeEventListener("keydown", closeOnEscape);
  }, [busy, onClose]);

  async function saveProfile(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setProfileSaving(true);
    setProfileError("");
    try {
      const updated = await updateCurrentUserDisplayName(displayName);
      setDisplayName(updated.display_name);
      onUserChange(updated);
    } catch (reason) {
      setProfileError(messageOf(reason, "Profile name could not be saved."));
    } finally {
      setProfileSaving(false);
    }
  }

  async function savePreferences(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setPreferencesSaving(true);
    setPreferencesError("");
    try {
      const updated = await updateCurrentUserPreferences(draft);
      setDraft({
        timezone: updated.timezone,
        persona: updated.persona,
        detail: updated.detail,
        tone: updated.tone,
        show_progress: updated.show_progress,
      });
      onPreferencesChange(updated);
    } catch (reason) {
      setPreferencesError(messageOf(reason, "Preferences could not be saved."));
    } finally {
      setPreferencesSaving(false);
    }
  }

  function useDeviceTimezone() {
    const timezone = Intl.DateTimeFormat().resolvedOptions().timeZone;
    if (timezone) setDraft((current) => ({ ...current, timezone }));
  }

  return (
    <div
      className="account-settings-backdrop"
      role="presentation"
      onMouseDown={(event) => {
        if (event.currentTarget === event.target && !busy) onClose();
      }}
    >
      <section
        className="account-settings"
        role="dialog"
        aria-modal="true"
        aria-labelledby="account-settings-title"
      >
        <header className="account-settings-header">
          <div>
            <span>Audrey account</span>
            <h2 id="account-settings-title">Settings</h2>
          </div>
          <button type="button" onClick={onClose} disabled={busy} aria-label="Close settings">×</button>
        </header>

        <form className="settings-section" onSubmit={(event) => void saveProfile(event)}>
          <div className="settings-section-heading">
            <h3>Profile</h3>
            <p>This is the name Audrey displays in the application.</p>
          </div>
          <label>
            <span>Profile name</span>
            <input
              autoFocus
              maxLength={100}
              value={displayName}
              onChange={(event) => setDisplayName(event.target.value)}
              disabled={profileSaving}
            />
          </label>
          {profileError ? <p className="settings-error" role="alert">{profileError}</p> : null}
          <button type="submit" disabled={profileSaving || !displayName.trim()}>
            {profileSaving ? "Saving…" : "Save profile"}
          </button>
        </form>

        <form className="settings-section" onSubmit={(event) => void savePreferences(event)}>
          <div className="settings-section-heading">
            <h3>Audrey preferences</h3>
            <p>Saved by Audrey and applied to every new native response.</p>
          </div>
          <label>
            <span>Timezone</span>
            <div className="timezone-control">
              <input
                aria-label="Timezone"
                value={draft.timezone}
                maxLength={100}
                onChange={(event) => setDraft((current) => ({
                  ...current,
                  timezone: event.target.value,
                }))}
                placeholder="America/Denver"
                disabled={preferencesSaving}
              />
              <button type="button" onClick={useDeviceTimezone} disabled={preferencesSaving}>
                Use this device
              </button>
            </div>
            <small>Use an IANA name such as America/Denver or Europe/London.</small>
          </label>
          <label>
            <span>Persona and style</span>
            <textarea
              aria-label="Persona and style"
              rows={4}
              maxLength={4000}
              value={draft.persona}
              onChange={(event) => setDraft((current) => ({
                ...current,
                persona: event.target.value,
              }))}
              placeholder="For example: Be direct, practical, and explain uncommon terms."
              disabled={preferencesSaving}
            />
            <small>{draft.persona.length}/4000 characters</small>
          </label>
          <div className="settings-pair">
            <label>
              <span>Response detail</span>
              <select
                value={draft.detail}
                onChange={(event) => setDraft((current) => ({
                  ...current,
                  detail: event.target.value as UserPreferencesUpdate["detail"],
                }))}
                disabled={preferencesSaving}
              >
                <option value="concise">Concise</option>
                <option value="balanced">Balanced</option>
                <option value="detailed">Detailed</option>
              </select>
            </label>
            <label>
              <span>Tone</span>
              <select
                value={draft.tone}
                onChange={(event) => setDraft((current) => ({
                  ...current,
                  tone: event.target.value as UserPreferencesUpdate["tone"],
                }))}
                disabled={preferencesSaving}
              >
                <option value="natural">Natural</option>
                <option value="professional">Professional</option>
                <option value="casual">Casual</option>
              </select>
            </label>
          </div>
          <label className="settings-checkbox">
            <input
              type="checkbox"
              checked={draft.show_progress}
              onChange={(event) => setDraft((current) => ({
                ...current,
                show_progress: event.target.checked,
              }))}
              disabled={preferencesSaving}
            />
            <span>Show the live stage and source summary above the composer</span>
          </label>
          {preferencesError ? (
            <p className="settings-error" role="alert">{preferencesError}</p>
          ) : null}
          <button type="submit" disabled={preferencesSaving || !draft.timezone.trim()}>
            {preferencesSaving ? "Saving…" : "Save preferences"}
          </button>
        </form>
      </section>
    </div>
  );
}

function messageOf(reason: unknown, fallback: string): string {
  return reason instanceof Error ? reason.message : fallback;
}
