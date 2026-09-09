import { useEffect, useState, type FormEvent } from "react";

import {
  createPersonalToken,
  exportChatHistory,
  getAccountDataPurge,
  listPersonalTokens,
  requestAccountDataPurge,
  revokePersonalToken,
  updateCurrentUserDisplayName,
  updateCurrentUserPreferences,
  type AccountPurgeStatus,
  type CurrentUser,
  type PersonalTokenRecord,
  type PersonalTokenScope,
  type UserPreferences,
  type UserPreferencesUpdate,
} from "./api";

export function AccountSettings({
  user,
  preferences,
  onUserChange,
  onPreferencesChange,
  onDataPurgeAttempted,
  onClose,
}: {
  user: CurrentUser;
  preferences: UserPreferences;
  onUserChange: (user: CurrentUser) => void;
  onPreferencesChange: (preferences: UserPreferences) => void;
  onDataPurgeAttempted: () => void;
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
  const [tokensOpen, setTokensOpen] = useState(false);
  const [tokens, setTokens] = useState<PersonalTokenRecord[]>([]);
  const [tokensLoading, setTokensLoading] = useState(false);
  const [tokenCreating, setTokenCreating] = useState(false);
  const [revokingTokenId, setRevokingTokenId] = useState("");
  const [revokeConfirmId, setRevokeConfirmId] = useState("");
  const [tokenName, setTokenName] = useState("");
  const [tokenExpiresInDays, setTokenExpiresInDays] = useState("90");
  const [tokenScopes, setTokenScopes] = useState<PersonalTokenScope[]>([
    "compat:full",
  ]);
  const [issuedToken, setIssuedToken] = useState("");
  const [tokenCopied, setTokenCopied] = useState(false);
  const [tokenError, setTokenError] = useState("");
  const [dataExporting, setDataExporting] = useState(false);
  const [dataMessage, setDataMessage] = useState("");
  const [dataError, setDataError] = useState("");
  const [purgeConfirmOpen, setPurgeConfirmOpen] = useState(false);
  const [purgeConfirmation, setPurgeConfirmation] = useState("");
  const [purgeRequesting, setPurgeRequesting] = useState(false);
  const [purgePolling, setPurgePolling] = useState(false);
  const [purgeIdempotencyKey, setPurgeIdempotencyKey] = useState("");
  const [purgeStatus, setPurgeStatus] = useState<AccountPurgeStatus | null>(null);
  const tokenBusy = tokensLoading || tokenCreating || Boolean(revokingTokenId);
  const dataBusy = dataExporting || purgeRequesting;
  const busy = profileSaving || preferencesSaving || tokenBusy || dataBusy
    || Boolean(issuedToken);

  useEffect(() => {
    function closeOnEscape(event: KeyboardEvent) {
      if (event.key === "Escape" && !busy) onClose();
    }
    window.addEventListener("keydown", closeOnEscape);
    return () => window.removeEventListener("keydown", closeOnEscape);
  }, [busy, onClose]);

  useEffect(() => {
    if (!purgeStatus || purgeStatus.status !== "pending") return undefined;
    let active = true;
    let checking = false;
    const timer = window.setInterval(() => {
      if (checking) return;
      checking = true;
      setPurgePolling(true);
      void getAccountDataPurge(purgeStatus.purge_id)
        .then((updated) => {
          if (!active) return;
          setPurgeStatus(updated);
          setDataError("");
        })
        .catch((reason: unknown) => {
          if (active) {
            setDataError(messageOf(
              reason,
              "Deletion status could not be refreshed. Audrey will keep retrying cleanup.",
            ));
          }
        })
        .finally(() => {
          checking = false;
          if (active) setPurgePolling(false);
        });
    }, 1500);
    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, [purgeStatus]);

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
  async function loadTokens() {
    setTokensLoading(true);
    setTokenError("");
    try {
      const response = await listPersonalTokens();
      setTokens(response.items.filter(({ revoked_at }) => !revoked_at));
    } catch (reason) {
      setTokenError(messageOf(reason, "Personal tokens could not be loaded."));
    } finally {
      setTokensLoading(false);
    }
  }

  async function openTokens() {
    setTokensOpen(true);
    await loadTokens();
  }

  function toggleTokenScope(scope: PersonalTokenScope, checked: boolean) {
    setTokenScopes((current) => {
      if (checked) {
        return current.includes(scope) ? current : [...current, scope];
      }
      return current.filter((item) => item !== scope);
    });
  }

  async function issueToken(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setTokenCreating(true);
    setTokenError("");
    setIssuedToken("");
    setTokenCopied(false);
    try {
      const created = await createPersonalToken({
        name: tokenName.trim(),
        scopes: tokenScopes,
        expires_in_days: Number(tokenExpiresInDays),
      });
      const { token, ...record } = created;
      setTokens((current) => [
        record,
        ...current.filter(({ id }) => id !== record.id),
      ]);
      setIssuedToken(token);
      setTokenName("");
    } catch (reason) {
      setTokenError(messageOf(reason, "Personal token could not be created."));
    } finally {
      setTokenCreating(false);
    }
  }

  async function revokeToken(tokenId: string) {
    setRevokingTokenId(tokenId);
    setTokenError("");
    try {
      await revokePersonalToken(tokenId);
      setTokens((current) => current.filter(({ id }) => id !== tokenId));
      setRevokeConfirmId("");
    } catch (reason) {
      setTokenError(messageOf(reason, "Personal token could not be revoked."));
    } finally {
      setRevokingTokenId("");
    }
  }

  async function copyIssuedToken() {
    try {
      if (!navigator.clipboard) throw new Error("Clipboard access is unavailable.");
      await navigator.clipboard.writeText(issuedToken);
      setTokenCopied(true);
    } catch {
      setTokenError("Copy failed. Select the token and copy it manually.");
    }
  }

  async function downloadChatExport() {
    setDataExporting(true);
    setDataError("");
    setDataMessage("");
    try {
      const archive = await exportChatHistory();
      const exportedAt = new Date().toISOString();
      downloadJson(
        { ...archive, exported_at: exportedAt },
        `audrey-chat-history-${exportedAt.slice(0, 10)}.json`,
      );
      const noun = archive.items.length === 1 ? "message" : "messages";
      setDataMessage(`Downloaded ${archive.items.length} archived ${noun}.`);
    } catch (reason) {
      setDataError(messageOf(reason, "Chat history could not be exported."));
    } finally {
      setDataExporting(false);
    }
  }

  async function requestDataPurge(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (purgeConfirmation !== ACCOUNT_PURGE_CONFIRMATION) return;
    const idempotencyKey = purgeIdempotencyKey || newPurgeIdempotencyKey();
    if (!purgeIdempotencyKey) setPurgeIdempotencyKey(idempotencyKey);
    setPurgeRequesting(true);
    setDataError("");
    setDataMessage("");
    try {
      const status = await requestAccountDataPurge(
        purgeConfirmation,
        idempotencyKey,
      );
      setPurgeStatus(status);
      setTokens([]);
      setIssuedToken("");
      setTokenCopied(false);
      onDataPurgeAttempted();
    } catch (reason) {
      setDataError(messageOf(reason, "Audrey data deletion could not be started."));
      onDataPurgeAttempted();
    } finally {
      setPurgeRequesting(false);
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
        <div className="account-settings-header">
          <div>
            <span>Audrey account</span>
            <h2 id="account-settings-title">Settings</h2>
          </div>
          <button type="button" onClick={onClose} disabled={busy} aria-label="Close settings">×</button>
        </div>

        {purgeStatus ? (
          <section className="settings-section purge-receipt" aria-labelledby="purge-status-title">
            <div className="settings-section-heading">
              <h3 id="purge-status-title">{purgeStatusTitle(purgeStatus.status)}</h3>
              <p>{purgeStatusDescription(purgeStatus.status)}</p>
            </div>
            <dl className="purge-progress">
              <div>
                <dt>Files remaining</dt>
                <dd>{purgeStatus.files.pending}</dd>
              </div>
              <div>
                <dt>Disk items remaining</dt>
                <dd>{purgeStatus.paths.pending}</dd>
              </div>
              <div>
                <dt>Archive cleanup</dt>
                <dd>{purgeStatus.sidecar.completed ? "Complete" : "Pending"}</dd>
              </div>
            </dl>
            {purgePolling ? <p className="data-message" role="status">Checking cleanup…</p> : null}
            {dataError ? <p className="settings-error" role="alert">{dataError}</p> : null}
          </section>
        ) : (
          <>
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
        <section className="settings-section" aria-labelledby="personal-token-title">
          <div className="settings-section-heading">
            <h3 id="personal-token-title">Personal access tokens</h3>
            <p>
              Create expiring credentials for API clients. Audrey shows each
              secret once and never stores it in this browser.
            </p>
          </div>
          {!tokensOpen ? (
            <button type="button" onClick={() => void openTokens()}>
              Manage personal tokens
            </button>
          ) : (
            <div className="token-manager">
              <form className="token-create-form" onSubmit={(event) => void issueToken(event)}>
                <label>
                  <span>Token name</span>
                  <input
                    aria-label="Token name"
                    autoComplete="off"
                    maxLength={80}
                    required
                    value={tokenName}
                    onChange={(event) => setTokenName(event.target.value)}
                    disabled={tokenCreating}
                    placeholder="Laptop API client"
                  />
                </label>
                <label>
                  <span>Expires after</span>
                  <input
                    aria-label="Token lifetime in days"
                    type="number"
                    min={1}
                    max={365}
                    required
                    value={tokenExpiresInDays}
                    onChange={(event) => setTokenExpiresInDays(event.target.value)}
                    disabled={tokenCreating}
                  />
                  <small>Choose between 1 and 365 days.</small>
                </label>
                <fieldset className="token-scope-fieldset">
                  <legend>Access</legend>
                  <label className="settings-checkbox">
                    <input
                      type="checkbox"
                      checked={tokenScopes.includes("compat:full")}
                      onChange={(event) => toggleTokenScope("compat:full", event.target.checked)}
                      disabled={tokenCreating}
                    />
                    <span>Use Audrey's OpenAI-compatible API</span>
                  </label>
                  <label className="settings-checkbox">
                    <input
                      type="checkbox"
                      checked={tokenScopes.includes("account:read")}
                      onChange={(event) => toggleTokenScope("account:read", event.target.checked)}
                      disabled={tokenCreating}
                    />
                    <span>Read Audrey account details and preferences</span>
                  </label>
                </fieldset>
                <button
                  type="submit"
                  disabled={
                    tokenCreating
                    || !tokenName.trim()
                    || tokenScopes.length === 0
                    || !validTokenExpiry(tokenExpiresInDays)
                  }
                >
                  {tokenCreating ? "Creating…" : "Create token"}
                </button>
              </form>

              {issuedToken ? (
                <div className="token-secret" role="status" aria-live="polite">
                  <strong>Copy this token now. Audrey cannot show it again.</strong>
                  <label>
                    <span>New personal token</span>
                    <textarea
                      aria-label="New personal token"
                      value={issuedToken}
                      readOnly
                      rows={3}
                      spellCheck={false}
                    />
                  </label>
                  <div className="token-actions">
                    <button type="button" onClick={() => void copyIssuedToken()}>
                      {tokenCopied ? "Copied" : "Copy token"}
                    </button>
                    <button
                      type="button"
                      onClick={() => {
                        setIssuedToken("");
                        setTokenCopied(false);
                      }}
                    >
                      I saved it
                    </button>
                  </div>
                </div>
              ) : null}

              {tokenError ? <p className="settings-error" role="alert">{tokenError}</p> : null}
              {tokensLoading ? (
                <p className="token-empty" role="status">Loading personal tokens…</p>
              ) : tokens.length === 0 ? (
                <p className="token-empty">No active personal tokens.</p>
              ) : (
                <ul className="token-list" aria-label="Personal access tokens">
                  {tokens.map((record) => (
                    <li key={record.id}>
                      <div className="token-record-heading">
                        <strong>{record.name}</strong>
                        <span className={tokenExpired(record) ? "token-status expired" : "token-status"}>
                          {tokenExpired(record) ? "Expired" : "Active"}
                        </span>
                      </div>
                      <dl>
                        <div>
                          <dt>Access</dt>
                          <dd>{record.scopes.map(tokenScopeLabel).join(", ")}</dd>
                        </div>
                        <div>
                          <dt>Expires</dt>
                          <dd><time dateTime={record.expires_at}>{formatTokenTime(record.expires_at)}</time></dd>
                        </div>
                        <div>
                          <dt>Last used</dt>
                          <dd>
                            {record.last_used_at ? (
                              <time dateTime={record.last_used_at}>
                                {formatTokenTime(record.last_used_at)}
                              </time>
                            ) : "Never"}
                          </dd>
                        </div>
                      </dl>
                      {revokeConfirmId === record.id ? (
                        <div className="token-actions">
                          <button
                            className="danger-button"
                            type="button"
                            onClick={() => void revokeToken(record.id)}
                            disabled={Boolean(revokingTokenId)}
                          >
                            {revokingTokenId === record.id ? "Revoking…" : "Confirm revoke"}
                          </button>
                          <button
                            type="button"
                            onClick={() => setRevokeConfirmId("")}
                            disabled={Boolean(revokingTokenId)}
                          >
                            Keep token
                          </button>
                        </div>
                      ) : (
                        <button
                          className="token-revoke-button"
                          type="button"
                          aria-label={`Revoke ${record.name}`}
                          onClick={() => setRevokeConfirmId(record.id)}
                          disabled={tokenBusy}
                        >
                          Revoke
                        </button>
                      )}
                    </li>
                  ))}
                </ul>
              )}
            </div>
          )}
        </section>

        <section className="settings-section" aria-labelledby="account-data-title">
          <div className="settings-section-heading">
            <h3 id="account-data-title">Your Audrey data</h3>
            <p>
              Download the current chat-search archive or permanently remove
              Audrey-owned personal data from every managed store.
            </p>
          </div>
          <div className="data-export-control">
            <div>
              <strong>Export chat history</strong>
              <p>
                Downloads all currently archived messages as JSON. Uploaded
                file contents, memories, and conversations still awaiting
                archive delivery are not included.
              </p>
            </div>
            <button
              type="button"
              onClick={() => void downloadChatExport()}
              disabled={dataBusy}
            >
              {dataExporting ? "Preparing export…" : "Download chat history"}
            </button>
          </div>
          {dataMessage ? <p className="data-message" role="status">{dataMessage}</p> : null}
          {dataError ? <p className="settings-error" role="alert">{dataError}</p> : null}

          <div className="data-danger-zone">
            <div>
              <strong>Delete all Audrey data</strong>
              <p>
                Permanently deletes conversations, runs, personal tokens,
                uploads, memories, and search history, and resets Audrey
                preferences. Your sign-in identity and profile remain.
              </p>
            </div>
            {!purgeConfirmOpen ? (
              <button
                className="danger-button"
                type="button"
                onClick={() => setPurgeConfirmOpen(true)}
                disabled={dataBusy || Boolean(issuedToken)}
              >
                Delete Audrey data
              </button>
            ) : (
              <form className="purge-confirmation" onSubmit={(event) => void requestDataPurge(event)}>
                <label>
                  <span>Type <code>{ACCOUNT_PURGE_CONFIRMATION}</code> to continue</span>
                  <input
                    aria-label="Deletion confirmation"
                    autoComplete="off"
                    spellCheck={false}
                    value={purgeConfirmation}
                    onChange={(event) => setPurgeConfirmation(event.target.value)}
                    disabled={purgeRequesting}
                  />
                </label>
                <div className="token-actions">
                  <button
                    className="danger-button"
                    type="submit"
                    disabled={purgeRequesting || purgeConfirmation !== ACCOUNT_PURGE_CONFIRMATION}
                  >
                    {purgeRequesting ? "Starting deletion…" : "Delete all Audrey data"}
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      setPurgeConfirmOpen(false);
                      setPurgeConfirmation("");
                    }}
                    disabled={purgeRequesting}
                  >
                    Keep my data
                  </button>
                </div>
              </form>
            )}
          </div>
        </section>
          </>
        )}

      </section>
    </div>
  );
}

const ACCOUNT_PURGE_CONFIRMATION = "DELETE ALL MY AUDREY DATA";

function validTokenExpiry(value: string): boolean {
  const days = Number(value);
  return Number.isInteger(days) && days >= 1 && days <= 365;
}

function tokenExpired(record: PersonalTokenRecord): boolean {
  return Date.parse(record.expires_at) <= Date.now();
}

function tokenScopeLabel(scope: PersonalTokenScope): string {
  return scope === "compat:full" ? "OpenAI-compatible API" : "Account read";
}

function formatTokenTime(value: string): string {
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return value;
  return new Intl.DateTimeFormat(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(parsed);
}

function downloadJson(value: unknown, filename: string) {
  const blob = new Blob([`${JSON.stringify(value, null, 2)}\n`], {
    type: "application/json",
  });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.hidden = true;
  document.body.append(anchor);
  anchor.click();
  anchor.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 0);
}

function newPurgeIdempotencyKey(): string {
  if (typeof globalThis.crypto.randomUUID === "function") {
    return `native-ui-${globalThis.crypto.randomUUID()}`;
  }
  const values = new Uint32Array(4);
  globalThis.crypto.getRandomValues(values);
  return `native-ui-${Array.from(values, (value) => value.toString(16)).join("-")}`;
}

function purgeStatusTitle(status: string): string {
  if (status === "completed") return "Deletion complete";
  if (status === "attention_required") return "Deletion needs attention";
  return "Deletion is in progress";
}

function purgeStatusDescription(status: string): string {
  if (status === "completed") {
    return "Audrey removed the requested data. Your sign-in identity remains available for a fresh start.";
  }
  if (status === "attention_required") {
    return "The data remains hidden, but one or more cleanup operations need an administrator repair.";
  }
  return "Audrey has hidden the pre-deletion data and will keep retrying durable cleanup in the background.";
}


function messageOf(reason: unknown, fallback: string): string {
  return reason instanceof Error ? reason.message : fallback;
}
