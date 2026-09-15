import { useEffect, useState } from "react";

import {
  approveAdminUser,
  denyAdminUser,
  listAdminModels,
  listAdminUsers,
  updateAdminModel,
  updateAdminUser,
  type AccessGroup,
  type AdminModel,
  type AdminUser,
} from "./api";

export function AdminPanel({
  currentUserId,
  onChanged,
  onClose,
}: {
  currentUserId: string;
  onChanged: () => void;
  onClose: () => void;
}) {
  const [users, setUsers] = useState<AdminUser[]>([]);
  const [models, setModels] = useState<AdminModel[]>([]);
  const [loading, setLoading] = useState(true);
  const [busyKey, setBusyKey] = useState("");
  const [denyConfirmId, setDenyConfirmId] = useState("");
  const [error, setError] = useState("");
  const busy = loading || Boolean(busyKey);

  useEffect(() => {
    let active = true;
    Promise.all([listAdminUsers(), listAdminModels()])
      .then(([userResponse, modelResponse]) => {
        if (!active) return;
        setUsers(userResponse.items);
        setModels(modelResponse.items);
      })
      .catch((reason: unknown) => {
        if (active) setError(messageOf(reason, "Administration data could not be loaded."));
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    function closeOnEscape(event: KeyboardEvent) {
      if (event.key === "Escape" && !busy) onClose();
    }
    window.addEventListener("keydown", closeOnEscape);
    return () => window.removeEventListener("keydown", closeOnEscape);
  }, [busy, onClose]);

  function replaceUser(updated: AdminUser) {
    setUsers((current) => current.map((user) =>
      user.id === updated.id ? updated : user,
    ));
    onChanged();
  }

  async function approve(user: AdminUser) {
    setBusyKey(`user:${user.id}`);
    setError("");
    try {
      replaceUser(await approveAdminUser(user.id, false));
    } catch (reason) {
      setError(messageOf(reason, "The account could not be approved."));
    } finally {
      setBusyKey("");
    }
  }

  async function deny(user: AdminUser) {
    setBusyKey(`user:${user.id}`);
    setError("");
    try {
      replaceUser(await denyAdminUser(user.id));
      setDenyConfirmId("");
    } catch (reason) {
      setError(messageOf(reason, "The account could not be denied."));
    } finally {
      setBusyKey("");
    }
  }

  async function setUserStatus(user: AdminUser, status: "active" | "disabled") {
    setBusyKey(`user:${user.id}`);
    setError("");
    const groups = status === "active"
      ? normalizedGroups(user.groups)
      : user.groups;
    try {
      replaceUser(await updateAdminUser(user.id, { status, groups }));
    } catch (reason) {
      setError(messageOf(reason, "The account status could not be changed."));
    } finally {
      setBusyKey("");
    }
  }

  async function setGroup(user: AdminUser, group: "testers" | "admins", checked: boolean) {
    setBusyKey(`user:${user.id}`);
    setError("");
    const groups = new Set(normalizedGroups(user.groups));
    if (checked) groups.add(group);
    else groups.delete(group);
    try {
      replaceUser(await updateAdminUser(user.id, {
        groups: [...groups].sort() as AccessGroup[],
      }));
    } catch (reason) {
      setError(messageOf(reason, "The access groups could not be changed."));
    } finally {
      setBusyKey("");
    }
  }

  async function setModel(
    model: AdminModel,
    patch: { enabled?: boolean; audience?: AccessGroup },
  ) {
    setBusyKey(`model:${model.id}`);
    setError("");
    try {
      const updated = await updateAdminModel(model.id, {
        enabled: patch.enabled ?? model.enabled,
        audience: patch.audience ?? model.audience,
      });
      setModels((current) => current.map((item) =>
        item.id === updated.id ? updated : item,
      ));
      onChanged();
    } catch (reason) {
      setError(messageOf(reason, "The model policy could not be changed."));
    } finally {
      setBusyKey("");
    }
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
        className="account-settings admin-panel"
        role="dialog"
        aria-modal="true"
        aria-labelledby="admin-panel-title"
      >
        <div className="account-settings-header">
          <div>
            <span>Audrey administration</span>
            <h2 id="admin-panel-title">Access control</h2>
          </div>
          <button type="button" onClick={onClose} disabled={busy} aria-label="Close administration">×</button>
        </div>

        {error ? <p className="settings-error admin-error" role="alert">{error}</p> : null}
        {loading ? <p className="admin-loading" role="status">Loading access controls…</p> : (
          <>
            <section className="settings-section" aria-labelledby="admin-users-title">
              <div className="settings-section-heading">
                <h3 id="admin-users-title">Accounts</h3>
                <p>Approve new sign-ins and assign Audrey-owned access groups.</p>
              </div>
              <div className="admin-record-list">
                {users.map((user) => {
                  const rowBusy = busyKey === `user:${user.id}`;
                  const isSelf = user.id === currentUserId;
                  return (
                    <article className="admin-record" key={user.id}>
                      <div className="admin-record-heading">
                        <div>
                          <strong>{user.display_name || user.email}</strong>
                          <span>{user.email}</span>
                        </div>
                        <span className={`account-status ${user.status}`}>{user.status}</span>
                      </div>
                      <p className="admin-record-meta">
                        {user.auth_provider || "unknown provider"}
                        {user.last_seen_at ? ` · Last seen ${formatTime(user.last_seen_at)}` : ""}
                      </p>
                      {user.status === "pending" ? (
                        <div className="admin-record-actions">
                          <button type="button" onClick={() => void approve(user)} disabled={rowBusy}>
                            {rowBusy ? "Updating…" : "Approve"}
                          </button>
                          {denyConfirmId === user.id ? (
                            <>
                              <button className="danger-button" type="button" onClick={() => void deny(user)} disabled={rowBusy}>
                                Confirm deny
                              </button>
                              <button type="button" onClick={() => setDenyConfirmId("")} disabled={rowBusy}>Keep pending</button>
                            </>
                          ) : (
                            <button className="danger-button" type="button" onClick={() => setDenyConfirmId(user.id)} disabled={rowBusy}>
                              Deny
                            </button>
                          )}
                        </div>
                      ) : (
                        <div className="admin-account-controls">
                          <label className="settings-checkbox">
                            <input
                              type="checkbox"
                              checked={user.groups.includes("testers")}
                              onChange={(event) => void setGroup(user, "testers", event.target.checked)}
                              disabled={rowBusy || user.status !== "active"}
                            />
                            <span>Tester models</span>
                          </label>
                          <label className="settings-checkbox">
                            <input
                              type="checkbox"
                              checked={user.groups.includes("admins")}
                              onChange={(event) => void setGroup(user, "admins", event.target.checked)}
                              disabled={rowBusy || user.status !== "active" || isSelf}
                            />
                            <span>Administrator</span>
                          </label>
                          <button
                            className={user.status === "active" ? "danger-button" : ""}
                            type="button"
                            onClick={() => void setUserStatus(
                              user,
                              user.status === "active" ? "disabled" : "active",
                            )}
                            disabled={rowBusy || isSelf}
                          >
                            {rowBusy
                              ? "Updating…"
                              : user.status === "active" ? "Disable" : "Reactivate"}
                          </button>
                        </div>
                      )}
                    </article>
                  );
                })}
              </div>
            </section>

            <section className="settings-section" aria-labelledby="admin-models-title">
              <div className="settings-section-heading">
                <h3 id="admin-models-title">Models</h3>
                <p>Control which access group can see and invoke each server-defined model.</p>
              </div>
              <div className="admin-record-list">
                {models.map((model) => {
                  const rowBusy = busyKey === `model:${model.id}`;
                  return (
                    <article className="admin-record admin-model-record" key={model.id}>
                      <div className="admin-record-heading">
                        <div>
                          <strong>{model.label}</strong>
                          <span>{model.kind === "direct" ? model.concrete_model : model.id}</span>
                        </div>
                        <button
                          type="button"
                          aria-pressed={model.enabled}
                          onClick={() => void setModel(model, { enabled: !model.enabled })}
                          disabled={rowBusy}
                        >
                          {rowBusy ? "Saving…" : model.enabled ? "Enabled" : "Disabled"}
                        </button>
                      </div>
                      <p>{model.description}</p>
                      <label className="admin-model-audience">
                        <span>Available to</span>
                        <select
                          value={model.audience}
                          onChange={(event) => void setModel(model, {
                            audience: event.target.value as AccessGroup,
                          })}
                          disabled={rowBusy}
                        >
                          <option value="users">Users</option>
                          <option value="testers">Testers</option>
                          <option value="admins">Administrators</option>
                        </select>
                      </label>
                    </article>
                  );
                })}
              </div>
            </section>
          </>
        )}
      </section>
    </div>
  );
}

function normalizedGroups(groups: AccessGroup[]): AccessGroup[] {
  return [...new Set<AccessGroup>(["users", ...groups])].sort();
}

function formatTime(value: string): string {
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return value;
  return new Intl.DateTimeFormat(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(parsed);
}

function messageOf(reason: unknown, fallback: string): string {
  return reason instanceof Error ? reason.message : fallback;
}
