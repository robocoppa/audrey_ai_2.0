import { useEffect, useState } from "react";

import {
  approveAdminUser,
  denyAdminUser,
  listAdminModels,
  listAdminUsers,
  resetAdminModelPolicy,
  updateAdminModel,
  updateAdminUser,
  type AccessGroup,
  type AdminModel,
  type AdminUser,
} from "./api";

type AdminView = "accounts" | "models";
type AccountFilter = "all" | "pending" | "active" | "disabled";
type ModelFilter = "all" | "workflow" | "direct";
type ModelStatusFilter = "all" | "enabled" | "disabled";
type AccessRole = "user" | "tester" | "admin";

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
  const [modelSource, setModelSource] = useState<"ollama" | "configuration">("ollama");
  const [modelWarning, setModelWarning] = useState("");
  const [loading, setLoading] = useState(true);
  const [busyKey, setBusyKey] = useState("");
  const [denyConfirmId, setDenyConfirmId] = useState("");
  const [error, setError] = useState("");
  const [view, setView] = useState<AdminView>("accounts");
  const [accountQuery, setAccountQuery] = useState("");
  const [accountFilter, setAccountFilter] = useState<AccountFilter>("all");
  const [modelQuery, setModelQuery] = useState("");
  const [modelFilter, setModelFilter] = useState<ModelFilter>("all");
  const [modelStatusFilter, setModelStatusFilter] = useState<ModelStatusFilter>("all");
  const busy = loading || Boolean(busyKey);
  const pendingCount = users.filter(({ status }) => status === "pending").length;
  const enabledDirectCount = models.filter((model) =>
    model.kind === "direct" && model.enabled
  ).length;
  const filteredUsers = users.filter((user) => {
    const query = accountQuery.trim().toLocaleLowerCase();
    const matchesQuery = !query
      || user.display_name.toLocaleLowerCase().includes(query)
      || user.email.toLocaleLowerCase().includes(query);
    return matchesQuery && (accountFilter === "all" || user.status === accountFilter);
  });
  const filteredModels = models.filter((model) => {
    const query = modelQuery.trim().toLocaleLowerCase();
    const matchesQuery = !query
      || model.label.toLocaleLowerCase().includes(query)
      || model.concrete_model.toLocaleLowerCase().includes(query);
    const matchesType = modelFilter === "all" || model.kind === modelFilter;
    const matchesStatus = modelStatusFilter === "all"
      || model.enabled === (modelStatusFilter === "enabled");
    return matchesQuery && matchesType && matchesStatus;
  });

  useEffect(() => {
    let active = true;
    Promise.all([listAdminUsers(), listAdminModels()])
      .then(([userResponse, modelResponse]) => {
        if (!active) return;
        setUsers(userResponse.items);
        setModels(modelResponse.items);
        setModelSource(modelResponse.source);
        setModelWarning(modelResponse.warning);
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

  async function approve(user: AdminUser, tester: boolean) {
    setBusyKey(`user:${user.id}`);
    setError("");
    try {
      replaceUser(await approveAdminUser(user.id, tester));
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

  async function setRole(user: AdminUser, role: AccessRole) {
    setBusyKey(`user:${user.id}`);
    setError("");
    try {
      replaceUser(await updateAdminUser(user.id, {
        groups: groupsForRole(role),
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

  async function resetModel(model: AdminModel) {
    setBusyKey(`model:${model.id}`);
    setError("");
    try {
      const updated = await resetAdminModelPolicy(model.id);
      setModels((current) => current.map((item) =>
        item.id === updated.id ? updated : item,
      ));
      onChanged();
    } catch (reason) {
      setError(messageOf(reason, "The model policy could not be reset."));
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
            <div className="admin-tabs" role="tablist" aria-label="Administration sections">
              <button
                type="button"
                role="tab"
                id="admin-tab-accounts"
                aria-controls="admin-accounts-panel"
                aria-selected={view === "accounts"}
                onClick={() => setView("accounts")}
              >
                Accounts <span>{users.length}</span>
                {pendingCount > 0 ? <em>{pendingCount} pending</em> : null}
              </button>
              <button
                type="button"
                role="tab"
                id="admin-tab-models"
                aria-controls="admin-models-panel"
                aria-selected={view === "models"}
                onClick={() => setView("models")}
              >
                Models <span>{models.length}</span>
                <em>{enabledDirectCount} direct enabled</em>
              </button>
            </div>

            {view === "accounts" ? (
            <section
              className="settings-section admin-workspace"
              id="admin-accounts-panel"
              role="tabpanel"
              aria-labelledby="admin-tab-accounts"
            >
              <div className="settings-section-heading">
                <h3 id="admin-users-title">Accounts</h3>
                <p>Approve sign-ins, assign one clear access role, and suspend accounts.</p>
              </div>
              <div className="admin-toolbar admin-account-toolbar">
                <label>
                  <span>Find an account</span>
                  <input
                    type="search"
                    value={accountQuery}
                    onChange={(event) => setAccountQuery(event.target.value)}
                    placeholder="Name or email"
                  />
                </label>
                <label>
                  <span>Status</span>
                  <select
                    value={accountFilter}
                    onChange={(event) => setAccountFilter(event.target.value as AccountFilter)}
                  >
                    <option value="all">All accounts</option>
                    <option value="pending">Pending</option>
                    <option value="active">Active</option>
                    <option value="disabled">Disabled</option>
                  </select>
                </label>
              </div>
              <p className="admin-role-guide">
                <strong>User</strong> gets Audrey workflows. <strong>Tester</strong> can receive
                preview models. <strong>Administrator</strong> manages accounts and model access.
              </p>
              <div className="admin-record-list">
                {filteredUsers.map((user) => {
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
                          <button type="button" onClick={() => void approve(user, false)} disabled={rowBusy}>
                            {rowBusy ? "Updating…" : "Approve as user"}
                          </button>
                          <button type="button" onClick={() => void approve(user, true)} disabled={rowBusy}>
                            Approve as tester
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
                          <label className="admin-role-control">
                            <span>Role</span>
                            <select
                              aria-label={`Role for ${user.display_name || user.email}`}
                              value={roleForUser(user)}
                              onChange={(event) => void setRole(
                                user,
                                event.target.value as AccessRole,
                              )}
                              disabled={rowBusy || user.status !== "active" || isSelf}
                            >
                              <option value="user">User</option>
                              <option value="tester">Tester</option>
                              <option value="admin">Administrator</option>
                            </select>
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
                {filteredUsers.length === 0 ? (
                  <p className="admin-empty">No accounts match these filters.</p>
                ) : null}
              </div>
            </section>
            ) : null}

            {view === "models" ? (
            <section
              className="settings-section admin-workspace"
              id="admin-models-panel"
              role="tabpanel"
              aria-labelledby="admin-tab-models"
            >
              <div className="settings-section-heading">
                <h3 id="admin-models-title">Models</h3>
                <p>Publish Audrey workflows and Ollama models to the appropriate role.</p>
              </div>
              <div className="admin-inventory-status">
                <span className={`admin-source ${modelSource}`}>{modelSource === "ollama" ? "Live Ollama inventory" : "Configured fallback"}</span>
                <p>
                  Ollama models start enabled for administrators only. Changes below are durable
                  Audrey policy overrides and do not alter Ollama itself.
                </p>
                {modelWarning ? <p className="admin-inventory-warning" role="status">{modelWarning}</p> : null}
              </div>
              <div className="admin-toolbar">
                <label>
                  <span>Find a model</span>
                  <input
                    type="search"
                    value={modelQuery}
                    onChange={(event) => setModelQuery(event.target.value)}
                    placeholder="Name or Ollama tag"
                  />
                </label>
                <label>
                  <span>Type</span>
                  <select
                    value={modelFilter}
                    onChange={(event) => setModelFilter(event.target.value as ModelFilter)}
                  >
                    <option value="all">All models</option>
                    <option value="workflow">Audrey workflows</option>
                    <option value="direct">Direct Ollama</option>
                  </select>
                </label>
                <label>
                  <span>State</span>
                  <select
                    value={modelStatusFilter}
                    onChange={(event) => setModelStatusFilter(
                      event.target.value as ModelStatusFilter,
                    )}
                  >
                    <option value="all">Any state</option>
                    <option value="enabled">Enabled</option>
                    <option value="disabled">Disabled</option>
                  </select>
                </label>
              </div>
              <div className="admin-record-list">
                {filteredModels.map((model) => {
                  const rowBusy = busyKey === `model:${model.id}`;
                  return (
                    <article className="admin-record admin-model-record" key={model.id}>
                      <div className="admin-record-heading">
                        <div>
                          <strong>{model.label}</strong>
                          <span>{model.kind === "direct" ? model.concrete_model : model.id}</span>
                        </div>
                        <div className="admin-model-state">
                          <span>{model.kind === "direct" ? "Direct" : "Workflow"}</span>
                          {model.policy_overridden ? <em>Customized</em> : <em>Default</em>}
                        </div>
                      </div>
                      <p>{model.description}</p>
                      <div className="admin-model-controls">
                        <button
                          type="button"
                          aria-pressed={model.enabled}
                          onClick={() => void setModel(model, { enabled: !model.enabled })}
                          disabled={rowBusy}
                        >
                          {rowBusy ? "Saving…" : model.enabled ? "Enabled" : "Disabled"}
                        </button>
                        <label className="admin-model-audience">
                          <span>Minimum role</span>
                          <select
                            value={model.audience}
                            onChange={(event) => void setModel(model, {
                              audience: event.target.value as AccessGroup,
                            })}
                            disabled={rowBusy}
                          >
                            <option value="users">All users</option>
                            <option value="testers">Testers + admins</option>
                            <option value="admins">Admins only</option>
                          </select>
                        </label>
                        {model.policy_overridden ? (
                          <button type="button" onClick={() => void resetModel(model)} disabled={rowBusy}>
                            Reset default
                          </button>
                        ) : null}
                      </div>
                    </article>
                  );
                })}
                {filteredModels.length === 0 ? (
                  <p className="admin-empty">No models match these filters.</p>
                ) : null}
              </div>
            </section>
            ) : null}
          </>
        )}
      </section>
    </div>
  );
}

function roleForUser(user: AdminUser): AccessRole {
  if (user.groups.includes("admins") || user.role === "admin") return "admin";
  if (user.groups.includes("testers")) return "tester";
  return "user";
}

function groupsForRole(role: AccessRole): AccessGroup[] {
  if (role === "admin") return ["admins", "users"];
  if (role === "tester") return ["testers", "users"];
  return ["users"];
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
