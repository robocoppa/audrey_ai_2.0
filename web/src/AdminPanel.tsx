import { useEffect, useState } from "react";

import {
  approveAdminUser,
  createPendingAdminUser,
  createAdminRole,
  deleteAdminRole,
  listAdminRoles,
  removeAdminModelPortrait,
  updateAdminModelProfile,
  updateAdminRole,
  uploadAdminModelPortrait,
  deleteAdminUser,
  denyAdminUser,
  listAdminModels,
  listAdminUsers,
  resetAdminModelPolicy,
  setAdminModelOrder,
  updateAdminModel,
  updateAdminUser,
  type AccessGroup,
  type AdminModel,
  type AdminRole,
  type AdminUser,
} from "./api";

type AdminView = "accounts" | "models" | "roles";
type AccountFilter = "all" | "pending" | "active" | "disabled";
type ModelFilter = "all" | "workflow" | "direct";
type ModelStatusFilter = "all" | "enabled" | "disabled";
type AccessRole = string;

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
  const [roles, setRoles] = useState<AdminRole[]>([]);
  const [newRoleId, setNewRoleId] = useState("");
  const [newRoleName, setNewRoleName] = useState("");
  const [newRoleDescription, setNewRoleDescription] = useState("");
  const [editingRoleId, setEditingRoleId] = useState("");
  const [roleName, setRoleName] = useState("");
  const [roleDescription, setRoleDescription] = useState("");
  const [deleteRoleId, setDeleteRoleId] = useState("");
  const [editingModel, setEditingModel] = useState<AdminModel | null>(null);
  const [modelName, setModelName] = useState("");
  const [modelRoles, setModelRoles] = useState<AccessGroup[]>([]);
  const [modelPortrait, setModelPortrait] = useState<File | null>(null);
  const [modelSource, setModelSource] = useState<"ollama" | "configuration">("ollama");
  const [modelWarning, setModelWarning] = useState("");
  const [loading, setLoading] = useState(true);
  const [busyKey, setBusyKey] = useState("");
  const [denyConfirmId, setDenyConfirmId] = useState("");
  const [deleteConfirmId, setDeleteConfirmId] = useState("");
  const [error, setError] = useState("");
  const [view, setView] = useState<AdminView>("accounts");
  const [accountQuery, setAccountQuery] = useState("");
  const [accountFilter, setAccountFilter] = useState<AccountFilter>("all");
  const [newAccountEmail, setNewAccountEmail] = useState("");
  const [newAccountName, setNewAccountName] = useState("");
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
  const modelOrderLocked = Boolean(modelQuery.trim()) || modelStatusFilter !== "all";
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
    Promise.all([listAdminUsers(), listAdminModels(), listAdminRoles()])
      .then(([userResponse, modelResponse, roleResponse]) => {
        if (!active) return;
        setUsers(userResponse.items);
        setModels(modelResponse.items);
        setRoles(roleResponse.items);
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
    if (!users.some((user) => user.deletion_pending)) return;
    const timer = window.setInterval(() => {
      void listAdminUsers().then((response) => setUsers(response.items)).catch(() => {
        // The worker keeps retrying; a later refresh can reconcile the list.
      });
    }, 3000);
    return () => window.clearInterval(timer);
  }, [users]);

  useEffect(() => {
    function closeOnEscape(event: KeyboardEvent) {
      if (event.key === "Escape" && !busy) {
        if (editingModel) setEditingModel(null);
        else onClose();
      }
    }
    window.addEventListener("keydown", closeOnEscape);
    return () => window.removeEventListener("keydown", closeOnEscape);
  }, [busy, editingModel, onClose]);

  function replaceUser(updated: AdminUser) {
    setUsers((current) => current.map((user) =>
      user.id === updated.id ? updated : user,
    ));
    onChanged();
  }

  async function addPendingAccount(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setBusyKey("new-account");
    setError("");
    try {
      const created = await createPendingAdminUser(newAccountEmail.trim(), newAccountName.trim());
      setUsers((current) => [created, ...current]);
      setNewAccountEmail("");
      setNewAccountName("");
      setAccountFilter("pending");
      setAccountQuery("");
      onChanged();
    } catch (reason) {
      setError(messageOf(reason, "The pending account could not be added."));
    } finally {
      setBusyKey("");
    }
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

  async function deleteUser(user: AdminUser) {
    setBusyKey(`user:${user.id}`);
    setError("");
    try {
      await deleteAdminUser(user.id);
      setUsers((current) => current.map((item) =>
        item.id === user.id ? { ...item, status: "disabled", deletion_pending: true } : item,
      ));
      setDeleteConfirmId("");
      onChanged();
    } catch (reason) {
      setError(messageOf(reason, "The account could not be deleted."));
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

  async function addRole(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setBusyKey("new-role");
    setError("");
    try {
      const created = await createAdminRole({
        id: newRoleId.trim().toLowerCase(),
        name: newRoleName.trim(),
        description: newRoleDescription.trim(),
      });
      setRoles((current) => [...current, created]);
      setNewRoleId("");
      setNewRoleName("");
      setNewRoleDescription("");
    } catch (reason) {
      setError(messageOf(reason, "The role could not be created."));
    } finally {
      setBusyKey("");
    }
  }

  async function saveRole(role: AdminRole) {
    setBusyKey(`role:${role.id}`);
    setError("");
    try {
      const updated = await updateAdminRole(role.id, {
        name: roleName.trim(),
        description: roleDescription.trim(),
      });
      setRoles((current) => current.map((item) => item.id === role.id ? updated : item));
      setEditingRoleId("");
    } catch (reason) {
      setError(messageOf(reason, "The role could not be saved."));
    } finally {
      setBusyKey("");
    }
  }

  async function removeRole(role: AdminRole) {
    setBusyKey(`role:${role.id}`);
    setError("");
    try {
      await deleteAdminRole(role.id);
      setRoles((current) => current.filter((item) => item.id !== role.id));
      setDeleteRoleId("");
    } catch (reason) {
      setError(messageOf(reason, "The role could not be deleted."));
    } finally {
      setBusyKey("");
    }
  }

  async function moveModel(model: AdminModel, direction: -1 | 1) {
    const siblings = models.filter((item) => item.kind === model.kind);
    const index = siblings.findIndex((item) => item.id === model.id);
    const target = index + direction;
    if (modelOrderLocked || busy || target < 0 || target >= siblings.length) return;
    const reordered = [...siblings];
    [reordered[index], reordered[target]] = [reordered[target], reordered[index]];
    setBusyKey("model-order");
    setError("");
    try {
      await setAdminModelOrder(model.kind, reordered.map((item) => item.id));
      setModels((current) => {
        const byId = new Map(current.map((item) => [item.id, item]));
        const ordered = reordered.map((item) => byId.get(item.id) ?? item);
        let next = 0;
        return current.map((item) => item.kind === model.kind ? ordered[next++] : item);
      });
      onChanged();
    } catch (reason) {
      setError(messageOf(reason, "The model order could not be saved."));
    } finally {
      setBusyKey("");
    }
  }

  async function changeModelVisibility(model: AdminModel, visibility: "public" | "private") {
    setBusyKey(`model:${model.id}`);
    setError("");
    try {
      const updated = await updateAdminModelProfile(model.id, {
        visibility,
        roles: visibility === "public"
          ? (model.roles.length ? model.roles : ["users"])
          : model.roles,
        display_name: model.label,
      });
      setModels((current) => current.map((item) => item.id === model.id ? updated : item));
      onChanged();
    } catch (reason) {
      setError(messageOf(reason, "The model visibility could not be changed."));
    } finally {
      setBusyKey("");
    }
  }

  function openModelEditor(model: AdminModel) {
    setEditingModel(model);
    setModelName(model.label);
    setModelRoles(model.roles);
    setModelPortrait(null);
  }

  async function saveModelEditor(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!editingModel) return;
    setBusyKey(`model:${editingModel.id}`);
    setError("");
    try {
      let updated = await updateAdminModelProfile(editingModel.id, {
        visibility: editingModel.visibility,
        roles: modelRoles,
        display_name: modelName.trim(),
      });
      if (modelPortrait) {
        updated = await uploadAdminModelPortrait(editingModel.id, modelPortrait);
      }
      setModels((current) => current.map((item) => item.id === updated.id ? updated : item));
      setEditingModel(null);
      setModelPortrait(null);
      onChanged();
    } catch (reason) {
      setError(messageOf(reason, "The model settings could not be saved."));
    } finally {
      setBusyKey("");
    }
  }

  async function removePortrait() {
    if (!editingModel) return;
    setBusyKey(`model:${editingModel.id}`);
    setError("");
    try {
      const updated = await removeAdminModelPortrait(editingModel.id);
      setModels((current) => current.map((item) => item.id === updated.id ? updated : item));
      setEditingModel(updated);
      onChanged();
    } catch (reason) {
      setError(messageOf(reason, "The portrait could not be removed."));
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
              <button
                type="button"
                role="tab"
                id="admin-tab-roles"
                aria-controls="admin-roles-panel"
                aria-selected={view === "roles"}
                onClick={() => setView("roles")}
              >
                Roles <span>{roles.length}</span>
                <em>Manage access groups</em>
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
                <p>Approve sign-ins, assign one clear access role, suspend or permanently delete accounts.</p>
              </div>
              <form className="admin-add-account" onSubmit={(event) => void addPendingAccount(event)}>
                <label>
                  <span>Add a pending account by email</span>
                  <input
                    type="email"
                    required
                    value={newAccountEmail}
                    onChange={(event) => setNewAccountEmail(event.target.value)}
                    placeholder="email@example.com"
                    disabled={busy}
                  />
                </label>
                <label>
                  <span>Name (optional)</span>
                  <input
                    type="text"
                    maxLength={100}
                    value={newAccountName}
                    onChange={(event) => setNewAccountName(event.target.value)}
                    disabled={busy}
                  />
                </label>
                <button type="submit" disabled={busy || !newAccountEmail.trim()}>
                  {busyKey === "new-account" ? "Adding…" : "Add pending"}
                </button>
              </form>
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
                    <article className="admin-record admin-user-record" key={user.id}>
                      <div
                        className="admin-record-heading"
                        title={`${user.auth_provider || "unknown provider"}${user.last_seen_at ? ` · Last seen ${formatTime(user.last_seen_at)}` : ""}`}
                      >
                        <div>
                          <strong>{user.display_name || user.email}</strong>
                          <span>{user.email}</span>
                        </div>
                        <span className={`account-status ${user.status}`}>
                          {user.deletion_pending ? "deleting" : user.status}
                        </span>
                      </div>
                      <p className="admin-record-meta admin-visually-hidden">
                        {user.auth_provider || "unknown provider"}
                        {user.last_seen_at ? ` · Last seen ${formatTime(user.last_seen_at)}` : ""}
                      </p>
                      {user.deletion_pending ? (
                        <p className="admin-deletion-progress" title="This account will disappear when data cleanup completes.">Purging data…</p>
                      ) : user.status === "pending" ? (
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
                              {roles.filter((role) => !role.system).map((role) => (
                                <option key={role.id} value={role.id}>{role.name}</option>
                              ))}
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
                      {!isSelf && !user.deletion_pending ? (
                        <>
                          <button
                            className="admin-delete-trigger danger-button"
                            type="button"
                            aria-label={`Delete account ${user.display_name || user.email}`}
                            title="Delete account"
                            aria-expanded={deleteConfirmId === user.id}
                            onClick={() => setDeleteConfirmId(user.id)}
                            disabled={rowBusy}
                          >
                            <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                              <path d="M4 7h16M9 7V4h6v3m3 0-1 13H7L6 7m4 4v6m4-6v6" />
                            </svg>
                          </button>
                          {deleteConfirmId === user.id ? (
                            <div className="admin-delete-confirmation" role="group" aria-label={`Confirm deletion of ${user.email}`}>
                              <span>Delete this account and all its Audrey data permanently? They can sign in again if their identity provider still allows access.</span>
                              <button className="danger-button" type="button" onClick={() => void deleteUser(user)} disabled={rowBusy}>
                                {rowBusy ? "Deleting…" : "Yes, delete account"}
                              </button>
                              <button type="button" onClick={() => setDeleteConfirmId("")} disabled={rowBusy}>Cancel</button>
                            </div>
                          ) : null}
                        </>
                      ) : null}
                    </article>
                  );
                })}
                {filteredUsers.length === 0 ? (
                  <p className="admin-empty">No accounts match these filters.</p>
                ) : null}
              </div>
            </section>
            ) : null}

            {view === "roles" ? (
            <section
              className="settings-section admin-workspace"
              id="admin-roles-panel"
              role="tabpanel"
              aria-labelledby="admin-tab-roles"
            >
              <div className="settings-section-heading">
                <h3>Roles</h3>
                <p>Create roles for account assignments and Public model access. Built-in roles are protected.</p>
              </div>
              <form className="admin-role-form" onSubmit={(event) => void addRole(event)}>
                <label>
                  <span>Role ID</span>
                  <input
                    value={newRoleId}
                    onChange={(event) => setNewRoleId(event.target.value)}
                    placeholder="researchers"
                    pattern="[a-z][a-z0-9_-]{1,31}"
                    maxLength={32}
                    required
                    disabled={busy}
                  />
                </label>
                <label>
                  <span>Name</span>
                  <input
                    value={newRoleName}
                    onChange={(event) => setNewRoleName(event.target.value)}
                    placeholder="Researchers"
                    maxLength={60}
                    required
                    disabled={busy}
                  />
                </label>
                <label>
                  <span>Description (optional)</span>
                  <input
                    value={newRoleDescription}
                    onChange={(event) => setNewRoleDescription(event.target.value)}
                    maxLength={240}
                    disabled={busy}
                  />
                </label>
                <button type="submit" disabled={busy}>Add role</button>
              </form>
              <div className="admin-record-list">
                {roles.map((role) => (
                  <article className="admin-record admin-role-record" key={role.id}>
                    {editingRoleId === role.id ? (
                      <form className="admin-role-edit" onSubmit={(event) => {
                        event.preventDefault();
                        void saveRole(role);
                      }}>
                        <label>
                          <span>Name</span>
                          <input value={roleName} onChange={(event) => setRoleName(event.target.value)} maxLength={60} required />
                        </label>
                        <label>
                          <span>Description</span>
                          <input value={roleDescription} onChange={(event) => setRoleDescription(event.target.value)} maxLength={240} />
                        </label>
                        <button type="submit" disabled={busy}>Save</button>
                        <button type="button" onClick={() => setEditingRoleId("")} disabled={busy}>Cancel</button>
                      </form>
                    ) : (
                      <>
                        <div className="admin-record-heading">
                          <div>
                            <strong>{role.name}</strong>
                            <span>{role.id} · {role.user_count} users{role.description ? ` · ${role.description}` : ""}</span>
                          </div>
                          <span>{role.system ? "Built-in" : "Custom"}</span>
                        </div>
                        {!role.system ? (
                          <div className="admin-role-actions">
                            <button type="button" disabled={busy} onClick={() => {
                              setEditingRoleId(role.id);
                              setRoleName(role.name);
                              setRoleDescription(role.description);
                            }}>Edit</button>
                            {deleteRoleId === role.id ? (
                              <>
                                <button className="danger-button" type="button" disabled={busy} onClick={() => void removeRole(role)}>
                                  Confirm delete
                                </button>
                                <button type="button" disabled={busy} onClick={() => setDeleteRoleId("")}>Cancel</button>
                              </>
                            ) : (
                              <button className="danger-button" type="button" disabled={busy} onClick={() => setDeleteRoleId(role.id)}>
                                Delete
                              </button>
                            )}
                          </div>
                        ) : null}
                      </>
                    )}
                  </article>
                ))}
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
                <p>Use the arrows to set the order within Audrey workflows or Other models. Clear search and state filters to reorder.</p>
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
                  const siblings = models.filter((item) => item.kind === model.kind);
                  const orderIndex = siblings.findIndex((item) => item.id === model.id);
                  return (
                    <article className="admin-record admin-model-record" key={model.id}>
                      <div className="admin-record-heading" title={model.description}>
                        <div>
                          <strong>{model.label}</strong>
                          <span>{model.kind === "direct" ? model.concrete_model : model.id}</span>
                        </div>
                        <div className="admin-model-state">
                          <span>{model.kind === "direct" ? "Direct" : "Workflow"}</span>
                          {model.policy_overridden ? <em>Customized</em> : <em>Default</em>}
                        </div>
                      </div>
                      <p className="admin-visually-hidden">{model.description}</p>
                      <div className="admin-model-controls">
                        <div className="admin-model-order-controls" aria-label={`Order for ${model.label}`}>
                          <button
                            type="button"
                            aria-label={`Move ${model.label} up`}
                            title="Move up"
                            onClick={() => void moveModel(model, -1)}
                            disabled={busy || modelOrderLocked || orderIndex === 0}
                          >↑</button>
                          <button
                            type="button"
                            aria-label={`Move ${model.label} down`}
                            title="Move down"
                            onClick={() => void moveModel(model, 1)}
                            disabled={busy || modelOrderLocked || orderIndex === siblings.length - 1}
                          >↓</button>
                        </div>
                        <button
                          type="button"
                          aria-pressed={model.enabled}
                          onClick={() => void setModel(model, { enabled: !model.enabled })}
                          disabled={rowBusy}
                        >
                          {rowBusy ? "Saving…" : model.enabled ? "Enabled" : "Disabled"}
                        </button>
                        <label className="admin-model-audience">
                          <span>Visibility</span>
                          <select
                            value={model.visibility}
                            onChange={(event) => void changeModelVisibility(
                              model, event.target.value as "public" | "private",
                            )}
                            disabled={rowBusy}
                          >
                            <option value="public">Public</option>
                            <option value="private">Private</option>
                          </select>
                        </label>
                        <button type="button" onClick={() => openModelEditor(model)} disabled={rowBusy}>
                          Edit…
                        </button>
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

            {editingModel ? (
              <div className="admin-editor-backdrop" role="presentation" onMouseDown={(event) => {
                if (event.currentTarget === event.target && !busy) setEditingModel(null);
              }}>
                <section className="admin-model-editor" role="dialog" aria-modal="true" aria-labelledby="admin-model-editor-title">
                  <div className="admin-editor-heading">
                    <div>
                      <span>Model settings</span>
                      <h3 id="admin-model-editor-title">{editingModel.label}</h3>
                    </div>
                    <button type="button" aria-label="Close model editor" onClick={() => setEditingModel(null)} disabled={busy}>×</button>
                  </div>
                  <form onSubmit={(event) => void saveModelEditor(event)}>
                    <label>
                      <span>Display name</span>
                      <input value={modelName} onChange={(event) => setModelName(event.target.value)} maxLength={80} required />
                    </label>
                    <div className="admin-portrait-editor">
                      {editingModel.portrait_url ? (
                        <img src={editingModel.portrait_url} alt="" />
                      ) : (
                        <span className="admin-portrait-placeholder" aria-hidden="true">✦</span>
                      )}
                      <label>
                        <span>Portrait (PNG, JPEG or WebP, up to 2 MB)</span>
                        <input
                          type="file"
                          accept="image/png,image/jpeg,image/webp"
                          onChange={(event) => setModelPortrait(event.target.files?.[0] ?? null)}
                        />
                      </label>
                      {editingModel.portrait_url ? (
                        <button type="button" onClick={() => void removePortrait()} disabled={busy}>Remove portrait</button>
                      ) : null}
                    </div>
                    <fieldset>
                      <legend>Roles allowed when Public</legend>
                      <p>Private models are always limited to administrators, regardless of these selections.</p>
                      <div className="admin-role-checks">
                        {roles.filter((role) => role.id !== "admins").map((role) => (
                          <label key={role.id}>
                            <input
                              type="checkbox"
                              checked={modelRoles.includes(role.id)}
                              onChange={(event) => setModelRoles((current) =>
                                event.target.checked
                                  ? [...current, role.id]
                                  : current.filter((id) => id !== role.id),
                              )}
                            />
                            {role.name}
                          </label>
                        ))}
                      </div>
                    </fieldset>
                    <div className="admin-editor-actions">
                      <button type="submit" disabled={busy || !modelName.trim() || (editingModel.visibility === "public" && modelRoles.length === 0)}>
                        {busy ? "Saving…" : "Save model"}
                      </button>
                      <button type="button" onClick={() => setEditingModel(null)} disabled={busy}>Cancel</button>
                    </div>
                  </form>
                </section>
              </div>
            ) : null}
          </>
        )}
      </section>
    </div>
  );
}

function roleForUser(user: AdminUser): AccessRole {
  if (user.groups.includes("admins") || user.role === "admin") return "admin";
  const custom = user.groups.find((group) => !["users", "testers", "admins"].includes(group));
  if (custom) return custom;
  if (user.groups.includes("testers")) return "tester";
  return "user";
}

function groupsForRole(role: AccessRole): AccessGroup[] {
  if (role === "admin") return ["admins", "users"];
  if (role === "tester") return ["testers", "users"];
  if (role === "user") return ["users"];
  return ["users", role];
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
