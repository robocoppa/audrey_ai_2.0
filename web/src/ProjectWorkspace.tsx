import { useEffect, useMemo, useState, type FormEvent } from "react";

import {
  addProjectFile,
  createProject,
  createProjectConversation,
  deleteProject,
  listFiles,
  removeProjectFile,
  updateProject,
  type AudreyFile,
  type AudreyProject,
  type AudreyProjectFile,
  type Conversation,
  type ProjectLimits,
} from "./api";

type ProjectFileKind = "all" | AudreyFile["kind"];

const PROJECT_FILE_FOLDERS: ReadonlyArray<{
  kind: ProjectFileKind;
  label: string;
  symbol: string;
}> = [
  { kind: "all", label: "All ready files", symbol: "▦" },
  { kind: "text", label: "Documents", symbol: "≡" },
  { kind: "image", label: "Images", symbol: "◫" },
  { kind: "audio", label: "Audio", symbol: "♪" },
  { kind: "video", label: "Videos", symbol: "▶" },
];

export function NewProjectDialog({
  limits,
  onClose,
  onCreated,
}: {
  limits: ProjectLimits;
  onClose: () => void;
  onCreated: (project: AudreyProject) => void;
}) {
  const [name, setName] = useState("");
  const [instructions, setInstructions] = useState("");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    const returnTo = document.activeElement;
    return () => {
      if (returnTo instanceof HTMLElement) returnTo.focus();
    };
  }, []);

  useEffect(() => {
    function closeOnEscape(event: KeyboardEvent) {
      if (event.key === "Escape" && !saving) onClose();
    }
    window.addEventListener("keydown", closeOnEscape);
    return () => window.removeEventListener("keydown", closeOnEscape);
  }, [onClose, saving]);

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const projectName = name.trim();
    if (!projectName || saving) return;
    setSaving(true);
    setError("");
    try {
      onCreated(await createProject(projectName, instructions.trim()));
    } catch (reason) {
      setError(messageOf(reason));
    } finally {
      setSaving(false);
    }
  }

  return (
    <div
      className="project-dialog-backdrop"
      role="presentation"
      onMouseDown={(event) => {
        if (event.currentTarget === event.target && !saving) onClose();
      }}
    >
      <section
        className="project-editor-dialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby="new-project-title"
      >
        <header>
          <div>
            <span>Projects</span>
            <h2 id="new-project-title">New project</h2>
          </div>
          <button type="button" onClick={onClose} disabled={saving} aria-label="Close new project">×</button>
        </header>
        <form onSubmit={(event) => void submit(event)}>
          <label>
            <span>Project name</span>
            <input
              value={name}
              onChange={(event) => setName(event.target.value)}
              maxLength={limits.max_name_chars}
              autoFocus
              required
            />
          </label>
          <label>
            <span>Project instructions <small>Optional</small></span>
            <textarea
              value={instructions}
              onChange={(event) => setInstructions(event.target.value)}
              maxLength={limits.max_instructions_chars}
              rows={5}
              placeholder="Tell Audrey how to work inside this project."
            />
          </label>
          {error ? <p className="project-error" role="alert">{error}</p> : null}
          <div className="project-dialog-actions">
            <button type="button" onClick={onClose} disabled={saving}>Cancel</button>
            <button className="project-primary-button" type="submit" disabled={!name.trim() || saving}>
              {saving ? "Creating…" : "Create project"}
            </button>
          </div>
        </form>
      </section>
    </div>
  );
}

export function ProjectHome({
  project,
  conversations,
  files,
  filesLoading,
  limits,
  defaultModelId,
  onProjectChange,
  onProjectDeleted,
  onConversationCreated,
  onConversationSelected,
  onFilesChange,
}: {
  project: AudreyProject;
  conversations: Conversation[];
  files: AudreyProjectFile[];
  filesLoading: boolean;
  limits: ProjectLimits;
  defaultModelId: string;
  onProjectChange: (project: AudreyProject) => void;
  onProjectDeleted: (projectId: string) => void;
  onConversationCreated: (conversation: Conversation) => void;
  onConversationSelected: (conversation: Conversation) => void;
  onFilesChange: (files: AudreyProjectFile[]) => void;
}) {
  const [editing, setEditing] = useState(false);
  const [name, setName] = useState(project.name);
  const [instructions, setInstructions] = useState(project.instructions);
  const [saving, setSaving] = useState(false);
  const [creatingConversation, setCreatingConversation] = useState(false);
  const [removingFileId, setRemovingFileId] = useState<string | null>(null);
  const [filesOpen, setFilesOpen] = useState(false);
  const [confirmingDelete, setConfirmingDelete] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    setName(project.name);
    setInstructions(project.instructions);
    setEditing(false);
    setConfirmingDelete(false);
    setError("");
  }, [project.id, project.instructions, project.name]);

  async function saveProject(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const nextName = name.trim();
    if (!nextName || saving) return;
    setSaving(true);
    setError("");
    try {
      const updated = await updateProject(project.id, {
        name: nextName,
        instructions: instructions.trim(),
      });
      onProjectChange(updated);
      setEditing(false);
    } catch (reason) {
      setError(messageOf(reason));
    } finally {
      setSaving(false);
    }
  }

  async function startConversation() {
    if (creatingConversation) return;
    setCreatingConversation(true);
    setError("");
    try {
      onConversationCreated(await createProjectConversation(project.id, defaultModelId));
    } catch (reason) {
      setError(messageOf(reason));
    } finally {
      setCreatingConversation(false);
    }
  }

  async function removeFile(file: AudreyProjectFile) {
    if (removingFileId) return;
    setRemovingFileId(file.id);
    setError("");
    try {
      await removeProjectFile(project.id, file.id);
      onFilesChange(files.filter(({ id }) => id !== file.id));
    } catch (reason) {
      setError(messageOf(reason));
    } finally {
      setRemovingFileId(null);
    }
  }

  async function removeProject() {
    if (deleting) return;
    setDeleting(true);
    setError("");
    try {
      await deleteProject(project.id);
      onProjectDeleted(project.id);
    } catch (reason) {
      setError(messageOf(reason));
      setConfirmingDelete(false);
    } finally {
      setDeleting(false);
    }
  }

  return (
    <div className="project-home">
      <header className="project-home-header">
        <div>
          <span>Project</span>
          <h1>{project.name}</h1>
          <p>{project.instructions || "No project instructions yet."}</p>
        </div>
        <div className="project-home-actions">
          <button
            className="project-primary-button"
            type="button"
            onClick={() => void startConversation()}
            disabled={creatingConversation}
          >
            {creatingConversation ? "Creating…" : "New conversation in project"}
          </button>
          <button type="button" onClick={() => setEditing((current) => !current)}>
            {editing ? "Close edit" : "Edit project"}
          </button>
        </div>
      </header>

      {editing ? (
        <form className="project-edit-form" onSubmit={(event) => void saveProject(event)}>
          <label>
            <span>Project name</span>
            <input
              value={name}
              onChange={(event) => setName(event.target.value)}
              maxLength={limits.max_name_chars}
              required
            />
          </label>
          <label>
            <span>Project instructions</span>
            <textarea
              value={instructions}
              onChange={(event) => setInstructions(event.target.value)}
              maxLength={limits.max_instructions_chars}
              rows={5}
            />
          </label>
          <div className="project-dialog-actions">
            <button type="button" onClick={() => {
              setName(project.name);
              setInstructions(project.instructions);
              setEditing(false);
            }}>Cancel</button>
            <button className="project-primary-button" type="submit" disabled={!name.trim() || saving}>
              {saving ? "Saving…" : "Save project"}
            </button>
          </div>
        </form>
      ) : null}

      {error ? <p className="project-error" role="alert">{error}</p> : null}

      <div className="project-home-grid">
        <section className="project-panel" aria-labelledby="project-conversations-title">
          <header>
            <div>
              <span>Conversations</span>
              <h2 id="project-conversations-title">Project conversations</h2>
            </div>
            <small>{conversations.length}</small>
          </header>
          {conversations.length ? (
            <ul className="project-home-conversations">
              {conversations.map((conversation) => (
                <li key={conversation.id}>
                  <button type="button" onClick={() => onConversationSelected(conversation)}>
                    <strong>{conversation.title || "New conversation"}</strong>
                    <small>{conversation.archived_at ? "Archived" : formatActivity(conversation)}</small>
                  </button>
                </li>
              ))}
            </ul>
          ) : (
            <p className="project-empty">No conversations in this view.</p>
          )}
        </section>

        <section className="project-panel" aria-labelledby="project-files-title">
          <header>
            <div>
              <span>Shared context</span>
              <h2 id="project-files-title">Project files</h2>
            </div>
            <small>{files.length}/{limits.max_files}</small>
          </header>
          <p className="project-panel-copy">
            Select Ready files from My Files for this project.
          </p>
          <button className="project-manage-files" type="button" onClick={() => setFilesOpen(true)}>
            Manage project files
          </button>
          {filesLoading ? <p className="project-empty" role="status">Loading project files…</p> : null}
          {!filesLoading && files.length === 0 ? (
            <p className="project-empty">No files selected.</p>
          ) : null}
          {files.length ? (
            <ul className="project-file-summary">
              {files.map((file) => (
                <li key={file.id}>
                  <span className="project-file-symbol" aria-hidden="true">{kindSymbol(file.kind)}</span>
                  <span title={file.filename}>{file.filename}</span>
                  <button
                    type="button"
                    onClick={() => void removeFile(file)}
                    disabled={removingFileId !== null}
                    aria-label={`Remove ${file.filename} from project`}
                  >
                    {removingFileId === file.id ? "…" : "Remove"}
                  </button>
                </li>
              ))}
            </ul>
          ) : null}
        </section>
      </div>

      <section className="project-danger-zone" aria-labelledby="delete-project-title">
        <div>
          <h2 id="delete-project-title">Delete project</h2>
          <p>Conversations and files will be kept. Conversations return to ordinary history.</p>
        </div>
        {confirmingDelete ? (
          <div className="project-delete-confirmation" role="alert">
            <span>Delete <strong>{project.name}</strong>?</span>
            <button type="button" onClick={() => setConfirmingDelete(false)} disabled={deleting}>Cancel</button>
            <button className="danger-button confirming-delete" type="button" onClick={() => void removeProject()} disabled={deleting}>
              {deleting ? "Deleting…" : "Confirm delete project"}
            </button>
          </div>
        ) : (
          <button className="danger-button" type="button" onClick={() => setConfirmingDelete(true)}>
            Delete project
          </button>
        )}
      </section>

      {filesOpen ? (
        <ProjectFilePicker
          project={project}
          selected={files}
          limits={limits}
          onChange={onFilesChange}
          onClose={() => setFilesOpen(false)}
        />
      ) : null}
    </div>
  );
}

function ProjectFilePicker({
  project,
  selected,
  limits,
  onChange,
  onClose,
}: {
  project: AudreyProject;
  selected: AudreyProjectFile[];
  limits: ProjectLimits;
  onChange: (files: AudreyProjectFile[]) => void;
  onClose: () => void;
}) {
  const [files, setFiles] = useState<AudreyFile[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [search, setSearch] = useState("");
  const [kind, setKind] = useState<ProjectFileKind>("all");
  const [mutatingId, setMutatingId] = useState<string | null>(null);

  useEffect(() => {
    const returnTo = document.activeElement;
    return () => {
      if (returnTo instanceof HTMLElement) returnTo.focus();
    };
  }, []);

  useEffect(() => {
    let active = true;
    listFiles()
      .then((listing) => {
        if (active) setFiles(listing.items.filter(({ status }) => status === "ready"));
      })
      .catch((reason: unknown) => {
        if (active) setError(messageOf(reason));
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => { active = false; };
  }, []);

  useEffect(() => {
    function closeOnEscape(event: KeyboardEvent) {
      if (event.key === "Escape" && mutatingId === null) onClose();
    }
    window.addEventListener("keydown", closeOnEscape);
    return () => window.removeEventListener("keydown", closeOnEscape);
  }, [mutatingId, onClose]);

  const selectedIds = useMemo(() => new Set(selected.map(({ id }) => id)), [selected]);
  const visible = useMemo(() => {
    const query = search.trim().toLocaleLowerCase();
    return files.filter((file) => {
      if (kind !== "all" && file.kind !== kind) return false;
      return !query || file.filename.toLocaleLowerCase().includes(query);
    });
  }, [files, kind, search]);

  async function toggle(file: AudreyFile) {
    if (mutatingId) return;
    setMutatingId(file.id);
    setError("");
    try {
      if (selectedIds.has(file.id)) {
        await removeProjectFile(project.id, file.id);
        onChange(selected.filter(({ id }) => id !== file.id));
      } else {
        onChange([...selected, await addProjectFile(project.id, file.id)]);
      }
    } catch (reason) {
      setError(messageOf(reason));
    } finally {
      setMutatingId(null);
    }
  }

  return (
    <div
      className="file-manager-backdrop project-file-backdrop"
      role="presentation"
      onMouseDown={(event) => {
        if (event.currentTarget === event.target && mutatingId === null) onClose();
      }}
    >
      <section className="file-manager project-file-picker" role="dialog" aria-modal="true" aria-labelledby="project-file-picker-title">
        <header className="file-manager-header">
          <div>
            <span>{project.name}</span>
            <h2 id="project-file-picker-title">Choose project files</h2>
          </div>
          <button className="file-manager-close" type="button" onClick={onClose} disabled={mutatingId !== null} aria-label="Close project files">×</button>
        </header>
        <p className="project-file-picker-copy">
          Select Ready files from My Files. {selected.length} of {limits.max_files} selected.
        </p>
        {error ? <p className="file-manager-error" role="alert">{error}</p> : null}
        {loading ? <p className="file-manager-status" role="status">Loading ready files…</p> : null}
        {!loading ? (
          <div className="file-explorer">
            <aside className="file-explorer-sidebar" aria-label="Project file folders">
              <span>My Files</span>
              {PROJECT_FILE_FOLDERS.map((folder) => {
                const count = folder.kind === "all"
                  ? files.length
                  : files.filter(({ kind: fileKind }) => fileKind === folder.kind).length;
                return (
                  <button
                    type="button"
                    key={folder.kind}
                    aria-label={`${folder.label} (${count})`}
                    aria-pressed={kind === folder.kind}
                    onClick={() => setKind(folder.kind)}
                  >
                    <span className="file-folder-symbol" aria-hidden="true">{folder.symbol}</span>
                    <span>{folder.label}</span>
                    <small>{count}</small>
                  </button>
                );
              })}
            </aside>
            <div className="file-explorer-content">
              <label className="project-file-search">
                <span>Search ready files</span>
                <input
                  type="search"
                  value={search}
                  onChange={(event) => setSearch(event.target.value)}
                  placeholder="Search filenames"
                  autoFocus
                />
              </label>
              {visible.length ? (
                <ul className="project-picker-files">
                  {visible.map((file) => {
                    const isSelected = selectedIds.has(file.id);
                    const atLimit = !isSelected && selected.length >= limits.max_files;
                    return (
                      <li key={file.id}>
                        <button
                          type="button"
                          aria-pressed={isSelected}
                          onClick={() => void toggle(file)}
                          disabled={mutatingId !== null || atLimit}
                        >
                          <span className="project-file-symbol" aria-hidden="true">{kindSymbol(file.kind)}</span>
                          <span>
                            <strong>{file.filename}</strong>
                            <small>{kindLabel(file.kind)} · {formatBytes(file.bytes)}</small>
                          </span>
                          <span className="project-file-check" aria-hidden="true">
                            {mutatingId === file.id ? "…" : isSelected ? "✓" : "+"}
                          </span>
                        </button>
                      </li>
                    );
                  })}
                </ul>
              ) : (
                <p className="project-empty">No Ready files match this view.</p>
              )}
            </div>
          </div>
        ) : null}
        <footer className="project-picker-footer">
          <span>{selected.length} selected</span>
          <button className="project-primary-button" type="button" onClick={onClose} disabled={mutatingId !== null}>Done</button>
        </footer>
      </section>
    </div>
  );
}

function formatActivity(conversation: Conversation) {
  const stamp = conversation.last_message_at || conversation.created_at;
  const parsed = new Date(stamp);
  return Number.isNaN(parsed.valueOf())
    ? "No messages yet"
    : new Intl.DateTimeFormat(undefined, { month: "short", day: "numeric" }).format(parsed);
}

function kindSymbol(kind: AudreyFile["kind"]) {
  if (kind === "image") return "◫";
  if (kind === "audio") return "♪";
  if (kind === "video") return "▶";
  return "≡";
}

function kindLabel(kind: AudreyFile["kind"]) {
  if (kind === "image") return "Image";
  if (kind === "audio") return "Audio";
  if (kind === "video") return "Video";
  return "Document";
}

function formatBytes(bytes: number) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function messageOf(reason: unknown) {
  return reason instanceof Error ? reason.message : "The request could not be completed.";
}
