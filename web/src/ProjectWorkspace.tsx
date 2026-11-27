import { FileDownloadLink } from "./FileDownloadLink";
import { FilePreviewButton } from "./FilePreview";
import { useEffect, useId, useMemo, useRef, useState, type FormEvent, type ReactNode } from "react";

import {
  addProjectFile,
  createProject,
  createProjectConversation,
  deleteProject,
  listFiles,
  listProjectFiles,
  removeProjectFile,
  updateProject,
  uploadFile,
  uploadPrivateProjectFile,
  PRIVATE_DOCUMENT_EXTENSIONS,
  uploadPrecheck,
  type AudreyFile,
  type AgentTask,
  type PublicWatch,
  type AudreyProject,
  type AudreyProjectFile,
  type Conversation,
  type ProjectLimits,
} from "./api";

import { UploadProgress } from "./UploadProgress";
import { TaxProjectSection } from "./TaxProjectSection";
import { ProjectKnowledgeSection } from "./ProjectKnowledgeSection";
import { type KnowledgeEntry } from "./api";
import { ProjectWatches } from "./ProjectWatches";
import { ProjectAgents } from "./ProjectAgents";
import { SidebarItemButton } from "./SidebarItemButton";
import "./ProjectWorkspace.css";

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
  privateAvailable,
  onClose,
  onCreated,
}: {
  limits: ProjectLimits;
  privateAvailable: boolean;
  onClose: () => void;
  onCreated: (project: AudreyProject) => void;
}) {
  const [name, setName] = useState("");
  const [modelId, setModelId] = useState<"auto" | "private">("auto");
  const [instructions, setInstructions] = useState("");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");
  const dialog = useRef<HTMLElement | null>(null);
  const returnTo = useRef(document.activeElement instanceof HTMLElement ? document.activeElement : null);
  const lifetime = useRef(0);
  const pending = useRef(false);

  useEffect(() => {
    const generation = ++lifetime.current;
    const launcher = returnTo.current;
    if (!dialog.current?.contains(document.activeElement)) dialog.current?.querySelector<HTMLInputElement>("input")?.focus();
    return () => {
      lifetime.current = generation + 1;
      if (launcher?.isConnected) launcher.focus();
    };
  }, []);

  useEffect(() => {
    function keepFocus(event: KeyboardEvent) {
      if (event.key === "Escape") {
        event.preventDefault();
        event.stopPropagation();
        if (!saving) onClose();
        return;
      }
      if (event.key !== "Tab" || !dialog.current) return;
      const controls = Array.from(dialog.current.querySelectorAll<HTMLElement>(
        'button:not([disabled]), input:not([disabled]):not([type="hidden"]), textarea:not([disabled]), select:not([disabled]), a[href], [tabindex]:not([tabindex="-1"])',
      ));
      const first = controls[0];
      const last = controls.at(-1);
      if (!first || !last) {
        event.preventDefault();
        dialog.current.focus();
      } else if (event.shiftKey && (document.activeElement === first || !dialog.current.contains(document.activeElement))) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && (document.activeElement === last || !dialog.current.contains(document.activeElement))) {
        event.preventDefault();
        first.focus();
      }
    }
    window.addEventListener("keydown", keepFocus, true);
    return () => window.removeEventListener("keydown", keepFocus, true);
  }, [onClose, saving]);

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const projectName = name.trim();
    if (!projectName || pending.current) return;
    const generation = lifetime.current;
    pending.current = true;
    setSaving(true);
    setError("");
    try {
      const project = await createProject(projectName, instructions.trim(), modelId);
      if (lifetime.current === generation) onCreated(project);
    } catch (reason) {
      if (lifetime.current === generation) setError(messageOf(reason));
    } finally {
      if (lifetime.current === generation) {
        pending.current = false;
        setSaving(false);
      }
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
        ref={dialog}
        className="project-editor-dialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby="new-project-title"
        tabIndex={-1}
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
          {privateAvailable ? (
            <fieldset className="project-processing-choice">
              <legend>Processing</legend>
              <div>
                <button type="button" aria-pressed={modelId === "auto"} onClick={() => setModelId("auto")} disabled={saving}>
                  <strong>Auto</strong><span>Available local and cloud models</span>
                </button>
                <button type="button" aria-pressed={modelId === "private"} onClick={() => setModelId("private")} disabled={saving}>
                  <strong>Private</strong><span>Local models and document tools</span>
                </button>
              </div>
              <p>This choice applies to every conversation and file in the project and cannot be changed later.</p>
            </fieldset>
          ) : null}
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
  visible = true,
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
  onNewProjectAgent,
  onOpenProjectAgent,
  onNewProjectWatch,
  onOpenProjectWatch,
}: {
  visible?: boolean;
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
  onNewProjectAgent?: (project: AudreyProject, signal?: AbortSignal) => void | Promise<void>;
  onResearchProjectKnowledge?: (project: AudreyProject, entry: KnowledgeEntry, signal?: AbortSignal) => void | Promise<void>;
  onOpenProjectAgent?: (task: AgentTask, signal?: AbortSignal) => void | Promise<void>;
  onNewProjectWatch?: (project: AudreyProject, signal?: AbortSignal) => void | Promise<void>;
  onOpenProjectWatch?: (watch: PublicWatch, signal?: AbortSignal) => void | Promise<void>;
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
  const [uploadingFiles, setUploadingFiles] = useState(false);
  const [uploadingName, setUploadingName] = useState("");
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadNotice, setUploadNotice] = useState("");
  const [error, setError] = useState("");
  const projectUploadRef = useRef<HTMLInputElement>(null);
  // Keep the canonical save response available until the parent supplies new saved fields.
  const savedProjectFields = useRef({ id: project.id, name: project.name, instructions: project.instructions });
  const draftProjectId = useRef(project.id);
  const editDirty = useRef(false);
  const isPrivate = project.private_required === true;

  useEffect(() => {
    savedProjectFields.current = { id: project.id, name: project.name, instructions: project.instructions };
    let current = true;
    queueMicrotask(() => {
      if (!current) return;
      if (draftProjectId.current !== project.id) {
        draftProjectId.current = project.id;
        editDirty.current = false;
        setEditing(false);
        setConfirmingDelete(false);
        setUploadNotice("");
        setError("");
      }
      // Late initial details can fill the form; background refreshes must keep user edits.
      if (!editDirty.current) {
        setName(project.name);
        setInstructions(project.instructions);
      }
    });
    return () => { current = false; };
  }, [project.id, project.instructions, project.name]);

  function resetProjectDraft() {
    const saved = savedProjectFields.current.id === project.id ? savedProjectFields.current : project;
    draftProjectId.current = project.id;
    editDirty.current = false;
    setName(saved.name);
    setInstructions(saved.instructions);
  }

  const hasProcessingFiles = files.some(({ status }) =>
    status === "pending" || status === "processing",
  );

  useEffect(() => {
    if (!hasProcessingFiles) return;
    let active = true;
    let refreshing = false;
    const refreshFiles = () => {
      if (refreshing || document.visibilityState !== "visible") return;
      refreshing = true;
      listProjectFiles(project.id)
        .then((listing) => {
          if (active) onFilesChange(listing.items);
        })
        .catch((reason: unknown) => {
          if (active) setError(messageOf(reason));
        })
        .finally(() => { refreshing = false; });
    };
    const timer = window.setInterval(refreshFiles, 5000);
    document.addEventListener("visibilitychange", refreshFiles);
    return () => {
      active = false;
      window.clearInterval(timer);
      document.removeEventListener("visibilitychange", refreshFiles);
    };
  }, [hasProcessingFiles, onFilesChange, project.id]);

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
      if (savedProjectFields.current.id !== project.id) return;
      savedProjectFields.current = { id: updated.id, name: updated.name, instructions: updated.instructions };
      editDirty.current = false;
      setName(updated.name);
      setInstructions(updated.instructions);
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
      onConversationCreated(await createProjectConversation(project.id, isPrivate ? "private" : defaultModelId));
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

  async function uploadToProject(selected: FileList) {
    const chosen = Array.from(selected);
    if (!chosen.length || uploadingFiles) return;
    const remaining = limits.max_files - files.length;
    if (remaining <= 0 || chosen.length > remaining) {
      setError(`This project has room for ${Math.max(0, remaining)} more file${remaining === 1 ? "" : "s"}.`);
      if (projectUploadRef.current) projectUploadRef.current.value = "";
      return;
    }
    setUploadingFiles(true);
    setUploadProgress(0);
    setUploadNotice("");
    setError("");
    let nextFiles = files;
    let processing = 0;
    try {
      const library = await listFiles();
      for (const file of chosen) {
        const precheck = uploadPrecheck(file, library.limits, isPrivate);
        if (precheck) throw new Error(`${file.name}: ${precheck}`);
        setUploadingName(file.name);
        setUploadProgress(0);
        if (isPrivate) {
          await uploadPrivateProjectFile(project.id, file, library.limits, setUploadProgress);
          nextFiles = (await listProjectFiles(project.id)).items;
        } else {
          const uploaded = await uploadFile(file, library.limits, setUploadProgress);
          const relation = await addProjectFile(project.id, uploaded.id);
          if (relation.status !== "ready") processing += 1;
          nextFiles = [relation, ...nextFiles];
        }
        onFilesChange(nextFiles);
      }
      setUploadNotice(
        processing
          ? `${chosen.length} file${chosen.length === 1 ? "" : "s"} added. Processing files become project context when Ready.`
          : `${chosen.length} file${chosen.length === 1 ? "" : "s"} added to this project.`,
      );
    } catch (reason) {
      setError(messageOf(reason));
    } finally {
      setUploadingFiles(false);
      setUploadingName("");
      setUploadProgress(0);
      if (projectUploadRef.current) projectUploadRef.current.value = "";
    }
  }

  return (
    <div className="project-home" data-private={isPrivate ? "true" : undefined}>
      <header className="project-home-header">
        <div className="project-home-title-line">
          <h1 className="workspace-page-title">{project.name}</h1>
          {isPrivate ? <span className="project-private-label">Private</span> : null}
          {project.instructions?.trim() ? <p title={project.instructions}>{project.instructions}</p> : null}
        </div>
        <div className="project-home-actions">
          <div className="project-management-actions">
            <button type="button" aria-label={editing ? "Close edit" : "Edit project"} onClick={() => {
              resetProjectDraft();
              setEditing((current) => !current);
            }} disabled={deleting}>
              {editing ? "Close edit" : <><span className="project-action-label-full">Edit project</span><span className="project-action-label-compact" aria-hidden="true">Edit</span></>}
            </button>
            <button
              className={confirmingDelete ? "danger-button confirming-delete" : "danger-button"}
              type="button"
              aria-label={confirmingDelete ? "Confirm delete project" : "Delete project"}
              aria-pressed={confirmingDelete}
              title={confirmingDelete
                ? isPrivate
                  ? "Click again to delete project. Conversations and files will be kept."
                  : "Click again to delete project. Linked watches stop; saved checks, conversations and files will be kept."
                : isPrivate
                  ? "Conversations and files will be kept. Conversations return to your chat history with their processing setting unchanged."
                  : "Linked watches stop; saved checks, conversations and files will be kept. Conversations return to your chat history with their processing setting unchanged."}
              onClick={() => {
                if (confirmingDelete) {
                  void removeProject();
                } else {
                  setConfirmingDelete(true);
                }
              }}
              onBlur={() => setConfirmingDelete(false)}
              onKeyDown={(event) => {
                if (event.key === "Escape") setConfirmingDelete(false);
              }}
              disabled={deleting}
            >
              {deleting ? "Deleting…" : confirmingDelete ? "✓" : <><span className="project-action-label-full">Delete project</span><span className="project-action-label-compact" aria-hidden="true">Delete</span></>}
            </button>
          </div>
        </div>
      </header>

      {editing ? (
        <form className="project-edit-form" onSubmit={(event) => void saveProject(event)}>
          <label>
            <span>Project name</span>
            <input
              value={name}
              onChange={(event) => { editDirty.current = true; setName(event.target.value); }}
              maxLength={limits.max_name_chars}
              required
            />
          </label>
          <label>
            <span>Project instructions</span>
            <textarea
              value={instructions}
              onChange={(event) => { editDirty.current = true; setInstructions(event.target.value); }}
              maxLength={limits.max_instructions_chars}
              rows={5}
            />
          </label>
          <div className="project-dialog-actions">
            <button type="button" onClick={() => {
              resetProjectDraft();
              setEditing(false);
            }}>Cancel</button>
            <button className="project-primary-button" type="submit" disabled={!name.trim() || saving}>
              {saving ? "Saving…" : "Save project"}
            </button>
          </div>
        </form>
      ) : null}

      {error ? <p className="project-error" role="alert">{error}</p> : null}

      <div className="project-home-layout">
        <ProjectActivity key={project.id}
          agents={!isPrivate && (onNewProjectAgent || onOpenProjectAgent) ? (active) => <ProjectAgents
            project={project} visible={visible && active}
            onNew={onNewProjectAgent} onOpen={onOpenProjectAgent} /> : undefined}
          watches={!isPrivate && (onNewProjectWatch || onOpenProjectWatch) ? (active) => <ProjectWatches
            project={project} visible={visible && active}
            onNew={onNewProjectWatch} onOpen={onOpenProjectWatch} /> : undefined}
          chats={
            <section className="project-panel project-conversations" aria-labelledby="project-conversations-title">
              <header>
                <div>
                  <h2 id="project-conversations-title">Project conversations</h2>
                </div>
                <div className="project-conversation-actions">
                  <small>{conversations.length}</small>
                  <button
                    className="project-primary-button"
                    type="button"
                    aria-label={creatingConversation ? "Creating…" : "New conversation in project"}
                    onClick={() => void startConversation()}
                    disabled={creatingConversation || deleting}
                  >{creatingConversation ? "Creating…" : "New conversation"}</button>
                </div>
              </header>
              {conversations.length ? (
                <ul className="project-home-conversations">
                  {conversations.map((conversation) => (
                    <li className="conversation-row" key={conversation.id}>
                      <SidebarItemButton
                        className="conversation project-home-conversation"
                        data-private={conversation.private_required ? "true" : undefined}
                        type="button"
                        onClick={() => onConversationSelected(conversation)}
                        title={conversation.title || "New conversation"}
                      >
                        <span className="sidebar-item-title">{conversation.title || "New conversation"}</span>
                        <small className="sidebar-item-meta">{conversation.archived_at ? "Archived" : formatActivity(conversation)}</small>
                      </SidebarItemButton>
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="project-empty">No conversations in this view.</p>
              )}
            </section>
          }
        />
        <div className="project-context">
          <ProjectKnowledgeSection project={project} visible={visible} />
          <section className="project-panel project-files" aria-labelledby="project-files-title">
            <header>
              <div>
                <h2 id="project-files-title">Project files</h2>
              </div>
              <small>{files.length}/{limits.max_files}</small>
            </header>
            <p className="project-panel-copy">
              {isPrivate
                ? "Local models use this project's instructions and ready Private documents. Upload a text document or choose from your Private files."
                : "Upload here or select Ready files already in My Files. Processing uploads become shared context when Ready."}
            </p>
            <div className="project-file-actions">
              <input
                ref={projectUploadRef}
                type="file"
                accept={isPrivate ? PRIVATE_DOCUMENT_EXTENSIONS.join(",") : undefined}
                multiple
                hidden
                disabled={uploadingFiles || files.length >= limits.max_files}
                onChange={(event) => event.target.files ? void uploadToProject(event.target.files) : undefined}
              />
              <button
                className="project-upload-files"
                type="button"
                onClick={() => projectUploadRef.current?.click()}
                disabled={uploadingFiles || files.length >= limits.max_files}
              >
                <span aria-hidden="true">＋</span>
                {uploadingFiles ? (uploadProgress >= 1 ? "Processing…" : "Uploading…") : "Upload file"}
              </button>
              <button className="project-manage-files" type="button" onClick={() => setFilesOpen(true)} disabled={uploadingFiles}>
                Choose from My Files
              </button>
            </div>
            {uploadingFiles ? (
              <UploadProgress filename={uploadingName} progress={uploadProgress} ariaLabel="Project file upload progress" />
            ) : null}
            {uploadNotice ? <p className="project-upload-notice" role="status">{uploadNotice}</p> : null}
            {filesLoading ? <p className="project-empty" role="status">Loading project files…</p> : null}
            {!filesLoading && files.length === 0 ? (
              <p className="project-empty">No files selected.</p>
            ) : null}
            {files.length ? (
              <ul className="project-file-summary">
                {files.map((file) => (
                  <li key={file.id}>
                    <span className="project-file-symbol" aria-hidden="true">{kindSymbol(file.kind)}</span>
                    <span className="project-file-details" title={file.filename}>
                      <span>{file.origin === "generated" ? <FileDownloadLink href={`/api/files/${encodeURIComponent(file.id)}/download`}>{file.filename}</FileDownloadLink> : file.filename}</span>
                      {file.origin !== "generated" && file.mime === "application/pdf" && file.status === "ready" ? <FilePreviewButton fileId={file.id} label={`Preview ${file.filename}`} /> : null}
                      {file.origin === "generated" ? <small>Created by Audrey</small> : null}
                      {file.status === "ready" ? null : <small>{projectFileStatus(file.status)}</small>}
                    </span>
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
      </div>
      {isPrivate ? <TaxProjectSection projectId={project.id} onConversationSelected={onConversationSelected} /> : null}

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

type ProjectActivityTab = "chats" | "agents" | "watches";

function ProjectActivity({ chats, agents, watches }: {
  chats: ReactNode;
  agents?: (active: boolean) => ReactNode;
  watches?: (active: boolean) => ReactNode;
}) {
  const [selected, setSelected] = useState<ProjectActivityTab>("chats");
  const id = useId();
  const buttons = useRef<Partial<Record<ProjectActivityTab, HTMLButtonElement | null>>>({});
  const tabs: { id: ProjectActivityTab; label: string }[] = [{ id: "chats", label: "Chats" }];
  if (agents) tabs.push({ id: "agents", label: "Agents" });
  if (watches) tabs.push({ id: "watches", label: "Watches" });
  const active = tabs.some((tab) => tab.id === selected) ? selected : "chats";

  return <section className="project-panel project-activity" aria-label="Project activity">
    <div className="project-activity-tabs" role="tablist" aria-label="Project activity">
      {tabs.map((tab, index) => <button key={tab.id} type="button" role="tab"
        id={`${id}-${tab.id}-tab`} aria-controls={`${id}-${tab.id}-panel`}
        aria-selected={active === tab.id} tabIndex={active === tab.id ? 0 : -1}
        ref={(node) => { buttons.current[tab.id] = node; }}
        onClick={() => setSelected(tab.id)}
        onKeyDown={(event) => {
          const next = event.key === "Home" ? 0 : event.key === "End" ? tabs.length - 1
            : event.key === "ArrowRight" ? (index + 1) % tabs.length
            : event.key === "ArrowLeft" ? (index - 1 + tabs.length) % tabs.length : null;
          if (next === null) return;
          event.preventDefault();
          setSelected(tabs[next].id);
          buttons.current[tabs[next].id]?.focus();
        }}>{tab.label}</button>)}
    </div>
    {tabs.map((tab) => <div className="project-activity-panel" key={tab.id} role="tabpanel"
      id={`${id}-${tab.id}-panel`} aria-labelledby={`${id}-${tab.id}-tab`}
      hidden={active !== tab.id} tabIndex={0}>
      {tab.id === "chats" ? chats : tab.id === "agents" ? agents?.(active === "agents") : watches?.(active === "watches")}
    </div>)}
  </section>;
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
        if (active) setFiles(listing.items.filter((file) => file.status === "ready"
          && Boolean(file.private_required) === Boolean(project.private_required)
          && (!project.private_required || file.kind === "text")));
      })
      .catch((reason: unknown) => {
        if (active) setError(messageOf(reason));
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => { active = false; };
  }, [project.private_required]);

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
          Select Ready {project.private_required ? "Private documents" : "files"} from My Files. {selected.length} of {limits.max_files} selected.
        </p>
        {error ? <p className="file-manager-error" role="alert">{error}</p> : null}
        {loading ? <p className="file-manager-status" role="status">Loading ready files…</p> : null}
        {!loading ? (
          <div className="file-explorer">
            <aside className="file-explorer-sidebar" aria-label="Project file folders">
              <span>My Files</span>
              {PROJECT_FILE_FOLDERS.filter((folder) => !project.private_required || ["all", "text"].includes(folder.kind)).map((folder) => {
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

function projectFileStatus(status: string) {
  if (status === "pending" || status === "processing") return "Processing";
  if (status === "failed") return "Processing failed";
  return status;
}

function formatBytes(bytes: number) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function messageOf(reason: unknown) {
  return reason instanceof Error ? reason.message : "The request could not be completed.";
}
