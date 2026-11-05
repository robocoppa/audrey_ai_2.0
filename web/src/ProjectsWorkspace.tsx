import { useEffect, useRef, useState } from "react";

import { NewProjectDialog, ProjectHome } from "./ProjectWorkspace";
import {
  getProject, listProjects, listProjectConversations, listProjectFiles,
  type AudreyModel, type AudreyProject, type AudreyProjectFile, type Conversation,
  type ProjectLimits, type AgentTask, type PublicWatch,
} from "./api";
import "./ProjectsWorkspace.css";

const DEFAULT_LIMITS: ProjectLimits = { max_name_chars: 100, max_instructions_chars: 4000, max_files: 20 };

export function ProjectsWorkspace({ visible, models, openRequest, onOpenConversation, onNewProjectAgent,
  onOpenProjectAgent, onNewProjectWatch, onOpenProjectWatch }: {
  visible: boolean;
  models: AudreyModel[];
  openRequest?: { project: AudreyProject; serial: number };
  onOpenConversation: (conversation: Conversation) => void;
  onNewProjectAgent: (project: AudreyProject, signal?: AbortSignal) => void | Promise<void>;
  onOpenProjectAgent: (task: AgentTask, signal?: AbortSignal) => void | Promise<void>;
  onNewProjectWatch: (project: AudreyProject, signal?: AbortSignal) => void | Promise<void>;
  onOpenProjectWatch: (watch: PublicWatch, signal?: AbortSignal) => void | Promise<void>;
}) {
  const [projects, setProjects] = useState<AudreyProject[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [project, setProject] = useState<AudreyProject | null>(null);
  const [files, setFiles] = useState<AudreyProjectFile[]>([]);
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [limits, setLimits] = useState(DEFAULT_LIMITS);
  const [nextCursor, setNextCursor] = useState<string | null>(null);
  const [conversationCursor, setConversationCursor] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [detailLoading, setDetailLoading] = useState(false);
  const [loadingMore, setLoadingMore] = useState(false);
  const [creating, setCreating] = useState(false);
  const [error, setError] = useState("");
  const [refresh, setRefresh] = useState(0);
  const openedRequest = useRef<number | null>(null);
  const selection = useRef<string | null>(null);
  const listGeneration = useRef(0);
  const lifetime = useRef(0);
  const active = useRef(visible);
  const defaultModelId = models.find(({ id }) => id === "auto")?.id ?? models[0]?.id ?? "auto";

  useEffect(() => { lifetime.current += 1; return () => { lifetime.current += 1; }; }, []);
  useEffect(() => { active.current = visible; }, [visible]);

  useEffect(() => {
    if (!visible) return;
    const controller = new AbortController();
    const generation = ++listGeneration.current;
    queueMicrotask(() => { if (!controller.signal.aborted) { setLoading(true); setError(""); } });
    void listProjects(undefined, controller.signal).then((page) => {
      if (controller.signal.aborted || generation !== listGeneration.current) return;
      setProjects((current) => {
        const selected = current.find(({ id }) => id === selection.current);
        return selected && !page.items.some(({ id }) => id === selected.id) ? [...page.items, selected] : page.items;
      });
      setNextCursor(page.next_cursor);
      if (page.limits) setLimits(page.limits);
    }).catch((reason: unknown) => { if (!controller.signal.aborted && generation === listGeneration.current) setError(messageOf(reason)); })
      .finally(() => { if (!controller.signal.aborted && generation === listGeneration.current) setLoading(false); });
    return () => controller.abort();
  }, [visible, refresh]);

  useEffect(() => {
    if (!visible || !openRequest || openedRequest.current === openRequest.serial) return;
    let active = true;
    queueMicrotask(() => {
      if (!active) return;
      openedRequest.current = openRequest.serial;
      const next = openRequest.project;
      setProjects((current) => [next, ...current.filter(({ id }) => id !== next.id)]);
      selection.current = next.id; setSelectedId(next.id); setProject(next);
    });
    return () => { active = false; };
  }, [visible, openRequest]);

  useEffect(() => {
    if (!visible || !selectedId) return;
    const controller = new AbortController();
    queueMicrotask(() => { if (!controller.signal.aborted) { setDetailLoading(true); setError(""); } });
    void Promise.all([getProject(selectedId, controller.signal), listProjectConversations(selectedId), listProjectFiles(selectedId, controller.signal)])
      .then(([current, threads, linked]) => {
        if (controller.signal.aborted) return;
        setProject(current); setConversations(threads.items); setConversationCursor(threads.next_cursor); setFiles(linked.items);
        if (linked.limits) setLimits(linked.limits);
        setProjects((items) => items.map((item) => item.id === current.id ? current : item));
      }).catch((reason: unknown) => {
        if (!controller.signal.aborted) { setProject(null); setConversations([]); setFiles([]); setError(messageOf(reason)); }
      }).finally(() => { if (!controller.signal.aborted) setDetailLoading(false); });
    return () => controller.abort();
  }, [visible, selectedId, refresh]);

  useEffect(() => { if (!visible) queueMicrotask(() => setCreating(false)); }, [visible]);

  function select(id: string) {
    selection.current = id || null; setSelectedId(selection.current);
    setError(""); setConversationCursor(null);
    if (!id) setDetailLoading(false);
    if (id !== project?.id) { setConversations([]); setFiles([]); }
  }

  async function loadMoreProjects() {
    if (!nextCursor || loadingMore) return;
    const generation = listGeneration.current;
    const mounted = lifetime.current;
    setLoadingMore(true);
    try {
      const page = await listProjects(nextCursor);
      if (mounted !== lifetime.current || generation !== listGeneration.current) return;
      setProjects((current) => [...current, ...page.items.filter((item) => !current.some(({ id }) => id === item.id))]);
      setNextCursor(page.next_cursor);
    } catch (reason) { if (mounted === lifetime.current && generation === listGeneration.current) setError(messageOf(reason)); }
    finally { if (mounted === lifetime.current) setLoadingMore(false); }
  }

  async function loadMoreConversations() {
    if (!selectedId || !conversationCursor || detailLoading) return;
    const id = selectedId;
    const mounted = lifetime.current;
    setDetailLoading(true);
    try {
      const page = await listProjectConversations(id, { cursor: conversationCursor });
      if (mounted !== lifetime.current || selection.current !== id) return;
      setConversations((current) => [...current, ...page.items.filter((item) => !current.some(({ id: saved }) => saved === item.id))]);
      setConversationCursor(page.next_cursor);
    } catch (reason) { if (mounted === lifetime.current && selection.current === id) setError(messageOf(reason)); }
    finally { if (mounted === lifetime.current && selection.current === id) setDetailLoading(false); }
  }

  return <div className="projects-workspace">
    <aside className="projects-directory" aria-label="Projects" inert={creating ? true : undefined}>
      <header><h2>Projects</h2><button type="button" onClick={() => setCreating(true)}>New project</button></header>
      <select className="projects-compact-picker" aria-label="Choose project" value={selectedId ?? ""} onChange={(event) => select(event.target.value)}>
        <option value="">Choose a project</option>{projects.map((item) => <option key={item.id} value={item.id}>{item.name}</option>)}
      </select>
      {loading ? <p className="project-empty" role="status">Loading projects…</p> : null}
      <nav className="projects-directory-list" aria-label="Project list">
        {projects.map((item) => <button key={item.id} type="button" aria-current={selectedId === item.id ? "page" : undefined}
          onClick={() => select(item.id)}><span>{item.name}</span>{item.private_required ? <small>Private</small> : null}</button>)}
      </nav>
      {!loading && projects.length === 0 ? <p className="project-empty">No projects yet.</p> : null}
      {nextCursor ? <button type="button" disabled={loadingMore} onClick={() => void loadMoreProjects()}>{loadingMore ? "Loading…" : "More projects"}</button> : null}
    </aside>
    <section className="projects-selected" aria-label="Selected project" inert={creating ? true : undefined}>
      {error ? <div className="projects-workspace-error" role="alert"><p>{error}</p><button type="button" onClick={() => setRefresh((value) => value + 1)}>Retry</button></div> : null}
      {project && project.id === selectedId ? <ProjectHome key={project.id} visible={visible} project={project} conversations={conversations}
        files={files} filesLoading={detailLoading} limits={limits} defaultModelId={defaultModelId}
        onProjectChange={(updated) => {
          listGeneration.current += 1; setLoading(false);
          if (selection.current === updated.id) setProject(updated);
          setProjects((current) => current.map((item) => item.id === updated.id ? updated : item));
        }}
        onProjectDeleted={(id) => {
          listGeneration.current += 1; setLoading(false);
          setProjects((current) => current.filter((item) => item.id !== id));
          if (selection.current === id) { select(""); setProject(null); }
        }}
        onConversationCreated={(conversation) => {
          if (selection.current !== project.id) return;
          setConversations((current) => [conversation, ...current.filter(({ id }) => id !== conversation.id)]);
          if (active.current) onOpenConversation(conversation);
        }}
        onConversationSelected={onOpenConversation}
        onFilesChange={(updated) => { if (selection.current === project.id) setFiles(updated); }}
        onNewProjectAgent={onNewProjectAgent} onOpenProjectAgent={onOpenProjectAgent}
        onNewProjectWatch={onNewProjectWatch} onOpenProjectWatch={onOpenProjectWatch} />
        : <div className="projects-workspace-empty"><h1>{detailLoading ? "Opening project…" : "Your projects"}</h1><p>{detailLoading ? "Loading saved conversations and files." : "Choose a project or create one to keep related work together."}</p></div>}
      {conversationCursor ? <button className="projects-more-conversations" type="button" disabled={detailLoading} onClick={() => void loadMoreConversations()}>More conversations</button> : null}
    </section>
    {creating ? <NewProjectDialog limits={limits} privateAvailable={models.some(({ id }) => id === "private")}
      onClose={() => setCreating(false)} onCreated={(created) => {
        listGeneration.current += 1;
        setLoading(false); setCreating(false); setProject(created);
        setProjects((current) => [created, ...current.filter(({ id }) => id !== created.id)]);
        select(created.id);
      }} /> : null}
  </div>;
}

function messageOf(reason: unknown) { return reason instanceof Error ? reason.message : "Project is unavailable. Try again."; }
