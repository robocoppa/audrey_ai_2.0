import { Children, isValidElement, useCallback, useEffect, useMemo, useRef, useState, type FormEvent } from "react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";

import {
  actOnAgentTask,
  answerAgentInput,
  ApiError,
  createAgentTask,
  deleteAgentTask,
  getAgentTask,
  getActivityUnreadCount,
  getProject,
  listAgentTasks,
  listProjects,
  type AgentTask,
  type AgentTaskStatus,
  type AgentInputRequest,
  type AudreyProject,
  type AudreyProjectFile,
  type PublicWatch,
  type WatchRun,
  type ProjectLimits,
} from "./api";
import "./AgentsWorkspace.css";
import { AgentOutputs } from "./AgentOutputs";
import { AgentRecentActivity } from "./AgentRecentActivity";
import { AgentFileInputsView, AgentFilePicker } from "./AgentFileInputs";
import { WatchesWorkspace } from "./WatchesWorkspace";
import { ActivityWorkspace } from "./ActivityWorkspace";
import { canAssignKnowledge, knowledgeTaskObjective, type KnowledgeAgentRequest } from "./knowledgeAgents";
import { NewProjectDialog } from "./ProjectWorkspace";

const STATUS_LABELS: Record<AgentTaskStatus, string> = {
  queued: "Queued", running: "Running", paused: "Paused", completed: "Completed",
  cancelled: "Cancelled", failed: "Failed", waiting_for_input: "Waiting for input",
};

const PROJECT_CONTEXT_HELP = "Uses Project instructions and knowledge. Select files separately when the agent needs to read their text.";

function stepLabel(step: AgentTask["steps"][number] | undefined): string | undefined {
  return step ? ({ plan: "Plan", research: "Work", review: "Review" }[step.id]) : undefined;
}

function isMoving(task: AgentTask) {
  return task.status === "queued" || task.status === "running";
}

function agentTitle(task: AgentTask) {
  return task.title?.trim() || task.objective;
}

function isFinished(task: AgentTask) {
  return ["completed", "cancelled", "failed"].includes(task.status);
}

function agentContextBlocked(task: AgentTask) {
  return Boolean(task.project_context && (!task.project_context.available || !task.project_context.current))
    || task.file_inputs?.current === false;
}

// Polls can finish after a user action. A saved newer revision always wins.
function mergeTasks(current: AgentTask[], incoming: AgentTask[], deleted: ReadonlySet<string>): AgentTask[] {
  const tasks = new Map(current.map((task) => [task.id, task]));
  for (const task of incoming) {
    if (deleted.has(task.id)) continue;
    const saved = tasks.get(task.id);
    if ((saved?.revision ?? -1) > task.revision) continue;
    if (task.summary && saved) {
      tasks.set(task.id, {
        ...task,
        result: saved.result,
        outputs: task.outputs ?? saved.outputs,
        file_inputs: task.file_inputs ?? saved.file_inputs,
        input_requests: task.revision === saved.revision ? saved.input_requests : task.input_requests,
        steps: task.steps.map((step) => ({ ...step, result: saved.steps.find(({ id }) => id === step.id)?.result ?? "" })),
        summary: task.revision === saved.revision ? saved.summary : true,
      });
    } else tasks.set(task.id, task);
  }
  return [...tasks.values()].sort((a, b) => b.created_at.localeCompare(a.created_at));
}

function messageOf(reason: unknown): string {
  return reason instanceof Error ? reason.message : "The request could not be completed.";
}

function objectiveTooLong(value: string): boolean {
  const text = value.trim();
  return Array.from(text).length > 6000 || new TextEncoder().encode(text).byteLength > 8000;
}

function newRequestId(): string {
  if (typeof crypto.randomUUID === "function") return crypto.randomUUID();
  const values = crypto.getRandomValues(new Uint32Array(4));
  return `agent-ui-${Array.from(values, (value) => value.toString(16)).join("-")}`;
}

interface AgentsWorkspaceProps {
  visible?: boolean;
  timezone?: string;
  onActiveCountChange?: (count: number) => void;
  navigationOpen?: boolean;
  onNavigationOpenChange?: (open: boolean) => void;
  composerProjectRequest?: { project: AudreyProject; serial: number };
  composerKnowledgeRequest?: KnowledgeAgentRequest;
  externalAgentRequest?: { task: AgentTask; serial: number };
  composerWatchProjectRequest?: { project: AudreyProject; serial: number };
  externalWatchRequest?: { watch: PublicWatch; serial: number };
  onOpenProject?: (projectId: string) => void;
}

interface AnswerDraft { text: string; revision: number }
function inputDraftKey(taskId: string, requestId: string) { return JSON.stringify([taskId, requestId]); }

export function AgentsWorkspace(props: AgentsWorkspaceProps) {
  const [view, setView] = useState<"tasks" | "watches" | "activity">("tasks");
  const [watchesOpened, setWatchesOpened] = useState(false);
  const [activityOpened, setActivityOpened] = useState(false);
  const [agentRequest, setAgentRequest] = useState<{ task: AgentTask; serial: number }>();
  const [watchRequest, setWatchRequest] = useState<{ watch: PublicWatch; run: WatchRun; serial: number }>();
  const requestSerial = useRef(0);
  const externalSerial = useRef<number | null>(null);
  const composerSerial = useRef<number | null>(null);
  const watchComposerSerial = useRef<number | null>(null);
  const watchExternalSerial = useRef<number | null>(null);
  const taskRevisions = useRef(new Map<string, number>());
  const receiveTasks = useCallback((tasks: AgentTask[]) => {
    taskRevisions.current = new Map(tasks.map((task) => [task.id, task.revision]));
  }, []);
  const [unread, setUnread] = useState(0);
  const unreadGeneration = useRef(0);
  const visible = props.visible ?? true;
  const receiveUnread = useCallback((count: number) => {
    unreadGeneration.current += 1;
    setUnread(count);
  }, []);
  useEffect(() => {
    const request = props.externalAgentRequest;
    if (!request || externalSerial.current === request.serial) return;
    externalSerial.current = request.serial;
    let current = true;
    queueMicrotask(() => {
      if (!current) return;
      setAgentRequest({ task: request.task, serial: ++requestSerial.current }); setView("tasks");
    });
    return () => { current = false; };
  }, [props.externalAgentRequest]);
  useEffect(() => {
    const request = !props.composerKnowledgeRequest || (props.composerProjectRequest?.serial ?? -1) > props.composerKnowledgeRequest.serial
      ? props.composerProjectRequest : props.composerKnowledgeRequest;
    if (!request || composerSerial.current === request.serial) return;
    composerSerial.current = request.serial;
    let current = true;
    queueMicrotask(() => { if (current) setView("tasks"); });
    return () => { current = false; };
  }, [props.composerProjectRequest, props.composerKnowledgeRequest]);
  useEffect(() => {
    const request = props.composerWatchProjectRequest;
    if (!request || watchComposerSerial.current === request.serial) return;
    watchComposerSerial.current = request.serial;
    let current = true;
    queueMicrotask(() => { if (current) { setWatchesOpened(true); setView("watches"); } });
    return () => { current = false; };
  }, [props.composerWatchProjectRequest]);
  useEffect(() => {
    const request = props.externalWatchRequest;
    if (!request || watchExternalSerial.current === request.serial) return;
    watchExternalSerial.current = request.serial;
    let current = true;
    queueMicrotask(() => { if (current) { setWatchesOpened(true); setView("watches"); } });
    return () => { current = false; };
  }, [props.externalWatchRequest]);
  useEffect(() => {
    if (!visible) return;
    const controller = new AbortController();
    let inFlight = false;
    async function refreshCount() {
      if (inFlight || controller.signal.aborted || document.visibilityState === "hidden") return;
      inFlight = true;
      const generation = unreadGeneration.current;
      try {
        const result = await getActivityUnreadCount(controller.signal);
        if (!controller.signal.aborted && generation === unreadGeneration.current) setUnread(result.unread_count);
      } catch {
        // Retain the last known count when this small background read fails.
      } finally { inFlight = false; }
    }
    void refreshCount();
    const timer = window.setInterval(() => void refreshCount(), 15000);
    window.addEventListener("focus", refreshCount);
    document.addEventListener("visibilitychange", refreshCount);
    return () => {
      controller.abort();
      window.clearInterval(timer);
      window.removeEventListener("focus", refreshCount);
      document.removeEventListener("visibilitychange", refreshCount);
    };
  }, [visible]);
  function navigate(next: typeof view) {
    if (next === "watches") setWatchesOpened(true);
    if (next === "activity") setActivityOpened(true);
    setView(next);
    props.onNavigationOpenChange?.(false);
  }
  return <section className="agents-hub" aria-label="Agents workspace">
    <nav className="agents-subnavigation" aria-label="Agent workspaces">
      <button type="button" aria-current={view === "tasks" ? "page" : undefined} onClick={() => navigate("tasks")}>Tasks</button>
      <button type="button" aria-current={view === "watches" ? "page" : undefined} onClick={() => navigate("watches")}>Watches</button>
      <button type="button" aria-current={view === "activity" ? "page" : undefined} onClick={() => navigate("activity")}>Activity{unread ? <span className="agents-active-count" aria-label={`${unread} unread notifications`}>{unread}</span> : null}</button>
    </nav>
    <div className="agents-hub-view" hidden={view !== "tasks"}><AgentTasksWorkspace {...props} onTasksChange={receiveTasks} openRequest={agentRequest} visible={visible && view === "tasks"} /></div>
    {watchesOpened ? <div className="agents-hub-view" hidden={view !== "watches"}><WatchesWorkspace
      visible={visible && view === "watches"}
      timezone={props.timezone}
      navigationOpen={props.navigationOpen}
      onNavigationOpenChange={props.onNavigationOpenChange}
      openRequest={watchRequest}
      composerProjectRequest={props.composerWatchProjectRequest}
      externalWatchRequest={props.externalWatchRequest}
      onOpenProject={props.onOpenProject}
    /></div> : null}
    {activityOpened ? <div className="agents-hub-view" hidden={view !== "activity"}><ActivityWorkspace visible={visible && view === "activity"} timezone={props.timezone} onUnreadChange={receiveUnread}
      isAgentResultCurrent={(task) => (taskRevisions.current.get(task.id) ?? -1) <= task.revision}
      onOpenAgent={(task) => { setAgentRequest({ task, serial: ++requestSerial.current }); navigate("tasks"); }}
      onOpenWatch={(watch, run) => { setWatchRequest({ watch, run, serial: ++requestSerial.current }); navigate("watches"); }}
    /></div> : null}
  </section>;
}

function AgentTasksWorkspace({
  visible = true,
  timezone,
  onActiveCountChange,
  navigationOpen,
  onNavigationOpenChange,
  openRequest,
  onTasksChange,
  composerProjectRequest,
  composerKnowledgeRequest,
  onOpenProject,
}: AgentsWorkspaceProps & { openRequest?: { task: AgentTask; serial: number }; onTasksChange: (tasks: AgentTask[]) => void }) {
  const [compactNavigation, setCompactNavigation] = useState(() =>
    window.matchMedia?.("(max-width: 760px)").matches ?? false);
  const [internalNavigationOpen, setInternalNavigationOpen] = useState(false);
  const [tasks, setTasks] = useState<AgentTask[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [objective, setObjective] = useState("");
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [refreshToken, setRefreshToken] = useState(0);
  const [detailRefreshToken, setDetailRefreshToken] = useState(0);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [busy, setBusy] = useState(false);
  const [answerDrafts, setAnswerDrafts] = useState<Record<string, AnswerDraft>>({});
  const [projects, setProjects] = useState<AudreyProject[]>([]);
  const [projectsCursor, setProjectsCursor] = useState<string | null>(null);
  const [projectsLoading, setProjectsLoading] = useState(false);
  const [projectsError, setProjectsError] = useState("");
  const [projectRefresh, setProjectRefresh] = useState(0);
  const [projectLimits, setProjectLimits] = useState<ProjectLimits>({ max_name_chars: 100, max_instructions_chars: 4000, max_files: 20 });
  const [creatingProject, setCreatingProject] = useState(false);
  const [composerProject, setComposerProject] = useState<AudreyProject | null>(null);
  const [composerFiles, setComposerFiles] = useState<{ projectId: string; files: AudreyProjectFile[] } | null>(null);
  const [composerOrigin, setComposerOrigin] = useState<{ id: string; revision: number } | null>(null);
  const [pendingKnowledgeRequest, setPendingKnowledgeRequest] = useState<KnowledgeAgentRequest | null>(null);
  const [unavailableProjectId, setUnavailableProjectId] = useState<string | null>(null);
  const [filter, setFilter] = useState<"all" | "active" | "finished">("all");
  const deletedIds = useRef(new Set<string>());
  const knownIds = useRef<Set<string> | null>(null);
  const mutationGeneration = useRef(0);
  const busyRef = useRef(false);
  const objectiveRef = useRef<HTMLTextAreaElement>(null);
  const newProjectRef = useRef<HTMLButtonElement>(null);
  const detailRef = useRef<HTMLElement>(null);
  const sidebarRef = useRef<HTMLElement>(null);
  const sidebarToggleRef = useRef<HTMLButtonElement>(null);
  const createRequest = useRef<{ key: string; requestId: string } | null>(null);
  const mounted = useRef(false);
  const openedRequest = useRef<number | null>(null);
  const openedComposerRequest = useRef<number | null>(null);
  const projectsRequest = useRef<AbortController | null>(null);
  const activeCount = tasks.filter(isMoving).length;
  const hasActive = activeCount > 0;
  const sidebarOpen = visible && compactNavigation && (navigationOpen ?? internalNavigationOpen);

  const setSidebarOpen = useCallback((open: boolean) => {
    setInternalNavigationOpen(open);
    onNavigationOpenChange?.(open);
  }, [onNavigationOpenChange]);

  const closeSidebar = useCallback(() => {
    setSidebarOpen(false);
    window.requestAnimationFrame(() => sidebarToggleRef.current?.focus({ preventScroll: true }));
  }, [setSidebarOpen]);

  useEffect(() => {
    const viewport = window.matchMedia?.("(max-width: 760px)");
    if (!viewport) return;
    const updateViewport = (event: MediaQueryListEvent) => {
      const focused = document.activeElement;
      const focusInSidebar = focused instanceof HTMLElement && sidebarRef.current?.contains(focused);
      const focusOnToggle = focused === sidebarToggleRef.current;
      setCompactNavigation(event.matches);
      // Chat and Agents share the app drawer state. A hidden workspace must not
      // dismiss the visible workspace's navigation when the viewport changes.
      if (!visible) return;
      setSidebarOpen(false);
      if (event.matches && focusInSidebar) {
        window.requestAnimationFrame(() => sidebarToggleRef.current?.focus({ preventScroll: true }));
      } else if (!event.matches && focusOnToggle) {
        window.requestAnimationFrame(() => objectiveRef.current?.focus({ preventScroll: true }));
      }
    };
    viewport.addEventListener("change", updateViewport);
    return () => viewport.removeEventListener("change", updateViewport);
  }, [setSidebarOpen, visible]);

  useEffect(() => {
    if (!sidebarOpen || creatingProject) return;
    sidebarRef.current?.querySelector<HTMLButtonElement>(".agents-sidebar-close")?.focus({ preventScroll: true });
    const handleKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault();
        closeSidebar();
      } else if (event.key === "Tab") {
        const controls = Array.from(sidebarRef.current?.querySelectorAll<HTMLElement>(
          'button:not([disabled]), textarea:not([disabled]), a[href], input:not([disabled]), select:not([disabled]), [tabindex="0"]',
        ) ?? []);
        const first = controls[0];
        const last = controls[controls.length - 1];
        if (event.shiftKey && document.activeElement === first) {
          event.preventDefault();
          last?.focus();
        } else if (!event.shiftKey && document.activeElement === last) {
          event.preventDefault();
          first?.focus();
        }
      }
    };
    document.addEventListener("keydown", handleKey);
    return () => document.removeEventListener("keydown", handleKey);
  }, [closeSidebar, sidebarOpen, creatingProject]);

  useEffect(() => {
    if (!visible) queueMicrotask(() => setCreatingProject(false));
  }, [visible]);

  useEffect(() => {
    mounted.current = true;
    return () => { mounted.current = false; };
  }, []);

  useEffect(() => { onActiveCountChange?.(activeCount); }, [activeCount, onActiveCountChange]);
  useEffect(() => { onTasksChange(tasks); }, [tasks, onTasksChange]);

  useEffect(() => {
    if (!visible) return;
    const controller = new AbortController(); projectsRequest.current?.abort(); projectsRequest.current = controller;
    queueMicrotask(() => { if (!controller.signal.aborted) { setProjectsLoading(true); setProjectsError(""); } });
    void listProjects(undefined, controller.signal).then((page) => {
      if (!controller.signal.aborted) { setProjects(page.items.filter((project) => !project.private_required)); setProjectsCursor(page.next_cursor); setProjectLimits(page.limits); }
    }).catch((reason: unknown) => {
      if (!controller.signal.aborted) setProjectsError(`Could not load Projects. ${messageOf(reason)}`);
    }).finally(() => { if (!controller.signal.aborted) setProjectsLoading(false); });
    return () => { controller.abort(); projectsRequest.current?.abort(); };
  }, [visible, projectRefresh]);

  useEffect(() => {
    const knowledgeRequest = composerKnowledgeRequest && composerKnowledgeRequest.serial >= (composerProjectRequest?.serial ?? -1) ? composerKnowledgeRequest : undefined;
    const request = knowledgeRequest ?? composerProjectRequest;
    if (!visible || !request || (openedComposerRequest.current ?? -1) >= request.serial) return;
    let current = true;
    queueMicrotask(() => {
      if (!current) return;
      openedComposerRequest.current = request.serial;
      let needsChoice = false;
      if (request.project.private_required) { setError("Private Projects are not available for agent tasks."); return; }
      if (knowledgeRequest) {
        if (!canAssignKnowledge(knowledgeRequest.project, knowledgeRequest.entry)) { setError("This entry is unavailable for an agent task. Refresh and review it before trying again."); return; }
        const nextObjective = knowledgeTaskObjective(knowledgeRequest.entry);
        if (objective.trim() && (objective !== nextObjective || composerProject?.id !== request.project.id)) {
          setPendingKnowledgeRequest(knowledgeRequest);
          needsChoice = true;
        } else {
          if (composerProject?.id !== request.project.id) setComposerFiles(null);
          setObjective(nextObjective); setComposerProject(request.project); setUnavailableProjectId(null);
          setComposerOrigin({ id: knowledgeRequest.entry.id, revision: knowledgeRequest.entry.revision }); setPendingKnowledgeRequest(null);
        }
      } else {
        if (composerProject?.id !== request.project.id) setComposerFiles(null);
        setComposerProject(request.project); setUnavailableProjectId(null); setComposerOrigin(null); setPendingKnowledgeRequest(null);
      }
      if (compactNavigation) setSidebarOpen(true);
      window.requestAnimationFrame(() => {
        const target = needsChoice ? sidebarRef.current?.querySelector<HTMLButtonElement>(".agents-knowledge-confirm button") : objectiveRef.current;
        target?.focus({ preventScroll: true });
      });
    });
    return () => { current = false; };
  }, [visible, composerProjectRequest, composerKnowledgeRequest, compactNavigation, setSidebarOpen, objective, composerProject?.id]);

  useEffect(() => {
    if (!visible || !openRequest || openedRequest.current === openRequest.serial) return;
    openedRequest.current = openRequest.serial;
    mutationGeneration.current += 1;
    knownIds.current?.add(openRequest.task.id);
    // The Activity reader has resolved the exact owned saved revision first.
    // Queue state updates so navigation effects can settle before focusing the result.
    const controller = new AbortController();
    queueMicrotask(() => {
      if (controller.signal.aborted) return;
      setTasks((current) => mergeTasks(current, [openRequest.task], deletedIds.current));
      setSelectedId(openRequest.task.id); setFilter("all"); setError(""); setNotice("");
      window.requestAnimationFrame(() => { if (compactNavigation) detailRef.current?.focus({ preventScroll: true }); });
    });
    return () => controller.abort();
  }, [visible, openRequest, compactNavigation]);

  useEffect(() => {
    const controller = new AbortController();
    let inFlight = false;
    async function refresh() {
      if (inFlight || controller.signal.aborted || document.visibilityState === "hidden") return;
      inFlight = true;
      setRefreshing(true);
      const generation = mutationGeneration.current;
      try {
        const { items } = await listAgentTasks(controller.signal);
        if (controller.signal.aborted) return;
        const incoming = items.filter((task) => !deletedIds.current.has(task.id));
        if (generation === mutationGeneration.current && !busyRef.current) {
          // This is the full retained catalog. Reconcile deletions from other sessions.
          const ids = new Set(incoming.map((task) => task.id));
          knownIds.current = ids;
          setTasks((current) => mergeTasks(current.filter((task) => ids.has(task.id)), incoming, deletedIds.current));
          setSelectedId((current) => current && ids.has(current) ? current : incoming[0]?.id ?? null);
        } else {
          setTasks((current) => mergeTasks(current, incoming, deletedIds.current));
          if (!busyRef.current) setRefreshToken((value) => value + 1);
        }
        setDetailRefreshToken((value) => value + 1);
      } catch (reason) {
        if (!controller.signal.aborted) setError(`Could not refresh agents. ${messageOf(reason)}`);
      } finally {
        inFlight = false;
        if (!controller.signal.aborted) {
          setLoading(false);
          setRefreshing(false);
        }
      }
    }
    void refresh();
    const timer = hasActive ? window.setInterval(() => void refresh(), 3000) : undefined;
    const onVisible = () => { if (visible || hasActive) void refresh(); };
    document.addEventListener("visibilitychange", onVisible);
    window.addEventListener("focus", onVisible);
    return () => {
      controller.abort();
      window.clearInterval(timer);
      document.removeEventListener("visibilitychange", onVisible);
      window.removeEventListener("focus", onVisible);
    };
  }, [visible, hasActive, refreshToken]);

  useEffect(() => {
    if (!selectedId || !visible) return;
    const controller = new AbortController();
    void getAgentTask(selectedId, controller.signal).then((task) => {
      if (!controller.signal.aborted && (!knownIds.current || knownIds.current.has(task.id))) {
        setTasks((current) => mergeTasks(current, [task], deletedIds.current));
      }
    }).catch((reason: unknown) => {
      if (!controller.signal.aborted) setError(messageOf(reason));
    });
    return () => controller.abort();
  }, [selectedId, visible, detailRefreshToken]);

  const dateFormatter = useMemo(() => {
    try {
      return new Intl.DateTimeFormat(undefined, { dateStyle: "medium", timeStyle: "short", timeZone: timezone });
    } catch {
      return new Intl.DateTimeFormat(undefined, { dateStyle: "medium", timeStyle: "short" });
    }
  }, [timezone]);

  function timestamp(value: string) {
    const date = new Date(value);
    return Number.isNaN(date.getTime()) ? "Time unavailable" : dateFormatter.format(date);
  }

  function showWorkflow() {
    if (!visible) return;
    if (compactNavigation) setSidebarOpen(false);
    window.requestAnimationFrame(() => {
      if (!detailRef.current) return;
      detailRef.current.scrollTop = 0;
      if (compactNavigation) detailRef.current.focus({ preventScroll: true });
    });
  }

  function selectTask(id: string) {
    setSelectedId(id);
    setNotice("");
    showWorkflow();
  }

  async function loadMoreProjects() {
    if (!projectsCursor || projectsLoading) return;
    const controller = new AbortController(); projectsRequest.current?.abort(); projectsRequest.current = controller;
    setProjectsLoading(true); setProjectsError("");
    try {
      const page = await listProjects(projectsCursor, controller.signal);
      if (controller.signal.aborted || !mounted.current) return;
      setProjects((current) => [...new Map([...current, ...page.items.filter((project) => !project.private_required)].map((project) => [project.id, project])).values()]);
      setProjectsCursor(page.next_cursor);
    } catch (reason) { if (!controller.signal.aborted && mounted.current) setProjectsError(`Could not load more Projects. ${messageOf(reason)}`); }
    finally { if (!controller.signal.aborted && mounted.current) setProjectsLoading(false); }
  }

  async function startTask(text: string, key: string) {
    if (busyRef.current || !text.trim() || objectiveTooLong(text) || pendingKnowledgeRequest || composerProject?.id === unavailableProjectId) return;
    const requestedProject = composerProject;
    const requestedOrigin = composerOrigin;
    const requestedFiles = composerFiles?.projectId === requestedProject?.id ? composerFiles?.files ?? [] : [];
    busyRef.current = true;
    mutationGeneration.current += 1;
    setBusy(true);
    setError("");
    setNotice("");
    let projectVerified = false;
    // Retain this identity after a lost response so Retry does not create duplicates.
    try {
      if (createRequest.current?.key !== key) {
        createRequest.current = { key, requestId: newRequestId() };
      }
      if (requestedProject) {
        const current = await getProject(requestedProject.id);
        if (!mounted.current) return;
        if (current.private_required) {
          setUnavailableProjectId(requestedProject.id);
          throw new ApiError(422, "This Project is Private. Choose another Project or None before starting an agent task.");
        }
        setComposerProject(current);
        projectVerified = true;
      }
      const task = requestedProject
        ? requestedFiles.length
          ? await createAgentTask(text.trim(), createRequest.current.requestId, requestedProject.id, requestedOrigin ?? undefined, requestedFiles.map(({ id }) => id))
          : requestedOrigin
          ? await createAgentTask(text.trim(), createRequest.current.requestId, requestedProject.id, requestedOrigin)
          : await createAgentTask(text.trim(), createRequest.current.requestId, requestedProject.id)
        : await createAgentTask(text.trim(), createRequest.current.requestId);
      if (!mounted.current) return;
      createRequest.current = null;
      knownIds.current?.add(task.id);
      setTasks((current) => mergeTasks(current, [task], deletedIds.current));
      setSelectedId(task.id);
      showWorkflow();
      setFilter("all");
      if (key.startsWith("new:")) { setObjective(""); setComposerOrigin(null); setComposerFiles(null); }
    } catch (reason) {
      if (mounted.current) {
        if (requestedProject && !projectVerified && reason instanceof ApiError && reason.status === 404) {
          setUnavailableProjectId(requestedProject.id);
          setError("The selected Project is unavailable. Choose another Project or None before starting.");
        } else setError(messageOf(reason));
      }
    } finally {
      busyRef.current = false;
      if (mounted.current) { setBusy(false); setRefreshToken((value) => value + 1); }
    }
  }

  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    void startTask(objective, `new:${JSON.stringify([objective.trim(), composerProject?.id ?? null, composerOrigin,
      composerFiles?.projectId === composerProject?.id ? composerFiles?.files.map(({ id }) => id) : []])}`);
  }

  async function act(task: AgentTask, action: "pause" | "resume" | "cancel" | "rerun") {
    if (busyRef.current || (action === "resume" || action === "rerun") && agentContextBlocked(task)) return;
    busyRef.current = true;
    mutationGeneration.current += 1;
    setBusy(true);
    setError("");
    setNotice("");
    try {
      const updated = await actOnAgentTask(task.id, action, task.revision);
      if (mounted.current) {
        setTasks((current) => mergeTasks(current, [updated], deletedIds.current));
        if (action === "rerun") {
          setSelectedId(updated.id);
          setFilter("all");
          showWorkflow();
        }
      }
    } catch (reason) {
      if (!mounted.current) return;
      if (reason instanceof ApiError && reason.status === 409) {
        try {
          const latest = await getAgentTask(task.id);
          if (mounted.current) {
            setTasks((current) => mergeTasks(current, [latest], deletedIds.current));
            setNotice("This agent changed while you were acting. Its latest saved state is shown; review it before trying again.");
          }
        } catch (reloadError) {
          if (mounted.current) setError(messageOf(reloadError));
        }
      } else setError(messageOf(reason));
    } finally {
      busyRef.current = false;
      if (mounted.current) { setBusy(false); setRefreshToken((value) => value + 1); }
    }
  }

  async function removeTask(task: AgentTask) {
    if (busyRef.current) return;
    busyRef.current = true;
    mutationGeneration.current += 1;
    setBusy(true);
    setError("");
    setNotice("");
    try {
      await deleteAgentTask(task.id, task.revision);
      if (!mounted.current) return;
      deletedIds.current.add(task.id);
      knownIds.current?.delete(task.id);
      setTasks((current) => current.filter((item) => item.id !== task.id));
      setSelectedId((current) => current === task.id ? null : current);
      setAnswerDrafts((current) => Object.fromEntries(Object.entries(current).filter(([key]) => JSON.parse(key)[0] !== task.id)));
    } catch (reason) {
      if (!mounted.current) return;
      if (reason instanceof ApiError && reason.status === 409) {
        try {
          const latest = await getAgentTask(task.id);
          if (mounted.current) {
            setTasks((current) => mergeTasks(current, [latest], deletedIds.current));
            setNotice("This agent changed while you were acting. Its latest saved state is shown; review it before trying again.");
          }
        } catch (reloadError) {
          if (mounted.current) setError(messageOf(reloadError));
        }
      } else setError(messageOf(reason));
    } finally {
      busyRef.current = false;
      if (mounted.current) { setBusy(false); setRefreshToken((value) => value + 1); }
    }
  }

  async function answerInput(task: AgentTask, request: AgentInputRequest, draft: AnswerDraft) {
    if (busyRef.current || agentContextBlocked(task) || task.status !== "waiting_for_input" || draft.revision !== task.revision
      || request.answer !== null || task.input_requests?.find((item) => item.answer === null)?.id !== request.id) return;
    busyRef.current = true; mutationGeneration.current += 1;
    setBusy(true); setError(""); setNotice("");
    try {
      const saved = await answerAgentInput(task.id, request.id, draft.revision, draft.text);
      if (!mounted.current) return;
      setTasks((current) => mergeTasks(current, [saved], deletedIds.current));
      setAnswerDrafts((current) => {
        const next = { ...current }; delete next[inputDraftKey(task.id, request.id)]; return next;
      });
      setNotice("Answer saved. This agent will continue planning with your answer.");
    } catch (reason) {
      if (!mounted.current) return;
      if (reason instanceof ApiError && reason.status === 409) {
        try {
          const latest = await getAgentTask(task.id);
          if (!mounted.current) return;
          setTasks((current) => mergeTasks(current, [latest], deletedIds.current));
          setNotice("This agent changed while you were answering. Review its latest question and saved answers before continuing.");
        } catch (reloadError) { if (mounted.current) setError(messageOf(reloadError)); }
      } else setError(messageOf(reason));
    } finally {
      busyRef.current = false;
      if (mounted.current) { setBusy(false); setRefreshToken((value) => value + 1); }
    }
  }

  const selected = tasks.find((task) => task.id === selectedId);
  const selectedInput = selected?.input_requests?.find((request) => request.answer === null);
  const selectedInputKey = selected && selectedInput ? inputDraftKey(selected.id, selectedInput.id) : "";
  const selectedInputDraft = selected ? answerDrafts[selectedInputKey] ?? { text: "", revision: selected.revision } : undefined;
  const composerProjects = composerProject && !projects.some((project) => project.id === composerProject.id) ? [composerProject, ...projects] : projects;
  const shownTasks = tasks.filter((task) => filter === "all"
    || (filter === "finished" ? isFinished(task) : !isFinished(task)));

  return (
    <section className="agents-workspace" aria-labelledby="agents-title" data-sidebar-open={sidebarOpen}>
      {compactNavigation ? <div className="agents-mobile-toolbar" inert={sidebarOpen ? true : undefined}>
        <button
          type="button"
          className="agents-secondary agents-sidebar-toggle"
          ref={sidebarToggleRef}
          aria-controls="agents-navigation"
          aria-expanded={sidebarOpen}
          onClick={() => setSidebarOpen(true)}
        >
          <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" aria-hidden="true"><path d="M4 5h16M4 12h16M4 19h16" /></svg>
          Agent controls
        </button>
        <span>Workflow</span>
      </div> : null}
      {sidebarOpen ? <button type="button" className="agents-sidebar-backdrop" tabIndex={-1} aria-label="Dismiss agent controls" onClick={closeSidebar} /> : null}
      <div className="agents-columns" inert={creatingProject ? true : undefined}>
        <aside
          className="agents-sidebar"
          id="agents-navigation"
          ref={sidebarRef}
          aria-label="Agent tasks"
          role={compactNavigation ? "dialog" : undefined}
          aria-modal={sidebarOpen ? true : undefined}
          aria-hidden={compactNavigation && !sidebarOpen ? true : undefined}
          inert={compactNavigation && !sidebarOpen ? true : undefined}
        >
          <header className="agents-header">
            <div>
              <p className="agents-eyebrow">Your tasks, in progress</p>
              <h1 id="agents-title">Agents</h1>
            </div>
            {compactNavigation
              ? <button type="button" className="agents-secondary agents-sidebar-close" aria-label="Close agent controls" onClick={closeSidebar}>✕</button>
              : <button type="button" className="agents-secondary" onClick={() => objectiveRef.current?.focus()}>New agent task</button>}
          </header>
          <div className="agents-sidebar-scroll">
            {sidebarOpen && error ? <p className="agents-error" role="alert">{error}</p> : null}
            {sidebarOpen && notice ? <p className="agents-notice" role="status">{notice}</p> : null}
            <form className="agents-composer" onSubmit={submit}>
              <div className="agents-project-heading">
                <label htmlFor="agent-project">Project <span className="agents-optional">Optional</span></label>
                <button ref={newProjectRef} className="agents-refresh agents-create-project" type="button" disabled={busy} onClick={() => setCreatingProject(true)}>New project</button>
              </div>
              <select id="agent-project" aria-label="Project" value={composerProject?.id ?? ""} disabled={busy} onChange={(event) => {
                setComposerProject(composerProjects.find((project) => project.id === event.target.value) ?? null); setUnavailableProjectId(null);
                setComposerOrigin(null); setPendingKnowledgeRequest(null); setComposerFiles(null);
              }}>
                <option value="">None</option>
                {composerProjects.map((project) => <option key={project.id} value={project.id}>{project.name}{project.id === unavailableProjectId ? " (unavailable)" : ""}</option>)}
              </select>
              <p>Agent tasks can use ordinary Projects. Private Projects are unavailable here.</p>
              {composerProject ? <p>{PROJECT_CONTEXT_HELP}</p> : null}
              {projectsLoading ? <p role="status">Loading Projects…</p> : null}
              {projectsError ? <div className="agents-project-error"><p role="alert">{projectsError}</p><button className="agents-secondary" type="button" disabled={projectsLoading} onClick={() => setProjectRefresh((value) => value + 1)}>Retry Projects</button></div> : null}
              {projectsCursor ? <button className="agents-secondary" type="button" disabled={projectsLoading} onClick={() => void loadMoreProjects()}>Load more Projects</button> : null}
              {composerProject?.id === unavailableProjectId ? <p className="agents-project-unavailable">The selected Project is unavailable. Choose another Project or None.</p> : null}
              {composerProject && composerProject.id !== unavailableProjectId ? <AgentFilePicker key={composerProject.id}
                projectId={composerProject.id} selected={composerFiles?.projectId === composerProject.id ? composerFiles.files : []}
                onChange={(files) => setComposerFiles({ projectId: composerProject.id, files })}
                visible={visible && (!compactNavigation || sidebarOpen)} disabled={busy} /> : null}
              {pendingKnowledgeRequest ? <div className="agents-knowledge-confirm" role="group" aria-label="Use Project Knowledge entry">
                <p>Your task draft is kept. Use this Project {pendingKnowledgeRequest.entry.kind} instead?</p>
                <blockquote>{pendingKnowledgeRequest.entry.text}</blockquote>
                <div><button className="agents-secondary" type="button" disabled={busy} onClick={() => {
                  if (composerProject?.id !== pendingKnowledgeRequest.project.id) setComposerFiles(null);
                  setObjective(knowledgeTaskObjective(pendingKnowledgeRequest.entry)); setComposerProject(pendingKnowledgeRequest.project);
                  setComposerOrigin({ id: pendingKnowledgeRequest.entry.id, revision: pendingKnowledgeRequest.entry.revision });
                  setUnavailableProjectId(null); setPendingKnowledgeRequest(null); setError(""); objectiveRef.current?.focus();
                }}>Use entry</button><button className="agents-secondary" type="button" disabled={busy} onClick={() => { setPendingKnowledgeRequest(null); objectiveRef.current?.focus(); }}>Keep my draft</button></div>
              </div> : null}
              {composerOrigin ? <div className="agents-knowledge-origin"><p>From Project Knowledge · Revision {composerOrigin.revision}. Agent work keeps the entry open. Updating, closing or deleting it stops linked agents from continuing.</p>
                <button className="agents-secondary" type="button" disabled={busy} onClick={() => { setComposerOrigin(null); objectiveRef.current?.focus(); }}>Remove entry link</button></div> : null}
              <label htmlFor="agent-objective">Task objective</label>
              <textarea
                id="agent-objective"
                ref={objectiveRef}
                rows={4}
                maxLength={6000}
                value={objective}
                onChange={(event) => setObjective(event.target.value)}
                placeholder="Describe the task and desired result."
                aria-describedby="agent-objective-help"
                required
                disabled={busy}
              />
              <p id="agent-objective-help">All agent steps use local models with no cloud fallback. You can leave this page and return to the saved progress and result.</p>
              {objectiveTooLong(objective) ? <p role="alert" className="agents-error">Shorten the task objective to 6,000 characters and 8,000 UTF-8 bytes before starting. {composerOrigin ? "Your full entry remains saved in Project Knowledge." : "Your draft is kept."}</p> : null}
              <button className="agents-primary" type="submit" disabled={busy || !objective.trim() || objectiveTooLong(objective) || Boolean(pendingKnowledgeRequest) || composerProject?.id === unavailableProjectId}>{busy ? "Saving…" : "Start agent"}</button>
              {objective || composerOrigin || composerFiles?.files.length ? <button className="agents-secondary" type="button" disabled={busy} onClick={() => { setObjective(""); setComposerOrigin(null); setComposerFiles(null); setPendingKnowledgeRequest(null); createRequest.current = null; objectiveRef.current?.focus(); }}>Clear draft</button> : null}
            </form>

            {selected && !isFinished(selected) ? (
              <section className="agents-current-controls" aria-labelledby="agent-current-controls-title">
                <h2 id="agent-current-controls-title">Current agent</h2>
                <p>{agentTitle(selected)}</p>
                <AgentTaskControls
                  key={`${selected.id}:active:${visible}:${sidebarOpen}`}
                  task={selected}
                  busy={busy}
                  onAction={(action) => void act(selected, action)}
                  onDelete={() => void removeTask(selected)}
                />
              </section>
            ) : null}

            <div className="agents-task-list-header">
              <h2>Saved tasks</h2>
              <button type="button" className="agents-refresh" onClick={() => { setError(""); setRefreshToken((value) => value + 1); }} disabled={refreshing} aria-label="Refresh agent tasks">{refreshing ? "Refreshing…" : "Refresh"}</button>
            </div>
            <div className="agents-filters" role="group" aria-label="Filter agent tasks">
              {([['all', 'All'], ['active', 'Active'], ['finished', 'Finished']] as const).map(([value, label]) => (
                <button key={value} type="button" aria-pressed={filter === value} onClick={() => setFilter(value)}>{label}</button>
              ))}
            </div>
            {loading ? <p className="agents-empty" role="status">Loading your agents…</p> : !shownTasks.length ? (
              <p className="agents-empty">{tasks.length ? "No tasks in this view." : "No agents yet. Start with a task objective above."}</p>
            ) : (
              <ul className="agents-task-list">
                {shownTasks.map((task) => (
                  <li key={task.id}>
                    <button type="button" className="agents-task-card" aria-pressed={selectedId === task.id} aria-label={`Open agent: ${agentTitle(task)}`} onClick={() => selectTask(task.id)}>
                      <span className={`agents-status agents-status-${task.status}`}>{STATUS_LABELS[task.status]}</span>
                      <strong>{agentTitle(task)}</strong>
                      {task.project_context ? <span className="agents-project-badge">Project: {task.project_context.project_name ?? "Unavailable"}</span> : null}
                      <span className="agents-task-stage">{task.status === "waiting_for_input" ? "Your answer is needed to continue" : stepLabel(task.steps.find((step) => step.status === "running")) ?? (task.status === "paused" ? "Completed steps saved" : task.status === "queued" ? "Waiting to start" : "Saved progress")}</span>
                      <time dateTime={task.updated_at}>Last activity {timestamp(task.updated_at)}</time>
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </aside>

        <section className="agents-workflow" aria-label="Selected agent" ref={detailRef} tabIndex={0} inert={sidebarOpen ? true : undefined}>
          {!sidebarOpen && error ? <p className="agents-error" role="alert">{error}</p> : null}
          {!sidebarOpen && notice ? <p className="agents-notice" role="status">{notice}</p> : null}
          <div className="agents-detail">
            {selected ? (
              <>
                <header className="agents-detail-header">
                  <div className="agents-detail-toolbar">
                    <div className="agents-detail-meta">
                      <span className={`agents-status agents-status-${selected.status}`}>{STATUS_LABELS[selected.status]}</span>
                      {selected.project_context ? <AgentProjectContextBadge task={selected} onOpenProject={onOpenProject} /> : null}
                    </div>
                    {isFinished(selected) ? <AgentTaskControls
                      key={`${selected.id}:${isFinished(selected) ? selected.revision : "active"}:${visible}`}
                      task={selected}
                      busy={busy}
                      onAction={(action) => void act(selected, action)}
                      onDelete={() => void removeTask(selected)}
                    /> : null}
                  </div>
                  <h2>{agentTitle(selected)}</h2>
                  <details className="agents-original-prompt"><summary>Original prompt</summary><p>{selected.objective}</p></details>
                  <p>Last activity <time dateTime={selected.updated_at}>{timestamp(selected.updated_at)}</time></p>
                </header>
                {selected.project_context && (!selected.project_context.available || !selected.project_context.current) ? <p className="agents-project-context-notice">
                  {!selected.project_context.available ? "This Project is no longer available. Continuing or running this agent again is unavailable." : "The Project context has changed. Start a new agent to use its current instructions and knowledge."}
                </p> : null}
                <AgentFileInputsView inputs={selected.file_inputs} />
                {selected.status === "paused" ? <p className="agents-notice">Completed steps are saved. Resume to continue from the next unfinished step.</p> : null}
                {selected.error ? <p className="agents-error" role="alert">{selected.error}</p> : null}
                {selected.status === "waiting_for_input" ? selectedInput && selectedInputDraft ? <AgentInputPanel request={selectedInput} revision={selected.revision} draft={selectedInputDraft} busy={busy} contextBlocked={agentContextBlocked(selected)}
                  onChange={(text) => setAnswerDrafts((current) => ({ ...current, [selectedInputKey]: { text, revision: current[selectedInputKey]?.revision ?? selected.revision } }))}
                  onUseLatest={() => setAnswerDrafts((current) => ({ ...current, [selectedInputKey]: { text: current[selectedInputKey]?.text ?? "", revision: selected.revision } }))}
                  onAnswer={() => void answerInput(selected, selectedInput, selectedInputDraft)} />
                  : <p className="agents-notice" role="status">{selected.summary ? "Loading the saved question…" : "The saved question is unavailable. Refresh this agent before answering."}</p> : null}
                {selected.input_requests?.some((request) => request.answer !== null) ? <details className="agents-input-history"><summary>Questions and answers</summary><ol>{selected.input_requests.filter((request) => request.answer !== null).map((request) => <li key={request.id}><h3>Planning question</h3><p>{request.question}</p><h4>Your answer</h4><p>{request.answer}</p></li>)}</ol></details> : null}
                <div className="agents-progress-panels">
                  <section className="agents-plan agents-progress-panel" aria-labelledby="agent-plan-title">
                    <header className="agents-panel-header"><h3 id="agent-plan-title">Plan and progress</h3></header>
                    <div className="agents-plan-scroll" role="region" aria-label="Saved plan and progress" tabIndex={0}>
                      <ol className="agents-step-list">
                        {selected.steps.map((step) => (
                          <li key={step.id} className={`agents-step agents-step-${step.status}`}>
                            <span className="agents-step-mark" aria-hidden="true">{step.status === "completed" ? "✓" : step.status === "running" ? "●" : "○"}</span>
                            <div>
                              <strong>{stepLabel(step)}</strong>
                              <span className="agents-step-status">{step.status === "completed" ? "Completed" : step.status === "running" ? "In progress" : "Pending"}</span>
                              {step.result ? <details><summary>View saved {step.id === "plan" ? "plan" : step.id === "research" ? "work" : "review"}</summary><AgentMarkdown text={step.result} /></details> : null}
                            </div>
                          </li>
                        ))}
                      </ol>
                    </div>
                  </section>
                  <AgentRecentActivity key={selected.id} task={selected} visible={visible && !sidebarOpen} timezone={timezone} />
                </div>
                <AgentOutputs key={`${selected.id}:${visible}`} outputs={selected.outputs} />
                <section className="agents-result" aria-labelledby="agent-result-title">
                  <header className="agents-result-header">
                    <h3 id="agent-result-title">Result</h3>
                  </header>
                  <div className="agents-result-body">
                    {selected.result ? <AgentMarkdown text={selected.result} /> : <p className="agents-result-empty">{selected.summary && selected.status === "completed" ? "Loading the saved result…" : selected.status === "failed" ? "This task stopped before a final result was saved. Any completed steps are available above." : selected.status === "cancelled" ? "This task was cancelled. Any completed steps are available above." : "The result will appear here when Audrey completes its review."}</p>}
                  </div>
                </section>
              </>
            ) : (
              <div className="agents-welcome">
                <span className="agents-welcome-mark" aria-hidden="true">✦</span>
                <h2>Work that continues while you’re away</h2>
                <p>Start a task or select a saved agent to see its objective, completed steps and result.</p>
                <ol><li>Plan the work</li><li>Work on the task</li><li>Review and save the result</li></ol>
              </div>
            )}
          </div>
        </section>
      </div>
      {creatingProject && visible ? <NewProjectDialog limits={projectLimits} privateAvailable={false}
        onClose={() => { setCreatingProject(false); window.requestAnimationFrame(() => newProjectRef.current?.focus()); }}
        onCreated={(project) => {
          projectsRequest.current?.abort();
          setProjects((current) => [project, ...current.filter((item) => item.id !== project.id)]);
          setProjectsLoading(false); setProjectsError(""); setProjectRefresh((value) => value + 1);
          setComposerProject(project); setComposerFiles(null); setComposerOrigin(null); setPendingKnowledgeRequest(null); setUnavailableProjectId(null);
          setCreatingProject(false); setNotice("Project created and selected. Your task draft is kept.");
          window.requestAnimationFrame(() => objectiveRef.current?.focus());
        }} /> : null}
    </section>
  );
}

function AgentProjectContextBadge({ task, onOpenProject }: { task: AgentTask; onOpenProject?: (projectId: string) => void }) {
  const context = task.project_context;
  if (!context) return null;
  const name = context.project_name ?? "Saved Project";
  const state = !context.available ? "Unavailable" : !context.current ? "Changed" : "";
  const description = [
    `Project: ${name}.`,
    state ? `Project context ${state.toLowerCase()}.` : "",
    context.instructions_included ? "Project instructions included." : "No Project instructions were selected.",
    `${context.knowledge.length} saved knowledge ${context.knowledge.length === 1 ? "revision" : "revisions"} pinned for this agent.`,
    context.origin_entry ? `Started from Project Knowledge · Revision ${context.origin_entry.revision}. Agent work does not complete or resolve the entry.` : "",
  ].filter(Boolean).join(" ");
  const contents = <><span className="agents-context-project-label">Project:</span><span className="agents-context-project-name">{name}</span>{state ? <span className="agents-context-project-state">{state}</span> : null}</>;
  const className = `agents-context-project${state ? " agents-context-project-changed" : ""}`;
  return context.available && onOpenProject
    ? <button type="button" className={className} title={description} aria-label={`Open Project: ${name}`} onClick={() => onOpenProject(context.project_id)}>{contents}</button>
    : <span className={className} title={description} aria-label={`Project: ${name}${state ? ` · ${state}` : ""}`}>{contents}</span>;
}

function AgentInputPanel({ request, revision, draft, busy, contextBlocked, onChange, onUseLatest, onAnswer }: {
  request: AgentInputRequest;
  revision: number;
  draft: AnswerDraft;
  busy: boolean;
  contextBlocked: boolean;
  onChange: (text: string) => void;
  onUseLatest: () => void;
  onAnswer: () => void;
}) {
  const bytes = new TextEncoder().encode(draft.text.trim()).byteLength;
  const changed = draft.revision !== revision;
  return <section className="agents-input-panel" aria-labelledby="agent-input-title">
    <h3 id="agent-input-title">Agent needs your input</h3>
    <p className="agents-input-question" id="agent-input-question">{request.question}</p>
    <p className="agents-input-help">This agent is waiting. Answer the planning question to continue the same saved task.</p>
    <form aria-label="Answer agent question" onSubmit={(event) => { event.preventDefault(); if (!busy && !contextBlocked && !changed && bytes > 0 && bytes <= 8000) onAnswer(); }}>
      <label htmlFor="agent-input-answer">Your answer</label>
      <textarea id="agent-input-answer" rows={4} maxLength={8000} aria-describedby="agent-input-question agent-input-size" value={draft.text} disabled={busy} onChange={(event) => onChange(event.target.value)} />
      <p id="agent-input-size" className="agents-input-help">{bytes.toLocaleString()} / 8,000 UTF-8 bytes</p>
      {bytes > 8000 ? <p className="agents-error" role="alert">Your answer exceeds the 8,000-byte limit. Shorten it before continuing.</p> : null}
      {changed ? <div className="agents-input-conflict"><p>This agent changed while you were answering. Review the current question before using your saved draft.</p><button className="agents-secondary" type="button" disabled={busy} onClick={onUseLatest}>Use my answer with this revision</button></div> : null}
      <button className="agents-primary" type="submit" disabled={busy || contextBlocked || changed || bytes === 0 || bytes > 8000}>{busy ? "Saving answer…" : "Answer and continue"}</button>
    </form>
  </section>;
}

function AgentTaskControls({ task, busy, onAction, onDelete }: {
  task: AgentTask;
  busy: boolean;
  onAction: (action: "pause" | "resume" | "cancel" | "rerun") => void;
  onDelete: () => void;
}) {
  const [confirmingDelete, setConfirmingDelete] = useState(false);

  useEffect(() => {
    const resetConfirmation = () => setConfirmingDelete(false);
    const onVisibilityChange = () => {
      if (document.visibilityState === "hidden") resetConfirmation();
    };
    window.addEventListener("blur", resetConfirmation);
    document.addEventListener("visibilitychange", onVisibilityChange);
    return () => {
      window.removeEventListener("blur", resetConfirmation);
      document.removeEventListener("visibilitychange", onVisibilityChange);
    };
  }, []);

  return (
    <div className="agents-task-actions" role="group" aria-label="Agent controls">
      {isMoving(task) ? <button type="button" className="agents-secondary" disabled={busy} onClick={() => onAction("pause")}>Pause</button> : null}
      {task.status === "paused" ? <button type="button" className="agents-primary" disabled={busy || agentContextBlocked(task)} onClick={() => onAction("resume")}>Resume</button> : null}
      {!isFinished(task) ? <button type="button" className="agents-secondary" disabled={busy} onClick={() => onAction("cancel")}>Cancel</button> : null}
      {isFinished(task) ? <>
        <button type="button" className="agents-primary" disabled={busy || agentContextBlocked(task)} onClick={() => { setConfirmingDelete(false); onAction("rerun"); }}>Run again</button>
        <button
          type="button"
          className={`agents-secondary danger-button${confirmingDelete ? " confirming-delete" : ""}`}
          aria-label={confirmingDelete ? "Confirm delete task" : "Delete task"}
          aria-pressed={confirmingDelete}
          title={confirmingDelete ? "Click again to delete this saved task and its result. Created files stay in My Files." : "Delete saved task and result. Created files stay in My Files."}
          disabled={busy}
          onClick={() => {
            if (confirmingDelete) {
              setConfirmingDelete(false);
              onDelete();
            } else setConfirmingDelete(true);
          }}
          onBlur={() => setConfirmingDelete(false)}
          onKeyDown={(event) => {
            if (event.key === "Escape") setConfirmingDelete(false);
          }}
        >{confirmingDelete ? "✓" : "Delete task"}</button>
      </> : null}
    </div>
  );
}

function AgentMarkdown({ text }: { text: string }) {
  return <div className="markdown-content agents-markdown"><Markdown remarkPlugins={[remarkGfm]} skipHtml components={{
    a: ({ href, children }) => <a href={href} target="_blank" rel="noopener noreferrer">{children}</a>,
    pre: ({ children }) => {
      const code = Children.toArray(children).find((child) => isValidElement(child));
      const language = isValidElement<{ className?: string }>(code)
        ? code.props.className?.match(/language-(\S+)/)?.[1]?.slice(0, 40) : undefined;
      return <div className="agents-code-block" data-language={language}>
        <div className="agents-code-header">{language || "Text"}</div>
        <pre tabIndex={0} aria-label={`${language || "Text"} code block`}>{children}</pre>
      </div>;
    },
    // A source can contain tracking images; reports show text and source links.
    img: ({ alt }) => <span>{alt}</span>,
  }}>{text}</Markdown></div>;
}
