import { useCallback, useEffect, useMemo, useRef, useState, type FormEvent } from "react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";

import {
  actOnAgentTask,
  ApiError,
  createAgentTask,
  deleteAgentTask,
  getAgentTask,
  getActivityUnreadCount,
  listAgentTasks,
  type AgentTask,
  type AgentTaskStatus,
  type PublicWatch,
  type WatchRun,
} from "./api";
import "./AgentsWorkspace.css";
import { AgentKnowledgeDialog } from "./AgentKnowledgeDialog";
import { WatchesWorkspace } from "./WatchesWorkspace";
import { ActivityWorkspace } from "./ActivityWorkspace";

const STATUS_LABELS: Record<AgentTaskStatus, string> = {
  queued: "Queued", running: "Running", paused: "Paused", completed: "Completed",
  cancelled: "Cancelled", failed: "Failed",
};

function isMoving(task: AgentTask) {
  return task.status === "queued" || task.status === "running";
}

function agentTitle(task: AgentTask) {
  return task.title?.trim() || task.objective;
}

function isFinished(task: AgentTask) {
  return ["completed", "cancelled", "failed"].includes(task.status);
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
}

export function AgentsWorkspace(props: AgentsWorkspaceProps) {
  const [view, setView] = useState<"tasks" | "watches" | "activity">("tasks");
  const [watchesOpened, setWatchesOpened] = useState(false);
  const [activityOpened, setActivityOpened] = useState(false);
  const [agentRequest, setAgentRequest] = useState<{ task: AgentTask; serial: number }>();
  const [watchRequest, setWatchRequest] = useState<{ watch: PublicWatch; run: WatchRun; serial: number }>();
  const requestSerial = useRef(0);
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
  const [knowledgeTask, setKnowledgeTask] = useState<AgentTask | null>(null);
  const [filter, setFilter] = useState<"all" | "active" | "finished">("all");
  const deletedIds = useRef(new Set<string>());
  const knownIds = useRef<Set<string> | null>(null);
  const mutationGeneration = useRef(0);
  const busyRef = useRef(false);
  const objectiveRef = useRef<HTMLTextAreaElement>(null);
  const detailRef = useRef<HTMLElement>(null);
  const sidebarRef = useRef<HTMLElement>(null);
  const sidebarToggleRef = useRef<HTMLButtonElement>(null);
  const createRequest = useRef<{ key: string; requestId: string } | null>(null);
  const mounted = useRef(false);
  const openedRequest = useRef<number | null>(null);
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
    if (!sidebarOpen) return;
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
  }, [closeSidebar, sidebarOpen]);

  useEffect(() => {
    mounted.current = true;
    return () => { mounted.current = false; };
  }, []);

  useEffect(() => { onActiveCountChange?.(activeCount); }, [activeCount, onActiveCountChange]);
  useEffect(() => { onTasksChange(tasks); }, [tasks, onTasksChange]);

  useEffect(() => {
    if (!visible || !openRequest || openedRequest.current === openRequest.serial) return;
    openedRequest.current = openRequest.serial;
    mutationGeneration.current += 1;
    knownIds.current?.add(openRequest.task.id);
    // The Activity reader has resolved the exact owned terminal revision first.
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

  async function startTask(text: string, key: string) {
    if (busyRef.current || !text.trim()) return;
    busyRef.current = true;
    mutationGeneration.current += 1;
    setBusy(true);
    setError("");
    setNotice("");
    // Retain this identity after a lost response so Retry does not create duplicates.
    try {
      if (createRequest.current?.key !== key) {
        createRequest.current = { key, requestId: newRequestId() };
      }
      const task = await createAgentTask(text.trim(), createRequest.current.requestId);
      if (!mounted.current) return;
      createRequest.current = null;
      knownIds.current?.add(task.id);
      setTasks((current) => mergeTasks(current, [task], deletedIds.current));
      setSelectedId(task.id);
      showWorkflow();
      setFilter("all");
      if (key.startsWith("new:")) setObjective("");
    } catch (reason) {
      if (mounted.current) setError(messageOf(reason));
    } finally {
      busyRef.current = false;
      if (mounted.current) { setBusy(false); setRefreshToken((value) => value + 1); }
    }
  }

  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    void startTask(objective, `new:${objective.trim()}`);
  }

  async function act(task: AgentTask, action: "pause" | "resume" | "cancel" | "rerun") {
    if (busyRef.current) return;
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

  const selected = tasks.find((task) => task.id === selectedId);
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
      <div className="agents-columns">
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
              <p className="agents-eyebrow">Your research, in progress</p>
              <h1 id="agents-title">Agents</h1>
            </div>
            {compactNavigation
              ? <button type="button" className="agents-secondary agents-sidebar-close" aria-label="Close agent controls" onClick={closeSidebar}>✕</button>
              : <button type="button" className="agents-secondary" onClick={() => objectiveRef.current?.focus()}>New agent</button>}
          </header>
          <div className="agents-sidebar-scroll">
            {sidebarOpen && error ? <p className="agents-error" role="alert">{error}</p> : null}
            {sidebarOpen && notice ? <p className="agents-notice" role="status">{notice}</p> : null}
            <form className="agents-composer" onSubmit={submit}>
              <label htmlFor="agent-objective">Research objective</label>
              <textarea
                id="agent-objective"
                ref={objectiveRef}
                rows={4}
                maxLength={6000}
                value={objective}
                onChange={(event) => setObjective(event.target.value)}
                placeholder="Compare the current public documentation for…"
                aria-describedby="agent-objective-help"
                required
                disabled={busy}
              />
              <p id="agent-objective-help">All agent steps use local models with no cloud fallback. Research uses public web sources. You can leave this page and return to the saved progress and report.</p>
              <button className="agents-primary" type="submit" disabled={busy || !objective.trim()}>{busy ? "Saving…" : "Start agent"}</button>
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
              <p className="agents-empty">{tasks.length ? "No tasks in this view." : "No agents yet. Start with a research objective above."}</p>
            ) : (
              <ul className="agents-task-list">
                {shownTasks.map((task) => (
                  <li key={task.id}>
                    <button type="button" className="agents-task-card" aria-pressed={selectedId === task.id} aria-label={`Open agent: ${agentTitle(task)}`} onClick={() => selectTask(task.id)}>
                      <span className={`agents-status agents-status-${task.status}`}>{STATUS_LABELS[task.status]}</span>
                      <strong>{agentTitle(task)}</strong>
                      <span className="agents-task-stage">{task.steps.find((step) => step.status === "running")?.label ?? (task.status === "paused" ? "Completed steps saved" : task.status === "queued" ? "Waiting to start" : "Saved progress")}</span>
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
                    <span className={`agents-status agents-status-${selected.status}`}>{STATUS_LABELS[selected.status]}</span>
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
                {selected.status === "paused" ? <p className="agents-notice">Completed steps are saved. Resume to continue from the next unfinished step.</p> : null}
                {selected.error ? <p className="agents-error" role="alert">{selected.error}</p> : null}
                <section className="agents-plan" aria-labelledby="agent-plan-title">
                  <h3 id="agent-plan-title">Plan and progress</h3>
                  <ol>
                    {selected.steps.map((step) => (
                      <li key={step.id} className={`agents-step agents-step-${step.status}`}>
                        <span className="agents-step-mark" aria-hidden="true">{step.status === "completed" ? "✓" : step.status === "running" ? "●" : "○"}</span>
                        <div>
                          <strong>{step.label}</strong>
                          <span className="agents-step-status">{step.status === "completed" ? "Completed" : step.status === "running" ? "In progress" : "Pending"}</span>
                          {step.result ? <details><summary>View saved {step.id === "plan" ? "plan" : step.id === "research" ? "research" : "review"}</summary><AgentMarkdown text={step.result} /></details> : null}
                        </div>
                      </li>
                    ))}
                  </ol>
                </section>
                <section className="agents-result" aria-labelledby="agent-result-title">
                  <h3 id="agent-result-title">Report</h3>
                  {selected.status === "completed" && selected.result ? <button className="agents-secondary" type="button" onClick={() => setKnowledgeTask(selected)}>Propose Project knowledge</button> : null}
                  {selected.result ? <AgentMarkdown text={selected.result} /> : <p>{selected.summary && selected.status === "completed" ? "Loading the saved report…" : selected.status === "failed" ? "This task stopped before a final report was saved. Any completed steps are available above." : selected.status === "cancelled" ? "This task was cancelled. Any completed steps are available above." : "The report will appear here when Audrey completes its review."}</p>}
                </section>
              </>
            ) : (
              <div className="agents-welcome">
                <span className="agents-welcome-mark" aria-hidden="true">✦</span>
                <h2>Work that continues while you’re away</h2>
                <p>Start a research task or select a saved agent to see its objective, completed steps, and report.</p>
                <ol><li>Plan the work</li><li>Research public sources</li><li>Review and save the report</li></ol>
              </div>
            )}
          </div>
        </section>
      </div>
      {knowledgeTask && visible ? <AgentKnowledgeDialog task={knowledgeTask} onClose={() => setKnowledgeTask(null)} /> : null}
    </section>
  );
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
      {task.status === "paused" ? <button type="button" className="agents-primary" disabled={busy} onClick={() => onAction("resume")}>Resume</button> : null}
      {!isFinished(task) ? <button type="button" className="agents-secondary" disabled={busy} onClick={() => onAction("cancel")}>Cancel</button> : null}
      {isFinished(task) ? <>
        <button type="button" className="agents-primary" disabled={busy} onClick={() => { setConfirmingDelete(false); onAction("rerun"); }}>Run again</button>
        <button
          type="button"
          className={`agents-secondary danger-button${confirmingDelete ? " confirming-delete" : ""}`}
          aria-label={confirmingDelete ? "Confirm delete task" : "Delete task"}
          aria-pressed={confirmingDelete}
          title={confirmingDelete ? "Click again to delete this saved task and its report." : "Delete saved task and report"}
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
    // A source can contain tracking images; reports show text and source links.
    img: ({ alt }) => <span>{alt}</span>,
  }}>{text}</Markdown></div>;
}
