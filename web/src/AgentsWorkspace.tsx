import { useEffect, useMemo, useRef, useState, type FormEvent } from "react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";

import {
  actOnAgentTask,
  ApiError,
  createAgentTask,
  deleteAgentTask,
  getAgentTask,
  listAgentTasks,
  type AgentTask,
  type AgentTaskStatus,
} from "./api";
import "./AgentsWorkspace.css";

const STATUS_LABELS: Record<AgentTaskStatus, string> = {
  queued: "Queued", running: "Running", paused: "Paused", completed: "Completed",
  cancelled: "Cancelled", failed: "Failed",
};

function isMoving(task: AgentTask) {
  return task.status === "queued" || task.status === "running";
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

export function AgentsWorkspace({
  visible = true,
  timezone,
  onActiveCountChange,
}: {
  visible?: boolean;
  timezone?: string;
  onActiveCountChange?: (count: number) => void;
}) {
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
  const [filter, setFilter] = useState<"all" | "active" | "finished">("all");
  const [confirmingDeleteId, setConfirmingDeleteId] = useState<string | null>(null);
  const deletedIds = useRef(new Set<string>());
  const knownIds = useRef<Set<string> | null>(null);
  const mutationGeneration = useRef(0);
  const busyRef = useRef(false);
  const objectiveRef = useRef<HTMLTextAreaElement>(null);
  const detailRef = useRef<HTMLElement>(null);
  const createRequest = useRef<{ key: string; requestId: string } | null>(null);
  const mounted = useRef(false);
  const activeCount = tasks.filter(isMoving).length;
  const hasActive = activeCount > 0;

  useEffect(() => {
    mounted.current = true;
    return () => { mounted.current = false; };
  }, []);

  useEffect(() => { onActiveCountChange?.(activeCount); }, [activeCount, onActiveCountChange]);

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

  function scrollToDetail() {
    if (visible && window.innerWidth <= 760) {
      window.requestAnimationFrame(() => detailRef.current?.scrollIntoView?.({ block: "start" }));
    }
  }

  function selectTask(id: string) {
    setSelectedId(id);
    setConfirmingDeleteId(null);
    setNotice("");
    scrollToDetail();
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
      scrollToDetail();
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

  async function act(task: AgentTask, action: "pause" | "resume" | "cancel") {
    if (busyRef.current) return;
    busyRef.current = true;
    mutationGeneration.current += 1;
    setBusy(true);
    setError("");
    setNotice("");
    try {
      const updated = await actOnAgentTask(task.id, action, task.revision);
      if (mounted.current) setTasks((current) => mergeTasks(current, [updated], deletedIds.current));
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
      setConfirmingDeleteId(null);
    } catch (reason) {
      if (!mounted.current) return;
      if (reason instanceof ApiError && reason.status === 409) {
        setConfirmingDeleteId(null);
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
    <section className="agents-workspace" aria-labelledby="agents-title">
      <header className="agents-header">
        <div>
          <p className="agents-eyebrow">Your research, in progress</p>
          <h1 id="agents-title">Agents</h1>
          <p>Give Audrey a research objective. It will plan, research public sources, and review a saved report.</p>
        </div>
        <button type="button" className="agents-secondary" onClick={() => objectiveRef.current?.focus()}>New agent</button>
      </header>

      {error ? <p className="agents-error" role="alert">{error}</p> : null}
      {notice ? <p className="agents-notice" role="status">{notice}</p> : null}

      <div className="agents-columns">
        <aside className="agents-sidebar" aria-label="Agent tasks">
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
            <p id="agent-objective-help">All agent steps use local Qwen 3.8 with no cloud fallback. Research uses public web sources. You can leave this page and return to the saved progress and report.</p>
            <button className="agents-primary" type="submit" disabled={busy || !objective.trim()}>{busy ? "Saving…" : "Start agent"}</button>
          </form>

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
                  <button type="button" className="agents-task-card" aria-pressed={selectedId === task.id} aria-label={`Open agent: ${task.objective}`} onClick={() => selectTask(task.id)}>
                    <span className={`agents-status agents-status-${task.status}`}>{STATUS_LABELS[task.status]}</span>
                    <strong>{task.objective}</strong>
                    <span className="agents-task-stage">{task.steps.find((step) => step.status === "running")?.label ?? (task.status === "paused" ? "Completed steps saved" : task.status === "queued" ? "Waiting to start" : "Saved progress")}</span>
                    <time dateTime={task.updated_at}>Last activity {timestamp(task.updated_at)}</time>
                  </button>
                </li>
              ))}
            </ul>
          )}
        </aside>

        <section className="agents-detail" aria-label="Selected agent" ref={detailRef}>
          {selected ? (
            <>
              <header className="agents-detail-header">
                <span className={`agents-status agents-status-${selected.status}`}>{STATUS_LABELS[selected.status]}</span>
                <h2>{selected.objective}</h2>
                <p>Last activity <time dateTime={selected.updated_at}>{timestamp(selected.updated_at)}</time></p>
                <div className="agents-task-actions" role="group" aria-label="Agent controls">
                  {isMoving(selected) ? <button type="button" className="agents-secondary" disabled={busy} onClick={() => void act(selected, "pause")}>Pause</button> : null}
                  {selected.status === "paused" ? <button type="button" className="agents-primary" disabled={busy} onClick={() => void act(selected, "resume")}>Resume</button> : null}
                  {!isFinished(selected) ? <button type="button" className="agents-secondary" disabled={busy} onClick={() => void act(selected, "cancel")}>Cancel</button> : null}
                  {isFinished(selected) ? <button type="button" className="agents-primary" disabled={busy} onClick={() => void startTask(selected.objective, `rerun:${selected.id}`)}>Run again</button> : null}
                  {isFinished(selected) && confirmingDeleteId !== selected.id ? <button type="button" className="agents-secondary" disabled={busy} onClick={() => setConfirmingDeleteId(selected.id)}>Delete task</button> : null}
                </div>
                {confirmingDeleteId === selected.id ? <div className="agents-delete-confirm" role="group" aria-label="Confirm task deletion">
                  <p>Delete this saved task and its report? This cannot be undone.</p>
                  <button type="button" className="agents-secondary agents-danger" disabled={busy} onClick={() => void removeTask(selected)}>Confirm delete task</button>
                  <button type="button" className="agents-secondary" disabled={busy} onClick={() => setConfirmingDeleteId(null)}>Keep task</button>
                </div> : null}
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
        </section>
      </div>
    </section>
  );
}

function AgentMarkdown({ text }: { text: string }) {
  return <div className="markdown-content agents-markdown"><Markdown remarkPlugins={[remarkGfm]} skipHtml components={{
    a: ({ href, children }) => <a href={href} target="_blank" rel="noopener noreferrer">{children}</a>,
    // A source can contain tracking images; reports show text and source links.
    img: ({ alt }) => <span>{alt}</span>,
  }}>{text}</Markdown></div>;
}
