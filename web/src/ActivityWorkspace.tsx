import { useEffect, useMemo, useRef, useState } from "react";
import {
  ApiError, getActivityAgentTask, getWatch, getWatchRun, listActivity, readActivity,
  type ActivityNotice, type AgentTask, type PublicWatch, type WatchRun,
} from "./api";
import "./ActivityWorkspace.css";

function label(item: ActivityNotice) {
  if (item.origin === "agent") return item.kind === "completed" ? "Agent completed" : "Agent failed";
  return item.kind === "change" ? "Relevant change" : "Watch failed";
}

export function ActivityWorkspace({ visible, timezone, onUnreadChange, onOpenAgent, onOpenWatch, isAgentResultCurrent }: {
  visible: boolean;
  timezone?: string;
  onUnreadChange: (count: number) => void;
  onOpenAgent: (task: AgentTask) => void;
  onOpenWatch: (watch: PublicWatch, run: WatchRun) => void;
  isAgentResultCurrent?: (task: AgentTask) => boolean;
}) {
  const [items, setItems] = useState<ActivityNotice[]>([]);
  const [loading, setLoading] = useState(true);
  const [opening, setOpening] = useState<string | null>(null);
  const [error, setError] = useState("");
  const [token, setToken] = useState(0);
  const unread = useRef(0);
  const generation = useRef(0);
  const targetRequest = useRef<AbortController | null>(null);
  const visibleRef = useRef(visible);
  const formatter = useMemo(() => {
    try { return new Intl.DateTimeFormat(undefined, { dateStyle: "medium", timeStyle: "short", timeZone: timezone }); }
    catch { return new Intl.DateTimeFormat(undefined, { dateStyle: "medium", timeStyle: "short" }); }
  }, [timezone]);

  useEffect(() => {
    visibleRef.current = visible;
    return () => { visibleRef.current = false; targetRequest.current?.abort(); };
  }, [visible]);

  useEffect(() => {
    if (!visible) return;
    const controller = new AbortController();
    let inFlight = false;
    async function refresh() {
      if (inFlight || targetRequest.current || controller.signal.aborted || document.visibilityState === "hidden") return;
      inFlight = true;
      const started = generation.current;
      try {
        const result = await listActivity(controller.signal);
        if (controller.signal.aborted || started !== generation.current) return;
        setItems(result.items); unread.current = result.unread_count; onUnreadChange(result.unread_count);
      } catch (reason) {
        if (!controller.signal.aborted) setError(reason instanceof Error ? reason.message : "Could not load Activity.");
      } finally { inFlight = false; if (!controller.signal.aborted) setLoading(false); }
    }
    void refresh();
    const timer = window.setInterval(() => void refresh(), 15000);
    const onVisible = () => { void refresh(); };
    window.addEventListener("focus", onVisible);
    document.addEventListener("visibilitychange", onVisible);
    return () => {
      controller.abort(); window.clearInterval(timer);
      window.removeEventListener("focus", onVisible);
      document.removeEventListener("visibilitychange", onVisible);
    };
  }, [visible, token, onUnreadChange]);

  async function open(item: ActivityNotice) {
    targetRequest.current?.abort();
    const controller = new AbortController(); targetRequest.current = controller;
    generation.current += 1;
    setOpening(`${item.origin}:${item.id}`); setError("");
    const isCurrent = () => !controller.signal.aborted && visibleRef.current && document.visibilityState !== "hidden";
    try {
      let openTarget: () => void;
      if (item.origin === "agent" && item.target.kind === "agent") {
        const task = await getActivityAgentTask(item.id, controller.signal);
        if (!isCurrent()) return;
        if (task.id !== item.target.task_id || task.revision !== item.target.revision || !["completed", "failed"].includes(task.status)) {
          throw new ApiError(409, "Agent has been run again; refresh Activity.");
        }
        if (isAgentResultCurrent && !isAgentResultCurrent(task)) throw new ApiError(409, "Agent has been run again; refresh Activity.");
        openTarget = () => {
          if (isAgentResultCurrent && !isAgentResultCurrent(task)) throw new ApiError(409, "Agent has been run again; refresh Activity.");
          onOpenAgent(task);
        };
      } else if (item.origin === "watch" && item.target.kind === "watch") {
        const [watch, run] = await Promise.all([
          getWatch(item.target.watch_id, controller.signal), getWatchRun(item.target.watch_id, item.target.run_id, controller.signal),
        ]);
        if (!isCurrent()) return;
        if (watch.id !== item.target.watch_id || run.id !== item.target.run_id || run.watch_id !== watch.id) {
          throw new ApiError(404, "The saved check is no longer available. Refresh Activity.");
        }
        openTarget = () => onOpenWatch(watch, run);
      } else throw new ApiError(404, "This saved activity is no longer available. Refresh Activity.");
      if (!item.read_at) {
        const saved = await readActivity(item.origin, item.id, controller.signal);
        if (!isCurrent()) return;
        generation.current += 1;
        setItems((current) => current.map((entry) => entry.id === item.id && entry.origin === item.origin ? saved : entry));
        unread.current = Math.max(0, unread.current - 1); onUnreadChange(unread.current);
      }
      if (isCurrent()) openTarget();
    } catch (reason) {
      if (!isCurrent()) return;
      if (reason instanceof ApiError && (reason.status === 404 || reason.status === 409)) {
        setError(item.origin === "agent" ? "This agent result is no longer current. It may have been run again or deleted; refresh Activity." : "This watch or saved check is no longer available. Refresh Activity.");
      } else setError(reason instanceof Error ? reason.message : "Could not open this saved activity.");
    } finally { if (targetRequest.current === controller) { targetRequest.current = null; setOpening(null); } }
  }

  useEffect(() => {
    const onHidden = () => { if (document.visibilityState === "hidden") targetRequest.current?.abort(); };
    document.addEventListener("visibilitychange", onHidden);
    return () => document.removeEventListener("visibilitychange", onHidden);
  }, []);

  return <section className="activity-workspace" aria-labelledby="activity-title">
    <header><h1 id="activity-title">Activity</h1><p>Completed agents, agent failures, relevant page changes and persistent watch failures. Open an item to view its saved result.</p><p>Unread activity appears first. Showing up to 50 current items.</p><button className="agents-secondary" type="button" onClick={() => { setError(""); setToken((value) => value + 1); }}>Refresh activity</button></header>
    {error ? <p className="agents-error" role="alert">{error}</p> : null}
    {loading ? <p role="status">Loading activity…</p> : !items.length ? <p className="agents-empty">No activity yet. Agent results and watch alerts will appear here.</p> : <ul>{items.map((item) => {
      const key = `${item.origin}:${item.id}`;
      const date = new Date(item.created_at);
      return <li key={key}><button type="button" className="activity-notice" data-unread={!item.read_at} disabled={opening === key} onClick={() => void open(item)}>
        <span>{item.read_at ? "Read" : "Unread"} · {label(item)}</span><strong>{item.title}</strong><p>{item.body}</p><time dateTime={item.created_at}>{Number.isNaN(date.getTime()) ? "Time unavailable" : formatter.format(date)}</time><span>{opening === key ? "Opening saved result…" : item.origin === "agent" ? "Open agent result →" : "Open saved check →"}</span>
      </button></li>;
    })}</ul>}
  </section>;
}
