import { useEffect, useId, useLayoutEffect, useMemo, useRef, useState } from "react";

import { ApiError, listAgentEvents, type AgentEvent, type AgentTask } from "./api";

const STAGES = { plan: "Plan", research: "Work", review: "Review", title: "Title" };
const TOOLS = { web_search: "Search web", web_fetch: "Read web page", create_file: "Create file", read_selected_file: "Read selected file" };
const STATES = { started: "Started", completed: "Completed", failed: "Failed", interrupted: "Interrupted" };

function mergeEvents(current: AgentEvent[], incoming: AgentEvent[], replace: boolean): AgentEvent[] {
  const saved = new Map(current.map((event) => [event.seq, event]));
  const merged = new Map(replace ? [] : saved);
  for (const event of incoming) {
    const previous = saved.get(event.seq);
    // A late snapshot cannot turn a finished operation back into a running one.
    merged.set(event.seq, previous && previous.state !== "started" ? previous : event);
  }
  return [...merged.values()].sort((a, b) => b.seq - a.seq).slice(0, 200);
}

function timing(event: AgentEvent): string {
  const parts: string[] = [];
  if (event.duration_ms !== null) parts.push(event.duration_ms < 1000 ? `${event.duration_ms} ms` : `${(event.duration_ms / 1000).toFixed(1)} s`);
  if (event.input_tokens !== null) parts.push(`${event.input_tokens.toLocaleString()} input tokens`);
  if (event.output_tokens !== null) parts.push(`${event.output_tokens.toLocaleString()} output tokens`);
  return parts.join(" · ");
}

export function AgentRecentActivity({ task, visible, timezone }: { task: Pick<AgentTask, "id" | "status" | "revision">; visible: boolean; timezone?: string }) {
  const [open, setOpen] = useState(true);
  const [expanded, setExpanded] = useState(false);
  const [pageVisible, setPageVisible] = useState(document.visibilityState !== "hidden");
  const [retry, setRetry] = useState(0);
  const session = useMemo(() => ({ taskId: task.id, ready: visible && open && pageVisible }), [task.id, visible, open, pageVisible]);
  const [snapshot, setSnapshot] = useState<{ session: typeof session; items: AgentEvent[]; hasMore: boolean; truncated: boolean }>();
  const [error, setError] = useState("");
  const [loadingOlder, setLoadingOlder] = useState(false);
  const [loading, setLoading] = useState(true);
  const live = useRef(session);
  const generation = useRef(0);
  const historyRequest = useRef<AbortController | null>(null);
  const list = useRef<HTMLOListElement | null>(null);
  const anchor = useRef<{ seq: string; top: number } | null>(null);
  const contentId = useId();
  const rows = snapshot?.session === session ? snapshot.items : [];
  const hasMore = snapshot?.session === session && snapshot.hasMore;
  const truncated = snapshot?.session === session && snapshot.truncated;
  const active = task.status === "queued" || task.status === "running";
  const time = (value: string) => new Date(value).toLocaleString(undefined, { timeZone: timezone, month: "short", day: "numeric", hour: "numeric", minute: "2-digit", second: "2-digit" });

  function rememberPosition() {
    const node = list.current;
    if (!node || node.scrollTop === 0) { anchor.current = null; return; }
    const top = node.getBoundingClientRect().top;
    const item = Array.from(node.children).find((child) => child.getBoundingClientRect().bottom > top);
    anchor.current = item ? { seq: item.getAttribute("data-seq") ?? "", top: item.getBoundingClientRect().top } : null;
  }
  useLayoutEffect(() => {
    if (anchor.current && list.current) {
      const item = Array.from(list.current.children).find((child) => child.getAttribute("data-seq") === anchor.current?.seq);
      if (item) list.current.scrollTop += item.getBoundingClientRect().top - anchor.current.top;
    }
    anchor.current = null;
  }, [snapshot]);
  useEffect(() => {
    const changed = () => setPageVisible(document.visibilityState !== "hidden");
    document.addEventListener("visibilitychange", changed);
    return () => document.removeEventListener("visibilitychange", changed);
  }, []);
  useEffect(() => {
    live.current = session;
    const epoch = ++generation.current;
    if (!session.ready) return;
    let request: AbortController | null = null;
    async function refresh() {
      if (request || document.visibilityState === "hidden") return;
      const controller = new AbortController(); request = controller;
      try {
        const page = await listAgentEvents(task.id, controller.signal);
        if (controller.signal.aborted || epoch !== generation.current || live.current !== session) return;
        rememberPosition();
        setSnapshot((saved) => ({ session,
          items: mergeEvents(saved?.session === session ? saved.items : [], page.items, !page.has_more),
          hasMore: saved?.session === session && saved.items.length > 50 ? saved.hasMore && page.has_more : page.has_more,
          truncated: page.truncated }));
        setError("");
      } catch (reason) {
        if (controller.signal.aborted || epoch !== generation.current || live.current !== session) return;
        historyRequest.current?.abort();
        setSnapshot(undefined);
        setError(reason instanceof ApiError && reason.status === 404
          ? "This agent’s activity is no longer available."
          : "Could not load this agent’s activity. Try again.");
      } finally {
        if (epoch === generation.current && !controller.signal.aborted) setLoading(false);
        if (request === controller) request = null;
      }
    }
    queueMicrotask(() => { if (epoch === generation.current) { setLoading(true); setLoadingOlder(false); setError(""); void refresh(); } });
    const timer = active ? window.setInterval(() => void refresh(), 3000) : null;
    return () => {
      generation.current += 1;
      request?.abort(); historyRequest.current?.abort();
      if (timer !== null) window.clearInterval(timer);
    };
  }, [session, task.id, task.revision, task.status, active, retry]);

  async function older() {
    const before = rows.at(-1)?.seq;
    if (!before || historyRequest.current && !historyRequest.current.signal.aborted) return;
    const controller = new AbortController(); historyRequest.current = controller;
    const epoch = generation.current;
    setLoadingOlder(true); setError("");
    try {
      const page = await listAgentEvents(task.id, controller.signal, before);
      if (controller.signal.aborted || generation.current !== epoch || live.current !== session) return;
      rememberPosition();
      setSnapshot((saved) => saved?.session === session ? { session, items: mergeEvents(saved.items, page.items, false), hasMore: page.has_more, truncated: page.truncated } : saved);
    } catch (reason) {
      if (controller.signal.aborted || generation.current !== epoch || live.current !== session) return;
      if (reason instanceof ApiError && reason.status === 404) setSnapshot(undefined);
      setError(reason instanceof ApiError && reason.status === 404 ? "This agent’s activity is no longer available." : "Could not load older activity. Try again.");
    } finally {
      if (historyRequest.current === controller) {
        historyRequest.current = null;
        if (!controller.signal.aborted) setLoadingOlder(false);
      }
    }
  }

  return <section className="agents-recent-activity" aria-label="Recent activity">
    <h3><button type="button" className="agents-activity-toggle" aria-expanded={open} aria-controls={contentId} onClick={() => setOpen((value) => !value)}>
      <span aria-hidden="true">{open ? "⌄" : "›"}</span>Recent activity
    </button></h3>
    {open ? <div id={contentId}>
      {session.ready && error ? <p className="agents-event-error" role="alert">{error} <button type="button" className="agents-secondary" onClick={() => setRetry((value) => value + 1)}>Retry activity</button></p> : null}
      {!rows.length ? !error || loading ? <p className="agents-event-empty">{!session.ready || loading ? "Loading activity…" : "No saved activity yet. Older tasks may not have activity history."}</p> : null : <>
        <ol className="agents-event-list" aria-label="Agent operations" ref={list} tabIndex={expanded ? 0 : undefined}>
          {(expanded ? rows : rows.slice(0, 6)).map((event) => <li key={event.seq} data-seq={event.seq} className={`agents-event-${event.state}`}>
            <div className="agents-event-line"><strong>{event.kind === "tool" && event.tool ? TOOLS[event.tool] : event.kind === "model" ? "Local model" : `${STAGES[event.stage]} stage`}</strong><span>{STATES[event.state]}</span></div>
            <div className="agents-event-meta">{STAGES[event.stage]} · Attempt {event.attempt} · <time dateTime={event.created_at}>{time(event.created_at)}</time></div>
            {timing(event) ? <p className="agents-event-usage">{timing(event)}</p> : null}
            {event.state === "interrupted" ? <p className="agents-event-usage">The outcome was not recorded.</p> : null}
          </li>)}
        </ol>
        <div className="agents-event-actions">
          {rows.length > 6 || hasMore ? <button type="button" className="agents-secondary" onClick={() => setExpanded((value) => !value)}>{expanded ? "Show recent only" : "Show more activity"}</button> : null}
          {expanded && hasMore && rows.length < 200 ? <button type="button" className="agents-secondary" disabled={loadingOlder} onClick={() => void older()}>{loadingOlder ? "Loading older activity…" : "Load older activity"}</button> : null}
        </div>
      </>}
      {truncated ? <p className="agents-event-empty">Older activity has been removed. Up to 200 operations are retained.</p> : null}
    </div> : null}
  </section>;
}
