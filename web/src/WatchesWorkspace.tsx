import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  ApiError, actOnWatch, createWatch, deleteWatch, getWatch, getWatchRun,
  listWatchRuns, listWatches, updateWatch,
  type PublicWatch, type WatchDefinition, type WatchRun, type WatchRunSummary,
} from "./api";
import { WatchForm } from "./WatchForm";
import "./WatchesWorkspace.css";

function messageOf(reason: unknown) { return reason instanceof Error ? reason.message : "The request could not be completed."; }
function runLabel(run: WatchRunSummary) {
  if (run.status === "completed") return run.outcome === "baseline" ? "Baseline saved" : run.outcome === "unchanged" ? "Unchanged" : run.outcome === "changed" ? "Changed" : "Completed";
  return { queued: "Queued", running: "Checking", failed: "Failed", cancelled: "Cancelled" }[run.status];
}
function mergeWatch(current: PublicWatch[], watch: PublicWatch) {
  const saved = current.find((item) => item.id === watch.id);
  if (saved && saved.revision > watch.revision) return current;
  return [watch, ...current.filter((item) => item.id !== watch.id)].sort((a, b) => b.created_at.localeCompare(a.created_at));
}
function scheduleLabel(watch: Pick<PublicWatch, "schedule">) {
  const schedule = watch.schedule;
  const cadence = schedule.kind === "interval" ? `Every ${schedule.interval_minutes} minutes` : schedule.kind === "daily" ? `Daily at ${schedule.time}` : `${["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][schedule.weekday ?? 0]} at ${schedule.time}`;
  return `${cadence} · ${schedule.timezone}`;
}

export function WatchesWorkspace({ visible, timezone, navigationOpen, onNavigationOpenChange, openRequest }: {
  visible: boolean;
  timezone?: string;
  navigationOpen?: boolean;
  onNavigationOpenChange?: (open: boolean) => void;
  openRequest?: { watch: PublicWatch; run: WatchRun; serial: number };
}) {
  const [watches, setWatches] = useState<PublicWatch[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [runs, setRuns] = useState<WatchRun[]>([]);
  const [run, setRun] = useState<WatchRun | null>(null);
  const [runId, setRunId] = useState<string | null>(null);
  const [editing, setEditing] = useState<PublicWatch | "new" | null>(null);
  const [conflict, setConflict] = useState<PublicWatch>();
  const [loading, setLoading] = useState(true);
  const [loadingRun, setLoadingRun] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [token, setToken] = useState(0);
  const [drawerOpen, setInternalDrawerOpen] = useState(false);
  const [compact, setCompact] = useState(() => window.matchMedia?.("(max-width: 760px)").matches ?? false);
  const sidebar = useRef<HTMLElement>(null);
  const sidebarButton = useRef<HTMLButtonElement>(null);
  const detail = useRef<HTMLElement>(null);
  const mounted = useRef(false);
  const busyRef = useRef(false);
  const generation = useRef(0);
  const deleted = useRef(new Set<string>());
  const createRequest = useRef<{ key: string; id: string } | null>(null);
  const openedRequest = useRef<number | null>(null);
  const hasScheduled = watches.some((watch) => watch.enabled || watch.active_run_id);
  const selected = watches.find((watch) => watch.id === selectedId);
  const sidebarOpen = visible && compact && (navigationOpen ?? drawerOpen);
  const setDrawerOpen = useCallback((open: boolean) => {
    setInternalDrawerOpen(open);
    onNavigationOpenChange?.(open);
  }, [onNavigationOpenChange]);

  useEffect(() => {
    mounted.current = true;
    return () => { mounted.current = false; };
  }, []);

  const dateFormatter = useMemo(() => {
    try { return new Intl.DateTimeFormat(undefined, { dateStyle: "medium", timeStyle: "short", timeZone: timezone }); }
    catch { return new Intl.DateTimeFormat(undefined, { dateStyle: "medium", timeStyle: "short" }); }
  }, [timezone]);
  function timestamp(value: string | null) {
    if (!value) return "Not scheduled";
    const date = new Date(value);
    return Number.isNaN(date.getTime()) ? "Time unavailable" : dateFormatter.format(date);
  }

  useEffect(() => {
    if (!visible) return;
    const controller = new AbortController();
    let inFlight = false;
    async function refresh() {
      if (inFlight || controller.signal.aborted || document.visibilityState === "hidden") return;
      inFlight = true;
      const started = generation.current;
      try {
        const catalog = await listWatches(controller.signal);
        if (controller.signal.aborted || started !== generation.current || busyRef.current) return;
        const items = catalog.items.filter((watch) => !deleted.current.has(watch.id));
        setWatches((current) => items.reduce(mergeWatch, current.filter((watch) => items.some((item) => item.id === watch.id))));
        setSelectedId((current) => current && items.some((watch) => watch.id === current) ? current : items[0]?.id ?? null);
      } catch (reason) {
        if (!controller.signal.aborted) setError(`Could not refresh watches. ${messageOf(reason)}`);
      } finally {
        inFlight = false;
        if (!controller.signal.aborted) setLoading(false);
      }
    }
    void refresh();
    const timer = hasScheduled ? window.setInterval(() => void refresh(), 15000) : undefined;
    const onVisible = () => { void refresh(); };
    document.addEventListener("visibilitychange", onVisible);
    window.addEventListener("focus", onVisible);
    return () => {
      controller.abort();
      window.clearInterval(timer);
      document.removeEventListener("visibilitychange", onVisible);
      window.removeEventListener("focus", onVisible);
    };
  }, [visible, hasScheduled, token]);

  const activeRunId = selected?.active_run_id;
  const latestRunId = selected?.latest_run?.id;
  const selectedRevision = selected?.revision;
  useEffect(() => {
    if (!visible || !selectedId || editing) return;
    const controller = new AbortController();
    let inFlight = false;
    async function refreshHistory() {
      if (inFlight || controller.signal.aborted || document.visibilityState === "hidden") return;
      inFlight = true;
      try {
        const result = await listWatchRuns(selectedId!, controller.signal);
        if (!controller.signal.aborted) {
          setRuns(result.items);
          setRunId((current) => current ?? result.items[0]?.id ?? null);
        }
      } catch (reason) { if (!controller.signal.aborted) setError(messageOf(reason)); }
      finally { inFlight = false; }
    }
    void refreshHistory();
    const timer = activeRunId ? window.setInterval(() => void refreshHistory(), 3000) : undefined;
    return () => { controller.abort(); window.clearInterval(timer); };
  }, [visible, selectedId, selectedRevision, activeRunId, latestRunId, token, editing]);

  useEffect(() => {
    if (!visible || !selectedId || !runId || editing) return;
    const controller = new AbortController();
    let inFlight = false;
    async function refreshRun() {
      if (inFlight || controller.signal.aborted || document.visibilityState === "hidden") return;
      inFlight = true;
      setLoadingRun(true);
      try {
        const saved = await getWatchRun(selectedId!, runId!, controller.signal);
        if (!controller.signal.aborted) setRun(saved);
      } catch (reason) { if (!controller.signal.aborted) setError(messageOf(reason)); }
      finally { inFlight = false; if (!controller.signal.aborted) setLoadingRun(false); }
    }
    void refreshRun();
    const timer = activeRunId === runId ? window.setInterval(() => void refreshRun(), 3000) : undefined;
    return () => { controller.abort(); window.clearInterval(timer); };
  }, [visible, selectedId, runId, activeRunId, selectedRevision, token, editing]);

  const closeSidebar = useCallback(() => {
    setDrawerOpen(false);
    window.requestAnimationFrame(() => sidebarButton.current?.focus({ preventScroll: true }));
  }, [setDrawerOpen]);
  useEffect(() => {
    const query = window.matchMedia?.("(max-width: 760px)");
    if (!query) return;
    const update = (event: MediaQueryListEvent) => {
      const focused = document.activeElement;
      const focusInSidebar = focused instanceof HTMLElement && sidebar.current?.contains(focused);
      setCompact(event.matches);
      // Hidden workspaces share the app drawer state with Chat and Tasks.
      if (!visible) return;
      setDrawerOpen(false);
      if (event.matches && focusInSidebar) window.requestAnimationFrame(() => sidebarButton.current?.focus());
      else if (!event.matches && focused === sidebarButton.current) {
        window.requestAnimationFrame(() => sidebar.current?.querySelector<HTMLButtonElement>(".watch-sidebar-scroll > button")?.focus());
      }
    };
    query.addEventListener("change", update);
    return () => query.removeEventListener("change", update);
  }, [setDrawerOpen, visible]);
  useEffect(() => {
    if (!sidebarOpen) return;
    sidebar.current?.querySelector<HTMLButtonElement>(".watch-sidebar-close")?.focus();
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") { event.preventDefault(); closeSidebar(); }
      if (event.key !== "Tab") return;
      const controls = [...(sidebar.current?.querySelectorAll<HTMLElement>('button:not([disabled]), [tabindex="0"]') ?? [])];
      const first = controls[0], last = controls[controls.length - 1];
      if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last?.focus(); }
      else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first?.focus(); }
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [sidebarOpen, closeSidebar]);

  function showDetail() {
    setDrawerOpen(false);
    window.requestAnimationFrame(() => { if (detail.current) { detail.current.scrollTop = 0; if (compact) detail.current.focus({ preventScroll: true }); } });
  }

  useEffect(() => {
    if (!visible || !openRequest || openedRequest.current === openRequest.serial) return;
    openedRequest.current = openRequest.serial;
    generation.current += 1;
    const controller = new AbortController();
    queueMicrotask(() => {
      if (controller.signal.aborted) return;
      setWatches((current) => mergeWatch(current, openRequest.watch));
      setSelectedId(openRequest.watch.id); setRunId(openRequest.run.id); setRun(openRequest.run);
      setEditing(null); setConflict(undefined); setError(""); setNotice("");
      window.requestAnimationFrame(() => { if (compact) detail.current?.focus({ preventScroll: true }); });
    });
    return () => controller.abort();
  }, [visible, openRequest, compact]);
  function selectWatch(id: string) {
    setSelectedId(id); setRunId(null); setRun(null); setRuns([]); setEditing(null); setError(""); setNotice(""); setConflict(undefined); showDetail();
  }
  function newWatch() { setEditing("new"); setConflict(undefined); setError(""); setNotice(""); showDetail(); }

  async function reloadConflict(id: string) {
    try {
      const latest = await getWatch(id);
      if (!mounted.current) return;
      setWatches((current) => mergeWatch(current, latest));
      setConflict(latest);
      setNotice("This watch changed. Review its latest saved state before trying again. Any edit draft is preserved.");
    } catch (reason) { if (mounted.current) setError(messageOf(reason)); }
  }

  async function save(definition: WatchDefinition, expectedRevision?: number) {
    if (busyRef.current) return;
    busyRef.current = true; generation.current += 1; setBusy(true); setError(""); setNotice("");
    const edit = editing && editing !== "new" ? editing : null;
    try {
      let saved: PublicWatch;
      if (edit) saved = await updateWatch(edit.id, definition, expectedRevision!);
      else {
        const key = JSON.stringify(definition);
        if (createRequest.current?.key !== key) createRequest.current = { key, id: crypto.randomUUID() };
        saved = await createWatch(definition, createRequest.current.id);
      }
      if (!mounted.current) return;
      createRequest.current = null;
      setWatches((current) => mergeWatch(current, saved)); setSelectedId(saved.id); setEditing(null); setConflict(undefined); setRunId(null); setRun(null); setRuns([]);
      setNotice(edit ? "Watch saved. The edited definition applies to future checks." : "Watch created. Its first scheduled check is in the future; use Run now for an immediate baseline.");
    } catch (reason) {
      if (!mounted.current) return;
      if (reason instanceof ApiError && reason.status === 409 && edit) await reloadConflict(edit.id);
      else setError(messageOf(reason));
    } finally { busyRef.current = false; if (mounted.current) { setBusy(false); setToken((value) => value + 1); } }
  }

  async function act(watch: PublicWatch, action: "pause" | "resume" | "run" | "delete") {
    if (busyRef.current) return;
    busyRef.current = true; generation.current += 1; setBusy(true); setError(""); setNotice("");
    try {
      if (action === "delete") {
        await deleteWatch(watch.id, watch.revision);
        if (!mounted.current) return;
        deleted.current.add(watch.id); setWatches((current) => current.filter((item) => item.id !== watch.id)); setSelectedId(null); setRunId(null); setRun(null); setRuns([]);
        setNotice("Watch and its saved history deleted.");
      } else {
        const updated = await actOnWatch(watch.id, action, watch.revision);
        if (!mounted.current) return;
        setWatches((current) => mergeWatch(current, updated));
        if (action === "run") { setRunId(updated.active_run_id); setRun(null); setNotice("Check queued. The saved schedule is unchanged."); }
        else setNotice(action === "pause" ? "Watch paused. Queued or running checks cannot publish a result after this change." : "Watch resumed. Its next scheduled check is shown below.");
      }
    } catch (reason) {
      if (!mounted.current) return;
      if (reason instanceof ApiError && reason.status === 409) await reloadConflict(watch.id);
      else setError(messageOf(reason));
    } finally { busyRef.current = false; if (mounted.current) { setBusy(false); setToken((value) => value + 1); } }
  }

  const errors = <>{error ? <p className="agents-error" role="alert">{error}</p> : null}{notice ? <p className="agents-notice" role="status">{notice}</p> : null}</>;
  return <section className="watches-workspace" data-sidebar-open={sidebarOpen} aria-labelledby="watches-title">
    {compact ? <div className="agents-mobile-toolbar" inert={sidebarOpen ? true : undefined}><button className="agents-secondary" type="button" ref={sidebarButton} aria-controls="watches-navigation" aria-expanded={sidebarOpen} onClick={() => setDrawerOpen(true)}>Watch controls</button><span>Page monitoring</span></div> : null}
    {sidebarOpen ? <button className="agents-sidebar-backdrop" type="button" tabIndex={-1} aria-label="Dismiss watch controls" onClick={closeSidebar} /> : null}
    <div className="watch-columns">
      <aside className="watch-sidebar" id="watches-navigation" ref={sidebar} aria-label="Saved watches" role={compact ? "dialog" : undefined} aria-modal={sidebarOpen ? true : undefined} aria-hidden={compact && !sidebarOpen ? true : undefined} inert={compact && !sidebarOpen ? true : undefined}>
        <header className="agents-header"><div><p className="agents-eyebrow">Keep an eye on what matters</p><h1 id="watches-title">Watches</h1></div>{compact ? <button type="button" className="agents-secondary watch-sidebar-close" aria-label="Close watch controls" onClick={closeSidebar}>✕</button> : null}</header>
        <div className="watch-sidebar-scroll"><button className="agents-primary" type="button" disabled={busy} onClick={newWatch}>New watch</button><p className="watch-help">Scheduled checks continue while you’re away. Only public pages are supported.</p><button className="agents-refresh" type="button" onClick={() => setToken((value) => value + 1)}>Refresh watches</button>{loading ? <p role="status">Loading watches…</p> : !watches.length ? <p className="agents-empty">No watches yet. Create one to follow a public page.</p> : <ul className="agents-task-list">{watches.map((watch) => <li key={watch.id}><button type="button" className="agents-task-card" aria-pressed={!editing && selectedId === watch.id} onClick={() => selectWatch(watch.id)} aria-label={`Open watch: ${watch.title}`}><span className={`agents-status agents-status-${watch.enabled ? "running" : "paused"}`}>{watch.enabled ? "Scheduled" : "Paused"}</span><strong>{watch.title}</strong><span className="agents-task-stage">{watch.active_run_id ? "Check in progress" : watch.latest_run ? runLabel(watch.latest_run) : "Awaiting baseline"}</span><span className="agents-task-stage">Next: {timestamp(watch.next_run_at)}</span></button></li>)}</ul>}</div>
      </aside>
      <section className="watch-detail-scroll" aria-label="Selected watch" tabIndex={0} ref={detail} inert={sidebarOpen ? true : undefined}>{errors}<div className="watch-detail">
        {editing ? <WatchForm key={editing === "new" ? "new" : editing.id} watch={editing === "new" ? undefined : editing} timezone={timezone} busy={busy} conflict={conflict} onSave={save} onCancel={() => { setEditing(null); setConflict(undefined); }} /> : selected ? <>
          <header className="watch-detail-header"><div className="agents-detail-toolbar"><span className={`agents-status agents-status-${selected.enabled ? "running" : "paused"}`}>{selected.enabled ? "Scheduled" : "Paused"}</span><WatchControls key={`${selected.id}:${selected.revision}:${visible}`} watch={selected} busy={busy} onAction={(action) => void act(selected, action)} onEdit={() => { setEditing(selected); setConflict(undefined); }} /></div><h2>{selected.title}</h2><a href={selected.url} target="_blank" rel="noopener noreferrer">{selected.url}</a><p>{selected.criterion}</p><p>{scheduleLabel(selected)}</p><p>Next scheduled check: <strong>{timestamp(selected.next_run_at)}</strong></p>{!selected.enabled ? <p>Run now checks this page once and leaves the schedule paused.</p> : null}</header>
          <section className="watch-history" aria-labelledby="watch-history-title"><h3 id="watch-history-title">Check history</h3><p className="watch-help">The first successful check saves a baseline. Later successful checks compare normalized page text; failed checks leave the baseline unchanged. Showing up to 50 recent checks.</p>{runs.length ? <ul>{runs.map((item) => <li key={item.id}><button type="button" aria-pressed={runId === item.id} onClick={() => { setRunId(item.id); setRun(null); }}><strong>{runLabel(item)}</strong><span>{item.kind === "catchup" ? "Catch-up after downtime" : item.kind === "manual" ? "Run now" : "Scheduled"}</span><time dateTime={item.scheduled_at}>{timestamp(item.scheduled_at)}</time></button></li>)}</ul> : <p className="agents-empty">No saved checks yet. Run now to establish a baseline, or wait for the first scheduled check.</p>}</section>
          {run && run.id === runId && run.watch_id === selected.id ? <WatchRunDetail run={run} timestamp={timestamp} /> : loadingRun ? <p role="status">Loading saved check…</p> : null}
        </> : <div className="agents-welcome"><span className="agents-welcome-mark" aria-hidden="true">◉</span><h2>Follow a public page</h2><p>Save a schedule and a criterion for the changes you care about. Audrey keeps check history and adds relevant changes to Activity.</p><button className="agents-primary" type="button" onClick={newWatch}>Create a watch</button></div>}
      </div></section>
    </div>
  </section>;
}

function WatchControls({ watch, busy, onAction, onEdit }: { watch: PublicWatch; busy: boolean; onAction: (action: "pause" | "resume" | "run" | "delete") => void; onEdit: () => void }) {
  const [confirming, setConfirming] = useState(false);
  useEffect(() => {
    const reset = () => setConfirming(false);
    const onVisible = () => { if (document.visibilityState === "hidden") reset(); };
    window.addEventListener("blur", reset); document.addEventListener("visibilitychange", onVisible);
    return () => { window.removeEventListener("blur", reset); document.removeEventListener("visibilitychange", onVisible); };
  }, []);
  return <div className="agents-task-actions watch-controls" role="group" aria-label="Watch controls"><button type="button" className="agents-primary" disabled={busy || Boolean(watch.active_run_id)} onClick={() => onAction("run")}>Run now</button><button type="button" className="agents-secondary" disabled={busy} onClick={() => onAction(watch.enabled ? "pause" : "resume")}>{watch.enabled ? "Pause" : "Resume"}</button><button type="button" className="agents-secondary" disabled={busy} onClick={onEdit}>Edit</button><button type="button" className={`agents-secondary danger-button${confirming ? " confirming-delete" : ""}`} disabled={busy} aria-label={confirming ? "Confirm delete watch" : "Delete watch"} aria-pressed={confirming} title={confirming ? "Click again to delete this watch, snapshots and history." : "Delete watch and history"} onBlur={() => setConfirming(false)} onKeyDown={(event) => { if (event.key === "Escape") setConfirming(false); }} onClick={() => { if (confirming) { setConfirming(false); onAction("delete"); } else setConfirming(true); }}>{confirming ? "✓" : "Delete watch"}</button></div>;
}

function WatchRunDetail({ run, timestamp }: { run: WatchRun; timestamp: (value: string | null) => string }) {
  return <section className="watch-run" aria-labelledby="watch-run-title"><h3 id="watch-run-title">{runLabel(run)}</h3><p>{run.kind === "catchup" ? "Catch-up after downtime" : run.kind === "manual" ? "Run now" : "Scheduled check"} · {timestamp(run.scheduled_at)}</p><p>Watch revision {run.watch_revision}{run.completed_at ? ` · Finished ${timestamp(run.completed_at)}` : ""}</p>{run.definition ? <details className="watch-saved-inputs"><summary>Saved check inputs</summary><dl><dt>Title</dt><dd>{run.definition.title}</dd><dt>Public page URL</dt><dd>{run.definition.url}</dd><dt>Change criterion</dt><dd>{run.definition.criterion}</dd><dt>Schedule</dt><dd>{scheduleLabel(run.definition)}</dd></dl></details> : null}{run.error ? <p className="agents-error" role="alert">{run.error}</p> : null}{run.status === "queued" || run.status === "running" ? <p role="status">This check is {run.status === "queued" ? "queued" : "in progress"}. Saved results will appear here.</p> : null}{run.outcome === "baseline" ? <p>This successful check established the saved baseline for future comparisons.</p> : null}{run.outcome === "unchanged" ? <p>The normalized page text is unchanged. No repeated notification was created.</p> : null}{run.outcome === "changed" ? <p>{run.significant === true ? "The local model found a change relevant to your criterion." : run.significant === false ? "The page changed; the local model did not find a change relevant to your criterion." : "The page changed."}</p> : null}{run.report ? <section><h4>{run.outcome === "changed" ? "Local model assessment" : "Check result"}</h4><p className="watch-plain-text">{run.report}</p></section> : null}{run.diff ? <details className="watch-diff"><summary>Exact normalized text diff</summary><pre>{run.diff}</pre></details> : null}</section>;
}
