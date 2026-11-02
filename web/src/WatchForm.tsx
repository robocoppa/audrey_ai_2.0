import { useEffect, useRef, useState, type FormEvent } from "react";
import { listProjects, previewWatchSchedule, type AudreyProject, type PublicWatch, type WatchDefinition, type WatchSchedule } from "./api";

export function WatchForm({ watch, timezone, busy, conflict, visible = true, projectRequest, onSave, onCancel }: {
  watch?: PublicWatch;
  timezone?: string;
  busy: boolean;
  conflict?: PublicWatch;
  visible?: boolean;
  projectRequest?: { project: AudreyProject; serial: number };
  onSave: (definition: WatchDefinition, expectedRevision?: number) => Promise<void>;
  onCancel: () => void;
}) {
  const [title, setTitle] = useState(watch?.title ?? "");
  const [url, setUrl] = useState(watch?.url ?? "");
  const [criterion, setCriterion] = useState(watch?.criterion ?? "");
  const [kind, setKind] = useState<WatchSchedule["kind"]>(watch?.schedule.kind ?? "daily");
  const [zone, setZone] = useState(watch?.schedule.timezone ?? timezone ?? Intl.DateTimeFormat().resolvedOptions().timeZone ?? "UTC");
  const [minutes, setMinutes] = useState(String(watch?.schedule.interval_minutes ?? 1440));
  const [time, setTime] = useState(watch?.schedule.time ?? "09:00");
  const [weekday, setWeekday] = useState(watch?.schedule.weekday ?? 0);
  const [baseRevision, setBaseRevision] = useState(watch?.revision);
  const [enabled, setEnabled] = useState(watch?.enabled ?? true);
  const [projectId, setProjectId] = useState(watch?.project?.id ?? "");
  const [selectedProjectName, setSelectedProjectName] = useState(watch?.project?.name ?? "");
  const [projects, setProjects] = useState<AudreyProject[]>([]);
  const [projectCursor, setProjectCursor] = useState<string | null>(null);
  const [projectsLoading, setProjectsLoading] = useState(false);
  const [projectsError, setProjectsError] = useState("");
  const [projectRefresh, setProjectRefresh] = useState(0);
  const projectRead = useRef<AbortController | null>(null);
  const appliedProject = useRef<number | null>(null);
  const [preview, setPreview] = useState<{ key: string; occurrences: string[] } | null>(null);
  const [previewing, setPreviewing] = useState(false);
  const [error, setError] = useState("");
  const previewRequest = useRef<AbortController | null>(null);
  const schedule: WatchSchedule = { kind, timezone: zone.trim(), ...(kind === "interval" ? { interval_minutes: Number(minutes) } : { time, ...(kind === "weekly" ? { weekday } : {}) }) };
  const scheduleKey = JSON.stringify(schedule);
  const previewCurrent = preview?.key === scheduleKey;
  const needsRebase = Boolean(conflict && conflict.revision !== baseRevision);
  useEffect(() => {
    if (visible) return;
    previewRequest.current?.abort();
    queueMicrotask(() => setPreviewing(false));
  }, [visible]);
  useEffect(() => () => previewRequest.current?.abort(), []);

  useEffect(() => {
    if (!visible) return;
    const controller = new AbortController();
    projectRead.current?.abort(); projectRead.current = controller;
    queueMicrotask(() => { if (!controller.signal.aborted) { setProjectsLoading(true); setProjectsError(""); } });
    void listProjects(undefined, controller.signal).then((page) => {
      if (!controller.signal.aborted) { setProjects(page.items.filter((project) => !project.private_required)); setProjectCursor(page.next_cursor); }
    }).catch((reason: unknown) => {
      if (!controller.signal.aborted) setProjectsError(reason instanceof Error ? reason.message : "Could not load Projects.");
    }).finally(() => { if (!controller.signal.aborted) setProjectsLoading(false); });
    return () => { controller.abort(); projectRead.current?.abort(); };
  }, [visible, projectRefresh]);

  useEffect(() => {
    if (!visible || !projectRequest || appliedProject.current === projectRequest.serial) return;
    const controller = new AbortController();
    queueMicrotask(() => {
      if (controller.signal.aborted) return;
      appliedProject.current = projectRequest.serial;
      if (projectRequest.project.private_required) { setError("Private Projects are not available for public page watches."); return; }
      setProjectId(projectRequest.project.id);
      setSelectedProjectName(projectRequest.project.name);
    });
    return () => controller.abort();
  }, [visible, projectRequest]);

  async function loadMoreProjects() {
    if (!visible || !projectCursor || projectsLoading) return;
    const controller = new AbortController(); projectRead.current?.abort(); projectRead.current = controller;
    setProjectsLoading(true); setProjectsError("");
    try {
      const page = await listProjects(projectCursor, controller.signal);
      if (!controller.signal.aborted) {
        setProjects((current) => [...new Map([...current, ...page.items.filter((project) => !project.private_required)].map((project) => [project.id, project])).values()]);
        setProjectCursor(page.next_cursor);
      }
    } catch (reason) { if (!controller.signal.aborted) setProjectsError(reason instanceof Error ? reason.message : "Could not load more Projects."); }
    finally { if (!controller.signal.aborted) setProjectsLoading(false); }
  }

  async function showPreview() {
    previewRequest.current?.abort();
    const controller = new AbortController();
    previewRequest.current = controller;
    setPreviewing(true);
    setError("");
    try {
      const result = await previewWatchSchedule(schedule, controller.signal);
      if (!controller.signal.aborted) setPreview({ key: scheduleKey, occurrences: result.occurrences });
    } catch (reason) {
      if (!controller.signal.aborted) setError(reason instanceof Error ? reason.message : "Could not preview this schedule.");
    } finally {
      if (!controller.signal.aborted) setPreviewing(false);
    }
  }

  function submit(event: FormEvent) {
    event.preventDefault();
    setError("");
    try {
      const address = new URL(url.trim());
      if (!["http:", "https:"].includes(address.protocol) || address.username || address.password || address.search || address.hash) {
        throw new Error("Use a public HTTP or HTTPS page without credentials, a query string or a fragment.");
      }
    } catch (reason) {
      setError(reason instanceof Error && reason.message.startsWith("Use a public") ? reason.message : "Enter a valid public HTTP or HTTPS URL.");
      return;
    }
    if (!previewCurrent || needsRebase) return;
    if (projectId && watch?.project?.id === projectId && !watch.project.available) {
      setError("Choose an available Project or None before saving.");
      return;
    }
    void onSave({ title: title.trim(), url: url.trim(), criterion: criterion.trim(), schedule, enabled, ...(projectId ? { project_id: projectId } : watch?.project ? { project_id: null } : {}) }, baseRevision);
  }

  return <form className="watch-form" onSubmit={submit} aria-label={watch ? "Edit watch" : "Create watch"}>
    <h2>{watch ? "Edit watch" : "New public page watch"}</h2>
    <p>Audrey checks one public page on a saved schedule. Local models assess changes against your criterion. Creating a watch schedules its first check for the future; Run now starts an immediate check.</p>
    <label>Project<select value={projectId} disabled={busy} onChange={(event) => { setProjectId(event.target.value); setSelectedProjectName(projects.find((project) => project.id === event.target.value)?.name ?? ""); }}><option value="">None</option>{projectId && !projects.some((project) => project.id === projectId) ? <option value={projectId}>{selectedProjectName || "Unavailable Project"}</option> : null}{projects.map((project) => <option key={project.id} value={project.id}>{project.name}</option>)}</select></label>
    <p className="watch-help">A Project organizes this public page watch. Its instructions, notes and files are not used for checks.</p>
    {projectsLoading ? <p className="watch-help" role="status">Loading Projects…</p> : null}
    {projectsError ? <p className="agents-error" role="alert">Could not load Projects. {projectsError} <button type="button" className="agents-secondary" disabled={busy || projectsLoading} onClick={() => setProjectRefresh((value) => value + 1)}>Retry Projects</button></p> : null}
    {projectCursor ? <button type="button" className="agents-secondary" disabled={busy || projectsLoading} onClick={() => void loadMoreProjects()}>Load more Projects</button> : null}
    <label>Title<input required maxLength={100} value={title} disabled={busy} onChange={(event) => setTitle(event.target.value)} /></label>
    <label>Public page URL<input type="url" required maxLength={2048} value={url} disabled={busy} placeholder="https://example.com/releases" onChange={(event) => setUrl(event.target.value)} /></label>
    <p className="watch-help">Use a public HTTP or HTTPS page without sign-in, credentials, a query string or a fragment.</p>
    <label>What changes matter?<textarea required maxLength={2000} rows={4} value={criterion} disabled={busy} placeholder="Tell me when a new stable release is announced." onChange={(event) => setCriterion(event.target.value)} /></label>
    <fieldset disabled={busy}>
      <legend>Schedule</legend>
      <label>Cadence<select value={kind} onChange={(event) => setKind(event.target.value as WatchSchedule["kind"])}><option value="interval">Every interval</option><option value="daily">Daily</option><option value="weekly">Weekly</option></select></label>
      {kind === "interval" ? <label>Interval in minutes<input type="number" required min={60} max={10080} step={1} value={minutes} onChange={(event) => setMinutes(event.target.value)} /></label> : <>
        {kind === "weekly" ? <label>Day<select value={weekday} onChange={(event) => setWeekday(Number(event.target.value))}>{["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"].map((day, index) => <option key={day} value={index}>{day}</option>)}</select></label> : null}
        <label>Local time<input type="time" required value={time} onChange={(event) => setTime(event.target.value)} /></label>
      </>}
      <label>Time zone<input required maxLength={100} value={zone} onChange={(event) => setZone(event.target.value)} placeholder="America/Denver" /></label>
      <button className="agents-secondary" type="button" disabled={previewing || !zone.trim()} onClick={() => void showPreview()}>{previewing ? "Previewing…" : "Preview next checks"}</button>
      {previewCurrent ? <div className="watch-schedule-preview" role="status"><strong>Next three checks (UTC)</strong><ol>{preview.occurrences.map((value) => <li key={value}><time dateTime={value}>{new Date(value).toISOString().replace("T", " ").replace(".000Z", " UTC")}</time></li>)}</ol></div> : <p className="watch-help">Preview the schedule before saving. Local times follow this time zone, including daylight saving changes.</p>}
    </fieldset>
    {needsRebase ? <section className="watch-conflict" role="status"><strong>This watch changed in another session.</strong><p>Your draft is preserved. Compare the latest saved definition before applying your draft.</p><details><summary>Latest saved definition</summary><p>{conflict?.title}</p><p>{conflict?.url}</p><p>{conflict?.criterion}</p><p>Project: {conflict?.project?.name ?? (conflict?.project ? "Unavailable Project" : "None")}</p><p>{conflict?.enabled ? "Scheduled" : "Paused"}</p><pre>{JSON.stringify(conflict?.schedule, null, 2)}</pre></details><button type="button" className="agents-secondary" onClick={() => { setBaseRevision(conflict?.revision); setEnabled(conflict?.enabled ?? enabled); }}>Use latest revision with my draft</button></section> : null}
    {error ? <p className="agents-error" role="alert">{error}</p> : null}
    <div className="watch-form-actions"><button className="agents-primary" type="submit" disabled={busy || !previewCurrent || needsRebase || !title.trim() || !criterion.trim() || !url.trim()}>{busy ? "Saving…" : watch ? "Save watch" : "Create watch"}</button><button className="agents-secondary" type="button" disabled={busy} onClick={onCancel}>Cancel</button></div>
  </form>;
}
