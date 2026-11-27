import { useEffect, useRef, useState, type FormEvent } from "react";
import { listProjectFiles, listProjects, previewWatchSchedule, type AudreyProject, type AudreyProjectFile, type PublicWatch, type WatchDefinition, type WatchSchedule, type WatchSourceKind } from "./api";

type WatchCadence = WatchSchedule["kind"] | "hourly";
type WatchFormSource = WatchSourceKind | "project_file_text";
type FileChoices = { projectId: string; filenames: string[] } | { projectId: string; error: true };
const maxWatchedTextBytes = 64_000;

function eligibleTextFilenames(files: AudreyProjectFile[]): string[] {
  const seen = new Set<string>();
  const filenames: string[] = [];
  // The Project listing returns the newest attachments first.
  for (const file of files) {
    const { filename } = file;
    if (seen.has(filename)) continue;
    seen.add(filename);
    if (file.status === "ready" && file.kind === "text" && file.bytes > 0 && file.bytes <= maxWatchedTextBytes
        && /^.+\.(txt|md|csv|tsv|json)$/iu.test(filename) && Array.from(filename).length <= 255
        && !["/", "\\", ":", "\0"].some((character) => filename.includes(character))) {
      filenames.push(filename);
    }
  }
  return filenames;
}

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
  const [sourceKind, setSourceKind] = useState<WatchFormSource>(watch?.private_required || watch?.source_filename ? "project_file_text" : watch?.source_kind ?? "public_page");
  const projectFiles = sourceKind !== "public_page";
  const fileText = sourceKind === "project_file_text";
  const fixedProject = Boolean(watch && projectFiles);
  const [title, setTitle] = useState(watch?.title ?? "");
  const [url, setUrl] = useState(watch?.url ?? "");
  const [criterion, setCriterion] = useState(watch?.criterion ?? "");
  const [cadence, setCadence] = useState<WatchCadence>(
    watch?.schedule.kind === "interval" && watch.schedule.interval_minutes === 60
      ? "hourly"
      : watch?.schedule.kind ?? "daily",
  );
  const [zone, setZone] = useState(watch?.schedule.timezone ?? timezone ?? Intl.DateTimeFormat().resolvedOptions().timeZone ?? "UTC");
  const [minutes, setMinutes] = useState(String(watch?.schedule.interval_minutes ?? 1440));
  const [time, setTime] = useState(watch?.schedule.time ?? "09:00");
  const [weekday, setWeekday] = useState(watch?.schedule.weekday ?? 0);
  const [baseRevision, setBaseRevision] = useState(watch?.revision);
  const [enabled, setEnabled] = useState(watch?.enabled ?? true);
  const [projectId, setProjectId] = useState(watch?.project?.id ?? "");
  const [fileSelection, setFileSelection] = useState({ projectId: watch?.project?.id ?? "", filename: watch?.source_filename ?? "" });
  const [fileChoices, setFileChoices] = useState<FileChoices | null>(null);
  const [fileRefresh, setFileRefresh] = useState(0);
  const [selectedProjectName, setSelectedProjectName] = useState(watch?.project?.name ?? "");
  const [selectedProjectPrivate, setSelectedProjectPrivate] = useState(Boolean(watch?.private_required || watch?.project?.private_required));
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
  const schedule: WatchSchedule = cadence === "hourly" || cadence === "interval"
    ? { kind: "interval", timezone: zone.trim(), interval_minutes: cadence === "hourly" ? 60 : Number(minutes) }
    : { kind: cadence, timezone: zone.trim(), time, ...(cadence === "weekly" ? { weekday } : {}) };
  const scheduleKey = JSON.stringify(schedule);
  const previewCurrent = preview?.key === scheduleKey;
  const needsRebase = Boolean(conflict && conflict.revision !== baseRevision);
  const projectPrivate = selectedProjectPrivate || Boolean(projects.find((project) => project.id === projectId)?.private_required);
  const eligibleProjects = projects.filter((project) => fileText || !project.private_required);
  const projectEligible = !projectPrivate || fileText;
  const projectUnavailable = Boolean(projectId && watch?.project?.id === projectId && !watch.project.available);
  const currentFileChoices = fileChoices?.projectId === projectId ? fileChoices : null;
  const filenames = currentFileChoices && "filenames" in currentFileChoices ? currentFileChoices.filenames : [];
  const sourceFilename = fileSelection.projectId === projectId ? fileSelection.filename : "";
  const fileReady = fixedProject ? Boolean(sourceFilename) : filenames.includes(sourceFilename);
  const sourceReady = projectFiles
    ? Boolean(projectId) && projectEligible && !projectUnavailable && (!fileText || fileReady)
    : Boolean(criterion.trim() && url.trim()) && projectEligible;
  useEffect(() => {
    if (visible) return;
    previewRequest.current?.abort();
    queueMicrotask(() => setPreviewing(false));
  }, [visible]);
  useEffect(() => () => previewRequest.current?.abort(), []);

  useEffect(() => {
    if (!visible || fixedProject) return;
    const controller = new AbortController();
    projectRead.current?.abort(); projectRead.current = controller;
    queueMicrotask(() => { if (!controller.signal.aborted) { setProjectsLoading(true); setProjectsError(""); } });
    void listProjects(undefined, controller.signal).then((page) => {
      if (!controller.signal.aborted) { setProjects(page.items); setProjectCursor(page.next_cursor); }
    }).catch((reason: unknown) => {
      if (!controller.signal.aborted) setProjectsError(reason instanceof Error ? reason.message : "Could not load Projects.");
    }).finally(() => { if (!controller.signal.aborted) setProjectsLoading(false); });
    return () => { controller.abort(); projectRead.current?.abort(); };
  }, [visible, fixedProject, projectRefresh]);

  useEffect(() => {
    if (!visible || !fileText || fixedProject || !projectId) return;
    const controller = new AbortController();
    queueMicrotask(() => {
      if (!controller.signal.aborted) setFileChoices(null);
    });
    void listProjectFiles(projectId, controller.signal).then(({ items }) => {
      if (!controller.signal.aborted) setFileChoices({ projectId, filenames: eligibleTextFilenames(items) });
    }).catch(() => {
      if (!controller.signal.aborted) setFileChoices({ projectId, error: true });
    });
    return () => controller.abort();
  }, [fileRefresh, fileText, fixedProject, projectId, visible]);

  useEffect(() => {
    if (!visible || !projectRequest || appliedProject.current === projectRequest.serial) return;
    const controller = new AbortController();
    queueMicrotask(() => {
      if (controller.signal.aborted) return;
      appliedProject.current = projectRequest.serial;
      if (projectRequest.project.private_required) setSourceKind("project_file_text");
      setProjectId(projectRequest.project.id);
      setSelectedProjectName(projectRequest.project.name);
      setSelectedProjectPrivate(Boolean(projectRequest.project.private_required));
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
        setProjects((current) => [...new Map([...current, ...page.items].map((project) => [project.id, project])).values()]);
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
    if (!previewCurrent || needsRebase || !title.trim()) return;
    const settings = { title: title.trim(), schedule, enabled };
    if (projectFiles) {
      if (sourceReady) void onSave({
        ...settings, source_kind: "project_files", project_id: projectId,
        ...(fileText ? { source_filename: sourceFilename } : {}),
        ...(projectPrivate ? { private_required: true as const } : {}),
      }, baseRevision);
      return;
    }
    const enteredUrl = url.trim();
    const pageUrl = /^[a-z][a-z\d+.-]*:/i.test(enteredUrl) ? enteredUrl : `https://${enteredUrl}`;
    try {
      const address = new URL(pageUrl);
      if (!/^https?:\/\/[^/]/i.test(pageUrl) || !["http:", "https:"].includes(address.protocol) || address.username || address.password || /[?#]/.test(pageUrl)) {
        throw new Error("Use a public HTTP or HTTPS page without credentials, a query string or a fragment.");
      }
    } catch (reason) {
      setError(reason instanceof Error && reason.message.startsWith("Use a public") ? reason.message : "Enter a valid public HTTP or HTTPS URL.");
      return;
    }
    if (projectUnavailable) {
      setError("Choose an available Project or None before saving.");
      return;
    }
    if (sourceReady) void onSave({ ...settings, url: pageUrl, criterion: criterion.trim(), ...(projectId ? { project_id: projectId } : watch?.project ? { project_id: null } : {}) }, baseRevision);
  }

  return <form className="watch-form" onSubmit={submit} aria-label={watch ? "Edit watch" : "Create watch"}>
    <h2>{watch ? "Edit watch" : "New watch"}</h2>
    <label>Watch source<select value={sourceKind} disabled={busy || Boolean(watch)} onChange={(event) => {
      const next = event.target.value as WatchFormSource;
      if (projectPrivate && next !== "project_file_text") {
        setProjectId(""); setSelectedProjectName(""); setSelectedProjectPrivate(false);
        setFileSelection({ projectId: "", filename: "" });
      }
      setSourceKind(next); setError("");
    }}>
      <option value="public_page">Public page</option><option value="project_files">Project files</option><option value="project_file_text">Project file text</option>
    </select></label>
    <p>{fileText ? "Audrey checks the newest Project attachment with the exact filename you choose. Keep the previous copy attached to compare versions. If that copy is removed, the next successful check starts a new baseline."
      : projectFiles ? "Audrey compares the Project’s attachment list and flags added or removed files."
      : "Audrey checks one public page. Local models assess changes against your criterion."}</p>
    {fileText ? <p className="watch-help">TXT, MD, CSV, TSV, or JSON files up to 64,000 bytes. Checks run locally without a model, web access or saving file contents in watch history.</p> : null}
    <label>Project<select value={projectEligible ? projectId : ""} disabled={busy || fixedProject} required={projectFiles} onChange={(event) => {
      const chosen = eligibleProjects.find((project) => project.id === event.target.value);
      setProjectId(chosen?.id ?? ""); setSelectedProjectName(chosen?.name ?? "");
      setSelectedProjectPrivate(Boolean(chosen?.private_required));
    }}><option value="" disabled={projectFiles}>{projectFiles ? "Choose a Project" : "None"}</option>{projectId && projectEligible && !eligibleProjects.some((project) => project.id === projectId) ? <option value={projectId}>{selectedProjectName || "Unavailable Project"}{projectPrivate ? " · Private" : ""}</option> : null}{eligibleProjects.map((project) => <option key={project.id} value={project.id}>{project.name}{project.private_required ? " · Private" : ""}</option>)}</select></label>
    {projectPrivate && fileText ? <p className="watch-help">Private · Checks follow the selected filename within this Project.</p> : null}
    {!projectFiles ? <p className="watch-help">A Project organizes this public page watch. Its instructions, notes and files are not used for checks.</p> : projectUnavailable ? <p className="agents-error" role="alert">This Project is unavailable. Create a new watch to monitor a different Project.</p> : null}
    {projectsLoading ? <p className="watch-help" role="status">Loading Projects…</p> : null}
    {projectsError ? <p className="agents-error" role="alert">Could not load Projects. {projectsError} <button type="button" className="agents-secondary" disabled={busy || projectsLoading} onClick={() => setProjectRefresh((value) => value + 1)}>Retry Projects</button></p> : null}
    {projectCursor ? <button type="button" className="agents-secondary" disabled={busy || projectsLoading} onClick={() => void loadMoreProjects()}>Load more Projects</button> : null}
    {fileText ? <>
      <label>Filename<select required value={fileReady ? sourceFilename : ""} disabled={busy || fixedProject || !filenames.length} onChange={(event) => setFileSelection({ projectId, filename: event.target.value })}>
        <option value="" disabled>Choose a file</option>
        {fixedProject ? <option value={sourceFilename}>{sourceFilename}</option> : filenames.map((name) => <option key={name} value={name}>{name}</option>)}
      </select></label>
      {!fixedProject && projectId && !currentFileChoices ? <p className="watch-help" role="status">Loading Project files…</p> : null}
      {currentFileChoices && "error" in currentFileChoices ? <p className="agents-error" role="alert">Could not load Project files. <button className="agents-secondary" type="button" disabled={busy} onClick={() => { setFileChoices(null); setFileRefresh((value) => value + 1); }}>Retry files</button></p> : null}
      {currentFileChoices && "filenames" in currentFileChoices && !filenames.length ? <p className="watch-help">Attach a ready TXT, MD, CSV, TSV, or JSON file of up to 64,000 bytes.</p> : null}
    </> : null}
    <label>Title<input required maxLength={100} value={title} disabled={busy} onChange={(event) => setTitle(event.target.value)} /></label>
    {!projectFiles ? <>
      <label>Public page URL<input type="text" inputMode="url" autoCapitalize="none" spellCheck={false} required maxLength={2048} value={url} disabled={busy} placeholder="example.com/releases" onChange={(event) => setUrl(event.target.value)} /></label>
      <p className="watch-help">A domain like example.com uses HTTPS by default. Use a public page without sign-in, a query string or a fragment.</p>
      <label>What changes matter?<textarea required maxLength={2000} rows={4} value={criterion} disabled={busy} placeholder="Tell me when a new stable release is announced." onChange={(event) => setCriterion(event.target.value)} /></label>
    </> : null}
    <fieldset disabled={busy}>
      <legend>Schedule</legend>
      <label>Cadence<select value={cadence} onChange={(event) => setCadence(event.target.value as WatchCadence)}>
        <option value="hourly">Hourly</option><option value="daily">Daily</option><option value="weekly">Weekly</option><option value="interval">Custom interval</option>
      </select></label>
      {cadence === "interval" ? <label>Interval in minutes<input type="number" required min={60} max={10080} step={1} value={minutes} onChange={(event) => setMinutes(event.target.value)} /></label> : null}
      {cadence === "daily" || cadence === "weekly" ? <>
        {cadence === "weekly" ? <label>Day<select value={weekday} onChange={(event) => setWeekday(Number(event.target.value))}>{["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"].map((day, index) => <option key={day} value={index}>{day}</option>)}</select></label> : null}
        <label>Local time<input type="time" required value={time} onChange={(event) => setTime(event.target.value)} /></label>
      </> : null}
      <label>Time zone<input required maxLength={100} value={zone} onChange={(event) => setZone(event.target.value)} placeholder="America/Denver" /></label>
      <button className="agents-secondary" type="button" disabled={previewing || !zone.trim()} onClick={() => void showPreview()}>{previewing ? "Previewing…" : "Preview next checks"}</button>
      {previewCurrent ? <div className="watch-schedule-preview" role="status"><strong>Next three checks (UTC)</strong><ol>{preview.occurrences.map((value) => <li key={value}><time dateTime={value}>{new Date(value).toISOString().replace("T", " ").replace(".000Z", " UTC")}</time></li>)}</ol></div> : <p className="watch-help">Preview the schedule before saving. Local times follow this time zone, including daylight saving changes.</p>}
    </fieldset>
    {needsRebase ? <section className="watch-conflict" role="status"><strong>This watch changed in another session.</strong><p>Your draft is preserved. Compare the latest saved definition before applying your draft.</p><details><summary>Latest saved definition</summary><p>{conflict?.title}</p><p>{conflict?.source_filename ?? conflict?.url}</p><p>{conflict?.criterion}</p><p>Project: {conflict?.project?.name ?? (conflict?.project ? "Unavailable Project" : "None")}</p><p>{conflict?.enabled ? "Scheduled" : "Paused"}</p><pre>{JSON.stringify(conflict?.schedule, null, 2)}</pre></details><button type="button" className="agents-secondary" onClick={() => { setBaseRevision(conflict?.revision); setEnabled(conflict?.enabled ?? enabled); }}>Use latest revision with my draft</button></section> : null}
    {error ? <p className="agents-error" role="alert">{error}</p> : null}
    <div className="watch-form-actions"><button className="agents-primary" type="submit" disabled={busy || !previewCurrent || needsRebase || !title.trim() || !sourceReady}>{busy ? "Saving…" : watch ? "Save watch" : "Create watch"}</button><button className="agents-secondary" type="button" disabled={busy} onClick={onCancel}>Cancel</button></div>
  </form>;
}
