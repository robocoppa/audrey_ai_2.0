import { useEffect, useRef, useState, type FormEvent } from "react";
import { previewWatchSchedule, type PublicWatch, type WatchDefinition, type WatchSchedule } from "./api";

export function WatchForm({ watch, timezone, busy, conflict, onSave, onCancel }: {
  watch?: PublicWatch;
  timezone?: string;
  busy: boolean;
  conflict?: PublicWatch;
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
  const [preview, setPreview] = useState<{ key: string; occurrences: string[] } | null>(null);
  const [previewing, setPreviewing] = useState(false);
  const [error, setError] = useState("");
  const previewRequest = useRef<AbortController | null>(null);
  const schedule: WatchSchedule = { kind, timezone: zone.trim(), ...(kind === "interval" ? { interval_minutes: Number(minutes) } : { time, ...(kind === "weekly" ? { weekday } : {}) }) };
  const scheduleKey = JSON.stringify(schedule);
  const previewCurrent = preview?.key === scheduleKey;
  const needsRebase = Boolean(conflict && conflict.revision !== baseRevision);
  useEffect(() => () => previewRequest.current?.abort(), []);

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
    void onSave({ title: title.trim(), url: url.trim(), criterion: criterion.trim(), schedule, enabled }, baseRevision);
  }

  return <form className="watch-form" onSubmit={submit} aria-label={watch ? "Edit watch" : "Create watch"}>
    <h2>{watch ? "Edit watch" : "New public page watch"}</h2>
    <p>Audrey checks one public page on a saved schedule. Local models assess changes against your criterion. Creating a watch schedules its first check for the future; Run now starts an immediate check.</p>
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
    {needsRebase ? <section className="watch-conflict" role="status"><strong>This watch changed in another session.</strong><p>Your draft is preserved. Compare the latest saved definition before applying your draft.</p><details><summary>Latest saved definition</summary><p>{conflict?.title}</p><p>{conflict?.url}</p><p>{conflict?.criterion}</p><p>{conflict?.enabled ? "Scheduled" : "Paused"}</p><pre>{JSON.stringify(conflict?.schedule, null, 2)}</pre></details><button type="button" className="agents-secondary" onClick={() => { setBaseRevision(conflict?.revision); setEnabled(conflict?.enabled ?? enabled); }}>Use latest revision with my draft</button></section> : null}
    {error ? <p className="agents-error" role="alert">{error}</p> : null}
    <div className="watch-form-actions"><button className="agents-primary" type="submit" disabled={busy || !previewCurrent || needsRebase || !title.trim() || !criterion.trim() || !url.trim()}>{busy ? "Saving…" : watch ? "Save watch" : "Create watch"}</button><button className="agents-secondary" type="button" disabled={busy} onClick={onCancel}>Cancel</button></div>
  </form>;
}
