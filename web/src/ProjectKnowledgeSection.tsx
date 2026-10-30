import { useEffect, useId, useRef, useState, type FormEvent } from "react";

import {
  ApiError,
  createProjectKnowledge,
  decideProjectKnowledge,
  deleteProjectKnowledge,
  getProjectKnowledgeHistory,
  listProjectKnowledge,
  updateProjectKnowledge,
  type AudreyProject,
  type KnowledgeEntry,
  type KnowledgeKind,
  type KnowledgeSourceRef,
  type ProjectKnowledgeList,
} from "./api";
import "./ProjectKnowledgeSection.css";

type Props = {
  project: AudreyProject;
  onChanged?: () => void;
  onOpenSource?: (source: KnowledgeSourceRef, entry: KnowledgeEntry, index: number) => void;
};
const KINDS: ReadonlyArray<{ value: KnowledgeKind; label: string }> = [
  { value: "fact", label: "Fact" }, { value: "decision", label: "Decision" },
  { value: "question", label: "Question" }, { value: "action", label: "Action" },
];

export function ProjectKnowledgeSection(props: Props) {
  return <ProjectKnowledge key={props.project.id} {...props} />;
}

function ProjectKnowledge({ project, onChanged, onOpenSource }: Props) {
  const headingId = useId();
  const active = useRef(true);
  const loadGeneration = useRef(0);
  const [loaded, setLoaded] = useState<ProjectKnowledgeList | null>(null);
  const [loadError, setLoadError] = useState("");
  const [loading, setLoading] = useState(true);
  const [version, setVersion] = useState(0);
  const [filter, setFilter] = useState<"all" | KnowledgeKind>("all");
  const [stateFilter, setStateFilter] = useState<"current" | "proposed" | "history">("current");
  const [adding, setAdding] = useState(false);
  const [kind, setKind] = useState<KnowledgeKind>("fact");
  const [text, setText] = useState("");
  const [pinned, setPinned] = useState(false);
  const [saving, setSaving] = useState(false);
  const [createError, setCreateError] = useState("");
  const [editingIds, setEditingIds] = useState<Set<string>>(() => new Set());
  const requestId = useRef<string | null>(null);

  useEffect(() => {
    active.current = true;
    return () => { active.current = false; };
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    const generation = ++loadGeneration.current;
    async function load() {
      try {
        const first = await listProjectKnowledge(project.id, undefined, controller.signal);
        const items = [...first.items];
        let cursor = first.next_cursor;
        const seen = new Set<string>();
        // The repository limits every Project to 200 current entries. Bound the
        // client too, even if a broken server repeats a cursor indefinitely.
        for (let page = 0; cursor && items.length < first.limits.max_entries && page < 3; page++) {
          if (seen.has(cursor)) throw new Error("The saved knowledge list could not be loaded completely.");
          seen.add(cursor);
          const next = await listProjectKnowledge(project.id, cursor, controller.signal);
          items.push(...next.items);
          cursor = next.next_cursor;
        }
        if (!controller.signal.aborted && generation === loadGeneration.current) {
          setLoaded({ ...first, items: [...new Map(items.map((entry) => [entry.id, entry])).values()], next_cursor: cursor });
          setLoadError("");
          setLoading(false);
        }
      } catch (reason) {
        if (!controller.signal.aborted && generation === loadGeneration.current) {
          setLoadError(messageOf(reason));
          setLoading(false);
        }
      }
    }
    void load();
    return () => controller.abort();
  }, [project.id, version]);

  function refresh() {
    setVersion((value) => value + 1);
  }
  function changed(entry?: KnowledgeEntry, deletedId?: string) {
    if (!active.current) return;
    // A list refresh started before this write must not replace its result with
    // an older page after the mutation has already completed.
    ++loadGeneration.current;
    if (entry || deletedId) setLoaded((current) => current ? {
      ...current,
      items: entry
        ? [...current.items.filter((item) => item.id !== entry.id), entry]
        : current.items.filter((item) => item.id !== deletedId),
    } : current);
    onChanged?.();
  }
  async function create(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!text.trim() || saving) return;
    setSaving(true);
    setCreateError("");
    requestId.current ??= crypto.randomUUID();
    try {
      const entry = await createProjectKnowledge(project.id, { kind, text: text.trim(), pinned, request_id: requestId.current });
      if (!active.current) return;
      changed(entry);
      setText("");
      setPinned(false);
      setAdding(false);
      requestId.current = null;
      setStateFilter("current");
      setFilter("all");
    } catch (reason) {
      if (active.current) setCreateError(messageOf(reason));
    } finally {
      if (active.current) setSaving(false);
    }
  }

  const entries = [...(loaded?.items ?? [])].filter((entry) => editingIds.has(entry.id) || (
    (filter === "all" || entry.kind === filter)
    && (stateFilter === "current" ? entry.approval_state === "approved"
      : stateFilter === "proposed" ? entry.approval_state === "proposed"
        : entry.approval_state === "rejected" || entry.approval_state === "superseded")),
  ).sort((a, b) => Number(b.pinned) - Number(a.pinned) || b.updated_at.localeCompare(a.updated_at) || a.id.localeCompare(b.id));
  const pendingCount = loaded?.items.filter((entry) => entry.approval_state === "proposed").length ?? 0;

  return <section className="project-panel project-knowledge" aria-labelledby={headingId}>
    <header><div><span>Saved in this project</span><h2 id={headingId}>Knowledge</h2></div>
      <button type="button" onClick={() => setAdding((value) => !value)} disabled={!loaded || saving} aria-expanded={adding}>
        {adding ? "Close new entry" : "Add entry"}
      </button>
    </header>
    <p className="project-panel-copy">Save facts, decisions, questions and actions for future project conversations. Approval records what you accept; it does not verify a claim.</p>
    {adding ? <form className="knowledge-editor knowledge-create" onSubmit={(event) => void create(event)}>
      <label>Entry type<select value={kind} disabled={saving} onChange={(event) => { setKind(event.target.value as KnowledgeKind); requestId.current = null; }}>
        {KINDS.map((item) => <option key={item.value} value={item.value}>{item.label}</option>)}
      </select></label>
      <label>Entry text<textarea value={text} maxLength={loaded?.limits.max_text_chars ?? 8000} rows={4} required disabled={saving}
        onChange={(event) => { setText(event.target.value); requestId.current = null; }} /></label>
      <label className="knowledge-check"><input type="checkbox" checked={pinned} disabled={saving} onChange={(event) => { setPinned(event.target.checked); requestId.current = null; }} />Pin for project context</label>
      <p className="knowledge-note">Entries you write are saved as approved. An action stays open until you complete it.</p>
      {createError ? <p role="alert" className="knowledge-error">{createError}</p> : null}
      <div className="knowledge-actions"><button type="submit" disabled={saving || !text.trim()}>{saving ? "Saving…" : "Save entry"}</button>
        <button type="button" disabled={saving} onClick={() => setAdding(false)}>Cancel</button></div>
    </form> : null}
    <div className="knowledge-filters">
      <label>Entry type<select value={filter} onChange={(event) => setFilter(event.target.value as typeof filter)}>
        <option value="all">All types</option>{KINDS.map((item) => <option key={item.value} value={item.value}>{item.label}s</option>)}
      </select></label>
      <label>Show<select value={stateFilter} onChange={(event) => setStateFilter(event.target.value as typeof stateFilter)}>
        <option value="current">Approved entries</option><option value="proposed">Proposals{pendingCount ? ` (${pendingCount})` : ""}</option><option value="history">Rejected and superseded</option>
      </select></label>
      <button type="button" onClick={refresh}>Refresh</button>
    </div>
    {pendingCount && stateFilter !== "proposed" ? <button className="knowledge-proposals-notice" type="button" onClick={() => setStateFilter("proposed")}>{pendingCount} {pendingCount === 1 ? "proposal needs" : "proposals need"} your review</button> : null}
    {loading ? <p className="project-panel-copy" role="status">Loading project knowledge…</p> : null}
    {loadError ? <p role="alert" className="knowledge-error">Could not refresh project knowledge. {loadError} <button type="button" onClick={refresh}>Try again</button></p> : null}
    {loaded?.next_cursor ? <p role="status" className="knowledge-note">Some entries could not be loaded. Refresh to try again.</p> : null}
    {!loading && loaded && !entries.length ? <p className="project-panel-copy">{stateFilter === "proposed" ? "No proposals need review." : stateFilter === "history" ? "No rejected or superseded entries." : "No saved entries match this view."}</p> : null}
    <div className="knowledge-entries">{entries.map((entry) => <KnowledgeCard key={entry.id} entry={entry} maxTextChars={loaded?.limits.max_text_chars ?? 8000}
      replacement={entry.replaces_id ? loaded?.items.find((item) => item.id === entry.replaces_id) : undefined}
      onChanged={changed} onConflict={refresh} onOpenSource={onOpenSource}
      onEditingChange={(editing) => setEditingIds((current) => { const next = new Set(current); if (editing) next.add(entry.id); else next.delete(entry.id); return next; })} />)}</div>
  </section>;
}

function KnowledgeCard({ entry, replacement, maxTextChars, onChanged, onConflict, onOpenSource, onEditingChange }: {
  entry: KnowledgeEntry;
  replacement?: KnowledgeEntry;
  maxTextChars: number;
  onChanged: (entry?: KnowledgeEntry, deletedId?: string) => void;
  onConflict: () => void;
  onOpenSource?: (source: KnowledgeSourceRef, entry: KnowledgeEntry, index: number) => void;
  onEditingChange: (editing: boolean) => void;
}) {
  const active = useRef(true);
  useEffect(() => { active.current = true; return () => { active.current = false; }; }, []);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(entry.text);
  const [draftKind, setDraftKind] = useState(entry.kind);
  const [baseRevision, setBaseRevision] = useState(entry.revision);
  const [deleteRevision, setDeleteRevision] = useState<number | null>(null);
  const [history, setHistory] = useState<{ revision: number; items: KnowledgeEntry[] } | null>(null);
  const [historyOpen, setHistoryOpen] = useState(false);
  const [historyLoading, setHistoryLoading] = useState(false);
  const editable = entry.approval_state === "approved" || entry.approval_state === "proposed";
  const draftConflict = editing && baseRevision !== entry.revision;
  const confirmingDelete = deleteRevision === entry.revision;

  useEffect(() => {
    function disarmDelete() { setDeleteRevision(null); }
    window.addEventListener("blur", disarmDelete);
    document.addEventListener("visibilitychange", disarmDelete);
    return () => { window.removeEventListener("blur", disarmDelete); document.removeEventListener("visibilitychange", disarmDelete); };
  }, []);

  async function mutate(operation: () => Promise<KnowledgeEntry | void>) {
    if (busy) return;
    setBusy(true);
    setError("");
    setDeleteRevision(null);
    try {
      const updated = await operation();
      if (!active.current) return;
      onChanged(updated || undefined, updated ? undefined : entry.id);
      setEditing(false);
      onEditingChange(false);
      setHistory(null);
      setHistoryOpen(false);
    } catch (reason) {
      if (!active.current) return;
      if (reason instanceof ApiError && reason.status === 409) {
        setError("This entry changed. The latest saved version is being loaded; your draft is kept.");
        onConflict();
      } else setError(messageOf(reason));
    } finally {
      if (active.current) setBusy(false);
    }
  }
  async function showHistory() {
    if (historyOpen) { setHistoryOpen(false); return; }
    setHistoryOpen(true);
    if (history?.revision === entry.revision || historyLoading) return;
    setHistoryLoading(true);
    setError("");
    try {
      const result = await getProjectKnowledgeHistory(entry.project_id, entry.id);
      if (active.current) setHistory({ revision: entry.revision, items: result.items });
    } catch (reason) {
      if (active.current) { setError(messageOf(reason)); setHistoryOpen(false); }
    } finally {
      if (active.current) setHistoryLoading(false);
    }
  }
  function beginEdit() {
    setDraft(entry.text);
    setDraftKind(entry.kind);
    setBaseRevision(entry.revision);
    setEditing(true);
    onEditingChange(true);
    setError("");
    setDeleteRevision(null);
  }

  return <article className="knowledge-card" aria-label={`${kindLabel(entry.kind)}: ${entry.text.slice(0, 100)}`} data-state={entry.approval_state}>
    <header><div className="knowledge-labels"><strong>{kindLabel(entry.kind)}</strong>
      {entry.pinned ? <span>Pinned</span> : null}
      <span>{approvalLabel(entry.approval_state)}</span>
      {entry.kind === "question" || entry.kind === "action" ? <span>{entry.work_status === "open" ? "Open" : entry.work_status === "resolved" ? "Resolved" : "Completed"}</span> : null}
      {entry.stale ? <span className="knowledge-stale">Sources need review</span> : null}
    </div><small>Revision {entry.revision}</small></header>
    {editing ? <form className="knowledge-editor" onSubmit={(event) => { event.preventDefault(); if (!draftConflict && draft.trim()) void mutate(() => updateProjectKnowledge(entry.project_id, entry.id, { expected_revision: baseRevision, kind: draftKind, text: draft.trim() })); }}>
      <label>Entry type<select value={draftKind} disabled={busy} onChange={(event) => setDraftKind(event.target.value as KnowledgeKind)}>{KINDS.map((item) => <option key={item.value} value={item.value}>{item.label}</option>)}</select></label>
      <label>Edit entry text<textarea value={draft} rows={4} maxLength={maxTextChars} disabled={busy} onChange={(event) => setDraft(event.target.value)} required /></label>
      {draftConflict ? <div className="knowledge-conflict"><p>Latest saved text · Revision {entry.revision}</p><blockquote>{entry.text}</blockquote>
        <button type="button" onClick={() => { setBaseRevision(entry.revision); setError(""); }}>Keep my draft and use latest revision</button></div> : null}
      <div className="knowledge-actions"><button type="submit" disabled={busy || draftConflict || !draft.trim()}>{busy ? "Saving…" : "Save changes"}</button>
        <button type="button" disabled={busy} onClick={() => { setEditing(false); onEditingChange(false); setError(""); }}>Cancel edit</button></div>
    </form> : <p className="knowledge-text">{entry.text}</p>}
    {replacement ? <div className="knowledge-replacement"><p>Proposed replacement for revision {entry.base_revision ?? replacement.revision}</p><blockquote>{replacement.text}</blockquote>
      {entry.base_revision !== replacement.revision ? <p>The original entry changed after this proposal. Review both before approving.</p> : null}</div> : entry.replaces_id ? <p className="knowledge-note">The entry this proposal replaces is unavailable.</p> : null}
    <p className="knowledge-origin">{entry.origin === "user" ? "Written by you" : "Proposed from an agent report · Not independently verified"}{entry.source_task_id ? <span> · Agent {entry.source_task_id}</span> : null}<span> · Saved {formatDate(entry.updated_at)}</span></p>
    {entry.sources.length ? <details className="knowledge-sources"><summary>{entry.sources.length} {entry.sources.length === 1 ? "source reference" : "source references"}</summary>
      <ul>{entry.sources.map((source, index) => <li key={`${source.source_id}:${source.revision}`}>
        {onOpenSource && !entry.stale ? <button type="button" onClick={() => onOpenSource(source, entry, index)}>View source</button> : null}
        <span>{source.source_id} · Revision {source.revision}</span>
      </li>)}</ul></details> : entry.origin === "agent_report" ? <p className="knowledge-note">No inspected evidence is attached. The report identifies the origin of this suggestion.</p> : null}
    {entry.source_warnings.length ? <ul className="knowledge-warnings">{entry.source_warnings.map((warning, index) => <li key={index}>{warning}</li>)}</ul> : null}
    {error ? <p className="knowledge-error" role="alert">{error}</p> : null}
    <div className="knowledge-actions">
      {entry.approval_state === "proposed" ? <><button type="button" disabled={busy || editing} onClick={() => void mutate(() => decideProjectKnowledge(entry.project_id, entry.id, "approve", entry.revision))}>Approve proposal</button>
        <button type="button" disabled={busy || editing} onClick={() => void mutate(() => decideProjectKnowledge(entry.project_id, entry.id, "reject", entry.revision))}>Reject proposal</button></> : null}
      {editable && !editing ? <button type="button" onClick={beginEdit} disabled={busy}>Edit</button> : null}
      {entry.approval_state === "approved" ? <><button type="button" disabled={busy || editing} aria-pressed={entry.pinned} onClick={() => void mutate(() => updateProjectKnowledge(entry.project_id, entry.id, { expected_revision: entry.revision, pinned: !entry.pinned }))}>{entry.pinned ? "Unpin" : "Pin"}</button>
        {entry.kind === "question" || entry.kind === "action" ? <button type="button" disabled={busy || editing} onClick={() => void mutate(() => updateProjectKnowledge(entry.project_id, entry.id, { expected_revision: entry.revision, work_status: entry.work_status === "open" ? entry.kind === "question" ? "resolved" : "completed" : "open" }))}>
          {entry.work_status !== "open" ? "Reopen" : entry.kind === "question" ? "Resolve question" : "Complete action"}
        </button> : null}</> : null}
      <button type="button" disabled={busy || historyLoading} aria-expanded={historyOpen} onClick={() => void showHistory()}>{historyLoading ? "Loading history…" : historyOpen ? "Hide history" : "History"}</button>
      <button type="button" className={`danger-button${confirmingDelete ? " confirming-delete" : ""}`} disabled={busy || editing}
        aria-label={confirmingDelete ? "Confirm delete entry" : "Delete entry"} title={confirmingDelete ? "Press again to delete this entry and its history" : "Delete entry"}
        onBlur={() => setDeleteRevision(null)} onKeyDown={(event) => { if (event.key === "Escape") setDeleteRevision(null); }}
        onClick={() => { if (confirmingDelete) void mutate(() => deleteProjectKnowledge(entry.project_id, entry.id, entry.revision)); else setDeleteRevision(entry.revision); }}>{confirmingDelete ? "✓" : "Delete"}</button>
    </div>
    {historyOpen ? <div className="knowledge-history" aria-label="Entry revision history"><p>Saved revision history{history && history.revision !== entry.revision ? ` through revision ${history.revision}` : ""}</p>
      {historyLoading ? <p role="status">Loading revisions…</p> : history?.items.map((revision) => <section key={revision.revision}><h4>Revision {revision.revision} · {approvalLabel(revision.approval_state)}</h4><p>{revision.text}</p><small>{formatDate(revision.updated_at)}</small></section>)}
      {!historyLoading && !history?.items.length ? <p>No earlier revisions are available.</p> : null}
    </div> : null}
  </article>;
}

function kindLabel(kind: KnowledgeKind) { return KINDS.find((item) => item.value === kind)?.label ?? kind; }
function approvalLabel(state: KnowledgeEntry["approval_state"]) {
  return state === "proposed" ? "Needs approval" : state === "approved" ? "Approved" : state === "rejected" ? "Rejected" : "Superseded";
}
function formatDate(value: string) { const date = new Date(value); return Number.isNaN(date.getTime()) ? value : date.toLocaleString(); }
function messageOf(reason: unknown) { return reason instanceof Error ? reason.message : "The request could not be completed."; }
