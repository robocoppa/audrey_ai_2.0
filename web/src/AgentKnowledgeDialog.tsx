import { useEffect, useId, useRef, useState, type FormEvent } from "react";
import { createPortal } from "react-dom";

import {
  createKnowledgeProposal, listProjectKnowledge, listProjects,
  type AgentTask, type AudreyProject, type KnowledgeEntry, type KnowledgeKind,
} from "./api";
import "./AgentKnowledgeDialog.css";

export function AgentKnowledgeDialog({ task, onClose }: { task: AgentTask; onClose: () => void }) {
  return <ProposalDialog key={task.id} task={task} onClose={onClose} />;
}

function ProposalDialog({ task, onClose }: { task: AgentTask; onClose: () => void }) {
  const titleId = useId();
  const active = useRef(true);
  const dialogRef = useRef<HTMLElement>(null);
  const closeRef = useRef<HTMLButtonElement>(null);
  const closeAction = useRef(onClose);
  const busyRef = useRef(false);
  const [projects, setProjects] = useState<AudreyProject[]>([]);
  const [projectsLoading, setProjectsLoading] = useState(true);
  const [projectsError, setProjectsError] = useState("");
  const [projectsPartial, setProjectsPartial] = useState(false);
  const [projectLoad, setProjectLoad] = useState(0);
  const [projectId, setProjectId] = useState("");
  const [kind, setKind] = useState<KnowledgeKind>("fact");
  const [text, setText] = useState("");
  const [pinned, setPinned] = useState(false);
  const [replacement, setReplacement] = useState<KnowledgeEntry | null>(null);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [error, setError] = useState("");
  const request = useRef<{ fingerprint: string; id: string } | null>(null);

  useEffect(() => { closeAction.current = onClose; }, [onClose]);
  useEffect(() => {
    active.current = true;
    const previousFocus = document.activeElement;
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    closeRef.current?.focus();
    function keydown(event: KeyboardEvent) {
      if (event.key === "Escape" && !busyRef.current) {
        event.preventDefault();
        closeAction.current();
      }
      if (event.key !== "Tab") return;
      const controls = [...(dialogRef.current?.querySelectorAll<HTMLElement>("button:not([disabled]), select:not([disabled]), textarea:not([disabled]), input:not([disabled]), [tabindex='0']") ?? [])];
      const first = controls[0];
      const last = controls.at(-1);
      if (event.shiftKey && (document.activeElement === first || !dialogRef.current?.contains(document.activeElement))) {
        event.preventDefault(); last?.focus();
      } else if (!event.shiftKey && (document.activeElement === last || !dialogRef.current?.contains(document.activeElement))) {
        event.preventDefault(); first?.focus();
      }
    }
    document.addEventListener("keydown", keydown);
    return () => {
      active.current = false;
      document.removeEventListener("keydown", keydown);
      document.body.style.overflow = previousOverflow;
      if (previousFocus instanceof HTMLElement && previousFocus.isConnected) previousFocus.focus();
    };
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    async function loadProjects() {
      try {
        const first = await listProjects();
        const items = [...first.items];
        let cursor = first.next_cursor;
        if (cursor && !controller.signal.aborted) {
          const second = await listProjects(cursor);
          items.push(...second.items);
          cursor = second.next_cursor;
        }
        if (!controller.signal.aborted) {
          setProjects([...new Map(items.slice(0, 200).map((project) => [project.id, project])).values()]);
          setProjectsPartial(Boolean(cursor));
          setProjectsError("");
          setProjectsLoading(false);
        }
      } catch (reason) {
        if (!controller.signal.aborted) {
          setProjectsError(messageOf(reason));
          setProjectsLoading(false);
        }
      }
    }
    void loadProjects();
    return () => controller.abort();
  }, [projectLoad]);

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (saving || !projectId || !text.trim() || saved) return;
    const payload = {
      task_id: task.id, task_revision: task.revision, kind, text: text.trim(), pinned,
      ...(replacement ? { replaces_id: replacement.id, base_revision: replacement.revision } : {}),
    };
    const fingerprint = JSON.stringify({ projectId, ...payload });
    if (request.current?.fingerprint !== fingerprint) request.current = { fingerprint, id: crypto.randomUUID() };
    setSaving(true);
    busyRef.current = true;
    setError("");
    try {
      await createKnowledgeProposal(projectId, { ...payload, request_id: request.current.id });
      if (active.current) setSaved(true);
    } catch (reason) {
      if (active.current) setError(messageOf(reason));
    } finally {
      busyRef.current = false;
      if (active.current) setSaving(false);
    }
  }

  return createPortal(<div className="agent-knowledge-backdrop" role="presentation" onMouseDown={(event) => { if (event.currentTarget === event.target && !saving) onClose(); }}>
    <section ref={dialogRef} className="agent-knowledge-dialog" role="dialog" aria-modal="true" aria-labelledby={titleId}>
      <header><div><span>Project knowledge</span><h2 id={titleId}>Propose from this report</h2></div>
        <button ref={closeRef} type="button" aria-label="Close knowledge proposal" onClick={onClose} disabled={saving}>×</button></header>
      {saved ? <div className="agent-knowledge-saved" role="status"><p>Proposal saved. Approve it in Project knowledge before reuse.</p><button type="button" onClick={onClose}>Done</button></div> : <form onSubmit={(event) => void submit(event)}>
        <p>Choose a statement from the report to review and save as a proposal. The report records where it came from; it does not independently verify the statement.</p>
        <label>Project<select value={projectId} disabled={saving || projectsLoading} required onChange={(event) => { setProjectId(event.target.value); setReplacement(null); setError(""); }}>
          <option value="">{projectsLoading ? "Loading projects…" : "Choose a project"}</option>
          {projects.map((project) => <option key={project.id} value={project.id}>{project.name}{project.private_required ? " · Private" : ""}</option>)}
        </select></label>
        {projectsError ? <p className="agent-knowledge-error" role="alert">Could not load projects. {projectsError} <button type="button" onClick={() => setProjectLoad((value) => value + 1)}>Try again</button></p> : null}
        {!projectsLoading && !projects.length && !projectsError ? <p>Create a project first, then return to this report.</p> : null}
        {projectsPartial ? <p className="agent-knowledge-note">Only the first 200 projects are shown.</p> : null}
        <label>Entry type<select value={kind} disabled={saving} onChange={(event) => setKind(event.target.value as KnowledgeKind)}>
          <option value="fact">Fact</option><option value="decision">Decision</option><option value="question">Question</option><option value="action">Action</option>
        </select></label>
        <label>Proposed text<textarea value={text} rows={5} maxLength={8000} required disabled={saving} placeholder="Write or paste the part you want to keep." onChange={(event) => setText(event.target.value)} /></label>
        {projectId ? <ReplacementPicker key={projectId} projectId={projectId} disabled={saving} replacement={replacement} onSelect={setReplacement} /> : null}
        <label className="agent-knowledge-check"><input type="checkbox" checked={pinned} disabled={saving} onChange={(event) => setPinned(event.target.checked)} />Pin after approval</label>
        <p className="agent-knowledge-note">This saves a pending proposal. It does not approve knowledge, start an action or run a model.</p>
        {error ? <p role="alert" className="agent-knowledge-error">{error}</p> : null}
        <div className="agent-knowledge-actions"><button type="button" disabled={saving} onClick={onClose}>Cancel</button><button type="submit" disabled={saving || !projectId || !text.trim()}>{saving ? "Saving…" : "Save proposal"}</button></div>
      </form>}
    </section>
  </div>, document.body);
}

function ReplacementPicker({ projectId, disabled, replacement, onSelect }: {
  projectId: string; disabled: boolean; replacement: KnowledgeEntry | null; onSelect: (entry: KnowledgeEntry | null) => void;
}) {
  const [entries, setEntries] = useState<KnowledgeEntry[] | null>(null);
  const [error, setError] = useState("");
  const [partial, setPartial] = useState(false);
  const [version, setVersion] = useState(0);
  useEffect(() => {
    const controller = new AbortController();
    async function load() {
      try {
        const first = await listProjectKnowledge(projectId, undefined, controller.signal);
        const items = [...first.items];
        const seen = new Set<string>();
        let cursor = first.next_cursor;
        for (let page = 0; cursor && page < 3 && items.length < 200; page++) {
          if (seen.has(cursor)) throw new Error("The entry list could not be loaded completely.");
          seen.add(cursor);
          const next = await listProjectKnowledge(projectId, cursor, controller.signal);
          items.push(...next.items);
          cursor = next.next_cursor;
        }
        if (!controller.signal.aborted) {
          setEntries(items.filter((entry) => entry.approval_state === "approved"));
          setPartial(Boolean(cursor));
          setError("");
        }
      } catch (reason) { if (!controller.signal.aborted) setError(messageOf(reason)); }
    }
    void load();
    return () => controller.abort();
  }, [projectId, version]);

  return <div className="agent-knowledge-replacement"><label>Replace an approved entry <small>Optional</small>
    <select value={replacement?.id ?? ""} disabled={disabled || !entries} onChange={(event) => onSelect(entries?.find((entry) => entry.id === event.target.value) ?? null)}>
      <option value="">{!entries && !error ? "Loading entries…" : "Save as a new entry"}</option>
      {entries?.map((entry) => <option value={entry.id} key={entry.id}>{entry.kind} · {entry.text.slice(0, 100)}</option>)}
    </select></label>
    {replacement ? <blockquote>{replacement.text}<small>Current revision {replacement.revision}. It stays active until the replacement is approved.</small></blockquote> : null}
    {error ? <p className="agent-knowledge-error" role="alert">Could not load replacement choices. <button type="button" disabled={disabled} onClick={() => setVersion((value) => value + 1)}>Try again</button></p> : null}
    {partial ? <p className="agent-knowledge-note">Some replacement choices could not be loaded.</p> : null}
  </div>;
}

function messageOf(reason: unknown) { return reason instanceof Error ? reason.message : "The request could not be completed."; }
