import { useCallback, useEffect, useId, useRef, useState, type FormEvent } from "react";

import { ApiError, getProjectKnowledgeText, saveProjectKnowledgeText, type AudreyProject, type ProjectKnowledgeText } from "./api";
import "./ProjectKnowledgeSection.css";

type Props = { project: AudreyProject; visible?: boolean; onChanged?: () => void };

export function ProjectKnowledgeSection(props: Props) {
  return <ProjectKnowledge key={props.project.id} {...props} />;
}

function ProjectKnowledge({ project, visible = true, onChanged }: Props) {
  const headingId = useId(), helpId = useId();
  const [saved, setSaved] = useState<ProjectKnowledgeText | null>(null);
  const [text, setText] = useState("");
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");
  const [conflict, setConflict] = useState<ProjectKnowledgeText | null>(null);
  const [refresh, setRefresh] = useState(0);
  const baseline = useRef<ProjectKnowledgeText | null>(null);
  const draft = useRef("");
  const lifetime = useRef(0);
  const pending = useRef(false);
  const saveRequest = useRef<AbortController | null>(null);

  const accept = useCallback((value: ProjectKnowledgeText) => {
    baseline.current = value;
    draft.current = value.text;
    setSaved(value); setText(value.text); setConflict(null); setError("");
  }, []);

  useEffect(() => {
    const generation = ++lifetime.current;
    return () => { lifetime.current = generation + 1; saveRequest.current?.abort(); };
  }, []);

  useEffect(() => {
    if (!visible) return;
    const controller = new AbortController();
    queueMicrotask(() => { if (!controller.signal.aborted) setLoading(true); });
    void getProjectKnowledgeText(project.id, controller.signal).then((value) => {
      if (controller.signal.aborted) return;
      const previous = baseline.current;
      if (!previous || draft.current === previous.text || draft.current === value.text) accept(value);
      else if (previous.revision !== value.revision) {
        setConflict(value);
        setError("Project knowledge changed elsewhere. Your edits are kept.");
      } else { setConflict(null); setError(""); }
    }).catch((reason: unknown) => {
      if (!controller.signal.aborted) setError(messageOf(reason));
    }).finally(() => { if (!controller.signal.aborted) setLoading(false); });
    return () => controller.abort();
  }, [accept, project.id, visible, refresh]);

  async function save(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const current = baseline.current;
    if (!current || loading || text === current.text || pending.current || conflict || Array.from(text).length > current.max_text_chars) return;
    const generation = lifetime.current;
    const controller = new AbortController();
    saveRequest.current = controller;
    pending.current = true;
    setSaving(true); setError("");
    try {
      const value = await saveProjectKnowledgeText(project.id, text, current.revision, controller.signal);
      if (lifetime.current !== generation || controller.signal.aborted) return;
      accept(value);
      onChanged?.();
    } catch (reason) {
      if (lifetime.current !== generation || controller.signal.aborted) return;
      if (reason instanceof ApiError && reason.status === 409) {
        setError("Project knowledge changed elsewhere. Your edits are kept.");
        setRefresh((value) => value + 1);
      } else setError(messageOf(reason));
    } finally {
      if (lifetime.current === generation && !controller.signal.aborted) {
        pending.current = false;
        setSaving(false);
      }
    }
  }

  const count = Array.from(text).length;
  const limit = saved?.max_text_chars ?? 8000;
  const tooLong = count > limit;
  const dirty = saved !== null && text !== saved.text;

  return <section className="project-panel project-knowledge" aria-labelledby={headingId}>
    <header><h2 id={headingId}>Project knowledge</h2></header>
    <form className="project-knowledge-form" onSubmit={(event) => void save(event)}>
      <p id={helpId} className="project-panel-copy">Notes for this project's {project.private_required ? "chats" : "chats and agents"}. Long notes may be shortened.</p>
      <textarea aria-label="Project knowledge" aria-describedby={helpId} value={text} rows={6}
        placeholder="Add background, preferences and details Audrey should remember."
        disabled={!saved || loading || saving}
        onChange={(event) => { draft.current = event.target.value; setText(event.target.value); }} />
      {tooLong ? <p className="knowledge-error" role="alert">Shorten this text to {limit.toLocaleString()} characters before saving. No text has been removed.</p> : null}
      {error ? <div className="knowledge-error" role="alert">{error}
        {!conflict ? <button type="button" disabled={loading || saving} onClick={() => setRefresh((value) => value + 1)}>Reload saved text</button> : null}
      </div> : null}
      {conflict ? <div className="knowledge-conflict">
        <p>Latest saved text</p><blockquote>{conflict.text || "No saved knowledge."}</blockquote>
        <div>
          <button type="button" onClick={() => accept(conflict)}>Use saved text</button>
          <button type="button" onClick={() => { baseline.current = conflict; setSaved(conflict); setConflict(null); setError(""); }}>Keep my edits</button>
        </div>
      </div> : null}
      <footer>
        <span role="status">{loading ? "Loading…" : saving ? "Saving…" : dirty ? "Unsaved changes" : saved ? "Saved" : "Not loaded"}</span>
        <small>{count.toLocaleString()} / {limit.toLocaleString()}</small>
        <button type="submit" disabled={!dirty || saving || loading || !!conflict || tooLong}>Save</button>
      </footer>
    </form>
  </section>;
}

function messageOf(reason: unknown): string {
  return reason instanceof Error ? reason.message : "Project knowledge is unavailable. Try again.";
}
