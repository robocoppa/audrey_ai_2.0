import { useEffect, useState } from "react";

import { listProjectFiles, type AgentFileInputs, type AudreyProjectFile } from "./api";

interface AgentFilePickerProps {
  projectId: string;
  selected: AudreyProjectFile[];
  onChange: (files: AudreyProjectFile[]) => void;
  visible: boolean;
  disabled: boolean;
}

export function AgentFilePicker({ projectId, selected, onChange, visible, disabled }: AgentFilePickerProps) {
  const [expanded, setExpanded] = useState(false);
  const [files, setFiles] = useState<AudreyProjectFile[]>([]);
  const [loading, setLoading] = useState(false);
  const [loaded, setLoaded] = useState(false);
  const [error, setError] = useState("");
  const [refresh, setRefresh] = useState(0);
  const [documentVisible, setDocumentVisible] = useState(() => document.visibilityState !== "hidden");

  useEffect(() => {
    const update = () => setDocumentVisible(document.visibilityState !== "hidden");
    document.addEventListener("visibilitychange", update);
    return () => document.removeEventListener("visibilitychange", update);
  }, []);

  useEffect(() => {
    if (!visible || !expanded || !documentVisible) return;
    const controller = new AbortController();
    queueMicrotask(() => {
      if (!controller.signal.aborted) { setLoading(true); setLoaded(false); setError(""); }
    });
    void listProjectFiles(projectId, controller.signal).then((page) => {
      if (!controller.signal.aborted) {
        setFiles(page.items.filter((file) => file.status === "ready"));
        setLoaded(true);
      }
    }).catch(() => {
      if (!controller.signal.aborted) setError("Could not load Project files. Your selections are kept.");
    }).finally(() => { if (!controller.signal.aborted) setLoading(false); });
    return () => controller.abort();
  }, [projectId, visible, expanded, documentVisible, refresh]);

  const missing = loaded ? selected.filter((file) => !files.some(({ id }) => id === file.id)) : [];
  return <div className="agents-file-picker">
    <button className="agents-secondary" type="button" disabled={disabled}
      aria-expanded={expanded} aria-controls="agent-project-file-choices" onClick={() => setExpanded((value) => !value)}>
      Project files <span className="agents-optional">{selected.length ? `${selected.length} / 3 selected` : "Optional"}</span>
    </button>
    {!expanded && selected.length ? <p className="agents-file-selection">{selected.map(({ filename }) => filename).join(", ")}</p> : null}
    {expanded ? <div id="agent-project-file-choices">
      <p id="agent-project-files-help">Choose up to 3 files. Audrey starts with short extracts and can read more saved text or transcripts when needed. Changed or removed files stop the task from continuing.</p>
      {loading ? <p role="status">Loading Project files…</p> : null}
      {error ? <p role="alert">{error}</p> : null}
      {loaded && !files.length ? <p>No ready files in this Project. Add files from the Project screen.</p> : null}
      {loaded ? <div role="group" aria-label="Choose Project files" aria-describedby="agent-project-files-help" className="agents-file-choices">
        {files.map((file) => {
          const checked = selected.some(({ id }) => id === file.id);
          return <label key={file.id}>
            <input type="checkbox" aria-label={file.filename} aria-describedby={`agent-file-${file.id}-meta`} checked={checked} disabled={disabled || (!checked && selected.length >= 3)} onChange={() => onChange(checked ? selected.filter(({ id }) => id !== file.id) : [...selected, file])} />
            <span><span className="agents-file-name">{file.filename}</span><span className="agents-file-meta" id={`agent-file-${file.id}-meta`}>{file.kind === "audio" || file.kind === "video" ? "Transcript" : "Saved text"}{file.origin === "generated" ? " · Working output" : ""}</span></span>
          </label>;
        })}
      </div> : null}
      {missing.map((file) => <p className="agents-file-missing" key={file.id}><span>{file.filename} is no longer ready in this Project.</span> <button className="agents-refresh" type="button" disabled={disabled} onClick={() => onChange(selected.filter(({ id }) => id !== file.id))}>Remove {file.filename}</button></p>)}
      {selected.length >= 3 ? <p>Three files selected. Remove one to choose another.</p> : null}
      <button className="agents-refresh" type="button" disabled={disabled || loading} onClick={() => setRefresh((value) => value + 1)}>Refresh files</button>
      {selected.length ? <button className="agents-refresh" type="button" disabled={disabled} onClick={() => onChange([])}>Clear selected files</button> : null}
    </div> : null}
  </div>;
}

export function AgentFileInputsView({ inputs }: { inputs?: AgentFileInputs }) {
  if (!inputs?.files.length) return null;
  return <details className="agents-project-context agents-file-inputs">
    <summary>{inputs.current ? `Input files · ${inputs.files.length}` : "Input files changed"}</summary>
    <p>This task starts with the extracts below and can read more of the same saved file versions when needed.</p>
    {!inputs.current ? <p className="agents-project-unavailable">A selected file changed or is unavailable. Start a new agent and choose its files again to continue.</p> : null}
    <ul>{inputs.files.map((file) => <li key={file.file_id}>
      <strong className="agents-file-name">{file.filename}</strong>
      <span className="agents-file-meta">{file.artifact === "transcript" ? "Transcript" : "Saved text"}{file.origin === "generated" ? " · Working output" : ""} · {file.chars.toLocaleString()} {file.truncated ? `of ${file.total_chars.toLocaleString()} characters · Shortened` : "characters"}</span>
    </li>)}</ul>
  </details>;
}
