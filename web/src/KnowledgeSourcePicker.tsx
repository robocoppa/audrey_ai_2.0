import { useEffect, useId, useRef, useState } from "react";

import {
  ApiError,
  getKnowledgeSourcePreview,
  listKnowledgeSourceCandidates,
  type KnowledgeSourceCandidate,
  type KnowledgeSourceRef,
  type SearchSource,
} from "./api";
import "./KnowledgeSourcePicker.css";

type Props = {
  projectId: string;
  sources: KnowledgeSourceRef[];
  onChange: (sources: KnowledgeSourceRef[]) => void;
  disabled?: boolean;
  visible?: boolean;
};
type SourceKind = KnowledgeSourceCandidate["kind"];
type PreviewRequest = {
  sourceId: string;
  revision?: string;
  originalRevision?: string;
  kind?: SourceKind;
  mode: "attach" | "saved" | "refresh";
  offset: number;
  previous: number[];
};
const MAX_SOURCES = 8;
const MAX_CANDIDATES = 500;
const MAX_PAGES = 20;

export function KnowledgeSourcePicker(props: Props) {
  return <SourcePicker key={props.projectId} {...props} />;
}

function SourcePicker({ visible = true, disabled = false, ...props }: Props) {
  const [expanded, setExpanded] = useState(false);
  const panelId = useId();
  if (!visible) return null;
  return <section className="knowledge-source-picker" aria-label="Knowledge source references">
    <button type="button" aria-expanded={expanded} aria-controls={panelId} disabled={disabled}
      onClick={() => setExpanded((current) => !current)}>Source references ({props.sources.length}/{MAX_SOURCES})</button>
    {expanded && !disabled ? <div id={panelId}><SourcePanel {...props} /></div> : null}
  </section>;
}

function SourcePanel({ projectId, sources, onChange }: Omit<Props, "disabled" | "visible">) {
  const [kind, setKind] = useState<SourceKind>("file");
  const [browseRequest, setBrowseRequest] = useState({ kind: "file" as SourceKind, cursor: undefined as string | undefined, serial: 0 });
  const [candidates, setCandidates] = useState<KnowledgeSourceCandidate[]>([]);
  const [nextCursor, setNextCursor] = useState<string | null>(null);
  const [browseBusy, setBrowseBusy] = useState(true);
  const [browseError, setBrowseError] = useState("");
  const [browseBounded, setBrowseBounded] = useState(false);
  const candidateItems = useRef<KnowledgeSourceCandidate[]>([]);
  const cursors = useRef(new Set<string>());
  const pages = useRef(0);
  const [previewRequest, setPreviewRequest] = useState<PreviewRequest | null>(null);
  const [previewResult, setPreviewResult] = useState<{ request: PreviewRequest; source: SearchSource } | null>(null);
  const [previewError, setPreviewError] = useState<{ request: PreviewRequest; message: string } | null>(null);
  const previewController = useRef<AbortController | null>(null);
  const headingId = useId();
  const selectId = useId();

  useEffect(() => {
    const controller = new AbortController();
    async function load() {
      try {
        const result = await listKnowledgeSourceCandidates(projectId, browseRequest.kind, browseRequest.cursor, controller.signal);
        if (controller.signal.aborted) return;
        if (result.items.some((item) => item.kind !== browseRequest.kind || !item.source_id)) {
          throw new Error("The source list changed unexpectedly. Try again.");
        }
        if (browseRequest.cursor) cursors.current.add(browseRequest.cursor);
        pages.current += 1;
        const repeated = Boolean(result.next_cursor && cursors.current.has(result.next_cursor));
        const merged = [...new Map([...candidateItems.current, ...result.items].map((item) => [item.source_id, item])).values()];
        const bounded = pages.current >= MAX_PAGES || merged.length >= MAX_CANDIDATES;
        candidateItems.current = merged.slice(0, MAX_CANDIDATES);
        setCandidates(candidateItems.current);
        setBrowseBounded(bounded);
        setNextCursor(repeated || bounded ? null : result.next_cursor);
        setBrowseError(repeated ? "The source list repeated a page. Refresh the source list to continue." : "");
      } catch (reason) {
        if (!controller.signal.aborted) setBrowseError(reason instanceof ApiError && reason.status === 422 && browseRequest.cursor
          ? "The source list changed. Refresh the source list to continue."
          : sourceError(reason, "The source list could not be loaded. Try again."));
      } finally {
        if (!controller.signal.aborted) setBrowseBusy(false);
      }
    }
    void load();
    return () => controller.abort();
  }, [browseRequest, projectId]);

  useEffect(() => {
    if (!previewRequest) return;
    const controller = new AbortController();
    previewController.current = controller;
    const request = previewRequest;
    async function load() {
      try {
        const source = await getKnowledgeSourcePreview(projectId, request.sourceId, request.revision, request.offset, controller.signal);
        if (controller.signal.aborted) return;
        if (source.source_id !== request.sourceId || !source.revision || (request.revision !== undefined && source.revision !== request.revision)
          || (request.kind && source.kind !== request.kind) || !["file", "message", "agent"].includes(source.kind)
          || source.target.kind !== source.kind || source.offset !== request.offset
          || (source.next_offset !== null && (source.next_offset <= source.offset || request.previous.includes(source.next_offset)))) {
          throw new ApiError(409, "The saved source changed.");
        }
        setPreviewResult({ request, source });
      } catch (reason) {
        if (!controller.signal.aborted) setPreviewError({ request, message: sourceError(reason, "The source preview could not be loaded. Try again.") });
      }
    }
    void load();
    return () => { controller.abort(); if (previewController.current === controller) previewController.current = null; };
  }, [previewRequest, projectId]);

  function changeKind(value: SourceKind) {
    previewController.current?.abort();
    setPreviewRequest(null);
    setKind(value);
    setCandidates([]);
    candidateItems.current = [];
    setNextCursor(null);
    setBrowseBusy(true);
    setBrowseError("");
    setBrowseBounded(false);
    cursors.current.clear();
    pages.current = 0;
    setBrowseRequest((current) => ({ kind: value, cursor: undefined, serial: current.serial + 1 }));
  }
  function loadMore() {
    if (browseBusy || !nextCursor || cursors.current.has(nextCursor)) return;
    setBrowseBusy(true);
    setBrowseError("");
    setBrowseRequest((current) => ({ ...current, cursor: nextCursor, serial: current.serial + 1 }));
  }
  function closePreview() {
    previewController.current?.abort();
    setPreviewRequest(null);
    setPreviewResult(null);
    setPreviewError(null);
  }
  function preview(request: Omit<PreviewRequest, "offset" | "previous">) {
    previewController.current?.abort();
    setPreviewRequest({ ...request, offset: 0, previous: [] });
  }
  const source = previewResult?.request === previewRequest ? previewResult.source : null;
  const sourceTextLength = source ? [...source.text].length : 0;
  const error = previewError?.request === previewRequest ? previewError.message : "";
  const attached = source ? sources.find((item) => item.source_id === source.source_id) : undefined;
  const mayRefresh = Boolean(source && previewRequest?.mode === "refresh" && attached?.revision === previewRequest.originalRevision);
  const canAttach = Boolean(source && !attached && sources.length < MAX_SOURCES && previewRequest?.mode === "attach");

  return <div className="knowledge-source-panel">
    <p className="knowledge-note">References keep a source ID and revision. Review the saved text before attaching; no text is copied into your entry.</p>
    {sources.length ? <ul className="knowledge-source-selected" aria-label="Attached source references">{sources.map((item) => <li key={item.source_id}>
      <span>{item.source_id}<small>Revision {item.revision}</small></span>
      <div>
        <button type="button" aria-label={`View saved version of ${item.source_id}`} onClick={() => preview({ sourceId: item.source_id, revision: item.revision, mode: "saved" })}>View saved version</button>
        <button type="button" aria-label={`Review current version of ${item.source_id}`} onClick={() => preview({ sourceId: item.source_id, originalRevision: item.revision, mode: "refresh" })}>Review current version</button>
        <button type="button" aria-label={`Remove reference to ${item.source_id}`} onClick={() => onChange(sources.filter((ref) => ref.source_id !== item.source_id))}>Remove</button>
      </div>
    </li>)}</ul> : <p className="knowledge-note">No source references attached.</p>}
    {sources.length >= MAX_SOURCES ? <p className="knowledge-note" role="status">Eight references are attached. Remove one before attaching another.</p> : null}
    <label className="knowledge-source-type" htmlFor={selectId}>Source type<select id={selectId} value={kind} onChange={(event) => changeKind(event.target.value as SourceKind)}>
      <option value="file">Files</option><option value="message">Chat messages</option><option value="agent">Agent reports</option>
    </select></label>
    <p className="knowledge-note">{kind === "agent" ? "Your saved agent reports, including reports outside this project." : kind === "message" ? "Saved messages in this project's conversations." : "Saved file text linked to this project."}</p>
    <div className="knowledge-source-candidates" aria-label="Available source references" aria-busy={browseBusy}>
      {candidates.map((item) => <button type="button" key={item.source_id} aria-label={`Preview ${item.title || "Untitled source"}`}
        onClick={() => preview({ sourceId: item.source_id, kind: item.kind, mode: "attach" })}>
        <span>{item.title || "Untitled source"}</span>{item.private_required ? <small>Private</small> : null}
      </button>)}
      {!browseBusy && !browseError && !candidates.length ? <p className="knowledge-note">No saved sources of this type are available.</p> : null}
    </div>
    {browseBusy ? <p className="knowledge-note" role="status">Loading sources…</p> : null}
    {browseError ? <div className="knowledge-error" role="alert"><p>{browseError}</p><div className="knowledge-source-actions">
      {!browseError.includes("repeated") ? <button type="button" onClick={() => { setBrowseBusy(true); setBrowseError(""); setBrowseRequest((current) => ({ ...current, serial: current.serial + 1 })); }}>Retry source list</button> : null}
      {browseRequest.cursor ? <button type="button" onClick={() => changeKind(kind)}>Refresh source list</button> : null}
    </div></div> : null}
    {nextCursor ? <button type="button" disabled={browseBusy} onClick={loadMore}>Load more sources</button> : null}
    {browseBounded ? <p className="knowledge-note">This view reached its source limit. Choose another source type or reopen to refresh.</p> : null}
    {previewRequest ? <section className="knowledge-source-preview" aria-labelledby={headingId}>
      <header><h4 id={headingId}>{source?.title || "Source preview"}</h4><button type="button" onClick={closePreview}>Close preview</button></header>
      {!source && !error ? <p className="knowledge-note" role="status">Loading source preview…</p> : null}
      {error ? <p className="knowledge-error" role="alert">{error} Your attached references are unchanged. <button type="button" onClick={() => setPreviewRequest({ ...previewRequest })}>Retry preview</button></p> : null}
      {source ? <>
        <p className="knowledge-note">{source.kind === "agent" ? "Generated report · " : source.kind === "file" ? "File text · " : "Saved chat message · "}Revision {source.revision}{source.private_required ? " · Private" : ""}</p>
        {source.availability === "text_only" ? <p className="knowledge-note">Original unavailable; retained text shown.</p> : null}
        {previewRequest.mode === "refresh" ? <p className="knowledge-note">Attached revision: {previewRequest.originalRevision}. Current revision: {source.revision}. The reference changes only when you choose Use current revision.</p> : null}
        <blockquote aria-label="Source preview text">{source.text}</blockquote>
        <div className="knowledge-source-pagination">
          <span>{source.offset + (sourceTextLength ? 1 : 0)}–{source.offset + sourceTextLength} of {source.total_chars} characters</span>
          {previewRequest.previous.length ? <button type="button" onClick={() => setPreviewRequest({ ...previewRequest, revision: source.revision, offset: previewRequest.previous.at(-1)!, previous: previewRequest.previous.slice(0, -1) })}>Previous source text</button> : null}
          {source.next_offset !== null ? <button type="button" onClick={() => setPreviewRequest({ ...previewRequest, revision: source.revision, offset: source.next_offset!, previous: [...previewRequest.previous, source.offset] })}>Next source text</button> : null}
        </div>
        <div className="knowledge-source-actions">
          {previewRequest.mode === "attach" ? <button type="button" disabled={!canAttach} onClick={() => { if (canAttach) onChange([...sources, { source_id: source.source_id, revision: source.revision }]); }}>{attached ? "Reference attached" : "Attach reference"}</button> : null}
          {previewRequest.mode === "refresh" ? <button type="button" disabled={!mayRefresh || source.revision === attached?.revision} onClick={() => { if (mayRefresh) onChange(sources.map((item) => item.source_id === source.source_id ? { source_id: source.source_id, revision: source.revision } : item)); }}>Use current revision</button> : null}
        </div>
      </> : null}
    </section> : null}
  </div>;
}

function sourceError(reason: unknown, fallback: string) {
  if (reason instanceof ApiError) {
    if (reason.status === 409) return "This source changed. Review its current version before replacing a reference.";
    if ([403, 404].includes(reason.status)) return "This source was deleted or is no longer available.";
  }
  return fallback;
}
