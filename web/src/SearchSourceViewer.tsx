import { useEffect, useId, useRef, useState } from "react";
import { createPortal } from "react-dom";

import { ApiError, getSearchSource, type SearchResult, type SearchSource } from "./api";

import "./SearchSourceViewer.css";

const PAGE_SIZE = 4000;

type ViewerProps = {
  result: SearchResult;
  onClose: () => void;
  loadSource?: (offset: number, signal: AbortSignal) => Promise<SearchSource>;
  presentation?: "modal" | "inline";
  onOpenWorkspace?: (result: SearchResult, signal?: AbortSignal) => void | Promise<void>;
};

export function SearchSourceViewer(props: ViewerProps) {
  return <SourceViewer key={JSON.stringify([props.result.id, props.result.source_id, props.result.revision, props.result.kind, props.result.target])} {...props} />;
}

function SourceViewer({ result, onClose, loadSource, presentation = "modal", onOpenWorkspace }: ViewerProps) {
  const [request, setRequest] = useState({ offset: result.offset, serial: 0 });
  const [source, setSource] = useState<SearchSource | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [retryable, setRetryable] = useState(false);
  const [opening, setOpening] = useState(false);
  const [navigationError, setNavigationError] = useState("");
  const navigation = useRef<AbortController | null>(null);
  const dialogRef = useRef<HTMLElement>(null);
  const closeRef = useRef<HTMLButtonElement>(null);
  const close = useRef(onClose);
  const titleId = useId();
  const inline = presentation === "inline";

  useEffect(() => {
    close.current = () => {
      navigation.current?.abort();
      onClose();
    };
  }, [onClose]);

  useEffect(() => () => { navigation.current?.abort(); }, []);

  useEffect(() => {
    if (inline) {
      closeRef.current?.focus();
      return;
    }
    const previousFocus = document.activeElement;
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    closeRef.current?.focus();
    function onKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") {
        event.preventDefault();
        close.current();
      } else if (event.key === "Tab") {
        const items = [...(dialogRef.current?.querySelectorAll<HTMLElement>("button:not([disabled]), [tabindex='0']") ?? [])];
        const first = items[0];
        const last = items.at(-1);
        if (event.shiftKey && (document.activeElement === first || !dialogRef.current?.contains(document.activeElement))) {
          event.preventDefault();
          last?.focus();
        } else if (!event.shiftKey && (document.activeElement === last || !dialogRef.current?.contains(document.activeElement))) {
          event.preventDefault();
          first?.focus();
        }
      }
    }
    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.removeEventListener("keydown", onKeyDown);
      document.body.style.overflow = previousOverflow;
      if (previousFocus instanceof HTMLElement && previousFocus.isConnected) previousFocus.focus();
    };
  }, [inline]);

  useEffect(() => {
    const controller = new AbortController();
    let current = true;
    queueMicrotask(() => {
      if (!current) return;
      setSource(null);
      setLoading(true);
      setError("");
      setNavigationError("");
      setOpening(false);
    });
    async function load() {
      try {
        const fresh = await (loadSource ? loadSource(request.offset, controller.signal) : getSearchSource(result, request.offset, controller.signal));
        if (!current || controller.signal.aborted) return;
        if (fresh.source_id !== result.source_id || fresh.revision !== result.revision || fresh.kind !== result.kind || fresh.target.kind !== fresh.kind || !sameTarget(fresh.target, result.target)) {
          throw new ApiError(409, "The saved source changed.");
        }
        setSource(fresh);
        setError("");
      } catch (reason) {
        if (!current || controller.signal.aborted) return;
        const boundaryError = sourceBoundaryError(reason, Boolean(loadSource));
        if (boundaryError) {
          setSource(null);
          setError(boundaryError);
          setRetryable(false);
        } else {
          setError(reason instanceof Error ? reason.message : "The saved source could not be loaded.");
          setRetryable(true);
        }
      } finally {
        if (current && !controller.signal.aborted) setLoading(false);
      }
    }
    void load();
    return () => { current = false; controller.abort(); navigation.current?.abort(); };
  }, [request, result, loadSource]);

  function loadPage(offset: number) {
    navigation.current?.abort();
    navigation.current = null;
    setOpening(false);
    setNavigationError("");
    closeRef.current?.focus();
    setLoading(true);
    setError("");
    setSource(null);
    setRequest((saved) => ({ offset, serial: saved.serial + 1 }));
  }

  async function openWorkspace() {
    if (!onOpenWorkspace || !source || loading || error || navigation.current && !navigation.current.signal.aborted) return;
    const controller = new AbortController();
    navigation.current = controller;
    setOpening(true);
    setNavigationError("");
    try {
      await onOpenWorkspace({ ...result, target: source.target, title: source.title, availability: source.availability, private_required: source.private_required }, controller.signal);
    } catch (reason) {
      if (!controller.signal.aborted && navigation.current === controller) {
        const boundaryError = sourceBoundaryError(reason, Boolean(loadSource));
        if (boundaryError) {
          setSource(null);
          setError(boundaryError);
          setRetryable(false);
          setNavigationError("");
          closeRef.current?.focus();
        } else {
          setNavigationError(reason instanceof Error ? reason.message : "The workspace could not be opened. Try again.");
        }
      }
    } finally {
      if (!controller.signal.aborted && navigation.current === controller) {
        navigation.current = null;
        setOpening(false);
      }
    }
  }

  const viewer = (
      <section className={`search-source-dialog${inline ? " search-source-inline" : ""}`} role={inline ? undefined : "dialog"} aria-modal={inline ? undefined : true} aria-labelledby={titleId} ref={dialogRef}>
        <header>
          <div>
            <span>Saved source · {result.kind === "message" ? "Chat message" : result.kind === "file" ? "File text" : result.kind === "knowledge" ? "Project knowledge" : "Agent work"}</span>
            <h2 id={titleId}>{source?.title || result.title || "Untitled"}</h2>
          </div>
          <button type="button" ref={closeRef} onClick={() => close.current()} aria-label={inline ? "Back to search results" : "Close saved source"}>{inline ? "←" : "×"}</button>
        </header>
        <div className="search-source-body" aria-busy={loading}>
          {loading ? <p className="search-source-status" role="status">Loading the saved source…</p> : null}
          {error ? <div className="search-source-error" role="alert"><p>{error}</p>{retryable ? <button type="button" onClick={() => loadPage(request.offset)}>Try again</button> : null}</div> : null}
          {source ? <>
            {onOpenWorkspace && !loading && !error ? <div className="search-source-workspace">
              <button type="button" disabled={opening} aria-busy={opening} onClick={() => { void openWorkspace(); }}>{opening ? "Opening…" : source.target.kind === "message" ? "Open conversation" : source.target.kind === "file" ? "Open file" : source.target.kind === "agent" ? "Open agent" : "Open Project"}</button>
              {navigationError ? <p role="alert">{navigationError}</p> : null}
            </div> : null}
            <div className="search-source-metadata">
              {source.private_required ? <span>Private</span> : null}
              {source.target.kind === "file" ? <span>{source.target.artifact === "summary" ? "Generated summary" : source.target.artifact === "transcript" ? "Transcript" : source.target.artifact === "visual" ? "Generated visual notes" : "Saved text"}</span> : null}
              {source.kind === "agent" ? <span>Generated report</span> : null}
              {source.target.kind === "agent" && source.target.step_id ? <span>Saved {source.target.step_id}</span> : null}
              {source.availability === "text_only" ? <span>Saved text retained · original unavailable</span> : null}
            </div>
            <div className="search-source-text" tabIndex={0} aria-label="Saved source text">
              {source.text ? <SourceText source={source} result={result} /> : <p>No saved text is available for this source.</p>}
            </div>
            <footer className="search-source-pagination" aria-label="Source text pages">
              <button type="button" disabled={loading || source.offset === 0} onClick={() => loadPage(Math.max(0, source.offset - PAGE_SIZE))}>Previous text</button>
              <span>{source.total_chars ? `Characters ${(source.offset + 1).toLocaleString()}–${Math.min(source.total_chars, source.offset + Array.from(source.text).length).toLocaleString()} of ${source.total_chars.toLocaleString()}` : "0 characters"}</span>
              <button type="button" disabled={loading || source.next_offset === null} onClick={() => { if (source.next_offset !== null) loadPage(source.next_offset); }}>Next text</button>
            </footer>
          </> : null}
        </div>
      </section>
  );

  if (inline) return viewer;

  return createPortal(
    <div className="search-source-backdrop" role="presentation" onMouseDown={(event) => { if (event.currentTarget === event.target) close.current(); }}>
      {viewer}
    </div>,
    document.body,
  );
}

function sourceBoundaryError(reason: unknown, projectSource: boolean): string | null {
  if (!(reason instanceof ApiError)) return null;
  if (reason.status === 403 || reason.status === 404) {
    return `This source was deleted or is no longer available to you. Close this view and ${projectSource ? "refresh Project knowledge" : "search again"}.`;
  }
  if (reason.status === 409) {
    return projectSource ? "This source changed or was detached. Close this view and refresh Project knowledge." : "This source changed after the search. Close this view and search again for its latest text.";
  }
  return null;
}

function sameTarget(left: SearchResult["target"], right: SearchResult["target"]): boolean {
  switch (left.kind) {
    case "message": return right.kind === "message" && left.conversation_id === right.conversation_id && left.message_id === right.message_id && left.sequence === right.sequence;
    case "file": return right.kind === "file" && left.file_id === right.file_id && left.artifact === right.artifact;
    case "agent": return right.kind === "agent" && left.task_id === right.task_id && left.step_id === right.step_id;
    case "knowledge": return right.kind === "knowledge" && left.project_id === right.project_id && left.entry_id === right.entry_id;
  }
}

function SourceText({ source, result }: { source: SearchSource; result: SearchResult }) {
  const text = Array.from(source.text);
  const excerptLength = Array.from(result.excerpt).length;
  const ranges = result.highlights.filter(({ start, end }) => Number.isInteger(start) && Number.isInteger(end) && start >= 0 && start < end && end <= excerptLength).map(({ start, end }) => ({
    start: Math.max(0, result.offset + start - source.offset),
    end: Math.min(text.length, result.offset + end - source.offset),
  })).filter(({ start, end }) => Number.isInteger(start) && Number.isInteger(end) && start < end).sort((left, right) => left.start - right.start);
  const merged: Array<{ start: number; end: number }> = [];
  for (const range of ranges) {
    const previous = merged.at(-1);
    if (previous && range.start <= previous.end) previous.end = Math.max(previous.end, range.end);
    else merged.push({ ...range });
  }
  let offset = 0;
  const content = merged.flatMap(({ start, end }) => {
    const before = text.slice(offset, start).join("");
    offset = end;
    return [before, <mark key={`${start}:${end}`}>{text.slice(start, end).join("")}</mark>];
  });
  return <>{content}{text.slice(offset).join("")}</>;
}
