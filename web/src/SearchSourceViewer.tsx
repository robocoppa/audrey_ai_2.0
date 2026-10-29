import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";

import { ApiError, getSearchSource, type SearchResult, type SearchSource } from "./api";

import "./SearchSourceViewer.css";

const PAGE_SIZE = 4000;

export function SearchSourceViewer({ result, onClose }: { result: SearchResult; onClose: () => void }) {
  return <SourceViewer key={`${result.id}:${result.revision}`} result={result} onClose={onClose} />;
}

function SourceViewer({ result, onClose }: { result: SearchResult; onClose: () => void }) {
  const [request, setRequest] = useState({ offset: result.offset, serial: 0 });
  const [source, setSource] = useState<SearchSource | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [retryable, setRetryable] = useState(false);
  const dialogRef = useRef<HTMLElement>(null);
  const closeRef = useRef<HTMLButtonElement>(null);
  const close = useRef(onClose);

  useEffect(() => { close.current = onClose; }, [onClose]);

  useEffect(() => {
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
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    let current = true;
    async function load() {
      try {
        const fresh = await getSearchSource(result, request.offset, controller.signal);
        if (!current || controller.signal.aborted) return;
        if (fresh.source_id !== result.source_id || fresh.revision !== result.revision || fresh.kind !== result.kind) {
          throw new ApiError(409, "The saved source changed.");
        }
        setSource(fresh);
        setError("");
      } catch (reason) {
        if (!current || controller.signal.aborted) return;
        if (reason instanceof ApiError && reason.status === 404) {
          setError("This source was deleted or is no longer available to you. Close this view and search again.");
          setRetryable(false);
        } else if (reason instanceof ApiError && reason.status === 409) {
          setError("This source changed after the search. Close this view and search again for its latest text.");
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
    return () => { current = false; controller.abort(); };
  }, [request, result]);

  function loadPage(offset: number) {
    closeRef.current?.focus();
    setLoading(true);
    setError("");
    setSource(null);
    setRequest((saved) => ({ offset, serial: saved.serial + 1 }));
  }

  return createPortal(
    <div className="search-source-backdrop" role="presentation" onMouseDown={(event) => { if (event.currentTarget === event.target) onClose(); }}>
      <section className="search-source-dialog" role="dialog" aria-modal="true" aria-labelledby="search-source-title" ref={dialogRef}>
        <header>
          <div>
            <span>Saved source · {result.kind === "message" ? "Chat message" : result.kind === "file" ? "File text" : "Agent work"}</span>
            <h2 id="search-source-title">{source?.title || result.title || "Untitled"}</h2>
          </div>
          <button type="button" ref={closeRef} onClick={onClose} aria-label="Close saved source">×</button>
        </header>
        <div className="search-source-body" aria-busy={loading}>
          {loading ? <p className="search-source-status" role="status">Loading the saved source…</p> : null}
          {error ? <div className="search-source-error" role="alert"><p>{error}</p>{retryable ? <button type="button" onClick={() => loadPage(request.offset)}>Try again</button> : null}</div> : null}
          {source ? <>
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
    </div>,
    document.body,
  );
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
