import { useEffect, useLayoutEffect, useRef, useState, type FormEvent } from "react";
import {
  ApiError, getWatchRunSnapshot, searchWatchRunSnapshot,
  WATCH_SNAPSHOT_MATCH_LIMIT, WATCH_SNAPSHOT_PAGE_SIZE, WATCH_SNAPSHOT_QUERY_LIMIT,
  type WatchRunSnapshotPage, type WatchSnapshotMatch, type WatchSnapshotSearchResult,
} from "./api";
import "./WatchPageSnapshot.css";

type SnapshotTarget = { watchId: string; runId: string };
type SnapshotResult = { offset: number; page: WatchRunSnapshotPage } | { offset: number; error: string };
type SnapshotFind = { result: WatchSnapshotSearchResult; index: number } | { error: string };

function matchesRequestedPage(page: WatchRunSnapshotPage, watchId: string, runId: string, offset: number) {
  if (!page || page.watch_id !== watchId || page.run_id !== runId
      || typeof page.source_url !== "string" || typeof page.text !== "string") return false;

  const { start, end, total_chars: total, next_offset: next, text } = page;
  const length = end - start;
  return start === offset && Number.isSafeInteger(end) && Number.isSafeInteger(total)
    && length >= 0 && length <= WATCH_SNAPSHOT_PAGE_SIZE && end <= total
    && text.length <= length * 2 && Array.from(text).length === length
    && (next === null ? end === total : next === end && end < total && length === WATCH_SNAPSHOT_PAGE_SIZE);
}

function validMatches(result: WatchSnapshotSearchResult, watchId: string, runId: string, total: number, queryLength: number) {
  if (!result || result.watch_id !== watchId || result.run_id !== runId || result.total_chars !== total
      || typeof result.truncated !== "boolean" || !Array.isArray(result.matches)
      || result.matches.length > WATCH_SNAPSHOT_MATCH_LIMIT
      || (result.truncated && result.matches.length !== WATCH_SNAPSHOT_MATCH_LIMIT)) return false;

  let previousEnd = 0;
  for (const match of result.matches) {
    if (!match || !Number.isSafeInteger(match.start) || !Number.isSafeInteger(match.end)
        || match.start < previousEnd || match.end > total || match.end - match.start !== queryLength) return false;
    previousEnd = match.end;
  }
  return true;
}

export function WatchPageSnapshot({ watchId, runId }: SnapshotTarget) {
  const [open, setOpen] = useState(false);
  return <details className="watch-page-snapshot" onToggle={(event) => setOpen(event.currentTarget.open)}>
    <summary>Saved page text</summary>
    {open ? <SnapshotReader watchId={watchId} runId={runId} /> : null}
  </details>;
}

function SnapshotReader({ watchId, runId }: SnapshotTarget) {
  const [offset, setOffset] = useState(0);
  const [result, setResult] = useState<SnapshotResult | null>(null);
  const [query, setQuery] = useState("");
  const [find, setFind] = useState<SnapshotFind | null>(null);
  const [finding, setFinding] = useState(false);
  const findRequest = useRef<AbortController | null>(null);

  useEffect(() => () => findRequest.current?.abort(), []);

  useEffect(() => {
    const controller = new AbortController();
    async function readPage() {
      try {
        const page = await getWatchRunSnapshot(watchId, runId, offset, controller.signal);
        if (controller.signal.aborted) return;
        if (!matchesRequestedPage(page, watchId, runId, offset)) {
          throw new Error("Unexpected saved page response");
        }
        setResult({ offset, page });
      } catch (reason) {
        if (controller.signal.aborted) return;
        setResult({
          offset,
          error: reason instanceof ApiError && reason.status === 404
            ? "Saved page text is unavailable for this check."
            : "Could not load saved page text. Close and reopen to try again.",
        });
      }
    }
    void readPage();
    return () => controller.abort();
  }, [watchId, runId, offset]);

  const shown = result?.offset === offset ? result : null;
  const page = shown && "page" in shown ? shown.page : null;
  const nextOffset = page?.next_offset;
  const showPagination = offset > 0 || Boolean(result && "page" in result && result.page.total_chars > WATCH_SNAPSHOT_PAGE_SIZE);
  const range = page && (page.total_chars === 0 ? "No text was saved for this check."
    : `Characters ${(page.start + 1).toLocaleString()}–${page.end.toLocaleString()} of ${page.total_chars.toLocaleString()}`);
  const lastPage = result && "page" in result ? result.page : null;
  const found = find && "result" in find ? find : null;
  const matches = found?.result.matches;
  const activeMatch = found && matches?.[found.index];
  const trimmedQuery = query.trim();
  const queryLength = Array.from(trimmedQuery).length;
  const canFind = Boolean(lastPage && queryLength && queryLength <= WATCH_SNAPSHOT_QUERY_LIMIT && !finding);

  function selectMatch(foundResult: WatchSnapshotSearchResult, index: number) {
    const match = foundResult.matches[index];
    setFind({ result: foundResult, index });
    if (match) setOffset(Math.floor(match.start / WATCH_SNAPSHOT_PAGE_SIZE) * WATCH_SNAPSHOT_PAGE_SIZE);
  }

  async function submitFind(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!canFind || !lastPage) return;
    const controller = new AbortController();
    findRequest.current?.abort();
    findRequest.current = controller;
    setFinding(true);
    setFind(null);
    try {
      const foundResult = await searchWatchRunSnapshot(watchId, runId, trimmedQuery, controller.signal);
      if (controller.signal.aborted) return;
      if (!validMatches(foundResult, watchId, runId, lastPage.total_chars, queryLength)) {
        throw new Error("Unexpected saved page match response");
      }
      selectMatch(foundResult, 0);
    } catch (reason) {
      if (!controller.signal.aborted) setFind({ error: reason instanceof ApiError && reason.status === 404
        ? "Saved page text is unavailable for this check."
        : "Could not find text in this check. Try again." });
    } finally {
      if (findRequest.current === controller) {
        findRequest.current = null;
        setFinding(false);
      }
    }
  }

  return <div className="watch-snapshot-reader">
    {lastPage ? <form className="watch-snapshot-find" role="search" aria-label="Find in saved page text" onSubmit={(event) => void submitFind(event)}>
      <input type="search" aria-label="Find text" placeholder="Find text" maxLength={WATCH_SNAPSHOT_QUERY_LIMIT * 2} value={query} onChange={(event) => {
        findRequest.current?.abort();
        findRequest.current = null;
        setFinding(false);
        setFind(null);
        setQuery(event.target.value);
      }} />
      <button type="submit" className="agents-secondary" aria-disabled={!canFind}>Find</button>
    </form> : null}
    {queryLength > WATCH_SNAPSHOT_QUERY_LIMIT ? <p role="status">Use up to {WATCH_SNAPSHOT_QUERY_LIMIT} characters.</p> : null}
    {finding ? <p role="status">Finding text…</p> : null}
    {find && "error" in find ? <p className="agents-error" role="alert">{find.error}</p> : null}
    {found && matches ? <div className="watch-snapshot-toolbar">
      <p role="status">{matches.length
        ? `Match ${found.index + 1} of ${matches.length}${found.result.truncated ? " shown. Use a longer phrase to narrow results." : ""}`
        : "No matches in this saved check."}</p>
      {matches.length > 1 ? <div className="watch-snapshot-pagination" role="group" aria-label="Saved page text matches">
        <button type="button" className="agents-secondary" aria-disabled={found.index === 0} onClick={() => {
          if (found.index > 0) selectMatch(found.result, found.index - 1);
        }}>Previous match</button>
        <button type="button" className="agents-secondary" aria-disabled={found.index === matches.length - 1} onClick={() => {
          if (found.index + 1 < matches.length) selectMatch(found.result, found.index + 1);
        }}>Next match</button>
      </div> : null}
    </div> : null}
    <div className="watch-snapshot-toolbar">
      {shown && "error" in shown
        ? <p className="agents-error" role="alert">{shown.error}</p>
        : <p role="status">{range ?? "Loading saved page text…"}</p>}
      {showPagination ? <div className="watch-snapshot-pagination" role="group" aria-label="Saved page text pages">
        <button type="button" className="agents-secondary" aria-disabled={offset === 0 || !shown} onClick={() => {
          if (offset > 0 && shown) setOffset(Math.max(0, offset - WATCH_SNAPSHOT_PAGE_SIZE));
        }}>Previous page</button>
        <button type="button" className="agents-secondary" aria-disabled={nextOffset == null} onClick={() => {
          if (nextOffset != null) setOffset(nextOffset);
        }}>Next page</button>
      </div> : null}
    </div>
    {page ? <>
      <p className="watch-snapshot-source" title={page.source_url}>{page.source_url}</p>
      {page.text ? <SnapshotText key={page.start} page={page} match={activeMatch || undefined} /> : null}
    </> : null}
  </div>;
}

function SnapshotText({ page, match }: { page: WatchRunSnapshotPage; match?: WatchSnapshotMatch }) {
  const highlightRef = useRef<HTMLElement>(null);
  const start = match ? Math.max(0, match.start - page.start) : 0;
  const end = match ? Math.min(page.end, match.end) - page.start : 0;
  const characters = end > start ? Array.from(page.text) : null;

  useLayoutEffect(() => {
    highlightRef.current?.scrollIntoView?.({ block: "nearest", inline: "nearest", behavior: "instant" });
  }, [start, end, page.text]);

  return <pre tabIndex={0} aria-label="Saved page text">{characters ? <>
    {characters.slice(0, start).join("")}
    <mark ref={highlightRef}>{characters.slice(start, end).join("")}</mark>
    {characters.slice(end).join("")}
  </> : page.text}</pre>;
}
