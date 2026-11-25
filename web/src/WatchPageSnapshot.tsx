import { useEffect, useState } from "react";
import { ApiError, getWatchRunSnapshot, WATCH_SNAPSHOT_PAGE_SIZE, type WatchRunSnapshotPage } from "./api";
import "./WatchPageSnapshot.css";

type SnapshotTarget = { watchId: string; runId: string };
type SnapshotResult = { offset: number; page: WatchRunSnapshotPage } | { offset: number; error: string };

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

  return <div className="watch-snapshot-reader">
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
      {page.text ? <pre key={page.start} tabIndex={0} aria-label="Saved page text">{page.text}</pre> : null}
    </> : null}
  </div>;
}
