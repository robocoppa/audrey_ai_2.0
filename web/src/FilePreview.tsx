import { useCallback, useEffect, useId, useRef, useState } from "react";
import { createPortal } from "react-dom";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";

import { ApiError, apiJson, getFilePreview, filePreviewQuery, type FilePreviewPosition, type AudreyFilePreview } from "./api";
import { downloadOwnedFile } from "./fileDownloads";
import { downloadTaxWorkbook, isTaxWorkbookUrl } from "./taxReviewApi";
import { PdfPreviewContent } from "./PdfPreview";
import "./FilePreview.css";

type PreviewSource = { fileId: string; downloadPath?: never } | { fileId?: never; downloadPath: string };

export function FilePreviewButton({ fileId, label = "Preview" }: { fileId: string; label?: string }) {
  const [open, setOpen] = useState(false);
  const close = useCallback(() => setOpen(false), []);
  return <span className="file-output-link"><button type="button" className="file-preview-trigger" aria-label={label}
    aria-haspopup="dialog" onClick={() => setOpen(true)}>
    <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7S2 12 2 12Z" /><circle cx="12" cy="12" r="3" /></svg>Preview
  </button>{open ? <FilePreviewDialog fileId={fileId} onClose={close} /> : null}</span>;
}

export function FilePreviewDialog({ fileId, downloadPath, onClose }: PreviewSource & { onClose: () => void }) {
  const titleId = useId();
  const dialog = useRef<HTMLElement>(null);
  const closeButton = useRef<HTMLButtonElement>(null);
  const downloadRequest = useRef<AbortController | null>(null);
  const [downloading, setDownloading] = useState(false);
  const [downloadError, setDownloadError] = useState("");
  useEffect(() => {
    const previous = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    closeButton.current?.focus({ preventScroll: true });
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault();
        event.stopImmediatePropagation();
        onClose();
      } else if (event.key === "Tab") {
        const buttons = Array.from(dialog.current?.querySelectorAll<HTMLElement>(
          'button:not(:disabled), input:not(:disabled):not([type="hidden"]), select:not(:disabled), a[href], [tabindex="0"]',
        ) ?? []);
        const first = buttons[0], last = buttons.at(-1);
        if (event.shiftKey && (document.activeElement === first || !dialog.current?.contains(document.activeElement))) {
          event.preventDefault(); last?.focus();
        } else if (!event.shiftKey && (document.activeElement === last || !dialog.current?.contains(document.activeElement))) {
          event.preventDefault(); first?.focus();
        }
      }
    };
    document.addEventListener("keydown", onKey, true);
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.removeEventListener("keydown", onKey, true);
      document.body.style.overflow = previousOverflow;
      if (previous?.isConnected) previous.focus({ preventScroll: true });
      downloadRequest.current?.abort();
    };
  }, [onClose]);

  async function download() {
    if (downloading) return;
    const controller = new AbortController();
    downloadRequest.current = controller;
    setDownloading(true);
    setDownloadError("");
    try {
      if (fileId) await downloadOwnedFile(`/api/files/${encodeURIComponent(fileId)}/download`, controller.signal);
      else if (isTaxWorkbookUrl(downloadPath)) await downloadTaxWorkbook(downloadPath, undefined, controller.signal);
      else throw new Error("Invalid download");
    }
    catch { if (!controller.signal.aborted) setDownloadError("Could not download the file. Try again."); }
    finally { if (!controller.signal.aborted) setDownloading(false); }
  }
  return createPortal(<div className="file-preview-backdrop" onMouseDown={(event) => {
    if (event.target === event.currentTarget) onClose();
  }}><section ref={dialog} className="file-preview-dialog" role="dialog" aria-modal="true" aria-labelledby={titleId}>
    <header className="file-preview-header">
      <div><span className="file-preview-eyebrow">Read only</span><h2 id={titleId}>File preview</h2></div>
      <div className="file-preview-actions"><button type="button" onClick={() => void download()} disabled={downloading}>
        {downloading ? "Downloading…" : "Download"}</button>
        <button ref={closeButton} className="file-preview-close" type="button" onClick={onClose} aria-label="Close file preview">×</button></div>
    </header>
    {downloadError ? <p className="file-preview-error" role="alert">{downloadError}</p> : null}
    <FilePreviewContent key={fileId ?? downloadPath} fileId={fileId} workbookPath={downloadPath} />
  </section></div>, document.body);
}

type PreviewContentProps = { fileId?: string; workbookPath?: string; showName?: boolean };
type ReadPreview = (position: FilePreviewPosition, signal: AbortSignal) => Promise<AudreyFilePreview>;

export function FilePreviewContent(props: PreviewContentProps) {
  return <PreviewContent key={props.fileId ?? props.workbookPath} {...props} />;
}

function PreviewContent({ fileId, workbookPath, showName = true }: PreviewContentProps) {
  const [attempt, setAttempt] = useState(0);
  const [preview, setPreview] = useState<AudreyFilePreview | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const [positions, setPositions] = useState<number[]>([]);
  const read = useCallback<ReadPreview>((position, signal) => fileId ? getFilePreview(fileId, signal, position)
    : isTaxWorkbookUrl(workbookPath)
      ? apiJson<AudreyFilePreview>(workbookPath.replace(/\/download$/u, "/preview") + filePreviewQuery(position), { signal })
      : Promise.reject(new Error("Invalid preview")), [fileId, workbookPath]);
  const paging = usePreviewPaging(read);
  const content = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const controller = new AbortController();
    read({}, controller.signal).then((value) => {
      if (!controller.signal.aborted) setPreview(value);
    }).catch((reason: unknown) => {
      if (!controller.signal.aborted) setError(reason instanceof ApiError && [404, 410].includes(reason.status)
        ? "This file is no longer available. It may have been deleted."
        : "Preview is unavailable. You can still download the file.");
    }).finally(() => { if (!controller.signal.aborted) setLoading(false); });
    return () => controller.abort();
  }, [read, attempt]);
  const retry = useCallback(() => { setLoading(true); setError(""); setAttempt((value) => value + 1); }, []);
  async function textPage(forward: boolean) {
    if (!preview) return;
    const offset = forward ? preview.next_offset : positions.at(-1);
    if (offset == null) return;
    const value = await paging.load({ text_offset: offset });
    if (!value) return;
    setPositions((history) => forward ? [...history, preview.offset ?? 0] : history.slice(0, -1));
    setPreview(value);
    content.current?.scrollTo?.({ top: 0 });
  }
  return <div ref={content} className="file-preview-content">
    {loading ? <p className="file-preview-status" role="status">Loading preview…</p> : null}
    {error ? <div className="file-preview-error" role="alert"><p>{error}</p><button type="button" onClick={retry}>Try again</button></div> : null}
    {preview && !error ? <>
      {showName ? <h3 className="file-preview-name">{preview.filename}</h3> : null}
      {preview.kind === "pdf" && fileId ? <PdfPreviewContent fileId={fileId} filename={preview.filename}
        metadata={{ id: preview.id, filename: preview.filename, format: "pdf", kind: "pdf", page_count: preview.page_count ?? 0, truncated: false }} />
        : preview.kind === "table" ? <SpreadsheetPreview sheets={preview.sheets ?? []} paging={paging} />
        : <>
          {((preview.offset ?? 0) > 0 || preview.next_offset != null) ? <PreviewPager busy={paging.busy}
            previous={positions.length > 0} next={preview.next_offset != null}
            onPrevious={() => void textPage(false)} onNext={() => void textPage(true)}
            label={`Section ${positions.length + 1} · Characters ${((preview.offset ?? 0) + 1).toLocaleString()}–${((preview.offset ?? 0) + (preview.text?.length ?? 0)).toLocaleString()}${preview.total_chars != null ? ` of ${preview.total_chars.toLocaleString()}` : ""}`} /> : null}
          {preview.kind === "document" ? <article className="file-preview-document" tabIndex={0} aria-label="Document content">{preview.paragraphs?.map((paragraph, index) => <p key={index}>{paragraph || "\u00a0"}</p>)}</article>
            : preview.format === "md" ? <article className="file-preview-document file-preview-markdown" tabIndex={0} aria-label="Document content"><Markdown remarkPlugins={[remarkGfm]} skipHtml components={{
              img: ({ alt }) => <span className="file-preview-omitted-image">{alt ? `Image: ${alt}` : "Image"}</span>,
              a: ({ href, children }) => <a href={href} target="_blank" rel="noreferrer noopener">{children}</a>,
            }}>{preview.text ?? ""}</Markdown></article>
              : <pre className="file-preview-text" tabIndex={0} aria-label="File content">{preview.text || "This file is empty."}</pre>}
        </>}
      {paging.error ? <p className="file-preview-error" role="alert">{paging.error}</p> : null}
      {preview.notice && preview.content_truncated !== false && (preview.kind !== "table" || preview.content_truncated === undefined || (preview.sheet_count ?? 0) > 8) ? <p className="file-preview-notice" role="status">{preview.notice}</p> : null}
    </> : null}
  </div>;
}

function usePreviewPaging(read: ReadPreview) {
  const controller = useRef<AbortController | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  useEffect(() => () => controller.current?.abort(), []);
  async function load(position: FilePreviewPosition) {
    if (controller.current) return;
    const request = new AbortController();
    controller.current = request;
    setBusy(true); setError("");
    try {
      const value = await read(position, request.signal);
      if (!request.signal.aborted) return value;
    } catch (reason) {
      if (!request.signal.aborted) setError(reason instanceof ApiError && [404, 410].includes(reason.status)
        ? "This file is no longer available. It may have been deleted."
        : "Could not load this section. Try again or download the file.");
    } finally {
      if (!request.signal.aborted) { controller.current = null; setBusy(false); }
    }
  }
  return { load, busy, error };
}

function PreviewPager({ busy, previous, next, onPrevious, onNext, label }: {
  busy: boolean; previous: boolean; next: boolean; onPrevious: () => void; onNext: () => void; label: string;
}) {
  return <nav className="file-preview-pagination" aria-label="Preview pages" aria-busy={busy}>
    <button type="button" disabled={busy || !previous} onClick={onPrevious}>Previous</button>
    <span className="file-preview-page-label" aria-live="polite">{busy ? "Loading section…" : label}</span>
    <button type="button" disabled={busy || !next} onClick={onNext}>Next</button>
  </nav>;
}

function SpreadsheetPreview({ sheets: initialSheets, paging }: {
  sheets: NonNullable<AudreyFilePreview["sheets"]>; paging: ReturnType<typeof usePreviewPaging>;
}) {
  const [sheets, setSheets] = useState(initialSheets);
  const [positions, setPositions] = useState<Record<number, number[]>>({});
  const grid = useRef<HTMLDivElement>(null);
  const [selected, setSelected] = useState(0);
  const tabId = useId();
  const sheet = sheets[selected] ?? sheets[0];
  const columns = Math.max(1, ...sheet?.rows.map((row) => row.length) ?? [1]);
  const rows = sheet?.rows ?? [];
  async function page(forward: boolean) {
    if (!sheet) return;
    const offset = forward ? sheet.next_row_offset : positions[selected]?.at(-1);
    if (offset == null) return;
    const value = await paging.load({ sheet: sheet.sheet_index ?? selected, row_offset: offset });
    const replacement = value?.sheets?.[0];
    if (!replacement) return;
    setSheets((items) => items.map((item, index) => index === selected ? replacement : item));
    setPositions((history) => ({ ...history, [selected]: forward ? [...(history[selected] ?? []), sheet.row_offset ?? 0] : history[selected].slice(0, -1) }));
    grid.current?.scrollTo?.({ top: 0 });
  }
  return <div className="file-preview-workbook">
    {sheets.length > 1 ? <div className="file-preview-tabs" role="tablist" aria-label="Worksheets">
      {sheets.map((item, index) => <button key={item.name} type="button" role="tab" id={`${tabId}-tab-${index}`}
        aria-controls={`${tabId}-sheet-${index}`} aria-selected={index === selected} tabIndex={index === selected ? 0 : -1}
        disabled={paging.busy} onClick={() => setSelected(index)} onKeyDown={(event) => {
          if (paging.busy) return;
          if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
          event.preventDefault();
          const next = event.key === "Home" ? 0 : event.key === "End" ? sheets.length - 1
            : (index + (event.key === "ArrowLeft" ? -1 : 1) + sheets.length) % sheets.length;
          setSelected(next);
          event.currentTarget.parentElement?.querySelectorAll<HTMLButtonElement>('button[role="tab"]')[next]?.focus();
        }}>{item.name}</button>)}
    </div> : null}
    <div ref={grid} className="file-preview-grid" role={sheets.length > 1 ? "tabpanel" : "region"}
      id={`${tabId}-sheet-${selected}`} aria-labelledby={sheets.length > 1 ? `${tabId}-tab-${selected}` : undefined}
      aria-label={sheets.length === 1 ? sheet?.name ?? "Table" : undefined} tabIndex={0}>
      {rows.length ? <table aria-label={sheet?.name ?? "Table"}><thead><tr><th scope="col" aria-label="Row" />
        {Array.from({ length: columns }, (_, index) => <th key={index} scope="col">{index < 26 ? String.fromCharCode(65 + index) : `A${String.fromCharCode(65 + index - 26)}`}</th>)}
      </tr></thead><tbody>{rows.map((row, index) => <tr key={index}><th scope="row">{(sheet?.row_offset ?? 0) + index + 1}</th>
        {Array.from({ length: columns }, (_, column) => <td key={column} className={typeof row[column] === "number" ? "file-preview-number" : undefined}>
          {row[column] == null ? "" : typeof row[column] === "boolean" ? (row[column] ? "TRUE" : "FALSE") : String(row[column])}
        </td>)}
      </tr>)}</tbody></table> : <p className="file-preview-status">This worksheet is empty.</p>}
    </div>
    {((sheet?.row_offset ?? 0) > 0 || sheet?.next_row_offset != null) ? <PreviewPager busy={paging.busy}
      previous={(positions[selected]?.length ?? 0) > 0} next={sheet?.next_row_offset != null}
      onPrevious={() => void page(false)} onNext={() => void page(true)}
      label={rows.length ? `Rows ${((sheet?.row_offset ?? 0) + 1).toLocaleString()}–${((sheet?.row_offset ?? 0) + rows.length).toLocaleString()}${sheet?.total_rows != null ? ` of ${sheet.total_rows.toLocaleString()}` : ""}` : "More rows available"} /> : null}
    {sheet?.content_truncated ? <p className="file-preview-grid-caption file-preview-notice" role="status">Some cells or columns are shortened. Download the file to see all contents.</p> : null}
    <div className="file-preview-grid-caption">{sheet?.name ?? "Table"} · {rows.length.toLocaleString()} {rows.length === 1 ? "row" : "rows"}{sheet?.truncated ? " in this section" : ""}</div>
  </div>;
}
