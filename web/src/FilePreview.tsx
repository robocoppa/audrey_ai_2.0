import { useCallback, useEffect, useId, useRef, useState } from "react";
import { createPortal } from "react-dom";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";

import { ApiError, apiJson, getFilePreview, type AudreyFilePreview } from "./api";
import { downloadOwnedFile } from "./fileDownloads";
import { downloadTaxWorkbook, isTaxWorkbookUrl } from "./taxReviewApi";
import "./FilePreview.css";

type PreviewSource = { fileId: string; downloadPath?: never } | { fileId?: never; downloadPath: string };

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
          'button:not(:disabled), a[href], [tabindex="0"]',
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

export function FilePreviewContent({ fileId, workbookPath, showName = true }: { fileId?: string; workbookPath?: string; showName?: boolean }) {
  const [attempt, setAttempt] = useState(0);
  const [preview, setPreview] = useState<AudreyFilePreview | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  useEffect(() => {
    const controller = new AbortController();
    const read = fileId ? getFilePreview(fileId, controller.signal)
      : isTaxWorkbookUrl(workbookPath)
        ? apiJson<AudreyFilePreview>(workbookPath.replace(/\/download$/u, "/preview"), { signal: controller.signal })
        : Promise.reject(new Error("Invalid preview"));
    read.then((value) => {
      if (!controller.signal.aborted) setPreview(value);
    }).catch((reason: unknown) => {
      if (!controller.signal.aborted) setError(reason instanceof ApiError && [404, 410].includes(reason.status)
        ? "This file is no longer available. It may have been deleted."
        : "Preview is unavailable. You can still download the file.");
    }).finally(() => { if (!controller.signal.aborted) setLoading(false); });
    return () => controller.abort();
  }, [fileId, workbookPath, attempt]);
  const retry = useCallback(() => { setLoading(true); setError(""); setAttempt((value) => value + 1); }, []);
  return <div className="file-preview-content">
    {loading ? <p className="file-preview-status" role="status">Loading preview…</p> : null}
    {error ? <div className="file-preview-error" role="alert"><p>{error}</p><button type="button" onClick={retry}>Try again</button></div> : null}
    {preview && !error ? <>
      {showName ? <h3 className="file-preview-name">{preview.filename}</h3> : null}
      {preview.kind === "table" ? <SpreadsheetPreview key={fileId} sheets={preview.sheets ?? []} />
        : preview.kind === "document" ? <article className="file-preview-document">{preview.paragraphs?.map((paragraph, index) => <p key={index}>{paragraph || "\u00a0"}</p>)}</article>
          : preview.format === "md" ? <article className="file-preview-document file-preview-markdown"><Markdown remarkPlugins={[remarkGfm]} skipHtml components={{
            img: ({ alt }) => <span className="file-preview-omitted-image">{alt ? `Image: ${alt}` : "Image"}</span>,
            a: ({ href, children }) => <a href={href} target="_blank" rel="noreferrer noopener">{children}</a>,
          }}>{preview.text ?? ""}</Markdown></article>
            : <pre className="file-preview-text">{preview.text || "This file is empty."}</pre>}
      {preview.notice ? <p className="file-preview-notice" role="status">{preview.notice}</p> : null}
    </> : null}
  </div>;
}

function SpreadsheetPreview({ sheets }: { sheets: NonNullable<AudreyFilePreview["sheets"]> }) {
  const [selected, setSelected] = useState(0);
  const tabId = useId();
  const sheet = sheets[selected] ?? sheets[0];
  const columns = Math.max(1, ...sheet?.rows.map((row) => row.length) ?? [1]);
  const rows = sheet?.rows ?? [];
  return <div className="file-preview-workbook">
    {sheets.length > 1 ? <div className="file-preview-tabs" role="tablist" aria-label="Worksheets">
      {sheets.map((item, index) => <button key={item.name} type="button" role="tab" id={`${tabId}-tab-${index}`}
        aria-controls={`${tabId}-sheet-${index}`} aria-selected={index === selected} tabIndex={index === selected ? 0 : -1}
        onClick={() => setSelected(index)} onKeyDown={(event) => {
          if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
          event.preventDefault();
          const next = event.key === "Home" ? 0 : event.key === "End" ? sheets.length - 1
            : (index + (event.key === "ArrowLeft" ? -1 : 1) + sheets.length) % sheets.length;
          setSelected(next);
          event.currentTarget.parentElement?.querySelectorAll<HTMLButtonElement>('button[role="tab"]')[next]?.focus();
        }}>{item.name}</button>)}
    </div> : null}
    <div className="file-preview-grid" role={sheets.length > 1 ? "tabpanel" : "region"}
      id={`${tabId}-sheet-${selected}`} aria-labelledby={sheets.length > 1 ? `${tabId}-tab-${selected}` : undefined}
      aria-label={sheets.length === 1 ? sheet?.name ?? "Table" : undefined} tabIndex={0}>
      {rows.length ? <table aria-label={sheet?.name ?? "Table"}><thead><tr><th scope="col" aria-label="Row" />
        {Array.from({ length: columns }, (_, index) => <th key={index} scope="col">{index < 26 ? String.fromCharCode(65 + index) : `A${String.fromCharCode(65 + index - 26)}`}</th>)}
      </tr></thead><tbody>{rows.map((row, index) => <tr key={index}><th scope="row">{index + 1}</th>
        {Array.from({ length: columns }, (_, column) => <td key={column} className={typeof row[column] === "number" ? "file-preview-number" : undefined}>
          {row[column] == null ? "" : typeof row[column] === "boolean" ? (row[column] ? "TRUE" : "FALSE") : String(row[column])}
        </td>)}
      </tr>)}</tbody></table> : <p className="file-preview-status">This worksheet is empty.</p>}
    </div>
    <div className="file-preview-grid-caption">{sheet?.name ?? "Table"} · {rows.length.toLocaleString()} {rows.length === 1 ? "row" : "rows"}{sheet?.truncated ? " shown" : ""}</div>
  </div>;
}
