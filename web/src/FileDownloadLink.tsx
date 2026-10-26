import { useCallback, useEffect, useRef, useState, type ReactNode } from "react";

import { ApiError } from "./api";
import { FilePreviewDialog } from "./FilePreview";
import { downloadOwnedFile, nativeFileId } from "./fileDownloads";
import { TaxWorkbookLink } from "./TaxWorkbookDownload";

export function FileDownloadLink({ href, title, children }: { href?: string; title?: string; children?: ReactNode }) {
  if (!nativeFileId(href)) return <TaxWorkbookLink href={href} title={title}>{children}</TaxWorkbookLink>;
  return <DownloadLink key={href} href={href!} title={title}>{children}</DownloadLink>;
}

function DownloadLink({ href, title, children }: { href: string; title?: string; children?: ReactNode }) {
  const request = useRef<AbortController | null>(null);
  const [previewOpen, setPreviewOpen] = useState(false);
  const closePreview = useCallback(() => setPreviewOpen(false), []);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  useEffect(() => () => request.current?.abort(), []);
  async function download() {
    if (busy) return;
    const controller = new AbortController();
    request.current = controller;
    setBusy(true);
    setError("");
    try {
      await downloadOwnedFile(href, controller.signal);
    } catch (reason) {
      if (!controller.signal.aborted) setError(reason instanceof ApiError && [404, 410].includes(reason.status)
        ? "This file is no longer available. It may have been deleted."
        : "Could not download the file. Try again.");
    } finally { if (!controller.signal.aborted) setBusy(false); }
  }
  return <span className="tax-workbook-link file-output-link"><a href={href} title={title} aria-busy={busy} aria-disabled={busy}
    onClick={(event) => { event.preventDefault(); void download(); }}>{busy ? "Downloading…" : children}</a>
    <button type="button" className="file-preview-trigger" aria-haspopup="dialog" onClick={() => setPreviewOpen(true)}>
      <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7S2 12 2 12Z" /><circle cx="12" cy="12" r="3" /></svg>Preview
    </button>
    {previewOpen ? <FilePreviewDialog fileId={nativeFileId(href)!} onClose={closePreview} /> : null}
    {error ? <span className="tax-workbook-error" role="alert">{error}</span> : null}
  </span>;
}
