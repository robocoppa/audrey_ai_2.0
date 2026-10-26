import { useCallback, useEffect, useRef, useState, type ReactNode } from "react";

import { FilePreviewDialog } from "./FilePreview";
import { ApiError } from "./api";
import { downloadTaxWorkbook, isTaxWorkbookUrl, prepareTaxWorkbook, type TaxCase } from "./taxReviewApi";
import "./TaxWorkbookDownload.css";

export function TaxWorkbookDownload({ taxCase, disabled = false, onReload }: {
  taxCase: TaxCase;
  disabled?: boolean;
  onReload: () => void;
}) {
  return <WorkbookDownload key={`${taxCase.id}:${taxCase.revision}`} taxCase={taxCase} disabled={disabled} onReload={onReload} />;
}

function WorkbookDownload({ taxCase, disabled, onReload }: {
  taxCase: TaxCase;
  disabled: boolean;
  onReload: () => void;
}) {
  const request = useRef<AbortController | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [stale, setStale] = useState(false);
  const [previewPath, setPreviewPath] = useState<string | null>(null);
  const closePreview = useCallback(() => setPreviewPath(null), []);
  const [preparingPreview, setPreparingPreview] = useState(false);
  useEffect(() => () => request.current?.abort(), []);

  async function download() {
    if (busy || disabled || stale || preparingPreview) return;
    const controller = new AbortController();
    request.current = controller;
    setBusy(true);
    setError("");
    try {
      const workbook = await prepareTaxWorkbook(taxCase, controller.signal);
      await downloadTaxWorkbook(workbook.download_url, workbook.filename, controller.signal);
    } catch (reason) {
      if (controller.signal.aborted) return;
      const changed = reason instanceof ApiError && reason.status === 409;
      setStale(changed);
      setError(changed ? "This review changed. Reload it before downloading." : "Could not download the workbook. Try again.");
    } finally {
      if (!controller.signal.aborted) setBusy(false);
    }
  }

  async function preview() {
    if (busy || disabled || stale || preparingPreview) return;
    const controller = new AbortController();
    request.current = controller;
    setPreparingPreview(true);
    setError("");
    try {
      const workbook = await prepareTaxWorkbook(taxCase, controller.signal);
      if (!controller.signal.aborted) setPreviewPath(workbook.download_url);
    } catch (reason) {
      if (controller.signal.aborted) return;
      const changed = reason instanceof ApiError && reason.status === 409;
      setStale(changed);
      setError(changed ? "This review changed. Reload it before previewing." : "Could not prepare the preview. Try again.");
    } finally { if (!controller.signal.aborted) setPreparingPreview(false); }
  }

  return <div className="tax-workbook-download">
    <div className="tax-workbook-actions"><button type="button" className="tax-workbook-button" disabled={disabled || busy || stale || preparingPreview}
      aria-busy={busy} onClick={() => void download()} title="Download saved facts, sources and calculations as an Excel workbook">
      <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 3v12m-4-4 4 4 4-4M5 16v4h14v-4" /></svg>
      {busy ? "Preparing workbook…" : "Download workbook"}
    </button>
    <button type="button" className="tax-workbook-button" disabled={disabled || busy || stale || preparingPreview}
      aria-label="Preview workbook" aria-haspopup="dialog" onClick={() => void preview()}>{preparingPreview ? "Preparing…" : "Preview"}</button></div>
    {previewPath ? <FilePreviewDialog downloadPath={previewPath} onClose={closePreview} /> : null}
    {taxCase.review?.stale ? <p className="tax-workbook-note">Includes the current saved information; the worksheet needs updating.</p> : null}
    {error ? <div className="tax-workbook-error" role="alert"><span>{error}</span>
      {stale ? <button type="button" disabled={disabled || busy} onClick={onReload}>Reload review</button> : null}
    </div> : null}
  </div>;
}

export function TaxWorkbookLink({ href, title, children }: { href?: string; title?: string; children?: ReactNode }) {
  if (!isTaxWorkbookUrl(href)) return <a href={href} title={title} target="_blank" rel="noreferrer noopener">{children}</a>;
  return <WorkbookLink key={href} href={href} title={title}>{children}</WorkbookLink>;
}

function WorkbookLink({ href, title, children }: { href: string; title?: string; children?: ReactNode }) {
  const request = useRef<AbortController | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [previewOpen, setPreviewOpen] = useState(false);
  const closePreview = useCallback(() => setPreviewOpen(false), []);
  useEffect(() => () => request.current?.abort(), []);

  async function download() {
    if (busy) return;
    const controller = new AbortController();
    request.current = controller;
    setBusy(true);
    setError("");
    try { await downloadTaxWorkbook(href, undefined, controller.signal); }
    catch (reason) {
      if (!controller.signal.aborted) setError(reason instanceof ApiError && [404, 410].includes(reason.status)
        ? "This saved download is no longer available. Use Download workbook in the Toolbox for the current review."
        : "Could not download the workbook. Try again.");
    }
    finally { if (!controller.signal.aborted) setBusy(false); }
  }

  return <span className="tax-workbook-link file-output-link"><a href={href} title={title} aria-busy={busy} aria-disabled={busy}
    onClick={(event) => { event.preventDefault(); void download(); }}>{busy ? "Preparing workbook…" : children}</a>
    <button type="button" className="file-preview-trigger" aria-haspopup="dialog" onClick={() => setPreviewOpen(true)}>Preview</button>
    {previewOpen ? <FilePreviewDialog downloadPath={href} onClose={closePreview} /> : null}
    {error ? <span className="tax-workbook-error" role="alert">{error}</span> : null}
  </span>;
}
