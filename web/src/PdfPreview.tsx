import { useEffect, useId, useRef, useState } from "react";

import { ApiError, apiJson } from "./api";
import "./PdfPreview.css";

export interface PdfPreviewMetadata {
  id: string;
  filename: string;
  format: "pdf";
  kind: "pdf";
  page_count: number;
  truncated: false;
}

type PdfPreviewProps = { fileId: string; filename?: string; metadata?: PdfPreviewMetadata };

/** Raster pages are fetched from the owned-file API, never from document URLs. */
export function PdfPreviewContent(props: PdfPreviewProps) {
  return <PdfReader key={props.fileId} {...props} />;
}

function validMetadata(value: unknown, fileId: string): value is PdfPreviewMetadata {
  if (!value || typeof value !== "object") return false;
  const info = value as Partial<PdfPreviewMetadata>;
  return info.id === fileId && typeof info.filename === "string" && info.format === "pdf" && info.kind === "pdf"
    && info.truncated === false && Number.isInteger(info.page_count) && info.page_count! >= 1 && info.page_count! <= 2000;
}

function previewError(reason: unknown): string {
  return reason instanceof ApiError && [404, 410].includes(reason.status)
    ? "This file is no longer available. It may have been deleted."
    : "PDF preview is unavailable. You can still download the file.";
}

function PdfReader({ fileId, filename, metadata }: PdfPreviewProps) {
  const [fetched, setFetched] = useState<PdfPreviewMetadata | null>(null);
  const [error, setError] = useState("");
  const [attempt, setAttempt] = useState(0);
  const [page, setPage] = useState(1);
  const [pageText, setPageText] = useState("1");
  const [zoom, setZoom] = useState(1);
  const pageId = useId();
  const zoomId = useId();
  const provided = validMetadata(metadata, fileId) ? metadata : undefined;
  const info = provided ?? fetched;
  useEffect(() => {
    if (provided) return;
    const controller = new AbortController();
    apiJson<PdfPreviewMetadata>(`/api/files/${encodeURIComponent(fileId)}/preview`, { signal: controller.signal })
      .then((value) => {
        if (!validMetadata(value, fileId)) throw new Error("Invalid PDF metadata");
        if (!controller.signal.aborted) setFetched(value);
      }).catch((reason: unknown) => {
        if (!controller.signal.aborted) setError(previewError(reason));
      });
    return () => controller.abort();
  }, [fileId, provided, attempt]);
  function goToPage(value: number) {
    if (!info) return;
    const next = Number.isInteger(value) ? Math.max(1, Math.min(info.page_count, value)) : page;
    setPage(next);
    setPageText(String(next));
  }
  return <section className="pdf-preview" aria-label="PDF preview">
    {!info && !error ? <p className="pdf-preview-status" role="status">Loading PDF preview…</p> : null}
    {error ? <div className="pdf-preview-error" role="alert"><p>{error}</p><button type="button" onClick={() => {
      setError(""); setAttempt((value) => value + 1);
    }}>Try again</button></div> : null}
    {info && !error ? <>
      <div className="pdf-preview-toolbar">
        <nav className="pdf-preview-navigation" aria-label="PDF pages">
          <button type="button" aria-label="Previous PDF page" disabled={page === 1} onClick={() => goToPage(page - 1)}>
            <span aria-hidden="true">‹</span><span className="pdf-preview-direction">Previous</span>
          </button>
          <form className="pdf-preview-page-form" onSubmit={(event) => { event.preventDefault(); goToPage(Number(pageText)); }}>
            <label htmlFor={pageId}>Page</label>
            <input id={pageId} className="pdf-preview-page-number" type="number" inputMode="numeric" min={1} max={info.page_count}
              step={1} value={pageText} onChange={(event) => setPageText(event.currentTarget.value)} aria-label="PDF page number" />
            <span className="pdf-preview-total">of {info.page_count.toLocaleString()}</span>
            <button type="submit" className="pdf-preview-go">Go</button>
          </form>
          <button type="button" aria-label="Next PDF page" disabled={page === info.page_count} onClick={() => goToPage(page + 1)}>
            <span className="pdf-preview-direction">Next</span><span aria-hidden="true">›</span>
          </button>
        </nav>
        <div className="pdf-preview-zoom"><label htmlFor={zoomId}>Zoom</label><select id={zoomId} value={zoom}
          onChange={(event) => setZoom(Number(event.currentTarget.value))}>
          <option value={1}>Fit width</option><option value={1.25}>125%</option><option value={1.5}>150%</option><option value={2}>200%</option>
        </select></div>
      </div>
      <PdfPage key={page} fileId={fileId} page={page} total={info.page_count} zoom={zoom} filename={filename ?? info.filename} />
    </> : null}
  </section>;
}

type PdfPageProps = { fileId: string; page: number; total: number; zoom: number; filename: string };

function PdfPage(props: PdfPageProps) {
  const [attempt, setAttempt] = useState(0);
  return <RasterPage key={attempt} {...props} retry={() => setAttempt((value) => value + 1)} />;
}

function RasterPage({ fileId, page, total, zoom, filename, retry }: PdfPageProps & { retry: () => void }) {
  const [loaded, setLoaded] = useState(false);
  const [error, setError] = useState(false);
  const image = useRef<HTMLImageElement>(null);
  useEffect(() => {
    const node = image.current;
    // Dropping the source also cancels a pending image request when leaving a page.
    return () => node?.removeAttribute("src");
  }, []);
  return <>
    <span className="pdf-preview-page-status" role="status" aria-live="polite">
      {error ? `Page ${page} could not be shown` : loaded ? `Page ${page} of ${total}` : `Loading page ${page}…`}
    </span>
    <div className={`pdf-preview-viewport${loaded ? "" : " pdf-preview-viewport-pending"}`}
      role="region" aria-label={`PDF page ${page}`} aria-busy={!loaded && !error} tabIndex={0}>
      {error ? <div className="pdf-preview-error" role="alert"><p>This PDF page could not be displayed. Try again or download the file.</p>
        <button type="button" onClick={retry}>Try again</button></div>
        : <>
          {!loaded ? <div className="pdf-preview-page-placeholder" aria-hidden="true"><span /></div> : null}
          <img ref={image} className={`pdf-preview-page pdf-preview-page-zoom-${Math.round(zoom * 100)}${loaded ? " is-loaded" : ""}`}
            src={`/api/files/${encodeURIComponent(fileId)}/preview/pages/${page}`}
            alt={`${filename} — page ${page} of ${total}`} draggable={false} hidden={!loaded}
            onLoad={() => setLoaded(true)} onError={() => { setLoaded(false); setError(true); }} />
        </>}
    </div>
  </>;
}
