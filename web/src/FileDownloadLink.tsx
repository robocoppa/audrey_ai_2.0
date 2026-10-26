import { useEffect, useRef, useState, type ReactNode } from "react";

import { ApiError, apiResponse } from "./api";
import { TaxWorkbookLink } from "./TaxWorkbookDownload";

function isFileDownloadUrl(path: string | undefined): path is string {
  return Boolean(path && /^\/api\/files\/[a-zA-Z0-9_-]+\/download$/u.test(path));
}

export function FileDownloadLink({ href, title, children }: { href?: string; title?: string; children?: ReactNode }) {
  if (!isFileDownloadUrl(href)) return <TaxWorkbookLink href={href} title={title}>{children}</TaxWorkbookLink>;
  return <DownloadLink key={href} href={href} title={title}>{children}</DownloadLink>;
}

function DownloadLink({ href, title, children }: { href: string; title?: string; children?: ReactNode }) {
  const request = useRef<AbortController | null>(null);
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
      const response = await apiResponse(href, { signal: controller.signal });
      const disposition = response.headers.get("Content-Disposition") ?? "";
      if (!/^attachment(?:;|$)/iu.test(disposition)) throw new Error("Missing attachment");
      const encoded = /filename\*=UTF-8''([^;]+)/iu.exec(disposition)?.[1];
      let filename = /filename="([^"]+)"/iu.exec(disposition)?.[1] ?? "download";
      if (encoded) { try { filename = decodeURIComponent(encoded); } catch { /* Use plain name. */ } }
      filename = filename.split(/[\\/]/u).pop()!.replace(/\p{Cc}/gu, "") || "download";
      const blob = await response.blob();
      if (controller.signal.aborted) return;
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = filename;
      document.body.append(anchor);
      try { anchor.click(); } finally {
        anchor.remove();
        window.setTimeout(() => URL.revokeObjectURL(url), 0);
      }
    } catch (reason) {
      if (!controller.signal.aborted) setError(reason instanceof ApiError && [404, 410].includes(reason.status)
        ? "This file is no longer available. It may have been deleted."
        : "Could not download the file. Try again.");
    } finally { if (!controller.signal.aborted) setBusy(false); }
  }
  return <span className="tax-workbook-link"><a href={href} title={title} aria-busy={busy} aria-disabled={busy}
    onClick={(event) => { event.preventDefault(); void download(); }}>{busy ? "Downloading…" : children}</a>
    {error ? <span className="tax-workbook-error" role="alert">{error}</span> : null}
  </span>;
}
