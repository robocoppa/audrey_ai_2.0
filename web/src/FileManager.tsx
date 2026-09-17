import { useCallback, useEffect, useRef, useState, type FormEvent } from "react";

import {
  deleteFile,
  fetchVideoFromUrl,
  listFiles,
  uploadFile,
  type AudreyFile,
  type AudreyFileList,
} from "./api";

export function FileManager({ onClose }: { onClose: () => void }) {
  const [listing, setListing] = useState<AudreyFileList | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [videoUrl, setVideoUrl] = useState("");
  const [fetchingUrl, setFetchingUrl] = useState(false);
  const [queuedUrl, setQueuedUrl] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [confirmingId, setConfirmingId] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const refreshInFlight = useRef<Promise<AudreyFileList> | null>(null);

  const refresh = useCallback(async () => {
    if (!refreshInFlight.current) {
      refreshInFlight.current = listFiles().finally(() => {
        refreshInFlight.current = null;
      });
    }
    const next = await refreshInFlight.current;
    setListing(next);
  }, []);

  useEffect(() => {
    let active = true;
    listFiles()
      .then((next) => {
        if (active) setListing(next);
      })
      .catch((reason: unknown) => {
        if (active) setError(messageOf(reason));
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, [refresh]);

  const hasMovingFiles = listing?.items.some((file) =>
    ["fetch_pending", "fetching", "pending", "processing"].includes(file.status),
  ) ?? false;

  useEffect(() => {
    if (!hasMovingFiles) return;
    const update = () => {
      if (document.visibilityState === "visible") {
        void refresh().catch((reason: unknown) => setError(messageOf(reason)));
      }
    };
    const timer = window.setInterval(update, 5000);
    document.addEventListener("visibilitychange", update);
    return () => {
      window.clearInterval(timer);
      document.removeEventListener("visibilitychange", update);
    };
  }, [hasMovingFiles, refresh]);

  useEffect(() => {
    function closeOnEscape(event: KeyboardEvent) {
      if (event.key === "Escape" && !uploading && !fetchingUrl && deletingId === null) onClose();
    }
    window.addEventListener("keydown", closeOnEscape);
    return () => window.removeEventListener("keydown", closeOnEscape);
  }, [deletingId, fetchingUrl, onClose, uploading]);

  async function submitUpload(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!selectedFile || !listing) return;
    setUploading(true);
    setProgress(0);
    setError("");
    try {
      await uploadFile(selectedFile, listing.limits, setProgress);
      setSelectedFile(null);
      if (inputRef.current) inputRef.current.value = "";
      setLoading(true);
      await refresh();
    } catch (reason) {
      setError(messageOf(reason));
    } finally {
      setLoading(false);
      setUploading(false);
    }
  }

  async function submitVideoUrl(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const url = videoUrl.trim();
    if (!url || fetchingUrl) return;
    setFetchingUrl(true);
    setQueuedUrl(false);
    setError("");
    try {
      await fetchVideoFromUrl(url);
      setVideoUrl("");
      setQueuedUrl(true);
      await refresh();
    } catch (reason) {
      setError(messageOf(reason));
    } finally {
      setFetchingUrl(false);
    }
  }

  async function remove(file: AudreyFile) {
    setDeletingId(file.id);
    setError("");
    try {
      await deleteFile(file.id);
      setConfirmingId(null);
      setLoading(true);
      await refresh();
    } catch (reason) {
      setError(messageOf(reason));
    } finally {
      setLoading(false);
      setDeletingId(null);
    }
  }

  const usage = listing && listing.limits.max_user_bytes > 0
    ? Math.min(100, (listing.total_bytes / listing.limits.max_user_bytes) * 100)
    : 0;
  const accept = listing?.limits.allowed_extensions.join(",") ?? undefined;
  const fetchHosts = listing?.limits.fetch_hosts ?? [];

  return (
    <div className="file-manager-backdrop" role="presentation" onMouseDown={(event) => {
      if (event.currentTarget === event.target && !uploading && !fetchingUrl && deletingId === null) onClose();
    }}>
      <section
        className="file-manager"
        role="dialog"
        aria-modal="true"
        aria-labelledby="file-manager-title"
      >
        <header className="file-manager-header">
          <div>
            <span>Knowledge and attachments</span>
            <h2 id="file-manager-title">Your files</h2>
          </div>
          <button
            className="file-manager-close"
            type="button"
            onClick={onClose}
            disabled={uploading || fetchingUrl || deletingId !== null}
            aria-label="Close files"
            autoFocus
          >
            ×
          </button>
        </header>

        {listing ? (
          <div className="file-quota" aria-label="Storage usage">
            <div>
              <span>{formatBytes(listing.total_bytes)} used</span>
              <span>{formatBytes(listing.limits.max_user_bytes)} limit</span>
            </div>
            <div className="file-quota-track" aria-hidden="true">
              <span style={{ width: `${usage}%` }} />
            </div>
          </div>
        ) : null}

        <form className="file-upload" onSubmit={(event) => void submitUpload(event)}>
          <label>
            <span>Choose a file</span>
            <input
              ref={inputRef}
              type="file"
              accept={accept}
              disabled={uploading}
              onChange={(event) => setSelectedFile(event.target.files?.[0] ?? null)}
            />
          </label>
          <button type="submit" disabled={!selectedFile || !listing || uploading}>
            {uploading ? `Uploading ${Math.round(progress * 100)}%` : "Upload"}
          </button>
          {uploading ? (
            <progress value={progress} max={1} aria-label="Upload progress" />
          ) : null}
          {selectedFile && listing ? (
            <small>
              {selectedFile.name} · {formatBytes(selectedFile.size)}
              {selectedFile.size > listing.limits.max_upload_bytes ? " · chunked upload" : ""}
            </small>
          ) : null}
        </form>

        <form className="file-url-form" onSubmit={(event) => void submitVideoUrl(event)}>
          <label htmlFor="file-video-url">Paste a video link</label>
          <div>
            <input
              id="file-video-url"
              type="url"
              value={videoUrl}
              onChange={(event) => {
                setVideoUrl(event.target.value);
                setQueuedUrl(false);
              }}
              placeholder={fetchHosts.length ? "Allowed: " + fetchHosts.join(", ") : "Video links are unavailable"}
              disabled={!listing || fetchHosts.length === 0 || fetchingUrl}
              required
            />
            <button type="submit" disabled={!videoUrl.trim() || fetchingUrl || fetchHosts.length === 0}>
              {fetchingUrl ? "Queueing…" : "Fetch video"}
            </button>
          </div>
          <small>Audrey downloads the video and prepares its summary for your private files.</small>
          {queuedUrl ? <p role="status">Queued. Watch the file below for download and summarization progress.</p> : null}
        </form>

        {error ? <p className="file-manager-error" role="alert">{error}</p> : null}
        {loading ? <p className="file-manager-status" role="status">Loading files…</p> : null}
        {!loading && listing?.items.length === 0 ? (
          <p className="file-manager-empty">No files yet. Upload a document, image, or video for Audrey to use.</p>
        ) : null}
        {listing?.items.length ? (
          <ul className="file-list">
            {listing.items.map((file) => (
              <li key={file.id}>
                <div className="file-kind" aria-hidden="true">{kindSymbol(file.kind)}</div>
                <div className="file-details">
                  <strong>{file.filename}</strong>
                  <span className="file-meta">
                    {file.kind} · {file.bytes || !["fetch_pending", "fetching"].includes(file.status) ? formatBytes(file.bytes) : "Size pending"}
                  </span>
                  <span className="file-status">{fileStatus(file)}</span>
                  {file.source_url ? (
                    <a href={file.source_url} target="_blank" rel="noopener noreferrer">Source video</a>
                  ) : null}
                  {file.transcript_source ? <small>Transcript: {transcriptLabel(file.transcript_source)}</small> : null}
                  {file.failure_reason ? <small className="file-failure">{file.failure_reason}</small> : null}
                  {file.source_freed_at ? <small>Original media reclaimed; derived text remains searchable.</small> : null}
                  {file.summary ? (
                    <details className="file-summary">
                      <summary>{summaryTeaser(file.summary)}</summary>
                      <p>{file.summary}</p>
                    </details>
                  ) : null}
                </div>
                <button
                  className={confirmingId === file.id ? "file-remove confirming-delete" : "file-remove"}
                  type="button"
                  aria-label={`${confirmingId === file.id ? "Confirm delete" : "Delete"} ${file.filename}`}
                  aria-pressed={confirmingId === file.id}
                  title={confirmingId === file.id ? "Click again to delete" : "Delete file"}
                  onClick={() => {
                    if (confirmingId === file.id) {
                      void remove(file);
                    } else {
                      setConfirmingId(file.id);
                    }
                  }}
                  onBlur={() => setConfirmingId((current) => current === file.id ? null : current)}
                  onKeyDown={(event) => {
                    if (event.key === "Escape") setConfirmingId(null);
                  }}
                  disabled={uploading || deletingId !== null}
                >
                  {deletingId === file.id ? "Deleting…" : confirmingId === file.id ? "✓" : "Remove"}
                </button>
              </li>
            ))}
          </ul>
        ) : null}
      </section>
    </div>
  );
}

function kindSymbol(kind: AudreyFile["kind"]): string {
  if (kind === "image") return "◫";
  if (kind === "video") return "▶";
  return "≡";
}

function fileStatus(file: AudreyFile): string {
  if (file.status === "fetch_pending") return "Waiting to download";
  if (file.status === "fetching") {
    const total = file.fetch_total_bytes;
    const done = file.fetch_downloaded_bytes;
    if (total > 0) return "Downloading " + Math.min(100, Math.round((done / total) * 100)) + "% (" + formatBytes(done) + " of " + formatBytes(total) + ")";
    if (done > 0) return "Downloading " + formatBytes(done) + " so far";
    return "Downloading";
  }
  if (file.status === "pending" || file.status === "processing") {
    return file.kind === "video" ? "Preparing summary" : "Processing";
  }
  return file.status.replaceAll("_", " ").replace(/^./, (letter) => letter.toUpperCase());
}

function transcriptLabel(source: string): string {
  return source === "auto_captions" ? "auto-captions" : source;
}

function summaryTeaser(summary: string): string {
  const text = summary.replace(/\s+/g, " ").trim();
  return text.length > 100 ? text.slice(0, 99).trimEnd() + "…" : text;
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  const units = ["KB", "MB", "GB", "TB"];
  let value = bytes / 1024;
  let unit = units[0];
  for (const next of units.slice(1)) {
    if (value < 1024) break;
    value /= 1024;
    unit = next;
  }
  return `${value < 10 ? value.toFixed(1) : Math.round(value)} ${unit}`;
}

function messageOf(reason: unknown): string {
  return reason instanceof Error ? reason.message : "The file operation did not complete.";
}
