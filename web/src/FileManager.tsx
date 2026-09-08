import { useCallback, useEffect, useRef, useState, type FormEvent } from "react";

import {
  deleteFile,
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
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [confirmingId, setConfirmingId] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const refresh = useCallback(async () => {
    const next = await listFiles();
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

  useEffect(() => {
    function closeOnEscape(event: KeyboardEvent) {
      if (event.key === "Escape" && !uploading && deletingId === null) onClose();
    }
    window.addEventListener("keydown", closeOnEscape);
    return () => window.removeEventListener("keydown", closeOnEscape);
  }, [deletingId, onClose, uploading]);

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

  return (
    <div className="file-manager-backdrop" role="presentation" onMouseDown={(event) => {
      if (event.currentTarget === event.target && !uploading && deletingId === null) onClose();
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
            disabled={uploading || deletingId !== null}
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
                  <span>
                    {file.kind} · {formatBytes(file.bytes)} · {statusLabel(file.status)}
                  </span>
                  {file.failure_reason ? <small>{file.failure_reason}</small> : null}
                  {file.source_freed_at ? <small>Original media reclaimed; derived text remains searchable.</small> : null}
                </div>
                {confirmingId === file.id ? (
                  <div className="file-delete-confirm" role="group" aria-label={`Delete ${file.filename}`}>
                    <button
                      className="danger-button"
                      type="button"
                      onClick={() => void remove(file)}
                      disabled={deletingId !== null}
                    >
                      {deletingId === file.id ? "Deleting…" : "Delete"}
                    </button>
                    <button type="button" onClick={() => setConfirmingId(null)}>Keep</button>
                  </div>
                ) : (
                  <button
                    className="file-remove"
                    type="button"
                    onClick={() => setConfirmingId(file.id)}
                    disabled={uploading || deletingId !== null}
                    aria-label={`Delete ${file.filename}`}
                  >
                    Remove
                  </button>
                )}
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

function statusLabel(status: string): string {
  return status.replaceAll("_", " ").replace(/^./, (letter) => letter.toUpperCase());
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
