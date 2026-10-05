import { useCallback, useEffect, useMemo, useRef, useState, type DragEvent, type FormEvent } from "react";

import {
  cancelDocumentJob,
  createProjectBrief,
  decideDocumentJob,
  deleteFile,
  fetchVideoFromUrl,
  getDocumentJob,
  getFileArtifact,
  getFileArtifactDownloadUrl,
  getFileDownloadUrl,
  getFileImageUrl,
  getFileText,
  listDocumentJobs,
  listFiles,
  uploadFile,
  uploadPrecheck,
  type AudreyFile,
  type AudreyFileArtifact,
  type AudreyFileArtifactKind,
  type AudreyFileText,
  type AudreyFileList,
  type DocumentJob,
} from "./api";

type FileKindFilter = "all" | AudreyFile["kind"];
type FileStatusFilter = "all" | "ready" | "active" | "failed";
type FileSort = "newest" | "oldest" | "name-asc" | "name-desc";

const fileNameCollator = new Intl.Collator(undefined, { numeric: true, sensitivity: "base" });
const FILE_FOLDERS: ReadonlyArray<{
  kind: FileKindFilter;
  label: string;
  symbol: string;
}> = [
  { kind: "all", label: "All files", symbol: "▦" },
  { kind: "text", label: "Documents", symbol: "≡" },
  { kind: "image", label: "Images", symbol: "◫" },
  { kind: "audio", label: "Audio", symbol: "♪" },
  { kind: "video", label: "Videos", symbol: "▶" },
];

export function FileManager({ onClose }: { onClose: () => void }) {
  const [listing, setListing] = useState<AudreyFileList | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [draggingFiles, setDraggingFiles] = useState(false);
  const [uploadResults, setUploadResults] = useState<string[]>([]);
  const [activeUpload, setActiveUpload] = useState("");
  const [videoUrl, setVideoUrl] = useState("");
  const [fetchingUrl, setFetchingUrl] = useState(false);
  const [queuedUrl, setQueuedUrl] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [confirmingId, setConfirmingId] = useState<string | null>(null);
  const [selectedFileId, setSelectedFileId] = useState<string | null>(null);
  const [fileSearch, setFileSearch] = useState("");
  const [kindFilter, setKindFilter] = useState<FileKindFilter>("all");
  const [statusFilter, setStatusFilter] = useState<FileStatusFilter>("all");
  const [fileSort, setFileSort] = useState<FileSort>("newest");
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
    if (!selectedFiles.length || !listing || uploading || fetchingUrl) return;
    setUploading(true);
    setProgress(0);
    setError("");
    setUploadResults([]);
    let uploaded = false;
    try {
      for (const file of selectedFiles) {
        const precheck = uploadPrecheck(file, listing.limits);
        if (precheck) {
          setUploadResults((results) => [...results, file.name + ": skipped — " + precheck]);
          continue;
        }
        setActiveUpload(file.name);
        setProgress(0);
        try {
          const result = await uploadFile(file, listing.limits, setProgress);
          uploaded = true;
          setUploadResults((results) => [...results, file.name + ": " + (result.status === "ready" ? "uploaded" : "stored; processing")]);
        } catch (reason) {
          setUploadResults((results) => [...results, file.name + ": failed — " + messageOf(reason)]);
        }
      }
      setSelectedFiles([]);
      if (inputRef.current) inputRef.current.value = "";
      if (uploaded) await refresh();
    } catch (reason) {
      setError(messageOf(reason));
    } finally {
      setActiveUpload("");
      setProgress(0);
      setUploading(false);
    }
  }

  function selectFiles(files: FileList | File[]) {
    if (uploading || fetchingUrl || files.length === 0) return;
    setSelectedFiles(Array.from(files));
    setUploadResults([]);
    setError("");
  }

  function dropFiles(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setDraggingFiles(false);
    if (event.dataTransfer.files.length) selectFiles(event.dataTransfer.files);
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
      if (selectedFileId === file.id) setSelectedFileId(null);
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
  const selectedFile = listing?.items.find((file) => file.id === selectedFileId) ?? null;
  const visibleFiles = useMemo(() => {
    const search = fileSearch.trim().toLocaleLowerCase();
    return (listing?.items ?? [])
      .map((file, index) => ({ file, index, uploaded: Date.parse(file.uploaded_at) || 0 }))
      .filter(({ file }) => {
        if (search && !file.filename.toLocaleLowerCase().includes(search)
          && !file.source_url.toLocaleLowerCase().includes(search)) return false;
        if (kindFilter !== "all" && file.kind !== kindFilter) return false;
        if (statusFilter === "ready" && file.status !== "ready") return false;
        if (statusFilter === "failed" && file.status !== "failed") return false;
        if (statusFilter === "active" && !["fetch_pending", "fetching", "pending", "processing"].includes(file.status)) return false;
        return true;
      })
      .sort((left, right) => {
        if (fileSort === "name-asc" || fileSort === "name-desc") {
          const byName = fileNameCollator.compare(left.file.filename, right.file.filename);
          if (byName) return fileSort === "name-asc" ? byName : -byName;
        }
        const byDate = fileSort === "oldest"
          ? left.uploaded - right.uploaded
          : right.uploaded - left.uploaded;
        return byDate || left.index - right.index;
      })
      .map(({ file }) => file);
  }, [fileSearch, fileSort, kindFilter, listing, statusFilter]);
  const browseChanged = fileSearch.trim() !== "" || kindFilter !== "all"
    || statusFilter !== "all" || fileSort !== "newest";

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

        {selectedFile ? (
          ["video", "audio"].includes(selectedFile.kind) ? (
            <MediaArtifactViewer
              key={selectedFile.id}
              file={selectedFile}
              onBack={() => setSelectedFileId(null)}
            />
          ) : selectedFile.kind === "image" ? (
            <ImagePreviewViewer
              key={selectedFile.id}
              file={selectedFile}
              onBack={() => setSelectedFileId(null)}
            />
          ) : (
            <DocumentTextViewer
              key={selectedFile.id}
              file={selectedFile}
              onBack={() => setSelectedFileId(null)}
            />
          )
        ) : (
          <>
        <details className="file-add-panel">
          <summary>
            <span className="file-add-symbol" aria-hidden="true">＋</span>
            <span>
              <strong>Add files</strong>
              <small>Upload from your device or fetch a video link</small>
            </span>
            <span className="file-add-chevron" aria-hidden="true">⌄</span>
          </summary>
          <div className="file-add-content">
        <form className="file-upload" onSubmit={(event) => void submitUpload(event)}>
          <div
            className={draggingFiles ? "file-drop-zone dragging" : "file-drop-zone"}
            role="region"
            aria-label="Drop files to upload"
            onDragOver={(event) => event.preventDefault()}
            onDragEnter={(event) => {
              event.preventDefault();
              if (!uploading && !fetchingUrl && event.dataTransfer.types.includes("Files")) setDraggingFiles(true);
            }}
            onDragLeave={() => setDraggingFiles(false)}
            onDrop={dropFiles}
          >Drop files here, or choose several below.</div>
          {listing ? (
            <p className="file-upload-hint">
              Up to {formatBytes(listing.limits.chunked_max_bytes || listing.limits.max_upload_bytes)} per file.
              {listing.limits.chunked_max_bytes > listing.limits.max_upload_bytes
                ? " Files over " + formatBytes(listing.limits.max_upload_bytes) + " upload in parts."
                : ""}
              {" Supported: " + listing.limits.allowed_extensions.join(", ") + "."}
              {" Scanned PDFs, audio, and videos become searchable after processing."}
            </p>
          ) : null}
          <label>
            <span>Choose files</span>
            <input
              ref={inputRef}
              type="file"
              multiple
              accept={accept}
              disabled={uploading || fetchingUrl}
              onChange={(event) => selectFiles(event.target.files ?? [])}
            />
          </label>
          <button type="submit" disabled={!selectedFiles.length || !listing || uploading || fetchingUrl}>
            {uploading ? "Uploading " + activeUpload + " · " + Math.round(progress * 100) + "%" : selectedFiles.length > 1 ? "Upload " + selectedFiles.length + " files" : "Upload"}
          </button>
          {uploading ? (
            <progress value={progress} max={1} aria-label="Upload progress" />
          ) : null}
          {selectedFiles.length > 0 ? (
            <small>
              {selectedFiles.length} selected · {formatBytes(selectedFiles.reduce((total, file) => total + file.size, 0))}
            </small>
          ) : null}
          {uploadResults.length ? (
            <ul className="file-upload-results" aria-label="Upload results" aria-live="polite">
              {uploadResults.map((result, index) => <li key={index}>{result}</li>)}
            </ul>
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
              disabled={!listing || fetchHosts.length === 0 || fetchingUrl || uploading}
              required
            />
            <button type="submit" disabled={!videoUrl.trim() || fetchingUrl || uploading || fetchHosts.length === 0}>
              {fetchingUrl ? "Queueing…" : "Fetch video"}
            </button>
          </div>
          <small>Audrey downloads the video and prepares its summary for your private files.</small>
          {queuedUrl ? <p role="status">Queued. Watch the file below for download and summarization progress.</p> : null}
        </form>
          </div>
        </details>

        <ProjectBriefBuilder onPublished={refresh} />

        {error ? <p className="file-manager-error" role="alert">{error}</p> : null}
        {loading ? <p className="file-manager-status" role="status">Loading files…</p> : null}
        {!loading && listing?.items.length === 0 ? (
          <p className="file-manager-empty">No files yet. Upload a document, image, audio recording, or video for Audrey to use.</p>
        ) : null}
        {!loading && listing?.items.length ? (
          <div className="file-explorer">
            <aside className="file-explorer-sidebar" aria-label="File folders">
              <span>Library</span>
              {FILE_FOLDERS.map((folder) => {
                const count = folder.kind === "all"
                  ? listing.items.length
                  : listing.items.filter(({ kind }) => kind === folder.kind).length;
                return (
                  <button
                    type="button"
                    key={folder.kind}
                    aria-label={`${folder.label} (${count})`}
                    aria-pressed={kindFilter === folder.kind}
                    onClick={() => setKindFilter(folder.kind)}
                  >
                    <span className="file-folder-symbol" aria-hidden="true">{folder.symbol}</span>
                    <span>{folder.label}</span>
                    <small>{count}</small>
                  </button>
                );
              })}
            </aside>
            <div className="file-explorer-content">
              <div className="file-browser">
                <div className="file-browser-controls">
                  <label>
                    <span>Search files</span>
                    <input
                      type="search"
                      value={fileSearch}
                      onChange={(event) => setFileSearch(event.target.value)}
                      placeholder="Filename or video link"
                    />
                  </label>
                  <label>
                    <span>Status</span>
                    <select value={statusFilter} onChange={(event) => setStatusFilter(event.target.value as FileStatusFilter)}>
                      <option value="all">All statuses</option>
                      <option value="ready">Ready</option>
                      <option value="active">In progress</option>
                      <option value="failed">Failed</option>
                    </select>
                  </label>
                  <label>
                    <span>Sort by</span>
                    <select value={fileSort} onChange={(event) => setFileSort(event.target.value as FileSort)}>
                      <option value="newest">Newest</option>
                      <option value="oldest">Oldest</option>
                      <option value="name-asc">Name A–Z</option>
                      <option value="name-desc">Name Z–A</option>
                    </select>
                  </label>
                </div>
                <div className="file-browser-result">
                  <span role="status">Showing {visibleFiles.length} of {listing.items.length} files</span>
                  {browseChanged ? (
                    <button type="button" onClick={() => {
                      setFileSearch("");
                      setKindFilter("all");
                      setStatusFilter("all");
                      setFileSort("newest");
                    }}>Clear filters</button>
                  ) : null}
                </div>
              </div>

              {visibleFiles.length === 0 ? (
                <p className="file-manager-empty">No files match these filters.</p>
              ) : (
                <ul className="file-list">
                  {visibleFiles.map((file) => (
                    <li key={file.id}>
                      <div className="file-kind" aria-hidden="true">{kindSymbol(file.kind)}</div>
                      <div className="file-details">
                        <strong title={file.filename}>{file.filename}</strong>
                        <div className="file-row-meta">
                          <span className="file-status">{fileStatus(file, listing.server_time)}</span>
                          <span>
                            {kindLabel(file.kind)} · {file.bytes || !["fetch_pending", "fetching"].includes(file.status)
                              ? formatBytes(file.bytes)
                              : "Size pending"}
                            {file.status === "ready" ? ` · ${file.chunks} ${file.chunks === 1 ? "chunk" : "chunks"}` : ""}
                            {file.transcript_source ? ` · ${transcriptLabel(file.transcript_source)}` : ""}
                          </span>
                          <time dateTime={file.uploaded_at} title={file.uploaded_at}>
                            {formatFileTime(file.uploaded_at)}
                          </time>
                        </div>
                        {file.failure_reason ? <small className="file-failure">{file.failure_reason}</small> : null}
                        {file.source_freed_at ? <small>Original reclaimed · derived text retained</small> : null}
                      </div>
                      <div className="file-actions">
                        {file.source_url ? (
                          <a href={file.source_url} target="_blank" rel="noopener noreferrer">Source</a>
                        ) : null}
                        {!file.source_freed_at && !["fetch_pending", "fetching"].includes(file.status) ? (
                          <a
                            className="file-download"
                            href={getFileDownloadUrl(file.id)}
                            download={file.filename}
                            aria-label={`Download original ${file.filename}`}
                          >Download</a>
                        ) : null}
                        {file.status === "ready" ? (
                          <button
                            className="file-view"
                            type="button"
                            onClick={() => setSelectedFileId(file.id)}
                            aria-label={`View ${file.kind === "image" ? "image" : file.kind === "video" ? "video text" : file.kind === "audio" ? "audio text" : "document text"} for ${file.filename}`}
                          >Open</button>
                        ) : null}
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
                          {deletingId === file.id ? "Deleting…" : confirmingId === file.id ? "✓" : "Delete"}
                        </button>
                      </div>
                    </li>
                  ))}
                </ul>
              )}
            </div>
          </div>
        ) : null}
          </>
        )}
      </section>
    </div>
  );
}

type BriefForm = {
  filename: string;
  title: string;
  preparedFor: string;
  preparedOn: string;
  summary: string;
  objectives: string;
  nextSteps: string;
};

function localDateValue(): string {
  const now = new Date();
  const local = new Date(now.getTime() - now.getTimezoneOffset() * 60_000);
  return local.toISOString().slice(0, 10);
}

const emptyBrief = (): BriefForm => ({
  filename: "Project brief.docx",
  title: "",
  preparedFor: "",
  preparedOn: localDateValue(),
  summary: "",
  objectives: "",
  nextSteps: "",
});

const activeDocumentStatuses = new Set(["awaiting_approval", "queued", "running"]);

function ProjectBriefBuilder({ onPublished }: { onPublished: () => Promise<AudreyFileList> }) {
  const [form, setForm] = useState<BriefForm>(emptyBrief);
  const [job, setJob] = useState<DocumentJob | null>(null);
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [panelOpen, setPanelOpen] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    let active = true;
    listDocumentJobs()
      .then(({ items }) => {
        if (!active) return;
        const current = (items ?? []).find((item) =>
          item.operation === "template_to_docx" && activeDocumentStatuses.has(item.status),
        );
        if (current) {
          setJob(current);
          setPanelOpen(true);
        }
      })
      .catch((reason: unknown) => {
        if (active) setError(messageOf(reason));
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => { active = false; };
  }, []);

  const jobId = job?.id ?? null;
  const jobStatus = job?.status ?? null;

  useEffect(() => {
    if (!jobId || !jobStatus || !["queued", "running"].includes(jobStatus)) return;
    let active = true;
    let refreshing = false;
    const check = async () => {
      if (refreshing) return;
      refreshing = true;
      try {
        const next = await getDocumentJob(jobId);
        if (!active) return;
        setJob(next);
        if (next.status === "succeeded") await onPublished();
      } catch (reason) {
        if (active) setError(messageOf(reason));
      } finally {
        refreshing = false;
      }
    };
    void check();
    const timer = window.setInterval(() => void check(), 1500);
    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, [jobId, jobStatus, onPublished]);

  function change<K extends keyof BriefForm>(field: K, value: BriefForm[K]) {
    setForm((current) => ({ ...current, [field]: value }));
    setError("");
  }

  async function requestDocument(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (submitting) return;
    const objectives = lines(form.objectives);
    const nextSteps = lines(form.nextSteps);
    if (!objectives.length || !nextSteps.length) {
      setError("Add at least one objective and one next step, one per line.");
      return;
    }
    if (objectives.length > 8 || nextSteps.length > 8) {
      setError("Use no more than eight objectives and eight next steps.");
      return;
    }
    if ([...objectives, ...nextSteps].some((item) => item.length > 300)) {
      setError("Keep each objective and next step under 300 characters.");
      return;
    }
    setSubmitting(true);
    setError("");
    try {
      const cleanName = form.filename.trim();
      const filename = cleanName.toLocaleLowerCase().endsWith(".docx")
        ? cleanName
        : cleanName + ".docx";
      const random = globalThis.crypto?.randomUUID?.()
        ?? `${Date.now()}-${Math.random().toString(16).slice(2)}`;
      const created = await createProjectBrief({
        template_id: "project-brief-v1",
        filename,
        fields: {
          title: form.title.trim(),
          prepared_for: form.preparedFor.trim(),
          prepared_on: form.preparedOn.trim(),
          summary: form.summary.trim(),
          objectives,
          next_steps: nextSteps,
        },
        idempotency_key: `project-brief-${random}`,
      });
      setJob(created);
      setPanelOpen(true);
    } catch (reason) {
      setError(messageOf(reason));
    } finally {
      setSubmitting(false);
    }
  }

  async function decide(decision: "approved" | "rejected") {
    if (!job || submitting) return;
    setSubmitting(true);
    setError("");
    try {
      setJob(await decideDocumentJob(job.id, job.operation_digest, decision));
    } catch (reason) {
      setError(messageOf(reason));
    } finally {
      setSubmitting(false);
    }
  }

  async function cancel() {
    if (!job || submitting) return;
    setSubmitting(true);
    setError("");
    try {
      setJob(await cancelDocumentJob(job.id));
    } catch (reason) {
      setError(messageOf(reason));
    } finally {
      setSubmitting(false);
    }
  }

  const objectiveLines = lines(form.objectives);
  const nextStepLines = lines(form.nextSteps);
  const valid = Boolean(
    form.filename.trim() && form.title.trim() && form.preparedFor.trim()
    && form.preparedOn.trim() && form.summary.trim()
    && objectiveLines.length > 0 && nextStepLines.length > 0,
  );
  const generating = job && ["queued", "running"].includes(job.status);

  return (
    <details
      className="file-add-panel file-document-panel"
      open={panelOpen}
      onToggle={(event) => setPanelOpen(event.currentTarget.open)}
    >
      <summary>
        <span className="file-add-symbol" aria-hidden="true">▤</span>
        <span>
          <strong>Create a document</strong>
          <small>Build a private Project Brief from a reviewed Audrey template</small>
        </span>
        <span className="file-add-chevron" aria-hidden="true">⌄</span>
      </summary>
      <div className="file-document-content">
        {loading ? <p role="status">Checking document requests…</p> : null}
        {!loading && !job ? (
          <form className="file-document-form" onSubmit={(event) => void requestDocument(event)}>
            <div className="file-document-form-heading">
              <div>
                <strong>Project Brief</strong>
                <small>Creates a new DOCX in My Files. Existing files are never changed.</small>
              </div>
              <span>Reviewed template</span>
            </div>
            <label>
              <span>Document name</span>
              <input value={form.filename} maxLength={255} required onChange={(event) => change("filename", event.target.value)} />
            </label>
            <label>
              <span>Title</span>
              <input value={form.title} maxLength={120} required onChange={(event) => change("title", event.target.value)} />
            </label>
            <label>
              <span>Prepared for</span>
              <input value={form.preparedFor} maxLength={120} required onChange={(event) => change("preparedFor", event.target.value)} />
            </label>
            <label>
              <span>Date</span>
              <input type="date" value={form.preparedOn} required onChange={(event) => change("preparedOn", event.target.value)} />
            </label>
            <label className="file-document-wide">
              <span>Summary</span>
              <textarea value={form.summary} maxLength={2000} required rows={3} onChange={(event) => change("summary", event.target.value)} />
            </label>
            <label>
              <span>Objectives · one per line</span>
              <textarea value={form.objectives} maxLength={2400} required rows={4} onChange={(event) => change("objectives", event.target.value)} />
              <small>Up to 8 items and 300 characters per item.</small>
            </label>
            <label>
              <span>Next steps · one per line</span>
              <textarea value={form.nextSteps} maxLength={2400} required rows={4} onChange={(event) => change("nextSteps", event.target.value)} />
              <small>Up to 8 items and 300 characters per item.</small>
            </label>
            <div className="file-document-actions file-document-wide">
              <small>Audrey will show a request summary and its exact fingerprint before generating anything.</small>
              <button type="submit" disabled={!valid || submitting}>
                {submitting ? "Preparing…" : "Review request"}
              </button>
            </div>
          </form>
        ) : null}

        {job ? (
          <div className="file-document-approval" aria-live="polite">
            <div className="file-document-form-heading">
              <div>
                <strong>{documentStatus(job)}</strong>
                <small>{job.summary}</small>
              </div>
              <span>{job.status.replaceAll("_", " ")}</span>
            </div>
            <p>{job.preview}</p>
            <dl>
              <div><dt>Request fingerprint</dt><dd>{job.operation_digest.slice(0, 16)}…</dd></div>
              <div><dt>Output</dt><dd>Private DOCX in My Files</dd></div>
            </dl>
            {job.status === "awaiting_approval" ? (
              <div className="file-document-actions">
                <small>Approval applies only to this request fingerprint and expires at {formatFileTime(job.approval.expires_at)}.</small>
                <div>
                  <button type="button" className="secondary" disabled={submitting} onClick={() => void decide("rejected")}>Reject</button>
                  <button type="button" disabled={submitting} onClick={() => void decide("approved")}>
                    {submitting ? "Submitting…" : "Approve and create"}
                  </button>
                </div>
              </div>
            ) : null}
            {generating ? (
              <div className="file-document-actions">
                <small>Audrey is generating, reopening, indexing, and verifying the document.</small>
                <button type="button" className="secondary" disabled={submitting} onClick={() => void cancel()}>
                  {submitting ? "Cancelling…" : "Cancel"}
                </button>
              </div>
            ) : null}
            {!activeDocumentStatuses.has(job.status) ? (
              <div className="file-document-actions">
                <small>{job.status === "succeeded"
                  ? "The verified document is ready below and can be added to any Project."
                  : job.error_code ? `Audrey stopped: ${job.error_code.replaceAll("_", " ")}.` : "No document was created."}</small>
                <button type="button" onClick={() => {
                  setJob(null);
                  setForm(emptyBrief());
                  setError("");
                }}>Create another</button>
              </div>
            ) : null}
          </div>
        ) : null}
        {error ? <p className="file-manager-error" role="alert">{error}</p> : null}
      </div>
    </details>
  );
}

function lines(value: string): string[] {
  return value.split("\n").map((line) => line.trim()).filter(Boolean);
}

function documentStatus(job: DocumentJob): string {
  if (job.status === "awaiting_approval") return "Review this document request";
  if (job.status === "queued") return "Document queued";
  if (job.status === "running") return "Creating and verifying document";
  if (job.status === "succeeded") return "Document created";
  if (job.status === "rejected") return "Request rejected";
  if (job.status === "cancelled") return "Request cancelled";
  if (job.status === "expired") return "Approval expired";
  return "Document generation stopped";
}

const artifactKinds: AudreyFileArtifactKind[] = ["summary", "transcript", "visual"];

function artifactLabel(artifact: AudreyFileArtifactKind): string {
  if (artifact === "visual") return "Visual notes";
  return artifact[0].toUpperCase() + artifact.slice(1);
}

function DocumentTextViewer({ file, onBack }: { file: AudreyFile; onBack: () => void }) {
  const [view, setView] = useState<"summary" | "transcript">("summary");

  return (
    <div className="file-artifact-viewer">
      <div className="file-artifact-heading">
        <button type="button" onClick={onBack}>← All files</button>
        <div>
          <h3>{file.filename}</h3>
          <small>Document text · {formatBytes(file.bytes)}</small>
        </div>
      </div>
      <div className="file-artifact-tabs" role="group" aria-label="Document text type">
        <button
          type="button"
          aria-pressed={view === "summary"}
          onClick={() => setView("summary")}
        >Summary</button>
        <button
          type="button"
          aria-pressed={view === "transcript"}
          onClick={() => setView("transcript")}
        >Transcript</button>
      </div>
      <ArtifactPage
        key={file.id + ":" + view}
        file={file}
        artifact={view === "summary" ? "summary" : undefined}
      />
    </div>
  );
}

function ImagePreviewViewer({ file, onBack }: { file: AudreyFile; onBack: () => void }) {
  const [failed, setFailed] = useState(false);
  return (
    <div className="file-artifact-viewer">
      <div className="file-artifact-heading">
        <button type="button" onClick={onBack}>← All files</button>
        <div>
          <h3>{file.filename}</h3>
          <small>Image preview · {formatBytes(file.bytes)}</small>
        </div>
      </div>
      <div className="file-artifact-body file-image-preview">
        {failed ? (
          <p className="file-manager-error" role="alert">Image preview is unavailable.</p>
        ) : (
          <img
            src={getFileImageUrl(file.id)}
            alt={`Preview of ${file.filename}`}
            onError={() => setFailed(true)}
          />
        )}
        {file.mime === "image/gif" ? <small>Animated images show their first frame.</small> : null}
      </div>
    </div>
  );
}

function MediaArtifactViewer({ file, onBack }: { file: AudreyFile; onBack: () => void }) {
  const [artifact, setArtifact] = useState<AudreyFileArtifactKind>("summary");
  const availableKinds = file.kind === "audio"
    ? artifactKinds.filter((kind) => kind !== "visual")
    : artifactKinds;
  const mediaLabel = file.kind === "audio" ? "Audio" : "Video";

  return (
    <div className="file-artifact-viewer">
      <div className="file-artifact-heading">
        <button type="button" onClick={onBack}>← All files</button>
        <div>
          <h3>{file.filename}</h3>
          <small>{mediaLabel} text · {formatBytes(file.bytes)}</small>
        </div>
      </div>
      <div className="file-artifact-tabs" role="group" aria-label={mediaLabel + " text type"}>
        {availableKinds.map((kind) => (
          <button
            key={kind}
            type="button"
            aria-pressed={artifact === kind}
            onClick={() => setArtifact(kind)}
          >{artifactLabel(kind)}</button>
        ))}
      </div>
      <ArtifactPage key={file.id + ":" + artifact} file={file} artifact={artifact} />
    </div>
  );
}

function ArtifactPage({ file, artifact }: { file: AudreyFile; artifact?: AudreyFileArtifactKind }) {
  const [page, setPage] = useState<AudreyFileArtifact | AudreyFileText | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    let active = true;
    const readPage = artifact ? getFileArtifact(file.id, artifact) : getFileText(file.id);
    readPage
      .then((result) => {
        if (active) setPage(result);
      })
      .catch((reason: unknown) => {
        if (active) setError(messageOf(reason));
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => { active = false; };
  }, [artifact, file.id]);

  async function loadMore() {
    if (loading || page?.next_offset == null) return;
    const nextOffset = page.next_offset;
    setLoading(true);
    setError("");
    try {
      const next = artifact
        ? await getFileArtifact(file.id, artifact, nextOffset)
        : await getFileText(file.id, nextOffset);
      setPage((current) => current && current.next_offset === next.offset
        ? { ...next, text: current.text + next.text, offset: 0 }
        : current);
    } catch (reason) {
      setError(messageOf(reason));
    } finally {
      setLoading(false);
    }
  }

  const fallbackSummary = artifact === "summary" && page?.total_chars === 0 && file.summary;
  const visibleText = fallbackSummary ? file.summary : page?.text;
  const downloadLabel = artifact && artifact !== "summary" && page && page.total_chars > 0
    ? artifactLabel(artifact)
    : "";

  return (
    <div className="file-artifact-body" aria-live="polite">
      {loading && !page ? <p role="status">Loading {artifact ?? "document text"}…</p> : null}
      {error ? <p className="file-manager-error" role="alert">{error}</p> : null}
      {artifact && downloadLabel ? (
        <a
          className="file-artifact-download"
          href={getFileArtifactDownloadUrl(file.id, artifact)}
          download
          aria-label={`Download ${downloadLabel.toLowerCase()} for ${file.filename}`}
        >Download {downloadLabel}</a>
      ) : null}
      {visibleText ? <div className="file-artifact-text">{visibleText}</div> : null}
      {!loading && !error && !visibleText ? (
        <p>{artifact === "visual"
          ? "No visual notes are available for this video."
          : artifact === "summary" && file.kind === "text"
            ? "No summary is available for this document."
            : artifact
              ? "No " + artifact + " is available for this "
                + (file.kind === "audio" ? "recording." : "video.")
              : "No extracted text is available for this document."}</p>
      ) : null}
      {fallbackSummary ? <small>Only the stored listing summary is available for this file.</small> : null}
      {page?.next_offset != null ? (
        <button type="button" onClick={() => void loadMore()} disabled={loading}>
          {loading ? "Loading more…" : "Load more"}
        </button>
      ) : null}
      {page && page.total_chars > 0 ? (
        <small>{Math.min(page.text.length, page.total_chars).toLocaleString()} of {page.total_chars.toLocaleString()} characters</small>
      ) : null}
    </div>
  );
}

function kindLabel(kind: AudreyFile["kind"]): string {
  if (kind === "text") return "Document";
  return kind[0].toUpperCase() + kind.slice(1);
}

function kindSymbol(kind: AudreyFile["kind"]): string {
  if (kind === "image") return "◫";
  if (kind === "video") return "▶";
  if (kind === "audio") return "♪";
  return "≡";
}

function fileStatus(file: AudreyFile, serverTime: string | undefined): string {
  // Both timestamps come from Audrey. Browser clock skew cannot invent a stalled job.
  const elapsed = elapsedSince(file.uploaded_at, serverTime);
  const age = elapsed === null ? "" : " · " + formatElapsed(elapsed) + " elapsed";
  if (file.status === "fetch_pending") return "Waiting to download" + age;
  if (file.status === "fetching") {
    const total = file.fetch_total_bytes;
    const done = file.fetch_downloaded_bytes;
    if (total > 0) return "Downloading " + Math.min(100, Math.round((done / total) * 100)) + "% (" + formatBytes(done) + " of " + formatBytes(total) + ")";
    if (done > 0) return "Downloading " + formatBytes(done) + " so far";
    return "Downloading" + age;
  }
  if (file.status === "pending" || file.status === "processing") {
    return (file.kind === "video"
      ? "Preparing summary"
      : file.kind === "audio" ? "Transcribing" : "Processing") + age;
  }
  return file.status.replaceAll("_", " ").replace(/^./, (letter) => letter.toUpperCase());
}

function elapsedSince(startedAt: string, serverTime: string | undefined): number | null {
  if (!serverTime) return null;
  const started = Date.parse(startedAt);
  const now = Date.parse(serverTime);
  if (!Number.isFinite(started) || !Number.isFinite(now)) return null;
  return Math.max(0, Math.round((now - started) / 1000));
}

function formatElapsed(seconds: number): string {
  if (seconds < 60) return seconds + "s";
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return minutes + "m " + String(seconds % 60).padStart(2, "0") + "s";
  return Math.floor(minutes / 60) + "h " + String(minutes % 60).padStart(2, "0") + "m";
}

function transcriptLabel(source: string): string {
  return source === "auto_captions" ? "auto-captions" : source;
}

function formatFileTime(value: string): string {
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return value || "unknown";
  return new Intl.DateTimeFormat(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(parsed);
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
