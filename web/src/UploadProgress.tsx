export function UploadProgress({
  filename,
  progress,
  ariaLabel = "Upload progress",
}: {
  filename: string;
  progress: number;
  ariaLabel?: string;
}) {
  const processing = progress >= 1;
  const percent = Math.min(99, Math.max(1, Math.floor(progress * 100)));
  const label = processing
    ? "Upload complete. Processing your file…"
    : progress <= 0 ? "Preparing upload…" : `Uploading · ${percent}%`;

  return (
    <div className="upload-status" role="status">
      <div className="upload-status-label">
        {processing ? <span className="upload-status-spinner" aria-hidden="true" /> : null}
        <span>{label}</span>
      </div>
      <span className="upload-status-filename" title={filename}>{filename}</span>
      <progress
        value={processing ? undefined : Math.max(0, progress)}
        max={1}
        aria-label={ariaLabel}
      />
    </div>
  );
}
