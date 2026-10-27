import { useEffect, useRef, useState } from "react";

import { getFileDownloadUrl, type AudreyFile } from "./api";
import "./MediaFilePlayer.css";

type MediaFile = Pick<AudreyFile, "id" | "filename" | "kind" | "source_freed_at">;

export function MediaFilePlayer({ file }: { file: MediaFile }) {
  return <Player key={file.id} file={file} />;
}

function Player({ file }: { file: MediaFile }) {
  const player = useRef<HTMLMediaElement | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const label = file.kind === "audio" ? "Audio" : "Video";
  useEffect(() => {
    const media = player.current;
    return () => {
      if (!media) return;
      media.pause();
      media.removeAttribute("src");
      media.load();
    };
  }, []);

  if (file.source_freed_at) return <div className="file-media-player file-media-unavailable" role="status">
    The original was reclaimed. The saved summary and transcript are still available.
  </div>;

  const props = {
    controls: true,
    preload: "metadata",
    src: `/api/files/${encodeURIComponent(file.id)}/media`,
    "aria-label": `${label} player for ${file.filename}`,
    onLoadedMetadata: () => setLoading(false),
    onCanPlay: () => setLoading(false),
    onError: () => {
      setLoading(false);
      const code = player.current?.error?.code;
      setError(code === 4 ? "This browser cannot play this file’s format. You can download the original instead."
        : code === 3 ? "This browser could not decode the file. You can download the original instead."
          : "Playback is unavailable. The file may no longer be available, or its format may not be supported by your browser.");
    },
  };
  return <div className="file-media-player">
    {file.kind === "audio" ? <audio ref={(node) => { player.current = node; }} {...props} />
      : <video ref={(node) => { player.current = node; }} {...props} playsInline />}
    {loading && !error ? <p className="file-media-status" role="status">Loading {label.toLowerCase()} preview…</p> : null}
    {error ? <p className="file-media-error" role="alert">{error}</p> : null}
    <a className="file-artifact-download" href={getFileDownloadUrl(file.id)} download={file.filename}
      aria-label={`Download original ${file.filename}`}>Download original</a>
  </div>;
}
