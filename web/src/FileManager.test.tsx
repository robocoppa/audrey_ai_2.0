import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";

import { FileManager } from "./FileManager";
import * as api from "./api";

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

it("offers original downloads only while the stored source exists", async () => {
  vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(JSON.stringify({
    items: [
      {
        id: "file / ready",
        filename: "notes.txt",
        mime: "text/plain",
        bytes: 12,
        uploaded_at: "2026-09-28T00:00:00+00:00",
        kind: "text",
        chunks: 1,
        status: "ready",
        failure_reason: "",
        duration_s: 0,
        summary: "",
        source_freed_at: "",
        leased_at: "",
        source_url: "",
        transcript_source: "",
        fetch_downloaded_bytes: 0,
        fetch_total_bytes: 0,
      },
      {
        id: "video_reclaimed",
        filename: "old-video.mp4",
        mime: "video/mp4",
        bytes: 42,
        uploaded_at: "2026-09-27T00:00:00+00:00",
        kind: "video",
        chunks: 2,
        status: "ready",
        failure_reason: "",
        duration_s: 10,
        summary: "A stored summary remains.",
        source_freed_at: "2026-09-28T00:00:00+00:00",
        leased_at: "",
        source_url: "https://example.com/video",
        transcript_source: "captions",
        fetch_downloaded_bytes: 0,
        fetch_total_bytes: 0,
      },
    ],
    total_bytes: 54,
    server_time: "2026-09-28T00:01:00+00:00",
    limits: {
      max_upload_bytes: 50_000_000,
      max_user_bytes: 1_000_000_000,
      allowed_extensions: [".txt", ".mp4"],
      chunked_max_bytes: 2_000_000_000,
      part_size: 8_000_000,
      fetch_hosts: [],
      max_images_per_turn: 4,
    },
  }), { headers: { "Content-Type": "application/json" } })));

  render(<FileManager onClose={() => undefined} />);

  const download = await screen.findByRole("link", { name: "Download original notes.txt" });
  expect(download).toHaveAttribute("href", "/api/files/file%20%2F%20ready/download");
  expect(download).toHaveAttribute("download", "notes.txt");
  expect(
    screen.queryByRole("link", { name: "Download original old-video.mp4" }),
  ).not.toBeInTheDocument();
});


it("offers downloads for transcript and visual notes but not summaries", async () => {
  const file = {
    id: "video / ready",
    filename: "recording.mp4",
    mime: "video/mp4",
    bytes: 42,
    uploaded_at: "2026-09-28T00:00:00+00:00",
    kind: "video",
    chunks: 2,
    status: "ready",
    failure_reason: "",
    duration_s: 10,
    summary: "A legacy listing summary remains visible.",
    source_freed_at: "2026-09-28T00:00:00+00:00",
    leased_at: "",
    source_url: "",
    transcript_source: "whisper",
    fetch_downloaded_bytes: 0,
    fetch_total_bytes: 0,
  };
  const listing = {
    items: [file],
    total_bytes: 0,
    server_time: "2026-09-28T00:01:00+00:00",
    limits: {
      max_upload_bytes: 50_000_000,
      max_user_bytes: 1_000_000_000,
      allowed_extensions: [".mp4"],
      chunked_max_bytes: 2_000_000_000,
      part_size: 8_000_000,
      fetch_hosts: [],
      max_images_per_turn: 4,
    },
  };
  const artifactText = {
    summary: "A useful summary of the recording.",
    transcript: "Transcript text.",
    visual: "Visual notes text.",
  };
  const fetchMock = vi.fn().mockImplementation((path: string) => {
    const kind = path.includes("/artifacts/transcript?")
      ? "transcript"
      : path.includes("/artifacts/visual?") ? "visual" : "summary";
    const text = artifactText[kind];
    const payload = path === "/api/files"
      ? listing
      : {
          id: file.id,
          artifact: kind,
          text,
          offset: 0,
          next_offset: null,
          total_chars: text.length,
        };
    return Promise.resolve(new Response(JSON.stringify(payload), {
      headers: { "Content-Type": "application/json" },
    }));
  });
  vi.stubGlobal("fetch", fetchMock);

  render(<FileManager onClose={() => undefined} />);

  fireEvent.click(await screen.findByRole("button", { name: "View video text for recording.mp4" }));
  expect(await screen.findByText("A useful summary of the recording.")).toBeVisible();
  expect(
    screen.queryByRole("link", { name: "Download summary for recording.mp4" }),
  ).not.toBeInTheDocument();

  fireEvent.click(screen.getByRole("button", { name: "Transcript" }));
  const download = await screen.findByRole("link", {
    name: "Download transcript for recording.mp4",
  });
  expect(download).toHaveAttribute(
    "href",
    "/api/files/video%20%2F%20ready/artifacts/transcript/download",
  );
  expect(download).toHaveAttribute("download", "");

  fireEvent.click(screen.getByRole("button", { name: "Visual notes" }));
  const visualDownload = await screen.findByRole("link", {
    name: "Download visual notes for recording.mp4",
  });
  expect(visualDownload).toHaveAttribute(
    "href",
    "/api/files/video%20%2F%20ready/artifacts/visual/download",
  );
});


it("shows audio summary and transcript without a visual-notes tab", async () => {
  const file = {
    id: "audio_ready",
    filename: "interview.mp3",
    mime: "audio/mpeg",
    bytes: 2048,
    uploaded_at: "2026-09-30T00:00:00+00:00",
    kind: "audio",
    chunks: 2,
    status: "ready",
    failure_reason: "",
    duration_s: 12,
    summary: "A short interview about the blue lantern.",
    source_freed_at: "",
    leased_at: "",
    source_url: "",
    transcript_source: "whisper",
    fetch_downloaded_bytes: 0,
    fetch_total_bytes: 0,
  };
  const listing = {
    items: [file],
    total_bytes: file.bytes,
    server_time: "2026-09-30T00:01:00+00:00",
    limits: {
      max_upload_bytes: 50_000_000,
      max_user_bytes: 1_000_000_000,
      allowed_extensions: [".mp3"],
      chunked_max_bytes: 2_000_000_000,
      part_size: 8_000_000,
      fetch_hosts: [],
      max_images_per_turn: 4,
    },
  };
  const fetchMock = vi.fn().mockImplementation((path: string) => {
    const transcript = path.includes("/artifacts/transcript?");
    const text = transcript
      ? "The blue lantern is ready."
      : "A short interview about the blue lantern.";
    return Promise.resolve(new Response(JSON.stringify(path === "/api/files"
      ? listing
      : {
          id: file.id,
          artifact: transcript ? "transcript" : "summary",
          text,
          offset: 0,
          next_offset: null,
          total_chars: text.length,
        }), { headers: { "Content-Type": "application/json" } }));
  });
  vi.stubGlobal("fetch", fetchMock);

  render(<FileManager onClose={() => undefined} />);

  fireEvent.click(await screen.findByRole("button", {
    name: "View audio text for interview.mp3",
  }));
  expect(await screen.findByText("A short interview about the blue lantern.")).toBeVisible();
  expect(screen.getByRole("group", { name: "Audio text type" })).toBeVisible();
  expect(screen.queryByRole("button", { name: "Visual notes" })).not.toBeInTheDocument();

  fireEvent.click(screen.getByRole("button", { name: "Transcript" }));
  expect(await screen.findByText("The blue lantern is ready.")).toBeVisible();
});


it("shows a PDF summary first and its extracted text under Transcript", async () => {
  const file = {
    id: "pdf_ready",
    filename: "inspection.pdf",
    mime: "application/pdf",
    bytes: 4096,
    uploaded_at: "2026-09-30T00:00:00+00:00",
    kind: "text",
    chunks: 3,
    status: "ready",
    failure_reason: "",
    duration_s: 0,
    summary: "The inspection found corrosion and recommends replacing the western seal.",
    source_freed_at: "",
    leased_at: "",
    source_url: "",
    transcript_source: "",
    fetch_downloaded_bytes: 0,
    fetch_total_bytes: 0,
  };
  const listing = {
    items: [file],
    total_bytes: file.bytes,
    server_time: "2026-09-30T00:01:00+00:00",
    limits: {
      max_upload_bytes: 50_000_000,
      max_user_bytes: 1_000_000_000,
      allowed_extensions: [".pdf"],
      chunked_max_bytes: 2_000_000_000,
      part_size: 8_000_000,
      fetch_hosts: [],
      max_images_per_turn: 4,
    },
  };
  const fetchMock = vi.fn().mockImplementation((path: string) => {
    if (path === "/api/files") {
      return Promise.resolve(new Response(JSON.stringify(listing), {
        headers: { "Content-Type": "application/json" },
      }));
    }
    if (path.endsWith("/preview")) return Promise.resolve(new Response(JSON.stringify({
      id: file.id, filename: file.filename, kind: "pdf", format: "pdf", page_count: 2, truncated: false,
    }), { headers: { "Content-Type": "application/json" } }));
    const summary = path.includes("/artifacts/summary?");
    const body = summary
      ? {
          id: file.id,
          artifact: "summary",
          text: file.summary,
          offset: 0,
          next_offset: null,
          total_chars: file.summary.length,
        }
      : {
          id: file.id,
          text: "Full extracted inspection text.",
          offset: 0,
          next_offset: null,
          total_chars: 31,
        };
    return Promise.resolve(new Response(JSON.stringify(body), {
      headers: { "Content-Type": "application/json" },
    }));
  });
  vi.stubGlobal("fetch", fetchMock);

  render(<FileManager onClose={() => undefined} />);

  fireEvent.click(await screen.findByRole("button", {
    name: "View document text for inspection.pdf",
  }));
  expect(screen.getByRole("group", { name: "Document text type" })).toBeVisible();
  expect(await screen.findByText(file.summary)).toBeVisible();

  fireEvent.click(screen.getByRole("button", { name: "Transcript" }));
  expect(await screen.findByText("Full extracted inspection text.")).toBeVisible();
  expect(fetchMock).toHaveBeenCalledWith("/api/files/pdf_ready/text?offset=0", expect.objectContaining({
    credentials: "same-origin",
    headers: { Accept: "application/json" },
  }));
  fireEvent.click(screen.getByRole("button", { name: "Original PDF" }));
  const page = await screen.findByAltText("inspection.pdf — page 1 of 2");
  expect(page).toHaveAttribute("src", "/api/files/pdf_ready/preview/pages/1");
  fireEvent.load(page);
  expect(page).toBeVisible();
  fireEvent.click(screen.getByRole("button", { name: "Summary" }));
  expect(await screen.findByText(file.summary)).toBeVisible();
});


it("shows processing after upload bytes finish while the server response is pending", async () => {
  const file: api.AudreyFile = {
    id: "file_ready", filename: "notes.txt", mime: "text/plain", bytes: 4,
    uploaded_at: "2026-10-19T00:00:00+00:00", kind: "text", chunks: 1,
    status: "ready", failure_reason: "", duration_s: 0, summary: "A short note.",
    source_freed_at: "", leased_at: "", source_url: "", transcript_source: "",
    fetch_downloaded_bytes: 0, fetch_total_bytes: 0,
  };
  const listing: api.AudreyFileList = {
    items: [], total_bytes: 0, server_time: "2026-10-19T00:01:00+00:00",
    limits: {
      max_upload_bytes: 50_000_000, max_user_bytes: 1_000_000_000,
      allowed_extensions: [".txt"], chunked_max_bytes: 2_000_000_000,
      part_size: 8_000_000, fetch_hosts: [], max_images_per_turn: 4,
    },
  };
  vi.spyOn(api, "listFiles")
    .mockResolvedValueOnce(listing)
    .mockResolvedValue({ ...listing, items: [file], total_bytes: file.bytes });
  let finishUpload: ((result: api.AudreyFileUpload) => void) | undefined;
  const pending = new Promise<api.AudreyFileUpload>((resolve) => { finishUpload = resolve; });
  vi.spyOn(api, "uploadFile").mockImplementation((_file, _limits, onProgress) => {
    onProgress?.(1);
    return pending;
  });

  render(<FileManager onClose={() => undefined} />);
  await screen.findByText("No files yet. Upload a document, image, audio recording, or video for Audrey to use.");
  fireEvent.click(screen.getByText("Add files"));
  fireEvent.change(screen.getByLabelText("Choose files"), {
    target: { files: [new File(["tiny"], "notes.txt", { type: "text/plain" })] },
  });
  fireEvent.click(screen.getByRole("button", { name: /^Upload$/ }));

  expect(await screen.findByText("Upload complete. Processing your file…")).toBeVisible();
  expect(screen.getByText("notes.txt", { selector: ".upload-status-filename" })).toBeVisible();
  expect(screen.getByRole("progressbar", { name: "Upload progress" })).not.toHaveAttribute("value");
  expect(screen.queryByText(/100%/)).not.toBeInTheDocument();

  await act(async () => {
    finishUpload?.(file);
    await pending;
  });
  expect(await screen.findByText("notes.txt: uploaded")).toBeVisible();
  await waitFor(() => expect(screen.queryByRole("progressbar")).not.toBeInTheDocument());
  expect(await screen.findByRole("link", { name: "Download original notes.txt" })).toBeVisible();
});

function previewFile(overrides: Partial<api.AudreyFile>): api.AudreyFile {
  return {
    id: "preview_file", filename: "preview.txt", mime: "text/plain", bytes: 4096,
    uploaded_at: "2026-10-27T00:00:00+00:00", kind: "text", chunks: 1, status: "ready",
    failure_reason: "", duration_s: 0, summary: "Saved summary.", source_freed_at: "",
    leased_at: "", source_url: "", transcript_source: "", fetch_downloaded_bytes: 0,
    fetch_total_bytes: 0, ...overrides,
  };
}

function listPreviewFiles(files: api.AudreyFile[]) {
  vi.spyOn(api, "listFiles").mockResolvedValue({
    items: files, total_bytes: 4096, server_time: "2026-10-27T00:00:01+00:00",
    limits: { max_upload_bytes: 50000000, max_user_bytes: 1000000000, allowed_extensions: [".pdf", ".png", ".mp3", ".mp4"],
      chunked_max_bytes: 2000000000, part_size: 8000000, fetch_hosts: [], max_images_per_turn: 4 },
  });
}

it("opens the actual PDF directly from its filename while Open keeps the summary default", async () => {
  const file = previewFile({ filename: "inspection.pdf", mime: "application/pdf" });
  listPreviewFiles([file]);
  vi.stubGlobal("fetch", vi.fn().mockImplementation(() => Promise.resolve(new Response(JSON.stringify({
    id: file.id, filename: file.filename, kind: "pdf", format: "pdf", page_count: 2, truncated: false,
  }), { headers: { "Content-Type": "application/json" } }))));
  vi.spyOn(api, "getFileArtifact").mockResolvedValue({ id: file.id, artifact: "summary", text: file.summary, offset: 0, next_offset: null, total_chars: file.summary.length });
  render(<FileManager onClose={() => undefined} />);
  fireEvent.click(await screen.findByRole("button", { name: "Preview inspection.pdf" }));
  const image = await screen.findByAltText("inspection.pdf — page 1 of 2");
  expect(image).toHaveAttribute("src", "/api/files/preview_file/preview/pages/1");
  expect(screen.getByRole("button", { name: "Original PDF" })).toHaveAttribute("aria-pressed", "true");
  expect(api.getFileArtifact).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole("button", { name: "← All files" }));
  fireEvent.click(await screen.findByRole("button", { name: "View document text for inspection.pdf" }));
  expect(await screen.findByText(file.summary)).toBeVisible();
  expect(screen.getByRole("button", { name: "Summary" })).toHaveAttribute("aria-pressed", "true");
});

it("opens images from their filenames and disables previews until processing finishes", async () => {
  listPreviewFiles([
    previewFile({ filename: "photo.png", mime: "image/png", kind: "image" }),
    previewFile({ id: "pending_file", filename: "processing.pdf", mime: "application/pdf", status: "processing" }),
  ]);
  render(<FileManager onClose={() => undefined} />);
  expect(await screen.findByRole("button", { name: "Preview processing.pdf" })).toBeDisabled();
  fireEvent.click(screen.getByRole("button", { name: "Preview photo.png" }));
  expect(await screen.findByAltText("Preview of photo.png")).toHaveAttribute("src", "/api/files/preview_file/image");
});

it.each(["audio", "video"] as const)("opens %s playback from a filename and stops it when switching to Summary", async (kind) => {
  vi.spyOn(HTMLMediaElement.prototype, "pause").mockImplementation(() => undefined);
  vi.spyOn(HTMLMediaElement.prototype, "load").mockImplementation(() => undefined);
  const file = previewFile({ filename: kind === "audio" ? "interview.mp3" : "walk.mp4", kind, mime: kind === "audio" ? "audio/mpeg" : "video/mp4" });
  listPreviewFiles([file]);
  vi.spyOn(api, "getFileArtifact").mockResolvedValue({ id: file.id, artifact: "summary", text: file.summary, offset: 0, next_offset: null, total_chars: file.summary.length });
  render(<FileManager onClose={() => undefined} />);
  fireEvent.click(await screen.findByRole("button", { name: `Preview ${file.filename}` }));
  const media = screen.getByLabelText(`${kind === "audio" ? "Audio" : "Video"} player for ${file.filename}`);
  expect(media).toHaveAttribute("src", "/api/files/preview_file/media");
  expect(screen.getByRole("button", { name: "Player" })).toHaveAttribute("aria-pressed", "true");
  expect(api.getFileArtifact).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole("button", { name: "Summary" }));
  expect(await screen.findByText(file.summary)).toBeVisible();
  expect(HTMLMediaElement.prototype.pause).toHaveBeenCalledOnce();
  expect(media).not.toHaveAttribute("src");
});

it("opens unavailable playback for reclaimed filenames and keeps their source and text accessible", async () => {
  const file = previewFile({ filename: "old.mp4", kind: "video", mime: "video/mp4", source_freed_at: "2026-10-27",
    source_url: "https://www.youtube.com/watch?v=old-video" });
  listPreviewFiles([file]);
  vi.spyOn(api, "getFileArtifact").mockResolvedValue({ id: file.id, artifact: "summary", text: file.summary, offset: 0, next_offset: null, total_chars: file.summary.length });
  render(<FileManager onClose={() => undefined} />);
  fireEvent.click(await screen.findByRole("button", { name: "Preview old.mp4" }));
  expect(screen.getByRole("button", { name: "Player" })).toHaveAttribute("aria-pressed", "true");
  expect(screen.getByRole("status")).toHaveTextContent("Playback unavailable");
  expect(screen.getByRole("link", { name: "Open source website" })).toHaveAttribute("href", file.source_url);
  expect(screen.queryByLabelText("Video player for old.mp4")).not.toBeInTheDocument();
  expect(api.getFileArtifact).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole("button", { name: "Summary" }));
  expect(await screen.findByText(file.summary)).toBeVisible();
  expect(screen.getByText(/The original file was removed to free storage/)).toBeVisible();
  fireEvent.click(screen.getByRole("button", { name: "Player" }));
  expect(screen.getByRole("status")).toHaveTextContent("Playback unavailable");
  fireEvent.click(screen.getByRole("button", { name: "← All files" }));
  fireEvent.click(await screen.findByRole("button", { name: "View video text for old.mp4" }));
  expect(await screen.findByText(file.summary)).toBeVisible();
  expect(screen.getByRole("button", { name: "Summary" })).toHaveAttribute("aria-pressed", "true");
});


it("opens a text document's contents from its filename while Open keeps Summary", async () => {
  const file = previewFile({ filename: "notes.txt" });
  listPreviewFiles([file]);
  vi.spyOn(api, "getFileText").mockResolvedValue({ id: file.id, text: "The complete document contents.", offset: 0, next_offset: null, total_chars: 31 });
  vi.spyOn(api, "getFileArtifact").mockResolvedValue({ id: file.id, artifact: "summary", text: file.summary, offset: 0, next_offset: null, total_chars: file.summary.length });
  render(<FileManager onClose={() => undefined} />);
  fireEvent.click(await screen.findByRole("button", { name: "Preview notes.txt" }));
  expect(await screen.findByText("The complete document contents.")).toBeVisible();
  expect(screen.getByRole("button", { name: "Transcript" })).toHaveAttribute("aria-pressed", "true");
  expect(api.getFileArtifact).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole("button", { name: "← All files" }));
  fireEvent.click(await screen.findByRole("button", { name: "View document text for notes.txt" }));
  expect(await screen.findByText(file.summary)).toBeVisible();
});
