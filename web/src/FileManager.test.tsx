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
