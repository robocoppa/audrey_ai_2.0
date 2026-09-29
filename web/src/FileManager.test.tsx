import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";

import { FileManager } from "./FileManager";

afterEach(() => {
  cleanup();
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
