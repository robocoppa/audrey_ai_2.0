import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";

import { MediaFilePlayer } from "./MediaFilePlayer";

const file = { id: "media / file", filename: "interview.mp3", kind: "audio" as const, source_freed_at: "" };

beforeEach(() => {
  vi.spyOn(HTMLMediaElement.prototype, "pause").mockImplementation(() => undefined);
  vi.spyOn(HTMLMediaElement.prototype, "load").mockImplementation(() => undefined);
});
afterEach(() => { cleanup(); vi.restoreAllMocks(); });

it("offers native audio controls with protected metadata loading and no autoplay", () => {
  const { unmount } = render(<MediaFilePlayer file={file} />);
  const audio = screen.getByLabelText("Audio player for interview.mp3");
  expect(audio.tagName).toBe("AUDIO");
  expect(audio).toHaveAttribute("controls");
  expect(audio).toHaveAttribute("preload", "metadata");
  expect(audio).toHaveAttribute("src", "/api/files/media%20%2F%20file/media");
  expect(audio).not.toHaveAttribute("autoplay");
  expect(screen.getByRole("status")).toHaveTextContent("Loading audio preview");
  fireEvent.loadedMetadata(audio);
  expect(screen.queryByRole("status")).not.toBeInTheDocument();
  expect(screen.getByRole("link", { name: "Download original interview.mp3" })).toHaveAttribute("download", "interview.mp3");
  unmount();
  expect(HTMLMediaElement.prototype.pause).toHaveBeenCalledOnce();
  expect(HTMLMediaElement.prototype.load).toHaveBeenCalledOnce();
  expect(audio).not.toHaveAttribute("src");
});

it("uses inline video and releases the previous playback when the selected file changes", () => {
  const { rerender } = render(<MediaFilePlayer file={{ ...file, kind: "video", filename: "walk.mp4" }} />);
  const first = screen.getByLabelText("Video player for walk.mp4");
  expect(first.tagName).toBe("VIDEO");
  expect(first).toHaveAttribute("playsinline");
  rerender(<MediaFilePlayer file={{ ...file, id: "another_video", kind: "video", filename: "run.mp4" }} />);
  expect(first).not.toHaveAttribute("src");
  expect(HTMLMediaElement.prototype.pause).toHaveBeenCalledOnce();
  expect(screen.getByLabelText("Video player for run.mp4")).toHaveAttribute("src", "/api/files/another_video/media");
});

it("explains unsupported formats and retains the original download", () => {
  render(<MediaFilePlayer file={file} />);
  const audio = screen.getByLabelText("Audio player for interview.mp3");
  Object.defineProperty(audio, "error", { configurable: true, value: { code: 4 } });
  fireEvent.error(audio);
  expect(screen.getByRole("alert")).toHaveTextContent("This browser cannot play this file’s format.");
  expect(screen.getByRole("link", { name: "Download original interview.mp3" })).toBeVisible();
  expect(screen.queryByRole("status")).not.toBeInTheDocument();
});

it("does not request reclaimed media and points to the remaining derived text", () => {
  const { container } = render(<MediaFilePlayer file={{ ...file, source_freed_at: "2026-10-27" }} />);
  expect(container.querySelector("audio,video")).toBeNull();
  expect(screen.getByRole("status")).toHaveTextContent("The original was reclaimed");
  expect(screen.queryByRole("link")).not.toBeInTheDocument();
});
