import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";

import { FileDownloadLink } from "./FileDownloadLink";
import { FilePreviewContent } from "./FilePreview";

function json(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
}
const document = { id: "file_123", filename: "weekend.docx", format: "docx", kind: "document", paragraphs: ["Weekend plan", "Read a book.", "Take a walk."], truncated: false };
const workbook = { id: "file_123", filename: "budget.xlsx", format: "xlsx", kind: "table", truncated: false, sheets: [
  { name: "Budget", rows: [["Item", "Cost"], ["Food", 40], ["Travel", 25]], truncated: false },
  { name: "Notes", rows: [["=NOT_A_FORMULA", "<script>unsafe</script>"], [null, true]], truncated: false },
] };
afterEach(() => { cleanup(); vi.restoreAllMocks(); vi.unstubAllGlobals(); });

it("opens a Word preview from chat without downloading and restores focus on Escape", async () => {
  const fetch = vi.fn().mockResolvedValue(json(document));
  vi.stubGlobal("fetch", fetch);
  render(<FileDownloadLink href="/api/files/file_123/download">Download weekend.docx</FileDownloadLink>);
  const trigger = screen.getByRole("button", { name: "Preview" });
  trigger.focus();
  fireEvent.click(trigger);
  const modal = await screen.findByRole("dialog", { name: "File preview" });
  expect(await within(modal).findByText("Read a book.")).toBeVisible();
  expect(fetch).toHaveBeenCalledOnce();
  expect(fetch.mock.calls[0]).toEqual(["/api/files/file_123/preview", expect.objectContaining({ credentials: "same-origin" })]);
  expect(within(modal).getByRole("button", { name: "Close file preview" })).toHaveFocus();
  fireEvent.keyDown(window.document, { key: "Escape" });
  expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  expect(trigger).toHaveFocus();
});

it("shows spreadsheet cells, preserves literal content and supports keyboard sheet switching", async () => {
  vi.stubGlobal("fetch", vi.fn().mockResolvedValue(json(workbook)));
  render(<FilePreviewContent fileId="file_123" />);
  const budget = await screen.findByRole("tab", { name: "Budget" });
  expect(screen.getByRole("cell", { name: "40" })).toBeVisible();
  expect(budget).toHaveAttribute("aria-selected", "true");
  budget.focus();
  fireEvent.keyDown(budget, { key: "ArrowRight" });
  const notes = screen.getByRole("tab", { name: "Notes" });
  expect(notes).toHaveFocus();
  expect(notes).toHaveAttribute("aria-selected", "true");
  expect(screen.getByRole("cell", { name: "=NOT_A_FORMULA" })).toBeVisible();
  expect(screen.getByRole("cell", { name: "<script>unsafe</script>" })).toBeVisible();
  expect(screen.getByRole("cell", { name: "TRUE" })).toBeVisible();
  expect(window.document.querySelector("script")).toBeNull();
});

it("renders Markdown while blocking automatic remote images and raw HTML", async () => {
  const fetch = vi.fn().mockResolvedValue(json({ id: "file_123", filename: "notes.md", format: "md", kind: "text", truncated: false,
    text: '# Working notes\n\n![remote image](https://tracker.example/pixel.png)\n\n<script>alert(1)</script>\n\n[Reference](https://example.com)' }));
  vi.stubGlobal("fetch", fetch);
  render(<FilePreviewContent fileId="file_123" />);
  expect(await screen.findByRole("heading", { name: "Working notes" })).toBeVisible();
  expect(screen.queryByRole("img")).not.toBeInTheDocument();
  expect(window.document.querySelector("script")).toBeNull();
  expect(screen.getByText("Image: remote image")).toBeVisible();
  expect(screen.getByRole("link", { name: "Reference" })).toHaveAttribute("rel", "noreferrer noopener");
  expect(fetch).toHaveBeenCalledOnce();
});

it("shows unavailable preview errors in place and permits retry", async () => {
  vi.stubGlobal("fetch", vi.fn().mockResolvedValueOnce(json({ detail: "Unavailable" }, 422)).mockResolvedValueOnce(json(document)));
  render(<FilePreviewContent fileId="file_123" />);
  expect(await screen.findByRole("alert")).toHaveTextContent("You can still download the file.");
  fireEvent.click(screen.getByRole("button", { name: "Try again" }));
  expect(await screen.findByText("Read a book.")).toBeVisible();
});

it("explains deleted files rather than navigating to an API error page", async () => {
  vi.stubGlobal("fetch", vi.fn().mockResolvedValue(json({ detail: "File not found." }, 404)));
  render(<FilePreviewContent fileId="file_123" />);
  expect(await screen.findByRole("alert")).toHaveTextContent("may have been deleted");
});

it("keeps loading abortable and closes on the backdrop", async () => {
  const fetch = vi.fn().mockReturnValue(new Promise(() => {}));
  vi.stubGlobal("fetch", fetch);
  render(<FileDownloadLink href="/api/files/file_123/download">Download notes</FileDownloadLink>);
  fireEvent.click(screen.getByRole("button", { name: "Preview" }));
  expect(await screen.findByText("Loading preview…")).toBeVisible();
  const signal = fetch.mock.calls[0][1].signal as AbortSignal;
  fireEvent.mouseDown(screen.getByRole("dialog").parentElement!);
  expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  await waitFor(() => expect(signal.aborted).toBe(true));
});

it("makes a partial spreadsheet preview explicit", async () => {
  vi.stubGlobal("fetch", vi.fn().mockResolvedValue(json({ ...workbook, truncated: true,
    notice: "Showing part of this file. Download it to see all contents." })));
  render(<FilePreviewContent fileId="file_123" />);
  expect(await screen.findByText("Showing part of this file. Download it to see all contents.")).toBeVisible();
});
