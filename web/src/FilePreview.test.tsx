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

it("pages worksheets independently with exact row numbers and preserves the last successful section on failure", async () => {
  const first = { ...workbook, content_truncated: false, truncated: true, notice: "Showing part of this file. Download it to see all contents.", sheets: [
    { ...workbook.sheets[0], sheet_index: 0, row_offset: 0, next_row_offset: 3, total_rows: null, truncated: true },
    { ...workbook.sheets[1], sheet_index: 1, row_offset: 0, next_row_offset: null, total_rows: 2 },
  ] };
  const second = { ...first, sheets: [{ name: "Budget", sheet_index: 0, rows: [["Rent", 800]], row_offset: 3, next_row_offset: null, total_rows: 4, truncated: true }] };
  const fetch = vi.fn().mockResolvedValueOnce(json(first)).mockResolvedValueOnce(json({ detail: "Unavailable" }, 503))
    .mockResolvedValueOnce(json(second)).mockResolvedValueOnce(json({ ...first, sheets: [first.sheets[0]] }));
  vi.stubGlobal("fetch", fetch);
  render(<FilePreviewContent fileId="file_123" />);
  expect(await screen.findByText("Rows 1–3")).toBeVisible();
  expect(screen.queryByText(first.notice)).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "Next" }));
  expect(await screen.findByRole("alert")).toHaveTextContent("Could not load this section.");
  expect(screen.getByRole("cell", { name: "Food" })).toBeVisible();
  fireEvent.click(screen.getByRole("button", { name: "Next" }));
  expect(await screen.findByRole("cell", { name: "Rent" })).toBeVisible();
  expect(screen.getByRole("rowheader", { name: "4" })).toBeVisible();
  expect(screen.getByText("Rows 4–4 of 4")).toBeVisible();
  expect(screen.getByRole("button", { name: "Next" })).toBeDisabled();
  expect(fetch.mock.calls[2][0]).toBe("/api/files/file_123/preview?sheet=0&row_offset=3");
  fireEvent.click(screen.getByRole("tab", { name: "Notes" }));
  expect(screen.getByRole("cell", { name: "TRUE" })).toBeVisible();
  fireEvent.click(screen.getByRole("tab", { name: "Budget" }));
  expect(screen.getByRole("cell", { name: "Rent" })).toBeVisible();
  fireEvent.click(screen.getByRole("button", { name: "Previous" }));
  expect(await screen.findByRole("cell", { name: "Food" })).toBeVisible();
  expect(fetch.mock.calls[3][0]).toBe("/api/files/file_123/preview?sheet=0&row_offset=0");
  expect(screen.getByRole("button", { name: "Previous" })).toBeDisabled();
});

it("browses contiguous text sections and returns to the exact previous offset", async () => {
  const first = { id: "file_123", filename: "long.txt", kind: "text", format: "txt", text: "First section.",
    offset: 0, next_offset: 14, total_chars: 27, truncated: true, content_truncated: false };
  const last = { ...first, text: "Last section.", offset: 14, next_offset: null };
  const fetch = vi.fn().mockResolvedValueOnce(json(first)).mockResolvedValueOnce(json(last)).mockResolvedValueOnce(json(first));
  vi.stubGlobal("fetch", fetch);
  render(<FilePreviewContent fileId="file_123" />);
  expect(await screen.findByText("First section.")).toBeVisible();
  fireEvent.click(screen.getByRole("button", { name: "Next" }));
  expect(await screen.findByText("Last section.")).toBeVisible();
  expect(screen.getByText("Section 2 · Characters 15–27 of 27")).toBeVisible();
  expect(fetch.mock.calls[1][0]).toBe("/api/files/file_123/preview?text_offset=14");
  fireEvent.click(screen.getByRole("button", { name: "Previous" }));
  expect(await screen.findByText("First section.")).toBeVisible();
  expect(fetch.mock.calls[2][0]).toBe("/api/files/file_123/preview?text_offset=0");
});

it("pages the same immutable Tax export rather than preparing a new revision", async () => {
  const first = { ...workbook, sheets: [{ ...workbook.sheets[1], sheet_index: 1, row_offset: 0, next_row_offset: 2 }] };
  const fetch = vi.fn().mockResolvedValueOnce(json(first)).mockResolvedValueOnce(json({ ...first,
    sheets: [{ ...first.sheets[0], rows: [["Retained public passage"]], row_offset: 2, next_row_offset: null }] }));
  vi.stubGlobal("fetch", fetch);
  render(<FilePreviewContent workbookPath="/api/conversations/con_tax/tax-review/exports/tax_export_1/download" />);
  expect(await screen.findByRole("button", { name: "Next" })).toBeEnabled();
  fireEvent.click(screen.getByRole("button", { name: "Next" }));
  expect(await screen.findByRole("cell", { name: "Retained public passage" })).toBeVisible();
  expect(fetch.mock.calls[1][0]).toBe("/api/conversations/con_tax/tax-review/exports/tax_export_1/preview?sheet=1&row_offset=2");
  expect(fetch.mock.calls.every(([, options]) => !options.method || options.method === "GET")).toBe(true);
});

it("cancels an in-flight page when the file changes and discards the old result", async () => {
  let completePage!: (value: Response) => void;
  const fetch = vi.fn().mockResolvedValueOnce(json({ ...workbook, sheets: [{ ...workbook.sheets[0], next_row_offset: 3 }] }))
    .mockImplementationOnce(() => new Promise<Response>((resolve) => { completePage = resolve; }))
    .mockResolvedValueOnce(json(document));
  vi.stubGlobal("fetch", fetch);
  const { rerender } = render(<FilePreviewContent fileId="file_123" />);
  fireEvent.click(await screen.findByRole("button", { name: "Next" }));
  const signal = fetch.mock.calls[1][1].signal as AbortSignal;
  expect(screen.getByRole("button", { name: "Next" })).toBeDisabled();
  rerender(<FilePreviewContent fileId="file_456" />);
  expect(await screen.findByText("Read a book.")).toBeVisible();
  expect(signal.aborted).toBe(true);
  completePage(json({ ...workbook, sheets: [{ name: "Old file", rows: [["Stale data"]], truncated: false }] }));
  await waitFor(() => expect(screen.queryByText("Stale data")).not.toBeInTheDocument());
});
