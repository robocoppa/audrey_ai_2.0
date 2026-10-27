import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";

import { PdfPreviewContent, type PdfPreviewMetadata } from "./PdfPreview";

const metadata: PdfPreviewMetadata = { id: "file_pdf", filename: "source.pdf", kind: "pdf", format: "pdf", page_count: 3, truncated: false };
function json(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
}
function pageImage() { return window.document.querySelector<HTMLImageElement>(".pdf-preview-page")!; }
afterEach(() => { cleanup(); vi.restoreAllMocks(); vi.unstubAllGlobals(); });

it("loads protected metadata and displays a PDF raster page without a download or remote viewer", async () => {
  const fetch = vi.fn().mockResolvedValue(json(metadata));
  vi.stubGlobal("fetch", fetch);
  render(<PdfPreviewContent fileId="file_pdf" />);
  expect(await screen.findByRole("region", { name: "PDF page 1" })).toHaveAttribute("aria-busy", "true");
  expect(fetch).toHaveBeenCalledWith("/api/files/file_pdf/preview", expect.objectContaining({ credentials: "same-origin" }));
  expect(pageImage()).toHaveAttribute("src", "/api/files/file_pdf/preview/pages/1");
  expect(window.document.querySelector("iframe,object,a[download]")).toBeNull();
  fireEvent.load(pageImage());
  expect(await screen.findByRole("img", { name: "source.pdf — page 1 of 3" })).toBeVisible();
  expect(screen.getByText("Page 1 of 3")).toBeVisible();
  expect(screen.getByRole("button", { name: "Previous PDF page" })).toBeDisabled();
});

it("reuses metadata, clears the old image on navigation, supports direct page choice and bounded zoom", () => {
  const fetch = vi.fn();
  vi.stubGlobal("fetch", fetch);
  render(<PdfPreviewContent fileId="file_pdf" metadata={metadata} />);
  const first = pageImage();
  fireEvent.load(first);
  fireEvent.change(screen.getByRole("combobox", { name: "Zoom" }), { target: { value: "1.5" } });
  expect(first).toHaveClass("pdf-preview-page-zoom-150", "is-loaded");
  expect(first).not.toHaveAttribute("style");
  expect(fetch).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole("button", { name: "Next PDF page" }));
  expect(first).not.toBeInTheDocument();
  expect(first).not.toHaveAttribute("src");
  expect(pageImage()).toHaveAttribute("src", "/api/files/file_pdf/preview/pages/2");
  expect(screen.queryByRole("img")).not.toBeInTheDocument();
  expect(screen.getByText("Loading page 2…")).toBeVisible();
  fireEvent.load(pageImage());
  expect(screen.getByRole("img", { name: "source.pdf — page 2 of 3" })).toHaveClass("pdf-preview-page-zoom-150");
  fireEvent.change(screen.getByRole("spinbutton", { name: "PDF page number" }), { target: { value: "3" } });
  fireEvent.click(screen.getByRole("button", { name: "Go" }));
  expect(pageImage()).toHaveAttribute("src", "/api/files/file_pdf/preview/pages/3");
  expect(screen.getByRole("button", { name: "Next PDF page" })).toBeDisabled();
  fireEvent.click(screen.getByRole("button", { name: "Previous PDF page" }));
  expect(screen.getByRole("spinbutton", { name: "PDF page number" })).toHaveValue(2);
});

it("shows an image failure in place and retries the same page without stale content", () => {
  render(<PdfPreviewContent fileId="file_pdf" metadata={metadata} />);
  const first = pageImage();
  fireEvent.error(first);
  expect(screen.getByRole("alert")).toHaveTextContent("Try again or download the file.");
  expect(screen.queryByRole("img")).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "Try again" }));
  expect(pageImage()).not.toBe(first);
  expect(pageImage()).toHaveAttribute("src", "/api/files/file_pdf/preview/pages/1");
  fireEvent.load(pageImage());
  expect(screen.getByRole("img")).toBeVisible();
});

it("explains missing files and retries a failed metadata request", async () => {
  const fetch = vi.fn().mockResolvedValueOnce(json({ detail: "Missing" }, 404)).mockResolvedValueOnce(json(metadata));
  vi.stubGlobal("fetch", fetch);
  render(<PdfPreviewContent fileId="file_pdf" />);
  expect(await screen.findByRole("alert")).toHaveTextContent("may have been deleted");
  fireEvent.click(screen.getByRole("button", { name: "Try again" }));
  expect(await screen.findByRole("region", { name: "PDF page 1" })).toBeInTheDocument();
  expect(fetch).toHaveBeenCalledTimes(2);
});

it("aborts metadata when the file changes and ignores a late response from the old file", async () => {
  let complete!: (value: Response) => void;
  const fetch = vi.fn().mockImplementationOnce(() => new Promise<Response>((resolve) => { complete = resolve; }))
    .mockResolvedValueOnce(json({ ...metadata, id: "second_pdf", filename: "second.pdf", page_count: 1 }));
  vi.stubGlobal("fetch", fetch);
  const { rerender } = render(<PdfPreviewContent fileId="file_pdf" />);
  const signal = fetch.mock.calls[0][1].signal as AbortSignal;
  rerender(<PdfPreviewContent fileId="second_pdf" />);
  expect(signal.aborted).toBe(true);
  await screen.findByRole("region", { name: "PDF page 1" });
  complete(json(metadata));
  await waitFor(() => expect(pageImage()).toHaveAttribute("src", "/api/files/second_pdf/preview/pages/1"));
  fireEvent.load(pageImage());
  expect(screen.getByRole("img", { name: "second.pdf — page 1 of 1" })).toBeVisible();
});

it("removes a loaded page immediately when another file is selected", async () => {
  const fetch = vi.fn().mockReturnValue(new Promise(() => {}));
  vi.stubGlobal("fetch", fetch);
  const { rerender } = render(<PdfPreviewContent fileId="file_pdf" metadata={metadata} />);
  const previous = pageImage();
  fireEvent.load(previous);
  rerender(<PdfPreviewContent fileId="second_pdf" />);
  expect(previous).not.toBeInTheDocument();
  expect(previous).not.toHaveAttribute("src");
  expect(await screen.findByText("Loading PDF preview…")).toBeVisible();
  expect(screen.queryByRole("img")).not.toBeInTheDocument();
});

it("refuses invalid metadata instead of offering a broken page range", async () => {
  vi.stubGlobal("fetch", vi.fn().mockResolvedValue(json({ ...metadata, page_count: 0 })));
  render(<PdfPreviewContent fileId="file_pdf" />);
  expect(await screen.findByRole("alert")).toHaveTextContent("PDF preview is unavailable.");
  expect(screen.queryByRole("navigation", { name: "PDF pages" })).not.toBeInTheDocument();
});


it("verifies invalid supplied metadata through the server rather than using its page range", async () => {
  const fetch = vi.fn().mockResolvedValue(json(metadata));
  vi.stubGlobal("fetch", fetch);
  render(<PdfPreviewContent fileId="file_pdf" metadata={{ ...metadata, page_count: 99999 }} />);
  expect(screen.queryByRole("navigation", { name: "PDF pages" })).not.toBeInTheDocument();
  expect(await screen.findByRole("region", { name: "PDF page 1" })).toBeInTheDocument();
  expect(fetch).toHaveBeenCalledOnce();
  expect(screen.getByText("of 3")).toBeVisible();
});
