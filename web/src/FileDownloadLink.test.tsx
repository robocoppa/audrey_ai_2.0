import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { FileDownloadLink } from "./FileDownloadLink";

afterEach(() => { cleanup(); vi.restoreAllMocks(); vi.unstubAllGlobals(); });

it("downloads a generated file through authenticated API and keeps the server filename", async () => {
  const create = vi.fn(() => "blob:download");
  vi.stubGlobal("URL", { createObjectURL: create, revokeObjectURL: vi.fn() });
  const names: string[] = [];
  vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(function (this: HTMLAnchorElement) { names.push(this.download); });
  const fetch = vi.fn().mockResolvedValue(new Response("saved output", { headers: {
    "Content-Disposition": "attachment; filename*=UTF-8''meeting%20notes.docx",
    "Content-Type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
  } }));
  vi.stubGlobal("fetch", fetch);
  render(<FileDownloadLink href="/api/files/file_123/download">Download notes</FileDownloadLink>);
  fireEvent.click(screen.getByRole("link", { name: "Download notes" }));
  await waitFor(() => expect(names).toEqual(["meeting notes.docx"]));
  expect(fetch).toHaveBeenCalledWith("/api/files/file_123/download", expect.objectContaining({ credentials: "same-origin" }));
  expect(await (create.mock.calls[0] as unknown as [Blob])[0].text()).toBe("saved output");
});

it("keeps deleted-file errors in chat", async () => {
  vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(JSON.stringify({ detail: "File not found." }), {
    status: 404, headers: { "Content-Type": "application/json" },
  })));
  render(<FileDownloadLink href="/api/files/file_123/download">Download notes</FileDownloadLink>);
  fireEvent.click(screen.getByRole("link", { name: "Download notes" }));
  expect(await screen.findByRole("alert")).toHaveTextContent("may have been deleted");
});

it("does not send API credentials to external or malformed links", () => {
  render(<FileDownloadLink href="https://example.com/notes">External</FileDownloadLink>);
  expect(screen.getByRole("link", { name: "External" })).toHaveAttribute("target", "_blank");
});
