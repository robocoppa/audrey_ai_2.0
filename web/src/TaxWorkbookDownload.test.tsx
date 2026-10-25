import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";

import { TaxWorkbookDownload, TaxWorkbookLink } from "./TaxWorkbookDownload";
import { TAX_WORKBOOK_CONTENT_TYPE, type TaxCase } from "./taxReviewApi";
import fixtures from "./test/fixtures/tax-review-cases.json";

const taxCase = { ...fixtures.group_reviewed, conversation_id: "con_tax" } as TaxCase;
const downloadPath = "/api/conversations/con_tax/tax-review/exports/tax_export_1/download";
function json(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
}
function prepared(overrides = {}) {
  return json({ export: { id: "tax_export_1", case_revision: taxCase.revision,
    review_revision: taxCase.review!.review_revision, filename: "tax-review-2025.xlsx", download_url: downloadPath, ...overrides } });
}
function mockDownload() {
  const createObjectURL = vi.fn((blob: Blob) => { void blob; return "blob:workbook"; });
  const revokeObjectURL = vi.fn();
  vi.stubGlobal("URL", { createObjectURL, revokeObjectURL });
  const names: string[] = [];
  const click = vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(function (this: HTMLAnchorElement) { names.push(this.download); });
  return { createObjectURL, revokeObjectURL, click, names };
}
afterEach(() => { cleanup(); vi.restoreAllMocks(); vi.unstubAllGlobals(); });

it("prepares the displayed revision and downloads with authenticated requests on demand", async () => {
  const download = mockDownload();
  const fetchMock = vi.fn().mockResolvedValueOnce(prepared())
    .mockResolvedValueOnce(new Response("workbook bytes", { headers: { "Content-Type": TAX_WORKBOOK_CONTENT_TYPE } }));
  vi.stubGlobal("fetch", fetchMock);
  render(<TaxWorkbookDownload taxCase={taxCase} onReload={vi.fn()} />);
  expect(fetchMock).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole("button", { name: "Download workbook" }));
  await waitFor(() => expect(download.click).toHaveBeenCalledOnce());
  expect(fetchMock.mock.calls[0][0]).toBe("/api/conversations/con_tax/tax-review/exports");
  expect(fetchMock.mock.calls[0][1]).toMatchObject({ method: "POST", credentials: "same-origin", body: JSON.stringify({ expected_revision: taxCase.revision }) });
  expect(fetchMock.mock.calls[1]).toEqual([downloadPath, expect.objectContaining({ credentials: "same-origin", headers: expect.objectContaining({ Accept: TAX_WORKBOOK_CONTENT_TYPE }) })]);
  expect(download.names[0]).toBe("tax-review-2025.xlsx");
  expect(await (download.createObjectURL.mock.calls[0][0] as Blob).text()).toBe("workbook bytes");
  await waitFor(() => expect(download.revokeObjectURL).toHaveBeenCalledWith("blob:workbook"));
});

it("keeps a stale case locked until the user reloads rather than exporting another revision", async () => {
  const fetchMock = vi.fn().mockResolvedValueOnce(json({ detail: "Review changed." }, 409));
  vi.stubGlobal("fetch", fetchMock);
  const onReload = vi.fn();
  render(<TaxWorkbookDownload taxCase={taxCase} onReload={onReload} />);
  fireEvent.click(screen.getByRole("button", { name: "Download workbook" }));
  expect(await screen.findByRole("alert")).toHaveTextContent("This review changed.");
  expect(screen.getByRole("button", { name: "Download workbook" })).toBeDisabled();
  expect(fetchMock).toHaveBeenCalledOnce();
  fireEvent.click(screen.getByRole("button", { name: "Reload review" }));
  expect(onReload).toHaveBeenCalledOnce();
  expect(fetchMock).toHaveBeenCalledOnce();
});

it("reports download failure in place and permits a deliberate retry", async () => {
  const download = mockDownload();
  const fetchMock = vi.fn().mockResolvedValueOnce(prepared()).mockResolvedValueOnce(json({ detail: "Unavailable" }, 503));
  vi.stubGlobal("fetch", fetchMock);
  render(<TaxWorkbookDownload taxCase={taxCase} onReload={vi.fn()} />);
  fireEvent.click(screen.getByRole("button", { name: "Download workbook" }));
  expect(await screen.findByRole("alert")).toHaveTextContent("Could not download the workbook.");
  expect(screen.getByRole("button", { name: "Download workbook" })).toBeEnabled();
  expect(download.click).not.toHaveBeenCalled();
});

it("refuses an off-origin prepared download instead of sending credentials or navigating away", async () => {
  const fetchMock = vi.fn().mockResolvedValueOnce(prepared({ download_url: "https://other.example/api/private.xlsx" }));
  vi.stubGlobal("fetch", fetchMock);
  render(<TaxWorkbookDownload taxCase={taxCase} onReload={vi.fn()} />);
  fireEvent.click(screen.getByRole("button", { name: "Download workbook" }));
  expect(await screen.findByRole("alert")).toHaveTextContent("Could not download");
  expect(fetchMock).toHaveBeenCalledOnce();
});

it("downloads a saved chat link directly and preserves the server's filename", async () => {
  const download = mockDownload();
  const fetchMock = vi.fn().mockResolvedValueOnce(new Response("saved bytes", { headers: {
    "Content-Type": TAX_WORKBOOK_CONTENT_TYPE,
    "Content-Disposition": 'attachment; filename="tax-review-revision-24.xlsx"',
  } }));
  vi.stubGlobal("fetch", fetchMock);
  render(<TaxWorkbookLink href={downloadPath}>Download saved workbook</TaxWorkbookLink>);
  fireEvent.click(screen.getByRole("link", { name: "Download saved workbook" }));
  await waitFor(() => expect(download.click).toHaveBeenCalledOnce());
  expect(fetchMock.mock.calls[0][0]).toBe(downloadPath);
  expect(download.names[0]).toBe("tax-review-revision-24.xlsx");
});

it("shows a failed saved chat link inline without opening a raw error page", async () => {
  const fetchMock = vi.fn().mockResolvedValueOnce(json({ detail: "Not found" }, 404));
  vi.stubGlobal("fetch", fetchMock);
  render(<TaxWorkbookLink href={downloadPath}>Download saved workbook</TaxWorkbookLink>);
  const link = screen.getByRole("link", { name: "Download saved workbook" });
  expect(link).not.toHaveAttribute("target");
  fireEvent.click(link);
  expect(await screen.findByRole("alert")).toHaveTextContent("This saved download is no longer available.");
  expect(screen.getByRole("alert")).toHaveTextContent("Download workbook in the Toolbox");
});
