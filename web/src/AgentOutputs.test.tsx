import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { AgentOutputs } from "./AgentOutputs";
import { type AgentOutput } from "./api";

const output: AgentOutput = { file_id: "file_owned-1", filename: "checklist.md", download_url: "/api/files/file_owned-1/download", created_at: "2026-11-02T10:00:00Z" };

function mockDownload() {
  const createObjectURL = vi.fn((blob: Blob) => { void blob; return "blob:owned-output"; });
  const revokeObjectURL = vi.fn();
  vi.stubGlobal("URL", { createObjectURL, revokeObjectURL });
  const click = vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => undefined);
  return { createObjectURL, revokeObjectURL, click };
}

afterEach(() => { cleanup(); vi.restoreAllMocks(); vi.unstubAllGlobals(); });

describe("AgentOutputs", () => {
  it("downloads only on demand through the authenticated file path and the server filename", async () => {
    const download = mockDownload();
    const fetchMock = vi.fn().mockResolvedValue(new Response("saved output bytes", { headers: { "Content-Disposition": 'attachment; filename="canonical-checklist.md"' } }));
    vi.stubGlobal("fetch", fetchMock);
    render(<AgentOutputs outputs={[output]} />);
    expect(screen.getByRole("heading", { name: "Files" })).toBeVisible();
    expect(fetchMock).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: "Download checklist.md" }));
    await waitFor(() => expect(download.click).toHaveBeenCalledOnce());
    expect(fetchMock).toHaveBeenCalledExactlyOnceWith(output.download_url, expect.objectContaining({ credentials: "same-origin", signal: expect.any(AbortSignal) }));
    expect(download.click.mock.instances[0]).toHaveAttribute("download", "canonical-checklist.md");
    expect(await download.createObjectURL.mock.calls[0][0]?.text()).toBe("saved output bytes");
    await waitFor(() => expect(download.revokeObjectURL).toHaveBeenCalledWith("blob:owned-output"));
  });

  it("renders filenames as plain text and ignores unsafe or duplicate navigation payloads", () => {
    const filename = "<img src=x onerror=alert(1)>.md";
    const { container } = render(<AgentOutputs outputs={[
      { ...output, filename }, output,
      { ...output, file_id: "file_other", download_url: "https://outside.example/download" },
      { ...output, file_id: "../wrong", download_url: "/api/files/..%2Fwrong/download" },
    ]} />);
    expect(screen.getByRole("button", { name: `Download ${filename}` })).toBeVisible();
    expect(screen.getAllByRole("button")).toHaveLength(1);
    expect(container.querySelector("img")).toBeNull();
    expect(container.querySelector("a")).toBeNull();
  });

  it("reports a removed file and allows an explicit retry without exposing the server error", async () => {
    const download = mockDownload();
    const fetchMock = vi.fn().mockResolvedValueOnce(new Response(JSON.stringify({ detail: "private storage path" }), { status: 404 }))
      .mockResolvedValueOnce(new Response("saved", { headers: { "Content-Disposition": 'attachment; filename="checklist.md"' } }));
    vi.stubGlobal("fetch", fetchMock);
    render(<AgentOutputs outputs={[output]} />);
    fireEvent.click(screen.getByRole("button", { name: "Download checklist.md" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("This file is no longer available. Check My Files.");
    expect(screen.queryByText("private storage path")).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Download checklist.md" }));
    await waitFor(() => expect(download.click).toHaveBeenCalledOnce());
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it("cancels late downloads when the task/file is removed", async () => {
    const download = mockDownload();
    let release!: (value: Response) => void;
    const fetchMock = vi.fn().mockImplementation(() => new Promise<Response>((resolve) => { release = resolve; }));
    vi.stubGlobal("fetch", fetchMock);
    const { rerender } = render(<AgentOutputs outputs={[output]} />);
    fireEvent.click(screen.getByRole("button", { name: "Download checklist.md" }));
    expect(screen.getByRole("button", { name: "Downloading checklist.md" })).toBeDisabled();
    const signal = fetchMock.mock.calls[0][1].signal as AbortSignal;
    rerender(<AgentOutputs outputs={[]} />);
    expect(signal.aborted).toBe(true);
    await act(async () => release(new Response("late", { headers: { "Content-Disposition": 'attachment; filename="late.md"' } })));
    expect(download.createObjectURL).not.toHaveBeenCalled();
    expect(download.click).not.toHaveBeenCalled();
  });

  it("uses the recorded current file even when an older identity is reused after a newer file", async () => {
    const download = mockDownload();
    const fetchMock = vi.fn().mockImplementation(async () => new Response("saved", { headers: { "Content-Disposition": 'attachment; filename="checklist.md"' } }));
    vi.stubGlobal("fetch", fetchMock);
    const previous = { ...output, file_id: "file_newer", download_url: "/api/files/file_newer/download", created_at: "2026-11-09T12:00:00Z", current: false };
    render(<AgentOutputs outputs={[previous, { ...output, current: true }]} />);
    const currentFile = within(screen.getByRole("list", { name: "Current files" })).getByRole("button", { name: /^Download checklist\.md, current file, saved / });
    expect(currentFile).toHaveTextContent("Current");
    expect(currentFile.querySelector("time")).toHaveAttribute("dateTime", output.created_at);
    const history = screen.getByText(/^Previous files/, { selector: "summary" }).closest("details")!;
    expect(history).not.toHaveAttribute("open");
    expect(within(history).getByRole("button", { hidden: true })).not.toBeVisible();
    fireEvent.click(currentFile);
    await waitFor(() => expect(download.click).toHaveBeenCalledOnce());
    expect(fetchMock.mock.calls[0][0]).toBe(output.download_url);
    fireEvent.click(within(history).getByText(/^Previous files/, { selector: "summary" }));
    fireEvent.click(within(history).getByRole("button", { name: /^Download checklist\.md, previous file, saved / }));
    await waitFor(() => expect(download.click).toHaveBeenCalledTimes(2));
    expect(fetchMock.mock.calls[1][0]).toBe(previous.download_url);
  });

  it("keeps legacy files unclassified and distinguishes same-name downloads without guessing from dates", () => {
    const other = { ...output, current: null, file_id: "file_legacy", download_url: "/api/files/file_legacy/download", created_at: "2026-11-09T10:00:00Z" };
    render(<AgentOutputs outputs={[output, other]} />);
    expect(screen.getByText("Version information is unavailable for these saved files.")).toBeVisible();
    expect(screen.queryByRole("list", { name: "Current files" })).not.toBeInTheDocument();
    expect(screen.queryByText(/^Previous files/, { selector: "summary" })).not.toBeInTheDocument();
    const files = within(screen.getByRole("list", { name: "Saved files" })).getAllByRole("button", { name: /^Download checklist\.md, saved file, saved / });
    expect(files).toHaveLength(2);
    expect(files[0]).toHaveAccessibleName(new RegExp(`file ${output.file_id}$`));
    expect(files[1]).toHaveAccessibleName(/file file_legacy$/);
    expect(screen.queryByText("Current")).not.toBeInTheDocument();
  });

  it("does not promote a previous file after the current file is removed and cancels its late download", async () => {
    const download = mockDownload();
    let release!: (value: Response) => void;
    const fetchMock = vi.fn().mockImplementation(() => new Promise<Response>((resolve) => { release = resolve; }));
    vi.stubGlobal("fetch", fetchMock);
    const previous = { ...output, file_id: "file_previous", download_url: "/api/files/file_previous/download", current: false };
    const { rerender } = render(<AgentOutputs outputs={[{ ...output, current: true }, previous]} />);
    fireEvent.click(within(screen.getByRole("list", { name: "Current files" })).getByRole("button"));
    const signal = fetchMock.mock.calls[0][1].signal as AbortSignal;
    rerender(<AgentOutputs outputs={[previous]} />);
    expect(signal.aborted).toBe(true);
    expect(screen.queryByRole("list", { name: "Current files" })).not.toBeInTheDocument();
    expect(screen.getByText("No current file is available. Previous files are below.")).toBeVisible();
    const history = screen.getByText(/^Previous files/, { selector: "summary" }).closest("details")!;
    expect(within(history).getByRole("button", { hidden: true })).not.toBeVisible();
    await act(async () => release(new Response("late", { headers: { "Content-Disposition": 'attachment; filename="deleted.md"' } })));
    expect(download.click).not.toHaveBeenCalled();
  });

  it("does not show an empty Files area", () => {
    render(<AgentOutputs outputs={[]} />);
    expect(screen.queryByRole("heading", { name: "Files" })).not.toBeInTheDocument();
  });
});
