import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { AgentFileInputsView, AgentFilePicker } from "./AgentFileInputs";
import * as api from "./api";

const limits = { max_name_chars: 100, max_instructions_chars: 4000, max_files: 20 };
function projectFile(id: string, patch: Partial<api.AudreyProjectFile> = {}): api.AudreyProjectFile {
  return { id, filename: `${id}.txt`, mime: "text/plain", kind: "text", bytes: 200,
    status: "ready", uploaded_at: "2026-11-03T12:00:00Z", added_at: "2026-11-03T12:00:00Z", ...patch };
}

afterEach(() => { cleanup(); vi.restoreAllMocks(); });

describe("Agent Project file chooser", () => {
  it("loads only on explicit expand, offers ready files and enforces the three-file selection limit", async () => {
    const files = [projectFile("one"), projectFile("two", { kind: "video" }), projectFile("three", { origin: "generated" }), projectFile("four"), projectFile("pending", { status: "processing" })];
    const read = vi.spyOn(api, "listProjectFiles").mockResolvedValue({ items: files, limits });
    const change = vi.fn();
    const { rerender } = render(<AgentFilePicker projectId="project-one" selected={[]} onChange={change} visible disabled={false} />);
    expect(read).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: "Project files Optional" }));
    const one = await screen.findByRole("checkbox", { name: "one.txt" });
    expect(screen.queryByRole("checkbox", { name: /pending/ })).not.toBeInTheDocument();
    expect(screen.getByRole("checkbox", { name: "two.txt" })).toBeVisible();
    expect(screen.getByRole("checkbox", { name: "three.txt" })).toBeVisible();
    fireEvent.click(one); expect(change).toHaveBeenLastCalledWith([files[0]]);
    rerender(<AgentFilePicker projectId="project-one" selected={files.slice(0, 3)} onChange={change} visible disabled={false} />);
    expect(screen.getByRole("checkbox", { name: "four.txt" })).toBeDisabled();
    expect(screen.getByRole("checkbox", { name: "one.txt" })).toBeEnabled();
    fireEvent.click(screen.getByRole("checkbox", { name: "one.txt" }));
    expect(change).toHaveBeenLastCalledWith(files.slice(1, 3));
  });

  it.each(["workspace", "page"])("aborts reads when the %s is hidden, discards late responses and revalidates on reopening", async (hidden) => {
    let visibility: DocumentVisibilityState = "visible";
    vi.spyOn(document, "visibilityState", "get").mockImplementation(() => visibility);
    let release!: (page: api.AudreyProjectFileList) => void;
    const read = vi.spyOn(api, "listProjectFiles").mockImplementationOnce(() => new Promise((resolve) => { release = resolve; })).mockResolvedValue({ items: [projectFile("current")], limits });
    const props = { projectId: "project-one", selected: [], onChange: vi.fn(), disabled: false };
    const { rerender } = render(<AgentFilePicker {...props} visible />);
    fireEvent.click(screen.getByRole("button", { name: "Project files Optional" }));
    await waitFor(() => expect(read).toHaveBeenCalledTimes(1));
    const signal = read.mock.calls[0][1]!;
    if (hidden === "workspace") rerender(<AgentFilePicker {...props} visible={false} />);
    else { visibility = "hidden"; fireEvent(document, new Event("visibilitychange")); }
    expect(signal.aborted).toBe(true);
    await act(async () => release({ items: [projectFile("stale")], limits }));
    expect(screen.queryByText("stale.txt")).not.toBeInTheDocument();
    if (hidden === "workspace") rerender(<AgentFilePicker {...props} visible />);
    else { visibility = "visible"; fireEvent(document, new Event("visibilitychange")); }
    expect(await screen.findByRole("checkbox", { name: "current.txt" })).toBeVisible();
    expect(read).toHaveBeenCalledTimes(2);
  });

  it("keeps selected files after load failure and lets users remove unavailable selections after refresh", async () => {
    const selected = projectFile("selected"); const change = vi.fn();
    const read = vi.spyOn(api, "listProjectFiles").mockRejectedValueOnce(new Error("unsafe server error")).mockResolvedValueOnce({ items: [], limits });
    render(<AgentFilePicker projectId="project-one" selected={[selected]} onChange={change} visible disabled={false} />);
    fireEvent.click(screen.getByRole("button", { name: "Project files 1 / 3 selected" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Your selections are kept.");
    expect(screen.queryByText("unsafe server error")).not.toBeInTheDocument(); expect(change).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: "Refresh files" }));
    fireEvent.click(await screen.findByRole("button", { name: "Remove selected.txt" }));
    expect(read).toHaveBeenCalledTimes(2); expect(change).toHaveBeenCalledWith([]);
  });
});

describe("Agent input file metadata", () => {
  it("shows frozen extract metadata, truncation and generated provenance without fetching source bodies", () => {
    const read = vi.spyOn(api, "listProjectFiles");
    render(<AgentFileInputsView inputs={{ current: true, files: [{ file_id: "one", filename: "<img onerror=alert(1)>.md", artifact: "text", chars: 1200, total_chars: 6000, truncated: true, origin: "generated" }] }} />);
    fireEvent.click(screen.getByText("Input files · 1", { selector: "summary" }));
    expect(screen.getByText("<img onerror=alert(1)>.md")).toBeVisible();
    expect(screen.getByText(/Saved text · Working output · 1,200 of 6,000 characters · Shortened/)).toBeVisible();
    expect(document.querySelector("img")).toBeNull(); expect(read).not.toHaveBeenCalled();
  });
});
