import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { useState } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { KnowledgeSourcePicker } from "./KnowledgeSourcePicker";
import * as api from "./api";
import type { KnowledgeSourceCandidate, KnowledgeSourceCandidateList, KnowledgeSourceRef, SearchSource } from "./api";

const reference = { source_id: "file_one", revision: "saved_revision" };
function candidate(patch: Partial<KnowledgeSourceCandidate> = {}): KnowledgeSourceCandidate {
  return { source_id: "file_one", kind: "file", title: "Research notes · Text", private_required: false, ...patch };
}
function source(patch: Partial<SearchSource> = {}): SearchSource {
  return {
    source_id: "file_one", revision: "current_revision", kind: "file", title: "Research notes · Text", text: "Saved reference text.",
    offset: 0, next_offset: null, total_chars: 21, target: { kind: "file", file_id: "file_one", artifact: "text" },
    availability: "available", private_required: false, ...patch,
  };
}
function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason: unknown) => void;
  const promise = new Promise<T>((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}
function Harness({ initial = [], changed = vi.fn() }: { initial?: KnowledgeSourceRef[]; changed?: (sources: KnowledgeSourceRef[]) => void }) {
  const [sources, setSources] = useState(initial);
  return <KnowledgeSourcePicker projectId="project_one" sources={sources} onChange={(next) => { changed(next); setSources(next); }} />;
}
function expand() { fireEvent.click(screen.getByRole("button", { name: /^Source references /u })); }
async function previewFirst() { fireEvent.click(await screen.findByRole("button", { name: "Preview Research notes · Text" })); }

beforeEach(() => {
  vi.spyOn(api, "listKnowledgeSourceCandidates").mockResolvedValue({ items: [candidate()], next_cursor: null });
  vi.spyOn(api, "getKnowledgeSourcePreview").mockResolvedValue(source());
});
afterEach(() => { cleanup(); vi.restoreAllMocks(); });

describe("KnowledgeSourcePicker", () => {
  it("does no canonical reads until the source panel is explicitly opened", async () => {
    render(<Harness />);
    expect(api.listKnowledgeSourceCandidates).not.toHaveBeenCalled();
    expect(api.getKnowledgeSourcePreview).not.toHaveBeenCalled();
    expand();
    await screen.findByRole("button", { name: "Preview Research notes · Text" });
    expect(api.listKnowledgeSourceCandidates).toHaveBeenCalledExactlyOnceWith("project_one", "file", undefined, expect.any(AbortSignal));
    expect(api.getKnowledgeSourcePreview).not.toHaveBeenCalled();
  });

  it("quotes source text safely and attaches only the ID and reviewed revision after an explicit click", async () => {
    const changed = vi.fn();
    const text = '<img src="https://example.com/track"> **claim** [link](https://example.com)';
    vi.mocked(api.getKnowledgeSourcePreview).mockResolvedValue(source({ text, availability: "text_only", private_required: true }));
    const { container } = render(<Harness changed={changed} />);
    expand();
    await previewFirst();
    expect(await screen.findByLabelText("Source preview text")).toHaveTextContent(text);
    expect(screen.getByText("Original unavailable; retained text shown.")).toBeVisible();
    expect(screen.getByText(/Revision current_revision · Private/u)).toBeVisible();
    expect(container.querySelectorAll("img, a, strong")).toHaveLength(0);
    expect(changed).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: "Attach reference" }));
    expect(changed).toHaveBeenCalledExactlyOnceWith([{ source_id: "file_one", revision: "current_revision" }]);
    expect(screen.getByRole("button", { name: "Reference attached" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Source references (1/8)" })).toBeVisible();
  });

  it("browses each canonical source kind without consulting Search", async () => {
    vi.mocked(api.listKnowledgeSourceCandidates)
      .mockResolvedValueOnce({ items: [candidate()], next_cursor: null })
      .mockResolvedValueOnce({ items: [candidate({ source_id: "message_one", kind: "message", title: "Project chat" })], next_cursor: null })
      .mockResolvedValueOnce({ items: [candidate({ source_id: "report_one", kind: "agent", title: "Outside project report" })], next_cursor: null });
    vi.spyOn(api, "searchContent");
    render(<Harness />);
    expand();
    await screen.findByRole("button", { name: "Preview Research notes · Text" });
    fireEvent.change(screen.getByLabelText("Source type"), { target: { value: "message" } });
    expect(await screen.findByRole("button", { name: "Preview Project chat" })).toBeVisible();
    expect(screen.queryByRole("button", { name: "Preview Research notes · Text" })).not.toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("Source type"), { target: { value: "agent" } });
    expect(await screen.findByRole("button", { name: "Preview Outside project report" })).toBeVisible();
    expect(api.listKnowledgeSourceCandidates).toHaveBeenLastCalledWith("project_one", "agent", undefined, expect.any(AbortSignal));
    expect(api.searchContent).not.toHaveBeenCalled();
  });

  it("deduplicates candidate pages and stops a repeated cursor", async () => {
    vi.mocked(api.listKnowledgeSourceCandidates)
      .mockResolvedValueOnce({ items: [candidate()], next_cursor: "cursor_one" })
      .mockResolvedValueOnce({ items: [candidate(), candidate({ source_id: "file_two", title: "Second source" })], next_cursor: "cursor_one" });
    render(<Harness />);
    expand();
    fireEvent.click(await screen.findByRole("button", { name: "Load more sources" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("repeated a page");
    expect(screen.getAllByRole("button", { name: "Preview Research notes · Text" })).toHaveLength(1);
    expect(screen.getByRole("button", { name: "Preview Second source" })).toBeVisible();
    expect(screen.queryByRole("button", { name: "Load more sources" })).not.toBeInTheDocument();
    expect(api.listKnowledgeSourceCandidates).toHaveBeenCalledTimes(2);
  });

  it("bounds even an oversized candidate response", async () => {
    vi.mocked(api.listKnowledgeSourceCandidates).mockResolvedValue({
      items: Array.from({ length: 501 }, (_, index) => candidate({ source_id: `file_${index}`, title: `Source ${index}` })), next_cursor: "more",
    });
    render(<Harness />);
    expand();
    await screen.findByRole("button", { name: "Preview Source 0" });
    expect(within(screen.getByLabelText("Available source references")).getAllByRole("button")).toHaveLength(500);
    expect(screen.queryByRole("button", { name: "Preview Source 500" })).not.toBeInTheDocument();
    expect(screen.getByText(/reached its source limit/u)).toBeVisible();
    expect(screen.queryByRole("button", { name: "Load more sources" })).not.toBeInTheDocument();
  });

  it("retries a failed next page without dropping prior candidates or attached references", async () => {
    const changed = vi.fn();
    vi.mocked(api.listKnowledgeSourceCandidates)
      .mockResolvedValueOnce({ items: [candidate()], next_cursor: "cursor_one" })
      .mockRejectedValueOnce(new TypeError("connection lost"))
      .mockResolvedValueOnce({ items: [candidate({ source_id: "file_two", title: "Second source" })], next_cursor: null });
    render(<Harness initial={[reference]} changed={changed} />);
    expand();
    fireEvent.click(await screen.findByRole("button", { name: "Load more sources" }));
    fireEvent.click(await screen.findByRole("button", { name: "Retry source list" }));
    expect(await screen.findByRole("button", { name: "Preview Second source" })).toBeVisible();
    expect(screen.getByRole("button", { name: "Preview Research notes · Text" })).toBeVisible();
    expect(api.listKnowledgeSourceCandidates).toHaveBeenLastCalledWith("project_one", "file", "cursor_one", expect.any(AbortSignal));
    expect(screen.getByRole("button", { name: "Remove reference to file_one" })).toBeVisible();
    expect(changed).not.toHaveBeenCalled();
  });

  it("refreshes an invalid paginated cursor from the first page without changing draft references", async () => {
    const changed = vi.fn();
    const pending = deferred<SearchSource>();
    vi.mocked(api.getKnowledgeSourcePreview).mockReturnValue(pending.promise);
    vi.mocked(api.listKnowledgeSourceCandidates)
      .mockResolvedValueOnce({ items: [candidate()], next_cursor: "expired_cursor" })
      .mockRejectedValueOnce(new api.ApiError(422, "Source cursor is invalid."))
      .mockResolvedValueOnce({ items: [candidate({ source_id: "file_two", title: "Fresh project source" })], next_cursor: null });
    render(<Harness initial={[reference]} changed={changed} />);
    expand();
    await previewFirst();
    const previewSignal = vi.mocked(api.getKnowledgeSourcePreview).mock.calls[0][4]!;
    fireEvent.click(screen.getByRole("button", { name: "Load more sources" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("The source list changed. Refresh the source list to continue.");
    fireEvent.click(await screen.findByRole("button", { name: "Refresh source list" }));
    expect(previewSignal.aborted).toBe(true);
    expect(await screen.findByRole("button", { name: "Preview Fresh project source" })).toBeVisible();
    expect(api.listKnowledgeSourceCandidates).toHaveBeenLastCalledWith("project_one", "file", undefined, expect.any(AbortSignal));
    expect(screen.queryByRole("button", { name: "Preview Research notes · Text" })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Remove reference to file_one" })).toBeVisible();
    expect(screen.queryByRole("button", { name: "Refresh source list" })).not.toBeInTheDocument();
    await act(async () => pending.resolve(source({ text: "Late preview before source refresh" })));
    expect(screen.queryByText("Late preview before source refresh")).not.toBeInTheDocument();
    expect(changed).not.toHaveBeenCalled();
  });

  it("enforces eight unique source IDs and makes removal explicit", async () => {
    const initial = Array.from({ length: 8 }, (_, index) => ({ source_id: `attached_${index}`, revision: "one" }));
    const changed = vi.fn();
    render(<Harness initial={initial} changed={changed} />);
    expand();
    await previewFirst();
    expect(await screen.findByRole("button", { name: "Attach reference" })).toBeDisabled();
    expect(changed).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: "Remove reference to attached_0" }));
    expect(changed).toHaveBeenLastCalledWith(initial.slice(1));
    fireEvent.click(screen.getByRole("button", { name: "Attach reference" }));
    expect(changed).toHaveBeenLastCalledWith([...initial.slice(1), { source_id: "file_one", revision: "current_revision" }]);
  });

  it("does not attach a second revision of an already attached source through browsing", async () => {
    const changed = vi.fn();
    render(<Harness initial={[reference]} changed={changed} />);
    expand();
    await previewFirst();
    expect(await screen.findByRole("button", { name: "Reference attached" })).toBeDisabled();
    expect(changed).not.toHaveBeenCalled();
  });

  it("pins saved review and uses an unpinned fresh read only for explicit revision replacement", async () => {
    const changed = vi.fn();
    const other = { source_id: "file_other", revision: "other_revision" };
    vi.mocked(api.getKnowledgeSourcePreview).mockResolvedValueOnce(source({ revision: reference.revision })).mockResolvedValueOnce(source());
    render(<Harness initial={[reference, other]} changed={changed} />);
    expand();
    fireEvent.click(screen.getByRole("button", { name: "View saved version of file_one" }));
    await screen.findByLabelText("Source preview text");
    expect(api.getKnowledgeSourcePreview).toHaveBeenLastCalledWith("project_one", "file_one", "saved_revision", 0, expect.any(AbortSignal));
    fireEvent.click(screen.getByRole("button", { name: "Review current version of file_one" }));
    expect(await screen.findByText(/Attached revision: saved_revision. Current revision: current_revision/u)).toBeVisible();
    expect(api.getKnowledgeSourcePreview).toHaveBeenLastCalledWith("project_one", "file_one", undefined, 0, expect.any(AbortSignal));
    expect(changed).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: "Use current revision" }));
    expect(changed).toHaveBeenCalledExactlyOnceWith([{ source_id: "file_one", revision: "current_revision" }, other]);
  });

  it("keeps current-version preview pagination pinned and can return to the prior page", async () => {
    vi.mocked(api.getKnowledgeSourcePreview)
      .mockResolvedValueOnce(source({ next_offset: 4000, total_chars: 4100 }))
      .mockResolvedValueOnce(source({ text: "Second page", offset: 4000, total_chars: 4100 }))
      .mockResolvedValueOnce(source({ next_offset: 4000, total_chars: 4100 }));
    render(<Harness initial={[reference]} />);
    expand();
    fireEvent.click(screen.getByRole("button", { name: "Review current version of file_one" }));
    fireEvent.click(await screen.findByRole("button", { name: "Next source text" }));
    expect(await screen.findByLabelText("Source preview text")).toHaveTextContent("Second page");
    expect(api.getKnowledgeSourcePreview).toHaveBeenLastCalledWith("project_one", "file_one", "current_revision", 4000, expect.any(AbortSignal));
    fireEvent.click(screen.getByRole("button", { name: "Previous source text" }));
    expect(await screen.findByLabelText("Source preview text")).toHaveTextContent("Saved reference text.");
    expect(api.getKnowledgeSourcePreview).toHaveBeenLastCalledWith("project_one", "file_one", "current_revision", 0, expect.any(AbortSignal));
  });

  it.each([403, 404, 409])("keeps attached references unchanged when preview hits HTTP %i", async (status) => {
    const changed = vi.fn();
    vi.mocked(api.getKnowledgeSourcePreview).mockRejectedValue(new api.ApiError(status, "sensitive server details"));
    render(<Harness initial={[reference]} changed={changed} />);
    expand();
    fireEvent.click(screen.getByRole("button", { name: "Review current version of file_one" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Your attached references are unchanged.");
    expect(screen.getByRole("alert")).not.toHaveTextContent("sensitive server details");
    expect(screen.queryByLabelText("Source preview text")).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Use current revision" })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Remove reference to file_one" })).toBeVisible();
    expect(changed).not.toHaveBeenCalled();
  });

  it.each([
    { revision: "wrong_revision" },
    { source_id: "foreign_source" },
    { next_offset: 0 },
  ])("rejects mismatched or nonprogressing pinned previews: %j", async (patch) => {
    vi.mocked(api.getKnowledgeSourcePreview).mockResolvedValue(source({ revision: reference.revision, ...patch }));
    const changed = vi.fn();
    render(<Harness initial={[reference]} changed={changed} />);
    expand();
    fireEvent.click(screen.getByRole("button", { name: "View saved version of file_one" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("This source changed.");
    expect(screen.queryByLabelText("Source preview text")).not.toBeInTheDocument();
    expect(changed).not.toHaveBeenCalled();
  });

  it.each(["close preview", "collapse", "disabled", "hidden", "project", "unmount", "kind"] as const)("aborts a pending preview on %s and ignores a late completion", async (action) => {
    const pending = deferred<SearchSource>();
    vi.mocked(api.getKnowledgeSourcePreview).mockReturnValue(pending.promise);
    const changed = vi.fn();
    const props = { projectId: "project_one", sources: [reference], onChange: changed };
    const { rerender, unmount } = render(<KnowledgeSourcePicker {...props} />);
    expand();
    fireEvent.click(screen.getByRole("button", { name: "Review current version of file_one" }));
    const signal = vi.mocked(api.getKnowledgeSourcePreview).mock.calls[0][4]!;
    if (action === "close preview") fireEvent.click(screen.getByRole("button", { name: "Close preview" }));
    else if (action === "collapse") expand();
    else if (action === "disabled") rerender(<KnowledgeSourcePicker {...props} disabled />);
    else if (action === "hidden") rerender(<KnowledgeSourcePicker {...props} visible={false} />);
    else if (action === "project") rerender(<KnowledgeSourcePicker {...props} projectId="project_two" />);
    else if (action === "kind") fireEvent.change(screen.getByLabelText("Source type"), { target: { value: "agent" } });
    else unmount();
    expect(signal.aborted).toBe(true);
    await act(async () => pending.resolve(source({ text: "Late private preview" })));
    expect(screen.queryByText("Late private preview")).not.toBeInTheDocument();
    expect(changed).not.toHaveBeenCalled();
  });

  it.each(["collapse", "disabled", "hidden", "project", "unmount"] as const)("aborts a pending candidate list on %s and ignores a late completion", async (action) => {
    const pending = deferred<KnowledgeSourceCandidateList>();
    vi.mocked(api.listKnowledgeSourceCandidates).mockReturnValue(pending.promise);
    const props = { projectId: "project_one", sources: [reference], onChange: vi.fn() };
    const { rerender, unmount } = render(<KnowledgeSourcePicker {...props} />);
    expand();
    const signal = vi.mocked(api.listKnowledgeSourceCandidates).mock.calls[0][3]!;
    if (action === "collapse") expand();
    else if (action === "disabled") rerender(<KnowledgeSourcePicker {...props} disabled />);
    else if (action === "hidden") rerender(<KnowledgeSourcePicker {...props} visible={false} />);
    else if (action === "project") rerender(<KnowledgeSourcePicker {...props} projectId="project_two" />);
    else unmount();
    expect(signal.aborted).toBe(true);
    await act(async () => pending.resolve({ items: [candidate({ title: "Late private title" })], next_cursor: null }));
    expect(screen.queryByText("Late private title")).not.toBeInTheDocument();
  });

  it("clears preview text immediately while another canonical preview is pending", async () => {
    const pending = deferred<SearchSource>();
    vi.mocked(api.getKnowledgeSourcePreview).mockResolvedValueOnce(source()).mockReturnValueOnce(pending.promise);
    render(<Harness initial={[reference]} />);
    expand();
    await previewFirst();
    await screen.findByLabelText("Source preview text");
    fireEvent.click(screen.getByRole("button", { name: "Review current version of file_one" }));
    expect(screen.queryByLabelText("Source preview text")).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Attach reference" })).not.toBeInTheDocument();
    await act(async () => pending.resolve(source()));
    expect(await screen.findByLabelText("Source preview text")).toBeVisible();
  });

  it("reopens collapsed sources with fresh canonical reads and preserves only the parent references", async () => {
    render(<Harness initial={[reference]} />);
    expand();
    await screen.findByRole("button", { name: "Preview Research notes · Text" });
    expand();
    expand();
    await waitFor(() => expect(api.listKnowledgeSourceCandidates).toHaveBeenCalledTimes(2));
    expect(screen.getByRole("button", { name: "Remove reference to file_one" })).toBeVisible();
    expect(api.getKnowledgeSourcePreview).not.toHaveBeenCalled();
  });
});
