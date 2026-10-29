import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { SearchWorkspace } from "./SearchWorkspace";
import * as api from "./api";
import type { SearchResponse, SearchResult } from "./api";

function result(patch: Partial<SearchResult> = {}): SearchResult {
  return {
    id: "message:msg_first:0", source_id: "msg_first", kind: "message", revision: "3",
    title: "Project decisions", excerpt: "We chose the local storage option.",
    highlights: [{ start: 13, end: 18 }], offset: 100, updated_at: "2026-10-29T16:00:00Z",
    project_id: null, private_required: false, availability: "available",
    target: { kind: "message", conversation_id: "con_first", message_id: "msg_first", sequence: 20 },
    ...patch,
  };
}

function response(patch: Partial<SearchResponse> = {}): SearchResponse {
  return { items: [], next_cursor: null, partial: false, indexing: false, warnings: [], ...patch };
}

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason: Error) => void;
  const promise = new Promise<T>((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}

function search(query = "local storage") {
  fireEvent.change(screen.getByRole("searchbox", { name: "Search your content" }), { target: { value: query } });
  fireEvent.submit(screen.getByRole("search"));
}

beforeEach(() => {
  vi.spyOn(api, "listProjects").mockResolvedValue({ items: [], next_cursor: null, limits: { max_name_chars: 100, max_instructions_chars: 5000, max_files: 20 } });
  vi.spyOn(api, "searchContent").mockResolvedValue(response());
});

afterEach(() => { cleanup(); vi.restoreAllMocks(); });

describe("SearchWorkspace", () => {
  it("waits for an explicit nonempty query before making a content request", async () => {
    render(<SearchWorkspace onOpenResult={vi.fn()} />);
    expect(screen.getByRole("heading", { name: "Search" })).toBeVisible();
    expect(screen.getByText("Your work, easier to find")).toBeVisible();
    expect(screen.getByRole("button", { name: "Search" })).toBeDisabled();
    fireEvent.change(screen.getByRole("searchbox"), { target: { value: "   " } });
    fireEvent.submit(screen.getByRole("search"));
    expect(api.searchContent).not.toHaveBeenCalled();
    fireEvent.change(screen.getByRole("searchbox"), { target: { value: "local storage" } });
    expect(api.searchContent).not.toHaveBeenCalled();
    fireEvent.submit(screen.getByRole("search"));
    expect(await screen.findByText("No matches found. Try fewer words or broader filters.")).toBeVisible();
    expect(api.searchContent).toHaveBeenCalledExactlyOnceWith("local storage", { kind: "all", project_id: undefined, archived: "all", since: undefined, until: undefined }, expect.any(AbortSignal));
  });

  it("opens the exact native message target and highlights matching text", async () => {
    const item = result();
    vi.mocked(api.searchContent).mockResolvedValue(response({ items: [item] }));
    const open = vi.fn();
    const { container } = render(<SearchWorkspace onOpenResult={open} />);
    search();
    fireEvent.click(await screen.findByRole("button", { name: "Open chat: Project decisions" }));
    expect(open).toHaveBeenCalledExactlyOnceWith(item);
    expect(container.querySelector("mark")).toHaveTextContent("local");
    expect(container.querySelector(".search-excerpt")).toHaveTextContent(item.excerpt);
  });

  it("keeps unsafe markup and Markdown inert and handles Unicode and overlapping highlights", async () => {
    const excerpt = "🗂 local <img src=x onerror=alert(1)> [link](https://example.test)";
    vi.mocked(api.searchContent).mockResolvedValue(response({ items: [result({
      title: "<script>alert(1)</script>", excerpt,
      highlights: [{ start: 2, end: 7 }, { start: 3, end: 5 }, { start: -2, end: 4 }, { start: 5, end: 9000 }],
    })] }));
    const { container } = render(<SearchWorkspace onOpenResult={vi.fn()} />);
    search();
    expect(await screen.findByRole("heading", { name: "<script>alert(1)</script>" })).toBeVisible();
    expect(container.querySelectorAll("mark")).toHaveLength(1);
    expect(container.querySelector("mark")).toHaveTextContent("local");
    expect(container.querySelector(".search-excerpt")).toHaveTextContent(excerpt);
    expect(container.querySelector("img, script, a")).toBeNull();
  });

  it("applies kind, project and archive filters to the last submitted query", async () => {
    vi.mocked(api.listProjects).mockResolvedValue({ items: [{ id: "prj_private", name: "Design", private_required: true, instructions: "", created_at: "", updated_at: "" }], next_cursor: null, limits: { max_name_chars: 100, max_instructions_chars: 5000, max_files: 20 } });
    render(<SearchWorkspace onOpenResult={vi.fn()} />);
    await screen.findByRole("option", { name: "Design · Private" });
    search();
    await waitFor(() => expect(api.searchContent).toHaveBeenCalledTimes(1));
    fireEvent.change(screen.getByRole("searchbox"), { target: { value: "unsubmitted change" } });
    fireEvent.click(screen.getByRole("button", { name: "Files" }));
    await waitFor(() => expect(api.searchContent).toHaveBeenLastCalledWith("local storage", expect.objectContaining({ kind: "file" }), expect.any(AbortSignal)));
    fireEvent.change(screen.getByRole("combobox", { name: "Project" }), { target: { value: "prj_private" } });
    await waitFor(() => expect(api.searchContent).toHaveBeenLastCalledWith("local storage", expect.objectContaining({ project_id: "prj_private", kind: "file" }), expect.any(AbortSignal)));
    fireEvent.change(screen.getByRole("combobox", { name: "Chat archive" }), { target: { value: "archived" } });
    await waitFor(() => expect(api.searchContent).toHaveBeenLastCalledWith("local storage", expect.objectContaining({ archived: "archived", project_id: "prj_private" }), expect.any(AbortSignal)));
    expect(screen.getByRole("button", { name: "Files" })).toHaveAttribute("aria-pressed", "true");
  });

  it("cancels an earlier search and ignores its late response", async () => {
    const first = deferred<SearchResponse>();
    const second = deferred<SearchResponse>();
    vi.mocked(api.searchContent).mockReturnValueOnce(first.promise).mockReturnValueOnce(second.promise);
    render(<SearchWorkspace onOpenResult={vi.fn()} />);
    search("first query");
    await waitFor(() => expect(api.searchContent).toHaveBeenCalledTimes(1));
    const firstSignal = vi.mocked(api.searchContent).mock.calls[0][2];
    search("second query");
    await waitFor(() => expect(api.searchContent).toHaveBeenCalledTimes(2));
    expect(firstSignal?.aborted).toBe(true);
    await act(async () => second.resolve(response({ items: [result({ title: "Current result" })] })));
    expect(await screen.findByRole("heading", { name: "Current result" })).toBeVisible();
    await act(async () => first.resolve(response({ items: [result({ title: "Stale result" })] })));
    expect(screen.queryByRole("heading", { name: "Stale result" })).not.toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Results for “second query”" })).toBeVisible();
  });

  it("ignores a late error from a replaced query", async () => {
    const first = deferred<SearchResponse>();
    vi.mocked(api.searchContent).mockReturnValueOnce(first.promise).mockResolvedValueOnce(response({ items: [result()] }));
    render(<SearchWorkspace onOpenResult={vi.fn()} />);
    search("first");
    search("second");
    await screen.findByRole("heading", { name: "Project decisions" });
    await act(async () => first.reject(new Error("Stale request failed")));
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("cancels hidden work and reruns the saved query when visible again", async () => {
    const pending = deferred<SearchResponse>();
    vi.mocked(api.searchContent).mockReturnValueOnce(pending.promise).mockResolvedValueOnce(response({ items: [result()] }));
    const { rerender } = render(<SearchWorkspace onOpenResult={vi.fn()} />);
    search();
    await waitFor(() => expect(api.searchContent).toHaveBeenCalledTimes(1));
    const signal = vi.mocked(api.searchContent).mock.calls[0][2];
    rerender(<SearchWorkspace visible={false} onOpenResult={vi.fn()} />);
    expect(signal?.aborted).toBe(true);
    await act(async () => pending.resolve(response({ items: [result({ title: "Hidden result" })] })));
    expect(screen.queryByRole("heading", { name: "Hidden result" })).not.toBeInTheDocument();
    rerender(<SearchWorkspace visible onOpenResult={vi.fn()} />);
    expect(await screen.findByRole("heading", { name: "Project decisions" })).toBeVisible();
  });

  it("explains partial indexing and preserves server warnings as plain text", async () => {
    vi.mocked(api.searchContent).mockResolvedValue(response({ items: [result()], partial: true, indexing: true, warnings: ["A file is being processed."] }));
    render(<SearchWorkspace onOpenResult={vi.fn()} />);
    search();
    expect(await screen.findByText(/still being indexed/u)).toBeVisible();
    expect(screen.getByText(/These results are partial/u)).toBeVisible();
    expect(screen.getByText("A file is being processed.")).toBeVisible();
    expect(screen.getByRole("button", { name: "Open chat: Project decisions" })).toBeEnabled();
  });

  it("labels retained file text and opens only its stored artifact target", async () => {
    const item = result({ id: "file:fil_video:transcript:0", source_id: "fil_video", kind: "file", title: "Trees.mp4", availability: "text_only", private_required: true, target: { kind: "file", file_id: "fil_video", artifact: "transcript" } });
    vi.mocked(api.searchContent).mockResolvedValue(response({ items: [item] }));
    const open = vi.fn();
    render(<SearchWorkspace onOpenResult={open} />);
    search();
    fireEvent.click(await screen.findByRole("button", { name: "Open file: Trees.mp4" }));
    expect(screen.getByText("Saved text · original unavailable")).toBeVisible();
    expect(screen.getByText("Transcript")).toBeVisible();
    expect(screen.getByText("Private", { exact: true })).toBeVisible();
    expect(open).toHaveBeenCalledWith(item);
  });

  it("labels generated summaries, visual notes, and saved agent reports", async () => {
    vi.mocked(api.searchContent).mockResolvedValue(response({ items: [
      result({ id: "summary", kind: "file", title: "Overview", target: { kind: "file", file_id: "fil_one", artifact: "summary" } }),
      result({ id: "agent", kind: "agent", title: "Crusades timeline", target: { kind: "agent", task_id: "agt_first", step_id: "research" } }),
      result({ id: "visual", kind: "file", title: "Scanned diagram", target: { kind: "file", file_id: "fil_visual", artifact: "visual" } }),
    ] }));
    render(<SearchWorkspace onOpenResult={vi.fn()} />);
    search();
    expect(await screen.findByText("Generated summary")).toBeVisible();
    expect(screen.getByText("Saved research")).toBeVisible();
    expect(screen.getByText("Generated report")).toBeVisible();
    expect(screen.getByText("Generated visual notes")).toBeVisible();
  });

  it("applies inclusive activity date filters and resets pagination for the saved query", async () => {
    vi.mocked(api.searchContent).mockResolvedValue(response({ items: [result()], next_cursor: "next-page" }));
    render(<SearchWorkspace onOpenResult={vi.fn()} />);
    search();
    fireEvent.click(await screen.findByRole("button", { name: "Load more results" }));
    await waitFor(() => expect(api.searchContent).toHaveBeenLastCalledWith("local storage", expect.objectContaining({ cursor: "next-page" }), expect.any(AbortSignal)));
    fireEvent.change(screen.getByLabelText("Updated from"), { target: { value: "2026-10-01" } });
    await waitFor(() => expect(api.searchContent).toHaveBeenLastCalledWith("local storage", expect.objectContaining({ since: "2026-10-01", cursor: undefined }), expect.any(AbortSignal)));
    fireEvent.change(screen.getByLabelText("Updated through"), { target: { value: "2026-10-29" } });
    await waitFor(() => expect(api.searchContent).toHaveBeenLastCalledWith("local storage", expect.objectContaining({ since: "2026-10-01", until: "2026-10-29", cursor: undefined }), expect.any(AbortSignal)));
    expect(screen.getByText("Activity dates use UTC. Both dates are included.")).toBeVisible();
    fireEvent.change(screen.getByLabelText("Updated from"), { target: { value: "" } });
    await waitFor(() => expect(api.searchContent).toHaveBeenLastCalledWith("local storage", expect.objectContaining({ since: undefined, until: "2026-10-29" }), expect.any(AbortSignal)));
  });

  it("blocks a reversed date range and uses the corrected dates when searching", async () => {
    render(<SearchWorkspace onOpenResult={vi.fn()} />);
    fireEvent.change(screen.getByLabelText("Updated from"), { target: { value: "2026-10-29" } });
    fireEvent.change(screen.getByLabelText("Updated through"), { target: { value: "2026-10-01" } });
    expect(screen.getByRole("alert")).toHaveTextContent("The start date must be on or before the end date.");
    search();
    expect(api.searchContent).not.toHaveBeenCalled();
    fireEvent.change(screen.getByLabelText("Updated through"), { target: { value: "2026-10-30" } });
    search();
    await waitFor(() => expect(api.searchContent).toHaveBeenCalledExactlyOnceWith("local storage", expect.objectContaining({ since: "2026-10-29", until: "2026-10-30" }), expect.any(AbortSignal)));
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("retries the failed request without losing the submitted query and filters", async () => {
    vi.mocked(api.searchContent).mockRejectedValueOnce(new api.ApiError(503, "Search is temporarily unavailable.")).mockResolvedValueOnce(response({ items: [result()] }));
    render(<SearchWorkspace onOpenResult={vi.fn()} />);
    fireEvent.click(screen.getByRole("button", { name: "Agents" }));
    search();
    expect(await screen.findByRole("alert")).toHaveTextContent("Search is temporarily unavailable.");
    fireEvent.click(screen.getByRole("button", { name: "Try again" }));
    expect(await screen.findByRole("heading", { name: "Project decisions" })).toBeVisible();
    expect(api.searchContent).toHaveBeenLastCalledWith("local storage", expect.objectContaining({ kind: "agent" }), expect.any(AbortSignal));
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("paginates with the saved query, deduplicates matches and retries a failed next page", async () => {
    vi.mocked(api.searchContent).mockResolvedValueOnce(response({ items: [result()], next_cursor: "next-page" }))
      .mockRejectedValueOnce(new Error("Connection lost"))
      .mockResolvedValueOnce(response({ items: [result(), result({ id: "second", title: "Another saved result" })] }));
    render(<SearchWorkspace onOpenResult={vi.fn()} />);
    search();
    fireEvent.click(await screen.findByRole("button", { name: "Load more results" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Connection lost");
    expect(screen.getByRole("heading", { name: "Project decisions" })).toBeVisible();
    fireEvent.click(screen.getByRole("button", { name: "Try again" }));
    expect(await screen.findByRole("heading", { name: "Another saved result" })).toBeVisible();
    expect(screen.getAllByRole("heading", { name: "Project decisions" })).toHaveLength(1);
    expect(api.searchContent).toHaveBeenLastCalledWith("local storage", expect.objectContaining({ cursor: "next-page" }), expect.any(AbortSignal));
    expect(screen.queryByRole("button", { name: "Load more results" })).not.toBeInTheDocument();
  });

  it("bounds visible results and explains how to narrow a large result set", async () => {
    vi.mocked(api.searchContent).mockResolvedValue(response({ items: Array.from({ length: 205 }, (_, index) => result({ id: `result_${index}` })), next_cursor: "more" }));
    render(<SearchWorkspace onOpenResult={vi.fn()} />);
    search();
    expect(await screen.findByText(/Showing the first 200 matches/u)).toBeVisible();
    expect(screen.getAllByRole("article")).toHaveLength(200);
    expect(screen.queryByRole("button", { name: "Load more results" })).not.toBeInTheDocument();
  });

  it("keeps search usable if Project filters cannot load", async () => {
    vi.mocked(api.listProjects).mockRejectedValue(new Error("Projects offline"));
    render(<SearchWorkspace onOpenResult={vi.fn()} />);
    expect(await screen.findByText(/Project filters are unavailable/u)).toBeVisible();
    search();
    expect(await screen.findByText("No matches found. Try fewer words or broader filters.")).toBeVisible();
  });

  it("aborts the pending request on unmount", async () => {
    vi.mocked(api.searchContent).mockReturnValue(new Promise(() => {}));
    const { unmount } = render(<SearchWorkspace onOpenResult={vi.fn()} />);
    search();
    await waitFor(() => expect(api.searchContent).toHaveBeenCalledTimes(1));
    const signal = vi.mocked(api.searchContent).mock.calls[0][2];
    unmount();
    expect(signal?.aborted).toBe(true);
  });
});
