import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { SearchSourceViewer } from "./SearchSourceViewer";
import * as api from "./api";
import type { SearchResult, SearchSource } from "./api";

function result(patch: Partial<SearchResult> = {}): SearchResult {
  return {
    id: "message:msg_one:100", source_id: "msg_one", revision: "3", kind: "message", title: "Design choices",
    excerpt: "🗂 local option", highlights: [{ start: 2, end: 7 }], offset: 100,
    updated_at: "2026-10-29T16:00:00Z", project_id: null, private_required: false, availability: "available",
    target: { kind: "message", conversation_id: "con_one", message_id: "msg_one", sequence: 20 }, ...patch,
  };
}

function source(patch: Partial<SearchSource> = {}): SearchSource {
  const item = result();
  return {
    source_id: item.source_id, revision: item.revision, kind: item.kind, title: item.title,
    text: "🗂 local option. More saved source content.", offset: 100, next_offset: null, total_chars: 140,
    target: item.target, availability: "available", private_required: false, ...patch,
  };
}

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason: unknown) => void;
  const promise = new Promise<T>((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}

beforeEach(() => { vi.spyOn(api, "getSearchSource").mockResolvedValue(source()); });
afterEach(() => { cleanup(); vi.restoreAllMocks(); });

describe("SearchSourceViewer", () => {
  it.each([
    { kind: "message", target: { kind: "message", conversation_id: "con_one", message_id: "msg_one", sequence: 20 }, label: "Open conversation" },
    { kind: "file", target: { kind: "file", file_id: "fil_one", artifact: "summary" }, label: "Open file" },
    { kind: "agent", target: { kind: "agent", task_id: "agt_one", step_id: "research" }, label: "Open agent" },
    { kind: "knowledge", target: { kind: "knowledge", project_id: "prj_one", entry_id: "knw_one" }, label: "Open Project" },
  ] satisfies Array<{ kind: SearchResult["kind"]; target: SearchResult["target"]; label: string }>) ("opens the canonical $kind workspace only after its source loads", async ({ kind, target, label }) => {
    const saved = deferred<SearchSource>();
    const item = result({ kind, target });
    const fresh = source({ kind, target, title: "Canonical title", private_required: true, availability: "text_only" });
    const open = vi.fn();
    vi.mocked(api.getSearchSource).mockReturnValue(saved.promise);
    render(<SearchSourceViewer presentation="inline" result={item} onClose={vi.fn()} onOpenWorkspace={open} />);
    expect(screen.queryByRole("button", { name: label })).not.toBeInTheDocument();
    await act(async () => saved.resolve(fresh));
    fireEvent.click(await screen.findByRole("button", { name: label }));
    expect(open).toHaveBeenCalledExactlyOnceWith({ ...item, target: fresh.target, title: fresh.title, private_required: true, availability: "text_only" }, expect.any(AbortSignal));
  });

  it("preserves the reader without workspace controls when no callback is provided", async () => {
    render(<SearchSourceViewer result={result()} onClose={vi.fn()} />);
    await screen.findByLabelText("Saved source text");
    expect(screen.queryByRole("button", { name: /^Open /u })).not.toBeInTheDocument();
  });

  it("clears navigation while a replacement source loader revalidates the same result", async () => {
    const pendingNavigation = deferred<void>();
    const pendingSource = deferred<SearchSource>();
    const item = result();
    const open = vi.fn().mockReturnValue(pendingNavigation.promise);
    const firstLoad = vi.fn().mockResolvedValue(source());
    const nextLoad = vi.fn().mockReturnValue(pendingSource.promise);
    const { rerender } = render(<SearchSourceViewer result={item} loadSource={firstLoad} onClose={vi.fn()} onOpenWorkspace={open} />);
    fireEvent.click(await screen.findByRole("button", { name: "Open conversation" }));
    const signal = open.mock.calls[0][1] as AbortSignal;
    rerender(<SearchSourceViewer result={item} loadSource={nextLoad} onClose={vi.fn()} onOpenWorkspace={open} />);
    expect(signal.aborted).toBe(true);
    await waitFor(() => expect(screen.queryByRole("button", { name: /Open conversation|Opening…/u })).not.toBeInTheDocument());
    await act(async () => pendingSource.resolve(source()));
    expect(await screen.findByRole("button", { name: "Open conversation" })).toBeEnabled();
    await act(async () => pendingNavigation.resolve());
    expect(screen.getByRole("button", { name: "Open conversation" })).toBeEnabled();
  });

  it("shows transient workspace errors without discarding the source and permits an explicit retry", async () => {
    const open = vi.fn().mockRejectedValueOnce(new TypeError("The connection was lost.")).mockResolvedValueOnce(undefined);
    render(<SearchSourceViewer result={result()} onClose={vi.fn()} onOpenWorkspace={open} />);
    fireEvent.click(await screen.findByRole("button", { name: "Open conversation" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("The connection was lost.");
    expect(screen.getByLabelText("Saved source text")).toBeVisible();
    fireEvent.click(screen.getByRole("button", { name: "Open conversation" }));
    await waitFor(() => expect(open).toHaveBeenCalledTimes(2));
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it.each([403, 404, 409])("clears saved source text and workspace controls when navigation discovers a %i boundary", async (status) => {
    const open = vi.fn().mockRejectedValue(new api.ApiError(status, "Stale canonical target"));
    render(<SearchSourceViewer presentation="inline" result={result()} onClose={vi.fn()} onOpenWorkspace={open} />);
    expect(await screen.findByLabelText("Saved source text")).toBeVisible();
    fireEvent.click(screen.getByRole("button", { name: "Open conversation" }));
    expect(await screen.findByRole("alert")).toHaveTextContent(status === 409 ? "This source changed after the search" : "This source was deleted or is no longer available");
    expect(screen.getByRole("alert")).toHaveTextContent("search again");
    expect(screen.queryByLabelText("Saved source text")).not.toBeInTheDocument();
    expect(screen.queryByText("🗂 local option. More saved source content.")).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /Open conversation|Opening…|Try again|Next text|Previous text/u })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Back to search results" })).toHaveFocus();
    expect(screen.getAllByRole("alert")).toHaveLength(1);
  });

  it("keeps Project source navigation conflicts terminal with the scoped refresh instruction", async () => {
    const load = vi.fn().mockResolvedValue(source());
    const open = vi.fn().mockRejectedValue(new api.ApiError(409, "Detached"));
    render(<SearchSourceViewer result={result()} loadSource={load} onClose={vi.fn()} onOpenWorkspace={open} />);
    fireEvent.click(await screen.findByRole("button", { name: "Open conversation" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("This source changed or was detached. Close this view and refresh Project knowledge.");
    expect(screen.queryByLabelText("Saved source text")).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /Open conversation|Try again/u })).not.toBeInTheDocument();
  });

  it("ignores a late navigation access failure after paging to a current source", async () => {
    const pending = deferred<void>();
    const open = vi.fn().mockReturnValue(pending.promise);
    vi.mocked(api.getSearchSource).mockResolvedValueOnce(source({ next_offset: 4100, total_chars: 5000 }))
      .mockResolvedValueOnce(source({ offset: 4100, text: "Current source page", total_chars: 5000 }));
    render(<SearchSourceViewer presentation="inline" result={result()} onClose={vi.fn()} onOpenWorkspace={open} />);
    fireEvent.click(await screen.findByRole("button", { name: "Open conversation" }));
    const signal = open.mock.calls[0][1] as AbortSignal;
    fireEvent.click(screen.getByRole("button", { name: "Next text" }));
    expect(signal.aborted).toBe(true);
    expect(await screen.findByLabelText("Saved source text")).toHaveTextContent("Current source page");
    await act(async () => pending.reject(new api.ApiError(404, "Late deleted response")));
    expect(screen.getByLabelText("Saved source text")).toHaveTextContent("Current source page");
    expect(screen.getByRole("button", { name: "Open conversation" })).toBeEnabled();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it.each(["unmount", "back", "page", "result"] as const)("aborts pending workspace navigation on %s and ignores its late completion", async (change) => {
    const pending = deferred<void>();
    const open = vi.fn().mockReturnValue(pending.promise);
    vi.mocked(api.getSearchSource).mockResolvedValue(source({ next_offset: 4100, total_chars: 5000 }));
    const close = vi.fn();
    const { rerender, unmount } = render(<SearchSourceViewer presentation="inline" result={result()} onClose={close} onOpenWorkspace={open} />);
    fireEvent.click(await screen.findByRole("button", { name: "Open conversation" }));
    expect(screen.getByRole("button", { name: "Opening…" })).toBeDisabled();
    fireEvent.click(screen.getByRole("button", { name: "Opening…" }));
    expect(open).toHaveBeenCalledOnce();
    const signal = open.mock.calls[0][1] as AbortSignal;
    if (change === "unmount") unmount();
    else if (change === "back") fireEvent.click(screen.getByRole("button", { name: "Back to search results" }));
    else if (change === "page") fireEvent.click(screen.getByRole("button", { name: "Next text" }));
    else rerender(<SearchSourceViewer presentation="inline" result={result({ revision: "4" })} onClose={close} onOpenWorkspace={open} />);
    expect(signal.aborted).toBe(true);
    await act(async () => pending.resolve());
    expect(open).toHaveBeenCalledOnce();
    if (change === "back") expect(close).toHaveBeenCalledOnce();
    if (change === "page") expect(await screen.findByRole("button", { name: "Open conversation" })).toBeEnabled();
  });

  it.each([
    { expected: { kind: "message", conversation_id: "con_one", message_id: "msg_one", sequence: 20 }, actual: { kind: "message", conversation_id: "con_other", message_id: "msg_one", sequence: 20 } },
    { expected: { kind: "message", conversation_id: "con_one", message_id: "msg_one", sequence: 20 }, actual: { kind: "message", conversation_id: "con_one", message_id: "msg_other", sequence: 20 } },
    { expected: { kind: "message", conversation_id: "con_one", message_id: "msg_one", sequence: 20 }, actual: { kind: "message", conversation_id: "con_one", message_id: "msg_one", sequence: 21 } },
    { expected: { kind: "file", file_id: "fil_one", artifact: "text" }, actual: { kind: "file", file_id: "fil_one", artifact: "summary" } },
    { expected: { kind: "file", file_id: "fil_one", artifact: "text" }, actual: { kind: "file", file_id: "fil_other", artifact: "text" } },
    { expected: { kind: "agent", task_id: "agt_one", step_id: "research" }, actual: { kind: "agent", task_id: "agt_one", step_id: "review" } },
    { expected: { kind: "agent", task_id: "agt_one" }, actual: { kind: "agent", task_id: "agt_other" } },
    { expected: { kind: "knowledge", project_id: "prj_one", entry_id: "knw_one" }, actual: { kind: "knowledge", project_id: "prj_other", entry_id: "knw_one" } },
    { expected: { kind: "knowledge", project_id: "prj_one", entry_id: "knw_one" }, actual: { kind: "knowledge", project_id: "prj_one", entry_id: "knw_other" } },
  ] satisfies Array<{ expected: SearchResult["target"]; actual: SearchResult["target"] }>) ("rejects changed target identity $actual", async ({ expected, actual }) => {
    const open = vi.fn();
    vi.mocked(api.getSearchSource).mockResolvedValue(source({ kind: actual.kind, target: actual, text: "Wrong target" }));
    render(<SearchSourceViewer result={result({ kind: expected.kind, target: expected })} onClose={vi.fn()} onOpenWorkspace={open} />);
    expect(await screen.findByRole("alert")).toHaveTextContent("This source changed after the search");
    expect(screen.queryByText("Wrong target")).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /^Open /u })).not.toBeInTheDocument();
    expect(open).not.toHaveBeenCalled();
  });

  it("embeds canonical text without a portal, modal controls, body lock, or global keyboard handling", async () => {
    const close = vi.fn();
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "auto";
    const { container, unmount } = render(<SearchSourceViewer presentation="inline" result={result()} onClose={close} />);
    const back = screen.getByRole("button", { name: "Back to search results" });
    expect(back).toHaveFocus();
    expect(await screen.findByLabelText("Saved source text")).toHaveTextContent("🗂 local option");
    expect(container.querySelector(".search-source-inline")).toBe(screen.getByRole("region", { name: "Design choices" }));
    expect(container.querySelector("mark")).toHaveTextContent("local");
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    expect(document.querySelector(".search-source-backdrop, [aria-modal]")).toBeNull();
    expect(document.body.style.overflow).toBe("auto");
    const escape = new KeyboardEvent("keydown", { key: "Escape", bubbles: true, cancelable: true });
    document.dispatchEvent(escape);
    expect(escape.defaultPrevented).toBe(false);
    expect(close).not.toHaveBeenCalled();
    const tab = new KeyboardEvent("keydown", { key: "Tab", shiftKey: true, bubbles: true, cancelable: true });
    document.dispatchEvent(tab);
    expect(tab.defaultPrevented).toBe(false);
    fireEvent.click(back);
    expect(close).toHaveBeenCalledOnce();
    unmount();
    expect(document.body.style.overflow).toBe("auto");
    document.body.style.overflow = previousOverflow;
  });

  it("keeps inline paging inside the source region and returns focus to its back control", async () => {
    const item = result();
    const nextPage = deferred<SearchSource>();
    vi.mocked(api.getSearchSource).mockResolvedValueOnce(source({ next_offset: 4100, total_chars: 5000 }))
      .mockReturnValueOnce(nextPage.promise);
    render(<SearchSourceViewer presentation="inline" result={item} onClose={vi.fn()} />);
    fireEvent.click(await screen.findByRole("button", { name: "Next text" }));
    expect(screen.getByRole("button", { name: "Back to search results" })).toHaveFocus();
    expect(screen.queryByLabelText("Saved source text")).not.toBeInTheDocument();
    expect(api.getSearchSource).toHaveBeenLastCalledWith(item, 4100, expect.any(AbortSignal));
    await act(async () => nextPage.resolve(source({ offset: 4100, text: "Later canonical text", total_chars: 5000 })));
    expect(await screen.findByLabelText("Saved source text")).toHaveTextContent("Later canonical text");
    expect(screen.getByRole("button", { name: "Back to search results" })).toHaveFocus();
    expect(document.querySelector(".search-source-backdrop")).toBeNull();
  });

  it.each([403, 404, 409])("clears inline source text when paging crosses a %i access or revision boundary", async (status) => {
    vi.mocked(api.getSearchSource).mockResolvedValueOnce(source({ next_offset: 4100, total_chars: 5000 }))
      .mockRejectedValueOnce(new api.ApiError(status, "Unavailable"));
    render(<SearchSourceViewer presentation="inline" result={result()} onClose={vi.fn()} />);
    fireEvent.click(await screen.findByRole("button", { name: "Next text" }));
    expect(await screen.findByRole("alert")).toHaveTextContent(status === 409 ? "This source changed" : "deleted or is no longer available");
    expect(screen.queryByLabelText("Saved source text")).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Try again" })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Back to search results" })).toBeVisible();
  });

  it("aborts an inline canonical read on collapse without moving focus owned by its parent", async () => {
    const pending = deferred<SearchSource>();
    vi.mocked(api.getSearchSource).mockReturnValue(pending.promise);
    const outside = document.createElement("button");
    document.body.append(outside);
    const { unmount } = render(<SearchSourceViewer presentation="inline" result={result()} onClose={vi.fn()} />);
    const signal = vi.mocked(api.getSearchSource).mock.calls[0][2];
    outside.focus();
    unmount();
    expect(signal?.aborted).toBe(true);
    expect(outside).toHaveFocus();
    await act(async () => pending.resolve(source()));
    expect(screen.queryByLabelText("Saved source text")).not.toBeInTheDocument();
    outside.remove();
  });

  it("keeps Project provenance paging within its scoped loader and fences detached sources", async () => {
    const load = vi.fn().mockResolvedValueOnce(source({ next_offset: 4100, total_chars: 5000 }))
      .mockRejectedValueOnce(new api.ApiError(409, "Detached"));
    render(<SearchSourceViewer result={result()} loadSource={load} onClose={vi.fn()} />);
    expect(await screen.findByLabelText("Saved source text")).toBeVisible();
    expect(api.getSearchSource).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: "Next text" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("refresh Project knowledge");
    expect(screen.queryByLabelText("Saved source text")).not.toBeInTheDocument();
    expect(load).toHaveBeenLastCalledWith(4100, expect.any(AbortSignal));
  });
  it("reads pinned canonical text at the returned offset and highlights Unicode matches", async () => {
    const item = result();
    render(<SearchSourceViewer result={item} onClose={vi.fn()} />);
    expect(await screen.findByLabelText("Saved source text")).toHaveTextContent("🗂 local option. More saved source content.");
    expect(api.getSearchSource).toHaveBeenCalledExactlyOnceWith(item, 100, expect.any(AbortSignal));
    expect(document.querySelector("mark")).toHaveTextContent("local");
    expect(screen.getByRole("dialog")).toHaveAccessibleName("Design choices");
    expect(screen.getByRole("button", { name: "Next text" })).toBeDisabled();
  });

  it("does not substitute the cached search excerpt when canonical loading fails", async () => {
    vi.mocked(api.getSearchSource).mockRejectedValue(new api.ApiError(404, "Not found"));
    render(<SearchSourceViewer result={result()} onClose={vi.fn()} />);
    expect(await screen.findByRole("alert")).toHaveTextContent("This source was deleted or is no longer available to you.");
    expect(screen.getByRole("alert")).toHaveTextContent("search again");
    expect(screen.queryByLabelText("Saved source text")).not.toBeInTheDocument();
    expect(screen.queryByText("🗂 local option")).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Try again" })).not.toBeInTheDocument();
  });

  it("asks for a new search when the source revision changed", async () => {
    vi.mocked(api.getSearchSource).mockRejectedValue(new api.ApiError(409, "Changed"));
    render(<SearchSourceViewer result={result()} onClose={vi.fn()} />);
    expect(await screen.findByRole("alert")).toHaveTextContent("This source changed after the search.");
    expect(screen.queryByLabelText("Saved source text")).not.toBeInTheDocument();
  });

  it("refuses a response that does not match the pinned source revision", async () => {
    vi.mocked(api.getSearchSource).mockResolvedValue(source({ revision: "4", text: "Unexpected newer content" }));
    render(<SearchSourceViewer result={result()} onClose={vi.fn()} />);
    expect(await screen.findByRole("alert")).toHaveTextContent("This source changed after the search.");
    expect(screen.queryByText("Unexpected newer content")).not.toBeInTheDocument();
  });

  it("retries a network failure using the same source and offset", async () => {
    vi.mocked(api.getSearchSource).mockRejectedValueOnce(new TypeError("Connection lost")).mockResolvedValueOnce(source());
    const item = result();
    render(<SearchSourceViewer result={item} onClose={vi.fn()} />);
    expect(await screen.findByRole("alert")).toHaveTextContent("Connection lost");
    fireEvent.click(screen.getByRole("button", { name: "Try again" }));
    expect(await screen.findByLabelText("Saved source text")).toBeVisible();
    expect(api.getSearchSource).toHaveBeenLastCalledWith(item, 100, expect.any(AbortSignal));
  });

  it("follows canonical page offsets and loads previous text without inventing page or media locators", async () => {
    vi.mocked(api.getSearchSource).mockResolvedValueOnce(source({ next_offset: 4100, total_chars: 9000 }))
      .mockResolvedValueOnce(source({ offset: 4100, next_offset: 8100, total_chars: 9000, text: "A later section" }))
      .mockResolvedValueOnce(source({ offset: 100, next_offset: 4100, total_chars: 9000 }));
    const item = result();
    render(<SearchSourceViewer result={item} onClose={vi.fn()} />);
    fireEvent.click(await screen.findByRole("button", { name: "Next text" }));
    expect(await screen.findByText("A later section")).toBeVisible();
    expect(api.getSearchSource).toHaveBeenLastCalledWith(item, 4100, expect.any(AbortSignal));
    expect(document.querySelector("mark")).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "Previous text" }));
    await waitFor(() => expect(api.getSearchSource).toHaveBeenLastCalledWith(item, 100, expect.any(AbortSignal)));
    expect(await screen.findByLabelText("Saved source text")).toHaveTextContent("local option");
    expect(screen.queryByRole("button", { name: /seek|page [0-9]/iu })).not.toBeInTheDocument();
  });

  it("clips a highlight to a source page boundary", async () => {
    const item = result({ offset: 10, excerpt: "abcdefghijkl", highlights: [{ start: 0, end: 10 }] });
    vi.mocked(api.getSearchSource).mockResolvedValue(source({ offset: 15, text: "fghijklmnop", total_chars: 26 }));
    render(<SearchSourceViewer result={item} onClose={vi.fn()} />);
    await screen.findByLabelText("Saved source text");
    expect(document.querySelector("mark")).toHaveTextContent("fghij");
  });

  it("renders saved source markup as plain text and labels retained Private summaries", async () => {
    const item = result({ kind: "file", target: { kind: "file", file_id: "fil_one", artifact: "summary" } });
    vi.mocked(api.getSearchSource).mockResolvedValue(source({ kind: "file", target: item.target, private_required: true, availability: "text_only", text: "<img src=x onerror=alert(1)> **summary**" }));
    render(<SearchSourceViewer result={item} onClose={vi.fn()} />);
    expect(await screen.findByLabelText("Saved source text")).toHaveTextContent("<img src=x onerror=alert(1)> **summary**");
    expect(document.querySelector("img, script, a")).toBeNull();
    expect(screen.getByText("Private", { exact: true })).toBeVisible();
    expect(screen.getByText("Generated summary")).toBeVisible();
    expect(screen.getByText("Saved text retained · original unavailable")).toBeVisible();
  });

  it.each(["agent", "visual"] as const)("identifies generated %s source content", async (kind) => {
    const target: SearchResult["target"] = kind === "agent" ? { kind: "agent", task_id: "agt_one", step_id: "research" } : { kind: "file", file_id: "fil_one", artifact: "visual" };
    const sourceKind = kind === "agent" ? "agent" : "file";
    vi.mocked(api.getSearchSource).mockResolvedValue(source({ kind: sourceKind, target }));
    render(<SearchSourceViewer result={result({ kind: sourceKind, target })} onClose={vi.fn()} />);
    expect(await screen.findByText(kind === "agent" ? "Generated report" : "Generated visual notes")).toBeVisible();
  });

  it("focuses the dialog, traps keyboard focus, dismisses with Escape, and returns focus on close", async () => {
    const opener = document.createElement("button");
    opener.textContent = "Source opener";
    document.body.append(opener);
    opener.focus();
    const close = vi.fn();
    const { unmount } = render(<SearchSourceViewer result={result()} onClose={close} />);
    const closeButton = screen.getByRole("button", { name: "Close saved source" });
    expect(closeButton).toHaveFocus();
    await screen.findByLabelText("Saved source text");
    fireEvent.keyDown(document, { key: "Tab", shiftKey: true });
    expect(screen.getByRole("button", { name: "Previous text" })).toHaveFocus();
    fireEvent.keyDown(document, { key: "Tab" });
    expect(closeButton).toHaveFocus();
    fireEvent.keyDown(document, { key: "Escape" });
    expect(close).toHaveBeenCalledOnce();
    unmount();
    expect(opener).toHaveFocus();
    opener.remove();
  });

  it("dismisses from the close button or backdrop without closing on content clicks", async () => {
    const close = vi.fn();
    render(<SearchSourceViewer result={result()} onClose={close} />);
    fireEvent.mouseDown(screen.getByRole("dialog"));
    expect(close).not.toHaveBeenCalled();
    fireEvent.mouseDown(document.querySelector(".search-source-backdrop")!);
    fireEvent.click(screen.getByRole("button", { name: "Close saved source" }));
    expect(close).toHaveBeenCalledTimes(2);
  });

  it("cancels pending canonical reads and ignores late results after close", async () => {
    const pending = deferred<SearchSource>();
    vi.mocked(api.getSearchSource).mockReturnValue(pending.promise);
    const { unmount } = render(<SearchSourceViewer result={result()} onClose={vi.fn()} />);
    await waitFor(() => expect(api.getSearchSource).toHaveBeenCalled());
    const signal = vi.mocked(api.getSearchSource).mock.calls[0][2];
    unmount();
    expect(signal?.aborted).toBe(true);
    await act(async () => pending.resolve(source()));
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });
});
