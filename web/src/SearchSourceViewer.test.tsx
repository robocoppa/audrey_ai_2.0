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
  const promise = new Promise<T>((yes) => { resolve = yes; });
  return { promise, resolve };
}

beforeEach(() => { vi.spyOn(api, "getSearchSource").mockResolvedValue(source()); });
afterEach(() => { cleanup(); vi.restoreAllMocks(); });

describe("SearchSourceViewer", () => {
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
