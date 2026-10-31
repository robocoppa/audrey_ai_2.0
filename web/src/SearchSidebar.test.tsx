import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { SearchSidebar } from "./SearchSidebar";
import * as api from "./api";
import type { SearchResponse, SearchResult, SearchSource } from "./api";

const item: SearchResult = {
  id: "message:msg_first:0", source_id: "msg_first", kind: "message", revision: "3",
  title: "Project decisions", excerpt: "We chose the local storage option.",
  highlights: [{ start: 13, end: 18 }], offset: 100, updated_at: "2026-10-29T16:00:00Z",
  project_id: null, private_required: false, availability: "available",
  target: { kind: "message", conversation_id: "con_first", message_id: "msg_first", sequence: 20 },
};
const matches: SearchResponse = { items: [item], next_cursor: null, partial: false, indexing: false, warnings: [] };
const source: SearchSource = {
  source_id: item.source_id, kind: item.kind, revision: item.revision,
  title: item.title, text: "Saved source text", offset: 100, next_offset: 4100, total_chars: 8000,
  target: item.target, private_required: false, availability: "available",
};

function submit(query = "local storage") {
  fireEvent.change(screen.getByRole("searchbox", { name: "Search your content" }), { target: { value: query } });
  fireEvent.submit(screen.getByRole("search"));
}

beforeEach(() => {
  vi.spyOn(api, "listProjects").mockResolvedValue({ items: [], next_cursor: null, limits: { max_name_chars: 100, max_instructions_chars: 5000, max_files: 20 } });
  vi.spyOn(api, "searchContent").mockResolvedValue(matches);
  vi.spyOn(api, "getSearchSource").mockResolvedValue(source);
});
afterEach(() => { cleanup(); vi.restoreAllMocks(); });

describe("SearchSidebar", () => {
  it("preserves the query, filters and results when collapsed and revalidates on reopening", async () => {
    const onClose = vi.fn();
    const { rerender } = render(<SearchSidebar open modal={false} openRequest={{ query: "", serial: 1 }} onClose={onClose} />);
    expect(screen.getByRole("complementary", { name: "Search" })).toHaveAttribute("id", "app-search-sidebar");
    expect(screen.getAllByRole("heading", { name: "Search" })).toHaveLength(1);
    await waitFor(() => expect(screen.getByRole("searchbox")).toHaveFocus());
    fireEvent.click(screen.getByRole("button", { name: "Files" }));
    fireEvent.change(screen.getByLabelText("Updated from"), { target: { value: "2026-10-01" } });
    submit();
    expect(await screen.findByRole("heading", { name: item.title })).toBeVisible();
    fireEvent.click(screen.getByRole("button", { name: "Collapse Search" }));
    expect(onClose).toHaveBeenCalledOnce();
    rerender(<SearchSidebar open={false} modal={false} openRequest={{ query: "", serial: 1 }} onClose={onClose} />);
    expect(screen.queryByRole("searchbox")).not.toBeInTheDocument();
    await act(async () => {});
    expect(api.searchContent).toHaveBeenCalledTimes(1);
    rerender(<SearchSidebar open modal={false} openRequest={{ query: "", serial: 2 }} onClose={onClose} />);
    await waitFor(() => expect(api.searchContent).toHaveBeenCalledTimes(2));
    expect(screen.getByRole("searchbox")).toHaveValue("local storage");
    expect(screen.getByRole("button", { name: "Files" })).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByLabelText("Updated from")).toHaveValue("2026-10-01");
    expect(screen.getByRole("heading", { name: item.title })).toBeVisible();
    expect(api.searchContent).toHaveBeenLastCalledWith("local storage", expect.objectContaining({ kind: "file", since: "2026-10-01" }), expect.any(AbortSignal));
  });

  it("aborts hidden searches and ignores their late results until a current search completes", async () => {
    let resolve!: (value: SearchResponse) => void;
    vi.mocked(api.searchContent).mockReturnValueOnce(new Promise((done) => { resolve = done; }));
    const { rerender } = render(<SearchSidebar open modal={false} onClose={vi.fn()} />);
    submit();
    await waitFor(() => expect(api.searchContent).toHaveBeenCalledTimes(1));
    const signal = vi.mocked(api.searchContent).mock.calls[0][2];
    rerender(<SearchSidebar open={false} modal={false} onClose={vi.fn()} />);
    expect(signal?.aborted).toBe(true);
    await act(async () => { resolve(matches); });
    expect(screen.queryByText(item.title)).not.toBeInTheDocument();
    rerender(<SearchSidebar open modal={false} onClose={vi.fn()} />);
    expect(await screen.findByRole("heading", { name: item.title })).toBeVisible();
    expect(api.searchContent).toHaveBeenCalledTimes(2);
  });

  it("opens canonical source text inside the panel, pages it and returns to retained results", async () => {
    const close = vi.fn();
    render(<SearchSidebar open modal={false} onClose={close} />);
    submit();
    const trigger = await screen.findByRole("button", { name: `Open chat: ${item.title}` });
    trigger.focus();
    fireEvent.click(trigger);
    const panel = screen.getByRole("complementary", { name: "Search" });
    await waitFor(() => expect(within(panel).getByLabelText("Saved source text")).toHaveTextContent(source.text));
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    expect(screen.queryByRole("searchbox")).not.toBeInTheDocument();
    expect(api.getSearchSource).toHaveBeenCalledWith(item, item.offset, expect.any(AbortSignal));
    fireEvent.click(within(panel).getByRole("button", { name: "Next text" }));
    await waitFor(() => expect(api.getSearchSource).toHaveBeenLastCalledWith(item, 4100, expect.any(AbortSignal)));
    fireEvent.click(within(panel).getByRole("button", { name: "Back to search results" }));
    await waitFor(() => expect(trigger).toHaveFocus());
    expect(screen.getByRole("searchbox")).toHaveValue("local storage");
    expect(screen.getByRole("heading", { name: item.title })).toBeVisible();
    expect(close).not.toHaveBeenCalled();
  });

  it("uses Escape to return from a source before collapsing the sidebar", async () => {
    const close = vi.fn();
    render(<SearchSidebar open modal={false} onClose={close} />);
    submit();
    fireEvent.click(await screen.findByRole("button", { name: `Open chat: ${item.title}` }));
    const back = await screen.findByRole("button", { name: "Back to search results" });
    fireEvent.keyDown(back, { key: "Escape" });
    expect(screen.getByRole("searchbox")).toBeVisible();
    expect(close).not.toHaveBeenCalled();
    fireEvent.keyDown(screen.getByRole("searchbox"), { key: "Escape" });
    expect(close).toHaveBeenCalledOnce();
  });

  it("traps keyboard focus only in the overlay presentation", async () => {
    const { rerender } = render(<SearchSidebar open modal onClose={vi.fn()} />);
    expect(screen.getByRole("dialog", { name: "Search" })).toHaveAttribute("aria-modal", "true");
    await waitFor(() => expect(screen.getByRole("searchbox")).toHaveFocus());
    const first = screen.getByRole("button", { name: "Collapse Search" });
    const last = screen.getByLabelText("Updated through");
    first.focus();
    fireEvent.keyDown(first, { key: "Tab", shiftKey: true });
    expect(last).toHaveFocus();
    fireEvent.keyDown(last, { key: "Tab" });
    expect(first).toHaveFocus();
    rerender(<SearchSidebar open modal={false} onClose={vi.fn()} />);
    expect(screen.getByRole("complementary", { name: "Search" })).not.toHaveAttribute("aria-modal");
    last.focus();
    expect(fireEvent.keyDown(last, { key: "Tab" })).toBe(true);
    expect(last).toHaveFocus();
  });

  it("aborts a source read when collapsed and returns to the search on a new open request", async () => {
    vi.mocked(api.getSearchSource).mockReturnValue(new Promise(() => {}));
    const { rerender } = render(<SearchSidebar open modal={false} openRequest={{ query: "", serial: 1 }} onClose={vi.fn()} />);
    await act(async () => {});
    submit();
    fireEvent.click(await screen.findByRole("button", { name: `Open chat: ${item.title}` }));
    await waitFor(() => expect(api.getSearchSource).toHaveBeenCalledTimes(1));
    const signal = vi.mocked(api.getSearchSource).mock.calls[0][2];
    rerender(<SearchSidebar open={false} modal={false} openRequest={{ query: "", serial: 1 }} onClose={vi.fn()} />);
    expect(signal?.aborted).toBe(true);
    await act(async () => {});
    rerender(<SearchSidebar open modal={false} openRequest={{ query: "", serial: 2 }} onClose={vi.fn()} />);
    await waitFor(() => expect(screen.getByRole("searchbox")).toHaveFocus());
    expect(api.getSearchSource).toHaveBeenCalledTimes(1);
    expect(screen.queryByRole("button", { name: "Back to search results" })).not.toBeInTheDocument();
  });

  it("moves focus into an already-open panel when resizing into the overlay", async () => {
    const { rerender } = render(<><button>Chat composer</button><SearchSidebar open modal={false} onClose={vi.fn()} /></>);
    await act(async () => {});
    screen.getByRole("button", { name: "Chat composer" }).focus();
    rerender(<><button>Chat composer</button><SearchSidebar open modal onClose={vi.fn()} /></>);
    await waitFor(() => expect(screen.getByRole("searchbox")).toHaveFocus());
    expect(screen.getByRole("dialog", { name: "Search" })).toHaveAttribute("aria-modal", "true");
  });

  it("focuses the retained source when resizing into the overlay from the underlying app", async () => {
    const { rerender } = render(<><button>Chat composer</button><SearchSidebar open modal={false} onClose={vi.fn()} /></>);
    await act(async () => {});
    submit();
    fireEvent.click(await screen.findByRole("button", { name: `Open chat: ${item.title}` }));
    await screen.findByLabelText("Saved source text");
    screen.getByRole("button", { name: "Chat composer" }).focus();
    rerender(<><button>Chat composer</button><SearchSidebar open modal onClose={vi.fn()} /></>);
    const back = screen.getByRole("button", { name: "Back to search results" });
    await waitFor(() => expect(back).toHaveFocus());
    expect(screen.getByRole("dialog", { name: "Search" })).toHaveAttribute("aria-modal", "true");
    expect(screen.getByLabelText("Saved source text")).toHaveTextContent(source.text);
    expect(screen.queryByRole("searchbox")).not.toBeInTheDocument();
    expect(api.getSearchSource).toHaveBeenCalledTimes(1);
    const collapse = screen.getByRole("button", { name: "Collapse Search" });
    collapse.focus();
    fireEvent.keyDown(collapse, { key: "Tab", shiftKey: true });
    expect(screen.getByRole("button", { name: "Next text" })).toHaveFocus();
  });
});
