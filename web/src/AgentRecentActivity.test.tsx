import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { AgentRecentActivity } from "./AgentRecentActivity";
import * as api from "./api";

const task = { id: "agent-first", status: "running" as api.AgentTaskStatus, revision: 1 };
function event(patch: Partial<api.AgentEvent> = {}): api.AgentEvent {
  return { seq: 1, attempt: 1, stage: "research", kind: "model", state: "started", tool: null,
    created_at: "2026-11-03T14:00:00Z", finished_at: null, duration_ms: null, input_tokens: null, output_tokens: null, ...patch };
}
const page = (items: api.AgentEvent[], has_more = false, truncated = false): api.AgentEventsPage => ({ items, has_more, truncated });
afterEach(() => { cleanup(); vi.restoreAllMocks(); vi.useRealTimers(); });

describe("AgentRecentActivity", () => {
  it("labels a selected-file read without rendering source text", async () => {
    vi.spyOn(api, "listAgentEvents").mockResolvedValue(page([
      event({ kind: "tool", tool: "read_selected_file", state: "completed", finished_at: "2026-11-03T14:00:01Z" }),
    ]));
    render(<AgentRecentActivity task={task} visible />);
    expect(await screen.findByText("Read selected file")).toBeVisible();
    expect(screen.getByText("Completed")).toBeVisible();
  });
  it("shows honest local operation states and compact token/timing metadata", async () => {
    vi.spyOn(api, "listAgentEvents").mockResolvedValue(page([
      event({ seq: 4, kind: "tool", tool: "web_search", state: "interrupted", finished_at: "2026-11-03T14:00:02Z", duration_ms: 2000 }),
      event({ seq: 3, kind: "tool", tool: "web_fetch", state: "completed", finished_at: "2026-11-03T14:00:01Z", duration_ms: 950 }),
      event({ seq: 2, kind: "tool", tool: "create_file", state: "failed", finished_at: "2026-11-03T14:00:01Z" }),
      event({ input_tokens: 48, output_tokens: 7 }),
    ]));
    render(<AgentRecentActivity task={task} visible timezone="UTC" />);
    const operations = await screen.findByRole("list", { name: "Agent operations" });
    expect(within(operations).getByText("Local model")).toBeVisible();
    expect(within(operations).getByText("Started")).toBeVisible();
    expect(within(operations).getByText("Search web")).toBeVisible();
    expect(within(operations).getByText("Read web page")).toBeVisible();
    expect(within(operations).getByText("Create file")).toBeVisible();
    expect(within(operations).getByText("The outcome was not recorded.")).toBeVisible();
    expect(within(operations).getByText("48 input tokens · 7 output tokens")).toBeVisible();
    expect(within(operations).getByText("2.0 s")).toBeVisible();
    expect(screen.queryByText(/research/i)).not.toBeInTheDocument();
  });

  it("polls selected active tasks only and never regresses a terminal row to Started", async () => {
    vi.useFakeTimers();
    const read = vi.spyOn(api, "listAgentEvents").mockResolvedValueOnce(page([event()]))
      .mockResolvedValueOnce(page([event({ state: "completed", finished_at: "2026-11-03T14:00:01Z" })]))
      .mockResolvedValue(page([event()]));
    render(<AgentRecentActivity task={task} visible />);
    await act(async () => { await vi.advanceTimersByTimeAsync(0); });
    expect(screen.getByText("Started")).toBeVisible();
    await act(async () => { await vi.advanceTimersByTimeAsync(3000); });
    expect(screen.getByText("Completed")).toBeVisible();
    await act(async () => { await vi.advanceTimersByTimeAsync(3000); });
    expect(screen.getByText("Completed")).toBeVisible();
    expect(read).toHaveBeenCalledTimes(3);
  });

  it("refreshes terminal revisions, stops polling and ignores late replies after a task change", async () => {
    let release!: (value: api.AgentEventsPage) => void;
    const read = vi.spyOn(api, "listAgentEvents").mockImplementationOnce(() => new Promise((resolve) => { release = resolve; }))
      .mockResolvedValue(page([event({ stage: "review", state: "completed", finished_at: "2026-11-03T14:00:01Z" })]));
    const { rerender } = render(<AgentRecentActivity task={task} visible />);
    await waitFor(() => expect(read).toHaveBeenCalledTimes(1));
    const signal = read.mock.calls[0][1];
    rerender(<AgentRecentActivity task={{ ...task, id: "agent-second", status: "completed", revision: 4 }} visible />);
    expect(await screen.findByText("Completed")).toBeVisible();
    expect(signal?.aborted).toBe(true);
    await act(async () => release(page([event({ kind: "tool", tool: "web_search" })])));
    expect(screen.queryByText("Search web")).not.toBeInTheDocument();
    vi.useFakeTimers();
    await act(async () => { await vi.advanceTimersByTimeAsync(6000); });
    expect(read).toHaveBeenCalledTimes(2);
    rerender(<AgentRecentActivity task={{ ...task, id: "agent-second", status: "completed", revision: 5 }} visible />);
    await act(async () => { await vi.advanceTimersByTimeAsync(0); });
    expect(read).toHaveBeenCalledTimes(3);
  });

  it("aborts on collapse, hidden workspace and unmount, revalidating each reopening", async () => {
    let release!: (value: api.AgentEventsPage) => void;
    const read = vi.spyOn(api, "listAgentEvents").mockImplementationOnce(() => new Promise((resolve) => { release = resolve; }))
      .mockResolvedValue(page([event()]));
    const { rerender, unmount } = render(<AgentRecentActivity task={task} visible />);
    await waitFor(() => expect(read).toHaveBeenCalledTimes(1));
    fireEvent.click(screen.getByRole("button", { name: "Recent activity" }));
    expect(read.mock.calls[0][1]?.aborted).toBe(true);
    await act(async () => release(page([event({ kind: "tool", tool: "create_file" })])));
    expect(screen.queryByText("Create file")).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Recent activity" }));
    expect(await screen.findByText("Local model")).toBeVisible();
    expect(read).toHaveBeenCalledTimes(2);
    rerender(<AgentRecentActivity task={task} visible={false} />);
    expect(screen.queryByText("Local model")).not.toBeInTheDocument();
    rerender(<AgentRecentActivity task={task} visible />);
    expect(await screen.findByText("Local model")).toBeVisible();
    expect(read).toHaveBeenCalledTimes(3);
    const controllerSignal = read.mock.calls[2][1];
    unmount();
    // Completed requests need no cancellation; there is no timer after unmount.
    expect(controllerSignal).toBeInstanceOf(AbortSignal);
  });

  it("pauses requests while the browser page is hidden and revalidates on return", async () => {
    const visibility = vi.spyOn(document, "visibilityState", "get").mockReturnValue("visible");
    const read = vi.spyOn(api, "listAgentEvents").mockResolvedValue(page([event()]));
    render(<AgentRecentActivity task={task} visible />);
    await screen.findByText("Local model");
    visibility.mockReturnValue("hidden"); fireEvent(document, new Event("visibilitychange"));
    expect(screen.queryByText("Local model")).not.toBeInTheDocument();
    vi.useFakeTimers(); await act(async () => { await vi.advanceTimersByTimeAsync(9000); });
    expect(read).toHaveBeenCalledTimes(1);
    visibility.mockReturnValue("visible"); fireEvent(document, new Event("visibilitychange"));
    await act(async () => { await vi.advanceTimersByTimeAsync(0); });
    expect(read).toHaveBeenCalledTimes(2);
    expect(screen.getByText("Local model")).toBeVisible();
  });

  it("aborts an unfinished request on unmount and ignores its late completion", async () => {
    let release!: (value: api.AgentEventsPage) => void;
    const read = vi.spyOn(api, "listAgentEvents").mockImplementation(() => new Promise((resolve) => { release = resolve; }));
    const { unmount } = render(<AgentRecentActivity task={task} visible />);
    await waitFor(() => expect(read).toHaveBeenCalledOnce());
    const signal = read.mock.calls[0][1];
    unmount();
    expect(signal?.aborted).toBe(true);
    await act(async () => release(page([event()])));
    expect(screen.queryByRole("region", { name: "Recent activity" })).not.toBeInTheDocument();
  });

  it("ignores an older page that arrives after history becomes unavailable", async () => {
    let release!: (value: api.AgentEventsPage) => void;
    const read = vi.spyOn(api, "listAgentEvents").mockResolvedValueOnce(page(Array.from({ length: 7 }, (_, i) => event({ seq: 20 - i })), true))
      .mockImplementationOnce(() => new Promise((resolve) => { release = resolve; }))
      .mockRejectedValueOnce(new api.ApiError(404, "private detail"));
    const { rerender } = render(<AgentRecentActivity task={task} visible />);
    await screen.findByRole("list", { name: "Agent operations" });
    fireEvent.click(screen.getByRole("button", { name: "Show more activity" }));
    fireEvent.click(screen.getByRole("button", { name: "Load older activity" }));
    await waitFor(() => expect(read).toHaveBeenCalledTimes(2));
    const signal = read.mock.calls[1][1];
    rerender(<AgentRecentActivity task={{ ...task, revision: 2 }} visible />);
    expect(await screen.findByRole("alert")).toHaveTextContent("This agent’s activity is no longer available.");
    expect(signal?.aborted).toBe(true);
    await act(async () => release(page([event({ seq: 2, kind: "tool", tool: "create_file" })])));
    expect(screen.queryByRole("list", { name: "Agent operations" })).not.toBeInTheDocument();
    expect(screen.queryByText("Create file")).not.toBeInTheDocument();
  });

  it("shows only six rows until expanded and loads bounded older pages with the oldest sequence", async () => {
    const first = Array.from({ length: 50 }, (_, index) => event({ seq: 100 - index }));
    const read = vi.spyOn(api, "listAgentEvents").mockResolvedValueOnce(page(first, true))
      .mockResolvedValueOnce(page(Array.from({ length: 50 }, (_, index) => event({ seq: 50 - index })), false, true));
    render(<AgentRecentActivity task={{ ...task, status: "completed" }} visible />);
    const operations = await screen.findByRole("list", { name: "Agent operations" });
    expect(within(operations).getAllByRole("listitem")).toHaveLength(6);
    fireEvent.click(screen.getByRole("button", { name: "Show more activity" }));
    expect(within(operations).getAllByRole("listitem")).toHaveLength(50);
    fireEvent.click(screen.getByRole("button", { name: "Load older activity" }));
    await waitFor(() => expect(within(operations).getAllByRole("listitem")).toHaveLength(100));
    expect(read).toHaveBeenLastCalledWith(task.id, expect.any(AbortSignal), 51);
    expect(screen.queryByRole("button", { name: "Load older activity" })).not.toBeInTheDocument();
    expect(screen.getByText(/Older activity has been removed/)).toBeVisible();
    expect(document.activeElement).not.toBe(operations);
  });

  it("clears unavailable rows, hides raw server details and permits a safe explicit retry", async () => {
    const read = vi.spyOn(api, "listAgentEvents").mockResolvedValueOnce(page([event()]))
      .mockRejectedValueOnce(new api.ApiError(404, "private owner and path"))
      .mockResolvedValue(page([]));
    const { rerender } = render(<AgentRecentActivity task={task} visible />);
    await screen.findByText("Local model");
    rerender(<AgentRecentActivity task={{ ...task, revision: 2 }} visible />);
    expect(await screen.findByRole("alert")).toHaveTextContent("This agent’s activity is no longer available.");
    expect(screen.queryByText("Local model")).not.toBeInTheDocument();
    expect(screen.queryByText(/private owner/)).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Retry activity" }));
    expect(await screen.findByText(/No saved activity yet/)).toBeVisible();
    expect(read).toHaveBeenCalledTimes(3);
  });
});
