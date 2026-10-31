import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ActivityWorkspace } from "./ActivityWorkspace";
import * as api from "./api";

function notice(patch: Partial<api.ActivityNotice> = {}): api.ActivityNotice {
  return { id: "notice-1", origin: "agent", kind: "completed", title: "Public documentation comparison", body: "Your agent completed. Open its saved result.", created_at: "2026-10-30T15:00:00Z", read_at: null, target: { kind: "agent", task_id: "agent-1", revision: 4 }, ...patch };
}
function task(patch: Partial<api.AgentTask> = {}): api.AgentTask {
  return { id: "agent-1", objective: "Compare public documentation", title: "Public documentation comparison", revision: 4, status: "completed", result: "Full canonical report", error: "", attempts: 1, steps: [], created_at: "2026-10-30T15:00:00Z", updated_at: "2026-10-30T15:10:00Z", ...patch };
}
function watch(): api.PublicWatch {
  return { id: "watch-1", title: "Release page", url: "https://example.com/releases", criterion: "Stable release", revision: 1, enabled: true, schedule: { kind: "daily", time: "09:00", timezone: "UTC" }, next_run_at: "2026-10-31T09:00:00Z", active_run_id: null, latest_run: null, created_at: "2026-10-30T15:00:00Z", updated_at: "2026-10-30T15:00:00Z" };
}
function run(): api.WatchRun {
  return { id: "old-run", watch_id: "watch-1", watch_revision: 1, status: "completed", outcome: "changed", kind: "scheduled", scheduled_at: "2026-10-30T15:00:00Z", created_at: "2026-10-30T15:00:00Z", updated_at: "2026-10-30T15:01:00Z", completed_at: "2026-10-30T15:01:00Z", attempts: 1, report: "Canonical saved assessment", diff: "+ New release", significant: true, error: "" };
}
function workspace(props: Partial<Parameters<typeof ActivityWorkspace>[0]> = {}) {
  return render(<ActivityWorkspace visible onUnreadChange={vi.fn()} onOpenAgent={vi.fn()} onOpenWatch={vi.fn()} {...props} />);
}
beforeEach(() => {
  vi.spyOn(api, "listActivity").mockResolvedValue({ items: [notice()], unread_count: 3 });
  vi.spyOn(api, "getActivityAgentTask").mockResolvedValue(task());
  vi.spyOn(api, "getWatch").mockResolvedValue(watch());
  vi.spyOn(api, "getWatchRun").mockResolvedValue(run());
  vi.spyOn(api, "readActivity").mockImplementation(async () => notice({ read_at: "2026-10-30T16:00:00Z" }));
});
afterEach(() => { cleanup(); vi.restoreAllMocks(); vi.unstubAllGlobals(); vi.useRealTimers(); });

describe("unified Activity", () => {
  it("labels mixed notices and renders all saved text without HTML or Markdown execution", async () => {
    const unsafe = "<img src=x onerror=alert(1)> **Saved text**";
    vi.mocked(api.listActivity).mockResolvedValue({ items: [notice({ title: unsafe, body: unsafe }), notice({ id: "failure", kind: "failure" }), notice({ id: "change", origin: "watch", kind: "change", target: { kind: "watch", watch_id: "watch-1", run_id: "old-run" } }), notice({ id: "watch-failed", origin: "watch", kind: "failure", target: { kind: "watch", watch_id: "watch-1", run_id: "old-run" } })], unread_count: 4 });
    workspace();
    expect(await screen.findByText("Unread · Agent completed")).toBeVisible();
    expect(screen.getByText("Unread · Agent failed")).toBeVisible(); expect(screen.getByText("Unread · Relevant change")).toBeVisible(); expect(screen.getByText("Unread · Watch failed")).toBeVisible();
    expect(screen.getAllByText(unsafe)).toHaveLength(2); expect(document.querySelector('img[src="x"]')).toBeNull();
  });

  it("resolves the canonical terminal agent revision before marking read and opening", async () => {
    const open = vi.fn(), unread = vi.fn();
    let finish!: (value: api.AgentTask) => void;
    vi.mocked(api.getActivityAgentTask).mockImplementation(() => new Promise((resolve) => { finish = resolve; }));
    workspace({ onOpenAgent: open, onUnreadChange: unread });
    fireEvent.click(await screen.findByRole("button", { name: /Public documentation comparison/ }));
    expect(api.readActivity).not.toHaveBeenCalled(); expect(open).not.toHaveBeenCalled();
    await act(async () => finish(task()));
    await waitFor(() => expect(open).toHaveBeenCalledWith(task()));
    expect(api.getActivityAgentTask).toHaveBeenCalledWith("notice-1", expect.any(AbortSignal));
    expect(api.readActivity).toHaveBeenCalledWith("agent", "notice-1", expect.any(AbortSignal));
    expect(unread).toHaveBeenLastCalledWith(2);
  });

  it.each([404, 409])("keeps an unavailable or rerun agent unread after HTTP %s", async (status) => {
    vi.mocked(api.getActivityAgentTask).mockRejectedValue(new api.ApiError(status, "Gone")); const open = vi.fn();
    workspace({ onOpenAgent: open }); fireEvent.click(await screen.findByRole("button", { name: /Public documentation comparison/ }));
    expect(await screen.findByRole("alert")).toHaveTextContent(/run again or deleted; refresh Activity/);
    expect(api.readActivity).not.toHaveBeenCalled(); expect(open).not.toHaveBeenCalled();
  });

  it("rejects a mismatched task revision without silently opening the latest run", async () => {
    vi.mocked(api.getActivityAgentTask).mockResolvedValue(task({ revision: 8 })); const open = vi.fn();
    workspace({ onOpenAgent: open }); fireEvent.click(await screen.findByRole("button", { name: /Public documentation comparison/ }));
    expect(await screen.findByRole("alert")).toHaveTextContent("refresh Activity"); expect(api.readActivity).not.toHaveBeenCalled(); expect(open).not.toHaveBeenCalled();
  });

  it("opens the exact notified watch run before marking its notice read", async () => {
    const item = notice({ origin: "watch", kind: "change", target: { kind: "watch", watch_id: "watch-1", run_id: "old-run" } });
    vi.mocked(api.listActivity).mockResolvedValue({ items: [item], unread_count: 1 });
    vi.mocked(api.readActivity).mockResolvedValue({ ...item, read_at: "2026-10-30T16:00:00Z" }); const open = vi.fn();
    workspace({ onOpenWatch: open }); fireEvent.click(await screen.findByRole("button", { name: /Public documentation comparison/ }));
    await waitFor(() => expect(open).toHaveBeenCalledWith(watch(), run()));
    expect(api.getWatchRun).toHaveBeenCalledWith("watch-1", "old-run", expect.any(AbortSignal)); expect(api.readActivity).toHaveBeenCalledWith("watch", "notice-1", expect.any(AbortSignal));
  });

  it("keeps deleted watch results unread and displays a refresh action", async () => {
    vi.mocked(api.listActivity).mockResolvedValue({ items: [notice({ origin: "watch", kind: "failure", target: { kind: "watch", watch_id: "watch-1", run_id: "old-run" } })], unread_count: 1 });
    vi.mocked(api.getWatchRun).mockRejectedValue(new api.ApiError(404, "Gone")); const open = vi.fn(); workspace({ onOpenWatch: open });
    fireEvent.click(await screen.findByRole("button", { name: /Public documentation comparison/ }));
    expect(await screen.findByRole("alert")).toHaveTextContent("This watch or saved check is no longer available"); expect(api.readActivity).not.toHaveBeenCalled(); expect(open).not.toHaveBeenCalled();
  });

  it("aborts target navigation on hiding and ignores a late saved result", async () => {
    let finish!: (value: api.AgentTask) => void;
    vi.mocked(api.getActivityAgentTask).mockImplementation(() => new Promise((resolve) => { finish = resolve; })); const open = vi.fn(), unread = vi.fn();
    const view = workspace({ onOpenAgent: open, onUnreadChange: unread });
    fireEvent.click(await screen.findByRole("button", { name: /Public documentation comparison/ }));
    const signal = vi.mocked(api.getActivityAgentTask).mock.calls[0][1];
    view.rerender(<ActivityWorkspace visible={false} onOpenAgent={open} onOpenWatch={vi.fn()} onUnreadChange={unread} />);
    expect(signal?.aborted).toBe(true); await act(async () => finish(task()));
    expect(open).not.toHaveBeenCalled(); expect(api.readActivity).not.toHaveBeenCalled();
  });

  it("does not fetch a hidden feed and does not mark an already-read notice again", async () => {
    const view = workspace({ visible: false }); expect(api.listActivity).not.toHaveBeenCalled();
    vi.mocked(api.listActivity).mockResolvedValue({ items: [notice({ read_at: "2026-10-30T16:00:00Z" })], unread_count: 0 });
    const open = vi.fn(); view.rerender(<ActivityWorkspace visible onOpenAgent={open} onOpenWatch={vi.fn()} onUnreadChange={vi.fn()} />);
    fireEvent.click(await screen.findByRole("button", { name: /Public documentation comparison/ }));
    await waitFor(() => expect(open).toHaveBeenCalledOnce()); expect(api.readActivity).not.toHaveBeenCalled();
  });

  it("suppresses feed polling while a read is pending so its count drops exactly once", async () => {
    vi.useFakeTimers(); const unread = vi.fn();
    let finishRead!: (value: api.ActivityNotice) => void;
    vi.mocked(api.readActivity).mockImplementation(() => new Promise((resolve) => { finishRead = resolve; }));
    workspace({ onUnreadChange: unread }); await act(async () => {});
    fireEvent.click(screen.getByRole("button", { name: /Public documentation comparison/ })); await act(async () => {});
    vi.mocked(api.listActivity).mockResolvedValue({ items: [notice({ read_at: "2026-10-30T16:00:00Z" })], unread_count: 2 });
    await act(async () => { await vi.advanceTimersByTimeAsync(15000); });
    expect(api.listActivity).toHaveBeenCalledTimes(1); expect(unread).toHaveBeenLastCalledWith(3);
    await act(async () => finishRead(notice({ read_at: "2026-10-30T16:00:00Z" })));
    expect(unread).toHaveBeenLastCalledWith(2);
    await act(async () => { await vi.advanceTimersByTimeAsync(15000); });
    expect(api.listActivity).toHaveBeenCalledTimes(2); expect(unread).toHaveBeenLastCalledWith(2);
  });

  it("does not let an older feed restore an acknowledged notice or unread count", async () => {
    const unread = vi.fn(); let finishFeed!: (value: { items: api.ActivityNotice[]; unread_count: number }) => void;
    workspace({ onUnreadChange: unread }); await screen.findByRole("button", { name: /Public documentation comparison/ });
    vi.mocked(api.listActivity).mockImplementation(() => new Promise((resolve) => { finishFeed = resolve; }));
    fireEvent.click(screen.getByRole("button", { name: "Refresh activity" })); await waitFor(() => expect(finishFeed).toBeTypeOf("function"));
    fireEvent.click(screen.getByRole("button", { name: /Public documentation comparison/ }));
    await waitFor(() => expect(unread).toHaveBeenLastCalledWith(2));
    await act(async () => finishFeed({ items: [notice()], unread_count: 3 }));
    expect(unread).toHaveBeenLastCalledWith(2); expect(screen.getByText("Read · Agent completed")).toBeVisible();
  });
});
