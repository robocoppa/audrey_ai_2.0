import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import * as api from "./api";
import { WatchesWorkspace } from "./WatchesWorkspace";
import { WatchForm } from "./WatchForm";
import type { PublicWatch, WatchRun } from "./api";

function savedRun(patch: Partial<WatchRun> = {}): WatchRun {
  return { id: "run-1", watch_id: "watch-1", watch_revision: 1, status: "completed", outcome: "baseline", kind: "manual", scheduled_at: "2026-10-30T15:00:00Z", created_at: "2026-10-30T15:00:00Z", updated_at: "2026-10-30T15:01:00Z", completed_at: "2026-10-30T15:01:00Z", attempts: 1, report: "", diff: "", significant: null, error: "", ...patch };
}
function watch(patch: Partial<PublicWatch> = {}): PublicWatch {
  return { id: "watch-1", title: "Release announcements", url: "https://example.com/releases", criterion: "New stable releases", schedule: { kind: "daily", time: "09:00", timezone: "America/Denver" }, enabled: true, revision: 1, next_run_at: "2026-10-31T15:00:00Z", active_run_id: null, latest_run: savedRun(), created_at: "2026-10-30T14:00:00Z", updated_at: "2026-10-30T15:01:00Z", ...patch };
}
const occurrences = ["2026-10-31T15:00:00Z", "2026-11-01T16:00:00Z", "2026-11-02T16:00:00Z"];
function workspace(props: Partial<Parameters<typeof WatchesWorkspace>[0]> = {}) {
  return render(<WatchesWorkspace visible mode="watches" timezone="America/Denver" onUnreadChange={vi.fn()} onOpenWatch={vi.fn()} {...props} />);
}
function fillNewWatch() {
  fireEvent.change(screen.getByRole("textbox", { name: "Title" }), { target: { value: "Release announcements" } });
  fireEvent.change(screen.getByRole("textbox", { name: "Public page URL" }), { target: { value: "https://example.com/releases" } });
  fireEvent.change(screen.getByRole("textbox", { name: "What changes matter?" }), { target: { value: "New stable releases" } });
}
beforeEach(() => {
  vi.spyOn(api, "listWatches").mockResolvedValue({ items: [watch()] });
  vi.spyOn(api, "listWatchNotifications").mockResolvedValue({ items: [], unread_count: 0 });
  vi.spyOn(api, "listWatchRuns").mockResolvedValue({ items: [savedRun()] });
  vi.spyOn(api, "getWatchRun").mockResolvedValue(savedRun());
  vi.spyOn(api, "getWatch").mockResolvedValue(watch());
  vi.spyOn(api, "previewWatchSchedule").mockResolvedValue({ occurrences });
});
afterEach(() => { cleanup(); vi.restoreAllMocks(); vi.unstubAllGlobals(); vi.useRealTimers(); });

describe("watch creation and schedules", () => {
  it("requires a current UTC preview, saves explicitly and never runs at creation", async () => {
    const create = vi.spyOn(api, "createWatch").mockResolvedValue(watch({ latest_run: null }));
    const run = vi.spyOn(api, "actOnWatch");
    workspace();
    fireEvent.click(await screen.findByRole("button", { name: "New watch" }));
    fillNewWatch();
    expect(screen.getByRole("button", { name: "Create watch" })).toBeDisabled();
    fireEvent.click(screen.getByRole("button", { name: "Preview next checks" }));
    expect(await screen.findByText("Next three checks (UTC)")).toBeVisible();
    expect(screen.getByText("2026-11-01 16:00:00 UTC")).toBeVisible();
    fireEvent.change(screen.getByRole("textbox", { name: "Time zone" }), { target: { value: "UTC" } });
    expect(screen.getByRole("button", { name: "Create watch" })).toBeDisabled();
    fireEvent.click(screen.getByRole("button", { name: "Preview next checks" }));
    await waitFor(() => expect(screen.getByRole("button", { name: "Create watch" })).toBeEnabled());
    fireEvent.click(screen.getByRole("button", { name: "Create watch" }));
    await waitFor(() => expect(create).toHaveBeenCalledWith(expect.objectContaining({ enabled: true, schedule: { kind: "daily", time: "09:00", timezone: "UTC" } }), expect.any(String)));
    expect(run).not.toHaveBeenCalled();
    expect(await screen.findByText(/Watch created\. Its first scheduled check/)).toBeVisible();
  });

  it("keeps creation identity and draft when the save response is lost", async () => {
    const create = vi.spyOn(api, "createWatch").mockRejectedValueOnce(new TypeError("Connection lost")).mockResolvedValueOnce(watch());
    workspace();
    fireEvent.click(await screen.findByRole("button", { name: "New watch" })); fillNewWatch();
    fireEvent.click(screen.getByRole("button", { name: "Preview next checks" }));
    await screen.findByText("Next three checks (UTC)");
    fireEvent.click(screen.getByRole("button", { name: "Create watch" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Connection lost");
    expect(screen.getByRole("textbox", { name: "Title" })).toHaveValue("Release announcements");
    fireEvent.click(screen.getByRole("button", { name: "Create watch" }));
    await waitFor(() => expect(create).toHaveBeenCalledTimes(2));
    expect(create.mock.calls[0][1]).toBe(create.mock.calls[1][1]);
  });

  it("sends interval and weekly schedules with bounded values and Monday-based days", async () => {
    render(<WatchForm busy={false} timezone="UTC" onSave={vi.fn()} onCancel={vi.fn()} />);
    fireEvent.change(screen.getByRole("combobox", { name: "Cadence" }), { target: { value: "interval" } });
    expect(screen.getByRole("spinbutton", { name: "Interval in minutes" })).toHaveAttribute("min", "60");
    expect(screen.getByRole("spinbutton", { name: "Interval in minutes" })).toHaveAttribute("max", "10080");
    fireEvent.click(screen.getByRole("button", { name: "Preview next checks" }));
    await waitFor(() => expect(api.previewWatchSchedule).toHaveBeenCalledWith({ kind: "interval", timezone: "UTC", interval_minutes: 1440 }, expect.any(AbortSignal)));
    fireEvent.change(screen.getByRole("combobox", { name: "Cadence" }), { target: { value: "weekly" } });
    fireEvent.change(screen.getByRole("combobox", { name: "Day" }), { target: { value: "6" } });
    fireEvent.click(screen.getByRole("button", { name: "Preview next checks" }));
    await waitFor(() => expect(api.previewWatchSchedule).toHaveBeenLastCalledWith({ kind: "weekly", timezone: "UTC", time: "09:00", weekday: 6 }, expect.any(AbortSignal)));
  });

  it("rejects credentials/query/fragment URLs without sending a create request", async () => {
    const save = vi.fn();
    render(<WatchForm busy={false} onSave={save} onCancel={vi.fn()} />); fillNewWatch();
    fireEvent.click(screen.getByRole("button", { name: "Preview next checks" })); await screen.findByText("Next three checks (UTC)");
    fireEvent.change(screen.getByRole("textbox", { name: "Public page URL" }), { target: { value: "https://user:secret@example.com/page?token=x#private" } });
    fireEvent.submit(screen.getByRole("form", { name: "Create watch" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("without credentials, a query string or a fragment"); expect(save).not.toHaveBeenCalled();
  });
});

describe("saved watch controls and history", () => {
  it("shows the saved check's original URL, criterion and schedule after its watch is edited", async () => {
    const definition = { title: "Original public page", url: "https://example.com/original", criterion: "The original change criterion", schedule: { kind: "weekly" as const, weekday: 1, time: "14:30", timezone: "Europe/London" } };
    const historic = savedRun({ definition, report: "Saved baseline result." });
    vi.mocked(api.listWatches).mockResolvedValue({ items: [watch({ url: "https://example.com/edited", criterion: "The newly edited criterion", revision: 4 })] });
    vi.mocked(api.listWatchRuns).mockResolvedValue({ items: [historic] });
    vi.mocked(api.getWatchRun).mockResolvedValue(historic);
    workspace();
    expect(await screen.findByText("The newly edited criterion")).toBeVisible();
    fireEvent.click(await screen.findByText("Saved check inputs", { selector: "summary" }));
    const inputs = within(screen.getByText("Saved check inputs", { selector: "summary" }).parentElement!);
    expect(inputs.getByText(definition.title)).toBeVisible();
    expect(inputs.getByText(definition.url)).toBeVisible();
    expect(inputs.getByText(definition.criterion)).toBeVisible();
    expect(inputs.getByText("Tuesday at 14:30 · Europe/London")).toBeVisible();
    expect(screen.getByRole("link", { name: "https://example.com/edited" })).toBeVisible();
  });

  it.each(["baseline", "unchanged"] as const)("labels a %s report as a check result without implying a model call", async (outcome) => {
    const result = savedRun({ outcome, report: "The public page check completed." });
    vi.mocked(api.listWatchRuns).mockResolvedValue({ items: [result] });
    vi.mocked(api.getWatchRun).mockResolvedValue(result);
    workspace();
    expect(await screen.findByRole("heading", { name: "Check result" })).toBeVisible();
    expect(screen.queryByRole("heading", { name: "Local model assessment" })).not.toBeInTheDocument();
  });

  it("shows exact diff and local assessment as plain text with distinct change outcomes", async () => {
    const changed = savedRun({ outcome: "changed", significant: false, kind: "catchup", report: "<img src=x onerror=alert(1)> **Not relevant**", diff: "- old text\n+ new text" });
    vi.mocked(api.listWatchRuns).mockResolvedValue({ items: [changed] }); vi.mocked(api.getWatchRun).mockResolvedValue(changed);
    workspace();
    expect(await screen.findByRole("heading", { name: "Changed" })).toBeVisible();
    expect(screen.getByText(/did not find a change relevant/)).toBeVisible();
    expect(screen.getByText(changed.report)).toBeVisible(); expect(document.querySelector('img[src="x"]')).toBeNull();
    expect(screen.getByText(/- old text/, { selector: "pre" })).not.toBeVisible(); fireEvent.click(screen.getByText("Exact normalized text diff", { selector: "summary" }));
    expect(screen.getByText(/- old text/, { selector: "pre" })).toBeVisible(); expect(screen.getAllByText("Catch-up after downtime").length).toBeGreaterThan(0);
  });

  it("runs a paused watch once without changing its saved schedule", async () => {
    vi.mocked(api.listWatches).mockResolvedValue({ items: [watch({ enabled: false, next_run_at: null })] });
    const action = vi.spyOn(api, "actOnWatch").mockResolvedValue(watch({ revision: 2, enabled: false, next_run_at: null, active_run_id: "run-2" }));
    workspace(); fireEvent.click(await screen.findByRole("button", { name: "Run now" }));
    await waitFor(() => expect(action).toHaveBeenCalledWith("watch-1", "run", 1));
    expect(screen.getByRole("button", { name: "Resume" })).toBeVisible(); expect(screen.getByRole("button", { name: "Run now" })).toBeDisabled();
    expect(await screen.findByText("Check queued. The saved schedule is unchanged.")).toBeVisible();
  });

  it("requires the red delete check twice and resets confirmation on blur and revision conflicts", async () => {
    const remove = vi.spyOn(api, "deleteWatch").mockRejectedValue(new api.ApiError(409, "Changed"));
    vi.mocked(api.getWatch).mockResolvedValue(watch({ revision: 2 })); workspace();
    fireEvent.click(await screen.findByRole("button", { name: "Delete watch" }));
    const confirm = screen.getByRole("button", { name: "Confirm delete watch" }); expect(confirm).toHaveTextContent("✓"); expect(confirm).toHaveClass("danger-button");
    fireEvent.blur(confirm); expect(screen.getByRole("button", { name: "Delete watch" })).toBeVisible(); expect(remove).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: "Delete watch" })); fireEvent.click(screen.getByRole("button", { name: "Confirm delete watch" }));
    await waitFor(() => expect(remove).toHaveBeenCalledWith("watch-1", 1));
    expect(await screen.findByText(/This watch changed\. Review/)).toBeVisible();
    expect(screen.getByRole("button", { name: "Delete watch" })).toBeVisible();
  });

  it("preserves conflicting edits and explicitly rebases onto the newest paused definition", async () => {
    const update = vi.spyOn(api, "updateWatch").mockRejectedValueOnce(new api.ApiError(409, "Changed")).mockResolvedValueOnce(watch({ revision: 3, enabled: false }));
    vi.mocked(api.getWatch).mockResolvedValue(watch({ revision: 2, enabled: false })); workspace();
    fireEvent.click(await screen.findByRole("button", { name: "Edit" }));
    fireEvent.change(screen.getByRole("textbox", { name: "What changes matter?" }), { target: { value: "My unsaved criterion" } });
    fireEvent.click(screen.getByRole("button", { name: "Preview next checks" })); await screen.findByText("Next three checks (UTC)");
    fireEvent.click(screen.getByRole("button", { name: "Save watch" }));
    expect(await screen.findByText("This watch changed in another session.")).toBeVisible(); expect(screen.getByRole("textbox", { name: "What changes matter?" })).toHaveValue("My unsaved criterion"); expect(screen.getByRole("button", { name: "Save watch" })).toBeDisabled();
    fireEvent.click(screen.getByRole("button", { name: "Use latest revision with my draft" })); fireEvent.click(screen.getByRole("button", { name: "Save watch" }));
    await waitFor(() => expect(update).toHaveBeenLastCalledWith("watch-1", expect.objectContaining({ criterion: "My unsaved criterion", enabled: false }), 2));
  });

  it("ignores late history and source reads after selecting a different watch", async () => {
    const second = watch({ id: "watch-2", title: "Second page", latest_run: savedRun({ id: "run-2", watch_id: "watch-2" }) });
    vi.mocked(api.listWatches).mockResolvedValue({ items: [watch(), second] });
    let finishFirst!: (value: WatchRun) => void;
    vi.mocked(api.getWatchRun).mockImplementation((id) => id === "watch-1" ? new Promise((resolve) => { finishFirst = resolve; }) : Promise.resolve(savedRun({ id: "run-2", watch_id: "watch-2", report: "Current second report" })));
    vi.mocked(api.listWatchRuns).mockImplementation(async (id) => ({ items: [savedRun({ id: id === "watch-1" ? "run-1" : "run-2", watch_id: id })] }));
    workspace(); await waitFor(() => expect(finishFirst).toBeTypeOf("function"));
    fireEvent.click(screen.getByRole("button", { name: "Open watch: Second page" })); expect(await screen.findByText("Current second report")).toBeVisible();
    await act(async () => finishFirst(savedRun({ report: "Outdated first report" })));
    expect(screen.queryByText("Outdated first report")).not.toBeInTheDocument();
  });
});

describe("watch activity and visibility", () => {
  it("opens the exact notified run and marks the durable notice read", async () => {
    const item: api.WatchNotification = { id: "notice-1", watch_id: "watch-1", run_id: "old-run", kind: "change", title: "New release", body: "Relevant release was announced", created_at: "2026-10-30T15:01:00Z", read_at: null };
    vi.mocked(api.listWatchNotifications).mockResolvedValue({ items: [item], unread_count: 3 });
    vi.mocked(api.getWatchRun).mockResolvedValue(savedRun({ id: "old-run" }));
    const read = vi.spyOn(api, "readWatchNotification").mockResolvedValue({ ...item, read_at: "2026-10-30T16:00:00Z" }); const open = vi.fn(), unread = vi.fn();
    workspace({ mode: "activity", onOpenWatch: open, onUnreadChange: unread });
    fireEvent.click(await screen.findByRole("button", { name: /New release/ }));
    await waitFor(() => expect(open).toHaveBeenCalledOnce()); expect(api.getWatchRun).toHaveBeenCalledWith("watch-1", "old-run", expect.any(AbortSignal)); expect(read).toHaveBeenCalledWith("notice-1"); expect(unread).toHaveBeenCalledWith(3);
  });

  it("does not fetch or poll a hidden workspace and aborts outstanding visible reads", async () => {
    const view = workspace({ visible: false }); expect(api.listWatches).not.toHaveBeenCalled();
    view.rerender(<WatchesWorkspace visible mode="watches" onUnreadChange={vi.fn()} onOpenWatch={vi.fn()} />);
    await screen.findByRole("heading", { name: "Release announcements" });
    const signal = vi.mocked(api.listWatches).mock.calls.at(-1)?.[0];
    view.rerender(<WatchesWorkspace visible={false} mode="watches" onUnreadChange={vi.fn()} onOpenWatch={vi.fn()} />);
    expect(signal?.aborted).toBe(true);
  });

  it("opens mobile controls with trapped focus and closes them after selecting a watch", async () => {
    vi.stubGlobal("matchMedia", vi.fn(() => ({ matches: true, addEventListener: vi.fn(), removeEventListener: vi.fn() }))); workspace();
    await screen.findByRole("heading", { name: "Release announcements" }); fireEvent.click(screen.getByRole("button", { name: "Watch controls" }));
    const dialog = screen.getByRole("dialog", { name: "Saved watches" }); const close = within(dialog).getByRole("button", { name: "Close watch controls" }); expect(close).toHaveFocus();
    fireEvent.click(within(dialog).getByRole("button", { name: "Open watch: Release announcements" }));
    expect(screen.queryByRole("dialog", { name: "Saved watches" })).not.toBeInTheDocument();
    await waitFor(() => expect(screen.getByRole("region", { name: "Selected watch" })).toHaveFocus());
  });

  it("keeps the visible workspace's shared navigation when a hidden watch view resizes", async () => {
    let resize!: (event: MediaQueryListEvent) => void;
    vi.stubGlobal("matchMedia", vi.fn(() => ({ matches: false, addEventListener: (_name: string, listener: typeof resize) => { resize = listener; }, removeEventListener: vi.fn() })));
    const change = vi.fn();
    workspace({ visible: false, navigationOpen: true, onNavigationOpenChange: change });
    await act(async () => resize({ matches: true } as MediaQueryListEvent));
    expect(change).not.toHaveBeenCalled();
  });
});
