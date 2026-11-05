import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import * as api from "./api";
import { WatchesWorkspace } from "./WatchesWorkspace";
import { WatchForm } from "./WatchForm";
import type { AudreyProject, PublicWatch, WatchRun } from "./api";

function savedRun(patch: Partial<WatchRun> = {}): WatchRun {
  return { id: "run-1", watch_id: "watch-1", watch_revision: 1, status: "completed", outcome: "baseline", kind: "manual", scheduled_at: "2026-10-30T15:00:00Z", created_at: "2026-10-30T15:00:00Z", updated_at: "2026-10-30T15:01:00Z", completed_at: "2026-10-30T15:01:00Z", attempts: 1, report: "", diff: "", significant: null, error: "", ...patch };
}
function watch(patch: Partial<PublicWatch> = {}): PublicWatch {
  return { id: "watch-1", title: "Release announcements", url: "https://example.com/releases", criterion: "New stable releases", schedule: { kind: "daily", time: "09:00", timezone: "America/Denver" }, enabled: true, revision: 1, next_run_at: "2026-10-31T15:00:00Z", active_run_id: null, latest_run: savedRun(), created_at: "2026-10-30T14:00:00Z", updated_at: "2026-10-30T15:01:00Z", ...patch };
}
const occurrences = ["2026-10-31T15:00:00Z", "2026-11-01T16:00:00Z", "2026-11-02T16:00:00Z"];
const projectLimits = { max_name_chars: 100, max_instructions_chars: 10000, max_files: 100 };
function project(patch: Partial<AudreyProject> = {}): AudreyProject {
  return { id: "project-1", name: "Release research", instructions: "Project context", created_at: "2026-10-30T14:00:00Z", updated_at: "2026-10-30T14:00:00Z", ...patch };
}
function workspace(props: Partial<Parameters<typeof WatchesWorkspace>[0]> = {}) {
  return render(<WatchesWorkspace visible timezone="America/Denver" {...props} />);
}
function fillNewWatch() {
  fireEvent.change(screen.getByRole("textbox", { name: "Title" }), { target: { value: "Release announcements" } });
  fireEvent.change(screen.getByRole("textbox", { name: "Public page URL" }), { target: { value: "https://example.com/releases" } });
  fireEvent.change(screen.getByRole("textbox", { name: "What changes matter?" }), { target: { value: "New stable releases" } });
}
beforeEach(() => {
  vi.spyOn(api, "listProjects").mockResolvedValue({ items: [], next_cursor: null, limits: projectLimits });
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
  it("does not fetch or poll a hidden workspace and aborts outstanding visible reads", async () => {
    const view = workspace({ visible: false }); expect(api.listWatches).not.toHaveBeenCalled();
    view.rerender(<WatchesWorkspace visible />);
    await screen.findByRole("heading", { name: "Release announcements" });
    const signal = vi.mocked(api.listWatches).mock.calls.at(-1)?.[0];
    view.rerender(<WatchesWorkspace visible={false} />);
    expect(signal?.aborted).toBe(true);
  });

  it.each([false, true])("opens mobile controls with trapped focus and closes them after selecting a watch (sidebar tabs: %s)", async (integrated) => {
    vi.stubGlobal("matchMedia", vi.fn(() => ({ matches: true, addEventListener: vi.fn(), removeEventListener: vi.fn() })));
    workspace({ sidebarTabs: integrated ? <nav aria-label="Agent sections"><button type="button">New</button><button type="button">Watches</button><button type="button">Tasks</button></nav> : undefined });
    await screen.findByRole("heading", { name: "Release announcements" }); fireEvent.click(screen.getByRole("button", { name: "Watch controls" }));
    const dialog = screen.getByRole("dialog", { name: "Saved watches" }); const close = within(dialog).getByRole("button", { name: "Close watch controls" }); expect(close).toHaveFocus();
    if (integrated) {
      const firstTab = within(dialog).getByRole("button", { name: "New" });
      firstTab.focus(); fireEvent.keyDown(firstTab, { key: "Tab", shiftKey: true });
      expect(within(dialog).getByRole("button", { name: "Open watch: Release announcements" })).toHaveFocus();
      fireEvent.keyDown(document.activeElement!, { key: "Tab" });
      expect(firstTab).toHaveFocus();
    }
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

describe("Project watch navigation and drafts", () => {
  it("prefills a Project without creating or running, keeps the draft on another prefill and saves explicitly", async () => {
    const create = vi.spyOn(api, "createWatch").mockResolvedValue(watch({ project: { id: "project-2", name: "Another Project", available: true } }));
    const action = vi.spyOn(api, "actOnWatch");
    const view = workspace({ composerProjectRequest: { project: project(), serial: 1 } });
    const selector = await screen.findByRole("combobox", { name: "Project" });
    await waitFor(() => expect(selector).toHaveValue("project-1"));
    fillNewWatch();
    fireEvent.click(screen.getByRole("button", { name: "Preview next checks" }));
    await screen.findByText("Next three checks (UTC)");
    view.rerender(<WatchesWorkspace visible composerProjectRequest={{ project: project({ id: "project-2", name: "Another Project" }), serial: 2 }} />);
    await waitFor(() => expect(selector).toHaveValue("project-2"));
    expect(screen.getByRole("textbox", { name: "Title" })).toHaveValue("Release announcements");
    expect(screen.getByRole("textbox", { name: "What changes matter?" })).toHaveValue("New stable releases");
    expect(create).not.toHaveBeenCalled(); expect(action).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: "Create watch" }));
    await waitFor(() => expect(create).toHaveBeenCalledWith(expect.objectContaining({ project_id: "project-2", criterion: "New stable releases" }), expect.any(String)));
    expect(action).not.toHaveBeenCalled();
  });

  it("loads ordinary Projects across pages, excludes Private choices and keeps the selected draft on read failure", async () => {
    vi.mocked(api.listProjects).mockResolvedValueOnce({ items: [project(), project({ id: "private", name: "Private accounts", private_required: true })], next_cursor: "more", limits: projectLimits })
      .mockRejectedValueOnce(new Error("Project list unavailable"))
      .mockResolvedValueOnce({ items: [project()], next_cursor: "more", limits: projectLimits })
      .mockResolvedValueOnce({ items: [project({ id: "project-2", name: "Second Project" })], next_cursor: null, limits: projectLimits });
    workspace(); fireEvent.click(await screen.findByRole("button", { name: "New watch" }));
    expect(await screen.findByRole("option", { name: "Release research" })).toBeInTheDocument();
    expect(screen.queryByRole("option", { name: "Private accounts" })).not.toBeInTheDocument();
    fireEvent.change(screen.getByRole("combobox", { name: "Project" }), { target: { value: "project-1" } });
    fillNewWatch();
    fireEvent.click(screen.getByRole("button", { name: "Load more Projects" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Project list unavailable");
    expect(screen.getByRole("combobox", { name: "Project" })).toHaveValue("project-1");
    expect(screen.getByRole("textbox", { name: "Title" })).toHaveValue("Release announcements");
    fireEvent.click(screen.getByRole("button", { name: "Retry Projects" }));
    await waitFor(() => expect(screen.queryByRole("alert")).not.toBeInTheDocument());
    fireEvent.click(screen.getByRole("button", { name: "Load more Projects" }));
    expect(await screen.findByRole("option", { name: "Second Project" })).toBeInTheDocument();
    expect(api.listProjects).toHaveBeenLastCalledWith("more", expect.any(AbortSignal));
  });

  it("opens a freshly read existing watch and preserves the unsaved form for Resume draft", async () => {
    const create = vi.spyOn(api, "createWatch"); const action = vi.spyOn(api, "actOnWatch");
    const view = workspace();
    fireEvent.click(await screen.findByRole("button", { name: "New watch" })); fillNewWatch();
    fireEvent.change(screen.getByRole("textbox", { name: "Title" }), { target: { value: "Unsent watch draft" } });
    const external = watch({ id: "watch-2", title: "Fresh saved watch" });
    vi.mocked(api.getWatch).mockResolvedValue(external);
    view.rerender(<WatchesWorkspace visible externalWatchRequest={{ watch: { ...external, title: "Old watch title" }, serial: 1 }} />);
    expect(await screen.findByRole("heading", { name: "Fresh saved watch" })).toBeVisible();
    expect(api.getWatch).toHaveBeenCalledWith("watch-2", expect.any(AbortSignal));
    expect(create).not.toHaveBeenCalled(); expect(action).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: "Resume draft" }));
    expect(screen.getByRole("textbox", { name: "Title" })).toHaveValue("Unsent watch draft");
    expect(screen.getByRole("textbox", { name: "What changes matter?" })).toHaveValue("New stable releases");
  });

  it("aborts a pending external watch read when hidden and ignores its late completion", async () => {
    let finish!: (value: PublicWatch) => void;
    vi.mocked(api.getWatch).mockImplementation(() => new Promise((resolve) => { finish = resolve; }));
    const request = { watch: watch({ id: "watch-2" }), serial: 1 };
    const view = workspace({ externalWatchRequest: request });
    await waitFor(() => expect(api.getWatch).toHaveBeenCalled());
    const signal = vi.mocked(api.getWatch).mock.calls.at(-1)?.[1];
    view.rerender(<WatchesWorkspace visible={false} externalWatchRequest={request} />);
    expect(signal?.aborted).toBe(true);
    await act(async () => finish(watch({ id: "watch-2", title: "Stale hidden watch" })));
    expect(screen.queryByRole("heading", { name: "Stale hidden watch" })).not.toBeInTheDocument();
  });

  it("rejects a changed Project association between parent and child canonical watch reads", async () => {
    const incoming = watch({ id: "watch-2", title: "Project-linked watch", project: { id: "project-1", name: "First Project", available: true } });
    vi.mocked(api.getWatch).mockResolvedValue({ ...incoming, project: { id: "project-2", name: "Changed Project", available: true } });
    workspace({ externalWatchRequest: { watch: incoming, serial: 1 } });
    expect(await screen.findByRole("alert")).toHaveTextContent("Project changed or is unavailable");
    expect(screen.queryByRole("heading", { name: "Project-linked watch" })).not.toBeInTheDocument();
  });

  it("keeps an unavailable Project readable and editable but blocks Run now and Resume until detached", async () => {
    const unavailable = watch({ enabled: false, project: { id: "deleted", name: null, available: false } });
    vi.mocked(api.listWatches).mockResolvedValue({ items: [unavailable] });
    const update = vi.spyOn(api, "updateWatch").mockResolvedValue(watch({ project: null, revision: 2, enabled: false }));
    const action = vi.spyOn(api, "actOnWatch"); workspace();
    expect(await screen.findByRole("button", { name: "Run now" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Resume" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Delete watch" })).toBeEnabled();
    fireEvent.click(screen.getByRole("button", { name: "Edit" }));
    fireEvent.change(screen.getByRole("combobox", { name: "Project" }), { target: { value: "" } });
    fireEvent.click(screen.getByRole("button", { name: "Preview next checks" })); await screen.findByText("Next three checks (UTC)");
    fireEvent.click(screen.getByRole("button", { name: "Save watch" }));
    await waitFor(() => expect(update).toHaveBeenCalledWith("watch-1", expect.objectContaining({ project_id: null, enabled: false }), 1));
    expect(action).not.toHaveBeenCalled();
  });

  it("opens the current Project and preserves the saved check's original Project association", async () => {
    const current = { id: "project-current", name: "Current Project", available: true };
    const original = { id: "project-original", name: "Original Project", available: true };
    vi.mocked(api.listWatches).mockResolvedValue({ items: [watch({ project: current })] });
    vi.mocked(api.getWatchRun).mockResolvedValue(savedRun({ definition: { title: "Original title", url: "https://example.com/original", criterion: "Original criterion", schedule: watch().schedule, project: original } }));
    const open = vi.fn(); workspace({ onOpenProject: open });
    fireEvent.click(await screen.findByRole("button", { name: "Open Project" })); expect(open).toHaveBeenLastCalledWith("project-current");
    fireEvent.click(await screen.findByText("Saved check inputs", { selector: "summary" }));
    const saved = within(screen.getByText("Saved check inputs", { selector: "summary" }).parentElement!);
    expect(saved.getByText("Original Project")).toBeVisible();
    fireEvent.click(saved.getByRole("button", { name: "Open Project" })); expect(open).toHaveBeenLastCalledWith("project-original");
    expect(screen.getByRole("button", { name: "Run now" })).toBeEnabled();
  });
});
