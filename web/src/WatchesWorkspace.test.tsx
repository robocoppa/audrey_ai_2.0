import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import * as api from "./api";
import { WatchesWorkspace } from "./WatchesWorkspace";
import { WatchForm } from "./WatchForm";
import type { AudreyProject, AudreyProjectFile, AudreyProjectFileList, PublicWatch, WatchRun } from "./api";

function savedRun(patch: Partial<WatchRun> = {}): WatchRun {
  return { id: "run-1", watch_id: "watch-1", watch_revision: 1, status: "completed", outcome: "baseline", kind: "manual", scheduled_at: "2026-10-30T15:00:00Z", created_at: "2026-10-30T15:00:00Z", updated_at: "2026-10-30T15:01:00Z", completed_at: "2026-10-30T15:01:00Z", attempts: 1, report: "", diff: "", significant: null, error: "", ...patch };
}
function savedPage(run: WatchRun, text: string, start = 0, totalChars?: number): api.WatchRunSnapshotPage {
  const end = start + Array.from(text).length;
  const total = totalChars ?? end;
  return { watch_id: run.watch_id, run_id: run.id, source_url: "https://example.com/captured", start, end, total_chars: total, next_offset: end === total ? null : end, text };
}
const publicPageUrl = "https://example.com/releases";
function watch(patch: Partial<PublicWatch> = {}): PublicWatch {
  return { id: "watch-1", title: "Release announcements", url: publicPageUrl, criterion: "New stable releases", schedule: { kind: "daily", time: "09:00", timezone: "America/Denver" }, enabled: true, revision: 1, next_run_at: "2026-10-31T15:00:00Z", active_run_id: null, latest_run: savedRun(), created_at: "2026-10-30T14:00:00Z", updated_at: "2026-10-30T15:01:00Z", ...patch };
}
const occurrences = ["2026-10-31T15:00:00Z", "2026-11-01T16:00:00Z", "2026-11-02T16:00:00Z"];
const projectLimits = { max_name_chars: 100, max_instructions_chars: 10000, max_files: 100 };
function project(patch: Partial<AudreyProject> = {}): AudreyProject {
  return { id: "project-1", name: "Release research", instructions: "Project context", created_at: "2026-10-30T14:00:00Z", updated_at: "2026-10-30T14:00:00Z", ...patch };
}
function workspace(props: Partial<Parameters<typeof WatchesWorkspace>[0]> = {}) {
  return render(<WatchesWorkspace visible timezone="America/Denver" {...props} />);
}
function fillNewWatch(url = "https://example.com/releases") {
  fireEvent.change(screen.getByRole("textbox", { name: "Title" }), { target: { value: "Release announcements" } });
  fireEvent.change(screen.getByRole("textbox", { name: "Public page URL" }), { target: { value: url } });
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

describe("watch history navigation", () => {
  it("opens saved watches and check history before showing any creation fields", async () => {
    const create = vi.spyOn(api, "createWatch");
    const action = vi.spyOn(api, "actOnWatch");
    workspace({ historyRequest: 0 });
    expect(await screen.findByRole("heading", { name: "Release announcements" })).toBeVisible();
    expect(await screen.findByRole("heading", { name: "Check history" })).toBeVisible();
    const settings = within(screen.getByRole("region", { name: "Watch settings" }));
    expect(settings.getByRole("heading", { name: "Watched site" })).toBeVisible();
    expect(settings.getByRole("link", { name: publicPageUrl })).toHaveAttribute("href", publicPageUrl);
    expect(settings.getByRole("heading", { name: "Watch prompt" })).toBeVisible();
    expect(settings.getByText(watch().criterion)).toBeVisible();
    expect(screen.getByText("Next scheduled check:")).toBeVisible();
    expect(screen.getByRole("button", { name: "Open watch: Release announcements" })).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByRole("button", { name: "New watch" })).toBeVisible();
    expect(screen.queryByRole("form", { name: "Create watch" })).not.toBeInTheDocument();
    expect(api.previewWatchSchedule).not.toHaveBeenCalled();
    expect(create).not.toHaveBeenCalled();
    expect(action).not.toHaveBeenCalled();
  });

  it("keeps an empty history until New watch is explicitly selected", async () => {
    vi.mocked(api.listWatches).mockResolvedValue({ items: [] });
    const create = vi.spyOn(api, "createWatch");
    workspace({ historyRequest: 0 });
    expect(await screen.findByText("No watches yet. Create one to follow a public page or Project files.")).toBeVisible();
    expect(screen.getByRole("heading", { name: "Watch history" })).toBeVisible();
    expect(screen.queryByRole("textbox", { name: "Title" })).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "New watch" }));
    expect(await screen.findByRole("textbox", { name: "Title" })).toBeVisible();
    expect(screen.getByRole("combobox", { name: "Project" })).toBeVisible();
    expect(screen.getByRole("button", { name: "Create watch" })).toBeDisabled();
    fireEvent.click(screen.getByRole("button", { name: "Back to watches" }));
    expect(screen.getByRole("heading", { name: "Watch history" })).toBeVisible();
    expect(screen.queryByRole("textbox", { name: "Title" })).not.toBeInTheDocument();
    expect(create).not.toHaveBeenCalled();
    expect(api.previewWatchSchedule).not.toHaveBeenCalled();
  });

  it("returns to the selected history without losing a draft and keeps Cancel as explicit discard", async () => {
    const create = vi.spyOn(api, "createWatch");
    workspace();
    await screen.findByRole("heading", { name: "Release announcements" });
    fireEvent.click(screen.getByRole("button", { name: "New watch" }));
    fillNewWatch();
    fireEvent.change(screen.getByRole("textbox", { name: "Title" }), { target: { value: "My unsaved watch" } });
    fireEvent.click(screen.getByRole("button", { name: "Back to watches" }));
    expect(screen.getByRole("heading", { name: "Release announcements" })).toBeVisible();
    expect(screen.getByRole("heading", { name: "Check history" })).toBeVisible();
    expect(screen.queryByRole("textbox", { name: "Title" })).not.toBeInTheDocument();
    await waitFor(() => expect(screen.getByRole("region", { name: "Selected watch" })).toHaveFocus());
    fireEvent.click(screen.getByRole("button", { name: "Resume draft" }));
    expect(screen.getByRole("textbox", { name: "Title" })).toHaveValue("My unsaved watch");
    expect(screen.getByRole("textbox", { name: "Public page URL" })).toHaveValue("https://example.com/releases");
    fireEvent.click(screen.getByRole("button", { name: "Cancel" }));
    expect(screen.getByRole("heading", { name: "Release announcements" })).toBeVisible();
    expect(screen.queryByRole("button", { name: "Resume draft" })).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "New watch" }));
    expect(screen.getByRole("textbox", { name: "Title" })).toHaveValue("");
    expect(create).not.toHaveBeenCalled();
  });

  it("returns explicit Watches tab requests to history and reopens the retained new draft", async () => {
    const view = workspace({ historyRequest: 0 });
    await screen.findByRole("heading", { name: "Release announcements" });
    fireEvent.click(screen.getByRole("button", { name: "New watch" }));
    fillNewWatch();
    fireEvent.change(screen.getByRole("textbox", { name: "Title" }), { target: { value: "Retained through tabs" } });
    view.rerender(<WatchesWorkspace visible={false} historyRequest={0} />);
    view.rerender(<WatchesWorkspace visible historyRequest={1} />);
    expect(await screen.findByRole("heading", { name: "Release announcements" })).toBeVisible();
    expect(screen.queryByRole("textbox", { name: "Title" })).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "New watch" }));
    expect(screen.getByRole("textbox", { name: "Title" })).toHaveValue("Retained through tabs");
  });

  it("lets an explicit Project composer request take precedence over an initial or concurrent history request", async () => {
    const first = { project: project(), serial: 1 };
    const view = workspace({ historyRequest: 0, composerProjectRequest: first });
    const selector = await screen.findByRole("combobox", { name: "Project" });
    await waitFor(() => expect(selector).toHaveValue("project-1"));
    const second = { project: project({ id: "project-2", name: "Second Project" }), serial: 2 };
    view.rerender(<WatchesWorkspace visible historyRequest={1} composerProjectRequest={second} />);
    await waitFor(() => expect(selector).toHaveValue("project-2"));
    expect(screen.getByRole("textbox", { name: "Title" })).toBeVisible();
  });
});

describe("watch creation and schedules", () => {
  it.each([
    ["www.example.com/releases", "https://www.example.com/releases"],
    ["http://example.com/releases", "http://example.com/releases"],
  ])("saves %s with a current UTC preview and never runs at creation", async (input, expectedUrl) => {
    const create = vi.spyOn(api, "createWatch").mockResolvedValue(watch({ latest_run: null }));
    const run = vi.spyOn(api, "actOnWatch");
    workspace();
    fireEvent.click(await screen.findByRole("button", { name: "New watch" }));
    fillNewWatch(input);
    expect(screen.getByRole("button", { name: "Create watch" })).toBeDisabled();
    fireEvent.click(screen.getByRole("button", { name: "Preview next checks" }));
    expect(await screen.findByText("Next three checks (UTC)")).toBeVisible();
    expect(screen.getByText("2026-11-01 16:00:00 UTC")).toBeVisible();
    fireEvent.change(screen.getByRole("textbox", { name: "Time zone" }), { target: { value: "UTC" } });
    expect(screen.getByRole("button", { name: "Create watch" })).toBeDisabled();
    fireEvent.click(screen.getByRole("button", { name: "Preview next checks" }));
    await waitFor(() => expect(screen.getByRole("button", { name: "Create watch" })).toBeEnabled());
    fireEvent.click(screen.getByRole("button", { name: "Create watch" }));
    await waitFor(() => expect(create).toHaveBeenCalledWith(expect.objectContaining({
      url: expectedUrl,
      enabled: true,
      schedule: { kind: "daily", time: "09:00", timezone: "UTC" },
    }), expect.any(String)));
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

  it("previews and saves Hourly as a 60-minute interval and restores it when editing", async () => {
    const schedule = { kind: "interval", timezone: "America/Denver", interval_minutes: 60 } as const;
    const create = vi.spyOn(api, "createWatch").mockImplementation(async (definition) => {
      const saved = watch({ schedule: definition.schedule, latest_run: null });
      vi.mocked(api.listWatches).mockResolvedValue({ items: [saved] });
      return saved;
    });
    workspace();
    fireEvent.click(await screen.findByRole("button", { name: "New watch" }));
    fillNewWatch();
    const cadence = screen.getByRole("combobox", { name: "Cadence" });
    expect(within(cadence).getByRole("option", { name: "Hourly" })).toBeVisible();
    fireEvent.change(cadence, { target: { value: "hourly" } });
    expect(screen.queryByRole("spinbutton", { name: "Interval in minutes" })).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Preview next checks" }));
    await screen.findByText("Next three checks (UTC)");
    expect(api.previewWatchSchedule).toHaveBeenCalledWith(schedule, expect.any(AbortSignal));
    fireEvent.click(screen.getByRole("button", { name: "Create watch" }));
    expect(await screen.findByText("Hourly · America/Denver")).toBeVisible();
    expect(create).toHaveBeenCalledWith(expect.objectContaining({ schedule }), expect.any(String));
    fireEvent.click(screen.getByRole("button", { name: "Edit" }));
    expect(screen.getByRole("combobox", { name: "Cadence" })).toHaveValue("hourly");
  });

  it("sends interval and weekly schedules with bounded values and Monday-based days", async () => {
    render(<WatchForm busy={false} timezone="UTC" onSave={vi.fn()} onCancel={vi.fn()} />);
    fireEvent.change(screen.getByRole("combobox", { name: "Cadence" }), { target: { value: "interval" } });
    const minutes = screen.getByRole("spinbutton", { name: "Interval in minutes" });
    expect(minutes).toHaveAttribute("min", "60");
    expect(minutes).toHaveAttribute("max", "10080");
    fireEvent.change(minutes, { target: { value: "90" } });
    fireEvent.click(screen.getByRole("button", { name: "Preview next checks" }));
    await waitFor(() => expect(api.previewWatchSchedule).toHaveBeenCalledWith({ kind: "interval", timezone: "UTC", interval_minutes: 90 }, expect.any(AbortSignal)));
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
    expect(screen.getByRole("heading", { name: "Page changes" })).toBeVisible();
    expect(screen.queryByText("Page changes", { selector: "summary" })).not.toBeInTheDocument();
    expect(screen.getByText(outcome === "baseline" ? /saved the baseline for future comparisons/ : /page text is unchanged/)).toBeVisible();
  });

  it("switches saved checks immediately and ignores a late detail response for the previous card", async () => {
    const first = savedRun({ outcome: "unchanged", report: "First saved check result." });
    const second = savedRun({ id: "run-2", outcome: "unchanged", report: "Second saved check result." });
    let finishFirst!: (result: WatchRun) => void;
    let finishSecond!: (result: WatchRun) => void;
    vi.mocked(api.listWatchRuns).mockResolvedValue({ items: [first, second] });
    const getRun = vi.mocked(api.getWatchRun).mockImplementation((_watchId, id) => new Promise((resolve) => {
      if (id === first.id) finishFirst = resolve;
      else finishSecond = resolve;
    }));
    workspace();
    expect(await screen.findByText(first.report)).toBeVisible();
    const firstSignal = getRun.mock.calls[0][2];
    const history = screen.getByRole("region", { name: "Check history" });
    const secondCard = within(history).getAllByRole("button", { name: /^Unchanged/ })[1];

    fireEvent.click(secondCard);
    expect(screen.getByText(second.report)).toBeVisible();
    expect(screen.queryByText(first.report)).not.toBeInTheDocument();
    expect(screen.queryByText("Loading saved check…")).not.toBeInTheDocument();
    expect(firstSignal?.aborted).toBe(true);
    fireEvent.click(secondCard);
    expect(screen.getByText(second.report)).toBeVisible();

    await act(async () => finishFirst({ ...first, report: "Late first check response." }));
    expect(screen.queryByText("Late first check response.")).not.toBeInTheDocument();
    await act(async () => finishSecond({ ...second, report: "Revalidated second check." }));
    expect(screen.getByText("Revalidated second check.")).toBeVisible();
  });

  it("shows a relevant change using the exact saved before and after excerpts", async () => {
    const before = "Café 🧪\nSpring plan\nDetails";
    const after = "Café 🧪\nWinter plan\nDetails";
    const excerpt = (text: string) => ({ start: 31, end: 31 + Array.from(text).length, text });
    const evidenceJson = (oldText: string, newText: string) => JSON.stringify({
      offset_unit: "Unicode characters, zero-based, end-exclusive",
      coverage: "All changed text is included; surrounding text is context.",
      old: excerpt(oldText),
      new: excerpt(newText),
    });
    const changed = savedRun({
      outcome: "changed",
      significant: true,
      report: "The seasonal plan changed.",
      diff: evidenceJson(before, after),
    });
    const historic = savedRun({ id: "older-change", outcome: "changed", diff: evidenceJson("Details", "Details\nNew section") });
    vi.mocked(api.listWatches).mockResolvedValue({ items: [watch({ latest_run: changed })] });
    vi.mocked(api.listWatchRuns).mockResolvedValue({ items: [changed, historic] });
    vi.mocked(api.getWatchRun).mockImplementation(async (_, id) => id === historic.id ? historic : changed);
    workspace();
    expect(await screen.findByRole("heading", { name: "Relevant change" })).toBeVisible();
    const sidebar = screen.getByRole("button", { name: "Open watch: Release announcements" });
    expect(within(sidebar).getByText("Relevant change")).toBeVisible();
    const history = screen.getByRole("region", { name: "Check history" });
    expect(within(history).getByRole("button", { name: /^Relevant change/ })).toHaveAttribute("aria-pressed", "true");
    const evidence = screen.getByText("Page changes", { selector: "summary" });
    expect(evidence.parentElement).not.toHaveAttribute("open");
    fireEvent.click(evidence);
    expect(screen.getByText("Before", { exact: true })).toBeVisible();
    expect(screen.getByText("After", { exact: true })).toBeVisible();
    expect(screen.getByText("Spring", { selector: "mark" }).closest("pre")?.textContent).toBe(before);
    expect(screen.getByText("Winter", { selector: "mark" }).closest("pre")?.textContent).toBe(after);
    fireEvent.click(within(history).getByRole("button", { name: /^Changed/ }));
    expect(await screen.findByRole("heading", { name: "Changed" })).toBeVisible();
    const olderEvidence = screen.getByText("Page changes", { selector: "summary" });
    expect(olderEvidence.parentElement).not.toHaveAttribute("open");
    expect(screen.queryByText("Spring", { selector: "mark" })).not.toBeInTheDocument();
    fireEvent.click(olderEvidence);
    const oldExcerpt = screen.getByLabelText("Before excerpt");
    expect(oldExcerpt).toHaveTextContent("Details");
    expect(oldExcerpt.querySelector("mark")).toBeNull();
    expect(screen.getByText("New section", { selector: "mark" })).toBeVisible();
  });

  it("shows an older diff and local assessment as plain text for other page changes", async () => {
    const diff = "- old text\n+ <img src=old-diff onerror=alert(1)>";
    const changed = savedRun({ outcome: "changed", significant: false, kind: "catchup", report: "<img src=x onerror=alert(1)> **Not relevant**", diff });
    vi.mocked(api.listWatchRuns).mockResolvedValue({ items: [changed] });
    vi.mocked(api.getWatchRun).mockResolvedValue(changed);
    workspace();
    expect(await screen.findByRole("heading", { name: "Other page change" })).toBeVisible();
    expect(screen.getByText(/did not find a change relevant/)).toBeVisible();
    expect(screen.getByText(changed.report)).toBeVisible();
    const savedDiff = screen.getByText(/- old text/, { selector: "pre" });
    expect(savedDiff).not.toBeVisible();
    fireEvent.click(screen.getByText("Page changes", { selector: "summary" }));
    expect(savedDiff).toBeVisible();
    expect(savedDiff.textContent).toBe(diff);
    expect(document.querySelector('img[src="x"], img[src="old-diff"]')).toBeNull();
    expect(screen.getAllByText("Catch-up after downtime").length).toBeGreaterThan(0);
  });

  it("runs a paused watch once without changing its saved schedule", async () => {
    vi.mocked(api.listWatches).mockResolvedValue({ items: [watch({ enabled: false, next_run_at: null })] });
    const action = vi.spyOn(api, "actOnWatch").mockResolvedValue(watch({ revision: 2, enabled: false, next_run_at: null, active_run_id: "run-2" }));
    workspace(); fireEvent.click(await screen.findByRole("button", { name: "Run now" }));
    await waitFor(() => expect(action).toHaveBeenCalledWith("watch-1", "run", 1));
    expect(screen.getByRole("button", { name: "Resume" })).toBeVisible(); expect(screen.getByRole("button", { name: "Run now" })).toBeDisabled();
    expect(await screen.findByText("Check queued. The saved schedule is unchanged.")).toBeVisible();
  });

  it("requires confirmation for a paused baseline reset and opens the reset check beside earlier checks", async () => {
    const paused = watch({ enabled: false, next_run_at: null });
    const reset = savedRun({ id: "reset-1", rebaseline: true, watch_revision: 2 });
    const updated = { ...paused, revision: 2, latest_run: reset };
    vi.mocked(api.listWatches).mockResolvedValue({ items: [paused] });
    vi.mocked(api.getWatchRun).mockImplementation(async (_, runId) => runId === reset.id ? reset : savedRun());
    const action = vi.spyOn(api, "actOnWatch").mockImplementation(async () => {
      vi.mocked(api.listWatches).mockResolvedValue({ items: [updated] });
      vi.mocked(api.listWatchRuns).mockResolvedValue({ items: [reset, savedRun()] });
      return updated;
    });
    workspace();
    const history = await screen.findByRole("region", { name: "Check history" });
    const resetButton = within(history).getByRole("button", { name: "Reset baseline" });
    resetButton.focus();
    fireEvent.click(resetButton);
    expect(action).not.toHaveBeenCalled();
    const confirm = within(history).getByRole("button", { name: "Confirm baseline reset" });
    expect(confirm).toHaveAccessibleDescription(/comparison point once the check succeeds/);
    fireEvent.keyDown(confirm, { key: "Escape" });
    expect(resetButton).toHaveFocus();
    expect(resetButton).toHaveTextContent("Reset baseline");
    fireEvent.click(resetButton);
    fireEvent.click(within(history).getByRole("button", { name: "Confirm baseline reset" }));
    expect(action).toHaveBeenCalledExactlyOnceWith("watch-1", "rebaseline", 1);
    expect(await screen.findByText("This successful check saved a new baseline for future comparisons.")).toBeVisible();
    expect(within(history).getByRole("button", { name: /Baseline saved\s*Baseline reset/ })).toHaveAttribute("aria-pressed", "true");
    expect(within(history).getByRole("button", { name: /Baseline saved\s*Run now/ })).toBeVisible();
    expect(screen.getByRole("button", { name: "Resume" })).toBeVisible();
    const detail = within(screen.getByRole("region", { name: "Selected watch" }));
    expect(detail.getByText("Paused", { exact: true })).toBeVisible();
    expect(detail.queryByText("Next scheduled check:")).not.toBeInTheDocument();
  });

  it.each([
    { active_run_id: "running-check" },
    { project: { id: "project-1", name: null, available: false } },
  ])("disables reset while a check is active or the Project is unavailable (%j)", async (patch) => {
    vi.mocked(api.listWatches).mockResolvedValue({ items: [watch(patch)] });
    workspace();
    expect(await screen.findByRole("button", { name: "Reset baseline" })).toBeDisabled();
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
    workspace({ sidebarTabs: integrated ? <nav aria-label="Agent sections"><button type="button">Agents</button><button type="button">Watches</button></nav> : undefined });
    await screen.findByRole("heading", { name: "Release announcements" }); fireEvent.click(screen.getByRole("button", { name: "Watch controls" }));
    const dialog = screen.getByRole("dialog", { name: "Saved watches" }); const close = within(dialog).getByRole("button", { name: "Close watch controls" }); expect(close).toHaveFocus();
    if (integrated) {
      close.focus(); fireEvent.keyDown(close, { key: "Tab", shiftKey: true });
      expect(within(dialog).getByRole("button", { name: "Open watch: Release announcements" })).toHaveFocus();
      fireEvent.keyDown(document.activeElement!, { key: "Tab" });
      expect(close).toHaveFocus();
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

describe("saved watch page text", () => {
  it("finds text in the full capture and follows exact Unicode matches across pages", async () => {
    const run = savedRun();
    const first = "🌲" + "a".repeat(3997) + "Py";
    const second = "thon 3.15\n🐢 Python 3.14";
    const total = 4000 + Array.from(second).length;
    const later = 4000 + Array.from(second.split("Python")[0]).length;
    const read = vi.spyOn(api, "getWatchRunSnapshot").mockImplementation(async (_, __, offset) => offset
      ? savedPage(run, second, 4000)
      : savedPage(run, first, 0, total));
    const search = vi.spyOn(api, "searchWatchRunSnapshot").mockResolvedValue({
      watch_id: run.watch_id, run_id: run.id, total_chars: total, truncated: false,
      matches: [{ start: 3998, end: 4004 }, { start: later, end: later + 6 }],
    });
    workspace();
    fireEvent.click(await screen.findByText("Saved page text", { selector: "summary" }));
    const input = await screen.findByRole("searchbox", { name: "Find text" });
    fireEvent.change(input, { target: { value: "  pYtHoN  " } });
    expect(search).not.toHaveBeenCalled();
    fireEvent.submit(screen.getByRole("search", { name: "Find in saved page text" }));
    expect(await screen.findByText("Match 1 of 2")).toBeVisible();
    expect(search).toHaveBeenCalledExactlyOnceWith(run.watch_id, run.id, "pYtHoN", expect.any(AbortSignal));
    expect(screen.getByLabelText("Saved page text").querySelector("mark")).toHaveTextContent("Py");
    fireEvent.click(screen.getByRole("button", { name: "Next page" }));
    const secondPage = await screen.findByLabelText("Saved page text");
    expect(secondPage.querySelector("mark")).toHaveTextContent("thon");
    fireEvent.click(screen.getByRole("button", { name: "Next match" }));
    expect(screen.getByText("Match 2 of 2")).toBeVisible();
    expect(secondPage.querySelector("mark")).toHaveTextContent("Python");
    fireEvent.click(screen.getByRole("button", { name: "Previous match" }));
    expect((await screen.findByLabelText("Saved page text")).textContent).toBe(first);
    expect(read.mock.calls.map(([, , offset]) => offset)).toEqual([0, 4000, 0]);
  });

  it("cancels stale find results and shows a genuine empty result for the new query", async () => {
    const run = savedRun();
    const page = savedPage(run, "Python release notes");
    const empty = { watch_id: run.watch_id, run_id: run.id, total_chars: page.total_chars, matches: [], truncated: false };
    vi.spyOn(api, "getWatchRunSnapshot").mockResolvedValue(page);
    let finish!: (result: api.WatchSnapshotSearchResult) => void;
    const search = vi.spyOn(api, "searchWatchRunSnapshot")
      .mockImplementationOnce(() => new Promise((resolve) => { finish = resolve; }))
      .mockResolvedValueOnce(empty);
    workspace();
    fireEvent.click(await screen.findByText("Saved page text", { selector: "summary" }));
    const input = await screen.findByRole("searchbox", { name: "Find text" });
    const form = screen.getByRole("search", { name: "Find in saved page text" });
    fireEvent.change(input, { target: { value: "Python" } });
    fireEvent.submit(form);
    expect(screen.getByText("Finding text…")).toBeVisible();
    const signal = search.mock.calls[0][3];
    fireEvent.change(input, { target: { value: "No such version" } });
    expect(signal?.aborted).toBe(true);
    fireEvent.submit(form);
    expect(await screen.findByText("No matches in this saved check.")).toBeVisible();
    await act(async () => finish({ ...empty, matches: [{ start: 0, end: 6 }] }));
    expect(screen.queryByText("Match 1 of 1")).not.toBeInTheDocument();
    expect(screen.getByLabelText("Saved page text").querySelector("mark")).toBeNull();
  });

  it("fetches only when opened and pages through exact Unicode text without rendering markup", async () => {
    const run = savedRun({ outcome: "unchanged" });
    vi.mocked(api.listWatchRuns).mockResolvedValue({ items: [run] });
    vi.mocked(api.getWatchRun).mockResolvedValue(run);
    const firstText = "a".repeat(3999) + "🧪";
    const secondText = "<img src=snapshot.invalid />\nCafé 🐢";
    const total = 4000 + Array.from(secondText).length;
    const read = vi.spyOn(api, "getWatchRunSnapshot").mockImplementation(async (_, __, offset) => (
      offset === 0 ? savedPage(run, firstText, 0, total) : savedPage(run, secondText, 4000)
    ));
    const action = vi.spyOn(api, "actOnWatch");
    workspace();
    const summary = await screen.findByText("Saved page text", { selector: "summary" });
    expect(summary.parentElement).not.toHaveAttribute("open");
    expect(read).not.toHaveBeenCalled();
    fireEvent.click(summary);
    const first = await screen.findByLabelText("Saved page text");
    expect(first.textContent).toBe(firstText);
    expect(screen.getByRole("status")).toHaveTextContent(`Characters 1–4,000 of ${total.toLocaleString()}`);
    expect(screen.getByText("https://example.com/captured")).toBeVisible();
    expect(read).toHaveBeenCalledExactlyOnceWith(run.watch_id, run.id, 0, expect.any(AbortSignal));

    const previous = screen.getByRole("button", { name: "Previous page" });
    const next = screen.getByRole("button", { name: "Next page" });
    expect(previous).toHaveAttribute("aria-disabled", "true");
    fireEvent.click(next);
    expect(screen.queryByLabelText("Saved page text")).not.toBeInTheDocument();
    const second = await screen.findByLabelText("Saved page text");
    expect(second.textContent).toBe(secondText);
    expect(second.querySelector("img")).toBeNull();
    expect(screen.getByRole("status")).toHaveTextContent(`Characters 4,001–${total.toLocaleString()} of ${total.toLocaleString()}`);
    expect(next).toHaveAttribute("aria-disabled", "true");
    fireEvent.click(next);
    expect(read).toHaveBeenCalledTimes(2);
    fireEvent.click(previous);
    expect((await screen.findByLabelText("Saved page text")).textContent).toBe(firstText);
    expect(read.mock.calls.map(([, , offset]) => offset)).toEqual([0, 4000, 0]);
    expect(action).not.toHaveBeenCalled();
  });

  it("aborts a pending page and closes the disclosure when another check is selected", async () => {
    const firstRun = savedRun();
    const secondRun = savedRun({ id: "run-2", outcome: "changed" });
    vi.mocked(api.listWatchRuns).mockResolvedValue({ items: [firstRun, secondRun] });
    vi.mocked(api.getWatchRun).mockImplementation(async (_, id) => id === firstRun.id ? firstRun : secondRun);
    let finish!: (page: api.WatchRunSnapshotPage) => void;
    const read = vi.spyOn(api, "getWatchRunSnapshot").mockImplementation((_watchId, id) => id === firstRun.id
      ? new Promise((resolve) => { finish = resolve; })
      : Promise.resolve(savedPage(secondRun, "Second check page.")));
    workspace();
    const summary = await screen.findByText("Saved page text", { selector: "summary" });
    fireEvent.click(summary);
    expect(await screen.findByText("Loading saved page text…")).toBeVisible();
    const signal = read.mock.calls[0][3];
    fireEvent.click(within(screen.getByRole("region", { name: "Check history" })).getByRole("button", { name: /^Changed/ }));
    expect(signal?.aborted).toBe(true);
    const nextSummary = screen.getByText("Saved page text", { selector: "summary" });
    expect(nextSummary.parentElement).not.toHaveAttribute("open");
    await act(async () => finish(savedPage(firstRun, "Outdated first check page.")));
    expect(screen.queryByText("Outdated first check page.")).not.toBeInTheDocument();
    expect(read).toHaveBeenCalledTimes(1);
    fireEvent.click(nextSummary);
    expect(await screen.findByText("Second check page.")).toBeVisible();
  });

  it.each(["close", "hide", "drawer"] as const)("discards the pending page on %s", async (dismiss) => {
    if (dismiss === "drawer") {
      vi.stubGlobal("matchMedia", vi.fn(() => ({ matches: true, addEventListener: vi.fn(), removeEventListener: vi.fn() })));
    }
    let finish!: (page: api.WatchRunSnapshotPage) => void;
    const read = vi.spyOn(api, "getWatchRunSnapshot").mockImplementation(() => new Promise((resolve) => { finish = resolve; }));
    const view = workspace();
    const summary = await screen.findByText("Saved page text", { selector: "summary" });
    fireEvent.click(summary);
    expect(await screen.findByText("Loading saved page text…")).toBeVisible();
    const signal = read.mock.calls[0][3];
    if (dismiss === "close") fireEvent.click(summary);
    if (dismiss === "hide") view.rerender(<WatchesWorkspace visible={false} />);
    if (dismiss === "drawer") fireEvent.click(screen.getByRole("button", { name: "Watch controls" }));
    await waitFor(() => expect(signal?.aborted).toBe(true));
    await act(async () => finish(savedPage(savedRun(), "Discarded page.")));
    expect(screen.queryByText("Discarded page.")).not.toBeInTheDocument();
    if (dismiss === "hide") view.rerender(<WatchesWorkspace visible />);
    if (dismiss === "drawer") fireEvent.click(screen.getByRole("button", { name: "Close watch controls" }));
    expect(screen.getByText("Saved page text", { selector: "summary" }).parentElement).not.toHaveAttribute("open");
    expect(read).toHaveBeenCalledTimes(1);
  });

  it("shows unavailable and recoverable errors without showing a body or offering text for a failed check", async () => {
    const failed = savedRun({ id: "failed", status: "failed", outcome: null });
    vi.mocked(api.listWatchRuns).mockResolvedValue({ items: [savedRun(), failed] });
    vi.mocked(api.getWatchRun).mockImplementation(async (_, id) => id === failed.id ? failed : savedRun());
    const read = vi.spyOn(api, "getWatchRunSnapshot")
      .mockRejectedValueOnce(new api.ApiError(404, "Internal storage error detail"))
      .mockRejectedValueOnce(new api.ApiError(503, "Internal storage error detail"))
      .mockResolvedValueOnce(savedPage(savedRun(), "Recovered page."));
    workspace();
    const summary = await screen.findByText("Saved page text", { selector: "summary" });
    fireEvent.click(summary);
    expect(await screen.findByRole("alert")).toHaveTextContent("Saved page text is unavailable for this check.");
    expect(screen.queryByText("Internal storage error detail")).not.toBeInTheDocument();
    fireEvent.click(summary);
    await waitFor(() => expect(screen.queryByRole("alert")).not.toBeInTheDocument());
    fireEvent.click(summary);
    expect(await screen.findByRole("alert")).toHaveTextContent("Could not load saved page text. Close and reopen to try again.");
    fireEvent.click(summary);
    await waitFor(() => expect(screen.queryByRole("alert")).not.toBeInTheDocument());
    fireEvent.click(summary);
    expect(await screen.findByText("Recovered page.")).toBeVisible();
    fireEvent.click(within(screen.getByRole("region", { name: "Check history" })).getByRole("button", { name: /^Failed/ }));
    expect(screen.queryByText("Saved page text", { selector: "summary" })).not.toBeInTheDocument();
    expect(screen.queryByLabelText("Saved page text")).not.toBeInTheDocument();
    expect(read).toHaveBeenCalledTimes(3);
  });

  it.each([
    { run_id: "other-run" },
    { next_offset: 0, total_chars: 6000 },
    { end: 4000, total_chars: 4000 },
    { total_chars: null },
  ])("rejects an inconsistent response without rendering its text", async (override) => {
    const page = { ...savedPage(savedRun(), "Unverified text."), ...override } as api.WatchRunSnapshotPage;
    vi.spyOn(api, "getWatchRunSnapshot").mockResolvedValue(page);
    workspace();
    fireEvent.click(await screen.findByText("Saved page text", { selector: "summary" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Could not load saved page text.");
    expect(screen.queryByText("Unverified text.")).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Next page" })).not.toBeInTheDocument();
  });
});

describe("Project watch navigation and drafts", () => {
  it("creates a Project file watch from an explicit ordinary Project selection", async () => {
    const selected = project();
    vi.mocked(api.listProjects).mockResolvedValue({
      items: [selected, project({ id: "private", name: "Private files", private_required: true })],
      next_cursor: null,
      limits: projectLimits,
    });
    const saved = watch({ title: "Daily attachments", source_kind: "project_files", url: null,
      project: { id: selected.id, name: selected.name, available: true }, latest_run: null });
    const create = vi.spyOn(api, "createWatch").mockImplementation(async () => {
      vi.mocked(api.listWatches).mockResolvedValue({ items: [saved] });
      return saved;
    });
    workspace();
    fireEvent.click(await screen.findByRole("button", { name: "New watch" }));
    fireEvent.change(screen.getByRole("combobox", { name: "Watch source" }), { target: { value: "project_files" } });
    fireEvent.change(screen.getByRole("textbox", { name: "Title" }), { target: { value: saved.title } });
    expect(screen.queryByRole("textbox", { name: "Public page URL" })).not.toBeInTheDocument();
    expect(screen.queryByRole("textbox", { name: "What changes matter?" })).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Preview next checks" }));
    await screen.findByText("Next three checks (UTC)");
    const save = screen.getByRole("button", { name: "Create watch" });
    expect(save).toBeDisabled();
    expect(screen.queryByRole("option", { name: "Private files" })).not.toBeInTheDocument();
    fireEvent.change(screen.getByRole("combobox", { name: "Project" }), { target: { value: selected.id } });
    fireEvent.click(save);
    expect(await screen.findByRole("heading", { name: "Watched Project files" })).toBeVisible();
    expect(create).toHaveBeenCalledExactlyOnceWith({
      title: saved.title, schedule: saved.schedule, enabled: true, source_kind: "project_files", project_id: selected.id,
    }, expect.any(String));
  });

  it("loads the selected Project's latest ready text filenames and saves the exact name", async () => {
    const first = project();
    const second = project({ id: "project-2", name: "Release drafts" });
    vi.mocked(api.listProjects).mockResolvedValue({
      items: [first, second, project({ id: "private", name: "Private files", private_required: true })],
      next_cursor: null, limits: projectLimits,
    });
    const filename = "Release notes.MD";
    const file = { id: "notes", filename, kind: "text", mime: "text/markdown", bytes: 32, status: "ready",
      uploaded_at: "2026-10-30T14:00:00Z", added_at: "2026-10-30T14:00:00Z" } satisfies AudreyProjectFile;
    let finishFirst!: (page: AudreyProjectFileList) => void;
    const load = vi.spyOn(api, "listProjectFiles").mockImplementation((id) => id === first.id
      ? new Promise((resolve) => { finishFirst = resolve; })
      : Promise.resolve({ items: [
        file, { ...file, id: "previous" },
        { ...file, id: "pending", filename: "inventory.csv", status: "pending" },
        { ...file, id: "old inventory", filename: "inventory.csv" },
        { ...file, id: "unsupported", filename: "manual.pdf" },
        { ...file, id: "large", filename: "large.txt", bytes: 64_001 },
      ], limits: projectLimits }));
    const saved = watch({ title: "Release notes text", source_kind: "project_files", source_filename: filename, url: null,
      project: { id: second.id, name: second.name, available: true }, latest_run: null });
    const create = vi.spyOn(api, "createWatch").mockImplementation(async () => {
      vi.mocked(api.listWatches).mockResolvedValue({ items: [saved] });
      return saved;
    });

    workspace();
    fireEvent.click(await screen.findByRole("button", { name: "New watch" }));
    fireEvent.change(screen.getByRole("combobox", { name: "Watch source" }), { target: { value: "project_file_text" } });
    expect(load).not.toHaveBeenCalled();
    const projectChoice = screen.getByRole("combobox", { name: "Project" });
    await screen.findByRole("option", { name: first.name });
    expect(screen.queryByRole("option", { name: "Private files" })).not.toBeInTheDocument();
    fireEvent.change(projectChoice, { target: { value: first.id } });
    await waitFor(() => expect(load).toHaveBeenCalledWith(first.id, expect.any(AbortSignal)));
    const firstSignal = load.mock.calls[0][1];
    fireEvent.change(projectChoice, { target: { value: second.id } });
    expect(firstSignal?.aborted).toBe(true);
    await screen.findByRole("option", { name: filename });
    await act(async () => finishFirst({ items: [{ ...file, filename: "other Project.txt" }], limits: projectLimits }));
    const choice = screen.getByRole("combobox", { name: "Filename" });
    expect(within(choice).getAllByRole("option").map(({ textContent }) => textContent)).toEqual(["Choose a file", filename]);
    expect(screen.queryByRole("textbox", { name: "Public page URL" })).not.toBeInTheDocument();
    expect(screen.queryByRole("textbox", { name: "What changes matter?" })).not.toBeInTheDocument();
    fireEvent.change(screen.getByRole("textbox", { name: "Title" }), { target: { value: saved.title } });
    fireEvent.click(screen.getByRole("button", { name: "Preview next checks" }));
    await screen.findByText("Next three checks (UTC)");
    const save = screen.getByRole("button", { name: "Create watch" });
    expect(save).toBeDisabled();
    fireEvent.change(choice, { target: { value: filename } });
    fireEvent.click(save);
    expect(await screen.findByRole("heading", { name: "Watched Project file" })).toBeVisible();
    expect(create).toHaveBeenCalledExactlyOnceWith({
      title: saved.title, schedule: saved.schedule, enabled: true, source_kind: "project_files", project_id: second.id, source_filename: filename,
    }, expect.any(String));
  });

  it.each([undefined, "Release notes.MD"])("shows file changes and keeps a saved watch bound to its Project and source (%s)", async (sourceFilename) => {
    const selected = { id: "project-1", name: "Release research", available: true };
    const saved = watch({ source_kind: "project_files", source_filename: sourceFilename, url: null, project: selected });
    const report = sourceFilename ? "The saved file text changed." : "1 attachment added, 1 removed.";
    const changed = savedRun({ outcome: "changed", significant: true, report, definition: {
      title: saved.title, source_kind: "project_files", source_filename: sourceFilename, url: null, criterion: saved.criterion,
      schedule: saved.schedule, project: selected,
    } });
    vi.mocked(api.listWatches).mockResolvedValue({ items: [{ ...saved, latest_run: changed }] });
    vi.mocked(api.listWatchRuns).mockResolvedValue({ items: [changed] });
    vi.mocked(api.getWatchRun).mockResolvedValue(changed);
    const openProject = vi.fn();
    workspace({ onOpenProject: openProject });
    expect(await screen.findByRole("heading", { name: sourceFilename ? "File text changed" : "Files changed" })).toBeVisible();
    expect(screen.getByText(report)).toBeVisible();
    expect(screen.queryByText("Saved page text")).not.toBeInTheDocument();
    expect(screen.queryByText("Local model assessment")).not.toBeInTheDocument();
    expect(screen.queryByRole("link")).not.toBeInTheDocument();
    const settings = within(screen.getByRole("region", { name: "Watch settings" }));
    if (sourceFilename) {
      expect(settings.getByText(sourceFilename)).toBeVisible();
      const inputs = screen.getByText("Saved check inputs", { selector: "summary" });
      fireEvent.click(inputs);
      expect(within(inputs.parentElement!).getByText(sourceFilename)).toBeVisible();
    }
    fireEvent.click(settings.getByRole("button", { name: "Open Project" }));
    expect(openProject).toHaveBeenCalledWith(selected.id);
    fireEvent.click(screen.getByRole("button", { name: "Edit" }));
    const source = screen.getByRole("combobox", { name: "Watch source" });
    expect(source).toHaveValue(sourceFilename ? "project_file_text" : "project_files");
    expect(source).toBeDisabled();
    const projectChoice = screen.getByRole("combobox", { name: "Project" });
    expect(projectChoice).toHaveValue(selected.id);
    expect(projectChoice).toBeDisabled();
    if (sourceFilename) {
      const filename = screen.getByRole("combobox", { name: "Filename" });
      expect(filename).toHaveValue(sourceFilename);
      expect(filename).toBeDisabled();
    }
  });

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
