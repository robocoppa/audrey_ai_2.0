import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import * as api from "./api";
import { WatchForm } from "./WatchForm";
import { WatchesWorkspace } from "./WatchesWorkspace";

const limits = { max_name_chars: 100, max_instructions_chars: 10000, max_files: 100 };
const privateProject: api.AudreyProject = { id: "private-project", name: "Private accounts", private_required: true, instructions: "Private instructions", created_at: "2026-11-27T10:00:00Z", updated_at: "2026-11-27T10:00:00Z" };
const ordinaryProject: api.AudreyProject = { ...privateProject, id: "ordinary", name: "Ordinary Project", private_required: false };
const filename = "Budget notes.MD";
const file: api.AudreyProjectFile = { id: "private-file", filename, mime: "text/markdown", kind: "text", status: "ready", bytes: 40, uploaded_at: privateProject.created_at, added_at: privateProject.created_at };
const schedule: api.WatchSchedule = { kind: "daily", time: "09:00", timezone: "UTC" };
const saved: api.PublicWatch = { id: "private-watch", title: "Private budget check", source_kind: "project_files", source_filename: filename, private_required: true, url: null, criterion: "Notify me when the saved text in this Project file changes.", schedule, enabled: true, revision: 1, project: { id: privateProject.id, name: privateProject.name, available: true, private_required: true }, next_run_at: "2026-11-28T09:00:00Z", active_run_id: null, latest_run: null, created_at: privateProject.created_at, updated_at: privateProject.updated_at };

beforeEach(() => {
  vi.spyOn(api, "listProjects").mockResolvedValue({ items: [ordinaryProject, privateProject], next_cursor: null, limits });
  vi.spyOn(api, "listProjectFiles").mockResolvedValue({ items: [file], limits });
  vi.spyOn(api, "previewWatchSchedule").mockResolvedValue({ occurrences: ["2026-11-28T09:00:00Z", "2026-11-29T09:00:00Z", "2026-11-30T09:00:00Z"] });
});
afterEach(() => { cleanup(); vi.restoreAllMocks(); });

it("prefills an exact Private filename watch and sends only its explicitly chosen scope", async () => {
  const save = vi.fn().mockResolvedValue(undefined);
  render(<WatchForm timezone="UTC" busy={false} projectRequest={{ project: privateProject, serial: 1 }} onSave={save} onCancel={vi.fn()} />);
  await waitFor(() => expect(screen.getByRole("combobox", { name: "Watch source" })).toHaveValue("project_file_text"));
  expect(screen.getByRole("combobox", { name: "Project" })).toHaveValue(privateProject.id);
  expect(screen.getByRole("option", { name: "Private accounts · Private" })).toBeInTheDocument();
  fireEvent.change(screen.getByRole("combobox", { name: "Filename" }), { target: { value: (await screen.findByRole("option", { name: filename })).getAttribute("value") } });
  fireEvent.change(screen.getByRole("textbox", { name: "Title" }), { target: { value: saved.title } });
  expect(screen.getByText(/Checks run locally without a model, web access/)).toBeVisible();
  expect(screen.queryByRole("textbox", { name: "Public page URL" })).not.toBeInTheDocument();
  expect(screen.queryByRole("textbox", { name: "What changes matter?" })).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "Preview next checks" }));
  await screen.findByText("Next three checks (UTC)");
  fireEvent.click(screen.getByRole("button", { name: "Create watch" }));
  expect(save).toHaveBeenCalledExactlyOnceWith({ title: saved.title, schedule, enabled: true, source_kind: "project_files", project_id: privateProject.id, source_filename: filename, private_required: true }, undefined);
});

it.each(["public_page", "project_files"])("clears Private scope when changing to %s and requires a new eligible selection", async (nextSource) => {
  const save = vi.fn();
  render(<WatchForm timezone="UTC" busy={false} projectRequest={{ project: privateProject, serial: 1 }} onSave={save} onCancel={vi.fn()} />);
  await screen.findByRole("option", { name: filename });
  fireEvent.change(screen.getByRole("combobox", { name: "Filename" }), { target: { value: filename } });
  fireEvent.change(screen.getByRole("combobox", { name: "Watch source" }), { target: { value: nextSource } });
  expect(screen.getByRole("combobox", { name: "Project" })).toHaveValue("");
  expect(screen.queryByRole("option", { name: /Private accounts/ })).not.toBeInTheDocument();
  expect(screen.queryByRole("combobox", { name: "Filename" })).not.toBeInTheDocument();
  fireEvent.change(screen.getByRole("textbox", { name: "Title" }), { target: { value: "A fresh watch" } });
  fireEvent.click(screen.getByRole("button", { name: "Preview next checks" }));
  await screen.findByText("Next three checks (UTC)");
  if (nextSource === "public_page") {
    fireEvent.change(screen.getByRole("textbox", { name: "Public page URL" }), { target: { value: "example.com/releases" } });
    fireEvent.change(screen.getByRole("textbox", { name: "What changes matter?" }), { target: { value: "Stable releases" } });
    fireEvent.click(screen.getByRole("button", { name: "Create watch" }));
    expect(save).toHaveBeenCalledExactlyOnceWith({ title: "A fresh watch", schedule, enabled: true, url: "https://example.com/releases", criterion: "Stable releases" }, undefined);
  } else {
    expect(screen.getByRole("button", { name: "Create watch" })).toBeDisabled();
    expect(save).not.toHaveBeenCalled();
  }
});

it("permits Private Projects from later pages only for exact-name file text", async () => {
  vi.mocked(api.listProjects).mockResolvedValueOnce({ items: [ordinaryProject], next_cursor: "more", limits }).mockResolvedValueOnce({ items: [privateProject], next_cursor: null, limits });
  render(<WatchForm timezone="UTC" busy={false} onSave={vi.fn()} onCancel={vi.fn()} />);
  fireEvent.click(await screen.findByRole("button", { name: "Load more Projects" }));
  await waitFor(() => expect(api.listProjects).toHaveBeenCalledTimes(2));
  expect(screen.queryByRole("option", { name: /Private accounts/ })).not.toBeInTheDocument();
  fireEvent.change(screen.getByRole("combobox", { name: "Watch source" }), { target: { value: "project_files" } });
  expect(screen.queryByRole("option", { name: /Private accounts/ })).not.toBeInTheDocument();
  fireEvent.change(screen.getByRole("combobox", { name: "Watch source" }), { target: { value: "project_file_text" } });
  expect(await screen.findByRole("option", { name: "Private accounts · Private" })).toBeInTheDocument();
});

it("shows owned Private history without public links, content snapshots or filename changes on edit", async () => {
  const run: api.WatchRun = { id: "private-run", watch_id: saved.id, watch_revision: 1, status: "completed", outcome: "changed", kind: "manual", scheduled_at: saved.created_at, created_at: saved.created_at, updated_at: saved.updated_at, completed_at: saved.updated_at, attempts: 1, report: "The saved file text changed.", diff: "Do not render file contents.", significant: true, error: "", definition: { title: saved.title, source_kind: "project_files", source_filename: filename, private_required: true, url: "https://example.com/not-a-file-source", criterion: saved.criterion, schedule, project: saved.project } };
  vi.spyOn(api, "listWatches").mockResolvedValue({ items: [{ ...saved, latest_run: run }] });
  vi.spyOn(api, "listWatchRuns").mockResolvedValue({ items: [run] });
  vi.spyOn(api, "getWatchRun").mockResolvedValue(run);
  const snapshot = vi.spyOn(api, "getWatchRunSnapshot"), search = vi.spyOn(api, "searchWatchRunSnapshot"), open = vi.fn();
  render(<WatchesWorkspace visible timezone="UTC" onOpenProject={open} />);
  expect(await screen.findByRole("heading", { name: "File text changed" })).toBeVisible();
  expect(screen.getByText(run.report)).toBeVisible();
  expect(screen.getAllByText("Scheduled · Private")).toHaveLength(2);
  expect(screen.queryByRole("link")).not.toBeInTheDocument();
  expect(screen.queryByText("Saved page text")).not.toBeInTheDocument();
  expect(screen.queryByText(run.diff)).not.toBeInTheDocument();
  fireEvent.click(screen.getByText("Saved check inputs", { selector: "summary" }));
  expect(screen.getByText("Private · Project file text")).toBeVisible();
  const settings = within(screen.getByRole("region", { name: "Watch settings" }));
  fireEvent.click(settings.getByRole("button", { name: "Open Project" }));
  expect(open).toHaveBeenCalledWith(privateProject.id);
  fireEvent.click(screen.getByRole("button", { name: "Edit" }));
  expect(screen.getByRole("combobox", { name: "Watch source" })).toBeDisabled();
  expect(screen.getByRole("combobox", { name: "Project" })).toBeDisabled();
  expect(screen.getByRole("combobox", { name: "Filename" })).toBeDisabled();
  expect(screen.getByRole("combobox", { name: "Filename" })).toHaveValue(filename);
  expect(api.listProjectFiles).not.toHaveBeenCalled();
  expect(snapshot).not.toHaveBeenCalled(); expect(search).not.toHaveBeenCalled();
});
