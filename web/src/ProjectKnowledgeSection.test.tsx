import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { ProjectKnowledgeSection } from "./ProjectKnowledgeSection";
import * as api from "./api";

const project: api.AudreyProject = { id: "project-1", name: "Garden", instructions: "", created_at: "2026-11-04T12:00:00Z", updated_at: "2026-11-04T12:00:00Z" };
const saved: api.ProjectKnowledgeText = { text: "Budget: $472.35\nDelivery: Wednesday", revision: "version-1", max_text_chars: 8000 };

beforeEach(() => { vi.spyOn(api, "getProjectKnowledgeText").mockResolvedValue(saved); });
afterEach(() => { cleanup(); vi.restoreAllMocks(); vi.unstubAllGlobals(); });

async function loadedText() {
  const input = screen.getByRole("textbox", { name: "Project knowledge" });
  await waitFor(() => expect(input).toHaveValue(saved.text));
  return input;
}

it("shows existing knowledge in one editable textbox without entry controls", async () => {
  const write = vi.spyOn(api, "saveProjectKnowledgeText");
  render(<ProjectKnowledgeSection project={project} />);
  expect(await loadedText()).toBeEnabled();
  expect(screen.getAllByRole("textbox")).toHaveLength(1);
  expect(screen.queryByRole("combobox")).not.toBeInTheDocument();
  expect(screen.queryByRole("checkbox")).not.toBeInTheDocument();
  expect(screen.queryByRole("button", { name: /Add entry|Approve|Assign/ })).not.toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Save" })).toBeDisabled();
  expect(write).not.toHaveBeenCalled();
});

it.each(["Updated budget: $500", ""])("explicitly saves edits, including clearing the textbox (%j)", async (text) => {
  const write = vi.spyOn(api, "saveProjectKnowledgeText").mockResolvedValue({ ...saved, text, revision: "version-2" });
  const changed = vi.fn();
  render(<ProjectKnowledgeSection project={project} onChanged={changed} />);
  const input = await loadedText();
  fireEvent.change(input, { target: { value: text } });
  expect(write).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole("button", { name: "Save" }));
  await waitFor(() => expect(write).toHaveBeenCalledExactlyOnceWith(project.id, text, saved.revision, expect.any(AbortSignal)));
  await waitFor(() => expect(screen.getByRole("status")).toHaveTextContent("Saved"));
  expect(changed).toHaveBeenCalledOnce();
  expect(input).toHaveValue(text);
  expect(screen.getByRole("button", { name: "Save" })).toBeDisabled();
});

it("keeps drafts after a failed save and retries without duplicate pending writes", async () => {
  let fail!: (reason: Error) => void;
  const write = vi.spyOn(api, "saveProjectKnowledgeText").mockImplementationOnce(() => new Promise((_, reject) => { fail = reject; })).mockResolvedValue({ ...saved, text: "My draft", revision: "version-2" });
  render(<ProjectKnowledgeSection project={project} />);
  const input = await loadedText();
  fireEvent.change(input, { target: { value: "My draft" } });
  fireEvent.submit(input.closest("form")!); fireEvent.submit(input.closest("form")!);
  expect(write).toHaveBeenCalledOnce();
  await act(async () => fail(new api.ApiError(503, "Temporarily unavailable")));
  expect(screen.getByRole("alert")).toHaveTextContent("Temporarily unavailable");
  expect(input).toHaveValue("My draft");
  fireEvent.click(screen.getByRole("button", { name: "Save" }));
  await waitFor(() => expect(screen.getByRole("status")).toHaveTextContent("Saved"));
  expect(write).toHaveBeenCalledTimes(2);
});

it("preserves local edits on conflict and requires an explicit choice before overwriting", async () => {
  const latest = { ...saved, text: "Changed in another tab", revision: "version-2" };
  vi.mocked(api.getProjectKnowledgeText).mockResolvedValueOnce(saved).mockResolvedValue(latest);
  const write = vi.spyOn(api, "saveProjectKnowledgeText").mockRejectedValueOnce(new api.ApiError(409, "Changed")).mockResolvedValue({ ...latest, text: "My draft", revision: "version-3" });
  render(<ProjectKnowledgeSection project={project} />);
  const input = await loadedText();
  fireEvent.change(input, { target: { value: "My draft" } });
  fireEvent.click(screen.getByRole("button", { name: "Save" }));
  expect(await screen.findByText(latest.text)).toBeVisible();
  expect(input).toHaveValue("My draft");
  expect(screen.getByRole("button", { name: "Save" })).toBeDisabled();
  fireEvent.click(screen.getByRole("button", { name: "Keep my edits" }));
  fireEvent.click(screen.getByRole("button", { name: "Save" }));
  await waitFor(() => expect(write).toHaveBeenLastCalledWith(project.id, "My draft", latest.revision, expect.any(AbortSignal)));
  await waitFor(() => expect(screen.getByRole("status")).toHaveTextContent("Saved"));
});

it("revalidates on reopening and can adopt the latest saved text without overwriting it", async () => {
  const latest = { ...saved, text: "Latest knowledge", revision: "version-2" };
  vi.mocked(api.getProjectKnowledgeText).mockResolvedValueOnce(saved).mockResolvedValue(latest);
  const write = vi.spyOn(api, "saveProjectKnowledgeText");
  const view = render(<ProjectKnowledgeSection project={project} />);
  const input = await loadedText();
  fireEvent.change(input, { target: { value: "Local draft" } });
  view.rerender(<ProjectKnowledgeSection project={project} visible={false} />);
  view.rerender(<ProjectKnowledgeSection project={project} visible />);
  fireEvent.click(await screen.findByRole("button", { name: "Use saved text" }));
  expect(input).toHaveValue(latest.text);
  expect(write).not.toHaveBeenCalled();
});

it("shows oversized legacy notes fully and permits saving after shortening", async () => {
  const text = "x".repeat(8500);
  vi.mocked(api.getProjectKnowledgeText).mockResolvedValue({ ...saved, text });
  vi.spyOn(api, "saveProjectKnowledgeText").mockResolvedValue({ ...saved, text: "Shorter", revision: "version-2" });
  render(<ProjectKnowledgeSection project={project} />);
  const input = await screen.findByDisplayValue(text);
  expect(input).toHaveValue(text);
  expect(screen.getByRole("alert")).toHaveTextContent("No text has been removed");
  fireEvent.change(input, { target: { value: "Shorter" } });
  expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Save" })).toBeEnabled();
});

it("retries a failed initial read without enabling an empty overwrite", async () => {
  vi.mocked(api.getProjectKnowledgeText).mockRejectedValueOnce(new api.ApiError(503, "Offline")).mockResolvedValue(saved);
  render(<ProjectKnowledgeSection project={project} />);
  expect(await screen.findByRole("alert")).toHaveTextContent("Offline");
  expect(screen.getByRole("textbox", { name: "Project knowledge" })).toBeDisabled();
  fireEvent.click(screen.getByRole("button", { name: "Reload saved text" }));
  expect(await loadedText()).toBeEnabled();
});

it("does not apply a late saved response to a different Project", async () => {
  let finish!: (value: api.ProjectKnowledgeText) => void;
  const changed = vi.fn();
  const write = vi.spyOn(api, "saveProjectKnowledgeText").mockImplementation(() => new Promise((resolve) => { finish = resolve; }));
  const view = render(<ProjectKnowledgeSection project={project} onChanged={changed} />);
  const input = await loadedText();
  fireEvent.change(input, { target: { value: "Old Project draft" } });
  fireEvent.click(screen.getByRole("button", { name: "Save" }));
  const signal = vi.mocked(write).mock.calls[0][3];
  vi.mocked(api.getProjectKnowledgeText).mockResolvedValue({ ...saved, text: "Other Project" });
  view.rerender(<ProjectKnowledgeSection project={{ ...project, id: "project-2" }} onChanged={changed} />);
  expect(await screen.findByDisplayValue("Other Project")).toBeVisible();
  expect(signal?.aborted).toBe(true);
  await act(async () => finish({ ...saved, text: "Old Project draft", revision: "version-2" }));
  expect(changed).not.toHaveBeenCalled();
  expect(screen.getByRole("textbox", { name: "Project knowledge" })).toHaveValue("Other Project");
});

it("uses owner-scoped text endpoints and passes the saved revision with each write", async () => {
  vi.restoreAllMocks();
  const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify(saved), { status: 200, headers: { "Content-Type": "application/json" } }));
  vi.stubGlobal("fetch", fetchMock);
  await api.getProjectKnowledgeText("project one");
  expect(fetchMock).toHaveBeenLastCalledWith("/api/projects/project%20one/knowledge/text", expect.objectContaining({ cache: "no-store" }));
  fetchMock.mockResolvedValue(new Response(JSON.stringify(saved), { status: 200, headers: { "Content-Type": "application/json" } }));
  await api.saveProjectKnowledgeText("project one", "New text", saved.revision);
  expect(fetchMock).toHaveBeenLastCalledWith("/api/projects/project%20one/knowledge/text", expect.objectContaining({ method: "PUT", body: JSON.stringify({ text: "New text", expected_revision: saved.revision }) }));
});
