import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";

import { AgentKnowledgeDialog } from "./AgentKnowledgeDialog";
import { type AgentTask, type AudreyProject, type KnowledgeEntry } from "./api";

const task: AgentTask = { id: "task-1", objective: "Research dates", status: "completed", revision: 8, created_at: "2026-10-29T10:00:00Z", updated_at: "2026-10-29T11:00:00Z", steps: [], result: "A large completed report that must not be copied automatically.", error: "", attempts: 1 };
const project: AudreyProject = { id: "project-1", name: "History", instructions: "", created_at: "2026-10-29T10:00:00Z", updated_at: "2026-10-29T10:00:00Z" };
const saved: KnowledgeEntry = { id: "knowledge-1", project_id: project.id, kind: "decision", text: "Existing approved decision.", approval_state: "approved", work_status: "open", revision: 3, pinned: false, origin: "user", sources: [], source_task_id: null, created_at: "2026-10-29T10:00:00Z", updated_at: "2026-10-29T10:00:00Z", approved_at: "2026-10-29T10:00:00Z", stale: false, source_warnings: [], replaces_id: null, base_revision: null };
function json(body: unknown, status = 200) { return new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } }); }
function projects(items = [project]) { return { items, next_cursor: null, limits: {} }; }
function entries(items: KnowledgeEntry[] = []) { return { items, next_cursor: null, limits: { max_entries: 200, max_text_chars: 8000, max_sources: 8 } }; }
afterEach(() => { cleanup(); vi.restoreAllMocks(); vi.unstubAllGlobals(); });

async function chooseProject() {
  await waitFor(() => expect(screen.getByRole("combobox", { name: "Project" })).toBeEnabled());
  fireEvent.change(screen.getByRole("combobox", { name: "Project" }), { target: { value: project.id } });
}

it("keeps report text out of the editor and saves an explicit pending proposal", async () => {
  const fetchMock = vi.fn().mockResolvedValueOnce(json(projects())).mockResolvedValueOnce(json(entries())).mockResolvedValueOnce(json({ ...saved, approval_state: "proposed" }));
  vi.stubGlobal("fetch", fetchMock);
  render(<AgentKnowledgeDialog task={task} onClose={vi.fn()} />);
  expect(screen.getByRole("dialog", { name: "Propose from this report" })).toBeVisible();
  expect(screen.getByRole("textbox", { name: "Proposed text" })).toHaveValue("");
  expect(screen.queryByText(task.result)).not.toBeInTheDocument();
  await chooseProject();
  await screen.findByRole("option", { name: "Save as a new entry" });
  fireEvent.change(screen.getByRole("combobox", { name: "Entry type" }), { target: { value: "action" } });
  fireEvent.change(screen.getByRole("textbox", { name: "Proposed text" }), { target: { value: "  Compare the dates.  " } });
  fireEvent.click(screen.getByRole("button", { name: "Save proposal" }));
  expect(await screen.findByRole("status")).toHaveTextContent("Proposal saved. Approve it in Project knowledge before reuse.");
  const mutation = fetchMock.mock.calls.find(([, init]) => init?.method === "POST")!;
  expect(mutation[0]).toBe("/api/projects/project-1/knowledge/proposals");
  expect(JSON.parse(mutation[1].body as string)).toEqual({ task_id: task.id, task_revision: 8, kind: "action", text: "Compare the dates.", pinned: false, request_id: expect.any(String) });
  expect(fetchMock.mock.calls.filter(([, init]) => init?.method === "POST")).toHaveLength(1);
});

it("offers approved replacement entries and pins the proposal to their current revision", async () => {
  const fetchMock = vi.fn().mockResolvedValueOnce(json(projects()))
    .mockResolvedValueOnce(json(entries([saved, { ...saved, id: "proposal-2", approval_state: "proposed", text: "Pending entry." }])))
    .mockResolvedValueOnce(json({ ...saved, approval_state: "proposed" }));
  vi.stubGlobal("fetch", fetchMock);
  render(<AgentKnowledgeDialog task={task} onClose={vi.fn()} />);
  await chooseProject();
  const option = await screen.findByRole("option", { name: "decision · Existing approved decision." });
  expect(screen.queryByRole("option", { name: /Pending entry/u })).not.toBeInTheDocument();
  fireEvent.change(option.closest("select")!, { target: { value: saved.id } });
  expect(screen.getByText(/Current revision 3/u)).toBeVisible();
  fireEvent.change(screen.getByRole("textbox", { name: "Proposed text" }), { target: { value: "A revised decision." } });
  fireEvent.click(screen.getByRole("checkbox", { name: "Pin after approval" }));
  fireEvent.click(screen.getByRole("button", { name: "Save proposal" }));
  await screen.findByRole("status");
  const payload = JSON.parse(fetchMock.mock.calls[2][1].body as string);
  expect(payload).toMatchObject({ replaces_id: saved.id, base_revision: 3, pinned: true });
});

it("reuses idempotency for a failed retry and changes it when the user changes text", async () => {
  const fetchMock = vi.fn().mockResolvedValueOnce(json(projects())).mockResolvedValueOnce(json(entries()))
    .mockResolvedValueOnce(json({ detail: "Temporary failure" }, 503))
    .mockResolvedValueOnce(json({ detail: "Temporary failure" }, 503))
    .mockResolvedValueOnce(json(saved));
  vi.stubGlobal("fetch", fetchMock);
  render(<AgentKnowledgeDialog task={task} onClose={vi.fn()} />);
  await chooseProject();
  await screen.findByRole("option", { name: "Save as a new entry" });
  fireEvent.change(screen.getByRole("textbox", { name: "Proposed text" }), { target: { value: "A chosen statement." } });
  fireEvent.click(screen.getByRole("button", { name: "Save proposal" }));
  await screen.findByRole("alert");
  fireEvent.click(screen.getByRole("button", { name: "Save proposal" }));
  await waitFor(() => expect(fetchMock.mock.calls.filter(([, init]) => init?.method === "POST")).toHaveLength(2));
  await screen.findByRole("button", { name: "Save proposal" });
  fireEvent.change(screen.getByRole("textbox", { name: "Proposed text" }), { target: { value: "A corrected statement." } });
  fireEvent.click(screen.getByRole("button", { name: "Save proposal" }));
  await screen.findByRole("status");
  const payloads = fetchMock.mock.calls.filter(([, init]) => init?.method === "POST").map(([, init]) => JSON.parse(init.body as string));
  expect(payloads[0].request_id).toBe(payloads[1].request_id);
  expect(payloads[2].request_id).not.toBe(payloads[1].request_id);
});

it("traps focus, closes on Escape and restores the opening control", async () => {
  vi.stubGlobal("fetch", vi.fn().mockResolvedValue(json(projects())));
  const opener = document.createElement("button");
  document.body.append(opener);
  opener.focus();
  const onClose = vi.fn();
  const view = render(<AgentKnowledgeDialog task={task} onClose={onClose} />);
  const close = screen.getByRole("button", { name: "Close knowledge proposal" });
  expect(close).toHaveFocus();
  fireEvent.keyDown(document, { key: "Tab", shiftKey: true });
  expect(screen.getByRole("button", { name: "Cancel" })).toHaveFocus();
  fireEvent.keyDown(document, { key: "Tab" });
  expect(close).toHaveFocus();
  fireEvent.keyDown(document, { key: "Escape" });
  expect(onClose).toHaveBeenCalledTimes(1);
  view.unmount();
  expect(opener).toHaveFocus();
  opener.remove();
});

it("guards closing and editing while a proposal write is pending", async () => {
  let finish!: (response: Response) => void;
  const fetchMock = vi.fn().mockResolvedValueOnce(json(projects())).mockResolvedValueOnce(json(entries()))
    .mockImplementationOnce(() => new Promise<Response>((resolve) => { finish = resolve; }));
  vi.stubGlobal("fetch", fetchMock);
  const onClose = vi.fn();
  render(<AgentKnowledgeDialog task={task} onClose={onClose} />);
  await chooseProject();
  await screen.findByRole("option", { name: "Save as a new entry" });
  fireEvent.change(screen.getByRole("textbox", { name: "Proposed text" }), { target: { value: "A statement." } });
  fireEvent.click(screen.getByRole("button", { name: "Save proposal" }));
  expect(screen.getByRole("button", { name: "Close knowledge proposal" })).toBeDisabled();
  expect(screen.getByRole("textbox", { name: "Proposed text" })).toBeDisabled();
  fireEvent.keyDown(document, { key: "Escape" });
  expect(onClose).not.toHaveBeenCalled();
  finish(json(saved));
  await screen.findByRole("status");
  fireEvent.click(screen.getByRole("button", { name: "Done" }));
  expect(onClose).toHaveBeenCalledTimes(1);
});

it("clears replacement selection when changing projects and discards late choices", async () => {
  let finish!: (response: Response) => void;
  const other = { ...project, id: "project-2", name: "Other" };
  const fetchMock = vi.fn().mockResolvedValueOnce(json(projects([project, other])))
    .mockImplementationOnce(() => new Promise<Response>((resolve) => { finish = resolve; }))
    .mockResolvedValueOnce(json(entries()));
  vi.stubGlobal("fetch", fetchMock);
  render(<AgentKnowledgeDialog task={task} onClose={vi.fn()} />);
  await chooseProject();
  fireEvent.change(screen.getByRole("combobox", { name: "Project" }), { target: { value: other.id } });
  await screen.findByRole("option", { name: "Save as a new entry" });
  finish(json(entries([saved])));
  await waitFor(() => expect(screen.queryByRole("option", { name: /Existing approved decision/u })).not.toBeInTheDocument());
});

it("retries a failed projects list without losing the draft", async () => {
  vi.stubGlobal("fetch", vi.fn().mockResolvedValueOnce(json({ detail: "Unavailable" }, 503)).mockResolvedValueOnce(json(projects())));
  render(<AgentKnowledgeDialog task={task} onClose={vi.fn()} />);
  fireEvent.change(screen.getByRole("textbox", { name: "Proposed text" }), { target: { value: "Keep this draft." } });
  await screen.findByRole("alert");
  fireEvent.click(screen.getByRole("button", { name: "Try again" }));
  await screen.findByRole("option", { name: "History" });
  expect(screen.getByRole("textbox", { name: "Proposed text" })).toHaveValue("Keep this draft.");
});
