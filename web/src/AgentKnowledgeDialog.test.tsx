import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";

import { AgentKnowledgeDialog } from "./AgentKnowledgeDialog";
import { type AgentTask, type AudreyProject, type KnowledgeEntry } from "./api";

const task: AgentTask = { id: "task-1", objective: "Research dates", status: "completed", revision: 8, created_at: "2026-10-29T10:00:00Z", updated_at: "2026-10-29T11:00:00Z", steps: [], result: "A large completed report that must not be copied automatically.", error: "", attempts: 1 };
const project: AudreyProject = { id: "project-1", name: "History", instructions: "", created_at: "2026-10-29T10:00:00Z", updated_at: "2026-10-29T10:00:00Z" };
const saved: KnowledgeEntry = { id: "knowledge-1", project_id: project.id, kind: "decision", text: "Existing approved decision.", approval_state: "approved", work_status: "open", revision: 3, pinned: false, origin: "user", sources: [], source_task_id: null, created_at: "2026-10-29T10:00:00Z", updated_at: "2026-10-29T10:00:00Z", approved_at: "2026-10-29T10:00:00Z", stale: false, source_warnings: [], replaces_id: null, base_revision: null };
function json(body: unknown, status = 200) { return new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } }); }
function projects(items = [project]) { return { items, next_cursor: null, limits: {} }; }
function entries(items: KnowledgeEntry[] = []) { return { items, next_cursor: null, limits: { max_entries: 200, max_text_chars: 8000, max_sources: 8 } }; }
const suggestion = { kind: "question", text: "Which date is supported by the inspected source?" };
function drafts(suggestions = [suggestion]) { return { task_id: task.id, task_revision: task.revision, suggestions }; }
function mockDialogFetch(generate: () => Promise<Response>) {
  const fetchMock = vi.fn().mockImplementation((path: string, init?: RequestInit) => {
    if (path.endsWith("/knowledge-drafts")) return generate();
    if (path.endsWith("/proposals") && init?.method === "POST") return Promise.resolve(json({ ...saved, approval_state: "proposed" }));
    if (new URL(path, "http://test").pathname === "/api/projects") return Promise.resolve(json(projects()));
    return Promise.resolve(json(entries([saved])));
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}
afterEach(() => { cleanup(); vi.restoreAllMocks(); vi.unstubAllGlobals(); });

async function chooseProject() {
  await waitFor(() => expect(screen.getByRole("combobox", { name: "Project" })).toBeEnabled());
  fireEvent.change(screen.getByRole("combobox", { name: "Project" }), { target: { value: project.id } });
}
async function chooseReplacement() {
  const option = await screen.findByRole("option", { name: "decision · Existing approved decision." });
  fireEvent.change(option.closest("select")!, { target: { value: saved.id } });
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

it("generates only on an explicit request and uses a suggestion without saving or selecting a Project", async () => {
  let finish!: (response: Response) => void;
  const fetchMock = mockDialogFetch(() => new Promise<Response>((resolve) => { finish = resolve; }));
  render(<AgentKnowledgeDialog task={task} onClose={vi.fn()} />);
  await screen.findByRole("option", { name: project.name });
  expect(fetchMock.mock.calls.filter(([path]) => path.endsWith("/knowledge-drafts"))).toHaveLength(0);
  fireEvent.click(screen.getByRole("button", { name: "Generate suggestions" }));
  expect(screen.getByRole("status")).toHaveTextContent("Generating suggestions…");
  expect(screen.getByRole("button", { name: "Generate suggestions" })).toBeDisabled();
  expect(screen.getByRole("button", { name: "Close knowledge proposal" })).toBeEnabled();
  finish(json(drafts()));
  fireEvent.click(await screen.findByRole("button", { name: "Use suggestion" }));
  expect(screen.getByRole("textbox", { name: "Proposed text" })).toHaveValue(suggestion.text);
  expect(screen.getByRole("combobox", { name: "Entry type" })).toHaveValue("question");
  expect(screen.getByRole("combobox", { name: "Project" })).toHaveValue("");
  expect(screen.getByRole("button", { name: "Save proposal" })).toBeDisabled();
  expect(screen.getByText("Generated suggestions · Review before saving")).toBeVisible();
  const writes = fetchMock.mock.calls.filter(([, init]) => init?.method === "POST");
  expect(writes).toHaveLength(1);
  expect(JSON.parse(writes[0][1]!.body as string)).toEqual({ expected_revision: task.revision });
});

it("requires Replace draft and preserves Project, pin and replacement choices before explicitly saving pending knowledge", async () => {
  const fetchMock = mockDialogFetch(async () => json(drafts()));
  render(<AgentKnowledgeDialog task={task} onClose={vi.fn()} />);
  await chooseProject();
  await chooseReplacement();
  fireEvent.click(screen.getByRole("checkbox", { name: "Pin after approval" }));
  fireEvent.change(screen.getByRole("textbox", { name: "Proposed text" }), { target: { value: "Keep my existing draft." } });
  fireEvent.click(screen.getByRole("button", { name: "Generate suggestions" }));
  fireEvent.click(await screen.findByRole("button", { name: "Use suggestion" }));
  expect(screen.getByRole("textbox", { name: "Proposed text" })).toHaveValue("Keep my existing draft.");
  expect(screen.getByLabelText("Review draft replacement")).toContainElement(screen.getByRole("button", { name: "Replace draft" }));
  expect(screen.getByRole("button", { name: "Replace draft" })).toHaveFocus();
  fireEvent.click(screen.getByRole("button", { name: "Keep my draft" }));
  expect(screen.queryByRole("button", { name: "Replace draft" })).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "Use suggestion" }));
  fireEvent.click(screen.getByRole("button", { name: "Replace draft" }));
  expect(screen.getByRole("textbox", { name: "Proposed text" })).toHaveValue(suggestion.text);
  expect(screen.getByRole("checkbox", { name: "Pin after approval" })).toBeChecked();
  expect(screen.getByRole("combobox", { name: "Project" })).toHaveValue(project.id);
  expect(screen.getByRole("combobox", { name: "Replace an approved entry Optional" })).toHaveValue(saved.id);
  fireEvent.change(screen.getByRole("textbox", { name: "Proposed text" }), { target: { value: "My reviewed version of the suggestion." } });
  expect(fetchMock.mock.calls.filter(([path]) => path.endsWith("/proposals"))).toHaveLength(0);
  fireEvent.click(screen.getByRole("button", { name: "Save proposal" }));
  expect(await screen.findByRole("status")).toHaveTextContent("Proposal saved. Approve it in Project knowledge before reuse.");
  const writes = fetchMock.mock.calls.filter(([, init]) => init?.method === "POST");
  expect(writes).toHaveLength(2);
  expect(JSON.parse(writes[1][1]!.body as string)).toEqual({ task_id: task.id, task_revision: task.revision, kind: "question", text: "My reviewed version of the suggestion.", pinned: true, replaces_id: saved.id, base_revision: saved.revision, request_id: expect.any(String) });
});

it("cancels generation, ignores late output and retries only when requested", async () => {
  let finish!: (response: Response) => void;
  let calls = 0;
  const fetchMock = mockDialogFetch(() => ++calls === 1 ? new Promise<Response>((resolve) => { finish = resolve; }) : Promise.resolve(json(drafts())));
  render(<AgentKnowledgeDialog task={task} onClose={vi.fn()} />);
  fireEvent.change(screen.getByRole("textbox", { name: "Proposed text" }), { target: { value: "Keep this manual draft." } });
  fireEvent.click(screen.getByRole("button", { name: "Generate suggestions" }));
  const signal = fetchMock.mock.calls.find(([path]) => path.endsWith("/knowledge-drafts"))![1]!.signal as AbortSignal;
  fireEvent.click(screen.getByRole("button", { name: "Cancel generation" }));
  expect(signal.aborted).toBe(true);
  expect(screen.getByRole("textbox", { name: "Proposed text" })).toHaveValue("Keep this manual draft.");
  await act(async () => { finish(json(drafts([{ kind: "fact", text: "Discard late output" }]))); });
  expect(screen.queryByText("Discard late output")).not.toBeInTheDocument();
  expect(calls).toBe(1);
  fireEvent.click(screen.getByRole("button", { name: "Generate suggestions" }));
  await screen.findByRole("button", { name: "Use suggestion" });
  expect(calls).toBe(2);
  expect(screen.getByRole("textbox", { name: "Proposed text" })).toHaveValue("Keep this manual draft.");
});

it.each(["button", "Escape", "backdrop"])("aborts generation when closing with %s", async (method) => {
  let finish!: (response: Response) => void;
  const fetchMock = mockDialogFetch(() => new Promise<Response>((resolve) => { finish = resolve; }));
  const onClose = vi.fn();
  render(<AgentKnowledgeDialog task={task} onClose={onClose} />);
  fireEvent.click(screen.getByRole("button", { name: "Generate suggestions" }));
  const signal = fetchMock.mock.calls.find(([path]) => path.endsWith("/knowledge-drafts"))![1]!.signal as AbortSignal;
  if (method === "button") fireEvent.click(screen.getByRole("button", { name: "Close knowledge proposal" }));
  else if (method === "Escape") fireEvent.keyDown(document, { key: "Escape" });
  else fireEvent.mouseDown(document.querySelector(".agent-knowledge-backdrop")!);
  expect(onClose).toHaveBeenCalledTimes(1);
  expect(signal.aborted).toBe(true);
  await act(async () => { finish(json(drafts())); });
  expect(screen.queryByRole("button", { name: "Use suggestion" })).not.toBeInTheDocument();
});

it.each([404, 409, 503])("keeps manual choices and clears old suggestions after a generation HTTP %i", async (status) => {
  let calls = 0;
  mockDialogFetch(async () => ++calls === 1 ? json(drafts()) : json({ detail: "Generation could not finish." }, status));
  render(<AgentKnowledgeDialog task={task} onClose={vi.fn()} />);
  await chooseProject();
  await chooseReplacement();
  fireEvent.change(screen.getByRole("textbox", { name: "Proposed text" }), { target: { value: "Retain this choice." } });
  fireEvent.click(screen.getByRole("button", { name: "Generate suggestions" }));
  await screen.findByRole("button", { name: "Use suggestion" });
  fireEvent.click(screen.getByRole("button", { name: "Generate suggestions" }));
  await screen.findByRole("alert");
  expect(screen.queryByRole("button", { name: "Use suggestion" })).not.toBeInTheDocument();
  expect(screen.getByRole("textbox", { name: "Proposed text" })).toHaveValue("Retain this choice.");
  expect(screen.getByRole("combobox", { name: "Project" })).toHaveValue(project.id);
  expect(screen.getByRole("combobox", { name: "Replace an approved entry Optional" })).toHaveValue(saved.id);
  expect(calls).toBe(2);
});

it("accepts an empty generation and renders untrusted suggestions as plain editable text", async () => {
  const malicious = '<img src=x onerror="window.executed=true">';
  let calls = 0;
  mockDialogFetch(async () => json(drafts(++calls === 1 ? [] : [{ kind: "fact", text: malicious }])));
  render(<AgentKnowledgeDialog task={task} onClose={vi.fn()} />);
  fireEvent.click(screen.getByRole("button", { name: "Generate suggestions" }));
  await screen.findByText("No suggestions were found. You can write a proposal or generate again.");
  expect(screen.queryByRole("button", { name: "Use suggestion" })).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "Generate suggestions" }));
  expect(await screen.findByText(malicious)).toBeVisible();
  expect(document.querySelector(".agent-knowledge-suggestions img")).toBeNull();
  fireEvent.click(screen.getByRole("button", { name: "Use suggestion" }));
  expect(screen.getByRole("textbox", { name: "Proposed text" })).toHaveValue(malicious);
});

it("discards mismatched and malformed generated suggestions without replacing a manual draft", async () => {
  let calls = 0;
  mockDialogFetch(async () => json(++calls === 1 ? { ...drafts(), task_revision: 99 } : drafts([{ kind: "fact", text: "x".repeat(1201) }])));
  render(<AgentKnowledgeDialog task={task} onClose={vi.fn()} />);
  fireEvent.change(screen.getByRole("textbox", { name: "Proposed text" }), { target: { value: "My own statement." } });
  fireEvent.click(screen.getByRole("button", { name: "Generate suggestions" }));
  expect(await screen.findByRole("alert")).toHaveTextContent("The saved report changed.");
  expect(screen.queryByRole("button", { name: "Use suggestion" })).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "Generate suggestions" }));
  await waitFor(() => expect(screen.getByRole("alert")).toHaveTextContent("The generated suggestions could not be read."));
  expect(screen.getByRole("textbox", { name: "Proposed text" })).toHaveValue("My own statement.");
  expect(screen.queryByRole("button", { name: "Use suggestion" })).not.toBeInTheDocument();
});

it("aborts late generation on report revision changes and unmount", async () => {
  let finish!: (response: Response) => void;
  const fetchMock = mockDialogFetch(() => new Promise<Response>((resolve) => { finish = resolve; }));
  const view = render(<AgentKnowledgeDialog task={task} onClose={vi.fn()} />);
  fireEvent.click(screen.getByRole("button", { name: "Generate suggestions" }));
  const signal = fetchMock.mock.calls.find(([path]) => path.endsWith("/knowledge-drafts"))![1]!.signal as AbortSignal;
  view.rerender(<AgentKnowledgeDialog task={{ ...task, revision: 9 }} onClose={vi.fn()} />);
  expect(signal.aborted).toBe(true);
  await act(async () => { finish(json(drafts())); });
  expect(screen.queryByRole("button", { name: "Use suggestion" })).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "Generate suggestions" }));
  const secondSignal = fetchMock.mock.calls.filter(([path]) => path.endsWith("/knowledge-drafts"))[1][1]!.signal as AbortSignal;
  view.unmount();
  expect(secondSignal.aborted).toBe(true);
  await act(async () => { finish(json({ ...drafts(), task_revision: 9 })); });
});

it("aborts an unused generation when the user saves their manual pending proposal", async () => {
  let finish!: (response: Response) => void;
  const fetchMock = mockDialogFetch(() => new Promise<Response>((resolve) => { finish = resolve; }));
  render(<AgentKnowledgeDialog task={task} onClose={vi.fn()} />);
  await chooseProject();
  fireEvent.change(screen.getByRole("textbox", { name: "Proposed text" }), { target: { value: "Save my manual statement." } });
  fireEvent.click(screen.getByRole("button", { name: "Generate suggestions" }));
  const signal = fetchMock.mock.calls.find(([path]) => path.endsWith("/knowledge-drafts"))![1]!.signal as AbortSignal;
  fireEvent.click(screen.getByRole("button", { name: "Save proposal" }));
  await screen.findByText("Proposal saved. Approve it in Project knowledge before reuse.");
  expect(signal.aborted).toBe(true);
  await act(async () => { finish(json(drafts())); });
  expect(screen.queryByRole("button", { name: "Use suggestion" })).not.toBeInTheDocument();
});
