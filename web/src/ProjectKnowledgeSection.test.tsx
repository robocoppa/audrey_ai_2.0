import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";

import { ProjectKnowledgeSection } from "./ProjectKnowledgeSection";
import { type AudreyProject, type KnowledgeEntry, type AgentTask } from "./api";

const project: AudreyProject = { id: "project-1", name: "Research", instructions: "", created_at: "2026-10-29T10:00:00Z", updated_at: "2026-10-29T10:00:00Z" };
function entry(patch: Partial<KnowledgeEntry> = {}): KnowledgeEntry {
  return { id: "entry-1", project_id: project.id, kind: "decision", text: "Use local models for agents.", approval_state: "approved", work_status: "open", revision: 1, pinned: false, origin: "user", sources: [], source_task_id: null, created_at: "2026-10-29T10:00:00Z", updated_at: "2026-10-29T10:00:00Z", approved_at: "2026-10-29T10:00:00Z", stale: false, source_warnings: [], replaces_id: null, base_revision: null, ...patch };
}
const limits = { max_entries: 200, max_text_chars: 8000, max_sources: 8 };
function json(body: unknown, status = 200) { return new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } }); }
function listing(items: KnowledgeEntry[]) { return { items, next_cursor: null, limits }; }
afterEach(() => { cleanup(); vi.restoreAllMocks(); vi.unstubAllGlobals(); });

const reference = { source_id: "file:owned:text", revision: "text-v1" };
function sourcePreview(revision = reference.revision) {
  return { ...reference, revision, kind: "file", title: "Project source", text: "The inspected source passage.", offset: 0, next_offset: null, total_chars: 29, target: { kind: "file", file_id: "owned", artifact: "text" }, availability: "available", private_required: false };
}

async function attachSource() {
  fireEvent.click(screen.getByRole("button", { name: "Source references (0/8)" }));
  fireEvent.click(await screen.findByRole("button", { name: "Preview Project source" }));
  fireEvent.click(await screen.findByRole("button", { name: "Attach reference" }));
}

it("offers agent assignment for current approved open actions and questions including saved file sources", async () => {
  const entries = [
    entry({ id: "action", kind: "action", text: "Open action" }),
    entry({ id: "question", kind: "question", text: "Open question", sources: [{ source_id: "agent:owned:report", revision: "2" }] }),
    entry({ id: "closed", kind: "action", text: "Closed action", work_status: "completed" }),
    entry({ id: "file", kind: "action", text: "File-backed action", sources: [reference] }),
    entry({ id: "unsupported", kind: "action", text: "Unsupported source action", sources: [{ source_id: "file:owned:original", revision: "1" }] }),
    entry({ id: "stale", kind: "action", text: "Stale action", stale: true }),
    entry({ id: "fact", kind: "fact", text: "Fact entry" }),
    entry({ id: "pending", kind: "question", text: "Pending question", approval_state: "proposed" }),
  ];
  const fetchMock = vi.fn().mockResolvedValue(json(listing(entries))); vi.stubGlobal("fetch", fetchMock);
  const onResearch = vi.fn(); render(<ProjectKnowledgeSection project={project} onResearch={onResearch} />);
  expect(await screen.findAllByRole("button", { name: "Assign to agent" })).toHaveLength(3);
  fireEvent.click(within(screen.getByRole("article", { name: "Action: Open action" })).getByRole("button", { name: "Assign to agent" }));
  await waitFor(() => expect(onResearch).toHaveBeenCalledWith(project, entries[0], expect.any(AbortSignal)));
  fireEvent.click(within(screen.getByRole("article", { name: "Action: File-backed action" })).getByRole("button", { name: "Assign to agent" }));
  await waitFor(() => expect(onResearch).toHaveBeenCalledWith(project, entries[3], expect.any(AbortSignal)));
  expect(fetchMock.mock.calls).toHaveLength(1);
});

it("does not offer public research in Private Projects and cancels a hidden prefill", async () => {
  const saved = entry({ kind: "action" });
  vi.stubGlobal("fetch", vi.fn().mockResolvedValue(json(listing([saved]))));
  let finish!: () => void; const onResearch = vi.fn().mockImplementation(() => new Promise<void>((resolve) => { finish = resolve; }));
  const { rerender } = render(<ProjectKnowledgeSection project={project} onResearch={onResearch} />);
  fireEvent.click(await screen.findByRole("button", { name: "Assign to agent" }));
  const signal = onResearch.mock.calls[0][2] as AbortSignal;
  rerender(<ProjectKnowledgeSection project={project} visible={false} onResearch={onResearch} />);
  expect(signal.aborted).toBe(true); finish();
  rerender(<ProjectKnowledgeSection project={{ ...project, private_required: true }} onResearch={onResearch} />);
  expect(screen.queryByRole("button", { name: "Assign to agent" })).not.toBeInTheDocument();
});

it("lazily lists only agents linked to this entry and opens the same saved task", async () => {
  const saved = entry({ kind: "action" });
  const linked: AgentTask = { id: "agent-linked", title: "Existing entry research", objective: "Research entry", status: "completed", revision: 8, created_at: project.created_at, updated_at: project.updated_at, steps: [], result: "Report", error: "", attempts: 1, project_context: { project_id: project.id, project_name: project.name, available: true, current: true, instructions_included: false, knowledge: [], origin_entry: { id: saved.id, revision: saved.revision } } };
  const other = { ...linked, id: "agent-other", title: "Unrelated research", project_context: { ...linked.project_context!, origin_entry: { id: "another-entry", revision: 1 } } };
  const fetchMock = vi.fn((path: string) => Promise.resolve(json(path.startsWith("/api/agents") ? { items: [linked, other] } : listing([saved])))); vi.stubGlobal("fetch", fetchMock);
  const open = vi.fn(); render(<ProjectKnowledgeSection project={project} onOpenAgent={open} />);
  await screen.findByRole("button", { name: "Assigned agents" }); expect(fetchMock.mock.calls).toHaveLength(1);
  fireEvent.click(screen.getByRole("button", { name: "Assigned agents" }));
  fireEvent.click(await screen.findByRole("button", { name: "Open agent: Existing entry research" }));
  expect(open).toHaveBeenCalledWith(linked, expect.any(AbortSignal)); expect(screen.queryByText("Unrelated research")).not.toBeInTheDocument();
  expect(fetchMock).toHaveBeenCalledWith("/api/agents?project_id=project-1", expect.objectContaining({ cache: "no-store", signal: expect.any(AbortSignal) }));
  expect(fetchMock.mock.calls).toHaveLength(2);
});

it("discards late linked-agent lists after hiding the Project", async () => {
  let finish!: (response: Response) => void;
  const fetchMock = vi.fn<(path: string, options?: RequestInit) => Promise<Response>>((path: string) => path.startsWith("/api/agents") ? new Promise<Response>((resolve) => { finish = resolve; }) : Promise.resolve(json(listing([entry({ kind: "action" })]))));
  vi.stubGlobal("fetch", fetchMock); const open = vi.fn();
  const { rerender } = render(<ProjectKnowledgeSection project={project} onOpenAgent={open} />);
  fireEvent.click(await screen.findByRole("button", { name: "Assigned agents" }));
  const signal = (fetchMock.mock.calls.find(([path]) => path.startsWith("/api/agents"))![1] as RequestInit).signal;
  rerender(<ProjectKnowledgeSection project={project} visible={false} onOpenAgent={open} />); expect(signal?.aborted).toBe(true);
  finish(json({ items: [] })); expect(screen.queryByText("No agents have been assigned to this entry.")).not.toBeInTheDocument();
});

it("saves explicitly inspected references and preserves a new entry draft when closed", async () => {
  const saved = entry({ text: "Use the inspected date.", sources: [reference] });
  const fetchMock = vi.fn().mockImplementation((path: string, options?: RequestInit) => {
    if (path.includes("source-candidates")) return Promise.resolve(json({ items: [{ source_id: reference.source_id, kind: "file", title: "Project source", private_required: false }], next_cursor: null }));
    if (path.includes("source-preview")) return Promise.resolve(json(sourcePreview()));
    if (options?.method === "POST") return Promise.resolve(json(saved));
    return Promise.resolve(json(listing([])));
  });
  vi.stubGlobal("fetch", fetchMock);
  render(<ProjectKnowledgeSection project={project} />);
  await waitFor(() => expect(screen.getByRole("button", { name: "Add entry" })).toBeEnabled());
  fireEvent.click(screen.getByRole("button", { name: "Add entry" }));
  fireEvent.change(screen.getByLabelText("Entry text"), { target: { value: saved.text } });
  await attachSource();
  fireEvent.click(screen.getByRole("button", { name: "Cancel" }));
  fireEvent.click(screen.getByRole("button", { name: "Add entry" }));
  expect(screen.getByLabelText("Entry text")).toHaveValue(saved.text);
  expect(screen.getByRole("button", { name: "Source references (1/8)" })).toBeVisible();
  fireEvent.click(screen.getByRole("button", { name: "Save entry" }));
  await waitFor(() => expect(screen.queryByLabelText("Entry text")).not.toBeInTheDocument());
  const [, options] = fetchMock.mock.calls.find(([, init]) => init?.method === "POST")!;
  expect(JSON.parse(options!.body as string)).toEqual({ kind: "fact", text: saved.text, pinned: false, sources: [reference], request_id: expect.any(String) });
});

it("writes an explicit empty reference list when unlinking and preserves agent provenance", async () => {
  const proposed = entry({ approval_state: "proposed", origin: "agent_report", source_task_id: "task-1", sources: [reference] });
  const fetchMock = vi.fn().mockImplementation((path: string, options?: RequestInit) => Promise.resolve(json(path.includes("source-candidates") ? { items: [], next_cursor: null } : options?.method === "PATCH" ? { ...proposed, sources: [], revision: 2 } : listing([proposed]))));
  vi.stubGlobal("fetch", fetchMock);
  render(<ProjectKnowledgeSection project={project} />);
  fireEvent.click(await screen.findByRole("button", { name: "1 proposal needs your review" }));
  fireEvent.click(screen.getByRole("button", { name: "Edit" }));
  fireEvent.click(screen.getByRole("button", { name: "Source references (1/8)" }));
  fireEvent.click(screen.getByRole("button", { name: `Remove reference to ${reference.source_id}` }));
  fireEvent.click(screen.getByRole("button", { name: "Save changes" }));
  await waitFor(() => expect(screen.queryByLabelText("Edit entry text")).not.toBeInTheDocument());
  const [, options] = fetchMock.mock.calls.find(([, init]) => init?.method === "PATCH")!;
  expect(JSON.parse(options!.body as string)).toEqual({ expected_revision: 1, kind: "decision", text: proposed.text, sources: [] });
  expect(screen.getByText(/Proposed from an agent report/u)).toBeVisible();
  expect(screen.getByRole("button", { name: "Approve proposal" })).toBeVisible();
});

it("preserves saved references without reading or replacing them during a text-only edit", async () => {
  const original = entry({ sources: [reference] });
  const fetchMock = vi.fn().mockResolvedValueOnce(json(listing([original])))
    .mockResolvedValueOnce(json({ ...original, text: "Clearer wording", revision: 2 }));
  vi.stubGlobal("fetch", fetchMock);
  render(<ProjectKnowledgeSection project={project} />);
  fireEvent.click(await screen.findByRole("button", { name: "Edit" }));
  fireEvent.change(screen.getByLabelText("Edit entry text"), { target: { value: "Clearer wording" } });
  fireEvent.click(screen.getByRole("button", { name: "Save changes" }));
  await waitFor(() => expect(screen.queryByLabelText("Edit entry text")).not.toBeInTheDocument());
  expect(fetchMock).toHaveBeenCalledTimes(2);
  expect(JSON.parse(fetchMock.mock.calls[1][1].body as string)).toEqual({ expected_revision: 1, kind: "decision", text: "Clearer wording" });
  expect(screen.getByText("1 source reference")).toBeVisible();
});

it("keeps text and source drafts through a conflict and requires explicit rebase", async () => {
  const latestReference = { source_id: "message:elsewhere", revision: "message-v2" };
  const latest = entry({ text: "Latest saved text", sources: [latestReference], revision: 2 });
  let writes = 0;
  let reads = 0;
  const fetchMock = vi.fn().mockImplementation((path: string, options?: RequestInit) => {
    if (path.includes("source-candidates")) return Promise.resolve(json({ items: [], next_cursor: null }));
    if (options?.method === "PATCH") {
      writes++;
      return Promise.resolve(writes === 1 ? json({ detail: "Revision conflict" }, 409) : json({ ...latest, text: "My unsaved text", sources: [], revision: 3 }));
    }
    reads++;
    return Promise.resolve(json(listing([reads === 1 ? entry({ sources: [reference] }) : latest])));
  });
  vi.stubGlobal("fetch", fetchMock);
  render(<ProjectKnowledgeSection project={project} />);
  fireEvent.click(await screen.findByRole("button", { name: "Edit" }));
  fireEvent.change(screen.getByLabelText("Edit entry text"), { target: { value: "My unsaved text" } });
  fireEvent.click(screen.getByRole("button", { name: "Source references (1/8)" }));
  fireEvent.click(screen.getByRole("button", { name: `Remove reference to ${reference.source_id}` }));
  fireEvent.click(screen.getByRole("button", { name: "Save changes" }));
  const rebase = await screen.findByRole("button", { name: "Keep my draft and use latest revision" });
  expect(screen.getByLabelText("Edit entry text")).toHaveValue("My unsaved text");
  expect(screen.getByRole("button", { name: "Source references (0/8)" })).toBeVisible();
  expect(screen.getByRole("button", { name: "Save changes" })).toBeDisabled();
  expect(screen.getByText("Latest saved source references")).toBeVisible();
  expect(screen.getAllByText(`${latestReference.source_id} · Revision ${latestReference.revision}`)).toHaveLength(2);
  fireEvent.click(rebase);
  fireEvent.click(screen.getByRole("button", { name: "Save changes" }));
  await waitFor(() => expect(screen.queryByLabelText("Edit entry text")).not.toBeInTheDocument());
  const bodies = fetchMock.mock.calls.filter(([, init]) => init?.method === "PATCH").map(([, init]) => JSON.parse(init!.body as string));
  expect(bodies).toEqual([
    { expected_revision: 1, kind: "decision", text: "My unsaved text", sources: [] },
    { expected_revision: 2, kind: "decision", text: "My unsaved text", sources: [] },
  ]);
});

it("changes the creation idempotency key when selected references change after failure", async () => {
  const fetchMock = vi.fn().mockImplementation((path: string, options?: RequestInit) => {
    if (path.includes("source-candidates")) return Promise.resolve(json({ items: [{ source_id: reference.source_id, kind: "file", title: "Project source", private_required: false }], next_cursor: null }));
    if (path.includes("source-preview")) return Promise.resolve(json(sourcePreview()));
    if (options?.method === "POST") return Promise.resolve(json({ detail: "Temporary failure" }, 503));
    return Promise.resolve(json(listing([])));
  });
  vi.stubGlobal("fetch", fetchMock);
  render(<ProjectKnowledgeSection project={project} />);
  await waitFor(() => expect(screen.getByRole("button", { name: "Add entry" })).toBeEnabled());
  fireEvent.click(screen.getByRole("button", { name: "Add entry" }));
  fireEvent.change(screen.getByLabelText("Entry text"), { target: { value: "Keep this draft" } });
  fireEvent.click(screen.getByRole("button", { name: "Save entry" }));
  await screen.findByRole("alert");
  await attachSource();
  fireEvent.click(screen.getByRole("button", { name: "Save entry" }));
  await waitFor(() => expect(screen.getByRole("button", { name: "Save entry" })).toBeEnabled());
  const bodies = fetchMock.mock.calls.filter(([, init]) => init?.method === "POST").map(([, init]) => JSON.parse(init!.body as string));
  expect(bodies).toHaveLength(2);
  expect(bodies[0].request_id).not.toBe(bodies[1].request_id);
  expect(bodies[0]).not.toHaveProperty("sources");
  expect(bodies[1].sources).toEqual([reference]);
});

it("shows approved entries, pending proposals and their provenance as plain text", async () => {
  const saved = entry({ text: "<script>not executable</script>", pinned: true });
  const proposed = entry({ id: "proposal-1", text: "Adopt a second model.", approval_state: "proposed", origin: "agent_report", source_task_id: "task-1" });
  vi.stubGlobal("fetch", vi.fn().mockResolvedValue(json(listing([saved, proposed]))));
  render(<ProjectKnowledgeSection project={project} />);
  expect(await screen.findByText(saved.text)).toBeVisible();
  expect(screen.queryByText(proposed.text)).not.toBeInTheDocument();
  expect(screen.getByText("Pinned")).toBeVisible();
  fireEvent.click(screen.getByRole("button", { name: "1 proposal needs your review" }));
  expect(screen.getByText(proposed.text)).toBeVisible();
  expect(screen.getByText("Proposed from an agent report · Not independently verified", { exact: false })).toBeVisible();
  expect(screen.getByText(/No inspected evidence is attached/u)).toBeVisible();
  expect(document.querySelector("script")).toBeNull();
});

it("saves manual approved entries with an idempotency key and leaves new actions open", async () => {
  const saved = entry({ kind: "action", text: "Compare dates.", pinned: true });
  const fetchMock = vi.fn().mockResolvedValueOnce(json(listing([]))).mockResolvedValueOnce(json(saved));
  vi.stubGlobal("fetch", fetchMock);
  const onChanged = vi.fn();
  render(<ProjectKnowledgeSection project={project} onChanged={onChanged} />);
  await waitFor(() => expect(screen.getByRole("button", { name: "Add entry" })).toBeEnabled());
  fireEvent.click(screen.getByRole("button", { name: "Add entry" }));
  const form = screen.getByRole("button", { name: "Save entry" }).closest("form")!;
  fireEvent.change(within(form).getByLabelText("Entry type"), { target: { value: "action" } });
  fireEvent.change(within(form).getByLabelText("Entry text"), { target: { value: "  Compare dates.  " } });
  fireEvent.click(within(form).getByRole("checkbox"));
  fireEvent.click(within(form).getByRole("button", { name: "Save entry" }));
  expect(await screen.findByRole("button", { name: "Complete action" })).toBeEnabled();
  expect(screen.getByText("Compare dates.")).toBeVisible();
  const init = fetchMock.mock.calls[1][1] as RequestInit;
  expect(JSON.parse(init.body as string)).toEqual({ kind: "action", text: "Compare dates.", pinned: true, request_id: expect.any(String) });
  expect(onChanged).toHaveBeenCalledTimes(1);
});

it("requires a second press of the same red delete button and resets on blur and Escape", async () => {
  const fetchMock = vi.fn().mockResolvedValueOnce(json(listing([entry()]))).mockResolvedValueOnce(new Response(null, { status: 204 }));
  vi.stubGlobal("fetch", fetchMock);
  render(<ProjectKnowledgeSection project={project} />);
  const button = await screen.findByRole("button", { name: "Delete entry" });
  expect(button).toHaveClass("danger-button");
  fireEvent.click(button);
  expect(button).toHaveAccessibleName("Confirm delete entry");
  expect(button).toHaveTextContent("✓");
  expect(fetchMock).toHaveBeenCalledTimes(1);
  fireEvent.blur(button);
  expect(button).toHaveAccessibleName("Delete entry");
  fireEvent.click(button);
  fireEvent.keyDown(button, { key: "Escape" });
  expect(button).toHaveAccessibleName("Delete entry");
  fireEvent.click(button);
  fireEvent.click(button);
  await waitFor(() => expect(screen.queryByText("Use local models for agents.")).not.toBeInTheDocument());
  expect(fetchMock).toHaveBeenLastCalledWith("/api/projects/project-1/knowledge/entry-1?expected_revision=1", expect.objectContaining({ method: "DELETE" }));
});

it("refreshes conflicts while preserving unsaved text until the user accepts the newer revision", async () => {
  const latest = entry({ text: "The saved decision changed elsewhere.", revision: 2 });
  const fetchMock = vi.fn().mockResolvedValueOnce(json(listing([entry()])))
    .mockResolvedValueOnce(json({ detail: "Revision conflict" }, 409))
    .mockResolvedValueOnce(json(listing([latest])))
    .mockResolvedValueOnce(json({ ...latest, text: "My unsaved correction.", revision: 3 }));
  vi.stubGlobal("fetch", fetchMock);
  render(<ProjectKnowledgeSection project={project} />);
  fireEvent.click(await screen.findByRole("button", { name: "Edit" }));
  fireEvent.change(screen.getByLabelText("Edit entry text"), { target: { value: "My unsaved correction." } });
  fireEvent.click(screen.getByRole("button", { name: "Save changes" }));
  expect(await screen.findByText(latest.text)).toBeVisible();
  expect(screen.getByLabelText("Edit entry text")).toHaveValue("My unsaved correction.");
  expect(screen.getByRole("button", { name: "Save changes" })).toBeDisabled();
  fireEvent.click(screen.getByRole("button", { name: "Keep my draft and use latest revision" }));
  fireEvent.click(screen.getByRole("button", { name: "Save changes" }));
  await waitFor(() => expect(screen.queryByLabelText("Edit entry text")).not.toBeInTheDocument());
  const bodies = fetchMock.mock.calls.filter(([, options]) => options?.method === "PATCH").map(([, options]) => JSON.parse(options.body as string));
  expect(bodies).toEqual([{ expected_revision: 1, kind: "decision", text: "My unsaved correction." }, { expected_revision: 2, kind: "decision", text: "My unsaved correction." }]);
});

it("keeps proposals inactive and explicitly approves or rejects the saved revision", async () => {
  const proposed = entry({ approval_state: "proposed", origin: "agent_report", source_task_id: "task-1" });
  const fetchMock = vi.fn().mockResolvedValueOnce(json(listing([proposed])))
    .mockResolvedValueOnce(json({ ...proposed, approval_state: "approved", revision: 2 }));
  vi.stubGlobal("fetch", fetchMock);
  render(<ProjectKnowledgeSection project={project} />);
  fireEvent.click(await screen.findByRole("button", { name: "1 proposal needs your review" }));
  expect(screen.queryByRole("button", { name: "Pin" })).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "Approve proposal" }));
  await screen.findByText("No proposals need review.");
  expect(fetchMock).toHaveBeenLastCalledWith("/api/projects/project-1/knowledge/entry-1/decision", expect.objectContaining({ method: "POST", body: JSON.stringify({ decision: "approve", expected_revision: 1 }) }));
  fireEvent.change(screen.getByLabelText("Show"), { target: { value: "current" } });
  expect(screen.getByText("Approved")).toBeVisible();
});

it("allows questions to resolve and reopen without changing approval", async () => {
  const question = entry({ kind: "question", text: "Which date is correct?" });
  const fetchMock = vi.fn().mockResolvedValueOnce(json(listing([question])))
    .mockResolvedValueOnce(json({ ...question, work_status: "resolved", revision: 2 }))
    .mockResolvedValueOnce(json({ ...question, revision: 3 }));
  vi.stubGlobal("fetch", fetchMock);
  render(<ProjectKnowledgeSection project={project} />);
  fireEvent.click(await screen.findByRole("button", { name: "Resolve question" }));
  fireEvent.click(await screen.findByRole("button", { name: "Reopen" }));
  await screen.findByRole("button", { name: "Resolve question" });
  const bodies = fetchMock.mock.calls.slice(1).map(([, options]) => JSON.parse(options.body as string));
  expect(bodies).toEqual([{ expected_revision: 1, work_status: "resolved" }, { expected_revision: 2, work_status: "open" }]);
});

it("rejects proposals explicitly and keeps the rejected text in its own view", async () => {
  const proposed = entry({ approval_state: "proposed", origin: "agent_report" });
  const fetchMock = vi.fn().mockResolvedValueOnce(json(listing([proposed])))
    .mockResolvedValueOnce(json({ ...proposed, approval_state: "rejected", revision: 2 }));
  vi.stubGlobal("fetch", fetchMock);
  render(<ProjectKnowledgeSection project={project} />);
  fireEvent.click(await screen.findByRole("button", { name: "1 proposal needs your review" }));
  fireEvent.click(screen.getByRole("button", { name: "Reject proposal" }));
  await screen.findByText("No proposals need review.");
  fireEvent.change(screen.getByLabelText("Show"), { target: { value: "history" } });
  expect(screen.getByText("Rejected")).toBeVisible();
  expect(screen.queryByRole("button", { name: "Edit" })).not.toBeInTheDocument();
  expect(fetchMock).toHaveBeenLastCalledWith("/api/projects/project-1/knowledge/entry-1/decision", expect.objectContaining({ body: JSON.stringify({ decision: "reject", expected_revision: 1 }) }));
});

it("keeps an open draft mounted when filters change", async () => {
  vi.stubGlobal("fetch", vi.fn().mockResolvedValue(json(listing([entry()]))));
  render(<ProjectKnowledgeSection project={project} />);
  fireEvent.click(await screen.findByRole("button", { name: "Edit" }));
  fireEvent.change(screen.getByLabelText("Edit entry text"), { target: { value: "A draft I have not saved." } });
  fireEvent.change(screen.getByLabelText("Show"), { target: { value: "proposed" } });
  expect(screen.getByLabelText("Edit entry text")).toHaveValue("A draft I have not saved.");
  fireEvent.click(screen.getByRole("button", { name: "Cancel edit" }));
  expect(screen.queryByLabelText("Edit entry text")).not.toBeInTheDocument();
  expect(screen.getByText("No proposals need review.")).toBeVisible();
});

it("recovers from initial list failures with a retry", async () => {
  vi.stubGlobal("fetch", vi.fn().mockResolvedValueOnce(json({ detail: "Unavailable" }, 503)).mockResolvedValueOnce(json(listing([entry()]))));
  render(<ProjectKnowledgeSection project={project} />);
  expect(await screen.findByRole("alert")).toHaveTextContent("Could not refresh project knowledge.");
  fireEvent.click(screen.getByRole("button", { name: "Try again" }));
  expect(await screen.findByText("Use local models for agents.")).toBeVisible();
  expect(screen.queryByRole("alert")).not.toBeInTheDocument();
});

it("shows bounded revision history and source warnings without inventing source links", async () => {
  const source = { source_id: "message:owned", revision: "2" };
  const current = entry({ sources: [source], stale: true, source_warnings: ["The source was edited."] });
  const fetchMock = vi.fn().mockResolvedValueOnce(json(listing([current]))).mockResolvedValueOnce(json({ items: [entry({ text: "Earlier wording." }), current] }));
  vi.stubGlobal("fetch", fetchMock);
  const onOpenSource = vi.fn();
  render(<ProjectKnowledgeSection project={project} onOpenSource={onOpenSource} />);
  expect(await screen.findByText("Sources need review")).toBeVisible();
  expect(screen.getByText("The source was edited.")).toBeVisible();
  expect(screen.queryByRole("button", { name: "View source" })).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "History" }));
  expect(await screen.findByText("Earlier wording.")).toBeVisible();
  expect(fetchMock).toHaveBeenLastCalledWith("/api/projects/project-1/knowledge/entry-1/history", expect.objectContaining({ cache: "no-store" }));
});

it("shows saved reference revisions in history without reading historical source text", async () => {
  const original = entry({ sources: [reference] });
  const current = entry({ sources: [{ ...reference, revision: "text-v2" }], revision: 2 });
  const fetchMock = vi.fn().mockResolvedValueOnce(json(listing([current])))
    .mockResolvedValueOnce(json({ items: [original, current] }));
  vi.stubGlobal("fetch", fetchMock);
  render(<ProjectKnowledgeSection project={project} />);
  fireEvent.click(await screen.findByRole("button", { name: "History" }));
  const history = await screen.findByLabelText("Entry revision history");
  const references = await within(history).findAllByText("1 saved source reference");
  fireEvent.click(references[0]);
  expect(within(history).getByText(`${reference.source_id} · Revision text-v1`)).toBeVisible();
  expect(within(history).queryByRole("button", { name: "View source" })).not.toBeInTheDocument();
  expect(fetchMock).toHaveBeenCalledTimes(2);
});

it("passes a live source reference and entry identity to the source viewer", async () => {
  const source = { source_id: "file:owned:text", revision: "digest" };
  const saved = entry({ sources: [source] });
  vi.stubGlobal("fetch", vi.fn().mockResolvedValue(json(listing([saved]))));
  const onOpenSource = vi.fn();
  render(<ProjectKnowledgeSection project={project} onOpenSource={onOpenSource} />);
  fireEvent.click(await screen.findByText("1 source reference"));
  fireEvent.click(screen.getByRole("button", { name: "View source" }));
  expect(onOpenSource).toHaveBeenCalledWith(source, saved, 0);
});

it("drops old project content and ignores pending requests after switching projects", async () => {
  let finish!: (response: Response) => void;
  const fetchMock = vi.fn().mockResolvedValueOnce(json(listing([entry()])))
    .mockImplementationOnce(() => new Promise<Response>((resolve) => { finish = resolve; }))
    .mockResolvedValueOnce(json(listing([])));
  vi.stubGlobal("fetch", fetchMock);
  const onChanged = vi.fn();
  const view = render(<ProjectKnowledgeSection project={project} onChanged={onChanged} />);
  fireEvent.click(await screen.findByRole("button", { name: "Pin" }));
  view.rerender(<ProjectKnowledgeSection project={{ ...project, id: "project-2" }} onChanged={onChanged} />);
  expect(screen.queryByText("Use local models for agents.")).not.toBeInTheDocument();
  finish(json(entry({ pinned: true, revision: 2 })));
  await screen.findByText("No saved entries match this view.");
  expect(onChanged).not.toHaveBeenCalled();
});

it("loads bounded list pages before offering complete filters", async () => {
  const fetchMock = vi.fn().mockResolvedValueOnce(json({ ...listing([entry()]), next_cursor: "page2" }))
    .mockResolvedValueOnce(json(listing([entry({ id: "entry-2", kind: "fact", text: "A second saved fact." })])));
  vi.stubGlobal("fetch", fetchMock);
  render(<ProjectKnowledgeSection project={project} />);
  expect(await screen.findByText("A second saved fact.")).toBeVisible();
  expect(screen.getByText("Use local models for agents.")).toBeVisible();
  expect(fetchMock).toHaveBeenLastCalledWith("/api/projects/project-1/knowledge?limit=50&cursor=page2", expect.anything());
});
