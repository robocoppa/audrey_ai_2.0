import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";

import { ProjectKnowledgeSection } from "./ProjectKnowledgeSection";
import { type AudreyProject, type KnowledgeEntry } from "./api";

const project: AudreyProject = { id: "project-1", name: "Research", instructions: "", created_at: "2026-10-29T10:00:00Z", updated_at: "2026-10-29T10:00:00Z" };
function entry(patch: Partial<KnowledgeEntry> = {}): KnowledgeEntry {
  return { id: "entry-1", project_id: project.id, kind: "decision", text: "Use local models for agents.", approval_state: "approved", work_status: "open", revision: 1, pinned: false, origin: "user", sources: [], source_task_id: null, created_at: "2026-10-29T10:00:00Z", updated_at: "2026-10-29T10:00:00Z", approved_at: "2026-10-29T10:00:00Z", stale: false, source_warnings: [], replaces_id: null, base_revision: null, ...patch };
}
const limits = { max_entries: 200, max_text_chars: 8000, max_sources: 8 };
function json(body: unknown, status = 200) { return new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } }); }
function listing(items: KnowledgeEntry[]) { return { items, next_cursor: null, limits }; }
afterEach(() => { cleanup(); vi.restoreAllMocks(); vi.unstubAllGlobals(); });

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
