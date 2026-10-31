import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { ProjectHome } from "./ProjectWorkspace";
import * as api from "./api";

vi.mock("./ProjectKnowledgeSection", () => ({ ProjectKnowledgeSection: () => null }));
vi.mock("./TaxProjectSection", () => ({ TaxProjectSection: () => null }));
const project: api.AudreyProject = { id: "project-one", name: "Public releases", instructions: "Stable releases only.", created_at: "2026-10-31T12:00:00Z", updated_at: "2026-10-31T12:00:00Z" };
const task: api.AgentTask = { id: "agent-one", title: "Release comparison", objective: "Compare public releases.", status: "completed", revision: 6, created_at: project.created_at, updated_at: project.updated_at, steps: [], result: "Saved report", error: "", attempts: 1, project_context: { project_id: project.id, project_name: project.name, available: true, current: true, instructions_included: true, knowledge: [] } };
function props() { return { project, conversations: [], files: [], filesLoading: false, limits: { max_name_chars: 100, max_instructions_chars: 4000, max_files: 20 }, defaultModelId: "auto", onProjectChange: vi.fn(), onProjectDeleted: vi.fn(), onConversationCreated: vi.fn(), onConversationSelected: vi.fn(), onFilesChange: vi.fn() }; }
afterEach(() => { cleanup(); vi.restoreAllMocks(); });

it("lists only this Project's saved agents and offers explicit prefill without creating", async () => {
  const listing = vi.spyOn(api, "listAgentTasks").mockResolvedValue({ items: [task, { ...task, id: "other", title: "Other Project", project_context: { ...task.project_context!, project_id: "other-project" } }] });
  const create = vi.spyOn(api, "createAgentTask"), onNew = vi.fn(), onOpen = vi.fn();
  render(<ProjectHome {...props()} onNewProjectAgent={onNew} onOpenProjectAgent={onOpen} />);
  fireEvent.click(await screen.findByRole("button", { name: /Release comparison/ }));
  expect(onOpen).toHaveBeenCalledWith(task, expect.any(AbortSignal));
  expect(screen.queryByText("Other Project")).not.toBeInTheDocument();
  await waitFor(() => expect(screen.getByRole("button", { name: "New agent in Project" })).toBeEnabled());
  fireEvent.click(screen.getByRole("button", { name: "New agent in Project" }));
  expect(onNew).toHaveBeenCalledWith(project, expect.any(AbortSignal)); expect(create).not.toHaveBeenCalled();
  expect(listing).toHaveBeenCalledWith(expect.any(AbortSignal), project.id);
});

it("never exposes public agent creation or list reads inside Private Projects", () => {
  const listing = vi.spyOn(api, "listAgentTasks");
  render(<ProjectHome {...props()} project={{ ...project, private_required: true }} onNewProjectAgent={vi.fn()} onOpenProjectAgent={vi.fn()} />);
  expect(screen.queryByRole("button", { name: "New agent in Project" })).not.toBeInTheDocument(); expect(listing).not.toHaveBeenCalled();
});

it("aborts hidden navigation and refreshes without leaving controls disabled", async () => {
  vi.spyOn(api, "listAgentTasks").mockResolvedValue({ items: [task] });
  let signal: AbortSignal | undefined;
  const onOpen = vi.fn((_task: api.AgentTask, request?: AbortSignal) => { signal = request; return new Promise<void>(() => {}); });
  const base = props(); const { rerender } = render(<ProjectHome {...base} onOpenProjectAgent={onOpen} onNewProjectAgent={vi.fn()} />);
  fireEvent.click(await screen.findByRole("button", { name: /Release comparison/ }));
  expect(signal?.aborted).toBe(false);
  rerender(<ProjectHome {...base} visible={false} onOpenProjectAgent={onOpen} onNewProjectAgent={vi.fn()} />);
  expect(signal?.aborted).toBe(true);
  rerender(<ProjectHome {...base} visible onOpenProjectAgent={onOpen} onNewProjectAgent={vi.fn()} />);
  await waitFor(() => expect(screen.getByRole("button", { name: "New agent in Project" })).toBeEnabled());
});
