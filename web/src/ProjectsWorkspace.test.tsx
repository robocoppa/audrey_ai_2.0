import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { ProjectsWorkspace } from "./ProjectsWorkspace";
import * as api from "./api";

const project: api.AudreyProject = { id: "garden", name: "Garden project", instructions: "", created_at: "2026-11-04T12:00:00Z", updated_at: "2026-11-04T12:00:00Z" };
const other = { ...project, id: "other", name: "Other project" };
const limits = { max_name_chars: 100, max_instructions_chars: 4000, max_files: 20 };
const model: api.AudreyModel = { id: "auto", label: "Auto", description: "", kind: "workflow", mode: "auto", presentation: "auto", capabilities: [], enabled: true, audience: "users", visibility: "public", roles: [], portrait_url: "" };
const conversation: api.Conversation = { id: "thread", title: "Garden discussion", project_id: project.id, default_mode: "auto", default_model_id: "auto", created_at: project.created_at, updated_at: project.updated_at, last_message_at: project.updated_at, archived_at: null };
function props() { return { visible: true, models: [model], onOpenConversation: vi.fn(), onNewProjectAgent: vi.fn(), onOpenProjectAgent: vi.fn(), onNewProjectWatch: vi.fn(), onOpenProjectWatch: vi.fn() }; }
beforeEach(() => {
  vi.spyOn(api, "listProjects").mockResolvedValue({ items: [project, other], next_cursor: null, limits });
  vi.spyOn(api, "getProject").mockImplementation(async (id) => id === other.id ? other : project);
  vi.spyOn(api, "listProjectConversations").mockResolvedValue({ items: [conversation], next_cursor: null });
  vi.spyOn(api, "listProjectFiles").mockResolvedValue({ items: [], limits });
  vi.spyOn(api, "getProjectKnowledgeText").mockResolvedValue({ text: "Garden label: Fern Circle", revision: "v1", max_text_chars: 8000 });
  vi.spyOn(api, "listAgentTasks").mockResolvedValue({ items: [] });
  vi.spyOn(api, "listWatches").mockResolvedValue({ items: [] });
});
afterEach(() => { cleanup(); vi.restoreAllMocks(); });

it("opens Projects without creating work, and uses canonical Project conversations", async () => {
  const options = props();
  const create = vi.spyOn(api, "createProjectConversation").mockResolvedValue({ ...conversation, id: "new-thread" });
  render(<ProjectsWorkspace {...options} />);
  fireEvent.click(await screen.findByRole("button", { name: project.name }));
  expect(await screen.findByRole("heading", { name: project.name })).toBeVisible();
  expect(await screen.findByDisplayValue("Garden label: Fern Circle")).toBeVisible();
  expect(create).not.toHaveBeenCalled();
  const box = screen.getByRole("region", { name: "Project conversations" });
  fireEvent.click(await within(box).findByRole("button", { name: /Garden discussion/ }));
  expect(options.onOpenConversation).toHaveBeenCalledWith(conversation);
  fireEvent.click(within(box).getByRole("button", { name: "New conversation in project" }));
  await waitFor(() => expect(create).toHaveBeenCalledWith(project.id, "auto"));
  await waitFor(() => expect(options.onOpenConversation).toHaveBeenLastCalledWith(expect.objectContaining({ id: "new-thread", project_id: project.id })));
});

it("keeps the selected Project and unsaved Knowledge when hidden and reopened", async () => {
  const options = props();
  const view = render(<ProjectsWorkspace {...options} />);
  fireEvent.click(await screen.findByRole("button", { name: project.name }));
  const input = await screen.findByDisplayValue("Garden label: Fern Circle");
  fireEvent.change(input, { target: { value: "Unsaved garden notes" } });
  view.rerender(<ProjectsWorkspace {...options} visible={false} />);
  view.rerender(<ProjectsWorkspace {...options} visible />);
  await waitFor(() => expect(screen.getByRole("textbox", { name: "Project knowledge" })).toBeEnabled());
  expect(screen.getByRole("textbox", { name: "Project knowledge" })).toHaveValue("Unsaved garden notes");
  expect(screen.getByRole("heading", { name: project.name })).toBeVisible();
});

it("ignores a late Project response after selecting a different Project", async () => {
  let finish!: (value: api.AudreyProject) => void;
  vi.mocked(api.getProject).mockImplementation((id) => id === project.id ? new Promise((resolve) => { finish = resolve; }) : Promise.resolve(other));
  render(<ProjectsWorkspace {...props()} />);
  fireEvent.click(await screen.findByRole("button", { name: project.name }));
  await waitFor(() => expect(api.getProject).toHaveBeenCalledWith(project.id, expect.any(AbortSignal)));
  fireEvent.click(screen.getByRole("button", { name: other.name }));
  expect(await screen.findByRole("heading", { name: other.name })).toBeVisible();
  await act(async () => finish(project));
  expect(screen.queryByRole("heading", { name: project.name })).not.toBeInTheDocument();
});

it("creates and selects a Project with no implicit conversation or Agent", async () => {
  const created = { ...project, id: "created", name: "New garden" };
  const create = vi.spyOn(api, "createProject").mockResolvedValue(created);
  vi.mocked(api.getProject).mockResolvedValue(created);
  const conversationCreate = vi.spyOn(api, "createProjectConversation");
  const agentCreate = vi.spyOn(api, "createAgentTask");
  render(<ProjectsWorkspace {...props()} />);
  fireEvent.click(screen.getByRole("button", { name: "New project" }));
  const dialog = screen.getByRole("dialog", { name: "New project" });
  fireEvent.change(within(dialog).getByRole("textbox", { name: "Project name" }), { target: { value: created.name } });
  fireEvent.click(within(dialog).getByRole("button", { name: "Create project" }));
  expect(await screen.findByRole("heading", { name: created.name })).toBeVisible();
  expect(create).toHaveBeenCalledWith(created.name, "", "auto");
  expect(conversationCreate).not.toHaveBeenCalled(); expect(agentCreate).not.toHaveBeenCalled();
});

it("does not navigate when a pending conversation creation completes after Projects is hidden", async () => {
  let finish!: (value: api.Conversation) => void;
  vi.spyOn(api, "createProjectConversation").mockImplementation(() => new Promise((resolve) => { finish = resolve; }));
  const options = props();
  const view = render(<ProjectsWorkspace {...options} />);
  fireEvent.click(await screen.findByRole("button", { name: project.name }));
  fireEvent.click(await screen.findByRole("button", { name: "New conversation in project" }));
  view.rerender(<ProjectsWorkspace {...options} visible={false} />);
  await act(async () => finish(conversation));
  expect(options.onOpenConversation).not.toHaveBeenCalled();
});

it("refreshes an unavailable Project on explicit retry", async () => {
  vi.mocked(api.getProject).mockRejectedValueOnce(new api.ApiError(404, "Project not found")).mockResolvedValue(project);
  render(<ProjectsWorkspace {...props()} />);
  fireEvent.click(await screen.findByRole("button", { name: project.name }));
  expect(await screen.findByRole("alert")).toHaveTextContent("Project not found");
  fireEvent.click(screen.getByRole("button", { name: "Retry" }));
  expect(await screen.findByRole("heading", { name: project.name })).toBeVisible();
});
