import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { ProjectsWorkspace } from "./ProjectsWorkspace";
import * as api from "./api";

const project: api.AudreyProject = { id: "garden", name: "Garden project", instructions: "", created_at: "2026-11-04T12:00:00Z", updated_at: "2026-11-04T12:00:00Z", conversation_count: 7, last_activity_at: "2026-11-05T18:30:00Z" };
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
  vi.mocked(api.getProject).mockResolvedValueOnce({ ...project, conversation_count: 8 });
  fireEvent.click(within(box).getByRole("button", { name: "New conversation in project" }));
  await waitFor(() => expect(create).toHaveBeenCalledWith(project.id, "auto"));
  await waitFor(() => expect(options.onOpenConversation).toHaveBeenLastCalledWith(expect.objectContaining({ id: "new-thread", project_id: project.id })));
  await waitFor(() => expect(within(screen.getByRole("button", { name: project.name })).getByText("8 chats")).toBeVisible());
});

it("shows canonical chat totals and local activity labels without fetching every Project", async () => {
  const privateProject = { ...project, private_required: true };
  vi.mocked(api.listProjects).mockResolvedValue({ items: [privateProject, { ...other, conversation_count: 0 }], next_cursor: null, limits });
  render(<ProjectsWorkspace {...props()} />);
  const card = await screen.findByRole("button", { name: project.name });
  const activity = new Date(project.last_activity_at!);
  const fullActivity = new Intl.DateTimeFormat(undefined, { dateStyle: "full", timeStyle: "long" }).format(activity);
  expect(within(card).getByText("7 chats")).toBeVisible();
  expect(within(card).getByText("Private")).toBeVisible();
  expect(card).toHaveAccessibleDescription(`Private project. 7 chats. Last activity ${fullActivity}.`);
  expect(within(card).getByTitle(fullActivity)).toHaveAttribute("datetime", activity.toISOString());
  expect(within(card).getByTitle(fullActivity)).toHaveTextContent(new Intl.DateTimeFormat(undefined, { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" }).format(activity));
  expect(within(screen.getByRole("button", { name: other.name })).getByText("0 chats")).toBeVisible();
  expect(api.getProject).not.toHaveBeenCalled();
  expect(api.listProjectConversations).not.toHaveBeenCalled();
  fireEvent.click(card);
  await screen.findByRole("heading", { name: project.name });
  expect(within(card).getByText("7 chats")).toBeVisible();
  expect(api.listProjectConversations).toHaveBeenCalledTimes(1);
});

it("keeps loaded summary data across older detail responses and marks absent metadata honestly", async () => {
  const legacyProject: api.AudreyProject = { id: project.id, name: project.name, instructions: "", created_at: project.created_at, updated_at: project.updated_at };
  vi.mocked(api.getProject).mockResolvedValue(legacyProject);
  vi.mocked(api.listProjects).mockResolvedValue({ items: [project, { ...other, conversation_count: undefined, last_activity_at: undefined }], next_cursor: null, limits });
  render(<ProjectsWorkspace {...props()} />);
  const card = await screen.findByRole("button", { name: project.name });
  fireEvent.click(card);
  await screen.findByRole("heading", { name: project.name });
  expect(within(card).getByText("7 chats")).toBeVisible();
  const legacyCard = screen.getByRole("button", { name: other.name });
  expect(within(legacyCard).getByText("Chat count unavailable")).toBeVisible();
  expect(within(legacyCard).getByText("Activity unavailable")).toBeVisible();
  expect(within(legacyCard).queryByText("0 chats")).not.toBeInTheDocument();
});

it("updates the selected Project activity and count when a conversation is created", async () => {
  const created = { ...conversation, id: "new-thread", created_at: "2026-11-06T08:15:00Z", updated_at: "2026-11-06T08:15:00Z", last_message_at: null };
  vi.mocked(api.getProject).mockResolvedValueOnce(project).mockResolvedValueOnce({ ...project, conversation_count: 8, last_activity_at: created.created_at });
  vi.spyOn(api, "createProjectConversation").mockResolvedValue(created);
  render(<ProjectsWorkspace {...props()} />);
  fireEvent.click(await screen.findByRole("button", { name: project.name }));
  fireEvent.click(await screen.findByRole("button", { name: "New conversation in project" }));
  const card = screen.getByRole("button", { name: project.name });
  await waitFor(() => expect(within(card).getByText("8 chats")).toBeVisible());
  expect(card.querySelector("time")).toHaveAttribute("datetime", new Date(created.created_at).toISOString());
});

it("does not increment a refreshed count that already contains the newly created chat", async () => {
  let finishCreate!: (value: api.Conversation) => void;
  vi.spyOn(api, "createProjectConversation").mockImplementation(() => new Promise((resolve) => { finishCreate = resolve; }));
  const canonical = { ...project, conversation_count: 8 };
  vi.mocked(api.listProjects).mockResolvedValueOnce({ items: [project, other], next_cursor: null, limits })
    .mockResolvedValue({ items: [canonical, other], next_cursor: null, limits });
  vi.mocked(api.getProject).mockResolvedValueOnce(project).mockResolvedValue(canonical);
  const options = props();
  const view = render(<ProjectsWorkspace {...options} />);
  fireEvent.click(await screen.findByRole("button", { name: project.name }));
  await screen.findByRole("heading", { name: project.name });
  view.rerender(<ProjectsWorkspace {...options} visible={false} />);
  view.rerender(<ProjectsWorkspace {...options} visible />);
  fireEvent.click(screen.getByRole("button", { name: "New conversation in project" }));
  const card = screen.getByRole("button", { name: project.name });
  await waitFor(() => expect(within(card).getByText("8 chats")).toBeVisible());
  await act(async () => finishCreate({ ...conversation, id: "new-thread" }));
  await waitFor(() => expect(api.getProject).toHaveBeenCalledTimes(3));
  expect(within(card).getByText("8 chats")).toBeVisible();
  expect(within(card).queryByText("9 chats")).not.toBeInTheDocument();
});

it("ignores pre-creation details and older summaries while preserving Project content and drafts", async () => {
  let finishDetails!: (value: api.AudreyProject) => void;
  let finishSummary!: (value: api.AudreyProject) => void;
  vi.mocked(api.getProject).mockResolvedValueOnce(project)
    .mockImplementationOnce(() => new Promise((resolve) => { finishDetails = resolve; }))
    .mockImplementationOnce(() => new Promise((resolve) => { finishSummary = resolve; }))
    .mockResolvedValueOnce({ ...project, conversation_count: 9, name: "Unrelated snapshot name", instructions: "Unrelated snapshot instructions" });
  vi.spyOn(api, "createProjectConversation").mockResolvedValueOnce({ ...conversation, id: "first-new", title: "First new chat" })
    .mockResolvedValueOnce({ ...conversation, id: "second-new", title: "Second new chat" });
  const options = props();
  const view = render(<ProjectsWorkspace {...options} />);
  fireEvent.click(await screen.findByRole("button", { name: project.name }));
  const knowledge = await screen.findByDisplayValue("Garden label: Fern Circle");
  fireEvent.change(knowledge, { target: { value: "Keep this unsaved knowledge" } });
  view.rerender(<ProjectsWorkspace {...options} visible={false} />);
  view.rerender(<ProjectsWorkspace {...options} visible />);
  await waitFor(() => expect(api.getProject).toHaveBeenCalledTimes(2));
  fireEvent.click(screen.getByRole("button", { name: "New conversation in project" }));
  await waitFor(() => expect(api.getProject).toHaveBeenCalledTimes(3));
  await act(async () => finishDetails({ ...project, name: "Old detail name", conversation_count: 2 }));
  expect(screen.getByRole("heading", { name: project.name })).toBeVisible();
  expect(screen.getByRole("button", { name: /First new chat/ })).toBeVisible();
  fireEvent.click(screen.getByRole("button", { name: "New conversation in project" }));
  const card = screen.getByRole("button", { name: project.name });
  await waitFor(() => expect(within(card).getByText("9 chats")).toBeVisible());
  await act(async () => finishSummary({ ...project, conversation_count: 8, name: "Old summary name" }));
  expect(within(card).getByText("9 chats")).toBeVisible();
  expect(screen.getByRole("heading", { name: project.name })).toBeVisible();
  expect(screen.getByText("No project instructions yet.")).toBeVisible();
  expect(screen.getByRole("textbox", { name: "Project knowledge" })).toHaveValue("Keep this unsaved knowledge");
});

it("retains the last summary after a failed refresh and retries when Projects is reopened", async () => {
  vi.mocked(api.getProject).mockResolvedValueOnce(project).mockRejectedValueOnce(new api.ApiError(503, "Unavailable"))
    .mockResolvedValue({ ...project, conversation_count: 8 });
  vi.spyOn(api, "createProjectConversation").mockResolvedValue({ ...conversation, id: "new-thread" });
  const options = props();
  const view = render(<ProjectsWorkspace {...options} />);
  fireEvent.click(await screen.findByRole("button", { name: project.name }));
  fireEvent.click(await screen.findByRole("button", { name: "New conversation in project" }));
  await waitFor(() => expect(api.getProject).toHaveBeenCalledTimes(2));
  const card = screen.getByRole("button", { name: project.name });
  expect(within(card).getByText("7 chats")).toBeVisible();
  expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  view.rerender(<ProjectsWorkspace {...options} visible={false} />);
  view.rerender(<ProjectsWorkspace {...options} visible />);
  await waitFor(() => expect(within(card).getByText("8 chats")).toBeVisible());
});

it("discards a delayed summary after selecting another Project", async () => {
  let finishSummary!: (value: api.AudreyProject) => void;
  vi.mocked(api.getProject).mockResolvedValueOnce(project)
    .mockImplementationOnce(() => new Promise((resolve) => { finishSummary = resolve; }))
    .mockResolvedValue(other);
  vi.spyOn(api, "createProjectConversation").mockResolvedValue({ ...conversation, id: "new-thread" });
  render(<ProjectsWorkspace {...props()} />);
  fireEvent.click(await screen.findByRole("button", { name: project.name }));
  fireEvent.click(await screen.findByRole("button", { name: "New conversation in project" }));
  await waitFor(() => expect(api.getProject).toHaveBeenCalledTimes(2));
  const signal = vi.mocked(api.getProject).mock.calls[1][1];
  fireEvent.click(screen.getByRole("button", { name: other.name }));
  expect(await screen.findByRole("heading", { name: other.name })).toBeVisible();
  await act(async () => finishSummary({ ...project, conversation_count: 99 }));
  expect(signal?.aborted).toBe(true);
  expect(within(screen.getByRole("button", { name: project.name })).getByText("7 chats")).toBeVisible();
  expect(screen.getByRole("heading", { name: other.name })).toBeVisible();
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
