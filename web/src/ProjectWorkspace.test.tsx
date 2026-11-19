import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";

import { NewProjectDialog, ProjectHome } from "./ProjectWorkspace";
import * as api from "./api";

const limits = { max_name_chars: 100, max_instructions_chars: 4000, max_files: 20 };
const project: api.AudreyProject = { id: "project-new", name: "Fern Circle", instructions: "Use garden context.", created_at: "2026-11-04T12:00:00Z", updated_at: "2026-11-04T12:00:00Z" };

afterEach(() => { cleanup(); vi.restoreAllMocks(); });

it("creates a conversation from inside its Project conversations box", async () => {
  vi.spyOn(api, "getProjectKnowledgeText").mockResolvedValue({ text: "", revision: "empty", max_text_chars: 8000 });
  const conversation: api.Conversation = { id: "conversation-new", title: "New conversation", default_model_id: "auto", default_mode: "fast", created_at: project.created_at, updated_at: project.updated_at, project_id: project.id, last_message_at: null, archived_at: null };
  const create = vi.spyOn(api, "createProjectConversation").mockResolvedValue(conversation);
  const onCreated = vi.fn();
  render(<ProjectHome project={project} conversations={[]} files={[]} filesLoading={false} limits={limits}
    defaultModelId="auto" onProjectChange={vi.fn()} onProjectDeleted={vi.fn()} onConversationCreated={onCreated}
    onConversationSelected={vi.fn()} onFilesChange={vi.fn()} />);
  const box = screen.getByRole("tabpanel", { name: "Chats" });
  expect(screen.getAllByRole("button", { name: "New conversation in project" })).toHaveLength(1);
  fireEvent.click(within(box).getByRole("button", { name: "New conversation in project" }));
  await waitFor(() => expect(onCreated).toHaveBeenCalledWith(conversation));
  expect(create).toHaveBeenCalledWith(project.id, "auto");
});

function homeProps() {
  return { project, conversations: [], files: [], filesLoading: false, limits, defaultModelId: "auto",
    onProjectChange: vi.fn(), onProjectDeleted: vi.fn(), onConversationCreated: vi.fn(),
    onConversationSelected: vi.fn(), onFilesChange: vi.fn(), onNewProjectAgent: vi.fn(),
    onOpenProjectAgent: vi.fn(), onNewProjectWatch: vi.fn(), onOpenProjectWatch: vi.fn() };
}

it("shows one activity tab at a time while keeping knowledge edits and files available", async () => {
  vi.spyOn(api, "getProjectKnowledgeText").mockResolvedValue({ text: "Saved garden notes", revision: "v1", max_text_chars: 8000 });
  const save = vi.spyOn(api, "saveProjectKnowledgeText");
  const agents = vi.spyOn(api, "listAgentTasks").mockResolvedValue({ items: [] });
  const watches = vi.spyOn(api, "listWatches").mockResolvedValue({ items: [] });
  render(<ProjectHome {...homeProps()} />);
  const tabs = screen.getByRole("tablist", { name: "Project activity" });
  expect(within(tabs).getByRole("tab", { name: "Chats", selected: true })).toBeVisible();
  expect(screen.getAllByRole("tabpanel")).toHaveLength(1);
  expect(screen.getByRole("tabpanel", { name: "Chats" })).toBeVisible();
  expect(agents).not.toHaveBeenCalled(); expect(watches).not.toHaveBeenCalled();
  const notes = await screen.findByDisplayValue("Saved garden notes");
  fireEvent.change(notes, { target: { value: "Keep this unsaved knowledge draft" } });

  fireEvent.click(within(tabs).getByRole("tab", { name: "Agents" }));
  expect(screen.getAllByRole("tabpanel")).toHaveLength(1);
  expect(screen.getByRole("tabpanel", { name: "Agents" })).toBeVisible();
  expect(screen.queryByRole("button", { name: "New conversation in project" })).not.toBeInTheDocument();
  await waitFor(() => expect(agents).toHaveBeenCalledWith(expect.any(AbortSignal), project.id));
  expect(watches).not.toHaveBeenCalled();

  fireEvent.click(within(tabs).getByRole("tab", { name: "Watches" }));
  expect(screen.getAllByRole("tabpanel")).toHaveLength(1);
  expect(screen.getByRole("tabpanel", { name: "Watches" })).toBeVisible();
  await waitFor(() => expect(watches).toHaveBeenCalledWith(expect.any(AbortSignal), project.id));
  expect(agents.mock.calls[0][0]?.aborted).toBe(true);
  expect(notes).toHaveValue("Keep this unsaved knowledge draft");
  expect(screen.getByRole("region", { name: "Project files" })).toBeVisible();
  expect(screen.getByRole("button", { name: "Upload file" })).toBeVisible();
  expect(save).not.toHaveBeenCalled();

  fireEvent.click(within(tabs).getByRole("tab", { name: "Chats" }));
  expect(notes).toHaveValue("Keep this unsaved knowledge draft");
  expect(screen.getByRole("button", { name: "Save" })).toBeEnabled();
});

it("moves focus and selected content with arrow keys, Home and End", async () => {
  vi.spyOn(api, "getProjectKnowledgeText").mockResolvedValue({ text: "", revision: "empty", max_text_chars: 8000 });
  vi.spyOn(api, "listAgentTasks").mockResolvedValue({ items: [] });
  vi.spyOn(api, "listWatches").mockResolvedValue({ items: [] });
  render(<ProjectHome {...homeProps()} />);
  const tabs = screen.getByRole("tablist", { name: "Project activity" });
  const chats = within(tabs).getByRole("tab", { name: "Chats" });
  chats.focus();
  fireEvent.keyDown(chats, { key: "ArrowRight" });
  const agents = within(tabs).getByRole("tab", { name: "Agents", selected: true });
  expect(agents).toHaveFocus();
  expect(screen.getByRole("tabpanel", { name: "Agents" })).toBeVisible();
  expect(agents.getAttribute("aria-controls")).toBe(screen.getByRole("tabpanel", { name: "Agents" }).id);
  fireEvent.keyDown(agents, { key: "End" });
  const watches = within(tabs).getByRole("tab", { name: "Watches", selected: true });
  expect(watches).toHaveFocus();
  fireEvent.keyDown(watches, { key: "ArrowRight" });
  expect(chats).toHaveFocus();
  expect(chats).toHaveAttribute("aria-selected", "true");
  fireEvent.keyDown(chats, { key: "ArrowLeft" });
  expect(watches).toHaveFocus();
  fireEvent.keyDown(watches, { key: "Home" });
  expect(chats).toHaveFocus();
  expect(chats.tabIndex).toBe(0);
  expect(agents.tabIndex).toBe(-1);
  expect(watches.tabIndex).toBe(-1);
  await waitFor(() => expect(screen.getByRole("textbox", { name: "Project knowledge" })).toBeEnabled());
});

it("returns to Chats when changing Projects without reading hidden activity lists", async () => {
  vi.spyOn(api, "getProjectKnowledgeText").mockResolvedValue({ text: "", revision: "empty", max_text_chars: 8000 });
  const agents = vi.spyOn(api, "listAgentTasks").mockResolvedValue({ items: [] });
  const watches = vi.spyOn(api, "listWatches").mockResolvedValue({ items: [] });
  const props = homeProps();
  const view = render(<ProjectHome {...props} />);
  fireEvent.click(screen.getByRole("tab", { name: "Agents" }));
  await waitFor(() => expect(agents).toHaveBeenCalledWith(expect.any(AbortSignal), project.id));
  view.rerender(<ProjectHome {...props} project={{ ...project, id: "another-project", name: "Another garden" }} />);
  await waitFor(() => expect(screen.getByRole("tab", { name: "Chats" })).toHaveAttribute("aria-selected", "true"));
  expect(screen.getAllByRole("tabpanel")).toHaveLength(1);
  expect(agents).toHaveBeenCalledTimes(1);
  expect(agents.mock.calls[0][0]?.aborted).toBe(true);
  expect(watches).not.toHaveBeenCalled();
});

it("keeps keyboard focus inside the dialog and restores its launcher on close", () => {
  const onClose = vi.fn();
  function View({ open }: { open: boolean }) {
    return <><button type="button">Launch project</button>{open ? <NewProjectDialog limits={limits} privateAvailable={false} onClose={onClose} onCreated={vi.fn()} /> : null}</>;
  }
  const { rerender } = render(<View open={false} />);
  const launcher = screen.getByRole("button", { name: "Launch project" });
  launcher.focus();
  rerender(<View open />);
  const dialog = screen.getByRole("dialog", { name: "New project" });
  const name = within(dialog).getByRole("textbox", { name: "Project name" });
  expect(name).toHaveFocus();
  fireEvent.change(name, { target: { value: "Fern Circle" } });
  const close = within(dialog).getByRole("button", { name: "Close new project" });
  const create = within(dialog).getByRole("button", { name: "Create project" });
  close.focus();
  fireEvent.keyDown(close, { key: "Tab", shiftKey: true });
  expect(create).toHaveFocus();
  fireEvent.keyDown(create, { key: "Tab" });
  expect(close).toHaveFocus();
  fireEvent.keyDown(close, { key: "Escape" });
  expect(onClose).toHaveBeenCalledOnce();
  rerender(<View open={false} />);
  expect(launcher).toHaveFocus();
});

it("saves once, keeps focus contained and blocks dismissal while creating", async () => {
  let complete!: (value: api.AudreyProject) => void;
  const create = vi.spyOn(api, "createProject").mockImplementation(() => new Promise((resolve) => { complete = resolve; }));
  const onClose = vi.fn(), onCreated = vi.fn();
  render(<NewProjectDialog limits={limits} privateAvailable={false} onClose={onClose} onCreated={onCreated} />);
  const name = screen.getByRole("textbox", { name: "Project name" });
  fireEvent.change(name, { target: { value: " Fern Circle " } });
  fireEvent.change(screen.getByRole("textbox", { name: /Project instructions/ }), { target: { value: " Use garden context. " } });
  const form = name.closest("form")!;
  fireEvent.submit(form);
  fireEvent.submit(form);
  expect(create).toHaveBeenCalledExactlyOnceWith("Fern Circle", "Use garden context.", "auto");
  fireEvent.keyDown(name, { key: "Escape" });
  expect(onClose).not.toHaveBeenCalled();
  const instructions = screen.getByRole("textbox", { name: /Project instructions/ });
  instructions.focus();
  fireEvent.keyDown(instructions, { key: "Tab" });
  expect(name).toHaveFocus();
  await act(async () => complete(project));
  expect(onCreated).toHaveBeenCalledExactlyOnceWith(project);
});

it.each(["success", "failure"] as const)("ignores a late %s after the dialog unmounts", async (outcome) => {
  let complete!: (value: api.AudreyProject) => void;
  let fail!: (reason: Error) => void;
  vi.spyOn(api, "createProject").mockImplementation(() => new Promise((resolve, reject) => { complete = resolve; fail = reject; }));
  const onCreated = vi.fn();
  const { unmount } = render(<NewProjectDialog limits={limits} privateAvailable={false} onClose={vi.fn()} onCreated={onCreated} />);
  const name = screen.getByRole("textbox", { name: "Project name" });
  fireEvent.change(name, { target: { value: "Fern Circle" } });
  fireEvent.submit(name.closest("form")!);
  unmount();
  await act(async () => { if (outcome === "success") complete(project); else fail(new Error("Late rejection")); });
  expect(onCreated).not.toHaveBeenCalled();
});
