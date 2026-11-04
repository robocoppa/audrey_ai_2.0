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
  const box = screen.getByRole("region", { name: "Project conversations" });
  expect(screen.getAllByRole("button", { name: "New conversation in project" })).toHaveLength(1);
  fireEvent.click(within(box).getByRole("button", { name: "New conversation in project" }));
  await waitFor(() => expect(onCreated).toHaveBeenCalledWith(conversation));
  expect(create).toHaveBeenCalledWith(project.id, "auto");
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
