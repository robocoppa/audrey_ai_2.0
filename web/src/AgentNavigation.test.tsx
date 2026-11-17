import { act, cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { AgentsWorkspace } from "./AgentsWorkspace";
import * as api from "./api";
beforeEach(() => {
  vi.spyOn(api, "listProjects").mockResolvedValue({ items: [], next_cursor: null, limits: { max_name_chars: 100, max_instructions_chars: 4000, max_files: 20 } });
  vi.spyOn(api, "getActivityUnreadCount").mockResolvedValue({ unread_count: 2 });
  vi.spyOn(api, "listAgentTasks").mockResolvedValue({ items: [] });
  vi.spyOn(api, "listWatches").mockResolvedValue({ items: [] });
  vi.spyOn(api, "listActivity").mockResolvedValue({ items: [], unread_count: 2 });
});
afterEach(() => { cleanup(); vi.restoreAllMocks(); vi.useRealTimers(); });
it("opens Agents history with unread updates and keeps creation and watch bodies closed", async () => {
  render(<AgentsWorkspace />);
  expect(await screen.findByLabelText("2 unread notifications")).toBeVisible();
  const sections = within(screen.getByRole("navigation", { name: "Agent sections" }));
  expect(sections.getByRole("button", { name: /^Agents/ })).toHaveAttribute("aria-current", "page");
  expect(sections.getAllByRole("button")).toHaveLength(2);
  expect(sections.queryByRole("button", { name: "Tasks" })).not.toBeInTheDocument();
  expect(sections.queryByRole("button", { name: "New" })).not.toBeInTheDocument();
  expect(screen.getByRole("button", { name: "New agent" })).toBeVisible();
  expect(screen.queryByRole("textbox", { name: "Task objective" })).not.toBeInTheDocument();
  expect(api.getActivityUnreadCount).toHaveBeenCalledWith(expect.any(AbortSignal));
  expect(api.listWatches).not.toHaveBeenCalled();
  expect(api.listActivity).not.toHaveBeenCalled();
});

it("refreshes only the unread count while visible and stops when Agents is hidden", async () => {
  vi.useFakeTimers();
  const view = render(<AgentsWorkspace />);
  await act(async () => { await Promise.resolve(); });
  vi.mocked(api.getActivityUnreadCount).mockResolvedValue({ unread_count: 5 });
  await act(async () => { await vi.advanceTimersByTimeAsync(15000); });
  expect(screen.getByLabelText("5 unread notifications")).toBeVisible();
  view.rerender(<AgentsWorkspace visible={false} />);
  const calls = vi.mocked(api.getActivityUnreadCount).mock.calls.length;
  expect(vi.mocked(api.getActivityUnreadCount).mock.calls.at(-1)?.[0]?.aborted).toBe(true);
  await act(async () => { await vi.advanceTimersByTimeAsync(30000); });
  expect(api.getActivityUnreadCount).toHaveBeenCalledTimes(calls);
  expect(api.listWatches).not.toHaveBeenCalled();
  expect(api.listActivity).not.toHaveBeenCalled();
});

it("keeps Agent and Watch drafts while their tabs return to history and Updates stays inside Agents", async () => {
  render(<AgentsWorkspace />);
  fireEvent.click(screen.getByRole("button", { name: "New agent" }));
  fireEvent.change(screen.getByRole("textbox", { name: "Task objective" }), { target: { value: "Unsent task objective" } });
  expect(api.listWatches).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole("button", { name: "Watches" }));
  fireEvent.click(await screen.findByRole("button", { name: "New watch" }));
  fireEvent.change(screen.getByRole("textbox", { name: "Title" }), { target: { value: "Unsent watch title" } });
  fireEvent.click(within(screen.getByRole("navigation", { name: "Agent sections" })).getByRole("button", { name: /^Agents/ }));
  expect(screen.queryByRole("textbox", { name: "Task objective" })).not.toBeInTheDocument();
  const updates = document.querySelector<HTMLDetailsElement>(".agents-task-updates")!;
  fireEvent.click(updates.querySelector("summary")!);
  expect(await screen.findByRole("heading", { name: "Updates" })).toBeVisible();
  expect(api.listActivity).toHaveBeenCalledWith(expect.any(AbortSignal));
  expect(screen.queryByRole("button", { name: /^Activity/ })).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "New agent" }));
  expect(screen.getByRole("textbox", { name: "Task objective" })).toHaveValue("Unsent task objective");
  fireEvent.click(screen.getByRole("button", { name: "Watches" }));
  // The Watch workspace retains its draft without creating or checking a page.
  fireEvent.click(screen.getByRole("button", { name: "New watch" }));
  expect(screen.getByRole("textbox", { name: "Title" })).toHaveValue("Unsent watch title");
});
