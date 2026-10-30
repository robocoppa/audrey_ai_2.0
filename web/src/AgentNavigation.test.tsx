import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { AgentsWorkspace } from "./AgentsWorkspace";
import * as api from "./api";
beforeEach(() => {
  vi.spyOn(api, "getWatchUnreadCount").mockResolvedValue({ unread_count: 2 });
  vi.spyOn(api, "listAgentTasks").mockResolvedValue({ items: [] });
  vi.spyOn(api, "listWatches").mockResolvedValue({ items: [] });
  vi.spyOn(api, "listWatchNotifications").mockResolvedValue({ items: [], unread_count: 2 });
});
afterEach(() => { cleanup(); vi.restoreAllMocks(); vi.useRealTimers(); });
it("shows saved unread activity on a fresh Tasks view without loading watch or notification bodies", async () => {
  render(<AgentsWorkspace />);
  expect(await screen.findByLabelText("2 unread notifications")).toBeVisible();
  expect(screen.getByRole("button", { name: "Tasks" })).toHaveAttribute("aria-current", "page");
  expect(api.getWatchUnreadCount).toHaveBeenCalledWith(expect.any(AbortSignal));
  expect(api.listWatches).not.toHaveBeenCalled();
  expect(api.listWatchNotifications).not.toHaveBeenCalled();
});

it("refreshes only the unread count while visible and stops when Agents is hidden", async () => {
  vi.useFakeTimers();
  const view = render(<AgentsWorkspace />);
  await act(async () => { await Promise.resolve(); });
  vi.mocked(api.getWatchUnreadCount).mockResolvedValue({ unread_count: 5 });
  await act(async () => { await vi.advanceTimersByTimeAsync(15000); });
  expect(screen.getByLabelText("5 unread notifications")).toBeVisible();
  view.rerender(<AgentsWorkspace visible={false} />);
  const calls = vi.mocked(api.getWatchUnreadCount).mock.calls.length;
  expect(vi.mocked(api.getWatchUnreadCount).mock.calls.at(-1)?.[0]?.aborted).toBe(true);
  await act(async () => { await vi.advanceTimersByTimeAsync(30000); });
  expect(api.getWatchUnreadCount).toHaveBeenCalledTimes(calls);
  expect(api.listWatches).not.toHaveBeenCalled();
  expect(api.listWatchNotifications).not.toHaveBeenCalled();
});

it("keeps task and watch drafts across Tasks, Watches and Activity navigation", async () => {
  render(<AgentsWorkspace />);
  fireEvent.change(screen.getByRole("textbox", { name: "Research objective" }), { target: { value: "Unsent task objective" } });
  expect(api.listWatches).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole("button", { name: "Watches" }));
  fireEvent.click(await screen.findByRole("button", { name: "Create a watch" }));
  fireEvent.change(screen.getByRole("textbox", { name: "Title" }), { target: { value: "Unsent watch title" } });
  fireEvent.click(screen.getByRole("button", { name: /^Activity/ }));
  expect(await screen.findByRole("heading", { name: "Activity" })).toBeVisible();
  expect(screen.getByLabelText("2 unread notifications")).toBeVisible();
  fireEvent.click(screen.getByRole("button", { name: "Watches" }));
  expect(screen.getByRole("textbox", { name: "Title" })).toHaveValue("Unsent watch title");
  fireEvent.click(screen.getByRole("button", { name: "Tasks" }));
  expect(screen.getByRole("textbox", { name: "Research objective" })).toHaveValue("Unsent task objective");
  fireEvent.click(screen.getByRole("button", { name: "Watches" }));
  expect(screen.getByRole("textbox", { name: "Title" })).toHaveValue("Unsent watch title");
});
