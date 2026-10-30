import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { AgentsWorkspace } from "./AgentsWorkspace";
import * as api from "./api";
import type { AgentTask } from "./api";

function task(patch: Partial<AgentTask> = {}): AgentTask {
  return {
    id: "agt_first", objective: "Compare public documentation", status: "running", revision: 2,
    created_at: "2026-10-28T16:00:00Z", updated_at: "2026-10-28T16:01:00Z", attempts: 1,
    steps: [
      { id: "plan", label: "Plan the research", status: "completed", result: "Compare the official sources." },
      { id: "research", label: "Research public sources", status: "running", result: "" },
      { id: "review", label: "Review the report", status: "pending", result: "" },
    ],
    result: "", error: "", ...patch,
  };
}

beforeEach(() => {
  vi.spyOn(api, "listAgentTasks").mockResolvedValue({ items: [] });
  vi.spyOn(api, "getAgentTask").mockResolvedValue(task());
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
  vi.useRealTimers();
});

describe("AgentsWorkspace", () => {
  it("starts a saved public research task from the objective", async () => {
    const created = task({ status: "queued", revision: 1 });
    const create = vi.spyOn(api, "createAgentTask").mockImplementation(async () => {
      vi.mocked(api.listAgentTasks).mockResolvedValue({ items: [created] });
      return created;
    });
    vi.mocked(api.getAgentTask).mockResolvedValue(created);
    render(<AgentsWorkspace />);
    expect(await screen.findByText("No agents yet. Start with a research objective above.")).toBeVisible();
    expect(screen.getByRole("button", { name: "Start agent" })).toBeDisabled();
    fireEvent.change(screen.getByRole("textbox", { name: "Research objective" }), { target: { value: "  Compare public documentation  " } });
    fireEvent.click(screen.getByRole("button", { name: "Start agent" }));
    expect(await screen.findByRole("heading", { name: created.objective })).toBeVisible();
    expect(create).toHaveBeenCalledWith(created.objective, expect.any(String));
    expect(screen.getByRole("textbox", { name: "Research objective" })).toHaveValue("");
    expect(screen.getByRole("heading", { name: "Plan and progress" })).toBeVisible();
    expect(screen.getByText("Research public sources", { selector: "strong" })).toBeVisible();
    expect(screen.queryByRole("button", { name: "Schedule" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Delete task" })).not.toBeInTheDocument();
  });

  it("keeps the same creation identity when retrying after a lost response", async () => {
    const create = vi.spyOn(api, "createAgentTask")
      .mockRejectedValueOnce(new TypeError("The connection was lost."))
      .mockImplementationOnce(async () => {
        vi.mocked(api.listAgentTasks).mockResolvedValue({ items: [task()] });
        return task();
      });
    render(<AgentsWorkspace />);
    await screen.findByText("No agents yet. Start with a research objective above.");
    fireEvent.change(screen.getByRole("textbox", { name: "Research objective" }), { target: { value: "Compare public documentation" } });
    fireEvent.click(screen.getByRole("button", { name: "Start agent" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("The connection was lost.");
    fireEvent.click(screen.getByRole("button", { name: "Start agent" }));
    await screen.findByRole("heading", { name: "Compare public documentation" });
    expect(create.mock.calls[0][1]).toBe(create.mock.calls[1][1]);
  });

  it("uses saved revisions for pause, resume, and cancellation", async () => {
    let current = task();
    vi.mocked(api.listAgentTasks).mockImplementation(async () => ({ items: [current] }));
    vi.mocked(api.getAgentTask).mockImplementation(async () => current);
    const action = vi.spyOn(api, "actOnAgentTask").mockImplementation(async (_id, value) => {
      current = { ...current, revision: current.revision + 1, status: value === "pause" ? "paused" : value === "resume" ? "queued" : "cancelled" };
      return current;
    });
    render(<AgentsWorkspace />);
    const sidebar = within(screen.getByRole("complementary", { name: "Agent tasks" }));
    const detail = within(screen.getByRole("region", { name: "Selected agent" }));
    expect(sidebar.getByRole("textbox", { name: "Research objective" })).toBeVisible();
    fireEvent.click(await sidebar.findByRole("button", { name: "Pause" }));
    fireEvent.click(await sidebar.findByRole("button", { name: "Resume" }));
    fireEvent.click(await sidebar.findByRole("button", { name: "Cancel" }));
    expect(await detail.findByRole("button", { name: "Run again" })).toBeEnabled();
    expect(detail.getByRole("button", { name: "Delete task" })).toBeVisible();
    expect(sidebar.queryByRole("button", { name: "Run again" })).not.toBeInTheDocument();
    expect(action.mock.calls).toEqual([
      ["agt_first", "pause", 2], ["agt_first", "resume", 3], ["agt_first", "cancel", 4],
    ]);
  });

  it("opens a selected saved agent at the top of the workflow without moving sidebar focus", async () => {
    const first = task();
    const second = task({ id: "agt_second", objective: "A different research report" });
    vi.mocked(api.listAgentTasks).mockResolvedValue({ items: [first, second] });
    vi.mocked(api.getAgentTask).mockImplementation(async (id) => id === first.id ? first : second);
    render(<AgentsWorkspace />);
    await screen.findByRole("heading", { name: first.objective });
    const workflow = screen.getByRole("region", { name: "Selected agent" });
    workflow.scrollTop = 800;
    const savedTask = screen.getByRole("button", { name: `Open agent: ${second.objective}` });
    savedTask.focus();
    fireEvent.click(savedTask);
    expect(await screen.findByRole("heading", { name: second.objective })).toBeVisible();
    await waitFor(() => expect(workflow.scrollTop).toBe(0));
    expect(savedTask).toHaveFocus();
    expect(workflow).not.toContainElement(screen.getByRole("textbox", { name: "Research objective" }));
  });

  it("keeps mobile controls in a dismissible focus-trapped drawer and opens the workflow on selection", async () => {
    vi.stubGlobal("matchMedia", vi.fn(() => ({ matches: true, addEventListener: vi.fn(), removeEventListener: vi.fn() })));
    const saved = task();
    vi.mocked(api.listAgentTasks).mockResolvedValue({ items: [saved] });
    render(<AgentsWorkspace />);
    await screen.findByRole("heading", { name: saved.objective });
    const toggle = screen.getByRole("button", { name: "Agent controls" });
    expect(toggle).toHaveAttribute("aria-expanded", "false");
    expect(screen.queryByRole("dialog", { name: "Agent tasks" })).not.toBeInTheDocument();
    fireEvent.click(toggle);
    const drawer = screen.getByRole("dialog", { name: "Agent tasks" });
    const close = within(drawer).getByRole("button", { name: "Close agent controls" });
    const savedTask = within(drawer).getByRole("button", { name: `Open agent: ${saved.objective}` });
    expect(drawer).toHaveAttribute("aria-modal", "true");
    expect(screen.getByRole("region", { name: "Selected agent" })).toHaveAttribute("inert");
    expect(close).toHaveFocus();
    fireEvent.keyDown(close, { key: "Tab", shiftKey: true });
    expect(savedTask).toHaveFocus();
    fireEvent.keyDown(savedTask, { key: "Tab" });
    expect(close).toHaveFocus();
    fireEvent.keyDown(close, { key: "Escape" });
    await waitFor(() => expect(toggle).toHaveFocus());
    expect(toggle).toHaveAttribute("aria-expanded", "false");
    fireEvent.click(toggle);
    fireEvent.click(within(screen.getByRole("dialog", { name: "Agent tasks" })).getByRole("button", { name: `Open agent: ${saved.objective}` }));
    expect(toggle).toHaveAttribute("aria-expanded", "false");
    const workflow = screen.getByRole("region", { name: "Selected agent" });
    await waitFor(() => expect(workflow).toHaveFocus());
    expect(workflow).not.toHaveAttribute("inert");
  });

  it("keeps a failed mobile creation visible inside the controls drawer", async () => {
    vi.stubGlobal("matchMedia", vi.fn(() => ({ matches: true, addEventListener: vi.fn(), removeEventListener: vi.fn() })));
    vi.spyOn(api, "createAgentTask").mockRejectedValue(new Error("No local worker is available."));
    render(<AgentsWorkspace />);
    fireEvent.click(screen.getByRole("button", { name: "Agent controls" }));
    const controls = within(screen.getByRole("dialog", { name: "Agent tasks" }));
    fireEvent.change(controls.getByRole("textbox", { name: "Research objective" }), { target: { value: "Research public sources" } });
    fireEvent.click(controls.getByRole("button", { name: "Start agent" }));
    expect(await controls.findByRole("alert")).toHaveTextContent("No local worker is available.");
    expect(screen.getByRole("dialog", { name: "Agent tasks" })).toHaveAttribute("aria-modal", "true");
  });

  it("does not close Chat navigation when hidden Agents crosses its sidebar breakpoint", () => {
    let resize: ((event: MediaQueryListEvent) => void) | undefined;
    vi.stubGlobal("matchMedia", vi.fn(() => ({
      matches: true,
      addEventListener: (_name: string, listener: (event: MediaQueryListEvent) => void) => { resize = listener; },
      removeEventListener: vi.fn(),
    })));
    const navigationChanged = vi.fn();
    render(<AgentsWorkspace visible={false} navigationOpen onNavigationOpenChange={navigationChanged} />);
    act(() => resize?.({ matches: false } as MediaQueryListEvent));
    expect(navigationChanged).not.toHaveBeenCalled();
  });

  it("reloads a revision conflict without repeating the user's action", async () => {
    vi.mocked(api.listAgentTasks).mockResolvedValue({ items: [task()] });
    const action = vi.spyOn(api, "actOnAgentTask").mockRejectedValue(new api.ApiError(409, "Revision changed"));
    render(<AgentsWorkspace />);
    const pause = await screen.findByRole("button", { name: "Pause" });
    await waitFor(() => expect(api.getAgentTask).toHaveBeenCalled());
    vi.mocked(api.getAgentTask).mockResolvedValue(task({ status: "completed", revision: 5, result: "Reviewed report." }));
    fireEvent.click(pause);
    expect(await screen.findByText(/This agent changed while you were acting/u)).toBeVisible();
    expect(screen.getByRole("button", { name: "Run again" })).toBeEnabled();
    expect(screen.getByText("Reviewed report.")).toBeVisible();
    expect(action).toHaveBeenCalledOnce();
  });

  it.each(["failed", "completed", "cancelled"] as const)("runs the same %s agent again with its saved plan", async (status) => {
    const steps = task().steps.map((step) => status === "completed"
      ? { ...step, status: "completed" as const, result: step.result || `Saved ${step.id}.` }
      : step.id === "plan" ? step : { ...step, status: "pending" as const });
    const original = task({
      status, revision: 7, result: status === "completed" ? "Original report." : "",
      error: status === "failed" ? "Research stopped." : "", steps,
    });
    let current = original;
    vi.mocked(api.listAgentTasks).mockImplementation(async () => ({ items: [current] }));
    vi.mocked(api.getAgentTask).mockImplementation(async () => current);
    const create = vi.spyOn(api, "createAgentTask");
    const action = vi.spyOn(api, "actOnAgentTask").mockImplementation(async () => {
      current = {
        ...original, status: "queued", revision: 8, result: "", error: "",
        steps: original.steps.map((step) => step.id === "plan" ? step : { ...step, status: "pending", result: "" }),
      };
      return current;
    });
    render(<AgentsWorkspace />);
    const runAgain = await screen.findByRole("button", { name: "Run again" });
    fireEvent.click(screen.getByRole("button", { name: "Finished" }));
    fireEvent.click(runAgain);
    expect(await screen.findByRole("button", { name: "Pause" })).toBeEnabled();
    expect(action).toHaveBeenCalledExactlyOnceWith(original.id, "rerun", 7);
    expect(create).not.toHaveBeenCalled();
    expect(screen.getByRole("button", { name: /^All$/u })).toHaveAttribute("aria-pressed", "true");
    const cards = screen.getAllByRole("button", { name: `Open agent: ${original.objective}` });
    expect(cards).toHaveLength(1);
    expect(cards[0]).toHaveAttribute("aria-pressed", "true");
    expect(cards[0]).toHaveTextContent("Queued");
    expect(screen.getByRole("heading", { name: original.objective })).toBeVisible();
    fireEvent.click(screen.getByText("View saved plan"));
    expect(screen.getByText(original.steps[0].result)).toBeVisible();
    expect(screen.queryByText("Original report.")).not.toBeInTheDocument();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("reloads a rerun conflict on the same agent without creating or retrying a task", async () => {
    let current = task({ status: "failed", revision: 7, error: "Research stopped." });
    vi.mocked(api.listAgentTasks).mockImplementation(async () => ({ items: [current] }));
    vi.mocked(api.getAgentTask).mockImplementation(async () => current);
    const create = vi.spyOn(api, "createAgentTask");
    const action = vi.spyOn(api, "actOnAgentTask").mockImplementation(async () => {
      current = { ...current, status: "running", revision: 8, error: "" };
      throw new api.ApiError(409, "Revision changed");
    });
    render(<AgentsWorkspace />);
    fireEvent.click(await screen.findByRole("button", { name: "Run again" }));
    expect(await screen.findByText(/This agent changed while you were acting/u)).toBeVisible();
    expect(screen.getByRole("button", { name: "Pause" })).toBeEnabled();
    expect(action).toHaveBeenCalledExactlyOnceWith(current.id, "rerun", 7);
    expect(create).not.toHaveBeenCalled();
    expect(screen.getAllByRole("button", { name: `Open agent: ${current.objective}` })).toHaveLength(1);
    expect(screen.getByRole("button", { name: `Open agent: ${current.objective}` })).toHaveAttribute("aria-pressed", "true");
  });

  it("confirms terminal task deletion and prevents stale listings from restoring it", async () => {
    const saved = task({ status: "completed", revision: 8, result: "Saved report." });
    vi.mocked(api.listAgentTasks).mockResolvedValue({ items: [saved] });
    vi.mocked(api.getAgentTask).mockResolvedValue(saved);
    const remove = vi.spyOn(api, "deleteAgentTask").mockResolvedValue();
    render(<AgentsWorkspace />);
    const deleteButton = await screen.findByRole("button", { name: "Delete task" });
    expect(deleteButton).toHaveClass("danger-button");
    expect(deleteButton).toHaveAttribute("aria-pressed", "false");
    fireEvent.click(deleteButton);
    expect(remove).not.toHaveBeenCalled();
    const confirmButton = screen.getByRole("button", { name: "Confirm delete task" });
    expect(confirmButton).toBe(deleteButton);
    expect(confirmButton).toHaveTextContent("✓");
    expect(confirmButton).toHaveClass("danger-button", "confirming-delete");
    expect(confirmButton).toHaveAttribute("aria-pressed", "true");
    expect(screen.queryByRole("group", { name: "Confirm task deletion" })).not.toBeInTheDocument();
    fireEvent.click(confirmButton);
    expect(await screen.findByText("No agents yet. Start with a research objective above.")).toBeVisible();
    expect(remove).toHaveBeenCalledWith(saved.id, 8);
    expect(screen.queryByRole("heading", { name: saved.objective })).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Refresh agent tasks" }));
    await waitFor(() => expect(screen.getByRole("button", { name: "Refresh agent tasks" })).toBeEnabled());
    expect(screen.queryByRole("button", { name: `Open agent: ${saved.objective}` })).not.toBeInTheDocument();
  });

  it.each(["blur", "Escape"])("cancels the delete checkmark on %s", async (event) => {
    const saved = task({ status: "completed" });
    vi.mocked(api.listAgentTasks).mockResolvedValue({ items: [saved] });
    vi.mocked(api.getAgentTask).mockResolvedValue(saved);
    const remove = vi.spyOn(api, "deleteAgentTask").mockResolvedValue();
    render(<AgentsWorkspace />);
    const deleteButton = await screen.findByRole("button", { name: "Delete task" });
    fireEvent.click(deleteButton);
    expect(deleteButton).toHaveTextContent("✓");
    if (event === "blur") fireEvent.blur(deleteButton);
    else fireEvent.keyDown(deleteButton, { key: "Escape" });
    expect(screen.getByRole("button", { name: "Delete task" })).toBe(deleteButton);
    expect(deleteButton).toHaveAttribute("aria-pressed", "false");
    expect(remove).not.toHaveBeenCalled();
    fireEvent.click(deleteButton);
    expect(remove).not.toHaveBeenCalled();
    expect(screen.getByRole("button", { name: "Confirm delete task" })).toHaveTextContent("✓");
  });

  it("requires a fresh delete confirmation when switching saved agents", async () => {
    const first = task({ status: "completed" });
    const second = task({ id: "agt_second", objective: "Another saved task", status: "completed" });
    vi.mocked(api.listAgentTasks).mockResolvedValue({ items: [first, second] });
    vi.mocked(api.getAgentTask).mockImplementation(async (id) => id === first.id ? first : second);
    const remove = vi.spyOn(api, "deleteAgentTask").mockResolvedValue();
    render(<AgentsWorkspace />);
    fireEvent.click(await screen.findByRole("button", { name: "Delete task" }));
    expect(screen.getByRole("button", { name: "Confirm delete task" })).toHaveTextContent("✓");
    fireEvent.click(screen.getByRole("button", { name: `Open agent: ${second.objective}` }));
    expect(screen.getByRole("button", { name: "Delete task" })).toHaveAttribute("aria-pressed", "false");
    fireEvent.click(screen.getByRole("button", { name: `Open agent: ${first.objective}` }));
    expect(screen.getByRole("button", { name: "Delete task" })).toHaveAttribute("aria-pressed", "false");
    expect(remove).not.toHaveBeenCalled();
  });

  it("clears a pending delete confirmation when leaving the Agents workspace", async () => {
    const saved = task({ status: "completed" });
    vi.mocked(api.listAgentTasks).mockResolvedValue({ items: [saved] });
    vi.mocked(api.getAgentTask).mockResolvedValue(saved);
    const { rerender } = render(<AgentsWorkspace />);
    fireEvent.click(await screen.findByRole("button", { name: "Delete task" }));
    expect(screen.getByRole("button", { name: "Confirm delete task" })).toHaveTextContent("✓");
    rerender(<AgentsWorkspace visible={false} />);
    rerender(<AgentsWorkspace visible />);
    expect(screen.getByRole("button", { name: "Delete task" })).toHaveAttribute("aria-pressed", "false");
  });

  it("requires a fresh confirmation if a refreshed task revision changes", async () => {
    let saved = task({ status: "completed" });
    vi.mocked(api.listAgentTasks).mockImplementation(async () => ({ items: [saved] }));
    vi.mocked(api.getAgentTask).mockImplementation(async () => saved);
    const remove = vi.spyOn(api, "deleteAgentTask").mockResolvedValue();
    render(<AgentsWorkspace />);
    fireEvent.click(await screen.findByRole("button", { name: "Delete task" }));
    saved = { ...saved, revision: saved.revision + 1 };
    await waitFor(() => expect(screen.getByRole("button", { name: "Refresh agent tasks" })).toBeEnabled());
    fireEvent.click(screen.getByRole("button", { name: "Refresh agent tasks" }));
    await screen.findByRole("button", { name: "Delete task" });
    expect(screen.queryByRole("button", { name: "Confirm delete task" })).not.toBeInTheDocument();
    expect(remove).not.toHaveBeenCalled();
  });

  it("reloads a deletion conflict and requires a fresh confirmation", async () => {
    const saved = task({ status: "completed" });
    vi.mocked(api.listAgentTasks).mockResolvedValue({ items: [saved] });
    vi.mocked(api.getAgentTask).mockResolvedValue(saved);
    const remove = vi.spyOn(api, "deleteAgentTask").mockRejectedValue(new api.ApiError(409, "Revision changed"));
    render(<AgentsWorkspace />);
    fireEvent.click(await screen.findByRole("button", { name: "Delete task" }));
    vi.mocked(api.getAgentTask).mockResolvedValue({ ...saved, revision: 6 });
    fireEvent.click(screen.getByRole("button", { name: "Confirm delete task" }));
    await screen.findByText(/This agent changed while you were acting/u);
    expect(remove).toHaveBeenCalledOnce();
    expect(screen.queryByRole("button", { name: "Confirm delete task" })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Delete task" })).toBeEnabled();
  });

  it("retains selection and newer revisions when an older refresh completes", async () => {
    const first = task();
    const second = task({ id: "agt_second", objective: "Check release documentation", revision: 3 });
    vi.mocked(api.listAgentTasks).mockResolvedValue({ items: [first, second] });
    vi.mocked(api.getAgentTask).mockImplementation(async (id) => id === first.id ? first : second);
    vi.spyOn(api, "actOnAgentTask").mockResolvedValue({ ...second, status: "paused", revision: 4 });
    render(<AgentsWorkspace />);
    fireEvent.click(await screen.findByRole("button", { name: `Open agent: ${second.objective}` }));
    await screen.findByRole("heading", { name: second.objective });
    let finishRefresh!: (value: { items: AgentTask[] }) => void;
    vi.mocked(api.listAgentTasks).mockImplementationOnce(() => new Promise((resolve) => { finishRefresh = resolve; }));
    await waitFor(() => expect(screen.getByRole("button", { name: "Refresh agent tasks" })).toBeEnabled());
    fireEvent.click(screen.getByRole("button", { name: "Refresh agent tasks" }));
    fireEvent.click(screen.getByRole("button", { name: "Pause" }));
    await screen.findByRole("button", { name: "Resume" });
    await act(async () => { finishRefresh({ items: [first, second] }); });
    expect(screen.getByRole("heading", { name: second.objective })).toBeVisible();
    expect(screen.getByRole("button", { name: "Resume" })).toBeEnabled();
  });

  it("removes a task deleted in another session when the full catalog refreshes", async () => {
    vi.mocked(api.listAgentTasks).mockResolvedValue({ items: [task()] });
    const count = vi.fn();
    render(<AgentsWorkspace onActiveCountChange={count} />);
    await screen.findByRole("heading", { name: task().objective });
    await waitFor(() => expect(screen.getByRole("button", { name: "Refresh agent tasks" })).toBeEnabled());
    vi.mocked(api.listAgentTasks).mockResolvedValue({ items: [] });
    fireEvent.click(screen.getByRole("button", { name: "Refresh agent tasks" }));
    expect(await screen.findByText("No agents yet. Start with a research objective above.")).toBeVisible();
    expect(screen.queryByRole("heading", { name: task().objective })).not.toBeInTheDocument();
    expect(count).toHaveBeenLastCalledWith(0);
  });

  it("polls active tasks, reports the active count, and stops polling on unmount", async () => {
    vi.useFakeTimers();
    let current = task();
    vi.mocked(api.listAgentTasks).mockImplementation(async () => ({ items: [{ ...current, summary: true, result: "", steps: current.steps.map((step) => ({ ...step, result: "" })) }] }));
    vi.mocked(api.getAgentTask).mockImplementation(async () => current);
    const count = vi.fn();
    const { unmount } = render(<AgentsWorkspace onActiveCountChange={count} />);
    await act(async () => { await Promise.resolve(); });
    expect(count).toHaveBeenLastCalledWith(1);
    current = { ...current, status: "completed", revision: 4, result: "The completed report." };
    await act(async () => { await vi.advanceTimersByTimeAsync(3000); });
    expect(screen.getByText("The completed report.")).toBeVisible();
    expect(count).toHaveBeenLastCalledWith(0);
    unmount();
    const calls = vi.mocked(api.listAgentTasks).mock.calls.length;
    await act(async () => { await vi.advanceTimersByTimeAsync(6000); });
    expect(api.listAgentTasks).toHaveBeenCalledTimes(calls);
  });

  it("hydrates summary records from detail and preserves saved bodies during a refresh", async () => {
    const full = task({ status: "completed", result: "The saved detailed report." });
    const summary = { ...full, summary: true, result: "", steps: full.steps.map((step) => ({ ...step, result: "" })) };
    vi.mocked(api.listAgentTasks).mockResolvedValue({ items: [summary] });
    vi.mocked(api.getAgentTask).mockResolvedValue(full);
    render(<AgentsWorkspace />);
    expect(await screen.findByText("The saved detailed report.")).toBeVisible();
    fireEvent.click(screen.getByText("View saved plan"));
    expect(screen.getByText("Compare the official sources.")).toBeVisible();
    vi.mocked(api.getAgentTask).mockImplementation(() => new Promise(() => {}));
    fireEvent.click(screen.getByRole("button", { name: "Refresh agent tasks" }));
    await waitFor(() => expect(screen.getByRole("button", { name: "Refresh agent tasks" })).toBeEnabled());
    expect(screen.getByText("The saved detailed report.")).toBeVisible();
    expect(screen.getByText("Compare the official sources.")).toBeVisible();
  });

  it("recovers from a failed list refresh and renders report Markdown safely", async () => {
    vi.mocked(api.listAgentTasks).mockRejectedValueOnce(new Error("Temporarily unavailable"));
    const saved = task({ status: "completed", result: "## Findings\n[Source](https://example.com/report)\n<script>alert('unsafe')</script>\n[unsafe](javascript:alert)\n![tracking](https://example.com/tracker.png)" });
    vi.mocked(api.getAgentTask).mockResolvedValue(saved);
    render(<AgentsWorkspace />);
    expect(await screen.findByRole("alert")).toHaveTextContent("Temporarily unavailable");
    vi.mocked(api.listAgentTasks).mockResolvedValue({ items: [saved] });
    fireEvent.click(screen.getByRole("button", { name: "Refresh agent tasks" }));
    expect(await screen.findByRole("heading", { name: "Findings" })).toBeVisible();
    expect(screen.getByRole("link", { name: "Source" })).toHaveAttribute("rel", "noopener noreferrer");
    expect(document.querySelector("script")).not.toBeInTheDocument();
    expect(screen.queryByRole("img")).not.toBeInTheDocument();
    expect(screen.getByText("unsafe", { selector: "a" })).not.toHaveAttribute("href", expect.stringContaining("javascript:"));
  });
});
