import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";

import { PrivateTaskProgress } from "./PrivateTaskProgress";
import { actOnPrivateTask, getPrivateTask, type PrivateTask } from "./privateTasksApi";

vi.mock("./privateTasksApi", () => ({ getPrivateTask: vi.fn(), actOnPrivateTask: vi.fn() }));

function task(changes: Partial<PrivateTask> = {}): PrivateTask {
  return {
    id: "task_saved", conversation_id: "con_private", revision: 7,
    status: "running", stop_reason: "", progress: 1, updated_at: "2026-10-22T12:00:00Z", native_run_id: "run_saved",
    usage: { steps: 1, wall_seconds: 20, model_tokens: 50, tool_calls: 2 },
    limits: { max_steps: 24, max_wall_seconds: 600, max_model_tokens: 32000, max_tool_calls: 64 },
    ...changes,
  };
}

function props() {
  return { conversationId: "con_private", refreshVersion: 0, disabled: false, hasReview: false, onReview: vi.fn(), onChanged: vi.fn() };
}

beforeEach(() => { vi.mocked(getPrivateTask).mockResolvedValue({ task: null }); });
afterEach(() => { cleanup(); vi.clearAllMocks(); vi.useRealTimers(); });

it("loads saved task before offering generation and resumes only after an explicit choice", async () => {
  let resolve!: (value: { task: PrivateTask }) => void;
  vi.mocked(getPrivateTask).mockReturnValue(new Promise((value) => { resolve = value; }));
  const callbacks = props();
  render(<PrivateTaskProgress {...callbacks} />);
  expect(screen.queryByRole("button", { name: /Start|Resume/u })).not.toBeInTheDocument();
  await act(async () => resolve({ task: task({ status: "paused", stop_reason: "server_restart" }) }));
  expect(screen.getByRole("status")).toHaveTextContent("Review paused after a restart");
  expect(callbacks.onReview).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole("button", { name: "Resume review" }));
  expect(callbacks.onReview).toHaveBeenCalledOnce();
  expect(screen.getByLabelText("Saved review progress")).not.toHaveTextContent(/task_saved|run_saved|32000/u);
});

it("pauses an active native review with the current saved revision even while generation is disabled", async () => {
  vi.mocked(getPrivateTask).mockResolvedValue({ task: task() });
  vi.mocked(actOnPrivateTask).mockResolvedValue({ task: task({ revision: 8, status: "paused" }) });
  const callbacks = props();
  render(<PrivateTaskProgress {...callbacks} disabled />);
  fireEvent.click(await screen.findByRole("button", { name: "Pause" }));
  await waitFor(() => expect(actOnPrivateTask).toHaveBeenCalledWith("task_saved", 7, "pause", expect.any(AbortSignal)));
  expect(await screen.findByRole("button", { name: "Resume review" })).toBeDisabled();
  expect(callbacks.onChanged).toHaveBeenCalledOnce();
  expect(callbacks.onReview).not.toHaveBeenCalled();
});

it("cancels queued work without issuing another review turn", async () => {
  vi.mocked(getPrivateTask).mockResolvedValue({ task: task({ status: "queued" }) });
  vi.mocked(actOnPrivateTask).mockResolvedValue({ task: task({ revision: 8, status: "cancelled" }) });
  const callbacks = props();
  render(<PrivateTaskProgress {...callbacks} disabled />);
  fireEvent.click(await screen.findByRole("button", { name: "Cancel" }));
  expect(await screen.findByText("Review stopped · Saved work is retained")).toBeVisible();
  expect(actOnPrivateTask).toHaveBeenCalledWith("task_saved", 7, "cancel", expect.any(AbortSignal));
  expect(callbacks.onReview).not.toHaveBeenCalled();
});

it("keeps spent-budget work readable without a restart or resume button", async () => {
  vi.mocked(getPrivateTask).mockResolvedValue({ task: task({ status: "paused", stop_reason: "budget_exhausted" }) });
  render(<PrivateTaskProgress {...props()} />);
  expect(await screen.findByText("Review limit reached · Saved work is retained")).toBeVisible();
  expect(screen.queryByRole("button", { name: /Resume|Start|Continue/u })).not.toBeInTheDocument();
});

it("uses the normal review callback after input is saved and offers only one continue control", async () => {
  vi.mocked(getPrivateTask).mockResolvedValue({ task: task({ status: "waiting_for_user" }) });
  const callbacks = props();
  render(<PrivateTaskProgress {...callbacks} />);
  fireEvent.click(await screen.findByRole("button", { name: "Continue review" }));
  expect(callbacks.onReview).toHaveBeenCalledOnce();
  expect(screen.getAllByRole("button", { name: "Continue review" })).toHaveLength(1);
  expect(screen.queryByRole("button", { name: "Start review" })).not.toBeInTheDocument();
});

it("polls serially and refreshes saved case data when the task finishes", async () => {
  vi.useFakeTimers();
  vi.mocked(getPrivateTask).mockResolvedValueOnce({ task: task() })
    .mockResolvedValue({ task: task({ revision: 8, status: "completed" }) });
  const callbacks = props();
  render(<PrivateTaskProgress {...callbacks} />);
  await act(async () => { await Promise.resolve(); });
  expect(getPrivateTask).toHaveBeenCalledTimes(1);
  await act(async () => { await vi.advanceTimersByTimeAsync(2000); });
  expect(getPrivateTask).toHaveBeenCalledTimes(2);
  expect(screen.getByRole("status")).toHaveTextContent("Review saved");
  expect(callbacks.onChanged).toHaveBeenCalledOnce();
  await act(async () => { await vi.advanceTimersByTimeAsync(2000); });
  expect(callbacks.onChanged).toHaveBeenCalledOnce();
});

it("ignores out-of-order refresh results and aborts the superseded read", async () => {
  let oldResolve!: (value: { task: PrivateTask }) => void;
  vi.mocked(getPrivateTask).mockReturnValueOnce(new Promise((resolve) => { oldResolve = resolve; }))
    .mockResolvedValue({ task: task({ status: "paused" }) });
  const callbacks = props();
  const view = render(<PrivateTaskProgress {...callbacks} />);
  const signal = vi.mocked(getPrivateTask).mock.calls[0][1];
  view.rerender(<PrivateTaskProgress {...callbacks} refreshVersion={1} />);
  expect(await screen.findByRole("button", { name: "Resume review" })).toBeVisible();
  await act(async () => oldResolve({ task: task() }));
  expect(signal?.aborted).toBe(true);
  expect(screen.queryByRole("button", { name: "Pause" })).not.toBeInTheDocument();
});

it("offers a controlled retry on read failure without falling back to duplicate generation", async () => {
  vi.mocked(getPrivateTask).mockRejectedValueOnce(new Error("private backend details"))
    .mockResolvedValue({ task: null });
  render(<PrivateTaskProgress {...props()} />);
  expect(await screen.findByRole("alert")).toHaveTextContent("Could not load saved review progress.");
  expect(screen.queryByRole("button", { name: "Start review" })).not.toBeInTheDocument();
  expect(screen.getByRole("alert")).not.toHaveTextContent("private backend details");
  fireEvent.click(screen.getByRole("button", { name: "Reload progress" }));
  expect(await screen.findByRole("button", { name: "Start review" })).toBeEnabled();
});

it("keeps task controls unavailable on read-only archived views", async () => {
  vi.mocked(getPrivateTask).mockResolvedValue({ task: task() });
  render(<PrivateTaskProgress {...props()} readOnly />);
  expect(await screen.findByText("Audrey is reviewing your saved information")).toBeVisible();
  expect(screen.queryByRole("button")).not.toBeInTheDocument();
});

it("stops reading after unmount, including a read that resolves after cancellation", async () => {
  vi.useFakeTimers();
  let resolve!: (value: { task: PrivateTask }) => void;
  vi.mocked(getPrivateTask).mockReturnValue(new Promise((value) => { resolve = value; }));
  const view = render(<PrivateTaskProgress {...props()} />);
  const signal = vi.mocked(getPrivateTask).mock.calls[0][1];
  view.unmount();
  await act(async () => { resolve({ task: task() }); await vi.advanceTimersByTimeAsync(6000); });
  expect(signal?.aborted).toBe(true);
  expect(getPrivateTask).toHaveBeenCalledTimes(1);
});
