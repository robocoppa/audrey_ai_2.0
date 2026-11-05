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

function waitingTask(patch: Partial<AgentTask> = {}): AgentTask {
  return task({ status: "waiting_for_input", revision: 6, steps: [{ id: "plan", label: "Plan the research", status: "pending", result: "" }, { id: "research", label: "Research public sources", status: "pending", result: "" }, { id: "review", label: "Review the report", status: "pending", result: "" }], input_requests: [{ id: "input-1", step_id: "plan", question: "Which years should the comparison cover?", answer: null, created_at: "2026-10-30T15:00:00Z", answered_at: null }], ...patch });
}

const project = { id: "project-research", name: "Public release work", instructions: "Stable releases only.", created_at: "2026-10-31T10:00:00Z", updated_at: "2026-10-31T10:00:00Z" };
const projectContext = { project_id: project.id, project_name: project.name, available: true, current: true, instructions_included: true, knowledge: [{ id: "note-one", revision: 1 }] };
const knowledgeEntry: api.KnowledgeEntry = { id: "entry-open", project_id: project.id, kind: "action", text: "Compare public release dates.", approval_state: "approved", work_status: "open", revision: 3, pinned: false, origin: "user", sources: [], source_task_id: null, created_at: project.created_at, updated_at: project.updated_at, approved_at: project.updated_at, stale: false, source_warnings: [], replaces_id: null, base_revision: null };
const knowledgeRequest = { project, entry: knowledgeEntry, serial: 1 };
const savedOutput: api.AgentOutput = { file_id: "file_checklist", filename: "task-checklist.md", download_url: "/api/files/file_checklist/download", created_at: "2026-11-02T10:00:00Z" };
const selectedProjectFile: api.AudreyProjectFile = { id: "file-input", filename: "project-instructions.txt", mime: "text/plain", kind: "text", bytes: 900, status: "ready", uploaded_at: project.created_at, added_at: project.created_at };
const fileInputs: api.AgentFileInputs = { current: true, files: [{ file_id: selectedProjectFile.id, filename: selectedProjectFile.filename, artifact: "text", chars: 500, total_chars: 900, truncated: true, origin: "upload" }] };

beforeEach(() => {
  vi.spyOn(api, "listProjects").mockResolvedValue({ items: [], next_cursor: null, limits: { max_name_chars: 100, max_instructions_chars: 4000, max_files: 20 } });
  vi.spyOn(api, "getActivityUnreadCount").mockResolvedValue({ unread_count: 0 });
  vi.spyOn(api, "listAgentEvents").mockResolvedValue({ items: [], has_more: false, truncated: false });
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
  it("creates and selects an ordinary Project without losing the task draft or starting an agent", async () => {
    const created = { ...project, id: "project-created", name: "Garden preparations" };
    const createProject = vi.spyOn(api, "createProject").mockResolvedValue(created);
    vi.spyOn(api, "getProject").mockResolvedValue(created);
    const saved = task({ project_context: { ...projectContext, project_id: created.id, project_name: created.name } });
    vi.mocked(api.getAgentTask).mockResolvedValue(saved);
    const start = vi.spyOn(api, "createAgentTask").mockImplementation(async () => { vi.mocked(api.listAgentTasks).mockResolvedValue({ items: [saved] }); return saved; });
    render(<AgentsWorkspace />);
    fireEvent.change(screen.getByRole("textbox", { name: "Task objective" }), { target: { value: "Prepare my garden checklist" } });
    fireEvent.click(screen.getByRole("button", { name: "New project" }));
    const dialog = await screen.findByRole("dialog", { name: "New project" });
    expect(within(dialog).queryByRole("button", { name: /Private/ })).not.toBeInTheDocument();
    fireEvent.change(within(dialog).getByRole("textbox", { name: "Project name" }), { target: { value: "Garden preparations" } });
    fireEvent.change(within(dialog).getByRole("textbox", { name: "Project instructions Optional" }), { target: { value: "Keep the checklist short." } });
    fireEvent.click(within(dialog).getByRole("button", { name: "Create project" }));
    await waitFor(() => expect(createProject).toHaveBeenCalledWith(created.name, "Keep the checklist short.", "auto"));
    await waitFor(() => expect(screen.queryByRole("dialog", { name: "New project" })).not.toBeInTheDocument());
    expect(screen.getByRole("combobox", { name: "Project" })).toHaveValue(created.id);
    expect(screen.getByRole("textbox", { name: "Task objective" })).toHaveValue("Prepare my garden checklist");
    expect(start).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: "Start agent" }));
    await waitFor(() => expect(start).toHaveBeenCalledWith("Prepare my garden checklist", expect.any(String), created.id));
  });

  it("keeps the existing Project, selected files and objective after a failed or cancelled Project creation", async () => {
    vi.spyOn(api, "listProjectFiles").mockResolvedValue({ items: [selectedProjectFile], limits: { max_name_chars: 100, max_instructions_chars: 4000, max_files: 20 } });
    const createProject = vi.spyOn(api, "createProject").mockRejectedValue(new api.ApiError(503, "Project creation unavailable"));
    const start = vi.spyOn(api, "createAgentTask");
    render(<AgentsWorkspace composerProjectRequest={{ project, serial: 1 }} />);
    fireEvent.click(await screen.findByRole("button", { name: "Project files Optional" }));
    fireEvent.click(await screen.findByRole("checkbox", { name: selectedProjectFile.filename }));
    fireEvent.change(screen.getByRole("textbox", { name: "Task objective" }), { target: { value: "Keep my existing task" } });
    fireEvent.click(screen.getByRole("button", { name: "New project" }));
    const dialog = await screen.findByRole("dialog", { name: "New project" });
    fireEvent.change(within(dialog).getByRole("textbox", { name: "Project name" }), { target: { value: "Another Project" } });
    fireEvent.click(within(dialog).getByRole("button", { name: "Create project" }));
    expect(await within(dialog).findByRole("alert")).toHaveTextContent("Project creation unavailable");
    expect(within(dialog).getByRole("textbox", { name: "Project name" })).toHaveValue("Another Project");
    fireEvent.click(within(dialog).getByRole("button", { name: "Cancel" }));
    expect(screen.getByRole("textbox", { name: "Task objective" })).toHaveValue("Keep my existing task");
    expect(screen.getByRole("combobox", { name: "Project" })).toHaveValue(project.id);
    expect(screen.getByRole("checkbox", { name: selectedProjectFile.filename })).toBeChecked();
    expect(createProject).toHaveBeenCalledTimes(1); expect(start).not.toHaveBeenCalled();
  });

  it("starts only with explicitly selected Project files, keeps selections across navigation, then clears them on success", async () => {
    vi.spyOn(api, "listProjectFiles").mockResolvedValue({ items: [selectedProjectFile], limits: { max_name_chars: 100, max_instructions_chars: 4000, max_files: 20 } });
    vi.spyOn(api, "getProject").mockResolvedValue(project);
    const saved = task({ file_inputs: fileInputs, project_context: projectContext });
    vi.mocked(api.getAgentTask).mockResolvedValue(saved);
    const create = vi.spyOn(api, "createAgentTask").mockImplementation(async () => { vi.mocked(api.listAgentTasks).mockResolvedValue({ items: [saved] }); return saved; });
    const request = { project, serial: 1 };
    const { rerender } = render(<AgentsWorkspace composerProjectRequest={request} />);
    fireEvent.click(await screen.findByRole("button", { name: "Project files Optional" }));
    fireEvent.click(await screen.findByRole("checkbox", { name: "project-instructions.txt" }));
    fireEvent.change(screen.getByRole("textbox", { name: "Task objective" }), { target: { value: "Make a checklist from this file." } });
    rerender(<AgentsWorkspace composerProjectRequest={request} visible={false} />);
    rerender(<AgentsWorkspace composerProjectRequest={request} visible />);
    await waitFor(() => expect(screen.getByRole("checkbox", { name: "project-instructions.txt" })).toBeChecked());
    expect(create).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: "Start agent" }));
    await waitFor(() => expect(create).toHaveBeenCalledWith("Make a checklist from this file.", expect.any(String), project.id, undefined, [selectedProjectFile.id]));
    await waitFor(() => expect(screen.getByRole("textbox", { name: "Task objective" })).toHaveValue(""));
    expect(screen.getByRole("checkbox", { name: "project-instructions.txt" })).not.toBeChecked();
    fireEvent.click(await screen.findByText("Input files · 1", { selector: "summary" }));
    expect(screen.getByText(/500 of 900 characters · Shortened/)).toBeVisible();
  });

  it("keeps the objective and file selection after the server rejects unavailable saved text", async () => {
    vi.spyOn(api, "listProjectFiles").mockResolvedValue({ items: [selectedProjectFile], limits: { max_name_chars: 100, max_instructions_chars: 4000, max_files: 20 } });
    vi.spyOn(api, "getProject").mockResolvedValue(project);
    const create = vi.spyOn(api, "createAgentTask").mockRejectedValue(new api.ApiError(422, "Saved text is unavailable for a selected file. Choose another file."));
    render(<AgentsWorkspace composerProjectRequest={{ project, serial: 1 }} />);
    fireEvent.click(await screen.findByRole("button", { name: "Project files Optional" }));
    fireEvent.click(await screen.findByRole("checkbox", { name: "project-instructions.txt" }));
    fireEvent.change(screen.getByRole("textbox", { name: "Task objective" }), { target: { value: "Use this file" } });
    fireEvent.click(screen.getByRole("button", { name: "Start agent" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Saved text is unavailable");
    expect(screen.getByRole("textbox", { name: "Task objective" })).toHaveValue("Use this file");
    expect(screen.getByRole("checkbox", { name: "project-instructions.txt" })).toBeChecked();
    expect(create).toHaveBeenCalledTimes(1);
  });

  it.each(["change Project", "clear draft"])("discards file selections when users %s", async (action) => {
    vi.mocked(api.listProjects).mockResolvedValue({ items: [project], next_cursor: null, limits: { max_name_chars: 100, max_instructions_chars: 4000, max_files: 20 } });
    vi.spyOn(api, "listProjectFiles").mockResolvedValue({ items: [selectedProjectFile], limits: { max_name_chars: 100, max_instructions_chars: 4000, max_files: 20 } });
    vi.spyOn(api, "getProject").mockResolvedValue(project);
    const create = vi.spyOn(api, "createAgentTask").mockResolvedValue(task());
    render(<AgentsWorkspace composerProjectRequest={{ project, serial: 1 }} />);
    fireEvent.click(await screen.findByRole("button", { name: "Project files Optional" }));
    fireEvent.click(await screen.findByRole("checkbox", { name: "project-instructions.txt" }));
    fireEvent.change(screen.getByRole("textbox", { name: "Task objective" }), { target: { value: "Use my objective" } });
    if (action === "change Project") {
      fireEvent.change(screen.getByRole("combobox", { name: "Project" }), { target: { value: "" } });
      fireEvent.change(screen.getByRole("combobox", { name: "Project" }), { target: { value: project.id } });
      fireEvent.click(screen.getByRole("button", { name: "Project files Optional" }));
    } else {
      fireEvent.click(screen.getByRole("button", { name: "Clear draft" }));
      fireEvent.change(screen.getByRole("textbox", { name: "Task objective" }), { target: { value: "Fresh objective" } });
    }
    expect(await screen.findByRole("checkbox", { name: "project-instructions.txt" })).not.toBeChecked();
    fireEvent.click(screen.getByRole("button", { name: "Start agent" }));
    await waitFor(() => expect(create).toHaveBeenCalledTimes(1)); expect(create.mock.calls[0]).toHaveLength(3);
  });

  it.each(["completed", "paused", "waiting_for_input"] as const)("blocks continuation for stale file inputs on a %s task", async (status) => {
    const saved = status === "waiting_for_input" ? waitingTask({ file_inputs: { ...fileInputs, current: false } }) : task({ status, file_inputs: { ...fileInputs, current: false } });
    vi.mocked(api.listAgentTasks).mockResolvedValue({ items: [saved] }); vi.mocked(api.getAgentTask).mockResolvedValue(saved);
    const action = vi.spyOn(api, "actOnAgentTask"); const answer = vi.spyOn(api, "answerAgentInput");
    render(<AgentsWorkspace />);
    fireEvent.click(await screen.findByText("Input files changed", { selector: "summary" }));
    expect(screen.getByText(/A selected file changed or is unavailable/)).toBeVisible();
    if (status === "waiting_for_input") {
      fireEvent.change(screen.getByRole("textbox", { name: "Your answer" }), { target: { value: "Use this answer" } });
      expect(screen.getByRole("button", { name: "Answer and continue" })).toBeDisabled();
      fireEvent.submit(screen.getByRole("form", { name: "Answer agent question" }));
    } else expect(screen.getByRole("button", { name: status === "paused" ? "Resume" : "Run again" })).toBeDisabled();
    expect(action).not.toHaveBeenCalled(); expect(answer).not.toHaveBeenCalled();
  });

  it.each(["running", "paused", "failed", "cancelled", "completed"] as const)("keeps produced files available on a %s task with no result", async (status) => {
    const saved = task({ status, result: "", outputs: [savedOutput] });
    vi.mocked(api.listAgentTasks).mockResolvedValue({ items: [saved] }); vi.mocked(api.getAgentTask).mockResolvedValue(saved);
    render(<AgentsWorkspace />);
    expect(await screen.findByRole("button", { name: "Download task-checklist.md" })).toBeVisible();
    expect(screen.getByRole("region", { name: "Files" })).toHaveTextContent("Saved in My Files.");
    expect(screen.getByRole("heading", { name: "Result" })).toBeVisible();
  });

  it("keeps saved files through summaries, reloads newer detail and removes a canonically deleted file", async () => {
    const saved = task({ status: "completed", revision: 8, outputs: [savedOutput], file_inputs: { ...fileInputs, current: false } });
    vi.mocked(api.listAgentTasks).mockResolvedValue({ items: [saved] }); vi.mocked(api.getAgentTask).mockResolvedValue(saved);
    render(<AgentsWorkspace />);
    await screen.findByRole("button", { name: "Download task-checklist.md" });
    await waitFor(() => expect(screen.getByRole("button", { name: "Refresh agent tasks" })).toBeEnabled());
    let release!: (value: AgentTask) => void;
    vi.mocked(api.getAgentTask).mockImplementation(() => new Promise((resolve) => { release = resolve; }));
    const { outputs: _outputs, file_inputs: _inputs, ...summary } = saved;
    void _outputs; void _inputs;
    vi.mocked(api.listAgentTasks).mockResolvedValue({ items: [{ ...summary, revision: 9, summary: true }] });
    const calls = vi.mocked(api.getAgentTask).mock.calls.length;
    fireEvent.click(screen.getByRole("button", { name: "Refresh agent tasks" }));
    await waitFor(() => expect(api.getAgentTask).toHaveBeenCalledTimes(calls + 1));
    expect(screen.getAllByRole("button", { name: "Download task-checklist.md" })).toHaveLength(1);
    expect(screen.getByText("Input files changed", { selector: "summary" })).toBeVisible();
    expect(screen.getByRole("button", { name: "Run again" })).toBeDisabled();
    await act(async () => release({ ...summary, revision: 9 }));
    await waitFor(() => expect(screen.queryByRole("heading", { name: "Files" })).not.toBeInTheDocument());
    expect(screen.queryByText("Input files changed", { selector: "summary" })).not.toBeInTheDocument();
    expect(screen.getByRole("heading", { name: saved.objective })).toBeVisible();
  });

  it.each(["running", "failed", "completed"] as const)("uses Result for a general %s task", async (status) => {
    const saved = task({ status, result: status === "completed" ? "The drafted checklist." : "" });
    vi.mocked(api.listAgentTasks).mockResolvedValue({ items: [saved] }); vi.mocked(api.getAgentTask).mockResolvedValue(saved);
    render(<AgentsWorkspace />);
    expect(await screen.findByRole("heading", { name: "Result" })).toBeVisible();
    expect(screen.queryByRole("heading", { name: "Report" })).not.toBeInTheDocument();
    if (status === "running") expect(screen.getByText("The result will appear here when Audrey completes its review.")).toBeVisible();
    else if (status === "failed") expect(screen.getByText("This task stopped before a final result was saved. Any completed steps are available above.")).toBeVisible();
    else expect(screen.getByRole("button", { name: "Delete task" })).toHaveAttribute("title", "Delete saved task and result. Created files stay in My Files.");
  });

  it("describes an empty general task workflow with a saved result", async () => {
    render(<AgentsWorkspace />);
    expect(await screen.findByText("Review and save the result")).toBeVisible();
    expect(screen.queryByText("Review and save the report")).not.toBeInTheDocument();
  });
  it("prefills an open Knowledge entry without starting and sends its reviewed pair only on explicit Start", async () => {
    vi.spyOn(api, "getProject").mockResolvedValue(project);
    const origin = { id: knowledgeEntry.id, revision: 3 };
    const saved = task({ project_context: { ...projectContext, origin_entry: origin } });
    const create = vi.spyOn(api, "createAgentTask").mockImplementation(async () => { vi.mocked(api.listAgentTasks).mockResolvedValue({ items: [saved] }); return saved; });
    vi.mocked(api.getAgentTask).mockResolvedValue(saved);
    const update = vi.spyOn(api, "updateProjectKnowledge");
    render(<AgentsWorkspace composerKnowledgeRequest={knowledgeRequest} />);
    await waitFor(() => expect(screen.getByRole("textbox", { name: "Task objective" })).toHaveValue("Compare public release dates."));
    expect(screen.getByText(/From Project Knowledge · Revision 3/)).toBeVisible();
    expect(create).not.toHaveBeenCalled(); expect(update).not.toHaveBeenCalled();
    fireEvent.change(screen.getByRole("textbox", { name: "Task objective" }), { target: { value: "Research stable releases only." } });
    fireEvent.click(screen.getByRole("button", { name: "Start agent" }));
    await waitFor(() => expect(create).toHaveBeenCalledWith("Research stable releases only.", expect.any(String), project.id, origin));
    expect(update).not.toHaveBeenCalled();
    expect(await screen.findByLabelText(`Project: ${project.name}`)).toHaveAttribute("title", expect.stringContaining("Started from Project Knowledge · Revision 3"));
  });

  it("preserves a different composer draft until Use entry and supports Keep my draft", async () => {
    const create = vi.spyOn(api, "createAgentTask");
    const { rerender } = render(<AgentsWorkspace />);
    fireEvent.change(screen.getByRole("textbox", { name: "Task objective" }), { target: { value: "My unsent objective" } });
    rerender(<AgentsWorkspace composerKnowledgeRequest={knowledgeRequest} />);
    expect(await screen.findByRole("group", { name: "Use Project Knowledge entry" })).toBeVisible();
    expect(screen.getByRole("textbox", { name: "Task objective" })).toHaveValue("My unsent objective");
    expect(screen.getByRole("combobox", { name: "Project" })).toHaveValue("");
    expect(screen.getByRole("button", { name: "Start agent" })).toBeDisabled();
    fireEvent.click(screen.getByRole("button", { name: "Keep my draft" }));
    expect(screen.queryByRole("group", { name: "Use Project Knowledge entry" })).not.toBeInTheDocument();
    expect(screen.getByRole("textbox", { name: "Task objective" })).toHaveValue("My unsent objective");
    rerender(<AgentsWorkspace composerKnowledgeRequest={{ ...knowledgeRequest, serial: 2 }} />);
    fireEvent.click(await screen.findByRole("button", { name: "Use entry" }));
    expect(screen.getByRole("textbox", { name: "Task objective" })).toHaveValue("Compare public release dates.");
    expect(screen.getByRole("combobox", { name: "Project" })).toHaveValue(project.id); expect(create).not.toHaveBeenCalled();
  });

  it.each(["change Project", "remove entry link", "clear draft"])("clears the Knowledge origin when users %s", async (action) => {
    vi.spyOn(api, "getProject").mockResolvedValue(project);
    const create = vi.spyOn(api, "createAgentTask").mockResolvedValue(task());
    render(<AgentsWorkspace composerKnowledgeRequest={knowledgeRequest} />);
    await screen.findByRole("button", { name: "Remove entry link" });
    if (action === "change Project") fireEvent.change(screen.getByRole("combobox", { name: "Project" }), { target: { value: "" } });
    else fireEvent.click(screen.getByRole("button", { name: action === "clear draft" ? "Clear draft" : "Remove entry link" }));
    expect(screen.queryByRole("button", { name: "Remove entry link" })).not.toBeInTheDocument();
    if (action === "clear draft") fireEvent.change(screen.getByRole("textbox", { name: "Task objective" }), { target: { value: "A fresh objective" } });
    fireEvent.click(screen.getByRole("button", { name: "Start agent" }));
    await waitFor(() => expect(create).toHaveBeenCalledTimes(1));
    expect(create.mock.calls[0]).toHaveLength(action === "change Project" ? 2 : 3);
  });

  it("retains the objective and origin after an authoritative entry conflict", async () => {
    vi.spyOn(api, "getProject").mockResolvedValue(project);
    const create = vi.spyOn(api, "createAgentTask").mockRejectedValue(new api.ApiError(409, "The selected Knowledge entry changed. Review it before starting."));
    render(<AgentsWorkspace composerKnowledgeRequest={knowledgeRequest} />);
    await screen.findByRole("button", { name: "Remove entry link" });
    fireEvent.click(screen.getByRole("button", { name: "Start agent" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("The selected Knowledge entry changed");
    expect(screen.getByRole("textbox", { name: "Task objective" })).toHaveValue("Compare public release dates.");
    expect(screen.getByRole("button", { name: "Remove entry link" })).toBeVisible(); expect(create).toHaveBeenCalledTimes(1);
  });

  it("keeps an available Project selected after origin deletion and lets users remove its entry link", async () => {
    vi.spyOn(api, "getProject").mockResolvedValue(project);
    const create = vi.spyOn(api, "createAgentTask").mockRejectedValueOnce(new api.ApiError(404, "The selected Knowledge entry is unavailable.")).mockResolvedValueOnce(task());
    render(<AgentsWorkspace composerKnowledgeRequest={knowledgeRequest} />);
    await screen.findByRole("button", { name: "Remove entry link" }); fireEvent.click(screen.getByRole("button", { name: "Start agent" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("The selected Knowledge entry is unavailable");
    expect(screen.getByRole("combobox", { name: "Project" })).toHaveValue(project.id);
    expect(screen.getByRole("button", { name: "Start agent" })).toBeEnabled();
    expect(screen.queryByText(/The selected Project is unavailable/)).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Remove entry link" })); fireEvent.click(screen.getByRole("button", { name: "Start agent" }));
    await waitFor(() => expect(create).toHaveBeenCalledTimes(2)); expect(create.mock.calls[1]).toHaveLength(3);
  });

  it("rejects a Unicode objective over the backend byte limit while retaining its entry text", async () => {
    const text = "“date” ".repeat(850); const create = vi.spyOn(api, "createAgentTask");
    render(<AgentsWorkspace composerKnowledgeRequest={{ ...knowledgeRequest, entry: { ...knowledgeEntry, text } }} />);
    expect(await screen.findByRole("alert")).toHaveTextContent("8,000 UTF-8 bytes");
    expect(screen.getByRole("textbox", { name: "Task objective" })).toHaveValue(`${text}`);
    fireEvent.submit(screen.getByRole("button", { name: "Start agent" }).closest("form")!); expect(create).not.toHaveBeenCalled();
  });

  it("keeps an oversized entry intact and requires an edited bounded objective", async () => {
    const create = vi.spyOn(api, "createAgentTask"); const text = "a".repeat(7000);
    render(<AgentsWorkspace composerKnowledgeRequest={{ ...knowledgeRequest, entry: { ...knowledgeEntry, text } }} />);
    expect(await screen.findByRole("alert")).toHaveTextContent("6,000 characters");
    expect(screen.getByRole("textbox", { name: "Task objective" })).toHaveValue(`${text}`);
    fireEvent.submit(screen.getByRole("button", { name: "Start agent" }).closest("form")!); expect(create).not.toHaveBeenCalled();
  });

  it("applies only the latest composer navigation and never restores an older Knowledge origin", async () => {
    render(<AgentsWorkspace composerKnowledgeRequest={knowledgeRequest} composerProjectRequest={{ project, serial: 2 }} />);
    await waitFor(() => expect(screen.getByRole("combobox", { name: "Project" })).toHaveValue(project.id));
    expect(screen.getByRole("textbox", { name: "Task objective" })).toHaveValue("");
    expect(screen.queryByRole("button", { name: "Remove entry link" })).not.toBeInTheDocument();
  });
  it("keeps completed results free of Project Knowledge proposal controls", async () => {
    const saved = task({ status: "completed", result: "Completed source report.", revision: 8 });
    vi.mocked(api.listAgentTasks).mockResolvedValue({ items: [saved] });
    vi.mocked(api.getAgentTask).mockResolvedValue(saved);
    const generate = vi.spyOn(api, "generateAgentKnowledgeDrafts"), propose = vi.spyOn(api, "createProjectKnowledge");
    render(<AgentsWorkspace />);
    expect(await screen.findByText(saved.result)).toBeVisible();
    expect(screen.queryByRole("button", { name: "Propose Project knowledge" })).not.toBeInTheDocument();
    expect(screen.queryByRole("dialog", { name: "Propose from this report" })).not.toBeInTheDocument();
    expect(generate).not.toHaveBeenCalled(); expect(propose).not.toHaveBeenCalled();
  });

  it("creates an explicitly selected ordinary Project agent and excludes Private choices", async () => {
    vi.mocked(api.listProjects).mockResolvedValue({ items: [project, { ...project, id: "protected", name: "Private work", private_required: true }], next_cursor: null, limits: { max_name_chars: 100, max_instructions_chars: 4000, max_files: 20 } });
    vi.spyOn(api, "getProject").mockResolvedValue(project);
    const create = vi.spyOn(api, "createAgentTask").mockResolvedValue(task({ project_context: projectContext }));
    render(<AgentsWorkspace />);
    await screen.findByRole("option", { name: project.name });
    expect(screen.queryByRole("option", { name: "Private work" })).not.toBeInTheDocument();
    fireEvent.change(screen.getByRole("combobox", { name: "Project" }), { target: { value: project.id } });
    fireEvent.change(screen.getByRole("textbox", { name: "Task objective" }), { target: { value: "Compare public releases" } });
    fireEvent.click(screen.getByRole("button", { name: "Start agent" }));
    await waitFor(() => expect(create).toHaveBeenCalledWith("Compare public releases", expect.any(String), project.id));
  });

  it("prefills a Project without starting, preserving the objective across navigation", async () => {
    const create = vi.spyOn(api, "createAgentTask");
    const { rerender } = render(<AgentsWorkspace />);
    fireEvent.change(screen.getByRole("textbox", { name: "Task objective" }), { target: { value: "An unsent objective" } });
    rerender(<AgentsWorkspace composerProjectRequest={{ project, serial: 1 }} />);
    await waitFor(() => expect(screen.getByRole("combobox", { name: "Project" })).toHaveValue(project.id));
    expect(screen.getByRole("textbox", { name: "Task objective" })).toHaveValue("An unsent objective");
    rerender(<AgentsWorkspace visible={false} composerProjectRequest={{ project, serial: 1 }} />);
    rerender(<AgentsWorkspace visible composerProjectRequest={{ project, serial: 1 }} />);
    expect(screen.getByRole("combobox", { name: "Project" })).toHaveValue(project.id);
    expect(create).not.toHaveBeenCalled();
  });

  it("keeps unavailable Project selection explicit and never retries as an unscoped task", async () => {
    vi.spyOn(api, "getProject").mockRejectedValue(new api.ApiError(404, "Unavailable"));
    const create = vi.spyOn(api, "createAgentTask");
    render(<AgentsWorkspace composerProjectRequest={{ project, serial: 1 }} />);
    await waitFor(() => expect(screen.getByRole("combobox", { name: "Project" })).toHaveValue(project.id));
    fireEvent.change(screen.getByRole("textbox", { name: "Task objective" }), { target: { value: "A selected objective" } });
    fireEvent.click(screen.getByRole("button", { name: "Start agent" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("The selected Project is unavailable");
    expect(screen.getByRole("button", { name: "Start agent" })).toBeDisabled(); expect(create).not.toHaveBeenCalled();
  });

  it("opens a linked agent's Project and blocks rerun after its context changes", async () => {
    const saved = task({ status: "completed", result: "Saved report", project_context: { ...projectContext, current: false } });
    vi.mocked(api.listAgentTasks).mockResolvedValue({ items: [saved] }); vi.mocked(api.getAgentTask).mockResolvedValue(saved);
    const open = vi.fn(), control = vi.spyOn(api, "actOnAgentTask"); render(<AgentsWorkspace onOpenProject={open} />);
    fireEvent.click(await screen.findByRole("button", { name: `Open Project: ${project.name}` })); expect(open).toHaveBeenCalledWith(project.id);
    expect(screen.getByText("Changed")).toBeVisible();
    expect(screen.getByText("The Project context has changed. Start a new agent to use its current instructions and knowledge.")).toBeVisible();
    expect(screen.getByRole("button", { name: "Run again" })).toBeDisabled(); expect(control).not.toHaveBeenCalled();
  });

  it("places the saved Project beside the running status without a detail dropdown", async () => {
    const saved = task({ project_context: projectContext });
    vi.mocked(api.listAgentTasks).mockResolvedValue({ items: [saved] }); vi.mocked(api.getAgentTask).mockResolvedValue(saved);
    const open = vi.fn(); render(<AgentsWorkspace onOpenProject={open} />);
    const badge = await screen.findByRole("button", { name: `Open Project: ${project.name}` });
    const toolbar = badge.closest(".agents-detail-toolbar")!;
    expect(within(toolbar as HTMLElement).getByText("Running")).toBeVisible();
    expect(badge).toHaveAccessibleDescription(expect.stringContaining("1 saved knowledge revision pinned for this agent."));
    const detail = screen.getByRole("region", { name: "Selected agent" });
    expect(within(detail).queryByRole("combobox", { name: "Project" })).not.toBeInTheDocument();
    expect(detail.querySelector(".agents-project-context")).toBeNull();
    fireEvent.click(badge); expect(open).toHaveBeenCalledExactlyOnceWith(project.id);
    expect(screen.getByRole("combobox", { name: "Project" })).toBeVisible();
  });

  it("keeps an unavailable saved Project visible and continuation blocked", async () => {
    const saved = task({ status: "completed", project_context: { ...projectContext, available: false, project_name: null } });
    vi.mocked(api.listAgentTasks).mockResolvedValue({ items: [saved] }); vi.mocked(api.getAgentTask).mockResolvedValue(saved);
    render(<AgentsWorkspace onOpenProject={vi.fn()} />);
    expect(await screen.findByLabelText("Project: Saved Project · Unavailable")).toBeVisible();
    expect(screen.getByText("This Project is no longer available. Continuing or running this agent again is unavailable.")).toBeVisible();
    expect(screen.queryByRole("button", { name: /^Open Project:/ })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Run again" })).toBeDisabled();
  });

  it("opens a Project's existing agent without creating another task", async () => {
    const saved = task({ project_context: projectContext });
    vi.mocked(api.listAgentTasks).mockResolvedValue({ items: [saved] }); vi.mocked(api.getAgentTask).mockResolvedValue(saved);
    const create = vi.spyOn(api, "createAgentTask"); render(<AgentsWorkspace externalAgentRequest={{ task: saved, serial: 1 }} />);
    expect(await screen.findByRole("heading", { name: saved.objective })).toBeVisible();
    expect(screen.getAllByRole("button", { name: `Open agent: ${saved.objective}` })).toHaveLength(1); expect(create).not.toHaveBeenCalled();
  });

  it("shows a waiting agent's plain question, only Cancel, and includes it in Active without a moving badge", async () => {
    const saved = waitingTask({ input_requests: [{ ...waitingTask().input_requests![0], question: "<img src=x onerror=alert(1)> Which years?" }] });
    vi.mocked(api.listAgentTasks).mockResolvedValue({ items: [saved] }); vi.mocked(api.getAgentTask).mockResolvedValue(saved);
    const active = vi.fn(); render(<AgentsWorkspace onActiveCountChange={active} />);
    expect(await screen.findByRole("heading", { name: "Agent needs your input" })).toBeVisible();
    expect(screen.getByText(saved.input_requests![0].question)).toBeVisible(); expect(document.querySelector('img[src="x"]')).toBeNull();
    expect(screen.getByRole("button", { name: "Cancel" })).toBeVisible(); expect(screen.queryByRole("button", { name: "Resume" })).not.toBeInTheDocument(); expect(screen.queryByRole("button", { name: "Pause" })).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Active" })); expect(screen.getByRole("button", { name: `Open agent: ${saved.objective}` })).toBeVisible();
    expect(screen.getAllByText("Waiting for input").length).toBeGreaterThan(0); expect(active).toHaveBeenLastCalledWith(0);
    expect(screen.getByRole("button", { name: "Answer and continue" })).toBeDisabled();
  });

  it("answers and continues the same task, preserving the objective and saved questions as plain text", async () => {
    const saved = waitingTask(), response = "Use 2000–2026. <b>Keep this plain text</b>";
    const answered = { ...saved, status: "queued" as const, revision: 7, input_requests: saved.input_requests!.map((input) => ({ ...input, answer: response, answered_at: saved.updated_at })) };
    vi.mocked(api.listAgentTasks).mockResolvedValue({ items: [saved] }); vi.mocked(api.getAgentTask).mockResolvedValue(saved);
    const answer = vi.spyOn(api, "answerAgentInput").mockImplementation(async () => { vi.mocked(api.listAgentTasks).mockResolvedValue({ items: [answered] }); vi.mocked(api.getAgentTask).mockResolvedValue(answered); return answered; });
    const create = vi.spyOn(api, "createAgentTask"); render(<AgentsWorkspace />);
    fireEvent.change(await screen.findByRole("textbox", { name: "Your answer" }), { target: { value: response } });
    fireEvent.click(screen.getByRole("button", { name: "Answer and continue" }));
    await waitFor(() => expect(answer).toHaveBeenCalledWith(saved.id, "input-1", 6, response));
    const confirmation = await screen.findByText(/Answer saved\. This agent will continue/);
    expect(confirmation).toBeVisible();
    expect(confirmation).toHaveAttribute("role", "status");
    expect(confirmation.closest(".agents-detail-header")).not.toBeNull();
    expect(screen.getAllByRole("button", { name: `Open agent: ${saved.objective}` })).toHaveLength(1); expect(create).not.toHaveBeenCalled();
    fireEvent.click(screen.getByLabelText("Questions and answers"));
    expect(screen.getByText(response)).toBeVisible(); expect(document.querySelector(".agents-input-history b")).toBeNull();
  });

  it("keeps a delayed answer confirmation off another selected task", async () => {
    const first = waitingTask(), second = task({ id: "agent-second", objective: "Prepare a welcome note" });
    const answered = { ...first, status: "queued" as const, revision: 7, input_requests: first.input_requests!.map((input) => ({ ...input, answer: "Use 2026.", answered_at: first.updated_at })) };
    vi.mocked(api.listAgentTasks).mockResolvedValue({ items: [first, second] });
    vi.mocked(api.getAgentTask).mockImplementation(async (id) => id === first.id ? first : second);
    let finishAnswer!: (saved: AgentTask) => void;
    vi.spyOn(api, "answerAgentInput").mockImplementation(() => new Promise((resolve) => { finishAnswer = resolve; }));
    render(<AgentsWorkspace />);
    fireEvent.change(await screen.findByRole("textbox", { name: "Your answer" }), { target: { value: "Use 2026." } });
    fireEvent.click(screen.getByRole("button", { name: "Answer and continue" }));
    fireEvent.click(screen.getByRole("button", { name: `Open agent: ${second.objective}` }));
    expect(await screen.findByRole("heading", { name: second.objective })).toBeVisible();
    vi.mocked(api.listAgentTasks).mockResolvedValue({ items: [answered, second] });
    await act(async () => { finishAnswer(answered); });
    expect(screen.queryByText(/Answer saved\. This agent will continue/)).not.toBeInTheDocument();
    expect(screen.getByRole("heading", { name: second.objective })).toBeVisible();
  });

  it("preserves separate answer drafts across task selection, hidden tabs and same-revision summaries", async () => {
    const first = waitingTask(), second = waitingTask({ id: "agent-second", objective: "Second research objective", input_requests: [{ ...waitingTask().input_requests![0], id: "input-2", question: "Which comparison should be used?" }] });
    vi.mocked(api.listAgentTasks).mockResolvedValue({ items: [first, second] }); vi.mocked(api.getAgentTask).mockImplementation(async (id) => id === first.id ? first : second);
    vi.spyOn(api, "listActivity").mockResolvedValue({ items: [], unread_count: 0 }); render(<AgentsWorkspace />);
    fireEvent.change(await screen.findByRole("textbox", { name: "Your answer" }), { target: { value: "First answer draft" } });
    fireEvent.click(screen.getByRole("button", { name: `Open agent: ${second.objective}` }));
    await screen.findByText("Which comparison should be used?"); fireEvent.change(screen.getByRole("textbox", { name: "Your answer" }), { target: { value: "Second answer draft" } });
    fireEvent.click(screen.getByRole("button", { name: "Activity" })); await screen.findByRole("heading", { name: "Activity" }); fireEvent.click(screen.getByRole("button", { name: "Tasks" }));
    expect(await screen.findByRole("textbox", { name: "Your answer" })).toHaveValue("Second answer draft");
    vi.mocked(api.listAgentTasks).mockResolvedValue({ items: [{ ...first, summary: true, input_requests: [] }, { ...second, summary: true, input_requests: [] }] });
    fireEvent.click(screen.getByRole("button", { name: `Open agent: ${first.objective}` }));
    fireEvent.click(screen.getByRole("button", { name: "Refresh agent tasks" }));
    await waitFor(() => expect(screen.getByRole("textbox", { name: "Your answer" })).toHaveValue("First answer draft"));
  });

  it("keeps a conflicted answer draft and requires explicit review before using a newer revision", async () => {
    const saved = waitingTask(), latest = waitingTask({ revision: 8 });
    vi.mocked(api.listAgentTasks).mockResolvedValue({ items: [saved] }); vi.mocked(api.getAgentTask).mockResolvedValue(saved);
    const answer = vi.spyOn(api, "answerAgentInput").mockRejectedValueOnce(new api.ApiError(409, "Changed")); render(<AgentsWorkspace />);
    fireEvent.change(await screen.findByRole("textbox", { name: "Your answer" }), { target: { value: "Preserved answer" } });
    vi.mocked(api.getAgentTask).mockResolvedValue(latest); vi.mocked(api.listAgentTasks).mockResolvedValue({ items: [latest] });
    fireEvent.click(screen.getByRole("button", { name: "Answer and continue" }));
    expect(await screen.findByText(/This agent changed while you were answering\. Review its latest question and saved answers/)).toBeVisible();
    expect(screen.getByRole("textbox", { name: "Your answer" })).toHaveValue("Preserved answer"); expect(screen.getByRole("button", { name: "Answer and continue" })).toBeDisabled();
    fireEvent.submit(screen.getByRole("form", { name: "Answer agent question" })); expect(answer).toHaveBeenCalledTimes(1);
    fireEvent.click(screen.getByRole("button", { name: "Use my answer with this revision" })); expect(screen.getByRole("button", { name: "Answer and continue" })).toBeEnabled();
  });

  it("enforces the answer's UTF-8 byte limit before sending and retains the draft after request errors", async () => {
    const saved = waitingTask(); vi.mocked(api.listAgentTasks).mockResolvedValue({ items: [saved] }); vi.mocked(api.getAgentTask).mockResolvedValue(saved);
    const answer = vi.spyOn(api, "answerAgentInput").mockRejectedValue(new TypeError("Connection lost")); render(<AgentsWorkspace />);
    const input = await screen.findByRole("textbox", { name: "Your answer" }); fireEvent.change(input, { target: { value: "🙂".repeat(2001) } });
    expect(await screen.findByRole("alert")).toHaveTextContent("8,000-byte limit"); expect(screen.getByRole("button", { name: "Answer and continue" })).toBeDisabled();
    fireEvent.submit(screen.getByRole("form", { name: "Answer agent question" })); expect(answer).not.toHaveBeenCalled();
    fireEvent.change(input, { target: { value: "A valid retained answer" } }); fireEvent.click(screen.getByRole("button", { name: "Answer and continue" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Connection lost"); expect(screen.getByRole("textbox", { name: "Your answer" })).toHaveValue("A valid retained answer");
  });

  it("never applies a previous question's draft when a poll supplies a different pending question", async () => {
    const saved = waitingTask(), latest = waitingTask({ revision: 9, input_requests: [{ ...waitingTask().input_requests![0], id: "input-new", question: "Which source should take priority?" }] });
    vi.mocked(api.listAgentTasks).mockResolvedValue({ items: [saved] }); vi.mocked(api.getAgentTask).mockResolvedValue(saved); const answer = vi.spyOn(api, "answerAgentInput"); render(<AgentsWorkspace />);
    fireEvent.change(await screen.findByRole("textbox", { name: "Your answer" }), { target: { value: "Answer intended for the previous question" } });
    vi.mocked(api.listAgentTasks).mockResolvedValue({ items: [latest] }); vi.mocked(api.getAgentTask).mockResolvedValue(latest); fireEvent.click(screen.getByRole("button", { name: "Refresh agent tasks" }));
    expect(await screen.findByText("Which source should take priority?")).toBeVisible(); expect(screen.getByRole("textbox", { name: "Your answer" })).toHaveValue("");
    fireEvent.submit(screen.getByRole("form", { name: "Answer agent question" })); expect(answer).not.toHaveBeenCalled();
  });

  it("retains the full waiting question, draft and answered history during same-revision summary refreshes", async () => {
    const pending = waitingTask().input_requests![0];
    const saved = waitingTask({ input_requests: [{ ...pending, id: "earlier-input", question: "Which edition?", answer: "The public edition", answered_at: pending.created_at }, pending] });
    vi.mocked(api.listAgentTasks).mockResolvedValue({ items: [saved] }); vi.mocked(api.getAgentTask).mockResolvedValue(saved); render(<AgentsWorkspace />);
    fireEvent.change(await screen.findByRole("textbox", { name: "Your answer" }), { target: { value: "Retained while detail reloads" } });
    fireEvent.click(screen.getByLabelText("Questions and answers")); expect(screen.getByText("The public edition")).toBeVisible();
    vi.mocked(api.getAgentTask).mockImplementation(() => new Promise(() => {}));
    vi.mocked(api.listAgentTasks).mockResolvedValue({ items: [{ ...saved, summary: true, input_requests: [] }] });
    fireEvent.click(screen.getByRole("button", { name: "Refresh agent tasks" }));
    await waitFor(() => expect(screen.getByRole("button", { name: "Refresh agent tasks" })).toBeEnabled());
    expect(screen.getByText(pending.question)).toBeVisible(); expect(screen.getByRole("textbox", { name: "Your answer" })).toHaveValue("Retained while detail reloads"); expect(screen.getByText("The public edition")).toBeVisible();
  });
  it("opens Activity's saved agent in the existing task card and preserves the composer draft", async () => {
    const saved = task({ status: "completed", revision: 4, result: "The saved complete report.", title: "Official public documentation comparison" });
    vi.mocked(api.listAgentTasks).mockResolvedValue({ items: [saved] }); vi.mocked(api.getAgentTask).mockResolvedValue(saved);
    const item: api.ActivityNotice = { id: "notice-1", origin: "agent", kind: "completed", title: saved.title!, body: "The agent completed.", target: { kind: "agent", task_id: saved.id, revision: saved.revision }, created_at: saved.updated_at, read_at: null };
    const feed = vi.spyOn(api, "listActivity").mockResolvedValue({ items: [item], unread_count: 1 });
    vi.spyOn(api, "getActivityAgentTask").mockResolvedValue(saved);
    vi.spyOn(api, "readActivity").mockResolvedValue({ ...item, read_at: saved.updated_at });
    const create = vi.spyOn(api, "createAgentTask");
    render(<AgentsWorkspace />);
    await screen.findByRole("heading", { name: saved.title });
    expect(feed).not.toHaveBeenCalled(); expect(api.getActivityUnreadCount).toHaveBeenCalled();
    fireEvent.change(screen.getByRole("textbox", { name: "Task objective" }), { target: { value: "My next unsaved agent objective" } });
    fireEvent.click(screen.getByRole("button", { name: "Activity" }));
    fireEvent.click(await screen.findByRole("button", { name: /Open agent result/ }));
    expect(await screen.findByRole("heading", { name: saved.title })).toBeVisible();
    expect(screen.getByRole("textbox", { name: "Task objective" })).toHaveValue("My next unsaved agent objective");
    expect(screen.getAllByRole("button", { name: `Open agent: ${saved.title}` })).toHaveLength(1);
    expect(create).not.toHaveBeenCalled();
  });

  it("does not let an older count-only request overwrite a newer Activity count", async () => {
    let finishCount!: (value: { unread_count: number }) => void;
    vi.mocked(api.getActivityUnreadCount).mockImplementation(() => new Promise((resolve) => { finishCount = resolve; }));
    vi.spyOn(api, "listActivity").mockResolvedValue({ items: [], unread_count: 7 });
    render(<AgentsWorkspace />);
    fireEvent.click(screen.getByRole("button", { name: "Activity" }));
    expect(await screen.findByLabelText("7 unread notifications")).toBeVisible();
    await act(async () => finishCount({ unread_count: 1 }));
    expect(screen.getByLabelText("7 unread notifications")).toBeVisible();
  });

  it("rejects an older Activity result when the same task already has a newer cached revision", async () => {
    const current = task({ revision: 17, status: "completed", result: "The newer cached report." });
    vi.mocked(api.listAgentTasks).mockResolvedValue({ items: [current] }); vi.mocked(api.getAgentTask).mockResolvedValue(current);
    const item: api.ActivityNotice = { id: "old-notice", origin: "agent", kind: "completed", title: "Older result", body: "The agent completed.", target: { kind: "agent", task_id: current.id, revision: 15 }, created_at: current.updated_at, read_at: null };
    vi.spyOn(api, "listActivity").mockResolvedValue({ items: [item], unread_count: 1 });
    vi.spyOn(api, "getActivityAgentTask").mockResolvedValue({ ...current, revision: 15, result: "The older saved report." });
    const read = vi.spyOn(api, "readActivity"); render(<AgentsWorkspace />);
    await screen.findByText("The newer cached report."); fireEvent.click(screen.getByRole("button", { name: "Activity" }));
    fireEvent.click(await screen.findByRole("button", { name: /Older result/ }));
    expect(await screen.findByRole("alert")).toHaveTextContent("refresh Activity");
    expect(read).not.toHaveBeenCalled(); expect(screen.getByRole("heading", { name: "Activity" })).toBeVisible();
    expect(screen.queryByText("The older saved report.")).not.toBeInTheDocument();
  });

  it("keeps a Watch creation draft and schedule preview across Activity navigation", async () => {
    vi.spyOn(api, "listWatches").mockResolvedValue({ items: [] });
    vi.spyOn(api, "listActivity").mockResolvedValue({ items: [], unread_count: 0 });
    vi.spyOn(api, "previewWatchSchedule").mockResolvedValue({ occurrences: ["2026-11-01T09:00:00Z", "2026-11-02T09:00:00Z", "2026-11-03T09:00:00Z"] });
    render(<AgentsWorkspace timezone="UTC" />);
    fireEvent.click(screen.getByRole("button", { name: "Watches" }));
    fireEvent.click(await screen.findByRole("button", { name: "New watch" }));
    fireEvent.change(screen.getByRole("textbox", { name: "Title" }), { target: { value: "Unsaved release watch" } });
    fireEvent.click(screen.getByRole("button", { name: "Preview next checks" }));
    await screen.findByText("Next three checks (UTC)");
    fireEvent.click(screen.getByRole("button", { name: "Activity" }));
    await screen.findByText("No activity yet. Agent results and watch alerts will appear here.");
    fireEvent.click(screen.getByRole("button", { name: "Watches" }));
    expect(screen.getByRole("textbox", { name: "Title" })).toHaveValue("Unsaved release watch");
    expect(screen.getByText("Next three checks (UTC)")).toBeVisible();
  });

  it("uses the saved model title and keeps the unchanged original prompt expandable", async () => {
    const original = "Give me an exhaustive timeline of the Crusades, including the major campaigns, their context, and the relevant sources.";
    const saved = task({ title: "Crusades Timeline and Major Campaigns", objective: original });
    vi.mocked(api.listAgentTasks).mockResolvedValue({ items: [saved] });
    vi.mocked(api.getAgentTask).mockResolvedValue(saved);
    render(<AgentsWorkspace />);
    expect(await screen.findByRole("heading", { name: saved.title })).toBeVisible();
    expect(screen.getByRole("button", { name: `Open agent: ${saved.title}` })).toBeVisible();
    const prompt = screen.getByText(original, { selector: ".agents-original-prompt > p" });
    expect(prompt).not.toBeVisible();
    fireEvent.click(screen.getByText("Original prompt", { selector: "summary" }));
    expect(prompt).toBeVisible();
  });
  it("starts a saved public research task from the objective", async () => {
    const created = task({ status: "queued", revision: 1 });
    const create = vi.spyOn(api, "createAgentTask").mockImplementation(async () => {
      vi.mocked(api.listAgentTasks).mockResolvedValue({ items: [created] });
      return created;
    });
    vi.mocked(api.getAgentTask).mockResolvedValue(created);
    render(<AgentsWorkspace />);
    expect(await screen.findByText("No agents yet. Start with a task objective above.")).toBeVisible();
    expect(screen.getByRole("button", { name: "Start agent" })).toBeDisabled();
    fireEvent.change(screen.getByRole("textbox", { name: "Task objective" }), { target: { value: "  Compare public documentation  " } });
    fireEvent.click(screen.getByRole("button", { name: "Start agent" }));
    expect(await screen.findByRole("heading", { name: created.objective })).toBeVisible();
    expect(create).toHaveBeenCalledWith(created.objective, expect.any(String));
    expect(screen.getByRole("textbox", { name: "Task objective" })).toHaveValue("");
    expect(screen.getByRole("heading", { name: "Plan and progress" })).toBeVisible();
    expect(screen.getByText("Work", { selector: "strong" })).toBeVisible();
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
    await screen.findByText("No agents yet. Start with a task objective above.");
    fireEvent.change(screen.getByRole("textbox", { name: "Task objective" }), { target: { value: "Compare public documentation" } });
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
    expect(sidebar.getByRole("textbox", { name: "Task objective" })).toBeVisible();
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
    expect(workflow).not.toContainElement(screen.getByRole("textbox", { name: "Task objective" }));
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
    fireEvent.change(controls.getByRole("textbox", { name: "Task objective" }), { target: { value: "Research public sources" } });
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
      error: status === "failed" ? "Research stopped." : "", steps, outputs: [savedOutput],
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
    expect(screen.getAllByRole("button", { name: "Download task-checklist.md" })).toHaveLength(1);
    fireEvent.click(screen.getByLabelText("View saved plan"));
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
    expect(await screen.findByText("No agents yet. Start with a task objective above.")).toBeVisible();
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
    expect(await screen.findByText("No agents yet. Start with a task objective above.")).toBeVisible();
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
    fireEvent.click(screen.getByLabelText("View saved plan"));
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


it("discards an old catalog error while starting a new Agent", async () => {
  let rejectOld!: (reason: Error) => void;
  let finish!: (value: AgentTask) => void;
  const created = task({ id: "agt_new", title: "New garden task" });
  vi.mocked(api.listAgentTasks).mockImplementationOnce(() => new Promise((_, reject) => { rejectOld = reject; })).mockResolvedValue({ items: [created] });
  vi.mocked(api.getAgentTask).mockResolvedValue(created);
  vi.spyOn(api, "createAgentTask").mockImplementation(() => new Promise((resolve) => { finish = resolve; }));
  render(<AgentsWorkspace />);
  await waitFor(() => expect(api.listAgentTasks).toHaveBeenCalled());
  fireEvent.change(screen.getByRole("textbox", { name: "Task objective" }), { target: { value: "New garden task" } });
  fireEvent.click(screen.getByRole("button", { name: "Start agent" }));
  await act(async () => rejectOld(new api.ApiError(409, "Agent task changed. Refresh before trying again.")));
  expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  await act(async () => finish(created));
  expect(await screen.findByRole("heading", { name: created.title })).toBeVisible();
  expect(screen.queryByRole("alert")).not.toBeInTheDocument();
});

it("discards the previous task's late detail conflict during deletion and permits a new task", async () => {
  let rejectOld!: (reason: Error) => void;
  let finishDelete!: () => void;
  const old = task({ id: "agt_old", title: "Old completed task", status: "completed", result: "Old report" });
  const created = task({ id: "agt_new", title: "New task" });
  vi.mocked(api.listAgentTasks).mockResolvedValue({ items: [old] });
  vi.mocked(api.getAgentTask).mockImplementationOnce(() => new Promise((_, reject) => { rejectOld = reject; })).mockResolvedValue(created);
  vi.spyOn(api, "deleteAgentTask").mockImplementation(() => new Promise((resolve) => { finishDelete = resolve; }));
  vi.spyOn(api, "createAgentTask").mockImplementation(async () => { vi.mocked(api.listAgentTasks).mockResolvedValue({ items: [created] }); return created; });
  render(<AgentsWorkspace />);
  expect(await screen.findByRole("heading", { name: old.title })).toBeVisible();
  await waitFor(() => expect(api.getAgentTask).toHaveBeenCalled());
  fireEvent.click(screen.getByRole("button", { name: "Delete task" }));
  fireEvent.click(screen.getByRole("button", { name: "Confirm delete task" }));
  await act(async () => rejectOld(new api.ApiError(409, "Agent task changed. Refresh before trying again.")));
  expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  vi.mocked(api.listAgentTasks).mockResolvedValue({ items: [] });
  await act(async () => finishDelete());
  fireEvent.change(screen.getByRole("textbox", { name: "Task objective" }), { target: { value: "New task" } });
  fireEvent.click(screen.getByRole("button", { name: "Start agent" }));
  expect(await screen.findByRole("heading", { name: created.title })).toBeVisible();
  expect(screen.queryByRole("alert")).not.toBeInTheDocument();
});
