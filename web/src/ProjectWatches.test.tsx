import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { ProjectHome } from "./ProjectWorkspace";
import * as api from "./api";

vi.mock("./ProjectKnowledgeSection", () => ({ ProjectKnowledgeSection: () => null }));
vi.mock("./TaxProjectSection", () => ({ TaxProjectSection: () => null }));
const project: api.AudreyProject = { id: "project-one", name: "Public releases", instructions: "Stable releases only.", created_at: "2026-11-01T12:00:00Z", updated_at: "2026-11-01T12:00:00Z" };
const watch: api.PublicWatch = { id: "watch-one", title: "Release announcements", url: "https://example.com/releases", criterion: "New stable release", schedule: { kind: "daily", time: "09:00", timezone: "UTC" }, enabled: true, revision: 1, next_run_at: "2026-11-02T09:00:00Z", active_run_id: null, latest_run: null, created_at: project.created_at, updated_at: project.updated_at, project: { id: project.id, name: project.name, available: true } };
function props() { return { project, conversations: [], files: [], filesLoading: false, limits: { max_name_chars: 100, max_instructions_chars: 4000, max_files: 20 }, defaultModelId: "auto", onProjectChange: vi.fn(), onProjectDeleted: vi.fn(), onConversationCreated: vi.fn(), onConversationSelected: vi.fn(), onFilesChange: vi.fn() }; }
afterEach(() => { cleanup(); vi.restoreAllMocks(); });

it("lists and opens the same Project watch before its first check and prefills without creating", async () => {
  const listing = vi.spyOn(api, "listWatches").mockResolvedValue({ items: [watch, { ...watch, id: "other", title: "Other Project watch", project: { ...watch.project!, id: "other-project" } }] });
  const create = vi.spyOn(api, "createWatch"), action = vi.spyOn(api, "actOnWatch"), onNew = vi.fn(), onOpen = vi.fn();
  render(<ProjectHome {...props()} onNewProjectWatch={onNew} onOpenProjectWatch={onOpen} />);
  expect(listing).toHaveBeenCalledWith(expect.any(AbortSignal), project.id);
  fireEvent.click(await screen.findByRole("button", { name: /Release announcements/ }));
  expect(onOpen).toHaveBeenCalledWith(watch, expect.any(AbortSignal));
  expect(screen.queryByText("Other Project watch")).not.toBeInTheDocument();
  await waitFor(() => expect(screen.getByRole("button", { name: "New watch in Project" })).toBeEnabled());
  fireEvent.click(screen.getByRole("button", { name: "New watch in Project" }));
  expect(onNew).toHaveBeenCalledWith(project, expect.any(AbortSignal));
  expect(create).not.toHaveBeenCalled(); expect(action).not.toHaveBeenCalled();
});

it("keeps Private Projects outside public-page watches and explains ordinary Project deletion", () => {
  const listing = vi.spyOn(api, "listWatches");
  const base = props(); const { rerender } = render(<ProjectHome {...base} project={{ ...project, private_required: true }} onNewProjectWatch={vi.fn()} onOpenProjectWatch={vi.fn()} />);
  expect(screen.queryByRole("heading", { name: "Project watches" })).not.toBeInTheDocument(); expect(listing).not.toHaveBeenCalled();
  rerender(<ProjectHome {...base} />);
  expect(screen.getByRole("button", { name: "Delete project" }).title).toContain("Linked watches stop; saved checks");
  fireEvent.click(screen.getByRole("button", { name: "Delete project" }));
  expect(screen.getByRole("button", { name: "Confirm delete project" })).toHaveTextContent("✓");
  expect(screen.getByRole("button", { name: "Confirm delete project" }).title).toContain("Linked watches stop; saved checks");
});

it("aborts hidden links and late list responses, then recovers on refresh", async () => {
  let resolve!: (value: { items: api.PublicWatch[] }) => void;
  const listing = vi.spyOn(api, "listWatches").mockResolvedValueOnce({ items: [watch] }).mockImplementationOnce(() => new Promise((done) => { resolve = done; })).mockResolvedValue({ items: [watch] });
  let signal: AbortSignal | undefined;
  const onOpen = vi.fn((_watch: api.PublicWatch, current?: AbortSignal) => { signal = current; return new Promise<void>(() => {}); });
  const base = props(); const { rerender } = render(<ProjectHome {...base} onOpenProjectWatch={onOpen} onNewProjectWatch={vi.fn()} />);
  fireEvent.click(await screen.findByRole("button", { name: /Release announcements/ }));
  rerender(<ProjectHome {...base} visible={false} onOpenProjectWatch={onOpen} onNewProjectWatch={vi.fn()} />);
  expect(signal?.aborted).toBe(true);
  rerender(<ProjectHome {...base} visible onOpenProjectWatch={onOpen} onNewProjectWatch={vi.fn()} />);
  await waitFor(() => expect(listing).toHaveBeenCalledTimes(2));
  const listSignal = listing.mock.calls[1][0];
  rerender(<ProjectHome {...base} visible={false} onOpenProjectWatch={onOpen} onNewProjectWatch={vi.fn()} />);
  expect(listSignal?.aborted).toBe(true);
  await act(async () => resolve({ items: [{ ...watch, title: "Late stale watch" }] }));
  expect(screen.queryByText("Late stale watch")).not.toBeInTheDocument();
  rerender(<ProjectHome {...base} visible onOpenProjectWatch={onOpen} onNewProjectWatch={vi.fn()} />);
  await waitFor(() => expect(screen.getByRole("button", { name: "New watch in Project" })).toBeEnabled());
  fireEvent.click(screen.getByRole("button", { name: "Refresh Project watches" }));
  await waitFor(() => expect(listing).toHaveBeenCalledTimes(4));
});

it("blocks a saved watch whose Project is unavailable", async () => {
  vi.spyOn(api, "listWatches").mockResolvedValue({ items: [{ ...watch, project: { ...watch.project!, available: false } }] });
  const onOpen = vi.fn(); render(<ProjectHome {...props()} onOpenProjectWatch={onOpen} />);
  fireEvent.click(await screen.findByRole("button", { name: /Release announcements/ }));
  expect(await screen.findByRole("alert")).toHaveTextContent("unavailable in this Project"); expect(onOpen).not.toHaveBeenCalled();
});
