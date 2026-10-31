import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { type ComponentProps, useState } from "react";
import { afterEach, expect, it, vi } from "vitest";
import { ChatWorkspace } from "./ChatWorkspace";
import type { Conversation, ConversationMessage } from "./api";

const timestamp = "2026-10-31T12:00:00Z";
const draft: Conversation = { id: "draft", title: "Draft conversation", default_mode: "auto", default_model_id: "auto", created_at: timestamp, updated_at: timestamp, last_message_at: null, archived_at: null };
const saved: Conversation = { ...draft, id: "saved", title: "Saved conversation", last_message_at: timestamp };
const archived: Conversation = { ...saved, id: "private", title: "Archived Private notes", default_mode: "local", default_model_id: "private", private_required: true, project_id: "private-project", archived_at: timestamp };
const userMessage: ConversationMessage = { id: "user-one", run_id: null, sequence: 1, role: "user", status: "completed", content: "Saved question", created_at: timestamp, updated_at: timestamp, attachments: [] };
const answerMessage: ConversationMessage = { ...userMessage, id: "answer-one", sequence: 2, role: "assistant", content: "Saved answer" };

function props(): ComponentProps<typeof ChatWorkspace> {
  return {
    visible: true, navigationOpen: false, onNavigationOpenChange: vi.fn(), mobileMenuHost: null,
    mobileNewChatHost: null, menuActions: null, menuFooter: null,
    user: { id: "owner", display_name: "Owner", email: "alice@example.com", role: "user", status: "active", groups: ["users"], auth_provider: "cloudflare_access" },
    preferences: { timezone: "UTC", persona: "", detail: "balanced", tone: "natural", show_progress: true, created_at: timestamp, updated_at: timestamp },
    models: [{ id: "auto", label: "Auto", description: "Choose automatically.", kind: "workflow", mode: "auto", presentation: "auto", capabilities: ["text", "files", "tools"], enabled: true, audience: "users", visibility: "public", roles: ["users"], portrait_url: "" }], skills: [],
  };
}

function mockApi(options: { delayedHistory?: boolean; initialDraft?: Conversation } = {}) {
  let finishHistory: ((value: Response) => void) | undefined;
  const requests = vi.fn(async (path: string, requestOptions?: RequestInit) => {
    if (requestOptions?.method && requestOptions.method !== "GET") throw new Error(`Unexpected mutation: ${requestOptions.method} ${path}`);
    const payload = path.startsWith("/api/conversations?")
      ? { items: [options.initialDraft ?? draft, saved], next_cursor: null }
      : path === "/api/conversations/draft/messages" ? { items: [] }
        : /\/api\/conversations\/(saved|private)\/messages/.test(path) ? { items: [userMessage, answerMessage] }
          : /\/api\/conversations\/[^/]+\/(tax-review|private-task)/.test(path) ? null
            : { items: [], next_cursor: null };
    if (options.delayedHistory && path.startsWith("/api/conversations?")) {
      return new Promise<Response>((done) => { finishHistory = done; });
    }
    return new Response(JSON.stringify(payload), { status: payload === null ? 404 : 200, headers: { "Content-Type": "application/json" } });
  });
  vi.stubGlobal("fetch", requests);
  return { requests, resolveHistory: (items: Conversation[]) => finishHistory?.(new Response(JSON.stringify({ items, next_cursor: null }), { headers: { "Content-Type": "application/json" } })) };
}

afterEach(() => { cleanup(); vi.restoreAllMocks(); vi.unstubAllGlobals(); });

it("opens the exact saved message repeatedly while keeping an existing unsent draft mounted", async () => {
  const { requests } = mockApi();
  const base = props(); const { rerender, container } = render(<ChatWorkspace {...base} />);
  const composer = await screen.findByRole("textbox", { name: "Ask Audrey" });
  fireEvent.change(composer, { target: { value: "Keep this draft" } });
  rerender(<ChatWorkspace {...base} conversationOpenRequest={{ conversation: saved, messageId: answerMessage.id, serial: 1 }} />);
  const answer = await screen.findByText(answerMessage.content);
  const message = answer.closest("[data-message-id]")!;
  await waitFor(() => expect(message).toHaveFocus());
  expect(message).toHaveAttribute("data-message-id", answerMessage.id);
  expect(composer).toBeInTheDocument();
  expect(composer).toHaveValue("Keep this draft");
  expect(composer).not.toBeVisible();
  const runtime = message.closest(".conversation-thread-slot");
  rerender(<ChatWorkspace {...base} conversationOpenRequest={{ conversation: saved, messageId: userMessage.id, serial: 2 }} />);
  const question = await screen.findByText(userMessage.content);
  await waitFor(() => expect(question.closest("[data-message-id]")).toHaveFocus());
  expect(question.closest(".conversation-thread-slot")).toBe(runtime);
  expect(container.querySelectorAll(".conversation-thread-slot")).toHaveLength(2);
  const drafts = screen.getByRole("navigation", { name: "Draft conversations" });
  expect(within(drafts).getAllByRole("button")).toHaveLength(1);
  fireEvent.click(within(drafts).getByRole("button", { name: "Open draft conversation Draft conversation" }));
  expect(screen.getByRole("textbox", { name: "Ask Audrey" })).toBe(composer);
  expect(composer).toBeVisible(); expect(composer).toHaveValue("Keep this draft");
  expect(screen.queryByRole("navigation", { name: "Draft conversations" })).not.toBeInTheDocument();
  expect(requests.mock.calls.some(([, init]) => init?.method && init.method !== "GET")).toBe(false);
});

it("makes a retained Private draft reachable from the mobile Menu after Search navigation", async () => {
  vi.stubGlobal("matchMedia", vi.fn(() => ({ matches: true, addEventListener: vi.fn(), removeEventListener: vi.fn() })));
  const { requests } = mockApi({ initialDraft: { ...draft, private_required: true, default_model_id: "private", default_mode: "local" } });
  const base = props();
  function MobileWorkspace({ request }: { request?: ComponentProps<typeof ChatWorkspace>["conversationOpenRequest"] }) {
    const [open, setOpen] = useState(false);
    const [menuHost, setMenuHost] = useState<HTMLDivElement | null>(null);
    return <><div ref={setMenuHost} /><ChatWorkspace {...base} navigationOpen={open} onNavigationOpenChange={setOpen}
      mobileMenuHost={menuHost} conversationOpenRequest={request} /></>;
  }
  const { rerender } = render(<MobileWorkspace />);
  const composer = await screen.findByRole("textbox", { name: "Ask Audrey" });
  fireEvent.change(composer, { target: { value: "Keep this Private draft" } });
  rerender(<MobileWorkspace request={{ conversation: saved, messageId: answerMessage.id, serial: 5 }} />);
  expect(await screen.findByText(answerMessage.content)).toBeVisible();
  expect(screen.queryByRole("navigation", { name: "Draft conversations" })).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "Menu" }));
  const drawer = screen.getByRole("dialog", { name: "Conversations" });
  const drafts = within(drawer).getByRole("navigation", { name: "Draft conversations" });
  const resume = within(drafts).getByRole("button", { name: "Open draft conversation Draft conversation" });
  expect(resume).toHaveAttribute("data-private", "true");
  fireEvent.click(resume);
  expect(screen.getByRole("textbox", { name: "Ask Audrey" })).toBe(composer);
  expect(composer).toBeVisible(); expect(composer).toHaveValue("Keep this Private draft");
  expect(screen.queryByRole("dialog", { name: "Conversations" })).not.toBeInTheDocument();
  expect(requests.mock.calls.some(([, init]) => init?.method && init.method !== "GET")).toBe(false);
});

it("waits until visible and keeps archived Private Project conversations read only", async () => {
  const { requests } = mockApi();
  const base = props(); const request = { conversation: archived, messageId: answerMessage.id, serial: 3 };
  const { rerender, container } = render(<ChatWorkspace {...base} visible={false} conversationOpenRequest={request} />);
  await screen.findByRole("textbox", { name: "Ask Audrey" });
  expect(container.querySelector('[data-message-id="answer-one"]')).not.toBeInTheDocument();
  rerender(<ChatWorkspace {...base} conversationOpenRequest={request} />);
  expect(await screen.findByText("This conversation is archived. Restore it to continue.")).toBeVisible();
  expect(screen.getByRole("button", { name: "Restore" })).toBeEnabled();
  fireEvent.focus(screen.getByRole("note", { name: "Private processing" }));
  expect(screen.getByRole("tooltip")).toHaveTextContent("Leave this Private project before choosing another processing mode.");
  await waitFor(() => expect(container.querySelector('[data-message-id="answer-one"]')).toHaveFocus());
  expect(requests.mock.calls.some(([, init]) => (init as RequestInit | undefined)?.method === "POST")).toBe(false);
});

it("does not let a late empty history page replace a Search selection or create a conversation", async () => {
  const api = mockApi({ delayedHistory: true });
  const base = props(); const { container } = render(<ChatWorkspace {...base} conversationOpenRequest={{ conversation: saved, messageId: userMessage.id, serial: 4 }} />);
  expect(await screen.findByText(userMessage.content)).toBeVisible();
  await act(async () => api.resolveHistory([]));
  await waitFor(() => expect(container.querySelector('[data-message-id="user-one"]')).toHaveFocus());
  expect(screen.getByRole("heading", { name: saved.title })).toBeVisible();
  expect(api.requests.mock.calls.some(([path]) => path === "/api/conversations")).toBe(false);
});
