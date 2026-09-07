import { HttpAgent, type AgentSubscriber } from "@ag-ui/client";
import {
  AssistantRuntimeProvider,
  ComposerPrimitive,
  ExportedMessageRepository,
  MessagePrimitive,
  ThreadPrimitive,
  type ThreadHistoryAdapter,
  type ThreadMessageLike,
  type TextMessagePartProps,
  type ToolCallMessagePartProps,
} from "@assistant-ui/react";
import { useAgUiRuntime } from "@assistant-ui/react-ag-ui";
import { useEffect, useMemo, useRef, useState } from "react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";

import autoPortrait from "../../images/audrey2.png";
import deepPortrait from "../../images/audrey7.png";
import fastPortrait from "../../images/audrey3.png";
import researchPortrait from "../../images/search.png";
import videoPortrait from "../../images/audrey8.png";
import cloudPortrait from "../../images/cloudModel.png";
import localPortrait from "../../images/localModel.png";

import {
  createConversation,
  deleteConversation,
  listConversations,
  listMessages,
  updateConversation,
  updateConversationMode,
  type AudreyMode,
  type Conversation,
  type ConversationMessage,
  type CurrentUser,
} from "./api";
import { latestActionFetch } from "./agentTransport";

const MODES: ReadonlyArray<{
  value: AudreyMode;
  label: string;
  portrait: string;
}> = [
  { value: "auto", label: "Auto", portrait: autoPortrait },
  { value: "fast", label: "Fast", portrait: fastPortrait },
  { value: "deep", label: "Deep", portrait: deepPortrait },
  { value: "research", label: "Research", portrait: researchPortrait },
  { value: "local", label: "Local only", portrait: localPortrait },
  { value: "cloud", label: "Cloud", portrait: cloudPortrait },
  { value: "video", label: "Video", portrait: videoPortrait },
];

type ThreadState =
  | { status: "idle" }
  | { status: "loading" }
  | { status: "ready"; messages: ConversationMessage[] }
  | { status: "error"; message: string };

type RunActivity = {
  status: "idle" | "running" | "complete" | "cancelled" | "error";
  label: string;
  detail: string;
  sourceCount: number;
  latestSource: string;
};

type ConversationView = "active" | "archived";

const IDLE_ACTIVITY: RunActivity = {
  status: "idle",
  label: "Ready",
  detail: "",
  sourceCount: 0,
  latestSource: "",
};

export function ChatWorkspace({ user }: { user: CurrentUser }) {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [openedConversations, setOpenedConversations] = useState<Conversation[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [creating, setCreating] = useState(false);
  const [loadingMore, setLoadingMore] = useState(false);
  const [nextCursor, setNextCursor] = useState<string | null>(null);
  const [view, setView] = useState<ConversationView>("active");
  const [searchInput, setSearchInput] = useState("");
  const [searchQuery, setSearchQuery] = useState("");
  const [error, setError] = useState("");
  const listKeyRef = useRef("");
  const selectedIdRef = useRef<string | null>(null);

  function selectConversation(conversation: Conversation | null) {
    if (conversation) {
      setOpenedConversations((current) => upsertConversation(current, conversation));
    }
    selectedIdRef.current = conversation?.id ?? null;
    setSelectedId(selectedIdRef.current);
  }

  useEffect(() => {
    const timer = window.setTimeout(() => {
      const nextSearch = searchInput.trim();
      if (nextSearch === searchQuery) return;
      setLoading(true);
      setConversations([]);
      selectConversation(null);
      setNextCursor(null);
      setError("");
      setSearchQuery(nextSearch);
    }, 250);
    return () => window.clearTimeout(timer);
  }, [searchInput, searchQuery]);

  useEffect(() => {
    let active = true;
    const requestKey = `${view}\n${searchQuery}`;
    listKeyRef.current = requestKey;
    listConversations({ archived: view === "archived", search: searchQuery })
      .then(({ items, next_cursor }) => {
        if (!active) return;
        setConversations(items);
        setNextCursor(next_cursor);
        const selected = items.find(({ id }) => id === selectedIdRef.current)
          ?? items[0]
          ?? null;
        selectConversation(selected);
      })
      .catch((reason: unknown) => {
        if (active) setError(messageOf(reason));
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, [searchQuery, view]);

  const selected = conversations.find(({ id }) => id === selectedId) ?? null;
  const renderedConversations = selected
    && !openedConversations.some(({ id }) => id === selected.id)
    ? [...openedConversations, selected]
    : openedConversations;

  function changeView(nextView: ConversationView) {
    if (nextView === view) return;
    setLoading(true);
    setConversations([]);
    selectConversation(null);
    setNextCursor(null);
    setError("");
    setView(nextView);
  }

  async function startConversation() {
    setCreating(true);
    setError("");
    try {
      const conversation = await createConversation("auto");
      const visibleImmediately = view === "active" && !searchQuery;
      if (visibleImmediately) {
        setConversations((current) => [conversation, ...current]);
      } else {
        setLoading(true);
        setSearchInput("");
        setSearchQuery("");
        setView("active");
      }
      selectConversation(conversation);
    } catch (reason) {
      setError(messageOf(reason));
    } finally {
      setCreating(false);
    }
  }

  function replaceConversation(updated: Conversation) {
    setOpenedConversations((current) => upsertConversation(current, updated));
    if (
      searchQuery
      && !updated.title.toLocaleLowerCase().includes(searchQuery.toLocaleLowerCase())
    ) {
      removeFromCurrentView(updated.id);
      return;
    }
    setConversations((current) =>
      current.map((conversation) =>
        conversation.id === updated.id ? updated : conversation,
      ),
    );
  }

  function removeFromCurrentView(conversationId: string, closeThread = false) {
    const remaining = conversations.filter(({ id }) => id !== conversationId);
    setConversations(remaining);
    if (closeThread) {
      setOpenedConversations((current) =>
        current.filter(({ id }) => id !== conversationId),
      );
    }
    if (selectedId === conversationId) {
      selectConversation(remaining[0] ?? null);
    }
  }

  async function loadMore() {
    if (!nextCursor || loadingMore) return;
    setLoadingMore(true);
    setError("");
    const expectedKey = listKeyRef.current;
    try {
      const page = await listConversations({
        archived: view === "archived",
        cursor: nextCursor,
        search: searchQuery,
      });
      if (listKeyRef.current !== expectedKey) return;
      setConversations((current) => [...current, ...page.items]);
      setNextCursor(page.next_cursor);
    } catch (reason) {
      if (listKeyRef.current === expectedKey) {
        setError(messageOf(reason));
      }
    } finally {
      setLoadingMore(false);
    }
  }

  return (
    <div className="workspace">
      <aside className="sidebar" aria-label="Conversations">
        <div className="sidebar-heading">
          <div>
            <span>Workspace</span>
            <strong>{user.display_name || user.email}</strong>
          </div>
          <button
            className="new-conversation"
            type="button"
            onClick={startConversation}
            disabled={creating}
          >
            {creating ? "Creating…" : "+ New"}
          </button>
        </div>

        <label className="conversation-search">
          <span>Search titles</span>
          <input
            type="search"
            value={searchInput}
            onChange={(event) => setSearchInput(event.target.value)}
            placeholder="Search conversations"
            maxLength={200}
          />
        </label>
        <div className="conversation-views" aria-label="Conversation view">
          <button
            type="button"
            aria-pressed={view === "active"}
            onClick={() => changeView("active")}
          >
            Active
          </button>
          <button
            type="button"
            aria-pressed={view === "archived"}
            onClick={() => changeView("archived")}
          >
            Archived
          </button>
        </div>

        {loading ? <p className="sidebar-status">Loading conversations…</p> : null}
        {!loading && conversations.length === 0 ? (
          <p className="sidebar-status">
            {searchQuery
              ? "No matching conversation titles."
              : view === "archived"
                ? "No archived conversations."
                : "No conversations yet."}
          </p>
        ) : null}
        <nav className="conversation-list" aria-label="Conversation history">
          {conversations.map((conversation) => (
            <button
              className={conversation.id === selectedId ? "conversation active" : "conversation"}
              type="button"
              key={conversation.id}
              onClick={() => selectConversation(conversation)}
              aria-current={conversation.id === selectedId ? "page" : undefined}
            >
              <span>{conversation.title || "Untitled conversation"}</span>
              <small>{modeLabel(conversation.default_mode)}</small>
            </button>
          ))}
        </nav>
        {nextCursor ? (
          <button
            className="load-conversations"
            type="button"
            onClick={() => void loadMore()}
            disabled={loadingMore}
          >
            {loadingMore ? "Loading…" : "Load older"}
          </button>
        ) : null}
        {error ? <p className="sidebar-error" role="alert">{error}</p> : null}
      </aside>

      <section className="chat-column" aria-label="Audrey conversation">
        {renderedConversations.map((opened) => (
          <div
            className="conversation-thread-slot"
            hidden={opened.id !== selectedId}
            key={opened.id}
          >
            <ConversationThread
              conversation={opened}
              onConversationChange={replaceConversation}
              onRemoveFromView={removeFromCurrentView}
            />
          </div>
        ))}
        {!selected ? (
          <div className="empty-workspace">
            <span>Audrey</span>
            <h1>What shall we work through?</h1>
            <p>Start a conversation to use Audrey's native run pipeline.</p>
            <button type="button" onClick={startConversation} disabled={creating}>
              Start a conversation
            </button>
          </div>
        ) : null}
      </section>
    </div>
  );
}

function ConversationThread({
  conversation,
  onConversationChange,
  onRemoveFromView,
}: {
  conversation: Conversation;
  onConversationChange: (conversation: Conversation) => void;
  onRemoveFromView: (conversationId: string, closeThread?: boolean) => void;
}) {
  const [thread, setThread] = useState<ThreadState>({ status: "idle" });
  const [mode, setMode] = useState<AudreyMode>(conversation.default_mode);
  const [editingTitle, setEditingTitle] = useState(false);
  const [titleDraft, setTitleDraft] = useState(conversation.title);
  const [mutation, setMutation] = useState<"mode" | "rename" | "archive" | "delete" | null>(null);
  const [mutationError, setMutationError] = useState("");
  const [confirmingDelete, setConfirmingDelete] = useState(false);
  const [runActive, setRunActive] = useState(false);
  const archived = conversation.archived_at !== null;

  useEffect(() => {
    let active = true;
    listMessages(conversation.id)
      .then(({ items }) => {
        if (active) setThread({ status: "ready", messages: items });
      })
      .catch((reason: unknown) => {
        if (active) setThread({ status: "error", message: messageOf(reason) });
      });
    return () => {
      active = false;
    };
  }, [conversation.id]);

  async function changeMode(next: AudreyMode) {
    if (next === mode || runActive) return;
    setMutation("mode");
    setMutationError("");
    try {
      const updated = await updateConversationMode(conversation.id, next);
      const messages = await listMessages(conversation.id);
      setThread({ status: "ready", messages: messages.items });
      setMode(next);
      onConversationChange(updated);
    } catch (reason) {
      setMutationError(messageOf(reason));
    } finally {
      setMutation(null);
    }
  }

  async function renameConversation(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const title = titleDraft.trim();
    if (!title || title === conversation.title) {
      setTitleDraft(conversation.title);
      setEditingTitle(false);
      return;
    }
    setMutation("rename");
    setMutationError("");
    try {
      onConversationChange(await updateConversation(conversation.id, { title }));
      setEditingTitle(false);
    } catch (reason) {
      setMutationError(messageOf(reason));
    } finally {
      setMutation(null);
    }
  }

  async function toggleArchived() {
    setMutation("archive");
    setMutationError("");
    try {
      await updateConversation(conversation.id, { archived: !archived });
      onRemoveFromView(conversation.id, true);
    } catch (reason) {
      setMutationError(messageOf(reason));
    } finally {
      setMutation(null);
    }
  }

  async function removeConversation() {
    setMutation("delete");
    setMutationError("");
    try {
      await deleteConversation(conversation.id);
      onRemoveFromView(conversation.id, true);
    } catch (reason) {
      setMutationError(messageOf(reason));
      setConfirmingDelete(false);
    } finally {
      setMutation(null);
    }
  }

  return (
    <>
      <div className="thread-header-shell">
        <header className="thread-header">
          <div className="thread-title">
            <span>Conversation</span>
            {editingTitle ? (
              <form className="rename-conversation" onSubmit={(event) => void renameConversation(event)}>
                <input
                  aria-label="Conversation title"
                  value={titleDraft}
                  onChange={(event) => setTitleDraft(event.target.value)}
                  maxLength={200}
                  autoFocus
                />
                <button type="submit" disabled={mutation !== null}>Save</button>
                <button
                  type="button"
                  onClick={() => {
                    setTitleDraft(conversation.title);
                    setEditingTitle(false);
                  }}
                >
                  Cancel
                </button>
              </form>
            ) : (
              <div className="conversation-title-display">
                <h1>{conversation.title || "Untitled conversation"}</h1>
                <button
                  className="conversation-title-button"
                  type="button"
                  onClick={() => setEditingTitle(true)}
                  aria-label="Rename conversation"
                >
                  Edit
                </button>
              </div>
            )}
          </div>
          <div className="thread-controls">
            <div className="conversation-actions">
              <button
                type="button"
                onClick={() => void toggleArchived()}
                disabled={runActive || mutation !== null}
              >
                {archived ? "Restore" : "Archive"}
              </button>
              {!confirmingDelete ? (
                <button
                  className="danger-button"
                  type="button"
                  onClick={() => setConfirmingDelete(true)}
                  disabled={runActive || mutation !== null}
                >
                  Delete
                </button>
              ) : (
                <div className="delete-confirmation" role="group" aria-label="Confirm deletion">
                  <span>Delete permanently?</span>
                  <button
                    className="danger-button"
                    type="button"
                    onClick={() => void removeConversation()}
                    disabled={mutation !== null}
                  >
                    Yes, delete
                  </button>
                  <button type="button" onClick={() => setConfirmingDelete(false)}>Keep</button>
                </div>
              )}
            </div>
          </div>
        </header>
        {mutationError ? <p className="conversation-mutation-error" role="alert">{mutationError}</p> : null}
      </div>

      {thread.status === "loading" || thread.status === "idle" ? (
        <div className="thread-loading" role="status">Loading thread…</div>
      ) : null}
      {thread.status === "error" ? (
        <div className="thread-loading thread-error" role="alert">{thread.message}</div>
      ) : null}
      {thread.status === "ready" ? (
        <AudreyThread
          conversationId={conversation.id}
          mode={mode}
          initialMessages={thread.messages}
          readOnly={archived}
          modeDisabled={runActive || mutation !== null}
          onModeChange={changeMode}
          onRunActiveChange={setRunActive}
        />
      ) : null}
    </>
  );
}

function AudreyThread({
  conversationId,
  mode,
  initialMessages,
  readOnly,
  modeDisabled,
  onModeChange,
  onRunActiveChange,
}: {
  conversationId: string;
  mode: AudreyMode;
  initialMessages: ConversationMessage[];
  readOnly: boolean;
  modeDisabled: boolean;
  onModeChange: (mode: AudreyMode) => Promise<void>;
  onRunActiveChange: (active: boolean) => void;
}) {
  const [runError, setRunError] = useState("");
  const [activity, setActivity] = useState<RunActivity>(IDLE_ACTIVITY);
  const history = useMemo<ThreadHistoryAdapter>(
    () => ({
      load: () => Promise.resolve(
        ExportedMessageRepository.fromArray(toThreadMessages(initialMessages)),
      ),
      // Audrey persists both sides of a turn before and after the server run.
      // Runtime history writes are therefore deliberately browser-local no-ops.
      append: () => Promise.resolve(),
    }),
    [initialMessages],
  );
  const agent = useMemo(
    () =>
      new HttpAgent({
        url: `/api/agent?mode=${encodeURIComponent(mode)}`,
        threadId: conversationId,
        fetch: latestActionFetch,
      }),
    [conversationId, mode],
  );
  useEffect(() => {
    const subscriber: AgentSubscriber = {
      onRunInitialized: () => {
        onRunActiveChange(true);
        setRunError("");
        setActivity({
          status: "running",
          label: "Starting",
          detail: "Preparing Audrey's run",
          sourceCount: 0,
          latestSource: "",
        });
      },
      onStepStartedEvent: ({ event }) => {
        setActivity((current) => ({
          ...current,
          status: "running",
          label: stageLabel(event.stepName),
          detail: "",
        }));
      },
      onStepFinishedEvent: ({ event }) => {
        setActivity((current) => ({
          ...current,
          label: stageLabel(event.stepName),
          detail: "Stage complete",
        }));
      },
      onCustomEvent: ({ event }) => {
        if (event.name === "audrey.stage.progress") {
          const value = recordOf(event.value);
          const stage = stringOf(value.stage);
          const delta = stringOf(value.delta);
          setActivity((current) => ({
            ...current,
            status: "running",
            label: stage ? stageLabel(stage) : current.label,
            detail: delta || current.detail,
          }));
        }
        if (event.name === "audrey.source.observed") {
          const value = recordOf(event.value);
          const title = stringOf(value.title);
          setActivity((current) => ({
            ...current,
            sourceCount: current.sourceCount + 1,
            latestSource: title || current.latestSource,
          }));
        }
      },
      onRunFinishedEvent: () => {
        onRunActiveChange(false);
        setActivity((current) => ({
          ...current,
          status: "complete",
          label: "Complete",
          detail: "Response finished",
        }));
      },
      onRunErrorEvent: ({ event }) => {
        onRunActiveChange(false);
        const cancelled = event.code === "cancelled_by_user" || isAbortMessage(event.message);
        setActivity((current) => ({
          ...current,
          status: cancelled ? "cancelled" : "error",
          label: cancelled ? "Stopped" : "Run failed",
          detail: cancelled ? "Run cancelled" : event.message || "The response did not finish cleanly",
        }));
      },
      onRunFailed: ({ error }) => {
        onRunActiveChange(false);
        const cancelled = isAbortError(error);
        setActivity((current) => ({
          ...current,
          status: cancelled ? "cancelled" : "error",
          label: cancelled ? "Stopped" : "Connection failed",
          detail: cancelled ? "Run cancelled" : error.message,
        }));
      },
    };
    const subscription = agent.subscribe(subscriber);
    return () => subscription.unsubscribe();
  }, [agent, onRunActiveChange]);
  const runtime = useAgUiRuntime({
    agent,
    adapters: { history },
    showThinking: false,
    onError: (reason) => {
      onRunActiveChange(false);
      setRunError(reason.message);
    },
    onCancel: () => {
      onRunActiveChange(false);
      setRunError("");
      setActivity((current) => ({
        ...current,
        status: "cancelled",
        label: "Stopped",
        detail: "Run cancelled",
      }));
    },
  });

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <ThreadPrimitive.Root className="thread-root">
        <ThreadPrimitive.Viewport className="thread-viewport">
          <ThreadPrimitive.Empty>
            <div className="thread-empty">
              <span>Ready</span>
              <h2>Ask Audrey anything.</h2>
              <p>The server will load this conversation's canonical history.</p>
            </div>
          </ThreadPrimitive.Empty>
          <ThreadPrimitive.Messages
            components={{
              UserMessage,
              AssistantMessage,
            }}
          />
          <ThreadPrimitive.ScrollToBottom className="scroll-bottom" aria-label="Scroll to latest message">
            ↓
          </ThreadPrimitive.ScrollToBottom>
          <ThreadPrimitive.ViewportFooter className="composer-dock">
            {readOnly ? (
              <p className="archived-notice" role="status">
                This conversation is archived. Restore it to continue.
              </p>
            ) : (
              <>
                {runError ? <p className="run-error" role="alert">{runError}</p> : null}
                <RunActivityStatus activity={activity} />
                <ComposerModelPicker
                  mode={mode}
                  disabled={modeDisabled}
                  onChange={onModeChange}
                />
                <ComposerPrimitive.Root className="composer">
                  <ComposerPrimitive.Input
                    className="composer-input"
                    aria-label="Message Audrey"
                    placeholder="Message Audrey…"
                    rows={1}
                  />
                  <div className="composer-actions">
                    <ComposerPrimitive.Cancel className="cancel-button">Stop</ComposerPrimitive.Cancel>
                    <ComposerPrimitive.Send className="send-button" aria-label="Send message">↑</ComposerPrimitive.Send>
                  </div>
                </ComposerPrimitive.Root>
                <p className="composer-hint">Enter to send · Shift+Enter for a new line</p>
              </>
            )}
          </ThreadPrimitive.ViewportFooter>
        </ThreadPrimitive.Viewport>
      </ThreadPrimitive.Root>
    </AssistantRuntimeProvider>
  );
}

function RunActivityStatus({ activity }: { activity: RunActivity }) {
  if (activity.status === "idle") return null;
  const sourceLabel = activity.sourceCount === 1 ? "1 source" : `${activity.sourceCount} sources`;
  return (
    <div className="run-activity" data-status={activity.status} role="status" aria-live="polite">
      <span className="run-activity-dot" aria-hidden="true" />
      <strong>{activity.label}</strong>
      {activity.detail ? <span>{activity.detail}</span> : null}
      {activity.sourceCount > 0 ? (
        <span className="run-sources">
          {sourceLabel}{activity.latestSource ? ` · ${activity.latestSource}` : ""}
        </span>
      ) : null}
    </div>
  );
}

function UserMessage() {
  return (
    <MessagePrimitive.Root className="message message-user">
      <div className="message-label">You</div>
      <MessagePrimitive.Parts components={{ Text: MarkdownText }} />
    </MessagePrimitive.Root>
  );
}

function AssistantMessage() {
  return (
    <MessagePrimitive.Root className="message message-assistant">
      <div className="message-label">Audrey</div>
      <MessagePrimitive.Parts
        components={{ Text: MarkdownText, tools: { Fallback: ToolActivity } }}
      />
    </MessagePrimitive.Root>
  );
}

function ComposerModelPicker({
  mode,
  disabled,
  onChange,
}: {
  mode: AudreyMode;
  disabled: boolean;
  onChange: (mode: AudreyMode) => Promise<void>;
}) {
  const selected = modeDetails(mode);
  return (
    <label className="composer-model-picker">
      <img src={selected.portrait} alt="" aria-hidden="true" />
      <span className="model-picker-copy">
        <span>Audrey</span>
        <strong>{selected.label}</strong>
      </span>
      <span className="model-picker-control">
        <span>Model</span>
        <select
          aria-label="Audrey model"
          value={mode}
          disabled={disabled}
          onChange={(event) => void onChange(event.target.value as AudreyMode)}
        >
          {MODES.map((item) => (
            <option key={item.value} value={item.value}>{item.label}</option>
          ))}
        </select>
      </span>
    </label>
  );
}

function MarkdownText({ text }: TextMessagePartProps) {
  return (
    <div className="markdown-content">
      <Markdown
        remarkPlugins={[remarkGfm]}
        skipHtml
        components={{
          a: ({ href, title, children }) => (
            <a href={href} title={title} target="_blank" rel="noreferrer noopener">
              {children}
            </a>
          ),
        }}
      >
        {text}
      </Markdown>
    </div>
  );
}

function ToolActivity({ toolName, args, result, status }: ToolCallMessagePartProps) {
  const finished = status.type === "complete";
  return (
    <details className="tool-activity">
      <summary>
        <span className={finished ? "tool-dot complete" : "tool-dot"} aria-hidden="true" />
        {toolName} · {finished ? "complete" : "running"}
      </summary>
      <pre>{JSON.stringify({ arguments: args, ...(result === undefined ? {} : { result }) }, null, 2)}</pre>
    </details>
  );
}

function toThreadMessages(messages: ConversationMessage[]): ThreadMessageLike[] {
  return messages.flatMap<ThreadMessageLike>((message) => {
    if (message.role === "user") {
      return [{ id: message.id, role: "user", content: message.content }];
    }
    if (message.role === "assistant") {
      return [{ id: message.id, role: "assistant", content: message.content }];
    }
    return [];
  });
}

function upsertConversation(
  conversations: Conversation[],
  updated: Conversation,
): Conversation[] {
  const index = conversations.findIndex(({ id }) => id === updated.id);
  if (index < 0) return [...conversations, updated];
  if (conversations[index] === updated) return conversations;
  return conversations.map((conversation) =>
    conversation.id === updated.id ? updated : conversation,
  );
}

function modeLabel(mode: AudreyMode): string {
  return modeDetails(mode).label;
}

function modeDetails(mode: AudreyMode) {
  return MODES.find((item) => item.value === mode) ?? MODES[0];
}

function messageOf(reason: unknown): string {
  return reason instanceof Error ? reason.message : "Audrey is unavailable.";
}

function recordOf(value: unknown): Record<string, unknown> {
  return value !== null && typeof value === "object"
    ? value as Record<string, unknown>
    : {};
}

function stringOf(value: unknown): string {
  return typeof value === "string" ? value : "";
}

function stageLabel(stage: string): string {
  const words = stage.replaceAll("_", " ").trim();
  return words ? words.charAt(0).toUpperCase() + words.slice(1) : "Working";
}

function isAbortError(error: Error): boolean {
  return error.name === "AbortError"
    || isAbortMessage(error.message);
}

function isAbortMessage(message: string | undefined): boolean {
  return message === "Fetch is aborted"
    || message === "signal is aborted without reason";
}
