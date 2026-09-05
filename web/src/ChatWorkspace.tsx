import { HttpAgent, type AgentSubscriber, type Message } from "@ag-ui/client";
import {
  AssistantRuntimeProvider,
  ComposerPrimitive,
  MessagePrimitive,
  ThreadPrimitive,
  type ToolCallMessagePartProps,
} from "@assistant-ui/react";
import { useAgUiRuntime } from "@assistant-ui/react-ag-ui";
import { useEffect, useMemo, useState } from "react";

import {
  createConversation,
  listConversations,
  listMessages,
  updateConversationMode,
  type AudreyMode,
  type Conversation,
  type ConversationMessage,
  type CurrentUser,
} from "./api";
import { latestActionFetch } from "./agentTransport";

const MODES: ReadonlyArray<{ value: AudreyMode; label: string }> = [
  { value: "auto", label: "Auto" },
  { value: "fast", label: "Fast" },
  { value: "deep", label: "Deep" },
  { value: "research", label: "Research" },
  { value: "local", label: "Local only" },
  { value: "cloud", label: "Cloud" },
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

const IDLE_ACTIVITY: RunActivity = {
  status: "idle",
  label: "Ready",
  detail: "",
  sourceCount: 0,
  latestSource: "",
};

export function ChatWorkspace({ user }: { user: CurrentUser }) {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [creating, setCreating] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    let active = true;
    listConversations()
      .then(({ items }) => {
        if (!active) return;
        setConversations(items);
        setSelectedId((current) => current ?? items[0]?.id ?? null);
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
  }, []);

  const selected = conversations.find(({ id }) => id === selectedId) ?? null;

  async function startConversation() {
    setCreating(true);
    setError("");
    try {
      const conversation = await createConversation("auto");
      setConversations((current) => [conversation, ...current]);
      setSelectedId(conversation.id);
    } catch (reason) {
      setError(messageOf(reason));
    } finally {
      setCreating(false);
    }
  }

  function replaceConversation(updated: Conversation) {
    setConversations((current) =>
      current.map((conversation) =>
        conversation.id === updated.id ? updated : conversation,
      ),
    );
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

        {loading ? <p className="sidebar-status">Loading conversations…</p> : null}
        {!loading && conversations.length === 0 ? (
          <p className="sidebar-status">No conversations yet.</p>
        ) : null}
        <nav className="conversation-list" aria-label="Conversation history">
          {conversations.map((conversation) => (
            <button
              className={conversation.id === selectedId ? "conversation active" : "conversation"}
              type="button"
              key={conversation.id}
              onClick={() => setSelectedId(conversation.id)}
              aria-current={conversation.id === selectedId ? "page" : undefined}
            >
              <span>{conversation.title || "Untitled conversation"}</span>
              <small>{modeLabel(conversation.default_mode)}</small>
            </button>
          ))}
        </nav>
        {error ? <p className="sidebar-error" role="alert">{error}</p> : null}
      </aside>

      <section className="chat-column" aria-label="Audrey conversation">
        {selected ? (
          <ConversationThread
            key={selected.id}
            conversation={selected}
            onConversationChange={replaceConversation}
          />
        ) : (
          <div className="empty-workspace">
            <span>Audrey</span>
            <h1>What shall we work through?</h1>
            <p>Start a conversation to use Audrey's native run pipeline.</p>
            <button type="button" onClick={startConversation} disabled={creating}>
              Start a conversation
            </button>
          </div>
        )}
      </section>
    </div>
  );
}

function ConversationThread({
  conversation,
  onConversationChange,
}: {
  conversation: Conversation;
  onConversationChange: (conversation: Conversation) => void;
}) {
  const [thread, setThread] = useState<ThreadState>({ status: "idle" });
  const [mode, setMode] = useState<AudreyMode>(conversation.default_mode);

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
    const previous = mode;
    setMode(next);
    try {
      onConversationChange(await updateConversationMode(conversation.id, next));
    } catch {
      setMode(previous);
    }
  }

  return (
    <>
      <header className="thread-header">
        <div>
          <span>Conversation</span>
          <h1>{conversation.title || "Untitled conversation"}</h1>
        </div>
        <label className="mode-picker">
          <span>Mode</span>
          <select value={mode} onChange={(event) => void changeMode(event.target.value as AudreyMode)}>
            {MODES.map((item) => (
              <option key={item.value} value={item.value}>{item.label}</option>
            ))}
          </select>
        </label>
      </header>

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
        />
      ) : null}
    </>
  );
}

function AudreyThread({
  conversationId,
  mode,
  initialMessages,
}: {
  conversationId: string;
  mode: AudreyMode;
  initialMessages: ConversationMessage[];
}) {
  const [runError, setRunError] = useState("");
  const [activity, setActivity] = useState<RunActivity>(IDLE_ACTIVITY);
  const agent = useMemo(
    () =>
      new HttpAgent({
        url: `/api/agent?mode=${encodeURIComponent(mode)}`,
        threadId: conversationId,
        initialMessages: toAgUiMessages(initialMessages),
        fetch: latestActionFetch,
      }),
    [conversationId, initialMessages, mode],
  );
  useEffect(() => {
    const subscriber: AgentSubscriber = {
      onRunInitialized: () => {
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
        setActivity((current) => ({
          ...current,
          status: "complete",
          label: "Complete",
          detail: "Response finished",
        }));
      },
      onRunErrorEvent: ({ event }) => {
        const cancelled = event.code === "cancelled_by_user" || isAbortMessage(event.message);
        setActivity((current) => ({
          ...current,
          status: cancelled ? "cancelled" : "error",
          label: cancelled ? "Stopped" : "Run failed",
          detail: cancelled ? "Run cancelled" : event.message || "The response did not finish cleanly",
        }));
      },
      onRunFailed: ({ error }) => {
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
  }, [agent]);
  const runtime = useAgUiRuntime({
    agent,
    showThinking: false,
    onError: (reason) => setRunError(reason.message),
    onCancel: () => {
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
            {runError ? <p className="run-error" role="alert">{runError}</p> : null}
            <RunActivityStatus activity={activity} />
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
      <MessagePrimitive.Parts />
    </MessagePrimitive.Root>
  );
}

function AssistantMessage() {
  return (
    <MessagePrimitive.Root className="message message-assistant">
      <div className="message-label">Audrey</div>
      <MessagePrimitive.Parts components={{ tools: { Fallback: ToolActivity } }} />
    </MessagePrimitive.Root>
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

function toAgUiMessages(messages: ConversationMessage[]): Message[] {
  return messages.flatMap<Message>((message) => {
    if (message.role === "user") {
      return [{ id: message.id, role: "user", content: message.content }];
    }
    if (message.role === "assistant") {
      return [{ id: message.id, role: "assistant", content: message.content }];
    }
    return [];
  });
}

function modeLabel(mode: AudreyMode): string {
  return MODES.find((item) => item.value === mode)?.label ?? mode;
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
