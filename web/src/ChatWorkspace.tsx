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
  useAuiState,
} from "@assistant-ui/react";
import { useAgUiRuntime } from "@assistant-ui/react-ag-ui";
import { createContext, useContext, useEffect, useMemo, useRef, useState } from "react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";

import autoPortrait from "./assets/models/audrey2.png";
import deepPortrait from "./assets/models/audrey7.png";
import fastPortrait from "./assets/models/audrey3.png";
import researchPortrait from "./assets/models/audrey8.png";
import videoPortrait from "./assets/models/audrey9.png";
import cloudPortrait from "./assets/models/cloudModel.png";
import localPortrait from "./assets/models/localModel.png";

import { AudreyLoader } from "./AudreyLoader";
import {
  cancelRun,
  createConversation,
  deleteConversation,
  discardEmptyConversation,
  getConversation,
  getRun,
  listConversations,
  listFiles,
  getFile,
  listMessages,
  uploadFile,
  uploadPrecheck,
  updateConversation,
  updateConversationModel,
  type AudreyFile,
  type AudreyFileLimits,
  type AudreyModel,
  type Conversation,
  type ConversationMessage,
  type MessageSource,
  type CurrentUser,
  type UserPreferences,
} from "./api";
import { latestActionFetch } from "./agentTransport";
import { FileManager } from "./FileManager";

const MODEL_PORTRAITS: Readonly<Record<string, string>> = {
  auto: autoPortrait,
  fast: fastPortrait,
  deep: deepPortrait,
  research: researchPortrait,
  video: videoPortrait,
  cloud: cloudPortrait,
  local: localPortrait,
};


type ThreadState =
  | { status: "idle" }
  | { status: "loading" }
  | { status: "ready"; messages: ConversationMessage[] }
  | { status: "error"; message: string };

type RunActivity = {
  status: "idle" | "running" | "complete" | "cancelled" | "error";
  label: string;
  detail: string;
  sources: RunSource[];
};

type RunSource = {
  id: string;
  title: string;
  url: string;
};

type LastAttempt = {
  text: string;
  attachmentIds: string[];
};

type ConversationView = "active" | "archived";

const SavedSourcesContext = createContext<ReadonlyMap<string, MessageSource[]>>(new Map());

const IDLE_ACTIVITY: RunActivity = {
  status: "idle",
  label: "Ready",
  detail: "",
  sources: [],
};

export function ChatWorkspace({
  user,
  preferences,
  models,
}: {
  user: CurrentUser;
  preferences: UserPreferences;
  models: AudreyModel[];
}) {
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
  const [managingFiles, setManagingFiles] = useState(false);
  const [confirmDeleteId, setConfirmDeleteId] = useState<string | null>(null);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const listKeyRef = useRef("");
  const selectedIdRef = useRef<string | null>(null);
  const defaultModelId = models.find(({ id }) => id === "auto")?.id ?? models[0]?.id ?? null;
  const catalogUnavailable = defaultModelId === null;
  const canBrowseDirectModels = models.some((model) =>
    model.kind === "direct"
    && (user.role === "admin"
      || user.groups.includes("admins")
      || model.roles?.some((role) => user.groups.includes(role))),
  );

  function selectConversation(conversation: Conversation | null) {
    const previousId = selectedIdRef.current;
    if (previousId && previousId !== conversation?.id) {
      const previous = openedConversations.find(({ id }) => id === previousId)
        ?? conversations.find(({ id }) => id === previousId);
      if (previous?.last_message_at === null) {
        void discardEmptyConversation(previousId)
          .then(() => {
            setConversations((current) => current.filter(({ id }) => id !== previousId));
            setOpenedConversations((current) => current.filter(({ id }) => id !== previousId));
          })
          .catch(async () => {
            // The server may have accepted a message after the list snapshot.
            try {
              replaceConversation(await getConversation(previousId));
            } catch {
              // A deleted draft has no history row to restore.
            }
          });
      }
    }
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
    if (!defaultModelId) return;
    let active = true;
    const requestKey = `${view}\n${searchQuery}`;
    listKeyRef.current = requestKey;
    listConversations({ archived: view === "archived", search: searchQuery })
      .then(async ({ items, next_cursor }) => {
        if (!active) return;
        if (view === "active" && !searchQuery && items.length === 0) {
          const conversation = await createConversation(defaultModelId);
          if (!active) return;
          setConversations([conversation]);
          setNextCursor(null);
          selectConversation(conversation);
          return;
        }
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
  }, [defaultModelId, searchQuery, view]);

  const selected = conversations.find(({ id }) => id === selectedId) ?? null;
  const historyConversations = conversations.filter(({ last_message_at }) => last_message_at !== null);
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
    if (!defaultModelId) return;
    setCreating(true);
    setError("");
    try {
      const conversation = await createConversation(defaultModelId);
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
      if (remaining.length === 0 && view === "active" && !searchQuery) {
        void startConversation();
      }
    }
  }

  async function deleteFromSidebar(conversationId: string) {
    setDeletingId(conversationId);
    setError("");
    try {
      await deleteConversation(conversationId);
      setConfirmDeleteId(null);
      removeFromCurrentView(conversationId, true);
    } catch (reason) {
      setError(messageOf(reason));
    } finally {
      setDeletingId(null);
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
          <div className="sidebar-heading-actions">
            <button
              className="manage-files"
              type="button"
              onClick={() => setManagingFiles(true)}
            >
              Files
            </button>
            <button
              className="new-conversation"
              type="button"
              onClick={startConversation}
              disabled={creating || loading || catalogUnavailable}
            >
              {creating ? "Creating…" : "+ New"}
            </button>
          </div>
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

        {!catalogUnavailable && loading ? (
          <p className="sidebar-status">Loading conversations…</p>
        ) : null}
        {catalogUnavailable ? (
          <p className="sidebar-error" role="alert">No models are enabled for this account.</p>
        ) : null}
        {!catalogUnavailable && !loading && historyConversations.length === 0 ? (
          <p className="sidebar-status">
            {searchQuery
              ? "No matching conversation titles."
              : view === "archived"
                ? "No archived conversations."
                : "No conversations yet."}
          </p>
        ) : null}
        <nav className="conversation-list" aria-label="Conversation history">
          {historyConversations.map((conversation) => (
            <div className="conversation-row" key={conversation.id}>
              <button
                className={conversation.id === selectedId ? "conversation active" : "conversation"}
                type="button"
                onClick={() => selectConversation(conversation)}
                aria-current={conversation.id === selectedId ? "page" : undefined}
              >
                <span>{conversation.title || "New conversation"}</span>
                <small>{modelLabel(models, conversation)}</small>
              </button>
              <button
                className={confirmDeleteId === conversation.id
                  ? "conversation-delete confirming-delete"
                  : "conversation-delete"}
                type="button"
                aria-label={`${confirmDeleteId === conversation.id ? "Confirm delete" : "Delete"} conversation ${conversation.title || "New conversation"}`}
                aria-pressed={confirmDeleteId === conversation.id}
                title={confirmDeleteId === conversation.id ? "Click again to delete" : "Delete conversation"}
                onClick={() => {
                  if (confirmDeleteId === conversation.id) {
                    void deleteFromSidebar(conversation.id);
                  } else {
                    setConfirmDeleteId(conversation.id);
                  }
                }}
                onBlur={() => setConfirmDeleteId((current) =>
                  current === conversation.id ? null : current)}
                onKeyDown={(event) => {
                  if (event.key === "Escape") setConfirmDeleteId(null);
                }}
                disabled={deletingId !== null}
              >
                {deletingId === conversation.id ? (
                  <span aria-hidden="true">…</span>
                ) : confirmDeleteId === conversation.id ? (
                  <span aria-hidden="true">✓</span>
                ) : (
                  <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                    <path d="M4 7h16M9 7V4h6v3m3 0-1 13H7L6 7m4 4v6m4-6v6" />
                  </svg>
                )}
              </button>
            </div>
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
        {catalogUnavailable ? (
          <div className="catalog-unavailable">
            <h2>No models available</h2>
            <p>An Audrey administrator can enable a model from the Admin panel.</p>
          </div>
        ) : renderedConversations.map((opened) => (
          <div
            className="conversation-thread-slot"
            hidden={opened.id !== selectedId}
            key={opened.id}
          >
            <ConversationThread
              conversation={opened}
              models={models}
              canBrowseDirectModels={canBrowseDirectModels}
              showProgress={preferences.show_progress}
              onConversationChange={replaceConversation}
              onRemoveFromView={removeFromCurrentView}
            />
          </div>
        ))}
        {!catalogUnavailable && !selected && (loading || creating) ? (
          <AudreyLoader label="Opening conversation" showPortrait={false} />
        ) : null}
      </section>
      {managingFiles ? <FileManager onClose={() => setManagingFiles(false)} /> : null}
    </div>
  );
}

function ConversationThread({
  conversation,
  models,
  canBrowseDirectModels,
  showProgress,
  onConversationChange,
  onRemoveFromView,
}: {
  conversation: Conversation;
  models: AudreyModel[];
  canBrowseDirectModels: boolean;
  showProgress: boolean;
  onConversationChange: (conversation: Conversation) => void;
  onRemoveFromView: (conversationId: string, closeThread?: boolean) => void;
}) {
  const [thread, setThread] = useState<ThreadState>({ status: "idle" });
  const [modelId, setModelId] = useState(
    models.some(({ id }) => id === conversation.default_model_id)
      ? conversation.default_model_id
      : (models.find(({ id }) => id === "auto") ?? models[0]).id,
  );
  const [editingTitle, setEditingTitle] = useState(false);
  const [titleDraft, setTitleDraft] = useState(conversation.title);
  const [mutation, setMutation] = useState<"model" | "rename" | "archive" | "delete" | null>(null);
  const [mutationError, setMutationError] = useState("");
  const [confirmingDelete, setConfirmingDelete] = useState(false);
  const [runActive, setRunActive] = useState(false);
  const [recoveredRunId, setRecoveredRunId] = useState<string | null>(null);
  const [recoveryError, setRecoveryError] = useState("");
  const [stoppingRecovered, setStoppingRecovered] = useState(false);
  const [threadRevision, setThreadRevision] = useState(0);
  const onConversationChangeRef = useRef(onConversationChange);
  useEffect(() => { onConversationChangeRef.current = onConversationChange; }, [onConversationChange]);
  const archived = conversation.archived_at !== null;
  const selectedModelId = models.some(({ id }) => id === modelId)
    ? modelId
    : (models.find(({ id }) => id === "auto") ?? models[0]).id;

  useEffect(() => {
    let active = true;
    async function restoreThread() {
      try {
        let { items } = await listMessages(conversation.id);
        const latestAssistant = items.filter(({ role }) => role === "assistant").at(-1);
        if (latestAssistant && latestAssistant.status !== "completed" && latestAssistant.run_id) {
          const run = await getRun(latestAssistant.run_id);
          if (run.conversation_id !== conversation.id) throw new Error("Run belongs to another conversation.");
          if (run.status === "running") {
            if (active) {
              setRecoveredRunId(run.id);
              setRunActive(true);
            }
          } else {
            items = (await listMessages(conversation.id)).items;
          }
        }
        if (active) setThread({ status: "ready", messages: items });
      } catch (reason) {
        if (active) setThread({ status: "error", message: messageOf(reason) });
      }
    }
    void restoreThread();
    return () => { active = false; };
  }, [conversation.id]);

  useEffect(() => {
    if (!recoveredRunId) return;
    let active = true;
    let checking = false;
    async function checkRun() {
      if (checking || !active) return;
      checking = true;
      try {
        const run = await getRun(recoveredRunId!);
        if (!active) return;
        if (run.conversation_id !== conversation.id) throw new Error("Run belongs to another conversation.");
        setRecoveryError("");
        if (run.status !== "running") {
          const messages = await listMessages(conversation.id);
          if (!active) return;
          setThread({ status: "ready", messages: messages.items });
          setThreadRevision((revision) => revision + 1);
          setRecoveredRunId(null);
          setRunActive(false);
          void getConversation(conversation.id).then((updated) => {
            if (active) {
              setTitleDraft(updated.title);
              onConversationChangeRef.current(updated);
            }
          }).catch(() => {});
        }
      } catch (reason) {
        if (active) setRecoveryError(messageOf(reason));
      } finally {
        checking = false;
      }
    }
    const timer = window.setInterval(() => {
      if (document.visibilityState !== "hidden") void checkRun();
    }, 3000);
    document.addEventListener("visibilitychange", checkRun);
    return () => {
      active = false;
      window.clearInterval(timer);
      document.removeEventListener("visibilitychange", checkRun);
    };
  }, [conversation.id, recoveredRunId]);

  async function stopRecoveredRun() {
    if (!recoveredRunId || stoppingRecovered) return;
    setStoppingRecovered(true);
    setRecoveryError("");
    try {
      const run = await cancelRun(recoveredRunId);
      if (run.status !== "running") {
        const messages = await listMessages(conversation.id);
        setThread({ status: "ready", messages: messages.items });
        setThreadRevision((revision) => revision + 1);
        setRecoveredRunId(null);
        setRunActive(false);
      }
    } catch (reason) {
      setRecoveryError(messageOf(reason));
    } finally {
      setStoppingRecovered(false);
    }
  }

  async function changeModel(next: string) {
    if (next === modelId || runActive) return;
    setMutation("model");
    setMutationError("");
    try {
      const updated = await updateConversationModel(conversation.id, next);
      const messages = await listMessages(conversation.id);
      setThread({ status: "ready", messages: messages.items });
      setModelId(updated.default_model_id);
      onConversationChange(updated);
    } catch (reason) {
      setMutationError(messageOf(reason));
    } finally {
      setMutation(null);
    }
  }

  async function refreshAutomaticTitle() {
    try {
      const updated = await getConversation(conversation.id);
      setTitleDraft(updated.title);
      onConversationChange(updated);
    } catch {
      // Canonical state remains correct; a later list or refresh will reconcile it.
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
                <h1>{conversation.title || "New conversation"}</h1>
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
              <button
                className={confirmingDelete ? "danger-button confirming-delete" : "danger-button"}
                type="button"
                aria-label={confirmingDelete ? "Confirm delete conversation" : "Delete conversation"}
                aria-pressed={confirmingDelete}
                title={confirmingDelete ? "Click again to delete" : "Delete conversation"}
                onClick={() => {
                  if (confirmingDelete) {
                    void removeConversation();
                  } else {
                    setConfirmingDelete(true);
                  }
                }}
                onBlur={() => setConfirmingDelete(false)}
                onKeyDown={(event) => {
                  if (event.key === "Escape") setConfirmingDelete(false);
                }}
                disabled={runActive || mutation !== null}
              >
                {mutation === "delete" ? "Deleting…" : confirmingDelete ? "✓" : "Delete"}
              </button>
            </div>
          </div>
        </header>
        {mutationError ? <p className="conversation-mutation-error" role="alert">{mutationError}</p> : null}
      </div>

      {thread.status === "loading" || thread.status === "idle" ? (
        <AudreyLoader label="Loading conversation" showPortrait={false} />
      ) : null}
      {thread.status === "error" ? (
        <div className="thread-loading thread-error" role="alert">{thread.message}</div>
      ) : null}
      {recoveredRunId ? (
        <div className="recovered-run" role="status" aria-label="Audrey is answering">
          <span className="recovered-run-orb" aria-hidden="true" />
          <button type="button" onClick={() => void stopRecoveredRun()} disabled={stoppingRecovered}>
            {stoppingRecovered ? "Stopping…" : "Stop current run"}
          </button>
          {recoveryError ? <span>Could not check run: {recoveryError}</span> : null}
        </div>
      ) : null}
      {thread.status === "ready" ? (
        <AudreyThread
          key={threadRevision}
          conversationId={conversation.id}
          models={models}
          canBrowseDirectModels={canBrowseDirectModels}
          modelId={selectedModelId}
          showProgress={showProgress}
          initialMessages={thread.messages}
          readOnly={archived}
          modeDisabled={runActive || mutation !== null}
          onModelChange={changeModel}
          onRunActiveChange={setRunActive}
          onRunStarted={() => void refreshAutomaticTitle()}
        />
      ) : null}
    </>
  );
}

function AudreyThread({
  conversationId,
  models,
  canBrowseDirectModels,
  modelId,
  showProgress,
  initialMessages,
  readOnly,
  modeDisabled,
  onModelChange,
  onRunActiveChange,
  onRunStarted,
}: {
  conversationId: string;
  canBrowseDirectModels: boolean;
  models: AudreyModel[];
  modelId: string;
  showProgress: boolean;
  initialMessages: ConversationMessage[];
  readOnly: boolean;
  modeDisabled: boolean;
  onModelChange: (modelId: string) => Promise<void>;
  onRunActiveChange: (active: boolean) => void;
  onRunStarted: () => void;
}) {
  const [runError, setRunError] = useState("");
  const [activity, setActivity] = useState<RunActivity>(IDLE_ACTIVITY);
  const restoredIncomplete = initialMessages.filter(({ role }) => role === "assistant").at(-1)?.status === "incomplete";
  const [lastAttempt, setLastAttempt] = useState<LastAttempt | null>(() => {
    const assistant = initialMessages.filter(({ role }) => role === "assistant").at(-1);
    if (assistant?.status !== "incomplete" || !assistant.run_id) return null;
    const user = [...initialMessages].reverse().find((message) =>
      message.role === "user" && message.run_id === assistant.run_id);
    return user ? {
      text: user.content,
      attachmentIds: user.attachments.map(({ id }) => id),
    } : null;
  });
  const [retrying, setRetrying] = useState(false);
  const [queuedRetry, setQueuedRetry] = useState<{ text: string } | null>(null);
  const dispatchedRetryRef = useRef<object | null>(null);
  const [attachmentPickerOpen, setAttachmentPickerOpen] = useState(false);
  const [attachmentFiles, setAttachmentFiles] = useState<AudreyFile[]>([]);
  const [attachmentsLoading, setAttachmentsLoading] = useState(false);
  const [attachmentError, setAttachmentError] = useState("");
  const [selectedAttachments, setSelectedAttachments] = useState<AudreyFile[]>([]);
  const [imageLimit, setImageLimit] = useState<number | null>(null);
  const [fileLimits, setFileLimits] = useState<AudreyFileLimits | null>(null);
  const [uploadingFile, setUploadingFile] = useState("");
  const [uploadProgress, setUploadProgress] = useState(0);
  const [pendingAttachment, setPendingAttachment] = useState<AudreyFile | null>(null);
  const [uploadIssue, setUploadIssue] = useState("");
  const uploadInputRef = useRef<HTMLInputElement>(null);
  const composerInputRef = useRef<HTMLTextAreaElement>(null);
  const attachmentBusy = Boolean(uploadingFile) || pendingAttachment !== null;
  const submissionBlocked = modeDisabled || attachmentBusy || retrying || Boolean(uploadIssue);
  const selectedImageCount = selectedAttachments.filter(({ kind }) => kind === "image").length;
  const selectedModel = modelDetails(models, modelId);
  const supportsFiles = selectedModel.capabilities.includes("files");
  const attachmentIds = useMemo(
    () => selectedAttachments.map(({ id }) => id),
    [selectedAttachments],
  );
  const savedSources = useMemo(() => new Map(
    initialMessages.filter(({ role }) => role === "assistant")
      .map(({ id, sources }) => [id, sources ?? []] as const),
  ), [initialMessages]);
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
        url: `/api/agent?model=${encodeURIComponent(modelId)}`,
        threadId: conversationId,
        fetch: (url, init) => latestActionFetch(url, init, attachmentIds),
      }),
    [attachmentIds, conversationId, modelId],
  );
  async function changeModel(nextModelId: string) {
    if (attachmentBusy || retrying) return;
    await onModelChange(nextModelId);
    const nextModel = modelDetails(models, nextModelId);
    if (!nextModel.capabilities.includes("files")) {
      setAttachmentPickerOpen(false);
      setSelectedAttachments([]);
    }
  }
  const onRunStartedRef = useRef(onRunStarted);
  useEffect(() => {
    onRunStartedRef.current = onRunStarted;
  }, [onRunStarted]);
  useEffect(() => {
    const subscriber: AgentSubscriber = {
      onRunInitialized: () => {
        onRunActiveChange(true);
        setRetrying(false);
        setRunError("");
        setActivity({
          status: "running",
          label: "Starting",
          detail: "Preparing Audrey's run",
          sources: [],
        });
      },
      onRunStartedEvent: () => {
        setAttachmentPickerOpen(false);
        onRunStartedRef.current();
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
          const id = stringOf(value.sourceId) || stringOf(value.url);
          if (id) {
            const source = {
              id,
              title: stringOf(value.title),
              url: safeSourceUrl(stringOf(value.url)),
            };
            setActivity((current) => current.sources.some((item) => item.id === id)
              ? current
              : { ...current, sources: [...current.sources, source] });
          }
        }
      },
      onRunFinishedEvent: () => {
        onRunActiveChange(false);
        setRetrying(false);
        setLastAttempt(null);
        setSelectedAttachments([]);
        setActivity((current) => ({
          ...current,
          status: "complete",
          label: "Complete",
          detail: "Response finished",
        }));
      },
      onRunErrorEvent: ({ event }) => {
        onRunActiveChange(false);
        setRetrying(false);
        setSelectedAttachments([]);
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
        setRetrying(false);
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
      setRetrying(false);
      setRunError(reason.message);
      setActivity((current) => current.status === "error" ? current : {
        ...current,
        status: "error",
        label: "Run failed",
        detail: reason.message,
      });
    },
    onCancel: () => {
      onRunActiveChange(false);
      setSelectedAttachments([]);
      setRunError("");
      setActivity((current) => ({
        ...current,
        status: "cancelled",
        label: "Stopped",
        detail: "Run cancelled",
      }));
    },
  });

  async function retryLastQuestion() {
    if (!lastAttempt || retrying || modeDisabled || attachmentBusy || readOnly) return;
    setRetrying(true);
    setRunError("");
    try {
      if (lastAttempt.attachmentIds.length > 0 && !supportsFiles) {
        throw new Error("The selected model accepts text only. Choose a file-capable model to retry this question.");
      }
      const attachments = await Promise.all(lastAttempt.attachmentIds.map((id) => getFile(id)));
      if (attachments.some(({ status }) => status !== "ready")) {
        throw new Error("An attached file is no longer ready. Choose a ready file before sending again.");
      }
      if (imageLimit !== null && attachments.filter(({ kind }) => kind === "image").length > imageLimit) {
        throw new Error("The image limit changed. Remove an image before sending again.");
      }
      setSelectedAttachments(attachments);
      setUploadIssue("");
      setQueuedRetry({ text: lastAttempt.text });
    } catch (reason) {
      setRetrying(false);
      setRunError("Could not retry: " + messageOf(reason));
      setActivity((current) => ({ ...current, status: "error", label: "Run failed" }));
    }
  }

  useEffect(() => {
    if (!queuedRetry || dispatchedRetryRef.current === queuedRetry) return;
    // Defer until the runtime has installed the agent for the validated IDs.
    const timer = window.setTimeout(() => {
      if (dispatchedRetryRef.current === queuedRetry) return;
      dispatchedRetryRef.current = queuedRetry;
      setActivity({ status: "running", label: "Retrying", detail: "Starting another run", sources: [] });
      try {
        runtime.thread.append(queuedRetry.text);
      } catch (reason) {
        setRunError("Could not retry: " + messageOf(reason));
        setActivity((current) => ({ ...current, status: "error", label: "Run failed" }));
      } finally {
        setQueuedRetry(null);
        setRetrying(false);
      }
    }, 0);
    return () => window.clearTimeout(timer);
  }, [queuedRetry, runtime]);

  async function toggleAttachmentPicker() {
    if (attachmentPickerOpen) {
      setAttachmentPickerOpen(false);
      return;
    }
    setAttachmentPickerOpen(true);
    setAttachmentsLoading(true);
    setAttachmentError("");
    try {
      const listing = await listFiles();
      setAttachmentFiles(listing.items.filter(({ status }) => status === "ready"));
      setFileLimits(listing.limits);
      setImageLimit(Number.isInteger(listing.limits.max_images_per_turn)
        ? Math.max(0, listing.limits.max_images_per_turn)
        : null);
    } catch (reason) {
      setAttachmentError(messageOf(reason));
    } finally {
      setAttachmentsLoading(false);
    }
  }

  function toggleAttachment(file: AudreyFile) {
    if (attachmentBusy) return;
    setSelectedAttachments((current) => {
      if (current.some(({ id }) => id === file.id)) {
        return current.filter(({ id }) => id !== file.id);
      }
      if (current.length >= 10) return current;
      if (
        file.kind === "image"
        && imageLimit !== null
        && current.filter(({ kind }) => kind === "image").length >= imageLimit
      ) return current;
      return [...current, file];
    });
  }

  async function uploadFromChat(file: File) {
    if (!fileLimits || attachmentBusy || selectedAttachments.length >= 10) return;
    const precheck = uploadPrecheck(file, fileLimits);
    if (precheck) {
      setUploadIssue(file.name + ": " + precheck);
      return;
    }
    setUploadIssue("");
    setUploadingFile(file.name);
    setUploadProgress(0);
    let stored = false;
    try {
      const uploaded = await uploadFile(file, fileLimits, setUploadProgress);
      stored = true;
      // The upload response has no failure details; fetch the owned row before attaching.
      const row = await getFile(uploaded.id);
      if (row.status === "ready") {
        if (row.kind === "image" && imageLimit !== null && selectedImageCount >= imageLimit) {
          setUploadIssue(row.filename + " was uploaded to Files, but the image limit is full. Remove an image to attach it.");
          setAttachmentFiles((current) => [row, ...current]);
          return;
        }
        setSelectedAttachments((current) => [...current, row]);
        setAttachmentFiles((current) => [row, ...current.filter(({ id }) => id !== row.id)]);
      } else {
        setPendingAttachment(row);
      }
      setAttachmentPickerOpen(false);
    } catch (reason) {
      setUploadIssue(stored
        ? file.name + " was uploaded to Files, but could not be attached: " + messageOf(reason)
        : file.name + ": " + messageOf(reason));
    } finally {
      setUploadingFile("");
      setUploadProgress(0);
    }
  }

  const pendingId = pendingAttachment?.id;
  const pendingStatus = pendingAttachment?.status;
  useEffect(() => {
    if (!pendingId || pendingStatus === "failed") return;
    const fileId = pendingId;
    let active = true;
    let checking = false;
    async function checkReady() {
      if (!active || checking || document.visibilityState === "hidden") return;
      checking = true;
      try {
        const row = await getFile(fileId);
        if (!active) return;
        if (row.status === "ready") {
          setSelectedAttachments((current) => current.some(({ id }) => id === row.id)
            ? current : [...current, row]);
          setAttachmentFiles((current) => [row, ...current.filter(({ id }) => id !== row.id)]);
          setPendingAttachment(null);
        } else {
          setPendingAttachment(row);
        }
      } catch {
        // A transient list/read failure should not silently detach a processing video.
      } finally {
        checking = false;
      }
    }
    const timer = window.setInterval(() => void checkReady(), 5000);
    document.addEventListener("visibilitychange", checkReady);
    return () => {
      active = false;
      window.clearInterval(timer);
      document.removeEventListener("visibilitychange", checkReady);
    };
  }, [pendingId, pendingStatus]);

  return (
    <SavedSourcesContext.Provider value={savedSources}>
    <AssistantRuntimeProvider runtime={runtime}>
      <ThreadPrimitive.Root className="thread-root">
        <ThreadPrimitive.Viewport className="thread-viewport">
          <ThreadPrimitive.Messages
            components={{
              UserMessage,
              AssistantMessage,
            }}
          />
          <ThreadPrimitive.ViewportFooter className="composer-dock">
            <ThreadPrimitive.ScrollToBottom
              className="scroll-bottom"
              aria-label="Scroll to latest message"
            >
              <svg viewBox="0 0 24 24" aria-hidden="true">
                <path d="M12 5v14m5.5-5.5L12 19l-5.5-5.5" />
              </svg>
            </ThreadPrimitive.ScrollToBottom>
            {readOnly ? (
              <p className="archived-notice" role="status">
                This conversation is archived. Restore it to continue.
              </p>
            ) : (
              <>
                {(showProgress || activity.status === "error") ? (
                  <RunActivityStatus activity={activity} error={runError} />
                ) : null}
                {lastAttempt && (restoredIncomplete || activity.status === "error" || runError) ? (
                  <button
                    className="retry-button"
                    type="button"
                    onClick={() => void retryLastQuestion()}
                    disabled={retrying || modeDisabled || attachmentBusy}
                  >{retrying ? "Checking attachments…" : "Retry last question"}</button>
                ) : null}
                <ThreadPrimitive.Empty>
                  <ComposerModelPicker
                    models={models}
                    canBrowseDirectModels={canBrowseDirectModels}
                    modelId={modelId}
                    disabled={modeDisabled || attachmentBusy || retrying}
                    onChange={changeModel}
                  />
                </ThreadPrimitive.Empty>
                {selectedAttachments.length > 0 || pendingAttachment ? (
                  <div className="selected-attachments" aria-label="Selected attachments">
                    {selectedAttachments.map((file) => (
                      <button
                        type="button"
                        key={file.id}
                        onClick={() => toggleAttachment(file)}
                        disabled={modeDisabled || attachmentBusy}
                        aria-label={`Remove attachment ${file.filename}`}
                      >
                        <span aria-hidden="true">×</span>
                        {file.filename}
                      </button>
                    ))}
                    {pendingAttachment ? (
                      <button
                        type="button"
                        className="pending-attachment"
                        onClick={() => setPendingAttachment(null)}
                        aria-label={`Remove attachment ${pendingAttachment.filename}`}
                      >
                        <span aria-hidden="true">×</span>
                        {pendingAttachment.filename} · {pendingAttachment.status === "failed"
                          ? "Processing failed" : "Processing…"}
                      </button>
                    ) : null}
                  </div>
                ) : null}
                {uploadingFile ? (
                  <p className="attachment-status" role="status">
                    Uploading {uploadingFile} · {Math.round(uploadProgress * 100)}%
                  </p>
                ) : null}
                {pendingAttachment ? (
                  <p className="attachment-status" role="status">
                    {pendingAttachment.status === "failed"
                      ? `${pendingAttachment.filename}: ${pendingAttachment.failure_reason || "Processing failed"}. Remove it to continue.`
                      : `${pendingAttachment.filename} is processing. You can write your question; send is available when it is ready.`}
                  </p>
                ) : null}
                {uploadIssue ? (
                  <div className="attachment-issue" role="alert">
                    <span>{uploadIssue}</span>
                    <button type="button" onClick={() => setUploadIssue("")}>Dismiss</button>
                  </div>
                ) : null}
                {attachmentPickerOpen ? (
                  <section className="attachment-picker" aria-label="Choose attachments">
                    <header>
                      <strong>Attach your files</strong>
                      <span>
                        {selectedAttachments.length}/10 files
                        {imageLimit === null ? "" : " · " + selectedImageCount + "/" + imageLimit + " images"}
                      </span>
                    </header>
                    <div className="attachment-upload">
                      <button
                        type="button"
                        onClick={() => uploadInputRef.current?.click()}
                        disabled={!fileLimits || attachmentBusy || selectedAttachments.length >= 10}
                      >Upload from device</button>
                      <input
                        ref={uploadInputRef}
                        type="file"
                        aria-label="Choose a file from your device"
                        tabIndex={-1}
                        accept={fileLimits?.allowed_extensions.join(",")}
                        onChange={(event) => {
                          const file = event.currentTarget.files?.[0];
                          event.currentTarget.value = "";
                          if (file) void uploadFromChat(file);
                        }}
                      />
                    </div>
                    {attachmentsLoading ? <p role="status">Loading files…</p> : null}
                    {attachmentError ? <p className="attachment-error" role="alert">{attachmentError}</p> : null}
                    {!attachmentsLoading && !attachmentError && attachmentFiles.length === 0 ? (
                      <p>No ready files yet. Upload one here to ask about it.</p>
                    ) : null}
                    {attachmentFiles.length > 0 ? (
                      <div className="attachment-options">
                        {attachmentFiles.map((file) => {
                          const selected = selectedAttachments.some(({ id }) => id === file.id);
                          return (
                            <button
                              type="button"
                              key={file.id}
                              aria-pressed={selected}
                              onClick={() => toggleAttachment(file)}
                              disabled={attachmentBusy || (!selected && (
                                selectedAttachments.length >= 10
                                || (file.kind === "image"
                                  && imageLimit !== null
                                  && selectedImageCount >= imageLimit)
                              ))}
                            >
                              <span aria-hidden="true">{selected ? "✓" : "+"}</span>
                              <span>{file.filename}</span>
                              <small>{file.kind} · {formatBytes(file.bytes)}</small>
                            </button>
                          );
                        })}
                      </div>
                    ) : null}
                  </section>
                ) : null}
                <ComposerPrimitive.Root
                  className="composer"
                  onSubmitCapture={(event) => {
                    if (submissionBlocked) {
                      event.preventDefault();
                      event.stopPropagation();
                      return;
                    }
                    const text = composerInputRef.current?.value.trim() ?? "";
                    if (text) setLastAttempt({ text, attachmentIds: selectedAttachments.map(({ id }) => id) });
                  }}
                >
                  <ThreadPrimitive.If empty={false}>
                    <ComposerModelPicker
                      compact
                      models={models}
                      canBrowseDirectModels={canBrowseDirectModels}
                      modelId={modelId}
                      disabled={modeDisabled || attachmentBusy || retrying}
                      onChange={changeModel}
                    />
                  </ThreadPrimitive.If>
                  <button
                    className="attach-button"
                    type="button"
                    onClick={() => void toggleAttachmentPicker()}
                    disabled={modeDisabled || retrying || !supportsFiles}
                    title={supportsFiles ? "Attach files" : `${selectedModel.label} accepts text only`}
                    aria-label={attachmentPickerOpen ? "Close attachment picker" : "Attach files"}
                    aria-expanded={attachmentPickerOpen}
                  >
                    <svg viewBox="0 0 24 24" aria-hidden="true">
                      <path d="m9.5 12.5 5.4-5.4a3 3 0 0 1 4.2 4.2l-7.5 7.5a5 5 0 0 1-7.1-7.1l7.2-7.2" />
                    </svg>
                  </button>
                  <ComposerPrimitive.Input
                    ref={composerInputRef}
                    className="composer-input"
                    aria-label="Ask Audrey"
                    placeholder="Ask Audrey…"
                    rows={1}
                  />
                  <div className="composer-actions">
                    <ComposerPrimitive.Cancel className="cancel-button">Stop</ComposerPrimitive.Cancel>
                    <ComposerPrimitive.Send
                      className="send-button"
                      aria-label="Send message"
                      disabled={submissionBlocked}
                    >
                      <svg viewBox="0 0 24 24" aria-hidden="true">
                        <path d="M12 19V5M6.5 10.5 12 5l5.5 5.5" />
                      </svg>
                    </ComposerPrimitive.Send>
                  </div>
                </ComposerPrimitive.Root>
              </>
            )}
          </ThreadPrimitive.ViewportFooter>
        </ThreadPrimitive.Viewport>
      </ThreadPrimitive.Root>
    </AssistantRuntimeProvider>
    </SavedSourcesContext.Provider>
  );
}

function RunActivityStatus({ activity, error }: { activity: RunActivity; error: string }) {
  if (activity.status === "idle") return null;
  const sourceCount = activity.sources.length;
  const sourceLabel = sourceCount === 1 ? "1 source found" : String(sourceCount) + " sources found";
  const latestSource = activity.sources.at(-1)?.title;
  return (
    <div className="run-activity" data-status={activity.status} role={activity.status === "error" ? "alert" : "status"} aria-live="polite">
      <span className="run-activity-dot" aria-hidden="true" />
      <strong>{activity.label}</strong>
      {error || activity.detail ? <span>{error || activity.detail}</span> : null}
      {sourceCount > 0 ? (
        <details className="run-sources">
          <summary>{sourceLabel}{latestSource ? " · " + latestSource : ""}</summary>
          <ul>
            {activity.sources.map(({ id, title, url }, index) => (
              <li key={id}>
                {url ? (
                  <a href={url} target="_blank" rel="noreferrer noopener">
                    {title || url}
                  </a>
                ) : (title || "Source " + (index + 1))}
              </li>
            ))}
          </ul>
        </details>
      ) : null}
    </div>
  );
}

function safeSourceUrl(raw: string): string {
  try {
    const url = new URL(raw);
    if (url.protocol !== "http:" && url.protocol !== "https:") return "";
    url.username = "";
    url.password = "";
    url.search = "";
    url.hash = "";
    return url.toString();
  } catch {
    return "";
  }
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
  const messageId = useAuiState((state) => state.message.id);
  const sources = useContext(SavedSourcesContext).get(messageId) ?? [];
  return (
    <MessagePrimitive.Root className="message message-assistant">
      <div className="message-label">Audrey</div>
      <MessagePrimitive.Parts
        components={{ Text: MarkdownText, tools: { Fallback: HiddenToolActivity } }}
      />
      {sources.length > 0 ? (
        <details className="saved-sources">
          <summary>{sources.length === 1 ? "1 source found" : `${sources.length} sources found`}</summary>
          <p>Observed during this run; the answer may cite a different set.</p>
          <ul>
            {sources.map(({ id, title, url }) => {
              const safeUrl = safeSourceUrl(url);
              return <li key={id}>{safeUrl ? (
                <a href={safeUrl} target="_blank" rel="noreferrer noopener">{title || safeUrl}</a>
              ) : (title || "Source")}</li>;
            })}
          </ul>
        </details>
      ) : null}
    </MessagePrimitive.Root>
  );
}

function ComposerModelPicker({
  compact = false,
  models,
  canBrowseDirectModels,
  modelId,
  disabled,
  onChange,
}: {
  compact?: boolean;
  models: AudreyModel[];
  canBrowseDirectModels: boolean;
  modelId: string;
  disabled: boolean;
  onChange: (modelId: string) => Promise<void>;
}) {
  const selected = modelDetails(models, modelId);
  const [directMenuOpen, setDirectMenuOpen] = useState(false);
  const anchorRef = useRef<HTMLDivElement>(null);
  const selectRef = useRef<HTMLSelectElement>(null);
  const firstDirectRef = useRef<HTMLButtonElement>(null);
  const workflowModels = models.filter(({ kind }) => kind === "workflow");
  const directModels = canBrowseDirectModels
    ? models.filter(({ kind }) => kind === "direct")
    : [];
  const directMenuVisible = directMenuOpen && !disabled && directModels.length > 0;

  useEffect(() => {
    if (!directMenuVisible) return;
    firstDirectRef.current?.focus();
    const closeOnOutsideClick = (event: PointerEvent) => {
      if (!anchorRef.current?.contains(event.target as Node)) {
        setDirectMenuOpen(false);
      }
    };
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key !== "Escape") return;
      event.stopPropagation();
      setDirectMenuOpen(false);
      selectRef.current?.focus();
    };
    document.addEventListener("pointerdown", closeOnOutsideClick);
    document.addEventListener("keydown", closeOnEscape);
    return () => {
      document.removeEventListener("pointerdown", closeOnOutsideClick);
      document.removeEventListener("keydown", closeOnEscape);
    };
  }, [directMenuVisible]);

  const select = (
    <select
      ref={selectRef}
      aria-label="Audrey model"
      aria-expanded={directMenuVisible}
      title={`${selected.label}: ${selected.description}`}
      value={modelId}
      disabled={disabled}
      onChange={(event) => {
        if (event.target.value === "__other_models__") {
          setDirectMenuOpen(true);
          return;
        }
        setDirectMenuOpen(false);
        void onChange(event.target.value);
      }}
    >
      <optgroup label="Audrey">
        {workflowModels.map((item) => (
          <option key={item.id} value={item.id}>{item.label}</option>
        ))}
      </optgroup>
      {selected.kind === "direct" && canBrowseDirectModels ? (
        <optgroup label="Selected direct model">
          <option value={selected.id}>{selected.id.slice("direct/".length)}</option>
        </optgroup>
      ) : null}
      {directModels.length > 0 ? (
        <option value="__other_models__">Other models...</option>
      ) : null}
    </select>
  );

  const control = (
    <div className="model-picker-menu-anchor" ref={anchorRef}>
      <label className={compact ? "compact-model-picker" : "model-picker-control"}>
        {select}
      </label>
      {directMenuVisible ? (
        <div className="direct-model-menu" role="menu" aria-label="Other models">
          <header>
            <strong>Other models</strong>
            <button
              className="direct-model-menu-close"
              type="button"
              aria-label="Close other models"
              onClick={() => {
                setDirectMenuOpen(false);
                selectRef.current?.focus();
              }}
            >
              ×
            </button>
          </header>
          <div className="direct-model-options">
            {directModels.map((item, index) => (
              <button
                key={item.id}
                ref={index === 0 ? firstDirectRef : undefined}
                type="button"
                role="menuitem"
                onClick={() => {
                  setDirectMenuOpen(false);
                  void onChange(item.id);
                }}
              >
                {item.id.slice("direct/".length)}
              </button>
            ))}
          </div>
        </div>
      ) : null}
    </div>
  );
  if (compact) {
    return (
      <div className="compact-model-picker-shell">
        {control}
      </div>
    );
  }
  return (
    <div className="composer-model-picker">
      <img src={selected.portrait} alt="" aria-hidden="true" />
      {control}
      <span className="model-description" aria-live="polite">
        {selected.description}
      </span>
    </div>
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

function HiddenToolActivity() {
  return null;
}

function toThreadMessages(messages: ConversationMessage[]): ThreadMessageLike[] {
  return messages.flatMap<ThreadMessageLike>((message) => {
    const content = messageWithAttachments(message);
    if (message.role === "user") {
      return [{ id: message.id, role: "user", content }];
    }
    if (message.role === "assistant") {
      return [{
        id: message.id,
        role: "assistant",
        content: message.content ? [{ type: "text" as const, text: message.content }] : [],
      }];
    }
    return [];
  });
}

function messageWithAttachments(message: ConversationMessage): string {
  if (message.role !== "user" || !message.attachments?.length) return message.content;
  const files = message.attachments
    .map(({ filename }) => `📎 ${escapeMarkdown(filename)}`)
    .join("\n");
  return `${message.content}\n\n${files}`;
}

function escapeMarkdown(value: string): string {
  return value
    .replaceAll("\r", " ")
    .replaceAll("\n", " ")
    .replace(/([\\`*_{}[\]()<>#+\-.!|])/g, "\\$1");
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  const units = ["KB", "MB", "GB"];
  let value = bytes / 1024;
  let unit = units[0];
  for (const next of units.slice(1)) {
    if (value < 1024) break;
    value /= 1024;
    unit = next;
  }
  return `${value < 10 ? value.toFixed(1) : Math.round(value)} ${unit}`;
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

function modelLabel(models: AudreyModel[], conversation: Conversation): string {
  return models.find(({ id }) => id === conversation.default_model_id)?.label
    ?? models[0]?.label
    ?? conversation.default_mode;
}

function modelDetails(models: AudreyModel[], modelId: string) {
  const model = models.find(({ id }) => id === modelId) ?? models[0];
  return {
    ...model,
    portrait: model.portrait_url || MODEL_PORTRAITS[model.presentation] || autoPortrait,
  };
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
