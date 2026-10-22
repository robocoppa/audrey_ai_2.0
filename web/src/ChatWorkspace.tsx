import { HttpAgent, type AgentSubscriber } from "@ag-ui/client";
import {
  AssistantRuntimeProvider,
  ComposerPrimitive,
  ExportedMessageRepository,
  MessagePrimitive,
  ThreadPrimitive,
  type CompleteAttachment,
  type CreateAttachment,
  type ThreadHistoryAdapter,
  type ThreadMessageLike,
  type TextMessagePartProps,
  useAui,
  useAuiState,
} from "@assistant-ui/react";
import { useAgUiRuntime } from "@assistant-ui/react-ag-ui";
import {
  createContext,
  isValidElement,
  type ReactNode,
  type RefObject,
  type SetStateAction,
  useCallback,
  useContext,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { createPortal } from "react-dom";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";

import autoPortrait from "./assets/models/audrey2.png";
import deepPortrait from "./assets/models/audrey7.png";
import fastPortrait from "./assets/models/audrey3.png";
import researchPortrait from "./assets/models/audrey8.png";
import videoPortrait from "./assets/models/audrey9.png";
import cloudPortrait from "./assets/models/cloudModel.png";
import localPortrait from "./assets/models/localModel.png";
import privatePortrait from "./assets/models/audrey5.png";

import { AudreyLoader } from "./AudreyLoader";
import { ScrollToLatest } from "./ScrollToLatest";
import { SwipeConversationRow } from "./SwipeConversationRow";
import { UploadProgress } from "./UploadProgress";
import { TaxReview } from "./TaxReview";
import { PrivateTaskProgress } from "./PrivateTaskProgress";
import type { PrivateTask } from "./privateTasksApi";
import { PrivateModeNotice } from "./PrivateModeNotice";
import { TaxReviewQuestions } from "./TaxReviewQuestions";
import { TAX_REVIEW_PROMPT, type TaxCase } from "./taxReviewApi";
import {
  ApiError,
  cancelRun,
  createConversation,
  deleteConversation,
  discardEmptyConversation,
  getConversation,
  getRun,
  getFileImageUrl,
  listConversations,
  listFiles,
  getFile,
  listMessages,
  listProjectConversations,
  listProjectFiles,
  listProjects,
  PRIVATE_DOCUMENT_EXTENSIONS,
  uploadFile,
  uploadPrecheck,
  updateConversation,
  updateConversationModel,
  type AudreyFile,
  type AudreyFileLimits,
  type AudreyModel,
  type AudreyProject,
  type AudreyProjectFile,
  type Conversation,
  type ConversationMessage,
  type MessageAttachment,
  type MessageModelUsage,
  type MessageSource,
  type ProjectLimits,
  type SkillSummary,
  type CurrentUser,
  type UserPreferences,
} from "./api";
import { latestActionFetch } from "./agentTransport";
import { upsertConversation } from "./conversationHistory";
import { NewProjectDialog, ProjectHome } from "./ProjectWorkspace";

const PRIVATE_MODEL: AudreyModel = {
  id: "private", label: "Private", description: "Local models and tools for private conversations.",
  kind: "workflow", mode: "local", presentation: "local", capabilities: ["text", "files", "tools"],
  enabled: true, audience: "users", visibility: "public", roles: ["users"], portrait_url: "",
};

const LEGACY_LOCAL_MODEL: AudreyModel = {
  ...PRIVATE_MODEL, id: "local", label: "Local", description: "Local model workflow.",
};

const MODEL_PORTRAITS: Readonly<Record<string, string>> = {
  auto: autoPortrait,
  fast: fastPortrait,
  deep: deepPortrait,
  research: researchPortrait,
  video: videoPortrait,
  cloud: cloudPortrait,
  local: localPortrait,
};


function ProjectIcon({ className }: { className?: string }) {
  return (
    <svg
      className={className}
      viewBox="0 0 24 24"
      width="17"
      height="17"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      focusable="false"
    >
      <path d="m12 3 10 5-10 5L2 8Z" />
      <path d="m2 12 10 5 10-5" />
      <path d="m2 16 10 5 10-5" />
    </svg>
  );
}


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
  models: MessageModelUsage[];
  tools: RunTool[];
};

type RunSource = {
  id: string;
  title: string;
  url: string;
};

type RunTool = {
  id: string;
  name: string;
  status: "running" | "succeeded" | "failed" | "incomplete";
};

type AnswerObservations = Pick<RunActivity, "sources" | "models" | "tools">;

type ActivityLedger = {
  activity: RunActivity;
  answerMessageId: string | null;
  answers: ReadonlyMap<string, AnswerObservations>;
};

type LastAttempt = {
  text: string;
  attachmentIds: string[];
};

type ConversationView = "active" | "archived";
type AttachmentKindFilter = "all" | AudreyFile["kind"];

const ATTACHMENT_FOLDERS: ReadonlyArray<{
  kind: AttachmentKindFilter;
  label: string;
  symbol: string;
}> = [
  { kind: "all", label: "All", symbol: "▦" },
  { kind: "text", label: "Documents", symbol: "≡" },
  { kind: "image", label: "Images", symbol: "◫" },
  { kind: "audio", label: "Audio", symbol: "♪" },
  { kind: "video", label: "Videos", symbol: "▶" },
];

const SavedSourcesContext = createContext<ReadonlyMap<string, MessageSource[]>>(new Map());
const SavedModelsContext = createContext<ReadonlyMap<string, MessageModelUsage[]>>(new Map());
const SavedToolsContext = createContext<ReadonlyMap<string, RunTool[]>>(new Map());

type ComposerDraft = { text: string; notice: string };

const IDLE_ACTIVITY: RunActivity = {
  status: "idle",
  label: "Ready",
  detail: "",
  sources: [],
  models: [],
  tools: [],
};

const DEFAULT_PROJECT_LIMITS: ProjectLimits = {
  max_name_chars: 100,
  max_instructions_chars: 4_000,
  max_files: 20,
};

export function ChatWorkspace({
  user,
  preferences,
  models,
  skills,
  navigationOpen,
  onNavigationOpenChange: setNavigationOpen,
  mobileMenuHost,
  mobileNewChatHost,
  menuActions,
  menuFooter,
}: {
  navigationOpen: boolean;
  onNavigationOpenChange: (open: boolean) => void;
  mobileMenuHost: HTMLElement | null;
  mobileNewChatHost: HTMLElement | null;
  menuActions: ReactNode;
  menuFooter: ReactNode;
  user: CurrentUser;
  preferences: UserPreferences;
  models: AudreyModel[];
  skills: SkillSummary[];
}) {
  const [compactNavigation, setCompactNavigation] = useState(() =>
    window.matchMedia?.("(max-width: 1100px)").matches ?? false);
  const [projectsExpanded, setProjectsExpanded] = useState(false);
  const navigationRef = useRef<HTMLElement>(null);
  const navigationToggleRef = useRef<HTMLButtonElement>(null);
  const lastFocusedRef = useRef<HTMLElement | null>(null);

  const closeNavigation = useCallback(() => {
    setNavigationOpen(false);
    if (compactNavigation && navigationOpen) {
      window.requestAnimationFrame(() => navigationToggleRef.current?.focus({ preventScroll: true }));
    }
  }, [compactNavigation, navigationOpen, setNavigationOpen]);

  useEffect(() => {
    const viewport = window.matchMedia?.("(max-width: 1100px)");
    if (!viewport) return;
    const rememberFocus = (event: FocusEvent) => {
      if (event.target instanceof HTMLElement && event.target !== document.body) {
        lastFocusedRef.current = event.target;
      }
    };
    const updateViewport = (event: MediaQueryListEvent) => {
      // A CSS breakpoint can hide a control before this event fires, moving
      // activeElement to body. Retain the last focused control for restoration.
      const focused = document.activeElement === document.body
        ? lastFocusedRef.current
        : document.activeElement;
      const focusInHistory = focused instanceof HTMLElement && navigationRef.current?.contains(focused);
      const focusInCompactControls = focused instanceof HTMLElement
        && focused.closest(".topbar-mobile-menu, .topbar-mobile-new-chat, .conversation-drawer-header, .session-controls-drawer, .mobile-menu-footer, .sidebar-menu-footer, .projects-category-toggle");
      const focusInDesktopActions = focused instanceof HTMLElement
        && focused.closest(".session-controls:not(.session-controls-drawer)");
      setCompactNavigation(event.matches);
      setNavigationOpen(false);
      if (event.matches && (focusInHistory || focusInDesktopActions)) {
        window.requestAnimationFrame(() => navigationToggleRef.current?.focus({ preventScroll: true }));
      } else if (!event.matches && focusInCompactControls) {
        window.requestAnimationFrame(() => navigationRef.current?.querySelector<HTMLButtonElement>(".new-conversation")?.focus({ preventScroll: true }));
      }
    };
    document.addEventListener("focusin", rememberFocus);
    viewport.addEventListener("change", updateViewport);
    return () => {
      document.removeEventListener("focusin", rememberFocus);
      viewport.removeEventListener("change", updateViewport);
    };
  }, [setNavigationOpen]);

  useEffect(() => {
    if (!compactNavigation || !navigationOpen) return;
    const navigation = navigationRef.current;
    navigation?.querySelector<HTMLButtonElement>(".conversation-drawer-close")?.focus({ preventScroll: true });
    const handleKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault();
        closeNavigation();
      } else if (event.key === "Tab" && navigation) {
        const actions = Array.from(navigation.querySelectorAll<HTMLElement>(
          'button:not([disabled]), a[href], input:not([disabled]), select:not([disabled]), [tabindex="0"]',
        )).filter((action) => action.getClientRects().length > 0);
        const first = actions[0];
        const last = actions[actions.length - 1];
        if (event.shiftKey && document.activeElement === first) {
          event.preventDefault();
          last?.focus();
        } else if (!event.shiftKey && document.activeElement === last) {
          event.preventDefault();
          first?.focus();
        }
      }
    };
    document.addEventListener("keydown", handleKey);
    return () => document.removeEventListener("keydown", handleKey);
  }, [closeNavigation, compactNavigation, navigationOpen]);

  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [openedConversations, setOpenedConversations] = useState<Conversation[]>([]);
  const [replacedDrafts, setReplacedDrafts] = useState<Record<string, ComposerDraft>>({});
  const navigationRevisionRef = useRef(0);
  const [projects, setProjects] = useState<AudreyProject[]>([]);
  const [projectLimits, setProjectLimits] = useState<ProjectLimits>(DEFAULT_PROJECT_LIMITS);
  const [projectsLoading, setProjectsLoading] = useState(true);
  const [projectsLoadingMore, setProjectsLoadingMore] = useState(false);
  const [projectsNextCursor, setProjectsNextCursor] = useState<string | null>(null);
  const [projectsError, setProjectsError] = useState("");
  const [selectedProjectId, setSelectedProjectId] = useState<string | null>(null);
  const [expandedProjectId, setExpandedProjectId] = useState<string | null>(null);
  const [projectConversations, setProjectConversations] = useState<Conversation[]>([]);
  const [projectConversationsLoading, setProjectConversationsLoading] = useState(false);
  const [projectConversationsNextCursor, setProjectConversationsNextCursor] = useState<string | null>(null);
  const [projectFiles, setProjectFiles] = useState<AudreyProjectFile[]>([]);
  const [projectFilesLoading, setProjectFilesLoading] = useState(false);
  const [projectError, setProjectError] = useState("");
  const [newProjectOpen, setNewProjectOpen] = useState(false);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [composerFocusId, setComposerFocusId] = useState<string | null>(null);
  const composerFocused = useCallback((conversationId: string) => {
    setComposerFocusId((current) => current === conversationId ? null : current);
  }, []);
  const [loading, setLoading] = useState(true);
  const [creating, setCreating] = useState(false);
  const [loadingMore, setLoadingMore] = useState(false);
  const [nextCursor, setNextCursor] = useState<string | null>(null);
  const [view, setView] = useState<ConversationView>("active");
  const [searchInput, setSearchInput] = useState("");
  const [searchQuery, setSearchQuery] = useState("");
  const [error, setError] = useState("");
  const [confirmDeleteId, setConfirmDeleteId] = useState<string | null>(null);
  const [archivingId, setArchivingId] = useState<string | null>(null);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const listKeyRef = useRef("");
  const selectedIdRef = useRef<string | null>(null);
  const selectedProjectIdRef = useRef<string | null>(null);
  const defaultModelId = models.find(({ id }) => id === "auto")?.id ?? models[0]?.id ?? null;
  const catalogUnavailable = defaultModelId === null;
  const canBrowseDirectModels = models.some((model) =>
    model.kind === "direct"
    && (user.role === "admin"
      || user.groups.includes("admins")
      || model.roles?.some((role) => user.groups.includes(role))),
  );

  const removeFromCurrentViewRef = useRef(removeFromCurrentView);
  useEffect(() => {
    removeFromCurrentViewRef.current = removeFromCurrentView;
  });

  function selectConversation(conversation: Conversation | null) {
    navigationRevisionRef.current += 1;
    setComposerFocusId((current) => current === conversation?.id ? current : null);
    const previousId = selectedIdRef.current;
    if (previousId && previousId !== conversation?.id) {
      const previous = openedConversations.find(({ id }) => id === previousId)
        ?? conversations.find(({ id }) => id === previousId);
      if (previous?.last_message_at === null) {
        void discardEmptyConversation(previousId)
          .then(() => {
            setConversations((current) => current.filter(({ id }) => id !== previousId));
            setOpenedConversations((current) => current.filter(({ id }) => id !== previousId));
            setProjectConversations((current) => current.filter(({ id }) => id !== previousId));
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
    let active = true;
    setProjectsLoading(true);
    setProjectsError("");
    listProjects()
      .then(({ items, next_cursor, limits }) => {
        if (!active) return;
        setProjects(items);
        setProjectsNextCursor(next_cursor);
        if (limits) setProjectLimits(limits);
      })
      .catch((reason: unknown) => {
        if (active) setProjectsError(actionError("Could not load projects", reason));
      })
      .finally(() => {
        if (active) setProjectsLoading(false);
      });
    return () => { active = false; };
  }, []);

  useEffect(() => {
    if (!selectedProjectId) {
      setProjectConversations([]);
      setProjectConversationsNextCursor(null);
      setProjectFiles([]);
      setProjectError("");
      return;
    }
    let active = true;
    setProjectConversations([]);
    setProjectConversationsNextCursor(null);
    setProjectFiles([]);
    setProjectConversationsLoading(true);
    setProjectFilesLoading(true);
    setProjectError("");
    Promise.all([
      listProjectConversations(selectedProjectId, {
        archived: view === "archived",
        search: searchQuery,
      }),
      listProjectFiles(selectedProjectId),
    ])
      .then(([conversationPage, filePage]) => {
        if (!active) return;
        setProjectConversations(conversationPage.items);
        setProjectConversationsNextCursor(conversationPage.next_cursor);
        setProjectFiles(filePage.items);
        if (filePage.limits) setProjectLimits(filePage.limits);
      })
      .catch((reason: unknown) => {
        if (active) setProjectError(actionError("Could not open project", reason));
      })
      .finally(() => {
        if (!active) return;
        setProjectConversationsLoading(false);
        setProjectFilesLoading(false);
      });
    return () => { active = false; };
  }, [searchQuery, selectedProjectId, view]);

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
        const ordinaryItems = items.filter(({ project_id }) => !project_id);
        if (
          view === "active"
          && !searchQuery
          && items.length === 0
          && selectedProjectIdRef.current === null
        ) {
          const conversation = await createConversation(defaultModelId);
          if (!active) return;
          setConversations([conversation, ...items]);
          setNextCursor(next_cursor);
          selectConversation(conversation);
          return;
        }
        if (!active) return;
        setConversations(items);
        setNextCursor(next_cursor);
        if (selectedProjectIdRef.current !== null && selectedIdRef.current === null) return;
        const nextSelected = items.find(({ id }) => id === selectedIdRef.current)
          ?? ordinaryItems[0]
          ?? items[0]
          ?? null;
        if (nextSelected) {
          openConversation(nextSelected, { dismissNavigation: false });
        } else {
          selectConversation(null);
        }
      })
      .catch((reason: unknown) => {
        if (active) setError(actionError("Could not load conversation history", reason));
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, [defaultModelId, searchQuery, view]);

  const selected = openedConversations.find(({ id }) => id === selectedId)
    ?? conversations.find(({ id }) => id === selectedId)
    ?? null;
  const selectedProject = projects.find(({ id }) => id === selectedProjectId) ?? null;
  const historyConversations = conversations.filter(({ last_message_at, project_id }) =>
    last_message_at !== null && (!project_id || Boolean(searchQuery)),
  );
  const projectHistoryConversations = projectConversations.filter(
    ({ last_message_at }) => last_message_at !== null,
  );
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

  function clearProjectSelection() {
    selectedProjectIdRef.current = null;
    setSelectedProjectId(null);
    setExpandedProjectId(null);
  }

  function openProject(project: AudreyProject) {
    closeNavigation();
    selectedProjectIdRef.current = project.id;
    setSelectedProjectId(project.id);
    setExpandedProjectId(project.id);
    selectConversation(null);
  }

  function openConversation(conversation: Conversation, { dismissNavigation = true } = {}) {
    if (dismissNavigation && navigationOpen) closeNavigation();
    const projectId = conversation.project_id ?? null;
    selectedProjectIdRef.current = projectId;
    setSelectedProjectId(projectId);
    setExpandedProjectId(projectId);
    selectConversation(conversation);
  }

  async function startConversation({ focusComposer = false, preserveNavigation = false } = {}) {
    if (!defaultModelId) return;
    const expectedNavigation = navigationRevisionRef.current;
    setCreating(true);
    setError("");
    try {
      const conversation = await createConversation(defaultModelId);
      if (preserveNavigation && navigationRevisionRef.current !== expectedNavigation) {
        void discardEmptyConversation(conversation.id).catch(() => {});
        return;
      }
      clearProjectSelection();
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
      if (focusComposer) setComposerFocusId(conversation.id);
      closeNavigation();
    } catch (reason) {
      if (!preserveNavigation || navigationRevisionRef.current === expectedNavigation) {
        setError(actionError("Could not open a new conversation", reason));
      }
    } finally {
      setCreating(false);
    }
  }

  function replaceConversation(updated: Conversation) {
    if (selectedIdRef.current === updated.id) {
      const projectId = updated.project_id ?? null;
      selectedProjectIdRef.current = projectId;
      setSelectedProjectId(projectId);
      setExpandedProjectId(projectId);
    }
    setOpenedConversations((current) => upsertConversation(current, updated));
    setProjectConversations((current) =>
      updated.project_id === selectedProjectId
        ? upsertConversation(current, updated)
        : current.filter(({ id }) => id !== updated.id),
    );
    if (
      searchQuery
      && !updated.title.toLocaleLowerCase().includes(searchQuery.toLocaleLowerCase())
    ) {
      removeFromCurrentView(updated.id);
      return;
    }
    setConversations((current) => upsertConversation(current, updated));
  }

  function replaceDraftConversation(previousId: string, updated: Conversation, draft: ComposerDraft) {
    setReplacedDrafts((current) => ({ ...current, [updated.id]: draft }));
    setConversations((current) => upsertConversation(current.filter(({ id }) => id !== previousId), updated));
    setOpenedConversations((current) => upsertConversation(current.filter(({ id }) => id !== previousId), updated));
    setProjectConversations((current) => current.filter(({ id }) => id !== previousId));
    if (selectedIdRef.current === previousId) {
      // The server replaced the empty draft. Do not read/discard its deleted ID.
      selectedIdRef.current = null;
      openConversation(updated);
      setComposerFocusId(updated.id);
    }
  }

  function removeFromCurrentView(conversationId: string, closeThread = false, openNewDraft = false) {
    const remaining = conversations.filter(({ id }) => id !== conversationId);
    setConversations((current) => current.filter(({ id }) => id !== conversationId));
    setProjectConversations((current) => current.filter(({ id }) => id !== conversationId));
    if (closeThread) {
      setOpenedConversations((current) =>
        current.filter(({ id }) => id !== conversationId),
      );
    }
    if (selectedIdRef.current === conversationId) {
      if (openNewDraft) {
        // Remove the deleted thread before creating its replacement. A later
        // navigation must win if the create request is still in flight.
        selectedIdRef.current = null;
        clearProjectSelection();
        selectConversation(null);
        void startConversation({ focusComposer: true, preserveNavigation: true });
      } else if (selectedProjectIdRef.current) {
        selectConversation(null);
      } else {
        const nextOrdinary = remaining.find(({ project_id }) => !project_id) ?? null;
        selectConversation(nextOrdinary);
        if (!nextOrdinary && view === "active" && !searchQuery) {
          void startConversation();
        }
      }
    }
  }

  function closeNewProject() {
    setNewProjectOpen(false);
    if (compactNavigation) {
      window.requestAnimationFrame(() => navigationToggleRef.current?.focus({ preventScroll: true }));
    }
  }

  function projectCreated(project: AudreyProject) {
    setProjects((current) => [project, ...current]);
    closeNewProject();
    openProject(project);
  }

  function projectChanged(updated: AudreyProject) {
    setProjects((current) => current.map((project) =>
      project.id === updated.id ? updated : project,
    ));
  }

  function projectDeleted(projectId: string) {
    const retained = projectConversations.map((conversation) => ({
      ...conversation,
      project_id: null,
    }));
    setProjects((current) => current.filter(({ id }) => id !== projectId));
    setConversations((current) => current.map((conversation) =>
      conversation.project_id === projectId
        ? { ...conversation, project_id: null }
        : conversation,
    ));
    setOpenedConversations((current) => current.map((conversation) =>
      conversation.project_id === projectId
        ? { ...conversation, project_id: null }
        : conversation,
    ));
    setProjectConversations([]);
    setProjectFiles([]);
    clearProjectSelection();
    const nextConversation = retained[0]
      ?? conversations.find(({ project_id }) => !project_id)
      ?? null;
    selectConversation(nextConversation);
    if (!nextConversation && view === "active" && !searchQuery) {
      void startConversation();
    }
  }

  function projectConversationCreated(conversation: Conversation) {
    setConversations((current) => upsertConversation(current, conversation));
    setProjectConversations((current) => upsertConversation(current, conversation));
    openConversation(conversation);
    setComposerFocusId(conversation.id);
  }

  async function archiveFromSidebar(conversation: Conversation) {
    setArchivingId(conversation.id);
    setError("");
    try {
      await updateConversation(conversation.id, { archived: conversation.archived_at === null });
      removeFromCurrentViewRef.current(conversation.id, true);
    } catch (reason) {
      setError(messageOf(reason));
    } finally {
      setArchivingId(null);
    }
  }

  async function deleteFromSidebar(conversationId: string) {
    setDeletingId(conversationId);
    setError("");
    try {
      await deleteConversation(conversationId);
      setConfirmDeleteId(null);
      removeFromCurrentViewRef.current(conversationId, true, true);
    } catch (reason) {
      setError(actionError("Could not delete conversation", reason));
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
        setError(actionError("Could not load older conversations", reason));
      }
    } finally {
      setLoadingMore(false);
    }
  }

  async function loadMoreProjects() {
    if (!projectsNextCursor || projectsLoadingMore) return;
    setProjectsLoadingMore(true);
    setProjectsError("");
    try {
      const page = await listProjects(projectsNextCursor);
      setProjects((current) => [...current, ...page.items]);
      setProjectsNextCursor(page.next_cursor);
      if (page.limits) setProjectLimits(page.limits);
    } catch (reason) {
      setProjectsError(messageOf(reason));
    } finally {
      setProjectsLoadingMore(false);
    }
  }

  async function loadMoreProjectConversations() {
    if (!selectedProjectId || !projectConversationsNextCursor) return;
    const expectedProjectId = selectedProjectId;
    setProjectConversationsLoading(true);
    setProjectError("");
    try {
      const page = await listProjectConversations(expectedProjectId, {
        archived: view === "archived",
        cursor: projectConversationsNextCursor,
        search: searchQuery,
      });
      if (selectedProjectIdRef.current !== expectedProjectId) return;
      setProjectConversations((current) => [...current, ...page.items]);
      setProjectConversationsNextCursor(page.next_cursor);
    } catch (reason) {
      if (selectedProjectIdRef.current === expectedProjectId) {
        setProjectError(messageOf(reason));
      }
    } finally {
      if (selectedProjectIdRef.current === expectedProjectId) {
        setProjectConversationsLoading(false);
      }
    }
  }

  const projectNavigation = (
    <section className="projects-sidebar" aria-labelledby="projects-sidebar-title">
      <header>
        <h2 id="projects-sidebar-title">
          {compactNavigation ? (
            <button
              className="projects-category-toggle menu-navigation-action"
              type="button"
              aria-expanded={projectsExpanded}
              aria-controls="sidebar-projects"
              onClick={() => setProjectsExpanded((expanded) => !expanded)}
            >
              <ProjectIcon />
              <span>My projects</span>
              <span aria-hidden="true">{projectsExpanded ? "⌄" : "›"}</span>
            </button>
          ) : "Projects"}
        </h2>
        <button className="project-create-button" type="button" onClick={() => {
          setNavigationOpen(false);
          setNewProjectOpen(true);
        }} aria-label="New project" title="Add project">
          <span aria-hidden="true">＋</span>
          {compactNavigation ? null : <span>New</span>}
        </button>
      </header>
      <div id="sidebar-projects" hidden={compactNavigation && !projectsExpanded}>
        {projectsLoading ? <p className="sidebar-status" role="status">Loading projects…</p> : null}
        {!projectsLoading && projects.length === 0 ? (
          <p className="sidebar-status">No projects yet.</p>
        ) : null}
        {projects.length ? (
          <ul className="project-sidebar-list">
            {projects.map((project) => {
              const expanded = expandedProjectId === project.id;
              const current = selectedProjectId === project.id;
              return (
                <li key={project.id} className={current ? "project-sidebar-row current" : "project-sidebar-row"}>
                  <div className="project-sidebar-main">
                    <button
                      className="project-sidebar-link"
                      type="button"
                      onClick={() => openProject(project)}
                      aria-current={current && selectedId === null ? "page" : undefined}
                      title={project.name}
                    >
                      <ProjectIcon className="project-icon" />
                      <span>{project.name}</span>
                    </button>
                    <button
                      className="project-sidebar-toggle"
                      type="button"
                      onClick={() => {
                        if (expanded) {
                          setExpandedProjectId(null);
                        } else {
                          openProject(project);
                        }
                      }}
                      aria-label={`${expanded ? "Collapse" : "Expand"} project ${project.name}`}
                      aria-expanded={expanded}
                    >
                      <span aria-hidden="true">{expanded ? "⌄" : "›"}</span>
                    </button>
                  </div>
                  {expanded ? (
                    <nav className="project-conversation-list" aria-label={`${project.name} conversations`}>
                      {projectConversationsLoading ? (
                        <p className="sidebar-status" role="status">Loading conversations…</p>
                      ) : null}
                      {!projectConversationsLoading && projectHistoryConversations.length === 0 ? (
                        <p className="sidebar-status">
                          {searchQuery ? "No matching titles." : view === "archived" ? "No archived conversations." : "No conversations yet."}
                        </p>
                      ) : null}
                      {projectHistoryConversations.map((conversation) => (
                        <SwipeConversationRow
                          key={`${conversation.id}-${compactNavigation && navigationOpen}`}
                          title={conversation.title || "New conversation"}
                          enabled={compactNavigation && navigationOpen}
                          disabled={deletingId !== null || archivingId !== null}
                          archiveLabel={conversation.archived_at ? "Restore" : "Archive"}
                          onArchive={() => void archiveFromSidebar(conversation)}
                          onDelete={() => void deleteFromSidebar(conversation.id)}
                        >
                          <button
                            className={conversation.id === selectedId ? "project-conversation active" : "project-conversation"}
                            data-private={conversation.private_required ? "true" : undefined}
                            type="button"
                            onClick={() => openConversation(conversation)}
                            aria-current={conversation.id === selectedId ? "page" : undefined}
                            title={conversation.title || "New conversation"}
                          >
                            <span>{conversation.title || "New conversation"}</span>
                            <small>{modelLabel(models, conversation)}</small>
                          </button>
                        </SwipeConversationRow>
                      ))}
                      {projectConversationsNextCursor ? (
                        <button
                          className="project-load-more"
                          type="button"
                          onClick={() => void loadMoreProjectConversations()}
                          disabled={projectConversationsLoading}
                        >
                          {projectConversationsLoading ? "Loading…" : "Load older"}
                        </button>
                      ) : null}
                    </nav>
                  ) : null}
                </li>
              );
            })}
          </ul>
        ) : null}
        {projectsNextCursor ? (
          <button className="project-load-more" type="button" onClick={() => void loadMoreProjects()} disabled={projectsLoadingMore}>
            {projectsLoadingMore ? "Loading…" : "Load more projects"}
          </button>
        ) : null}
        {projectsError ? <p className="sidebar-error">{projectsError}</p> : null}
      </div>
    </section>
  );

  return (
    <div className="workspace" data-navigation-open={navigationOpen}>
      {mobileMenuHost ? createPortal(
        <button
          id="app-mobile-menu-toggle"
          className="workspace-header-button"
          type="button"
          ref={navigationToggleRef}
          aria-label="Menu"
          aria-expanded={navigationOpen}
          aria-controls="conversation-navigation"
          onClick={() => setNavigationOpen(true)}
        >
          <svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" aria-hidden="true">
            <path d="M4 5h16M4 12h16M4 19h16" />
          </svg>
        </button>, mobileMenuHost,
      ) : null}
      {mobileNewChatHost ? createPortal(
        <button
          className="workspace-header-button"
          type="button"
          aria-label="New chat"
          onClick={() => void startConversation({ focusComposer: true })}
          disabled={creating || loading || catalogUnavailable}
        >
          <svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" aria-hidden="true">
            <path d="M12 5v14M5 12h14" />
          </svg>
        </button>, mobileNewChatHost,
      ) : null}
      {compactNavigation && !navigationOpen && error ? (
        <p className="workspace-mobile-error" role="alert">{error}</p>
      ) : null}
      {compactNavigation && navigationOpen ? (
        <button className="conversation-drawer-backdrop" type="button" tabIndex={-1} aria-label="Dismiss conversation history" onClick={closeNavigation} />
      ) : null}
      {newProjectOpen ? (
        <NewProjectDialog
          limits={projectLimits}
          privateAvailable={models.some(({ id }) => id === "private")}
          onClose={closeNewProject}
          onCreated={projectCreated}
        />
      ) : null}
      <aside
        className="sidebar"
        id="conversation-navigation"
        ref={navigationRef}
        aria-label="Conversations"
        role={compactNavigation ? "dialog" : undefined}
        aria-modal={compactNavigation && navigationOpen ? true : undefined}
        aria-hidden={compactNavigation && !navigationOpen ? true : undefined}
        inert={compactNavigation && !navigationOpen ? true : undefined}
      >
        <div className="conversation-drawer-header">
          <strong className="drawer-user-name" title={user.display_name || user.email}>
            {user.display_name.trim() || user.email.split("@", 1)[0]} - My chats
          </strong>
          <button className="conversation-drawer-close" type="button" aria-label="Close conversation history" onClick={closeNavigation}>
            <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" aria-hidden="true">
              <path d="M6 6l12 12M6 18 18 6" />
            </svg>
          </button>
        </div>
        <div className="sidebar-content">
          {compactNavigation ? menuActions : null}
          <div className="sidebar-primary-action">
            <button
              className="new-conversation"
              type="button"
              onClick={() => void startConversation({ focusComposer: true })}
              disabled={creating || loading || catalogUnavailable}
            >
              <svg viewBox="0 0 24 24" width="17" height="17" fill="none" stroke="currentColor" strokeWidth="1.9" strokeLinecap="round" aria-hidden="true">
                <path d="M12 5v14M5 12h14" />
              </svg>
              <span>{creating ? "Creating…" : "New conversation"}</span>
            </button>
          </div>

          {!compactNavigation ? projectNavigation : null}

          <label className="conversation-search">
            <input
              type="search"
              aria-label="Search titles"
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
              <SwipeConversationRow
                key={`${conversation.id}-${compactNavigation && navigationOpen}`}
                title={conversation.title || "New conversation"}
                enabled={compactNavigation && navigationOpen}
                disabled={deletingId !== null || archivingId !== null}
                archiveLabel={conversation.archived_at ? "Restore" : "Archive"}
                onArchive={() => void archiveFromSidebar(conversation)}
                onDelete={() => void deleteFromSidebar(conversation.id)}
              >
                <div className="conversation-row">
                  <button
                    className={conversation.id === selectedId ? "conversation active" : "conversation"}
                    data-private={conversation.private_required === true || conversation.default_model_id === "private" ? "true" : undefined}
                    type="button"
                    onClick={() => openConversation(conversation)}
                    aria-current={conversation.id === selectedId ? "page" : undefined}
                  >
                    <span>{conversation.title || "New conversation"}</span>
                    <small>{modelLabel(models, conversation)}</small>
                  </button>
                  {!compactNavigation ? <button
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
                  </button> : null}
                </div>
              </SwipeConversationRow>
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
        </div>
        {compactNavigation ? (
          <div className="sidebar-menu-footer">
            {projectNavigation}
            {menuFooter}
          </div>
        ) : null}
      </aside>

      <section className="chat-column" aria-label="Audrey conversation" inert={compactNavigation && navigationOpen ? true : undefined}>
        {catalogUnavailable ? (
          <div className="catalog-unavailable">
            <h2>No models available</h2>
            <p>An Audrey administrator can enable a model from the Admin panel.</p>
          </div>
        ) : null}
        {!catalogUnavailable && selectedProject && !selected ? (
          <ProjectHome
            project={selectedProject}
            conversations={projectConversations}
            files={projectFiles}
            filesLoading={projectFilesLoading}
            limits={projectLimits}
            defaultModelId={defaultModelId ?? "auto"}
            onProjectChange={projectChanged}
            onProjectDeleted={projectDeleted}
            onConversationCreated={projectConversationCreated}
            onConversationSelected={openConversation}
            onFilesChange={setProjectFiles}
          />
        ) : null}
        {!catalogUnavailable && projectError && selectedProject && !selected ? (
          <p className="project-page-error" role="alert">{projectError}</p>
        ) : null}
        {!catalogUnavailable ? renderedConversations.map((opened) => (
          <div
            className="conversation-thread-slot"
            hidden={opened.id !== selectedId}
            key={opened.id}
          >
            <ConversationThread
              conversation={opened}
              isSelected={opened.id === selectedId}
              focusComposer={composerFocusId === opened.id && opened.id === selectedId && !navigationOpen}
              onComposerFocused={composerFocused}
              compactNavigation={compactNavigation}
              models={models}
              projects={projects}
              skills={skills}
              canBrowseDirectModels={canBrowseDirectModels}
              showProgress={preferences.show_progress}
              onConversationChange={replaceConversation}
              replacementDraft={replacedDrafts[opened.id]}
              onDraftConsumed={(conversationId) => setReplacedDrafts((current) => {
                const remaining = { ...current };
                delete remaining[conversationId];
                return remaining;
              })}
              onDraftReplaced={replaceDraftConversation}
              onRemoveFromView={removeFromCurrentView}
              onOpenProject={(projectId) => {
                const project = projects.find(({ id }) => id === projectId);
                if (project) openProject(project);
              }}
            />
          </div>
        )) : null}
        {!catalogUnavailable && !selectedProject && !selected && (loading || creating) ? (
          <AudreyLoader label="Opening conversation" showPortrait={false} />
        ) : null}
        {!catalogUnavailable && selectedProjectId && !selectedProject && projectsLoading ? (
          <AudreyLoader label="Opening project" showPortrait={false} />
        ) : null}
      </section>
    </div>
  );
}

function ConversationThread({
  conversation,
  isSelected,
  focusComposer,
  onComposerFocused,
  compactNavigation,
  models,
  projects,
  skills,
  canBrowseDirectModels,
  showProgress,
  onConversationChange,
  replacementDraft,
  onDraftConsumed,
  onDraftReplaced,
  onRemoveFromView,
  onOpenProject,
}: {
  conversation: Conversation;
  isSelected: boolean;
  focusComposer: boolean;
  onComposerFocused: (conversationId: string) => void;
  compactNavigation: boolean;
  models: AudreyModel[];
  projects: AudreyProject[];
  skills: SkillSummary[];
  canBrowseDirectModels: boolean;
  showProgress: boolean;
  onConversationChange: (conversation: Conversation) => void;
  replacementDraft?: ComposerDraft;
  onDraftConsumed: (conversationId: string) => void;
  onDraftReplaced: (previousId: string, conversation: Conversation, draft: ComposerDraft) => void;
  onRemoveFromView: (conversationId: string, closeThread?: boolean, openNewDraft?: boolean) => void;
  onOpenProject: (projectId: string) => void;
}) {
  const [thread, setThread] = useState<ThreadState>({ status: "idle" });
  const privateRequired = conversation.private_required === true || conversation.default_model_id === "private";
  const [acceptedMessage, setAcceptedMessage] = useState(false);
  const [pendingDraft, setPendingDraft] = useState(replacementDraft);
  const privateLocked = privateRequired && (Boolean(conversation.project_id)
    || acceptedMessage || conversation.last_message_at !== null
    || (thread.status === "ready" && thread.messages.some(({ role }) => role === "user")));
  const availableModels = privateRequired && !models.some(({ id }) => id === "private")
    ? [...models, PRIVATE_MODEL] : models;
  const conversationModels = privateLocked ? [availableModels.find(({ id }) => id === "private") ?? PRIVATE_MODEL]
    : conversation.default_model_id === "local" && !models.some(({ id }) => id === "local")
      ? [...availableModels, LEGACY_LOCAL_MODEL] : availableModels;
  const [modelId, setModelId] = useState(
    conversationModels.some(({ id }) => id === conversation.default_model_id)
      ? conversation.default_model_id
      : (models.find(({ id }) => id === "auto") ?? models[0]).id,
  );
  const [editingTitle, setEditingTitle] = useState(false);
  const [titleDraft, setTitleDraft] = useState(conversation.title);
  const [mutation, setMutation] = useState<"model" | "rename" | "project" | "archive" | "delete" | null>(null);
  const [mutationError, setMutationError] = useState("");
  const [confirmingDelete, setConfirmingDelete] = useState(false);
  const [runActive, setRunActive] = useState(false);
  const [recoveredRunId, setRecoveredRunId] = useState<string | null>(null);
  const [durableReviewPresent, setDurableReviewPresent] = useState(false);
  const [recoveryError, setRecoveryError] = useState("");
  const [stoppingRecovered, setStoppingRecovered] = useState(false);
  const [threadRevision, setThreadRevision] = useState(0);
  const [titleRunId, setTitleRunId] = useState<string | null>(null);
  const titleRefreshVersionRef = useRef(0);
  const onConversationChangeRef = useRef(onConversationChange);
  useEffect(() => { onConversationChangeRef.current = onConversationChange; }, [onConversationChange]);
  const archived = conversation.archived_at !== null;
  const selectedModelId = privateRequired ? "private" : conversationModels.some(({ id }) => id === modelId)
    ? modelId
    : (models.find(({ id }) => id === "auto") ?? models[0]).id;
  const conversationProject = projects.find(({ id }) => id === conversation.project_id) ?? null;

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
            if (active) setTitleRunId(run.id);
          }
        } else if (active && latestAssistant?.run_id) {
          // A reload can land after the answer was saved but before its
          // auxiliary title finished. Showing history must not wait for it.
          setTitleRunId(latestAssistant.run_id);
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
          setTitleRunId(run.id);
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

  useEffect(() => () => {
    // Invalidate an older run-start title read when navigation or a manual
    // edit changes which response may update this thread.
    titleRefreshVersionRef.current += 1;
  }, [conversation.id, isSelected, editingTitle, mutation]);

  useEffect(() => {
    if (!titleRunId || !isSelected || editingTitle || mutation !== null) return;
    let active = true;
    let checks = 0;
    let timer: number | undefined;
    const controller = new AbortController();
    // Title generation has a short server budget. A failed or older server
    // must never leave a perpetual poll attached to a finished conversation.
    const deadline = window.setTimeout(() => {
      active = false;
      controller.abort();
      window.clearTimeout(timer);
      setTitleRunId(null);
    }, 20_000);

    async function checkTitle() {
      try {
        const run = await getRun(titleRunId!, controller.signal);
        if (!active) return;
        if (run.conversation_id !== conversation.id) throw new Error("Run belongs to another conversation.");
        checks += 1;
        if (run.title_pending) {
          if (checks < 20) timer = window.setTimeout(() => void checkTitle(), 1000);
          else setTitleRunId(null);
          return;
        }
        const updated = await getConversation(conversation.id, controller.signal);
        if (!active) return;
        setTitleDraft(updated.title);
        onConversationChangeRef.current(updated);
        setTitleRunId(null);
      } catch {
        if (active) setTitleRunId(null);
        // A later navigation or reload reconciles canonical state. Failure
        // of an auxiliary title refresh does not turn the answer into an error.
      }
    }
    void checkTitle();
    return () => {
      active = false;
      controller.abort();
      window.clearTimeout(timer);
      window.clearTimeout(deadline);
    };
  }, [conversation.id, titleRunId, isSelected, editingTitle, mutation]);

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

  async function changeModel(next: string, draft: ComposerDraft): Promise<boolean> {
    if (next === modelId || runActive || (privateLocked && next !== "private")) return false;
    setMutation("model");
    setMutationError("");
    try {
      const updated = await updateConversationModel(conversation.id, next);
      if (updated.id !== conversation.id) {
        onDraftReplaced(conversation.id, updated, draft);
        return true;
      }
      const messages = await listMessages(conversation.id);
      setThread({ status: "ready", messages: messages.items });
      setModelId(updated.default_model_id);
      onConversationChange(updated);
      return true;
    } catch (reason) {
      setMutationError(messageOf(reason));
      return false;
    } finally {
      setMutation(null);
    }
  }

  async function refreshAutomaticTitle() {
    if (!isSelected || editingTitle || mutation !== null) return;
    const version = titleRefreshVersionRef.current;
    try {
      const updated = await getConversation(conversation.id);
      if (version !== titleRefreshVersionRef.current) return;
      setTitleDraft(updated.title);
      onConversationChangeRef.current(updated);
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

  async function moveToProject(projectId: string | null) {
    if ((conversation.project_id ?? null) === projectId || runActive) return;
    setMutation("project");
    setMutationError("");
    try {
      onConversationChange(await updateConversation(conversation.id, {
        project_id: projectId,
      }));
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
      onRemoveFromView(conversation.id, true, true);
    } catch (reason) {
      setMutationError(actionError("Could not delete conversation", reason));
      setConfirmingDelete(false);
    } finally {
      setMutation(null);
    }
  }

  const conversationHeader = (
    <header className="thread-header">
      <div className="thread-title">
        {conversationProject ? (
          <nav className="project-breadcrumb" aria-label="Project breadcrumb">
            <button type="button" onClick={() => onOpenProject(conversationProject.id)}>{conversationProject.name}</button>
            <span aria-hidden="true">›</span>
            <span>Conversation</span>
          </nav>
        ) : (
          <span>Conversation</span>
        )}
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
      {compactNavigation && mutationError ? (
        <p className="conversation-mutation-error" role="alert">{mutationError}</p>
      ) : null}
    </header>
  );
  const projectPicker = (
    <ComposerProjectPicker
      projects={projects.filter((project) => Boolean(project.private_required) === privateRequired)}
      projectId={conversation.project_id ?? null}
      projectName={conversationProject?.name}
      disabled={runActive || mutation !== null}
      onChange={moveToProject}
    />
  );

  return (
    <>
      <div className="thread-header-shell">
        {!compactNavigation ? conversationHeader : null}
        {mutationError ? <p className="conversation-mutation-error" role="alert">{mutationError}</p> : null}
        {recoveredRunId && !durableReviewPresent ? (
          <div className="recovered-run" role="status" aria-label="Audrey is answering">
            <span className="recovered-run-orb" aria-hidden="true" />
            <div className="recovered-run-copy">
              <strong>Audrey is thinking</strong>
              <span>Your answer will appear here when it is ready.</span>
            </div>
            <button type="button" onClick={() => void stopRecoveredRun()} disabled={stoppingRecovered}>
              {stoppingRecovered ? "Stopping…" : "Stop run"}
            </button>
            {recoveryError ? <span>Could not check run: {recoveryError}</span> : null}
          </div>
        ) : null}
      </div>

      {thread.status === "loading" || thread.status === "idle" ? (
        <AudreyLoader label="Loading conversation" showPortrait={false} />
      ) : null}
      {thread.status === "error" ? (
        <div className="thread-loading thread-error" role="alert">{thread.message}</div>
      ) : null}
      {thread.status === "ready" ? (
        <AudreyThread
          key={threadRevision}
          conversationId={conversation.id}
          focusComposer={focusComposer}
          onComposerFocused={onComposerFocused}
          projectPicker={projectPicker}
          models={conversationModels}
          privateRequired={privateRequired}
          privateLockReason={privateLocked ? conversation.project_id
            ? "Leave this Private project before choosing another processing mode."
            : "Private stays selected after the first message." : ""}
          skills={skills}
          canBrowseDirectModels={canBrowseDirectModels}
          modelId={selectedModelId}
          showProgress={showProgress}
          initialMessages={thread.messages}
          replacementDraft={pendingDraft}
          onDraftRestored={() => {
            setPendingDraft(undefined);
            onDraftConsumed(conversation.id);
          }}
          readOnly={archived}
          modeDisabled={runActive || mutation !== null}
          recoveredRunActive={recoveredRunId !== null}
          onDurableReviewPresenceChange={setDurableReviewPresent}
          onModelChange={changeModel}
          onRunActiveChange={setRunActive}
          onRunStarted={() => {
            setAcceptedMessage(true);
            void refreshAutomaticTitle();
          }}
          onRunFinished={(runId) => {
            titleRefreshVersionRef.current += 1;
            setTitleRunId(runId);
          }}
        />
      ) : null}
    </>
  );
}

function AudreyThread({
  privateRequired,
  privateLockReason,
  conversationId,
  focusComposer,
  onComposerFocused,
  projectPicker,
  models,
  skills,
  canBrowseDirectModels,
  modelId,
  showProgress,
  initialMessages,
  replacementDraft,
  onDraftRestored,
  readOnly,
  modeDisabled,
  recoveredRunActive,
  onDurableReviewPresenceChange,
  onModelChange,
  onRunActiveChange,
  onRunStarted,
  onRunFinished,
}: {
  privateRequired: boolean;
  privateLockReason: string;
  conversationId: string;
  focusComposer: boolean;
  onComposerFocused: (conversationId: string) => void;
  projectPicker: ReactNode;
  canBrowseDirectModels: boolean;
  models: AudreyModel[];
  skills: SkillSummary[];
  modelId: string;
  showProgress: boolean;
  initialMessages: ConversationMessage[];
  replacementDraft?: ComposerDraft;
  onDraftRestored: () => void;
  readOnly: boolean;
  modeDisabled: boolean;
  recoveredRunActive: boolean;
  onDurableReviewPresenceChange: (present: boolean) => void;
  onModelChange: (modelId: string, draft: ComposerDraft) => Promise<boolean>;
  onRunActiveChange: (active: boolean) => void;
  onRunStarted: () => void;
  onRunFinished: (runId: string) => void;
}) {
  const isPrivate = privateRequired || modelId === "private";
  const [runError, setRunError] = useState("");
  const [skillId, setSkillId] = useState("");
  const [taxCase, setTaxCase] = useState<TaxCase | null>(null);
  const [privateTask, setPrivateTask] = useState<PrivateTask | null | undefined>(undefined);
  const acceptPrivateTask = useCallback((task: PrivateTask | null) => {
    setPrivateTask(task);
    onDurableReviewPresenceChange(task !== null);
  }, [onDurableReviewPresenceChange]);
  const [taxReviewExpanded, setTaxReviewExpanded] = useState(false);
  const [taxReviewRefresh, setTaxReviewRefresh] = useState(0);
  const [queuedTaxReview, setQueuedTaxReview] = useState<{ text: string } | null>(null);
  const pendingTaxReviewRef = useRef<object | null>(null);
  const dispatchedTaxReviewRef = useRef<object | null>(null);
  // A review uses saved case documents; the user's unsent composer files stay untouched.
  const preserveComposerAttachmentsRef = useRef(false);
  const [activityLedger, setActivityLedger] = useState<ActivityLedger>({
    activity: IDLE_ACTIVITY,
    answerMessageId: null,
    answers: new Map(),
  });
  const activity = activityLedger.activity;
  const setActivity = useCallback((
    update: SetStateAction<RunActivity>,
    messageId?: string | null,
  ) => {
    setActivityLedger((current) => {
      const next = typeof update === "function" ? update(current.activity) : update;
      const answerMessageId = messageId === undefined ? current.answerMessageId : messageId;
      let answers = current.answers;
      if (answerMessageId && ["complete", "cancelled", "error"].includes(next.status)) {
        answers = new Map(answers).set(answerMessageId, {
          sources: next.sources,
          models: next.models,
          tools: settleRunningTools(next.tools),
        });
      }
      return { activity: next, answerMessageId, answers };
    });
  }, []);
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
  const [queuedRetry, setQueuedRetry] = useState<{
    text: string;
    attachments: AudreyFile[];
  } | null>(null);
  const dispatchedRetryRef = useRef<object | null>(null);
  const [activePicker, setActivePicker] = useState<ComposerPickerName | null>(null);
  const attachmentPickerOpen = activePicker === "files";
  const skillPickerOpen = activePicker === "skills";
  const [attachmentFiles, setAttachmentFiles] = useState<AudreyFile[]>([]);
  const [attachmentsLoading, setAttachmentsLoading] = useState(false);
  const [attachmentError, setAttachmentError] = useState("");
  const [modelChangeNotice, setModelChangeNotice] = useState(replacementDraft?.notice ?? "");
  const [attachmentSearch, setAttachmentSearch] = useState("");
  const [attachmentKind, setAttachmentKind] = useState<AttachmentKindFilter>("all");
  const [selectedAttachments, setSelectedAttachments] = useState<AudreyFile[]>([]);
  const [imageLimit, setImageLimit] = useState<number | null>(null);
  const [fileLimits, setFileLimits] = useState<AudreyFileLimits | null>(null);
  const [uploadingFile, setUploadingFile] = useState("");
  const [uploadProgress, setUploadProgress] = useState(0);
  const [pendingAttachment, setPendingAttachment] = useState<AudreyFile | null>(null);
  const [uploadIssue, setUploadIssue] = useState("");
  const uploadInputRef = useRef<HTMLInputElement>(null);
  const attachButtonRef = useRef<HTMLButtonElement>(null);
  const skillButtonRef = useRef<HTMLButtonElement>(null);
  const composerInputRef = useRef<HTMLTextAreaElement>(null);
  useEffect(() => {
    if (!focusComposer || readOnly) return;
    // Wait until the loaded draft is mounted and drawer focus restoration ends.
    const frame = window.requestAnimationFrame(() => {
      const input = composerInputRef.current;
      if (!input || input.closest("[hidden], [inert]")) return;
      input.focus({ preventScroll: true });
      onComposerFocused(conversationId);
    });
    return () => window.cancelAnimationFrame(frame);
  }, [conversationId, focusComposer, onComposerFocused, readOnly]);
  const [activeRunId, setActiveRunId] = useState<string | null>(null);
  const userRequestedCancelRef = useRef(false);
  const attachmentBusy = Boolean(uploadingFile) || pendingAttachment !== null;
  const durableReviewActive = privateTask?.status === "queued" || privateTask?.status === "running";
  const durableRetryHandled = Boolean(isPrivate && taxCase && (privateTask === undefined
    || privateTask && ["queued", "running", "paused", "waiting_for_user"].includes(privateTask.status)));
  const submissionBlocked = modeDisabled || attachmentBusy || retrying || Boolean(uploadIssue) || queuedTaxReview !== null || durableReviewActive;
  const selectedImageCount = selectedAttachments.filter(({ kind }) => kind === "image").length;
  const visibleAttachmentFiles = useMemo(() => {
    const search = attachmentSearch.trim().toLocaleLowerCase();
    return attachmentFiles.filter((file) => (
      (attachmentKind === "all" || file.kind === attachmentKind)
      && (!search || file.filename.toLocaleLowerCase().includes(search))
    ));
  }, [attachmentFiles, attachmentKind, attachmentSearch]);
  const selectedModel = modelDetails(models, modelId);
  const selectedSkillMode = skillModeForModel(selectedModel);
  const compatibleSkills = isPrivate ? [] : skills.filter((skill) =>
    selectedSkillMode !== null && skill.supported_modes.includes(selectedSkillMode),
  );
  const selectedSkill = skills.find((skill) => skill.id === skillId);
  const effectiveSkillId = isPrivate
    ? skillId === "tax-review" && taxCase ? "tax-review" : ""
    : skillId === "tax-review" ? "" : skillId;
  const skillControlValue = isPrivate ? effectiveSkillId === "tax-review" ? "Tax review" : "Automatic" : selectedSkill?.name
    ?? (selectedSkillMode === null ? "Not available" : "Automatic");
  const acceptTaxCase = useCallback((value: TaxCase | null) => {
    setTaxCase((current) => current && value && current.id === value.id && current.revision > value.revision
      ? current : value);
  }, []);
  const supportsFiles = selectedModel.capabilities.includes("files");
  useEffect(() => {
    if (!skillId) return;
    if (isPrivate && skillId === "tax-review") return;
    const selectedSkill = skills.find((skill) => skill.id === skillId);
    if (
      isPrivate
      || skillId === "tax-review"
      || !selectedSkill
      || selectedSkill.availability !== "available"
      || selectedSkillMode === null
      || !selectedSkill.supported_modes.includes(selectedSkillMode)
    ) {
      setSkillId("");
    }
  }, [isPrivate, selectedSkillMode, skillId, skills]);
  const attachmentIds = useMemo(
    () => selectedAttachments.map(({ id }) => id),
    [selectedAttachments],
  );
  const savedSources = useMemo(() => {
    const sources = new Map(initialMessages.filter(({ role }) => role === "assistant")
      .map(({ id, sources }) => [id, sources ?? []] as const));
    for (const [id, answer] of activityLedger.answers) sources.set(id, answer.sources);
    return sources;
  }, [initialMessages, activityLedger.answers]);
  const savedModels = useMemo(() => {
    const models = new Map(initialMessages.filter(({ role }) => role === "assistant")
      .map(({ id, models }) => [id, models ?? []] as const));
    for (const [id, answer] of activityLedger.answers) models.set(id, answer.models);
    return models;
  }, [initialMessages, activityLedger.answers]);
  const savedTools = useMemo(() => {
    const tools = new Map<string, RunTool[]>(initialMessages.filter(({ role }) => role === "assistant")
      .map(({ id, tool_calls }) => [id, tool_calls ?? []] as const));
    for (const [id, answer] of activityLedger.answers) tools.set(id, answer.tools);
    return tools;
  }, [initialMessages, activityLedger.answers]);
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
  const fetchAgentAction = useCallback((url: string, init: RequestInit) => latestActionFetch(
    url, init, preserveComposerAttachmentsRef.current ? [] : attachmentIds, setActiveRunId,
  ), [attachmentIds]);
  const agent = useMemo(
    () =>
      // HttpAgent stores this fetch callback; it reads the current request guard only when called, never during render.
      // eslint-disable-next-line react-hooks/refs
      new HttpAgent({
        url: `/api/agent?model=${encodeURIComponent(modelId)}${
          effectiveSkillId ? `&skill=${encodeURIComponent(effectiveSkillId)}` : ""
        }`,
        threadId: conversationId,
        fetch: fetchAgentAction,
      }),
    [fetchAgentAction, conversationId, modelId, effectiveSkillId],
  );
  async function changeModel(nextModelId: string) {
    if (attachmentBusy || retrying) return;
    setActivePicker(null);
    const changesProtection = (nextModelId === "private") !== isPrivate;
    const notice = changesProtection && selectedAttachments.length > 0
      ? "Attachments were removed when you changed processing mode. Your files remain in My Files." : "";
    const changed = await onModelChange(nextModelId, { text: runtime.thread.composer.getState().text, notice });
    if (!changed) return;
    setModelChangeNotice(notice);
    const nextModel = modelDetails(models, nextModelId);
    const nextSkillMode = skillModeForModel(nextModel);
    const selectedSkill = skills.find((skill) => skill.id === skillId);
    if (
      selectedSkill
      && (nextSkillMode === null || !selectedSkill.supported_modes.includes(nextSkillMode))
    ) {
      setSkillId("");
    }
    if (!nextModel.capabilities.includes("files") || (nextModelId === "private") !== isPrivate) {
      void runtime.thread.composer.clearAttachments();
      setActivePicker(null);
      setSelectedAttachments([]);
    }
  }
  const onRunStartedRef = useRef(onRunStarted);
  const onRunFinishedRef = useRef(onRunFinished);
  useEffect(() => {
    onRunStartedRef.current = onRunStarted;
    onRunFinishedRef.current = onRunFinished;
  }, [onRunStarted, onRunFinished]);
  useEffect(() => {
    const subscriber: AgentSubscriber = {
      onRunInitialized: () => {
        userRequestedCancelRef.current = false;
        onRunActiveChange(true);
        setRetrying(false);
        setRunError("");
        setActivity({
          status: "running",
          label: "Starting",
          detail: "Preparing Audrey's run",
          sources: [],
          models: [],
          tools: [],
        }, null);
      },
      onRunStartedEvent: () => {
        setActivePicker(null);
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
        if (event.name === "audrey.model.used") {
          const model = stringOf(recordOf(event.value).model);
          if (model) {
            setActivity((current) => {
              const existing = current.models.find(({ model: name }) => name === model);
              if (!existing) {
                return { ...current, models: [...current.models, { model, calls: 1 }] };
              }
              return {
                ...current,
                models: current.models.map((item) => (
                  item.model === model ? { ...item, calls: item.calls + 1 } : item
                )),
              };
            });
          }
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
      onTextMessageStartEvent: ({ event }) => {
        if (event.role === "assistant") setActivity((current) => current, event.messageId);
      },
      onToolCallStartEvent: ({ event }) => {
        setActivity((current) => {
          const tool: RunTool = {
            id: event.toolCallId,
            name: event.toolCallName,
            status: "running",
          };
          const index = current.tools.findIndex(({ id }) => id === tool.id);
          if (index < 0) return { ...current, tools: [...current.tools, tool] };
          const tools = [...current.tools];
          tools[index] = tool;
          return { ...current, tools };
        });
      },
      onToolCallResultEvent: ({ event }) => {
        const status = toolResultFailed(event.content) ? "failed" : "succeeded";
        setActivity((current) => ({
          ...current,
          tools: current.tools.map((tool) => (
            tool.id === event.toolCallId ? { ...tool, status } : tool
          )),
        }));
      },
      onRunFinishedEvent: ({ event }) => {
        onRunFinishedRef.current(event.runId);
        setTaxReviewRefresh((current) => current + 1);
        setActiveRunId(null);
        userRequestedCancelRef.current = false;
        onRunActiveChange(false);
        setRetrying(false);
        setLastAttempt(null);
        if (pendingTaxReviewRef.current === dispatchedTaxReviewRef.current) pendingTaxReviewRef.current = null;
        if (!preserveComposerAttachmentsRef.current) setSelectedAttachments([]);
        setActivity((current) => ({
          ...current,
          status: "complete",
          label: "Complete",
          detail: "Response finished",
          tools: settleRunningTools(current.tools),
        }));
      },
      onRunErrorEvent: ({ event }) => {
        setTaxReviewRefresh((current) => current + 1);
        setActiveRunId(null);
        userRequestedCancelRef.current = false;
        onRunActiveChange(false);
        setRetrying(false);
        if (pendingTaxReviewRef.current === dispatchedTaxReviewRef.current) pendingTaxReviewRef.current = null;
        if (!preserveComposerAttachmentsRef.current) setSelectedAttachments([]);
        const paused = event.code === "private_task_paused";
        const cancelled = event.code === "cancelled_by_user" || isAbortMessage(event.message);
        if (paused) setRunError("");
        setActivity((current) => ({
          ...current,
          status: paused ? "complete" : cancelled ? "cancelled" : "error",
          label: paused ? "Review paused" : cancelled ? "Stopped" : "Run failed",
          detail: paused ? "Saved work is retained" : cancelled ? "Run cancelled" : event.message || "The response did not finish cleanly",
          tools: settleRunningTools(current.tools),
        }));
      },
      onRunFailed: ({ error }) => {
        if (pendingTaxReviewRef.current === dispatchedTaxReviewRef.current) pendingTaxReviewRef.current = null;
        onRunActiveChange(false);
        setRetrying(false);
        const paused = stringOf(recordOf(error).code) === "private_task_paused";
        const cancelled = isAbortError(error);
        if (!cancelled) setActiveRunId(null);
        if (paused) setRunError("");
        setActivity((current) => ({
          ...current,
          status: paused ? "complete" : cancelled ? "cancelled" : "error",
          label: paused ? "Review paused" : cancelled ? "Stopped" : "Connection failed",
          detail: paused ? "Saved work is retained" : cancelled ? "Run cancelled" : error.message,
          tools: settleRunningTools(current.tools),
        }));
      },
    };
    const subscription = agent.subscribe(subscriber);
    return () => subscription.unsubscribe();
  }, [agent, onRunActiveChange, setActivity]);
  const runtime = useAgUiRuntime({
    agent,
    adapters: { history },
    showThinking: false,
    onError: (reason) => {
      if (pendingTaxReviewRef.current === dispatchedTaxReviewRef.current) pendingTaxReviewRef.current = null;
      onRunActiveChange(false);
      setRetrying(false);
      if (stringOf(recordOf(reason).code) === "private_task_paused") {
        setRunError("");
        setActivity((current) => ({ ...current, status: "complete", label: "Review paused", detail: "Saved work is retained" }));
        return;
      }
      setRunError(reason.message);
      setActivity((current) => current.status === "error" ? current : {
        ...current,
        status: "error",
        label: "Run failed",
        detail: reason.message,
      });
    },
    onCancel: () => {
      const explicitlyStopped = userRequestedCancelRef.current;
      userRequestedCancelRef.current = false;
      const runId = activeRunId;
      if (pendingTaxReviewRef.current === dispatchedTaxReviewRef.current) pendingTaxReviewRef.current = null;
      setTaxReviewRefresh((current) => current + 1);
      if (!preserveComposerAttachmentsRef.current) setSelectedAttachments([]);
      setRunError("");
      if (!explicitlyStopped || !runId) {
        onRunActiveChange(false);
        setActivity((current) => ({
          ...current,
          status: "cancelled",
          label: "Stopped",
          detail: "Run cancelled",
        }));
        return;
      }
      onRunActiveChange(true);
      setActivity((current) => ({
        ...current,
        status: "running",
        label: "Stopping",
        detail: "Finishing cancellation",
      }));
      void cancelRun(runId).then(() => {
        setActiveRunId(null);
        onRunActiveChange(false);
        setActivity((current) => ({
          ...current,
          status: "cancelled",
          label: "Stopped",
          detail: "Run cancelled",
        }));
      }).catch((reason) => {
        onRunActiveChange(false);
        const detail = messageOf(reason);
        setRunError("Could not stop run: " + detail);
        setActivity((current) => ({
          ...current,
          status: "error",
          label: "Stop failed",
          detail,
        }));
      });
    },
  });

  const restoredDraftRef = useRef(false);
  useEffect(() => {
    if (!replacementDraft || restoredDraftRef.current) return;
    restoredDraftRef.current = true;
    runtime.thread.composer.setText(replacementDraft.text);
    onDraftRestored();
  }, [onDraftRestored, replacementDraft, runtime]);

  function startTaxReview(text = TAX_REVIEW_PROMPT) {
    if (!taxCase || !isPrivate || readOnly || pendingTaxReviewRef.current
      || ["update_available", "unavailable"].includes(taxCase.reference_update?.status ?? "")) return;
    if (text === TAX_REVIEW_PROMPT && (submissionBlocked || recoveredRunActive
      || activity.status === "running" || runtime.thread.getState().isRunning)) return;
    const request = { text };
    pendingTaxReviewRef.current = request;
    preserveComposerAttachmentsRef.current = true;
    setQueuedTaxReview(request);
    setLastAttempt(null);
    setSkillId("tax-review");
    setActivePicker(null);
  }

  useEffect(() => {
    if (!queuedTaxReview || effectiveSkillId !== "tax-review" || dispatchedTaxReviewRef.current === queuedTaxReview
      || readOnly || modeDisabled || attachmentBusy || retrying || recoveredRunActive || activity.status === "running") return;
    // useAgUiRuntime installs the newly selected agent in its effect before this dispatch.
    const timer = window.setTimeout(() => {
      if (pendingTaxReviewRef.current !== queuedTaxReview || dispatchedTaxReviewRef.current === queuedTaxReview) return;
      if (runtime.thread.getState().isRunning) return;
      dispatchedTaxReviewRef.current = queuedTaxReview;
      setRunError("");
      setActivity({ status: "running", label: "Starting review", detail: "Reviewing your saved tax case", sources: [], models: [], tools: [] });
      try {
        runtime.thread.append({ role: "user", content: [{ type: "text", text: queuedTaxReview.text }], attachments: [] });
      } catch (reason) {
        pendingTaxReviewRef.current = null;
        setRunError("Could not start review: " + messageOf(reason));
        setActivity((current) => ({ ...current, status: "error", label: "Run failed" }));
      } finally {
        setQueuedTaxReview(null);
      }
    }, 0);
    return () => window.clearTimeout(timer);
  }, [queuedTaxReview, effectiveSkillId, readOnly, modeDisabled, attachmentBusy, retrying, recoveredRunActive, activity.status, runtime, setActivity]);

  async function retryLastQuestion() {
    if (!lastAttempt || retrying || modeDisabled || attachmentBusy || readOnly) return;
    preserveComposerAttachmentsRef.current = false;
    setRetrying(true);
    setRunError("");
    try {
      if (lastAttempt.attachmentIds.length > 0 && !supportsFiles) {
        throw new Error("The selected model accepts text only. Choose a file-capable model to retry this question.");
      }
      const attachments = await Promise.all(lastAttempt.attachmentIds.map((id) => getFile(id)));
      if (attachments.some((file) => !attachmentAllowed(file, isPrivate))) {
        throw new Error("These files are not available in this conversation. Choose files from its file picker.");
      }
      if (attachments.some(({ status }) => status !== "ready")) {
        throw new Error("An attached file is no longer ready. Choose a ready file before sending again.");
      }
      if (imageLimit !== null && attachments.filter(({ kind }) => kind === "image").length > imageLimit) {
        throw new Error("The image limit changed. Remove an image before sending again.");
      }
      setSelectedAttachments(attachments);
      setUploadIssue("");
      setQueuedRetry({ text: lastAttempt.text, attachments });
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
      setActivity({
        status: "running",
        label: "Retrying",
        detail: "Starting another run",
        sources: [],
        models: [],
        tools: [],
      });
      try {
        runtime.thread.append({
          role: "user",
          content: [{ type: "text", text: queuedRetry.text }],
          attachments: queuedRetry.attachments.map(toCompleteAttachment),
        });
      } catch (reason) {
        setRunError("Could not retry: " + messageOf(reason));
        setActivity((current) => ({ ...current, status: "error", label: "Run failed" }));
      } finally {
        setQueuedRetry(null);
        setRetrying(false);
      }
    }, 0);
    return () => window.clearTimeout(timer);
  }, [queuedRetry, runtime, setActivity]);

  async function toggleAttachmentPicker() {
    if (attachmentPickerOpen) {
      setActivePicker(null);
      return;
    }
    setActivePicker("files");
    setAttachmentSearch("");
    setAttachmentKind("all");
    setAttachmentsLoading(true);
    setAttachmentError("");
    try {
      const listing = await listFiles();
      setAttachmentFiles(listing.items.filter((file) => file.status === "ready" && attachmentAllowed(file, isPrivate)));
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

  function toggleSkillPicker() {
    if (skillPickerOpen) {
      setActivePicker(null);
      return;
    }
    setActivePicker("skills");
  }

  function toggleAttachment(file: AudreyFile) {
    if (attachmentBusy) return;
    const selected = selectedAttachments.some(({ id }) => id === file.id);
    if (!selected && selectedAttachments.length >= 10) return;
    if (
      !selected
      && file.kind === "image"
      && imageLimit !== null
      && selectedImageCount >= imageLimit
    ) return;
    void (selected ? removeComposerAttachment(file.id) : addComposerAttachment(file))
      .then(() => {
        setSelectedAttachments((current) => selected
          ? current.filter(({ id }) => id !== file.id)
          : current.some(({ id }) => id === file.id) ? current : [...current, file]);
      })
      .catch((reason) => setUploadIssue(
        `Could not ${selected ? "remove" : "attach"} ${file.filename}: ${messageOf(reason)}`,
      ));
  }

  async function addComposerAttachment(file: AudreyFile) {
    const composer = runtime.thread.composer;
    if (composer.getState().attachments.some(({ id }) => id === file.id)) return;
    await composer.addAttachment(toCreateAttachment(file));
  }

  async function removeComposerAttachment(fileId: string) {
    const composer = runtime.thread.composer;
    const index = composer.getState().attachments.findIndex(({ id }) => id === fileId);
    if (index >= 0) await composer.getAttachmentByIndex(index).remove();
  }

  async function selectUploadedAttachment(file: AudreyFile) {
    await addComposerAttachment(file);
    setSelectedAttachments((current) => current.some(({ id }) => id === file.id)
      ? current : [...current, file]);
  }

  async function uploadFromChat(file: File) {
    if (!fileLimits || attachmentBusy || selectedAttachments.length >= 10) return;
    const precheck = uploadPrecheck(file, fileLimits, isPrivate);
    if (precheck) {
      setUploadIssue(file.name + ": " + precheck);
      return;
    }
    setUploadIssue("");
    setUploadingFile(file.name);
    setUploadProgress(0);
    setActivePicker(null);
    let stored = false;
    try {
      const uploaded = await uploadFile(file, fileLimits, setUploadProgress, isPrivate ? conversationId : undefined);
      stored = true;
      // The upload response has no failure details; fetch the owned row before attaching.
      const row = await getFile(uploaded.id);
      if (!attachmentAllowed(row, isPrivate)) throw new Error("This file cannot be attached in this conversation.");
      if (row.status === "ready") {
        if (row.kind === "image" && imageLimit !== null && selectedImageCount >= imageLimit) {
          setUploadIssue(row.filename + " was uploaded to Files, but the image limit is full. Remove an image to attach it.");
          setAttachmentFiles((current) => [row, ...current]);
          return;
        }
        await selectUploadedAttachment(row);
        setAttachmentFiles((current) => [row, ...current.filter(({ id }) => id !== row.id)]);
      } else {
        setPendingAttachment(row);
      }
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
          const composer = runtime.thread.composer;
          if (!composer.getState().attachments.some(({ id }) => id === row.id)) {
            await composer.addAttachment(toCreateAttachment(row));
          }
          if (!active) return;
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
  }, [pendingId, pendingStatus, runtime]);

  return (
    <ComposerPickerContext.Provider value={{ activePicker, setActivePicker }}>
    <SavedSourcesContext.Provider value={savedSources}>
      <SavedModelsContext.Provider value={savedModels}>
      <SavedToolsContext.Provider value={savedTools}>
        <AssistantRuntimeProvider runtime={runtime}>
      <ThreadPrimitive.Root className="thread-root">
        <ThreadPrimitive.Viewport className="thread-viewport">
          <ThreadPrimitive.Messages
            components={{
              UserMessage,
              AssistantMessage,
            }}
          />
          {isPrivate && taxCase && !readOnly ? <TaxReviewQuestions
            key={`${taxCase.id}:${taxCase.tax_year}`}
            conversationId={conversationId}
            taxCase={taxCase}
            answerVersion={activityLedger.answers.size}
            disabled={submissionBlocked || recoveredRunActive || activity.status === "running"
              || ["update_available", "unavailable"].includes(taxCase.reference_update?.status ?? "")}
            onCaseChange={(value) => { acceptTaxCase(value); setTaxReviewRefresh((current) => current + 1); }}
            onReload={() => setTaxReviewRefresh((current) => current + 1)}
            onChoice={(_value, feedback) => startTaxReview(feedback)}
          /> : null}
          <ThreadPrimitive.If empty={false}>
            <div className="thread-message-spacer" aria-hidden="true" />
          </ThreadPrimitive.If>
          <ThreadPrimitive.ViewportFooter
            className="composer-dock"
          >
            <ScrollToLatest />
            {readOnly ? (
              <p className="archived-notice" role="status">
                This conversation is archived. Restore it to continue.
              </p>
            ) : (
              <>
                {((showProgress && activity.status !== "complete") || activity.status === "error") ? (
                  <RunActivityStatus activity={activity} error={runError} />
                ) : null}
                {isPrivate ? <TaxReview
                  conversationId={conversationId}
                  expanded={taxReviewExpanded}
                  active={effectiveSkillId === "tax-review"}
                  disabled={submissionBlocked || activity.status === "running" || recoveredRunActive}
                  refreshVersion={taxReviewRefresh}
                  onExpandedChange={setTaxReviewExpanded}
                  onCaseChange={acceptTaxCase}
                  onSaved={() => setSkillId("tax-review")}
                  onReview={() => startTaxReview()}
                  taskControls={taxCase ? <PrivateTaskProgress
                    key={conversationId}
                    conversationId={conversationId}
                    refreshVersion={taxReviewRefresh}
                    disabled={submissionBlocked || activity.status === "running" || recoveredRunActive
                      || ["update_available", "unavailable"].includes(taxCase.reference_update?.status ?? "")}
                    hasReview={Boolean(taxCase.review)}
                    onReview={() => startTaxReview()}
                    onChanged={() => setTaxReviewRefresh((current) => current + 1)}
                    onTaskChange={acceptPrivateTask}
                  /> : undefined}
                /> : null}
                {isPrivate && taxCase && selectedAttachments.length > 0 ? <p className="attachment-status">Tax review uses files already sent here or added to the project. Send newly selected files before reviewing them.</p> : null}
                {lastAttempt && !durableRetryHandled && (restoredIncomplete || activity.status === "error" || runError) ? (
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
                  <UploadProgress filename={uploadingFile} progress={uploadProgress} />
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
                <ComposerPrimitive.Root
                  className="composer"
                  onSubmitCapture={(event) => {
                    if (submissionBlocked) {
                      event.preventDefault();
                      event.stopPropagation();
                      return;
                    }
                    preserveComposerAttachmentsRef.current = false;
                    const text = composerInputRef.current?.value.trim() ?? "";
                    if (text) setLastAttempt({ text, attachmentIds: selectedAttachments.map(({ id }) => id) });
                  }}
                >
                  <div className="composer-input-row">
                    <ComposerPrimitive.Input
                      ref={composerInputRef}
                      className="composer-input"
                      aria-label="Ask Audrey"
                      placeholder="Ask Audrey…"
                      rows={1}
                    />
                    <div className="composer-actions">
                      {!recoveredRunActive ? (
                        <ComposerPrimitive.Cancel
                          className="cancel-button"
                          onClick={() => { userRequestedCancelRef.current = true; }}
                        >Stop</ComposerPrimitive.Cancel>
                      ) : null}
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
                  </div>
                  <div className="composer-control-rail" aria-label="Message options">
                    <ComposerModelPicker
                      compact
                      models={models}
                      canBrowseDirectModels={canBrowseDirectModels}
                      modelId={modelId}
                      disabled={modeDisabled || attachmentBusy || retrying}
                      onChange={changeModel}
                    />
                    {projectPicker}
                    <button
                      ref={attachButtonRef}
                      className="composer-rail-control attach-button"
                      type="button"
                      onClick={() => void toggleAttachmentPicker()}
                      disabled={modeDisabled || retrying || !supportsFiles}
                      title={supportsFiles
                        ? "Upload a new file or choose ready files from My Files"
                        : `${selectedModel.label} accepts text only`}
                      aria-label={attachmentPickerOpen ? "Close file picker" : "Add files"}
                      aria-expanded={attachmentPickerOpen}
                    >
                      <span className="composer-control-label">Files</span>
                      <strong>
                        {selectedAttachments.length > 0
                          ? `${selectedAttachments.length} selected`
                          : "Add files"}
                      </strong>
                    </button>
                    <button
                      ref={skillButtonRef}
                      className="composer-rail-control skill-button"
                      type="button"
                      onClick={toggleSkillPicker}
                      disabled={submissionBlocked || selectedSkillMode === null}
                      title={isPrivate ? "Choose automatic local tools or configure a Tax review" : selectedSkillMode === null
                        ? `${selectedModel.label} does not support Audrey skills`
                        : "Choose automatic tool use or a focused Audrey skill"}
                      aria-label={`Tools and skills: ${skillControlValue}`}
                      aria-expanded={skillPickerOpen}
                    >
                      <span className="composer-control-label">Tools &amp; skills</span>
                      <strong>{skillControlValue}</strong>
                    </button>
                  </div>
                </ComposerPrimitive.Root>
                {modelChangeNotice ? <p className="attachment-status" role="status">{modelChangeNotice}</p> : null}
                {isPrivate ? <PrivateModeNotice lockReason={privateLockReason} /> : null}
                {attachmentPickerOpen ? (
                  <ComposerPickerPanel
                    className="attachment-picker"
                    label="Choose attachments"
                    role="region"
                    triggerRef={attachButtonRef}
                  >
                    <header>
                      <div>
                        <strong>Add files to this message</strong>
                        <span>{isPrivate
                          ? "Upload a document privately or choose a ready Private document."
                          : "Upload a new file or choose ready files from My Files."}</span>
                        <small>
                          {selectedAttachments.length}/10 files
                          {isPrivate || imageLimit === null ? "" : " · " + selectedImageCount + "/" + imageLimit + " images"}
                        </small>
                      </div>
                      <button
                        type="button"
                        className="attachment-picker-close"
                        aria-label="Hide attachment picker"
                        onClick={() => {
                          setActivePicker(null);
                          attachButtonRef.current?.focus({ preventScroll: true });
                        }}
                      >
                        <svg viewBox="0 0 24 24" aria-hidden="true">
                          <path d="m6.5 9 5.5 5.5L17.5 9" />
                        </svg>
                      </button>
                    </header>
                    <div className="attachment-picker-actions">
                      <input
                        type="search"
                        aria-label="Search ready files"
                        placeholder="Search your files"
                        value={attachmentSearch}
                        onChange={(event) => setAttachmentSearch(event.target.value)}
                      />
                      <div className="attachment-upload">
                        <button
                          type="button"
                          onClick={() => uploadInputRef.current?.click()}
                          disabled={!fileLimits || attachmentBusy || selectedAttachments.length >= 10}
                        >Upload</button>
                        <input
                          ref={uploadInputRef}
                          type="file"
                          aria-label="Choose a file from your device"
                          tabIndex={-1}
                          accept={(isPrivate ? PRIVATE_DOCUMENT_EXTENSIONS : fileLimits?.allowed_extensions)?.join(",")}
                          onChange={(event) => {
                            const file = event.currentTarget.files?.[0];
                            event.currentTarget.value = "";
                            if (file) void uploadFromChat(file);
                          }}
                        />
                      </div>
                    </div>
                    {attachmentFiles.length > 0 ? (
                      <div className="attachment-folders" role="group" aria-label="File types">
                        {ATTACHMENT_FOLDERS.filter((folder) => !isPrivate || ["all", "text"].includes(folder.kind)).map((folder) => {
                          const count = folder.kind === "all"
                            ? attachmentFiles.length
                            : attachmentFiles.filter(({ kind }) => kind === folder.kind).length;
                          return (
                            <button
                              type="button"
                              key={folder.kind}
                              aria-pressed={attachmentKind === folder.kind}
                              onClick={() => setAttachmentKind(folder.kind)}
                            >
                              <span aria-hidden="true">{folder.symbol}</span>
                              {folder.label}
                              <small>{count}</small>
                            </button>
                          );
                        })}
                      </div>
                    ) : null}
                    {attachmentsLoading ? <p role="status">Loading files…</p> : null}
                    {attachmentError ? <p className="attachment-error" role="alert">{attachmentError}</p> : null}
                    {!attachmentsLoading && !attachmentError && attachmentFiles.length === 0 ? (
                      <p>No ready files yet. Upload one here to ask about it.</p>
                    ) : null}
                    {!attachmentsLoading && attachmentFiles.length > 0 ? (
                      <div className="attachment-picker-result" role="status">
                        {visibleAttachmentFiles.length} of {attachmentFiles.length} ready files
                      </div>
                    ) : null}
                    {!attachmentsLoading && attachmentFiles.length > 0 && visibleAttachmentFiles.length === 0 ? (
                      <p>No files match this view.</p>
                    ) : null}
                    {visibleAttachmentFiles.length > 0 ? (
                      <ul className="attachment-options">
                        {visibleAttachmentFiles.map((file) => {
                          const selected = selectedAttachments.some(({ id }) => id === file.id);
                          return (
                            <li key={file.id}>
                              <button
                                type="button"
                                aria-pressed={selected}
                                onClick={() => toggleAttachment(file)}
                                disabled={attachmentBusy || (!selected && (
                                  selectedAttachments.length >= 10
                                  || (file.kind === "image"
                                    && imageLimit !== null
                                    && selectedImageCount >= imageLimit)
                                ))}
                              >
                                <span className="attachment-file-kind" aria-hidden="true">
                                  {attachmentKindSymbol(file.kind)}
                                </span>
                                <span className="attachment-file-name">{file.filename}</span>
                                <small>{attachmentKindLabel(file.kind)} · {formatBytes(file.bytes)}</small>
                                <span className="attachment-file-selection" aria-hidden="true">
                                  {selected ? "✓" : "+"}
                                </span>
                              </button>
                            </li>
                          );
                        })}
                      </ul>
                    ) : null}
                  </ComposerPickerPanel>
                ) : null}
                {skillPickerOpen ? (
                  <ComposerPickerPanel
                    className="skill-picker"
                    label="Choose how Audrey uses tools"
                    triggerRef={skillButtonRef}
                  >
                    <header>
                      <div>
                        <strong>Choose how Audrey uses tools</strong>
                        <span>
                          {isPrivate ? "Audrey automatically uses available local tools. Web, saved memory and media tools are unavailable in Private."
                            : "Skills give Audrey focused instructions and may narrow the tools available for this message."}
                        </span>
                      </div>
                      <button
                        type="button"
                        className="attachment-picker-close"
                        aria-label="Hide tools and skills picker"
                        onClick={() => {
                          setActivePicker(null);
                          skillButtonRef.current?.focus({ preventScroll: true });
                        }}
                      >
                        <svg viewBox="0 0 24 24" aria-hidden="true">
                          <path d="m6.5 9 5.5 5.5L17.5 9" />
                        </svg>
                      </button>
                    </header>
                    <div className="skill-options" role="group" aria-label="Tools and skills choices">
                      <button
                        type="button"
                        aria-pressed={!skillId}
                        onClick={() => {
                          setSkillId("");
                          setTaxReviewExpanded(false);
                          setActivePicker(null);
                          skillButtonRef.current?.focus({ preventScroll: true });
                        }}
                      >
                        <strong>Automatic</strong>
                        <span>{isPrivate ? "Audrey chooses available local tools when they are useful." : "Audrey chooses the available tools when they are useful."}</span>
                      </button>
                      {isPrivate ? <button
                        type="button"
                        aria-pressed={effectiveSkillId === "tax-review"}
                        onClick={() => {
                          if (taxCase) setSkillId("tax-review");
                          setTaxReviewExpanded(true);
                          setActivePicker(null);
                        }}
                      >
                        <strong>Tax review</strong>
                        <span>Configure a saved US federal tax case, review proposed facts and track missing information.</span>
                      </button> : null}
                      {compatibleSkills.map((skill) => (
                        <button
                          type="button"
                          key={skill.id}
                          aria-pressed={skillId === skill.id}
                          disabled={skill.availability !== "available"}
                          onClick={() => {
                            setSkillId(skill.id);
                            setActivePicker(null);
                            skillButtonRef.current?.focus({ preventScroll: true });
                          }}
                        >
                          <strong>
                            {skill.name}{skill.availability === "available" ? "" : " (unavailable)"}
                          </strong>
                          <span>{skill.description}</span>
                        </button>
                      ))}
                    </div>
                  </ComposerPickerPanel>
                ) : null}
              </>
            )}
          </ThreadPrimitive.ViewportFooter>
        </ThreadPrimitive.Viewport>
      </ThreadPrimitive.Root>
        </AssistantRuntimeProvider>
      </SavedToolsContext.Provider>
      </SavedModelsContext.Provider>
    </SavedSourcesContext.Provider>
    </ComposerPickerContext.Provider>
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
        <ExclusiveRunDetails
          className="run-sources"
          label={sourceLabel + (latestSource ? " · " + latestSource : "")}
        >
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
        </ExclusiveRunDetails>
      ) : null}
      {activity.models.length > 0 ? (
        <ModelSummary models={activity.models} className="run-models" />
      ) : null}
      {activity.tools.length > 0 ? (
        <ToolSummary tools={activity.tools} className="run-tools" />
      ) : null}
    </div>
  );
}

function ModelSummary({
  models,
  className,
}: {
  models: readonly MessageModelUsage[];
  className: string;
}) {
  const label = models.length === 1 ? "1 model" : `${models.length} models`;
  return (
    <ExclusiveRunDetails className={className} label={label}>
      <ul>
        {models.map(({ model, calls }) => (
          <li key={model}>
            <span>{model}</span>
            <small>{calls === 1 ? "1 call" : `${calls} calls`}</small>
          </li>
        ))}
      </ul>
    </ExclusiveRunDetails>
  );
}

function ToolSummary({
  tools,
  className,
}: {
  tools: readonly Pick<RunTool, "name" | "status">[];
  className: string;
}) {
  const groups = summarizeTools(tools);
  const label = tools.length === 1 ? "1 tool call" : `${tools.length} tool calls`;
  return (
    <ExclusiveRunDetails className={className} label={label}>
      <ul>
        {groups.map(({ name, count, status }) => (
          <li key={name}>
            <span>{name}</span>
            <small>{count > 1 ? ` × ${count}` : ""} · {toolStatusLabel(status)}</small>
          </li>
        ))}
      </ul>
    </ExclusiveRunDetails>
  );
}

const RUN_DETAILS_OPEN_EVENT = "audrey:run-details-open";

function ExclusiveRunDetails({
  className,
  label,
  children,
}: {
  className: string;
  label: string;
  children: ReactNode;
}) {
  const [open, setOpen] = useState(false);
  const detailsRef = useRef<HTMLDetailsElement>(null);

  useEffect(() => {
    const closeForPeer = (event: Event) => {
      if ((event as CustomEvent<HTMLDetailsElement>).detail !== detailsRef.current) {
        setOpen(false);
      }
    };
    document.addEventListener(RUN_DETAILS_OPEN_EVENT, closeForPeer);
    return () => document.removeEventListener(RUN_DETAILS_OPEN_EVENT, closeForPeer);
  }, []);

  useEffect(() => {
    if (!open || !detailsRef.current) return;
    document.dispatchEvent(new CustomEvent(RUN_DETAILS_OPEN_EVENT, {
      detail: detailsRef.current,
    }));
    const closeOnOutsideClick = (event: MouseEvent) => {
      const target = event.target;
      if (target instanceof Node && !detailsRef.current?.contains(target)) setOpen(false);
    };
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") setOpen(false);
    };
    document.addEventListener("click", closeOnOutsideClick);
    document.addEventListener("keydown", closeOnEscape);
    return () => {
      document.removeEventListener("click", closeOnOutsideClick);
      document.removeEventListener("keydown", closeOnEscape);
    };
  }, [open]);

  return (
    <details
      ref={detailsRef}
      className={className}
      open={open}
      onToggle={(event) => setOpen(event.currentTarget.open)}
    >
      <summary>{label}</summary>
      {children}
    </details>
  );
}

function summarizeTools(tools: readonly Pick<RunTool, "name" | "status">[]) {
  const groups = new Map<string, { name: string; count: number; statuses: RunTool["status"][] }>();
  for (const tool of tools) {
    const group = groups.get(tool.name);
    if (group) {
      group.count += 1;
      group.statuses.push(tool.status);
    } else {
      groups.set(tool.name, { name: tool.name, count: 1, statuses: [tool.status] });
    }
  }
  return [...groups.values()].map(({ name, count, statuses }) => ({
    name,
    count,
    status: statuses.includes("failed")
      ? "failed" as const
      : statuses.includes("running")
        ? "running" as const
        : statuses.includes("incomplete")
          ? "incomplete" as const
          : "succeeded" as const,
  }));
}

function toolStatusLabel(status: RunTool["status"]): string {
  return status === "succeeded" ? "complete" : status;
}

function toolResultFailed(content: string): boolean {
  try {
    const result = recordOf(JSON.parse(content));
    return stringOf(result.status) === "failed" || stringOf(result.error) === "tool_failed";
  } catch {
    return false;
  }
}

function settleRunningTools(tools: RunTool[]): RunTool[] {
  return tools.map((tool) => tool.status === "running"
    ? { ...tool, status: "incomplete" }
    : tool);
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
      <MessagePrimitive.Attachments>
        {({ attachment }) => <ChatAttachment attachment={attachment} />}
      </MessagePrimitive.Attachments>
    </MessagePrimitive.Root>
  );
}

function ChatAttachment({ attachment }: { attachment: CompleteAttachment }) {
  const [imageFailed, setImageFailed] = useState(false);
  if (attachment.type === "image") {
    return (
      <figure className="chat-attachment chat-image-attachment">
        {imageFailed ? (
          <div
            className="chat-image-unavailable"
            role="img"
            aria-label={`${attachment.name} preview unavailable`}
          >
            <span aria-hidden="true">◫</span>
            <small>Preview unavailable</small>
          </div>
        ) : (
          <img
            src={getFileImageUrl(attachment.id)}
            alt={attachment.name}
            loading="lazy"
            onError={() => setImageFailed(true)}
          />
        )}
        <figcaption>{attachment.name}</figcaption>
      </figure>
    );
  }
  const video = attachment.contentType?.startsWith("video/") ?? false;
  return (
    <div className="chat-attachment chat-file-attachment" aria-label={`Attached file ${attachment.name}`}>
      <span className="chat-file-icon" aria-hidden="true">{video ? "▶" : "▤"}</span>
      <span>
        <strong>{attachment.name}</strong>
        <small>{video ? "Video" : "File"}</small>
      </span>
    </div>
  );
}

function AssistantMessage() {
  const messageId = useAuiState((state) => state.message.id);
  const sources = useContext(SavedSourcesContext).get(messageId) ?? [];
  const models = useContext(SavedModelsContext).get(messageId) ?? [];
  const tools = useContext(SavedToolsContext).get(messageId) ?? [];
  return (
    <MessagePrimitive.Root className="message message-assistant">
      <div className="message-label">Audrey</div>
      <MessagePrimitive.Parts
        components={{ Text: MarkdownText, tools: { Fallback: HiddenToolActivity } }}
      />
      {sources.length > 0 || models.length > 0 || tools.length > 0 ? (
        <div className="saved-run-details">
          {sources.length > 0 ? (
            <ExclusiveRunDetails
              className="saved-sources"
              label={sources.length === 1 ? "1 source found" : `${sources.length} sources found`}
            >
              <p>Observed during this run; the answer may cite a different set.</p>
              <ul>
                {sources.map(({ id, title, url }) => {
                  const safeUrl = safeSourceUrl(url);
                  return <li key={id}>{safeUrl ? (
                    <a href={safeUrl} target="_blank" rel="noreferrer noopener">{title || safeUrl}</a>
                  ) : (title || "Source")}</li>;
                })}
              </ul>
            </ExclusiveRunDetails>
          ) : null}
          {models.length > 0 ? (
            <ModelSummary models={models} className="saved-models" />
          ) : null}
          {tools.length > 0 ? <ToolSummary tools={tools} className="saved-tools" /> : null}
        </div>
      ) : null}
      <AnswerCopyAction />
    </MessagePrimitive.Root>
  );
}

function AnswerCopyAction() {
  const aui = useAui();
  const hasText = useAuiState((state) =>
    state.message.parts.some((part) => part.type === "text" && part.text.length > 0));
  const { status, copy } = useClipboardFeedback();
  if (!hasText) return null;

  const label = status === "copied"
    ? "Answer copied"
    : status === "failed"
      ? "Could not copy answer"
      : "Copy answer";
  return (
    <div className="message-actions" aria-label="Message actions">
      <button
        type="button"
        className="copy-action"
        aria-label={label}
        onClick={() => void copy(aui.message.getCopyText())}
      >
        {status === "copied" ? "Copied" : status === "failed" ? "Copy failed" : "Copy"}
      </button>
    </div>
  );
}

function skillModeForModel(model: AudreyModel): "auto" | "fast" | "deep" | null {
  if (model.kind === "direct" || model.mode === "direct") return null;
  if (model.mode === "fast") return "fast";
  if (["deep", "research", "local", "cloud"].includes(model.mode)) return "deep";
  return "auto";
}

type ComposerPickerName = "files" | "skills" | "model" | "other-models" | "project";

const ComposerPickerContext = createContext<{
  activePicker: ComposerPickerName | null;
  setActivePicker: (picker: ComposerPickerName | null) => void;
}>({ activePicker: null, setActivePicker: () => {} });

// Message options float above their own controls. Portaling prevents a
// long menu from resizing the dock or being clipped by the scrolling thread.
function ComposerPickerPanel({
  className,
  label,
  role = "dialog",
  triggerRef,
  children,
}: {
  className: string;
  label: string;
  role?: "dialog" | "region" | "menu";
  triggerRef: RefObject<HTMLButtonElement | null>;
  children: ReactNode;
}) {
  const { activePicker, setActivePicker } = useContext(ComposerPickerContext);
  const panelRef = useRef<HTMLElement>(null);

  useLayoutEffect(() => {
    const trigger = triggerRef.current;
    const panel = panelRef.current;
    const composer = trigger?.closest<HTMLElement>(".composer");
    if (!trigger || !composer || !panel) return;
    const viewport = composer.closest<HTMLElement>(".thread-viewport");
    const visualViewport = window.visualViewport;
    const place = () => {
      const box = composer.getBoundingClientRect();
      const control = trigger.getBoundingClientRect();
      const thread = viewport?.getBoundingClientRect();
      const viewLeft = visualViewport?.offsetLeft ?? 0;
      const viewTop = visualViewport?.offsetTop ?? 0;
      const viewWidth = visualViewport?.width ?? window.innerWidth;
      const viewHeight = visualViewport?.height ?? window.innerHeight;
      const topLimit = Math.max(viewTop + 8, (thread?.top ?? viewTop) + 8);
      const bottom = Math.min(control.top - 8, viewTop + viewHeight - 8);
      const leftLimit = Math.max(viewLeft + 8, (thread?.left ?? viewLeft) + 8);
      const rightLimit = Math.min(viewLeft + viewWidth - 8, (thread?.right ?? viewLeft + viewWidth) - 8);
      const width = Math.max(0, Math.min(576, box.width, rightLimit - leftLimit));
      const alignedLeft = activePicker === "model" || activePicker === "other-models"
        ? control.left
        : activePicker === "skills" ? control.right - width
        : control.left + (control.width - width) / 2;
      const left = Math.max(leftLimit, Math.min(alignedLeft, rightLimit - width));
      panel.style.left = `${left}px`;
      panel.style.bottom = `${window.innerHeight - bottom}px`;
      panel.style.width = `${width}px`;
      panel.style.maxHeight = `${Math.max(0, Math.min(352, bottom - topLimit))}px`;
      panel.style.visibility = "visible";
    };
    place();
    const observer = new ResizeObserver(place);
    observer.observe(composer);
    observer.observe(trigger);
    if (composer.parentElement) observer.observe(composer.parentElement);
    if (viewport) observer.observe(viewport);
    window.addEventListener("resize", place);
    window.addEventListener("scroll", place, true);
    visualViewport?.addEventListener("resize", place);
    visualViewport?.addEventListener("scroll", place);
    const firstChoice = panel.querySelector<HTMLButtonElement>(
      '.composer-choice-options button[aria-pressed="true"]:not(:disabled), .composer-choice-options button[aria-checked="true"]:not(:disabled), .skill-options button[aria-pressed="true"]:not(:disabled)',
    ) ?? panel.querySelector<HTMLButtonElement>(
      '.composer-choice-options button:not(:disabled), .skill-options button:not(:disabled)',
    ) ?? panel.querySelector<HTMLButtonElement>('button:not(:disabled)');
    firstChoice?.focus({ preventScroll: true });
    if (firstChoice) {
      const choiceBox = firstChoice.getBoundingClientRect();
      const panelBox = panel.getBoundingClientRect();
      if (choiceBox.bottom > panelBox.bottom) panel.scrollTop += choiceBox.bottom - panelBox.bottom + 12;
    }
    return () => {
      observer.disconnect();
      window.removeEventListener("resize", place);
      window.removeEventListener("scroll", place, true);
      visualViewport?.removeEventListener("resize", place);
      visualViewport?.removeEventListener("scroll", place);
    };
  }, [activePicker, triggerRef]);

  useEffect(() => {
    const outside = (event: Event) => {
      const target = event.target;
      if (!(target instanceof Node)) return;
      if (panelRef.current?.contains(target) || triggerRef.current?.contains(target)) return;
      setActivePicker(null);
    };
    const escape = (event: KeyboardEvent) => {
      if (event.key !== "Escape") return;
      event.preventDefault();
      event.stopPropagation();
      setActivePicker(null);
      triggerRef.current?.focus({ preventScroll: true });
    };
    document.addEventListener("pointerdown", outside);
    document.addEventListener("focusin", outside);
    document.addEventListener("keydown", escape);
    return () => {
      document.removeEventListener("pointerdown", outside);
      document.removeEventListener("focusin", outside);
      document.removeEventListener("keydown", escape);
    };
  }, [setActivePicker, triggerRef]);

  return createPortal(
    <section
      ref={panelRef}
      className={`composer-picker ${className}`}
      role={role}
      aria-label={label}
      onKeyDown={(event) => {
        if (!(event.target instanceof HTMLButtonElement)) return;
        if (!["ArrowDown", "ArrowUp", "Home", "End"].includes(event.key)) return;
        const choices = Array.from(event.currentTarget.querySelectorAll<HTMLButtonElement>(
          '.composer-choice-options button:not(:disabled), .skill-options button:not(:disabled)',
        ));
        if (!choices.length) return;
        event.preventDefault();
        const current = choices.indexOf(event.target);
        const index = event.key === "Home" ? 0 : event.key === "End" ? choices.length - 1
          : current < 0 ? (event.key === "ArrowUp" ? choices.length - 1 : 0)
          : (current + (event.key === "ArrowUp" ? -1 : 1) + choices.length) % choices.length;
        choices[index].focus({ preventScroll: true });
        choices[index].scrollIntoView({ block: "nearest" });
      }}
    >{children}</section>,
    document.body,
  );
}

function ComposerPickerClose({
  label,
  triggerRef,
}: {
  label: string;
  triggerRef: RefObject<HTMLButtonElement | null>;
}) {
  const { setActivePicker } = useContext(ComposerPickerContext);
  return (
    <button
      type="button"
      className="attachment-picker-close"
      aria-label={label}
      onClick={() => {
        setActivePicker(null);
        triggerRef.current?.focus({ preventScroll: true });
      }}
    >
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <path d="m6.5 9 5.5 5.5L17.5 9" />
      </svg>
    </button>
  );
}

function ComposerProjectPicker({
  projects,
  projectId,
  projectName,
  disabled,
  disabledReason,
  onChange,
}: {
  projects: AudreyProject[];
  projectId: string | null;
  projectName?: string;
  disabled: boolean;
  disabledReason?: string;
  onChange: (id: string | null) => Promise<void>;
}) {
  const { activePicker, setActivePicker } = useContext(ComposerPickerContext);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const open = activePicker === "project" && !disabled;
  const choices = [
    { id: "", name: "No project" },
    ...(projectId && !projects.some(({ id }) => id === projectId)
      ? [{ id: projectId, name: projectName ?? "Current project" }] : []),
    ...projects,
  ];
  return (
    <>
      <button
        ref={triggerRef}
        type="button"
        className="composer-rail-control composer-project-picker"
        aria-label="Move conversation to project"
        title={disabledReason}
        aria-haspopup="dialog"
        aria-expanded={open}
        data-value={projectId ?? ""}
        disabled={disabled}
        onClick={() => setActivePicker(open ? null : "project")}
      >
        <span className="composer-control-label">Project</span>
        <strong>{projectName ?? (projectId ? "Current project" : "No project")}</strong>
      </button>
      {open ? (
        <ComposerPickerPanel className="project-picker" label="Choose a project" triggerRef={triggerRef}>
          <header>
            <div><strong>Choose a project</strong><span>Group this chat with a project's files and instructions.</span></div>
            <ComposerPickerClose label="Hide project picker" triggerRef={triggerRef} />
          </header>
          <div className="composer-choice-options">
            {choices.map((project) => (
              <button
                key={project.id}
                type="button"
                aria-label={project.name}
                aria-pressed={(projectId ?? "") === project.id}
                onClick={() => {
                  setActivePicker(null);
                  triggerRef.current?.focus({ preventScroll: true });
                  void onChange(project.id || null);
                }}
              ><strong>{project.name}</strong></button>
            ))}
          </div>
        </ComposerPickerPanel>
      ) : null}
    </>
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
  const { activePicker, setActivePicker } = useContext(ComposerPickerContext);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const hasPrivate = models.some(({ id }) => id === "private");
  const workflowModels = models.filter(({ kind, id }) => kind === "workflow"
    && (id !== "local" || !hasPrivate || modelId === "local"));
  const directModels = canBrowseDirectModels ? models.filter(({ kind }) => kind === "direct") : [];
  const open = !disabled && ["model", "other-models"].includes(activePicker ?? "");
  const direct = activePicker === "other-models";
  if (!compact) {
    return (
      <div className="composer-model-picker">
        <img src={selected.portrait} alt="" aria-hidden="true" />
        <span className="model-description" aria-live="polite">{selected.description}</span>
      </div>
    );
  }
  const choose = (id: string) => {
    setActivePicker(null);
    triggerRef.current?.focus({ preventScroll: true });
    void onChange(id);
  };
  return (
    <>
      <button
        ref={triggerRef}
        className="composer-rail-control compact-model-picker"
        type="button"
        aria-label="Audrey model"
        aria-haspopup="dialog"
        aria-expanded={open}
        data-value={modelId}
        title={`${selected.label}: ${selected.description}`}
        disabled={disabled}
        onClick={() => setActivePicker(open ? null : "model")}
      >
        <span className="composer-control-label">Model</span>
        <strong>{selected.kind === "direct" ? selected.id.slice("direct/".length) : selected.label}</strong>
      </button>
      {open ? (
        <ComposerPickerPanel
          className={direct ? "direct-model-menu" : "model-picker"}
          label={direct ? "Other models" : "Choose an Audrey model"}
          role={direct ? "menu" : "dialog"}
          triggerRef={triggerRef}
        >
          <header>
            <div>
              <strong>{direct ? "Other models" : "Choose an Audrey model"}</strong>
              <span>{direct ? "Use an available model directly." : "Choose how Audrey approaches your question."}</span>
            </div>
            <ComposerPickerClose label={direct ? "Close other models" : "Hide model picker"} triggerRef={triggerRef} />
          </header>
          <div className="composer-choice-options">
            {(direct ? directModels : workflowModels).map((item) => (
              <button
                key={item.id}
                type="button"
                role={direct ? "menuitemradio" : undefined}
                aria-label={direct ? item.id.slice("direct/".length) : item.label}
                aria-pressed={direct ? undefined : item.id === modelId}
                aria-checked={direct ? item.id === modelId : undefined}
                onClick={() => choose(item.id)}
              >
                <strong>{direct ? item.id.slice("direct/".length) : item.label}</strong>
                {!direct ? <span>{item.description}</span> : null}
              </button>
            ))}
          </div>
          {direct || directModels.length > 0 ? (
            <button
              type="button"
              className="composer-picker-browse"
              onClick={() => setActivePicker(direct ? "model" : "other-models")}
            >{direct ? "Back to Audrey models" : "Other models..."}</button>
          ) : null}
        </ComposerPickerPanel>
      ) : null}
    </>
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
          pre: ({ children }) => <MarkdownCodeBlock>{children}</MarkdownCodeBlock>,
        }}
      >
        {text}
      </Markdown>
    </div>
  );
}

function MarkdownCodeBlock({ children }: { children: ReactNode }) {
  const { status, copy } = useClipboardFeedback();
  const language = codeLanguage(children);
  const code = reactNodeText(children).replace(/\n$/, "");
  const codeDescription = language ? language + " code" : "code";
  const label = status === "copied"
    ? (language ? language + " " : "") + "code copied"
    : status === "failed"
      ? "Could not copy " + codeDescription
      : "Copy " + codeDescription;
  return (
    <div className="markdown-code-block">
      <div className="code-block-header">
        <span>{language ?? "Code"}</span>
        <button
          type="button"
          className="copy-action"
          aria-label={label}
          onClick={() => void copy(code)}
        >
          {status === "copied" ? "Copied" : status === "failed" ? "Copy failed" : "Copy"}
        </button>
      </div>
      <pre>{children}</pre>
    </div>
  );
}

function reactNodeText(node: ReactNode): string {
  if (typeof node === "string" || typeof node === "number") return String(node);
  if (Array.isArray(node)) return node.map(reactNodeText).join("");
  if (isValidElement<{ children?: ReactNode }>(node)) return reactNodeText(node.props.children);
  return "";
}

function codeLanguage(node: ReactNode): string | null {
  if (Array.isArray(node)) {
    for (const child of node) {
      const language = codeLanguage(child);
      if (language) return language;
    }
    return null;
  }
  if (!isValidElement<{ children?: ReactNode; className?: string }>(node)) return null;
  const languageClass = node.props.className
    ?.split(/\s+/)
    .find((className) => className.startsWith("language-"));
  return languageClass?.slice("language-".length) || codeLanguage(node.props.children);
}

function useClipboardFeedback() {
  const [status, setStatus] = useState<"idle" | "copied" | "failed">("idle");
  const resetTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => () => {
    if (resetTimer.current) clearTimeout(resetTimer.current);
  }, []);

  const copy = async (text: string) => {
    if (resetTimer.current) clearTimeout(resetTimer.current);
    try {
      if (!navigator.clipboard) throw new Error("Clipboard access is unavailable.");
      await navigator.clipboard.writeText(text);
      setStatus("copied");
    } catch {
      setStatus("failed");
    }
    resetTimer.current = setTimeout(() => setStatus("idle"), 2_000);
  };

  return { status, copy };
}

function HiddenToolActivity() {
  return null;
}

function toThreadMessages(messages: ConversationMessage[]): ThreadMessageLike[] {
  return messages.flatMap<ThreadMessageLike>((message) => {
    if (message.role === "user") {
      return [{
        id: message.id,
        role: "user",
        content: message.content,
        attachments: (message.attachments ?? []).map(toCompleteAttachment),
      }];
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

function toCompleteAttachment(file: MessageAttachment | AudreyFile): CompleteAttachment {
  return {
    id: file.id,
    type: attachmentPresentationType(file),
    name: file.filename,
    contentType: file.mime,
    content: [],
    status: { type: "complete" },
  };
}

function toCreateAttachment(file: MessageAttachment | AudreyFile): CreateAttachment {
  return {
    id: file.id,
    type: attachmentPresentationType(file),
    name: file.filename,
    contentType: file.mime,
    // The server loads file bytes by authenticated attachment id. The runtime
    // carries only display metadata and the minimized transport strips it.
    content: [],
  };
}

function attachmentPresentationType(file: MessageAttachment | AudreyFile) {
  return file.kind === "image"
    ? "image" as const
    : ["video", "audio"].includes(file.kind) ? "file" as const : "document" as const;
}
function attachmentKindLabel(kind: AudreyFile["kind"]): string {
  if (kind === "text") return "Document";
  return kind[0].toUpperCase() + kind.slice(1);
}

function attachmentAllowed(file: AudreyFile, privateMode: boolean): boolean {
  return privateMode ? file.private_required === true && file.kind === "text" : !file.private_required;
}

function attachmentKindSymbol(kind: AudreyFile["kind"]): string {
  return ATTACHMENT_FOLDERS.find((folder) => folder.kind === kind)?.symbol ?? "≡";
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

function modelLabel(models: AudreyModel[], conversation: Conversation): string {
  if (conversation.private_required || conversation.default_model_id === "private") return "Private";
  if (conversation.default_model_id === "local") return "Local";
  return models.find(({ id }) => id === conversation.default_model_id)?.label
    ?? models[0]?.label
    ?? conversation.default_mode;
}

function modelDetails(models: AudreyModel[], modelId: string) {
  const model = models.find(({ id }) => id === modelId) ?? models[0];
  return {
    ...model,
    portrait: model.id === "private" ? privatePortrait
      : model.portrait_url || MODEL_PORTRAITS[model.presentation] || autoPortrait,
  };
}

function actionError(action: string, reason: unknown): string {
  const status = reason instanceof ApiError ? ` (HTTP ${reason.status})` : "";
  return `${action}${status}: ${messageOf(reason)}`;
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
