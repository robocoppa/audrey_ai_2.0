import { lazy, Suspense, useEffect, useState } from "react";

import { AudreyLoader } from "./AudreyLoader";
import builtryteWordmark from "./assets/brand/builtryte-wordmark.png";
import { AccountSettings } from "./AccountSettings";
import {
  ApiError,
  getCurrentUser,
  getCurrentUserPreferences,
  type CurrentUser,
  type UserPreferences,
} from "./api";

const ChatWorkspace = lazy(() =>
  import("./ChatWorkspace").then((module) => ({ default: module.ChatWorkspace })),
);

type SessionState =
  | { status: "loading" }
  | { status: "ready"; user: CurrentUser; preferences: UserPreferences }
  | { status: "unauthenticated" }
  | { status: "error"; message: string };

const ACCESS_BOOTSTRAP_DELAYS_MS = [0, 500, 1_000, 2_000, 4_000, 7_500, 15_000];

function hasCloudflareAccessMessage() {
  return new URLSearchParams(window.location.search).has("__cf_access_message");
}

function clearCloudflareAccessMessage() {
  const search = new URLSearchParams(window.location.search);
  if (!search.has("__cf_access_message")) return;
  search.delete("__cf_access_message");
  const query = search.toString();
  window.history.replaceState(
    window.history.state,
    "",
    `${window.location.pathname}${query ? `?${query}` : ""}${window.location.hash}`,
  );
}

export function App() {
  const [session, setSession] = useState<SessionState>({ status: "loading" });
  const [workspaceRevision, setWorkspaceRevision] = useState(0);
  const [sessionRevision, setSessionRevision] = useState(0);
  const [showAccessHandoff, setShowAccessHandoff] = useState(hasCloudflareAccessMessage);

  useEffect(() => {
    let active = true;

    async function initializeSession() {
      let lastError: unknown;
      for (const delayMs of ACCESS_BOOTSTRAP_DELAYS_MS) {
        if (delayMs) {
          await new Promise((resolve) => window.setTimeout(resolve, delayMs));
        }
        if (!active) return;
        try {
          const user = await getCurrentUser();
          const preferences = await getCurrentUserPreferences();
          if (active) {
            clearCloudflareAccessMessage();
            setSession({ status: "ready", user, preferences });
          }
          return;
        } catch (error) {
          lastError = error;
          const retryable = error instanceof ApiError
            && (error.status === 401 || error.status === 403);
          if (!retryable) break;
        }
      }

      if (!active) return;
      if (lastError instanceof ApiError
        && (lastError.status === 401 || lastError.status === 403)) {
        setSession({ status: "unauthenticated" });
        return;
      }
      setSession({
        status: "error",
        message: lastError instanceof Error
          ? lastError.message
          : "Audrey is unavailable.",
      });
    }

    void initializeSession();

    return () => {
      active = false;
    };
  }, [sessionRevision]);

  const retrySession = () => {
    setShowAccessHandoff(true);
    setSession({ status: "loading" });
    setSessionRevision((current) => current + 1);
  };

  if (session.status === "loading") {
    if (showAccessHandoff) {
      return (
        <AudreyLoader
          fullscreen
          label="Finishing secure sign-in"
          message="Finishing your secure sign-in"
          detail="Cloudflare Access is confirming this browser. Audrey will open automatically."
        />
      );
    }
    return <AudreyLoader fullscreen />;
  }

  if (session.status !== "ready") {
    return <SessionTimeout session={session} onRetry={retrySession} />;
  }

  return (
    <div className="app-shell app-shell-ready">
      <svg className="brand-filter" aria-hidden="true">
        <filter id="light-wordmark-on-dark" colorInterpolationFilters="sRGB">
          <feColorMatrix
            type="matrix"
            values="0 0 0 0 0.843  0 0 0 0 0.886  0 0 0 0 0.945  -1.5 -1.5 -1.5 0 4.45"
          />
        </filter>
      </svg>
      <header className="topbar">
        <a className="brand" href="/" aria-label="Audrey home">
          <span className="brand-wordmark" aria-hidden="true">
            <img src={builtryteWordmark} alt="" />
          </span>
          <span className="brand-product">Ask Audrey</span>
        </a>
        <ReadySessionControls
          user={session.user}
          preferences={session.preferences}
          onUserChange={(user) => {
            setSession((current) =>
              current.status === "ready" ? { ...current, user } : current,
            );
          }}
          onPreferencesChange={(preferences) => {
            setSession((current) =>
              current.status === "ready" ? { ...current, preferences } : current,
            );
          }}
          onDataPurgeAttempted={() => {
            setWorkspaceRevision((current) => current + 1);
            void Promise.all([getCurrentUser(), getCurrentUserPreferences()])
              .then(([user, preferences]) => {
                setSession({ status: "ready", user, preferences });
              })
              .catch(() => undefined);
          }}
        />
      </header>

      <main className="native-main">
        <Suspense fallback={<AudreyLoader fullscreen label="Loading Audrey workspace" />}>
          <ChatWorkspace
            key={workspaceRevision}
            user={session.user}
            preferences={session.preferences}
          />
        </Suspense>
      </main>
    </div>
  );
}

function SessionTimeout({
  session,
  onRetry,
}: {
  session:
    | { status: "unauthenticated" }
    | { status: "error"; message: string };
  onRetry: () => void;
}) {
  const timedOut = session.status === "unauthenticated";
  const title = timedOut
    ? "Sign-in is taking longer than expected"
    : "Audrey could not finish opening";
  const detail = timedOut
    ? "Cloudflare Access did not establish this browser session within 30 seconds."
    : `The session check ended early. ${session.message}`;

  return (
    <main className="session-timeout" aria-labelledby="session-timeout-title">
      <div className="session-timeout-body">
        <span className="session-timeout-mark" aria-hidden="true">
          <span />
        </span>
        <p className="session-timeout-kicker">Secure connection</p>
        <h1 id="session-timeout-title">{title}</h1>
        <p className="session-timeout-detail" role="alert">
          {detail}
        </p>
        <div className="session-timeout-actions">
          <button
            className="session-retry-button"
            type="button"
            onClick={onRetry}
          >
            Retry session
          </button>
          <a className="logout-button" href="/cdn-cgi/access/logout">
            Log out
          </a>
        </div>
      </div>
    </main>
  );
}

function ReadySessionControls({
  user,
  preferences,
  onUserChange,
  onPreferencesChange,
  onDataPurgeAttempted,
}: {
  user: CurrentUser;
  preferences: UserPreferences;
  onUserChange: (user: CurrentUser) => void;
  onPreferencesChange: (preferences: UserPreferences) => void;
  onDataPurgeAttempted: () => void;
}) {
  const [settingsOpen, setSettingsOpen] = useState(false);

  return (
    <div className="session-controls" aria-label="Signed in user">
      <button
        className="session-name"
        type="button"
        aria-label="Open account settings"
        title={`${user.display_name || user.email} · Account settings`}
        onClick={() => setSettingsOpen(true)}
      >
        {firstName(user)}
      </button>
      {user.auth_provider === "cloudflare_access" ? (
        <a className="logout-button" href="/cdn-cgi/access/logout">Log out</a>
      ) : null}
      {settingsOpen ? (
        <AccountSettings
          user={user}
          preferences={preferences}
          onUserChange={onUserChange}
          onPreferencesChange={onPreferencesChange}
          onDataPurgeAttempted={onDataPurgeAttempted}
          onClose={() => setSettingsOpen(false)}
        />
      ) : null}
    </div>
  );
}

function firstName(user: CurrentUser): string {
  const displayName = user.display_name.trim();
  if (displayName) return displayName.split(/\s+/u)[0];
  return user.email.split("@", 1)[0] || "Account";
}
