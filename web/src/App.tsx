import { lazy, Suspense, useEffect, useState } from "react";

import { AudreyLoader } from "./AudreyLoader";
import { AdminPanel } from "./AdminPanel";
import builtryteWordmark from "./assets/brand/builtryte-wordmark.png";
import { AccountSettings } from "./AccountSettings";
import {
  ApiError,
  getCurrentUser,
  getCurrentUserPreferences,
  listModels,
  type AudreyModel,
  type CurrentUser,
  type UserPreferences,
} from "./api";

const ChatWorkspace = lazy(() =>
  import("./ChatWorkspace").then((module) => ({ default: module.ChatWorkspace })),
);

type SessionState =
  | { status: "loading" }
  | {
      status: "ready";
      user: CurrentUser;
      preferences: UserPreferences;
      models: AudreyModel[];
    }
  | { status: "restricted"; user: CurrentUser }
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
          if (!active) return;
          clearCloudflareAccessMessage();
          setShowAccessHandoff(false);
          if (user.status !== "active") {
            if (active) setSession({ status: "restricted", user });
            return;
          }
          const [preferences, catalog] = await Promise.all([
            getCurrentUserPreferences(),
            listModels(),
          ]);
          if (active) {
            setSession({ status: "ready", user, preferences, models: catalog.items });
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

  if (session.status === "restricted") {
    return <AccountAccessState user={session.user} onRetry={retrySession} />;
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
            void Promise.all([getCurrentUser(), getCurrentUserPreferences(), listModels()])
              .then(([user, preferences, catalog]) => {
                setSession({ status: "ready", user, preferences, models: catalog.items });
              })
              .catch(() => undefined);
          }}
          onAdministrationChanged={() => {
            void listModels().then(({ items }) => {
              setSession((current) =>
                current.status === "ready" ? { ...current, models: items } : current,
              );
            });
          }}
        />
      </header>

      <main className="native-main">
        <Suspense fallback={<AudreyLoader fullscreen label="Loading Audrey workspace" />}>
          <ChatWorkspace
            key={workspaceRevision}
            user={session.user}
            preferences={session.preferences}
            models={session.models}
          />
        </Suspense>
      </main>
    </div>
  );
}

function AccountAccessState({
  user,
  onRetry,
}: {
  user: CurrentUser;
  onRetry: () => void;
}) {
  const pending = user.status === "pending";
  return (
    <main className="session-timeout" aria-labelledby="account-access-title">
      <div className="session-timeout-body">
        <span className="session-timeout-mark" aria-hidden="true"><span /></span>
        <p className="session-timeout-kicker">Audrey account</p>
        <h1 id="account-access-title">
          {pending ? "Approval is pending" : "This account is disabled"}
        </h1>
        <p className="session-timeout-detail" role="status">
          {pending
            ? "Your secure sign-in is complete. An Audrey administrator must approve this account before the workspace opens."
            : "Your secure sign-in is valid, but this Audrey account no longer has application access."}
        </p>
        <p className="account-access-email">{user.email}</p>
        <div className="session-timeout-actions">
          <button className="session-retry-button" type="button" onClick={onRetry}>
            Check again
          </button>
          <a className="logout-button" href="/cdn-cgi/access/logout">Log out</a>
        </div>
      </div>
    </main>
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
  onAdministrationChanged,
}: {
  user: CurrentUser;
  preferences: UserPreferences;
  onUserChange: (user: CurrentUser) => void;
  onPreferencesChange: (preferences: UserPreferences) => void;
  onDataPurgeAttempted: () => void;
  onAdministrationChanged: () => void;
}) {
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [adminOpen, setAdminOpen] = useState(false);

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
      {user.groups.includes("admins") ? (
        <button
          className="admin-button"
          type="button"
          onClick={() => setAdminOpen(true)}
        >
          Admin
        </button>
      ) : null}
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
      {adminOpen ? (
        <AdminPanel
          currentUserId={user.id}
          onChanged={onAdministrationChanged}
          onClose={() => setAdminOpen(false)}
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
