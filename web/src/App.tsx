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

export function App() {
  const [session, setSession] = useState<SessionState>({ status: "loading" });

  useEffect(() => {
    let active = true;

    Promise.all([getCurrentUser(), getCurrentUserPreferences()])
      .then(([user, preferences]) => {
        if (active) setSession({ status: "ready", user, preferences });
      })
      .catch((error: unknown) => {
        if (!active) return;
        if (error instanceof ApiError && (error.status === 401 || error.status === 403)) {
          setSession({ status: "unauthenticated" });
          return;
        }
        setSession({
          status: "error",
          message: error instanceof Error ? error.message : "Audrey is unavailable.",
        });
      });

    return () => {
      active = false;
    };
  }, []);

  if (session.status === "loading") {
    return <AudreyLoader fullscreen />;
  }

  return (
    <div
      className={session.status === "ready" ? "app-shell app-shell-ready" : "app-shell"}
    >
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
        <SessionControls
          session={session}
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
        />
      </header>

      {session.status === "ready" ? (
        <main className="native-main">
          <Suspense fallback={<AudreyLoader fullscreen label="Loading Audrey workspace" />}>
            <ChatWorkspace user={session.user} preferences={session.preferences} />
          </Suspense>
        </main>
      ) : (
        <main className="welcome" aria-labelledby="welcome-title">
          <div className="eyebrow">Private intelligence, on your terms</div>
          <h1 id="welcome-title">A quieter place to think.</h1>
          <p>
            This is Audrey's first native application surface. Conversations,
            runs, tools, and files will live here without making another chat UI
            the system of record.
          </p>
          {session.status === "unauthenticated" ? (
            <p className="notice" role="alert">
              Sign in through the Audrey access page, then reload this tab.
            </p>
          ) : null}
          {session.status === "error" ? (
            <p className="notice notice-error" role="alert">
              Audrey could not load your session. {session.message}
            </p>
          ) : null}
        </main>
      )}

      {session.status !== "ready" ? (
        <footer className="footer">Native preview · Open WebUI remains available</footer>
      ) : null}
    </div>
  );
}

function SessionControls({
  session,
  onUserChange,
  onPreferencesChange,
}: {
  session: SessionState;
  onUserChange: (user: CurrentUser) => void;
  onPreferencesChange: (preferences: UserPreferences) => void;
}) {
  if (session.status === "loading") {
    return null;
  }
  if (session.status === "ready") {
    return (
      <ReadySessionControls
        user={session.user}
        preferences={session.preferences}
        onUserChange={onUserChange}
        onPreferencesChange={onPreferencesChange}
      />
    );
  }
  return <span className="session-badge session-badge-offline">Not connected</span>;
}

function ReadySessionControls({
  user,
  preferences,
  onUserChange,
  onPreferencesChange,
}: {
  user: CurrentUser;
  preferences: UserPreferences;
  onUserChange: (user: CurrentUser) => void;
  onPreferencesChange: (preferences: UserPreferences) => void;
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
