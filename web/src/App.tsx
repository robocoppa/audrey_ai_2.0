import { lazy, Suspense, useEffect, useState, type FormEvent } from "react";

import builtryteWordmark from "./assets/brand/builtryte-wordmark.png";
import {
  ApiError,
  getCurrentUser,
  updateCurrentUserDisplayName,
  type CurrentUser,
} from "./api";

const ChatWorkspace = lazy(() =>
  import("./ChatWorkspace").then((module) => ({ default: module.ChatWorkspace })),
);

type SessionState =
  | { status: "loading" }
  | { status: "ready"; user: CurrentUser }
  | { status: "unauthenticated" }
  | { status: "error"; message: string };

export function App() {
  const [session, setSession] = useState<SessionState>({ status: "loading" });

  useEffect(() => {
    let active = true;

    getCurrentUser()
      .then((user) => {
        if (active) setSession({ status: "ready", user });
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

  return (
    <div className="app-shell">
      <header className="topbar">
        <a className="brand" href="/" aria-label="Audrey home">
          <span className="brand-wordmark" aria-hidden="true">
            <img src={builtryteWordmark} alt="" />
          </span>
          <span className="brand-product">Audrey</span>
        </a>
        <SessionControls
          session={session}
          onUserChange={(user) => setSession({ status: "ready", user })}
        />
      </header>

      {session.status === "ready" ? (
        <main className="native-main">
          <Suspense fallback={<div className="thread-loading" role="status">Loading workspace…</div>}>
            <ChatWorkspace user={session.user} />
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
}: {
  session: SessionState;
  onUserChange: (user: CurrentUser) => void;
}) {
  if (session.status === "loading") {
    return <span className="session-badge">Checking session…</span>;
  }
  if (session.status === "ready") {
    return <ReadySessionControls user={session.user} onUserChange={onUserChange} />;
  }
  return <span className="session-badge session-badge-offline">Not connected</span>;
}

function ReadySessionControls({
  user,
  onUserChange,
}: {
  user: CurrentUser;
  onUserChange: (user: CurrentUser) => void;
}) {
  const [editing, setEditing] = useState(false);
  const [displayName, setDisplayName] = useState(user.display_name);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");

  async function saveProfile(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setSaving(true);
    setError("");
    try {
      const updated = await updateCurrentUserDisplayName(displayName);
      onUserChange(updated);
      setDisplayName(updated.display_name);
      setEditing(false);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Profile name could not be saved.");
    } finally {
      setSaving(false);
    }
  }

  function cancelEditing() {
    setDisplayName(user.display_name);
    setError("");
    setEditing(false);
  }

  return (
    <div className="session-controls" aria-label="Signed in user">
      <div className="profile-editor">
        {editing ? (
          <form className="profile-name-form" aria-label="Edit profile name" onSubmit={saveProfile}>
            <input
              aria-label="Profile name"
              autoFocus
              maxLength={100}
              value={displayName}
              onChange={(event) => setDisplayName(event.target.value)}
              placeholder="Your name"
              disabled={saving}
            />
            <button type="submit" disabled={saving || !displayName.trim()}>
              {saving ? "Saving…" : "Save"}
            </button>
            <button type="button" onClick={cancelEditing} disabled={saving}>
              Cancel
            </button>
          </form>
        ) : (
          <button
            className="session-name"
            type="button"
            aria-label="Edit profile name"
            title={`${user.display_name || user.email} · Edit profile name`}
            onClick={() => setEditing(true)}
          >
            {firstName(user)}
          </button>
        )}
        {error ? <span className="profile-name-error" role="alert">{error}</span> : null}
      </div>
      {user.auth_provider === "cloudflare_access" ? (
        <a className="logout-button" href="/cdn-cgi/access/logout">Log out</a>
      ) : null}
    </div>
  );
}

function firstName(user: CurrentUser): string {
  const displayName = user.display_name.trim();
  if (displayName) return displayName.split(/\s+/u)[0];
  return user.email.split("@", 1)[0] || "Account";
}
