import { lazy, Suspense, useEffect, useState } from "react";

import { ApiError, getCurrentUser, type CurrentUser } from "./api";

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
        <a className="brand" href="/app/" aria-label="Audrey home">
          Audrey
        </a>
        <SessionBadge session={session} />
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

function SessionBadge({ session }: { session: SessionState }) {
  if (session.status === "loading") {
    return <span className="session-badge">Checking session…</span>;
  }
  if (session.status === "ready") {
    return <span className="session-badge session-badge-ready">Authenticated</span>;
  }
  return <span className="session-badge session-badge-offline">Not connected</span>;
}
