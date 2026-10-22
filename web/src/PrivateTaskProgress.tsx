import { useEffect, useRef, useState } from "react";

import { actOnPrivateTask, getPrivateTask, type PrivateTask } from "./privateTasksApi";
import "./PrivateTaskProgress.css";

function taskLabel(task: PrivateTask): string {
  switch (task.status) {
    case "queued": return "Review queued";
    case "running": return "Audrey is reviewing your saved information";
    case "waiting_for_user": return "Review saved · Your input is needed";
    case "completed": return "Review saved";
    case "cancelled": return "Review stopped · Saved work is retained";
    case "failed": return "Review could not finish · Saved work is retained";
    case "paused":
      if (task.stop_reason === "budget_exhausted") return "Review limit reached · Saved work is retained";
      if (task.stop_reason === "server_restart" || task.stop_reason === "lease_expired") return "Review paused after a restart · Saved work is retained";
      if (task.stop_reason === "inputs_changed") return "Review paused · Your information changed";
      if (task.stop_reason === "usage_unavailable") return "Review paused · Model usage could not be measured";
      return "Review paused · Saved work is retained";
  }
}

export function PrivateTaskProgress({
  conversationId, refreshVersion, disabled, readOnly = false, hasReview, onReview, onChanged, onTaskChange,
}: {
  conversationId: string;
  refreshVersion: number;
  disabled: boolean;
  readOnly?: boolean;
  hasReview: boolean;
  onReview: () => void;
  onChanged?: () => void;
  onTaskChange?: (task: PrivateTask | null) => void;
}) {
  const [snapshot, setSnapshot] = useState<{ conversationId: string; task: PrivateTask | null; loaded: boolean }>({
    conversationId, task: null, loaded: false,
  });
  const [loadError, setLoadError] = useState(false);
  const [actionError, setActionError] = useState(false);
  const [busy, setBusy] = useState(false);
  const [reloadVersion, setReloadVersion] = useState(0);
  const busyRef = useRef(false);
  const requestVersion = useRef(0);
  const requestRef = useRef<AbortController | null>(null);
  const scopeRef = useRef<object | null>(null);
  const onChangedRef = useRef(onChanged);
  const onTaskChangeRef = useRef(onTaskChange);
  useEffect(() => { onTaskChangeRef.current = onTaskChange; }, [onTaskChange]);
  const lastTaskRef = useRef<PrivateTask | null>(null);
  useEffect(() => { onChangedRef.current = onChanged; }, [onChanged]);

  useEffect(() => {
    const scope = {};
    scopeRef.current = scope;
    let timer: number | undefined;
    let controller: AbortController | undefined;
    async function poll() {
      if (scopeRef.current !== scope) return;
      if (!busyRef.current) {
        controller = new AbortController();
        requestRef.current = controller;
        const version = ++requestVersion.current;
        try {
          const { task } = await getPrivateTask(conversationId, controller.signal);
          if (controller.signal.aborted || scopeRef.current !== scope || version !== requestVersion.current) return;
          const previous = lastTaskRef.current;
          lastTaskRef.current = task;
          setSnapshot({ conversationId, task, loaded: true });
          setLoadError(false);
          if (!previous || previous.id !== task?.id || previous.revision !== task?.revision) onTaskChangeRef.current?.(task);
          if (previous && task && previous.id === task.id && previous.status !== task.status
            && !["queued", "running"].includes(task.status)) onChangedRef.current?.();
        } catch {
          if (!controller.signal.aborted && scopeRef.current === scope && version === requestVersion.current) {
            setLoadError(true);
          }
        } finally {
          if (scopeRef.current === scope) timer = window.setTimeout(() => void poll(), 2000);
        }
      } else {
        timer = window.setTimeout(() => void poll(), 2000);
      }
    }
    void poll();
    return () => {
      scopeRef.current = null;
      controller?.abort();
      requestRef.current?.abort();
      window.clearTimeout(timer);
      busyRef.current = false;
    };
  }, [conversationId, refreshVersion, reloadVersion]);

  const task = snapshot.conversationId === conversationId ? snapshot.task : null;
  const loading = !snapshot.loaded || snapshot.conversationId !== conversationId;
  const active = task?.status === "queued" || task?.status === "running";
  const paused = task?.status === "paused";
  const resumable = paused && task.stop_reason !== "budget_exhausted";

  async function act(action: "pause" | "cancel") {
    if (!task || busyRef.current || readOnly || loadError) return;
    const scope = scopeRef.current;
    const controller = new AbortController();
    requestRef.current?.abort();
    requestRef.current = controller;
    const version = ++requestVersion.current;
    busyRef.current = true;
    setBusy(true);
    setActionError(false);
    try {
      const { task: saved } = await actOnPrivateTask(task.id, task.revision, action, controller.signal);
      if (controller.signal.aborted || scopeRef.current !== scope || version !== requestVersion.current) return;
      lastTaskRef.current = saved;
      setSnapshot({ conversationId, task: saved, loaded: true });
      onTaskChangeRef.current?.(saved);
      onChangedRef.current?.();
    } catch {
      if (!controller.signal.aborted && scopeRef.current === scope) setActionError(true);
    } finally {
      if (scopeRef.current === scope) {
        busyRef.current = false;
        setBusy(false);
      }
    }
  }

  return <div className="private-task-progress" aria-label="Saved review progress">
    <div className="private-task-row">
      <span role="status" className="private-task-label">
        {active ? <span className="private-task-spinner" aria-hidden="true" /> : null}
        {loading ? "Loading saved review…" : task ? taskLabel(task) : "Check your saved information and supporting rules."}
      </span>
      {!readOnly && !loading && !loadError ? <div className="private-task-actions">
        {active ? <>
          <button type="button" disabled={busy} onClick={() => void act("pause")}>Pause</button>
          <button type="button" disabled={busy} onClick={() => void act("cancel")}>Cancel</button>
        </> : <>
          {(!paused || resumable) ? <button type="button" disabled={busy || disabled} onClick={onReview}>
            {resumable ? "Resume review" : task || hasReview ? "Continue review" : "Start review"}
          </button> : null}
          {paused || task?.status === "waiting_for_user" ? <button type="button" disabled={busy} onClick={() => void act("cancel")}>Cancel</button> : null}
        </>}
      </div> : null}
    </div>
    {loadError || actionError ? <div className="private-task-error" role="alert">
      <span>{loadError ? "Could not load saved review progress." : "Could not change this review. Reload its progress and try again."}</span>
      <button type="button" disabled={busy} onClick={() => { setActionError(false); setReloadVersion((value) => value + 1); }}>Reload progress</button>
    </div> : null}
  </div>;
}
