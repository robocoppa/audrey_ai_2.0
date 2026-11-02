import { useEffect, useRef, useState } from "react";
import { listWatches, type AudreyProject, type PublicWatch } from "./api";
import "./ProjectWatches.css";

interface ProjectWatchesProps {
  project: AudreyProject;
  visible: boolean;
  onNew?: (project: AudreyProject, signal?: AbortSignal) => void | Promise<void>;
  onOpen?: (watch: PublicWatch, signal?: AbortSignal) => void | Promise<void>;
}

export function ProjectWatches({ project, visible, onNew, onOpen }: ProjectWatchesProps) {
  const [items, setItems] = useState<PublicWatch[]>([]);
  const [loading, setLoading] = useState(true);
  const [opening, setOpening] = useState(false);
  const [error, setError] = useState("");
  const [refresh, setRefresh] = useState(0);
  const link = useRef<AbortController | null>(null);
  useEffect(() => {
    if (!visible) return;
    const controller = new AbortController();
    queueMicrotask(() => {
      if (!controller.signal.aborted) { setLoading(true); setOpening(false); setError(""); }
    });
    void listWatches(controller.signal, project.id).then(({ items }) => {
      if (!controller.signal.aborted) setItems(items.filter((watch) => watch.project?.id === project.id));
    }).catch((reason: unknown) => {
      if (!controller.signal.aborted) setError(messageOf(reason));
    }).finally(() => { if (!controller.signal.aborted) setLoading(false); });
    return () => { controller.abort(); link.current?.abort(); };
  }, [visible, project.id, refresh]);

  async function follow(watch?: PublicWatch) {
    if (!visible || project.private_required) return;
    if (watch && (watch.project?.id !== project.id || !watch.project.available)) {
      setError("This saved watch is unavailable in this Project. Refresh before trying again."); return;
    }
    link.current?.abort();
    const controller = new AbortController(); link.current = controller;
    setOpening(true); setError("");
    try {
      if (watch) await onOpen?.(watch, controller.signal);
      else await onNew?.(project, controller.signal);
    } catch (reason) {
      if (!controller.signal.aborted) setError(messageOf(reason));
    } finally {
      if (link.current === controller) { link.current = null; if (!controller.signal.aborted) setOpening(false); }
    }
  }

  return <section className="project-panel project-watches" aria-labelledby="project-watches-title">
    <header><div><span>Public-page monitoring</span><h2 id="project-watches-title">Project watches</h2></div>
      {onNew ? <button className="project-primary-button" type="button" disabled={opening} onClick={() => void follow()}>New watch in Project</button> : null}
    </header>
    {error ? <p className="project-error" role="alert">{error}</p> : null}
    {loading ? <p role="status">Loading Project watches…</p> : items.length ? <ul className="project-home-conversations">{items.map((watch) => <li key={watch.id} className="conversation-row">
      <button className="conversation project-home-conversation" type="button" disabled={opening || !onOpen} onClick={() => void follow(watch)}><span>{watch.title}</span><small>{watch.active_run_id ? "Checking" : watch.enabled ? "Scheduled" : "Paused"}</small></button>
    </li>)}</ul> : <p className="project-empty">No watches in this Project yet. Create one to monitor a public page for changes.</p>}
    <button type="button" disabled={loading} onClick={() => setRefresh((value) => value + 1)}>Refresh Project watches</button>
  </section>;
}

function messageOf(reason: unknown) {
  return reason instanceof Error ? reason.message : "The Project watches could not be loaded.";
}
