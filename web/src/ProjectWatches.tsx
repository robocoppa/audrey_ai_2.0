import { useEffect, useRef, useState } from "react";
import { listWatches, type AudreyProject, type PublicWatch } from "./api";
import { SidebarItemButton } from "./SidebarItemButton";
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
    <header>
      <h2 id="project-watches-title">Project watches</h2>
      <div className="project-watch-actions">
        <button type="button" className="project-watch-refresh" aria-label="Refresh Project watches" disabled={loading} onClick={() => setRefresh((value) => value + 1)}>Refresh</button>
        {onNew ? <button className="project-primary-button" type="button" aria-label="New watch in Project" disabled={opening} onClick={() => void follow()}>New watch</button> : null}
      </div>
    </header>
    {error ? <p className="project-error" role="alert">{error}</p> : null}
    {loading ? <p className="project-empty" role="status">Loading Project watches…</p> : items.length ? <ul className="project-home-conversations">{items.map((watch) => <li key={watch.id} className="conversation-row">
      <SidebarItemButton className="project-watch-card" disabled={opening || !onOpen} onClick={() => void follow(watch)}>
        <span className="sidebar-item-title">{watch.title}</span>
        <small className="sidebar-item-meta project-watch-status">{watch.source_kind === "project_files" ? "Project files · " : ""}{watch.active_run_id ? "Checking" : watch.enabled ? "Scheduled" : "Paused"}</small>
      </SidebarItemButton>
    </li>)}</ul> : <p className="project-empty">No watches in this Project yet. Create one to follow a public page or files attached to this Project.</p>}
  </section>;
}

function messageOf(reason: unknown) {
  return reason instanceof Error ? reason.message : "The Project watches could not be loaded.";
}
