import { useEffect, useRef, useState } from "react";

import { listAgentTasks, type AgentTask, type AgentTaskStatus, type AudreyProject } from "./api";
import { SidebarItemButton } from "./SidebarItemButton";
import "./ProjectAgents.css";

const STATUS_LABELS: Record<AgentTaskStatus, string> = {
  queued: "Queued", running: "Running", paused: "Paused", completed: "Completed",
  failed: "Failed", cancelled: "Cancelled", waiting_for_input: "Waiting for input",
};

export function ProjectAgents({ project, visible, onNew, onOpen }: {
  project: AudreyProject;
  visible: boolean;
  onNew?: (project: AudreyProject, signal?: AbortSignal) => void | Promise<void>;
  onOpen?: (task: AgentTask, signal?: AbortSignal) => void | Promise<void>;
}) {
  const [items, setItems] = useState<AgentTask[]>([]);
  const [loading, setLoading] = useState(true);
  const [opening, setOpening] = useState(false);
  const [error, setError] = useState("");
  const [refresh, setRefresh] = useState(0);
  const link = useRef<AbortController | null>(null);
  useEffect(() => {
    if (!visible) return;
    const controller = new AbortController();
    queueMicrotask(() => { if (!controller.signal.aborted) { setLoading(true); setOpening(false); setError(""); } });
    void listAgentTasks(controller.signal, project.id).then(({ items }) => {
      if (!controller.signal.aborted) setItems(items.filter((task) => task.project_context?.project_id === project.id));
    }).catch((reason: unknown) => { if (!controller.signal.aborted) setError(messageOf(reason)); })
      .finally(() => { if (!controller.signal.aborted) setLoading(false); });
    return () => { controller.abort(); link.current?.abort(); };
  }, [visible, project.id, refresh]);

  async function follow(task?: AgentTask) {
    if (!visible) return;
    if (task && (task.project_context?.project_id !== project.id || !task.project_context.available)) {
      setError("This saved agent is unavailable in this Project. Refresh before trying again."); return;
    }
    link.current?.abort();
    const controller = new AbortController(); link.current = controller;
    setOpening(true); setError("");
    try { if (task) await onOpen?.(task, controller.signal); else await onNew?.(project, controller.signal); }
    catch (reason) { if (!controller.signal.aborted) setError(messageOf(reason)); }
    finally { if (link.current === controller) { link.current = null; if (!controller.signal.aborted) setOpening(false); } }
  }

  return <section className="project-panel project-agents" aria-labelledby="project-agents-title">
    <header>
      <h2 id="project-agents-title">Project agents</h2>
      <div className="project-agent-actions">
        <button type="button" className="project-agent-refresh" aria-label="Refresh Project agents" disabled={loading} onClick={() => setRefresh((value) => value + 1)}>Refresh</button>
        {onNew ? <button className="project-primary-button" type="button" aria-label="New agent in Project" disabled={opening} onClick={() => void follow()}>New agent</button> : null}
      </div>
    </header>
    {error ? <p className="project-error" role="alert">{error}</p> : null}
    {loading ? <p className="project-empty" role="status">Loading Project agents…</p> : items.length ? <ul className="project-home-conversations">{items.map((task) => <li key={task.id} className="conversation-row">
      <SidebarItemButton className="project-agent-card" disabled={opening || !onOpen} onClick={() => void follow(task)}>
        <span className="sidebar-item-title">{task.title || task.objective}</span>
        <small className="sidebar-item-meta project-agent-status">{STATUS_LABELS[task.status]}</small>
      </SidebarItemButton>
    </li>)}</ul> : <p className="project-empty">No agents in this Project yet. Start one with its instructions and knowledge.</p>}
  </section>;
}

function messageOf(reason: unknown) {
  return reason instanceof Error ? reason.message : "The request could not be completed.";
}
