import type { ReactNode } from "react";

import "./SidebarHeader.css";

/** Fixed sidebar actions keep navigation independent of history/content height. */
export function SidebarHeader({ action, children }: { action: ReactNode; children?: ReactNode }) {
  return <div className="workspace-sidebar-header">{action}{children}</div>;
}

/** Native button semantics remain owned by the filter or workspace navigation. */
export function SidebarSectionTabs({ label, className, children }: {
  label: string;
  className?: string;
  children: ReactNode;
}) {
  return <nav className={`sidebar-section-tabs${className ? ` ${className}` : ""}`} aria-label={label}>{children}</nav>;
}

/** A single compact label, optionally with an existing refresh action. */
export function SidebarHeading({ id, level = 2, children, action }: {
  id?: string;
  level?: 1 | 2;
  children: ReactNode;
  action?: ReactNode;
}) {
  const Heading = level === 1 ? "h1" : "h2";
  return <header className="workspace-sidebar-heading"><Heading id={id}>{children}</Heading>{action}</header>;
}
