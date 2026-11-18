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
