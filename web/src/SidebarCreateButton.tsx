import type { ComponentPropsWithRef } from "react";

import "./SidebarCreateButton.css";

/** The shared entry point for creating work from a sidebar. */
export function SidebarCreateButton({ children, className, type = "button", ...props }: ComponentPropsWithRef<"button">) {
  return <button {...props} type={type} className={`sidebar-create-button${className ? ` ${className}` : ""}`}>
    <svg viewBox="0 0 24 24" width="17" height="17" fill="none" stroke="currentColor" strokeWidth="1.9" strokeLinecap="round" aria-hidden="true">
      <path d="M12 5v14M5 12h14" />
    </svg>
    <span>{children}</span>
  </button>;
}
