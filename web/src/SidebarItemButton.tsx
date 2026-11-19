import type { ComponentPropsWithRef } from "react";

import "./SidebarItemButton.css";

/** Shared appearance; native selection, labels and actions remain caller-owned. */
export function SidebarItemButton({ className, type = "button", ...props }: ComponentPropsWithRef<"button">) {
  return <button {...props} type={type} className={`sidebar-item${className ? ` ${className}` : ""}`} />;
}
