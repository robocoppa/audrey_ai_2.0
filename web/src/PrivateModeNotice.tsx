import { useEffect, useId, useLayoutEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";

import "./PrivateModeNotice.css";

export function PrivateModeNotice({ lockReason = "" }: { lockReason?: string }) {
  const tooltipId = useId();
  const badgeRef = useRef<HTMLSpanElement>(null);
  const tooltipRef = useRef<HTMLDivElement>(null);
  const closeTimer = useRef<number | undefined>(undefined);
  const [open, setOpen] = useState(false);
  const [position, setPosition] = useState({ top: 0, left: 0, maxHeight: 0 });

  function reveal() {
    window.clearTimeout(closeTimer.current);
    setOpen(true);
  }

  function hideSoon() {
    window.clearTimeout(closeTimer.current);
    closeTimer.current = window.setTimeout(() => setOpen(false), 120);
  }

  useEffect(() => () => window.clearTimeout(closeTimer.current), []);

  useLayoutEffect(() => {
    if (!open) return;
    function place() {
      const anchor = badgeRef.current?.getBoundingClientRect();
      if (!anchor) return;
      const width = Math.min(340, window.innerWidth - 24);
      setPosition({
        top: anchor.bottom + 8,
        left: Math.max(12, Math.min(anchor.right - width, window.innerWidth - width - 12)),
        maxHeight: Math.max(80, window.innerHeight - anchor.bottom - 20),
      });
    }
    place();
    window.addEventListener("resize", place);
    window.addEventListener("scroll", place, true);
    return () => {
      window.removeEventListener("resize", place);
      window.removeEventListener("scroll", place, true);
    };
  }, [open]);

  useEffect(() => {
    if (!open) return;
    function dismiss(event: PointerEvent) {
      const target = event.target as Node;
      if (!badgeRef.current?.contains(target) && !tooltipRef.current?.contains(target)) setOpen(false);
    }
    function escape(event: KeyboardEvent) {
      if (event.key === "Escape") {
        window.clearTimeout(closeTimer.current);
        setOpen(false);
      }
    }
    document.addEventListener("pointerdown", dismiss);
    document.addEventListener("keydown", escape);
    return () => {
      document.removeEventListener("pointerdown", dismiss);
      document.removeEventListener("keydown", escape);
    };
  }, [open]);

  return <div className="private-mode-notice">
    <span
      className="private-mode-badge"
      ref={badgeRef}
      role="note"
      tabIndex={0}
      aria-label="Private processing"
      aria-describedby={open ? tooltipId : undefined}
      onPointerEnter={reveal}
      onPointerLeave={hideSoon}
      onPointerDown={(event) => { if (event.pointerType === "touch") reveal(); }}
      onFocus={reveal}
      onBlur={() => setOpen(false)}
    >
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true">
        <rect x="5" y="10" width="14" height="11" rx="3" />
        <path d="M8 10V7a4 4 0 0 1 8 0v3M12 14v3" strokeLinecap="round" />
      </svg>
      <span>Private processing</span>
    </span>
    {open ? createPortal(<div
      className="private-mode-tooltip"
      id={tooltipId}
      role="tooltip"
      ref={tooltipRef}
      style={position}
      onPointerEnter={reveal}
      onPointerLeave={hideSoon}
    >
      <strong>Local AI, stored on this server</strong>
      <p>Private uses local models and tools. Conversations and files are saved on this server.</p>
      <p>It is not device-only or anonymous: trusted server administrators may access stored data, and your browser connection passes through Cloudflare.</p>
      <p>Documents and Private projects only; web, saved memory and media are unavailable.</p>
      <p>Private stays selected after the first message.{lockReason ? ` ${lockReason}` : ""}</p>
    </div>, document.body) : null}
  </div>;
}
