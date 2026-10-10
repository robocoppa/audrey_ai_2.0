import { useEffect, useState } from "react";

const PULL_THRESHOLD = 120;
const PRIMARY_SCROLL_AREA = ".conversation-thread-slot:not([hidden]) .thread-viewport, .chat-column > .project-home";
const INTERACTIVE_AREA = 'button, a, input, select, textarea, [contenteditable]:not([contenteditable="false"]), .composer, [role="dialog"], [role="menu"], .saved-run-details, .run-activity';
const OPEN_OVERLAY = '[aria-modal="true"], [role="dialog"]:not([aria-hidden="true"]):not([hidden]), .conversation-drawer-backdrop';
const BUSY_OR_OPEN_PICKER = ".attachment-status, .attachment-picker, .skill-picker, .direct-model-menu, .saved-sources[open], .saved-models[open], .saved-tools[open]";

function reloadPage() {
  window.location.reload();
}

type Gesture = {
  id: number;
  startX: number;
  startY: number;
  area: HTMLElement;
  claimed: boolean;
};

export function PullToRefresh({ onRefresh = reloadPage }: { onRefresh?: () => void }) {
  const [pull, setPull] = useState<{ distance: number; top: number; refreshing: boolean } | null>(null);

  useEffect(() => {
    let gesture: Gesture | null = null;
    let refreshing = false;

    function eligible(area: HTMLElement) {
      return (window.matchMedia?.("(max-width: 1100px)").matches ?? window.innerWidth <= 1100)
        && document.querySelector(PRIMARY_SCROLL_AREA) === area
        && area.scrollTop <= 1
        && !Array.from(document.querySelectorAll(OPEN_OVERLAY)).some((overlay) =>
          !overlay.closest('[hidden], [aria-hidden="true"], [inert]'))
        && !area.querySelector(BUSY_OR_OPEN_PICKER)
        && !document.activeElement?.matches('input, textarea, select, [contenteditable]:not([contenteditable="false"])')
        && window.getSelection()?.isCollapsed !== false;
    }

    function cancel() {
      gesture = null;
      if (!refreshing) setPull(null);
    }

    function touchStart(event: TouchEvent) {
      cancel();
      if (refreshing || event.touches.length !== 1 || !(event.target instanceof Element)) return;
      const area = document.querySelector<HTMLElement>(PRIMARY_SCROLL_AREA);
      if (!area || !eligible(area) || event.target.closest(INTERACTIVE_AREA)) return;
      if (!area.contains(event.target) && !event.target.closest(".topbar")) return;
      const touch = event.touches[0];
      gesture = { id: touch.identifier, startX: touch.clientX, startY: touch.clientY, area, claimed: false };
    }

    function distanceFor(touch: Touch) {
      if (!gesture || !eligible(gesture.area)) return null;
      const down = touch.clientY - gesture.startY;
      const across = Math.abs(touch.clientX - gesture.startX);
      return down > 0 && across <= down / 1.5 ? down : null;
    }

    function touchMove(event: TouchEvent) {
      if (!gesture) return;
      const touchId = gesture.id;
      const touch = Array.from(event.touches).find(({ identifier }) => identifier === touchId);
      const distance = event.touches.length === 1 && touch ? distanceFor(touch) : null;
      if (distance === null || !event.cancelable) {
        cancel();
        return;
      }
      // Own only a downward pull at the boundary; other scrolling stays native.
      event.preventDefault();
      gesture.claimed = true;
      setPull({ distance, top: gesture.area.getBoundingClientRect().top + 12, refreshing: false });
    }

    function touchEnd(event: TouchEvent) {
      if (!gesture) return;
      const touchId = gesture.id;
      const touch = Array.from(event.changedTouches).find(({ identifier }) => identifier === touchId);
      const distance = event.touches.length === 0 && touch ? distanceFor(touch) : null;
      if (!gesture.claimed || distance === null || distance < PULL_THRESHOLD) {
        cancel();
        return;
      }
      const top = gesture.area.getBoundingClientRect().top + 12;
      gesture = null;
      refreshing = true;
      setPull({ distance: PULL_THRESHOLD, top, refreshing: true });
      onRefresh();
    }

    document.addEventListener("touchstart", touchStart, { capture: true, passive: true });
    document.addEventListener("touchmove", touchMove, { capture: true, passive: false });
    document.addEventListener("touchend", touchEnd, { capture: true, passive: true });
    document.addEventListener("touchcancel", cancel, { capture: true, passive: true });
    window.addEventListener("resize", cancel);
    return () => {
      document.removeEventListener("touchstart", touchStart, true);
      document.removeEventListener("touchmove", touchMove, true);
      document.removeEventListener("touchend", touchEnd, true);
      document.removeEventListener("touchcancel", cancel, true);
      window.removeEventListener("resize", cancel);
    };
  }, [onRefresh]);

  if (!pull || pull.distance < 8) return null;
  const ready = pull.distance >= PULL_THRESHOLD;
  return (
    <div
      className="pull-to-refresh"
      role="status"
      aria-live="polite"
      data-ready={ready}
      data-refreshing={pull.refreshing}
      style={{ top: pull.top, transform: `translate(-50%, ${Math.min(pull.distance * 0.3, 36)}px)` }}
    >
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <path d="M20 7v5h-5M4 17v-5h5" />
        <path d="M6.2 8a7 7 0 0 1 11.5-2L20 9M4 15l2.3 3A7 7 0 0 0 17.8 16" />
      </svg>
      <span>{pull.refreshing ? "Refreshing…" : ready ? "Release to refresh" : "Pull to refresh"}</span>
    </div>
  );
}
