import { useEffect, useRef, useState } from "react";
import type { PointerEvent, ReactNode } from "react";

const SWIPE_THRESHOLD = 80;
const ACTION_WIDTH = 176;

type Action = "archive" | "delete";
type Gesture = { id: number; x: number; y: number; claimed: boolean };

export function SwipeConversationRow({
  children,
  title,
  enabled,
  disabled = false,
  archiveLabel = "Archive",
  onArchive,
  onDelete,
}: {
  children: ReactNode;
  title: string;
  enabled: boolean;
  disabled?: boolean;
  archiveLabel?: "Archive" | "Restore";
  onArchive: () => void;
  onDelete: () => void;
}) {
  const rowRef = useRef<HTMLDivElement>(null);
  const gestureRef = useRef<Gesture | null>(null);
  const ignoreClickUntil = useRef(0);
  const keyboardReveal = useRef(false);
  const [action, setAction] = useState<Action | null>(null);
  const [distance, setDistance] = useState(0);

  useEffect(() => {
    function dismiss(event: Event) {
      if (event instanceof KeyboardEvent) {
        if (event.key !== "Escape" || !rowRef.current?.querySelector(".swipe-conversation-actions")) return;
        event.preventDefault();
        event.stopPropagation();
        rowRef.current.querySelector<HTMLButtonElement>(".swipe-conversation-content button")?.focus();
      }
      if (event.type === "pointerdown" && event.target instanceof Node
        && rowRef.current?.contains(event.target)) return;
      gestureRef.current = null;
      setAction(null);
      setDistance(0);
    }
    document.addEventListener("pointerdown", dismiss);
    document.addEventListener("keydown", dismiss, true);
    return () => {
      document.removeEventListener("pointerdown", dismiss);
      document.removeEventListener("keydown", dismiss, true);
    };
  }, []);

  useEffect(() => {
    if (action && keyboardReveal.current) {
      keyboardReveal.current = false;
      rowRef.current?.querySelector<HTMLButtonElement>(".swipe-conversation-action")?.focus();
    }
  }, [action]);

  function cancelGesture() {
    gestureRef.current = null;
    setDistance(0);
  }

  function pointerStart(event: PointerEvent<HTMLDivElement>) {
    if (!enabled || disabled || event.button !== 0
      || (event.target instanceof Element && event.target.closest(".swipe-conversation-actions"))) return;
    if (!event.isPrimary) {
      cancelGesture();
      return;
    }
    setAction(null);
    gestureRef.current = { id: event.pointerId, x: event.clientX, y: event.clientY, claimed: false };
  }

  function pointerMove(event: PointerEvent<HTMLDivElement>) {
    const gesture = gestureRef.current;
    if (!gesture || gesture.id !== event.pointerId) return;
    if (!enabled || disabled) {
      cancelGesture();
      return;
    }
    const horizontal = event.clientX - gesture.x;
    const vertical = event.clientY - gesture.y;
    if (!gesture.claimed) {
      if (Math.abs(vertical) > 8 && Math.abs(vertical) >= Math.abs(horizontal) / 1.5) {
        cancelGesture();
        return;
      }
      if (Math.abs(horizontal) < 12 || Math.abs(horizontal) <= Math.abs(vertical) * 1.5) return;
      gesture.claimed = true;
      event.currentTarget.setPointerCapture?.(event.pointerId);
    }
    event.preventDefault();
    ignoreClickUntil.current = Date.now() + 500;
    setDistance(Math.max(-120, Math.min(120, horizontal)));
  }

  function pointerEnd(event: PointerEvent<HTMLDivElement>) {
    const gesture = gestureRef.current;
    if (!gesture || gesture.id !== event.pointerId) return;
    if (gesture.claimed) {
      event.preventDefault();
      ignoreClickUntil.current = Date.now() + 500;
    }
    const horizontal = event.clientX - gesture.x;
    const vertical = event.clientY - gesture.y;
    cancelGesture();
    if (!enabled || disabled || !gesture.claimed || Math.abs(horizontal) < SWIPE_THRESHOLD
      || Math.abs(horizontal) <= Math.abs(vertical) * 1.5) return;
    if (horizontal > 0) {
      onArchive();
    } else {
      setAction("delete");
    }
  }

  const visibleAction = enabled && !disabled ? action : null;
  const offset = enabled && !disabled
    ? visibleAction === "archive" ? ACTION_WIDTH : visibleAction === "delete" ? -ACTION_WIDTH : distance
    : 0;

  return (
    <div
      ref={rowRef}
      className="swipe-conversation-row"
      data-enabled={enabled}
      data-dragging={enabled && !disabled && distance !== 0}
      data-action={visibleAction ?? undefined}
      onPointerDown={pointerStart}
      onPointerMove={pointerMove}
      onPointerUp={pointerEnd}
      onPointerCancel={cancelGesture}
      onLostPointerCapture={cancelGesture}
      onClickCapture={(event) => {
        if (event.target instanceof Element && event.target.closest(".swipe-conversation-actions")) return;
        if (Date.now() < ignoreClickUntil.current) {
          event.preventDefault();
          event.stopPropagation();
          return;
        }
        if (visibleAction) {
          event.preventDefault();
          event.stopPropagation();
          setAction(null);
        }
      }}
      onKeyDown={(event) => {
        if (!enabled || disabled || (event.target instanceof Element
          && event.target.closest(".swipe-conversation-actions"))) return;
        if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
        event.preventDefault();
        keyboardReveal.current = true;
        setAction(event.key === "ArrowLeft" ? "delete" : "archive");
      }}
    >
      {enabled && !disabled && distance !== 0 && !visibleAction ? (
        <span className="swipe-conversation-preview" data-direction={distance > 0 ? "archive" : "delete"} aria-hidden="true">
          {distance > 0 ? archiveLabel : "Delete"}
        </span>
      ) : null}
      {visibleAction ? (
        <div className="swipe-conversation-actions" data-action={visibleAction}>
          <button
            className="swipe-conversation-action"
            type="button"
            aria-label={visibleAction === "delete" ? `Confirm delete conversation ${title}` : `${archiveLabel} conversation ${title}`}
            onClick={() => {
              setAction(null);
              if (visibleAction === "archive") onArchive();
              else onDelete();
            }}
          >
            {visibleAction === "archive" ? archiveLabel : "Delete"}
          </button>
          <button
            className="swipe-conversation-cancel"
            type="button"
            aria-label="Cancel conversation action"
            onClick={() => {
              setAction(null);
              rowRef.current?.querySelector<HTMLButtonElement>(".swipe-conversation-content button")?.focus();
            }}
          >Cancel</button>
        </div>
      ) : null}
      <div className="swipe-conversation-content" style={{ transform: offset ? `translateX(${offset}px)` : undefined, touchAction: enabled ? "pan-y" : undefined }}>
        {children}
      </div>
    </div>
  );
}
