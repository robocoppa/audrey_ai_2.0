import { useEffect, useId, useLayoutEffect, useRef, useState, type ReactNode } from "react";

import "./ConversationToolbox.css";

export function ConversationToolbox({ available, open, onOpenChange, content, children, pendingCount = 0 }: {
  available: boolean;
  pendingCount?: number;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  content: ReactNode;
  children: ReactNode;
}) {
  const [compact, setCompact] = useState(() => window.matchMedia?.("(max-width: 900px)").matches ?? false);
  const dialogRef = useRef<HTMLDialogElement>(null);
  const closeRef = useRef<HTMLButtonElement>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const titleId = useId();
  const panelId = useId();
  const visible = available && open;
  useEffect(() => {
    const media = window.matchMedia?.("(max-width: 900px)");
    if (!media) return;
    const update = () => setCompact(media.matches);
    media.addEventListener("change", update);
    return () => media.removeEventListener("change", update);
  }, []);
  useLayoutEffect(() => {
    const dialog = dialogRef.current;
    if (!compact || !dialog) return;
    if (visible && !dialog.open) {
      dialog.showModal();
      closeRef.current?.focus({ preventScroll: true });
    } else if (!visible && dialog.open) dialog.close();
    return () => { if (dialog.open) dialog.close(); };
  }, [compact, visible]);
  function close() {
    dialogRef.current?.close();
    onOpenChange(false);
    triggerRef.current?.focus({ preventScroll: true });
  }
  const contents = <>
    <header className="conversation-toolbox-header">
      <h2 id={titleId}><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 8h16v12H4zM8 8V5h8v3M4 13h16M10 12v3h4v-3" /></svg>Toolbox</h2>
      <button ref={closeRef} type="button" aria-label="Hide toolbox" onClick={close}>
        <svg viewBox="0 0 24 24" aria-hidden="true"><path d="m6 6 12 12M18 6 6 18" /></svg>
      </button>
    </header>
    <div className="conversation-toolbox-content">{content}</div>
  </>;
  return <div className="conversation-with-toolbox" data-toolbox-open={visible && !compact}>
    <div className="conversation-toolbox-main">
      {available ? <div className="conversation-toolbox-toolbar">
        <button ref={triggerRef} type="button" aria-label="Toolbox" title={pendingCount ? `${pendingCount} details to review` : "Toolbox"} aria-controls={panelId} aria-expanded={visible}
          aria-haspopup={compact ? "dialog" : undefined} onClick={() => onOpenChange(!visible)}>
          <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 8h16v12H4zM8 8V5h8v3M4 13h16M10 12v3h4v-3" /></svg>Toolbox{pendingCount ? <span className="toolbox-pending-count" aria-hidden="true">{pendingCount}</span> : null}
        </button>
      </div> : null}
      {children}
    </div>
    {compact ? <dialog ref={dialogRef} id={panelId} className="conversation-toolbox conversation-toolbox-drawer"
      aria-labelledby={titleId} aria-modal={visible ? true : undefined} onCancel={(event) => { event.preventDefault(); close(); }}
      onClick={(event) => { if (event.target === event.currentTarget) {
        const bounds = event.currentTarget.getBoundingClientRect();
        if (event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom) close();
      } }}>{contents}</dialog>
      : <aside id={panelId} className="conversation-toolbox" aria-labelledby={titleId} hidden={!visible}>{contents}</aside>}
  </div>;
}
