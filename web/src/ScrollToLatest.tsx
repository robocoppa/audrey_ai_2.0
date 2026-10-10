import { ThreadPrimitive, useAuiState } from "@assistant-ui/react";
import { useEffect, useRef, useState } from "react";

export function ScrollToLatest() {
  const messageCount = useAuiState((state) => state.thread.messages.length);
  const anchorRef = useRef<HTMLSpanElement>(null);
  const [conversationOverflows, setConversationOverflows] = useState(false);

  useEffect(() => {
    const element = anchorRef.current?.closest<HTMLElement>(".thread-viewport");
    if (!element || messageCount === 0) return;
    const viewport = element;

    let frame: number | null = null;
    const observedMessages = new Set<Element>();
    const resizeObserver = typeof ResizeObserver === "undefined"
      ? null
      : new ResizeObserver(scheduleMeasure);

    function measure() {
      frame = null;
      const messages = Array.from(viewport.querySelectorAll<HTMLElement>(".message"));
      if (resizeObserver) {
        const currentMessages = new Set<Element>(messages);
        for (const message of observedMessages) {
          if (!currentMessages.has(message)) {
            resizeObserver.unobserve(message);
            observedMessages.delete(message);
          }
        }
        for (const message of messages) {
          if (!observedMessages.has(message)) {
            resizeObserver.observe(message);
            observedMessages.add(message);
          }
        }
      }

      const first = messages[0];
      const last = messages[messages.length - 1];
      const viewportStyle = getComputedStyle(viewport);
      const dock = anchorRef.current?.closest<HTMLElement>(".composer-dock");
      const dockStyle = dock ? getComputedStyle(dock) : null;
      const composerHeight = dock?.querySelector(".composer")?.getBoundingClientRect().height ?? 0;
      const padding = (parseFloat(viewportStyle.paddingTop) || 0)
        + (parseFloat(viewportStyle.paddingBottom) || 0)
        + (parseFloat(dockStyle?.paddingTop ?? "") || 0)
        + (parseFloat(dockStyle?.paddingBottom ?? "") || 0)
        + (parseFloat(dockStyle?.marginTop ?? "") || 0);
      const availableHeight = Math.max(0, viewport.clientHeight - composerHeight - padding);
      const messageHeight = first && last
        ? last.getBoundingClientRect().bottom - first.getBoundingClientRect().top
        : 0;

      // Welcome portraits, pickers, and activity notices are not conversation history.
      setConversationOverflows(messageHeight > availableHeight + 4);
    }

    function scheduleMeasure() {
      if (frame === null) frame = window.requestAnimationFrame(measure);
    }

    resizeObserver?.observe(viewport);
    const composer = viewport.querySelector(".composer");
    if (composer) resizeObserver?.observe(composer);
    const mutations = new MutationObserver(scheduleMeasure);
    mutations.observe(viewport, { childList: true, subtree: true, characterData: true });
    window.addEventListener("resize", scheduleMeasure);
    scheduleMeasure();

    return () => {
      if (frame !== null) window.cancelAnimationFrame(frame);
      resizeObserver?.disconnect();
      mutations.disconnect();
      window.removeEventListener("resize", scheduleMeasure);
    };
  }, [messageCount]);

  return (
    <>
      <span ref={anchorRef} hidden aria-hidden="true" />
      {messageCount > 0 && conversationOverflows ? (
        <ThreadPrimitive.ScrollToBottom
          className="scroll-bottom"
          aria-label="Scroll to latest message"
        >
          <svg viewBox="0 0 24 24" aria-hidden="true">
            <path d="M12 5v14m5.5-5.5L12 19l-5.5-5.5" />
          </svg>
        </ThreadPrimitive.ScrollToBottom>
      ) : null}
    </>
  );
}
