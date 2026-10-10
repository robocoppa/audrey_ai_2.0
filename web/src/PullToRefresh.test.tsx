import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { PullToRefresh } from "./PullToRefresh";

type Point = { identifier: number; clientX: number; clientY: number };

function touch(target: Element, type: string, points: Point[]) {
  const event = new Event(type, { bubbles: true, cancelable: true });
  Object.defineProperties(event, {
    touches: { value: type === "touchend" || type === "touchcancel" ? [] : points },
    changedTouches: { value: points },
  });
  fireEvent(target, event);
  return event;
}

function gesture(target: Element, distance: number, horizontal = 0) {
  touch(target, "touchstart", [{ identifier: 1, clientX: 100, clientY: 100 }]);
  const move = touch(target, "touchmove", [{
    identifier: 1, clientX: 100 + horizontal, clientY: 100 + distance,
  }]);
  return { move, release: () => touch(target, "touchend", [{
    identifier: 1, clientX: 100 + horizontal, clientY: 100 + distance,
  }]) };
}

function workspace() {
  const refresh = vi.fn();
  const view = render(
    <>
      <PullToRefresh onRefresh={refresh} />
      <div className="conversation-thread-slot" hidden>
        <div className="thread-viewport"><div data-testid="hidden-content">Older chat</div></div>
      </div>
      <div className="conversation-thread-slot">
        <div className="thread-viewport" data-testid="viewport">
          <div data-testid="content">Conversation content</div>
          <button type="button">Message action</button>
          <a href="#source">Source link</a>
          <div className="composer"><textarea aria-label="Message" /></div>
        </div>
      </div>
    </>,
  );
  const viewport = screen.getByTestId("viewport");
  const content = screen.getByTestId("content");
  return { ...view, refresh, viewport, content };
}

describe("deliberate mobile pull to refresh", () => {
  beforeEach(() => {
    vi.stubGlobal("matchMedia", vi.fn(() => ({
      matches: true,
      media: "(max-width: 1100px)",
      onchange: null,
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      addListener: vi.fn(),
      removeListener: vi.fn(),
      dispatchEvent: vi.fn(),
    })));
  });

  afterEach(() => {
    cleanup();
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  it("shows progress for a short pull and clears it without refreshing on release", () => {
    const { content, refresh } = workspace();
    const pull = gesture(content, 119);

    expect(screen.getByRole("status")).toHaveTextContent("Pull to refresh");
    expect(refresh).not.toHaveBeenCalled();
    pull.release();
    expect(screen.queryByRole("status")).not.toBeInTheDocument();
    expect(refresh).not.toHaveBeenCalled();
  });

  it("arms at 120 pixels and refreshes only once when the finger is released", () => {
    const { content, refresh } = workspace();
    const pull = gesture(content, 120);

    expect(screen.getByRole("status")).toHaveTextContent("Release to refresh");
    expect(refresh).not.toHaveBeenCalled();
    pull.release();
    expect(refresh).toHaveBeenCalledOnce();
    expect(screen.getByRole("status")).toHaveTextContent("Refreshing");
    pull.release();
    gesture(content, 180).release();
    expect(refresh).toHaveBeenCalledOnce();
  });

  it("allows retracting an armed pull below the threshold to cancel it", () => {
    const { content, refresh } = workspace();
    gesture(content, 160);
    expect(screen.getByRole("status")).toHaveTextContent("Release to refresh");

    touch(content, "touchmove", [{ identifier: 1, clientX: 100, clientY: 150 }]);
    expect(screen.getByRole("status")).toHaveTextContent("Pull to refresh");
    touch(content, "touchend", [{ identifier: 1, clientX: 100, clientY: 150 }]);
    expect(screen.queryByRole("status")).not.toBeInTheDocument();
    expect(refresh).not.toHaveBeenCalled();
  });

  it("cancels an armed gesture when the browser cancels touch", () => {
    const { content, refresh } = workspace();
    gesture(content, 160);
    touch(content, "touchcancel", [{ identifier: 1, clientX: 100, clientY: 260 }]);
    touch(content, "touchend", [{ identifier: 1, clientX: 100, clientY: 260 }]);

    expect(screen.queryByRole("status")).not.toBeInTheDocument();
    expect(refresh).not.toHaveBeenCalled();
  });

  it("does not capture horizontal, upward, or multi-finger scrolling", () => {
    const { content, refresh } = workspace();
    for (const [down, across] of [[150, 200], [-150, 0]]) {
      const pull = gesture(content, down, across);
      expect(pull.move.defaultPrevented).toBe(false);
      pull.release();
      expect(screen.queryByRole("status")).not.toBeInTheDocument();
    }
    gesture(content, 60);
    touch(content, "touchmove", [
      { identifier: 1, clientX: 100, clientY: 260 },
      { identifier: 2, clientX: 120, clientY: 260 },
    ]);
    touch(content, "touchend", [{ identifier: 1, clientX: 100, clientY: 260 }]);
    expect(screen.queryByRole("status")).not.toBeInTheDocument();
    expect(refresh).not.toHaveBeenCalled();
  });

  it("leaves a mostly horizontal first movement native even when its downward distance is tiny", () => {
    const { content, refresh } = workspace();
    const pull = gesture(content, 4, 10);
    expect(pull.move.defaultPrevented).toBe(false);
    touch(content, "touchmove", [{ identifier: 1, clientX: 110, clientY: 250 }]);
    touch(content, "touchend", [{ identifier: 1, clientX: 110, clientY: 250 }]);
    expect(screen.queryByRole("status")).not.toBeInTheDocument();
    expect(refresh).not.toHaveBeenCalled();
  });

  it("ignores dialogs belonging to a hidden conversation when refreshing the active chat", () => {
    const { content, refresh, container } = workspace();
    const dialog = document.createElement("div");
    dialog.setAttribute("role", "dialog");
    dialog.setAttribute("aria-modal", "true");
    dialog.textContent = "Inactive conversation dialog";
    container.querySelector(".conversation-thread-slot[hidden]")!.append(dialog);

    gesture(content, 120).release();
    expect(refresh).toHaveBeenCalledOnce();
  });

  it("requires the gesture to start at the top, even if scrolling reaches the top later", () => {
    const { content, viewport, refresh } = workspace();
    viewport.scrollTop = 80;
    touch(content, "touchstart", [{ identifier: 1, clientX: 100, clientY: 100 }]);
    viewport.scrollTop = 0;
    const move = touch(content, "touchmove", [{ identifier: 1, clientX: 100, clientY: 280 }]);
    touch(content, "touchend", [{ identifier: 1, clientX: 100, clientY: 280 }]);

    expect(move.defaultPrevented).toBe(false);
    expect(screen.queryByRole("status")).not.toBeInTheDocument();
    expect(refresh).not.toHaveBeenCalled();
  });

  it("ignores hidden chats and interactive or overlay surfaces", () => {
    const { viewport, refresh } = workspace();
    const surfaces = [
      screen.getByTestId("hidden-content"),
      screen.getByRole("button", { name: "Message action" }),
      screen.getByRole("link", { name: "Source link" }),
      screen.getByRole("textbox", { name: "Message" }),
    ];
    for (const surface of surfaces) {
      gesture(surface, 180).release();
      expect(screen.queryByRole("status")).not.toBeInTheDocument();
    }
    for (const role of ["dialog", "menu", "navigation"]) {
      const overlay = document.createElement("div");
      overlay.setAttribute("role", role);
      overlay.textContent = "Overlay content";
      (role === "navigation" ? document.body : viewport).append(overlay);
      gesture(overlay, 180).release();
      expect(screen.queryByRole("status")).not.toBeInTheDocument();
      overlay.remove();
    }
    expect(refresh).not.toHaveBeenCalled();
  });

  it("preserves editing and selected text instead of arming a refresh", () => {
    const { content, refresh } = workspace();
    const input = screen.getByRole("textbox", { name: "Message" });
    input.focus();
    gesture(content, 180).release();
    expect(refresh).not.toHaveBeenCalled();
    input.blur();

    vi.spyOn(window, "getSelection").mockReturnValue({
      isCollapsed: false, toString: () => "Selected answer",
    } as Selection);
    gesture(content, 180).release();
    expect(screen.queryByRole("status")).not.toBeInTheDocument();
    expect(refresh).not.toHaveBeenCalled();
  });

  it("does not refresh while the current chat is uploading a file", () => {
    const { content, viewport, refresh } = workspace();
    for (const className of ["attachment-status", "attachment-picker", "skill-picker", "direct-model-menu"]) {
      const blocked = document.createElement("div");
      blocked.className = className;
      blocked.textContent = "Upload or open chat picker";
      viewport.append(blocked);
      gesture(content, 180).release();
      expect(screen.queryByRole("status")).not.toBeInTheDocument();
      blocked.remove();
    }
    expect(refresh).not.toHaveBeenCalled();
  });

  it("supports the project page only when it starts at the top", () => {
    const refresh = vi.fn();
    render(
      <>
        <PullToRefresh onRefresh={refresh} />
        <div className="chat-column"><div className="project-home" data-testid="project-content">
          Project overview
        </div></div>
      </>,
    );
    const project = screen.getByTestId("project-content");
    project.scrollTop = 100;
    gesture(project, 140).release();
    expect(refresh).not.toHaveBeenCalled();
    project.scrollTop = 0;
    gesture(project, 140).release();
    expect(refresh).toHaveBeenCalledOnce();
  });

  it("does not arm on a wide desktop viewport and removes its listeners on unmount", () => {
    vi.mocked(window.matchMedia).mockReturnValue({ matches: false } as MediaQueryList);
    const { content, refresh, unmount } = workspace();
    gesture(content, 180).release();
    expect(screen.queryByRole("status")).not.toBeInTheDocument();
    expect(refresh).not.toHaveBeenCalled();

    unmount();
    vi.mocked(window.matchMedia).mockReturnValue({ matches: true } as MediaQueryList);
    const remainingWorkspace = document.createElement("div");
    remainingWorkspace.innerHTML = '<div class="conversation-thread-slot"><div class="thread-viewport">Remaining chat</div></div>';
    document.body.append(remainingWorkspace);
    gesture(remainingWorkspace.querySelector(".thread-viewport")!, 180).release();
    expect(refresh).not.toHaveBeenCalled();
    remainingWorkspace.remove();
  });
});
