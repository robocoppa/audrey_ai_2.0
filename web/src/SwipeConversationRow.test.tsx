import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { SwipeConversationRow } from "./SwipeConversationRow";

function pointer(target: Element, type: string, x: number, y = 100, values = {}) {
  const event = new Event(type, { bubbles: true, cancelable: true });
  Object.defineProperties(event, Object.fromEntries(Object.entries({
    pointerId: 1, pointerType: "touch", isPrimary: true, button: 0, clientX: x, clientY: y, ...values,
  }).map(([key, value]) => [key, { value }])));
  fireEvent(target, event);
  return event;
}

function row(overrides: { enabled?: boolean; disabled?: boolean; archiveLabel?: "Archive" | "Restore" } = {}) {
  const archive = vi.fn();
  const remove = vi.fn();
  const open = vi.fn();
  const view = render(
    <SwipeConversationRow enabled title="Example chat" onArchive={archive} onDelete={remove} {...overrides}>
      <button type="button" onClick={open}>Example chat</button>
    </SwipeConversationRow>,
  );
  const target = screen.getByRole("button", { name: "Example chat" });
  function swipe(horizontal: number, vertical = 0) {
    pointer(target, "pointerdown", 150);
    const move = pointer(target, "pointermove", 150 + horizontal, 100 + vertical);
    return {
      move,
      release: () => pointer(target, "pointerup", 150 + horizontal, 100 + vertical),
    };
  }
  return { ...view, target, archive, remove, open, swipe };
}

describe("mobile conversation swipes", () => {
  afterEach(() => {
    cleanup();
    vi.restoreAllMocks();
  });

  it("archives on release after a deliberate right swipe and suppresses its generated click", () => {
    const { target, swipe, archive, remove, open } = row();
    const pull = swipe(80);
    expect(pull.move.defaultPrevented).toBe(true);
    expect(archive).not.toHaveBeenCalled();
    pull.release();
    expect(archive).toHaveBeenCalledOnce();
    expect(remove).not.toHaveBeenCalled();
    fireEvent.click(target);
    expect(open).not.toHaveBeenCalled();
    pointer(target, "pointerup", 250);
    expect(archive).toHaveBeenCalledOnce();
  });

  it("keeps ordinary taps and vertical scrolling while cancelling short horizontal swipes", () => {
    const { target, swipe, archive, remove, open } = row();
    fireEvent.click(target);
    expect(open).toHaveBeenCalledOnce();
    const vertical = swipe(15, 100);
    expect(vertical.move.defaultPrevented).toBe(false);
    vertical.release();
    const short = swipe(79);
    short.release();
    fireEvent.click(target);
    expect(open).toHaveBeenCalledOnce();
    expect(archive).not.toHaveBeenCalled();
    expect(remove).not.toHaveBeenCalled();
    expect(screen.queryByRole("button", { name: "Confirm delete conversation Example chat" })).not.toBeInTheDocument();
  });

  it("reveals delete after a left swipe and requires the explicit Delete button", () => {
    const { swipe, target, archive, remove, open } = row();
    swipe(-120).release();
    fireEvent.click(target);
    expect(remove).not.toHaveBeenCalled();
    const confirm = screen.getByRole("button", { name: "Confirm delete conversation Example chat" });
    fireEvent.click(confirm);
    expect(remove).toHaveBeenCalledOnce();
    expect(archive).not.toHaveBeenCalled();
    expect(open).not.toHaveBeenCalled();
    expect(screen.queryByRole("button", { name: "Cancel conversation action" })).not.toBeInTheDocument();
  });

  it("cancels delete with its Cancel button, an outside touch, or Escape", () => {
    const { swipe, remove } = row();
    swipe(-120).release();
    fireEvent.click(screen.getByRole("button", { name: "Cancel conversation action" }));
    expect(screen.queryByRole("button", { name: "Confirm delete conversation Example chat" })).not.toBeInTheDocument();
    swipe(-120).release();
    pointer(document.body, "pointerdown", 200);
    expect(screen.queryByRole("button", { name: "Confirm delete conversation Example chat" })).not.toBeInTheDocument();
    swipe(-120).release();
    fireEvent.keyDown(document, { key: "Escape" });
    expect(screen.queryByRole("button", { name: "Confirm delete conversation Example chat" })).not.toBeInTheDocument();
    expect(remove).not.toHaveBeenCalled();
  });

  it("does not commit cancelled, retracted, or multiple-finger gestures", () => {
    const { target, swipe, archive, remove } = row();
    swipe(140);
    pointer(target, "pointercancel", 290);
    pointer(target, "pointerup", 290);
    swipe(140);
    pointer(target, "pointermove", 190);
    pointer(target, "pointerup", 190);
    swipe(-140);
    pointer(target, "pointerdown", 200, 100, { pointerId: 2, isPrimary: false });
    pointer(target, "pointerup", 10);
    expect(archive).not.toHaveBeenCalled();
    expect(remove).not.toHaveBeenCalled();
    expect(screen.queryByRole("button", { name: "Confirm delete conversation Example chat" })).not.toBeInTheDocument();
  });

  it.each([{ enabled: false }, { disabled: true }])("does not enable gestures for %s", (overrides) => {
    const { swipe, archive, remove, target } = row(overrides);
    swipe(140).release();
    swipe(-140).release();
    fireEvent.keyDown(target, { key: "ArrowLeft" });
    expect(archive).not.toHaveBeenCalled();
    expect(remove).not.toHaveBeenCalled();
    expect(screen.queryByRole("button", { name: "Confirm delete conversation Example chat" })).not.toBeInTheDocument();
  });

  it("makes keyboard actions available without archiving or deleting from arrow keys", () => {
    const { target, archive, remove } = row({ archiveLabel: "Restore" });
    expect(screen.queryByRole("button", { name: "Restore conversation Example chat" })).not.toBeInTheDocument();
    fireEvent.keyDown(target, { key: "ArrowRight" });
    const restore = screen.getByRole("button", { name: "Restore conversation Example chat" });
    expect(restore).toHaveFocus();
    expect(archive).not.toHaveBeenCalled();
    fireEvent.click(restore);
    expect(archive).toHaveBeenCalledOnce();
    fireEvent.keyDown(target, { key: "ArrowLeft" });
    const confirm = screen.getByRole("button", { name: "Confirm delete conversation Example chat" });
    expect(confirm).toHaveFocus();
    expect(remove).not.toHaveBeenCalled();
    const drawerEscape = vi.fn();
    document.addEventListener("keydown", drawerEscape);
    fireEvent.keyDown(confirm, { key: "Escape" });
    document.removeEventListener("keydown", drawerEscape);
    expect(screen.queryByRole("button", { name: "Confirm delete conversation Example chat" })).not.toBeInTheDocument();
    expect(target).toHaveFocus();
    expect(drawerEscape).not.toHaveBeenCalled();
  });

  it("supports mouse dragging in the compact drawer and leaves desktop rows unchanged", () => {
    const { target, archive, open } = row();
    pointer(target, "pointerdown", 150, 100, { pointerType: "mouse" });
    pointer(target, "pointermove", 290, 100, { pointerType: "mouse" });
    pointer(target, "pointerup", 290, 100, { pointerType: "mouse" });
    fireEvent.click(target);
    expect(archive).toHaveBeenCalledOnce();
    expect(open).not.toHaveBeenCalled();
    cleanup();
    const desktop = row({ enabled: false });
    pointer(desktop.target, "pointerdown", 150, 100, { pointerType: "mouse" });
    pointer(desktop.target, "pointermove", 290, 100, { pointerType: "mouse" });
    pointer(desktop.target, "pointerup", 290, 100, { pointerType: "mouse" });
    fireEvent.click(desktop.target);
    expect(desktop.archive).not.toHaveBeenCalled();
    expect(desktop.open).toHaveBeenCalledOnce();
  });

  it("checks the final release position so a last-moment retraction or vertical movement cancels", () => {
    const { target, swipe, archive, remove } = row();
    swipe(140);
    pointer(target, "pointerup", 190);
    swipe(-140);
    pointer(target, "pointerup", 110);
    swipe(140);
    pointer(target, "pointerup", 290, 260);
    expect(archive).not.toHaveBeenCalled();
    expect(remove).not.toHaveBeenCalled();
    expect(screen.queryByRole("button", { name: "Confirm delete conversation Example chat" })).not.toBeInTheDocument();
  });
});
