import { act, cleanup, render, screen } from "@testing-library/react";
import type { ComponentProps } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { ScrollToLatest } from "./ScrollToLatest";

const state = vi.hoisted(() => ({ messageCount: 0 }));

vi.mock("@assistant-ui/react", () => ({
  useAuiState: (select: (value: { thread: { messages: unknown[] } }) => number) =>
    select({ thread: { messages: Array.from({ length: state.messageCount }) } }),
  ThreadPrimitive: {
    ScrollToBottom: (props: ComponentProps<"button">) => <button {...props} />,
  },
}));

let measureAfterResize: ResizeObserverCallback | undefined;
const disconnect = vi.fn();

function conversation(messageHeights: number[], viewportHeight = 600) {
  state.messageCount = messageHeights.length;
  return render(
    <div
      className="thread-viewport"
      data-height={viewportHeight}
      style={{ paddingTop: 16, paddingBottom: 16 }}
    >
      {messageHeights.map((height, index) => (
        <div
          className="message"
          key={index}
          data-top={messageHeights.slice(0, index).reduce((total, item) => total + item, 0)}
          data-height={height}
        >
          Message {index + 1}
        </div>
      ))}
      <div
        className="composer-dock"
        style={{ paddingTop: 8, paddingBottom: 16, marginTop: 16 }}
      >
        <ScrollToLatest />
        <div className="composer-model-picker" data-height="800">Portrait</div>
        <form className="composer" data-height="120" />
        <section className="attachment-picker" data-height="800">Files</section>
      </div>
    </div>,
  );
}

async function measure() {
  await act(async () => {
    vi.runOnlyPendingTimers();
  });
}

describe("ScrollToLatest conversation overflow", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    disconnect.mockClear();
    measureAfterResize = undefined;
    vi.spyOn(window, "requestAnimationFrame").mockImplementation((callback) =>
      window.setTimeout(() => callback(0), 0));
    vi.spyOn(window, "cancelAnimationFrame").mockImplementation((frame) => {
      window.clearTimeout(frame);
    });
    vi.spyOn(HTMLElement.prototype, "getBoundingClientRect").mockImplementation(function (this: HTMLElement) {
      return new DOMRect(0, Number(this.dataset.top ?? 0), 360, Number(this.dataset.height ?? 0));
    });
    vi.spyOn(HTMLElement.prototype, "clientHeight", "get").mockImplementation(function (this: HTMLElement) {
      return Number(this.dataset.height ?? 0);
    });
    vi.stubGlobal("ResizeObserver", class {
      constructor(callback: ResizeObserverCallback) { measureAfterResize = callback; }
      observe = vi.fn();
      unobserve = vi.fn();
      disconnect = disconnect;
    });
  });

  afterEach(() => {
    cleanup();
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
    vi.useRealTimers();
  });

  it("hides the jump for an empty welcome even when its portrait and picker overflow", async () => {
    conversation([], 300);
    await measure();

    expect(screen.queryByRole("button", { name: "Scroll to latest message" })).not.toBeInTheDocument();
  });

  it("hides the jump for one short turn that fits the reading area", async () => {
    conversation([40, 80]);
    await measure();

    expect(screen.queryByRole("button", { name: "Scroll to latest message" })).not.toBeInTheDocument();
  });

  it("allows the native jump action when actual conversation messages overflow", async () => {
    conversation([40, 700, 40, 120]);
    await measure();

    expect(screen.getByRole("button", { name: "Scroll to latest message" })).toBeInTheDocument();
  });

  it("remeasures after viewport growth and disconnects on unmount", async () => {
    const view = conversation([40, 700]);
    await measure();
    expect(screen.getByRole("button", { name: "Scroll to latest message" })).toBeInTheDocument();

    view.container.querySelector<HTMLElement>(".thread-viewport")!.dataset.height = "1200";
    measureAfterResize?.([], {} as ResizeObserver);
    await measure();
    expect(screen.queryByRole("button", { name: "Scroll to latest message" })).not.toBeInTheDocument();

    view.unmount();
    expect(disconnect).toHaveBeenCalledOnce();
  });
});
