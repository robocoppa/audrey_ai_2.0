import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Page, type Route } from "@playwright/test";

const CONVERSATION_ID = "con_browser_test";

test("runs a native turn with typed stage, tool, and source activity", async ({ page }) => {
  let requestBody: Record<string, unknown> | null = null;
  await mockAudreyApi(page, async (route) => {
    requestBody = route.request().postDataJSON() as Record<string, unknown>;
    await route.fulfill({
      status: 200,
      contentType: "text/event-stream",
      body: aguiStream([
        { type: "RUN_STARTED", timestamp: 1, threadId: CONVERSATION_ID, runId: "run_browser" },
        { type: "TEXT_MESSAGE_START", timestamp: 2, messageId: "msg_browser" },
        { type: "STEP_STARTED", timestamp: 3, stepName: "researching" },
        {
          type: "CUSTOM",
          timestamp: 4,
          name: "audrey.stage.progress",
          value: { stage: "researching", delta: "Checking current sources" },
        },
        {
          type: "CUSTOM",
          timestamp: 5,
          name: "audrey.source.observed",
          value: { sourceId: "source_1", title: "Official source" },
        },
        {
          type: "TOOL_CALL_START",
          timestamp: 6,
          toolCallId: "tool_1",
          toolCallName: "web_search",
          parentMessageId: "msg_browser",
        },
        {
          type: "TOOL_CALL_ARGS",
          timestamp: 7,
          toolCallId: "tool_1",
          delta: JSON.stringify({ query: "Audrey native UI" }),
        },
        { type: "TOOL_CALL_END", timestamp: 8, toolCallId: "tool_1" },
        {
          type: "TOOL_CALL_RESULT",
          timestamp: 9,
          messageId: "tool_result_1",
          toolCallId: "tool_1",
          content: JSON.stringify({ results: 1 }),
        },
        { type: "STEP_FINISHED", timestamp: 10, stepName: "researching" },
        {
          type: "TEXT_MESSAGE_CONTENT",
          timestamp: 11,
          messageId: "msg_browser",
          delta: "Browser-native answer.",
        },
        { type: "TEXT_MESSAGE_END", timestamp: 12, messageId: "msg_browser" },
        {
          type: "RUN_FINISHED",
          timestamp: 13,
          threadId: CONVERSATION_ID,
          runId: "run_browser",
          outcome: { type: "success" },
        },
      ]),
    });
  });

  await page.goto("./");
  await expect(page.getByRole("heading", { name: "Browser smoke" })).toBeVisible();
  await expect(page.getByRole("combobox", { name: "Mode" })).toHaveValue("fast");

  const composer = page.getByRole("textbox", { name: "Message Audrey" });
  await composer.fill("Exercise the native browser path");
  await composer.press("Enter");

  await expect(page.getByText("Browser-native answer.")).toBeVisible();
  await expect(page.getByText("web_search · complete")).toBeVisible();
  await expect(page.getByText("1 source · Official source")).toBeVisible();
  await expect(page.getByText("Complete", { exact: true })).toBeVisible();
  expect(requestBody).toMatchObject({ threadId: CONVERSATION_ID });
  expect(requestBody?.messages).toHaveLength(1);

  const secrets = await page.evaluate(() => ({
    local: Object.keys(localStorage),
    session: Object.keys(sessionStorage),
  }));
  expect(secrets).toEqual({ local: [], session: [] });

  const accessibility = await new AxeBuilder({ page }).analyze();
  expect(accessibility.violations.map(({ id }) => id)).toEqual([]);
});

test("cancels an active browser run without leaving an error state", async ({ page }) => {
  await installHangingAgent(page);
  await mockAudreyApi(page);
  await page.goto("./");

  const composer = page.getByRole("textbox", { name: "Message Audrey" });
  await composer.fill("Keep this run open");
  await composer.press("Enter");
  await page.getByRole("button", { name: "Stop" }).click();

  await expect(page.getByText("Stopped")).toBeVisible();
  await expect.poll(
    () => page.evaluate(() => Boolean((window as Window & { __cancelObserved?: boolean }).__cancelObserved)),
  ).toBe(true);
  await expect(page.getByRole("alert")).toHaveCount(0);
});

test("surfaces an expired session during a run", async ({ page }) => {
  await mockAudreyApi(page, (route) =>
    route.fulfill({
      status: 401,
      contentType: "application/json",
      body: JSON.stringify({ detail: "Session expired." }),
    }),
  );
  await page.goto("./");

  const composer = page.getByRole("textbox", { name: "Message Audrey" });
  await composer.fill("Attempt after expiry");
  await composer.press("Enter");

  await expect(page.getByRole("alert")).toBeVisible();
  await expect(page.getByText("Connection failed")).toBeVisible();
});

async function mockAudreyApi(
  page: Page,
  agentHandler?: (route: Route) => Promise<void> | void,
) {
  await page.route("**/api/**", async (route) => {
    const url = new URL(route.request().url());
    if (url.pathname === "/api/me") {
      await json(route, {
        id: "usr_browser_test",
        email: "alice@example.com",
        display_name: "Alice",
        role: "user",
        status: "active",
        auth_provider: "cloudflare_access",
      });
      return;
    }
    if (url.pathname === "/api/conversations" && route.request().method() === "GET") {
      await json(route, {
        items: [{
          id: CONVERSATION_ID,
          title: "Browser smoke",
          default_mode: "fast",
          created_at: "2026-09-04T00:00:00Z",
          updated_at: "2026-09-04T00:00:00Z",
          last_message_at: null,
          archived_at: null,
        }],
        next_cursor: null,
      });
      return;
    }
    if (url.pathname === `/api/conversations/${CONVERSATION_ID}/messages`) {
      await json(route, { items: [], next_cursor: null });
      return;
    }
    if (url.pathname === "/api/agent" && agentHandler) {
      await agentHandler(route);
      return;
    }
    await route.abort("failed");
  });
}

async function json(route: Route, payload: unknown) {
  await route.fulfill({
    status: 200,
    contentType: "application/json",
    body: JSON.stringify(payload),
  });
}

function aguiStream(events: ReadonlyArray<Record<string, unknown>>): string {
  return events.map((event) => `data: ${JSON.stringify(event)}\n\n`).join("");
}

async function installHangingAgent(page: Page) {
  await page.addInitScript(() => {
    const originalFetch = window.fetch.bind(window);
    window.fetch = (input, init) => {
      const url = new URL(typeof input === "string" ? input : input instanceof URL ? input.href : input.url, location.href);
      if (url.pathname !== "/api/agent") return originalFetch(input, init);

      const encoder = new TextEncoder();
      let streamController: ReadableStreamDefaultController<Uint8Array>;
      const stream = new ReadableStream<Uint8Array>({
        start(controller) {
          streamController = controller;
          controller.enqueue(encoder.encode(
            `data: ${JSON.stringify({
              type: "RUN_STARTED",
              timestamp: Date.now(),
              threadId: "con_browser_test",
              runId: "run_hanging",
            })}\n\n`,
          ));
        },
      });
      init?.signal?.addEventListener("abort", () => {
        (window as Window & { __cancelObserved?: boolean }).__cancelObserved = true;
        streamController.error(new DOMException("Fetch is aborted", "AbortError"));
      }, { once: true });
      return Promise.resolve(new Response(stream, {
        status: 200,
        headers: { "Content-Type": "text/event-stream" },
      }));
    };
  });
}
