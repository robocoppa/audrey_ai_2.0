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
  await expect(page.getByRole("option", { name: "Video" })).toHaveCount(1);

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

test("keeps history and an active run alive while switching conversations", async ({ page }) => {
  const secondConversationId = "con_browser_second";
  const firstConversation = browserConversation("Running conversation");
  const secondConversation = {
    ...browserConversation("Second conversation"),
    id: secondConversationId,
  };
  await installNavigableAgent(page);
  await page.route("**/api/**", async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    if (url.pathname === "/api/me") {
      await json(route, browserUser());
      return;
    }
    if (url.pathname === "/api/conversations" && request.method() === "GET") {
      await json(route, {
        items: [firstConversation, secondConversation],
        next_cursor: null,
      });
      return;
    }
    if (url.pathname === `/api/conversations/${CONVERSATION_ID}/messages`) {
      await json(route, { items: canonicalBrowserTurn(), next_cursor: null });
      return;
    }
    if (url.pathname === `/api/conversations/${secondConversationId}/messages`) {
      await json(route, { items: [], next_cursor: null });
      return;
    }
    await route.abort("failed");
  });

  await page.goto("./");
  await expect(page.getByText("Canonical mode answer.")).toBeVisible();

  const composer = page.getByRole("textbox", { name: "Message Audrey" });
  await composer.fill("Keep this prompt while I visit another chat");
  await composer.press("Enter");
  await expect(page.getByText("Planning", { exact: true })).toBeVisible();
  await expect(page.getByText("Preparing the background answer")).toBeVisible();

  await page.getByRole("button", { name: /Second conversation/ }).click();
  await expect(page.getByRole("heading", { name: "Second conversation" })).toBeVisible();
  await expect(page.getByText("Canonical mode answer.")).toBeHidden();
  expect(await page.evaluate(() =>
    Boolean((window as Window & { __navigationRunAborted?: boolean }).__navigationRunAborted),
  )).toBe(false);

  await page.getByRole("button", { name: /Running conversation/ }).click();
  await expect(page.getByText("Canonical mode answer.")).toBeVisible();
  await expect(page.getByText("Keep this prompt while I visit another chat")).toBeVisible();
  await expect(page.getByText("Preparing the background answer")).toBeVisible();

  await page.evaluate(() => {
    (window as Window & { __finishNavigationRun?: () => void }).__finishNavigationRun?.();
  });
  await expect(page.getByText("Background response survived navigation.")).toBeVisible();
  await expect(page.getByText("Complete", { exact: true })).toBeVisible();

  await page.getByRole("button", { name: /Second conversation/ }).click();
  await page.getByRole("button", { name: /Running conversation/ }).click();
  await expect(page.getByText("Canonical mode answer.")).toBeVisible();
  await expect(page.getByText("Keep this prompt while I visit another chat")).toBeVisible();
  await expect(page.getByText("Background response survived navigation.")).toBeVisible();
});

test("searches, renames, archives, restores, and deletes a conversation", async ({ page }) => {
  let deleted = false;
  let conversation = browserConversation("Lifecycle conversation");
  const searchQueries: string[] = [];

  await page.route("**/api/**", async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    if (url.pathname === "/api/me") {
      await json(route, browserUser());
      return;
    }
    if (url.pathname === "/api/conversations" && request.method() === "GET") {
      const archived = url.searchParams.get("archived") === "true";
      const search = url.searchParams.get("q") ?? "";
      searchQueries.push(search);
      const matches = !deleted
        && Boolean(conversation.archived_at) === archived
        && conversation.title.toLocaleLowerCase().includes(search.toLocaleLowerCase());
      await json(route, { items: matches ? [conversation] : [], next_cursor: null });
      return;
    }
    if (url.pathname === `/api/conversations/${CONVERSATION_ID}/messages`) {
      await json(route, { items: [], next_cursor: null });
      return;
    }
    if (url.pathname === `/api/conversations/${CONVERSATION_ID}` && request.method() === "PATCH") {
      const patch = request.postDataJSON() as { title?: string; archived?: boolean };
      conversation = {
        ...conversation,
        ...(patch.title === undefined ? {} : { title: patch.title }),
        ...(patch.archived === undefined
          ? {}
          : { archived_at: patch.archived ? "2026-09-05T00:00:00Z" : null }),
      };
      await json(route, conversation);
      return;
    }
    if (url.pathname === `/api/conversations/${CONVERSATION_ID}` && request.method() === "DELETE") {
      deleted = true;
      await route.fulfill({ status: 204, body: "" });
      return;
    }
    await route.abort("failed");
  });

  await page.goto("./");
  await expect(page.getByRole("heading", { name: "Lifecycle conversation" })).toBeVisible();

  await page.getByRole("searchbox", { name: "Search titles" }).fill("life");
  await expect.poll(() => searchQueries.at(-1)).toBe("life");

  await page.getByRole("button", { name: "Rename conversation" }).click();
  await page.getByRole("textbox", { name: "Conversation title" }).fill("Lifecycle renamed");
  await page.getByRole("button", { name: "Save" }).click();
  await expect(page.getByRole("heading", { name: "Lifecycle renamed" })).toBeVisible();

  await page.getByRole("button", { name: "Archive", exact: true }).click();
  await expect(page.getByText("No matching conversation titles.")).toBeVisible();

  await page.getByRole("button", { name: "Archived" }).click();
  await expect(page.getByRole("heading", { name: "Lifecycle renamed" })).toBeVisible();
  await expect(page.getByRole("textbox", { name: "Message Audrey" })).toHaveCount(0);
  await expect(page.getByText("This conversation is archived. Restore it to continue.")).toBeVisible();

  await page.getByRole("button", { name: "Restore" }).click();
  await expect(page.getByText("No matching conversation titles.")).toBeVisible();

  await page.getByRole("button", { name: "Active" }).click();
  await expect(page.getByRole("heading", { name: "Lifecycle renamed" })).toBeVisible();
  await page.getByRole("button", { name: "Delete" }).click();
  await expect(page.getByRole("group", { name: "Confirm deletion" })).toBeVisible();
  await page.getByRole("button", { name: "Yes, delete" }).click();
  await expect(page.getByText("No matching conversation titles.")).toBeVisible();
});

test("loads older conversation pages without replacing the current page", async ({ page }) => {
  const older = { ...browserConversation("Older conversation"), id: "con_older" };
  await page.route("**/api/**", async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    if (url.pathname === "/api/me") {
      await json(route, browserUser());
      return;
    }
    if (url.pathname === "/api/conversations" && request.method() === "GET") {
      const cursor = url.searchParams.get("cursor");
      await json(route, cursor
        ? { items: [older], next_cursor: null }
        : {
            items: [browserConversation("Newest conversation")],
            next_cursor: "older-page",
          });
      return;
    }
    if (url.pathname.endsWith("/messages")) {
      await json(route, { items: [], next_cursor: null });
      return;
    }
    await route.abort("failed");
  });

  await page.goto("./");
  const history = page.getByRole("navigation", { name: "Conversation history" });
  await expect(history.getByText("Newest conversation", { exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Load older" }).click();
  await expect(history.getByText("Older conversation", { exact: true })).toBeVisible();
  await expect(history.getByText("Newest conversation", { exact: true })).toBeVisible();
  await expect(page.getByRole("button", { name: "Load older" })).toHaveCount(0);
});

test("keeps canonical messages when changing mode", async ({ page }) => {
  let completed = false;
  let conversation = browserConversation("Mode persistence");
  const agentModes: string[] = [];

  await page.route("**/api/**", async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    if (url.pathname === "/api/me") {
      await json(route, browserUser());
      return;
    }
    if (url.pathname === "/api/conversations" && request.method() === "GET") {
      await json(route, { items: [conversation], next_cursor: null });
      return;
    }
    if (url.pathname === `/api/conversations/${CONVERSATION_ID}/messages`) {
      await json(route, {
        items: completed ? canonicalBrowserTurn() : [],
        next_cursor: null,
      });
      return;
    }
    if (url.pathname === `/api/conversations/${CONVERSATION_ID}` && request.method() === "PATCH") {
      const patch = request.postDataJSON() as { default_mode?: "fast" | "deep" };
      conversation = {
        ...conversation,
        default_mode: patch.default_mode ?? conversation.default_mode,
      };
      await json(route, conversation);
      return;
    }
    if (url.pathname === "/api/agent") {
      completed = true;
      agentModes.push(url.searchParams.get("mode") ?? "");
      await route.fulfill({
        status: 200,
        contentType: "text/event-stream",
        body: aguiStream(canonicalBrowserEvents()),
      });
      return;
    }
    await route.abort("failed");
  });

  await page.goto("./");
  const composer = page.getByRole("textbox", { name: "Message Audrey" });
  await composer.fill("First mode turn");
  await composer.press("Enter");
  await expect(page.getByText("Canonical mode answer.")).toBeVisible();

  await page.getByRole("combobox", { name: "Mode" }).selectOption("deep");
  await expect(page.getByRole("combobox", { name: "Mode" })).toHaveValue("deep");
  await expect(page.getByText("Canonical mode answer.")).toBeVisible();
  expect(agentModes).toEqual(["fast"]);
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

function browserUser() {
  return {
    id: "usr_browser_test",
    email: "alice@example.com",
    display_name: "Alice",
    role: "user",
    status: "active",
    auth_provider: "cloudflare_access",
  };
}

function browserConversation(title: string) {
  return {
    id: CONVERSATION_ID,
    title,
    default_mode: "fast" as const,
    created_at: "2026-09-04T00:00:00Z",
    updated_at: "2026-09-04T00:00:00Z",
    last_message_at: null,
    archived_at: null as string | null,
  };
}

function canonicalBrowserTurn() {
  return [
    {
      id: "msg_mode_user",
      run_id: "run_mode",
      sequence: 1,
      role: "user",
      status: "completed",
      content: "First mode turn",
      created_at: "2026-09-05T00:00:00Z",
      updated_at: "2026-09-05T00:00:00Z",
    },
    {
      id: "msg_mode_assistant",
      run_id: "run_mode",
      sequence: 2,
      role: "assistant",
      status: "completed",
      content: "Canonical mode answer.",
      created_at: "2026-09-05T00:00:01Z",
      updated_at: "2026-09-05T00:00:01Z",
    },
  ];
}

function canonicalBrowserEvents(): ReadonlyArray<Record<string, unknown>> {
  return [
    { type: "RUN_STARTED", timestamp: 1, threadId: CONVERSATION_ID, runId: "run_mode" },
    { type: "TEXT_MESSAGE_START", timestamp: 2, messageId: "msg_mode_assistant" },
    {
      type: "TEXT_MESSAGE_CONTENT",
      timestamp: 3,
      messageId: "msg_mode_assistant",
      delta: "Canonical mode answer.",
    },
    { type: "TEXT_MESSAGE_END", timestamp: 4, messageId: "msg_mode_assistant" },
    {
      type: "RUN_FINISHED",
      timestamp: 5,
      threadId: CONVERSATION_ID,
      runId: "run_mode",
      outcome: { type: "success" },
    },
  ];
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

async function installNavigableAgent(page: Page) {
  await page.addInitScript(() => {
    const originalFetch = window.fetch.bind(window);
    window.fetch = (input, init) => {
      const url = new URL(
        typeof input === "string"
          ? input
          : input instanceof URL
            ? input.href
            : input.url,
        location.href,
      );
      if (url.pathname !== "/api/agent") return originalFetch(input, init);

      const encoder = new TextEncoder();
      const stream = new ReadableStream<Uint8Array>({
        start(controller) {
          const emit = (event: Record<string, unknown>) => {
            controller.enqueue(encoder.encode(`data: ${JSON.stringify(event)}\n\n`));
          };
          emit({
            type: "RUN_STARTED",
            timestamp: 1,
            threadId: "con_browser_test",
            runId: "run_navigation",
          });
          emit({
            type: "TEXT_MESSAGE_START",
            timestamp: 2,
            messageId: "msg_navigation_assistant",
          });
          emit({ type: "STEP_STARTED", timestamp: 3, stepName: "planning" });
          emit({
            type: "CUSTOM",
            timestamp: 4,
            name: "audrey.stage.progress",
            value: {
              stage: "planning",
              delta: "Preparing the background answer",
            },
          });
          const controls = window as Window & {
            __finishNavigationRun?: () => void;
            __navigationRunAborted?: boolean;
          };
          controls.__navigationRunAborted = false;
          controls.__finishNavigationRun = () => {
            emit({ type: "STEP_FINISHED", timestamp: 5, stepName: "planning" });
            emit({
              type: "TEXT_MESSAGE_CONTENT",
              timestamp: 6,
              messageId: "msg_navigation_assistant",
              delta: "Background response survived navigation.",
            });
            emit({
              type: "TEXT_MESSAGE_END",
              timestamp: 7,
              messageId: "msg_navigation_assistant",
            });
            emit({
              type: "RUN_FINISHED",
              timestamp: 8,
              threadId: "con_browser_test",
              runId: "run_navigation",
              outcome: { type: "success" },
            });
            controller.close();
          };
          init?.signal?.addEventListener("abort", () => {
            controls.__navigationRunAborted = true;
            controller.error(new DOMException("Fetch is aborted", "AbortError"));
          }, { once: true });
        },
      });
      return Promise.resolve(new Response(stream, {
        status: 200,
        headers: { "Content-Type": "text/event-stream" },
      }));
    };
  });
}
