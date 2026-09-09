import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Page, type Route } from "@playwright/test";

const CONVERSATION_ID = "con_browser_test";

test("centers Audrey Auto with a text-free orbit while the session loads", async ({ page }) => {
  let releaseSession: () => void = () => undefined;
  const sessionGate = new Promise<void>((resolve) => {
    releaseSession = resolve;
  });
  await page.route("**/api/**", async (route) => {
    const url = new URL(route.request().url());
    if (url.pathname === "/api/me/preferences") {
      await sessionGate;
      await json(route, browserPreferences());
      return;
    }
    if (url.pathname === "/api/me") {
      await sessionGate;
      await json(route, browserUser());
      return;
    }
    if (url.pathname === "/api/conversations" && route.request().method() === "GET") {
      await json(route, { items: [browserConversation("Loading workspace")], next_cursor: null });
      return;
    }
    if (url.pathname === `/api/conversations/${CONVERSATION_ID}/messages`) {
      await json(route, { items: [], next_cursor: null });
      return;
    }
    await route.abort("failed");
  });

  await page.goto("./", { waitUntil: "domcontentloaded" });

  const loader = page.getByRole("status", { name: "Loading Audrey" });
  const portrait = loader.locator("img");
  const orbit = loader.locator(".audrey-loading-orbit");
  await expect(loader).toBeVisible();
  await expect(loader).toHaveText("");
  await expect(portrait).toBeVisible();
  await expect.poll(
    () => portrait.evaluate((image) => (image as HTMLImageElement).naturalWidth),
  ).toBeGreaterThan(0);
  await expect.poll(
    () => orbit.evaluate((element) => getComputedStyle(element).animationName),
  ).toBe("audrey-loading-orbit");
  await expect.poll(
    () => portrait.evaluate((image) => getComputedStyle(image).opacity),
  ).toBe("0.8");

  const loaderBox = await loader.boundingBox();
  const portraitBox = await portrait.boundingBox();
  const viewport = page.viewportSize();
  expect(loaderBox).not.toBeNull();
  expect(portraitBox).not.toBeNull();
  expect(viewport).not.toBeNull();
  expect(Math.abs((portraitBox?.x ?? 0) + (portraitBox?.width ?? 0) / 2 - (viewport?.width ?? 0) / 2))
    .toBeLessThan(2);
  expect(Math.abs((portraitBox?.y ?? 0) + (portraitBox?.height ?? 0) / 2 - (viewport?.height ?? 0) / 2))
    .toBeLessThan(2);

  releaseSession();
  await expect(page.getByRole("link", { name: "Audrey home" })).toBeVisible();
});

test("recovers a transient Access bootstrap rejection", async ({ page }) => {
  const sessionRequests: string[] = [];
  let identityReads = 0;

  await page.route("**/api/**", async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    if (url.pathname === "/api/me") {
      identityReads += 1;
      sessionRequests.push(url.pathname);
      if (identityReads === 1) {
        await route.fulfill({
          status: 401,
          contentType: "application/json",
          body: JSON.stringify({ detail: "Missing bearer token." }),
        });
        return;
      }
      await json(route, browserUser());
      return;
    }
    if (url.pathname === "/api/me/preferences") {
      sessionRequests.push(url.pathname);
      await json(route, browserPreferences());
      return;
    }
    if (url.pathname === "/api/conversations" && request.method() === "GET") {
      await json(route, {
        items: [browserConversation("Recovered Access session")],
        next_cursor: null,
      });
      return;
    }
    if (url.pathname === `/api/conversations/${CONVERSATION_ID}/messages`) {
      await json(route, { items: [], next_cursor: null });
      return;
    }
    await route.abort("failed");
  });

  await page.goto("./");

  await expect(page.getByRole("heading", { name: "Recovered Access session" })).toBeVisible();
  expect(sessionRequests).toEqual([
    "/api/me",
    "/api/me",
    "/api/me/preferences",
  ]);
  await expect(page.getByText("A quieter place to think.")).toHaveCount(0);
  await expect(page.getByText("Not connected")).toHaveCount(0);
});

test("restores a saved conversation on hard refresh without showing a landing page", async ({ page }) => {
  let conversationListReads = 0;
  let messageReads = 0;
  let releaseReloadList: () => void = () => undefined;
  let releaseReloadMessages: () => void = () => undefined;
  const reloadListGate = new Promise<void>((resolve) => {
    releaseReloadList = resolve;
  });
  const reloadMessagesGate = new Promise<void>((resolve) => {
    releaseReloadMessages = resolve;
  });

  await page.route("**/api/**", async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    if (url.pathname === "/api/me/preferences") {
      await json(route, browserPreferences());
      return;
    }
    if (url.pathname === "/api/me") {
      await json(route, browserUser());
      return;
    }
    if (url.pathname === "/api/conversations" && request.method() === "GET") {
      conversationListReads += 1;
      if (conversationListReads === 2) await reloadListGate;
      await json(route, {
        items: [browserConversation("Hard refresh history")],
        next_cursor: null,
      });
      return;
    }
    if (url.pathname === `/api/conversations/${CONVERSATION_ID}/messages`) {
      messageReads += 1;
      if (messageReads === 2) await reloadMessagesGate;
      await json(route, { items: canonicalBrowserTurn(), next_cursor: null });
      return;
    }
    await route.abort("failed");
  });

  await page.goto("./");
  await expect(page.getByRole("heading", { name: "Hard refresh history" })).toBeVisible();
  await expect(page.getByText("Canonical mode answer.")).toBeVisible();

  await page.reload({ waitUntil: "domcontentloaded" });
  await expect.poll(() => conversationListReads).toBe(2);
  const opening = page.getByRole("status", { name: "Opening conversation" });
  await expect(opening).toBeAttached();
  await expect(opening.locator("img, .audrey-loading-orbit")).toHaveCount(0);
  await expect(page.getByText("What shall we work through?")).toHaveCount(0);

  releaseReloadList();
  const loading = page.getByRole("status", { name: "Loading conversation" });
  await expect(loading).toBeAttached();
  await expect(loading.locator("img, .audrey-loading-orbit")).toHaveCount(0);

  releaseReloadMessages();
  await expect(page.getByRole("heading", { name: "Hard refresh history" })).toBeVisible();
  await expect(page.getByText("Canonical mode answer.")).toBeVisible();
});

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
          delta: "Browser-native **answer** with `code`.\n\n- First\n- Second",
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
  await expect(page).toHaveTitle("Audrey by Builtryte");
  await expect.poll(
    () => page.evaluate(() => getComputedStyle(document.documentElement).colorScheme),
  ).toBe("dark");
  await expect.poll(
    () => page.evaluate(() => getComputedStyle(document.documentElement).backgroundColor),
  ).toBe("rgb(7, 16, 31)");
  const brandWordmark = page.locator(".brand-wordmark img");
  await expect(brandWordmark).toBeVisible();
  await expect(page.locator("#light-wordmark-on-dark feColorMatrix")).toHaveAttribute(
    "values",
    /0\.843.*0\.886.*0\.945/u,
  );
  await expect.poll(
    () => brandWordmark.evaluate((image) => (image as HTMLImageElement).naturalWidth),
  ).toBeGreaterThan(0);
  await expect(brandWordmark).toHaveAttribute("src", /builtryte-wordmark/u);
  await expect(page.locator('link[rel="icon"]')).toHaveAttribute(
    "href",
    /builtryte-favicon/u,
  );
  await expect(page.getByRole("heading", { name: "Browser smoke" })).toBeVisible();
  await expect(page.locator(".brand-product")).toHaveText("Ask Audrey");
  await expect.poll(
    () => page.locator(".brand-product").evaluate(
      (element) => Number.parseFloat(getComputedStyle(element).fontSize),
    ),
  ).toBeGreaterThanOrEqual(16);
  await expect(page.getByRole("combobox", { name: "Audrey model" })).toHaveValue("fast");
  await expect(page.getByRole("option", { name: "Video" })).toHaveCount(1);
  await expect(page.getByLabel("Signed in user")).toContainText("Alice");
  await expect(page.getByRole("link", { name: "Log out" })).toHaveAttribute(
    "href",
    "/cdn-cgi/access/logout",
  );
  const portrait = page.locator(".composer-model-picker img");
  await expect(portrait).toBeVisible();
  const modelPicker = page.getByRole("combobox", { name: "Audrey model" });
  await modelPicker.selectOption("research");
  await expect(portrait).toHaveAttribute("src", /audrey8/u);
  await modelPicker.selectOption("video");
  await expect(portrait).toHaveAttribute("src", /audrey9/u);
  await expect.poll(() => portrait.evaluate((image) => (image as HTMLImageElement).naturalWidth)).toBeGreaterThan(0);
  await modelPicker.selectOption("fast");
  await expect(portrait).toHaveAttribute("src", /audrey3/u);
  await expect.poll(() => portrait.evaluate((image) => (image as HTMLImageElement).naturalWidth)).toBeGreaterThan(0);
  await expect(page.getByText("Quick, direct answers for everyday questions and tasks.")).toBeVisible();
  await expect(page.locator(".model-picker-copy")).toHaveCount(0);
  await expect(page.getByRole("heading", { name: "Ask Audrey", exact: true })).toHaveCount(0);
  await expect(page.getByText("The server will load this conversation's canonical history.")).toHaveCount(0);
  const portraitBox = await portrait.boundingBox();
  const pickerBox = await page.getByRole("combobox", { name: "Audrey model" }).boundingBox();
  const composerBox = await page.locator(".composer").boundingBox();
  const viewport = page.viewportSize();
  expect(portraitBox).not.toBeNull();
  expect(pickerBox).not.toBeNull();
  expect(composerBox).not.toBeNull();
  expect(viewport).not.toBeNull();
  expect(portraitBox?.width ?? 0).toBeGreaterThanOrEqual(200);
  expect((pickerBox?.y ?? 0) - ((portraitBox?.y ?? 0) + (portraitBox?.height ?? 0)))
    .toBeGreaterThanOrEqual(18);
  expect((pickerBox?.y ?? 0) - ((portraitBox?.y ?? 0) + (portraitBox?.height ?? 0)))
    .toBeLessThanOrEqual(22);
  await expect.poll(
    () => portrait.evaluate((element) => Number.parseFloat(getComputedStyle(element).opacity)),
  ).toBe(0.8);
  await portrait.hover();
  await expect.poll(
    () => portrait.evaluate((element) => Number.parseFloat(getComputedStyle(element).opacity)),
  ).toBe(1);
  await expect(page.locator(".composer-model-picker")).toHaveCSS("isolation", "isolate");
  await expect(page.locator(".model-picker-control")).toHaveCSS("z-index", "1");
  expect((composerBox?.y ?? 0) + (composerBox?.height ?? 0)).toBeLessThanOrEqual(viewport?.height ?? 0);
  expect(Math.abs(
    (portraitBox?.x ?? 0) + (portraitBox?.width ?? 0) / 2
      - ((composerBox?.x ?? 0) + (composerBox?.width ?? 0) / 2),
  )).toBeLessThan(2);

  await expect.poll(
    () => page.locator(".model-description").evaluate((element) => Number.parseFloat(getComputedStyle(element).fontSize)),
  ).toBeGreaterThanOrEqual(14);
  await expect.poll(
    () => page.getByRole("combobox", { name: "Audrey model" }).evaluate((element) => Number.parseFloat(getComputedStyle(element).fontSize)),
  ).toBeGreaterThanOrEqual(14);
  await expect(page.getByRole("button", { name: "Send message" }).locator("svg")).toBeVisible();

  const composer = page.getByRole("textbox", { name: "Ask Audrey" });
  await composer.focus();
  await expect.poll(() => composer.evaluate((input) => getComputedStyle(input).outlineStyle)).toBe("none");
  await expect.poll(() => page.locator(".composer").evaluate((root) => getComputedStyle(root).boxShadow)).not.toBe("none");
  await composer.fill("Exercise the native browser path");
  await composer.press("Enter");

  const assistantMessage = page.locator(".message-assistant");
  await expect(assistantMessage.getByText("answer", { exact: true })).toHaveJSProperty("tagName", "STRONG");
  await expect(assistantMessage.getByText("code", { exact: true })).toHaveJSProperty("tagName", "CODE");
  await expect(assistantMessage.getByRole("listitem")).toHaveCount(2);
  await expect(assistantMessage).not.toContainText("**answer**");
  await expect(page.getByText("web_search · complete")).toBeVisible();
  await expect(page.getByText("1 source · Official source")).toBeVisible();
  await expect(page.getByText("Complete", { exact: true })).toBeVisible();
  await expect(page.locator(".composer-model-picker")).toHaveCount(0);
  await expect(page.locator(".composer .compact-model-picker")).toBeVisible();
  await expect(page.locator(".model-description")).toHaveCount(0);
  expect(requestBody).toMatchObject({ threadId: CONVERSATION_ID });
  expect(requestBody?.messages).toHaveLength(1);

  const secrets = await page.evaluate(() => ({
    local: Object.keys(localStorage),
    session: Object.keys(sessionStorage),
  }));
  expect(secrets).toEqual({ local: [], session: [] });

  const accessibility = await new AxeBuilder({ page }).analyze();
  expect(accessibility.violations).toEqual([]);
});

test("hides the run summary when the saved presentation preference is off", async ({ page }) => {
  await mockAudreyApi(
    page,
    async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "text/event-stream",
        body: aguiStream([
          { type: "RUN_STARTED", timestamp: 1, threadId: CONVERSATION_ID, runId: "run_quiet" },
          { type: "TEXT_MESSAGE_START", timestamp: 2, messageId: "msg_quiet" },
          { type: "STEP_STARTED", timestamp: 3, stepName: "thinking" },
          {
            type: "CUSTOM",
            timestamp: 4,
            name: "audrey.stage.progress",
            value: { stage: "thinking", delta: "Working quietly" },
          },
          { type: "STEP_FINISHED", timestamp: 5, stepName: "thinking" },
          {
            type: "TEXT_MESSAGE_CONTENT",
            timestamp: 6,
            messageId: "msg_quiet",
            delta: "Quiet preference answer.",
          },
          { type: "TEXT_MESSAGE_END", timestamp: 7, messageId: "msg_quiet" },
          {
            type: "RUN_FINISHED",
            timestamp: 8,
            threadId: CONVERSATION_ID,
            runId: "run_quiet",
            outcome: { type: "success" },
          },
        ]),
      });
    },
    [],
    browserPreferences({ show_progress: false }),
  );

  await page.goto("./");
  const composer = page.getByRole("textbox", { name: "Ask Audrey" });
  await composer.fill("Use quiet progress");
  await composer.press("Enter");

  await expect(page.getByText("Quiet preference answer.")).toBeVisible();
  await expect(page.locator(".run-activity")).toHaveCount(0);
  await expect(page.getByText("Working quietly")).toHaveCount(0);
});

test("summarizes a new conversation from its first prompt", async ({ page }) => {
  let created = false;
  let createBody: unknown = null;
  let conversation = {
    ...browserConversation(""),
    default_mode: "auto" as const,
  };

  await page.route("**/api/**", async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    if (url.pathname === "/api/me/preferences") {
      await json(route, browserPreferences());
      return;
    }
    if (url.pathname === "/api/me") {
      await json(route, browserUser());
      return;
    }
    if (url.pathname === "/api/conversations" && request.method() === "GET") {
      await json(route, { items: created ? [conversation] : [], next_cursor: null });
      return;
    }
    if (url.pathname === "/api/conversations" && request.method() === "POST") {
      createBody = request.postDataJSON();
      created = true;
      await json(route, conversation);
      return;
    }
    if (url.pathname === `/api/conversations/${CONVERSATION_ID}/messages`) {
      await json(route, { items: [], next_cursor: null });
      return;
    }
    if (
      url.pathname === `/api/conversations/${CONVERSATION_ID}`
      && request.method() === "GET"
    ) {
      await json(route, conversation);
      return;
    }
    if (url.pathname === "/api/agent") {
      conversation = {
        ...conversation,
        title: "Weekend Hiking Trip Planning",
      };
      await route.fulfill({
        status: 200,
        contentType: "text/event-stream",
        body: aguiStream([
          {
            type: "RUN_STARTED",
            timestamp: 1,
            threadId: CONVERSATION_ID,
            runId: "run_automatic_title",
          },
          { type: "TEXT_MESSAGE_START", timestamp: 2, messageId: "msg_title" },
          {
            type: "TEXT_MESSAGE_CONTENT",
            timestamp: 3,
            messageId: "msg_title",
            delta: "Here is the hiking plan.",
          },
          { type: "TEXT_MESSAGE_END", timestamp: 4, messageId: "msg_title" },
          {
            type: "RUN_FINISHED",
            timestamp: 5,
            threadId: CONVERSATION_ID,
            runId: "run_automatic_title",
            outcome: { type: "success" },
          },
        ]),
      });
      return;
    }
    await route.abort("failed");
  });

  await page.goto("./");
  await expect(page.getByRole("heading", { name: "New conversation" })).toBeVisible();
  await expect(page.locator(".composer-model-picker img")).toBeVisible();
  await expect(page.getByRole("textbox", { name: "Ask Audrey" })).toBeVisible();
  await expect(page.getByText("What shall we work through?")).toHaveCount(0);

  const prompt = "Plan a weekend hiking trip with a packing list";
  await page.getByRole("textbox", { name: "Ask Audrey" }).fill(prompt);
  await page.getByRole("textbox", { name: "Ask Audrey" }).press("Enter");

  const summaryTitle = "Weekend Hiking Trip Planning";
  await expect(page.getByRole("heading", { name: summaryTitle })).toBeVisible();
  await expect(
    page.getByRole("navigation", { name: "Conversation history" }).getByText(summaryTitle),
  ).toBeVisible();
  expect(createBody).toEqual({ default_mode: "auto" });
});

test("keeps the composer docked and returns to the latest message", async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 720 });
  await mockAudreyApi(page, undefined, scrollingBrowserHistory());
  await page.goto("./");

  const viewport = page.locator(".thread-viewport");
  const dock = page.locator(".composer-dock");
  const jump = page.getByRole("button", { name: "Scroll to latest message" });

  await expect.poll(() => viewport.evaluate(
    (element) => element.scrollHeight > element.clientHeight + 100,
  )).toBe(true);
  await expect(jump).toBeHidden();

  const dockAtLatest = await dock.boundingBox();
  const viewportBox = await viewport.boundingBox();
  expect(dockAtLatest).not.toBeNull();
  expect(viewportBox).not.toBeNull();

  await viewport.evaluate((element) => element.scrollTo({ top: 0 }));
  await expect(jump).toBeVisible();
  await expect(jump.locator("svg")).toBeVisible();

  const dockWhileReading = await dock.boundingBox();
  expect(dockWhileReading).not.toBeNull();
  expect(Math.abs((dockWhileReading?.y ?? 0) - (dockAtLatest?.y ?? 0))).toBeLessThan(2);
  expect(
    (dockWhileReading?.y ?? 0) + (dockWhileReading?.height ?? 0),
  ).toBeLessThanOrEqual(
    (viewportBox?.y ?? 0) + (viewportBox?.height ?? 0) + 1,
  );

  await jump.click();
  await expect.poll(() => viewport.evaluate((element) => (
    element.scrollTop + element.clientHeight >= element.scrollHeight - 2
  ))).toBe(true);
  await expect(jump).toBeHidden();
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
    if (url.pathname === "/api/me/preferences") {
      await json(route, browserPreferences());
      return;
    }
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

  const composer = page.getByRole("textbox", { name: "Ask Audrey" });
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
    if (url.pathname === "/api/me/preferences") {
      await json(route, browserPreferences());
      return;
    }
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
  await expect(page.getByRole("textbox", { name: "Ask Audrey" })).toHaveCount(0);
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
    if (url.pathname === "/api/me/preferences") {
      await json(route, browserPreferences());
      return;
    }
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
    if (url.pathname === "/api/me/preferences") {
      await json(route, browserPreferences());
      return;
    }
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
  const composer = page.getByRole("textbox", { name: "Ask Audrey" });
  await expect(page.locator(".composer-model-picker img")).toBeVisible();
  await composer.fill("First mode turn");
  await composer.press("Enter");
  await expect(page.getByText("Canonical mode answer.")).toBeVisible();

  await expect(page.locator(".composer-model-picker")).toHaveCount(0);
  await expect(page.locator(".composer .compact-model-picker")).toBeVisible();
  await page.getByRole("combobox", { name: "Audrey model" }).selectOption("deep");
  await expect(page.getByRole("combobox", { name: "Audrey model" })).toHaveValue("deep");
  await expect(page.getByText("A reasoning panel for complex problems and careful analysis.")).toHaveCount(0);
  await expect(page.getByText("Canonical mode answer.")).toBeVisible();
  expect(agentModes).toEqual(["fast"]);
});

test("sets and retains the current user's profile name", async ({ page }) => {
  let profile = { ...browserUser(), display_name: "" };
  const profilePatches: unknown[] = [];

  await page.route("**/api/**", async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    if (url.pathname === "/api/me" && request.method() === "PATCH") {
      const payload = request.postDataJSON() as { display_name: string };
      profilePatches.push(payload);
      profile = { ...profile, display_name: payload.display_name.trim() };
      await json(route, profile);
      return;
    }
    if (url.pathname === "/api/me/preferences") {
      await json(route, browserPreferences());
      return;
    }
    if (url.pathname === "/api/me") {
      await json(route, profile);
      return;
    }
    if (url.pathname === "/api/conversations" && request.method() === "GET") {
      await json(route, { items: [browserConversation("Profile conversation")], next_cursor: null });
      return;
    }
    if (url.pathname === `/api/conversations/${CONVERSATION_ID}/messages`) {
      await json(route, { items: [], next_cursor: null });
      return;
    }
    await route.abort("failed");
  });

  await page.goto("./");
  const profileButton = page.getByRole("button", { name: "Open account settings" });
  await expect(profileButton).toHaveText("alice");
  await profileButton.click();
  await page.getByRole("textbox", { name: "Profile name" }).fill("Alice Builder");
  await page.getByRole("button", { name: "Save profile" }).click();

  await expect(profileButton).toHaveText("Alice");
  await expect(page.locator(".sidebar-heading strong")).toHaveText("Alice Builder");
  expect(profilePatches).toEqual([{ display_name: "Alice Builder" }]);

  await page.reload();
  await expect(page.getByRole("button", { name: "Open account settings" })).toHaveText("Alice");
  await expect(page.getByRole("heading", { name: "Profile conversation" })).toBeVisible();
  const accessibility = await new AxeBuilder({ page }).analyze();
  expect(accessibility.violations).toEqual([]);
});

test("saves and reloads native Audrey preferences", async ({ page }) => {
  let preferences = browserPreferences();
  const updates: unknown[] = [];

  await page.route("**/api/**", async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    if (url.pathname === "/api/me/preferences" && request.method() === "PUT") {
      const payload = request.postDataJSON() as Record<string, unknown>;
      updates.push(payload);
      preferences = browserPreferences(payload);
      await json(route, preferences);
      return;
    }
    if (url.pathname === "/api/me/preferences") {
      await json(route, preferences);
      return;
    }
    if (url.pathname === "/api/me") {
      await json(route, browserUser());
      return;
    }
    if (url.pathname === "/api/conversations" && request.method() === "GET") {
      await json(route, { items: [browserConversation("Preferences conversation")], next_cursor: null });
      return;
    }
    if (url.pathname === `/api/conversations/${CONVERSATION_ID}/messages`) {
      await json(route, { items: [], next_cursor: null });
      return;
    }
    await route.abort("failed");
  });

  await page.goto("./");
  await page.getByRole("button", { name: "Open account settings" }).click();
  const dialog = page.getByRole("dialog", { name: "Settings" });
  await dialog.getByRole("textbox", { name: "Timezone" }).fill("America/Denver");
  await dialog.getByRole("textbox", { name: "Persona and style" }).fill(
    "Be direct and explain uncommon terms.",
  );
  await dialog.getByRole("combobox", { name: "Response detail" }).selectOption("detailed");
  await dialog.getByRole("combobox", { name: "Tone" }).selectOption("professional");
  await dialog.getByRole("checkbox", {
    name: "Show the live stage and source summary above the composer",
  }).uncheck();
  await dialog.getByRole("button", { name: "Save preferences" }).click();

  await expect.poll(() => updates.length).toBe(1);
  expect(updates[0]).toEqual({
    timezone: "America/Denver",
    persona: "Be direct and explain uncommon terms.",
    detail: "detailed",
    tone: "professional",
    show_progress: false,
  });

  await page.reload();
  await page.getByRole("button", { name: "Open account settings" }).click();
  const reloaded = page.getByRole("dialog", { name: "Settings" });
  await expect(reloaded.getByRole("textbox", { name: "Timezone" })).toHaveValue(
    "America/Denver",
  );
  await expect(reloaded.getByRole("textbox", { name: "Persona and style" })).toHaveValue(
    "Be direct and explain uncommon terms.",
  );
  await expect(reloaded.getByRole("combobox", { name: "Response detail" })).toHaveValue(
    "detailed",
  );
  await expect(reloaded.getByRole("combobox", { name: "Tone" })).toHaveValue(
    "professional",
  );
  await expect(reloaded.getByRole("checkbox", {
    name: "Show the live stage and source summary above the composer",
  })).not.toBeChecked();
});
test("manages personal tokens through the production browser bundle", async ({ page }) => {
  const requests: unknown[] = [];
  let tokens = [{
    id: "pat_browser_existing",
    name: "Existing client",
    scopes: ["account:read"],
    created_at: "2026-09-08T00:00:00Z",
    expires_at: "2099-10-08T00:00:00Z",
    last_used_at: null,
    revoked_at: null,
  }];

  await page.route("**/api/**", async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    if (url.pathname === "/api/tokens" && request.method() === "POST") {
      const payload = request.postDataJSON() as {
        name: string;
        scopes: string[];
        expires_in_days: number;
      };
      requests.push(payload);
      const created = {
        id: "pat_browser_created",
        name: payload.name,
        scopes: payload.scopes,
        created_at: "2026-09-08T00:00:00Z",
        expires_at: "2099-10-08T00:00:00Z",
        last_used_at: null,
        revoked_at: null,
      };
      tokens = [created, ...tokens];
      await route.fulfill({
        status: 201,
        contentType: "application/json",
        body: JSON.stringify({
          ...created,
          token: "aud_pat_browser_created.one-time-secret",
        }),
      });
      return;
    }
    if (url.pathname === "/api/tokens" && request.method() === "GET") {
      await json(route, { items: tokens });
      return;
    }
    if (url.pathname === "/api/tokens/pat_browser_created" && request.method() === "DELETE") {
      tokens = tokens.filter(({ id }) => id !== "pat_browser_created");
      await json(route, { id: "pat_browser_created", revoked: true });
      return;
    }
    if (url.pathname === "/api/me/preferences") {
      await json(route, browserPreferences());
      return;
    }
    if (url.pathname === "/api/me") {
      await json(route, browserUser());
      return;
    }
    if (url.pathname === "/api/conversations" && request.method() === "GET") {
      await json(route, { items: [browserConversation("Token conversation")], next_cursor: null });
      return;
    }
    if (url.pathname === `/api/conversations/${CONVERSATION_ID}/messages`) {
      await json(route, { items: [], next_cursor: null });
      return;
    }
    await route.abort("failed");
  });

  await page.goto("./");
  await page.getByRole("button", { name: "Open account settings" }).click();
  const dialog = page.getByRole("dialog", { name: "Settings" });
  await dialog.getByRole("button", { name: "Manage personal tokens" }).click();
  await expect(dialog.getByText("Existing client")).toBeVisible();

  await dialog.getByRole("textbox", { name: "Token name" }).fill("Browser CLI");
  await dialog.getByRole("spinbutton", { name: "Token lifetime in days" }).fill("30");
  await dialog.getByRole("button", { name: "Create token" }).click();

  await expect(dialog.getByRole("textbox", { name: "New personal token" })).toHaveValue(
    "aud_pat_browser_created.one-time-secret",
  );
  await expect(dialog.getByRole("button", { name: "Close settings" })).toBeDisabled();
  expect(requests).toEqual([{
    name: "Browser CLI",
    scopes: ["compat:full"],
    expires_in_days: 30,
  }]);
  expect(await page.evaluate(() => ({
    local: localStorage.length,
    session: sessionStorage.length,
  }))).toEqual({ local: 0, session: 0 });

  await dialog.getByRole("button", { name: "I saved it" }).click();
  await expect(dialog.getByRole("button", { name: "Close settings" })).toBeEnabled();
  await dialog.getByRole("button", { name: "Revoke Browser CLI" }).click();
  await dialog.getByRole("button", { name: "Confirm revoke" }).click();
  await expect(dialog.getByText("Browser CLI")).toHaveCount(0);
  await expect(dialog.getByText("Existing client")).toBeVisible();

  const accessibility = await new AxeBuilder({ page }).analyze();
  expect(accessibility.violations).toEqual([]);
});

test("exports archived chat and durably deletes Audrey data", async ({ page }) => {
  const userDataAuthorization: Array<string | undefined> = [];
  const purgeRequests: Array<{
    body: unknown;
    idempotencyKey: string | undefined;
  }> = [];
  let purgeStatusReads = 0;
  const exportMessage = {
    message_id: "msg_browser_export",
    conversation_id: "con_browser_export",
    conversation_title: "Browser export",
    conversation_created_at: "2026-09-09T00:00:00Z",
    conversation_updated_at: "2026-09-09T00:01:00Z",
    role: "assistant",
    content: "Portable archived answer.",
    created_at: "2026-09-09T00:01:00Z",
    archived_at: "2026-09-09T00:02:00Z",
    partial: false,
    virtual_model: "audrey_fast",
    concrete_model: "browser-model",
    prompt_tokens: 10,
    completion_tokens: 20,
  };
  const purgeReceipt = (status: "pending" | "completed") => ({
    schema_version: 1,
    purge_id: "purge_browser_data",
    cutoff_at: "2026-09-09T00:03:00Z",
    requested_at: "2026-09-09T00:03:00Z",
    status,
    completed_at: status === "completed" ? "2026-09-09T00:03:02Z" : "",
    files: {
      pending: status === "completed" ? 0 : 1,
      attempts: 1,
      with_error: 0,
      completed: status === "completed" ? 1 : 0,
    },
    paths: {
      pending: status === "completed" ? 0 : 1,
      attempts: 1,
      with_error: 0,
      completed: status === "completed" ? 1 : 0,
    },
    local_delivery: { completed: true, attempts: 1, with_error: false },
    sidecar: {
      acknowledged: true,
      completed: status === "completed",
      status,
      attempts: 1,
      with_error: false,
    },
  });

  await page.route("**/v1/me/**", async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    userDataAuthorization.push(request.headers().authorization);
    if (url.pathname === "/v1/me/chat-history/export") {
      await json(route, {
        schema_version: 1,
        items: [exportMessage],
        next_cursor: null,
      });
      return;
    }
    if (url.pathname === "/v1/me/data-purge" && request.method() === "POST") {
      purgeRequests.push({
        body: request.postDataJSON(),
        idempotencyKey: request.headers()["idempotency-key"],
      });
      await route.fulfill({
        status: 202,
        contentType: "application/json",
        body: JSON.stringify(purgeReceipt("pending")),
      });
      return;
    }
    if (url.pathname === "/v1/me/data-purge/purge_browser_data") {
      purgeStatusReads += 1;
      await json(route, purgeReceipt("completed"));
      return;
    }
    await route.abort("failed");
  });
  await page.route("**/api/**", async (route) => {
    const url = new URL(route.request().url());
    if (url.pathname === "/api/me/preferences") {
      await json(route, browserPreferences());
      return;
    }
    if (url.pathname === "/api/me") {
      await json(route, browserUser());
      return;
    }
    if (url.pathname === "/api/conversations" && route.request().method() === "GET") {
      await json(route, { items: [browserConversation("Data controls conversation")], next_cursor: null });
      return;
    }
    if (url.pathname === `/api/conversations/${CONVERSATION_ID}/messages`) {
      await json(route, { items: [], next_cursor: null });
      return;
    }
    await route.abort("failed");
  });

  await page.goto("./");
  await page.getByRole("button", { name: "Open account settings" }).click();
  const dialog = page.getByRole("dialog", { name: "Settings" });
  const downloadPromise = page.waitForEvent("download");
  await dialog.getByRole("button", { name: "Download chat history" }).click();
  const download = await downloadPromise;
  expect(download.suggestedFilename()).toMatch(
    /^audrey-chat-history-\d{4}-\d{2}-\d{2}\.json$/u,
  );
  const stream = await download.createReadStream();
  if (!stream) throw new Error("Chat export download stream was unavailable.");
  const chunks: Buffer[] = [];
  for await (const chunk of stream) chunks.push(Buffer.from(chunk));
  const artifact = JSON.parse(Buffer.concat(chunks).toString("utf8")) as {
    schema_version: number;
    items: Array<{ message_id: string; content: string }>;
  };
  expect(artifact.schema_version).toBe(1);
  expect(artifact.items).toEqual([expect.objectContaining({
    message_id: "msg_browser_export",
    content: "Portable archived answer.",
  })]);
  await expect(dialog.getByText("Downloaded 1 archived message.")).toBeVisible();

  await dialog.getByRole("button", { name: "Delete Audrey data" }).click();
  const deleteButton = dialog.getByRole("button", { name: "Delete all Audrey data" });
  await expect(deleteButton).toBeDisabled();
  await dialog.getByRole("textbox", { name: "Deletion confirmation" }).fill(
    "DELETE ALL MY AUDREY DATA",
  );
  await expect(deleteButton).toBeEnabled();
  await deleteButton.click();

  await expect(dialog.getByRole("heading", { name: "Deletion is in progress" })).toBeVisible();
  await expect(dialog.getByRole("heading", { name: "Deletion complete" })).toBeVisible();
  expect(purgeStatusReads).toBeGreaterThanOrEqual(1);
  expect(purgeRequests).toEqual([{
    body: { confirmation: "DELETE ALL MY AUDREY DATA" },
    idempotencyKey: expect.stringMatching(/^native-ui-/u),
  }]);
  expect(userDataAuthorization.every((value) => value === undefined)).toBe(true);
  expect(await page.evaluate(() => ({
    local: localStorage.length,
    session: sessionStorage.length,
  }))).toEqual({ local: 0, session: 0 });

  const accessibility = await new AxeBuilder({ page }).analyze();
  expect(accessibility.violations).toEqual([]);
});



test("cancels an active browser run without leaving an error state", async ({ page }) => {
  await installHangingAgent(page);
  await mockAudreyApi(page);
  await page.goto("./");

  const composer = page.getByRole("textbox", { name: "Ask Audrey" });
  await composer.fill("Keep this run open");
  await composer.press("Enter");
  await page.getByRole("button", { name: "Stop" }).click();

  await expect(page.getByText("Stopped")).toBeVisible();
  await expect.poll(
    () => page.evaluate(() => Boolean((window as Window & { __cancelObserved?: boolean }).__cancelObserved)),
  ).toBe(true);
  await expect(page.getByRole("alert")).toHaveCount(0);
});

test("attaches an owner file through the minimized AG-UI request", async ({ page }) => {
  let requestBody: Record<string, unknown> | null = null;
  const existingMessages = [
    {
      id: "msg_attached_user",
      run_id: "run_attached",
      sequence: 1,
      role: "user",
      status: "completed",
      content: "Review my notes.",
      created_at: "2026-09-07T00:00:00Z",
      updated_at: "2026-09-07T00:00:00Z",
      attachments: [{
        id: "file_existing",
        filename: "field-notes.txt",
        mime: "text/plain",
        kind: "text",
        bytes: 42,
      }],
    },
    {
      id: "msg_attached_assistant",
      run_id: "run_attached",
      sequence: 2,
      role: "assistant",
      status: "completed",
      content: "I reviewed them.",
      created_at: "2026-09-07T00:00:01Z",
      updated_at: "2026-09-07T00:00:01Z",
      attachments: [],
    },
  ];
  await page.route("**/api/**", async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    if (url.pathname === "/api/me/preferences") {
      await json(route, browserPreferences());
      return;
    }
    if (url.pathname === "/api/me") {
      await json(route, browserUser());
      return;
    }
    if (url.pathname === "/api/conversations" && request.method() === "GET") {
      await json(route, { items: [browserConversation("Attached notes")], next_cursor: null });
      return;
    }
    if (url.pathname === `/api/conversations/${CONVERSATION_ID}/messages`) {
      await json(route, { items: existingMessages, next_cursor: null });
      return;
    }
    if (url.pathname === "/api/files" && request.method() === "GET") {
      await json(route, browserFileListing([browserFile("file_existing", "field-notes.txt", 42)]));
      return;
    }
    if (url.pathname === "/api/agent") {
      requestBody = request.postDataJSON() as Record<string, unknown>;
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
  await expect(page.locator(".message-user").first()).toContainText("📎 field-notes.txt");
  await page.getByRole("button", { name: "Attach files" }).click();
  const picker = page.getByRole("region", { name: "Choose attachments" });
  await picker.getByRole("button", { name: /field-notes\.txt/ }).click();
  await expect(page.getByRole("button", { name: "Remove attachment field-notes.txt" })).toBeVisible();

  const composer = page.getByRole("textbox", { name: "Ask Audrey" });
  await composer.fill("Use the attached notes again.");
  await composer.press("Enter");

  await expect(page.getByText("Complete", { exact: true })).toBeVisible();
  expect(requestBody).toMatchObject({
    threadId: CONVERSATION_ID,
    attachmentIds: ["file_existing"],
  });
  expect(requestBody?.messages).toHaveLength(1);
  await expect(page.getByRole("button", { name: "Remove attachment field-notes.txt" })).toHaveCount(0);
});

test("manages owner-bound files without a browser bearer token", async ({ page }) => {
  let files = [browserFile("file_existing", "field-notes.txt", 42)];
  const authorizationHeaders: Array<string | undefined> = [];
  await page.route("**/api/**", async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    authorizationHeaders.push(request.headers().authorization);
    if (url.pathname === "/api/me/preferences") {
      await json(route, browserPreferences());
      return;
    }
    if (url.pathname === "/api/me") {
      await json(route, browserUser());
      return;
    }
    if (url.pathname === "/api/conversations" && request.method() === "GET") {
      await json(route, { items: [browserConversation("Browser smoke")], next_cursor: null });
      return;
    }
    if (url.pathname === `/api/conversations/${CONVERSATION_ID}/messages`) {
      await json(route, { items: [], next_cursor: null });
      return;
    }
    if (url.pathname === "/api/files" && request.method() === "GET") {
      await json(route, browserFileListing(files));
      return;
    }
    if (url.pathname === "/api/files" && request.method() === "POST") {
      files = [...files, browserFile("file_uploaded", "new-notes.txt", 12)];
      await json(route, {
        id: "file_uploaded",
        filename: "new-notes.txt",
        mime: "text/plain",
        bytes: 12,
        kind: "text",
        chunks: 1,
        status: "ready",
      });
      return;
    }
    if (url.pathname === "/api/files/file_uploaded" && request.method() === "DELETE") {
      files = files.filter(({ id }) => id !== "file_uploaded");
      await json(route, { id: "file_uploaded", deleted: true, pending_cleanup: false });
      return;
    }
    await route.abort("failed");
  });

  await page.goto("./");
  await page.getByRole("button", { name: "Files", exact: true }).click();

  const dialog = page.getByRole("dialog", { name: "Your files" });
  await expect(dialog).toBeVisible();
  await expect(dialog.getByText("field-notes.txt")).toBeVisible();
  await dialog.getByLabel("Choose a file").setInputFiles({
    name: "new-notes.txt",
    mimeType: "text/plain",
    buffer: Buffer.from("native bytes"),
  });
  await dialog.getByRole("button", { name: "Upload", exact: true }).click();
  await expect(dialog.getByText("new-notes.txt")).toBeVisible();

  await dialog.getByRole("button", { name: "Delete new-notes.txt" }).click();
  const confirmation = dialog.getByRole("group", { name: "Delete new-notes.txt" });
  await confirmation.getByRole("button", { name: "Delete", exact: true }).click();
  await expect(dialog.getByText("new-notes.txt")).toHaveCount(0);
  expect(authorizationHeaders.every((value) => value === undefined)).toBe(true);
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

  const composer = page.getByRole("textbox", { name: "Ask Audrey" });
  await composer.fill("Attempt after expiry");
  await composer.press("Enter");

  await expect(page.getByRole("alert")).toBeVisible();
  await expect(page.getByText("Connection failed")).toBeVisible();
});

async function mockAudreyApi(
  page: Page,
  agentHandler?: (route: Route) => Promise<void> | void,
  messages: ReadonlyArray<Record<string, unknown>> = [],
  preferences = browserPreferences(),
) {
  let conversation: Record<string, unknown> = browserConversation("Browser smoke");
  await page.route("**/api/**", async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    if (url.pathname === "/api/me/preferences") {
      await json(route, preferences);
      return;
    }
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
    if (url.pathname === "/api/conversations" && request.method() === "GET") {
      await json(route, {
        items: [conversation],
        next_cursor: null,
      });
      return;
    }
    if (url.pathname === `/api/conversations/${CONVERSATION_ID}/messages`) {
      await json(route, { items: messages, next_cursor: null });
      return;
    }
    if (url.pathname === `/api/conversations/${CONVERSATION_ID}` && request.method() === "PATCH") {
      conversation = {
        ...conversation,
        ...(request.postDataJSON() as Record<string, unknown>),
      };
      await json(route, conversation);
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

function browserPreferences(overrides: Record<string, unknown> = {}) {
  return {
    timezone: "UTC",
    persona: "",
    detail: "balanced",
    tone: "natural",
    show_progress: true,
    created_at: "2026-09-04T00:00:00Z",
    updated_at: "2026-09-04T00:00:00Z",
    ...overrides,
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

function browserFile(id: string, filename: string, bytes: number) {
  return {
    id,
    filename,
    mime: "text/plain",
    bytes,
    uploaded_at: "2026-09-07T00:00:00Z",
    kind: "text" as const,
    chunks: 1,
    status: "ready",
    failure_reason: "",
    duration_s: 0,
    summary: "",
    source_freed_at: "",
    leased_at: "",
    source_url: "",
    transcript_source: "",
    fetch_downloaded_bytes: 0,
    fetch_total_bytes: 0,
  };
}

function browserFileListing(files: ReturnType<typeof browserFile>[]) {
  return {
    items: files,
    total_bytes: files.reduce((total, file) => total + file.bytes, 0),
    server_time: "2026-09-07T00:00:00Z",
    limits: {
      max_upload_bytes: 50 * 1024 * 1024,
      max_user_bytes: 1024 * 1024 * 1024,
      allowed_extensions: [".txt"],
      chunked_max_bytes: 2 * 1024 * 1024 * 1024,
      part_size: 8 * 1024 * 1024,
    },
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

function scrollingBrowserHistory() {
  return Array.from({ length: 18 }, (_, index) => ({
    id: `msg_scroll_${index + 1}`,
    run_id: `run_scroll_${Math.floor(index / 2) + 1}`,
    sequence: index + 1,
    role: index % 2 === 0 ? "user" : "assistant",
    status: "completed",
    content: `${index % 2 === 0 ? "Question" : "Answer"} ${index + 1}: ${
      "A deliberately long conversation entry that makes the message viewport overflow. ".repeat(3)
    }`,
    created_at: "2026-09-05T00:00:00Z",
    updated_at: "2026-09-05T00:00:00Z",
  }));
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
