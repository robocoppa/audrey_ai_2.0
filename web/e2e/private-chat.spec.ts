import { expect, test, type Page, type Route } from "@playwright/test";

const ID = "con_private_browser";
const DATE = "2026-10-19T12:00:00Z";

async function json(route: Route, body: unknown, status = 200) {
  await route.fulfill({ status, contentType: "application/json", body: JSON.stringify(body) });
}

function file(id: string, filename: string, privateRequired: boolean, kind = "text") {
  return {
    id, filename, private_required: privateRequired, kind, mime: "text/plain", bytes: 3,
    chunks: 1, status: "ready", uploaded_at: DATE, failure_reason: "", duration_s: 0,
    summary: "", source_freed_at: "", leased_at: "", source_url: "", transcript_source: "",
    fetch_downloaded_bytes: 0, fetch_total_bytes: 0,
  };
}

async function mockPrivateApi(page: Page, options: {
  initialPrivate?: boolean; omitPrivate?: boolean; rejectUpgrade?: boolean; saved?: boolean; older?: boolean; conflictingPortrait?: boolean;
} = {}) {
  let conversation = {
    id: ID, title: "Private browser check", private_required: Boolean(options.initialPrivate),
    default_model_id: options.initialPrivate ? "private" : "auto",
    default_mode: options.initialPrivate ? "local" : "auto", project_id: null,
    created_at: DATE, updated_at: DATE, last_message_at: options.saved ? DATE : null, archived_at: null,
  };
  let deleted = false;
  const retiredIds = new Set<string>();
  const staleReads: string[] = [];
  const older = { ...conversation, id: "con_older", title: "An older chat", private_required: false, default_model_id: "auto", default_mode: "auto", last_message_at: DATE };
  const uploads: string[] = [];
  const agents: Array<{ model: string | null; skill: string | null; body: Record<string, unknown> }> = [];
  const uploaded = file("file_private_uploaded", "private.txt", true);
  const models = ["auto", "cloud", ...(options.omitPrivate ? [] : ["private"])].map((id) => ({
    id, label: id === "private" ? "Private" : id === "cloud" ? "Cloud" : "Auto",
    description: id === "private" ? "Local models and tools." : "Standard workflow.",
    kind: "workflow", mode: id === "private" ? "local" : id, presentation: "local",
    capabilities: ["text", "files", "tools"], enabled: true, audience: "users",
    visibility: "public", roles: ["users"], portrait_url: id === "private" && options.conflictingPortrait ? "/wrong-private-portrait.png" : "",
  }));
  await page.route("**/api/**", async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    if (url.pathname === "/api/me") return json(route, {
      id: "usr_private_browser", email: "alice@example.com", display_name: "Alice", role: "user",
      status: "active", groups: ["users"], auth_provider: "cloudflare_access",
    });
    if (url.pathname === "/api/me/preferences") return json(route, {
      timezone: "UTC", persona: "", detail: "balanced", tone: "natural", show_progress: true,
      created_at: DATE, updated_at: DATE,
    });
    if (url.pathname === "/api/models") return json(route, { items: models });
    if (url.pathname === "/api/skills") return json(route, {
      enabled: true, status: "ready", items: [{
        id: "video-analysis", name: "Video analysis", description: "Inspect videos.", version: 1,
        supported_modes: ["auto", "fast", "deep"], availability: "available",
      }],
    });
    if (url.pathname === "/api/projects") return json(route, { items: [], next_cursor: null });
    if (url.pathname === "/api/conversations") {
      if (request.method() === "POST") {
        conversation = { ...conversation, id: "con_new_browser", title: "New conversation", private_required: false, default_model_id: "auto", default_mode: "auto", last_message_at: null };
        deleted = false;
        return json(route, conversation, 201);
      }
      return json(route, { items: [...(deleted ? [] : [conversation]), ...(options.older ? [older] : [])], next_cursor: null });
    }
    const id = url.pathname.split("/")[3];
    if (retiredIds.has(id) && request.method() === "GET") {
      staleReads.push(url.pathname);
      return json(route, { detail: "Conversation not found." }, 404);
    }
    if (url.pathname.endsWith("/messages")) return json(route, { items: (options.saved && id === ID) || id === older.id ? [
      { id: "msg_saved_user_" + id, run_id: null, sequence: 1, role: "user", status: "completed", content: "Saved question.", created_at: DATE, updated_at: DATE, attachments: [] },
      { id: "msg_saved_answer_" + id, run_id: null, sequence: 2, role: "assistant", status: "completed", content: "Saved answer.", created_at: DATE, updated_at: DATE, attachments: [] },
    ] : [], next_cursor: null });
    if (id === older.id) return json(route, older);
    if (url.pathname === `/api/conversations/${conversation.id}`) {
      if (request.method() === "PATCH") {
        if (options.rejectUpgrade) return json(route, { detail: "Start a new conversation to use Private." }, 409);
        const next = request.postDataJSON().model_id;
        const replacing = conversation.private_required && next !== "private";
        if (replacing) retiredIds.add(conversation.id);
        conversation = { ...conversation, id: replacing ? "con_replaced_browser" : conversation.id,
          private_required: next === "private", default_mode: next === "private" ? "local" : next, default_model_id: next };
      }
      if (request.method() === "DELETE") {
        retiredIds.add(conversation.id);
        deleted = true;
        return route.fulfill({ status: 204 });
      }
      return json(route, conversation);
    }
    if (url.pathname === "/api/files" && request.method() === "GET") return json(route, {
      items: [
        file("file_standard", "ordinary.txt", false),
        file("file_protected", "protected.txt", true),
        file("file_protected_image", "protected.png", true, "image"),
      ],
      total_bytes: 9, server_time: DATE, limits: {
        max_upload_bytes: 1024, max_user_bytes: 4096, allowed_extensions: [".txt", ".png"],
        chunked_max_bytes: 4096, part_size: 1024, fetch_hosts: ["www.youtube.com"], max_images_per_turn: 4,
      },
    });
    if (request.method() === "POST" && (url.pathname.includes("/files") || url.pathname.includes("upload-sessions"))) {
      uploads.push(url.pathname);
      return json(route, uploaded);
    }
    if (url.pathname === `/api/files/${uploaded.id}`) return json(route, uploaded);
    if (url.pathname === "/api/agent") {
      agents.push({ model: url.searchParams.get("model"), skill: url.searchParams.get("skill"), body: request.postDataJSON() });
      const events = [
        { type: "RUN_STARTED", threadId: conversation.id, runId: "run_private" },
        { type: "TEXT_MESSAGE_START", messageId: "msg_private" },
        { type: "TEXT_MESSAGE_CONTENT", messageId: "msg_private", delta: "Private document answer." },
        { type: "TEXT_MESSAGE_END", messageId: "msg_private" },
        { type: "RUN_FINISHED", threadId: conversation.id, runId: "run_private", outcome: { type: "success" } },
      ];
      return route.fulfill({ status: 200, contentType: "text/event-stream", body: events.map((event) => `data: ${JSON.stringify(event)}\n\n`).join("") });
    }
    if (url.pathname === "/api/runs/run_private") return json(route, {
      id: "run_private", conversation_id: conversation.id, status: "succeeded", title_pending: false,
    });
    await route.abort("failed");
  });
  return { uploads, agents, staleReads };
}

async function choosePrivate(page: Page) {
  await page.getByRole("button", { name: "Audrey model", exact: true }).click();
  await page.getByRole("dialog", { name: "Choose an Audrey model" }).getByRole("button", { name: "Private", exact: true }).click();
}

for (const viewport of [{ width: 1280, height: 900 }, { width: 390, height: 844 }]) {
test(`Private keeps local controls, protected documents and upload routing through the first answer at ${viewport.width}px`, async ({ page }) => {
  await page.setViewportSize(viewport);
  const observed = await mockPrivateApi(page);
  await page.goto("./");
  await choosePrivate(page);
  await expect(page.getByText("Private uses local models and tools.", { exact: false })).toBeVisible();
  await expect(page.getByRole("textbox", { name: "Ask Audrey" })).toBeInViewport();
  await expect(page.getByRole("button", { name: "Tools and skills: Local only" })).toBeInViewport();
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(viewport.width);
  await expect(page.getByRole("button", { name: "Move conversation to project" })).toBeEnabled();
  await page.getByRole("button", { name: "Audrey model", exact: true }).click();
  const models = page.getByRole("dialog", { name: "Choose an Audrey model" });
  await expect(models.getByRole("button", { name: "Private", exact: true })).toBeVisible();
  await expect(models.getByRole("button", { name: "Cloud", exact: true })).toBeVisible();
  await page.keyboard.press("Escape");
  await page.getByRole("button", { name: "Tools and skills: Local only" }).click();
  const skills = page.getByRole("dialog", { name: "Choose how Audrey uses tools" });
  await expect(skills.getByRole("button", { name: /Automatic/u })).toBeVisible();
  await expect(skills.getByRole("button", { name: /Video analysis/u })).toHaveCount(0);
  await page.keyboard.press("Escape");
  await page.getByRole("button", { name: "Add files", exact: true }).click();
  const files = page.getByRole("region", { name: "Choose attachments" });
  await expect(files.getByText("protected.txt", { exact: true })).toBeVisible();
  await expect(files.getByRole("button", { name: "Upload", exact: true })).toBeInViewport();
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(viewport.width);
  await expect(files.getByText("ordinary.txt", { exact: true })).toHaveCount(0);
  await expect(files.getByText("protected.png", { exact: true })).toHaveCount(0);
  await page.getByLabel("Choose a file from your device").setInputFiles({ name: "private.txt", mimeType: "text/plain", buffer: Buffer.from("one") });
  await expect(page.getByRole("button", { name: "Remove attachment private.txt" })).toBeVisible();
  expect(observed.uploads).toEqual([`/api/conversations/${ID}/files`]);
  const composer = page.getByRole("textbox", { name: "Ask Audrey" });
  await composer.fill("Read this document privately.");
  await composer.press("Enter");
  await expect(page.getByText("Private document answer.")).toBeVisible();
  expect(observed.agents).toHaveLength(1);
  expect(observed.agents[0]).toMatchObject({ model: "private", skill: null, body: { attachmentIds: ["file_private_uploaded"] } });
  await page.getByRole("button", { name: "Audrey model", exact: true }).click();
  await expect(page.getByRole("dialog", { name: "Choose an Audrey model" }).getByRole("button", { name: "Cloud", exact: true })).toHaveCount(0);
  await page.keyboard.press("Escape");
  await expect(page.getByRole("textbox", { name: "Ask Audrey" })).toBeInViewport();
  await expect(page.getByRole("button", { name: "Tools and skills: Local only" })).toBeInViewport();
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(viewport.width);
  if (viewport.width === 390) await page.screenshot({ path: "/tmp/audrey-private-mobile.png" });
});
}

test("Private survives a catalog change and refuses media uploads before sending bytes", async ({ page }) => {
  const observed = await mockPrivateApi(page, { initialPrivate: true, omitPrivate: true });
  await page.goto("./");
  await expect(page.getByRole("button", { name: "Audrey model", exact: true })).toContainText("Private");
  await page.reload();
  await expect(page.getByRole("button", { name: "Audrey model", exact: true })).toContainText("Private");
  await page.getByRole("button", { name: "Add files", exact: true }).click();
  const input = page.getByLabel("Choose a file from your device");
  await expect(input).toHaveAttribute("accept", ".txt,.md,.csv,.pdf,.docx,.xlsx,.html,.htm");
  await input.setInputFiles({ name: "image.png", mimeType: "image/png", buffer: Buffer.from("one") });
  await expect(page.getByRole("alert")).toContainText("Private accepts documents only");
  expect(observed.uploads).toEqual([]);
});

test("ordinary attachment browsing excludes protected files", async ({ page }) => {
  await mockPrivateApi(page);
  await page.goto("./");
  await page.getByRole("button", { name: "Add files", exact: true }).click();
  const files = page.getByRole("region", { name: "Choose attachments" });
  await expect(files.getByText("ordinary.txt", { exact: true })).toBeVisible();
  await expect(files.getByText("protected.txt", { exact: true })).toHaveCount(0);
  await expect(files.getByText("protected.png", { exact: true })).toHaveCount(0);
});

test("a refused Private upgrade preserves the ordinary conversation and explains the next step", async ({ page }) => {
  await mockPrivateApi(page, { rejectUpgrade: true });
  await page.goto("./");
  await choosePrivate(page);
  await expect(page.getByRole("alert")).toContainText("Start a new conversation to use Private.");
  await expect(page.getByRole("button", { name: "Audrey model", exact: true })).toContainText("Auto");
  await expect(page.getByRole("button", { name: "Move conversation to project" })).toBeEnabled();
});


test("an empty Private draft can change models without leaking attachments or losing unsent text", async ({ page }) => {
  const observed = await mockPrivateApi(page, { initialPrivate: true, conflictingPortrait: true });
  await page.goto("./");
  await expect(page.locator(".composer-model-picker img")).toHaveAttribute("src", /localModel/u);
  await page.getByRole("button", { name: "Add files", exact: true }).click();
  await page.getByRole("region", { name: "Choose attachments" }).getByText("protected.txt", { exact: true }).click();
  const composer = page.getByRole("textbox", { name: "Ask Audrey" });
  await composer.fill("Keep this unsent question.");
  await page.getByRole("button", { name: "Audrey model", exact: true }).click();
  await page.getByRole("dialog", { name: "Choose an Audrey model" }).getByRole("button", { name: "Cloud", exact: true }).click();
  await expect(page.getByRole("button", { name: "Audrey model", exact: true })).toContainText("Cloud");
  await expect(composer).toHaveValue("Keep this unsent question.");
  await expect(page.getByRole("button", { name: "Remove attachment protected.txt" })).toHaveCount(0);
  await expect(page.getByRole("status").filter({ hasText: "Attachments were removed" })).toBeVisible();
  expect(observed.staleReads).toEqual([]);
  await composer.press("Enter");
  await expect(page.getByText("Private document answer.")).toBeVisible();
  expect(observed.agents[0]).toMatchObject({ model: "cloud", body: { threadId: "con_replaced_browser" } });
  expect(observed.agents[0].body.attachmentIds ?? []).toEqual([]);
  await expect(composer).toHaveValue("");
});

test("a saved Private conversation stays locked after reopening", async ({ page }) => {
  await mockPrivateApi(page, { initialPrivate: true, saved: true });
  await page.goto("./");
  await expect(page.getByText("Saved answer.")).toBeVisible();
  await page.getByRole("button", { name: "Audrey model", exact: true }).click();
  const picker = page.getByRole("dialog", { name: "Choose an Audrey model" });
  await expect(picker.getByRole("button", { name: "Private", exact: true })).toBeVisible();
  await expect(picker.getByRole("button", { name: "Cloud", exact: true })).toHaveCount(0);
});

test("deleting the selected Private conversation opens a fresh draft instead of an older chat", async ({ page }) => {
  const observed = await mockPrivateApi(page, { initialPrivate: true, saved: true, older: true });
  await page.goto("./");
  await expect(page.getByText("Saved answer.")).toBeVisible();
  await page.getByRole("button", { name: "Delete conversation", exact: true }).click();
  await page.getByRole("button", { name: "Confirm delete conversation", exact: true }).click();
  await expect(page.getByRole("heading", { name: "New conversation", exact: true })).toBeVisible();
  await expect(page.getByRole("button", { name: "Audrey model", exact: true })).toContainText("Auto");
  await expect(page.getByRole("textbox", { name: "Ask Audrey" })).toBeFocused();
  await expect(page.getByText("Saved answer.")).toHaveCount(0);
  await expect(page.locator(".composer-model-picker img")).toBeVisible();
  await expect(page.getByRole("alert")).toHaveCount(0);
  await expect(page.getByRole("navigation", { name: "Conversation history" }).getByRole("button", { name: /^An older chat/u })).toBeVisible();
  expect(observed.staleReads).toEqual([]);
});


test("navigation wins if deleting a Private conversation is still creating its new draft", async ({ page }) => {
  await mockPrivateApi(page, { initialPrivate: true, saved: true, older: true });
  let releaseCreate = () => {};
  let createStarted = false;
  let discarded = false;
  const createGate = new Promise<void>((resolve) => { releaseCreate = resolve; });
  await page.route("**/api/conversations", async (route) => {
    if (route.request().method() !== "POST") return route.fallback();
    createStarted = true;
    await createGate;
    return route.fallback();
  });
  await page.route("**/api/conversations/con_new_browser/empty", async (route) => {
    discarded = true;
    return route.fulfill({ status: 204 });
  });
  await page.goto("./");
  await expect(page.getByText("Saved answer.")).toBeVisible();
  await page.getByRole("button", { name: "Delete conversation", exact: true }).click();
  await page.getByRole("button", { name: "Confirm delete conversation", exact: true }).click();
  await expect.poll(() => createStarted).toBe(true);
  await page.getByRole("navigation", { name: "Conversation history" }).getByRole("button", { name: /^An older chat/u }).click();
  const composer = page.getByRole("textbox", { name: "Ask Audrey" });
  await composer.fill("Keep the newly selected chat.");
  releaseCreate();
  await expect.poll(() => discarded).toBe(true);
  await expect(page.getByRole("heading", { name: "An older chat", exact: true })).toBeVisible();
  await expect(composer).toHaveValue("Keep the newly selected chat.");
  await expect(page.getByRole("alert")).toHaveCount(0);
});
