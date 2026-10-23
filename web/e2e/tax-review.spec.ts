import { expect, test, type Locator, type Page, type Route } from "@playwright/test";

import type { PrivateTask } from "../src/privateTasksApi";
import { TAX_REVIEW_PROMPT, type TaxCase } from "../src/taxReviewApi";
// Generated through the real repository operations and checklist serializer.
import fixtures from "../src/test/fixtures/tax-review-cases.json" with { type: "json" };
import fileFixture from "../src/test/fixtures/tax-review-files.json" with { type: "json" };

function fixtureCase(state: keyof typeof fixtures): TaxCase {
  return { ...fixtures[state], conversation_id: ID, project_id: null } as TaxCase;
}

const DATE = "2026-10-20T12:00:00Z";
const ID = "con_tax_private";
const OTHER = "con_ordinary";

async function json(route: Route, body: unknown, status = 200) {
  await route.fulfill({ status, contentType: "application/json", body: JSON.stringify(body) });
}

async function mockTaxApi(page: Page, options: { savedTask?: boolean; historyCount?: number } = {}) {
  let taxCase: TaxCase | null = options.savedTask ? fixtureCase("created") : null;
  let privateTask: PrivateTask | null = options.savedTask ? {
    id: "task_saved_review", conversation_id: ID, revision: 7, status: "running", stop_reason: "",
    progress: 1, updated_at: DATE, native_run_id: "run_tax_saved",
    usage: { steps: 1, wall_seconds: 30, model_tokens: 100, tool_calls: 2 },
    limits: { max_steps: 24, max_wall_seconds: 600, max_model_tokens: 32000, max_tool_calls: 64 },
  } : null;

  const requests: Array<{ path: string; body: Record<string, unknown> }> = [];
  const skills: Array<string | null> = [];
  const agentBodies: Array<Record<string, unknown>> = [];
  const conversation = {
    id: ID, title: "Private tax review", private_required: true, default_mode: "local", default_model_id: "private",
    project_id: null, created_at: DATE, updated_at: DATE, last_message_at: null as string | null, archived_at: null,
  };
  const ordinary = { ...conversation, id: OTHER, title: "Ordinary conversation", private_required: false, default_mode: "auto", default_model_id: "auto", last_message_at: DATE };
  await page.route("**/api/**", async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    const path = url.pathname;
    if (path === "/api/me") return json(route, { id: "usr_tax", email: "alice@example.com", display_name: "Alice", role: "user", status: "active", groups: ["users"], auth_provider: "cloudflare_access" });
    if (path === "/api/me/preferences") return json(route, { timezone: "UTC", persona: "", detail: "balanced", tone: "natural", show_progress: true, created_at: DATE, updated_at: DATE });
    if (path === "/api/models") return json(route, { items: ["auto", "private"].map((id) => ({ id, label: id === "private" ? "Private" : "Auto", description: "Audrey workflow", kind: "workflow", mode: id === "private" ? "local" : "auto", presentation: "local", capabilities: ["text", "files", "tools"], enabled: true, audience: "users", visibility: "public", roles: ["users"], portrait_url: "" })) });
    if (path === "/api/skills") return json(route, { enabled: true, status: "ready", items: [] });
    if (path === "/api/capabilities") return json(route, { status: "degraded", generated_at: DATE, chat: { status: "available" }, tools: { status: "degraded" }, knowledge: { status: "available" }, skills: { status: "available" } });
    if (path === "/api/files") return json(route, fileFixture);
    if (path === "/api/projects") return json(route, { items: [], next_cursor: null });
    if (path === "/api/conversations") return json(route, request.method() === "POST" ? conversation : { items: [conversation, ordinary], next_cursor: null });
    if (path === `/api/conversations/${ID}`) return json(route, conversation);
    if (path === `/api/conversations/${OTHER}`) return json(route, ordinary);
    if (path.endsWith("/messages")) return json(route, { items: options.savedTask ? [
      { id: "msg_tax_saved_user", run_id: "run_tax_saved", sequence: 1, role: "user", status: "completed", content: TAX_REVIEW_PROMPT, created_at: DATE, updated_at: DATE, attachments: [] },
      { id: "msg_tax_saved_assistant", run_id: "run_tax_saved", sequence: 2, role: "assistant", status: privateTask?.status === "running" ? "in_progress" : "incomplete", content: "Saved review checkpoint.", created_at: DATE, updated_at: DATE, attachments: [] },
    ] : [], next_cursor: null });
    if (path === `/api/conversations/${ID}/private-task`) return json(route, { task: privateTask });
    if (path === "/api/private-tasks/task_saved_review/actions") {
      const body = request.postDataJSON();
      requests.push({ path, body });
      if (!privateTask || body.expected_revision !== privateTask.revision) return json(route, { detail: "Review changed; reload it." }, 409);
      privateTask = { ...privateTask, revision: privateTask.revision + 1,
        status: body.action === "pause" ? "paused" : "cancelled", stop_reason: body.action === "pause" ? "owner_paused" : "owner_cancelled" };
      return json(route, { task: privateTask });
    }
    if (path === `/api/conversations/${ID}/tax-review`) {
      if (request.method() === "GET") return json(route, { case: taxCase });
      const body = request.postDataJSON();
      requests.push({ path, body });
      taxCase = { ...fixtureCase("created"), tax_year: body.tax_year, goal: body.goal };
      return json(route, taxCase, 201);
    }
    if (path === `/api/conversations/${ID}/tax-review/actions`) {
      const body = request.postDataJSON();
      requests.push({ path, body });
      if (!taxCase || body.expected_revision !== taxCase.revision) return json(route, { detail: "Case changed; reload it." }, 409);
      if (body.operation === "update_reference_release") {
        if (taxCase.reference_update?.status !== "update_available" || body.data.reference_release !== taxCase.reference_update.release_id) return json(route, { detail: "Library changed; reload it." }, 409);
        taxCase = { ...taxCase, reference_release: body.data.reference_release, revision: taxCase.revision + 1, reference_update: { status: "current" } };
      } else if (body.operation === "confirm_facts") {
        taxCase = fixtureCase("group_confirmed");
      } else if (body.operation === "change_year") {
        taxCase = { ...taxCase, tax_year: body.data.tax_year, revision: taxCase.revision + 1,
          facts: taxCase.facts.map((fact) => ({ ...fact, case_tax_year: fact.case_tax_year ?? taxCase!.tax_year,
            status: fact.status === "confirmed" ? "proposed" : fact.status })) };
      } else if (body.operation === "reject_fact") {
        taxCase = { ...taxCase, revision: taxCase.revision + 1, facts: taxCase.facts.map((fact) => fact.id === body.data.fact_id ? { ...fact, status: "rejected" } : fact) };
      } else {
        taxCase = fixtureCase(body.operation === "confirm_fact" ? "confirmed" : "resolved");
      }
      return json(route, taxCase);
    }
    if (path === "/api/agent") {
      agentBodies.push(request.postDataJSON());
      conversation.last_message_at = DATE;
      skills.push(url.searchParams.get("skill"));
      const runId = "run_tax_" + skills.length;
      const body = request.postDataJSON();
      const last = body.messages?.at(-1)?.content ?? "";
      if (taxCase && url.searchParams.get("skill") === "tax-review") {
        if (last.startsWith("I confirm these details")) {
          taxCase = fixtureCase("group_reviewed");
          // A long saved choice history proves the rail scrolls independently.
          const prior = taxCase.facts[0];
          taxCase.facts = [...Array.from({ length: options.historyCount ?? 0 }, (_, index) => ({
            ...prior, id: `tax_earlier_${index}`, label: `Earlier confirmed detail ${index + 1}`,
            confirmed_at: `2026-10-19T12:00:${String(index).padStart(2, "0")}Z`,
          })), ...taxCase.facts];
        }
        else if (!last.startsWith("The saved ")) taxCase = fixtureCase("group_proposed");
      }
      if (privateTask) privateTask = { ...privateTask, revision: privateTask.revision + 1,
        status: "completed", stop_reason: "", native_run_id: runId };
      const messageId = "msg_" + runId;
      const events = [
        { type: "RUN_STARTED", threadId: ID, runId },
        { type: "TEXT_MESSAGE_START", messageId, role: "assistant" },
        { type: "TEXT_MESSAGE_CONTENT", messageId, delta: "Review findings saved " + skills.length + "." },
        { type: "TEXT_MESSAGE_END", messageId },
        { type: "RUN_FINISHED", threadId: ID, runId, outcome: { type: "success" } },
      ];
      return route.fulfill({ status: 200, contentType: "text/event-stream", body: events.map((event) => `data: ${JSON.stringify(event)}\n\n`).join("") });
    }
    if (path.startsWith("/api/runs/run_tax_")) return json(route, { id: path.split("/").at(-1), conversation_id: ID,
      status: options.savedTask && privateTask?.status === "running" ? "running"
        : options.savedTask && privateTask?.status === "paused" ? "cancelled" : "succeeded",
      error_code: privateTask?.status === "paused" ? "private_task_paused" : "", title_pending: false });
    await route.abort("failed");
  });
  return { requests, skills, agentBodies, offerLibraryUpdate() {
    if (!taxCase) throw new Error("Create the case before offering a library update.");
    taxCase = { ...taxCase, reference_update: { status: "update_available", release_id: "rel_updated_library" } };
  } };
}

function toolbox(page: Page, width: number): Locator {
  return page.getByRole(width <= 900 ? "dialog" : "complementary", { name: "Toolbox", exact: true });
}

async function openToolbox(page: Page, width: number) {
  const panel = toolbox(page, width);
  const trigger = page.locator(".conversation-toolbox-toolbar > button");
  await expect(trigger).toBeAttached();
  // A saved desktop case may open the rail before its hidden trigger is read.
  // Check and open atomically so that loading cannot cause a second toggle.
  await trigger.evaluate((button) => {
    if (!button.closest("[hidden]")) button.click();
  });
  await expect(panel).toBeVisible();
  await expect(trigger).toBeHidden();
  return panel;
}

async function closeToolbox(panel: Locator) {
  await panel.getByRole("button", { name: "Hide toolbox", exact: true }).click();
  await expect(panel).not.toBeVisible();
  await expect(panel.page().getByRole("button", { name: "Toolbox", exact: true })).toBeFocused();
}

for (const viewport of [{ width: 1280, height: 900 }, { width: 1280, height: 600 }, { width: 390, height: 667 }]) {
  test(`Toolbox setup and saved choice history work at ${viewport.width}x${viewport.height}`, async ({ page }, testInfo) => {
    await page.setViewportSize(viewport);
    await page.clock.setFixedTime(new Date(DATE));
    const observed = await mockTaxApi(page, { historyCount: 8 });
    await page.goto("./");
    await page.getByRole("button", { name: "Tools and skills: Automatic" }).click();
    await page.getByRole("dialog", { name: "Choose how Audrey uses tools" }).getByRole("button", { name: /Tax review/u }).click();
    const rail = toolbox(page, viewport.width);
    await expect(rail).toBeVisible();
    await expect(page.getByRole("button", { name: "Toolbox", exact: true })).toHaveCount(0);
    const settings = rail.getByRole("region", { name: "Tax review settings" });
    await expect(settings).toBeVisible();
    await expect(page.getByRole("dialog", { name: "Tax review settings" })).toHaveCount(0);
    const bounds = await rail.boundingBox();
    expect(bounds!.x).toBeGreaterThanOrEqual(0);
    expect(bounds!.y).toBeGreaterThanOrEqual(0);
    expect(bounds!.y + bounds!.height).toBeLessThanOrEqual(viewport.height);
    if (viewport.width > 900) {
      const chat = await page.locator(".thread-viewport").boundingBox();
      expect(chat!.x + chat!.width).toBeLessThanOrEqual(bounds!.x + 1);
    }
    await expect(rail.getByText(/Tax review can make mistakes/u)).toBeVisible();
    await expect(settings.getByLabel("Tax year", { exact: true })).toHaveValue("2025");
    await expect(settings.getByLabel("Review goal", { exact: true })).not.toBeFocused();
    await settings.getByLabel("Review goal", { exact: true }).fill("Review deductions");
    await settings.getByRole("button", { name: "Use Tax review" }).click();
    await expect(settings).toHaveCount(0);
    const panel = rail.getByRole("region", { name: "Tax review case" });
    const settingsButton = panel.getByRole("button", { name: /Tax review · 2025/u });
    await expect(settingsButton).toHaveAttribute("aria-expanded", "false");
    await expect(settingsButton).not.toHaveAttribute("aria-haspopup", "dialog");
    await expect(rail.getByRole("button", { name: /Start review|Continue review/u })).toHaveCount(0);
    expect(observed.agentBodies).toHaveLength(0);
    if (viewport.width <= 900) {
      await page.keyboard.press("Escape");
      await expect(rail).not.toBeVisible();
      await expect(page.getByRole("button", { name: "Toolbox", exact: true })).toBeFocused();
    } else await closeToolbox(rail);
    await openToolbox(page, viewport.width);
    expect(observed.agentBodies).toHaveLength(0);
    await closeToolbox(rail);
    const tab = page.getByRole("button", { name: "Toolbox", exact: true });
    const tabBounds = await tab.boundingBox();
    expect(tabBounds!.width).toBeGreaterThanOrEqual(44);
    expect(tabBounds!.height).toBeGreaterThan(tabBounds!.width * 2);
    expect(Math.abs(tabBounds!.x + tabBounds!.width - viewport.width)).toBeLessThan(1);
    expect(Math.abs(tabBounds!.y + tabBounds!.height / 2 - viewport.height / 2)).toBeLessThan(1);
    const sendBounds = await page.getByRole("button", { name: "Send message", exact: true }).boundingBox();
    expect(sendBounds!.x + sendBounds!.width).toBeLessThanOrEqual(tabBounds!.x);
    await page.screenshot({ path: testInfo.outputPath("toolbox-tab.png") });
    await expect(page.getByText("Some features degraded", { exact: true })).toHaveCount(0);

    const composer = page.getByRole("textbox", { name: "Ask Audrey" });
    await composer.fill(TAX_REVIEW_PROMPT);
    await page.getByRole("button", { name: "Send message", exact: true }).click();
    await expect(page.getByText("Review findings saved 1.")).toBeVisible();
    expect(JSON.stringify(observed.agentBodies[0])).toContain(TAX_REVIEW_PROMPT);
    // Attachments and text entered after the first response must survive the
    // explicit confirmation followup without being sent with that followup.
    await page.getByRole("button", { name: "Add files", exact: true }).click();
    const files = page.getByRole("region", { name: "Choose attachments" });
    await files.getByRole("button", { name: /unsent-private.txt/u }).click();
    await files.getByRole("button", { name: "Hide attachment picker" }).click();
    await composer.fill("Keep this unsent message and file.");
    await openToolbox(page, viewport.width);
    const question = rail.getByRole("region", { name: "Tax review choices" });
    await expect(page.locator(".thread-viewport > .tax-review-question")).toHaveCount(0);
    await expect(question.getByText("Wages", { exact: true })).toBeVisible();
    await expect(question.getByText("Filing status", { exact: true })).toBeVisible();
    await question.getByRole("button", { name: "Confirm these details" }).click();
    await expect(question.getByRole("list", { name: "Previous choices" }).getByText("Wages", { exact: true })).toBeVisible();
    await expect(question.getByRole("button", { name: "Confirm these details" })).toHaveCount(0);
    await expect(question.locator(".tax-choice-answered")).toHaveCount(10);
    expect(observed.requests[1].body).toEqual({ expected_revision: fixtures.group_proposed.revision,
      operation: "confirm_facts", data: { fact_ids: fixtures.group_proposed.facts.map(({ id }) => id) } });
    expect(observed.skills).toEqual(["tax-review", "tax-review"]);
    expect(JSON.stringify(observed.agentBodies[1])).toContain("I confirm these details for my 2025 tax review: Wages: 55000.00; Filing status: Single.");
    expect(observed.agentBodies[1]).not.toHaveProperty("attachmentIds");
    await expect(rail.getByRole("button", { name: /Start review|Continue review/u })).toHaveCount(0);

    const flow = question.locator(".tax-question-flow");
    await expect.poll(async () => flow.evaluate((box) => box.scrollHeight > box.clientHeight)).toBe(true);
    const pageScroll = await page.evaluate(() => window.scrollY);
    await flow.evaluate((box) => { box.scrollTop = 0; });
    await expect(question.getByText("Earlier confirmed detail 1", { exact: true })).toBeInViewport();
    expect(await page.evaluate(() => window.scrollY)).toBe(pageScroll);
    await flow.evaluate((box) => { box.scrollTop = box.scrollHeight; });
    await expect(question.getByText("Filing status", { exact: true })).toBeInViewport();
    await page.screenshot({ path: testInfo.outputPath("toolbox.png") });
    await closeToolbox(rail);
    await expect(page.getByText("Review findings saved 2.")).toBeVisible();
    await expect(composer).toHaveValue("Keep this unsent message and file.");
    await expect(page.getByRole("button", { name: "Remove attachment unsent-private.txt" })).toBeVisible();
    await openToolbox(page, viewport.width);
    expect(observed.agentBodies).toHaveLength(2);
    const worksheetToggle = panel.getByText("Review worksheet", { exact: true });
    await expect(worksheetToggle).toBeInViewport({ ratio: 1 });
    await worksheetToggle.click();
    await expect(worksheetToggle).toBeInViewport({ ratio: 1 });
    await expect(panel.getByText(fixtures.group_reviewed.review.summary, { exact: true })).toBeVisible();
    const worksheet = panel.locator(".tax-review-worksheet");
    await expect.poll(() => worksheet.evaluate((box) => box.scrollHeight > box.clientHeight)).toBe(true);
    await worksheet.evaluate((box) => { box.scrollTop = box.scrollHeight; });
    await expect(worksheetToggle).toBeInViewport({ ratio: 1 });
    expect(await page.evaluate(() => window.scrollY)).toBe(pageScroll);
    await worksheetToggle.click();
    await expect(worksheet).not.toHaveAttribute("open");
    await expect(worksheetToggle).toBeInViewport({ ratio: 1 });
    await page.reload();
    await openToolbox(page, viewport.width);
    await expect(question.locator(".tax-choice-answered")).toHaveCount(10);
    await expect(question.getByRole("button", { name: "Confirm these details" })).toHaveCount(0);
    await expect(rail.getByRole("button", { name: /Start review|Continue review/u })).toHaveCount(0);
    await panel.getByRole("button", { name: /Tax review · 2025/u }).click();
    await expect(settings).toBeVisible();
    await panel.getByRole("button", { name: /Tax review · 2025/u }).click();
    await expect(settings).toHaveCount(0);
    expect(observed.agentBodies).toHaveLength(2);
    expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(viewport.width);
  });
}

for (const viewport of [{ width: 1280, height: 600 }, { width: 390, height: 667 }]) {
  test(`Saved Private review pauses, reloads and resumes in Toolbox at ${viewport.width}px`, async ({ page }) => {
    await page.setViewportSize(viewport);
    await page.clock.setFixedTime(new Date(DATE));
    const observed = await mockTaxApi(page, { savedTask: true });
    await page.goto("./");
    const rail = await openToolbox(page, viewport.width);
    const panel = rail.getByRole("region", { name: "Tax review case" });
    const progress = panel.getByLabel("Saved review progress");
    await expect(progress.getByRole("status")).toHaveText("Audrey is reviewing your saved information");
    await expect(progress.getByRole("button", { name: "Pause", exact: true })).toBeInViewport();
    await expect(progress.getByRole("button", { name: "Cancel", exact: true })).toBeInViewport();
    await expect(panel.getByRole("button", { name: /Start review|Continue review/u })).toHaveCount(0);
    await expect(page.getByRole("button", { name: "Retry last question" })).toHaveCount(0);
    await progress.getByRole("button", { name: "Pause", exact: true }).click();
    await expect(progress.getByRole("status")).toHaveText("Review paused · Saved work is retained");
    await expect(progress.getByRole("button", { name: "Resume review" })).toBeEnabled();
    expect(observed.requests[0]).toEqual({ path: "/api/private-tasks/task_saved_review/actions",
      body: { expected_revision: 7, action: "pause" } });
    expect(observed.agentBodies).toHaveLength(0);
    await closeToolbox(rail);
    await page.reload();
    await openToolbox(page, viewport.width);
    await expect(progress.getByRole("button", { name: "Resume review" })).toBeEnabled();
    await expect(page.getByRole("button", { name: "Retry last question" })).toHaveCount(0);
    await expect(page.getByText("Run failed", { exact: true })).toHaveCount(0);
    expect(observed.agentBodies).toHaveLength(0);
    await closeToolbox(rail);
    const composer = page.getByRole("textbox", { name: "Ask Audrey" });
    await composer.fill("Keep this unsent draft while the review resumes.");
    await openToolbox(page, viewport.width);
    await progress.getByRole("button", { name: "Resume review" }).click();
    await expect(progress.getByRole("status")).toHaveText("Review saved");
    await expect(panel.getByRole("button", { name: /Start review|Continue review/u })).toHaveCount(0);
    expect(observed.skills).toEqual(["tax-review"]);
    expect(JSON.stringify(observed.agentBodies[0])).toContain(TAX_REVIEW_PROMPT);
    expect(JSON.stringify(observed.agentBodies[0])).not.toContain("Keep this unsent draft");
    await closeToolbox(rail);
    await expect(page.getByText("Review findings saved 1.")).toBeVisible();
    await expect(composer).toHaveValue("Keep this unsent draft while the review resumes.");
    await openToolbox(page, viewport.width);
    expect(observed.agentBodies).toHaveLength(1);
    expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(viewport.width);
  });
}
