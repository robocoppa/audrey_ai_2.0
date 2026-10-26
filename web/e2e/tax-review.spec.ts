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

async function mockTaxApi(page: Page, options: {
  savedTask?: boolean;
  savedCase?: boolean;
  savedFailure?: "tax-review" | "automatic";
  holdCase?: boolean;
  caseError?: boolean;
  failFirstAgent?: boolean;
  historyCount?: number;
  autoProject?: boolean;
} = {}) {
  let taxCase: TaxCase | null = options.savedTask || options.savedCase || options.savedFailure ? fixtureCase("created") : null;
  let caseError = options.caseError ?? false;
  let caseReads = 0;
  let releaseCase!: () => void;
  const caseGate = new Promise<void>((resolve) => { releaseCase = resolve; });
  if (!options.holdCase) releaseCase();
  const failedRuns = new Set(options.savedFailure ? ["run_tax_saved"] : []);
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
    project_id: null as string | null, created_at: DATE, updated_at: DATE, last_message_at: options.savedCase || options.savedTask || options.savedFailure ? DATE : null as string | null, archived_at: null,
  };
  const taxProject = { id: "proj_tax", name: "Tax review 2025", instructions: "", private_required: true, created_at: DATE, updated_at: DATE };
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
    if (path === "/api/projects") return json(route, { items: conversation.project_id ? [taxProject] : [], next_cursor: null });
    if (path === "/api/projects/proj_tax") return json(route, taxProject);
    if (path === "/api/projects/proj_tax/conversations") return json(route, { items: [conversation], next_cursor: null });
    if (path === "/api/projects/proj_tax/files") return json(route, { items: [], limits: { max_name_chars: 100, max_instructions_chars: 4000, max_files: 20 } });
    if (path === "/api/projects/proj_tax/tax-reviews") return json(route, { items: taxCase ? [taxCase] : [] });
    if (path === "/api/conversations") return json(route, request.method() === "POST" ? conversation : { items: [conversation, ordinary], next_cursor: null });
    if (path === `/api/conversations/${ID}`) return json(route, conversation);
    if (path === `/api/conversations/${OTHER}`) return json(route, ordinary);
    if (path.endsWith("/messages")) return json(route, { items: path === `/api/conversations/${ID}/messages` && (options.savedTask || options.savedFailure) ? [
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
      if (request.method() === "GET") {
        caseReads += 1;
        await caseGate;
        return caseError ? json(route, { detail: "Tax settings temporarily unavailable." }, 503) : json(route, { case: taxCase });
      }
      const body = request.postDataJSON();
      requests.push({ path, body });
      if (options.autoProject) conversation.project_id = taxProject.id;
      taxCase = { ...fixtureCase("created"), tax_year: body.tax_year, goal: body.goal, project_id: conversation.project_id };
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
      } else if (body.operation === "edit_fact") {
        taxCase = { ...taxCase, revision: taxCase.revision + 1, facts: taxCase.facts.map((fact) => fact.id === body.data.fact_id ? { ...fact, label: body.data.label, value: body.data.value, status: "confirmed" } : fact) };
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
      if (options.failFirstAgent && skills.length === 1) {
        failedRuns.add(runId);
        const events = [
          { type: "RUN_STARTED", threadId: ID, runId },
          { type: "RUN_ERROR", code: "provider_error", message: "Run failed." },
        ];
        return route.fulfill({ status: 200, contentType: "text/event-stream", body: events.map((event) => `data: ${JSON.stringify(event)}\n\n`).join("") });
      }
      const body = request.postDataJSON();
      const last = body.messages?.at(-1)?.content ?? "";
      if (taxCase && url.searchParams.get("skill") === "tax-review") {
        if (last.startsWith("I confirm these details")) {
          taxCase = fixtureCase("group_reviewed");
          // A long current facts proves the rail scrolls independently.
          const prior = taxCase.facts[0];
          taxCase.facts = [...Array.from({ length: options.historyCount ?? 0 }, (_, index) => ({
            ...prior, id: `tax_earlier_${index}`, label: `Earlier confirmed detail ${index + 1}`,
            confirmed_at: `2026-10-19T12:00:${String(index).padStart(2, "0")}Z`,
          })), ...taxCase.facts];
        }
        else if (last === "I was 40.") {
          taxCase = { ...taxCase, revision: taxCase.revision + 1, missing_items: [],
            facts: [...taxCase.facts, { ...taxCase.facts[0], id: "age_fact", label: "Age at year-end", value: "40", status: "confirmed" }] };
        } else if (!last.startsWith("I updated ") && !last.startsWith("The saved ")) taxCase = fixtureCase("group_proposed");
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
    if (path.startsWith("/api/runs/run_tax_")) {
      const runId = path.split("/").at(-1)!;
      const skillId = runId === "run_tax_saved" ? options.savedFailure === "automatic" ? "" : "tax-review"
        : skills[Number(runId.slice("run_tax_".length)) - 1] ?? "";
      return json(route, { id: runId, conversation_id: ID, skill_id: skillId,
        status: failedRuns.has(runId) ? "failed" : options.savedTask && privateTask?.status === "running" ? "running"
          : options.savedTask && privateTask?.status === "paused" ? "cancelled" : "succeeded",
        error_code: failedRuns.has(runId) ? "provider_error" : privateTask?.status === "paused" ? "private_task_paused" : "", title_pending: false });
    }
    await route.abort("failed");
  });
  return { requests, skills, agentBodies, releaseCase,
    get caseReads() { return caseReads; },
    setCaseError(value: boolean) { caseError = value; },
    askQuestion() {
      if (!taxCase) throw new Error("Create the case first.");
      taxCase = { ...taxCase, revision: taxCase.revision + 1, missing_items: [{ id: "need_age", label: "Age",
        question: "What was your age at the end of 2025?", status: "unresolved" }] };
    }, offerLibraryUpdate() {
      if (!taxCase) throw new Error("Create the case before offering a library update.");
      taxCase = { ...taxCase, reference_update: { status: "update_available", release_id: "rel_updated_library" } };
    }, saveProjectWorksheet() {
      taxCase = { ...fixtureCase("group_reviewed"), project_id: conversation.project_id };
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

async function chooseToolsMode(page: Page, mode: "Automatic" | "Tax review") {
  await page.getByRole("button", { name: /^Tools and skills:/u }).click();
  await page.getByRole("dialog", { name: "Choose how Audrey uses tools" })
    .getByRole("button", { name: new RegExp(`^${mode}`, "u") }).click();
}

for (const viewport of [{ width: 1280, height: 900 }, { width: 1280, height: 600 }, { width: 390, height: 667 }]) {
  test(`Toolbox setup and current facts work at ${viewport.width}x${viewport.height}`, async ({ page }, testInfo) => {
    await page.setViewportSize(viewport);
    await page.clock.setFixedTime(new Date(DATE));
    const observed = await mockTaxApi(page, { historyCount: 8 });
    await page.goto("./");
    const automaticTools = page.getByRole("button", { name: "Tools and skills: Automatic" });
    await expect(automaticTools).toBeVisible();
    const composer = page.getByRole("textbox", { name: "Ask Audrey" });
    const setupDraft = "Help me organize my tax documents.";
    await composer.fill(setupDraft);
    await expect(page.getByRole("button", { name: "Send message", exact: true })).toBeEnabled();
    await automaticTools.click();
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
    await closeToolbox(rail);
    await composer.fill(setupDraft);
    await expect(page.getByText("Finish Tax review setup in the Toolbox before sending.", { exact: true })).toBeVisible();
    await expect(page.getByRole("button", { name: "Send message", exact: true })).toBeDisabled();
    await composer.press("Enter");
    await expect(composer).toHaveValue(setupDraft);
    expect(observed.agentBodies).toHaveLength(0);
    await openToolbox(page, viewport.width);
    await expect(settings).toBeVisible();
    await settings.getByLabel("Review goal", { exact: true }).fill("Review deductions");
    await expect(settings.getByRole("button", { name: "Use Tax review" })).toBeEnabled();
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

    await expect(composer).toHaveValue(setupDraft);
    await expect(page.getByRole("button", { name: "Send message", exact: true })).toBeEnabled();
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
    const question = rail.getByRole("region", { name: "Tax review facts" });
    await expect(page.locator(".thread-viewport > .tax-review-question")).toHaveCount(0);
    await expect(question.getByText("Wages", { exact: true })).toBeVisible();
    await expect(question.getByText("Filing status", { exact: true })).toBeVisible();
    await question.getByRole("button", { name: "Confirm these details" }).click();
    await expect(question.getByRole("table", { name: "Saved tax facts" }).getByText("Wages", { exact: true })).toBeVisible();
    await expect(question.getByRole("button", { name: "Confirm these details" })).toHaveCount(0);
    await expect(question.locator(".tax-facts-table tbody > tr")).toHaveCount(10);
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
    await expect(question.locator(".tax-facts-table tbody > tr")).toHaveCount(10);
    await expect(question.getByRole("button", { name: "Confirm these details" })).toHaveCount(0);
    await expect(rail.getByRole("button", { name: /Start review|Continue review/u })).toHaveCount(0);
    await panel.getByRole("button", { name: /Tax review · 2025/u }).click();
    await expect(settings).toBeVisible();
    await panel.getByRole("button", { name: /Tax review · 2025/u }).click();
    await expect(settings).toHaveCount(0);
    expect(observed.agentBodies).toHaveLength(2);
    // A correction replaces its table row and sends saved owner feedback. It
    // never appends another historical confirmation card.
    await question.getByRole("button", { name: "Edit Wages", exact: true }).click();
    await question.getByLabel("Value for Wages", { exact: true }).fill("52000.00");
    await question.getByRole("button", { name: "Save fact", exact: true }).click();
    await expect(question.getByRole("table", { name: "Saved tax facts" })).toHaveText(/52000.00/u);
    await expect(question.locator(".tax-facts-table tbody > tr")).toHaveCount(10);
    expect(observed.requests.at(-1)?.body).toMatchObject({ operation: "edit_fact", data: { label: "Wages", value: "52000.00" } });
    await expect.poll(() => observed.agentBodies.length).toBe(3);
    observed.askQuestion();
    await page.reload();
    await openToolbox(page, viewport.width);
    await expect(question.getByText("What was your age at the end of 2025?", { exact: true })).toBeVisible();
    await question.getByLabel("Your answer", { exact: true }).fill("I was 40.");
    await question.getByRole("button", { name: "Send answer", exact: true }).click();
    await expect(question.getByRole("table", { name: "Saved tax facts" })).toHaveText(/Age at year-end40/u);
    await expect(question.getByRole("button", { name: "Send answer", exact: true })).toHaveCount(0);
    expect(JSON.stringify(observed.agentBodies[3])).toContain("I was 40.");
    expect(observed.requests.some(({ body }) => body.operation === "resolve_missing_item")).toBe(false);
    await page.screenshot({ path: testInfo.outputPath("toolbox-current-facts.png") });
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


for (const viewport of [{ width: 1280, height: 800 }, { width: 390, height: 760 }]) {
  test(`Tax case creates a visible Private project and preserves the current draft at ${viewport.width}px`, async ({ page }, testInfo) => {
    await page.setViewportSize(viewport);
    const observed = await mockTaxApi(page, { autoProject: true });
    await page.goto("./");
    const composer = page.getByRole("textbox", { name: "Ask Audrey" });
    await composer.fill("Keep my unsent tax question.");
    await page.getByRole("button", { name: "Tools and skills: Automatic" }).click();
    await page.getByRole("dialog", { name: "Choose how Audrey uses tools" }).getByRole("button", { name: /Tax review/u }).click();
    const rail = toolbox(page, viewport.width);
    const settings = rail.getByRole("region", { name: "Tax review settings" });
    await settings.getByLabel("Review goal", { exact: true }).fill("Review deductions");
    await settings.getByRole("button", { name: "Use Tax review" }).click();
    await expect(page.getByRole("button", { name: "Move conversation to project", exact: true })).toContainText("Tax review 2025");
    if (viewport.width <= 900) await closeToolbox(rail);
    await expect(composer).toHaveValue("Keep my unsent tax question.");
    expect(observed.agentBodies).toHaveLength(0);
    observed.saveProjectWorksheet();
    if (viewport.width <= 900) await page.getByRole("button", { name: "Menu", exact: true }).click();
    await page.locator(".project-sidebar-link").filter({ hasText: "Tax review 2025" }).click();
    const project = page.getByRole("region", { name: "Tax reviews", exact: true });
    await expect(project.getByRole("table", { name: "2025 saved tax facts" })).toBeVisible();
    await project.getByText("Saved worksheet", { exact: true }).click();
    await expect(project.getByText((fixtures.group_reviewed as TaxCase).review!.summary, { exact: true })).toBeVisible();
    await expect(page.getByRole("button", { name: "Upload file to project", exact: true })).toBeVisible();
    await expect(page.getByRole("button", { name: "Choose from My Files", exact: true })).toBeVisible();
    expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(viewport.width);
    await page.screenshot({ path: testInfo.outputPath("tax-project-overview.png"), fullPage: true });
    await project.getByRole("button", { name: "Open review conversation", exact: true }).click();
    await expect(composer).toHaveValue("Keep my unsent tax question.");
    await expect(page.getByRole("button", { name: "Audrey model", exact: true })).toContainText("Private");
  });
}

test("a saved workbook link appears after the tax answer without replacing the next draft", async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 900 });
  await page.clock.setFixedTime(new Date(DATE));
  await mockTaxApi(page);
  await page.goto("./");
  await page.getByRole("button", { name: "Tools and skills: Automatic" }).click();
  await page.getByRole("dialog", { name: "Choose how Audrey uses tools" }).getByRole("button", { name: /Tax review/u }).click();
  const rail = toolbox(page, 1280);
  await rail.getByLabel("Review goal", { exact: true }).fill("Review deductions");
  await rail.getByRole("button", { name: "Use Tax review" }).click();
  await closeToolbox(rail);
  let release!: () => void;
  let observed!: () => void;
  const canonicalReady = new Promise<void>((resolve) => { observed = resolve; });
  const gate = new Promise<void>((resolve) => { release = resolve; });
  const path = `/api/conversations/${ID}/tax-review/exports/tax_export_1/download`;
  await page.route(`**/api/conversations/${ID}/messages?*`, async (route) => {
    observed();
    await gate;
    return json(route, { items: [
      { id: "msg_user_review", run_id: "run_tax_1", role: "user", status: "completed", content: TAX_REVIEW_PROMPT, sequence: 1, created_at: DATE, updated_at: DATE, attachments: [] },
      { id: "msg_run_tax_1", run_id: "run_tax_1", role: "assistant", status: "completed", content: `Review findings saved 1.\n\n[Download workbook](${path})`, sequence: 2, created_at: DATE, updated_at: DATE, attachments: [] },
    ], next_cursor: null });
  });
  await page.route(`**${path}`, (route) => route.fulfill({ status: 200,
    contentType: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    headers: { "Content-Disposition": 'attachment; filename="tax-review-2025.xlsx"' }, body: "workbook bytes" }));
  const composer = page.getByRole("textbox", { name: "Ask Audrey" });
  await composer.fill(TAX_REVIEW_PROMPT);
  await page.getByRole("button", { name: "Send message", exact: true }).click();
  await expect(page.getByText("Review findings saved 1.", { exact: true })).toBeVisible();
  await canonicalReady;
  await page.getByRole("button", { name: "Add files", exact: true }).click();
  const files = page.getByRole("region", { name: "Choose attachments" });
  await files.getByRole("button", { name: /unsent-private.txt/u }).click();
  await files.getByRole("button", { name: "Hide attachment picker" }).click();
  await composer.fill("Keep this next question unsent.");
  release();
  const link = page.getByRole("link", { name: "Download workbook", exact: true });
  await expect(link).toBeVisible();
  await expect(composer).toHaveValue("Keep this next question unsent.");
  await expect(page.getByRole("button", { name: "Remove attachment unsent-private.txt" })).toBeVisible();
  const downloaded = page.waitForEvent("download");
  await link.click();
  expect((await downloaded).suggestedFilename()).toBe("tax-review-2025.xlsx");
  await expect(composer).toHaveValue("Keep this next question unsent.");
});


test("saved Tax review remains selected after navigation and reload for a normal composer turn", async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 900 });
  const observed = await mockTaxApi(page, { savedCase: true });
  await page.goto("./");
  await expect(page.getByRole("button", { name: "Tools and skills: Tax review" })).toBeVisible();
  const composer = page.getByRole("textbox", { name: "Ask Audrey" });
  await composer.fill("Keep my tax draft while I check another conversation.");
  const history = page.getByRole("navigation", { name: "Conversation history", exact: true });
  await history.getByRole("button", { name: /^Ordinary conversation/u }).click();
  await expect(page.getByRole("button", { name: "Tools and skills: Automatic" })).toBeVisible();
  await history.getByRole("button", { name: /^Private tax review/u }).click();
  await expect(page.getByRole("button", { name: "Tools and skills: Tax review" })).toBeVisible();
  await expect(composer).toHaveValue("Keep my tax draft while I check another conversation.");
  await page.reload();
  await expect(page.getByRole("button", { name: "Tools and skills: Tax review" })).toBeVisible();
  await composer.fill("Check the arithmetic in my saved tax case.");
  await composer.press("Enter");
  await expect(page.getByText("Review findings saved 1.", { exact: true })).toBeVisible();
  expect(observed.skills).toEqual(["tax-review"]);
  expect(JSON.stringify(observed.agentBodies[0])).toContain("Check the arithmetic in my saved tax case.");
});

test("an explicit Automatic choice survives tax settings changes and conversation navigation", async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 900 });
  const observed = await mockTaxApi(page, { savedCase: true });
  await page.goto("./");
  await expect(page.getByRole("button", { name: "Tools and skills: Tax review" })).toBeVisible();
  await chooseToolsMode(page, "Automatic");
  const rail = await openToolbox(page, 1280);
  await rail.getByRole("button", { name: /Tax review · 2025/u }).click();
  const settings = rail.getByRole("region", { name: "Tax review settings", exact: true });
  await settings.getByLabel("Tax year", { exact: true }).fill("2024");
  await settings.getByRole("button", { name: "Save year", exact: true }).click();
  await expect(settings).toHaveCount(0);
  await expect(rail.getByRole("button", { name: /Tax review · 2024/u })).toBeVisible();
  await expect(page.getByRole("button", { name: "Tools and skills: Automatic" })).toBeVisible();
  const history = page.getByRole("navigation", { name: "Conversation history", exact: true });
  await history.getByRole("button", { name: /^Ordinary conversation/u }).click();
  await history.getByRole("button", { name: /^Private tax review/u }).click();
  await expect(page.getByRole("button", { name: "Tools and skills: Automatic" })).toBeVisible();
  const composer = page.getByRole("textbox", { name: "Ask Audrey" });
  await composer.fill("Use the automatic local tools for this question.");
  await composer.press("Enter");
  await expect(page.getByText("Review findings saved 1.", { exact: true })).toBeVisible();
  expect(observed.skills).toEqual([null]);
});

test("a pending saved-case lookup blocks Send and Enter until Tax review is restored", async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 900 });
  const observed = await mockTaxApi(page, { savedCase: true, holdCase: true });
  try {
    await page.goto("./");
    await expect.poll(() => observed.caseReads).toBeGreaterThan(0);
    const tools = page.getByRole("button", { name: "Tools and skills: Loading tax settings" });
    await expect(tools).toBeEnabled();
    const composer = page.getByRole("textbox", { name: "Ask Audrey" });
    await composer.fill("Review my saved figures after settings load.");
    await expect(page.getByRole("button", { name: "Send message", exact: true })).toBeDisabled();
    await composer.press("Enter");
    await expect(composer).toHaveValue("Review my saved figures after settings load.");
    expect(observed.agentBodies).toHaveLength(0);
    observed.releaseCase();
    await expect(page.getByRole("button", { name: "Tools and skills: Tax review" })).toBeVisible();
    await expect(page.getByRole("button", { name: "Send message", exact: true })).toBeEnabled();
    await composer.press("Enter");
    await expect(page.getByText("Review findings saved 1.", { exact: true })).toBeVisible();
    expect(observed.skills).toEqual(["tax-review"]);
  } finally {
    observed.releaseCase();
  }
});

test("choosing Automatic deliberately bypasses a pending saved-case lookup", async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 900 });
  const observed = await mockTaxApi(page, { savedCase: true, holdCase: true });
  try {
    await page.goto("./");
    await expect.poll(() => observed.caseReads).toBeGreaterThan(0);
    await chooseToolsMode(page, "Automatic");
    const composer = page.getByRole("textbox", { name: "Ask Audrey" });
    await composer.fill("Use Automatic while the saved settings are unavailable.");
    await expect(page.getByRole("button", { name: "Send message", exact: true })).toBeEnabled();
    await composer.press("Enter");
    await expect(page.getByText("Review findings saved 1.", { exact: true })).toBeVisible();
    expect(observed.skills).toEqual([null]);
    observed.releaseCase();
    const rail = await openToolbox(page, 1280);
    await expect(rail.getByRole("button", { name: /Tax review · 2025/u })).toBeVisible();
    await expect(page.getByRole("button", { name: "Tools and skills: Automatic" })).toBeVisible();
  } finally {
    observed.releaseCase();
  }
});

test("a failed saved-case lookup blocks ordinary sends until reloaded or explicitly set to Automatic", async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 900 });
  const observed = await mockTaxApi(page, { savedCase: true, caseError: true });
  const loadError = page.getByText("Could not load tax review settings. Reload settings or choose Automatic to continue.", { exact: true });
  const composer = page.getByRole("textbox", { name: "Ask Audrey" });
  await page.goto("./");
  await expect(loadError).toBeVisible();
  await composer.fill("Wait for the saved tax settings.");
  await expect(page.getByRole("button", { name: "Send message", exact: true })).toBeDisabled();
  await composer.press("Enter");
  await expect(composer).toHaveValue("Wait for the saved tax settings.");
  expect(observed.agentBodies).toHaveLength(0);
  observed.setCaseError(false);
  await page.getByRole("button", { name: "Reload tax settings", exact: true }).click();
  await expect(page.getByRole("button", { name: "Tools and skills: Tax review" })).toBeVisible();
  await expect(loadError).toHaveCount(0);
  await composer.press("Enter");
  await expect(page.getByText("Review findings saved 1.", { exact: true })).toBeVisible();
  expect(observed.skills).toEqual(["tax-review"]);

  observed.setCaseError(true);
  await page.reload();
  await expect(loadError).toBeVisible();
  await chooseToolsMode(page, "Automatic");
  await composer.fill("Continue with Automatic by my choice.");
  await expect(page.getByRole("button", { name: "Send message", exact: true })).toBeEnabled();
  await composer.press("Enter");
  await expect(page.getByText("Review findings saved 2.", { exact: true })).toBeVisible();
  expect(observed.skills).toEqual(["tax-review", null]);
});

for (const originalSkill of ["tax-review", "automatic"] as const) {
  test(`retry keeps the original ${originalSkill} skill after the picker changes`, async ({ page }) => {
    await page.setViewportSize({ width: 1280, height: 900 });
    const observed = await mockTaxApi(page, { savedCase: true, failFirstAgent: true });
    await page.goto("./");
    await expect(page.getByRole("button", { name: "Tools and skills: Tax review" })).toBeVisible();
    if (originalSkill === "automatic") await chooseToolsMode(page, "Automatic");
    const composer = page.getByRole("textbox", { name: "Ask Audrey" });
    await composer.fill(TAX_REVIEW_PROMPT);
    await composer.press("Enter");
    const retry = page.getByRole("button", { name: "Retry last question", exact: true });
    await expect(retry).toBeEnabled();
    await chooseToolsMode(page, originalSkill === "tax-review" ? "Automatic" : "Tax review");
    await retry.click();
    await expect(page.getByText("Review findings saved 2.", { exact: true })).toBeVisible();
    const expectedSkill = originalSkill === "tax-review" ? "tax-review" : null;
    expect(observed.skills).toEqual([expectedSkill, expectedSkill]);
    expect(JSON.stringify(observed.agentBodies[1])).toContain(TAX_REVIEW_PROMPT);
  });

  test(`retry restores the saved ${originalSkill} run skill after reload`, async ({ page }) => {
    await page.setViewportSize({ width: 1280, height: 900 });
    const observed = await mockTaxApi(page, { savedFailure: originalSkill });
    await page.goto("./");
    await expect(page.getByRole("button", { name: "Tools and skills: Tax review" })).toBeVisible();
    await page.reload();
    await expect(page.getByRole("button", { name: "Tools and skills: Tax review" })).toBeVisible();
    // The current mode deliberately differs from the original failed request.
    if (originalSkill === "tax-review") await chooseToolsMode(page, "Automatic");
    const retry = page.getByRole("button", { name: "Retry last question", exact: true });
    await expect(retry).toBeEnabled();
    await retry.click();
    await expect(page.getByText("Review findings saved 1.", { exact: true })).toBeVisible();
    expect(observed.skills).toEqual([originalSkill === "tax-review" ? "tax-review" : null]);
    expect(JSON.stringify(observed.agentBodies[0])).toContain(TAX_REVIEW_PROMPT);
  });
}


test("a generated file link appears in Automatic chat without replacing the next draft", async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 900 });
  await mockTaxApi(page);
  await page.goto("./");
  await expect(page.getByRole("button", { name: "Tools and skills: Automatic" })).toBeVisible();
  const path = "/api/files/7f45cd33-d591-4f69-9ac6-42efb004eb9a/download";
  await page.route(`**/api/conversations/${ID}/messages?*`, async (route) => json(route, { items: [
    { id: "msg_user_output", run_id: "run_tax_1", role: "user", status: "completed", content: "Create notes.md", sequence: 1, created_at: DATE, updated_at: DATE, attachments: [] },
    { id: "msg_run_tax_1", run_id: "run_tax_1", role: "assistant", status: "completed", content: `Review findings saved 1.\n\n[Download notes.md](${path})`, sequence: 2, created_at: DATE, updated_at: DATE, attachments: [] },
  ], next_cursor: null }));
  await page.route(`**${path}`, (route) => route.fulfill({ status: 200, contentType: "text/markdown",
    headers: { "Content-Disposition": 'attachment; filename="notes.md"' }, body: "# Working notes" }));
  const composer = page.getByRole("textbox", { name: "Ask Audrey" });
  await composer.fill("Create notes.md");
  await page.getByRole("button", { name: "Send message", exact: true }).click();
  const link = page.getByRole("link", { name: "Download notes.md", exact: true });
  await expect(link).toBeVisible();
  await composer.fill("Keep this question unsent.");
  const downloaded = page.waitForEvent("download");
  await link.click();
  expect((await downloaded).suggestedFilename()).toBe("notes.md");
  await expect(composer).toHaveValue("Keep this question unsent.");
});

for (const viewport of [{ width: 1280, height: 900 }, { width: 390, height: 844 }]) {
  test(`generated spreadsheet preview fits ${viewport.width}px, keeps the draft and never downloads`, async ({ page }) => {
    await page.setViewportSize(viewport);
    await mockTaxApi(page);
    const fileId = "file_preview";
    await page.route(`**/api/conversations/${ID}/messages?*`, (route) => json(route, { items: [
      { id: "msg_preview_user", run_id: "run_tax_1", role: "user", status: "completed", content: "Create a budget", sequence: 1, created_at: DATE, updated_at: DATE, attachments: [] },
      { id: "msg_preview_saved", run_id: "run_tax_1", role: "assistant", status: "completed", content: `[Download budget.xlsx](/api/files/${fileId}/download)`, sequence: 2, created_at: DATE, updated_at: DATE, attachments: [] },
    ], next_cursor: null }));
    await page.route(`**/api/files/${fileId}/preview`, (route) => json(route, {
      id: fileId, filename: "budget.xlsx", format: "xlsx", kind: "table", truncated: false,
      sheets: [
        { name: "Budget", rows: [["Item", "Cost", "A longer note for the spreadsheet preview"],
          ...Array.from({ length: 40 }, (_, index) => [`Item ${index + 1}`, index * 5, "A saved working value."])], truncated: false },
        { name: "Notes", rows: [["Private working output", "Read only"]], truncated: false },
      ],
    }));
    const downloads: string[] = [];
    page.on("download", (download) => downloads.push(download.suggestedFilename()));
    await page.goto("./");
    const composer = page.getByRole("textbox", { name: "Ask Audrey" });
    await composer.fill("Keep my unsent next question.");
    const trigger = page.getByRole("button", { name: "Preview", exact: true });
    await trigger.click();
    const dialog = page.getByRole("dialog", { name: "File preview" });
    await expect(dialog).toBeVisible();
    await expect(dialog.getByRole("cell", { name: "Item 1", exact: true })).toBeVisible();
    const bounds = await dialog.boundingBox();
    expect(bounds!.x).toBeGreaterThanOrEqual(0);
    expect(bounds!.y).toBeGreaterThanOrEqual(0);
    expect(bounds!.x + bounds!.width).toBeLessThanOrEqual(viewport.width);
    expect(bounds!.y + bounds!.height).toBeLessThanOrEqual(viewport.height);
    await dialog.getByRole("tab", { name: "Notes", exact: true }).click();
    await expect(dialog.getByRole("cell", { name: "Private working output" })).toBeVisible();
    await page.screenshot({ path: `/tmp/audrey-file-preview-${viewport.width}.png` });
    await dialog.getByRole("button", { name: "Close file preview" }).click();
    await expect(dialog).not.toBeVisible();
    await expect(trigger).toBeFocused();
    await expect(composer).toHaveValue("Keep my unsent next question.");
    expect(downloads).toEqual([]);
  });
}
