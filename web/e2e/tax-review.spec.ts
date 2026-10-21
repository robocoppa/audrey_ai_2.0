import { expect, test, type Page, type Route } from "@playwright/test";

import type { TaxCase } from "../src/taxReviewApi";
// Generated through the real repository operations and checklist serializer.
import fixtures from "../src/test/fixtures/tax-review-cases.json" with { type: "json" };

function fixtureCase(state: keyof typeof fixtures): TaxCase {
  return { ...fixtures[state], conversation_id: ID, project_id: null } as TaxCase;
}

const DATE = "2026-10-20T12:00:00Z";
const ID = "con_tax_private";
const OTHER = "con_ordinary";

async function json(route: Route, body: unknown, status = 200) {
  await route.fulfill({ status, contentType: "application/json", body: JSON.stringify(body) });
}

async function mockTaxApi(page: Page) {
  let taxCase: TaxCase | null = null;
  const requests: Array<{ path: string; body: Record<string, unknown> }> = [];
  const skills: Array<string | null> = [];
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
    if (path === "/api/projects") return json(route, { items: [], next_cursor: null });
    if (path === "/api/conversations") return json(route, request.method() === "POST" ? conversation : { items: [conversation, ordinary], next_cursor: null });
    if (path === `/api/conversations/${ID}`) return json(route, conversation);
    if (path === `/api/conversations/${OTHER}`) return json(route, ordinary);
    if (path.endsWith("/messages")) return json(route, { items: [], next_cursor: null });
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
      if (body.operation === "change_year") {
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
      conversation.last_message_at = DATE;
      skills.push(url.searchParams.get("skill"));
      const runId = "run_tax_" + skills.length;
      if (taxCase && url.searchParams.get("skill") === "tax-review") taxCase = fixtureCase("proposed");
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
    if (path.startsWith("/api/runs/run_tax_")) return json(route, { id: path.split("/").at(-1), conversation_id: ID, status: "succeeded", title_pending: false });
    await route.abort("failed");
  });
  return { requests, skills };
}

for (const viewport of [{ width: 1280, height: 900 }, { width: 390, height: 844 }]) {
  test(`Tax review has quiet settings and conversational choices at ${viewport.width}px`, async ({ page }) => {
    await page.setViewportSize(viewport);
    await page.clock.setFixedTime(new Date(DATE));
    const observed = await mockTaxApi(page);
    await page.goto("./");
    await page.getByRole("button", { name: "Tools and skills: Automatic" }).click();
    const picker = page.getByRole("dialog", { name: "Choose how Audrey uses tools" });
    await picker.getByRole("button", { name: /Tax review/u }).click();
    const panel = page.getByRole("region", { name: "Tax review case" });
    await expect(panel.getByLabel("Tax year", { exact: true })).toHaveValue("2025");
    await panel.getByLabel("Review goal", { exact: true }).fill("Review deductions");
    await panel.getByRole("button", { name: "Use Tax review" }).click();
    await expect(page.getByRole("button", { name: "Tools and skills: Tax review" })).toBeVisible();
    expect(observed.requests[0].body).toEqual({ tax_year: 2025, jurisdiction: "US-federal", goal: "Review deductions" });
    await expect(panel.getByRole("button", { name: /Tax review · 2025/u })).toHaveAttribute("aria-expanded", "false");
    await expect(page.getByText("Some features degraded", { exact: true })).toHaveCount(0);
    await expect(panel).not.toContainText("Checklist");
    await expect(panel).not.toContainText("Reference release");
    const composer = page.getByRole("textbox", { name: "Ask Audrey" });
    await expect(composer).toBeInViewport();
    expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(viewport.width);
    await composer.fill("Review my case.");
    await composer.press("Enter");
    await expect(page.getByText("Review findings saved 1.")).toBeVisible();
    const question = page.getByRole("article", { name: "Audrey's tax review questions" });
    await expect(question.getByText("Is this correct for your 2025 return?")).toBeVisible();
    await expect(question.getByText("55000.00", { exact: true })).toBeVisible();
    await expect(panel.getByRole("button", { name: "Confirm Wages" })).toHaveCount(0);
    await expect(page.locator(".thread-viewport > .tax-review-question")).toBeVisible();
    await question.getByRole("button", { name: "Confirm Wages" }).click();
    await expect(question.getByRole("status")).toContainText("I've saved your confirmation.");
    await question.getByRole("button", { name: "Mark W-2 box 1 addressed" }).click();
    await expect(question.getByRole("status")).toContainText("I've marked that as addressed.");
    expect(observed.requests[1].body).toEqual({ expected_revision: fixtures.proposed.revision, operation: "confirm_fact", data: { fact_id: fixtures.proposed.facts[0].id } });
    expect(observed.requests[2].body).toEqual({ expected_revision: fixtures.confirmed.revision, operation: "resolve_missing_item", data: { id: fixtures.proposed.missing_items[0].id } });
    await page.getByRole("button", { name: "Tools and skills: Tax review" }).click();
    await picker.getByRole("button", { name: /Automatic/u }).click();
    await composer.fill("An ordinary Private question.");
    await composer.press("Enter");
    await expect(page.getByText("Review findings saved 2.")).toBeVisible();
    await expect(page.getByText("I've marked that as addressed.", { exact: true })).toHaveCount(0);
    expect(observed.skills).toEqual(["tax-review", null]);
    await page.reload();
    await expect(page.getByRole("button", { name: "Tools and skills: Automatic" })).toBeVisible();
    await expect(page.getByText("Some features degraded", { exact: true })).toHaveCount(0);
    await panel.getByRole("button", { name: /Tax review · 2025/u }).click();
    await panel.getByLabel("Tax year", { exact: true }).fill("2024");
    await panel.getByRole("button", { name: "Save year" }).click();
    await expect(panel.getByRole("button", { name: /Tax review · 2024/u })).toHaveAttribute("aria-expanded", "false");
    await expect(question.getByText("Is this correct for your 2024 return?")).toBeVisible();
    await expect(question.getByText("From your 2025 review. Please check whether this still applies.")).toBeVisible();
    await question.getByRole("button", { name: "Not correct: Wages" }).click();
    await expect(question.getByRole("status")).toContainText("I won't use that detail.");
    expect(observed.requests[3].body.operation).toBe("change_year");
    expect(observed.requests[4].body.operation).toBe("reject_fact");
    if (viewport.width <= 1100) await page.locator("#app-mobile-menu-toggle").click();
    await page.locator(".conversation-list .conversation").filter({ hasText: "Ordinary conversation" }).click();
    if (viewport.width > 1100) await expect(page.getByText("Some features degraded", { exact: true }).first()).toBeVisible();
    else {
      await page.locator("#app-mobile-menu-toggle").click();
      await expect(page.getByText("Some features degraded", { exact: true }).last()).toBeVisible();
      await page.getByRole("button", { name: "Close conversation history" }).click();
    }
    expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(viewport.width);
  });
}
