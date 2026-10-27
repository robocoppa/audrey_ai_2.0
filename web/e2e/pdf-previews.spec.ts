import { readFileSync } from "node:fs";
import { expect, test, type Page, type Route } from "@playwright/test";

const DATE = "2026-10-26T12:00:00Z";
const CSP = "default-src 'self'; connect-src 'self'; img-src 'self' data:; style-src 'self'; font-src 'self'; object-src 'none'; base-uri 'self'; frame-ancestors 'none'; form-action 'self'";
const PNG = readFileSync(new URL("./fixtures/pdf-preview.png", import.meta.url));

async function json(route: Route, body: unknown, status = 200) {
  await route.fulfill({ status, contentType: "application/json", body: JSON.stringify(body), headers: { "Cache-Control": "private, no-store" } });
}

async function mockPdf(page: Page, privateRequired: boolean) {
  const conversation = { id: "con_pdf", title: "Inspection review", private_required: privateRequired,
    default_mode: privateRequired ? "local" : "auto", default_model_id: privateRequired ? "private" : "auto",
    project_id: "proj_pdf", created_at: DATE, updated_at: DATE, last_message_at: DATE, archived_at: null };
  const project = { id: "proj_pdf", name: "Inspection project", private_required: privateRequired, instructions: "", created_at: DATE, updated_at: DATE };
  const file = { id: "file_pdf", filename: "inspection.pdf", origin: "upload", private_required: privateRequired,
    kind: "text", mime: "application/pdf", bytes: 4096, chunks: 1, status: "ready", uploaded_at: DATE,
    summary: "The inspection recommends replacing the western seal.", failure_reason: "", source_url: "", source_freed_at: "", leased_at: "", duration_s: 0,
    transcript_source: "", fetch_downloaded_bytes: 0, fetch_total_bytes: 0 };
  const limits = { max_name_chars: 100, max_instructions_chars: 4000, max_files: 20 };
  let failSecondPage = true;
  const pageRequests: number[] = [];
  await page.route("http://127.0.0.1:4173/", async (route) => {
    const response = await route.fetch();
    await route.fulfill({ response, headers: { ...response.headers(), "Content-Security-Policy": CSP } });
  });
  await page.route("**/api/**", async (route) => {
    const path = new URL(route.request().url()).pathname;
    if (path === "/api/me") return json(route, { id: "usr_pdf", email: "reader@example.com", display_name: "Reader", role: "user", status: "active", groups: ["users"], auth_provider: "cloudflare_access" });
    if (path === "/api/me/preferences") return json(route, { timezone: "UTC", persona: "", detail: "balanced", tone: "natural", show_progress: true, created_at: DATE, updated_at: DATE });
    if (path === "/api/models") return json(route, { items: ["auto", "private"].map((id) => ({ id, label: id === "private" ? "Private" : "Auto", description: "Audrey workflow", kind: "workflow", mode: id === "private" ? "local" : "auto", presentation: "local", capabilities: ["text", "files", "tools"], enabled: true, audience: "users", visibility: "public", roles: ["users"], portrait_url: "" })) });
    if (path === "/api/skills") return json(route, { enabled: true, status: "ready", items: [] });
    if (path === "/api/capabilities") return json(route, { status: "ready", generated_at: DATE, chat: { status: "available" }, tools: { status: "available" }, knowledge: { status: "available" }, skills: { status: "available" } });
    if (path === "/api/conversations") return json(route, { items: [conversation], next_cursor: null });
    if (path === "/api/conversations/con_pdf") return json(route, conversation);
    if (path === "/api/conversations/con_pdf/messages") return json(route, { items: [
      { id: "msg_pdf_user", run_id: "run_pdf", sequence: 1, role: "user", status: "completed", content: "Review this inspection.", attachments: [file], created_at: DATE, updated_at: DATE },
      { id: "msg_pdf_answer", run_id: "run_pdf", sequence: 2, role: "assistant", status: "completed", content: "The document is ready to review.", attachments: [], created_at: DATE, updated_at: DATE },
    ], next_cursor: null });
    if (path.endsWith("/tax-review")) return json(route, { case: null });
    if (path.endsWith("/private-task")) return json(route, { task: null });
    if (path === "/api/projects") return json(route, { items: [project], limits, next_cursor: null });
    if (path === "/api/projects/proj_pdf") return json(route, project);
    if (path === "/api/projects/proj_pdf/conversations") return json(route, { items: [conversation], next_cursor: null });
    if (path === "/api/projects/proj_pdf/files") return json(route, { items: [{ ...file, added_at: DATE }], limits });
    if (path === "/api/projects/proj_pdf/tax-reviews") return json(route, { items: [] });
    if (path === "/api/files") return json(route, { items: [file], total_bytes: file.bytes, server_time: DATE, limits: {
      max_upload_bytes: 50_000_000, max_user_bytes: 1_000_000_000, allowed_extensions: [".pdf"], chunked_max_bytes: 2_000_000_000, part_size: 8_000_000, fetch_hosts: [], max_images_per_turn: 4,
    } });
    if (path === "/api/files/file_pdf/preview") return json(route, { id: file.id, filename: file.filename, kind: "pdf", format: "pdf", page_count: 2, truncated: false });
    const pageMatch = path.match(/^\/api\/files\/file_pdf\/preview\/pages\/(\d+)$/u);
    if (pageMatch) {
      const number = Number(pageMatch[1]);
      pageRequests.push(number);
      if (number === 2 && failSecondPage) { failSecondPage = false; return json(route, { detail: "Temporarily unavailable." }, 503); }
      return route.fulfill({ status: 200, contentType: "image/png", body: PNG, headers: { "Cache-Control": "private, no-store" } });
    }
    if (path === "/api/files/file_pdf/artifacts/summary") return json(route, { id: file.id, artifact: "summary", text: file.summary, offset: 0, next_offset: null, total_chars: file.summary.length });
    if (path === "/api/files/file_pdf/text") return json(route, { id: file.id, text: "Original extracted inspection text.", offset: 0, next_offset: null, total_chars: 35 });
    return json(route, { detail: "Not found." }, 404);
  });
  return { pageRequests, file };
}

for (const viewport of [{ width: 1280, height: 900 }, { width: 390, height: 844 }]) {
  test(`uploaded PDF previews work from chat, My files and a project at ${viewport.width}px`, async ({ page }) => {
    await page.setViewportSize(viewport);
    const { pageRequests, file } = await mockPdf(page, viewport.width === 390);
    const downloads: string[] = [];
    page.on("download", (download) => downloads.push(download.suggestedFilename()));
    await page.goto("./");
    const composer = page.getByRole("textbox", { name: "Ask Audrey" });
    await composer.fill("Keep my next question unsent.");
    const trigger = page.getByRole("button", { name: "Preview inspection.pdf", exact: true });
    await trigger.click();
    const preview = page.getByRole("dialog", { name: "File preview" });
    const first = preview.getByRole("img", { name: "inspection.pdf — page 1 of 2", exact: true });
    await expect(first).toBeVisible();
    await expect(first).toHaveCSS("display", "block");
    const viewportRegion = preview.getByRole("region", { name: "PDF page 1", exact: true });
    const imageBounds = await first.boundingBox();
    const regionBounds = await viewportRegion.boundingBox();
    expect(imageBounds!.width).toBeLessThan(regionBounds!.width);
    const modalBounds = await preview.boundingBox();
    expect(modalBounds!.y).toBeGreaterThanOrEqual(0);
    expect(modalBounds!.y + modalBounds!.height).toBeLessThanOrEqual(viewport.height);
    await page.screenshot({ path: `/tmp/audrey-pdf-preview-${viewport.width}.png` });
    await preview.getByRole("button", { name: "Next PDF page" }).click();
    await expect(preview.getByRole("alert")).toContainText("could not be displayed");
    await expect(preview.getByRole("img")).toHaveCount(0);
    await preview.getByRole("button", { name: "Try again" }).click();
    const second = preview.getByRole("img", { name: "inspection.pdf — page 2 of 2" });
    await expect(second).toBeVisible();
    expect(pageRequests.slice(0, 3)).toEqual([1, 2, 2]);
    await expect(preview.getByRole("button", { name: "Next PDF page" })).toBeDisabled();
    const initialWidth = (await second.boundingBox())!.width;
    await preview.getByLabel("Zoom", { exact: true }).selectOption("2");
    await expect.poll(async () => (await second.boundingBox())!.width).toBeCloseTo(initialWidth * 2, 0);
    await preview.getByLabel("PDF page number").fill("1");
    await preview.getByRole("button", { name: "Go", exact: true }).click();
    await expect(first).toBeVisible();
    await preview.getByRole("button", { name: "Close file preview" }).click();
    await expect(trigger).toBeFocused();
    await expect(composer).toHaveValue("Keep my next question unsent.");

    if (viewport.width === 390) await page.getByRole("button", { name: "Menu", exact: true }).click();
    await page.getByRole("button", { name: "My Files", exact: true }).click();
    const files = page.getByRole("dialog", { name: "Your files" });
    await files.getByRole("button", { name: "View document text for inspection.pdf" }).click();
    await expect(files.getByText(file.summary)).toBeVisible();
    await files.getByRole("button", { name: "Original PDF", exact: true }).click();
    await expect(files.getByRole("img", { name: "inspection.pdf — page 1 of 2" })).toBeVisible();
    await files.getByRole("button", { name: "Transcript", exact: true }).click();
    await expect(files.getByText("Original extracted inspection text.")).toBeVisible();
    await files.getByRole("button", { name: "Close files" }).click();
    if (viewport.width === 390) {
      await page.getByRole("button", { name: "Menu", exact: true }).click();
      const projects = page.getByRole("button", { name: "My projects", exact: true });
      if (await projects.getAttribute("aria-expanded") === "false") await projects.click();
      await page.locator(".projects-sidebar").getByRole("button", { name: "Inspection project", exact: true }).click();
    } else {
      await page.getByRole("navigation", { name: "Project breadcrumb" }).getByRole("button", { name: "Inspection project", exact: true }).click();
    }
    await page.getByRole("button", { name: "Preview inspection.pdf", exact: true }).click();
    await expect(preview.getByRole("img", { name: "inspection.pdf — page 1 of 2" })).toBeVisible();
    await preview.getByRole("button", { name: "Close file preview" }).click();
    expect(downloads).toEqual([]);
  });
}
