import { readFileSync } from "node:fs";
import { expect, test, type Page, type Route } from "@playwright/test";
import fixtures from "../src/test/fixtures/office-previews.json" with { type: "json" };

const DATE = "2026-10-27T12:00:00Z";
const CSP = "default-src 'self'; connect-src 'self'; img-src 'self' data:; style-src 'self'; font-src 'self'; object-src 'none'; base-uri 'self'; frame-ancestors 'none'; form-action 'self'";
async function json(route: Route, body: unknown) {
  await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(body) });
}
async function mockOffice(page: Page) {
  const conversation = { id: "con_office", title: "Launch files", private_required: true, default_mode: "local", default_model_id: "private", project_id: null, created_at: DATE, updated_at: DATE, last_message_at: DATE, archived_at: null };
  await page.route("http://127.0.0.1:4173/", async (route) => {
    const response = await route.fetch();
    await route.fulfill({ response, headers: { ...response.headers(), "Content-Security-Policy": CSP } });
  });
  await page.route("**/api/**", async (route) => {
    const path = new URL(route.request().url()).pathname;
    if (path === "/api/me") return json(route, { id: "usr_office", email: "reader@example.com", display_name: "Reader", role: "user", status: "active", groups: ["users"], auth_provider: "cloudflare_access" });
    if (path === "/api/me/preferences") return json(route, { timezone: "UTC", persona: "", detail: "balanced", tone: "natural", show_progress: true, created_at: DATE, updated_at: DATE });
    if (path === "/api/models") return json(route, { items: ["auto", "private"].map((id) => ({ id, label: id === "private" ? "Private" : "Auto", description: "Audrey workflow", kind: "workflow", mode: id === "private" ? "local" : "auto", presentation: "local", capabilities: ["text", "files", "tools"], enabled: true, audience: "users", visibility: "public", roles: ["users"], portrait_url: "" })) });
    if (path === "/api/skills") return json(route, { enabled: true, status: "ready", items: [] });
    if (path === "/api/capabilities") return json(route, { status: "ready", generated_at: DATE, chat: { status: "available" }, tools: { status: "available" }, knowledge: { status: "available" }, skills: { status: "available" } });
    if (path === "/api/projects") return json(route, { items: [], next_cursor: null });
    if (path === "/api/conversations") return json(route, { items: [conversation], next_cursor: null });
    if (path === "/api/conversations/con_office") return json(route, conversation);
    if (path.endsWith("/tax-review")) return json(route, { case: null });
    if (path.endsWith("/private-task")) return json(route, { task: null });
    if (path.endsWith("/messages")) return json(route, { items: [
      { id: "msg_office_user", run_id: "run_office", sequence: 1, role: "user", status: "completed", content: "Make the launch documents.", attachments: [], created_at: DATE, updated_at: DATE },
      { id: "msg_office_answer", run_id: "run_office", sequence: 2, role: "assistant", status: "completed", content: `[Download launch-plan.docx](/api/files/file_document/download)\n\n[Download launch-budget.xlsx](/api/files/file_workbook/download)`, attachments: [], created_at: DATE, updated_at: DATE },
    ], next_cursor: null });
    if (path === "/api/files/file_document/preview") return json(route, fixtures.document);
    if (path === "/api/files/file_workbook/preview") return json(route, fixtures.workbook);
    return route.abort("failed");
  });
}
for (const width of [1280, 390]) {
  test(`saved office formatting is readable at ${width}px without changing the chat draft`, async ({ page }) => {
    await page.setViewportSize({ width, height: 900 });
    await mockOffice(page);
    const downloads: string[] = [];
    page.on("download", (download) => downloads.push(download.suggestedFilename()));
    await page.goto("./");
    const composer = page.getByRole("textbox", { name: "Ask Audrey" });
    await composer.fill("Keep the next question unsent.");
    const triggers = page.getByRole("button", { name: "Preview", exact: true });
    await triggers.first().click();
    const modal = page.getByRole("dialog", { name: "File preview" });
    await expect(modal.getByRole("heading", { name: "Launch plan", exact: true })).toBeVisible();
    await expect(modal.getByRole("heading", { name: "Next steps", exact: true })).toBeVisible();
    await expect(modal.getByRole("listitem")).toHaveCount(2);
    await expect(modal.getByRole("columnheader", { name: "Owner", exact: true })).toBeVisible();
    await expect(modal.locator(".office-document-preview strong", { hasText: "review" })).toHaveCSS("font-weight", "700");
    await expect(modal.locator(".office-document-preview em", { hasText: "draft" })).toHaveCSS("font-style", "italic");
    await expect(modal.getByText(/Some formatting is simplified/u)).toHaveCount(0);
    await page.screenshot({ path: `/tmp/audrey-rich-docx-${width}.png` });
    await modal.getByRole("button", { name: "Close file preview" }).click();
    await triggers.last().click();
    await expect(modal.getByRole("cell", { name: "$1,250.50", exact: true })).toHaveCSS("text-align", "left");
    await expect(modal.getByRole("cell", { name: "25.00%", exact: true })).toHaveCSS("text-align", "left");
    await expect(modal.getByRole("cell", { name: "Amount", exact: true })).toHaveCSS("background-color", "rgb(22, 50, 79)");
    const bounds = await modal.boundingBox();
    expect(bounds!.x).toBeGreaterThanOrEqual(0);
    expect(bounds!.x + bounds!.width).toBeLessThanOrEqual(width);
    expect(bounds!.y + bounds!.height).toBeLessThanOrEqual(900);
    await page.screenshot({ path: `/tmp/audrey-rich-xlsx-${width}.png` });
    await modal.getByRole("button", { name: "Close file preview" }).click();
    await expect(composer).toHaveValue("Keep the next question unsent.");
    expect(downloads).toEqual([]);
  });
}

for (const width of [1280, 390]) {
  test(`My Files filename previews and media playback work at ${width}px`, async ({ page }) => {
    await page.setViewportSize({ width, height: 900 });
    await mockOffice(page);
    const audioBytes = readFileSync(new URL("./fixtures/preview-audio.wav", import.meta.url));
    const videoBytes = readFileSync(new URL("./fixtures/preview-video.mp4", import.meta.url));
    const imageBytes = readFileSync(new URL("./fixtures/pdf-preview.png", import.meta.url));
    const files = [
      { id: "file_image", filename: "picture.png", kind: "image", mime: "image/png" },
      { id: "file_audio", filename: "sample.wav", kind: "audio", mime: "audio/x-wav" },
      { id: "file_video", filename: "sample.mp4", kind: "video", mime: "video/mp4" },
    ].map((file) => ({ ...file, bytes: 2048, origin: "upload", status: "ready", uploaded_at: DATE, chunks: 1,
      source_freed_at: "", summary: "A synthetic sample.", failure_reason: "", duration_s: 3, source_url: "", leased_at: "", transcript_source: "", fetch_downloaded_bytes: 0, fetch_total_bytes: 0 }));
    const reclaimedVideo = { ...files[2], id: "file_reclaimed_video", filename: "archived.mp4",
      source_freed_at: "2026-09-18T20:48:29+00:00", source_url: "https://www.youtube.com/watch?v=reclaimed_fixture",
      summary: "Saved notes from the removed video." };
    files.push(reclaimedVideo);
    const sourceRequests: string[] = [];
    await page.route("https://www.youtube.com/**", (route) => {
      sourceRequests.push(route.request().url());
      return route.abort("blockedbyclient");
    });
    await page.route("**/api/files", (route) => json(route, { items: files, total_bytes: 6144, server_time: DATE, limits: { max_upload_bytes: 50000000, max_user_bytes: 1000000000, allowed_extensions: [".wav", ".mp4", ".png"], chunked_max_bytes: 2000000000, part_size: 8000000, fetch_hosts: [], max_images_per_turn: 4 } }));
    await page.route("**/api/files/file_image/image", (route) => route.fulfill({ status: 200, contentType: "image/png", body: imageBytes }));
    await page.route("**/api/files/*/artifacts/*?*", (route) => {
      const [, , , id, , artifact] = new URL(route.request().url()).pathname.split("/");
      const text = files.find((file) => file.id === id)!.summary;
      return json(route, { id, artifact, text, offset: 0, next_offset: null, total_chars: text.length });
    });
    const mediaRequests: string[] = [];
    await page.route("**/api/files/*/media", async (route) => {
      mediaRequests.push(new URL(route.request().url()).pathname);
      if (route.request().url().includes(reclaimedVideo.id)) return route.abort("failed");
      const audio = route.request().url().includes("file_audio");
      const data = audio ? audioBytes : videoBytes;
      const range = /^bytes=(\d+)-(\d*)$/u.exec(route.request().headers().range ?? "");
      const start = range ? Number(range[1]) : 0;
      const end = range?.[2] ? Math.min(Number(range[2]), data.length - 1) : data.length - 1;
      await route.fulfill({ status: range ? 206 : 200, contentType: audio ? "audio/wav" : "video/mp4",
        headers: { "Accept-Ranges": "bytes", "Content-Length": String(end - start + 1), "Cache-Control": "private, no-store",
          ...(range ? { "Content-Range": `bytes ${start}-${end}/${data.length}` } : {}) }, body: data.subarray(start, end + 1) });
    });
    const downloads: string[] = [];
    page.on("download", (download) => downloads.push(download.suggestedFilename()));
    await page.goto("./");
    if (width === 390) await page.getByRole("button", { name: "Menu", exact: true }).click();
    await page.getByRole("button", { name: "My Files", exact: true }).click();
    const dialog = page.getByRole("dialog", { name: "Your files" });
    await dialog.getByRole("button", { name: "Preview picture.png", exact: true }).click();
    const picture = dialog.getByRole("img", { name: "Preview of picture.png" });
    await expect(picture).toBeVisible();
    await expect.poll(() => picture.evaluate((image) => (image as HTMLImageElement).naturalWidth)).toBe(800);
    await dialog.getByRole("button", { name: "← All files" }).click();
    await dialog.getByRole("button", { name: "Preview archived.mp4", exact: true }).click();
    await expect(dialog.getByRole("button", { name: "Player", exact: true })).toHaveAttribute("aria-pressed", "true");
    await expect(dialog.getByRole("status")).toContainText("Playback unavailable. The original file was removed to free storage.");
    const source = dialog.getByRole("link", { name: "Open source website", exact: true });
    await expect(source).toHaveAttribute("href", reclaimedVideo.source_url);
    await expect(source).toHaveAttribute("target", "_blank");
    await expect(source).toHaveAttribute("rel", "noopener noreferrer");
    await expect(dialog.locator("audio,video")).toHaveCount(0);
    await expect(dialog.getByRole("link", { name: "Download original archived.mp4", exact: true })).toHaveCount(0);
    await page.screenshot({ path: `/tmp/audrey-reclaimed-player-${width}.png` });
    await dialog.getByRole("button", { name: "Summary", exact: true }).click();
    await expect(dialog.getByText(reclaimedVideo.summary, { exact: true })).toBeVisible();
    await dialog.getByRole("button", { name: "Player", exact: true }).click();
    await expect(dialog.getByRole("status")).toContainText("Playback unavailable.");
    await expect(dialog.locator("audio,video")).toHaveCount(0);
    await dialog.getByRole("button", { name: "← All files" }).click();
    await dialog.getByRole("button", { name: "View video text for archived.mp4", exact: true }).click();
    await expect(dialog.getByRole("button", { name: "Summary", exact: true })).toHaveAttribute("aria-pressed", "true");
    await expect(dialog.getByText(reclaimedVideo.summary, { exact: true })).toBeVisible();
    await expect(dialog.getByRole("button", { name: "Player", exact: true })).toHaveAttribute("aria-pressed", "false");
    expect(mediaRequests).not.toContain(`/api/files/${reclaimedVideo.id}/media`);
    await dialog.getByRole("button", { name: "← All files" }).click();
    await dialog.getByRole("button", { name: "Preview sample.wav", exact: true }).click();
    const audio = dialog.getByLabel("Audio player for sample.wav", { exact: true });
    await expect(audio).toHaveAttribute("controls", "");
    await expect(audio).not.toHaveAttribute("autoplay", "");
    await expect.poll(() => audio.evaluate((media) => (media as HTMLMediaElement).readyState)).toBeGreaterThanOrEqual(1);
    await audio.evaluate((media) => (media as HTMLMediaElement).play());
    await expect.poll(() => audio.evaluate((media) => (media as HTMLMediaElement).currentTime)).toBeGreaterThan(0);
    await audio.evaluate((media) => { (media as HTMLMediaElement).currentTime = 1; });
    await expect.poll(() => audio.evaluate((media) => (media as HTMLMediaElement).currentTime)).toBeGreaterThanOrEqual(1);
    const oldAudio = await audio.elementHandle();
    await dialog.getByRole("button", { name: "Summary", exact: true }).click();
    await expect(dialog.getByText("A synthetic sample.", { exact: true })).toBeVisible();
    expect(await oldAudio!.evaluate((media) => (media as HTMLMediaElement).paused && !media.getAttribute("src"))).toBe(true);
    await dialog.getByRole("button", { name: "← All files" }).click();
    await dialog.getByRole("button", { name: "Preview sample.mp4", exact: true }).click();
    const video = dialog.getByLabel("Video player for sample.mp4", { exact: true });
    await expect(video).toBeVisible();
    await expect(video).toHaveAttribute("playsinline", "");
    await expect.poll(() => video.evaluate((media) => (media as HTMLMediaElement).readyState)).toBeGreaterThanOrEqual(1);
    await video.evaluate((media) => (media as HTMLMediaElement).play());
    await expect.poll(() => video.evaluate((media) => (media as HTMLMediaElement).paused)).toBe(false);
    const bounds = await video.boundingBox();
    expect(bounds!.x).toBeGreaterThanOrEqual(0);
    expect(bounds!.x + bounds!.width).toBeLessThanOrEqual(width);
    await page.screenshot({ path: `/tmp/audrey-media-player-${width}.png` });
    const oldVideo = await video.elementHandle();
    await dialog.getByRole("button", { name: "Close files" }).click();
    expect(await oldVideo!.evaluate((media) => (media as HTMLMediaElement).paused && !media.getAttribute("src"))).toBe(true);
    expect(mediaRequests).not.toContain(`/api/files/${reclaimedVideo.id}/media`);
    expect(sourceRequests).toEqual([]);
    expect(downloads).toEqual([]);
  });
}
