import { expect, test, type Page, type Route } from "@playwright/test";
import type { Conversation } from "../src/api";

const DATE = "2026-10-20T12:00:00Z";
const LIMITS = { max_name_chars: 100, max_instructions_chars: 4000, max_files: 20 };
const FILE_LIMITS = { max_upload_bytes: 1024, max_user_bytes: 4096, chunked_max_bytes: 4096, part_size: 1024, allowed_extensions: [".txt", ".png"], fetch_hosts: [], max_images_per_turn: 4 };

async function json(route: Route, body: unknown, status = 200) {
  await route.fulfill({ status, contentType: "application/json", body: JSON.stringify(body) });
}

function document(id: string, privateRequired: boolean, kind = "text") {
  return { id, filename: id + (kind === "text" ? ".txt" : ".png"), private_required: privateRequired,
    kind, mime: "text/plain", bytes: 3, chunks: 1, status: "ready", uploaded_at: DATE,
    summary: "", failure_reason: "", source_url: "", source_freed_at: "", leased_at: "", duration_s: 0,
    transcript_source: "", fetch_downloaded_bytes: 0, fetch_total_bytes: 0 };
}

async function mockProjects(page: Page, legacyLocal = false, omitLocal = false) {
  const initialConversation: Conversation = { id: "con_existing", title: "Existing chat", default_model_id: legacyLocal ? "local" : "auto",
    default_mode: legacyLocal ? "local" : "auto", private_required: false, project_id: null,
    created_at: DATE, updated_at: DATE, last_message_at: DATE, archived_at: null };
  let project: { id: string; name: string; instructions: string; private_required: boolean; created_at: string; updated_at: string } | undefined;
  let projectConversation: typeof initialConversation | undefined;
  const projectFiles: ReturnType<typeof document>[] = [];
  const posts: Array<{ path: string; body: unknown }> = [];
  const uploadPaths: string[] = [];
  const models = ["auto", ...(omitLocal ? [] : ["local"]), "private", "cloud"].map((id) => ({
    id, label: id[0].toUpperCase() + id.slice(1), description: id + " workflow", kind: "workflow",
    mode: id === "private" ? "local" : id, presentation: "local", capabilities: ["text", "files", "tools"],
    enabled: true, audience: "users", visibility: "public", roles: ["users"], portrait_url: "",
  }));
  await page.route("**/api/**", async (route) => {
    const request = route.request();
    const path = new URL(request.url()).pathname;
    if (path === "/api/me") return json(route, { id: "usr_project", email: "alice@example.com", display_name: "Alice", role: "user", status: "active", groups: ["users"], auth_provider: "cloudflare_access" });
    if (path === "/api/me/preferences") return json(route, { timezone: "UTC", persona: "", detail: "balanced", tone: "natural", show_progress: true, created_at: DATE, updated_at: DATE });
    if (path === "/api/models") return json(route, { items: models });
    if (path === "/api/skills") return json(route, { enabled: true, status: "ready", items: [] });
    if (path === "/api/projects") {
      if (request.method() === "POST") {
        const body = request.postDataJSON();
        posts.push({ path, body });
        project = { id: "proj_private", name: body.name, instructions: body.instructions, private_required: body.model_id === "private", created_at: DATE, updated_at: DATE };
        return json(route, project, 201);
      }
      return json(route, { items: [...(project ? [project] : []), { id: "proj_ordinary", name: "General project", instructions: "", private_required: false, created_at: DATE, updated_at: DATE }], next_cursor: null, limits: LIMITS });
    }
    if (path === "/api/conversations") return json(route, { items: [initialConversation, ...(projectConversation ? [projectConversation] : [])], next_cursor: null });
    if (path === "/api/conversations/con_existing") return json(route, initialConversation);
    if (path.endsWith("/messages")) return json(route, { items: [], next_cursor: null });
    if (path === "/api/projects/proj_private/conversations") {
      if (request.method() === "POST") {
        const body = request.postDataJSON();
        posts.push({ path, body });
        projectConversation = { ...initialConversation, id: "con_project", title: "Private project chat", private_required: true, default_mode: "local", default_model_id: "private", project_id: "proj_private" };
        return json(route, projectConversation, 201);
      }
      return json(route, { items: projectConversation ? [projectConversation] : [], next_cursor: null });
    }
    if (path === "/api/projects/proj_private/tax-reviews") return json(route, { items: [] });
    if (path === "/api/projects/proj_private/files") return json(route, { items: projectFiles.map((file) => ({ ...file, added_at: DATE })), limits: LIMITS });
    if (path === "/api/projects/proj_private/uploads") {
      uploadPaths.push(path);
      const file = document("project-upload", true);
      projectFiles.push(file);
      return json(route, file);
    }
    if (path === "/api/files") return json(route, { items: [document("ordinary", false), document("protected", true), document("protected-image", true, "image")], total_bytes: 9, server_time: DATE, limits: FILE_LIMITS });
    if (path === "/api/conversations/con_project") return json(route, projectConversation);
    return route.abort("failed");
  });
  return { posts, uploadPaths };
}

for (const viewport of [{ width: 1280, height: 900 }, { width: 390, height: 844 }]) {
  test(`Private project creation, document upload and conversation survive reload at ${viewport.width}px`, async ({ page }) => {
    await page.setViewportSize(viewport);
    const observed = await mockProjects(page);
    await page.goto("./");
    if (viewport.width === 390) await page.getByRole("button", { name: "Menu", exact: true }).click();
    await page.getByRole("button", { name: "New project", exact: true }).click();
    const dialog = page.getByRole("dialog", { name: "New project", exact: true });
    await dialog.getByLabel("Project name", { exact: true }).fill("Private review");
    await dialog.getByRole("button", { name: /Private Local models/u }).click();
    await dialog.getByRole("button", { name: "Create project", exact: true }).click();
    await expect(page.getByRole("heading", { name: "Private review", exact: true })).toBeVisible();
    await expect(page.getByText("Private project", { exact: true })).toBeVisible();
    const actions = page.locator(".project-home-header .project-management-actions");
    const edit = actions.getByRole("button", { name: "Edit project", exact: true });
    const remove = actions.getByRole("button", { name: "Delete project", exact: true });
    await expect(edit).toBeInViewport();
    await expect(remove).toBeInViewport();
    await expect(remove).toHaveAccessibleDescription(/Conversations and files will be kept/u);
    const editBounds = await edit.boundingBox();
    const removeBounds = await remove.boundingBox();
    expect(editBounds).not.toBeNull();
    expect(removeBounds).not.toBeNull();
    expect(Math.abs(editBounds!.y - removeBounds!.y)).toBeLessThan(1);
    expect(Math.abs(editBounds!.height - removeBounds!.height)).toBeLessThan(1);
    expect(Math.abs(editBounds!.width - removeBounds!.width)).toBeLessThan(1);
    await remove.click();
    const confirm = actions.getByRole("button", { name: "Confirm delete project", exact: true });
    await expect(confirm).toHaveText("✓");
    await confirm.press("Escape");
    await expect(remove).toHaveAttribute("aria-pressed", "false");
    await remove.click();
    await edit.click();
    await expect(remove).toHaveAttribute("aria-pressed", "false");
    await actions.getByRole("button", { name: "Close edit", exact: true }).click();
    expect(observed.posts[0]).toEqual({ path: "/api/projects", body: { name: "Private review", instructions: "", model_id: "private" } });
    const fileInput = page.locator('.project-file-actions input[type="file"]');
    await expect(fileInput).toHaveAttribute("accept", ".txt,.md,.csv,.pdf,.docx,.xlsx,.html,.htm");
    await fileInput.setInputFiles({ name: "project.txt", mimeType: "text/plain", buffer: Buffer.from("one") });
    await expect(page.getByText("project-upload.txt", { exact: true })).toBeVisible();
    expect(observed.uploadPaths).toEqual(["/api/projects/proj_private/uploads"]);
    await page.getByRole("button", { name: "Choose from My Files", exact: true }).click();
    const picker = page.getByRole("dialog", { name: "Choose project files", exact: true });
    await expect(picker.getByText("protected.txt", { exact: true })).toBeVisible();
    await expect(picker.getByText("ordinary.txt", { exact: true })).toHaveCount(0);
    await expect(picker.getByText("protected-image.png", { exact: true })).toHaveCount(0);
    await picker.getByRole("button", { name: "Close project files", exact: true }).click();
    await page.getByRole("button", { name: "New conversation in project", exact: true }).click();
    await expect(page.getByRole("button", { name: "Audrey model", exact: true })).toContainText("Private");
    await page.getByRole("button", { name: "Move conversation to project", exact: true }).click();
    const projects = page.getByRole("dialog", { name: "Choose a project", exact: true });
    await expect(projects.getByRole("button", { name: "Private review", exact: true })).toBeVisible();
    await expect(projects.getByRole("button", { name: "General project", exact: true })).toHaveCount(0);
    await page.keyboard.press("Escape");
    expect(observed.posts.at(-1)).toEqual({ path: "/api/projects/proj_private/conversations", body: { model_id: "private" } });
    await page.reload();
    if (viewport.width === 390) {
      await page.getByRole("button", { name: "Menu", exact: true }).click();
      await page.getByRole("button", { name: "My projects", exact: false }).click();
    }
    await page.locator(".project-sidebar-link").filter({ hasText: "Private review" }).click();
    await page.locator(".project-home-conversation").filter({ hasText: "Private project chat" }).click();
    await expect(page.getByRole("button", { name: "Audrey model", exact: true })).toContainText("Private");
    expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(viewport.width);
  });
}

test("new workflow choices prefer Private while a saved Local conversation keeps Local", async ({ page }) => {
  await mockProjects(page);
  await page.goto("./");
  await page.getByRole("button", { name: "Audrey model", exact: true }).click();
  const picker = page.getByRole("dialog", { name: "Choose an Audrey model", exact: true });
  await expect(picker.getByRole("button", { name: "Private", exact: true })).toBeVisible();
  await expect(picker.getByRole("button", { name: "Local", exact: true })).toHaveCount(0);
});

for (const omitLocal of [false, true]) {
  test(`saved Local conversation stays Local with ${omitLocal ? "missing" : "present"} catalog entry`, async ({ page }) => {
    await mockProjects(page, true, omitLocal);
    await page.goto("./");
    await expect(page.getByRole("button", { name: "Audrey model", exact: true })).toContainText("Local");
    await page.getByRole("button", { name: "Audrey model", exact: true }).click();
    await expect(page.getByRole("dialog", { name: "Choose an Audrey model", exact: true }).getByRole("button", { name: "Local", exact: true })).toHaveAttribute("aria-pressed", "true");
    await page.reload();
    await expect(page.getByRole("button", { name: "Audrey model", exact: true })).toContainText("Local");
  });
}
