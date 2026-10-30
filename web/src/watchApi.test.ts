import { afterEach, expect, it, vi } from "vitest";
import { actOnWatch, createWatch, deleteWatch, getWatch, getWatchRun, getWatchUnreadCount, listWatchNotifications, listWatchRuns, listWatches, previewWatchSchedule, readWatchNotification, updateWatch, type WatchDefinition } from "./api";
afterEach(() => vi.unstubAllGlobals());
it("keeps watch reads cancellable and mutations revisioned with owner authentication", async () => {
  const fetchMock = vi.fn().mockImplementation(async () => new Response(JSON.stringify({ items: [], occurrences: [] }), { status: 200, headers: { "Content-Type": "application/json" } })); vi.stubGlobal("fetch", fetchMock);
  const signal = new AbortController().signal;
  const definition: WatchDefinition = { title: "New release", url: "https://example.com/releases", criterion: "Stable releases", enabled: true, schedule: { kind: "weekly", weekday: 0, time: "09:00", timezone: "America/Denver" } };
  await listWatches(signal); await getWatch("watch / one", signal); await previewWatchSchedule(definition.schedule, signal); await createWatch(definition, "request-1"); await updateWatch("watch / one", definition, 2); await actOnWatch("watch / one", "run", 3); await deleteWatch("watch / one", 4); await listWatchRuns("watch / one", signal); await getWatchRun("watch / one", "run / one", signal); await listWatchNotifications(signal); await getWatchUnreadCount(signal); await readWatchNotification("notice / one");
  expect(fetchMock).toHaveBeenCalledWith("/api/watches/watch%20%2F%20one", expect.objectContaining({ signal, cache: "no-store" }));
  expect(fetchMock).toHaveBeenCalledWith("/api/watches/preview", expect.objectContaining({ method: "POST", signal, body: JSON.stringify({ schedule: definition.schedule }) }));
  expect(fetchMock).toHaveBeenCalledWith("/api/watches", expect.objectContaining({ method: "POST", body: JSON.stringify({ ...definition, request_id: "request-1" }) }));
  expect(fetchMock).toHaveBeenCalledWith("/api/watches/watch%20%2F%20one", expect.objectContaining({ method: "PATCH", body: JSON.stringify({ ...definition, expected_revision: 2 }) }));
  expect(fetchMock).toHaveBeenCalledWith("/api/watches/watch%20%2F%20one/actions", expect.objectContaining({ method: "POST", body: JSON.stringify({ action: "run", expected_revision: 3 }) }));
  expect(fetchMock).toHaveBeenCalledWith("/api/watches/watch%20%2F%20one?expected_revision=4", expect.objectContaining({ method: "DELETE" }));
  expect(fetchMock).toHaveBeenCalledWith("/api/watches/watch%20%2F%20one/runs/run%20%2F%20one", expect.objectContaining({ signal, cache: "no-store" }));
  expect(fetchMock).toHaveBeenCalledWith("/api/notifications/unread", expect.objectContaining({ signal, cache: "no-store" }));
  expect(fetchMock).toHaveBeenCalledWith("/api/notifications/notice%20%2F%20one/read", expect.objectContaining({ method: "POST" }));
  for (const [, init] of fetchMock.mock.calls as [string, RequestInit][]) { expect(init.credentials).toBe("same-origin"); expect(init.headers).not.toHaveProperty("Authorization"); }
});
