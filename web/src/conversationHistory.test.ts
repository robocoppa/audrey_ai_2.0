import { describe, expect, test } from "vitest";
import type { Conversation } from "./api";
import { upsertConversation } from "./conversationHistory";

function conversation(id: string, lastMessage: string | null): Conversation {
  return {
    id, title: id, default_mode: "auto", default_model_id: "auto",
    created_at: "2026-10-19T12:00:00Z", updated_at: "2026-10-19T12:00:00Z",
    last_message_at: lastMessage, archived_at: null, project_id: null,
  };
}

const earlier = "2026-10-19T12:00:00Z";
const recent = "2026-10-20T12:00:00Z";

describe("conversation activity", () => {
  test("newly active conversations rise above existing history", () => {
    const old = conversation("older", earlier);
    const active = conversation("active", recent);
    const updated = { ...old, last_message_at: "2026-10-20T07:00:00-06:00" };
    const current = [active, old];
    expect(upsertConversation(current, updated).map(({ id }) => id)).toEqual(["older", "active"]);
    expect(current).toEqual([active, old]);
  });

  test("a first message inserts a hidden draft at its correct activity position", () => {
    const old = conversation("older", earlier);
    const draft = { ...conversation("draft", null), created_at: recent };
    expect(upsertConversation([old], draft).map(({ id }) => id)).toEqual(["draft", "older"]);
    expect(upsertConversation([old], { ...draft, last_message_at: recent }).map(({ id }) => id))
      .toEqual(["draft", "older"]);
  });

  test("a stale metadata refresh cannot erase or rewind known message activity", () => {
    const active = conversation("active", recent);
    const old = conversation("older", earlier);
    for (const staleTime of [earlier, null]) {
      const result = upsertConversation([active, old], {
        ...active, title: "Generated title", last_message_at: staleTime,
      });
      expect(result.map(({ id }) => id)).toEqual(["active", "older"]);
      expect(result[0]).toMatchObject({ title: "Generated title", last_message_at: recent });
    }
  });

  test("selecting or renaming an old conversation does not promote it", () => {
    const active = conversation("active", recent);
    const old = conversation("older", earlier);
    expect(upsertConversation([active, old], { ...old, title: "Renamed", updated_at: recent })
      .map(({ id }) => id)).toEqual(["active", "older"]);
  });
});
