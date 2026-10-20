import type { Conversation } from "./api";

function messageTime(conversation: Conversation): number {
  return Date.parse(conversation.last_message_at ?? conversation.created_at);
}

export function upsertConversation(
  conversations: Conversation[],
  updated: Conversation,
): Conversation[] {
  const previous = conversations.find(({ id }) => id === updated.id);
  if (previous === updated) return conversations;
  // An auxiliary title/model read may finish after a newer message snapshot.
  // Message activity is monotonic; metadata refreshes cannot move it backward.
  const merged = previous?.last_message_at
    && (!updated.last_message_at || messageTime(previous) > messageTime(updated))
    ? { ...updated, last_message_at: previous.last_message_at }
    : updated;
  // Match the server's activity ordering. Selecting or renaming an old chat
  // does not make it a newly active conversation.
  return [...conversations.filter(({ id }) => id !== updated.id), merged].sort((left, right) =>
    messageTime(right) - messageTime(left) || right.id.localeCompare(left.id),
  );
}
