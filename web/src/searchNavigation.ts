import {
  ApiError, getAgentTask, getConversation, getFile, getProject, getSearchSource,
  type AgentTask, type AudreyFile, type AudreyProject, type Conversation,
  type SearchResult, type SearchTarget,
} from "./api";

type WorkspaceTarget =
  | { kind: "message"; conversation: Conversation; messageId: string }
  | { kind: "file"; file: AudreyFile; artifact: Extract<SearchTarget, { kind: "file" }>["artifact"] }
  | { kind: "agent"; task: AgentTask }
  | { kind: "knowledge"; project: AudreyProject; entryId: string };

function sameTarget(left: SearchTarget, right: SearchTarget): boolean {
  if (left.kind !== right.kind) return false;
  switch (left.kind) {
    case "message": return right.kind === "message" && left.conversation_id === right.conversation_id
      && left.message_id === right.message_id && left.sequence === right.sequence;
    case "file": return right.kind === "file" && left.file_id === right.file_id && left.artifact === right.artifact;
    case "agent": return right.kind === "agent" && left.task_id === right.task_id && left.step_id === right.step_id;
    case "knowledge": return right.kind === "knowledge" && left.project_id === right.project_id && left.entry_id === right.entry_id;
  }
}

/** Read owned canonical work, then recheck the original source revision before following it. */
export async function resolveSearchWorkspace(result: SearchResult, signal: AbortSignal): Promise<WorkspaceTarget> {
  const changed = () => new ApiError(409, "This source changed after the search. Search again before opening its workspace.");
  if (signal.aborted) throw new DOMException("Navigation cancelled.", "AbortError");
  if (result.kind !== result.target.kind) throw changed();
  let destination: WorkspaceTarget;
  switch (result.target.kind) {
    case "message": {
      const conversation = await getConversation(result.target.conversation_id, signal);
      if (conversation.id !== result.target.conversation_id
        || (conversation.project_id ?? null) !== result.project_id
        || Boolean(conversation.private_required || conversation.default_model_id === "private") !== result.private_required) throw changed();
      destination = { kind: "message", conversation, messageId: result.target.message_id };
      break;
    }
    case "file": {
      const file = await getFile(result.target.file_id, signal);
      if (file.id !== result.target.file_id || Boolean(file.private_required) !== result.private_required) throw changed();
      destination = { kind: "file", file, artifact: result.target.artifact };
      break;
    }
    case "agent": {
      const task = await getAgentTask(result.target.task_id, signal);
      if (task.id !== result.target.task_id) throw changed();
      destination = { kind: "agent", task };
      break;
    }
    case "knowledge": {
      const project = await getProject(result.target.project_id, signal);
      if (project.id !== result.target.project_id || Boolean(project.private_required) !== result.private_required) throw changed();
      destination = { kind: "knowledge", project, entryId: result.target.entry_id };
      break;
    }
  }
  // This awaited check also fences deletion, detachment and replaced reports during the target read.
  if (signal.aborted) throw new DOMException("Navigation cancelled.", "AbortError");
  const source = await getSearchSource(result, result.offset, signal);
  if (signal.aborted) throw new DOMException("Navigation cancelled.", "AbortError");
  if (source.source_id !== result.source_id || source.revision !== result.revision || source.kind !== result.kind
    || source.private_required !== result.private_required || !sameTarget(source.target, result.target)) throw changed();
  return destination;
}
