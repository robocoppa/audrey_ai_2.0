import type { AudreyProject, KnowledgeEntry } from "./api";

export interface KnowledgeAgentRequest {
  project: AudreyProject;
  entry: KnowledgeEntry;
  serial: number;
}

export function canAssignKnowledge(project: AudreyProject, entry: KnowledgeEntry): boolean {
  return !project.private_required && entry.project_id === project.id
    && entry.approval_state === "approved" && entry.work_status === "open"
    && (entry.kind === "action" || entry.kind === "question") && !entry.stale
    && entry.sources.every(({ source_id }) => /^(?:message:[^:/\\\0]+|agent:[^:/\\\0]+:report|file:[^:/\\\0]+:(?:text|transcript|visual|summary))$/.test(source_id));
}

export function knowledgeTaskObjective(entry: KnowledgeEntry): string {
  return entry.text;
}
