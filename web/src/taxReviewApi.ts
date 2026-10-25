import { apiJson, apiResponse } from "./api";

export interface TaxFactProvenance {
  file_id: string;
  checksum: string;
  page: number;
  box?: string;
  original_text: string;
  normalization?: string;
}

export interface TaxEvidence {
  id: string;
  claim: string;
  document_id: string;
  reference_release: string;
  revision: string;
  tax_year: number | null;
  tax_year_status: "unknown" | "proposed" | "verified";
  jurisdiction: string;
  catalog: "current" | "prior" | "bulletins";
  page: number;
  section?: string;
  passage: string;
  conflicts_with: string[];
}

export interface TaxCalculation {
  id: string;
  label: string;
  inputs: Record<string, string>;
  expression: string;
  rounding: string;
  result: string;
  fact_ids: string[];
  evidence_ids: string[];
  case_tax_year?: number;
  status?: "recorded" | "needs_recheck";
  recheck_for_tax_year?: number;
}

export interface TaxFact {
  id: string;
  label: string;
  value: unknown;
  kind: string;
  status: "proposed" | "confirmed" | "rejected";
  case_tax_year?: number;
  confirmed_tax_year?: number;
  confirmed_at?: string;
  confirmed_by?: string;
  source_message_id?: string;
  rejected_tax_year?: number;
  rejected_at?: string;
  created_at?: string;
  provenance?: TaxFactProvenance | null;
}

export const TAX_REVIEW_PROMPT = "Review my saved tax case. Use the confirmed facts and selected documents, retain official evidence, check the arithmetic, and save a review worksheet. Show unresolved questions rather than guessing.";

export interface TaxReviewWorksheet {
  summary: string;
  findings: Array<{
    label: string;
    detail: string;
    fact_ids: string[];
    evidence_ids: string[];
    calculation_ids: string[];
    return_item_ids: string[];
  }>;
  next_question: string;
  status: "needs_input" | "limited_evidence" | "review_saved";
  stale: boolean;
  limitations: string[];
  tax_year: number;
  reference_release: string;
  review_revision: number;
  document_scope?: string;
  scope?: "evidence_and_arithmetic";
  eligibility_status?: "not_determined";
  notice?: string;
}

export const TAX_REVIEW_SCOPE_NOTICE = "This worksheet reviews selected sources and calculations; it does not determine eligibility for deductions or credits. It does not establish that a tax return is complete or legally correct.";

export function taxReviewScopeNotice(review: TaxReviewWorksheet): string {
  // Older responses can contain a generic notice without the eligibility limit.
  return review.scope === "evidence_and_arithmetic" && review.notice?.trim()
    ? review.notice
    : TAX_REVIEW_SCOPE_NOTICE;
}

export interface TaxCase {
  id: string;
  conversation_id: string;
  project_id: string | null;
  tax_year: number;
  jurisdiction: "US-federal";
  goal: string;
  reference_release: string;
  reference_update?:
    | { status: "current" | "unavailable" }
    | { status: "update_available"; release_id: string };
  revision: number;
  review?: TaxReviewWorksheet | null;
  checklist_version: number;
  facts: TaxFact[];
  files?: Array<{ file_id: string; checksum: string; revision: string; filename: string }>;
  evidence: TaxEvidence[];
  calculations: TaxCalculation[];
  missing_items: Array<{ id: string; label: string; question?: string; status: "unresolved" | "resolved"; resolved_at?: string; created_at?: string }>;
  return_items: unknown[];
  checklist: {
    status: "needs_information" | "ready_for_review";
    checks: Array<{
      id: string;
      label: string;
      status: "needs_information" | "passed";
      issues: string[];
    }>;
    next_steps: string[];
    document_scope?: string;
    notice?: string;
  };
  created_at: string;
  updated_at: string;
}

export interface TaxCaseSetup {
  tax_year: number;
  jurisdiction: "US-federal";
  goal: string;
}

export function defaultTaxYear(now = new Date()): number {
  return now.getFullYear() - 1;
}

function casePath(conversationId: string): string {
  return `/api/conversations/${encodeURIComponent(conversationId)}/tax-review`;
}

export function getTaxCase(conversationId: string, signal?: AbortSignal): Promise<{ case: TaxCase | null }> {
  return apiJson(casePath(conversationId), { signal });
}

export function saveTaxCase(conversationId: string, setup: TaxCaseSetup): Promise<TaxCase> {
  return apiJson(casePath(conversationId), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(setup),
  });
}

export function createTaxCaseProject(conversationId: string): Promise<TaxCase> {
  return apiJson(`${casePath(conversationId)}/project`, { method: "POST" });
}

export function listProjectTaxCases(projectId: string, signal?: AbortSignal): Promise<{ items: TaxCase[] }> {
  return apiJson(`/api/projects/${encodeURIComponent(projectId)}/tax-reviews`, { signal });
}

export type TaxCaseAction =
  | { operation: "confirm_facts"; data: { fact_ids: string[] } }
  | { operation: "reject_fact"; data: { fact_id: string } }
  | { operation: "edit_fact"; data: { fact_id: string; label: string; value: string } }
  | { operation: "change_year"; data: { tax_year: number } }
  | { operation: "update_reference_release"; data: { reference_release: string } }
  | { operation: "confirm_fact"; data: { fact_id: string } }
  | { operation: "resolve_missing_item"; data: { id: string } };

export function actOnTaxCase(
  conversationId: string,
  expectedRevision: number,
  action: TaxCaseAction,
): Promise<TaxCase> {
  return apiJson(`${casePath(conversationId)}/actions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ expected_revision: expectedRevision, ...action }),
  });
}

export interface TaxWorkbookExport {
  id: string;
  case_revision: number;
  review_revision: number;
  filename: string;
  download_url: string;
}

export const TAX_WORKBOOK_CONTENT_TYPE = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet";

export function isTaxWorkbookUrl(path: string | undefined): path is string {
  return Boolean(path && /^\/api\/conversations\/[^/?#]+\/tax-review\/exports\/[^/?#]+\/download$/u.test(path));
}

export async function prepareTaxWorkbook(taxCase: TaxCase, signal?: AbortSignal): Promise<TaxWorkbookExport> {
  const result = await apiJson<{ export: TaxWorkbookExport }>(`${casePath(taxCase.conversation_id)}/exports`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ expected_revision: taxCase.revision }),
    signal,
  });
  const item = result.export;
  if (!isTaxWorkbookUrl(item.download_url)
    || item.download_url !== `${casePath(taxCase.conversation_id)}/exports/${encodeURIComponent(item.id)}/download`
    || item.case_revision !== taxCase.revision) {
    throw new Error("The workbook did not match this saved review.");
  }
  return item;
}

export async function downloadTaxWorkbook(path: string, filename?: string, signal?: AbortSignal): Promise<void> {
  if (!isTaxWorkbookUrl(path)) throw new Error("The workbook link is invalid.");
  const response = await apiResponse(path, { signal, headers: { Accept: TAX_WORKBOOK_CONTENT_TYPE } });
  if (response.headers.get("Content-Type")?.split(";")[0].trim() !== TAX_WORKBOOK_CONTENT_TYPE) {
    throw new Error("The server did not return a workbook.");
  }
  const blob = await response.blob();
  if (signal?.aborted) return;
  const disposition = response.headers.get("Content-Disposition") ?? "";
  const encodedName = /filename\*=UTF-8''([^;]+)/iu.exec(disposition)?.[1];
  let headerName = /filename="([^"]+)"/iu.exec(disposition)?.[1];
  if (encodedName) {
    try { headerName = decodeURIComponent(encodedName); } catch { /* Use the plain filename. */ }
  }
  const name = (filename || headerName || "tax-review.xlsx").split(/[\\/]/u).pop()!.replace(/\p{Cc}/gu, "");
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = name || "tax-review.xlsx";
  document.body.append(anchor);
  try { anchor.click(); } finally {
    anchor.remove();
    window.setTimeout(() => URL.revokeObjectURL(url), 0);
  }
}
