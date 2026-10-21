import { apiJson } from "./api";

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
  provenance?: TaxFactProvenance | null;
}

export interface TaxCase {
  id: string;
  conversation_id: string;
  project_id: string | null;
  tax_year: number;
  jurisdiction: "US-federal";
  goal: string;
  reference_release: string;
  revision: number;
  checklist_version: number;
  facts: TaxFact[];
  files?: Array<{ file_id: string; checksum: string; revision: string; filename: string }>;
  evidence: TaxEvidence[];
  calculations: TaxCalculation[];
  missing_items: Array<{ id: string; label: string; status: "unresolved" | "resolved" }>;
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

export type TaxCaseAction =
  | { operation: "reject_fact"; data: { fact_id: string } }
  | { operation: "change_year"; data: { tax_year: number } }
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
