import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";

import { TaxReview } from "./TaxReview";
import type { TaxCase } from "./taxReviewApi";

// Snapshots were created through TaxCasesRepository.create/update and tax_case_checklist.
import fixtures from "./test/fixtures/tax-review-cases.json";

const taxCase = fixtures.proposed as TaxCase;

function json(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
}

function props() {
  return { conversationId: "con_private", expanded: true, active: false, disabled: false, refreshVersion: 0, onExpandedChange: vi.fn(), onCaseChange: vi.fn(), onSaved: vi.fn() };
}

afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

it("saves explicit year and federal goal without an upload or model request", async () => {
  const fetchMock = vi.fn().mockResolvedValueOnce(json({ case: null })).mockResolvedValueOnce(json(fixtures.created));
  vi.stubGlobal("fetch", fetchMock);
  const callbacks = props();
  render(<TaxReview {...callbacks} />);
  await waitFor(() => expect(screen.getByLabelText("Tax year")).toBeEnabled());
  expect(screen.getByLabelText("Tax year")).toHaveValue(null);
  fireEvent.change(screen.getByLabelText("Tax year"), { target: { value: "2025" } });
  fireEvent.change(screen.getByLabelText("Review goal"), { target: { value: "Review deductions" } });
  fireEvent.click(screen.getByRole("button", { name: "Save and select Tax review" }));
  await waitFor(() => expect(callbacks.onSaved).toHaveBeenCalledOnce());
  expect(fetchMock).toHaveBeenCalledTimes(2);
  expect(fetchMock.mock.calls[1][0]).toBe("/api/conversations/con_private/tax-review");
  expect(JSON.parse(fetchMock.mock.calls[1][1].body)).toEqual({ tax_year: 2025, jurisdiction: "US-federal", goal: "Review deductions" });
  expect(screen.getByText("Planning checklist, not a completed tax return. No upload required.")).toBeVisible();
  expect(screen.getByText("Review deductions", { exact: true })).toBeVisible();
  expect(screen.queryByText("Confirmed by you")).not.toBeInTheDocument();
});

it("confirms facts and resolves missing items only after the server accepts the current revision", async () => {
  const confirmed = fixtures.confirmed;
  const resolved = fixtures.resolved;
  const fetchMock = vi.fn().mockResolvedValueOnce(json({ case: taxCase })).mockResolvedValueOnce(json(confirmed)).mockResolvedValueOnce(json(resolved));
  vi.stubGlobal("fetch", fetchMock);
  render(<TaxReview {...props()} />);
  fireEvent.click(await screen.findByRole("button", { name: "Confirm Wages" }));
  expect(await screen.findByText("Confirmed by you")).toBeVisible();
  expect(JSON.parse(fetchMock.mock.calls[1][1].body)).toEqual({ expected_revision: taxCase.revision, operation: "confirm_fact", data: { fact_id: taxCase.facts[0].id } });
  fireEvent.click(screen.getByRole("button", { name: "Resolve W-2 box 1" }));
  expect(await screen.findByText("Resolved")).toBeVisible();
  expect(JSON.parse(fetchMock.mock.calls[2][1].body)).toEqual({ expected_revision: confirmed.revision, operation: "resolve_missing_item", data: { id: taxCase.missing_items[0].id } });
});

it("retains proposed facts when a stale revision is rejected and offers reload", async () => {
  const fetchMock = vi.fn().mockResolvedValueOnce(json({ case: taxCase })).mockResolvedValueOnce(json({ detail: "Case changed; reload it." }, 409)).mockResolvedValueOnce(json({ case: { ...taxCase, revision: 5 } }));
  vi.stubGlobal("fetch", fetchMock);
  render(<TaxReview {...props()} />);
  fireEvent.click(await screen.findByRole("button", { name: "Confirm Wages" }));
  expect(await screen.findByRole("alert")).toHaveTextContent("Case changed; reload it. (HTTP 409)");
  expect(screen.getByRole("button", { name: "Confirm Wages" })).toBeEnabled();
  expect(screen.queryByText("Confirmed by you")).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "Reload case" }));
  expect(await screen.findByText(/Revision 5/u)).toBeVisible();
});

it("refreshes saved findings after a completed run without selecting the case", async () => {
  const fetchMock = vi.fn().mockResolvedValueOnce(json({ case: taxCase })).mockResolvedValueOnce(json({ case: { ...taxCase, revision: 5, evidence: [...taxCase.evidence, { ...taxCase.evidence[0], id: "ev_rule" }] } }));
  vi.stubGlobal("fetch", fetchMock);
  const callbacks = props();
  const view = render(<TaxReview {...callbacks} />);
  await screen.findByText("1 evidence · 0 calculations");
  view.rerender(<TaxReview {...callbacks} refreshVersion={1} />);
  expect(await screen.findByText("2 evidence · 0 calculations")).toBeVisible();
  expect(callbacks.onSaved).not.toHaveBeenCalled();
});

it("keeps save unavailable after an unreadable case and displays the server failure", async () => {
  vi.stubGlobal("fetch", vi.fn().mockResolvedValue(json({ detail: "Reference library unavailable." }, 503)));
  render(<TaxReview {...props()} />);
  expect(await screen.findByRole("alert")).toHaveTextContent("Reference library unavailable. (HTTP 503)");
  expect(screen.getByRole("button", { name: "Save and select Tax review" })).toBeDisabled();
});

it("shows original document provenance before confirmation and retained rule passages and calculated results", async () => {
  vi.stubGlobal("fetch", vi.fn().mockResolvedValue(json({ case: fixtures.reviewable })));
  render(<TaxReview {...props()} />);
  await screen.findByRole("button", { name: "Confirm Wages" });
  fireEvent.click(screen.getByText("Source · Page 2 · Box 1", { exact: true }));
  expect(screen.getByText("Wages statement.txt", { exact: true })).toBeVisible();
  expect(screen.getByText("Box 1 Wages, tips, other compensation: $55,000.00", { exact: true })).toBeVisible();
  expect(screen.getByText(/Normalized value: 55000.00/u)).toBeVisible();
  expect(screen.getByTitle("a".repeat(64))).toBeVisible();
  fireEvent.click(screen.getByText("Evidence (1)", { exact: true }));
  expect(screen.getByText("A retained official passage for this test fixture.", { exact: true })).toBeVisible();
  expect(screen.getByText(/Tax year: 2025 · Applicability: proposed/u)).toBeVisible();
  fireEvent.click(screen.getByText("Calculations (1)", { exact: true }));
  expect(screen.getByText("wages - adjustment", { exact: true })).toBeVisible();
  expect(screen.getByText("wages", { exact: true })).toBeVisible();
  expect(screen.getByText("adjustment", { exact: true })).toBeVisible();
  expect(screen.getByText("54000.00", { exact: true })).toBeVisible();
  expect(screen.getByText(/Rounding: half_up_cent/u)).toBeVisible();
  expect(screen.getByText("Dependencies: 1 facts · 1 evidence", { exact: true })).toBeVisible();
  expect(screen.queryByText("Confirmed by you")).not.toBeInTheDocument();
});
