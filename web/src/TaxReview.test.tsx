import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";

import { TaxReview } from "./TaxReview";
import { defaultTaxYear, type TaxCase } from "./taxReviewApi";

// Snapshots were created through TaxCasesRepository.create/update and tax_case_checklist.
import fixtures from "./test/fixtures/tax-review-cases.json";

const taxCase = fixtures.proposed as TaxCase;

function json(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
}

function props() {
  return { conversationId: "con_private", expanded: true, active: false, disabled: false, refreshVersion: 0, onExpandedChange: vi.fn(), onCaseChange: vi.fn(), onSaved: vi.fn() };
}

afterEach(() => { cleanup(); vi.restoreAllMocks(); vi.unstubAllGlobals(); });

it("defaults to the most recent completed calendar year across year boundaries", () => {
  expect(defaultTaxYear(new Date(2026, 0, 1))).toBe(2025);
  expect(defaultTaxYear(new Date(2026, 11, 31))).toBe(2025);
  expect(defaultTaxYear(new Date(2027, 0, 1))).toBe(2026);
});

it("starts with the default year, allows changing it before creation and collapses after selecting", async () => {
  const saved = { ...fixtures.created, tax_year: 2024 };
  const fetchMock = vi.fn().mockResolvedValueOnce(json({ case: null })).mockResolvedValueOnce(json(saved));
  vi.stubGlobal("fetch", fetchMock);
  const callbacks = props();
  render(<TaxReview {...callbacks} />);
  await waitFor(() => expect(screen.getByLabelText("Tax year")).toBeEnabled());
  expect(screen.getByLabelText("Tax year")).toHaveValue(defaultTaxYear());
  expect(screen.getByText("US federal", { exact: true })).toBeVisible();
  expect(screen.queryByRole("combobox")).not.toBeInTheDocument();
  fireEvent.change(screen.getByLabelText("Tax year"), { target: { value: "2024" } });
  fireEvent.change(screen.getByLabelText("Review goal"), { target: { value: "Review deductions" } });
  fireEvent.click(screen.getByRole("button", { name: "Use Tax review" }));
  await waitFor(() => expect(callbacks.onSaved).toHaveBeenCalledOnce());
  expect(callbacks.onExpandedChange).toHaveBeenCalledWith(false);
  expect(callbacks.onCaseChange).toHaveBeenLastCalledWith(saved);
  expect(fetchMock).toHaveBeenCalledTimes(2);
  expect(fetchMock.mock.calls[1][0]).toBe("/api/conversations/con_private/tax-review");
  expect(JSON.parse(fetchMock.mock.calls[1][1].body)).toEqual({ tax_year: 2024, jurisdiction: "US-federal", goal: "Review deductions" });
});

it("saves a changed year on an existing case using the current revision without selecting it", async () => {
  const updated = { ...taxCase, tax_year: 2024, revision: taxCase.revision + 1 };
  const fetchMock = vi.fn().mockResolvedValueOnce(json({ case: taxCase })).mockResolvedValueOnce(json(updated));
  vi.stubGlobal("fetch", fetchMock);
  const callbacks = props();
  render(<TaxReview {...callbacks} />);
  await waitFor(() => expect(screen.getByLabelText("Tax year")).toHaveValue(2025));
  expect(screen.getByRole("button", { name: "Save year" })).toBeDisabled();
  expect(screen.queryByRole("textbox", { name: "Review goal" })).not.toBeInTheDocument();
  fireEvent.change(screen.getByLabelText("Tax year"), { target: { value: "2024" } });
  expect(screen.getByText("Saved information will need review for 2024.")).toBeVisible();
  fireEvent.click(screen.getByRole("button", { name: "Save year" }));
  await waitFor(() => expect(callbacks.onExpandedChange).toHaveBeenCalledWith(false));
  expect(JSON.parse(fetchMock.mock.calls[1][1].body)).toEqual({ expected_revision: taxCase.revision, operation: "change_year", data: { tax_year: 2024 } });
  expect(callbacks.onCaseChange).toHaveBeenLastCalledWith(updated);
  expect(callbacks.onSaved).not.toHaveBeenCalled();
});

it("keeps the saved year when a conflicting update is rejected and reloads current settings", async () => {
  const updated = { ...taxCase, tax_year: 2026, revision: taxCase.revision + 1 };
  const fetchMock = vi.fn().mockResolvedValueOnce(json({ case: taxCase })).mockResolvedValueOnce(json({ detail: "The case changed. Reload it." }, 409)).mockResolvedValueOnce(json({ case: updated }));
  vi.stubGlobal("fetch", fetchMock);
  const callbacks = props();
  render(<TaxReview {...callbacks} />);
  await waitFor(() => expect(screen.getByLabelText("Tax year")).toHaveValue(2025));
  fireEvent.change(screen.getByLabelText("Tax year"), { target: { value: "2024" } });
  fireEvent.click(screen.getByRole("button", { name: "Save year" }));
  expect(await screen.findByRole("alert")).toHaveTextContent("The case changed. Reload it. (HTTP 409)");
  expect(screen.getByRole("button", { name: "Tax review · 2025 Settings" })).toBeVisible();
  expect(callbacks.onExpandedChange).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole("button", { name: "Reload settings" }));
  await waitFor(() => expect(screen.getByLabelText("Tax year")).toHaveValue(2026));
  expect(callbacks.onCaseChange).toHaveBeenLastCalledWith(updated);
});

it.each(["1899", "2101"])("rejects unsupported tax year %s before sending it", async (year) => {
  const fetchMock = vi.fn().mockResolvedValueOnce(json({ case: taxCase }));
  vi.stubGlobal("fetch", fetchMock);
  render(<TaxReview {...props()} />);
  await waitFor(() => expect(screen.getByLabelText("Tax year")).toHaveValue(2025));
  fireEvent.change(screen.getByLabelText("Tax year"), { target: { value: year } });
  fireEvent.submit(screen.getByRole("button", { name: "Save year" }).closest("form")!);
  expect(await screen.findByRole("alert")).toHaveTextContent("Enter a tax year from 1900 to 2100.");
  expect(fetchMock).toHaveBeenCalledTimes(1);
});

it("keeps the collapsed settings summary free of review questions and technical details", async () => {
  vi.stubGlobal("fetch", vi.fn().mockResolvedValue(json({ case: fixtures.reviewable })));
  render(<TaxReview {...props()} expanded={false} />);
  expect(await screen.findByRole("button", { name: "Tax review · 2025 Settings" })).toHaveAttribute("aria-expanded", "false");
  expect(screen.queryByText("Wages", { exact: true })).not.toBeInTheDocument();
  expect(screen.queryByText(/Reference release|Checklist|SHA-256|evidence|calculations|Planning checklist/u)).not.toBeInTheDocument();
  expect(screen.queryByRole("button", { name: /Confirm|Resolve/u })).not.toBeInTheDocument();
  expect(screen.queryByRole("spinbutton")).not.toBeInTheDocument();
});

it("refreshes shared case data without selecting Tax review or displaying findings in settings", async () => {
  const updated = { ...taxCase, revision: taxCase.revision + 1 };
  const fetchMock = vi.fn().mockResolvedValueOnce(json({ case: taxCase })).mockResolvedValueOnce(json({ case: updated }));
  vi.stubGlobal("fetch", fetchMock);
  const callbacks = props();
  const view = render(<TaxReview {...callbacks} />);
  await waitFor(() => expect(callbacks.onCaseChange).toHaveBeenCalledWith(taxCase));
  view.rerender(<TaxReview {...callbacks} refreshVersion={1} />);
  await waitFor(() => expect(callbacks.onCaseChange).toHaveBeenLastCalledWith(updated));
  expect(callbacks.onSaved).not.toHaveBeenCalled();
  expect(screen.queryByText("Wages", { exact: true })).not.toBeInTheDocument();
});

it("keeps save unavailable after unreadable settings and offers a retry", async () => {
  vi.stubGlobal("fetch", vi.fn().mockResolvedValueOnce(json({ detail: "Reference library unavailable." }, 503)).mockResolvedValueOnce(json({ case: null })));
  render(<TaxReview {...props()} />);
  expect(await screen.findByRole("alert")).toHaveTextContent("Reference library unavailable. (HTTP 503)");
  expect(screen.getByRole("button", { name: "Use Tax review" })).toBeDisabled();
  fireEvent.click(screen.getByRole("button", { name: "Reload settings" }));
  await waitFor(() => expect(screen.getByRole("button", { name: "Use Tax review" })).toBeEnabled());
});
