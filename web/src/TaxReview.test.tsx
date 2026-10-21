import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";

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

beforeEach(() => {
  // jsdom does not implement the native dialog lifecycle.
  HTMLDialogElement.prototype.showModal = vi.fn(function (this: HTMLDialogElement) { this.open = true; });
  HTMLDialogElement.prototype.close = vi.fn(function (this: HTMLDialogElement) { this.open = false; });
});

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

it("keeps fact decisions in the conversation while settings stay collapsed", async () => {
  vi.stubGlobal("fetch", vi.fn().mockResolvedValue(json({ case: fixtures.reviewable })));
  render(<TaxReview {...props()} expanded={false} />);
  expect(await screen.findByRole("button", { name: "Tax review · 2025 Settings" })).toHaveAttribute("aria-expanded", "false");
  const panel = screen.getByRole("region", { name: "Tax review case" });
  expect(within(panel).queryByRole("region", { name: "Tax review choices" })).not.toBeInTheDocument();
  expect(within(panel).queryByRole("button", { name: "Confirm Wages" })).not.toBeInTheDocument();
  expect(screen.queryByText(/Reference release|Checklist|SHA-256|Planning checklist/u)).not.toBeInTheDocument();
  expect(screen.queryByRole("spinbutton")).not.toBeInTheDocument();
});

it("refreshes case data without selecting Tax review", async () => {
  const updated = { ...taxCase, revision: taxCase.revision + 1 };
  const fetchMock = vi.fn().mockResolvedValueOnce(json({ case: taxCase })).mockResolvedValueOnce(json({ case: updated }));
  vi.stubGlobal("fetch", fetchMock);
  const callbacks = props();
  const view = render(<TaxReview {...callbacks} />);
  await waitFor(() => expect(callbacks.onCaseChange).toHaveBeenCalledWith(taxCase));
  view.rerender(<TaxReview {...callbacks} refreshVersion={1} />);
  await waitFor(() => expect(callbacks.onCaseChange).toHaveBeenLastCalledWith(updated));
  expect(callbacks.onSaved).not.toHaveBeenCalled();
});

it("keeps save unavailable after unreadable settings and offers a retry", async () => {
  vi.stubGlobal("fetch", vi.fn().mockResolvedValueOnce(json({ detail: "Reference library unavailable." }, 503)).mockResolvedValueOnce(json({ case: null })));
  render(<TaxReview {...props()} />);
  expect(await screen.findByRole("alert")).toHaveTextContent("Reference library unavailable. (HTTP 503)");
  expect(screen.getByRole("button", { name: "Use Tax review" })).toBeDisabled();
  fireEvent.click(screen.getByRole("button", { name: "Reload settings" }));
  await waitFor(() => expect(screen.getByRole("button", { name: "Use Tax review" })).toBeEnabled());
});



it("offers a library update inside the collapsed box without exposing release IDs", async () => {
  const initial = { ...taxCase, reference_update: { status: "update_available", release_id: "rel_new" } };
  const updated = { ...taxCase, revision: taxCase.revision + 1, reference_release: "rel_new", reference_update: { status: "current" } };
  const fetchMock = vi.fn().mockResolvedValueOnce(json({ case: initial })).mockResolvedValueOnce(json(updated));
  vi.stubGlobal("fetch", fetchMock);
  const callbacks = props();
  render(<TaxReview {...callbacks} expanded={false} />);
  const panel = await screen.findByRole("region", { name: "Tax review case" });
  fireEvent.click(await within(panel).findByRole("button", { name: "Update IRS library" }));
  await waitFor(() => expect(callbacks.onCaseChange).toHaveBeenLastCalledWith(updated));
  expect(JSON.parse(fetchMock.mock.calls[1][1].body)).toEqual({
    expected_revision: taxCase.revision, operation: "update_reference_release", data: { reference_release: "rel_new" },
  });
  expect(panel).not.toHaveTextContent("rel_new");
  expect(within(panel).queryByRole("button", { name: "Update IRS library" })).not.toBeInTheDocument();
  expect(callbacks.onSaved).not.toHaveBeenCalled();
});

it("keeps saved settings readable during a library outage but pauses year and fact choices", async () => {
  vi.stubGlobal("fetch", vi.fn().mockResolvedValueOnce(json({ case: {
    ...taxCase, reference_update: { status: "unavailable" },
  } })));
  render(<TaxReview {...props()} />);
  expect(await screen.findByText(/The IRS library is unavailable/u)).toBeVisible();
  expect(screen.getByLabelText("Tax year")).toHaveValue(2025);
  expect(screen.getByLabelText("Tax year")).toBeDisabled();
  expect(screen.queryByRole("button", { name: "Confirm Wages" })).not.toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Reload library" })).toBeEnabled();
});

it("requires reload after a stale library choice and never silently adopts a newer target", async () => {
  const initial = { ...taxCase, reference_update: { status: "update_available", release_id: "rel_old_choice" } };
  const latest = { ...initial, reference_update: { status: "update_available", release_id: "rel_new_choice" } };
  const fetchMock = vi.fn().mockResolvedValueOnce(json({ case: initial }))
    .mockResolvedValueOnce(json({ detail: "The IRS library changed again. Reload Tax review before updating." }, 409))
    .mockResolvedValueOnce(json({ case: latest }));
  vi.stubGlobal("fetch", fetchMock);
  const callbacks = props();
  render(<TaxReview {...callbacks} expanded={false} />);
  fireEvent.click(await screen.findByRole("button", { name: "Update IRS library" }));
  expect(await screen.findByRole("alert")).toHaveTextContent("The IRS library changed again.");
  expect(screen.getByRole("button", { name: "Update IRS library" })).toBeDisabled();
  expect(callbacks.onCaseChange).not.toHaveBeenCalledWith(latest);
  fireEvent.click(screen.getByRole("button", { name: "Reload settings" }));
  await waitFor(() => expect(screen.getByRole("button", { name: "Update IRS library" })).toBeEnabled());
  expect(callbacks.onCaseChange).toHaveBeenLastCalledWith(latest);
  expect(fetchMock).toHaveBeenCalledTimes(3);
});


it("offers review of an unfinished case without confirming its proposed facts", async () => {
  const fetchMock = vi.fn().mockResolvedValueOnce(json({ case: taxCase }));
  vi.stubGlobal("fetch", fetchMock);
  const callbacks = { ...props(), onReview: vi.fn() };
  render(<TaxReview {...callbacks} expanded={false} />);
  fireEvent.click(await screen.findByRole("button", { name: "Start review" }));
  expect(callbacks.onReview).toHaveBeenCalledOnce();
  expect(fetchMock).toHaveBeenCalledTimes(1);
  expect(callbacks.onSaved).not.toHaveBeenCalled();
  expect(screen.queryByRole("button", { name: "Confirm Wages" })).not.toBeInTheDocument();
});

it.each(["update_available", "unavailable"] as const)("pauses review when the IRS library is %s", async (status) => {
  vi.stubGlobal("fetch", vi.fn().mockResolvedValueOnce(json({ case: { ...taxCase,
    reference_update: { status, ...(status === "update_available" ? { release_id: "rel_new" } : {}) },
  } })));
  render(<TaxReview {...props()} onReview={vi.fn()} expanded={false} />);
  expect(await screen.findByRole("button", { name: "Start review" })).toBeDisabled();
});

it("keeps the saved worksheet available on demand and warns when it needs updating", async () => {
  const saved = { ...taxCase, calculations: [{ id: "calc_private", label: "Adjusted total", expression: "1250 * 1.08", result: "1350" }], review: {
    summary: "The arithmetic is consistent. Filing information is incomplete.",
    findings: [{ label: "Adjustment", detail: "An eight-percent adjustment is recorded.", fact_ids: [], evidence_ids: [], calculation_ids: ["calc_private"], return_item_ids: [] }],
    next_question: "Did you receive any other income?", status: "needs_input", stale: true,
    limitations: ["This review does not cover state taxes."], tax_year: 2025, reference_release: "rel_hidden", review_revision: 3,
  } };
  vi.stubGlobal("fetch", vi.fn().mockResolvedValueOnce(json({ case: saved })));
  render(<TaxReview {...props()} onReview={vi.fn()} expanded={false} />);
  expect(await screen.findByRole("button", { name: "Continue review" })).toBeEnabled();
  expect(screen.getByText("Review needs updating")).toBeVisible();
  const details = screen.getByText("Review worksheet").closest("details")!;
  expect(details).not.toHaveAttribute("open");
  fireEvent.click(screen.getByText("Review worksheet"));
  expect(screen.getByText(saved.review.summary)).toBeVisible();
  expect(screen.getByText("Adjusted total: 1250 * 1.08 = 1350")).toBeVisible();
  expect(screen.getByText(saved.review.next_question)).toBeVisible();
  expect(screen.getByText(saved.review.limitations[0])).toBeVisible();
  expect(screen.getByRole("region", { name: "Tax review case" })).not.toHaveTextContent(/rel_hidden|calc_private|review_revision/u);
});

it("keeps review unavailable during an active run or upload", async () => {
  vi.stubGlobal("fetch", vi.fn().mockResolvedValueOnce(json({ case: taxCase })));
  const onReview = vi.fn();
  render(<TaxReview {...props()} onReview={onReview} disabled expanded={false} />);
  expect(await screen.findByRole("button", { name: "Start review" })).toBeDisabled();
  fireEvent.click(screen.getByRole("button", { name: "Start review" }));
  expect(onReview).not.toHaveBeenCalled();
});

it("opens settings independently of the composer and closes without saving", async () => {
  const fetchMock = vi.fn().mockResolvedValue(json({ case: null }));
  vi.stubGlobal("fetch", fetchMock);
  const callbacks = props();
  render(<TaxReview {...callbacks} />);
  const dialog = screen.getByRole("dialog", { name: "Tax review settings" });
  expect(dialog).toHaveAttribute("open");
  expect(screen.getByRole("heading", { name: "Tax review settings" })).toHaveFocus();
  expect(screen.getByLabelText("Review goal")).not.toHaveFocus();
  fireEvent.click(screen.getByRole("button", { name: "Close Tax review settings" }));
  expect(callbacks.onExpandedChange).toHaveBeenCalledWith(false);
  expect(callbacks.onSaved).not.toHaveBeenCalled();
  fireEvent(dialog, new Event("cancel", { cancelable: true }));
  expect(callbacks.onExpandedChange).toHaveBeenCalledTimes(2);
  await waitFor(() => expect(callbacks.onCaseChange).toHaveBeenCalledWith(null));
  expect(fetchMock).toHaveBeenCalledTimes(1);
});

it("fits settings to the visible viewport when a mobile keyboard reduces it", async () => {
  const viewport = Object.assign(new EventTarget(), { width: 390, height: 844, offsetLeft: 0, offsetTop: 0 });
  vi.stubGlobal("visualViewport", viewport);
  vi.stubGlobal("fetch", vi.fn().mockResolvedValue(json({ case: null })));
  render(<TaxReview {...props()} />);
  const dialog = screen.getByRole("dialog", { name: "Tax review settings" });
  expect(dialog.style.getPropertyValue("--tax-dialog-width")).toBe("366px");
  expect(dialog.style.getPropertyValue("--tax-dialog-height")).toBe("820px");
  viewport.height = 340;
  viewport.offsetTop = 30;
  viewport.dispatchEvent(new Event("resize"));
  expect(dialog.style.getPropertyValue("--tax-dialog-height")).toBe("316px");
  expect(dialog.style.getPropertyValue("--tax-dialog-top")).toBe("200px");
  await waitFor(() => expect(screen.getByLabelText("Tax year")).toBeEnabled());
});
