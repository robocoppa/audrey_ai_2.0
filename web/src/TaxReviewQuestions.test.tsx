import { useState } from "react";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";

import { TaxReviewQuestions } from "./TaxReviewQuestions";
import type { TaxCase } from "./taxReviewApi";
import fixtures from "./test/fixtures/tax-review-cases.json";

const proposed = fixtures.reviewable as TaxCase;
const factId = proposed.facts[0].id;
function json(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
}
function Harness({ initial = proposed, disabled = false, answerVersion = 0, onChange = vi.fn(), onReload = vi.fn(), onChoice = vi.fn() }) {
  const [taxCase, setTaxCase] = useState(initial);
  return <TaxReviewQuestions conversationId="con_private" taxCase={taxCase} disabled={disabled} answerVersion={answerVersion}
    onCaseChange={(value) => { onChange(value); setTaxCase(value); }} onReload={onReload} onChoice={onChoice} />;
}

afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

it("keeps only the current user choice without case diagnostics or private identifiers", () => {
  render(<Harness />);
  expect(screen.getByRole("region", { name: "Tax review choices" })).toBeVisible();
  expect(screen.getByText("Is this correct for your 2025 review?")).toBeVisible();
  expect(screen.getByText("55000.00", { exact: true })).toBeVisible();
  expect(screen.getByRole("button", { name: "Confirm Wages" })).toHaveTextContent("Confirm");
  fireEvent.click(screen.getByText(/Wages statement.txt · Page 2/u));
  expect(screen.getByText("Box 1 Wages, tips, other compensation: $55,000.00")).toBeVisible();
  expect(screen.queryByText(proposed.id)).not.toBeInTheDocument();
  expect(screen.queryByText(factId)).not.toBeInTheDocument();
  expect(screen.queryByText(/rule_evidence|SHA-256|Reference release|Checklist v/u)).not.toBeInTheDocument();
});

it("saves a confirmation with the current revision and changes the question only after acceptance", async () => {
  let resolve!: (value: Response) => void;
  const confirmed: TaxCase = { ...proposed, revision: proposed.revision + 1,
    facts: proposed.facts.map((fact) => ({ ...fact, status: "confirmed" as const })) };
  const fetchMock = vi.fn().mockReturnValue(new Promise<Response>((done) => { resolve = done; }));
  vi.stubGlobal("fetch", fetchMock);
  const onChange = vi.fn();
  render(<Harness onChange={onChange} />);
  fireEvent.click(screen.getByRole("button", { name: "Confirm Wages" }));
  expect(onChange).not.toHaveBeenCalled();
  expect(screen.getByRole("button", { name: "Confirm Wages" })).toBeDisabled();
  expect(JSON.parse(fetchMock.mock.calls[0][1].body)).toEqual({ expected_revision: proposed.revision, operation: "confirm_fact", data: { fact_id: factId } });
  resolve(json(confirmed));
  await waitFor(() => expect(onChange).toHaveBeenCalledWith(confirmed));
  expect(screen.getByRole("status")).toHaveTextContent("Confirmed.");
  expect(screen.queryByRole("button", { name: "Confirm Wages" })).not.toBeInTheDocument();
  expect(screen.getByRole("list", { name: "Previous choices" })).toHaveTextContent("ConfirmedWages55000.00");
});

it("records Not correct as a rejection and invites corrected details in chat", async () => {
  const rejected: TaxCase = { ...proposed, revision: proposed.revision + 1,
    facts: proposed.facts.map((fact) => ({ ...fact, status: "rejected" as const })) };
  const fetchMock = vi.fn().mockResolvedValue(json(rejected));
  vi.stubGlobal("fetch", fetchMock);
  render(<Harness />);
  fireEvent.click(screen.getByRole("button", { name: "Not correct: Wages" }));
  expect(await screen.findByRole("status")).toHaveTextContent("Noted.");
  expect(JSON.parse(fetchMock.mock.calls[0][1].body)).toEqual({ expected_revision: proposed.revision, operation: "reject_fact", data: { fact_id: factId } });
  expect(screen.queryByRole("button", { name: "Confirm Wages" })).not.toBeInTheDocument();
  expect(screen.getByRole("list", { name: "Previous choices" })).toHaveTextContent("Not correctWages55000.00");
});

it("requires refreshed details after a revision conflict without accepting the fact", async () => {
  vi.stubGlobal("fetch", vi.fn().mockResolvedValue(json({ detail: "Case changed" }, 409)));
  const onChange = vi.fn();
  const onReload = vi.fn();
  const onChoice = vi.fn();
  render(<Harness onChange={onChange} onReload={onReload} onChoice={onChoice} />);
  fireEvent.click(screen.getByRole("button", { name: "Confirm Wages" }));
  expect(await screen.findByRole("alert")).toHaveTextContent("These details changed.");
  expect(onChange).not.toHaveBeenCalled();
  expect(screen.getByRole("button", { name: "Confirm Wages" })).toBeDisabled();
  fireEvent.click(screen.getByRole("button", { name: "Reload details" }));
  expect(onReload).toHaveBeenCalledOnce();
});

it("distinguishes a fact from a previous return year before reconfirmation", () => {
  const previous = { ...proposed, tax_year: 2024,
    facts: proposed.facts.map((fact) => ({ ...fact, case_tax_year: 2025 })) };
  render(<Harness initial={previous} />);
  expect(screen.getByText("From 2025. Check that this still applies.")).toBeVisible();
});

it("disables fact choices while another run is active", () => {
  render(<Harness disabled />);
  expect(screen.getByRole("button", { name: "Confirm Wages" })).toBeDisabled();
  expect(screen.getByRole("button", { name: "Not correct: Wages" })).toBeDisabled();
});


it("keeps missing-item decisions concise and records explicit resolution", async () => {
  const initial: TaxCase = { ...proposed, facts: [], missing_items: [{ id: "missing_statement", label: "Year-end income statement", status: "unresolved" }] };
  const updated: TaxCase = { ...initial, revision: initial.revision + 1,
    missing_items: initial.missing_items.map((item) => ({ ...item, status: "resolved" as const })) };
  const fetchMock = vi.fn().mockResolvedValue(json(updated));
  vi.stubGlobal("fetch", fetchMock);
  render(<Harness initial={initial} />);
  expect(screen.getByText("Could you help with this?")).toBeVisible();
  fireEvent.click(screen.getByRole("button", { name: "Mark Year-end income statement addressed" }));
  expect(await screen.findByRole("status")).toHaveTextContent("Marked addressed.");
  expect(screen.getByRole("list", { name: "Previous choices" })).toHaveTextContent("AddressedYear-end income statement");
  expect(JSON.parse(fetchMock.mock.calls[0][1].body)).toEqual({ expected_revision: initial.revision, operation: "resolve_missing_item", data: { id: "missing_statement" } });
});

it("allows deferring a missing record without changing its saved state", () => {
  const initial: TaxCase = { ...proposed, facts: [], missing_items: [{ id: "missing_statement", label: "Year-end income statement", status: "unresolved" }] };
  const fetchMock = vi.fn();
  vi.stubGlobal("fetch", fetchMock);
  render(<Harness initial={initial} />);
  fireEvent.click(screen.getByRole("button", { name: "Later" }));
  expect(screen.getByRole("status")).toHaveTextContent("Left open for later.");
  expect(fetchMock).not.toHaveBeenCalled();
});


it("does not repeat a saved acknowledgement after the next answer", async () => {
  const updated: TaxCase = { ...proposed, facts: [], missing_items: [], revision: proposed.revision + 1 };
  vi.stubGlobal("fetch", vi.fn().mockResolvedValue(json(updated)));
  const view = render(<Harness answerVersion={1} />);
  fireEvent.click(screen.getByRole("button", { name: "Confirm Wages" }));
  expect(await screen.findByRole("status")).toHaveTextContent("Confirmed.");
  view.rerender(<Harness answerVersion={2} />);
  expect(screen.queryByText("Confirmed.")).not.toBeInTheDocument();
});

it("keeps stale choices locked during reload until a fresh case is delivered", async () => {
  vi.stubGlobal("fetch", vi.fn().mockResolvedValue(json({ detail: "Case changed" }, 409)));
  const onReload = vi.fn();
  const props = { conversationId: "con_private", taxCase: proposed, disabled: false, onCaseChange: vi.fn(), onReload };
  const view = render(<TaxReviewQuestions {...props} />);
  fireEvent.click(screen.getByRole("button", { name: "Confirm Wages" }));
  await screen.findByRole("alert");
  fireEvent.click(screen.getByRole("button", { name: "Reload details" }));
  expect(onReload).toHaveBeenCalledOnce();
  expect(screen.getByRole("button", { name: "Confirm Wages" })).toBeDisabled();
  view.rerender(<TaxReviewQuestions {...props} taxCase={{ ...proposed, revision: proposed.revision + 1 }} />);
  expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Confirm Wages" })).toBeEnabled();
});


it("groups related facts in one atomic confirmation and sends feedback only after acceptance", async () => {
  const next = { ...proposed.facts[0], id: "next_fact", label: "Filing status", value: "Single" };
  const initial = { ...proposed, facts: [...proposed.facts, next] };
  const updated = { ...initial, revision: initial.revision + 1,
    facts: initial.facts.map((fact) => ({ ...fact, status: "confirmed" as const })) };
  let resolve!: (value: Response) => void;
  const fetchMock = vi.fn().mockReturnValue(new Promise<Response>((done) => { resolve = done; }));
  vi.stubGlobal("fetch", fetchMock);
  const onChoice = vi.fn();
  render(<Harness initial={initial} onChoice={onChoice} />);
  expect(screen.getByText("Filing status", { exact: true })).toBeVisible();
  fireEvent.click(screen.getByRole("button", { name: "Confirm these details" }));
  expect(onChoice).not.toHaveBeenCalled();
  expect(JSON.parse(fetchMock.mock.calls[0][1].body)).toEqual({ expected_revision: initial.revision,
    operation: "confirm_facts", data: { fact_ids: initial.facts.map(({ id }) => id) } });
  resolve(json(updated));
  await waitFor(() => expect(onChoice).toHaveBeenCalledOnce());
  expect(onChoice).toHaveBeenCalledWith(updated, expect.stringContaining("Wages: 55000.00; Filing status: Single"));
});

it("never sends model feedback for a rejected save", async () => {
  vi.stubGlobal("fetch", vi.fn().mockResolvedValue(json({ detail: "Case changed" }, 409)));
  const onChoice = vi.fn();
  render(<Harness onChoice={onChoice} />);
  fireEvent.click(screen.getByRole("button", { name: "Confirm Wages" }));
  await screen.findByRole("alert");
  expect(onChoice).not.toHaveBeenCalled();
});

it("caps each confirmation to the visible group", async () => {
  const facts = Array.from({ length: 12 }, (_, index) => ({ ...proposed.facts[0], id: `fact_${index}`, label: `Fact ${index}` }));
  const initial = { ...proposed, facts };
  const fetchMock = vi.fn().mockResolvedValue(json({ ...initial, revision: initial.revision + 1,
    facts: facts.map((fact, index) => index < 10 ? { ...fact, status: "confirmed" } : fact) }));
  vi.stubGlobal("fetch", fetchMock);
  render(<Harness initial={initial} />);
  expect(screen.getByText("2 more details left to review.")).toBeVisible();
  expect(screen.queryByText("Fact 10", { exact: true })).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "Confirm these details" }));
  await waitFor(() => expect(JSON.parse(fetchMock.mock.calls[0][1].body).data.fact_ids).toHaveLength(10));
});

it("keeps confirmed and rejected answers above the next prompt", () => {
  const initial: TaxCase = { ...proposed, facts: [
    { ...proposed.facts[0], status: "confirmed", confirmed_at: "2026-10-22T10:00:00Z" },
    { ...proposed.facts[0], id: "old_status", label: "Filing status", value: "Married", status: "rejected", rejected_at: "2026-10-22T10:01:00Z" },
    { ...proposed.facts[0], id: "new_status", label: "Corrected filing status", value: "Single", status: "proposed" },
  ] };
  render(<Harness initial={initial} />);
  const history = screen.getByRole("list", { name: "Previous choices" });
  expect(history).toHaveTextContent("ConfirmedWages55000.00");
  expect(history).toHaveTextContent("Not correctFiling statusMarried");
  expect(history).not.toHaveTextContent("Corrected filing status");
  expect(screen.getByRole("button", { name: "Confirm Corrected filing status" })).toBeEnabled();
  expect(screen.queryByRole("button", { name: "Confirm Wages" })).not.toBeInTheDocument();
  expect(screen.queryByText("Audrey", { exact: true })).not.toBeInTheDocument();
});

it("restores saved choices on refresh without generation or another save", () => {
  const initial: TaxCase = { ...proposed, facts: proposed.facts.map((fact) => ({
    ...fact, status: "confirmed", confirmed_at: "2026-10-22T10:00:00Z",
  })), missing_items: [{ id: "provided", label: "Income statement", status: "resolved", resolved_at: "2026-10-22T10:02:00Z" }] };
  const fetchMock = vi.fn();
  const onChoice = vi.fn();
  vi.stubGlobal("fetch", fetchMock);
  const view = render(<Harness initial={initial} onChoice={onChoice} />);
  expect(screen.getByRole("list", { name: "Previous choices" })).toHaveTextContent("Wages");
  view.unmount();
  render(<Harness initial={initial} onChoice={onChoice} />);
  const history = screen.getByRole("list", { name: "Previous choices" });
  expect(history).toHaveTextContent("AddressedIncome statement");
  expect(screen.queryByRole("button")).not.toBeInTheDocument();
  expect(fetchMock).not.toHaveBeenCalled();
  expect(onChoice).not.toHaveBeenCalled();
});

it("keeps Later visible while showing the next unresolved prompt", () => {
  const initial: TaxCase = { ...proposed, facts: [], missing_items: [
    { id: "first", label: "Income statement", status: "unresolved" },
    { id: "second", label: "Charitable receipts", status: "unresolved" },
  ] };
  const onChoice = vi.fn();
  render(<Harness initial={initial} onChoice={onChoice} />);
  fireEvent.click(screen.getByRole("button", { name: "Later" }));
  expect(screen.getByRole("list", { name: "Previous choices" })).toHaveTextContent("LaterIncome statement");
  expect(screen.getAllByText("Income statement", { exact: true })).toHaveLength(1);
  expect(screen.getByRole("button", { name: "Mark Charitable receipts addressed" })).toBeEnabled();
  expect(onChoice).toHaveBeenCalledWith(initial, expect.stringContaining("Leave this item open for later: Income statement"));
});

it("replaces a deferred card with its saved resolution without a duplicate", () => {
  const initial: TaxCase = { ...proposed, facts: [], missing_items: [{ id: "first", label: "Income statement", status: "unresolved" }] };
  const props = { conversationId: "con_private", taxCase: initial, disabled: false, onCaseChange: vi.fn(), onReload: vi.fn() };
  const view = render(<TaxReviewQuestions {...props} />);
  fireEvent.click(screen.getByRole("button", { name: "Later" }));
  view.rerender(<TaxReviewQuestions {...props} taxCase={{ ...initial, revision: initial.revision + 1,
    missing_items: [{ ...initial.missing_items[0], status: "resolved", resolved_at: "2026-10-22T12:00:00Z" }] }} />);
  expect(screen.getByRole("list", { name: "Previous choices" })).toHaveTextContent("AddressedIncome statement");
  expect(screen.queryByText("Later", { exact: true })).not.toBeInTheDocument();
  expect(screen.getAllByText("Income statement", { exact: true })).toHaveLength(1);
});

it("orders saved choices chronologically across facts and resolved items", () => {
  const initial: TaxCase = { ...proposed, facts: [
    { ...proposed.facts[0], status: "confirmed", confirmed_at: "2026-10-22T10:03:00Z" },
    { ...proposed.facts[0], id: "old", label: "Filing status", status: "rejected", rejected_at: "2026-10-22T10:01:00Z" },
  ], missing_items: [{ id: "provided", label: "Income statement", status: "resolved", resolved_at: "2026-10-22T10:02:00Z" }] };
  render(<Harness initial={initial} />);
  const cards = screen.getByRole("list", { name: "Previous choices" }).children;
  expect(cards[0]).toHaveTextContent("Filing status");
  expect(cards[1]).toHaveTextContent("Income statement");
  expect(cards[2]).toHaveTextContent("Wages");
});

it("ignores a pending save after switching conversations or years", async () => {
  let resolve!: (value: Response) => void;
  vi.stubGlobal("fetch", vi.fn().mockReturnValue(new Promise<Response>((done) => { resolve = done; })));
  const onCaseChange = vi.fn();
  const onChoice = vi.fn();
  const props = { conversationId: "con_private", taxCase: proposed, disabled: false, onCaseChange, onReload: vi.fn(), onChoice };
  const view = render(<TaxReviewQuestions {...props} />);
  fireEvent.click(screen.getByRole("button", { name: "Confirm Wages" }));
  view.rerender(<TaxReviewQuestions {...props} conversationId="con_other" taxCase={{ ...proposed, tax_year: 2024 }} />);
  resolve(json({ ...proposed, facts: proposed.facts.map((fact) => ({ ...fact, status: "confirmed" })) }));
  await waitFor(() => expect(screen.getByRole("button", { name: "Confirm Wages" })).toBeEnabled());
  expect(onCaseChange).not.toHaveBeenCalled();
  expect(onChoice).not.toHaveBeenCalled();
  expect(screen.queryByRole("list", { name: "Previous choices" })).not.toBeInTheDocument();
});

it("keeps the scroll position while inspecting earlier choices", () => {
  const initial: TaxCase = { ...proposed, facts: proposed.facts.map((fact) => ({ ...fact, status: "confirmed" })) };
  const props = { conversationId: "con_private", taxCase: initial, disabled: false, onCaseChange: vi.fn(), onReload: vi.fn() };
  const view = render(<TaxReviewQuestions {...props} />);
  const flow = screen.getByRole("list", { name: "Previous choices" }).parentElement!;
  Object.defineProperties(flow, { scrollHeight: { value: 400, configurable: true }, clientHeight: { value: 100 } });
  flow.scrollTop = 50;
  fireEvent.scroll(flow);
  view.rerender(<TaxReviewQuestions {...props} taxCase={{ ...initial, missing_items: [{ id: "new", label: "Another detail", status: "unresolved" }] }} />);
  expect(flow.scrollTop).toBe(50);
  flow.scrollTop = 300;
  fireEvent.scroll(flow);
  view.rerender(<TaxReviewQuestions {...props} taxCase={{ ...initial, missing_items: [{ id: "next", label: "Next detail", status: "unresolved" }] }} />);
  expect(flow.scrollTop).toBe(400);
});
