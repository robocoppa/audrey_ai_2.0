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
function Harness({ initial = proposed, disabled = false, answerVersion = 0, onChange = vi.fn(), onReload = vi.fn() }) {
  const [taxCase, setTaxCase] = useState(initial);
  return <TaxReviewQuestions conversationId="con_private" taxCase={taxCase} disabled={disabled} answerVersion={answerVersion}
    onCaseChange={(value) => { onChange(value); setTaxCase(value); }} onReload={onReload} />;
}

afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

it("asks for relevant confirmation as Audrey without case diagnostics or private identifiers", () => {
  render(<Harness />);
  expect(screen.getByRole("article", { name: "Audrey's tax review questions" })).toBeVisible();
  expect(screen.getByText("Is this correct for your 2025 return?")).toBeVisible();
  expect(screen.getByText("55000.00", { exact: true })).toBeVisible();
  expect(screen.getByRole("button", { name: "Confirm Wages" })).toHaveTextContent("Yes, that's correct");
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
  expect(screen.getByRole("status")).toHaveTextContent("I've saved your confirmation.");
  expect(screen.queryByRole("button", { name: "Confirm Wages" })).not.toBeInTheDocument();
});

it("records Not correct as a rejection and invites corrected details in chat", async () => {
  const rejected: TaxCase = { ...proposed, revision: proposed.revision + 1,
    facts: proposed.facts.map((fact) => ({ ...fact, status: "rejected" as const })) };
  const fetchMock = vi.fn().mockResolvedValue(json(rejected));
  vi.stubGlobal("fetch", fetchMock);
  render(<Harness />);
  fireEvent.click(screen.getByRole("button", { name: "Not correct: Wages" }));
  expect(await screen.findByRole("status")).toHaveTextContent("I won't use that detail.");
  expect(JSON.parse(fetchMock.mock.calls[0][1].body)).toEqual({ expected_revision: proposed.revision, operation: "reject_fact", data: { fact_id: factId } });
  expect(screen.queryByRole("button", { name: "Confirm Wages" })).not.toBeInTheDocument();
});

it("requires refreshed details after a revision conflict without accepting the fact", async () => {
  vi.stubGlobal("fetch", vi.fn().mockResolvedValue(json({ detail: "Case changed" }, 409)));
  const onChange = vi.fn();
  const onReload = vi.fn();
  render(<Harness onChange={onChange} onReload={onReload} />);
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
  expect(screen.getByText("Is this correct for your 2024 return?")).toBeVisible();
  expect(screen.getByText("From your 2025 review. Please check whether this still applies.")).toBeVisible();
});

it("disables fact choices while another run is active", () => {
  render(<Harness disabled />);
  expect(screen.getByRole("button", { name: "Confirm Wages" })).toBeDisabled();
  expect(screen.getByRole("button", { name: "Not correct: Wages" })).toBeDisabled();
});


it("keeps missing-item decisions in the conversation and records explicit resolution", async () => {
  const initial: TaxCase = { ...proposed, facts: [], missing_items: [{ id: "missing_statement", label: "Year-end income statement", status: "unresolved" }] };
  const updated: TaxCase = { ...initial, revision: initial.revision + 1,
    missing_items: initial.missing_items.map((item) => ({ ...item, status: "resolved" as const })) };
  const fetchMock = vi.fn().mockResolvedValue(json(updated));
  vi.stubGlobal("fetch", fetchMock);
  render(<Harness initial={initial} />);
  expect(screen.getByText("Before we continue, could you help with this?")).toBeVisible();
  fireEvent.click(screen.getByRole("button", { name: "Mark Year-end income statement addressed" }));
  expect(await screen.findByRole("status")).toHaveTextContent("I've marked that as addressed.");
  expect(JSON.parse(fetchMock.mock.calls[0][1].body)).toEqual({ expected_revision: initial.revision, operation: "resolve_missing_item", data: { id: "missing_statement" } });
});

it("allows deferring a missing record without changing its saved state", () => {
  const initial: TaxCase = { ...proposed, facts: [], missing_items: [{ id: "missing_statement", label: "Year-end income statement", status: "unresolved" }] };
  const fetchMock = vi.fn();
  vi.stubGlobal("fetch", fetchMock);
  render(<Harness initial={initial} />);
  fireEvent.click(screen.getByRole("button", { name: "Not yet" }));
  expect(screen.getByRole("status")).toHaveTextContent("I'll leave that open.");
  expect(fetchMock).not.toHaveBeenCalled();
});


it("does not repeat a saved acknowledgement below a later answer", async () => {
  const updated: TaxCase = { ...proposed, facts: [], missing_items: [], revision: proposed.revision + 1 };
  vi.stubGlobal("fetch", vi.fn().mockResolvedValue(json(updated)));
  const view = render(<Harness answerVersion={1} />);
  fireEvent.click(screen.getByRole("button", { name: "Confirm Wages" }));
  expect(await screen.findByRole("status")).toHaveTextContent("I've saved your confirmation.");
  view.rerender(<Harness answerVersion={2} />);
  expect(screen.queryByText("I've saved your confirmation.")).not.toBeInTheDocument();
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
