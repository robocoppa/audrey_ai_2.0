import { useState } from "react";
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";

import { TaxReviewQuestions } from "./TaxReviewQuestions";
import type { TaxCase } from "./taxReviewApi";
import fixtures from "./test/fixtures/tax-review-cases.json";

const proposed = fixtures.reviewable as TaxCase;
const factId = proposed.facts[0].id;
const confirmed: TaxCase = { ...proposed, facts: proposed.facts.map((fact) => ({ ...fact, status: "confirmed" })) };
function json(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
}
function Harness({ initial = proposed, disabled = false, answerVersion = 0, onChange = vi.fn(), onReload = vi.fn(), onChoice = vi.fn() }) {
  const [taxCase, setTaxCase] = useState(initial);
  return <TaxReviewQuestions conversationId="con_private" taxCase={taxCase} disabled={disabled} answerVersion={answerVersion}
    onCaseChange={(value) => { onChange(value); setTaxCase(value); }} onReload={onReload} onChoice={onChoice} />;
}
const missing: TaxCase = { ...confirmed, missing_items: [{ id: "eligibility", label: "Basic deduction eligibility", status: "unresolved",
  question: "What was your age at the end of 2025, were you blind, and could someone claim you as a dependent?" }] };

afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

it("shows saved facts as editable current values without old choices or resolved prompts", () => {
  const initial: TaxCase = { ...confirmed, facts: [...confirmed.facts, { ...confirmed.facts[0], id: "old", label: "Old wages", status: "rejected" }],
    missing_items: [{ id: "done", label: "Confirm filing status", status: "resolved" }] };
  render(<Harness initial={initial} />);
  const table = screen.getByRole("table", { name: "Saved tax facts" });
  expect(within(table).getByRole("row", { name: /Wages 55000.00/u })).toBeVisible();
  expect(screen.getByRole("button", { name: "Edit Wages" })).toBeEnabled();
  expect(screen.queryByRole("button", { name: /Confirm/u })).not.toBeInTheDocument();
  expect(screen.queryByText(/Your choices|Addressed|Old wages|Confirm filing status/u)).not.toBeInTheDocument();
  expect(screen.queryByRole("button", { name: /Continue/u })).not.toBeInTheDocument();
});

it("keeps facts from a user message without asking for a second confirmation", () => {
  const initial: TaxCase = { ...confirmed, facts: confirmed.facts.map((fact) => ({ ...fact, provenance: null,
    confirmed_by: "user_message", source_message_id: "msg_current" })) };
  render(<Harness initial={initial} />);
  expect(screen.getByRole("table", { name: "Saved tax facts" })).toHaveTextContent("From your message");
  expect(screen.queryByRole("button", { name: /Confirm/u })).not.toBeInTheDocument();
  expect(screen.queryByText("Check these details")).not.toBeInTheDocument();
});

it("clearly asks whether an extracted amount is correct and exposes its source on demand", () => {
  render(<Harness />);
  expect(screen.getByText("Is this amount correct for 2025? Edit anything that needs correcting.")).toBeVisible();
  expect(screen.getByRole("table", { name: "Details to check" })).toHaveTextContent("Wages55000.00");
  expect(screen.getByRole("button", { name: "Confirm Wages" })).toHaveTextContent("This is correct");
  fireEvent.click(screen.getByText(/Wages statement.txt · Page 2/u));
  expect(screen.getByText("Box 1 Wages, tips, other compensation: $55,000.00")).toBeVisible();
  expect(screen.queryByText(proposed.id)).not.toBeInTheDocument();
  expect(screen.queryByText(factId)).not.toBeInTheDocument();
});

it("confirms extracted details only after server acceptance and moves them into the current facts table", async () => {
  let resolve!: (value: Response) => void;
  const updated = { ...confirmed, revision: proposed.revision + 1 };
  const fetchMock = vi.fn().mockReturnValue(new Promise<Response>((done) => { resolve = done; }));
  vi.stubGlobal("fetch", fetchMock);
  const onChoice = vi.fn();
  render(<Harness onChoice={onChoice} />);
  fireEvent.click(screen.getByRole("button", { name: "Confirm Wages" }));
  expect(onChoice).not.toHaveBeenCalled();
  expect(screen.getByRole("button", { name: "Confirm Wages" })).toBeDisabled();
  expect(JSON.parse(fetchMock.mock.calls[0][1].body)).toEqual({ expected_revision: proposed.revision, operation: "confirm_fact", data: { fact_id: factId } });
  resolve(json(updated));
  await waitFor(() => expect(onChoice).toHaveBeenCalledWith(updated, expect.stringContaining("Wages: 55000.00")));
  expect(screen.getByRole("table", { name: "Saved tax facts" })).toHaveTextContent("55000.00");
  expect(screen.queryByRole("table", { name: "Details to check" })).not.toBeInTheDocument();
});

it("updates the same current row with the owner correction and sends feedback after saving", async () => {
  const updated = { ...confirmed, revision: confirmed.revision + 1,
    facts: confirmed.facts.map((fact) => ({ ...fact, value: "52000.00" })) };
  let resolve!: (value: Response) => void;
  const fetchMock = vi.fn().mockReturnValue(new Promise<Response>((done) => { resolve = done; }));
  vi.stubGlobal("fetch", fetchMock);
  const onChange = vi.fn();
  const onChoice = vi.fn();
  render(<Harness initial={confirmed} onChange={onChange} onChoice={onChoice} />);
  fireEvent.click(screen.getByRole("button", { name: "Edit Wages" }));
  fireEvent.change(screen.getByLabelText("Value for Wages"), { target: { value: "52000.00" } });
  fireEvent.click(screen.getByRole("button", { name: "Save fact" }));
  expect(onChange).not.toHaveBeenCalled();
  expect(onChoice).not.toHaveBeenCalled();
  expect(JSON.parse(fetchMock.mock.calls[0][1].body)).toEqual({ expected_revision: confirmed.revision, operation: "edit_fact",
    data: { fact_id: factId, label: "Wages", value: "52000.00" } });
  resolve(json(updated));
  await waitFor(() => expect(onChoice).toHaveBeenCalledWith(updated, expect.stringContaining("52000.00")));
  expect(screen.getByRole("table", { name: "Saved tax facts" })).toHaveTextContent("52000.00");
  expect(screen.queryByText("55000.00", { exact: true })).not.toBeInTheDocument();
  expect(screen.getAllByRole("button", { name: "Edit Wages" })).toHaveLength(1);
});

it("cancels an edit without changing the saved value or generating an answer", () => {
  const fetchMock = vi.fn();
  const onChoice = vi.fn();
  vi.stubGlobal("fetch", fetchMock);
  render(<Harness initial={confirmed} onChoice={onChoice} />);
  fireEvent.click(screen.getByRole("button", { name: "Edit Wages" }));
  fireEvent.change(screen.getByLabelText("Value for Wages"), { target: { value: "wrong" } });
  fireEvent.click(screen.getByRole("button", { name: "Cancel" }));
  expect(screen.getByText("55000.00", { exact: true })).toBeVisible();
  expect(fetchMock).not.toHaveBeenCalled();
  expect(onChoice).not.toHaveBeenCalled();
});

it("preserves an edit on a revision conflict and requires a reload before retrying", async () => {
  vi.stubGlobal("fetch", vi.fn().mockResolvedValue(json({ detail: "Case changed" }, 409)));
  const onChange = vi.fn();
  const onReload = vi.fn();
  const onChoice = vi.fn();
  render(<Harness initial={confirmed} onChange={onChange} onReload={onReload} onChoice={onChoice} />);
  fireEvent.click(screen.getByRole("button", { name: "Edit Wages" }));
  fireEvent.change(screen.getByLabelText("Value for Wages"), { target: { value: "52000.00" } });
  fireEvent.click(screen.getByRole("button", { name: "Save fact" }));
  expect(await screen.findByRole("alert")).toHaveTextContent("These details changed.");
  expect(screen.getByLabelText("Value for Wages")).toHaveValue("52000.00");
  expect(screen.getByRole("button", { name: "Save fact" })).toBeDisabled();
  expect(onChange).not.toHaveBeenCalled();
  expect(onChoice).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole("button", { name: "Reload details" }));
  expect(onReload).toHaveBeenCalledOnce();
});

it("does not overwrite a fresh case with an edit begun against an older revision", () => {
  const fetchMock = vi.fn();
  vi.stubGlobal("fetch", fetchMock);
  const props = { conversationId: "con_private", taxCase: confirmed, disabled: false, onCaseChange: vi.fn(), onReload: vi.fn() };
  const view = render(<TaxReviewQuestions {...props} />);
  fireEvent.click(screen.getByRole("button", { name: "Edit Wages" }));
  fireEvent.change(screen.getByLabelText("Value for Wages"), { target: { value: "52000.00" } });
  view.rerender(<TaxReviewQuestions {...props} taxCase={{ ...confirmed, revision: confirmed.revision + 1 }} />);
  expect(screen.getByRole("button", { name: "Save fact" })).toBeDisabled();
  expect(screen.getByLabelText("Value for Wages")).toHaveValue("52000.00");
  expect(screen.getByText(/changed while you were editing/u)).toBeVisible();
  expect(fetchMock).not.toHaveBeenCalled();
});

it("asks a clear missing question and sends the actual answer without marking it resolved first", () => {
  const fetchMock = vi.fn();
  const onChoice = vi.fn().mockReturnValue(true);
  vi.stubGlobal("fetch", fetchMock);
  render(<Harness initial={missing} onChoice={onChoice} />);
  expect(screen.getByText(missing.missing_items[0].question!)).toBeVisible();
  expect(screen.queryByText(/Could you help|provided this|Addressed/u)).not.toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Send answer" })).toBeDisabled();
  fireEvent.change(screen.getByLabelText("Your answer"), { target: { value: "40, not blind, and not a dependent." } });
  fireEvent.click(screen.getByRole("button", { name: "Send answer" }));
  expect(onChoice).toHaveBeenCalledWith(missing, "40, not blind, and not a dependent.");
  expect(fetchMock).not.toHaveBeenCalled();
  expect(screen.getByLabelText("Your answer")).toHaveValue("");
});

it("does not turn a generated question into a statement by the user", () => {
  const initial: TaxCase = { ...missing, missing_items: [{ ...missing.missing_items[0], question: "Were your wages $50,000?" }] };
  const onChoice = vi.fn().mockReturnValue(true);
  render(<Harness initial={initial} onChoice={onChoice} />);
  fireEvent.change(screen.getByLabelText("Your answer"), { target: { value: "No" } });
  fireEvent.click(screen.getByRole("button", { name: "Send answer" }));
  expect(onChoice).toHaveBeenCalledWith(initial, "No");
  expect(onChoice.mock.calls[0][1]).not.toContain("50,000");
});

it("keeps an answer draft when another queued review prevents sending", () => {
  render(<Harness initial={missing} onChoice={vi.fn().mockReturnValue(false)} />);
  fireEvent.change(screen.getByLabelText("Your answer"), { target: { value: "40" } });
  fireEvent.click(screen.getByRole("button", { name: "Send answer" }));
  expect(screen.getByLabelText("Your answer")).toHaveValue("40");
});

it("uses a current saved review question for older missing records", () => {
  const initial: TaxCase = { ...missing, missing_items: [{ id: "old", label: "Eligibility", status: "unresolved" }],
    review: { ...(fixtures.group_reviewed as TaxCase).review!, status: "needs_input", next_question: "How old were you at the end of 2025?" } };
  render(<Harness initial={initial} />);
  expect(screen.getByText("How old were you at the end of 2025?")).toBeVisible();
  expect(screen.queryByText("Eligibility", { exact: true })).not.toBeInTheDocument();
});

it("does not keep resolved questions or a stale saved question on the screen", () => {
  const initial: TaxCase = { ...confirmed, missing_items: [{ id: "done", label: "Eligibility", question: "How old?", status: "resolved" }],
    review: { ...(fixtures.group_reviewed as TaxCase).review!, status: "needs_input", stale: true, next_question: "How old?" } };
  render(<Harness initial={initial} />);
  expect(screen.queryByText("How old?")).not.toBeInTheDocument();
  expect(screen.queryByRole("button", { name: "Send answer" })).not.toBeInTheDocument();
  expect(screen.getByRole("table", { name: "Saved tax facts" })).toBeVisible();
});

it("disables edits, confirmations and answers while a run is active", () => {
  render(<Harness initial={missing} disabled />);
  expect(screen.getByRole("button", { name: "Edit Wages" })).toBeDisabled();
  expect(screen.getByLabelText("Your answer")).toBeDisabled();
});

it("groups only the visible proposed facts in one atomic confirmation", async () => {
  const facts = Array.from({ length: 12 }, (_, index) => ({ ...proposed.facts[0], id: `fact_${index}`, label: `Fact ${index}` }));
  const initial = { ...proposed, facts };
  const fetchMock = vi.fn().mockResolvedValue(json({ ...initial, revision: initial.revision + 1,
    facts: facts.map((fact, index) => index < 10 ? { ...fact, status: "confirmed" } : fact) }));
  vi.stubGlobal("fetch", fetchMock);
  render(<Harness initial={initial} />);
  expect(screen.getByText("2 more details to check.")).toBeVisible();
  expect(screen.queryByText("Fact 10", { exact: true })).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "Confirm these details" }));
  await waitFor(() => expect(JSON.parse(fetchMock.mock.calls[0][1].body).data.fact_ids).toHaveLength(10));
});

it("ignores a pending edit after switching conversations or years", async () => {
  let resolve!: (value: Response) => void;
  vi.stubGlobal("fetch", vi.fn().mockReturnValue(new Promise<Response>((done) => { resolve = done; })));
  const onCaseChange = vi.fn();
  const onChoice = vi.fn();
  const props = { conversationId: "con_private", taxCase: confirmed, disabled: false, onCaseChange, onReload: vi.fn(), onChoice };
  const view = render(<TaxReviewQuestions {...props} />);
  fireEvent.click(screen.getByRole("button", { name: "Edit Wages" }));
  fireEvent.click(screen.getByRole("button", { name: "Save fact" }));
  view.rerender(<TaxReviewQuestions {...props} conversationId="con_other" taxCase={{ ...confirmed, tax_year: 2024 }} />);
  resolve(json({ ...confirmed, revision: confirmed.revision + 1 }));
  await waitFor(() => expect(screen.getByRole("button", { name: "Edit Wages" })).toBeEnabled());
  expect(onCaseChange).not.toHaveBeenCalled();
  expect(onChoice).not.toHaveBeenCalled();
});

it("ignores an older pending response after a newer case revision arrived", async () => {
  let resolve!: (value: Response) => void;
  vi.stubGlobal("fetch", vi.fn().mockReturnValue(new Promise<Response>((done) => { resolve = done; })));
  const onCaseChange = vi.fn();
  const onChoice = vi.fn();
  const props = { conversationId: "con_private", taxCase: confirmed, disabled: false, onCaseChange, onReload: vi.fn(), onChoice };
  const view = render(<TaxReviewQuestions {...props} />);
  fireEvent.click(screen.getByRole("button", { name: "Edit Wages" }));
  fireEvent.click(screen.getByRole("button", { name: "Save fact" }));
  view.rerender(<TaxReviewQuestions {...props} taxCase={{ ...confirmed, revision: confirmed.revision + 2 }} />);
  resolve(json({ ...confirmed, revision: confirmed.revision + 1 }));
  await waitFor(() => expect(screen.getByRole("button", { name: "Cancel" })).toBeEnabled());
  expect(onCaseChange).not.toHaveBeenCalled();
  expect(onChoice).not.toHaveBeenCalled();
});

it("restores current facts after refresh without a save or generation", () => {
  const fetchMock = vi.fn();
  const onChoice = vi.fn();
  vi.stubGlobal("fetch", fetchMock);
  const view = render(<Harness initial={confirmed} onChoice={onChoice} />);
  view.unmount();
  render(<Harness initial={confirmed} onChoice={onChoice} />);
  expect(screen.getByRole("table", { name: "Saved tax facts" })).toHaveTextContent("55000.00");
  expect(fetchMock).not.toHaveBeenCalled();
  expect(onChoice).not.toHaveBeenCalled();
});

it("renders explicit false and zero as values rather than missing information", () => {
  const initial: TaxCase = { ...confirmed, facts: [{ ...confirmed.facts[0], label: "Profile", value: { blind: false, dependents: 0 } }] };
  render(<Harness initial={initial} />);
  expect(screen.getByText("blind: No; dependents: 0")).toBeVisible();
});

it("preserves scroll position while inspecting saved facts", () => {
  const props = { conversationId: "con_private", taxCase: confirmed, disabled: false, onCaseChange: vi.fn(), onReload: vi.fn() };
  const view = render(<TaxReviewQuestions {...props} />);
  const flow = screen.getByRole("table", { name: "Saved tax facts" }).parentElement!;
  Object.defineProperties(flow, { scrollHeight: { value: 400, configurable: true }, clientHeight: { value: 100 } });
  flow.scrollTop = 50;
  fireEvent.scroll(flow);
  view.rerender(<TaxReviewQuestions {...props} taxCase={missing} />);
  expect(flow.scrollTop).toBe(50);
});
