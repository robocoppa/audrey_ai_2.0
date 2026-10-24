import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";

import { TaxProjectSection } from "./TaxProjectSection";
import { TAX_REVIEW_SCOPE_NOTICE, type TaxCase } from "./taxReviewApi";
import fixtures from "./test/fixtures/tax-review-cases.json";

function json(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
}
const saved = { ...fixtures.group_reviewed, conversation_id: "con_tax", project_id: "proj_tax" } as TaxCase;
afterEach(() => { cleanup(); vi.restoreAllMocks(); vi.unstubAllGlobals(); });

it("shows canonical current facts and saved output without creating editable copies", async () => {
  const fetchMock = vi.fn().mockResolvedValue(json({ items: [saved] }));
  vi.stubGlobal("fetch", fetchMock);
  render(<TaxProjectSection projectId="proj_tax" onConversationSelected={vi.fn()} />);
  const table = await screen.findByRole("table", { name: "2025 saved tax facts" });
  for (const fact of saved.facts.filter(({ status }) => status !== "rejected")) {
    expect(within(table).getByRole("rowheader", { name: fact.label })).toBeVisible();
  }
  expect(screen.queryByRole("textbox")).not.toBeInTheDocument();
  expect(screen.queryByRole("button", { name: /Confirm|Edit/u })).not.toBeInTheDocument();
  const worksheet = screen.getByText("Saved worksheet").closest("details")!;
  expect(worksheet).not.toHaveAttribute("open");
  fireEvent.click(screen.getByText("Saved worksheet"));
  expect(within(worksheet).getByText(saved.review!.summary)).toBeVisible();
  expect(fetchMock.mock.calls[0][0]).toBe("/api/projects/proj_tax/tax-reviews");
  expect(fetchMock.mock.calls.every((call) => !call[1]?.method || call[1].method === "GET")).toBe(true);
});

it("keeps a saved project worksheet's scope visible before its findings without adding confirmation", async () => {
  const checked = { ...saved, review: { ...saved.review!, status: "review_saved", stale: false,
    next_question: "", verification: { status: "checked" },
  } };
  vi.stubGlobal("fetch", vi.fn().mockResolvedValue(json({ items: [checked] })));
  render(<TaxProjectSection projectId="proj_tax" onConversationSelected={vi.fn()} />);
  fireEvent.click(await screen.findByText("Saved worksheet"));
  const worksheet = screen.getByText("Saved worksheet").closest("details")!;
  const scope = within(worksheet).getByText(TAX_REVIEW_SCOPE_NOTICE);
  expect(scope).toBeVisible();
  expect(scope.compareDocumentPosition(within(worksheet).getByText(checked.review.summary)) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
  expect(within(worksheet).queryByRole("button")).not.toBeInTheDocument();
  expect(within(worksheet).queryByRole("checkbox")).not.toBeInTheDocument();
  expect(within(worksheet).queryByRole("textbox")).not.toBeInTheDocument();
});

it("opens the source conversation using its current project membership", async () => {
  const conversation = { id: "con_tax", project_id: "proj_tax", private_required: true };
  const fetchMock = vi.fn().mockResolvedValueOnce(json({ items: [saved] })).mockResolvedValueOnce(json(conversation));
  vi.stubGlobal("fetch", fetchMock);
  const onConversationSelected = vi.fn();
  render(<TaxProjectSection projectId="proj_tax" onConversationSelected={onConversationSelected} />);
  fireEvent.click(await screen.findByRole("button", { name: "Open review conversation" }));
  await waitFor(() => expect(onConversationSelected).toHaveBeenCalledWith(conversation));
  expect(fetchMock.mock.calls[1][0]).toBe("/api/conversations/con_tax");
});

it("hides another project's facts while a replacement project is loading", async () => {
  vi.stubGlobal("fetch", vi.fn().mockResolvedValueOnce(json({ items: [saved] })).mockImplementationOnce(() => new Promise(() => {})));
  const view = render(<TaxProjectSection projectId="proj_tax" onConversationSelected={vi.fn()} />);
  await screen.findByRole("table");
  view.rerender(<TaxProjectSection projectId="proj_other" onConversationSelected={vi.fn()} />);
  expect(screen.queryByRole("table")).not.toBeInTheDocument();
  expect(screen.queryByText("Saved worksheet")).not.toBeInTheDocument();
});

it("keeps ordinary Private projects uncluttered and reports loading failures with a retry", async () => {
  const fetchMock = vi.fn().mockResolvedValueOnce(json({ detail: "Unavailable" }, 503)).mockResolvedValueOnce(json({ items: [] }));
  vi.stubGlobal("fetch", fetchMock);
  render(<TaxProjectSection projectId="proj_empty" onConversationSelected={vi.fn()} />);
  expect(await screen.findByRole("alert")).toHaveTextContent("Could not load the saved tax reviews.");
  fireEvent.click(screen.getByRole("button", { name: "Try again" }));
  await waitFor(() => expect(screen.queryByRole("heading", { name: "Tax reviews" })).not.toBeInTheDocument());
});

it("does not open a stale conversation moved out of this Private project", async () => {
  vi.stubGlobal("fetch", vi.fn().mockResolvedValueOnce(json({ items: [saved] })).mockResolvedValueOnce(json({ id: "con_tax", project_id: "elsewhere", private_required: true })));
  const onConversationSelected = vi.fn();
  render(<TaxProjectSection projectId="proj_tax" onConversationSelected={onConversationSelected} />);
  fireEvent.click(await screen.findByRole("button", { name: "Open review conversation" }));
  expect(await screen.findByRole("alert")).toHaveTextContent("Could not open this review.");
  expect(onConversationSelected).not.toHaveBeenCalled();
});


it("does not navigate to an old case after the user changes projects", async () => {
  let complete!: (response: Response) => void;
  vi.stubGlobal("fetch", vi.fn().mockResolvedValueOnce(json({ items: [saved] }))
    .mockImplementationOnce(() => new Promise<Response>((resolve) => { complete = resolve; }))
    .mockResolvedValueOnce(json({ items: [] })));
  const onConversationSelected = vi.fn();
  const view = render(<TaxProjectSection projectId="proj_tax" onConversationSelected={onConversationSelected} />);
  fireEvent.click(await screen.findByRole("button", { name: "Open review conversation" }));
  view.rerender(<TaxProjectSection projectId="proj_other" onConversationSelected={onConversationSelected} />);
  complete(json({ id: "con_tax", project_id: "proj_tax", private_required: true }));
  await waitFor(() => expect(screen.queryByText("Opening…")).not.toBeInTheDocument());
  expect(onConversationSelected).not.toHaveBeenCalled();
});
