import { cleanup, render, screen, within } from "@testing-library/react";
import { afterEach, expect, it } from "vitest";

import { OfficeDocumentPreview, type OfficeDocumentBlock } from "./OfficeDocumentPreview";

afterEach(cleanup);

it("renders saved headings, inline emphasis, lists and tables as document structure", () => {
  const blocks: OfficeDocumentBlock[] = [
    { kind: "heading", level: 1, runs: [{ text: "Weekend plan" }] },
    { kind: "paragraph", runs: [{ text: "Remember ", bold: true }, { text: "comfortable shoes.", italic: true }] },
    { kind: "paragraph", list: "bullet", runs: [{ text: "Read a book" }] },
    { kind: "paragraph", list: "bullet", runs: [{ text: "Take a walk" }] },
    { kind: "heading", level: 2, runs: [{ text: "Preparation" }] },
    { kind: "paragraph", list: "number", list_start: 3, runs: [{ text: "Pack lunch" }] },
    { kind: "paragraph", list: "number", runs: [{ text: "Check weather" }] },
    { kind: "table", header: true, rows: [[{ runs: [{ text: "Item" }] }, { runs: [{ text: "Cost" }] }],
      [{ runs: [{ text: "Food" }] }, { runs: [{ text: "$40.00", bold: true }] }]] },
    { kind: "quote", runs: [{ text: "Leave time to rest." }] },
    { kind: "code", runs: [{ text: "items = 2\nprint(items)" }] },
  ];
  const { container } = render(<OfficeDocumentPreview blocks={blocks} paragraphs={["Plain fallback"]} />);
  expect(screen.getByRole("heading", { level: 1, name: "Weekend plan" })).toBeVisible();
  expect(screen.getByText("Remember").closest("strong")).not.toBeNull();
  expect(screen.getByText("comfortable shoes.").closest("em")).not.toBeNull();
  const lists = screen.getAllByRole("list");
  expect(within(lists[0]).getAllByRole("listitem")).toHaveLength(2);
  expect(lists[0].tagName).toBe("UL");
  expect(lists[1].tagName).toBe("OL");
  expect(lists[1]).toHaveAttribute("start", "3");
  expect(screen.getByRole("columnheader", { name: "Cost" })).toBeVisible();
  expect(screen.getByRole("cell", { name: "$40.00" })).toBeVisible();
  expect(container.querySelector("blockquote")).toHaveTextContent("Leave time to rest.");
  expect(container.querySelector("pre code")).toHaveTextContent("items = 2");
  expect(screen.queryByText("Plain fallback")).not.toBeInTheDocument();
});

it("keeps literal text inert and never interprets document content as HTML or remote resources", () => {
  const { container } = render(<OfficeDocumentPreview blocks={[
    { kind: "paragraph", runs: [{ text: '<img src="https://tracker.example/pixel">', bold: true }] },
    { kind: "table", rows: [[{ runs: [{ text: '<script>alert("unsafe")</script>' }] }]] },
    { kind: "paragraph", runs: [{ text: "=NOT_A_FORMULA", code: true }] },
  ]} />);
  expect(screen.getByText('<img src="https://tracker.example/pixel">')).toBeVisible();
  expect(screen.getByRole("cell", { name: '<script>alert("unsafe")</script>' })).toBeVisible();
  expect(screen.getByText("=NOT_A_FORMULA").closest("code")).not.toBeNull();
  expect(container.querySelector("img,script,a,[style]")).toBeNull();
});

it("preserves older plain document previews and empty paragraphs", () => {
  const { container, rerender } = render(<OfficeDocumentPreview paragraphs={["An older document", "", "Last paragraph"]} />);
  expect(screen.getByText("An older document")).toBeVisible();
  expect(container.querySelectorAll("p")).toHaveLength(3);
  expect(container.querySelectorAll("p")[1].textContent).toBe("\u00a0");
  rerender(<OfficeDocumentPreview paragraphs={[]} />);
  expect(screen.getByText("This document is empty.")).toBeVisible();
});


it("preserves explicit numbering jumps from the saved Word list", () => {
  render(<OfficeDocumentPreview blocks={[
    { kind: "paragraph", list: "number", list_start: 4, runs: [{ text: "Fourth step" }] },
    { kind: "paragraph", list: "number", list_start: 7, runs: [{ text: "Seventh step" }] },
  ]} />);
  expect(screen.getByText("Fourth step").closest("li")).toHaveAttribute("value", "4");
  expect(screen.getByText("Seventh step").closest("li")).toHaveAttribute("value", "7");
});
