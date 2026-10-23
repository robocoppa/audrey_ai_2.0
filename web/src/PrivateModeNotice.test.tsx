import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";

import { PrivateModeNotice } from "./PrivateModeNotice";

afterEach(() => { cleanup(); vi.useRealTimers(); });

it("presents privacy as a status with details on keyboard focus, without a button or input", () => {
  render(<PrivateModeNotice />);
  const status = screen.getByRole("note", { name: "Private processing" });
  expect(screen.queryByRole("button")).not.toBeInTheDocument();
  expect(screen.queryByRole("tooltip")).not.toBeInTheDocument();
  fireEvent.focus(status);
  const details = screen.getByRole("tooltip");
  expect(status).toHaveAttribute("aria-describedby", details.id);
  expect(details).toHaveTextContent("Conversations and files are saved on this server");
  expect(details).toHaveTextContent("trusted server administrators");
  expect(details).toHaveTextContent("Cloudflare");
  expect(details).toHaveTextContent("web, saved memory and media are unavailable");
  expect(details).toHaveTextContent("Private stays selected after the first message");
  expect(screen.queryByRole("textbox")).not.toBeInTheDocument();
  fireEvent.keyDown(document, { key: "Escape" });
  expect(screen.queryByRole("tooltip")).not.toBeInTheDocument();
});

it("keeps hover details readable when the pointer moves from the badge to the tooltip", () => {
  vi.useFakeTimers();
  render(<PrivateModeNotice />);
  const status = screen.getByRole("note");
  fireEvent.pointerEnter(status);
  const details = screen.getByRole("tooltip");
  fireEvent.pointerLeave(status);
  fireEvent.pointerEnter(details);
  act(() => vi.advanceTimersByTime(200));
  expect(details).toBeVisible();
  fireEvent.pointerLeave(details);
  act(() => vi.advanceTimersByTime(200));
  expect(screen.queryByRole("tooltip")).not.toBeInTheDocument();
});

it("dismisses opened details on an outside touch and retains project restrictions", () => {
  render(<PrivateModeNotice lockReason="Leave this Private project before choosing another processing mode." />);
  fireEvent.focus(screen.getByRole("note"));
  expect(screen.getByRole("tooltip")).toHaveTextContent("Leave this Private project");
  fireEvent.pointerDown(document.body, { pointerType: "touch" });
  expect(screen.queryByRole("tooltip")).not.toBeInTheDocument();
});
