import { useEffect, useRef, useState } from "react";

import { ApiError } from "./api";
import { actOnTaxCase, type TaxCase, type TaxCaseAction } from "./taxReviewApi";
import "./TaxReviewQuestions.css";

export function TaxReviewQuestions({
  conversationId, taxCase, disabled, answerVersion = 0, onCaseChange, onReload, onChoice,
}: {
  conversationId: string;
  taxCase: TaxCase;
  disabled: boolean;
  answerVersion?: number;
  onCaseChange: (value: TaxCase) => void;
  onReload: () => void;
  onChoice?: (value: TaxCase, feedback: string) => void;
}) {
  const [busy, setBusy] = useState<string | null>(null);
  const [notice, setNotice] = useState<{ text: string; answerVersion: number } | null>(null);
  const noticeText = !disabled && notice?.answerVersion === answerVersion ? notice.text : "";
  const [error, setError] = useState<{ message: string; snapshot: TaxCase } | null>(null);
  const [deferred, setDeferred] = useState<string[]>([]);
  const currentError = error?.snapshot === taxCase ? error.message : "";
  const active = useRef(true);
  const saving = useRef(false);
  useEffect(() => {
    active.current = true;
    return () => { active.current = false; };
  }, []);
  const proposed = taxCase.facts.filter(({ status }) => status === "proposed");
  const visible = proposed.slice(0, 10);
  const missing = taxCase.missing_items.find(({ id, status }) => status !== "resolved" && !deferred.includes(id));

  async function act(action: TaxCaseAction, itemId: string, success: string, feedback: string) {
    if (saving.current || disabled || currentError) return;
    saving.current = true;
    setBusy(itemId);
    setNotice(null);
    try {
      const updated = await actOnTaxCase(conversationId, taxCase.revision, action);
      if (!active.current) return;
      onCaseChange(updated);
      setNotice({ text: success, answerVersion });
      onChoice?.(updated, feedback);
    } catch (reason) {
      if (!active.current) return;
      setError({ snapshot: taxCase, message: reason instanceof ApiError && reason.status === 409
        ? "These details changed. Reload them before making a choice."
        : "I couldn't save that choice. Please reload the details and try again." });
    } finally {
      saving.current = false;
      if (active.current) setBusy(null);
    }
  }

  function confirmDetails() {
    const fact = visible[0];
    if (!fact) return;
    const action: TaxCaseAction = visible.length === 1
      ? { operation: "confirm_fact", data: { fact_id: fact.id } }
      : { operation: "confirm_facts", data: { fact_ids: visible.map(({ id }) => id) } };
    const details = visible.map((item) => `${item.label}: ${valueOf(item.value).slice(0, 160)}`).join("; ");
    return act(action, "confirm", "Confirmed.",
      `I confirm these details for my ${taxCase.tax_year} tax review: ${details}. Continue from my saved case.`);
  }

  function rejectDetail(factId: string, label: string) {
    return act({ operation: "reject_fact", data: { fact_id: factId } }, factId, "Noted.",
      `The saved ${label} is not correct for my ${taxCase.tax_year} tax review. Ask me for the correct information before using it.`);
  }

  if (!visible.length && !missing && !noticeText && !currentError) return null;
  return <section className="message message-assistant tax-review-question" aria-label="Tax review choices">
    <div className="message-label">Audrey</div>
    <div className="tax-question-card">
    {visible.length ? <>
      <p className="tax-question-intro">{visible.length === 1
        ? `Is this correct for your ${taxCase.tax_year} review?`
        : `Are these details correct for your ${taxCase.tax_year} review?`}</p>
      <ul className="tax-question-facts">
        {visible.map((fact) => <li key={fact.id}>
          <div className="tax-question-value"><strong>{fact.label}</strong><span>{valueOf(fact.value)}</span>
            <button type="button" className="tax-question-change" aria-label={`Not correct: ${fact.label}`}
              disabled={disabled || busy !== null || Boolean(currentError)}
              onClick={() => void rejectDetail(fact.id, fact.label)}>{visible.length === 1 ? "Not correct" : "Change"}</button>
          </div>
          {fact.case_tax_year !== undefined && fact.case_tax_year !== taxCase.tax_year
            ? <p className="tax-question-context">From {fact.case_tax_year}. Check that this still applies.</p> : null}
          {fact.provenance ? <details className="tax-question-source">
            <summary>{taxCase.files?.find(({ file_id }) => file_id === fact.provenance?.file_id)?.filename || "Uploaded document"} · Page {fact.provenance.page}{fact.provenance.box ? ` · ${fact.provenance.box}` : ""}</summary>
            <blockquote>{fact.provenance.original_text}</blockquote>
          </details> : null}
        </li>)}
      </ul>
      <div className="tax-question-actions">
        <button type="button" className="tax-question-confirm"
          aria-label={visible.length === 1 ? `Confirm ${visible[0].label}` : "Confirm these details"}
          disabled={disabled || busy !== null || Boolean(currentError)} onClick={() => void confirmDetails()}>
          <span aria-hidden="true">✓</span> {busy === "confirm" ? "Saving…" : visible.length === 1 ? "Confirm" : "Confirm details"}
        </button>
      </div>
      {proposed.length > visible.length ? <p className="tax-question-context">{proposed.length - visible.length} more {proposed.length - visible.length === 1 ? "detail" : "details"} left to review.</p> : null}
    </> : null}
    {!visible.length && missing ? <>
      <p className="tax-question-intro">Could you help with this?</p>
      <ul className="tax-question-facts"><li>
        <div className="tax-question-value"><strong>{missing.label}</strong></div>
        <div className="tax-question-actions">
          <button type="button" className="tax-question-confirm" aria-label={`Mark ${missing.label} addressed`}
            disabled={disabled || busy !== null || Boolean(currentError)}
            onClick={() => void act({ operation: "resolve_missing_item", data: { id: missing.id } }, missing.id, "Marked addressed.", `I have addressed this item in my ${taxCase.tax_year} tax review: ${missing.label}. Check my saved case and continue.`)}>
            {busy === missing.id ? "Saving…" : "I’ve provided this"}
          </button>
          <button type="button" disabled={disabled || busy !== null || Boolean(currentError)} onClick={() => {
            setDeferred((current) => [...current, missing.id]);
            setNotice({ text: "Left open for later.", answerVersion });
            onChoice?.(taxCase, `Leave this item open for later: ${missing.label}. Continue with the rest of my tax review without asking for it again this turn.`);
          }}>Later</button>
        </div>
      </li></ul>
    </> : null}
    {noticeText ? <p className="tax-question-notice" role="status">{noticeText}</p> : null}
    {currentError ? <div className="tax-question-error" role="alert"><p>{currentError}</p><button type="button" disabled={disabled || busy !== null} onClick={onReload}>Reload details</button></div> : null}
    </div>
  </section>;
}

function valueOf(value: unknown): string {
  return typeof value === "string" ? value : JSON.stringify(value) ?? "Unknown";
}
