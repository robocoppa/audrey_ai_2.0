import { useEffect, useRef, useState } from "react";

import { ApiError } from "./api";
import { actOnTaxCase, type TaxCase, type TaxCaseAction } from "./taxReviewApi";
import "./TaxReviewQuestions.css";

export function TaxReviewQuestions({
  conversationId, taxCase, disabled, answerVersion = 0, onCaseChange, onReload,
}: {
  conversationId: string;
  taxCase: TaxCase;
  disabled: boolean;
  answerVersion?: number;
  onCaseChange: (value: TaxCase) => void;
  onReload: () => void;
}) {
  const [busy, setBusy] = useState<string | null>(null);
  const [notice, setNotice] = useState<{ text: string; answerVersion: number } | null>(null);
  const noticeText = !disabled && notice?.answerVersion === answerVersion ? notice.text : "";
  const [error, setError] = useState<{ message: string; snapshot: TaxCase } | null>(null);
  const [deferred, setDeferred] = useState<string[]>([]);
  const currentError = error?.snapshot === taxCase ? error.message : "";
  const active = useRef(true);
  useEffect(() => {
    active.current = true;
    return () => { active.current = false; };
  }, []);
  const proposed = taxCase.facts.filter(({ status }) => status === "proposed");
  const visible = proposed.slice(0, 3);
  const missing = taxCase.missing_items.find(({ id, status }) => status !== "resolved" && !deferred.includes(id));

  async function act(action: TaxCaseAction, itemId: string, success: string) {
    if (busy || disabled || currentError) return;
    setBusy(itemId);
    setNotice(null);
    try {
      const updated = await actOnTaxCase(conversationId, taxCase.revision, action);
      if (!active.current) return;
      onCaseChange(updated);
      setNotice({ text: success, answerVersion });
    } catch (reason) {
      if (!active.current) return;
      setError({ snapshot: taxCase, message: reason instanceof ApiError && reason.status === 409
        ? "These details changed. Reload them before making a choice."
        : "I couldn't save that choice. Please reload the details and try again." });
    } finally {
      if (active.current) setBusy(null);
    }
  }

  function choose(factId: string, confirm: boolean) {
    return act({ operation: confirm ? "confirm_fact" : "reject_fact", data: { fact_id: factId } }, factId,
      confirm ? "I've saved your confirmation."
        : "I won't use that detail. Tell me the correct information in your next message.");
  }

  if (!visible.length && !missing && !noticeText) return null;
  return <article className="message message-assistant tax-review-question" aria-label="Audrey's tax review questions">
    <div className="message-label">Audrey</div>
    {visible.length ? <>
      <p className="tax-question-intro">{proposed.length === 1
        ? `Is this correct for your ${taxCase.tax_year} return?`
        : `Could you check these details for your ${taxCase.tax_year} return?`}</p>
      <ul className="tax-question-facts">
        {visible.map((fact) => <li key={fact.id}>
          <div className="tax-question-value"><strong>{fact.label}</strong><span>{valueOf(fact.value)}</span></div>
          {fact.case_tax_year !== undefined && fact.case_tax_year !== taxCase.tax_year
            ? <p className="tax-question-context">From your {fact.case_tax_year} review. Please check whether this still applies.</p> : null}
          {fact.provenance ? <details className="tax-question-source">
            <summary>{taxCase.files?.find(({ file_id }) => file_id === fact.provenance?.file_id)?.filename || "Uploaded document"} · Page {fact.provenance.page}{fact.provenance.box ? ` · ${fact.provenance.box}` : ""}</summary>
            <blockquote>{fact.provenance.original_text}</blockquote>
          </details> : null}
          <div className="tax-question-actions">
            <button type="button" className="tax-question-confirm" aria-label={`Confirm ${fact.label}`}
              disabled={disabled || busy !== null || Boolean(currentError)} onClick={() => void choose(fact.id, true)}>
              <span aria-hidden="true">✓</span> {busy === fact.id ? "Saving…" : "Yes, that's correct"}
            </button>
            <button type="button" aria-label={`Not correct: ${fact.label}`}
              disabled={disabled || busy !== null || Boolean(currentError)} onClick={() => void choose(fact.id, false)}>Not correct</button>
          </div>
        </li>)}
      </ul>
      {proposed.length > visible.length ? <p className="tax-question-context">{proposed.length - visible.length} more {proposed.length - visible.length === 1 ? "detail" : "details"} to review after these.</p> : null}
    </> : null}
    {!visible.length && missing ? <>
      <p className="tax-question-intro">Before we continue, could you help with this?</p>
      <ul className="tax-question-facts"><li>
        <div className="tax-question-value"><strong>{missing.label}</strong></div>
        <div className="tax-question-actions">
          <button type="button" className="tax-question-confirm" aria-label={`Mark ${missing.label} addressed`}
            disabled={disabled || busy !== null || Boolean(currentError)}
            onClick={() => void act({ operation: "resolve_missing_item", data: { id: missing.id } }, missing.id, "I've marked that as addressed.")}>
            {busy === missing.id ? "Saving…" : "Mark addressed"}
          </button>
          <button type="button" disabled={disabled || busy !== null || Boolean(currentError)} onClick={() => {
            setDeferred((current) => [...current, missing.id]);
            setNotice({ text: "I'll leave that open. You can add the information when you're ready.", answerVersion });
          }}>Not yet</button>
        </div>
      </li></ul>
    </> : null}
    {noticeText ? <p className="tax-question-notice" role="status">{noticeText}</p> : null}
    {currentError ? <div className="tax-question-error" role="alert"><p>{currentError}</p><button type="button" disabled={disabled || busy !== null} onClick={onReload}>Reload details</button></div> : null}
  </article>;
}

function valueOf(value: unknown): string {
  return typeof value === "string" ? value : JSON.stringify(value) ?? "Unknown";
}
