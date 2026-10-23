import { useEffect, useLayoutEffect, useRef, useState } from "react";

import { ApiError } from "./api";
import { actOnTaxCase, type TaxCase, type TaxCaseAction, type TaxFact } from "./taxReviewApi";
import "./TaxReviewQuestions.css";

type Props = {
  conversationId: string;
  taxCase: TaxCase;
  disabled: boolean;
  answerVersion?: number;
  onCaseChange: (value: TaxCase) => void;
  onReload: () => void;
  onChoice?: (value: TaxCase, feedback: string) => boolean | void;
};

type Edit = { id: string; label: string; value: string; revision: number };

export function TaxReviewQuestions(props: Props) {
  // A pending response must not update another conversation, case or return year.
  return <CaseFacts key={`${props.conversationId}:${props.taxCase.id}:${props.taxCase.tax_year}`} {...props} />;
}

function CaseFacts({
  conversationId, taxCase, disabled, answerVersion = 0, onCaseChange, onReload, onChoice,
}: Props) {
  const [busy, setBusy] = useState<string | null>(null);
  const [notice, setNotice] = useState<{ text: string; answerVersion: number } | null>(null);
  const noticeText = !disabled && notice?.answerVersion === answerVersion ? notice.text : "";
  const [error, setError] = useState<{ message: string; revision: number } | null>(null);
  const [edit, setEdit] = useState<Edit | null>(null);
  const [answer, setAnswer] = useState("");
  const currentError = error?.revision === taxCase.revision ? error.message : "";
  const active = useRef(true);
  const saving = useRef(false);
  const currentCase = useRef(taxCase);
  useLayoutEffect(() => { currentCase.current = taxCase; }, [taxCase]);
  const flow = useRef<HTMLDivElement>(null);
  const following = useRef(true);
  useEffect(() => {
    active.current = true;
    return () => { active.current = false; };
  }, []);
  const confirmed = taxCase.facts.filter(({ status }) => status === "confirmed");
  const proposed = taxCase.facts.filter(({ status }) => status === "proposed");
  const visible = proposed.slice(0, 10);
  const missing = taxCase.missing_items.find(({ status }) => status !== "resolved");
  const reviewQuestion = !taxCase.review?.stale && taxCase.review?.status === "needs_input"
    ? taxCase.review.next_question.trim() : "";
  const question = missing ? missing.question?.trim() || reviewQuestion
    || `Please tell Audrey about: ${missing.label}` : reviewQuestion;
  const locked = disabled || busy !== null || Boolean(currentError);
  const activityKey = JSON.stringify([
    confirmed.map(({ id, value }) => [id, value]), visible.map(({ id }) => id), question,
  ]);
  useLayoutEffect(() => {
    // Keep the user's place while they inspect facts; follow a new question
    // only when they were already at the bottom of the panel.
    if (following.current && flow.current) flow.current.scrollTop = flow.current.scrollHeight;
  }, [activityKey]);

  async function act(action: TaxCaseAction, itemId: string, success: string, feedback: string, revision = taxCase.revision) {
    if (saving.current || disabled || currentError) return;
    saving.current = true;
    setBusy(itemId);
    setNotice(null);
    try {
      const updated = await actOnTaxCase(conversationId, revision, action);
      if (!active.current || currentCase.current.revision > updated.revision) return;
      onCaseChange(updated);
      setEdit(null);
      setNotice({ text: success, answerVersion });
      onChoice?.(updated, feedback);
    } catch (reason) {
      if (!active.current) return;
      setError({ revision: taxCase.revision, message: reason instanceof ApiError && reason.status === 409
        ? "These details changed. Reload them before saving."
        : "I couldn't save that change. Please reload the details and try again." });
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
    return act(action, "confirm", "Details saved.",
      `I confirm these details for my ${taxCase.tax_year} tax review: ${details}. Continue from my saved case.`);
  }

  function saveEdit() {
    if (!edit || edit.revision !== taxCase.revision || !edit.label.trim() || !edit.value.trim()) return;
    return act({ operation: "edit_fact", data: { fact_id: edit.id, label: edit.label.trim(), value: edit.value.trim() } },
      edit.id, "Fact updated.", `I updated ${edit.label.trim()} in my ${taxCase.tax_year} tax review to: ${edit.value.trim()}. Use the saved correction and recheck anything that depends on it.`, edit.revision);
  }

  function factTable(facts: TaxFact[], name: string) {
    return <table className="tax-facts-table" aria-label={name}>
      <colgroup><col /><col /><col /></colgroup>
      <thead><tr><th scope="col">Fact</th><th scope="col">Value</th><th scope="col" aria-label="Actions" /></tr></thead>
      <tbody>{facts.map((fact) => <tr key={fact.id}>
        {edit?.id === fact.id ? <td colSpan={3} className="tax-fact-edit-cell">
          <form className="tax-fact-editor" onSubmit={(event) => { event.preventDefault(); void saveEdit(); }}>
            <label>Fact name<input aria-label="Fact name" value={edit.label} maxLength={300} disabled={locked}
              onChange={(event) => setEdit({ ...edit, label: event.target.value })} /></label>
            <label>Value for {fact.label}<textarea aria-label={`Value for ${fact.label}`} rows={3} value={edit.value} maxLength={4000} disabled={locked}
              onChange={(event) => setEdit({ ...edit, value: event.target.value })} /></label>
            {edit.revision !== taxCase.revision ? <p className="tax-question-context">This fact changed while you were editing. Cancel and reopen it to use the latest value.</p> : null}
            <div className="tax-question-actions">
              <button type="submit" className="tax-question-confirm" disabled={locked || edit.revision !== taxCase.revision || !edit.label.trim() || !edit.value.trim()}>{busy === fact.id ? "Saving…" : "Save fact"}</button>
              <button type="button" disabled={busy !== null} onClick={() => setEdit(null)}>Cancel</button>
            </div>
          </form>
        </td> : <>
          <th scope="row">{fact.label}
            {fact.confirmed_by === "user_message" && fact.source_message_id ? <span className="tax-fact-origin">From your message</span> : null}
            {fact.case_tax_year !== undefined && fact.case_tax_year !== taxCase.tax_year
              ? <span className="tax-fact-year">From {fact.case_tax_year}</span> : null}
          </th>
          <td><span className="tax-fact-value">{valueOf(fact.value)}</span>
            {fact.provenance ? <details className="tax-question-source">
              <summary>{taxCase.files?.find(({ file_id }) => file_id === fact.provenance?.file_id)?.filename || "Uploaded document"} · Page {fact.provenance.page}{fact.provenance.box ? ` · ${fact.provenance.box}` : ""}</summary>
              <blockquote>{fact.provenance.original_text}</blockquote>
            </details> : null}
          </td>
          <td><button type="button" className="tax-fact-edit" aria-label={`Edit ${fact.label}`} disabled={locked || edit !== null}
            onClick={() => { following.current = false; setEdit({ id: fact.id, label: fact.label, value: editValue(fact.value), revision: taxCase.revision }); }}>Edit</button></td>
        </>}
      </tr>)}</tbody>
    </table>;
  }

  if (!confirmed.length && !visible.length && !question && !noticeText && !currentError) return null;
  return <section className="tax-review-question" aria-label="Tax review facts">
    <h3 className="tax-question-heading">Your facts</h3>
    <div className="tax-question-flow" ref={flow} tabIndex={0} onScroll={() => {
      const box = flow.current;
      if (box) following.current = box.scrollHeight - box.scrollTop - box.clientHeight < 48;
    }}>
      {confirmed.length ? factTable(confirmed, "Saved tax facts") : null}
      {visible.length ? <div className="tax-question-card">
        <h4>Check these details</h4>
        <p className="tax-question-intro">{visible.length === 1
          ? `Is this ${isAmount(visible[0]) ? "amount" : "detail"} correct for ${taxCase.tax_year}?`
          : `Are these details correct for ${taxCase.tax_year}?`} Edit anything that needs correcting.</p>
        {factTable(visible, "Details to check")}
        <div className="tax-question-actions">
          <button type="button" className="tax-question-confirm"
            aria-label={visible.length === 1 ? `Confirm ${visible[0].label}` : "Confirm these details"}
            disabled={locked || edit !== null} onClick={() => void confirmDetails()}>
            <span aria-hidden="true">✓</span> {busy === "confirm" ? "Saving…" : visible.length === 1 ? "This is correct" : "These are correct"}
          </button>
        </div>
        {proposed.length > visible.length ? <p className="tax-question-context">{proposed.length - visible.length} more details to check.</p> : null}
      </div> : null}
      {!visible.length && question ? <form className="tax-question-card tax-question-answer" onSubmit={(event) => {
        event.preventDefault();
        if (locked || !answer.trim() || !onChoice) return;
        const accepted = onChoice(taxCase, answer.trim());
        if (accepted !== false) { setAnswer(""); setNotice(null); }
      }}>
        <h4>Audrey needs to know</h4>
        <p className="tax-question-intro">{question}</p>
        <label>Your answer<textarea aria-label="Your answer" rows={3} value={answer} maxLength={4000} disabled={locked}
          onChange={(event) => setAnswer(event.target.value)} /></label>
        <div className="tax-question-actions"><button type="submit" className="tax-question-confirm" disabled={locked || !answer.trim() || !onChoice}>Send answer</button></div>
        <p className="tax-question-context">You can also answer in the conversation.</p>
      </form> : null}
      {noticeText ? <p className="tax-question-notice" role="status">{noticeText}</p> : null}
      {currentError ? <div className="tax-question-error" role="alert"><p>{currentError}</p><button type="button" disabled={disabled || busy !== null} onClick={onReload}>Reload details</button></div> : null}
    </div>
  </section>;
}

function isAmount(fact: TaxFact): boolean {
  return /wages|withhold|income|amount|deduction|payment|credit|dollars/iu.test(fact.label)
    || (typeof fact.value === "string" && /^\$?[\d,]+\.\d{2}$/u.test(fact.value.trim()));
}

function editValue(value: unknown): string {
  return typeof value === "string" ? value : JSON.stringify(value) ?? "";
}

function valueOf(value: unknown): string {
  if (value === null || value === undefined) return "Not provided";
  if (typeof value === "boolean") return value ? "Yes" : "No";
  if (typeof value === "string" || typeof value === "number") return String(value);
  if (Array.isArray(value)) return value.map(valueOf).join("; ");
  if (typeof value === "object") return Object.entries(value).map(([name, item]) => `${name.replaceAll("_", " ")}: ${valueOf(item)}`).join("; ");
  return "Not provided";
}
