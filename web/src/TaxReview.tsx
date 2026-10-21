import { useEffect, useId, useState, type FormEvent } from "react";

import { ApiError } from "./api";
import {
  actOnTaxCase,
  getTaxCase,
  saveTaxCase,
  type TaxCase,
  type TaxCaseAction,
} from "./taxReviewApi";
import "./TaxReview.css";

export function TaxReview({
  conversationId,
  expanded,
  active,
  disabled,
  refreshVersion,
  onExpandedChange,
  onCaseChange,
  onSaved,
}: {
  conversationId: string;
  expanded: boolean;
  active: boolean;
  disabled: boolean;
  refreshVersion: number;
  onExpandedChange: (expanded: boolean) => void;
  onCaseChange: (value: TaxCase | null) => void;
  onSaved: () => void;
}) {
  const panelId = useId();
  const [taxCase, setTaxCase] = useState<TaxCase | null>(null);
  const [year, setYear] = useState("");
  const [goal, setGoal] = useState("");
  const [loadedKey, setLoadedKey] = useState("");
  const [loadFailed, setLoadFailed] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [reloadVersion, setReloadVersion] = useState(0);
  const loadKey = `${conversationId}:${refreshVersion}:${reloadVersion}`;
  const loading = loadedKey !== loadKey;

  useEffect(() => {
    const controller = new AbortController();
    getTaxCase(conversationId, controller.signal).then(({ case: value }) => {
      if (controller.signal.aborted) return;
      setTaxCase(value);
      onCaseChange(value);
      setYear(value ? String(value.tax_year) : "");
      setGoal(value?.goal ?? "");
      setError("");
      setLoadFailed(false);
    }).catch((reason: unknown) => {
      if (!controller.signal.aborted) {
        setLoadFailed(true);
        setError("Could not load tax review: " + messageOf(reason));
      }
    }).finally(() => {
      if (!controller.signal.aborted) setLoadedKey(loadKey);
    });
    return () => controller.abort();
  }, [conversationId, loadKey, onCaseChange]);

  function acceptCase(value: TaxCase) {
    setTaxCase(value);
    onCaseChange(value);
    setYear(String(value.tax_year));
    setGoal(value.goal);
  }

  async function save(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (busy || disabled || loading) return;
    const taxYear = Number(year);
    if (!/^\d{4}$/u.test(year) || !Number.isInteger(taxYear) || taxYear < 1900 || taxYear > 2100) {
      setError("Enter a tax year from 1900 to 2100.");
      return;
    }
    if (!goal.trim() || goal.length > 2000) {
      setError("Enter a review goal of 1 to 2,000 characters.");
      return;
    }
    setBusy(true);
    setError("");
    try {
      acceptCase(await saveTaxCase(conversationId, {
        tax_year: taxYear, jurisdiction: "US-federal", goal: goal.trim(),
      }));
      onSaved();
    } catch (reason) {
      setError("Could not save tax review: " + messageOf(reason));
    } finally {
      setBusy(false);
    }
  }

  async function act(action: TaxCaseAction) {
    if (!taxCase || busy || disabled || loading) return;
    setBusy(true);
    setError("");
    try {
      acceptCase(await actOnTaxCase(conversationId, taxCase.revision, action));
    } catch (reason) {
      setError("Could not update tax review: " + messageOf(reason));
    } finally {
      setBusy(false);
    }
  }

  if (!expanded && !taxCase) return null;
  const locked = busy || disabled || loading;
  return (
    <section className="tax-review" aria-label="Tax review case">
      <button
        type="button"
        className="tax-review-toggle"
        aria-expanded={expanded}
        aria-controls={panelId}
        onClick={() => onExpandedChange(!expanded)}
      >
        <strong>Tax review{taxCase ? ` · ${taxCase.tax_year}` : " setup"}</strong>
        <span>{active ? "Selected" : taxCase ? "Saved case" : "Configure case"} <span aria-hidden="true">{expanded ? "⌃" : "⌄"}</span></span>
      </button>
      {expanded ? <div id={panelId} className="tax-review-content">
        <p className="tax-review-note">Planning checklist, not a completed tax return. No upload required.</p>
        {!active && taxCase ? <p className="tax-review-note">Choose Tax review in Tools &amp; skills to use this case for your next message.</p> : null}
        {loading ? <p role="status">Loading tax review…</p> : null}
        {!taxCase ? <form onSubmit={(event) => void save(event)}>
          <div className="tax-review-fields">
            <label>Tax year
              <input aria-label="Tax year" inputMode="numeric" type="number" min="1900" max="2100" step="1" required value={year} onChange={(event) => setYear(event.target.value)} disabled={locked} placeholder="2025" />
            </label>
            <label>Jurisdiction
              <select aria-label="Tax jurisdiction" value="US-federal" disabled={locked}>
                <option value="US-federal">US federal</option>
              </select>
            </label>
          </div>
          <label>Review goal
            <textarea aria-label="Review goal" rows={2} maxLength={2000} required value={goal} onChange={(event) => setGoal(event.target.value)} disabled={locked} placeholder="What would you like to review?" />
          </label>
          <div className="tax-review-actions">
            <span>{goal.length.toLocaleString()} / 2,000 characters</span>
            <button type="submit" disabled={locked || loadFailed}>{busy ? "Saving…" : "Save and select Tax review"}</button>
          </div>
        </form> : <dl className="tax-review-scope">
          <div><dt>Tax year</dt><dd>{taxCase.tax_year}</dd></div>
          <div><dt>Jurisdiction</dt><dd>US federal</dd></div>
          <div><dt>Review goal</dt><dd>{taxCase.goal}</dd></div>
        </dl>}
        {error ? <div className="tax-review-error" role="alert">
          <span>{error}</span>
          <button type="button" disabled={locked} onClick={() => setReloadVersion((current) => current + 1)}>Reload case</button>
        </div> : null}
        {taxCase ? <div className="tax-review-details">
          <p className="tax-review-summary"><strong>{taxCase.checklist.status === "ready_for_review" ? "Ready for review" : "Needs information"}</strong><span>{taxCase.evidence.length} evidence · {taxCase.calculations.length} calculations</span></p>
          <ul className="tax-review-checks" aria-label="Tax review checklist">
            {taxCase.checklist.checks.map((check) => <li key={check.id} data-status={check.status}>
              <span aria-hidden="true">{check.status === "passed" ? "✓" : "○"}</span>
              <div><strong>{check.label}</strong><span>{check.status.replaceAll("_", " ")}{check.issues.length ? ` · ${check.issues.join(" ")}` : ""}</span></div>
            </li>)}
          </ul>
          {taxCase.facts.length ? <details open>
            <summary>Facts ({taxCase.facts.length})</summary>
            <ul className="tax-review-records">
              {taxCase.facts.map((fact) => <li key={fact.id}>
                <div>
                  <strong>{fact.label}</strong><span>{displayValue(fact.value)}</span>
                  <small>{fact.status === "confirmed" ? "Confirmed by you" : "Proposed · confirm only if correct"}</small>
                  {fact.provenance ? <details className="tax-review-source">
                    <summary>Source · Page {fact.provenance.page}{fact.provenance.box ? ` · ${fact.provenance.box}` : ""}</summary>
                    <p>{taxCase.files?.find(({ file_id }) => file_id === fact.provenance?.file_id)?.filename || fact.provenance.file_id}</p>
                    <small>File {fact.provenance.file_id} · SHA-256 <code title={fact.provenance.checksum}>{fact.provenance.checksum.slice(0, 16)}…</code></small>
                    <blockquote>{fact.provenance.original_text}</blockquote>
                    <p>Normalized value: {displayValue(fact.value)}{fact.provenance.normalization ? ` · ${fact.provenance.normalization}` : ""}</p>
                  </details> : null}
                </div>
                {fact.status === "proposed" ? <button type="button" disabled={locked} aria-label={`Confirm ${fact.label}`} onClick={() => void act({ operation: "confirm_fact", data: { fact_id: fact.id } })}>Confirm</button> : null}
              </li>)}
            </ul>
          </details> : null}
          {taxCase.evidence.length ? <details>
            <summary>Evidence ({taxCase.evidence.length})</summary>
            <ul className="tax-review-records tax-review-evidence">
              {taxCase.evidence.map((item) => <li key={item.id}><div>
                <strong>{item.claim}</strong>
                <span>Document <code title={item.document_id}>{shortIdentity(item.document_id)}</code> · Revision {item.revision} · Page {item.page}{item.section ? ` · ${item.section}` : ""}</span>
                <small>Tax year: {item.tax_year ?? "unknown"} · Applicability: {item.tax_year_status} · {item.jurisdiction}</small>
                <blockquote>{item.passage}</blockquote>
                <small>Reference release: {item.reference_release} · {item.catalog}{item.conflicts_with.length ? ` · ${item.conflicts_with.length} conflicting evidence` : ""}</small>
              </div></li>)}
            </ul>
          </details> : null}
          {taxCase.calculations.length ? <details>
            <summary>Calculations ({taxCase.calculations.length})</summary>
            <ul className="tax-review-records tax-review-calculations">
              {taxCase.calculations.map((item) => <li key={item.id}><div>
                <strong>{item.label}</strong>
                <code>{item.expression}</code>
                <dl className="tax-review-inputs">{Object.entries(item.inputs).map(([name, value]) => <div key={name}><dt>{name}</dt><dd>{value}</dd></div>)}</dl>
                <span>Result: <strong>{item.result}</strong> · Rounding: {item.rounding}</span>
                <small>Dependencies: {item.fact_ids.length} facts · {item.evidence_ids.length} evidence</small>
              </div></li>)}
            </ul>
          </details> : null}
          {taxCase.missing_items.length ? <details open>
            <summary>Missing items ({taxCase.missing_items.filter(({ status }) => status !== "resolved").length} open)</summary>
            <ul className="tax-review-records">
              {taxCase.missing_items.map((item) => <li key={item.id}>
                <div><strong>{item.label}</strong><small>{item.status === "resolved" ? "Resolved" : "Open"}</small></div>
                {item.status !== "resolved" ? <button type="button" disabled={locked} aria-label={`Resolve ${item.label}`} onClick={() => void act({ operation: "resolve_missing_item", data: { id: item.id } })}>Resolve</button> : null}
              </li>)}
            </ul>
          </details> : null}
          {taxCase.checklist.next_steps.length ? <div><strong>Next steps</strong><ul className="tax-review-next">{taxCase.checklist.next_steps.map((step, index) => <li key={index}>{step}</li>)}</ul></div> : null}
          {taxCase.checklist.document_scope ? <p className="tax-review-note">{taxCase.checklist.document_scope}</p> : null}
          <p className="tax-review-release">Reference release: {taxCase.reference_release || "Not pinned"} · Checklist v{taxCase.checklist_version} · Revision {taxCase.revision}</p>
        </div> : null}
      </div> : null}
    </section>
  );
}

function shortIdentity(value: string): string {
  return /^[0-9a-f]{64}$/u.test(value) ? value.slice(0, 16) + "…" : value;
}

function displayValue(value: unknown): string {
  return typeof value === "string" ? value : JSON.stringify(value) ?? "Unknown";
}

function messageOf(reason: unknown): string {
  if (reason instanceof ApiError) return `${reason.message} (HTTP ${reason.status})`;
  return reason instanceof Error ? reason.message : "Please try again.";
}
