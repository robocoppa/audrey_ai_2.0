import { useEffect, useId, useState, type FormEvent } from "react";

import { ApiError } from "./api";
import {
  actOnTaxCase,
  defaultTaxYear,
  getTaxCase,
  saveTaxCase,
  type TaxCase,
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
  const [year, setYear] = useState(() => String(defaultTaxYear()));
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
      setYear(value ? String(value.tax_year) : String(defaultTaxYear()));
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

  async function save(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (busy || disabled || loading || loadFailed) return;
    const taxYear = Number(year);
    if (!/^\d{4}$/u.test(year) || !Number.isInteger(taxYear) || taxYear < 1900 || taxYear > 2100) {
      setError("Enter a tax year from 1900 to 2100.");
      return;
    }
    if (!taxCase && (!goal.trim() || goal.length > 2000)) {
      setError("Enter a review goal of 1 to 2,000 characters.");
      return;
    }
    setBusy(true);
    setError("");
    try {
      const value = taxCase
        ? await actOnTaxCase(conversationId, taxCase.revision, {
          operation: "change_year", data: { tax_year: taxYear },
        })
        : await saveTaxCase(conversationId, {
          tax_year: taxYear, jurisdiction: "US-federal", goal: goal.trim(),
        });
      setTaxCase(value);
      onCaseChange(value);
      setYear(String(value.tax_year));
      setGoal(value.goal);
      if (!taxCase) onSaved();
      onExpandedChange(false);
    } catch (reason) {
      setError("Could not save tax review: " + messageOf(reason));
    } finally {
      setBusy(false);
    }
  }

  if (!expanded && !taxCase) return null;
  const locked = busy || disabled || loading || loadFailed;
  const yearChanged = Boolean(taxCase && year !== String(taxCase.tax_year));
  const hasSavedWork = taxCase && [
    taxCase.facts, taxCase.evidence, taxCase.calculations, taxCase.return_items, taxCase.missing_items,
  ].some((items) => items.length > 0);
  return (
    <section className="tax-review" aria-label="Tax review case" data-active={active}>
      <button
        type="button"
        className="tax-review-toggle"
        aria-expanded={expanded}
        aria-controls={expanded ? panelId : undefined}
        onClick={() => onExpandedChange(!expanded)}
      >
        <strong>Tax review · {taxCase?.tax_year ?? year}</strong>{" "}
        <span>Settings <svg viewBox="0 0 24 24" aria-hidden="true"><path d={expanded ? "m7 14 5-5 5 5" : "m7 10 5 5 5-5"} /></svg></span>
      </button>
      {expanded ? <div id={panelId} className="tax-review-content">
        {loading ? <p className="tax-review-note" role="status">Loading settings…</p> : null}
        <form onSubmit={(event) => void save(event)}>
          <div className="tax-review-fields">
            <label>Tax year
              <input aria-label="Tax year" inputMode="numeric" type="number" min="1900" max="2100" step="1" required value={year} onChange={(event) => setYear(event.target.value)} disabled={locked} />
            </label>
            <span className="tax-review-jurisdiction">US federal</span>
          </div>
          {!taxCase ? <label>Review goal
            <textarea aria-label="Review goal" rows={2} maxLength={2000} required value={goal} onChange={(event) => setGoal(event.target.value)} disabled={locked} placeholder="What would you like to review?" />
          </label> : <div className="tax-review-goal"><span>Review goal</span><p>{taxCase.goal}</p></div>}
          {yearChanged && hasSavedWork ? <p className="tax-review-note">Saved information will need review for {year}.</p> : null}
          <div className="tax-review-actions">
            <button type="submit" disabled={locked || Boolean(taxCase && !yearChanged)}>{busy ? "Saving…" : taxCase ? "Save year" : "Use Tax review"}</button>
          </div>
        </form>
        {error ? <div className="tax-review-error" role="alert">
          <span>{error}</span>
          <button type="button" disabled={busy || disabled || loading} onClick={() => setReloadVersion((current) => current + 1)}>Reload settings</button>
        </div> : null}
      </div> : null}
    </section>
  );
}

function messageOf(reason: unknown): string {
  if (reason instanceof ApiError) return `${reason.message} (HTTP ${reason.status})`;
  return reason instanceof Error ? reason.message : "Please try again.";
}
