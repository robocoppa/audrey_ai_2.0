import { useEffect, useId, useRef, useState, type FormEvent, type ReactNode } from "react";

import { ApiError } from "./api";
import {
  actOnTaxCase,
  createTaxCaseProject,
  defaultTaxYear,
  getTaxCase,
  saveTaxCase,
  taxReviewScopeNotice,
  type TaxCase,
} from "./taxReviewApi";
import { TaxWorkbookDownload } from "./TaxWorkbookDownload";
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
  taskControls,
  presentation = "dialog",
  children,
}: {
  conversationId: string;
  expanded: boolean;
  active: boolean;
  disabled: boolean;
  refreshVersion: number;
  onExpandedChange: (expanded: boolean) => void;
  onCaseChange: (value: TaxCase | null) => void;
  onSaved: () => void;
  onReview?: () => void;
  taskControls?: ReactNode;
  presentation?: "dialog" | "toolbox";
  children?: ReactNode;
}) {
  const panelId = useId();
  const dialogRef = useRef<HTMLDialogElement>(null);
  const titleRef = useRef<HTMLHeadingElement>(null);
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
  const referenceUpdate = taxCase?.reference_update;
  const referenceBlocked = referenceUpdate?.status === "update_available" || referenceUpdate?.status === "unavailable";

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

  useEffect(() => {
    if (!expanded || presentation === "toolbox") return;
    const dialog = dialogRef.current;
    if (!dialog) return;
    const viewport = window.visualViewport;
    function fit() {
      if (!dialog) return;
      const left = viewport?.offsetLeft ?? 0;
      const top = viewport?.offsetTop ?? 0;
      const width = viewport?.width ?? window.innerWidth;
      const height = viewport?.height ?? window.innerHeight;
      dialog.style.setProperty("--tax-dialog-left", `${left + width / 2}px`);
      dialog.style.setProperty("--tax-dialog-top", `${top + height / 2}px`);
      dialog.style.setProperty("--tax-dialog-width", `${Math.max(0, width - 24)}px`);
      dialog.style.setProperty("--tax-dialog-height", `${Math.max(0, height - 24)}px`);
    }
    fit();
    dialog.showModal();
    // Start on the heading so opening settings does not summon a mobile keyboard.
    titleRef.current?.focus();
    window.addEventListener("resize", fit);
    viewport?.addEventListener("resize", fit);
    viewport?.addEventListener("scroll", fit);
    return () => {
      window.removeEventListener("resize", fit);
      viewport?.removeEventListener("resize", fit);
      viewport?.removeEventListener("scroll", fit);
      dialog.close();
    };
  }, [expanded, presentation]);

  async function save(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (busy || disabled || loading || loadFailed || referenceBlocked) return;
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

  async function groupCase() {
    if (!taxCase || taxCase.project_id || busy || disabled || loading || loadFailed) return;
    setBusy(true);
    setError("");
    try {
      const value = await createTaxCaseProject(conversationId);
      setTaxCase(value);
      onCaseChange(value);
    } catch (reason) {
      setError("Could not create the Private project: " + messageOf(reason));
    } finally {
      setBusy(false);
    }
  }

  async function updateLibrary() {
    if (!taxCase || referenceUpdate?.status !== "update_available" || busy || disabled || loading || loadFailed) return;
    setBusy(true);
    setError("");
    try {
      const value = await actOnTaxCase(conversationId, taxCase.revision, {
        operation: "update_reference_release", data: { reference_release: referenceUpdate.release_id },
      });
      setTaxCase(value);
      onCaseChange(value);
    } catch (reason) {
      setError("Could not update the IRS library: " + messageOf(reason));
      // Keep the advertised choice locked until a fresh snapshot is loaded.
      setLoadFailed(true);
    } finally {
      setBusy(false);
    }
  }

  if (!expanded && !taxCase) return null;
  const locked = busy || disabled || loading || loadFailed || referenceBlocked;
  const yearChanged = Boolean(taxCase && year !== String(taxCase.tax_year));
  const hasSavedWork = taxCase && [
    taxCase.facts, taxCase.evidence, taxCase.calculations, taxCase.return_items, taxCase.missing_items,
  ].some((items) => items.length > 0);
  const settings = <>
        <div className="tax-review-content">
        {presentation === "dialog" ? <p className="tax-review-disclaimer">Tax review can make mistakes. Check the sources and figures before filing; Audrey does not prepare or file your return.</p> : null}
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
          {!taxCase ? <p className="tax-review-note">Your case, documents and saved worksheets stay together in a Private project.</p> : null}
          <div className="tax-review-actions">
            <button type="submit" disabled={locked || Boolean(taxCase && !yearChanged)}>{busy ? "Saving…" : taxCase ? "Save year" : "Use Tax review"}</button>
          </div>
        </form>
        {taxCase && !taxCase.project_id ? <div className="tax-review-project">
          <p className="tax-review-note">Keep this case, its documents and saved worksheets together.</p>
          <button type="button" disabled={busy || disabled || loading || loadFailed} onClick={() => void groupCase()}>
            {busy ? "Saving…" : "Create Private project"}
          </button>
        </div> : null}
        </div>
        {error ? <div className="tax-review-error" role="alert">
          <span>{error}</span>
          <button type="button" disabled={busy || disabled || loading} onClick={() => setReloadVersion((current) => current + 1)}>Reload settings</button>
        </div> : null}
  </>;
  return (
    <section className="tax-review" aria-label="Tax review case" data-active={active} data-presentation={presentation}>
      <button
        type="button"
        className="tax-review-toggle"
        aria-haspopup={presentation === "dialog" ? "dialog" : undefined}
        aria-expanded={expanded}
        aria-controls={expanded ? panelId : undefined}
        onClick={() => onExpandedChange(!expanded)}
      >
        <strong>Tax review · {taxCase?.tax_year ?? year}</strong>{" "}
        <span>Settings</span>
      </button>
      <div className="tax-review-body">
        {!expanded || presentation === "toolbox" ? <p className="tax-review-disclaimer">Tax review can make mistakes. Check the sources and figures before filing; Audrey does not prepare or file your return.</p> : null}
        {taxCase ? taskControls ?? <div className="tax-review-start">
          <span className="tax-review-note">{taxCase.review
            ? taxCase.review.stale ? "Review needs updating"
              : taxCase.review.status === "needs_input" ? "Review saved · Your input is needed"
                : taxCase.review.status === "limited_evidence" ? "Review saved · Evidence is limited" : "Review saved"
            : "Check your saved information and supporting rules."}</span>
        </div> : null}
        {referenceBlocked ? <div className="tax-review-library" role="status">
          <p>{referenceUpdate?.status === "update_available"
            ? "An updated IRS library is available. Update the library to continue; saved rules and related calculations will need rechecking."
            : "The IRS library is unavailable. Your saved review is still here; try loading it again before continuing."}</p>
          {referenceUpdate?.status === "update_available" ? <button type="button"
            disabled={busy || disabled || loading || loadFailed} onClick={() => void updateLibrary()}>
            {busy ? "Updating…" : "Update IRS library"}
          </button> : <button type="button" disabled={busy || disabled || loading}
            onClick={() => setReloadVersion((current) => current + 1)}>Reload library</button>}
        </div> : null}
      {expanded && presentation === "toolbox" ? <div id={panelId} className="tax-review-inline-settings" role="region" aria-label="Tax review settings">
        {settings}
      </div> : null}
      {expanded && presentation === "dialog" ? <dialog
        ref={dialogRef}
        id={panelId}
        className="tax-review-dialog"
        aria-labelledby={`${panelId}-title`}
        onCancel={(event) => { event.preventDefault(); onExpandedChange(false); }}
        onClick={(event) => {
          if (event.target !== event.currentTarget) return;
          const bounds = event.currentTarget.getBoundingClientRect();
          if (event.clientX < bounds.left || event.clientX > bounds.right
            || event.clientY < bounds.top || event.clientY > bounds.bottom) onExpandedChange(false);
        }}
      >
        <header className="tax-review-dialog-header">
          <h2 ref={titleRef} id={`${panelId}-title`} tabIndex={-1} autoFocus>Tax review settings</h2>
          <button type="button" aria-label="Close Tax review settings" onClick={() => onExpandedChange(false)}>
            <svg viewBox="0 0 24 24" aria-hidden="true"><path d="m6 6 12 12M6 18 18 6" /></svg>
          </button>
        </header>
        {settings}
      </dialog> : null}
        {children}
        {taxCase?.review ? <TaxWorkbookDownload taxCase={taxCase} disabled={loading || loadFailed}
          onReload={() => setReloadVersion((current) => current + 1)} /> : null}
        {taxCase?.review ? <details className="tax-review-worksheet">
          <summary>Review worksheet</summary>
          <p className="tax-review-note">{taxReviewScopeNotice(taxCase.review)}</p>
          <p>{taxCase.review.summary}</p>
          {taxCase.review.findings.length ? <ul>{taxCase.review.findings.map((finding, index) => <li key={index}>
            <strong>{finding.label}</strong><p>{finding.detail}</p>
            {finding.evidence_ids.map((id) => taxCase.evidence.find((evidence) => evidence.id === id)).filter((item) => item !== undefined)
              .map((evidence) => <details className="tax-question-source" key={evidence.id}>
                <summary>IRS source · Page {evidence.page}{evidence.section ? ` · ${evidence.section}` : ""}</summary>
                <blockquote>{evidence.passage}</blockquote>
              </details>)}
            {finding.calculation_ids.map((id) => taxCase.calculations.find((calculation) => calculation.id === id)).filter((item) => item !== undefined)
              .map((calculation) => <p className="tax-review-note" key={calculation.id}>{calculation.label}: {calculation.expression} = {calculation.result}</p>)}
          </li>)}</ul> : null}
          <p className="tax-review-note">{taxCase.review.document_scope ?? "Selected passages only; this is not a complete return review."}</p>
          {taxCase.review.next_question ? <p>{taxCase.review.next_question}</p> : null}
          {taxCase.review.limitations.length ? <ul className="tax-review-note">{taxCase.review.limitations.map((limitation, index) => <li key={index}>{limitation}</li>)}</ul> : null}
        </details> : null}
        {!expanded && error ? <div className="tax-review-error" role="alert">
          <span>{error}</span>
          <button type="button" disabled={busy || disabled || loading} onClick={() => setReloadVersion((current) => current + 1)}>Reload settings</button>
        </div> : null}
      </div>
    </section>
  );
}

function messageOf(reason: unknown): string {
  if (reason instanceof ApiError) return `${reason.message} (HTTP ${reason.status})`;
  return reason instanceof Error ? reason.message : "Please try again.";
}
