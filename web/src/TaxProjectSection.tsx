import { useEffect, useId, useRef, useState } from "react";

import { getConversation, type Conversation } from "./api";
import { listProjectTaxCases, type TaxCase } from "./taxReviewApi";
import "./TaxProjectSection.css";

type Props = {
  projectId: string;
  onConversationSelected: (conversation: Conversation) => void;
};

export function TaxProjectSection(props: Props) {
  return <ProjectCases key={props.projectId} {...props} />;
}

function ProjectCases({ projectId, onConversationSelected }: Props) {
  const headingId = useId();
  const active = useRef(true);
  useEffect(() => {
    active.current = true;
    return () => { active.current = false; };
  }, []);
  const [version, setVersion] = useState(0);
  const [loaded, setLoaded] = useState<{ projectId: string; items: TaxCase[]; error: boolean } | null>(null);
  const [opening, setOpening] = useState<string | null>(null);
  const [openError, setOpenError] = useState("");
  useEffect(() => {
    const controller = new AbortController();
    listProjectTaxCases(projectId, controller.signal).then(({ items }) => {
      if (!controller.signal.aborted) setLoaded({ projectId, items, error: false });
    }).catch(() => {
      if (!controller.signal.aborted) setLoaded({ projectId, items: [], error: true });
    });
    return () => controller.abort();
  }, [projectId, version]);

  const current = loaded?.projectId === projectId ? loaded : null;
  // Ordinary Private projects keep their existing layout when no tax case is saved.
  if (!current || (!current.error && !current.items.length)) return null;

  async function openCase(taxCase: TaxCase) {
    if (opening) return;
    setOpening(taxCase.id);
    setOpenError("");
    try {
      const conversation = await getConversation(taxCase.conversation_id);
      if (!active.current) return;
      if (conversation.project_id !== projectId || !conversation.private_required) {
        throw new Error("This conversation has moved.");
      }
      onConversationSelected(conversation);
    } catch {
      if (!active.current) return;
      setOpenError("Could not open this review. Reload the project and try again.");
    } finally {
      if (active.current) setOpening(null);
    }
  }

  return <section className="project-panel tax-project" aria-labelledby={headingId}>
    <header><div><span>Saved locally in this project</span><h2 id={headingId}>Tax reviews</h2></div>
      <small>{current.items.length || ""}</small></header>
    {current.error ? <div className="tax-project-error" role="alert">
      <p>Could not load the saved tax reviews.</p>
      <button type="button" onClick={() => setVersion((value) => value + 1)}>Try again</button>
    </div> : <div className="tax-project-cases">{current.items.map((taxCase) => {
      const facts = taxCase.facts.filter(({ status }) => status !== "rejected");
      const review = taxCase.review;
      return <article className="tax-project-case" key={taxCase.id} aria-label={`${taxCase.tax_year} tax review`}>
        <header><div><h3>{taxCase.tax_year} · US federal</h3><p>{taxCase.goal}</p></div>
          <button type="button" onClick={() => void openCase(taxCase)} disabled={opening !== null}>
            {opening === taxCase.id ? "Opening…" : "Open review conversation"}
          </button></header>
        {facts.length ? <div className="tax-project-table-scroll"><table aria-label={`${taxCase.tax_year} saved tax facts`}>
          <thead><tr><th scope="col">Fact</th><th scope="col">Value</th><th scope="col">Status</th></tr></thead>
          <tbody>{facts.map((fact) => <tr key={fact.id}>
            <th scope="row">{fact.label}</th><td>{factValue(fact.value)}</td>
            <td><span data-pending={fact.status === "proposed" ? "true" : undefined}>{fact.status === "confirmed" ? "Saved" : "Needs review"}</span></td>
          </tr>)}</tbody>
        </table></div> : <p className="project-panel-copy">Facts you share with Audrey will appear here.</p>}
        {review ? <details className="tax-project-worksheet">
          <summary><span>Saved worksheet</span><small>{review.stale ? "Needs updating" : review.status === "needs_input" ? "Your input is needed" : review.status === "limited_evidence" ? "Evidence is limited" : "Saved"}</small></summary>
          <div><p>{review.summary}</p>
            {review.findings.map((finding, index) => <section key={index}><h4>{finding.label}</h4><p>{finding.detail}</p>
              {finding.calculation_ids.map((id) => taxCase.calculations.find((calculation) => calculation.id === id))
                .filter((calculation) => calculation !== undefined).map((calculation) => <p className="tax-project-calculation" key={calculation.id}>
                  {calculation.label}: {calculation.expression} = {calculation.result}{calculation.status === "needs_recheck" ? " · Needs rechecking" : ""}
                </p>)}
              {finding.evidence_ids.map((id) => taxCase.evidence.find((evidence) => evidence.id === id))
                .filter((evidence) => evidence !== undefined).map((evidence) => <details className="tax-project-source" key={evidence.id}>
                  <summary>IRS source · Page {evidence.page}{evidence.section ? ` · ${evidence.section}` : ""}</summary>
                  <blockquote>{evidence.passage}</blockquote>
                </details>)}
            </section>)}
            <p className="project-panel-copy">{review.document_scope ?? "Selected passages only; this is not a complete return review."}</p>
            {review.next_question ? <p>{review.next_question}</p> : null}
            {review.limitations.length ? <ul className="project-panel-copy">{review.limitations.map((limitation, index) => <li key={index}>{limitation}</li>)}</ul> : null}
          </div>
        </details> : <p className="project-panel-copy">Worksheets are saved here as Audrey completes them.</p>}
      </article>;
    })}</div>}
    {openError ? <p className="tax-project-error" role="alert">{openError}</p> : null}
  </section>;
}

function factValue(value: unknown): string {
  if (value === null || value === undefined || value === "") return "Not provided";
  if (typeof value === "boolean") return value ? "Yes" : "No";
  if (Array.isArray(value)) return value.map(factValue).join("; ");
  if (typeof value === "object") return Object.entries(value).map(([key, item]) => `${key.replaceAll("_", " ")}: ${factValue(item)}`).join("; ");
  return String(value);
}
