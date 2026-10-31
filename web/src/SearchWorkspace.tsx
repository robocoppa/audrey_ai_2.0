import { useCallback, useEffect, useRef, useState, type FormEvent } from "react";

import {
  listProjects,
  searchContent,
  type AudreyProject,
  type SearchArchiveFilter,
  type SearchKind,
  type SearchOptions,
  type SearchResponse,
  type SearchResult,
} from "./api";

import "./SearchWorkspace.css";

const KINDS: ReadonlyArray<{ value: SearchKind; label: string }> = [
  { value: "all", label: "All" },
  { value: "message", label: "Chats" },
  { value: "file", label: "Files" },
  { value: "agent", label: "Agents" },
  { value: "knowledge", label: "Knowledge" },
];
const MAX_RESULTS = 200;

interface SearchRequest {
  query: string;
  options: SearchOptions;
  append: boolean;
  serial: number;
}

export function SearchWorkspace({ visible = true, presentation = "page", openRequest, onOpenResult }: {
  visible?: boolean;
  presentation?: "page" | "sidebar";
  openRequest?: { query: string; serial: number };
  onOpenResult: (result: SearchResult) => void;
}) {
  const [query, setQuery] = useState("");
  const [kind, setKind] = useState<SearchKind>("all");
  const [projectId, setProjectId] = useState("");
  const [archived, setArchived] = useState<SearchArchiveFilter>("all");
  const [since, setSince] = useState("");
  const [until, setUntil] = useState("");
  const [dateError, setDateError] = useState("");
  const [projects, setProjects] = useState<AudreyProject[]>([]);
  const [projectNotice, setProjectNotice] = useState("");
  const [request, setRequest] = useState<SearchRequest | null>(null);
  const [response, setResponse] = useState<SearchResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const serial = useRef(0);
  const requestController = useRef<AbortController | null>(null);
  const queryInput = useRef<HTMLInputElement | null>(null);
  const handledOpenRequest = useRef<number | null>(null);
  const incomingQuery = openRequest?.query.trim() ?? "";
  const incomingSearchSerial = incomingQuery ? openRequest?.serial : undefined;

  useEffect(() => {
    let current = true;
    async function loadProjects() {
      const found = new Map<string, AudreyProject>();
      const seenCursors = new Set<string>();
      let cursor: string | undefined;
      try {
        for (let page = 0; page < 5; page += 1) {
          const result = await listProjects(cursor);
          if (!current) return;
          result.items.forEach((project) => found.set(project.id, project));
          setProjects([...found.values()].slice(0, 500));
          if (!result.next_cursor) return;
          if (found.size >= 500 || seenCursors.has(result.next_cursor)) break;
          seenCursors.add(result.next_cursor);
          cursor = result.next_cursor;
        }
        if (current) setProjectNotice("Showing the first 500 projects. All projects remain searchable.");
      } catch {
        if (current) setProjectNotice("Project filters are unavailable. You can still search all your content.");
      }
    }
    void loadProjects();
    return () => { current = false; };
  }, []);

  useEffect(() => {
    if (!visible || !request || (incomingSearchSerial !== undefined && handledOpenRequest.current !== incomingSearchSerial)) return;
    let current = true;
    const controller = new AbortController();
    requestController.current = controller;
    queueMicrotask(() => { if (current) setLoading(true); });
    async function search() {
      try {
        const next = await searchContent(request!.query, request!.options, controller.signal);
        if (!current || controller.signal.aborted || request!.serial !== serial.current) return;
        setResponse((saved) => request!.append && saved ? {
          ...next,
          items: [...new Map([...saved.items, ...next.items].map((item) => [item.id, item])).values()].slice(0, MAX_RESULTS),
        } : { ...next, items: next.items.slice(0, MAX_RESULTS) });
        setError("");
      } catch (reason) {
        if (!current || controller.signal.aborted || request!.serial !== serial.current) return;
        setError(reason instanceof Error ? reason.message : "Search could not be completed. Please try again.");
      } finally {
        if (current && !controller.signal.aborted && request!.serial === serial.current) setLoading(false);
      }
    }
    void search();
    return () => {
      current = false;
      controller.abort();
      if (requestController.current === controller) requestController.current = null;
    };
  }, [request, visible, incomingQuery, incomingSearchSerial]);

  const begin = useCallback((queryText: string, options: SearchOptions, append = false) => {
    requestController.current?.abort();
    serial.current += 1;
    setError("");
    setLoading(true);
    if (!append) setResponse(null);
    setRequest({ query: queryText, options, append, serial: serial.current });
  }, []);

  const validDates = useCallback((options: SearchOptions) => {
    if (options.since && options.until && options.since > options.until) {
      setDateError("The start date must be on or before the end date.");
      return false;
    }
    setDateError("");
    return true;
  }, []);

  useEffect(() => {
    if (!visible || !openRequest || handledOpenRequest.current === openRequest.serial) return;
    let current = true;
    // Wait until the workspace is visible before focusing or starting a header query.
    queueMicrotask(() => {
      if (!current) return;
      handledOpenRequest.current = openRequest.serial;
      queryInput.current?.focus();
      const text = openRequest.query.trim();
      if (!text) return;
      setQuery(text);
      const options = { kind, project_id: projectId || undefined, archived, since: since || undefined, until: until || undefined };
      if (validDates(options)) begin(text, options);
    });
    return () => { current = false; };
  }, [visible, openRequest, kind, projectId, archived, since, until, begin, validDates]);

  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!query.trim()) return;
    const options = currentOptions();
    if (validDates(options)) begin(query.trim(), options);
  }

  function filter(patch: SearchOptions) {
    const options = { ...currentOptions(), ...patch, cursor: undefined };
    if (validDates(options) && request) begin(request.query, options);
  }

  function currentOptions(): SearchOptions {
    return { kind, project_id: projectId || undefined, archived, since: since || undefined, until: until || undefined };
  }

  function retry() {
    if (request) begin(request.query, request.options, request.append);
  }

  const results = response?.items ?? [];
  const atLimit = results.length >= MAX_RESULTS;
  const coverageWarnings = [...new Set(response?.warnings.map((warning) => warning.trim()).filter(Boolean) ?? [])];
  if (response?.partial && !response.indexing && !coverageWarnings.length) coverageWarnings.push("Some saved content is not included in these results.");

  return (
    <section className={`search-workspace${presentation === "sidebar" ? " search-workspace-sidebar" : ""}`} aria-labelledby={presentation === "page" ? "search-title" : undefined} aria-label={presentation === "sidebar" ? "Search your saved work" : undefined}>
      {presentation === "page" ? <header className="search-header">
        <p className="search-eyebrow">Find your saved work</p>
        <h1 id="search-title">Search</h1>
        <p>Find words in your chats, files, and saved agent work. Search stays local.</p>
      </header> : null}

      <form className="search-form" role="search" onSubmit={submit}>
        <label htmlFor="content-search-query">Search your content</label>
        <div className="search-input-row">
          <input
            id="content-search-query"
            ref={queryInput}
            type="search"
            value={query}
            maxLength={256}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Find a phrase, decision, or detail…"
            autoComplete="off"
            aria-describedby="content-search-help"
          />
          <button className="search-primary" type="submit" disabled={!query.trim()}>Search</button>
        </div>
        <p id="content-search-help">Search matches saved text. Submit a query to see results.</p>
      </form>

      <div className="search-filter-bar">
        <div className="search-kind-filters" role="group" aria-label="Search source">
          {KINDS.map(({ value, label }) => (
            <button key={value} type="button" aria-pressed={kind === value} onClick={() => { setKind(value); filter({ kind: value }); }}>{label}</button>
          ))}
        </div>
        <div className="search-select-filters">
          <label>
            <span>Project</span>
            <select value={projectId} onChange={(event) => { setProjectId(event.target.value); filter({ project_id: event.target.value || undefined }); }}>
              <option value="">All projects</option>
              {projects.map((project) => <option key={project.id} value={project.id}>{project.name}{project.private_required ? " · Private" : ""}</option>)}
            </select>
          </label>
          <label>
            <span>Chat archive</span>
            <select value={archived} onChange={(event) => { const value = event.target.value as SearchArchiveFilter; setArchived(value); filter({ archived: value }); }}>
              <option value="all">All chats</option>
              <option value="active">Active chats</option>
              <option value="archived">Archived chats</option>
            </select>
          </label>
        </div>
      </div>
      <fieldset className="search-date-filters" aria-describedby="search-date-help">
        <legend>Search dates</legend>
        <div>
          <label><span>Updated from</span><input type="date" value={since} onChange={(event) => { setSince(event.target.value); filter({ since: event.target.value || undefined }); }} /></label>
          <label><span>Updated through</span><input type="date" value={until} onChange={(event) => { setUntil(event.target.value); filter({ until: event.target.value || undefined }); }} /></label>
        </div>
        <p id="search-date-help">Activity dates use UTC. Both dates are included.</p>
        {dateError ? <p className="search-date-error" role="alert">{dateError}</p> : null}
      </fieldset>
      {projectNotice ? <p className="search-notice">{projectNotice}</p> : null}

      <section className="search-results" aria-label="Search results" aria-busy={loading}>
        {request ? <h2>Results for “{request.query}”</h2> : null}
        {error ? <div className="search-error" role="alert"><p>{error}</p><button type="button" className="search-secondary" onClick={retry}>Try again</button></div> : null}
        {loading ? <p className="search-state" role="status">{request?.append ? "Loading more results…" : "Searching your saved content…"}</p> : null}
        {response?.indexing ? <p className="search-indexing" role="status">Indexing saved content. Search again for more matches.</p> : null}
        {coverageWarnings.length === 1 ? <p className="search-coverage-note">{coverageWarnings[0]}</p> : null}
        {coverageWarnings.length > 1 ? <details className="search-coverage"><summary>Search coverage: {coverageWarnings.length} limitations</summary><ul>{coverageWarnings.map((warning) => <li key={warning}>{warning}</li>)}</ul></details> : null}
        {!request ? <div className="search-welcome"><span aria-hidden="true">⌕</span><h2>Your work, easier to find</h2><p>Search the contents of saved messages, file text, and agent reports.</p></div> : null}
        {response && !results.length && !loading && !error ? <p className="search-state">No matches found. Try fewer words or broader filters.</p> : null}
        {results.length ? <ul className="search-result-list">
          {results.map((result) => <li key={result.id}>
            <article className="search-result-card">
              <div className="search-result-meta">
                <span className={`search-source-badge search-source-${result.kind}`}>{result.kind === "message" ? "Chat" : result.kind === "file" ? "File" : result.kind === "knowledge" ? "Knowledge" : "Agent"}</span>
                {result.private_required ? <span className="search-private">Private</span> : null}
                {result.target.kind === "file" ? <span>{artifactLabel(result.target.artifact)}</span> : null}
                {result.kind === "agent" ? <span>Generated report</span> : null}
                {result.target.kind === "knowledge" && "stale" in result.target && result.target.stale ? <span>Source needs review</span> : null}
                {result.target.kind === "agent" && result.target.step_id ? <span>Saved {result.target.step_id}</span> : null}
                {result.availability === "text_only" ? <span>Saved text · original unavailable</span> : null}
                <time dateTime={result.updated_at}>{timestamp(result.updated_at)}</time>
              </div>
              <h3>{result.title || "Untitled"}</h3>
              <p className="search-excerpt"><HighlightedExcerpt text={result.excerpt} ranges={result.highlights} /></p>
              <button type="button" className="search-open" onClick={() => onOpenResult(result)} aria-label={`Open ${result.kind === "message" ? "chat" : result.kind}: ${result.title || "Untitled"}`}>Open {result.kind === "message" ? "message" : result.kind === "file" ? "saved text" : result.kind === "knowledge" ? "knowledge" : "agent"}<span aria-hidden="true"> ↗</span></button>
            </article>
          </li>)}
        </ul> : null}
        {response?.next_cursor && !atLimit ? <div className="search-pagination"><button type="button" className="search-secondary" disabled={loading} onClick={() => { if (request) begin(request.query, { ...request.options, cursor: response.next_cursor! }, true); }}>Load more results</button></div> : null}
        {atLimit && response?.next_cursor ? <p className="search-state">Showing the first {MAX_RESULTS} matches. Narrow your query or filters to find more.</p> : null}
      </section>
    </section>
  );
}

function HighlightedExcerpt({ text, ranges }: { text: string; ranges: SearchResult["highlights"] }) {
  // Backend offsets count Unicode code points, including supplementary characters.
  const characters = Array.from(text);
  const valid = ranges.filter(({ start, end }) => Number.isInteger(start) && Number.isInteger(end) && start >= 0 && start < end && end <= characters.length).sort((left, right) => left.start - right.start);
  const merged: Array<{ start: number; end: number }> = [];
  for (const range of valid) {
    const previous = merged.at(-1);
    if (previous && range.start <= previous.end) previous.end = Math.max(previous.end, range.end);
    else merged.push({ ...range });
  }
  let offset = 0;
  const parts = merged.flatMap(({ start, end }) => {
    const before = characters.slice(offset, start).join("");
    offset = end;
    return [before, <mark key={`${start}:${end}`}>{characters.slice(start, end).join("")}</mark>];
  });
  return <>{parts}{characters.slice(offset).join("")}</>;
}

function artifactLabel(artifact: "text" | "transcript" | "visual" | "summary") {
  return artifact === "transcript" ? "Transcript" : artifact === "visual" ? "Generated visual notes" : artifact === "summary" ? "Generated summary" : "Saved text";
}

function timestamp(value: string) {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? "" : date.toLocaleString(undefined, { month: "short", day: "numeric", year: "numeric" });
}
