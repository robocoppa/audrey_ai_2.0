import { useEffect, useRef, useState } from "react";

import { ApiError, isAgentOutput, type AgentOutput } from "./api";
import { downloadOwnedFile } from "./fileDownloads";

export function AgentOutputs({ outputs }: { outputs?: AgentOutput[] }) {
  const seen = new Set<string>();
  // The API validates metadata too; keep navigation/fixture payloads behind the same URL fence.
  const files = Array.isArray(outputs) ? outputs.filter((output) => {
    if (!isAgentOutput(output) || seen.has(output.file_id) || seen.size >= 8) return false;
    seen.add(output.file_id);
    return true;
  }) : [];
  if (!files.length) return null;
  const current = files.filter((output) => output.current === true);
  const previous = files.filter((output) => output.current === false);
  const unclassified = files.filter((output) => output.current == null);
  const repeatedNames = new Set(files.filter((output, index) => files.some((other, otherIndex) =>
    index !== otherIndex && other.filename === output.filename)).map((output) => output.filename));
  function renderFile(output: AgentOutput) {
    return <AgentOutputDownload key={output.file_id} output={output} distinguish={repeatedNames.has(output.filename)} />;
  }
  return <section className="agents-outputs" aria-labelledby="agent-files-title">
    <h3 id="agent-files-title">Files</h3>
    <p>Saved in My Files. Created files stay there if you delete this task.</p>
    {current.length ? <ul aria-label="Current files">{current.map(renderFile)}</ul> : null}
    {unclassified.length ? <div className="agents-output-unclassified">
      <p>Version information is unavailable for these saved files.</p>
      <ul aria-label="Saved files">{unclassified.map(renderFile)}</ul>
    </div> : null}
    {!current.length && !unclassified.length ? <p>No current file is available. Previous files are below.</p> : null}
    {previous.length ? <details className="agents-output-history">
      <summary>Previous files <span>({previous.length})</span></summary>
      <ul aria-label="Previous files">{previous.map(renderFile)}</ul>
    </details> : null}
  </section>;
}

function AgentOutputDownload({ output, distinguish }: { output: AgentOutput; distinguish: boolean }) {
  const request = useRef<AbortController | null>(null);
  const [downloading, setDownloading] = useState(false);
  const [error, setError] = useState("");
  const version = output.current === true ? "Current" : output.current === false ? "Previous" : "Saved";
  const savedAt = new Date(output.created_at).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "medium" });
  const downloadLabel = `${downloading ? "Downloading" : "Download"} ${output.filename}`
    + (distinguish ? `, ${version.toLowerCase()} file, saved ${savedAt}, file ${output.file_id}` : "");
  useEffect(() => () => request.current?.abort(), []);
  async function download() {
    if (request.current && !request.current.signal.aborted) return;
    const controller = new AbortController();
    request.current = controller;
    setDownloading(true); setError("");
    try { await downloadOwnedFile(output.download_url, controller.signal); }
    catch (reason) {
      if (!controller.signal.aborted) setError(reason instanceof ApiError && reason.status === 404
        ? "This file is no longer available. Check My Files."
        : "Could not download this file. Try again.");
    } finally {
      if (request.current === controller) {
        request.current = null;
        if (!controller.signal.aborted) setDownloading(false);
      }
    }
  }
  return <li>
    <button className="agents-secondary agents-output-download" type="button" disabled={downloading}
      aria-label={downloadLabel} onClick={() => void download()}>
      <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 3v12m-5-5 5 5 5-5M4 15v5h16v-5" /></svg>
      <span className="agents-output-name">{output.filename}<span className="agents-output-version">
        <span className={output.current === true ? "agents-output-current" : undefined}>{version}</span>
        <span aria-hidden="true"> · </span><time dateTime={output.created_at}>{savedAt}</time>
      </span></span>
      <span className="agents-output-action">{downloading ? "Downloading…" : "Download"}</span>
    </button>
    {error ? <p className="agents-output-error" role="alert">{error}</p> : null}
  </li>;
}
