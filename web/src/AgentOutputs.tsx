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
  return <section className="agents-outputs" aria-labelledby="agent-files-title">
    <h3 id="agent-files-title">Files</h3>
    <p>Saved in My Files. Created files stay there if you delete this task.</p>
    <ul>{files.map((output) => <AgentOutputDownload key={output.file_id} output={output} />)}</ul>
  </section>;
}

function AgentOutputDownload({ output }: { output: AgentOutput }) {
  const request = useRef<AbortController | null>(null);
  const [downloading, setDownloading] = useState(false);
  const [error, setError] = useState("");
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
      aria-label={`${downloading ? "Downloading" : "Download"} ${output.filename}`} onClick={() => void download()}>
      <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 3v12m-5-5 5 5 5-5M4 15v5h16v-5" /></svg>
      <span>{output.filename}</span><span className="agents-output-action">{downloading ? "Downloading…" : "Download"}</span>
    </button>
    {error ? <p className="agents-output-error" role="alert">{error}</p> : null}
  </li>;
}
