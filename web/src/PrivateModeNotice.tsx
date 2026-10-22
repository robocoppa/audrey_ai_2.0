import "./PrivateModeNotice.css";

export function PrivateModeNotice({ lockReason }: { lockReason: string }) {
  return <aside className="private-mode-notice" aria-label="Private mode privacy">
    <p>Local AI processing · Conversations and files are saved on this server.</p>
    <details>
      <summary>Privacy details</summary>
      <p>Private uses local models and tools. It is not device-only or anonymous: trusted server administrators may access stored data, and your browser connection passes through Cloudflare.</p>
      <p>Documents and Private projects only; web, saved memory and media are unavailable.</p>
    </details>
    {lockReason ? <p className="private-mode-lock">{lockReason}</p> : null}
  </aside>;
}
