import autoPortrait from "./assets/models/audrey2.png";

export function AudreyLoader({
  fullscreen = false,
  label = "Loading Audrey",
  showPortrait = true,
  message,
  detail,
}: {
  fullscreen?: boolean;
  label?: string;
  showPortrait?: boolean;
  message?: string;
  detail?: string;
}) {
  return (
    <div
      className={fullscreen ? "audrey-loader audrey-loader-fullscreen" : "audrey-loader"}
      role="status"
      aria-label={label}
    >
      <div className="audrey-loader-stack">
        {showPortrait ? (
          <div className="audrey-loading-portrait" aria-hidden="true">
            <span className="audrey-loading-orbit" />
            <img src={autoPortrait} alt="" />
          </div>
        ) : null}
        {message ? (
          <div className="audrey-loader-copy">
            <strong>{message}</strong>
            {detail ? <span>{detail}</span> : null}
          </div>
        ) : null}
      </div>
    </div>
  );
}
