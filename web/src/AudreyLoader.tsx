import type { CSSProperties } from "react";

import autoPortrait from "./assets/models/audrey2.png";

export function AudreyLoader({
  fullscreen = false,
  label = "Loading",
  showPortrait = true,
  message,
  detail,
  rimPassSeconds = 2,
}: {
  fullscreen?: boolean;
  label?: string;
  showPortrait?: boolean;
  message?: string;
  detail?: string;
  rimPassSeconds?: number;
}) {
  return (
    <div
      className={fullscreen ? "audrey-loader audrey-loader-fullscreen" : "audrey-loader"}
      role="status"
      aria-label={label}
      style={{
        "--audrey-rim-pass": `${rimPassSeconds}s`,
        "--audrey-rim-cycle": `${rimPassSeconds * 2}s`,
      } as CSSProperties}
    >
      <div className="audrey-loader-stack">
        {showPortrait ? (
          <div className="audrey-loader-portrait-stack">
            <div className="audrey-loading-portrait" aria-hidden="true">
              <svg className="audrey-loading-rim" viewBox="0 0 100 100" focusable="false">
                <circle
                  className="audrey-loading-rim-fill"
                  cx="50"
                  cy="50"
                  r="50"
                  pathLength="100"
                />
              </svg>
              <span className="audrey-loading-orbit" />
              <div className="audrey-loading-photo">
                <img src={autoPortrait} alt="" />
              </div>
            </div>
            {!message ? (
              <span className="audrey-loading-label" aria-hidden="true">
                Loading
                <span className="audrey-loading-dots">
                  <span>.</span>
                  <span className="audrey-loading-dot dot-two">.</span>
                  <span className="audrey-loading-dot dot-three">.</span>
                </span>
              </span>
            ) : null}
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
