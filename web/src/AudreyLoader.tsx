import autoPortrait from "./assets/models/audrey2.png";

export function AudreyLoader({
  fullscreen = false,
  label = "Loading Audrey",
}: {
  fullscreen?: boolean;
  label?: string;
}) {
  return (
    <div
      className={fullscreen ? "audrey-loader audrey-loader-fullscreen" : "audrey-loader"}
      role="status"
      aria-label={label}
    >
      <div className="audrey-loading-portrait" aria-hidden="true">
        <span className="audrey-loading-orbit" />
        <img src={autoPortrait} alt="" />
      </div>
    </div>
  );
}
