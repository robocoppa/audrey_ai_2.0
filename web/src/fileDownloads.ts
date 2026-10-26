import { apiResponse } from "./api";

export function nativeFileId(path: string | undefined): string | null {
  return path ? /^\/api\/files\/([a-zA-Z0-9_-]+)\/download$/u.exec(path)?.[1] ?? null : null;
}

export async function downloadOwnedFile(path: string, signal?: AbortSignal): Promise<void> {
  if (!nativeFileId(path)) throw new Error("Invalid file download path.");
  const response = await apiResponse(path, { signal });
  const disposition = response.headers.get("Content-Disposition") ?? "";
  if (!/^attachment(?:;|$)/iu.test(disposition)) throw new Error("Missing attachment");
  const encoded = /filename\*=UTF-8''([^;]+)/iu.exec(disposition)?.[1];
  let filename = /filename="([^"]+)"/iu.exec(disposition)?.[1] ?? "download";
  if (encoded) { try { filename = decodeURIComponent(encoded); } catch { /* Use plain name. */ } }
  filename = filename.split(/[\\/]/u).pop()!.replace(/\p{Cc}/gu, "") || "download";
  const blob = await response.blob();
  if (signal?.aborted) return;
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.append(anchor);
  try { anchor.click(); } finally {
    anchor.remove();
    window.setTimeout(() => URL.revokeObjectURL(url), 0);
  }
}
