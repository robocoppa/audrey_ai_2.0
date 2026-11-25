type Excerpt = { start: number; end: number; text: string };

function isExcerpt(value: unknown): value is Excerpt {
  if (typeof value !== "object" || value === null) return false;
  const { start, end, text } = value as Partial<Excerpt>;
  return typeof start === "number" && Number.isSafeInteger(start) && start >= 0
    && typeof end === "number" && Number.isSafeInteger(end)
    && typeof text === "string" && end - start === Array.from(text).length;
}

function comparedExcerpts(diff: string) {
  let saved: { old?: unknown; new?: unknown } | null;
  try {
    saved = JSON.parse(diff);
  } catch {
    return null;
  }
  if (!isExcerpt(saved?.old) || !isExcerpt(saved.new) || saved.old.start !== saved.new.start) return null;

  // The saved offsets count Unicode characters, not JavaScript string units.
  const before = Array.from(saved.old.text);
  const after = Array.from(saved.new.text);
  const sharedLength = Math.min(before.length, after.length);
  let prefix = 0;
  while (prefix < sharedLength && before[prefix] === after[prefix]) prefix++;
  if (prefix === before.length && prefix === after.length) return null;

  let suffix = 0;
  while (suffix < sharedLength - prefix && before.at(-1 - suffix) === after.at(-1 - suffix)) suffix++;
  return [before, after].map((characters) => ({
    leading: characters.slice(0, prefix).join(""),
    changed: characters.slice(prefix, characters.length - suffix).join(""),
    trailing: characters.slice(characters.length - suffix).join(""),
  }));
}

export function WatchChangeEvidence({ diff }: { diff: string }) {
  const excerpts = comparedExcerpts(diff);
  return <details className="watch-diff">
    <summary>Page changes</summary>
    {excerpts ? <div className="watch-diff-excerpts">
      {excerpts.map(({ leading, changed, trailing }, index) => {
        const label = index === 0 ? "Before" : "After";
        return <figure key={label} className="watch-diff-excerpt" data-side={label.toLowerCase()}>
          <figcaption>{label}</figcaption>
          <pre tabIndex={0} aria-label={`${label} excerpt`}>{leading}{changed && <mark>{changed}</mark>}{trailing}</pre>
        </figure>;
      })}
    </div> : <pre tabIndex={0} aria-label="Saved page changes">{diff}</pre>}
  </details>;
}
