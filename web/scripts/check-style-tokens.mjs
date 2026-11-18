import { readdir, readFile } from "node:fs/promises";
import path from "node:path";
import process from "node:process";
import { fileURLToPath, URL } from "node:url";

const sourceRoot = fileURLToPath(new URL("../src/", import.meta.url));

// Keep offsets/newlines intact so findings point at the original source line.
function withoutComments(source) {
  return source.replace(/\/\*[\s\S]*?\*\//gu, (comment) => comment.replace(/[^\n]/gu, " "));
}

function* declarationValues(text) {
  let start = 0, parentheses = 0, quote = "", escaped = false;
  for (let index = 0; index < text.length; index += 1) {
    const character = text[index];
    if (quote) {
      if (escaped) escaped = false;
      else if (character === "\\") escaped = true;
      else if (character === quote) quote = "";
      continue;
    }
    if (character === '"' || character === "'") { quote = character; continue; }
    if (character === "(") parentheses += 1;
    else if (character === ")") parentheses -= 1;
    else if (!parentheses && character === "{") start = index + 1;
    else if (!parentheses && (character === ";" || character === "}")) {
      const declaration = text.slice(start, index);
      const match = /^\s*[\w-]+\s*:\s*/u.exec(declaration);
      if (match) {
        const value = declaration.slice(match[0].length).replace(/(["'])(?:\\.|(?!\1)[^\\])*?\1/gu, (value) => " ".repeat(value.length));
        yield { value, offset: start + match[0].length };
      }
      start = index + 1;
    }
  }
}

export function checkStyleSources(sources) {
  const defined = new Set();
  for (const [name, source] of sources) {
    const text = name.endsWith(".css") ? withoutComments(source) : source;
    const patterns = name.endsWith(".css")
      ? [/(--[\w-]+)\s*:/gu, /@property\s+(--[\w-]+)/gu]
      : [/["'](--[\w-]+)["']\s*:/gu, /\.setProperty\(\s*["'](--[\w-]+)["']/gu];
    for (const pattern of patterns) for (const match of text.matchAll(pattern)) defined.add(match[1]);
  }
  const issues = [];
  for (const [name, source] of sources) {
    if (!name.endsWith(".css")) continue;
    const text = withoutComments(source);
    const report = (index, message) => issues.push(`${name}:${text.slice(0, index).split("\n").length}: ${message}`);
    for (const { value, offset } of declarationValues(text)) {
      for (const match of value.matchAll(/var\(\s*(--[\w-]+)/gu)) {
        if (!defined.has(match[1])) report(offset + match.index, `Undefined style variable ${match[1]}.`);
      }
      if (path.basename(name) === "theme.css") continue;
      // Include colors inside gradients/shadows, excluding selectors/quoted text.
      // Semantic keywords (currentColor, Canvas, etc.) remain valid.
      for (const match of value.matchAll(/#[\da-f]{3,8}\b|\b(?:rgba?|hsla?|hwb|lab|lch|oklab|oklch)\(\s*(?!var\b|from\b)[\d.+-]|\bcolor\(\s*(?!from\b)[\w-]+\s+[\d.+-]/giu)) {
        report(offset + match.index, "Use a semantic theme token instead of a literal color.");
      }
    }
  }
  return issues;
}

async function readSources(directory) {
  const entries = await readdir(directory, { withFileTypes: true });
  const sources = [];
  for (const entry of entries) {
    const filename = path.join(directory, entry.name);
    if (entry.isDirectory()) sources.push(...await readSources(filename));
    else if (/\.(?:css|tsx?)$/u.test(entry.name) && !/\.(?:test|spec)\./u.test(entry.name)) {
      sources.push([path.relative(sourceRoot, filename), await readFile(filename, "utf8")]);
    }
  }
  return sources;
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const sources = await readSources(sourceRoot);
  const issues = checkStyleSources(sources);
  if (issues.length) {
    process.stderr.write(`${issues.join("\n")}\n`);
    process.exitCode = 1;
  } else {
    process.stdout.write(`PASS style tokens: ${sources.filter(([name]) => name.endsWith(".css")).length} stylesheets; centralized colors and defined variables.\n`);
  }
}
