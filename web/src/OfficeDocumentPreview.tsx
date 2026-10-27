import type { ReactNode } from "react";

import "./OfficeDocumentPreview.css";

export interface OfficeTextRun {
  text: string;
  bold?: boolean;
  italic?: boolean;
  code?: boolean;
}

interface OfficeParagraphBlock {
  kind: "paragraph" | "heading" | "code" | "quote";
  runs: OfficeTextRun[];
  level?: 1 | 2 | 3;
  list?: "bullet" | "number";
  list_start?: number;
}

interface OfficeTableBlock {
  kind: "table";
  rows: { runs: OfficeTextRun[] }[][];
  header?: boolean;
}

export type OfficeDocumentBlock = OfficeParagraphBlock | OfficeTableBlock;

function InlineRuns({ runs }: { runs: OfficeTextRun[] }) {
  return runs.map((run, index) => {
    let content: ReactNode = run.text;
    if (run.code) content = <code>{content}</code>;
    if (run.italic) content = <em>{content}</em>;
    if (run.bold) content = <strong>{content}</strong>;
    return <span key={index}>{content}</span>;
  });
}

function Block({ block }: { block: OfficeDocumentBlock }) {
  if (block.kind === "table") {
    return <div className="office-preview-table-wrap"><table aria-label="Document table">
      <tbody>{block.rows.map((row, index) => <tr key={index}>{row.map((cell, column) => (
        block.header && index === 0 ? <th key={column} scope="col"><InlineRuns runs={cell.runs} /></th>
          : <td key={column}><InlineRuns runs={cell.runs} /></td>
      ))}</tr>)}</tbody>
    </table></div>;
  }
  const content = <InlineRuns runs={block.runs} />;
  if (block.kind === "heading") {
    if (block.level === 1) return <h1>{content}</h1>;
    if (block.level === 2) return <h2>{content}</h2>;
    return <h3>{content}</h3>;
  }
  if (block.kind === "code") return <pre className="office-preview-code"><code>{block.runs.map((run) => run.text).join("")}</code></pre>;
  if (block.kind === "quote") return <blockquote>{content}</blockquote>;
  return <p>{block.runs.length ? content : "\u00a0"}</p>;
}

export function OfficeDocumentPreview({ blocks, paragraphs }: {
  blocks?: OfficeDocumentBlock[];
  paragraphs?: string[];
}) {
  const contents: ReactNode[] = [];
  if (blocks?.length) {
    for (let index = 0; index < blocks.length; index++) {
      const block = blocks[index];
      if (block.kind === "paragraph" && block.list) {
        const entries: OfficeParagraphBlock[] = [block];
        while (index + 1 < blocks.length) {
          const next = blocks[index + 1];
          if (next.kind !== "paragraph" || next.list !== block.list) break;
          entries.push(next);
          index++;
        }
        const items = entries.map((entry, item) => <li key={item} value={block.list === "number" ? entry.list_start : undefined}><InlineRuns runs={entry.runs} /></li>);
        contents.push(block.list === "number" ? <ol key={index} start={block.list_start}>{items}</ol>
          : <ul key={index}>{items}</ul>);
      } else contents.push(<Block key={index} block={block} />);
    }
  } else {
    for (const [index, paragraph] of (paragraphs ?? []).entries()) {
      contents.push(<p key={index}>{paragraph || "\u00a0"}</p>);
    }
  }
  return <article className="file-preview-document office-document-preview" tabIndex={0} aria-label="Document content">
    {contents.length ? contents : <p>This document is empty.</p>}
  </article>;
}
