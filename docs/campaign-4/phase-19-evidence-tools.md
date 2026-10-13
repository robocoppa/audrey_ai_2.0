# Phase 19 — calculations and deeper evidence tools

Approved October 13, 2026. English requests only. These tools extend ordinary
Audrey chat and use the existing Tools and Sources summaries.

## Build order

| Slice | Capability | Acceptance |
|---|---|---|
| 19A | Calculator and unit conversion | Exact decimal totals, conversions, bounded linear/quadratic equations; invalid expressions fail clearly |
| 19B | Read-only CSV and Excel analysis | Inspect sheets/columns, filter rows, aggregate/group values, find duplicates; no partial totals |
| 19C | Inspect a PDF page | Render the requested page and describe visible charts, labels, diagrams and layout |
| 19D | Open a previous conversation | Search first, then read the owned saved exchange with explicit paging |
| 19E | Inspect a video moment | Describe a frame at the requested timestamp, with its position and evidence limits |

## Contracts

- Discovery and dispatch declare every new capability together. File/history
  reads bind the authenticated owner; callers cannot choose another user's files.
- Calculations accept a bounded mathematical expression or a typed conversion,
  never executable code. Equation solving initially covers real linear and
  quadratic equations; other equations receive an explicit unsupported result.
- Tables accept owned CSV/XLSX originals, with bounded parsing and explicit
  sheet/row coverage. CSV already uploads; XLSX gains readable ingestion.
  Default inspection chooses the first worksheet with cells and names it;
  header/result pages expose continuation offsets. Intake indexes an explicitly
  labeled first-usable-sheet preview. Formula cells use stored values, and
  missing cached values stay unknown.
  Reading and analysis do not create or edit spreadsheets.
- PDF pages and video frames are inspected through the existing vision-model
  pool and fairness gate. Their descriptions are visual observations for the
  answering model, with page/timestamp attribution. A still frame does not
  establish motion or speech.
- Missing/reclaimed originals produce a clear unavailable result. No automatic
  video re-download, source deletion, or changed retention policy.
- Previous conversations come from canonical saved user/assistant messages.
  Foreign/deleted conversations remain unavailable; paging names unread content.
- Tool calls use existing run events. Sources identify the file and inspected
  page/frame; calculation arguments and private result bodies are not copied
  into browser activity payloads.
- Vision tools release a local answer worker's GPU gate while dispatching,
  with a bounded timeout ladder covering rendering and vision generation.
- Existing Hermes adapters, passthrough model settings and authentication stay
  outside this feature. Users ask normal questions; no new picker is needed.

## Verification and deployment

Hermetic laptop checks cover numerical correctness, table rows/aggregates,
owner isolation, pagination, renderer failures and timeout/cancellation cleanup,
and tool discovery/dispatch/provenance. Fake vision responses prove plumbing;
the browser check proves deployed perception and answer quality.

After pull on Tower, rebuild `audrey` and `custom-tools`. The UI and media
workers need no rebuild for these tool additions. Manual acceptance uses a real
CSV/XLSX, a PDF with a diagram, an existing saved chat, and a retained video.
A calculator request needs no upload. The user confirms visible tool calls and
answers before live acceptance is marked passed.

Status: all five tools built; focused hermetic checks, real local rendering,
scoped lint and locked dependencies pass. Tower/browser acceptance pending.
