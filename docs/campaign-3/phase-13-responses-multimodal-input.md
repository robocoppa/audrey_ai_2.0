# Campaign 3 Phase 13 — Responses inputs, output, tools, and storage

**Status:** 13B–13F complete and accepted. 13A implementation is complete; its distinct inline-image live proof remains pending.

This extends [the Responses foundation](phase-05-responses-api.md) through existing authenticated generation paths. Admission and owner/policy failures finish before provider dispatch or SSE. Claimed `user` values never select the file/storage owner.

## Slice 13A - typed text and inline image parts

Easy-input messages accept nonblank `input_text` on system/developer/user/assistant roles. User messages also accept inline base64 JPEG/PNG/WEBP `input_image` data URLs with `detail: auto|low|high|original`. Malformed/empty data, GIF/unknown image types, wrong-role images, unsupported schemes, and unknown part types are rejected.

Completed and streamed requests share the existing vision adapter. The remaining live proof is one inline-image answer plus the unsupported-`file://` admission guard; passed owned/remote image checks are not repeated.

## Slice 13B - structured text output

`text.format.type: json_schema` accepts a name, optional description/strict flag, and bounded object-root schema. The admitted schema applies only to the final answer call. Audrey validates the returned JSON before success.

Fast structured calls skip ReAct and prose escalation. Deep/research workers retain their normal gathering, while final synthesis/writing is constrained. Sources appendices and progress are excluded from structured output.

The subset includes objects/arrays/scalars, properties/required, boolean additionalProperties, enums/constants, local references/definitions, anyOf, and basic length/numeric bounds. Admission caps serialized size, depth, nodes, properties, and anyOf width. Strict objects require every declared property and `additionalProperties: false`. Unsupported keywords/non-object roots return 400.

Schema violations return completed HTTP 502 or streamed `response.failed`/`structured_output_invalid`. Truncated output is incomplete rather than a schema violation. Legacy `json_object` is unsupported.

## Slice 13C - owned file references

User input accepts `input_image.file_id` or `input_file.file_id` from the caller's Audrey uploads; these are not OpenAI-hosted Files ids. Image sources are exclusive. Owned images use bounded metadata-free JPEG previews; documents contribute extracted/OCR text quoted as user evidence.

Missing, foreign, and deleted ids share 404. Pending/failed/wrong-kind files return 422; reclaimed/missing originals return 410; unreadable content returns 409. The reader supports PDF, DOCX, HTML, Markdown, CSV, RST, and plain text. PDF page images/charts are not inspected.

| Admission budget | Per file | Whole request |
|---|---|---|
| File-reference parts | — | 10 |
| Images, including inline/history | — | Configured `vision.max_images_per_turn` (default 4) |
| Document original bytes | 20 MiB | 32 MiB |
| Extracted document characters | 100,000 | 200,000 |
| Extracted document tokens | 8,000 | 12,000 |
| Preview/inline image bytes | — | 6 MiB |
| Adapted prompt including instructions/history | — | 250,000 characters / 16,000 tokens |

Repeated references count again. Limits use `cl100k_base` for admission; usage remains provider-reported. Oversized input returns 413 without silent truncation. Native paged document viewing retains its separate reader limits.

## Slice 13D - bounded public URL inputs

User `input_image.image_url` and `input_file.file_url` accept public HTTP(S) sources, exclusive with file ids. Remote inputs are temporary: no My Files entry, ingestion job, embedding, or project reference is created. Caller credentials/cookies/headers are never sent to source hosts.

### Fetch and parser contract

- Admit only HTTP:80 or HTTPS:443. Reject embedded credentials, unsupported schemes, malformed/control/backslash URLs, scoped IPv6, and URLs over 4,096 characters.
- Resolve and vet all initial destinations before requesting. Every address must be globally routable; private/loopback/link-local/Tailscale/shared/special and transition-network addresses are blocked. DNS failure closes admission.
- Connect to a vetted numeric IP while preserving Host/TLS SNI and hostname certificate verification. Connection failures may try another vetted IP. DNS cannot silently change the connected destination.
- Follow at most three manual redirects, revalidating URL and all DNS answers each time. No HTTPS downgrade. Connections are not reused across hops.
- Ignore environment proxies; use no cookie jar or full-URL INFO logging. Signed query parameters are removed from quoted evidence/archives.
- Disable compression; reject non-identity Content-Encoding. Stream bytes under limits, including absent lengths; reject oversized declared lengths, empty bodies, and non-200 final responses.
- Sniff bytes with libmagic; declared type must match supported content (with admitted text/octet-stream aliases). Header/extension alone cannot admit a blob.
- Isolated parser processes enforce resource budgets and are killed/reaped on timeout/cancellation; private temporary directories are removed. This is resource containment, not an OS sandbox.

| Remote bound | Limit |
|---|---|
| Image body | 6 MiB |
| All remote bodies | 32 MiB per request |
| Remote admission including queue/DNS/download/parser | 45 seconds |
| Concurrent admissions | 4 per backend worker |
| Each fetch | 30 seconds total; 5-second connect, 10-second read/write/pool |
| Parser wall / CPU / address space | 15 seconds / 10 seconds / 512 MiB |
| Image pixels / preview dimensions | 50 million / 1600 × 1600 |
| DOCX expansion | 1,000 entries / 32 MiB |

Phase 13C limits also apply to mixed owned/remote requests. PDFs require a text layer; scanned PDFs use the upload/OCR flow. Unsupported, encrypted, corrupt, and empty inputs are rejected.

Resource excess returns 413; blocked/invalid content 422; fetch/unexpected parser failure 502; timeout 504; parser startup failure 503. All are ordinary JSON responses before generation or SSE.

## Slice 13E - client-executed function tools

Client functions use permitted `audrey_passthrough/<exact-tag>` models with actual Ollama `tools` capability. Existing passthrough enablement/role/allow-list gates apply before file hydration/SSE. Capability lookup failure returns 502; unsupported models/virtual-pipeline client tools return 400. Native `direct/...` publication is separate.

Audrey validates requested functions but never dispatches them through its server tool registry. The client approves and executes calls. Virtual models retain server-managed tools. Passthrough scheduling, thinking policy, sampling, token accounting, and authenticated identity remain shared. Unretained passthrough calls do not create compatibility chat history.

### Request, validation, and replay

Definitions use flat `type: function`, name, description, parameters, and optional strict. Generated items contain `type: function_call`, stable `fc_` item id, distinct `call_id`, name, completed status, and JSON-string arguments. Function-only output has no empty assistant message.

Stateless continuation supplies original input, every prior output item, and one `function_call_output` result per call. Each result must match a unique preceding unanswered call; all pending calls need results before later messages/generation. Adjacent parallel calls plus assistant text form one provider assistant turn. Historical calls may replay without current definitions.

Only `tool_choice: auto|none` is supported; omission means auto. None withholds definitions from the provider for final synthesis but still admits submitted schemas. `parallel_tool_calls: false` rejects multi-call batches before exposure. Required/forced choices, built-ins, combined skills, combined structured text, and background execution are unsupported.

Generated names/JSON-object arguments must match advertised definitions and the bounded schema subset. Strict requires every object property and `additionalProperties: false`; omitted strict is false, without OpenAI's default normalization. Validation does not guarantee a tool call on every auto draw.

| Function budget | Limit |
|---|---|
| Definitions | 16 / 64 KiB aggregate JSON |
| Names / description | 1–64 letters, digits, underscores, hyphens / 1,024 characters |
| Arguments | 64 KiB per call |
| Calls per pending/output group | 8 |
| Replay items | 128 |
| Result text | 100,000 characters per item |
| Effective prompt, arguments/results/definitions/history | 250,000 characters / 16,000 tokens |

Prompt admission occurs before file access, after document hydration, and after vision descriptions. Existing multimodal limits remain authoritative.

### Streaming and interruption

Stable indexed items use output-item and function-argument delta/done events. Audrey buffers native calls through confirmed successful provider termination, validates the whole batch, then emits one complete argument delta per call. Clients execute completed call items only after `response.completed`;
failed or incomplete responses must not execute tools.

Invalid batches expose no executable items: completed 502 or streamed failure. Token-limit stops discard buffered calls and return incomplete partial text/usage with `max_output_tokens`. Unconfirmed EOF fails with `responses_stream_interrupted`. Cancellation closes the provider before releasing GPU/user slots. Reasoning/native progress are not answer text.

## Slice 13F - saved Responses and text/function continuation

`store: true` opts into canonical owner-scoped SQLite retention. Omitted/null/false stays stateless; this differs from OpenAI's default. Completed objects echo store and previous_response_id. Only successful terminal responses are retained; failed/incomplete/cancelled generations cannot become parents.

| Operation | Contract |
|---|---|
| GET `/v1/responses/{id}` | Exact saved terminal object; no generation |
| DELETE `/v1/responses/{id}` | Deleted response object; descendants removed too |
| `previous_response_id` | Owned saved history plus only new input; independently opt into saving the child |
| Missing/foreign/deleted/expired id | Identical 404 `responses_not_found` before provider/SSE |

Schema 21 stores immutable terminal/replay snapshots with account and compound parent/owner foreign keys. Provider reasoning is not saved. Records expire after 30 days; pruning/deletion cascades through descendants, even if a child is newer. Account privacy purge removes saved Responses. No in-progress retrieval/list/Conversations/background API is added.

The provider receives full combined history. Chaining reduces caller payload/bookkeeping; it does not by itself reduce input tokens or demonstrate cache savings.

Request instructions, model, tools, skill, metadata, sampling, and output ceilings are not inherited; supply them again. System/developer messages originally inside input remain history. Pending function calls require matching results, and historical arguments are checked against newly supplied matching definitions. Function replay requires a permitted tool-capable passthrough model.

Stored/chained input permits text, typed input_text, assistant output, and function call/results only. File ids, inline/remote images, and remote documents return 400 before hydration. Unretained multimodal input remains supported.

| Storage bound | Limit |
|---|---|
| Chain depth / replay items | 32 responses / 128 items |
| Retained request JSON / terminal-plus-replay JSON | 1 MiB / 2 MiB |
| Per owner | 100 records / 30 MiB |
| Combined admitted prompt | 250,000 characters / 16,000 tokens |

No history is silently dropped. Oversized input fails 413 before dispatch; owner quotas are atomic at completion. Success is committed before completed HTTP/SSE or executable function groups are exposed. Storage failure returns 503 or streamed failure; quota uses `responses_storage_limit`; parent deletion during generation uses `responses_not_found`. Already streamed text can remain visible. Cancellation waits for bounded in-flight SQLite saving and removes an unseen record before propagating.
