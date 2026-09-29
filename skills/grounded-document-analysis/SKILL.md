---
id: grounded-document-analysis
name: Grounded document analysis
description: Read, compare, and explain the user's uploaded documents with explicit evidence coverage.
version: 1
allowed_tools:
  - list_my_files
  - get_file_text
  - kb_search
  - kb_image_search
supported_modes:
  - auto
  - fast
  - deep
resources: []
---
Treat the user's uploaded files as the evidence for the answer. Use the available file and knowledge tools according to their descriptions, and do not fill gaps in retrieved evidence from general knowledge.

Resolve the evidence set before drawing conclusions. If the user has not identified a file precisely, inspect their files and either establish the exact match or name the plausible matches that need clarification. When the request spans multiple files, retrieve evidence for every requested file independently.

Keep an internal evidence ledger while working: one entry per file, the relevant evidence retrieved from it, and any unread or unavailable portion. Use that ledger to keep claims attached to the file that supports them. Missing evidence in one file must remain missing; content from another file cannot stand in for it.

For comparisons, present each file's position or facts separately before synthesizing similarities, differences, or conflicts. Name the supporting file wherever attribution could otherwise be ambiguous.

Check the returned coverage information before characterizing a whole document. Continue reading when the task requires complete coverage and the remaining content is reachable. If the available rounds or context do not permit a complete read, state which file and portion were covered and which portion remains unread.

Use image evidence only when the user's question depends on visual material. If text and image evidence conflict, report the conflict instead of silently choosing one.
