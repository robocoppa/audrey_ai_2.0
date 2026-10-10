# Campaign 4 Phase 18 — admin controls and project presentation

## 18A — complete

**Status:** User accepted October 7, 2026.

Account role/status/delete controls use fixed columns, including an empty delete
slot on the current administrator's row. Model rows place a drag handle on the
left, identity in the middle, and consistent role access, Edit, and Reset default
columns on the right. Reset default stays visible and is disabled on default rows.

The role dropdown allows multiple built-in/custom role selections with Save
access and Cancel. It closes on outside click or Escape. Administrators retain
implicit access; an empty selection means administrators only. Public/Private and
Enabled are removed from the model-row interface. Existing stored publication
fields and legacy APIs remain compatible. Editing changes the name/portrait;
role availability is controlled directly in the row.

Saving role access to a previously disabled model reactivates it. The optional
`enabled` field on the publication PATCH is persisted with the profile in one
transaction, retaining the existing audience and portrait. Invalid roles or a
storage failure cannot partly reactivate it. Omitted activation preserves the
old state for other callers.

Pointer dragging scrolls at the panel edges, shows an insertion line and saves the complete order within
the model's kind. Keyboard arrow keys on the same handle provide equivalent
movement. Workflows and direct models retain separate durable orders. Search
locks ordering while it hides rows. Failed saves retain the prior visible order.

Project home uses **Upload file to project**, equally readable file-action labels,
and conversation rows styled like the sidebar. Narrow layouts wrap controls
without changing upload, membership, grounding, or navigation behavior.

Upload, membership, grounding, and navigation behavior remain accepted.
