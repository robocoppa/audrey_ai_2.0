# Campaign 3 Phase 12 - sidebar navigation

**Status:** Slice 12A is laptop-complete and awaiting native browser
acceptance.

## User direction

Simplify the conversation sidebar and create a durable place for secondary
workspace actions.

## Slice 12A

### Top of the sidebar

- Replace the separate Files and `+ New` controls with one **New conversation**
  button.
- Make the button span the available sidebar width.
- Remove the **Workspace** label from the sidebar.
- Remove the username currently displayed in the sidebar.

### Bottom utility section

- Add a distinct section anchored at the bottom-left of the sidebar.
- Move **Files** into that section and preserve the existing Files dialog.
- Give the section a simple structure that can accept more workspace actions in
  later slices without redesigning the conversation history.
- Do not add placeholder buttons for future actions.

## Boundaries

- Preserve new-conversation creation, empty-draft cleanup, conversation search,
  active/archived history, selection, rename, archive, restore, and deletion.
- Preserve the current Files dialog and owner-bound file behavior; only its
  launch location changes.
- Keep account and session actions in their established surfaces unless the
  sidebar change exposes a concrete layout conflict.
- Retain usable keyboard focus, touch targets, narrow-screen behavior, and a
  scrollable conversation history between the top action and bottom utility
  section.

## Acceptance gate

1. The sidebar top contains one full-width **New conversation** button.
2. The old Files and `+ New` pair is gone.
3. **Workspace** and the sidebar username are gone.
4. Conversation history fills the middle and scrolls independently when long.
5. A visually distinct bottom utility section contains **Files**.
6. Files opens the unchanged native Files dialog.
7. Empty drafts, normal conversation creation, history actions, and mobile
   layout still pass.

## Implementation

- The sidebar begins with one full-width **New conversation** button and a
  familiar plus icon.
- The old Workspace heading, sidebar username, top Files button, and `+ New`
  label are removed.
- Conversation search, Active/Archived views, and the independently scrollable
  conversation list keep their existing behavior.
- A semantic bottom utility section contains one full-width Files action with a
  folder icon. It launches the existing Files dialog without changing file
  ownership, listing, or management behavior.
- The utility section sits after the flexible history region, so it remains
  anchored while long history scrolls. The narrow-screen sidebar band is taller
  to retain usable room for history between the primary and utility actions.

## Automated contracts

- The application test now addresses **New conversation** by its final
  accessible name when the model catalog is unavailable.
- The native browser contract requires the new action to fill its container,
  confirms Workspace and the username are absent, and verifies the Files
  utility is wide, below the primary action, and flush with the sidebar bottom.
- The empty-draft browser contract uses the new action name and still requires
  each abandoned empty conversation to be discarded.
- Existing Files launch, conversation search, active/archive, history action,
  keyboard, and narrow-screen contracts remain in the suite.

The laptop has the installed frontend dependency tree but no Node runtime, so
TypeScript, Vitest, build, and Playwright run in the normal `audrey-ui` build
environment.

**Laptop result, 2026-10-01:** The full hermetic backend suite passed 3,098
tests with one existing FastAPI deprecation warning, and the diff check is
clean. No Python file changed, so there is no changed-file Ruff target.

## Native browser gate

1. Confirm the sidebar starts with one full-width **New conversation** button.
   Confirm Workspace and the username no longer appear in the sidebar.
2. Click **New conversation**, then click it again without sending anything.
   Confirm only the latest empty draft remains and no empty item is added to
   history. Send a short message and confirm the conversation appears once.
3. Search history and switch between Active and Archived. With enough history
   to scroll, confirm only the middle list scrolls while the top action and
   bottom Files section remain in place.
4. Click **Files** in the bottom section. Confirm the existing compact Files
   dialog opens and can be closed normally.
5. Use keyboard Tab to reach New conversation and Files; confirm both have a
   visible focus state and activate with Enter.
6. Narrow the browser to a phone-sized width. Confirm both sidebar actions,
   search, views, and the horizontal conversation list remain reachable, then
   reopen a conversation and send a message.

## Completion gate

Close Slice 12A after the native browser gate passes. Expanded Responses API
Item 7 remains next.
