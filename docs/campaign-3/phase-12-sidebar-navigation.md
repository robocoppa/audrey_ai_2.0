# Campaign 3 Phase 12 - sidebar navigation

**Status:** Slice 12A is complete and live-accepted. The user confirmed all
navigation checks passed on October 7, 2026.

## User direction

Simplify the conversation sidebar. A follow-up correction moves file access to
the top bar, where it is visible without reading as part of the account group.

## Slice 12A

### Top of the sidebar

- Replace the separate Files and `+ New` controls with one **New conversation**
  button.
- Make the button span the available sidebar width.
- Remove the **Workspace** label from the sidebar.
- Remove the username currently displayed in the sidebar.

### Top-bar file action

- Remove Files from the sidebar.
- Add a noticeable **My Files** button to the top bar.
- Place it in its own visual group with space and a separator before the
  username, Admin Panel when present, and Log out.
- Preserve the existing Files dialog.

## Boundaries

- Preserve new-conversation creation, empty-draft cleanup, conversation search,
  active/archived history, selection, rename, archive, restore, and deletion.
- Preserve the current Files dialog and owner-bound file behavior; only its
  launch location changes.
- Keep account and session actions together in their own labeled group.
- Retain usable keyboard focus, touch targets, narrow-screen behavior, and a
  scrollable conversation history below the primary action.

## Acceptance gate

1. The sidebar top contains one full-width **New conversation** button.
2. The old Files and `+ New` pair is gone.
3. **Workspace** and the sidebar username are gone.
4. Conversation history fills the remaining sidebar and scrolls independently
   when long.
5. A visually distinct **My Files** action appears in the top bar, separated
   from the account actions.
6. My Files opens the unchanged native Files dialog.
7. Empty drafts, normal conversation creation, history actions, and mobile
   layout still pass.

## Implementation

- The sidebar begins with one full-width **New conversation** button and a
  familiar plus icon.
- The old Workspace heading, sidebar username, top Files button, and `+ New`
  label are removed.
- Conversation search, Active/Archived views, and the independently scrollable
  conversation list keep their existing behavior.
- **My Files** now lives in the app shell rather than the conversation
  workspace. It has a folder icon, accented button treatment, extra spacing,
  and a vertical separator before the separately labeled account group.
- The app shell owns the Files dialog, so it remains available even when the
  conversation model catalog is unavailable.
- Removing the bottom utility section returns the narrow-screen sidebar to its
  previous compact height. At phone width, the top bar wraps its session
  controls to a second row rather than clipping My Files.

## Automated contracts

- The application test now addresses **New conversation** by its final
  accessible name when the model catalog is unavailable.
- The native browser contract requires New conversation to fill its container,
  confirms Workspace and the username are absent from the sidebar, and verifies
  My Files is outside the sidebar with visible separation from account actions.
- The mobile contract requires My Files to remain inside the viewport.
- The empty-draft browser contract uses the new action name and still requires
  each abandoned empty conversation to be discarded.
- Existing Files launch, conversation search, active/archive, history action,
  keyboard, and narrow-screen contracts remain in the suite.

The laptop has the installed frontend dependency tree but no Node runtime, so
TypeScript, Vitest, build, and Playwright run in the normal `audrey-ui` build
environment.

**Current laptop result, 2026-10-01:** The combined Phase 12 correction and
following Responses slice passed the full 3,106-test hermetic suite with one
existing FastAPI deprecation warning. The diff check and changed-file Ruff
pass.

## Native browser result

**Passed, 2026-10-07.** The user confirmed all navigation checks passed,
including the sidebar/header placement, My Files launch, keyboard and
narrow-screen behavior, empty drafts, and history. Slice 12A is closed;
these checks must not be requested again for unchanged behavior.
