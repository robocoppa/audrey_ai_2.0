# Campaign 4 Phase 12 — navigation

**Status:** Desktop navigation accepted. Compact/mobile followup built;
browser acceptance pending.

The desktop sidebar starts with one full-width **New conversation** action.
Workspace, the sidebar username, and the old Files/+ New pair are removed.
Conversation history scrolls independently below the primary action.

**My Files** lives in the desktop top bar as an accented folder action,
separated from username, Admin Panel, and Log out. The app shell owns its dialog.

## Compact and mobile layout

- At widths up to 1,100 px, the top bar contains matching **Menu** and
  **New chat** icon buttons with the builtryte logo between them. Hide
  **Ask Audrey** and the desktop account controls.
- Menu opens one left drawer: Files/account/admin/logout actions first,
  the current chat's rename/archive/delete controls next, a collapsible
  **Projects** category, then **Chats**, search, active/archive views, and
  history. Archive/Delete are absent from the main compact chat screen.
- The drawer starts closed, showing the last conversation or the existing
  new-chat portrait/composer. Selection, new-chat creation, outside tap,
  Close, and Escape dismiss it. Opening a global dialog closes it; closing
  that dialog restores focus to Menu. Background chat/header are inert while
  open, and keyboard focus stays inside.
- Resizing to desktop restores the sidebar without remounting chats.
  Project selection sits beside Model in the composer; Files and Tools follow.
  The jump-to-latest arrow is shown only when actual messages exceed the
  available chat space, never because the empty portrait or an open picker
  makes the viewport overflow.
- Chat, files, profile, project, and administration content fit compact
  viewports and scroll within their own regions. Page overscroll is suppressed
  to prevent accidental native pull-to-refresh; ordinary scrolling remains.
- Profile opens with Close settings focused instead of its name input.
  Loading periods cycle one, two, three every 1.8 seconds; reduced motion shows
  three static periods. The accessible status label does not animate.

Conversation creation, empty-draft cleanup, search, rename, archive/restore,
deletion, and active-run retention keep their existing behavior.
