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
- Menu opens one left drawer with the user's name at the top left, **My Files**,
  a collapsible **Projects** category, then **Chats**, search, active/archive
  views, and history. **Admin Panel** (administrators only) and **Log out** stay
  at the bottom while history scrolls. Account settings and the current-chat
  action section are absent from this drawer.
- Any compact chat-history row, including expanded Project chats, supports
  horizontal swipes: right archives (or restores in Archived); left reveals
  **Delete** and **Cancel**. Delete requires the explicit tap. Short/vertical
  gestures preserve scrolling and selection. Keyboard arrow keys reveal the
  equivalent actions; Escape or an outside tap cancels. Visible row trash
  buttons are removed from compact layouts. Desktop controls retain their
  existing behavior.
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
  viewports and scroll within their own regions.
- Mobile chat and Project pages support deliberate pull-to-refresh from the
  top: pull down at least 120 px, then release when the indicator says
  **Release to refresh**. Short, reversed, sideways, and cancelled gestures
  do not refresh. Editing, upload activity, menus, and dialogs are excluded;
  normal scrolling remains native. The app handles this gesture so the
  browser's shorter native trigger does not bypass the threshold.
- Profile opens with Close settings focused instead of its name input.
  Loading periods cycle one, two, three every 1.8 seconds; reduced motion shows
  three static periods. The accessible status label does not animate.

Conversation creation, empty-draft cleanup, search, desktop rename,
archive/restore, deletion, and active-run retention keep their existing behavior.
