# Campaign 4 Phase 12 — navigation

**Status:** Desktop navigation accepted. Compact/mobile followup built;
browser acceptance pending.

The desktop sidebar starts with one full-width **New conversation** action.
Workspace, the sidebar username, and the old Files/+ New pair are removed.
Conversation history scrolls independently below the primary action.

**My Files** lives in the desktop top bar as an accented folder action,
separated from username, Admin Panel, and Log out. The app shell owns its dialog.

## Compact and mobile layout

- At widths up to 1,100 px, hide **Ask Audrey** and group My Files, account
  settings, administration, logout, and any health notice under a top-right
  hamburger. Outside click, Escape, leaving focus, or selecting an action closes
  it. Closing a dialog restores focus to the visible opener.
- At widths up to 900 px, history is hidden initially. The page shows the last
  conversation or the existing new-chat portrait/composer. Compact **Chats**
  and **New chat** actions remain available above the conversation.
- Chats opens a scrollable drawer containing Projects, search, active/archive
  views, and history. Selection, new-chat creation, outside tap, Close, and
  Escape dismiss it. Background chat is inert while open; keyboard focus stays
  inside. Resizing to desktop restores the sidebar without remounting chats.
- Chat, files, profile, project, and administration content fit compact
  viewports and scroll within their own regions. Page overscroll is suppressed
  to prevent accidental native pull-to-refresh; ordinary scrolling remains.
- Profile opens with Close settings focused instead of its name input.
  Loading periods cycle one, two, three every 1.8 seconds; reduced motion shows
  three static periods. The accessible status label does not animate.

Conversation creation, empty-draft cleanup, search, rename, archive/restore,
deletion, and active-run retention keep their existing behavior.
