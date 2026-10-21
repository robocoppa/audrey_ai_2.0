# Campaign 4 Phase 12 — navigation

**Status:** Earlier desktop navigation accepted. Desktop/mobile followup built;
browser acceptance pending.

Desktop Archive/Delete use larger action buttons. Composer controls,
including Project, have larger targets; model focus highlights only
the outer control, without a second glow on the select text. An archive/delete
request finishing after navigation preserves the newly selected chat and draft.
Deleting the selected conversation removes its cached thread and opens a fresh
focused draft. It does not select an older chat or expose a stale deleted-ID
error. A later navigation wins if the replacement draft is still being created.
Deleting an unselected sidebar chat preserves the currently open conversation.
If a conversation DELETE receives a server/proxy error or loses its transport
response, the browser checks that same owned ID once, with a five-second
deadline. Only the canonical API's explicit not-found response confirms deletion; an existing chat, failed check,
authentication refusal or generic proxy 404 preserves the original failure.
DELETE and new-chat POST are never automatically repeated. Other errors identify
the failed action and HTTP status without adding request queries or chat contents.

The desktop sidebar starts with one full-width **New conversation** action.
Workspace, the sidebar username, and the old Files/+ New pair are removed.
Conversation history scrolls independently below the primary action. Rows use
last-message time (creation time for an empty draft), newest first, matching
the server list. New and resumed chats rise immediately after message activity;
selection and metadata-only changes do not promote an older chat. Stale title
or detail responses cannot erase a newer known message timestamp.
Private conversation rows use a subtle red background, with stronger hover and
selected states, in desktop history and the mobile drawer, including Archived.

**My Files** lives in the desktop top bar as an accented folder action,
separated from username, Admin Panel, and Log out. The app shell owns its dialog.

The desktop wordmark is centered over the sidebar. Its divider follows that
sidebar boundary, with equal space between the visible logo and divider,
and between the divider and **AUDREY**. The product label uses spaced
capitals, a larger initial A, and a muted slate color, centered vertically
beside the wordmark.

## Compact and mobile layout

- At widths up to 1,100 px, the top bar contains matching **Menu** and
  **New chat** icon buttons with the builtryte logo between them. Hide
  **AUDREY** and the desktop account controls.
- Menu opens one left drawer headed **Username - My chats**, followed by a
  search box with hint text, **Active/Archived**, and history. No separate search
  label or swipe instructions are displayed; the search retains its accessible name.
- The bottom group contains **My projects**, **My Files** (folder icon retained),
  **Admin Panel** (administrators only), then **Log out**, with matching colors,
  typography, and targets. Projects uses a stacked-layers icon, also shared
  by individual project rows on desktop and mobile. My projects stays collapsible,
  with an adjacent plus
  button to create a project and separators above/below its section. Expanded
  projects scroll independently so the footer remains reachable. Account settings
  and the current-chat action section are absent from this drawer.
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
  While Audrey checks the signed-in Cloudflare account, the portrait loader
  shows **Processing Cloudflare Authentication** and **Please wait...** on
  separate lines, including ordinary sign-in returns without a URL flag.
  The portrait rim fills clockwise over 60 seconds, then its moving light
  erases the rim over another 60 seconds and repeats. Ordinary loading uses
  two seconds per pass and starts a fresh cycle after authentication.
  This is a decorative waiting animation, not measured authentication progress.
  The light sits directly on the photo's edge: the photo obscures the inner
  half of the small moving orb, and the backlight hugs the rim and radiates
  outward without a gap, with a larger orb and a wider, softer fade. Reduced motion shows a static rim. Hovering over the
  main-page portrait adds the same close backlight without brightening the photo.

Conversation creation, empty-draft cleanup, search, desktop rename,
archive/restore, deletion, and active-run retention keep their existing behavior.
