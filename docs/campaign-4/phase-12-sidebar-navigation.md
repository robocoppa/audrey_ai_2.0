# Campaign 4 Phase 12 — navigation

**Status:** Closed by owner October 27, 2026. Desktop/mobile navigation,
Project controls and interrupted-deletion recovery are implemented. Closure
does not assert passes for previously unreported browser polish.

## Conversation and sidebar behavior

The desktop sidebar has New conversation and independently scrolling history,
ordered by last-message activity (creation time for empty drafts). Selection or
metadata changes do not promote old chats; stale responses cannot erase newer
activity. Private rows retain a distinct red background in active/archived views.
My Files opens from the desktop top bar. Project selection lives in the composer.

Archive/Delete completion preserves later navigation and its draft. Deleting
the selected conversation clears its cached thread and opens a focused new draft;
deleting another conversation preserves the current one. A later navigation wins
if replacement-draft creation is still pending.

After a failed/lost DELETE response, the browser checks the same owned ID once
with a five-second deadline. Only canonical not-found confirms deletion. An
existing chat, failed check, authentication refusal or generic proxy 404 retains
the failure. DELETE and new-chat POST are never repeated automatically. Errors
identify action/status without request queries or chat contents.

## Compact and mobile layout

At widths up to 1,100 px, Menu/New chat flank the logo. The left drawer provides
username, search, Active/Archived history, collapsible Projects, My Files,
administrator controls when permitted and Log out. Project lists scroll without
hiding the footer. The drawer starts closed; selecting/creating a chat, outside
tap, Close or Escape dismisses it. Global dialogs close it and restore Menu
focus afterward. Background content is inert and focus stays inside while open.
Resizing preserves the chat and restores the desktop sidebar.

Compact history rows support right-swipe archive/restore and left-swipe
Delete/Cancel; deletion requires an explicit tap. Vertical/short gestures keep
scrolling. Keyboard arrows expose equivalent actions; Escape/outside interaction
cancels. Desktop controls keep their ordinary behavior.

The jump-to-latest action appears only when actual messages overflow. Chat and
dialog content scroll in their own regions. Deliberate pull-to-refresh starts
at the top, requires 120 px and release confirmation, and excludes editing,
uploads, menus and dialogs. Short, reversed, sideways or cancelled gestures
retain normal scrolling.

Settings initially focuses Close instead of a text field. Loading/authentication
status remains accessible and honors reduced motion. Authentication uses its
explicit Cloudflare waiting label; portrait/rim animation is decorative rather
than measured progress. Ordinary loading starts a fresh faster cycle. Header,
composer and drawer controls retain usable compact touch/focus targets.
