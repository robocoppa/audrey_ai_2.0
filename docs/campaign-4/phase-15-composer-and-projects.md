# Campaign 4 Phase 15 — composer and Projects

**Status:** Composer and Projects accepted, including direct project upload.
Shared overlay menus are built; browser acceptance is pending.

## Composer

The centered dock has a 56 rem desktop maximum: message field plus Send/Stop above an equal-height **Model**, **Project**, **Add files**, and **Tools & skills** rail. Compact screens use two columns, keeping Project beside Model and Files beside Tools. All four controls share the same surface, value color/typography, spacing, and outer focus treatment on mobile and desktop. Project selection moves the current conversation through the existing owner-scoped API.

Add files explains uploading a new file or choosing Ready files from My Files. Tools & skills uses server names/descriptions and explains Automatic; it selects a skill that may narrow tools rather than pretending to be an individual-tool permission editor.

Model, Project, Files, and Tools & skills open in the same dark panel above
the message field. The panel floats over the conversation; opening it keeps
the composer and empty-chat portrait in place. Its width and scrollable height
fit the visible viewport, including narrow screens and keyboard changes.
Only one picker opens at a time. Outside click, keyboard focus leaving the
picker, Escape, and the close button dismiss it. Escape, Close, and choosing a
model, project, or skill restore focus without scrolling the chat. Files stays
open for multiple selections. Model choices retain descriptions and the
role-filtered Other models list. Project keeps No project and owned projects.
Arrow keys navigate model, project, and skill choices. Open panels suspend
pull-to-refresh. Mobile history uses the Phase 12 drawer.

The empty draft stays centered; the composer pins to the bottom after the first
message. Existing attachment limits/progress, model availability, Stop, and retry
remain intact.

## Personal Project contract

Projects group owned conversations and share optional instructions plus references to the same owned files in My Files. They do not copy files or inject other project conversations into each chat.

Server-owned limits are 100 name characters, 4,000 instruction characters, and 20 file references. Project/conversation/file access is owner scoped; foreign objects return the same 404 as missing objects. Membership is a context choice, not new authorization to use a file.

Project CRUD, conversation creation/listing, move/ungroup, file selection/removal, and pagination use the owner-scoped API. Deleting a project transactionally ungroups conversations and removes references while retaining conversations and files. Deleting a file prunes project references.

The sidebar has compact collapsible Projects. Project home provides instructions, conversations, file management, and clear non-destructive deletion confirmation. Ordinary search/history and New conversation remain available.

## Upload and processing

Project home offers **Upload file to project** and **Choose from My Files**
with matching readable labels. Its conversations use the sidebar conversation
row styling, including stacked title/activity and hover treatment. These 18A
presentation changes and existing project behavior are accepted.

New uploads join immediately; Pending/Processing membership survives refresh/restart. Only Ready files become grounding evidence. Visible active states refresh every five seconds; failed rows remain visible for explicit removal, while missing references are pruned.

My Files has a clearer **Add files** action and styled **Choose files** control.

## Grounding and snapshots

At run creation, the server snapshots the owned project's instructions and currently Ready file ids, revalidating them against the owner's upload catalog. Changes apply to later runs; an active run retains its snapshot. A concurrent conversation move during run creation returns 409 without creating messages.

Project guidance is below Audrey's platform instructions and before conversation messages; it is excluded from classification and complexity routing. Filenames and passages are quoted/JSON-encoded as untrusted evidence.

Automatic text retrieval searches only selected owned non-image files, promotes each represented file's best hit, and caps context at eight passages and 3,000 tokens. Evidence uses the normal saved-source contract. Project images are listed but require a message attachment for visual inspection.

Workflow models retain existing scoped tools. Direct models receive the bounded text context without gaining tools. Ordinary conversations do not receive project context.

Sharing, nested project folders, and automatic cross-conversation memory are outside this phase.
