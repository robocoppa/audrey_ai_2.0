# Campaign 3 Phase 15 - composer controls and Projects

**Status:** Slice 15A passed user acceptance on 2026-10-02. Slice 15B is
laptop-complete and awaits its targeted deployment and restart-persistence gate.

## Goal

Make the message composer narrower, clearer, and visually balanced, then add
owner-scoped Projects that group conversations and give every conversation in
a project the same instructions and selected file context.

This phase keeps the two concerns in separate deployable slices. The composer
change can ship and be judged on its own. Projects then land through storage,
navigation, and grounding boundaries without hiding a large data migration
inside a visual change.

## Product contract

A Project is an owner-bound workspace with:

- a name;
- optional project instructions;
- references to ready files already owned in **My Files**;
- conversations that belong to that project.

Project files are references to the existing owner-scoped files. Audrey does
not copy their bytes or create a second file library. Adding a file to a
project makes it eligible as shared context for every conversation in that
project. A normal per-message attachment still adds one-off context and does
not automatically add the file to the project.

Project membership is a context choice, not an authorization boundary. The
user still owns and can explicitly use every file in My Files. Cross-owner
project, conversation, and file access must continue to return the same
not-found result used by Audrey's current ownership boundaries.

Deleting a project ungroups its conversations and leaves the conversations
and files intact. Deleting a file removes its project memberships. Project
changes affect runs started after the change; an in-progress run keeps the
project snapshot captured when it began.

The first version is personal. Shared projects, project members, nested
project folders, and automatic cross-conversation project memory are outside
this phase. Conversations are grouped together, but one conversation is not
silently injected into another.

## Slice 15A - composer control rail

### Layout

Turn the composer into two clear rows:

1. The upper row contains the expanding message field and the Send or Stop
   action.
2. A lower control rail contains the model, file, and tools/skills controls.

The attachment control and tools/skills control therefore sit below the text
field instead of consuming its writing width. Keep all rail controls the same
height, border treatment, spacing, and baseline. The left and right padding of
the composer must match, and wrapped mobile rows must retain the same gaps.

Reduce the desktop composer dock from its current 64 rem maximum to an initial
56 rem maximum. Judge the exact value in the deployed browser at ordinary and
wide desktop sizes; retain the existing fluid full-width behavior on smaller
screens.

### Explanatory copy

Use visible words rather than relying on the paperclip or a vague default:

- **Add files** - opens a picker headed **Add files to this message**, with the
  explanation **Upload a new file or choose ready files from My Files.**
- **Tools & skills: Automatic** - opens a picker headed **Choose how Audrey
  uses tools**, with Automatic explained as **Audrey chooses the available
  tools when they are useful.**
- Skill choices use the server-provided name and description. The current
  control selects a skill that can narrow Audrey's tools, so the UI must not
  claim it is a raw individual-tool permission editor.
- **Model: _name_** - keeps the existing model selection in the same control
  rail so the three selectors read as one balanced set.

The compact control text may shorten on narrow screens, but its accessible
name and expanded picker must retain the full explanation.

### Interaction and accessibility

- Anchor the file and tools/skills pickers above their matching controls.
- Opening either picker closes the other. Clicking elsewhere or pressing
  Escape closes it and restores focus to its control.
- Preserve attachment upload progress, ready-file filtering, the ten-file and
  image limits, disabled-model behavior, retry, Send, and Stop behavior.
- Preserve the empty-conversation centered state and the bottom-pinned dock
  after the first message.
- Keep a visible keyboard focus state and accurate `aria-expanded`, dialog or
  listbox semantics, control labels, and disabled explanations.

### Slice 15A acceptance

1. On a wide desktop, the composer is visibly narrower than the current
   64 rem layout and centered under the conversation.
2. The message field occupies the full upper row except for Send or Stop.
3. Model, Add files, and Tools & skills form an aligned lower rail with equal
   control height and symmetric outer spacing.
4. A first-time user can tell from the visible labels how to add files and what
   Automatic tools means without hovering an icon.
5. File and tools/skills pickers are mutually exclusive and close on outside
   click and Escape.
6. Upload one real file, select it, send a question, stop one active run, and
   retry one failed or cancelled question. Each existing behavior still works.
7. Repeat the layout and keyboard checks at a narrow viewport; controls wrap
   without clipping, overlap, or horizontal scrolling.

### Slice 15A implementation result

**Completed on the laptop, 2026-10-02.** The composer dock now has a 56 rem
desktop maximum and a two-row card: the expanding message field plus Send/Stop
sit above an equal-width Model, Files, and Tools & skills rail. The former
paperclip-only action is labeled **Add files**, and its picker explains that it
can upload or select ready My Files. The former native skill select is a
described Tools & skills picker with Automatic and server-provided skill
descriptions.

The file and tools/skills pickers are mutually exclusive, close on outside
click or Escape, and restore keyboard focus. At 560 px the rail wraps to two
columns, and at 400 px it stacks into one column. Focused Vitest and Playwright
contracts cover the explanatory copy, skill choice, equal desktop sizing,
narrow-screen containment, dismissal, and focus behavior.

The user accepted the deployed UI slice on 2026-10-02 while continuing normal
use and will reopen any regression found during that testing. Node remains
deferred on the laptop, so no local TypeScript, Vitest, build, or Playwright
result is claimed.

## Slice 15B - Projects storage and owner-scoped API

Add a schema migration after schema 18 with:

- `app_projects`: project id, owner id, name, instructions, created and updated
  timestamps;
- `app_project_files`: project id, file id, added timestamp, and a unique
  project/file pair;
- a nullable project id on `app_conversations`, indexed with owner and
  activity for project conversation lists.

The file relation stores only the existing file id. Because file metadata is
owned by the upload store, every add, list, run, and removal path revalidates
the file against the authenticated owner rather than trusting the relation.

Add owner-scoped application routes to:

- create, list, read, rename, update instructions, and delete a project;
- list, add, and remove project files;
- create a conversation inside a project;
- move an existing conversation into a project or back to no project;
- list conversations for one project without weakening archive, search, or
  pagination behavior.

Use explicit limits for the first contract: a 100-character name, 4,000
characters of instructions, and 20 ready files per project. Keep these limits
in one server-owned contract returned to the client. Reject duplicate,
non-ready, missing, and cross-owner file membership without revealing whether
another owner's object exists.

Project deletion must be transactional: clear the conversations' project id,
delete the project's file relations, then delete the project. Conversation and
file deletion semantics remain otherwise unchanged, including empty-draft
cleanup.

### Slice 15B verification

- Migration from schema 18 preserves every existing conversation with a null
  project id.
- CRUD, rename, instructions, pagination, and limits have repository and route
  coverage.
- Cross-owner project, file, and conversation mutations return 404.
- A conversation can move between two owned projects and back to no project.
- Project deletion retains its conversations and files but removes all
  memberships.
- File deletion removes or harmlessly prunes every project reference.
- Restart persistence and backup verification include the new tables and
  conversation field.

### Slice 15B implementation result

**Completed on the laptop, 2026-10-02.** Schema 19 adds owner-scoped projects,
unique project/file references, and nullable conversation membership. The
native API now supports project CRUD and pagination, project file selection,
project conversation creation and listing, and moving or ungrouping existing
conversations. Ready-file and owner checks are resolved through the existing
native file authority. File deletion prunes project references, while project
deletion transactionally ungroups conversations and keeps both conversations
and files.

Backup inspection now counts both project tables, and restart snapshots retain
the conversation project id. Repository and route coverage includes the schema
18 migration, limits, duplicate and non-ready files, cross-owner 404 behavior,
move and ungroup behavior, transactional deletion, and native file cleanup.
The focused backend set passes 145 tests. The full hermetic suite passes 3,113
tests with the existing FastAPI deprecation warning; changed-file Ruff and
Python compilation pass.

`tests/smoke/smoke_native_projects.py` is the targeted live gate. Its capture
step creates disposable projects and one conversation, references one existing
Ready file without changing it, and proves cross-owner and duplicate rejection.
After Audrey restarts, verify checks project, conversation, and file-reference
persistence, then proves project deletion retains the conversation and file and
removes all temporary records. This live result remains pending.

## Slice 15C - Projects navigation and management

Add a compact **Projects** section to the conversation sidebar below **New
conversation**. It contains a **New project** action and collapsible project
rows. Keep ordinary recent and archived conversation views available.

Selecting a project opens a project home view with:

- project name and optional instructions;
- **New conversation in project**;
- a compact list of its conversations;
- a **Project files** control that reuses the My Files explorer for selection;
- rename, edit instructions, move conversation, remove file, and delete
  project actions.

Creating a conversation while a project is selected assigns it immediately.
Existing conversation actions gain **Move to project** and **Remove from
project**. The conversation header shows the current project as a small
breadcrumb that returns to the project home.

The sidebar must stay compact when many projects exist: project rows collapse,
long names ellipsize, only the selected project's conversation list expands,
and the sidebar retains its own scrolling boundary. Project deletion requires
an explicit confirmation that says conversations and files will be kept.

### Slice 15C acceptance

1. Create and rename a project, add two ready files, and edit its instructions.
2. Start two conversations inside it and confirm both remain grouped after a
   hard refresh and Audrey restart.
3. Move an existing ordinary conversation into the project and remove it
   again without losing messages.
4. Search, active/archive views, empty-draft cleanup, and the global New
   conversation action still behave as before.
5. Delete the project and confirm its conversations return to the ordinary
   list and both files remain in My Files.
6. Verify desktop, narrow-screen, keyboard, focus, empty, loading, and error
   states manually in the native browser.

## Slice 15D - project instructions and file context

At native run creation, load the conversation's owner-scoped project and take
an immutable snapshot of its name, instructions, and current ready file ids.
The pipeline receives that typed snapshot; the browser never supplies trusted
project context with a run request.

For each project run:

1. Put the project instructions in a dedicated user-owned context section
   below Audrey's platform instructions and above conversation messages.
2. Provide a compact manifest of selected filenames and file kinds.
3. Search only the selected project files for passages relevant to the newest
   question, using the existing owner namespace and indexes.
4. Add a bounded set of retrieved passages to model context and persist the
   selected private-file evidence through the existing answer-provenance
   contract.
5. Treat filenames and retrieved contents as data, never as instructions.

Do not concatenate every project file into every prompt. Retrieval must have a
server-owned result and token budget. Documents and indexed audio/video text
use their existing chunks and artifacts. Relevant image context must reuse the
existing bounded preview and vision path; if no safe visual slot is available,
Audrey says the image needs to be attached to that message instead of
pretending it inspected it.

Workflow models may continue with the existing owner-scoped file tools after
the initial project retrieval. Direct models receive the bounded retrieved
context but do not gain tool access. A project file outside the relevant
retrieval results remains available for an explicit attachment or a later,
more specific question.

Project context must never bleed into an ordinary conversation or another
project. Changing project files or instructions affects the next run and does
not rewrite saved messages or a run already in progress.

### Slice 15D acceptance

Use two real documents with distinct facts and one project instruction:

1. Add both documents to a project and start a new project conversation.
2. Ask a question whose answer needs one fact from each document. Confirm the
   answer follows the project instruction and names both files in its saved
   source evidence.
3. Hard refresh and ask a follow-up without attaching either document. Confirm
   the shared project context still works.
4. Ask the same question in an ordinary conversation. Confirm Audrey does not
   automatically inject the project's instructions or selected file evidence.
5. Remove one file and change the instruction. Confirm the next run uses the
   new instruction and no longer retrieves the removed file.
6. Start a run, then change the project in another tab. Confirm the active run
   finishes with its starting snapshot and only the following run sees the
   change.
7. Verify a second user receives 404 for the project and cannot add their file
   to it or move a conversation into it.

## Delivery order

1. Build and accept Slice 15A independently.
2. Land Slice 15B with schema and API tests before exposing Projects in the
   browser.
3. Build Slice 15C on the owner-scoped API.
4. Add Slice 15D grounding, then run the real two-document browser gate.

Do not mark Phase 15 complete until all four slices pass their targeted native
gates. The laptop Node installation remains deferred, so frontend build,
Vitest, and Playwright evidence comes from the deployed `audrey-ui` build
environment unless that decision changes.
