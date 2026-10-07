# Campaign 3 Phase 15 — composer and Projects

**Status:** Complete and accepted, including direct project upload.

## Composer

The centered dock has a 56 rem desktop maximum and two rows: message field plus Send/Stop above an equal-height Model, **Add files**, and **Tools & skills** rail. Narrow screens wrap the controls without clipping.

Add files explains uploading a new file or choosing Ready files from My Files. Tools & skills uses server names/descriptions and explains Automatic; it selects a skill that may narrow tools rather than pretending to be an individual-tool permission editor.

File and skill pickers open above their controls, are mutually exclusive, and close on outside click/Escape with focus restoration. The empty draft stays centered; the composer pins to the bottom after the first message. Existing attachment limits/progress, model availability, Stop, and retry remain intact.

## Personal Project contract

Projects group owned conversations and share optional instructions plus references to the same owned files in My Files. They do not copy files or inject other project conversations into each chat.

Server-owned limits are 100 name characters, 4,000 instruction characters, and 20 file references. Project/conversation/file access is owner scoped; foreign objects return the same 404 as missing objects. Membership is a context choice, not new authorization to use a file.

Project CRUD, conversation creation/listing, move/ungroup, file selection/removal, and pagination use the owner-scoped API. Deleting a project transactionally ungroups conversations and removes references while retaining conversations and files. Deleting a file prunes project references.

The sidebar has compact collapsible Projects. Project home provides instructions, conversations, file management, and clear non-destructive deletion confirmation. Ordinary search/history and New conversation remain available.

## Upload and processing

Project home offers **Upload to project** and **Choose from My Files**. New uploads join immediately; Pending/Processing membership survives refresh/restart. Only Ready files become grounding evidence. Visible active states refresh every five seconds; failed rows remain visible for explicit removal, while missing references are pruned.

My Files has a clearer **Add files** action and styled **Choose files** control.

## Grounding and snapshots

At run creation, the server snapshots the owned project's instructions and currently Ready file ids, revalidating them against the owner's upload catalog. Changes apply to later runs; an active run retains its snapshot. A concurrent conversation move during run creation returns 409 without creating messages.

Project guidance is below Audrey's platform instructions and before conversation messages; it is excluded from classification and complexity routing. Filenames and passages are quoted/JSON-encoded as untrusted evidence.

Automatic text retrieval searches only selected owned non-image files, promotes each represented file's best hit, and caps context at eight passages and 3,000 tokens. Evidence uses the normal saved-source contract. Project images are listed but require a message attachment for visual inspection.

Workflow models retain existing scoped tools. Direct models receive the bounded text context without gaining tools. Ordinary conversations do not receive project context.

Sharing, nested project folders, and automatic cross-conversation memory are outside this phase.
