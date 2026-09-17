# Phase 2E: import Audrey chat-history exports

This is a one-time operator migration from the JSON downloaded by **Settings →
Download chat history** into Audrey's canonical native conversation history.
It does not read Open WebUI's database or add a runtime dependency on OWUI.
Nothing is imported automatically when the new code is deployed.

The Settings export contains the messages currently in Audrey's *chat-search
archive*. It is not a complete account backup: messages still awaiting archive
delivery, uploads, memories, attachments, and any conversations absent from
that archive are not in the file. Check archive repair before exporting. The
imported conversations start **Archived** so they do not displace current chat.
They can be opened, restored, or deleted through the normal native UI.
The operator command currently accepts at most 100 MiB and 100,000 messages
per export; larger histories need a separately planned batch migration.

## Before deploying or applying

1. Take and verify the normal pre-deploy application SQLite backup **before**
   starting the updated backend. Backend startup applies schema v12; the
   import command's own backup happens later and cannot roll back that schema
   migration. Use a SQLite-aware online backup, not a raw copy of a live WAL
   database.
2. Download one chat-history JSON from the intended user's Settings page. New
   downloads include `audrey_user_id` and `account_email` owner markers. Store
   the file privately in a location visible to the `audrey` container, such as
   the bind-mounted `/data/imports/` directory. The file contains chat text.
3. Obtain the exact `usr_...` identifier from that user's `/api/me` or the
   Admin Panel. Verify that identifier and the target account email together.
   Do not select by email alone: historical duplicate-email accounts may exist.

## Preview, then apply

From the Audrey compose directory on the Unraid host, replace the example
file, id, email, and backup name with the verified values. The preview is
read-only, including against a schema-v11 database:

```bash
docker compose exec audrey audrey-admin import-chat-export --file /data/imports/audrey-chat-history-YYYY-MM-DD.json --user-id usr_REPLACE_ME --email alice@example.com
```

Read the JSON counts before proceeding. `new_conversations` and `new_messages`
are what apply would add; `existing_messages` are idempotent matches.
`skipped_native_conversations` means the source ID is already a canonical
conversation; `skipped_deleted_conversations` means an earlier import was
deliberately deleted and will not be resurrected. An owner mismatch, altered
source message, or late older message fails instead of silently merging.

Apply makes another preview, then creates a **new, non-overwriting, mode-0600
SQLite online backup** and verifies its integrity before writing imported
history. Choose a unique backup filename in an existing private directory:

```bash
docker compose exec audrey audrey-admin import-chat-export --file /data/imports/audrey-chat-history-YYYY-MM-DD.json --user-id usr_REPLACE_ME --email alice@example.com --apply --backup-to /data/backups/audrey-app-before-chat-import-YYYYMMDD.sqlite
```

An older export without owner markers may still be previewed. Applying it
requires `--allow-unbound` **after independently verifying its owner**; never
use that override to resolve an ambiguous email. Partially applied imports
can be rerun with the same source file and a *new* backup path. Each
conversation commits atomically and provenance prevents duplicate messages.

## Live smoke and rollback

After a rebuild, use a disposable account and a small owner-bound export:

1. Confirm preview counts, apply, and verify that the backup file exists,
   remains private, and passes integrity checking. Do not print chat content.
2. Find the imported conversation under **Archived**, open its messages,
   restore it, and confirm the expected user/assistant order. A partial
   assistant message should appear as incomplete rather than a completed run.
3. Repeat preview and apply with a new backup path; expect zero new messages.
   Delete the disposable imported conversation and repeat preview; it should
   count as skipped-deleted and not reappear.
4. Sign in as a different disposable user and confirm the imported history is
   absent. Confirm existing native conversations have not duplicated.

The command never deletes source archive records. If import results are
unexpected, stop and inspect the preview/output before further writes. Restore
the verified pre-import SQLite backup only through the normal maintenance
procedure, with Audrey stopped and the current database preserved separately;
the pre-deploy backup is required if rolling back the schema as well. A full
account-data purge removes import provenance along with canonical history.
