# Lesson 6a — The application database

Lesson 4 put the application database on the system map as an *authority*, the
place where accounts, conversations, messages and runs are true by definition,
and Lesson 5 showed startup opening it before anything else runs. This lesson
looks inside. It answers two questions: Where does a conversation live, and what
keeps that data safe while the code around it keeps changing?

The answer rests on four guarantees, and the read-along takes them in turn:
writes that happen completely or not at all, rules the database enforces itself,
a schema that can change under existing rows, and backups that are checked by
restoring them.

## 1. Context

### 1.1 One file, many tables

Audrey keeps its canonical records in SQLite: one file, named by
[`application.sqlite_path`](../../config.yaml#L76) and opened through Python's
built-in `sqlite3` module. There's no database server to run. That suits one
machine serving a handful of people, with one container writing the data
(Lesson 5), and SQLite still provides what matters here: transactions,
constraints and online backups.

What lives in it:

| What it holds | Tables |
|---|---|
| Accounts and access | `app_users`, `external_identities`, `personal_access_tokens` |
| Conversations | `app_conversations`, `app_runs`, `app_messages` |
| Preferences | `user_preferences` |
| Archive hand-offs | `app_chat_projections` |
| Projects and other features | `app_projects`, `app_tax_cases`, `app_private_tasks` and more |

This lesson reads the database through conversations, messages and runs, the
tables every chat request touches, with preferences as its simplest example.
What it teaches about transactions, schema rules, migrations and backups applies
to every table above. Accounts and access, the delivery of archive hand-offs, and
Projects and the other features each get their own explanation in later lessons.

Some data lives elsewhere: Uploaded files are tracked in their own SQLite
database, saved memories live in Qdrant, and the searchable chat archive belongs
to the tools sidecar. Lesson 4 sorted those into authorities and projections,
and this lesson returns to that distinction at the end.

### 1.2 The files

| Job | File |
|---|---|
| The connection, its settings and the migration runner | [`store.py`](../../src/audrey/app_state/store.py) |
| Every schema change, in order | [`migrations.py`](../../src/audrey/app_state/migrations.py) |
| The commit-or-rollback boundary for writes | [`transactions.py`](../../src/audrey/app_state/transactions.py) |
| Conversations, messages, runs, preferences and archive receipts | [`repositories.py`](../../src/audrey/app_state/repositories.py) |
| The typed records callers get back | [`records.py`](../../src/audrey/app_state/records.py) |
| The worker that hands finished turns to the archive | [`chat_projection.py`](../../src/audrey/chat_projection.py) |
| Operator commands, including backup and verification | [`admin_cli.py`](../../src/audrey/admin_cli.py) |

The other repositories in `src/audrey/app_state/` share the same connection and
lock, and follow the same rules.

## 2. Read-along

### 2.1 One connection, one writer

The store opens a single connection when Audrey starts and shares it with every
repository. Its constructor, at
[`store.py:103-112`](../../src/audrey/app_state/store.py#L103-L112):

```python
self._lock = threading.RLock()
...
self._conn = sqlite3.connect(self.path, check_same_thread=False)
self._conn.row_factory = sqlite3.Row
with self._lock:
    self._conn.execute("PRAGMA foreign_keys = ON")
    self._conn.execute("PRAGMA busy_timeout = 5000")
    self._conn.execute("PRAGMA journal_mode = WAL")
    self._migrate_locked()
self.preferences = PreferencesRepository(self._conn, self._lock)
...
```

Each setting has a reason:

- **WAL** lets readers keep reading a consistent snapshot while a write is in
  progress. Only one connection writes at a time.
- **The busy timeout** makes a writer wait up to five seconds for the write lock
  instead of failing. That matters when an operator's `audrey-admin` command
  writes while Audrey does.
- **Foreign keys** stay off in SQLite unless each connection turns them on, and
  Audrey's ownership rules (§2.2) depend on them.

Sharing one connection takes two more pieces. SQLite calls block, so each
repository method that a request calls runs its SQL on a worker thread with
`asyncio.to_thread`, and the event loop stays free. Those threads run at the
same time, so the lock makes them take turns on the connection. Every write has
the same shape:

```text
repository method (async)            what routes call
 -> asyncio.to_thread                the event loop keeps running
     -> validate the input           no lock needed yet
     -> take the store's lock        one thread at a time
         -> BEGIN IMMEDIATE          the write lock, up front
         -> the SQL
         -> commit, or roll back on any error
 -> a typed record
```

`BEGIN IMMEDIATE` takes the write lock before the transaction reads anything, so
the checks a write makes stay true until it commits. Every write ends in exactly
one commit or one rollback, because on a shared connection a transaction left
open would make every later write fail. Short writes get that boundary from one
helper, [`write_transaction`](../../src/audrey/app_state/transactions.py#L11).

Two other connections open the same file. The service route behind the
chat-history tool reads through its own read-only connection
([`_connection`](../../src/audrey/routes/tool_history.py#L66)), inside one read
transaction, so a page never mixes two moments
([`_read_page`](../../src/audrey/routes/tool_history.py#L91)). And the
`audrey-admin` command writes when an operator grants the first administrator or
imports an exported chat history, taking turns with Audrey at the write lock.

One write skips `BEGIN IMMEDIATE` on purpose. When a model-generated title
arrives for a new conversation
([`apply_generated_title`](../../src/audrey/app_state/repositories.py#L255)),
the `UPDATE` applies only if the title and `updated_at` are still what they were
when the first run started. If the person renamed the conversation meanwhile,
nothing matches and the generated title is dropped. That's optimistic
concurrency: No lock is held while the model thinks, and one conditional
`UPDATE` checks and writes in a single step.

### 2.2 Rules the schema enforces itself

Python checks a write first, so a normal request gets a clear error. The schema
checks it again inside SQLite, and those checks hold even when a code path
forgets one. The runs table shows most of them, at
[`migrations.py:475-500`](../../src/audrey/app_state/migrations.py#L475-L500):

```sql
CREATE TABLE app_runs (
  run_id             TEXT PRIMARY KEY,
  conversation_id    TEXT NOT NULL,
  user_id            TEXT NOT NULL,
  ...
  UNIQUE (run_id, conversation_id, user_id),
  CHECK (
    (status = 'running' AND completed_at IS NULL)
    OR (status != 'running' AND completed_at IS NOT NULL)
  ),
  FOREIGN KEY (conversation_id, user_id)
    REFERENCES app_conversations(conversation_id, user_id) ON DELETE CASCADE
);
```

- **`CHECK` constraints** limit values and tie columns together: A run is
  `running` exactly when it has no completion time.
- **The owner is part of every key.** A run points at its conversation by
  conversation id *and* owner id, and messages point at their run the same way
  ([`migrations.py:173`](../../src/audrey/app_state/migrations.py#L173 'FOREIGN KEY (conversation_id, user_id)')),
  so no row can attach to another person's conversation.
- **Cascades** make deleting a conversation delete its runs, its messages and
  everything attached to them, in one statement.
- **Unique indexes** allow one user message and one assistant message per run
  ([`migrations.py:182`](../../src/audrey/app_state/migrations.py#L182 'CREATE UNIQUE INDEX IF NOT EXISTS idx_app_messages_run_user')),
  and one message per position in a conversation.
- **Triggers** make a finished run's outcome final
  ([`migrations.py:530`](../../src/audrey/app_state/migrations.py#L530 'CREATE TRIGGER trg_app_runs_terminal_immutable')),
  and stop a Private conversation, Project or run from being made ordinary
  ([`PRIVATE_PROCESSING_GUARDS_SQL`](../../src/audrey/app_state/migrations.py#L5)).

### 2.3 Records, repositories and the owner

SQL against this database lives in the repositories. Routes, the run manager and
the pipeline call repository methods and get typed records back, such as the
[`StartedRun`](../../src/audrey/app_state/records.py#L147) that `begin_run`
returns. Records are frozen dataclasses: a copy of what the database held at one
moment, which no code can edit and mistake for a change to the data. That keeps
validation, the thread hop and the transaction in one place, where nothing
outside the repositories can skip them. The history route from §2.1 is the one
exception, and it only reads.

Every query made on a person's behalf filters by `user_id`, and that id always
comes from the authenticated caller, never from the request. A conversation that
doesn't exist and one that belongs to someone else therefore look the same, as
no row, and [`get_conversation`](../../src/audrey/routes/app/conversations.py#L342)
answers both with the same 404. A different answer for "exists, but not yours"
would let anyone probe which ids are real.

### 2.4 A turn's life in the database

Lesson 4 traced one message through the platform. Here it is as the database
sees it:

```text
Send
 -> begin_run                one write transaction
      check: the conversation is yours and not archived, your account is
             active, no other run is in progress
      INSERT the run           status 'running'
      INSERT your message and its attachments
      INSERT the assistant message, empty, status 'in_progress'
      UPDATE the conversation's timestamps
    COMMIT
 -> the answer streams       in memory only; the database still says 'running'
 -> finish_run               one write transaction
      UPDATE the run: outcome, models, token counts, only if still 'running'
      UPDATE the assistant message: its text, 'completed' or 'incomplete'
      INSERT the sources, tool calls and an archive receipt
    COMMIT
```

Four things to notice:

- **Checks run inside the transaction.** `begin_run` reads after
  `BEGIN IMMEDIATE`
  ([`_begin_run_sync`](../../src/audrey/app_state/repositories.py#L792)), so
  nothing can change between a check and the inserts that depend on it. Two tabs
  pressing Send at once produce one run and one 409 Conflict.
- **A run ends once.** `finish_run` refuses a second outcome in Python, its
  `UPDATE` matches only a run that is still running
  ([`_finish_run_sync`](../../src/audrey/app_state/repositories.py#L1093)), and
  the trigger from §2.2 backs both up. Whatever text had streamed is saved: the
  whole answer, or the part that arrived, marked incomplete.
- **The archive receipt commits with the answer.** It's a row saying this turn
  still has to reach the searchable archive
  ([`_insert_native_projection_locked`](../../src/audrey/app_state/repositories.py#L1275)),
  and Private turns never get one. Because both commit together, no turn is
  saved without its receipt. A later lesson follows the delivery.
- **A crash leaves an honest ending.** If the process dies mid-answer, the run
  still says `running`. Before serving anyone, the next startup's
  [`recover_interrupted_runs`](../../src/audrey/app_state/repositories.py#L1238)
  marks it failed with `server_restart`, and its message incomplete.

### 2.5 Migrations

The code changes with every release, but the database file stays, and every
existing copy has to reach the shape the new code expects before the first
request. [`MIGRATIONS`](../../src/audrey/app_state/migrations.py#L39) is the
schema's whole history, an ordered list of numbered SQL scripts. Each database
records the versions it has applied, and at startup the store applies whatever
is missing, each in its own transaction together with the row that records it
([`_migrate_locked`](../../src/audrey/app_state/store.py#L123)). A migration is
either applied and recorded, or neither. A failure stops startup: Audrey doesn't
serve requests until every migration has applied. The highest recorded version
is the **schema version**, which startup logs
([`main.py:115`](../../src/audrey/main.py#L115 '"application_store: ready sqlite=%s schema=%d",')).

Migrations come in three kinds:

1. **New tables and indexes**, the common case. Existing rows are untouched.
2. **New columns**, always with a default, so existing rows get a value and
   older code keeps working. SQLite can't add a column only if it's missing, so
   the store adds these one at a time and skips any that exist
   ([`_apply_additive_column_migration_locked`](../../src/audrey/app_state/store.py#L184)).
3. **Table rebuilds**, for changes SQLite can't make in place, such as a new
   allowed value in a `CHECK`. The migration renames the old table aside,
   creates the new one, copies the rows and drops the old table
   ([`migrations.py:471`](../../src/audrey/app_state/migrations.py#L471 'PRAGMA legacy_alter_table = ON;')).

A rebuild's dangerous step is the drop. Messages reference runs with
`ON DELETE CASCADE`, and with foreign keys on, renaming a table drags those
references along, so dropping the renamed table would delete every message. So
rebuilds take their own path
([`_apply_table_rebuild_migration_locked`](../../src/audrey/app_state/store.py#L155)):
Foreign keys go off before the transaction starts, the rename leaves other
tables pointing at the table's name, and a full foreign-key check runs before the
commit, rolling the whole rebuild back if any row now points at nothing.

The rules for the next migration follow:

- **Append, never edit.** A database that recorded a version never runs it
  again, so an edited migration reaches only new databases.
- **Give new columns defaults.**
- **Register the special kinds** in
  [`_FOREIGN_KEYS_OFF_MIGRATIONS`](../../src/audrey/app_state/store.py#L61) or
  [`_ADDITIVE_COLUMN_MIGRATIONS`](../../src/audrey/app_state/store.py#L62).
- **Back up first.** Nothing reverses a migration, and one that runs cleanly but
  does the wrong thing commits like any other.

### 2.6 Backups and rebuilds

Copying the file isn't a backup while Audrey runs. Recent commits may still sit
in the WAL file, and copying files that are changing can catch them at different
moments. The operator command `audrey-admin backup-app-state` uses SQLite's
online backup instead, which copies one consistent snapshot while Audrey keeps
working
([`_online_application_backup`](../../src/audrey/admin_cli.py#L133)). It never
overwrites an existing file, makes the copy readable only by its owner, opens
production read-only, checks the copy's integrity, and deletes the copy if
anything failed.

A backup is only as good as a restore you've seen work, so
`verify-app-state-backup`
([`_verify_application_backup`](../../src/audrey/admin_cli.py#L261)) restores
one into a throwaway file, never into production. It checks the restored copy's
integrity and foreign keys, and that its schema version and record counts match
the backup's.

Audrey doesn't back itself up. The operator takes a backup, and verifies it,
before deploying a release with new migrations. The one bulk import an operator
can run won't start without a backup path, and takes that backup itself before
writing
([`admin_cli.py:367`](../../src/audrey/admin_cli.py#L367 '_online_application_backup(path, backup_to)')).

The archive receipts close the loop on Lesson 4's distinction.
[`ChatProjectionPromoter.rebuild`](../../src/audrey/chat_projection.py#L92)
marks every receipt undelivered, and the worker delivers every finished turn
again. A turn delivered twice does no harm, because each delivery carries the
receipt's stable id. Nothing works the other way: The archive can't rebuild the
database. **Back up authorities; rebuild projections.**

## 3. Comprehension questions

**1. Bob requests `GET /api/conversations/con_…` with an id copied from
Alice's screen. What comes back, and why?**

A 404, the same answer as for an id that doesn't exist. The query filters by
Bob's user id, taken from the authenticated caller, so Alice's conversation
isn't found (§2.3). A 403 would confirm that the id is real. And if a write path
ever forgot the filter, the foreign keys that carry the owner would still stop
Bob's rows from attaching to Alice's conversation (§2.2).

**2. The power fails once while `begin_run` is between its inserts, and once
while an answer is streaming. What does the database hold after each
restart?**

After the first, nothing from that turn, because the transaction never
committed. After the second, the run and an empty assistant message, since the
streamed text lived only in memory. Before anyone sees them, startup marks the
run failed with `server_restart` and the message incomplete (§2.4).

**3. A release adds a new allowed run mode. Why is that a table rebuild, and
why must foreign keys be off while it runs?**

The allowed modes are a `CHECK` constraint, which SQLite can't change in place,
so the table is recreated and its rows copied (§2.5). With foreign keys on,
renaming the old table would drag the messages' references along, and dropping
it would delete every message attached to a run. The rebuild path turns them
off first and checks every foreign key before it commits.

**4. A teammate adds a column to `app_messages` by editing the migration that
creates the table, instead of adding a new migration. The tests pass. What
happens on the server?**

Nothing. The server's database recorded that migration long ago and never runs
it again, so the column never appears there, and the first query that uses it
fails. The tests passed because they start from empty databases. A new migration
that adds the column, with a default, is the fix (§2.5).

**5. An operator copies `audrey_app.sqlite` with `cp` while Audrey runs, and
later restores it. Recent conversations are missing, or the file won't open.
Why, and what should they have done?**

Recent commits can sit in the WAL file until a checkpoint moves them, so the
main file alone misses them, and copying files while they change can catch them
at different moments. `audrey-admin backup-app-state` takes one consistent
snapshot while Audrey runs, and `verify-app-state-backup` proves it restores
(§2.6).

## When you're ready for the next lesson

Every query in this lesson filtered by a `user_id` taken from an authenticated
caller. How Audrey decides who that caller is, and what they may use, gets a
lesson of its own later. The next lesson returns to the answer pipeline and
opens the model layer:
[Lesson 6 — The model layer I: choosing a model](lesson-06b-the-model-layer.md).
It traces what happens when a pipeline node says "I need a general-purpose
model": which choices a person may pick, how the registry ranks candidates, and
how the health tracker steps around failing ones. The lesson after it follows
the call itself to Ollama.
