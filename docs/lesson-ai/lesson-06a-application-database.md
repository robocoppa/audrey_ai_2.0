# Lesson 6a — The application database

Lesson 4 put the application database on the system map as an *authority*, the
place where accounts, conversations, messages and runs are true by definition,
and Lesson 5 showed startup opening it before anything else runs. This lesson
looks inside. It answers two questions: Where does a conversation live, and what
keeps that data safe while the code around it keeps changing?

The answer rests on three guarantees, and the read-along takes them in turn:
writes that happen completely or not at all, rules the database enforces itself,
and a schema that can change under existing rows.

## 1. Context

### 1.1 The canonical records

Audrey's **canonical records** are the original copy of each person's account,
conversations, settings and Projects. Every other copy defers to them: The
browser shows one, and the searchable chat archive keeps one arranged for
search, but when a copy and the records disagree, the records are right.

An account holds the person's email, display name and role, whether an
administrator has approved or disabled them, and any personal access tokens that
let scripts and bots act for them. A conversation is a titled thread that
remembers the model chosen for it. Its messages are the person's questions and
Audrey's answers, in order, with their attachments, the sources each answer drew
on and the tools it called, and every turn adds a **run**, which records the
model that answered, the tokens it used and how the turn ended. Settings are a
time zone, a persona and a preferred response style. A Project gathers
conversations and files under its own instructions.

Audrey uses these records on almost every request. Each request starts by
loading the caller's account and checking that it's active, and the
conversation list, an open conversation and a Project page are all read from
them. Sending a message writes to them when the turn starts and again when the
answer is saved (§2.4). In between, Audrey builds what the model reads from the
records: the conversation's stored messages
([`_history_messages`](../../src/audrey/routes/app/runs.py#L1278)), leaving out
answers that didn't finish, plus the person's settings and, in a Project, its
instructions. The browser sends only the newest message (Lesson 4), so nothing
it holds can change what the model is told about the past.

The records live in SQLite: one file, named by
[`application.sqlite_path`](../../config.yaml#L76) and opened through Python's
built-in `sqlite3` module. There's no database server to run. That suits one
machine serving a handful of people, with one container writing the data
(Lesson 5), and SQLite still provides what matters here: transactions and
constraints.

What lives in it:

| What it holds | Tables |
|---|---|
| Accounts and access | `app_users`, `external_identities`, `personal_access_tokens` |
| Conversations | `app_conversations`, `app_runs`, `app_messages` |
| Preferences | `user_preferences` |
| Archive hand-offs | `app_chat_projections` |
| Projects and other features | `app_projects`, `app_tax_cases`, `app_private_tasks` and more |

This lesson reads the database through conversations, messages and runs, the
tables every chat request touches. What it teaches about transactions, schema
rules and migrations applies to every table above. Accounts and access, the
delivery of archive hand-offs, and Projects and the other features each get
their own explanation in later lessons.

Some data lives elsewhere: Uploaded files are tracked in their own SQLite
database, saved memories live in Qdrant, and the searchable chat archive belongs
to the tools sidecar.

### 1.2 The files

| Job | File |
|---|---|
| The connection, its settings and the migration runner | [`store.py`](../../src/audrey/app_state/store.py) |
| Every schema change, in order | [`migrations.py`](../../src/audrey/app_state/migrations.py) |
| The commit-or-rollback boundary for writes | [`transactions.py`](../../src/audrey/app_state/transactions.py) |
| Conversations, messages, runs, preferences and archive receipts | [`repositories.py`](../../src/audrey/app_state/repositories.py) |
| The typed records callers get back | [`records.py`](../../src/audrey/app_state/records.py) |

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
repository method (async)            # what routes call
 -> asyncio.to_thread                # the event loop keeps running
     -> validate the input           # no lock needed yet
     -> take the store's lock        # one thread at a time
         -> BEGIN IMMEDIATE          # the write lock, up front
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
 -> begin_run                    # one write transaction
      check: the conversation is yours and not archived, your account is
             active, no other run is in progress
      INSERT the run             # status 'running'
      INSERT your message and its attachments
      INSERT the assistant message, empty, status 'in_progress'
      UPDATE the conversation's timestamps
    COMMIT
 -> the answer streams           # to the browser, held in memory, not yet saved
 -> finish_run                   # one write transaction
      UPDATE the run: outcome, models, token counts, only if still 'running'
      UPDATE the assistant message: its text, 'completed' or 'incomplete'
      INSERT the sources, tool calls and an archive receipt
    COMMIT
```

The answer is the assistant's reply, and streaming is how it reaches the screen
(Lesson 4). The model writes the reply in pieces, usually a few words each. Each
piece reaches the run as an event, and so do the sources consulted, the tool
calls and the models used along the way. The run holds those events in memory,
and the browser's stream reads them from there, so the person watches the reply
being written. The database sees none of it until the run ends: The assistant
message stays empty, and then
[`_persist_terminal`](../../src/audrey/routes/app/runs.py#L1017) joins the text
pieces and hands them to `finish_run` with everything else the run collected.
That's one write per turn, however long the answer.

The run belongs to Audrey's process, not to the browser's connection, so
closing or reloading the tab doesn't stop the answer. A reloaded page reads the
conversation from the database, finds the run still `running`, says Audrey is
still answering, and shows the answer once it's saved.

Four things to notice:

- **Checks run inside the transaction.** `begin_run` reads after
  `BEGIN IMMEDIATE`
  ([`_begin_run_sync`](../../src/audrey/app_state/repositories.py#L792)), so
  nothing can change between a check and the inserts that depend on it. Two tabs
  pressing Send at once produce one run and one 409 Conflict.
- **A run ends once.** It ends in one of three ways: The model finishes
  (`succeeded`), the person presses Stop (`cancelled`), or something fails
  (`failed`). Whichever comes first, `finish_run` saves the text written by
  then: the whole answer, marked `completed`, or the part written before the
  Stop or the failure, marked `incomplete`. `finish_run` refuses a second
  outcome in Python, its `UPDATE` matches only a run that is still running
  ([`_finish_run_sync`](../../src/audrey/app_state/repositories.py#L1093)), and
  the trigger from §2.2 backs both up.
- **The archive receipt commits with the answer.** It's a row saying this turn
  still has to reach the searchable archive
  ([`_insert_native_projection_locked`](../../src/audrey/app_state/repositories.py#L1275)),
  and Private turns never get one. Because both commit together, no turn is
  saved without its receipt. A later lesson follows the delivery.
- **A crash leaves an honest ending.** If the process dies mid-answer, the text
  in memory dies with it, and the database still holds what `begin_run` wrote:
  a run that says `running` and an empty assistant message. Before serving
  anyone, the next startup's
  [`recover_interrupted_runs`](../../src/audrey/app_state/repositories.py#L1238)
  marks the run failed with `server_restart` and the message incomplete. The
  person sees an interrupted turn they can retry, not an answer that never
  comes.

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

Most migrations add new tables, indexes or columns and leave existing rows as
they are. The rules for the next one follow:

- **Append, never edit.** A database that recorded a version never runs it
  again, so an edited migration reaches only new databases.
- **Give new columns defaults,** so existing rows get a value and older code
  keeps working.
- **Back up first.** Nothing reverses a migration, and one that runs cleanly but
  does the wrong thing commits like any other.

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

**3. A teammate adds a column to `app_messages` by editing the migration that
creates the table, instead of adding a new migration. The tests pass. What
happens on the server?**

Nothing. The server's database recorded that migration long ago and never runs
it again, so the column never appears there, and the first query that uses it
fails. The tests passed because they start from empty databases. A new migration
that adds the column, with a default, is the fix (§2.5).

## When you're ready for the next lesson

Every query in this lesson filtered by a `user_id` taken from an authenticated
caller. The next lesson is about that caller:
[Lesson 6b — Identity and access](lesson-06b-identity-and-access.md). It covers
how Audrey knows who is calling, through Cloudflare Access, a personal access
token or its own service token, and what decides what each caller may do:
account status, groups, token scopes and which models they may use.
