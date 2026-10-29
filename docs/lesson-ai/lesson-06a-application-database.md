# Lesson 6a — The application database

Lesson 4 put the application database on the system map as an *authority*: the
place where accounts, conversations, messages and runs are true by definition.
Lesson 5 showed startup opening it before anything else touches the data. This
lesson opens the database itself.

It answers two questions. Where does a conversation actually live? And what
makes it safe to change that storage while real people's data is in it? The
answers are four guarantees, each with a mechanism behind it: writes that happen
completely or not at all, rules the database enforces even when the Python code
has a bug, a schema that can change under existing rows, and backups that have
been restored before anyone needs them.

## 1. Context

### 1.1 Why SQLite, and what lives in it

Most databases you may have heard of, such as PostgreSQL, are *servers*: a
separate program that applications connect to over the network. **SQLite** is
not a server. It's a library that reads and writes a single file, and Python
ships with it as the `sqlite3` module. Audrey's application database is the file
named by [`application.sqlite_path`](../../config.yaml#L76), on the `/data`
volume that Lesson 5 showed only Audrey writes.

That fits Audrey, which runs on one machine for a handful of people. What a
database server is best at, many machines writing at once, isn't needed. SQLite
has what Audrey does need: transactions that survive a power cut, constraints
checked on every write, and a way to back up a database while it's in use.

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

Three things are *not* in it. The records of uploaded files live in a second
SQLite database, the uploads database; saved memories live in Qdrant; and the
searchable chat archive belongs to the tools sidecar. Lesson 4 sorted those into
authorities and projections, and this lesson returns to that distinction only at
the end (§2.8), to show how the archive is rebuilt from the database.

### 1.2 The files

| Job | File | What it holds |
|---|---|---|
| The store | [`store.py`](../../src/audrey/app_state/store.py) | `ApplicationStore`: the connection, its lock and settings, and the migration runner. Its identity and administration methods belong to a later lesson |
| The schema's history | [`migrations.py`](../../src/audrey/app_state/migrations.py) | `MIGRATIONS`: every change to the schema, in order |
| The write boundary | [`transactions.py`](../../src/audrey/app_state/transactions.py) | `write_transaction`: commit or roll back, as a context manager |
| The SQL | [`repositories.py`](../../src/audrey/app_state/repositories.py) | Conversations, messages and runs; preferences; archive receipts |
| What callers get back | [`records.py`](../../src/audrey/app_state/records.py) | Frozen dataclasses such as `ConversationRecord` and `RunRecord` |
| | [`titles.py`](../../src/audrey/app_state/titles.py) | The title a new conversation takes from its first message |
| The archive hand-off | [`chat_projection.py`](../../src/audrey/chat_projection.py) | The worker that delivers archive receipts |
| Operator commands | [`admin_cli.py`](../../src/audrey/admin_cli.py) | `audrey-admin`, including the online backup and its verification |

The other repositories in `src/audrey/app_state/`, for Projects, tax review,
Private tasks, stored responses and imported history, share the same connection
and lock and follow the same rules. The lessons on those features read them.

## 2. Read-along

The read-along starts with the two ideas everything else rests on: transactions
and the write-ahead log (§2.1–2.2). Then it reads how Audrey uses them
(§2.3–2.6), how the schema changes (§2.7), and how the data is backed up (§2.8).

### 2.1 Concept spotlight: transactions

A database holds **tables** of **rows**, and SQL statements read them (`SELECT`)
or change them (`INSERT`, `UPDATE`, `DELETE`). One statement changes one table,
but one user action often needs several changes. When you press Send, Audrey
records a **run** (one attempt at answering, with its status, model and token
counts), your message, a record of each file you attached, an empty assistant
message for the answer to fill, and new timestamps on the conversation.

Suppose the power fails after the second of those five writes. The database
would hold a message tied to a run that will never answer, and every piece of
code that reads it later would have to cope with a shape that should never
exist. A **transaction** removes that possibility. It groups statements so that
they take effect together or not at all:

```text
BEGIN
  INSERT the run
  INSERT your message
  INSERT its attachments
  INSERT the empty assistant message
  UPDATE the conversation's timestamps
COMMIT          all five become visible at once
```

`ROLLBACK` in place of `COMMIT` discards all five. So does a crash before
`COMMIT`: The next time the database opens, SQLite throws away whatever the
unfinished transaction wrote. The commit is the line. Before it, nothing
happened; after it, everything did, and it survives a power cut. Those
properties are often listed as ACID: **atomic** (all or nothing), **consistent**
(the database's rules hold at every commit), **isolated** (no one sees a
half-finished transaction) and **durable** (a commit survives a crash).

In Python's `sqlite3` module, a connection runs a statement with `execute` and
ends a transaction with `commit()` or `rollback()`. Values never go into the SQL
text itself. They travel separately, as parameters, each marked by a `?`:

```python
connection.execute(
    "UPDATE app_conversations SET title = ? WHERE user_id = ? AND conversation_id = ?",
    (title, user_id, conversation_id),
)
```

The database treats a parameter strictly as data. A conversation titled
`'; DROP TABLE app_runs; --` stays a strange title, where the same text pasted
into the SQL would have become a command. That attack is **SQL injection**, and
parameters close it. The store passes every value that comes from a request
this way.

### 2.2 Concept spotlight: one writer, many readers

A transaction in progress raises a second question: What do other connections
see meanwhile, and can they read at all? SQLite's answer depends on its
**journal mode**, and Audrey uses **write-ahead logging**, WAL for short.

In WAL mode a writer doesn't change the database file in place. It appends the
pages it changes to a second file beside it, `audrey_app.sqlite-wal`, and a
commit is a marker in that log. A reader notes how far the log reached when it
started, and reads that **snapshot**: the main file plus the log up to the mark.
It never sees half of a transaction, and it never waits for the writer. From
time to time SQLite copies logged pages back into the main file, a
**checkpoint**. (A third small file, `audrey_app.sqlite-shm`, is an index into
the log that the connections share.) Three rules follow:

- Any number of connections can read at once.
- Only one connection writes at a time. The others wait for the **write lock**.
- Readers don't block the writer, and the writer doesn't block readers.

The store's constructor switches WAL on, along with two other settings, at
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

`row_factory = sqlite3.Row` makes result rows readable by column name, as
`row["title"]`. A **PRAGMA** is SQLite's statement for settings:

- **`foreign_keys = ON`.** A foreign key says "this column must match a row in
  that table" (§2.4). For compatibility with old databases, SQLite checks
  foreign keys only on connections that ask, so every connection that writes
  must switch it on.
- **`busy_timeout = 5000`.** When another connection holds the write lock, wait
  up to five seconds for it rather than failing at once.
- **`journal_mode = WAL`.** Unlike the other two, this setting is saved in the
  database file, so it holds for every later connection too.

Then the constructor brings the schema up to date (§2.7) and creates the
repositories, all sharing one connection and one lock (§2.3).

One more term: **`BEGIN IMMEDIATE`**. A plain `BEGIN` doesn't take the write
lock until the transaction first writes. A transaction that reads, decides and
then writes can find, at its first write, that another connection wrote in
between; SQLite then refuses the write, and the transaction fails halfway
through, after decisions based on what it read. `BEGIN IMMEDIATE` takes the
write lock at the start, so everything the transaction reads stays current until
it commits. Every write in the store but one begins this way, and the exception
(§2.3) is a single statement.

Who else opens the file? A **reader**: The service route that lets the
chat-history tool read a saved conversation opens its own read-only connection,
in [`_connection`](../../src/audrey/routes/tool_history.py#L66), and reads the
conversation inside one read transaction, in
[`_read_page`](../../src/audrey/routes/tool_history.py#L91):

```python
# A single read transaction prevents a delete or purge between the
# owner lookup and content read from producing an inconsistent page.
connection.execute("BEGIN")
```

Every query in that transaction reads the same snapshot, so a deletion that
commits halfway through can't produce a page assembled from two different
moments. And a **writer**: The `audrey-admin` command, which an operator runs
inside the `audrey` container to grant the first administrator or import an
exported chat history. Its writes and Audrey's take turns at the write lock, and
the busy timeout makes each wait for the other instead of failing.

### 2.3 One connection, one lock, one thread hop

The store opens one connection and shares it among all its repositories. SQLite
allows one writer at a time anyway, so a second connection inside the same
process would only add a second place to wait. Sharing raises two problems:

1. **`sqlite3` calls block.** A query runs to completion before Python moves on.
   Run inside an `async def`, it would freeze the event loop, and every other
   request with it: Lesson 1's "sync call inside an async function" hazard. So
   each repository method that request handlers call is an `async` wrapper that
   moves the real work onto a worker thread with `asyncio.to_thread`.
2. **Worker threads run at the same time, and one connection must not be used
   by two threads at once.** `check_same_thread=False` tells `sqlite3` not to
   enforce that rule itself, which leaves it to the store: Every method takes
   `self._lock` before touching `self._conn`. The lock is a `threading.RLock`,
   a re-entrant lock, which a thread that already holds it can take again
   without deadlocking on itself.

Preferences are the simplest table, so they show the whole pattern.
[`PreferencesRepository.replace`](../../src/audrey/app_state/repositories.py#L82)
is the `async` method routes call. It hands its arguments to `_replace_sync`,
which validates the timezone and the persona's length and then writes, at
[`repositories.py:115-132`](../../src/audrey/app_state/repositories.py#L115-L132):

```python
with self._lock:
    self._conn.execute("BEGIN IMMEDIATE")
    try:
        cursor = self._conn.execute(
            "UPDATE user_preferences SET timezone = ?, persona = ?, "
            "response_preferences_json = ?, updated_at = ? WHERE user_id = ?",
            (timezone, persona, preferences_json, now, user_id),
        )
        if cursor.rowcount == 0:
            self._conn.rollback()
            return None
        row = self._preference_row_locked(user_id)
        assert row is not None
        self._conn.commit()
    except BaseException:
        self._conn.rollback()
        raise
return _preferences_from_row(row)
```

The shape every write in the store shares:

```text
replace(...)                              # async; what routes call
 -> asyncio.to_thread(_replace_sync)      # the event loop keeps running
     -> validate and normalize            # no lock needed yet
     -> with self._lock                   # one thread at a time on the connection
         -> BEGIN IMMEDIATE               # the write lock, up front
         -> UPDATE ... WHERE user_id = ?
         -> no row changed: rollback, return None
         -> read the row back, commit
         -> any exception: rollback, re-raise
 -> UserPreferences                       # a typed record, not a database row
```

Three details carry weight. **Validation runs before the lock**, because
checking a timezone needs no database and shouldn't hold up anyone else's query.
**`rowcount` answers "did this exist?"** An `UPDATE` that matches no row changes
nothing, so the method rolls back and returns `None`, which a route turns into a
404 (§2.5). And **the names tell you where you are**: A `_sync` method runs on a
worker thread, and a `_locked` helper such as `_preference_row_locked` expects
its caller to hold the lock already.

The rollback isn't tidiness. A statement that fails normally undoes only itself,
not the transaction around it: The connection stays inside that transaction until
something commits or rolls it back. On a connection the whole process shares, an
abandoned transaction is everyone's problem, because the next `BEGIN IMMEDIATE`,
in any repository, fails with "cannot start a transaction within a transaction".
So every write ends in exactly one commit or one rollback, whichever way it
leaves, and `except BaseException` makes sure even unusual exceptions count.

Short writes get the same boundary from a helper,
[`transactions.py:10-20`](../../src/audrey/app_state/transactions.py#L10-L20):

```python
@contextmanager
def write_transaction(connection: sqlite3.Connection) -> Iterator[None]:
    """Commit one write or roll it back without leaving the connection busy."""
    connection.execute("BEGIN IMMEDIATE")
    try:
        yield
        connection.commit()
    except BaseException:
        if connection.in_transaction:
            connection.rollback()
        raise
```

`@contextmanager` builds a context manager, the construct Lesson 1 introduced,
from a generator: The code before `yield` runs when the `with` block is entered,
the code after it runs when the block ends normally, and an exception inside the
block surfaces at the `yield`. A caller writes
`with self._lock, write_transaction(self._conn):`, which enters the lock first
and leaves it last, so the transaction always ends before the lock is released.

#### Writing only if nothing changed

Audrey names a new conversation twice. Inside the transaction that starts its
first run, it takes its first message, tidied by
[`fallback_conversation_title`](../../src/audrey/app_state/titles.py#L11), as a
title. After that first answer succeeds, a model writes a better one. That takes
a few seconds, and the person may rename the conversation meanwhile. Holding the
lock while a model thinks would stall every other query, so the write checks
instead that nothing has changed, at
[`repositories.py:302-304`](../../src/audrey/app_state/repositories.py#L302-L304):

```python
"UPDATE app_conversations SET title = ?, updated_at = ? "
"WHERE user_id = ? AND conversation_id = ? "
"AND title = ? AND updated_at = ? AND archived_at IS NULL "
```

The expected title and `updated_at` are the conversation as it stood when its
first run started. If the person renamed it since, no row matches, `rowcount` is
0, and the generated title is quietly dropped. This is **optimistic
concurrency**: No lock is held during the slow work, and the write checks its
own assumption, losing gracefully if it no longer holds. It's the store's one
write without `BEGIN IMMEDIATE`, and it doesn't need one, because a single
`UPDATE` is its own transaction and checks its conditions as it writes.

### 2.4 Rules the schema enforces itself

Python checks a write before sending it, and the schema checks it again inside
SQLite. Python's checks give clear errors early; the schema's still hold when a
code path forgets one, a migration copies rows, or a script writes a table
directly. Here is `app_runs` from the migration that last rebuilt it (later ones
only added columns), at
[`migrations.py:475-500`](../../src/audrey/app_state/migrations.py#L475-L500):

```sql
CREATE TABLE app_runs (
  run_id             TEXT PRIMARY KEY,
  conversation_id    TEXT NOT NULL,
  user_id            TEXT NOT NULL,
  ...
  status             TEXT NOT NULL CHECK (status IN
                       ('running', 'succeeded', 'cancelled', 'failed')),
  started_at         TEXT NOT NULL,
  completed_at       TEXT,
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

Read it as a list of promises:

- **`CHECK` constraints** reject values outside a list and tie columns together.
  The last one says a run is `running` exactly when it has no completion time.
- **The owner is part of the key.** A run refers to its conversation by the
  *pair* of conversation id and owner id, and `app_conversations` declares that
  pair unique, so a run can only belong to a conversation with the same owner.
  Messages do the same one level down, at
  [`migrations.py:173-176`](../../src/audrey/app_state/migrations.py#L173-L176):

  ```sql
  FOREIGN KEY (conversation_id, user_id)
    REFERENCES app_conversations(conversation_id, user_id) ON DELETE CASCADE,
  FOREIGN KEY (run_id, conversation_id, user_id)
    REFERENCES app_runs(run_id, conversation_id, user_id) ON DELETE CASCADE
  ```

  A bug that mixed up two people's ids would fail loudly at the insert, rather
  than file Alice's message under Bob's conversation.
- **`ON DELETE CASCADE`** makes deletion follow the keys. Deleting a
  conversation deletes its runs and messages, and the messages take their
  attachments, sources and tool calls with them, all in one transaction.
- **Unique indexes** state "at most one" rules. Two partial ones, which apply
  only to rows matching a condition, allow at most one user message and one
  assistant message per run
  ([`migrations.py:182-186`](../../src/audrey/app_state/migrations.py#L182-L186 'CREATE UNIQUE INDEX IF NOT EXISTS idx_app_messages_run_user')),
  and `UNIQUE (conversation_id, sequence_no)` keeps two messages from sharing a
  position in a conversation.
- **Triggers** run SQL when a row changes. This one, at
  [`migrations.py:530-535`](../../src/audrey/app_state/migrations.py#L530-L535),
  makes a run's outcome final:

  ```sql
  CREATE TRIGGER trg_app_runs_terminal_immutable
  BEFORE UPDATE OF status ON app_runs
  WHEN OLD.status != 'running'
  BEGIN
    SELECT RAISE(ABORT, 'terminal run outcome is immutable');
  END;
  ```

  Once a run has succeeded, failed or been cancelled, any change to its status
  aborts. Private processing has guards of the same kind: Triggers refuse to
  turn a Private conversation or Project into an ordinary one, and freeze each
  run's Private flag
  ([`PRIVATE_PROCESSING_GUARDS_SQL`](../../src/audrey/app_state/migrations.py#L5)).

Python repeats most of these checks (§2.6), so a normal request gets a clear
error message. The schema's version of each rule is the one that still holds on
the day Python's doesn't.

### 2.5 Records and repositories: SQL stays inside

All of Audrey's SQL against this database lives in the repositories. Routes, the
run manager and the pipeline call repository methods and get typed records back;
none of them holds the connection. The history route from §2.2 is the one
exception, and it can only read.

A **record** is a frozen, slotted dataclass, the kind Lesson 1 introduced. Here
is what `begin_run` returns, at
[`records.py:146-153`](../../src/audrey/app_state/records.py#L146-L153):

```python
@dataclass(frozen=True, slots=True)
class StartedRun:
    """Records created atomically before Audrey starts streaming a response."""

    conversation: ConversationRecord
    run: RunRecord
    user_message: MessageRecord
    assistant_message: MessageRecord
```

Frozen matters here. A record is a copy of what the database held at one moment;
if code could assign `record.title = ...`, it might believe it had changed the
conversation. With frozen records, the only way to change data is a repository
method, which is where the transaction lives. Each record is built from a row by
a small function such as
[`_conversation_from_row`](../../src/audrey/app_state/repositories.py#L1793).
The conversion is spelled out because SQLite's types are few: It has no
boolean, so `private_required` is stored as 0 or 1 and becomes `bool(...)`, and
SQL's `NULL` becomes `None`.

The split earns its keep because every rule in this lesson lives inside these
methods, and code that can't reach the connection can't skip them. One place
validates input, moves work off the event loop, bounds transactions and scopes
queries by owner.

#### Owner scoping: one answer for "missing" and "not yours"

Every query made on a person's behalf filters by `user_id`, and that id comes
from the authenticated caller, never from the request body. Here is the route
that returns one conversation, at
[`conversations.py:341-353`](../../src/audrey/routes/app/conversations.py#L341-L353):

```python
@router.get("/{conversation_id}", response_model=ConversationResponse)
async def get_conversation(
    conversation_id: str,
    request: Request,
    principal: Principal = Depends(_conversation_access),
) -> ConversationResponse:
    record = await _store(request).conversations.get(
        user_id=principal.user_id,
        conversation_id=conversation_id,
    )
    if record is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return _conversation_response(record)
```

`principal` is the authenticated caller, the `Principal` that Lesson 4 showed
authentication producing. The repository's query, in
[`_conversation_row_locked`](../../src/audrey/app_state/repositories.py#L1319),
asks for a row matching both ids. A conversation that doesn't exist and one that
belongs to someone else give the same result, no row, so the caller gets the
same 404 either way. That's deliberate: A different answer for "exists, but not
yours", such as 403 Forbidden, would let anyone test which ids are real.

### 2.6 A turn's life in the database

Lesson 4 traced one message through the whole platform. Here it is as the
database sees it:

```text
Send
 -> begin_run                  # one write transaction
      check: the conversation is yours and not archived, your account is
             active, no other run is in progress
      first message and no title: the fallback title from your message
      INSERT the run           # status 'running'
      INSERT your message      # the next sequence number
      INSERT its attachments, and the assistant message: empty, 'in_progress'
      UPDATE the conversation's timestamps
    COMMIT -> StartedRun
 -> the answer streams         # in memory only; the database still says 'running'
 -> finish_run                 # one write transaction
      UPDATE the run: outcome, models, token counts, only if still 'running'
      UPDATE the assistant message: its text, 'completed' or 'incomplete'
      INSERT the models used, the sources, the tool calls, an archive receipt
    COMMIT -> FinishedRun
```

**The checks run inside the transaction.** `begin_run` does all its reading
after `BEGIN IMMEDIATE`, so nothing can change between a check and the inserts
that depend on it. Two clicks on Send, or two browser tabs, can't both pass the
"no other run" check: The second waits for the lock, finds the first one's run,
and gets 409 Conflict. The next sequence number, `MAX(sequence_no) + 1`, is safe
for the same reason, with the unique constraint from §2.4 behind it.

**One outcome, guarded twice.** `finish_run` refuses a second ending in Python,
then writes with a condition that would refuse it again, in
[`_finish_run_sync`](../../src/audrey/app_state/repositories.py#L1093):

```python
if str(existing["status"]) != "running":
    raise RunAlreadyTerminalError("run already has a terminal outcome")
cursor = self._conn.execute(
    "UPDATE app_runs SET status = ?, completed_at = ?, finish_reason = ?, "
    "error_code = ?, virtual_model = ?, concrete_model = ?, "
    "prompt_tokens = ?, completion_tokens = ? "
    "WHERE user_id = ? AND run_id = ? AND status = 'running'",
    ...
```

The trigger from §2.4 stands behind both. Whatever text had streamed is saved
with the outcome: the whole answer when the run succeeded, or the part that
arrived, marked `incomplete`, when it was stopped or failed.

**The receipt rides in the same transaction.** Before committing, `finish_run`
inserts an archive receipt with
[`_insert_native_projection_locked`](../../src/audrey/app_state/repositories.py#L1275):
a row in `app_chat_projections` saying this turn still has to reach the
searchable archive. (Private turns never get one.) A worker delivers receipts
later, and a later lesson follows that delivery. What matters here is that the
receipt commits with the answer, so no turn is ever saved but forgotten by the
archive, or announced to the archive but not saved.

**After a crash.** A run that was streaming when the process died is left saying
`running`, with an empty assistant message, because the streamed text lived only
in memory. Lesson 5 showed startup calling
[`recover_interrupted_runs`](../../src/audrey/app_state/repositories.py#L1238)
before it serves anyone. In one transaction, it marks each such message
incomplete, then each such run failed with the error code `server_restart`, and
writes their archive receipts. The order matters: It finds the messages through
their runs' `running` status, so the messages have to change first.

### 2.7 Migrations: changing the schema under real data

The code changes with every release; the database file stays. A release may need
a new table, a new column or a new allowed value, and every existing database,
whether the server's, a laptop's or a restored backup, has to reach the shape
the new code expects before the first request, without losing a row.

[`MIGRATIONS`](../../src/audrey/app_state/migrations.py#L39) is the schema's
whole history: an ordered tuple of `(version, sql)` pairs, from the first
`CREATE TABLE` onward. Each database records the versions it has applied in a
table of its own, `app_schema_migrations`, and the runner applies whatever is
missing, at
[`store.py:132-149`](../../src/audrey/app_state/store.py#L132-L149):

```python
for version, sql in MIGRATIONS:
    if version in applied:
        continue
    stamp = _utc_now()
    if version in _ADDITIVE_COLUMN_MIGRATIONS:
        self._apply_additive_column_migration_locked(version, stamp)
        continue
    if version in _FOREIGN_KEYS_OFF_MIGRATIONS:
        self._apply_table_rebuild_migration_locked(version, sql, stamp)
        continue
    try:
        self._conn.executescript(
            "BEGIN IMMEDIATE;\n"
            f"{sql}\n"
            "INSERT INTO app_schema_migrations(version, applied_at) "
            f"VALUES ({version}, '{stamp}');\n"
            "COMMIT;"
        )
    ...
```

`executescript` runs a script of several statements in one call. Each missing
version runs in its own transaction together with the row that records it, so a
migration is either applied and recorded, or neither. (The f-string breaks
§2.1's rule safely: The version and the timestamp come from Audrey's own code,
never from a request.) The **schema version** is the highest version recorded
([`schema_version`](../../src/audrey/app_state/store.py#L233)), and startup logs
it ([`main.py:115`](../../src/audrey/main.py#L115 '"application_store: ready sqlite=%s schema=%d",')). If a migration
fails, its transaction rolls back, nothing is recorded, and the exception stops
startup: Audrey won't serve requests until every migration has applied.

Migrations come in three kinds:

1. **New tables and indexes**, with `CREATE TABLE IF NOT EXISTS` and
   `CREATE INDEX IF NOT EXISTS`. Most migrations are this kind, and existing
   rows are untouched.
2. **New columns**, with `ALTER TABLE ... ADD COLUMN` and always a default, so
   existing rows get a value and code written before the column keeps working.
   SQLite has no "add this column unless it's there", so the store adds these
   columns one at a time and skips any that already exist
   ([`_apply_additive_column_migration_locked`](../../src/audrey/app_state/store.py#L184)).
3. **Table rebuilds**, for changes SQLite can't make in place, such as
   changing a `CHECK`. Allowing a new run mode, and adding a
   `requested_model_id` column filled in from existing rows, took one, at
   [`migrations.py:471-528`](../../src/audrey/app_state/migrations.py#L471-L528):

```sql
PRAGMA legacy_alter_table = ON;
ALTER TABLE app_runs RENAME TO app_runs_before_model_selection;
CREATE TABLE app_runs (
  ...
INSERT INTO app_runs
  ...
FROM app_runs_before_model_selection;
DROP TABLE app_runs_before_model_selection;
```

It renames the old table aside, creates the new one, copies every row (using
`CASE` expressions to fill the new column), drops the old table, and recreates
the table's index and trigger.

The dangerous step is the `DROP`. `app_messages` has a foreign key to `app_runs`
with `ON DELETE CASCADE`. With foreign keys on, SQLite rewrites that reference
when a table is renamed, so the messages would follow the old table to its new
name, and dropping it would delete every message attached to a run. So rebuilds
take their own path, at
[`store.py:163-182`](../../src/audrey/app_state/store.py#L163-L182):

```python
self._conn.execute("PRAGMA foreign_keys = OFF")
try:
    self._conn.executescript(f"BEGIN IMMEDIATE;\n{sql}\n")
    violations = self._conn.execute("PRAGMA foreign_key_check").fetchall()
    if violations:
        raise sqlite3.IntegrityError(
            f"migration {version} produced foreign-key violations"
        )
    ...
finally:
    self._conn.execute("PRAGMA legacy_alter_table = OFF")
    self._conn.execute("PRAGMA foreign_keys = ON")
```

Three moves make it safe. **Foreign keys go off before the transaction
begins**, because SQLite ignores that setting inside a transaction; with them
off, the `DROP` deletes no children. **`legacy_alter_table = ON`**, set by the
migration itself, keeps the rename from rewriting other tables' references, so
the messages go on referring to the name `app_runs`, which the new table takes.
And **`PRAGMA foreign_key_check` runs before the commit**: With enforcement off,
nothing stopped a bad copy, so the method checks the whole database. If any row
now points at nothing, it raises, and the rollback undoes the entire rebuild.
Either way, the `finally` turns enforcement back on.

The rules for the next migration follow:

- **Append; never edit.** A database that has recorded version N never runs N
  again, so an edit to an old migration reaches only brand-new databases. Tests
  pass, and every existing database silently lacks the change.
- **Give new columns defaults**, for the existing rows and for code that
  predates the column.
- **Register the special kinds.** A rebuild joins
  [`_FOREIGN_KEYS_OFF_MIGRATIONS`](../../src/audrey/app_state/store.py#L61),
  copies every column and recreates indexes and triggers. A column addition
  joins [`_ADDITIVE_COLUMN_MIGRATIONS`](../../src/audrey/app_state/store.py#L62),
  and the method that applies it needs the column's definition too.
- **Expect no undo.** Nothing in Audrey reverses a migration, which is one more
  reason to prefer changes that only add, and the reason to take a backup
  (§2.8) before deploying a release with new migrations: A migration that runs
  cleanly but does the wrong thing commits like any other.

### 2.8 Backups and verified restores

Copying the database file isn't a backup while Audrey runs. The newest commits
can sit in `audrey_app.sqlite-wal` until a checkpoint moves them, so a copy of
the main file alone can miss them, and copying both files while Audrey writes
can catch them at different moments.

SQLite's **online backup** copies a database page by page from one consistent
snapshot while other connections keep reading and writing. Audrey wraps it in
[`_online_application_backup`](../../src/audrey/admin_cli.py#L133); its core,
at [`admin_cli.py:142-159`](../../src/audrey/admin_cli.py#L142-L159):

```python
descriptor = os.open(destination, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
...
uri = f"file:{quote(str(source.resolve()), safe='/')}?mode=ro"
with (
    closing(sqlite3.connect(uri, uri=True)) as source_db,
    closing(sqlite3.connect(destination)) as backup_db,
):
    source_db.backup(backup_db)
    integrity = backup_db.execute("PRAGMA integrity_check").fetchone()
    ...
except BaseException:
    if created:
        destination.unlink(missing_ok=True)
    raise
```

Read it for what it refuses to do:

- **Overwrite.** `O_CREAT | O_EXCL` creates the file only if nothing exists at
  that path, in a single step, so an older backup stays intact.
- **Expose.** Mode `0o600` lets only the file's owner open it, and it holds
  every conversation in Audrey.
- **Touch production.** The `mode=ro` connection can't change the live database.
- **Trust itself.** `PRAGMA integrity_check` walks the copy's structure and
  answers `ok` or a list of problems.
- **Leave a half-done copy.** On any failure the partial file is deleted, so a
  broken backup can't later pass for a good one.

An operator runs it as `audrey-admin backup-app-state --to <new path>`
([`_backup_application_state`](../../src/audrey/admin_cli.py#L161)), which
prints the size, the integrity result and the schema version as JSON.

A backup is only as good as a restore you've seen work, so a second command,
`verify-app-state-backup`
([`_verify_application_backup`](../../src/audrey/admin_cli.py#L261)), restores
one into a throwaway file and checks it. It refuses outright if the backup *is*
the production database, or if the restore path is production or the backup
itself. It inspects the backup read-only and immutable
([`_inspect_application_database`](../../src/audrey/admin_cli.py#L191)) for
integrity, foreign keys, schema version and record counts, and requires the
schema version this build expects. Then it restores the backup with the same
online backup, inspects the copy the same way, and requires matching counts and
mode `0o600`. No step opens production for writing, and the output holds counts
and paths, never content.

**When backups happen.** Audrey doesn't back itself up: An operator runs the
command, and the rule is to take a backup, and verify it, before deploying a
release whose migrations haven't run yet. The one bulk write an operator can
make, importing an exported chat history with `import-chat-export --apply`,
enforces that rule for itself. It refuses to run without `--backup-to`, and
takes that backup before it writes a row
([`admin_cli.py:367`](../../src/audrey/admin_cli.py#L367 '_online_application_backup(path, backup_to)')).

**Rebuilding a projection from the authority.** Back to Lesson 4's distinction,
now from inside the database. The archive receipts from §2.6 are what make the
chat archive rebuildable:
[`ChatProjectionPromoter.rebuild`](../../src/audrey/chat_projection.py#L92)
marks every receipt undelivered with one `UPDATE`
([`_reset_all_sync`](../../src/audrey/app_state/repositories.py#L1479)), and
the worker delivers every finished turn again. Each delivery carries the
receipt's stable id, so delivering a turn twice does no harm. Lesson 4 named the
admin route that starts it,
[`rebuild_canonical_chat_projection`](../../src/audrey/routes/admin.py#L329).
Nothing works in the other direction: The archive can't rebuild the application
database. That asymmetry is the rule in practice: **Back up authorities;
rebuild projections.**

## 3. Comprehension questions

**1. Bob opens the browser's developer tools and requests
`GET /api/conversations/con_…` with an id copied from Alice's screen. What comes
back, and why that answer?**

A 404, "Conversation not found." The route passes Bob's user id, taken from the
authenticated principal and not from the request, to `conversations.get`, whose
query requires both ids to match (§2.5). Alice's conversation doesn't match
Bob's id, so the repository returns `None`, exactly as for an id that doesn't
exist. A 403 for "exists, but isn't yours" would let anyone confirm which ids
are real. And if a write path ever forgot the owner filter, the foreign keys
that carry the owner (§2.4) would still stop a row of Bob's from attaching to
Alice's conversation.

**2. The power fails twice: once while `begin_run` is between its inserts, and
once while a model is halfway through an answer. What does the database hold
after each restart?**

After the first, nothing from that turn: The transaction never committed, so
SQLite discards its writes when the database next opens (§2.1), and the person
can send the message again. After the second, `begin_run` had committed, so the
run says `running` and the assistant message is empty and `in_progress`; the
streamed text lived only in memory. Before the first request, startup's
`recover_interrupted_runs` marks the message incomplete and the run failed with
`server_restart`, and writes the turn's archive receipt, in one transaction
(§2.6). The person sees an honest failed turn, not one that spins forever.

**3. A release adds a new allowed value to `app_runs.mode`. Why is that
migration a table rebuild, and what would happen if it ran with foreign keys
on?**

The allowed values are a `CHECK` constraint, which SQLite can't alter in place,
so the table is recreated with the new constraint and its rows copied across
(§2.7). With foreign keys on, renaming `app_runs` aside would rewrite the
messages' foreign key to follow the renamed table, and dropping that table would
cascade: Every message attached to a run would be deleted, with its attachments,
sources and tool calls. That's why the rebuild path turns foreign keys off
before the transaction begins, the migration keeps references on the name with
`legacy_alter_table`, and `foreign_key_check` runs before the commit.

**4. A teammate needs a new column on `app_messages`. Rather than add a
migration, they edit the migration that creates that table to include it. The
tests pass. What happens when the release reaches the server?**

Nothing changes there. The server's database recorded that version long ago, so
the runner skips it, and even run again, `CREATE TABLE IF NOT EXISTS` wouldn't
touch an existing table. The tests passed because they start from empty
databases, which run the edited migration. In production, the first query that
names the new column fails. The rule from §2.7 prevents it: Never edit a
migration that has run anywhere; append a new version that adds the column, with
a default.

**5. Before an upgrade, an operator copies `audrey_app.sqlite` with `cp` while
Audrey is running. After a failed upgrade they restore the copy, and the most
recent conversations are missing, or the file won't open. Why, and what should
they have done?**

Recent commits can live in `audrey_app.sqlite-wal` until a checkpoint moves them
into the main file, so a copy of the main file alone doesn't have them, and
copying both while Audrey writes can catch them at different moments (§2.8). The
operator should have run `audrey-admin backup-app-state`, which takes one
consistent snapshot through SQLite's online backup while Audrey keeps running,
checks it and never overwrites, and then `verify-app-state-backup`, which proves
the backup restores, into a throwaway file rather than over production.

## When you're ready for the next lesson

Every query in this lesson filtered by a `user_id` taken from an authenticated
caller. How Audrey decides who that caller is, and what they may use, gets a
lesson of its own later. The next lesson returns to the answer pipeline and
opens the model layer:
[Lesson 6 — The model layer I: choosing a model](lesson-06-the-model-layer.md).
It traces what happens when a pipeline node says "I need a general-purpose
model": which choices a person may pick, how the registry ranks candidates, and
how the health tracker steps around failing ones. The lesson after it follows
the call itself to Ollama.
