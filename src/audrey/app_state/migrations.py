"""Ordered, additive migrations for Audrey's canonical application database."""

from __future__ import annotations

MIGRATIONS: tuple[tuple[int, str], ...] = (
    (
        1,
        """
        CREATE TABLE IF NOT EXISTS app_users (
          user_id            TEXT PRIMARY KEY,
          storage_namespace  TEXT NOT NULL UNIQUE,
          current_email      TEXT NOT NULL,
          display_name       TEXT NOT NULL DEFAULT '',
          role               TEXT NOT NULL CHECK (role IN ('user', 'admin')),
          status             TEXT NOT NULL DEFAULT 'active'
                             CHECK (status IN ('active', 'disabled')),
          created_at         TEXT NOT NULL,
          updated_at         TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS external_identities (
          provider       TEXT NOT NULL,
          subject        TEXT NOT NULL,
          user_id        TEXT NOT NULL,
          email          TEXT NOT NULL,
          created_at     TEXT NOT NULL,
          last_seen_at   TEXT NOT NULL,
          PRIMARY KEY (provider, subject),
          FOREIGN KEY (user_id) REFERENCES app_users(user_id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_external_identities_user
          ON external_identities(user_id);
        """,
    ),
    (
        2,
        """
        CREATE TABLE IF NOT EXISTS personal_access_tokens (
          token_id       TEXT PRIMARY KEY,
          user_id        TEXT NOT NULL,
          name           TEXT NOT NULL,
          secret_hash    TEXT NOT NULL UNIQUE,
          scopes_json    TEXT NOT NULL,
          created_at     TEXT NOT NULL,
          expires_at     TEXT NOT NULL,
          last_used_at   TEXT,
          revoked_at     TEXT,
          FOREIGN KEY (user_id) REFERENCES app_users(user_id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_personal_access_tokens_user
          ON personal_access_tokens(user_id);
        """,
    ),
    (
        3,
        """
        CREATE TABLE IF NOT EXISTS user_preferences (
          user_id                    TEXT PRIMARY KEY,
          timezone                   TEXT NOT NULL DEFAULT 'UTC',
          persona                    TEXT NOT NULL DEFAULT '',
          response_preferences_json TEXT NOT NULL DEFAULT '{}',
          created_at                 TEXT NOT NULL,
          updated_at                 TEXT NOT NULL,
          FOREIGN KEY (user_id) REFERENCES app_users(user_id) ON DELETE CASCADE
        );

        INSERT OR IGNORE INTO user_preferences
          (user_id, timezone, persona, response_preferences_json, created_at, updated_at)
        SELECT user_id, 'UTC', '', '{}', created_at, updated_at FROM app_users;

        CREATE TABLE IF NOT EXISTS app_conversations (
          conversation_id TEXT PRIMARY KEY,
          user_id          TEXT NOT NULL,
          title            TEXT NOT NULL DEFAULT '',
          default_mode     TEXT NOT NULL DEFAULT 'auto'
                           CHECK (default_mode IN
                             ('auto', 'fast', 'deep', 'research', 'local', 'cloud')),
          created_at       TEXT NOT NULL,
          updated_at       TEXT NOT NULL,
          last_message_at  TEXT,
          archived_at      TEXT,
          UNIQUE (conversation_id, user_id),
          FOREIGN KEY (user_id) REFERENCES app_users(user_id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_app_conversations_user_activity
          ON app_conversations(user_id, last_message_at DESC, created_at DESC);

        CREATE TABLE IF NOT EXISTS app_runs (
          run_id            TEXT PRIMARY KEY,
          conversation_id   TEXT NOT NULL,
          user_id           TEXT NOT NULL,
          mode              TEXT NOT NULL CHECK (mode IN
                              ('auto', 'fast', 'deep', 'research', 'local', 'cloud')),
          status            TEXT NOT NULL CHECK (status IN
                              ('running', 'succeeded', 'cancelled', 'failed')),
          started_at        TEXT NOT NULL,
          completed_at      TEXT,
          finish_reason     TEXT NOT NULL DEFAULT '',
          error_code        TEXT NOT NULL DEFAULT '',
          virtual_model     TEXT NOT NULL DEFAULT '',
          concrete_model    TEXT NOT NULL DEFAULT '',
          prompt_tokens     INTEGER NOT NULL DEFAULT 0 CHECK (prompt_tokens >= 0),
          completion_tokens INTEGER NOT NULL DEFAULT 0 CHECK (completion_tokens >= 0),
          UNIQUE (run_id, conversation_id, user_id),
          CHECK (
            (status = 'running' AND completed_at IS NULL)
            OR (status != 'running' AND completed_at IS NOT NULL)
          ),
          FOREIGN KEY (conversation_id, user_id)
            REFERENCES app_conversations(conversation_id, user_id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_app_runs_conversation_started
          ON app_runs(user_id, conversation_id, started_at, run_id);

        CREATE TRIGGER IF NOT EXISTS trg_app_runs_terminal_immutable
        BEFORE UPDATE OF status ON app_runs
        WHEN OLD.status != 'running'
        BEGIN
          SELECT RAISE(ABORT, 'terminal run outcome is immutable');
        END;

        CREATE TABLE IF NOT EXISTS app_messages (
          message_id      TEXT PRIMARY KEY,
          conversation_id TEXT NOT NULL,
          user_id         TEXT NOT NULL,
          run_id          TEXT,
          sequence_no     INTEGER NOT NULL CHECK (sequence_no > 0),
          role            TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'tool')),
          status          TEXT NOT NULL CHECK (status IN
                            ('in_progress', 'completed', 'incomplete')),
          content         TEXT NOT NULL DEFAULT '',
          created_at      TEXT NOT NULL,
          updated_at      TEXT NOT NULL,
          UNIQUE (conversation_id, sequence_no),
          FOREIGN KEY (conversation_id, user_id)
            REFERENCES app_conversations(conversation_id, user_id) ON DELETE CASCADE,
          FOREIGN KEY (run_id, conversation_id, user_id)
            REFERENCES app_runs(run_id, conversation_id, user_id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_app_messages_owner_order
          ON app_messages(user_id, conversation_id, sequence_no);

        CREATE UNIQUE INDEX IF NOT EXISTS idx_app_messages_run_user
          ON app_messages(run_id) WHERE role = 'user';

        CREATE UNIQUE INDEX IF NOT EXISTS idx_app_messages_run_assistant
          ON app_messages(run_id) WHERE role = 'assistant';
        """,
    ),
    (
        4,
        """
        CREATE TABLE IF NOT EXISTS app_chat_projections (
          projection_id       TEXT PRIMARY KEY,
          user_id             TEXT NOT NULL,
          conversation_id     TEXT NOT NULL,
          user_message_id     TEXT,
          assistant_message_id TEXT,
          partial             INTEGER NOT NULL DEFAULT 0
                              CHECK (partial IN (0, 1)),
          virtual_model       TEXT NOT NULL DEFAULT '',
          concrete_model      TEXT NOT NULL DEFAULT '',
          prompt_tokens       INTEGER NOT NULL DEFAULT 0
                              CHECK (prompt_tokens >= 0),
          completion_tokens   INTEGER NOT NULL DEFAULT 0
                              CHECK (completion_tokens >= 0),
          created_at          TEXT NOT NULL,
          enqueued_at         TEXT,
          attempts            INTEGER NOT NULL DEFAULT 0
                              CHECK (attempts >= 0),
          last_attempt_at     TEXT,
          last_error          TEXT NOT NULL DEFAULT '',
          next_attempt_at     TEXT NOT NULL,
          CHECK (user_message_id IS NOT NULL OR assistant_message_id IS NOT NULL),
          UNIQUE (user_message_id, assistant_message_id),
          FOREIGN KEY (conversation_id, user_id)
            REFERENCES app_conversations(conversation_id, user_id) ON DELETE CASCADE,
          FOREIGN KEY (user_message_id)
            REFERENCES app_messages(message_id) ON DELETE CASCADE,
          FOREIGN KEY (assistant_message_id)
            REFERENCES app_messages(message_id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_app_chat_projections_due
          ON app_chat_projections(enqueued_at, next_attempt_at, created_at);

        CREATE INDEX IF NOT EXISTS idx_app_chat_projections_user
          ON app_chat_projections(user_id, enqueued_at, next_attempt_at);

        CREATE TABLE IF NOT EXISTS app_chat_projection_deletions (
          deletion_id       TEXT PRIMARY KEY,
          user_id           TEXT NOT NULL,
          conversation_id   TEXT NOT NULL,
          requested_at      TEXT NOT NULL,
          completed_at      TEXT,
          attempts          INTEGER NOT NULL DEFAULT 0
                            CHECK (attempts >= 0),
          last_attempt_at   TEXT,
          last_error        TEXT NOT NULL DEFAULT '',
          next_attempt_at   TEXT NOT NULL,
          UNIQUE (user_id, conversation_id),
          FOREIGN KEY (user_id) REFERENCES app_users(user_id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_app_chat_projection_deletions_due
          ON app_chat_projection_deletions(
            completed_at, next_attempt_at, requested_at
          );

        CREATE INDEX IF NOT EXISTS idx_app_chat_projection_deletions_user
          ON app_chat_projection_deletions(
            user_id, completed_at, next_attempt_at
          );

        """,
    ),
)

__all__ = ["MIGRATIONS"]
