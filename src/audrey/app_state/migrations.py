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
    (
        5,
        """
        PRAGMA legacy_alter_table = ON;

        ALTER TABLE app_runs RENAME TO app_runs_before_video_mode;

        CREATE TABLE app_runs (
          run_id            TEXT PRIMARY KEY,
          conversation_id   TEXT NOT NULL,
          user_id           TEXT NOT NULL,
          mode              TEXT NOT NULL CHECK (mode IN
                              ('auto', 'fast', 'deep', 'research', 'local', 'cloud',
                               'video')),
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

        INSERT INTO app_runs
          (run_id, conversation_id, user_id, mode, status, started_at,
           completed_at, finish_reason, error_code, virtual_model, concrete_model,
           prompt_tokens, completion_tokens)
        SELECT run_id, conversation_id, user_id, mode, status, started_at,
               completed_at, finish_reason, error_code, virtual_model, concrete_model,
               prompt_tokens, completion_tokens
        FROM app_runs_before_video_mode;

        DROP TABLE app_runs_before_video_mode;

        CREATE INDEX idx_app_runs_conversation_started
          ON app_runs(user_id, conversation_id, started_at, run_id);

        CREATE TRIGGER trg_app_runs_terminal_immutable
        BEFORE UPDATE OF status ON app_runs
        WHEN OLD.status != 'running'
        BEGIN
          SELECT RAISE(ABORT, 'terminal run outcome is immutable');
        END;

        ALTER TABLE app_conversations
          RENAME TO app_conversations_before_video_mode;

        CREATE TABLE app_conversations (
          conversation_id TEXT PRIMARY KEY,
          user_id          TEXT NOT NULL,
          title            TEXT NOT NULL DEFAULT '',
          default_mode     TEXT NOT NULL DEFAULT 'auto'
                           CHECK (default_mode IN
                             ('auto', 'fast', 'deep', 'research', 'local', 'cloud',
                              'video')),
          created_at       TEXT NOT NULL,
          updated_at       TEXT NOT NULL,
          last_message_at  TEXT,
          archived_at      TEXT,
          UNIQUE (conversation_id, user_id),
          FOREIGN KEY (user_id) REFERENCES app_users(user_id) ON DELETE CASCADE
        );

        INSERT INTO app_conversations
          (conversation_id, user_id, title, default_mode, created_at, updated_at,
           last_message_at, archived_at)
        SELECT conversation_id, user_id, title, default_mode, created_at, updated_at,
               last_message_at, archived_at
        FROM app_conversations_before_video_mode;

        DROP TABLE app_conversations_before_video_mode;

        CREATE INDEX idx_app_conversations_user_activity
          ON app_conversations(user_id, last_message_at DESC, created_at DESC);

        PRAGMA legacy_alter_table = OFF;
        """,
    ),
    (
        6,
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_app_messages_owner_id
          ON app_messages(message_id, conversation_id, user_id);

        CREATE TABLE IF NOT EXISTS app_message_attachments (
          message_id      TEXT NOT NULL,
          conversation_id TEXT NOT NULL,
          user_id         TEXT NOT NULL,
          position        INTEGER NOT NULL CHECK (position >= 0),
          file_id         TEXT NOT NULL,
          filename        TEXT NOT NULL,
          mime            TEXT NOT NULL,
          kind            TEXT NOT NULL CHECK (kind IN ('text', 'image', 'video')),
          bytes           INTEGER NOT NULL CHECK (bytes >= 0),
          PRIMARY KEY (message_id, position),
          UNIQUE (message_id, file_id),
          FOREIGN KEY (message_id, conversation_id, user_id)
            REFERENCES app_messages(message_id, conversation_id, user_id)
            ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_app_message_attachments_owner
          ON app_message_attachments(user_id, conversation_id, message_id, position);
        """,
    ),
    (
        7,
        """
        PRAGMA legacy_alter_table = ON;

        ALTER TABLE app_users RENAME TO app_users_before_access_groups;

        CREATE TABLE app_users (
          user_id            TEXT PRIMARY KEY,
          storage_namespace  TEXT NOT NULL UNIQUE,
          current_email      TEXT NOT NULL,
          display_name       TEXT NOT NULL DEFAULT '',
          role               TEXT NOT NULL CHECK (role IN ('user', 'admin')),
          status             TEXT NOT NULL DEFAULT 'active'
                             CHECK (status IN ('pending', 'active', 'disabled')),
          created_at         TEXT NOT NULL,
          updated_at         TEXT NOT NULL
        );

        INSERT INTO app_users
          (user_id, storage_namespace, current_email, display_name, role, status,
           created_at, updated_at)
        SELECT user_id, storage_namespace, current_email, display_name, role, status,
               created_at, updated_at
        FROM app_users_before_access_groups;

        DROP TABLE app_users_before_access_groups;

        CREATE TABLE IF NOT EXISTS access_groups (
          group_id    TEXT PRIMARY KEY,
          label       TEXT NOT NULL,
          description TEXT NOT NULL DEFAULT '',
          system      INTEGER NOT NULL DEFAULT 1 CHECK (system IN (0, 1)),
          created_at  TEXT NOT NULL
        );

        INSERT OR IGNORE INTO access_groups
          (group_id, label, description, system, created_at)
        VALUES
          ('users', 'Users', 'Approved Audrey users.', 1,
           strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')),
          ('testers', 'Testers', 'Users with preview feature and model access.', 1,
           strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')),
          ('admins', 'Admins', 'Audrey administrators.', 1,
           strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now'));

        CREATE TABLE IF NOT EXISTS user_group_memberships (
          user_id            TEXT NOT NULL,
          group_id           TEXT NOT NULL,
          granted_by_user_id TEXT,
          granted_at         TEXT NOT NULL,
          PRIMARY KEY (user_id, group_id),
          FOREIGN KEY (user_id) REFERENCES app_users(user_id) ON DELETE CASCADE,
          FOREIGN KEY (group_id) REFERENCES access_groups(group_id) ON DELETE RESTRICT,
          FOREIGN KEY (granted_by_user_id) REFERENCES app_users(user_id) ON DELETE SET NULL
        );

        INSERT OR IGNORE INTO user_group_memberships
          (user_id, group_id, granted_by_user_id, granted_at)
        SELECT user_id, 'users', NULL, created_at FROM app_users;

        INSERT OR IGNORE INTO user_group_memberships
          (user_id, group_id, granted_by_user_id, granted_at)
        SELECT user_id, 'admins', NULL, created_at
        FROM app_users WHERE role = 'admin';

        CREATE INDEX IF NOT EXISTS idx_user_group_memberships_group
          ON user_group_memberships(group_id, user_id);

        CREATE TABLE IF NOT EXISTS admin_audit_events (
          event_id      TEXT PRIMARY KEY,
          actor_user_id TEXT,
          target_type   TEXT NOT NULL CHECK (target_type IN ('user', 'model')),
          target_id     TEXT NOT NULL,
          action        TEXT NOT NULL,
          before_json   TEXT NOT NULL DEFAULT '{}',
          after_json    TEXT NOT NULL DEFAULT '{}',
          created_at    TEXT NOT NULL,
          FOREIGN KEY (actor_user_id) REFERENCES app_users(user_id) ON DELETE SET NULL
        );

        CREATE INDEX IF NOT EXISTS idx_admin_audit_events_created
          ON admin_audit_events(created_at DESC, event_id DESC);

        CREATE TABLE IF NOT EXISTS model_access_policies (
          model_id           TEXT PRIMARY KEY,
          enabled            INTEGER NOT NULL CHECK (enabled IN (0, 1)),
          audience           TEXT NOT NULL
                             CHECK (audience IN ('users', 'testers', 'admins')),
          updated_by_user_id TEXT,
          updated_at         TEXT NOT NULL,
          FOREIGN KEY (updated_by_user_id) REFERENCES app_users(user_id) ON DELETE SET NULL
        );

        PRAGMA legacy_alter_table = OFF;
        """,
    ),
    (
        8,
        """
        PRAGMA legacy_alter_table = ON;

        ALTER TABLE app_runs RENAME TO app_runs_before_model_selection;

        CREATE TABLE app_runs (
          run_id             TEXT PRIMARY KEY,
          conversation_id    TEXT NOT NULL,
          user_id            TEXT NOT NULL,
          mode               TEXT NOT NULL CHECK (mode IN
                               ('auto', 'fast', 'deep', 'research', 'local', 'cloud',
                                'video', 'direct')),
          requested_model_id TEXT NOT NULL DEFAULT 'auto',
          status             TEXT NOT NULL CHECK (status IN
                               ('running', 'succeeded', 'cancelled', 'failed')),
          started_at         TEXT NOT NULL,
          completed_at       TEXT,
          finish_reason      TEXT NOT NULL DEFAULT '',
          error_code         TEXT NOT NULL DEFAULT '',
          virtual_model      TEXT NOT NULL DEFAULT '',
          concrete_model     TEXT NOT NULL DEFAULT '',
          prompt_tokens      INTEGER NOT NULL DEFAULT 0 CHECK (prompt_tokens >= 0),
          completion_tokens  INTEGER NOT NULL DEFAULT 0 CHECK (completion_tokens >= 0),
          UNIQUE (run_id, conversation_id, user_id),
          CHECK (
            (status = 'running' AND completed_at IS NULL)
            OR (status != 'running' AND completed_at IS NOT NULL)
          ),
          FOREIGN KEY (conversation_id, user_id)
            REFERENCES app_conversations(conversation_id, user_id) ON DELETE CASCADE
        );

        INSERT INTO app_runs
          (run_id, conversation_id, user_id, mode, requested_model_id, status,
           started_at, completed_at, finish_reason, error_code, virtual_model,
           concrete_model, prompt_tokens, completion_tokens)
        SELECT run_id, conversation_id, user_id, mode,
               CASE
                 WHEN virtual_model != '' THEN
                   CASE virtual_model
                     WHEN 'audrey_auto' THEN 'auto'
                     WHEN 'audrey_fast' THEN 'fast'
                     WHEN 'audrey_deep' THEN 'deep'
                     WHEN 'audrey_research' THEN 'research'
                     WHEN 'audrey_local' THEN 'local'
                     WHEN 'audrey_cloud' THEN 'cloud'
                     WHEN 'audrey_video' THEN 'video'
                     ELSE virtual_model
                   END
                 ELSE mode
               END,
               status, started_at, completed_at, finish_reason, error_code,
               virtual_model, concrete_model, prompt_tokens, completion_tokens
        FROM app_runs_before_model_selection;

        DROP TABLE app_runs_before_model_selection;

        CREATE INDEX idx_app_runs_conversation_started
          ON app_runs(user_id, conversation_id, started_at, run_id);

        CREATE TRIGGER trg_app_runs_terminal_immutable
        BEFORE UPDATE OF status ON app_runs
        WHEN OLD.status != 'running'
        BEGIN
          SELECT RAISE(ABORT, 'terminal run outcome is immutable');
        END;

        ALTER TABLE app_conversations
          RENAME TO app_conversations_before_model_selection;

        CREATE TABLE app_conversations (
          conversation_id TEXT PRIMARY KEY,
          user_id          TEXT NOT NULL,
          title            TEXT NOT NULL DEFAULT '',
          default_mode     TEXT NOT NULL DEFAULT 'auto'
                           CHECK (default_mode IN
                             ('auto', 'fast', 'deep', 'research', 'local', 'cloud',
                              'video', 'direct')),
          default_model_id TEXT NOT NULL DEFAULT 'auto',
          created_at       TEXT NOT NULL,
          updated_at       TEXT NOT NULL,
          last_message_at  TEXT,
          archived_at      TEXT,
          UNIQUE (conversation_id, user_id),
          FOREIGN KEY (user_id) REFERENCES app_users(user_id) ON DELETE CASCADE
        );

        INSERT INTO app_conversations
          (conversation_id, user_id, title, default_mode, default_model_id,
           created_at, updated_at, last_message_at, archived_at)
        SELECT conversation_id, user_id, title, default_mode, default_mode,
               created_at, updated_at, last_message_at, archived_at
        FROM app_conversations_before_model_selection;

        DROP TABLE app_conversations_before_model_selection;

        CREATE INDEX idx_app_conversations_user_activity
          ON app_conversations(user_id, last_message_at DESC, created_at DESC);

        PRAGMA legacy_alter_table = OFF;
        """,
    ),
    (
        9,
        """
        CREATE TABLE IF NOT EXISTS account_deletion_requests (
          user_id           TEXT PRIMARY KEY,
          purge_id          TEXT NOT NULL UNIQUE,
          storage_namespace TEXT NOT NULL,
          requested_at      TEXT NOT NULL,
          FOREIGN KEY (user_id) REFERENCES app_users(user_id) ON DELETE CASCADE
        );
        """,
    ),
    (
        10,
        """
        CREATE TABLE IF NOT EXISTS model_publication_profiles (
          model_id           TEXT PRIMARY KEY,
          visibility         TEXT NOT NULL CHECK (visibility IN ('public', 'private')),
          roles_json         TEXT NOT NULL DEFAULT '[]',
          display_name       TEXT NOT NULL DEFAULT '',
          portrait_mime      TEXT NOT NULL DEFAULT '',
          portrait_data      BLOB,
          updated_by_user_id TEXT,
          updated_at         TEXT NOT NULL,
          FOREIGN KEY (updated_by_user_id) REFERENCES app_users(user_id) ON DELETE SET NULL
        );
        """,
    ),
)

__all__ = ["MIGRATIONS"]
