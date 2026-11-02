"""Installed API/source-reference checks using isolated temporary local fixtures.

Stream into the running backend with smoke-knowledge-sources-onbox.sh. This
constructs a separate ASGI application and disposable SQLite/upload directories;
it never opens the server's account database, contacts a network or runs a model.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace


class SmokeSetupError(Exception):
    pass


class SmokeFailureError(Exception):
    pass


def require(condition, explanation):
    if not condition:
        raise SmokeFailureError(explanation)


def load_runtime():
    try:
        import httpx
        from fastapi import FastAPI

        from audrey import knowledge_source_browser  # noqa: F401 - installed feature preflight
        from audrey.app_state import ApplicationStore
        from audrey.auth import require_principal
        from audrey.content_sources import ContentSourceReader
        from audrey.kb.extract import ocr_text_path
        from audrey.kb.uploads_db import UploadsDB
        from audrey.kb.user_store import sanitize_user
        from audrey.knowledge_sources import KnowledgeSourceResolver
        from audrey.routes.app import router
    except ImportError as error:
        raise SmokeSetupError("Installed backend lacks the source browser; rebuild audrey.") from error
    return SimpleNamespace(
        httpx=httpx, FastAPI=FastAPI, ApplicationStore=ApplicationStore,
        require_principal=require_principal, ContentSourceReader=ContentSourceReader,
        ocr_text_path=ocr_text_path, UploadsDB=UploadsDB, sanitize_user=sanitize_user,
        KnowledgeSourceResolver=KnowledgeSourceResolver, router=router,
    )


class Scenario:
    def __init__(self, runtime, root):
        self.runtime, self.root = runtime, root
        self.store = runtime.ApplicationStore(root / "isolated-app.sqlite")
        self.uploads = runtime.UploadsDB(root / "isolated-uploads.sqlite")
        self.client = None

    async def __aenter__(self):
        try:
            return await self._initialize()
        except BaseException:
            await self.__aexit__()
            raise

    async def _initialize(self):
        r = self.runtime
        self.owner, self.other = [await self.store.resolve_external_identity(
            provider="source-smoke", subject=name, email=f"{name}@example.com", display_name=name,
            role="user", auth_method="test",
        ) for name in ("alice", "bob")]
        self.project = await self.store.projects.create(user_id=self.owner.user_id, name="Source smoke")
        self.private = await self.store.projects.create(
            user_id=self.owner.user_id, name="Private source smoke", private_required=True,
        )
        self.current = self.owner
        reader = r.ContentSourceReader(self.store, self.uploads, self.root / "uploads")
        app = r.FastAPI()
        app.state.application_store = self.store
        app.state.knowledge_source_resolver = r.KnowledgeSourceResolver(self.store, reader)
        app.include_router(r.router)
        app.dependency_overrides[r.require_principal] = lambda: self.current
        self.client = r.httpx.AsyncClient(transport=r.httpx.ASGITransport(app=app), base_url="http://isolated")
        self.path = f"/api/projects/{self.project.project_id}/knowledge"
        self.private_path = f"/api/projects/{self.private.project_id}/knowledge"
        self.message_id = await self.message("msg_a", "Evidence " + "a" * 4500)
        await self.message("msg_b", "Second saved message.")
        await self.message("msg_outside", "Outside the selected Project.", outside=True)
        return self

    async def __aexit__(self, *_error):
        try:
            if self.client:
                await self.client.aclose()
        finally:
            self.uploads.close()
            self.store.close()

    async def message(self, message_id, text, *, outside=False):
        conversation = await self.store.conversations.create(
            user_id=self.owner.user_id, title="Source smoke message",
            project_id=None if outside else self.project.project_id,
        )
        with self.store._lock, self.store._conn:
            self.store._conn.execute(
                "INSERT INTO app_messages (message_id,conversation_id,user_id,sequence_no,role,status,"
                "content,created_at,updated_at) VALUES (?,?,?,1,'user','completed',?,?,?)",
                (message_id, conversation.conversation_id, self.owner.user_id, text,
                 "2026-11-01T10:00:00+00:00", "2026-11-01T10:00:00+00:00"),
            )
        return message_id

    async def request(self, method, path, *, status=200, **kwargs):
        response = await self.client.request(method, path, **kwargs)
        require(response.status_code == status, f"{method} source contract expected HTTP {status}, received {response.status_code}.")
        return response.json() if status in {200, 201} else None

    async def preview(self, source_id, **params):
        return await self.request("GET", self.path + "/source-preview", params={"source_id": source_id, **params})

    async def file(self, *, private=False):
        file_id = "private_file" if private else "retained_file"
        project = self.private if private else self.project
        directory = self.root / "uploads" / self.runtime.sanitize_user(self.owner.storage_namespace)
        directory.mkdir(parents=True, exist_ok=True)
        name, kind = ("protected.pdf", "text") if private else ("recording.mp4", "video")
        if private:
            source = directory / f"{file_id}.pdf"
            source.write_bytes(b"%PDF-isolated-fixture")
            self.runtime.ocr_text_path(source).write_text("Protected local saved text.", encoding="utf-8")
        else:
            (directory / f"{file_id}.transcript.txt").write_text("Retained recording transcript.", encoding="utf-8")
        with self.uploads._lock:
            self.uploads._record_locked(
                file_id, self.owner.storage_namespace, name, "application/pdf" if private else "video/mp4",
                32, kind, "", 0, "2026-11-01T10:00:00+00:00", "ready", private_required=private,
            )
        if private:
            # Bootstrap an already-processed protected upload in this disposable
            # fixture; live Private upload admission is outside this smoke.
            with self.store._lock, self.store._conn:
                self.store._conn.execute(
                    "INSERT INTO app_project_files(project_id,file_id,added_at) VALUES (?,?,?)",
                    (project.project_id, file_id, "2026-11-01T10:00:00+00:00"),
                )
        else:
            await self.store.projects.add_file(
                user_id=self.owner.user_id, project_id=project.project_id, file_id=file_id,
            )
        return file_id


async def scoped_candidates(s):
    first = await s.request("GET", s.path + "/source-candidates", params={"kind": "message", "limit": 1})
    require([item["source_id"] for item in first["items"]] == ["message:msg_a"] and first["next_cursor"],
            "Canonical message candidates or paging failed.")
    second = await s.request("GET", s.path + "/source-candidates", params={
        "kind": "message", "limit": 1, "cursor": first["next_cursor"],
    })
    require([item["source_id"] for item in second["items"]] == ["message:msg_b"] and second["next_cursor"] is None,
            "Candidate paging crossed Project scope or repeated a source.")
    await s.request("GET", s.private_path + "/source-candidates", status=422,
                    params={"kind": "message", "cursor": first["next_cursor"]})


async def preview_and_reference_edits(s):
    source_id = f"message:{s.message_id}"
    preview = await s.preview(source_id)
    require(len(preview["text"]) == 4000 and preview["next_offset"] == 4000,
            "Canonical preview did not enforce bounded paging.")
    tail = await s.preview(source_id, revision=preview["revision"], offset=4000)
    require(tail["next_offset"] is None and len(tail["text"]) == 509, "Preview continuation changed its text window.")
    ref = {"source_id": source_id, "revision": preview["revision"]}
    entry = await s.request("POST", s.path, status=201, json={
        "kind": "fact", "text": "User-authored source note.", "sources": [ref], "request_id": "source-smoke",
    })
    require(entry["sources"] == [ref] and not entry["stale"], "The inspected source reference was not attached.")
    with s.store._lock, s.store._conn:
        s.store._conn.execute("UPDATE app_messages SET content=?,updated_at=? WHERE message_id=?",
            ("Updated canonical source.", "2026-11-01T11:00:00+00:00", s.message_id))
    await s.request("GET", s.path + "/source-preview", status=409,
                    params={"source_id": source_id, "revision": ref["revision"]})
    stale = await s.request("GET", f"{s.path}/{entry['id']}")
    require(stale["stale"] and stale["text"] == entry["text"], "A stale reference lost its user-authored note or stale state.")
    current = await s.preview(source_id)
    new_ref = {"source_id": source_id, "revision": current["revision"]}
    require(new_ref != ref, "The changed source did not receive a new revision.")
    refreshed = await s.request("PATCH", f"{s.path}/{entry['id']}", json={
        "expected_revision": entry["revision"], "sources": [new_ref],
    })
    require(not refreshed["stale"] and refreshed["sources"] == [new_ref], "Explicit reference refresh failed.")
    unlinked = await s.request("PATCH", f"{s.path}/{entry['id']}", json={
        "expected_revision": refreshed["revision"], "sources": [],
    })
    require(unlinked["sources"] == [] and unlinked["text"] == entry["text"], "Explicit source removal changed the note.")


async def retained_text_and_detachment(s):
    file_id = await s.file()
    source_id = f"file:{file_id}:transcript"
    listing = await s.request("GET", s.path + "/source-candidates", params={"kind": "file"})
    require(source_id in {item["source_id"] for item in listing["items"]}, "Retained file text was missing from candidates.")
    preview = await s.preview(source_id)
    require(preview["availability"] == "text_only" and preview["text"] == "Retained recording transcript.",
            "A missing original hid its retained text or was reported available.")
    await s.store.projects.remove_file(user_id=s.owner.user_id, project_id=s.project.project_id, file_id=file_id)
    response = await s.client.get(s.path + "/source-preview", params={"source_id": source_id})
    require(response.status_code in {404, 409} and "Retained recording transcript" not in response.text,
            "A detached file remained readable through its former Project.")


async def private_and_owner_fences(s):
    file_id = await s.file(private=True)
    source_id = f"file:{file_id}:text"
    preview = await s.request("GET", s.private_path + "/source-preview", params={"source_id": source_id})
    require(preview["private_required"] and preview["text"] == "Protected local saved text.",
            "Private Project preview did not use its saved local text.")
    ordinary = await s.client.get(s.path + "/source-preview", params={"source_id": source_id})
    require(ordinary.status_code in {404, 409} and "Protected local" not in ordinary.text,
            "An ordinary Project exposed protected source text.")
    s.current = s.other
    try:
        await s.request("GET", s.path + "/source-candidates", status=404, params={"kind": "message"})
        await s.request("GET", s.private_path + "/source-preview", status=404, params={"source_id": source_id})
    finally:
        s.current = s.owner


CASES = (
    ("project_scoped_candidates_and_cursor", scoped_candidates),
    ("preview_revision_attach_refresh_remove", preview_and_reference_edits),
    ("retained_text_and_detachment", retained_text_and_detachment),
    ("private_and_owner_fences", private_and_owner_fences),
)


async def run(runtime, timeout_seconds):
    checks = []
    with tempfile.TemporaryDirectory(prefix="audrey-source-smoke-") as directory:
        async with Scenario(runtime, Path(directory)) as scenario:
            for name, case in CASES:
                try:
                    async with asyncio.timeout(timeout_seconds):
                        await case(scenario)
                except TimeoutError:
                    raise SmokeFailureError(f"{name} exceeded its deadline.") from None
                checks.append(name)
    require(not await asyncio.to_thread(Path(directory).exists), "Disposable smoke fixtures were not cleaned up.")
    checks.append("isolated_fixture_cleanup")
    return checks


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--timeout-seconds", type=float, default=10.0)
    args = parser.parse_args(argv)
    code, checks, message = 0, [], "Knowledge source smoke passed; isolated temporary data; no credentials/network/models."
    try:
        if not 0 < args.timeout_seconds <= 10:
            raise SmokeSetupError("Per-case timeout must be positive and at most ten seconds.")
        checks = asyncio.run(run(load_runtime(), args.timeout_seconds))
    except SmokeSetupError as error:
        code, message = 2, str(error)
    except SmokeFailureError as error:
        code, message = 1, str(error)
    except (KeyboardInterrupt, asyncio.CancelledError):
        code, message = 130, "Knowledge source smoke interrupted."
    except Exception:  # noqa: BLE001 - private paths, SQL and provider errors never enter output
        code, message = 3, "Knowledge source smoke harness crashed."
    if args.json:
        print(json.dumps({"exit_code": code, "checks": checks, "message": message}, sort_keys=True))
    else:
        for check in checks:
            print("PASS " + check)
        print(("PASS" if code == 0 else "FAIL") + f" {message} (exit {code})")
    return code


if __name__ == "__main__":
    sys.exit(main())
