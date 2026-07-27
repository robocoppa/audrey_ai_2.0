"""Phase 31 — custom-tools presents the KB service token to Audrey.

`_service_headers` decides whether the audrey client sends
`X-Audrey-Service-Token`. A blank token must yield no header (so a dev/local
custom-tools falls through to Audrey's user-bearer arm rather than sending an
empty secret).
"""

from __future__ import annotations

import sys

sys.path.insert(0, "tools-server")

from app import _service_headers


def test_service_headers_present_when_token_set():
    assert _service_headers("secret") == {"X-Audrey-Service-Token": "secret"}


def test_service_headers_absent_when_token_empty():
    assert _service_headers("") == {}
