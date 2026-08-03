"""Media processing helpers for video ingest (campaign 2, phase 32).

Deliberately free of audrey internals — no config, no metrics, no clients. The
media worker runs in its own container and imports from here, so anything that
reaches back into the app would have to be untangled at that point.
"""
