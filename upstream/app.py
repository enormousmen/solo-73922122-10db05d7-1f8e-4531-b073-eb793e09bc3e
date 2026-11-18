"""Demo artifact upstream with controllable readiness latency and slow downloads.

Run two copies (blue/green).  Every response identifies itself via
``X-Upstream-Name`` so the relay demo makes routing visible.

Each copy also serves a mutable release manifest at ``/release-manifest.json``
(path -> SHA-256) with a strong, content-derived ``ETag`` and honours
``If-None-Match`` (304), so the relay's ``/api/release-diff`` can be exercised
end to end.  ``POST /_control/manifest`` edits the manifest live.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import time
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

NAME = os.environ.get("UPSTREAM_NAME", "upstream")
PROBE_DEFAULT_DELAY = float(os.environ.get("UPSTREAM_PROBE_DELAY", "0"))
PROBE_DEFAULT_STATUS = int(os.environ.get("UPSTREAM_PROBE_STATUS", "200"))
MAX_PROBE_DELAY = 120.0
MAX_MANIFEST_ITEMS = 200
MAX_MANIFEST_PATH = 200
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")

app = FastAPI(title=f"Artifact upstream {NAME}")
state = {"probe_delay": PROBE_DEFAULT_DELAY, "probe_status": PROBE_DEFAULT_STATUS}


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _default_manifest(name: str) -> dict[str, str]:
    """Deterministic per-upstream manifest: shared entries, one upstream-unique
    artifact, and one entry whose digest differs between blue and green, so the
    relay's release-diff demo shows all three categories out of the box."""
    return {
        "artifacts/app.tar.gz": _sha256_text("app.tar.gz:v1"),
        "artifacts/runtime.tar": _sha256_text("runtime.tar:v1"),
        "docs/manual.pdf": _sha256_text("manual.pdf:v1"),
        "artifacts/config.json": _sha256_text(f"config:{name}"),
        f"artifacts/{name}-only.bin": _sha256_text(f"{name}-only:v1"),
    }


manifest_state: dict[str, str] = _default_manifest(NAME)


def _manifest_etag() -> str:
    """Strong validator derived from the manifest content: any edit moves it."""
    canonical = json.dumps(manifest_state, sort_keys=True, separators=(",", ":"))
    return '"' + hashlib.sha256(canonical.encode()).hexdigest()[:32] + '"'


def _if_none_match_matches(header_value: str, etag: str) -> bool:
    for token in header_value.split(","):
        token = token.strip()
        if token == "*" or token == etag or token == f"W/{etag}":
            return True
    return False


@app.get("/release-manifest.json")
async def release_manifest(request: Request):
    """Release manifest with strong ETag + conditional GET (304) support."""
    etag = _manifest_etag()
    inm = request.headers.get("if-none-match")
    if inm is not None and _if_none_match_matches(inm, etag):
        return Response(status_code=304, headers={"etag": etag, "cache-control": "no-cache"})
    body = json.dumps(manifest_state, indent=2, sort_keys=True).encode()
    return Response(
        body,
        media_type="application/json",
        headers={"etag": etag, "cache-control": "no-cache"},
    )


@app.post("/_control/manifest")
async def control_manifest_update(request: Request):
    """Edit the manifest, e.g.

    ``POST /_control/manifest {"set": {"p": "<64 hex>"}, "remove": ["q"], "reset": false}``
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "body must be a JSON object"}, status_code=400)
    if not isinstance(body, dict):
        return JSONResponse({"error": "body must be a JSON object"}, status_code=400)
    new = _default_manifest(NAME) if body.get("reset") else dict(manifest_state)
    remove = body.get("remove") or []
    if not isinstance(remove, list):
        return JSONResponse({"error": "'remove' must be a list of paths"}, status_code=400)
    for path in remove:
        new.pop(str(path), None)
    set_entries = body.get("set") or {}
    if not isinstance(set_entries, dict):
        return JSONResponse({"error": "'set' must be an object of path -> sha256"}, status_code=400)
    for path, digest in set_entries.items():
        if not isinstance(path, str) or not path or len(path) > MAX_MANIFEST_PATH:
            return JSONResponse({"error": f"invalid artifact path {path!r}"}, status_code=400)
        if not isinstance(digest, str) or not _SHA256_RE.match(digest):
            return JSONResponse(
                {"error": f"entry {path!r} is not a 64-char hex SHA-256 digest"}, status_code=400
            )
        new[path] = digest.lower()
    if len(new) > MAX_MANIFEST_ITEMS:
        return JSONResponse(
            {"error": f"manifest would exceed {MAX_MANIFEST_ITEMS} entries"}, status_code=400
        )
    manifest_state.clear()
    manifest_state.update(new)
    return await control_manifest_get()


@app.get("/_control/manifest")
async def control_manifest_get():
    return {
        "upstream": NAME,
        "items": len(manifest_state),
        "etag": _manifest_etag(),
        "manifest": manifest_state,
    }


@app.get("/ready")
async def ready():
    """Readiness endpoint (the relay's default probe path)."""
    return await _readiness()


@app.get("/healthz")
async def healthz():
    """Alias kept for probes/healthchecks that expect /healthz."""
    return await _readiness()


async def _readiness():
    """Readiness endpoint. A long delay mimics a slow control plane."""
    delay = state["probe_delay"]
    started = time.monotonic()
    if delay:
        await asyncio.sleep(delay)
    status = state["probe_status"]
    return JSONResponse(
        {"upstream": NAME, "status": "ok" if status == 200 else "not_ready"},
        status_code=status,
        headers={"x-upstream": NAME, "x-probe-delay": str(round(time.monotonic() - started, 3))},
    )


@app.post("/_control/probe")
async def control_probe(delay: float = 0.0, status: int = 200):
    """Set readiness latency/status, e.g. POST /_control/probe?delay=4&status=503."""
    if delay < 0 or delay > MAX_PROBE_DELAY:
        return JSONResponse({"error": "delay out of range"}, status_code=400)
    state["probe_delay"] = delay
    state["probe_status"] = status
    return {"upstream": NAME, "probe_delay": delay, "probe_status": status}


@app.get("/_control/probe")
async def control_probe_get():
    return {"upstream": NAME, **state}


@app.get("/artifact")
async def artifact(
    request: Request,
    bytes: int = 1024 * 1024,
    chunk: int = 65536,
    delay: float = 0.0,
):
    """Stream ``bytes`` bytes with ``delay`` seconds between chunks."""
    bytes = max(0, min(bytes, 512 * 1024 * 1024))
    chunk = max(1, min(chunk, 1024 * 1024))
    delay = max(0.0, min(delay, 10.0))

    def generate() -> AsyncIterator[bytes]:
        async def gen() -> AsyncIterator[bytes]:
            remaining = bytes
            seq = 0
            while remaining > 0:
                size = min(chunk, remaining)
                seed = f"{NAME}:{seq}:".encode()
                block = (seed * (size // len(seed) + 1))[:size]
                yield block
                remaining -= size
                seq += 1
                if delay:
                    await asyncio.sleep(delay)

        return gen()

    headers = {
        "x-upstream": NAME,
        "x-artifact-bytes": str(bytes),
        "x-artifact-delay": str(delay),
        # No Content-Length: exercises chunk-by-chunk relaying.
    }
    return StreamingResponse(generate(), media_type="application/octet-stream", headers=headers)


@app.get("/")
async def index():
    return {"upstream": NAME, "probe_delay": state["probe_delay"], "probe_status": state["probe_status"]}
