"""Controllable upstream used by the integration tests."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time

from starlette.applications import Starlette
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

PATTERN = b"0123456789abcdef"


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def default_manifest(name: str) -> dict[str, str]:
    """Shared entries + one upstream-unique artifact + one per-upstream digest,
    so a blue/green diff shows all three categories."""
    return {
        "artifacts/app.tar.gz": _sha256_text("app.tar.gz:v1"),
        "artifacts/runtime.tar": _sha256_text("runtime.tar:v1"),
        "artifacts/config.json": _sha256_text(f"config:{name}"),
        f"artifacts/{name}-only.bin": _sha256_text(f"{name}-only:v1"),
    }


def manifest_etag(manifest: dict[str, str]) -> str:
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    return '"' + hashlib.sha256(canonical.encode()).hexdigest()[:32] + '"'


async def _artifact(request):
    params = request.query_params
    size = int(params.get("bytes", 1024))
    chunk = max(1, int(params.get("chunk", 64)))
    delay = float(params.get("delay", 0))
    header_delay = float(params.get("header_delay", 0))
    stats = request.app.state.stats
    stats["started"] += 1
    state = {"finished": False}

    if header_delay:
        await asyncio.sleep(header_delay)

    async def gen():
        try:
            sent = 0
            while sent < size:
                n = min(chunk, size - sent)
                block = (PATTERN * (n // len(PATTERN) + 1))[:n]
                yield block
                sent += n
                if delay:
                    await asyncio.sleep(delay)
            state["finished"] = True
            stats["completed"] += 1
        except asyncio.CancelledError:
            stats["cancelled"] += 1
            raise
        finally:
            if not state["finished"]:
                stats["aborted"] += 1

    return StreamingResponse(gen(), media_type="application/octet-stream")


async def _fixed(request):
    """Fixed-length response: verifies content-length pass-through."""
    request.app.state.stats["started"] += 1
    body = b"fixed-length-body" * 128
    headers = {"content-length": str(len(body)), "x-upstream-fixed": "1"}
    request.app.state.stats["completed"] += 1
    return Response(body, media_type="application/octet-stream", headers=headers)


async def _healthz(request):
    app = request.app
    app.state.probes += 1
    started = time.monotonic()
    if app.state.probe_delay:
        await asyncio.sleep(app.state.probe_delay)
    status = app.state.probe_status
    return JSONResponse(
        {"status": "ok" if status == 200 else "not_ready"},
        status_code=status,
        headers={"x-probe-delay": f"{time.monotonic() - started:.3f}"},
    )


async def _control(request):
    app = request.app
    if "delay" in request.query_params:
        app.state.probe_delay = float(request.query_params["delay"])
    if "status" in request.query_params:
        app.state.probe_status = int(request.query_params["status"])
    return JSONResponse({"probe_delay": app.state.probe_delay, "probe_status": app.state.probe_status})


async def _release_manifest(request):
    """Manifest endpoint with strong ETag, conditional GET, and fault knobs.

    ``app.state.manifest_raw`` (when set) short-circuits everything and is
    served verbatim: ``{"status": int, "body": str, "etag": str | None}``.
    """
    app = request.app
    state = app.state
    inm = request.headers.get("if-none-match")
    state.manifest_requests.append(
        {"if_none_match": inm, "conditional": inm is not None}
    )
    if state.manifest_delay:
        await asyncio.sleep(state.manifest_delay)

    raw = state.manifest_raw
    if raw is not None:
        etag = raw.get("etag")
        if inm is not None and etag is not None and state.conditional_status is None:
            tokens = [t.strip() for t in inm.split(",")]
            if etag in tokens or "*" in tokens:
                return Response(status_code=304, headers={"etag": etag})
        status = raw.get("status", 200)
        if inm is not None and state.conditional_status is not None:
            status = state.conditional_status
        headers = {"etag": etag} if etag is not None else {}
        return Response(raw.get("body", ""), status_code=status, headers=headers)

    if state.mutate_on_conditional and inm is not None:
        # Simulate a publish racing the comparison: the manifest changes
        # between the relay's fetch and its conditional revalidation.
        state.manifest["artifacts/mutated.bin"] = _sha256_text(f"mutated:{time.monotonic()}")
    if inm is not None and state.conditional_status is not None:
        return Response(status_code=state.conditional_status)

    etag = manifest_etag(state.manifest)
    if inm is not None:
        tokens = [t.strip() for t in inm.split(",")]
        if etag in tokens or f"W/{etag}" in tokens or "*" in tokens:
            return Response(status_code=304, headers={"etag": etag})
    body = json.dumps(state.manifest, sort_keys=True)
    return Response(body, media_type="application/json", headers={"etag": etag})


async def _manifest_control(request):
    """Test control for the manifest endpoint.

    JSON body keys (all optional):
      set / remove / reset  — edit the manifest entries
      raw                   — verbatim response {status, body, etag} or null
      delay                 — seconds to wait before any manifest response
      mutate_on_conditional — change the manifest when a conditional GET arrives
      conditional_status    — force this status for conditional GETs
    """
    app = request.app
    state = app.state
    body = await request.json()
    if body.get("reset"):
        state.manifest = default_manifest(state.name)
    for path in body.get("remove") or []:
        state.manifest.pop(path, None)
    for path, digest in (body.get("set") or {}).items():
        state.manifest[path] = digest
    if "raw" in body:
        state.manifest_raw = body["raw"]
    if "delay" in body:
        state.manifest_delay = float(body["delay"])
    if "mutate_on_conditional" in body:
        state.mutate_on_conditional = bool(body["mutate_on_conditional"])
    if "conditional_status" in body:
        value = body["conditional_status"]
        state.conditional_status = None if value is None else int(value)
    return JSONResponse(
        {
            "name": state.name,
            "etag": manifest_etag(state.manifest),
            "items": len(state.manifest),
            "manifest": state.manifest,
            "raw": state.manifest_raw,
            "delay": state.manifest_delay,
            "mutate_on_conditional": state.mutate_on_conditional,
            "conditional_status": state.conditional_status,
        }
    )


async def _state(request):
    app = request.app
    return JSONResponse(
        {
            "name": app.state.name,
            "probe_delay": app.state.probe_delay,
            "probe_status": app.state.probe_status,
            "stats": app.state.stats,
            "probes": app.state.probes,
            "manifest_requests": app.state.manifest_requests,
        }
    )


def make_app(name: str) -> Starlette:
    app = Starlette(
        routes=[
            Route("/ready", _healthz),
            Route("/healthz", _healthz),
            Route("/artifact", _artifact),
            Route("/fixed", _fixed),
            Route("/release-manifest.json", _release_manifest),
            Route("/_control/probe", _control, methods=["POST", "GET"]),
            Route("/_control/manifest", _manifest_control, methods=["POST", "GET"]),
            Route("/_state", _state),
        ]
    )
    app.state.name = name
    app.state.probe_delay = 0.0
    app.state.probe_status = 200
    app.state.stats = {"started": 0, "completed": 0, "cancelled": 0, "aborted": 0}
    app.state.probes = 0
    app.state.manifest = default_manifest(name)
    app.state.manifest_raw = None
    app.state.manifest_delay = 0.0
    app.state.mutate_on_conditional = False
    app.state.conditional_status = None
    app.state.manifest_requests = []
    return app
