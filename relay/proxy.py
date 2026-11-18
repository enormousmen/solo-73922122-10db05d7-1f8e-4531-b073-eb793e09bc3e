"""FastAPI application: single-process streaming GET proxy + switch API."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager, suppress
from typing import Iterable

import aiohttp
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel, Field
from starlette.types import Receive, Scope, Send

from .config import Config
from .generations import Generation, HOP_BY_HOP, InFlightRequest, RelayState
from . import release_diff

log = logging.getLogger("relay.proxy")

JSON_HEADERS = {"cache-control": "no-store", "x-content-type-options": "nosniff"}
# Response headers that describe the upstream hop, not the payload.  ``date``
# and ``server`` are dropped as well: uvicorn emits its own, and duplicates
# would make the relay's own identity ambiguous.
DROP_RESPONSE_HEADERS = HOP_BY_HOP | {
    "connection",
    "keep-alive",
    "content-length",
    "transfer-encoding",
    "trailer",
    "upgrade",
    "date",
    "server",
}


def _header_pairs(raw_headers: Iterable[tuple[bytes, bytes]]) -> list[tuple[str, str]]:
    return [(k.decode("latin-1"), v.decode("latin-1")) for k, v in raw_headers]


def _connection_tokens(headers: list[tuple[str, str]]) -> set[str]:
    tokens: set[str] = set()
    for name, value in headers:
        if name.lower() == "connection":
            tokens.update(part.strip().lower() for part in value.split(",") if part.strip())
    return tokens


def _strip_hop_by_hop(headers: list[tuple[str, str]]) -> list[tuple[str, str]]:
    drop = set(HOP_BY_HOP) | _connection_tokens(headers)
    return [(k, v) for k, v in headers if k.lower() not in drop]


def _target_url(generation: Generation, scope: Scope) -> str:
    raw_path = scope.get("raw_path")
    path = raw_path.decode("latin-1") if raw_path else scope.get("path", "/")
    if not path.startswith("/"):
        path = "/" + path
    query = scope.get("query_string", b"")
    if query:
        path = path + "?" + query.decode("latin-1")
    return generation.base_url + path


class SwitchRequest(BaseModel):
    target: str = Field(..., description="Target name from RELAY_TARGETS", min_length=1)
    wait: bool = Field(False, description="Block until the readiness probe finishes")
    wait_timeout: float = Field(30.0, gt=0, le=30.0, description="Max seconds to wait")


class ReleaseDiffRequest(BaseModel):
    target: str = Field(..., description="Candidate target name from RELAY_TARGETS", min_length=1)


class RelayApp:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.state = RelayState(config)
        self.app = self._build()

    # -- lifespan -----------------------------------------------------------
    @asynccontextmanager
    async def lifespan(self, app: FastAPI):
        # Two independent sessions/connectors: a slow download on the proxy
        # session can never consume the connections a readiness probe needs.
        self.state.upstream_session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(
                total=None,
                connect=self.config.connect_timeout,
                sock_read=self.config.sock_read_timeout or None,
            ),
            auto_decompress=False,
            connector=aiohttp.TCPConnector(limit=0, enable_cleanup_closed=True),
            trust_env=False,
        )
        self.state.probe_session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.probe_timeout),
            auto_decompress=False,
            connector=aiohttp.TCPConnector(limit=16, enable_cleanup_closed=True),
            trust_env=False,
        )
        try:
            yield
        finally:
            await self.state.aclose()
            await self.state.upstream_session.close()
            await self.state.probe_session.close()

    # -- app ----------------------------------------------------------------
    def _build(self) -> FastAPI:
        app = FastAPI(
            title="Artifact Relay",
            version="1.0.0",
            description=(
                "Switchable, streaming HTTP GET proxy for internal artifact downloads.\n\n"
                "* `POST /api/switch` probes the candidate. HTTP 200 commits; any other "
                "result keeps the current target.\n"
                "* A newer accepted switch supersedes an unfinished candidate, and a late "
                "probe result can never flip traffic.\n"
                "* Each GET is pinned to the generation that was current when it was "
                "accepted. After a commit the previous generation drains for "
                "`RELAY_DRAIN_SECONDS` (default 5s); when the window expires only that "
                "generation's unfinished requests are interrupted. Requests on newer "
                "generations are untouched.\n"
                "* Response bodies are forwarded chunk by chunk, never buffered whole.\n"
                "* `POST /api/release-diff` compares the release manifests "
                "(`/release-manifest.json`) of the current generation and a candidate "
                "target. Both manifests must be HTTP 200 with a strong `ETag`; before "
                "publishing the diff the relay revalidates both with conditional "
                "requests and re-checks the current generation. Any change, a failed "
                "revalidation, or a generation switch yields 409 and no diff.\n\n"
                "Open **GET /demo** for a click-through download demo."
            ),
            lifespan=self.lifespan,
        )
        app.get("/api/status", tags=["relay"])(self.api_status)
        app.get("/api/targets", tags=["relay"])(self.api_targets)
        app.post("/api/switch", tags=["relay"])(self.api_switch)
        app.post("/api/release-diff", tags=["relay"])(self.api_release_diff)
        app.get("/api/operations/{operation_id}", tags=["relay"])(self.api_operation)
        app.get("/demo", response_class=HTMLResponse, tags=["demo"])(self.demo_page)
        app.get("/", response_class=HTMLResponse, tags=["demo"])(self.demo_page)
        # The GET catch-all must be registered last.
        app.api_route("/{path:path}", methods=["GET", "HEAD"], include_in_schema=False)(self.proxy_get)
        return app

    # -- control API --------------------------------------------------------
    async def api_status(self) -> JSONResponse:
        # Synchronous snapshot: never awaits a probe, a download, or a lock,
        # so a slow probe/download cannot block this endpoint.
        return JSONResponse(self.state.snapshot(), headers=JSON_HEADERS)

    async def api_targets(self) -> JSONResponse:
        return JSONResponse(
            {
                "targets": {name: t.base_url for name, t in self.config.targets.items()},
                "current": self.state.current.target_name,
                "current_generation": self.state.current.id,
                "probe_path": self.config.probe_path,
                "drain_seconds": self.config.drain_seconds,
            },
            headers=JSON_HEADERS,
        )

    async def api_switch(self, body: SwitchRequest) -> Response:
        if body.target not in self.config.targets:
            raise HTTPException(
                status_code=404,
                detail=f"unknown target {body.target!r}; known: {sorted(self.config.targets)}",
            )
        operation = await self.state.begin_switch(body.target)
        if body.wait:
            with suppress(asyncio.TimeoutError):
                await asyncio.wait_for(operation.changed.wait(), timeout=body.wait_timeout)
        payload = {
            "operation": operation.to_dict(),
            "current": self.state.current.target_name,
            "current_generation": self.state.current.id,
        }
        code = 202 if operation.status == "pending" else 200
        return JSONResponse(payload, status_code=code, headers=JSON_HEADERS)

    async def api_operation(self, operation_id: int) -> JSONResponse:
        for operation in self.state.operations:
            if operation.id == operation_id:
                return JSONResponse({"operation": operation.to_dict()}, headers=JSON_HEADERS)
        raise HTTPException(status_code=404, detail=f"no operation {operation_id}")

    async def api_release_diff(self, body: ReleaseDiffRequest) -> Response:
        """Diff the release manifests of the current generation and a candidate.

        The diff is only published when both manifests were fetched as 200 with
        strong ETags, both still revalidate to those ETags via conditional
        requests, and the current generation has not switched in the meantime.
        Every failure path returns an error object *without* any diff fields.
        """
        if body.target not in self.config.targets:
            raise HTTPException(
                status_code=404,
                detail=f"unknown target {body.target!r}; known: {sorted(self.config.targets)}",
            )
        candidate = self.config.targets[body.target]
        # Snapshot the generation that is current at accept time.  This runs
        # synchronously before the first await, so a concurrent commit either
        # happened before (we see the new generation) or after (we detect the
        # switch below and return 409).
        generation = self.state.current
        current_src = release_diff.Source(
            side="current", name=generation.target_name, base_url=generation.base_url
        )
        candidate_src = release_diff.Source(
            side="candidate", name=candidate.name, base_url=candidate.base_url
        )
        session = self.state.probe_session
        timeout = self.config.manifest_timeout
        try:
            current_manifest, candidate_manifest = await release_diff.fetch_pair(
                session, current_src, candidate_src, timeout
            )
        except release_diff.ManifestError as exc:
            log.info("release-diff vs %s aborted: %s", candidate.name, exc.detail)
            return JSONResponse(
                {
                    "error": exc.error,
                    "detail": exc.detail,
                    "side": exc.side,
                    "candidate": candidate.name,
                    "generation": generation.id,
                },
                status_code=502,
                headers=JSON_HEADERS,
            )
        # Both manifests are parsed; confirm they are still current before
        # publishing anything derived from them.
        try:
            await release_diff.revalidate_pair(
                session,
                current_src,
                current_manifest.etag,
                candidate_src,
                candidate_manifest.etag,
                timeout,
            )
        except release_diff.ManifestConflict as exc:
            log.info("release-diff vs %s conflicted: %s", candidate.name, exc.detail)
            return JSONResponse(
                {
                    "error": exc.error,
                    "detail": exc.detail,
                    "side": exc.side,
                    "candidate": candidate.name,
                    "generation_at_accept": generation.id,
                    "current_generation": self.state.current.id,
                },
                status_code=409,
                headers=JSON_HEADERS,
            )
        if self.state.current is not generation:
            return JSONResponse(
                {
                    "error": "generation_switched",
                    "detail": (
                        f"current generation switched from {generation.id} to "
                        f"{self.state.current.id} while comparing; rerun the diff"
                    ),
                    "candidate": candidate.name,
                    "generation_at_accept": generation.id,
                    "current_generation": self.state.current.id,
                },
                status_code=409,
                headers=JSON_HEADERS,
            )
        diff = release_diff.diff_manifests(current_manifest, candidate_manifest)
        payload = {
            "candidate": candidate.name,
            "current_target": generation.target_name,
            "current_generation": generation.id,
            "etags": {"current": current_manifest.etag, "candidate": candidate_manifest.etag},
            "summary": {
                "missing_in_candidate": len(diff["missing_in_candidate"]),
                "digest_mismatch": len(diff["digest_mismatch"]),
                "added_in_candidate": len(diff["added_in_candidate"]),
            },
            **diff,
        }
        return JSONResponse(payload, headers=JSON_HEADERS)

    async def demo_page(self) -> HTMLResponse:
        return HTMLResponse(DEMO_HTML)

    # -- GET proxy ----------------------------------------------------------
    async def proxy_get(self, request: Request) -> Response:
        scope = request.scope
        # Resolve and register with no await in between: a concurrent commit
        # either happened before this tick (new target) or happens after (this
        # request belongs to the old generation and drains normally).
        generation = self.state.current
        in_flight = generation.register(scope.get("path", "/"))

        request_headers = [
            (k, v)
            for k, v in _strip_hop_by_hop(_header_pairs(scope.get("headers", [])))
            if k.lower() not in ("host", "content-length")
        ]
        url = _target_url(generation, scope)

        disconnect_event = asyncio.Event()
        disconnect_task = asyncio.create_task(
            _watch_disconnect(request.receive, disconnect_event),
            name=f"relay-disconnect-{in_flight.id}",
        )
        abort_wait = asyncio.ensure_future(in_flight.event.wait())
        disconnect_wait = asyncio.ensure_future(disconnect_event.wait())
        upstream_response: aiohttp.ClientResponse | None = None
        handed_off = False
        try:
            get_task = asyncio.ensure_future(
                self.state.upstream_session.request(
                    scope.get("method", "GET"), url, headers=request_headers, allow_redirects=False
                )
            )
            done, _ = await asyncio.wait(
                {get_task, abort_wait, disconnect_wait},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if get_task not in done:
                get_task.cancel()
                with suppress(BaseException):
                    await get_task
                if disconnect_event.is_set():
                    return self._error_response(
                        499, "client_disconnected", "client went away", generation, in_flight
                    )
                return self._drain_response(in_flight, generation)
            try:
                upstream_response = get_task.result()
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                status = 504 if isinstance(exc, asyncio.TimeoutError) else 502
                return self._error_response(
                    status,
                    "upstream_timeout" if status == 504 else "upstream_error",
                    f"{type(exc).__name__}: {exc}",
                    generation,
                    in_flight,
                )
            except asyncio.CancelledError:
                raise

            response_headers = [
                (k, v)
                for k, v in _strip_hop_by_hop(list(upstream_response.headers.items()))
                if k.lower() not in DROP_RESPONSE_HEADERS and not k.lower().startswith("x-relay-")
            ]
            # Body framing is ours. If upstream declared a length we pass it on
            # (bytes are relayed verbatim because decompression is disabled).
            length = upstream_response.headers.get("Content-Length")
            if length is not None and upstream_response.status not in (204, 304):
                response_headers.append(("content-length", length))
            response_headers.extend(
                [
                    ("x-relay-target", generation.target_name),
                    ("x-relay-generation", str(generation.id)),
                    ("x-relay-request-id", str(in_flight.id)),
                ]
            )
            response = StreamingUpstreamResponse(
                relay=self,
                upstream=upstream_response,
                in_flight=in_flight,
                generation=generation,
                status_code=upstream_response.status,
                headers=response_headers,
                abort_wait=abort_wait,
                disconnect=disconnect_event,
                disconnect_task=disconnect_task,
                disconnect_wait=disconnect_wait,
            )
            handed_off = True
            return response
        finally:
            if not handed_off:
                disconnect_wait.cancel()
                if upstream_response is not None:
                    upstream_response.close()  # abort an unread body, no reuse
                disconnect_task.cancel()
                with suppress(BaseException):
                    await disconnect_task
                abort_wait.cancel()
                generation.unregister(in_flight)

    def _drain_response(self, in_flight: InFlightRequest, generation: Generation) -> JSONResponse:
        return JSONResponse(
            {
                "error": "generation_draining",
                "detail": "drain window expired before the upstream responded",
                "generation": generation.id,
                "target": generation.target_name,
                "request_id": in_flight.id,
            },
            status_code=503,
            headers={
                **JSON_HEADERS,
                "x-relay-generation": str(generation.id),
                "connection": "close",
            },
        )

    def _error_response(
        self,
        status: int,
        error: str,
        detail: str,
        generation: Generation,
        in_flight: InFlightRequest,
    ) -> JSONResponse:
        return JSONResponse(
            {
                "error": error,
                "detail": detail,
                "generation": generation.id,
                "target": generation.target_name,
                "request_id": in_flight.id,
            },
            status_code=status,
            headers={**JSON_HEADERS, "x-relay-generation": str(generation.id)},
        )


async def _watch_disconnect(receive: Receive, event: asyncio.Event) -> None:
    """Set ``event`` when the client disconnects; runs for the response lifetime.

    Uvicorn delivers ``http.disconnect`` only through ``receive()``; there is no
    other signal for an aborted download, so this task is what frees the
    upstream connection when the client goes away.
    """
    try:
        while not event.is_set():
            message = await receive()
            if message["type"] == "http.disconnect":
                event.set()
                return
    except asyncio.CancelledError:
        raise
    except Exception:  # pragma: no cover - defensive
        event.set()


class StreamingUpstreamResponse(Response):
    """Chunk-by-chunk ASGI response pinned to one upstream generation.

    Nothing is buffered.  When the generation's drain timer fires mid-body the
    response stops sending and returns without completing the ASGI response;
    uvicorn then closes that one client connection, which the client observes
    as a truncated transfer.  Other connections/generations are unaffected.
    """

    def __init__(
        self,
        *,
        relay: RelayApp,
        upstream: aiohttp.ClientResponse,
        in_flight: InFlightRequest,
        generation: Generation,
        status_code: int,
        headers: list[tuple[str, str]],
        abort_wait: asyncio.Future,
        disconnect: asyncio.Event,
        disconnect_task: asyncio.Task,
        disconnect_wait: asyncio.Future,
    ) -> None:
        self.relay = relay
        self.upstream = upstream
        self.in_flight = in_flight
        self.generation = generation
        self.abort_wait = abort_wait
        self.disconnect = disconnect
        self.disconnect_task = disconnect_task
        self.disconnect_wait = disconnect_wait
        self.media_type = None
        self.status_code = status_code
        self.background = None
        self.raw_headers = [(k.lower().encode("latin-1"), v.encode("latin-1")) for k, v in headers]

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        outcome = "completed"
        try:
            await send(
                {"type": "http.response.start", "status": self.status_code, "headers": self.raw_headers}
            )
            if self.status_code in (204, 304):
                await send({"type": "http.response.body", "body": b"", "more_body": False})
                return

            body = self.upstream.content.iter_chunked(self.relay.config.chunk_size)
            while True:
                abort_reason = self.in_flight.abort_reason
                if abort_reason is not None:
                    outcome = f"drain_expired ({abort_reason})"
                    break
                if self.disconnect.is_set():
                    outcome = "client_disconnected"
                    break
                chunk_task = asyncio.ensure_future(body.__anext__())
                done, _ = await asyncio.wait(
                    {chunk_task, self.abort_wait},
                    return_when=asyncio.FIRST_COMPLETED,
                    timeout=0.25,  # also gives the disconnect flag a polling cadence
                )
                if chunk_task not in done:
                    chunk_task.cancel()
                    with suppress(BaseException):
                        await chunk_task
                    continue
                try:
                    chunk = chunk_task.result()
                except StopAsyncIteration:
                    break
                except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                    outcome = f"upstream_error: {type(exc).__name__}: {exc}"
                    break
                except asyncio.CancelledError:
                    raise
                if not await self._send_body(send, chunk):
                    outcome = self.in_flight.abort_reason or "drain_expired"
                    break

            if outcome == "completed":
                await send({"type": "http.response.body", "body": b"", "more_body": False})
            else:
                # Deliberately no final body message: uvicorn closes this
                # connection and the client sees a truncated download, while
                # other in-flight generations keep streaming.
                log.warning(
                    "interrupting request %s on generation %s (%s): %s",
                    self.in_flight.id,
                    self.generation.id,
                    self.generation.target_name,
                    outcome,
                )
        finally:
            await self._release()

    async def _send_body(self, send: Send, body: bytes) -> bool:
        """Send one chunk; return False if the generation was drained first.

        ``send`` may block on HTTP flow control for a slow reader; waiting on
        the generation's abort signal alongside it keeps the drain deadline
        honest even then.
        """
        send_task = asyncio.ensure_future(
            send({"type": "http.response.body", "body": body, "more_body": True})
        )
        done, _ = await asyncio.wait(
            {send_task, self.abort_wait}, return_when=asyncio.FIRST_COMPLETED
        )
        if send_task not in done:
            send_task.cancel()
            with suppress(BaseException):
                await send_task
            return False
        return True

    async def _release(self) -> None:
        # ``close`` (not ``release``): an interrupted body must not leave a
        # half-read connection back in the pool.
        self.upstream.close()
        await self.upstream.wait_for_close()
        self.disconnect_task.cancel()
        self.disconnect_wait.cancel()
        with suppress(BaseException):
            await self.disconnect_task
        if not self.abort_wait.done():
            self.abort_wait.cancel()
        else:
            with suppress(BaseException):
                self.abort_wait.result()
        self.generation.unregister(self.in_flight)


DEMO_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Artifact Relay demo</title>
<style>
  body { font-family: system-ui, sans-serif; margin: 2rem auto; max-width: 62rem; padding: 0 1rem; }
  code, pre { background: #f3f4f6; padding: .15rem .3rem; border-radius: 4px; }
  table { border-collapse: collapse; width: 100%; margin: .5rem 0 1.5rem; }
  th, td { border: 1px solid #d1d5db; padding: .35rem .5rem; text-align: left; font-size: .9rem; }
  .row { display: flex; gap: .5rem; flex-wrap: wrap; align-items: center; margin: .5rem 0; }
  button { padding: .4rem .8rem; cursor: pointer; }
  .muted { color: #6b7280; }
</style>
</head>
<body>
<h1>Artifact Relay</h1>
<p class="muted">One process, two upstreams (blue/green). API docs: <a href="/docs">/docs</a>.</p>

<h2>Switch traffic</h2>
<div class="row">
  <button onclick="switchTo('blue')">Switch to blue</button>
  <button onclick="switchTo('green')">Switch to green</button>
  <button onclick="refresh()">Refresh status</button>
  <span id="switch-msg" class="muted"></span>
</div>

<h2>Current target</h2>
<pre id="current">loading…</pre>

<h2>Generations / in-flight</h2>
<table>
  <thead><tr><th>gen</th><th>target</th><th>state</th><th>in flight</th><th>drain left (s)</th></tr></thead>
  <tbody id="generations"></tbody>
</table>

<h2>Download demo</h2>
<p>Links go through the relay at its current target. Start a slow download, then switch.</p>
<div class="row">
  <a target="_blank" href="/artifact?bytes=1048576&chunk=65536&delay=0.2">1 MiB fast</a>
  <a target="_blank" href="/artifact?bytes=3145728&chunk=32768&delay=0.5">3 MiB slow (≈45s)</a>
  <a target="_blank" href="/artifact?bytes=10485760&chunk=65536&delay=0.05">10 MiB medium</a>
</div>
<p class="muted">Both upstreams expose <code>/artifact</code> and a probe-delay control:
  <code>curl -X POST 'http://localhost:9001/_control/probe?delay=4'</code>,
  <code>curl -X POST 'http://localhost:9002/_control/probe?delay=4'</code>.
  Add <code>&amp;status=503</code> to make the probe fail.</p>

<h2>Release diff</h2>
<p>Compare <code>/release-manifest.json</code> of the current generation with a candidate
  (both manifests are revalidated with conditional requests before the diff is shown).</p>
<div class="row">
  <button onclick="releaseDiff('blue')">Diff current vs blue</button>
  <button onclick="releaseDiff('green')">Diff current vs green</button>
</div>
<pre id="diff-result">no diff requested yet</pre>
<p class="muted">Edit a manifest live:
  <code>curl -X POST http://localhost:9001/_control/manifest -H 'content-type: application/json'
  -d '{"set":{"artifacts/x.bin":"&lt;64 hex chars&gt;"}}'</code>
  (<code>remove</code> takes a list of paths, <code>"reset":true</code> restores the default).</p>

<script>
async function refresh() {
  try {
    const r = await fetch('/api/status');
    const s = await r.json();
    document.getElementById('current').textContent =
      `target=${s.current.target} generation=${s.current.generation} base=${s.current.base_url}\\n` +
      `pending=${s.pending ? s.pending.target + ' #' + s.pending.id + ' (' + s.pending.status + ')' : 'none'}`;
    document.getElementById('generations').innerHTML = s.generations.map(g =>
      `<tr><td>${g.id}</td><td>${g.target}</td><td>${g.state}</td><td>${g.in_flight}</td>` +
      `<td>${g.drain_remaining_seconds ?? ''}</td></tr>`).join('');
  } catch (e) { /* transient */ }
}
async function switchTo(target) {
  document.getElementById('switch-msg').textContent = `probing ${target}…`;
  const r = await fetch('/api/switch', {
    method: 'POST', headers: {'content-type': 'application/json'},
    body: JSON.stringify({target, wait: true, wait_timeout: 6})
  });
  const j = await r.json();
  document.getElementById('switch-msg').textContent = r.status + ' ' + j.operation.status +
    (j.operation.reason ? ' — ' + j.operation.reason : '');
  refresh();
}
async function releaseDiff(target) {
  document.getElementById('diff-result').textContent = `comparing current vs ${target}…`;
  const r = await fetch('/api/release-diff', {
    method: 'POST', headers: {'content-type': 'application/json'},
    body: JSON.stringify({target})
  });
  const j = await r.json();
  document.getElementById('diff-result').textContent = r.status + '\n' + JSON.stringify(j, null, 2);
}
refresh();
setInterval(refresh, 1000);
</script>
</body>
</html>
"""
