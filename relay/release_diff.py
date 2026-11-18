"""Release-manifest fetching, validation, and diffing for ``POST /api/release-diff``.

A *release manifest* is a JSON object served by every upstream at
``/release-manifest.json`` mapping artifact paths to their SHA-256 hex digest.
The relay only trusts manifests that

* come back as HTTP 200 with a **strong** ``ETag`` (needed for revalidation),
* are at most :data:`MAX_MANIFEST_BYTES` bytes and :data:`MAX_MANIFEST_ITEMS`
  entries, and
* have the exact shape ``{path: sha256hex}``.

After both manifests are fetched, the relay re-requests each one with
``If-None-Match`` and only publishes a diff when both upstreams answer
``304 Not Modified`` — i.e. the compared snapshots are still current.  Any
change, any failed conditional request, or a generation switch (checked by the
caller) aborts the comparison with a conflict and **no partial diff**.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from typing import Any

import aiohttp

log = logging.getLogger("relay.release_diff")

MANIFEST_PATH = "/release-manifest.json"
MAX_MANIFEST_BYTES = 64 * 1024
MAX_MANIFEST_ITEMS = 200
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


class ManifestError(Exception):
    """The manifest could not be read or failed validation (HTTP 502).

    No diff may be produced from a read/format error, so the endpoint maps
    this to an error response without any diff fields.
    """

    def __init__(self, error: str, detail: str, *, side: str | None = None) -> None:
        super().__init__(detail)
        self.error = error
        self.detail = detail
        self.side = side


class ManifestConflict(Exception):
    """A manifest changed or could not be revalidated (HTTP 409)."""

    def __init__(self, error: str, detail: str, *, side: str | None = None) -> None:
        super().__init__(detail)
        self.error = error
        self.detail = detail
        self.side = side


@dataclass(frozen=True)
class Source:
    """One upstream to read a manifest from (current generation or candidate)."""

    side: str  # "current" | "candidate"
    name: str  # target name, for messages
    base_url: str


@dataclass(frozen=True)
class Manifest:
    etag: str
    items: dict[str, str]  # artifact path -> sha256 hex (lowercase)


def _strong_etag(headers: Any, *, side: str) -> str:
    """Extract a strong validator or raise.  Weak/missing ETags are rejected:
    they cannot anchor the conditional revalidation this flow relies on."""
    etag = headers.get("ETag")
    if etag is None:
        raise ManifestError(
            "manifest_etag_missing",
            "manifest response has no ETag; a strong ETag is required",
            side=side,
        )
    etag = etag.strip()
    if not etag or etag[:2].upper() == "W/":
        raise ManifestError(
            "manifest_etag_weak",
            f"manifest ETag {etag!r} is weak; a strong ETag is required",
            side=side,
        )
    if len(etag) < 2 or not (etag.startswith('"') and etag.endswith('"')):
        raise ManifestError(
            "manifest_etag_invalid",
            f"manifest ETag {etag!r} is not a quoted strong entity-tag",
            side=side,
        )
    return etag


def parse_manifest(body: bytes, *, side: str = "") -> dict[str, str]:
    """Validate ``body`` as a manifest and return ``{path: sha256}``.

    Digests are normalised to lowercase so the comparison is by value, not by
    spelling.  Anything outside the contract raises :class:`ManifestError`.
    """
    try:
        data = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
        raise ManifestError("manifest_invalid_json", f"manifest is not valid JSON: {exc}", side=side) from exc
    if not isinstance(data, dict):
        raise ManifestError(
            "manifest_invalid_shape",
            "manifest must be a JSON object mapping artifact path to sha256",
            side=side,
        )
    if len(data) > MAX_MANIFEST_ITEMS:
        raise ManifestError(
            "manifest_too_many_items",
            f"manifest has {len(data)} entries; the limit is {MAX_MANIFEST_ITEMS}",
            side=side,
        )
    items: dict[str, str] = {}
    for path, digest in data.items():
        if not isinstance(path, str) or not path:
            raise ManifestError("manifest_invalid_path", f"invalid artifact path {path!r}", side=side)
        if not isinstance(digest, str) or not _SHA256_RE.match(digest):
            raise ManifestError(
                "manifest_invalid_digest",
                f"entry {path!r} is not a 64-char hex SHA-256 digest",
                side=side,
            )
        items[path] = digest.lower()
    return items


async def _read_limited(response: aiohttp.ClientResponse, *, side: str) -> bytes:
    """Read the body, aborting as soon as it exceeds the manifest size cap."""
    declared = response.content_length
    if declared is not None and declared > MAX_MANIFEST_BYTES:
        raise ManifestError(
            "manifest_too_large",
            f"manifest is {declared} bytes; the limit is {MAX_MANIFEST_BYTES}",
            side=side,
        )
    chunks: list[bytes] = []
    size = 0
    async for chunk in response.content.iter_any():
        size += len(chunk)
        if size > MAX_MANIFEST_BYTES:
            raise ManifestError(
                "manifest_too_large",
                f"manifest exceeds the {MAX_MANIFEST_BYTES}-byte limit",
                side=side,
            )
        chunks.append(chunk)
    return b"".join(chunks)


async def fetch_manifest(
    session: aiohttp.ClientSession, source: Source, timeout: float
) -> Manifest:
    """GET one manifest and validate status, ETag, size, and shape."""
    url = source.base_url + MANIFEST_PATH
    try:
        async with session.get(
            url,
            timeout=aiohttp.ClientTimeout(total=timeout),
            allow_redirects=False,
        ) as response:
            if response.status != 200:
                raise ManifestError(
                    "manifest_unavailable",
                    f"GET {url} returned HTTP {response.status}",
                    side=source.side,
                )
            etag = _strong_etag(response.headers, side=source.side)
            body = await _read_limited(response, side=source.side)
    except ManifestError:
        raise
    except asyncio.TimeoutError as exc:
        raise ManifestError(
            "manifest_unavailable", f"timeout reading {url}", side=source.side
        ) from exc
    except aiohttp.ClientError as exc:
        raise ManifestError(
            "manifest_unavailable",
            f"GET {url} failed: {type(exc).__name__}: {exc}",
            side=source.side,
        ) from exc
    return Manifest(etag=etag, items=parse_manifest(body, side=source.side))


async def _fetch_quiet(
    session: aiohttp.ClientSession, source: Source, timeout: float
) -> "Manifest | ManifestError":
    try:
        return await fetch_manifest(session, source, timeout)
    except ManifestError as exc:
        return exc


async def fetch_pair(
    session: aiohttp.ClientSession,
    current: Source,
    candidate: Source,
    timeout: float,
) -> tuple[Manifest, Manifest]:
    """Fetch both manifests concurrently; raise the first :class:`ManifestError`.

    Both fetches always run to completion (or failure) so no upstream request
    is left dangling when one side errors out.
    """
    current_result, candidate_result = await asyncio.gather(
        _fetch_quiet(session, current, timeout),
        _fetch_quiet(session, candidate, timeout),
    )
    for result in (current_result, candidate_result):
        if isinstance(result, ManifestError):
            raise result
    return current_result, candidate_result  # type: ignore[return-value]


async def revalidate_manifest(
    session: aiohttp.ClientSession, source: Source, etag: str, timeout: float
) -> None:
    """Confirm via ``If-None-Match`` that the manifest is still ``etag``.

    Returns quietly on 304.  A 200 means the manifest changed; anything else
    means the conditional check itself failed.  Both raise
    :class:`ManifestConflict` — the caller must not publish a diff.
    """
    url = source.base_url + MANIFEST_PATH
    try:
        async with session.get(
            url,
            headers={"If-None-Match": etag},
            timeout=aiohttp.ClientTimeout(total=timeout),
            allow_redirects=False,
        ) as response:
            status = response.status
    except asyncio.TimeoutError as exc:
        raise ManifestConflict(
            "revalidation_failed", f"timeout revalidating {url}", side=source.side
        ) from exc
    except aiohttp.ClientError as exc:
        raise ManifestConflict(
            "revalidation_failed",
            f"conditional GET {url} failed: {type(exc).__name__}: {exc}",
            side=source.side,
        ) from exc
    if status == 304:
        return
    if status == 200:
        raise ManifestConflict(
            "manifest_changed",
            f"{source.name} manifest changed while comparing (ETag {etag} is stale)",
            side=source.side,
        )
    raise ManifestConflict(
        "revalidation_failed",
        f"conditional GET {url} returned HTTP {status}; cannot confirm the manifest is unchanged",
        side=source.side,
    )


async def _revalidate_quiet(
    session: aiohttp.ClientSession, source: Source, etag: str, timeout: float
) -> "ManifestConflict | None":
    try:
        await revalidate_manifest(session, source, etag, timeout)
        return None
    except ManifestConflict as exc:
        return exc


async def revalidate_pair(
    session: aiohttp.ClientSession,
    current: Source,
    current_etag: str,
    candidate: Source,
    candidate_etag: str,
    timeout: float,
) -> None:
    """Revalidate both manifests concurrently; raise the first conflict."""
    current_conflict, candidate_conflict = await asyncio.gather(
        _revalidate_quiet(session, current, current_etag, timeout),
        _revalidate_quiet(session, candidate, candidate_etag, timeout),
    )
    for conflict in (current_conflict, candidate_conflict):
        if conflict is not None:
            raise conflict


def diff_manifests(current: Manifest, candidate: Manifest) -> dict[str, Any]:
    """Compute the operator-facing diff between the two validated manifests."""
    missing = sorted(path for path in current.items if path not in candidate.items)
    added = sorted(path for path in candidate.items if path not in current.items)
    mismatched = sorted(
        path
        for path in current.items.keys() & candidate.items.keys()
        if current.items[path] != candidate.items[path]
    )
    return {
        "missing_in_candidate": missing,
        "added_in_candidate": added,
        "digest_mismatch": [
            {
                "path": path,
                "current_sha256": current.items[path],
                "candidate_sha256": candidate.items[path],
            }
            for path in mismatched
        ],
    }
