"""Tests for POST /api/release-diff: manifest fetch, validation, revalidation.

Integration tests run the real relay against two in-process upstreams over TCP.
The fake upstreams serve a mutable /release-manifest.json with a strong ETag
and conditional-request support, plus fault knobs for the error paths.
"""

from __future__ import annotations

import asyncio
import json

import aiohttp
import pytest

from relay import release_diff

from .test_proxy_integration import stack, status, switch

DIFF_KEYS = ("missing_in_candidate", "added_in_candidate", "digest_mismatch")


def assert_no_diff_fields(payload: dict) -> None:
    for key in DIFF_KEYS + ("summary", "etags"):
        assert key not in payload, f"error response must not contain {key!r}: {payload}"


async def release_diff_request(base: str, target: str) -> tuple[int, dict]:
    async with aiohttp.ClientSession() as session:
        async with session.post(base + "/api/release-diff", json={"target": target}) as response:
            return response.status, await response.json()


async def manifest_control(url: str, **body) -> dict:
    async with aiohttp.ClientSession() as session:
        async with session.post(url + "/_control/manifest", json=body) as response:
            return await response.json()


async def manifest_requests(url: str) -> list[dict]:
    async with aiohttp.ClientSession() as session:
        async with session.get(url + "/_state") as response:
            return (await response.json())["manifest_requests"]


# -- unit tests: manifest parsing and ETag validation (no network) -----------


def test_parse_manifest_accepts_valid_and_normalizes_case():
    body = json.dumps({"a.bin": "A" * 64, "dir/b.bin": "b" * 64}).encode()
    items = release_diff.parse_manifest(body)
    assert items == {"a.bin": "a" * 64, "dir/b.bin": "b" * 64}


def test_parse_manifest_rejects_invalid_json():
    with pytest.raises(release_diff.ManifestError) as excinfo:
        release_diff.parse_manifest(b"not json{")
    assert excinfo.value.error == "manifest_invalid_json"


def test_parse_manifest_rejects_non_object():
    with pytest.raises(release_diff.ManifestError) as excinfo:
        release_diff.parse_manifest(b'["a", "b"]')
    assert excinfo.value.error == "manifest_invalid_shape"


def test_parse_manifest_rejects_too_many_items():
    body = json.dumps({f"f{i}.bin": "a" * 64 for i in range(201)}).encode()
    with pytest.raises(release_diff.ManifestError) as excinfo:
        release_diff.parse_manifest(body)
    assert excinfo.value.error == "manifest_too_many_items"
    # Exactly at the limit is fine.
    ok = json.dumps({f"f{i}.bin": "a" * 64 for i in range(200)}).encode()
    assert len(release_diff.parse_manifest(ok)) == 200


def test_parse_manifest_rejects_bad_digest_and_empty_path():
    with pytest.raises(release_diff.ManifestError) as excinfo:
        release_diff.parse_manifest(json.dumps({"a.bin": "not-a-sha"}).encode())
    assert excinfo.value.error == "manifest_invalid_digest"
    with pytest.raises(release_diff.ManifestError) as excinfo:
        release_diff.parse_manifest(json.dumps({"a.bin": "a" * 63}).encode())
    assert excinfo.value.error == "manifest_invalid_digest"
    with pytest.raises(release_diff.ManifestError) as excinfo:
        release_diff.parse_manifest(json.dumps({"": "a" * 64}).encode())
    assert excinfo.value.error == "manifest_invalid_path"


def test_strong_etag_validation():
    assert release_diff._strong_etag({"ETag": '"abc123"'}, side="current") == '"abc123"'
    assert release_diff._strong_etag({"ETag": ' "abc123" '}, side="current") == '"abc123"'
    for headers, error in (
        ({}, "manifest_etag_missing"),
        ({"ETag": 'W/"abc"'}, "manifest_etag_weak"),
        ({"ETag": 'w/"abc"'}, "manifest_etag_weak"),
        ({"ETag": "abc"}, "manifest_etag_invalid"),
        ({"ETag": ""}, "manifest_etag_weak"),
    ):
        with pytest.raises(release_diff.ManifestError) as excinfo:
            release_diff._strong_etag(headers, side="current")
        assert excinfo.value.error == error, headers


# -- integration: happy paths -------------------------------------------------


async def test_release_diff_lists_missing_mismatch_and_added():
    async with stack() as s:
        code, payload = await release_diff_request(s["base"], "green")
        assert code == 200
        assert payload["candidate"] == "green"
        assert payload["current_target"] == "blue"
        assert payload["current_generation"] == 1
        # Default manifests: one upstream-unique artifact each way and one
        # digest that differs between blue and green.
        assert payload["missing_in_candidate"] == ["artifacts/blue-only.bin"]
        assert payload["added_in_candidate"] == ["artifacts/green-only.bin"]
        mismatch = payload["digest_mismatch"]
        assert [m["path"] for m in mismatch] == ["artifacts/config.json"]
        assert len(mismatch[0]["current_sha256"]) == 64
        assert len(mismatch[0]["candidate_sha256"]) == 64
        assert mismatch[0]["current_sha256"] != mismatch[0]["candidate_sha256"]
        assert payload["summary"] == {
            "missing_in_candidate": 1,
            "digest_mismatch": 1,
            "added_in_candidate": 1,
        }
        # Both ETags are strong (quoted) and echoed back for the operator.
        for etag in payload["etags"].values():
            assert etag.startswith('"') and etag.endswith('"')
        # The diff must not register as in-flight download traffic.
        snap = await status(s["base"])
        assert all(g["in_flight"] == 0 for g in snap["generations"])
        # Both upstreams saw exactly one plain GET and one conditional GET
        # carrying the ETag the relay had fetched.
        for url, side in ((s["blue"], "current"), (s["green"], "candidate")):
            requests = await manifest_requests(url)
            assert [r["conditional"] for r in requests] == [False, True]
            assert requests[1]["if_none_match"] == payload["etags"][side]


async def test_release_diff_identical_manifests_produce_empty_diff():
    async with stack() as s:
        async with aiohttp.ClientSession() as session:
            async with session.get(s["blue"] + "/release-manifest.json") as response:
                blue_manifest = await response.json()
        await manifest_control(
            s["green"], set=blue_manifest, remove=["artifacts/green-only.bin"]
        )
        code, payload = await release_diff_request(s["base"], "green")
        assert code == 200
        assert payload["missing_in_candidate"] == []
        assert payload["added_in_candidate"] == []
        assert payload["digest_mismatch"] == []
        assert payload["summary"] == {
            "missing_in_candidate": 0,
            "digest_mismatch": 0,
            "added_in_candidate": 0,
        }


async def test_release_diff_against_current_target_itself_is_empty():
    async with stack() as s:
        code, payload = await release_diff_request(s["base"], "blue")
        assert code == 200
        assert payload["candidate"] == "blue"
        assert payload["missing_in_candidate"] == []
        assert payload["added_in_candidate"] == []
        assert payload["digest_mismatch"] == []
        assert payload["etags"]["current"] == payload["etags"]["candidate"]


async def test_release_diff_reflects_manifest_edits():
    async with stack() as s:
        await manifest_control(
            s["green"],
            set={"artifacts/new-tool.tar": "c" * 64},
            remove=["artifacts/runtime.tar"],
        )
        code, payload = await release_diff_request(s["base"], "green")
        assert code == 200
        assert "artifacts/runtime.tar" in payload["missing_in_candidate"]
        assert "artifacts/new-tool.tar" in payload["added_in_candidate"]


# -- integration: validation / request errors ---------------------------------


async def test_release_diff_unknown_target_and_bad_body():
    async with stack() as s:
        async with aiohttp.ClientSession() as session:
            async with session.post(s["base"] + "/api/release-diff", json={"target": "nope"}) as r:
                assert r.status == 404
            async with session.post(s["base"] + "/api/release-diff", json={}) as r:
                assert r.status == 422


async def test_release_diff_is_in_openapi():
    async with stack() as s:
        async with aiohttp.ClientSession() as session:
            async with session.get(s["base"] + "/openapi.json") as r:
                spec = await r.json()
        assert "post" in spec["paths"]["/api/release-diff"]


# -- integration: 409 conflicts (no partial diff allowed) ---------------------


async def test_candidate_manifest_change_before_revalidation_returns_409():
    async with stack() as s:
        await manifest_control(s["green"], mutate_on_conditional=True)
        code, payload = await release_diff_request(s["base"], "green")
        assert code == 409
        assert payload["error"] == "manifest_changed"
        assert payload["side"] == "candidate"
        assert payload["generation_at_accept"] == 1
        assert_no_diff_fields(payload)
        # The relay really sent the conditional request that detected it.
        green_requests = await manifest_requests(s["green"])
        assert any(r["conditional"] for r in green_requests)


async def test_current_manifest_change_before_revalidation_returns_409():
    async with stack() as s:
        await manifest_control(s["blue"], mutate_on_conditional=True)
        code, payload = await release_diff_request(s["base"], "green")
        assert code == 409
        assert payload["error"] == "manifest_changed"
        assert payload["side"] == "current"
        assert_no_diff_fields(payload)


async def test_conditional_request_failure_returns_409():
    async with stack() as s:
        await manifest_control(s["green"], conditional_status=500)
        code, payload = await release_diff_request(s["base"], "green")
        assert code == 409
        assert payload["error"] == "revalidation_failed"
        assert payload["side"] == "candidate"
        assert_no_diff_fields(payload)


async def test_generation_switch_during_compare_returns_409():
    async with stack() as s:
        # Slow down blue's manifest endpoint so the switch can land mid-compare.
        await manifest_control(s["blue"], delay=0.5)
        diff_task = asyncio.create_task(release_diff_request(s["base"], "green"))
        await asyncio.sleep(0.15)
        code, switch_payload = await switch(s["base"], "green")
        assert code == 200 and switch_payload["operation"]["status"] == "committed"
        code, payload = await asyncio.wait_for(diff_task, timeout=10)
        assert code == 409
        assert payload["error"] == "generation_switched"
        assert payload["generation_at_accept"] == 1
        assert payload["current_generation"] == 2
        assert_no_diff_fields(payload)


# -- integration: 502 read/format errors (no diff allowed) --------------------


async def test_candidate_manifest_not_found_returns_502():
    async with stack() as s:
        await manifest_control(s["green"], raw={"status": 404, "body": "nope", "etag": None})
        code, payload = await release_diff_request(s["base"], "green")
        assert code == 502
        assert payload["error"] == "manifest_unavailable"
        assert payload["side"] == "candidate"
        assert_no_diff_fields(payload)


async def test_candidate_manifest_weak_etag_returns_502():
    async with stack() as s:
        await manifest_control(s["green"], raw={"status": 200, "body": "{}", "etag": 'W/"abc"'})
        code, payload = await release_diff_request(s["base"], "green")
        assert code == 502
        assert payload["error"] == "manifest_etag_weak"
        assert_no_diff_fields(payload)


async def test_candidate_manifest_missing_etag_returns_502():
    async with stack() as s:
        await manifest_control(s["green"], raw={"status": 200, "body": "{}", "etag": None})
        code, payload = await release_diff_request(s["base"], "green")
        assert code == 502
        assert payload["error"] == "manifest_etag_missing"
        assert_no_diff_fields(payload)


async def test_candidate_manifest_invalid_json_returns_502():
    async with stack() as s:
        await manifest_control(s["green"], raw={"status": 200, "body": "not json{", "etag": '"x"'})
        code, payload = await release_diff_request(s["base"], "green")
        assert code == 502
        assert payload["error"] == "manifest_invalid_json"
        assert_no_diff_fields(payload)


async def test_candidate_manifest_too_many_items_returns_502():
    async with stack() as s:
        big = json.dumps({f"f{i}.bin": "a" * 64 for i in range(201)})
        await manifest_control(s["green"], raw={"status": 200, "body": big, "etag": '"x"'})
        code, payload = await release_diff_request(s["base"], "green")
        assert code == 502
        assert payload["error"] == "manifest_too_many_items"
        assert_no_diff_fields(payload)


async def test_candidate_manifest_oversize_returns_502():
    async with stack() as s:
        huge = "{}" + " " * (64 * 1024 + 1)
        await manifest_control(s["green"], raw={"status": 200, "body": huge, "etag": '"x"'})
        code, payload = await release_diff_request(s["base"], "green")
        assert code == 502
        assert payload["error"] == "manifest_too_large"
        assert_no_diff_fields(payload)


async def test_current_side_manifest_error_returns_502_with_current_side():
    async with stack() as s:
        await manifest_control(s["blue"], raw={"status": 200, "body": "nope{", "etag": '"x"'})
        code, payload = await release_diff_request(s["base"], "green")
        assert code == 502
        assert payload["error"] == "manifest_invalid_json"
        assert payload["side"] == "current"
        assert_no_diff_fields(payload)


async def test_candidate_upstream_down_returns_502():
    async with stack() as s:
        await s["upstreams"]["green"].stop()
        code, payload = await release_diff_request(s["base"], "green")
        assert code == 502
        assert payload["error"] == "manifest_unavailable"
        assert payload["side"] == "candidate"
        assert_no_diff_fields(payload)
