"""Unit validation with controlled HTTP transports; no live or paid API calls.

Reader regressions restored from publication_cleanup_20261006/originals/tests/unit/
test_search_io_contract.py and test_claw_outcome_search_infra.py in the old runner.
Additional cases validate actual JSON parsing, cache persistence and credentials.
"""

import asyncio
import json

import httpx
import pytest

from reward_engine.adapters.research_cache import ResearchCache
from reward_engine.adapters.support import load_named_key, read_json, write_json
from reward_engine.adapters.tavily_review import TavilyReview

CATALOG = "https://skills.sh/author/repo/audit"
REVISION = "a" * 40
SECRET = "synthetic-reader-credential"


@pytest.fixture(autouse=True)
def forbid_live_network(monkeypatch):
    """Fail closed if a test accidentally constructs an unmocked HTTP client."""

    async def forbidden(*args, **kwargs):
        pytest.fail("Unit readers must use MockTransport, never the network")

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", forbidden)


def provider(handle, *, cache=None, key="test", github_token=None):
    return TavilyReview(
        key,
        cache=cache,
        github_token=github_token,
        client_factory=lambda **kwargs: httpx.AsyncClient(
            transport=httpx.MockTransport(handle), **kwargs
        ),
    )


def missing_key():
    raise AssertionError("This public or cached operation must not load credentials")


def preview():
    return httpx.Response(
        200,
        json={
            "results": [
                {
                    "url": CATALOG,
                    "raw_content": "# Catalog preview\n[Source](https://github.com/author/repo)",
                }
            ]
        },
    )


@pytest.mark.asyncio
async def test_github_issue_uses_original_markdown_and_bounded_comment_read():
    calls = []
    code = "```c\nif (p->value != 3 && (*ptr)[2] == 1) { return 0; }\n```"

    def handle(request):
        calls.append(str(request.url))
        assert (
            request.url.host == "api.github.com"
            and "authorization" not in request.headers
        )
        if request.url.path.endswith("/comments"):
            assert request.url.params["per_page"] == "30"
            return httpx.Response(
                200,
                json=[
                    {
                        "id": 42,
                        "body": "The comment above is incorrect; inspect the expression.",
                    }
                ],
                headers={"link": '<https://api.github.com/next>; rel="next"'},
            )
        return httpx.Response(
            200,
            json={
                "title": "Parser discussion",
                "body": code,
                "comments": 40,
                "comments_url": "https://api.github.com/repos/author/repo/issues/9/comments",
            },
        )

    result = await provider(handle, key=missing_key).extract(
        "https://github.com/author/repo/issues/9"
    )
    assert result["physical_requests"] == len(calls) == 2
    assert result["authored_markdown"] and result["comments_truncated"]
    assert code in result["results"][0]["raw_content"]
    assert "incorrect; inspect" in result["results"][0]["raw_content"]


@pytest.mark.asyncio
async def test_github_issue_does_not_follow_unexpected_comment_url_or_guess_comments():
    calls = []

    def handle(request):
        calls.append(str(request.url))
        return httpx.Response(
            200,
            json={
                "title": "Method",
                "body": "Original text.",
                "comments": 1,
                "comments_url": "https://untrusted.invalid/collect",
            },
        )

    result = await provider(handle, key=missing_key).extract(
        "https://github.com/author/repo/issues/9"
    )
    assert (
        len(calls) == 1
        and result["comments_read"] == 0
        and result["comments_truncated"]
    )
    assert result["results"][0]["raw_content"].endswith("Original text.")


@pytest.mark.asyncio
async def test_public_raw_and_its_cache_hit_need_no_credentials(tmp_path):
    calls = []

    def handle(request):
        calls.append(str(request.url))
        assert "authorization" not in request.headers
        return httpx.Response(200, text="# Audit\n[Reference](references/check.md)")

    io = provider(
        handle, cache=ResearchCache(tmp_path), key=missing_key, github_token=missing_key
    )
    url = "https://github.com/author/repo/blob/main/audit/SKILL.md"
    first, second = await io.extract(url), await io.extract(url)
    assert first["read_kind"] == "body" and first["physical_requests"] == 1
    assert len(first["reader_receipts"]) == 1
    row = first["results"][0]
    assert row["url"] == url and row["fetched_url"].startswith(
        "https://raw.githubusercontent.com/"
    )
    assert row["observed_links"][0]["url"].endswith("/audit/references/check.md")
    assert second["physical_requests"] == 0 and second["reader_receipts"] == []
    assert len(second["origin_reader_receipts"]) == len(calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["search", "github_search"])
async def test_search_cache_does_not_lookup_provider_credentials(tmp_path, method):
    calls, keys = [], []

    def key():
        keys.append(1)
        return "secret"

    def handle(request):
        calls.append(str(request.url))
        assert request.headers["authorization"] == "Bearer secret"
        payload = {"results": [{"url": "https://docs.example/a"}]}
        if method == "github_search":
            payload = {
                "items": [
                    {
                        "html_url": "https://github.com/a/b/blob/main/SKILL.md",
                        "path": "SKILL.md",
                        "repository": {"private": False},
                    }
                ]
            }
        return httpx.Response(200, json=payload)

    io = provider(handle, cache=ResearchCache(tmp_path), key=key, github_token=key)
    first = await getattr(io, method)("audit")
    io.key = io.github_token = missing_key
    second = await getattr(io, method)("audit")
    assert first["physical_requests"] == 1 and second["physical_requests"] == 0
    assert len(calls) == len(keys) == 1


@pytest.mark.asyncio
async def test_transient_origin_failure_is_not_cached_as_success(tmp_path):
    tree_calls = []

    def handle(request):
        if request.url.host == "api.tavily.com":
            return preview()
        if request.url.host == "api.github.com":
            tree_calls.append(1)
            if len(tree_calls) == 1:
                raise httpx.ReadTimeout("transient", request=request)
            return httpx.Response(
                200,
                json={
                    "sha": REVISION,
                    "tree": [{"type": "blob", "path": "skills/audit/SKILL.md"}],
                },
            )
        return httpx.Response(200, text="# Audit\nActual method")

    io = provider(handle, cache=ResearchCache(tmp_path))
    first, second, third = (
        await io.extract(CATALOG),
        await io.extract(CATALOG),
        await io.extract(CATALOG),
    )
    assert (
        first["read_kind"] == "navigation"
        and first["resolution_status"] == "unavailable"
    )
    assert first["reader_receipts"][-1]["error_type"] == "ReadTimeout"
    assert first["physical_requests"] == 2
    assert second["read_kind"] == "body" and second["physical_requests"] == 3
    assert second["skill_origin"]["status"] == "read"
    assert third["physical_requests"] == 0 and len(tree_calls) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case", ["different_directory", "missing", "ambiguous", "truncated"]
)
async def test_registry_unresolved_means_navigation_to_observed_files(case):
    calls = []
    paths = ["capabilities/checks/SKILL.md"] if case == "different_directory" else []
    if case == "ambiguous":
        paths = ["one/audit/SKILL.md", "two/audit/SKILL.md"]
    if case == "truncated":
        paths = ["skills/audit/SKILL.md"]

    def handle(request):
        calls.append(str(request.url))
        if request.url.host == "api.tavily.com":
            return preview()
        assert request.url.host == "api.github.com"  # No guessed raw file request.
        return httpx.Response(
            200,
            json={
                "sha": REVISION,
                "truncated": case == "truncated",
                "tree": [{"type": "blob", "path": path} for path in paths],
            },
        )

    result = await provider(handle).extract(CATALOG)
    assert result["read_kind"] == "navigation" and not result["cacheable"]
    assert result["skill_origin"]["observed_skill_files"] == len(paths)
    links = {row["url"] for row in result["observed_links"]}
    assert {
        f"https://raw.githubusercontent.com/author/repo/HEAD/{path}" for path in paths
    } <= links
    assert (
        result["physical_requests"] == len(result["reader_receipts"]) == len(calls) == 2
    )


@pytest.mark.asyncio
async def test_registry_candidate_can_be_read_when_directory_differs_from_skill_name():
    from reward_engine.core.outcome.skill_sources import skill_identity

    def handle(request):
        if request.url.host == "api.tavily.com":
            return preview()
        if request.url.host == "api.github.com":
            return httpx.Response(
                200,
                json={
                    "sha": REVISION,
                    "tree": [{"type": "blob", "path": "checks/SKILL.md"}],
                },
            )
        assert request.url.path == "/author/repo/HEAD/checks/SKILL.md"
        return httpx.Response(
            200,
            text="---\nname: audit\ndescription: Check claims\n---\nRead actual evidence",
        )

    io = provider(handle)
    index = await io.extract(CATALOG)
    file = next(
        row["url"] for row in index["observed_links"] if row["url"].endswith("SKILL.md")
    )
    body = await io.extract(file)
    assert index["read_kind"] == "navigation" and body["read_kind"] == "body"
    assert skill_identity(file, body["results"][0]["raw_content"])["name"] == "audit"


@pytest.mark.asyncio
async def test_directory_timeout_preserves_every_attempt_and_navigation():
    url = "https://github.com/author/repo/tree/main/audit"

    def handle(request):
        if request.url.host == "api.github.com":
            return httpx.Response(
                200,
                json=[
                    {
                        "type": "file",
                        "name": "SKILL.md",
                        "download_url": "https://raw.githubusercontent.com/author/repo/main/audit/SKILL.md",
                    }
                ],
            )
        if request.url.host == "raw.githubusercontent.com":
            raise httpx.ReadTimeout("temporary", request=request)
        return httpx.Response(
            200, json={"results": [{"url": url, "raw_content": "# Directory preview"}]}
        )

    result = await provider(handle).extract(url)
    assert (
        result["read_kind"] == "navigation"
        and result["resolution_status"] == "unavailable"
    )
    assert not result["cacheable"]
    assert [r["http_status"] for r in result["reader_receipts"]] == [200, None, 200]
    assert result["physical_requests"] == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("ambiguous", [False, True])
async def test_raw_404_resolution_counts_initial_tree_and_retry(ambiguous):
    def handle(request):
        if request.url.host == "api.github.com":
            paths = ["skills/audit/SKILL.md"] + (
                ["other/audit/SKILL.md"] if ambiguous else []
            )
            return httpx.Response(
                200, json={"tree": [{"path": p, "type": "blob"} for p in paths]}
            )
        if "/skills/audit/SKILL.md" in request.url.path:
            return httpx.Response(200, text="# Audit\nActual method")
        return httpx.Response(404)

    result = await provider(handle).extract(
        "https://github.com/author/repo/blob/main/audit/SKILL.md"
    )
    assert result["read_kind"] == ("unavailable" if ambiguous else "body")
    assert (
        result["physical_requests"]
        == len(result["reader_receipts"])
        == (2 if ambiguous else 3)
    )
    assert result["reader_receipts"][0]["http_status"] == 404


@pytest.mark.asyncio
async def test_resolver_never_exceeds_its_request_limit():
    def handle(request):
        if request.url.host == "api.tavily.com":
            return preview()
        return httpx.Response(
            200,
            json={
                "sha": REVISION,
                "tree": [{"path": "audit/SKILL.md", "type": "blob"}],
            },
        )

    io = provider(handle)
    io.MAX_READER_REQUESTS = 2
    result = await io.extract(CATALOG)
    assert result["physical_requests"] == 2 and result["read_kind"] == "navigation"
    assert result["resolution_status"] == "unavailable"


@pytest.mark.asyncio
async def test_registry_tree_object_sha_is_not_used_as_a_commit_ref():
    requests = []

    def handle(request):
        requests.append(str(request.url))
        if request.url.host == "api.tavily.com":
            return preview()
        if request.url.host == "api.github.com":
            assert request.url.path.endswith("/git/trees/HEAD")
            return httpx.Response(
                200,
                json={
                    "sha": REVISION,
                    "tree": [{"type": "blob", "path": "skills/audit/SKILL.md"}],
                },
            )
        assert request.url.path == "/author/repo/HEAD/skills/audit/SKILL.md"
        return httpx.Response(200, text="# Audit\nRead actual source evidence")

    result = await provider(handle).extract(CATALOG)
    assert result["read_kind"] == "body"
    assert result["skill_origin"] == {
        "status": "read",
        "path": "skills/audit/SKILL.md",
        "ref": "HEAD",
        "tree_sha": REVISION,
    }
    assert not any(REVISION in url for url in requests)


@pytest.mark.asyncio
async def test_same_batch_aliases_keep_each_actual_io_receipt(tmp_path):
    raw = "https://raw.githubusercontent.com/author/repo/main/audit/SKILL.md"
    alias = "https://github.com/author/repo/blob/main/audit/SKILL.md"
    requests = []

    def handle(request):
        requests.append(str(request.url))
        assert str(request.url) == raw
        return httpx.Response(200, text="# Audit\nSame observed source body")

    io = provider(
        handle, cache=ResearchCache(tmp_path), key=missing_key, github_token=missing_key
    )
    first = await asyncio.gather(io.extract(raw), io.extract(alias))
    assert [r["results"][0]["fetched_url"] for r in first] == [raw, raw]
    assert sum(r["physical_requests"] for r in first) == len(requests) == 2
    assert all(len(r["reader_receipts"]) == 1 for r in first)
    second = await asyncio.gather(io.extract(raw), io.extract(alias))
    assert sum(r["physical_requests"] for r in second) == 0
    assert all(r["reader_receipts"] == [] for r in second)


@pytest.mark.asyncio
async def test_cache_ttl_params_disabled_and_failed_results(tmp_path):
    now, calls = [0], []

    async def run():
        calls.append(1)
        return {"results": [{"url": "https://a.example", "raw_content": "original"}]}

    cache = ResearchCache(tmp_path, clock=lambda: now[0])

    async def get(p=1):
        return await cache.call(
            "extract", {"url": "https://a.example", "p": p}, run, ttl_s=10
        )

    assert not (await get())["io_receipt"]["cache_hit"]
    assert (await get())["io_receipt"]["cache_hit"] and len(calls) == 1
    await get(2)
    now[0] = 11
    await get()
    assert len(calls) == 3
    cache.VERSION = "new-reader"
    await get()
    assert len(calls) == 4
    disabled = ResearchCache()
    for _ in range(2):
        assert not (await disabled.call("extract", {}, run, ttl_s=10))["io_receipt"][
            "cache_hit"
        ]

    async def failure():
        return {"results": [], "failed_results": [{"url": "https://a.example"}]}

    for _ in range(2):
        assert not (await cache.call("bad", {}, failure, ttl_s=10))["io_receipt"][
            "cache_hit"
        ]


@pytest.mark.asyncio
async def test_cache_write_failure_does_not_erase_source(tmp_path):
    path = tmp_path / "not_a_directory"
    path.write_text("existing user file")

    async def good():
        return {"results": [{"raw_content": "actual source"}]}

    out = await ResearchCache(path).call("extract", {}, good, ttl_s=10)
    assert out["results"] and out["io_receipt"]["cache_write_error"]
    assert path.read_text() == "existing user file"


@pytest.mark.asyncio
async def test_github_transport_auth_public_filter_and_cache(tmp_path):
    import httpx

    from reward_engine.adapters.tavily_review import TavilyReview

    requests = []

    def handle(req):
        requests.append(req)
        assert req.headers["authorization"] == "Bearer secret"
        assert (
            req.url.path == "/search/code"
            and "filename:SKILL.md" in req.url.params["q"]
        )
        return httpx.Response(
            200,
            json={
                "incomplete_results": True,
                "items": [
                    {
                        "html_url": "https://github.com/public/repo/blob/main/SKILL.md",
                        "path": "SKILL.md",
                        "repository": {"private": False},
                    },
                    {
                        "html_url": "https://github.com/private/repo/blob/main/SKILL.md",
                        "path": "SKILL.md",
                        "repository": {"private": True},
                    },
                ],
            },
        )

    provider = TavilyReview(
        "",
        github_token="secret",
        cache=ResearchCache(tmp_path),
        client_factory=lambda **kw: httpx.AsyncClient(
            transport=httpx.MockTransport(handle), **kw
        ),
    )
    result = await provider.github_search("audit filename:SKILL.md")
    assert len(result["results"]) == 1 and result["incomplete_results"]
    assert "secret" not in str(result)
    cached = await provider.github_search("audit filename:SKILL.md")
    assert cached["io_receipt"]["cache_hit"] and len(requests) == 1


@pytest.mark.asyncio
async def test_tavily_search_sends_contract_and_preserves_parsed_evidence(tmp_path):
    requests = []
    payload = {
        "results": [
            {
                "url": "https://docs.example/a",
                "title": "Audit",
                "content": "Evidence α",
                "score": 0.8,
            }
        ],
        "usage": {"credits": 1},
        "request_id": "controlled-search",
    }

    def handle(request):
        requests.append(request)
        assert request.method == "POST"
        assert str(request.url) == "https://api.tavily.com/search"
        assert request.headers["authorization"] == "Bearer " + SECRET
        assert json.loads(request.content) == {
            "query": "audit α",
            "search_depth": "basic",
            "max_results": 4,
            "include_answer": False,
            "include_raw_content": False,
            "include_usage": True,
        }
        return httpx.Response(200, json=payload)

    io = provider(handle, key=SECRET, cache=ResearchCache(tmp_path))
    first = await io.search("audit α")
    assert first["results"] == payload["results"]
    assert first["usage"] == {"credits": 1}
    assert first["request_id"] == "controlled-search"
    assert first["reader_receipts"] == [
        {
            "url": "https://api.tavily.com/search",
            "method": "POST",
            "http_status": 200,
            "cache_hit": False,
        }
    ]
    # Mutating the returned body must not mutate the persisted evidence.
    first["results"][0]["content"] = "caller mutation"
    io.key = missing_key
    cached = await io.search("audit α")
    assert cached["results"] == payload["results"]
    assert cached["physical_requests"] == 0 and len(requests) == 1
    assert cached["reader_receipts"] == []
    assert cached["origin_reader_receipts"] == first["reader_receipts"]
    assert SECRET not in json.dumps(cached)
    assert all(SECRET not in path.read_text() for path in tmp_path.iterdir())


@pytest.mark.asyncio
async def test_registry_search_filters_invalid_ids_and_requires_no_keys(tmp_path):
    requests = []
    payload = {
        "skills": [
            {"id": "author/repo/audit", "name": "Audit", "source": "author/repo"},
            {"id": "author/other"},
            {"id": "author/../audit"},
            {"id": "author/./audit"},
            {"id": "/author/repo"},
            {"id": "https://bad.example/a"},
            {"id": "author/repo?q=1"},
            {"id": "lonely"},
            {"id": 9},
            None,
        ]
    }

    def handle(request):
        requests.append(request)
        assert request.method == "GET" and request.url.host == "skills.sh"
        assert request.url.path == "/api/search"
        assert dict(request.url.params) == {"q": "audit reports", "limit": "20"}
        assert "authorization" not in request.headers
        return httpx.Response(200, json=payload)

    io = provider(
        handle, key=missing_key, github_token=missing_key, cache=ResearchCache(tmp_path)
    )
    result = await io.registry_search("audit reports")
    assert [r["url"] for r in result["results"]] == [
        "https://skills.sh/author/repo/audit",
        "https://skills.sh/author/other",
    ]
    assert result["results"][1]["title"] == "author/other"
    assert "Read original Skill before use" in result["results"][0]["content"]
    assert result["registry_response"] == payload
    assert result["physical_requests"] == len(result["reader_receipts"]) == 1
    assert (await io.registry_search("audit reports"))["physical_requests"] == 0
    assert len(requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["search", "github_search", "registry_search"])
@pytest.mark.parametrize("failure", ["status", "timeout", "json"])
async def test_search_failures_keep_sanitized_attempts_and_are_retried(
    tmp_path, method, failure
):
    requests = []

    def handle(request):
        requests.append(request)
        if len(requests) == 1:
            if failure == "timeout":
                raise httpx.ReadTimeout(SECRET, request=request)
            if failure == "status":
                return httpx.Response(503, text=SECRET)
            return httpx.Response(200, text="invalid JSON " + SECRET)
        return httpx.Response(
            200,
            json={
                "results": [{"url": "https://docs.example/a"}],
                "items": [
                    {
                        "html_url": "https://github.com/author/repo/blob/main/SKILL.md",
                        "path": "SKILL.md",
                        "repository": {"private": False},
                    }
                ],
                "skills": [{"id": "author/repo/audit"}],
            },
        )

    io = provider(
        handle, key=SECRET, github_token=SECRET, cache=ResearchCache(tmp_path)
    )
    expected = {
        "status": httpx.HTTPStatusError,
        "timeout": httpx.ReadTimeout,
        "json": ValueError,
    }[failure]
    with pytest.raises(expected) as caught:
        await getattr(io, method)("audit")
    receipts = caught.value.reader_receipts
    assert len(receipts) == 1
    assert (
        receipts[0]["http_status"]
        == {"status": 503, "timeout": None, "json": 200}[failure]
    )
    if failure == "timeout":
        assert receipts[0]["error_type"] == "ReadTimeout"
    assert SECRET not in json.dumps(receipts)
    assert not list(tmp_path.iterdir())
    assert (await getattr(io, method)("audit"))["physical_requests"] == 1
    assert (await getattr(io, method)("audit"))["physical_requests"] == 0
    assert len(requests) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["search", "github_search"])
async def test_missing_search_credentials_have_no_physical_receipt(method):
    def absent():
        raise ValueError("Required credential was not found")

    io = provider(lambda request: pytest.fail("No request may be sent"), key=absent)
    with pytest.raises(ValueError) as caught:
        await getattr(io, method)("audit")
    assert caught.value.reader_receipts == []


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", [[], {}, {"skills": None}, {"skills": {}}])
async def test_registry_invalid_shape_is_rejected_not_cached(tmp_path, payload):
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(200, json=payload)

    io = provider(handle, cache=ResearchCache(tmp_path), key=missing_key)
    for _ in range(2):
        with pytest.raises(ValueError, match="skills list") as caught:
            await io.registry_search("audit")
        assert caught.value.reader_receipts[0]["http_status"] == 200
    assert len(requests) == 2 and not list(tmp_path.iterdir())


@pytest.mark.asyncio
async def test_web_extract_selects_requested_row_and_resolves_explicit_links(tmp_path):
    url = "https://docs.example/guide/intro"
    body = "# Original α\n[Checks](checks.md)\n[Checks again](checks.md)\n[Local](#heading)"
    requests = []

    def handle(request):
        requests.append(request)
        assert str(request.url) == "https://api.tavily.com/extract"
        assert request.headers["authorization"] == "Bearer " + SECRET
        assert json.loads(request.content) == {
            "urls": [url],
            "extract_depth": "basic",
            "format": "markdown",
        }
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "url": "https://docs.example/unrequested",
                        "raw_content": "Wrong body",
                    },
                    {"url": url, "raw_content": body},
                ]
            },
        )

    io = provider(handle, key=SECRET, cache=ResearchCache(tmp_path))
    result = await io.extract(url)
    assert result["results"] == [
        {
            "url": url,
            "fetched_url": url,
            "raw_content": body,
            "read_kind": "body",
            "observed_links": [
                {"title": "Checks", "url": "https://docs.example/guide/checks.md"}
            ],
        }
    ]
    assert result["reader_limits"] == {"max_requests": 3, "max_links": 40}
    assert result["physical_requests"] == 1 and not result["links_truncated"]
    assert SECRET not in json.dumps(result)
    io.key = missing_key
    assert (await io.extract(url))["physical_requests"] == 0
    assert len(requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case", ["unmatched", "empty", "non_string", "invalid_rows", "partial_failure"]
)
async def test_extract_unusable_or_partial_evidence_is_never_cached(tmp_path, case):
    url = "https://docs.example/a"
    requests = []
    payload = {"results": [{"url": url, "raw_content": "# Evidence"}]}
    if case == "unmatched":
        payload["results"][0]["url"] = "https://docs.example/b"
    elif case in {"empty", "non_string"}:
        payload["results"][0]["raw_content"] = (
            " \n" if case == "empty" else {"text": "wrong type"}
        )
    elif case == "invalid_rows":
        payload["results"] = {}
    else:
        payload["failed_results"] = [{"url": url, "error": "partial"}]

    def handle(request):
        requests.append(request)
        return httpx.Response(200, json=payload)

    io = provider(handle, cache=ResearchCache(tmp_path))
    for _ in range(2):
        result = await io.extract(url)
        assert result["read_kind"] == (
            "body" if case == "partial_failure" else "unavailable"
        )
        assert result["cacheable"] is False
        assert result["physical_requests"] == 1
        if case == "partial_failure":
            assert result["failed_results"] == payload["failed_results"]
        else:
            assert result["results"][0]["raw_content"] == ""
    assert len(requests) == 2 and not list(tmp_path.iterdir())


@pytest.mark.asyncio
async def test_directory_reads_only_observed_skill_file_with_requested_ref():
    url = "https://github.com/author/repo/tree/release/audit"
    raw = "https://raw.githubusercontent.com/author/repo/release/audit/SKILL.md"
    requests = []

    def handle(request):
        requests.append(request)
        if request.url.host == "api.github.com":
            assert request.url.path == "/repos/author/repo/contents/audit"
            assert dict(request.url.params) == {"ref": "release"}
            assert request.headers["authorization"] == "Bearer " + SECRET
            return httpx.Response(
                200,
                json=[
                    {
                        "type": "file",
                        "name": "SKILL.md",
                        "download_url": raw,
                        "html_url": "https://github.com/author/repo/blob/release/audit/SKILL.md",
                    }
                ],
            )
        assert str(request.url) == raw
        assert "authorization" not in request.headers
        return httpx.Response(200, text="# Audit\n[Checks](references/check.md)")

    result = await provider(handle, key=missing_key, github_token=SECRET).extract(url)
    assert result["read_kind"] == "body"
    assert result["results"][0]["fetched_url"] == raw
    assert result["observed_links"][0]["url"] == raw.replace(
        "SKILL.md", "references/check.md"
    )
    assert result["physical_requests"] == len(requests) == 2
    assert SECRET not in json.dumps(result)


@pytest.mark.asyncio
async def test_directory_navigation_rejects_foreign_links_and_caps_observed_links():
    url = "https://github.com/author/repo/tree/main/skills"
    valid = [
        {"type": "dir", "name": f"skill-{i}", "html_url": url + f"/skill-{i}"}
        for i in range(45)
    ]

    def handle(request):
        assert request.url.host == "api.github.com"
        return httpx.Response(
            200,
            json=valid
            + [
                {"html_url": "https://github.com/author/other/tree/main"},
                {"html_url": "https://github.com.evil.example/author/repo/"},
                {"html_url": None},
                None,
            ],
        )

    result = await provider(handle, key=missing_key).extract(url)
    assert result["read_kind"] == "navigation" and result["links_truncated"]
    assert result["observed_links"] == [
        {"title": r["name"], "url": r["html_url"]} for r in valid[:40]
    ]
    assert result["physical_requests"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("max_requests", [0, 1, 2, 3])
async def test_registry_origin_budget_counts_every_transport_attempt(max_requests):
    requests = []

    def handle(request):
        requests.append(request)
        if request.url.host == "api.tavily.com":
            return preview()
        if request.url.host == "api.github.com":
            return httpx.Response(
                200,
                json={
                    "sha": REVISION,
                    "tree": [
                        {"type": "blob", "path": "skills/audit/SKILL.md"},
                    ],
                },
            )
        assert (
            str(request.url)
            == "https://raw.githubusercontent.com/author/repo/HEAD/skills/audit/SKILL.md"
        )
        return httpx.Response(200, text="# Audit\nOriginal method")

    io = provider(handle)
    io.MAX_READER_REQUESTS = max_requests
    result = await io.extract(CATALOG)
    assert (
        result["physical_requests"]
        == len(result["reader_receipts"])
        == len(requests)
        == max_requests
    )
    assert [r["url"] for r in result["reader_receipts"]] == [
        str(r.url) for r in requests
    ]
    if max_requests == 3:
        assert (
            result["read_kind"] == "body" and result["skill_origin"]["status"] == "read"
        )
    else:
        assert result["resolution_status"] == "unavailable" and not result["cacheable"]
        assert result["error_type"] == "ValueError"


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["unobserved", "invalid_sha", "invalid_tree"])
async def test_registry_origin_requires_observed_repo_and_valid_tree(tmp_path, case):
    requests = []

    def handle(request):
        requests.append(request)
        if request.url.host == "api.tavily.com":
            if case == "unobserved":
                return httpx.Response(
                    200,
                    json={
                        "results": [{"url": CATALOG, "raw_content": "# Preview only"}]
                    },
                )
            return preview()
        assert request.url.host == "api.github.com"
        return httpx.Response(
            200,
            json={
                "sha": "not-a-sha" if case == "invalid_sha" else REVISION,
                "tree": None
                if case == "invalid_tree"
                else [{"type": "blob", "path": "audit/SKILL.md"}],
            },
        )

    io = provider(handle, cache=ResearchCache(tmp_path))
    for _ in range(2):
        result = await io.extract(CATALOG)
        assert result["read_kind"] == "navigation" and not result["cacheable"]
        assert result["resolution_status"] == (
            "unavailable" if case == "invalid_tree" else "unresolved"
        )
        assert result["physical_requests"] == (1 if case == "unobserved" else 2)
    assert len(requests) == (2 if case == "unobserved" else 4)
    assert not list(tmp_path.iterdir())


@pytest.mark.asyncio
async def test_tree_resolution_ignores_unsafe_and_non_file_paths():
    def handle(request):
        if request.url.host == "api.tavily.com":
            return preview()
        assert request.url.host == "api.github.com"
        return httpx.Response(
            200,
            json={
                "sha": REVISION,
                "tree": [
                    {"type": "blob", "path": "checks/SKILL.md"},
                    {"type": "tree", "path": "audit/SKILL.md"},
                    {"type": "blob", "path": "../audit/SKILL.md"},
                    {"type": "blob", "path": "/audit/SKILL.md"},
                    {"type": "blob", "path": "./audit/SKILL.md"},
                    {"type": "blob", "path": 1},
                    None,
                ],
            },
        )

    result = await provider(handle).extract(CATALOG)
    assert result["skill_origin"]["observed_skill_files"] == 1
    assert result["skill_origin"]["matches"] == 0
    assert result["observed_links"] == [
        {
            "title": "checks/SKILL.md",
            "url": "https://raw.githubusercontent.com/author/repo/HEAD/checks/SKILL.md",
        },
        {"title": "Source", "url": "https://github.com/author/repo"},
    ]
    assert result["physical_requests"] == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "age,hit", [(0, True), (9.999, True), (10, False), (11, False), (-1, False)]
)
async def test_cache_ttl_boundary_and_clock_rollback(tmp_path, age, hit):
    now = [100.0]
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(200, text=f"# Body {len(requests)}")

    cache = ResearchCache(tmp_path, clock=lambda: now[0])
    io = provider(handle, key=missing_key)
    url = "https://raw.githubusercontent.com/author/repo/main/SKILL.md"
    first = await cache.call(
        "extract", {"url": url}, lambda: io._extract(url), ttl_s=10
    )
    now[0] += age
    second = await cache.call(
        "extract", {"url": url}, lambda: io._extract(url), ttl_s=10
    )
    assert second["io_receipt"]["cache_hit"] is hit
    assert len(requests) == (1 if hit else 2)
    assert second["physical_requests"] == (0 if hit else 1)
    assert second["io_receipt"]["timestamp"] == (100 if hit else now[0])
    if hit:
        assert second["reader_receipts"] == []
        assert second["origin_reader_receipts"] == first["reader_receipts"]
    else:
        assert second["results"][0]["raw_content"] == "# Body 2"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "corrupt",
    [
        "{not json",
        "null",
        "[]",
        "{}",
        '{"timestamp":"wrong","value":{}}',
        '{"timestamp":100,"value":{"results":[]}}',
        '{"timestamp":100,"value":{"results":[{}],"resolution_status":"unavailable"}}',
    ],
)
async def test_corrupt_or_unusable_cache_entry_is_replaced_by_real_read(
    tmp_path, corrupt
):
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(200, text=f"# Read {len(requests)}")

    io = provider(
        handle, key=missing_key, cache=ResearchCache(tmp_path, clock=lambda: 100)
    )
    url = "https://raw.githubusercontent.com/author/repo/main/SKILL.md"
    await io.extract(url)
    (path,) = tmp_path.iterdir()
    path.write_text(corrupt)
    result = await io.extract(url)
    assert not result["io_receipt"]["cache_hit"] and len(requests) == 2
    assert result["results"][0]["raw_content"] == "# Read 2"
    assert json.loads(path.read_text())["value"]["results"] == result["results"]
    assert (await io.extract(url))["physical_requests"] == 0
    assert len(list(tmp_path.iterdir())) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "extra",
    [
        {"failed_results": [{"error": "partial failure"}]},
        {"cacheable": False},
        {"read_kind": "unavailable"},
        {"resolution_status": "unresolved"},
        {"resolution_status": "unavailable"},
        {"skill_origin": {"status": "unresolved"}},
        {"skill_origin": {"status": "unavailable"}},
    ],
)
async def test_success_shaped_failure_receipts_are_not_cached(tmp_path, extra):
    calls = []

    async def execute():
        calls.append(1)
        return {"results": [{"raw_content": "Catalog preview"}], **extra}

    cache = ResearchCache(tmp_path)
    for _ in range(2):
        result = await cache.call("extract", {"url": CATALOG}, execute, ttl_s=10)
        assert result["io_receipt"]["cache_hit"] is False
    assert len(calls) == 2 and not list(tmp_path.iterdir())


@pytest.mark.asyncio
async def test_cache_key_uses_parameter_values_kind_and_version_not_dict_order(
    tmp_path,
):
    calls = []

    async def execute():
        calls.append(1)
        return {"results": [{"url": "https://docs.example/a"}]}

    cache = ResearchCache(tmp_path)
    first = await cache.call(
        "search", {"query": "audit", "limit": 4}, execute, ttl_s=10
    )
    reordered = await cache.call(
        "search", {"limit": 4, "query": "audit"}, execute, ttl_s=10
    )
    assert reordered["io_receipt"]["key"] == first["io_receipt"]["key"]
    assert reordered["io_receipt"]["cache_hit"] and len(calls) == 1
    other_query = await cache.call(
        "search", {"query": "check", "limit": 4}, execute, ttl_s=10
    )
    other_kind = await cache.call(
        "extract", {"query": "audit", "limit": 4}, execute, ttl_s=10
    )
    cache.VERSION = "future-format"
    other_version = await cache.call(
        "search", {"query": "audit", "limit": 4}, execute, ttl_s=10
    )
    assert (
        len(
            {
                r["io_receipt"]["key"]
                for r in [first, other_query, other_kind, other_version]
            }
        )
        == 4
    )
    assert len(calls) == 4


@pytest.mark.asyncio
async def test_cache_atomic_replace_failure_keeps_evidence_and_cleans_temp(
    tmp_path, monkeypatch
):
    def denied(source, destination):
        raise PermissionError("synthetic filesystem error")

    monkeypatch.setattr("reward_engine.adapters.research_cache.os.replace", denied)

    async def execute():
        return {
            "results": [{"raw_content": "Original evidence"}],
            "physical_requests": 1,
        }

    result = await ResearchCache(tmp_path).call("extract", {}, execute, ttl_s=10)
    assert result["results"] == [{"raw_content": "Original evidence"}]
    assert result["physical_requests"] == 1
    assert result["io_receipt"]["cache_write_error"] == "PermissionError"
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "assignment,expected",
    [
        ("READER_VALIDATION_KEY=plain-token", "plain-token"),
        ('export READER_VALIDATION_KEY="quoted token" # comment', "quoted token"),
        (
            "  READER_VALIDATION_KEY = 'token#literal' # trailing comment",
            "token#literal",
        ),
        ("READER_VALIDATION_KEY=token=with=equals", "token=with=equals"),
        (
            'READER_VALIDATION_KEY=""\nREADER_VALIDATION_KEY=second-token',
            "second-token",
        ),
    ],
)
def test_named_credential_parses_only_requested_assignment(
    tmp_path, monkeypatch, capsys, assignment, expected
):
    name = "READER_VALIDATION_KEY"
    monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("UNRELATED_READER_KEY", raising=False)
    env = tmp_path / "credentials.env"
    env.write_text(
        "UNRELATED_READER_KEY='unclosed ignored value\n# comment\n" + assignment + "\n"
    )
    assert load_named_key(name, env) == expected
    # Reading one credential does not import other assignments into the environment.
    import os

    assert "UNRELATED_READER_KEY" not in os.environ and name not in os.environ
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""


def test_named_credential_environment_precedes_missing_file(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setenv("READER_VALIDATION_KEY", SECRET)
    assert (
        load_named_key("READER_VALIDATION_KEY", tmp_path / "does-not-exist") == SECRET
    )
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""


@pytest.mark.parametrize(
    "value",
    ["", "''", "one two", "'$UNSET_SECRET'", "'${UNSET_SECRET}'", "'unterminated"],
)
def test_invalid_named_credential_is_rejected_without_exposing_file(
    tmp_path, monkeypatch, capsys, value
):
    name = "READER_VALIDATION_KEY"
    monkeypatch.setenv(name, "")
    env = tmp_path / "credentials.env"
    env.write_text(f"OTHER_KEY={SECRET}\n{name}={value}\n")
    with pytest.raises(ValueError) as caught:
        load_named_key(name, env)
    captured = capsys.readouterr()
    assert SECRET not in str(caught.value)
    assert captured.out == captured.err == ""


def test_named_credential_missing_name_does_not_leak_other_credentials(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("READER_VALIDATION_KEY", raising=False)
    env = tmp_path / "credentials.env"
    env.write_text(f"OTHER_KEY={SECRET}\n# READER_VALIDATION_KEY=commented-out\n")
    with pytest.raises(
        ValueError, match="Required credential READER_VALIDATION_KEY was not found"
    ) as caught:
        load_named_key("READER_VALIDATION_KEY", env)
    assert SECRET not in str(caught.value)


def test_json_support_round_trips_unicode_and_reports_invalid_json(tmp_path):
    path = tmp_path / "receipt.json"
    payload = {
        "title": "Evidence α",
        "reader_receipts": [{"http_status": 200}],
        "cache_hit": False,
    }
    write_json(path, payload)
    assert read_json(path) == payload
    text = path.read_text(encoding="utf-8")
    assert "Evidence α" in text and text.endswith("\n")
    path.write_text("{bad-json", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        read_json(path)
