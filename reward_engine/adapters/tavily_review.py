"""Public search and bounded source reading; credentials are loaded only for IO."""

import re
from urllib.parse import quote, urlparse

import httpx

from reward_engine.adapters.research_cache import ResearchCache
from reward_engine.core.outcome.skill_sources import observed_references


class TavilyReview:
    MAX_READER_REQUESTS = 3
    MAX_LINKS = 40

    def __init__(self, key, *, client_factory=httpx.AsyncClient, cache=None, github_token=None):
        self.key, self.client_factory = key, client_factory
        self.cache, self.github_token = cache or ResearchCache(), github_token

    def github_headers(self, *, required=False):
        token = self.github_token() if callable(self.github_token) else self.github_token
        if required and not token:
            raise ValueError("GitHub code search requires configured authentication")
        return {"Authorization": "Bearer " + token} if token else {}

    async def send(self, client, method, url, receipts, **kwargs):
        """Record attempts, including failures, without headers or response bodies."""
        receipt = {"url": url, "method": method.upper(), "cache_hit": False, "http_status": None}
        receipts.append(receipt)
        try:
            response = await getattr(client, method)(url, **kwargs)
            receipt.update(url=str(getattr(response, "url", url)),
                           http_status=getattr(response, "status_code", 200))
            return response
        except BaseException as exc:
            receipt["error_type"] = type(exc).__name__
            exc.reader_receipts = receipts
            raise

    async def request(self, endpoint, body, *, receipts=None):
        receipts = [] if receipts is None else receipts
        try:
            key = self.key() if callable(self.key) else self.key
            async with self.client_factory(timeout=45, trust_env=False) as client:
                response = await self.send(client, "post", "https://api.tavily.com/" + endpoint,
                    receipts, headers={"Authorization": "Bearer " + key}, json=body)
                response.raise_for_status()
                return {**response.json(), "reader_receipts": receipts, "physical_requests": len(receipts)}
        except Exception as exc:
            exc.reader_receipts = receipts
            raise

    async def search(self, query):
        params = {"query": query, "search_depth": "basic", "max_results": 4,
                  "include_answer": False, "include_raw_content": False, "include_usage": True}
        return await self.cache.call("tavily.search", params,
            lambda: self.request("search", params), ttl_s=3600)

    async def github_search(self, query):
        params = {"q": query + " is:public", "per_page": 10}
        async def execute():
            receipts = []
            try:
                async with self.client_factory(timeout=45, trust_env=False) as client:
                    response = await self.send(client, "get", "https://api.github.com/search/code", receipts,
                        params=params, headers={**self.github_headers(required=True),
                                               "Accept": "application/vnd.github+json"})
                    response.raise_for_status()
                    payload = response.json()
                    return {"results": [{"url": r["html_url"], "title": r["path"], "content": ""}
                        for r in payload.get("items", []) if r.get("repository", {}).get("private") is False],
                        "incomplete_results": payload.get("incomplete_results", False),
                        "reader_receipts": receipts, "physical_requests": len(receipts)}
            except Exception as exc:
                exc.reader_receipts = receipts
                raise
        return await self.cache.call("github.search", params, execute, ttl_s=3600)

    async def registry_search(self, query):
        """An index entry is a lead, never proof that its Skill still exists."""
        params = {"q": query, "limit": 20}
        async def execute():
            receipts = []
            try:
                async with self.client_factory(timeout=45, trust_env=False) as client:
                    response = await self.send(client, "get", "https://skills.sh/api/search", receipts, params=params)
                    response.raise_for_status()
                    data = response.json()
                    if not isinstance(data, dict) or not isinstance(data.get("skills"), list):
                        raise ValueError("Skill registry requires a skills list")
                    rows = [r for r in data["skills"] if isinstance(r, dict) and isinstance(r.get("id"), str)
                            and re.fullmatch(r"[\w.-]+(?:/[\w.-]+)+", r["id"])
                            and not any(p in {".", ".."} for p in r["id"].split("/"))]
                    return {"results": [{"url": "https://skills.sh/" + r["id"], "title": str(r.get("name", r["id"])),
                        "content": "Skill registry entry; source: " + str(r.get("source", "")) + ". Read original Skill before use."}
                        for r in rows], "registry_response": data,
                        "reader_receipts": receipts, "physical_requests": len(receipts)}
            except Exception as exc:
                exc.reader_receipts = receipts
                raise
        return await self.cache.call("skills.registry.search", params, execute, ttl_s=3600)

    async def extract(self, url):
        return await self.cache.call("extract", {"url": url, "reader": "observed-source-v4"},
                                     lambda: self._extract(url), ttl_s=86400)

    def result(self, url, text, fetched_url, *, kind="body", links=(), **extra):
        if kind == "body" and not text.strip():
            kind = "unavailable"
            extra["cacheable"] = False
        observed = [{"title": title, "url": link} for title, link in observed_references(fetched_url, text)]
        all_links = list({row["url"]: row for row in [*links, *observed]}.values())
        row = {"url": url, "raw_content": text, "fetched_url": fetched_url, "read_kind": kind,
               "observed_links": all_links[:self.MAX_LINKS]}
        return {"results": [row], "read_kind": kind, "observed_links": row["observed_links"],
                "links_truncated": len(all_links) > self.MAX_LINKS, **extra}

    async def tree(self, get, owner, repo, revision):
        response = await get(f"https://api.github.com/repos/{owner}/{repo}/git/trees/{quote(revision, safe='')}",
                             params={"recursive": "1"}, headers=self.github_headers())
        response.raise_for_status()
        tree = response.json()
        if not isinstance(tree, dict) or not isinstance(tree.get("tree"), list):
            raise ValueError("GitHub tree requires a tree list")
        paths = [r["path"] for r in tree["tree"] if isinstance(r, dict) and r.get("type") == "blob"
                 and isinstance(r.get("path"), str) and not r["path"].startswith("/")
                 and not any(p in {".", ".."} for p in r["path"].split("/"))]
        return tree, paths

    @staticmethod
    def file_url(owner, repo, revision, path):
        return f"https://raw.githubusercontent.com/{owner}/{repo}/{quote(revision, safe='')}/{quote(path, safe='/')}"

    async def registry_origin(self, url, result, get):
        parts = urlparse(url).path.strip("/").split("/")
        if len(parts) != 3 or not all(re.fullmatch(r"[\w.-]+", p) and p not in {".", ".."} for p in parts):
            return result
        owner, repo, name = parts
        origin = f"https://github.com/{owner}/{repo}"
        if not any(row["url"].rstrip("/").removesuffix(".git") == origin for row in result["observed_links"]):
            return {**result, "resolution_status": "unresolved", "cacheable": False}
        ref = "HEAD"
        tree, paths = await self.tree(get, owner, repo, ref)
        tree_sha = tree.get("sha", "")
        if not isinstance(tree_sha, str) or not re.fullmatch(r"[0-9a-fA-F]{40,64}", tree_sha):
            return {**result, "resolution_status": "unresolved", "cacheable": False}
        skills = [path for path in paths if path.split("/")[-1].lower() == "skill.md"]
        matches = [path for path in skills if path.split("/")[-2:] == [name, "SKILL.md"]]
        if not tree.get("truncated") and len(matches) == 1:
            # The trees API returns a tree object SHA, not a commit/ref. Keep
            # the observed HEAD ref for reading and record the tree separately.
            raw = self.file_url(owner, repo, ref, matches[0])
            response = await get(raw)
            response.raise_for_status()
            return self.result(url, response.text, raw, catalog_response=result,
                skill_origin={"status": "read", "path": matches[0], "ref": ref, "tree_sha": tree_sha})
        # A directory name need not equal frontmatter.name. Return observed files
        # for a content read, not a guessed match or a claim that no Skill exists.
        links = [{"title": path, "url": self.file_url(owner, repo, ref, path)} for path in skills]
        text = result["results"][0]["raw_content"]
        return self.result(url, text, url, kind="navigation", links=links, cacheable=False,
            resolution_status="unresolved", skill_origin={"status": "unresolved", "matches": len(matches),
                "observed_skill_files": len(skills), "truncated": bool(tree.get("truncated")),
                "ref": ref, "tree_sha": tree_sha})

    async def _extract(self, url):
        receipts, result, directory_links = [], None, []
        parsed = urlparse(url)
        parts = parsed.path.strip("/").split("/")
        directory = parsed.hostname == "github.com" and (len(parts) == 2 or (len(parts) >= 4 and parts[2] == "tree"))
        registry = parsed.hostname in {"skills.sh", "www.skills.sh"}
        resolution_error = None
        try:
            async with self.client_factory(timeout=45, trust_env=False) as client:
                async def get(target, **kwargs):
                    if len(receipts) >= self.MAX_READER_REQUESTS:
                        raise ValueError("Reader request limit reached")
                    return await self.send(client, "get", target, receipts, **kwargs)

                raw = url if parsed.hostname == "raw.githubusercontent.com" else None
                if parsed.hostname == "github.com" and len(parts) >= 5 and parts[2] == "blob":
                    raw = "https://raw.githubusercontent.com/" + "/".join(parts[:2] + parts[3:])
                if raw:
                    response = await get(raw)
                    if response.status_code == 404:
                        raw_parts = urlparse(raw).path.strip("/").split("/")
                        owner, repo, revision = raw_parts[:3]
                        wanted = "/".join(raw_parts[3:])
                        tree, paths = await self.tree(get, owner, repo, revision)
                        matches = [path for path in paths if path == wanted or path.endswith("/" + wanted)]
                        if not tree.get("truncated") and len(matches) == 1:
                            raw = self.file_url(owner, repo, revision, matches[0])
                            response = await get(raw)
                    response.raise_for_status()
                    result = self.result(url, response.text, raw)
                elif (parsed.hostname == 'github.com' and len(parts) == 4
                      and parts[2] in {'issues', 'pull'} and parts[3].isdigit()):
                    # The API returns authored Markdown, including literal code
                    # operators that generic HTML extraction can destroy.
                    owner, repo, _, number = parts
                    api = f'https://api.github.com/repos/{owner}/{repo}/issues/{number}'
                    response = await get(api, headers=self.github_headers())
                    response.raise_for_status()
                    issue = response.json()
                    if not isinstance(issue, dict) or not isinstance(issue.get('body'), str):
                        raise ValueError('GitHub issue requires original Markdown body')
                    text = '# ' + str(issue.get('title', '')) + '\n\n' + issue['body']
                    comments, truncated = [], bool(issue.get('comments'))
                    comments_url = issue.get('comments_url')
                    if truncated and comments_url == api + '/comments':
                        response = await get(comments_url, params={'per_page': 30}, headers=self.github_headers())
                        response.raise_for_status()
                        comments = response.json()
                        if not isinstance(comments, list):
                            raise ValueError('GitHub comments require a list')
                        truncated = 'rel="next"' in response.headers.get('link', '')
                        for comment in comments:
                            if isinstance(comment, dict) and isinstance(comment.get('body'), str):
                                text += '\n\n## Comment ' + str(comment.get('id', '')) + '\n\n' + comment['body']
                    result = self.result(url, text, url, authored_markdown=True, body_api_url=api,
                                         comments_read=len(comments), comments_truncated=truncated)
                elif directory:
                    owner, repo = parts[:2]
                    path = "/".join(parts[4:]) if len(parts) >= 4 else ""
                    try:
                        response = await get(f"https://api.github.com/repos/{owner}/{repo}/contents/{path}",
                            params={"ref": parts[3]} if len(parts) >= 4 else {}, headers=self.github_headers())
                        response.raise_for_status()
                        rows = response.json()
                        if not isinstance(rows, list):
                            raise ValueError("GitHub directory requires a list")
                        links = [{"url": r["html_url"], "title": r.get("name", "")} for r in rows
                            if isinstance(r, dict) and isinstance(r.get("html_url"), str)
                            and r["html_url"].startswith(f"https://github.com/{owner}/{repo}/")]
                        directory_links = links
                        matches = [r for r in rows if isinstance(r, dict) and r.get("type") == "file"
                                   and str(r.get("name", "")).lower() == "skill.md"]
                        raw = matches[0].get("download_url") if len(matches) == 1 else None
                        if isinstance(raw, str) and raw.startswith(f"https://raw.githubusercontent.com/{owner}/{repo}/") and not urlparse(raw).query:
                            response = await get(raw)
                            response.raise_for_status()
                            result = self.result(url, response.text, raw)
                        elif links and len(parts) >= 4:
                            text = "# Observed repository directory\n" + "\n".join(f"[{r['title']}]({r['url']})" for r in links[:self.MAX_LINKS])
                            result = self.result(url, text, url, kind="navigation", links=links)
                        elif matches:
                            resolution_error = "UnusableDownloadURL"
                    except (httpx.HTTPError, ValueError) as exc:
                        resolution_error = type(exc).__name__
                if result is None:
                    if len(receipts) >= self.MAX_READER_REQUESTS:
                        raise ValueError("Reader request limit reached")
                    page = await self.request("extract", {"urls": [url], "extract_depth": "basic", "format": "markdown"}, receipts=receipts)
                    rows = page.get("results", [])
                    row = next((r for r in rows if isinstance(r, dict) and r.get("url") == url), {}) if isinstance(rows, list) else {}
                    text = row.get("raw_content", "")
                    if not isinstance(text, str) or not text.strip():
                        result = self.result(url, "", url, kind="unavailable", cacheable=False,
                                             failed_results=page.get("failed_results", []))
                    else:
                        result = self.result(url, text, url, kind="navigation" if directory or registry else "body", links=directory_links)
                        if page.get("failed_results"):
                            result.update(failed_results=page["failed_results"], cacheable=False)
                        if registry:
                            result = await self.registry_origin(url, result, get)
                if resolution_error:
                    result.update(resolution_status="unavailable", error_type=resolution_error, cacheable=False)
        except (httpx.HTTPError, ValueError, TypeError) as exc:
            result = result or self.result(url, "", url, kind="unavailable")
            result.update(resolution_status="unavailable", error_type=type(exc).__name__, cacheable=False)
            if registry:
                result["skill_origin"] = {"status": "unavailable", "error_type": type(exc).__name__}
        result.update(reader_receipts=receipts, physical_requests=len(receipts),
                      reader_limits={"max_requests": self.MAX_READER_REQUESTS, "max_links": self.MAX_LINKS})
        return result
