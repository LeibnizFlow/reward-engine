"""Explicit public-search transport for native SERP clients without paid credentials."""

import xml.etree.ElementTree as ET

import httpx


def parse_rss(payload, maximum):
    root = ET.fromstring(payload)
    if root.tag != "rss":
        raise ValueError("Search did not return an RSS result document")
    rows = []
    for item in root.findall("./channel/item")[:maximum]:
        link = item.findtext("link", "")
        if not link.startswith(("https://", "http://")):
            raise ValueError("Search result has no public URL")
        rows.append(
            {
                "title": item.findtext("title", ""),
                "url": link,
                "description": item.findtext("description", ""),
                "date": item.findtext("pubDate", ""),
            }
        )
    return rows


async def search(query, maximum=10):
    async with httpx.AsyncClient(timeout=18, follow_redirects=True) as client:
        response = await client.get(
            "https://www.bing.com/search", params={"q": query, "format": "rss"}
        )
        response.raise_for_status()
        return {"data": {"organic_results": parse_rss(response.content, maximum)}}
