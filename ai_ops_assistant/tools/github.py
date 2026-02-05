import os
from typing import Any, Dict, List

import requests


def search_repositories(query: str, top_n: int = 5) -> Dict[str, Any]:
    try:
        top_n = int(top_n)
    except (TypeError, ValueError):
        top_n = 5
    url = "https://api.github.com/search/repositories"
    params = {"q": query, "sort": "stars", "order": "desc", "per_page": top_n}
    headers: Dict[str, str] = {"Accept": "application/vnd.github+json"}
    token = os.getenv("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    response = requests.get(url, params=params, headers=headers, timeout=20)
    response.raise_for_status()
    data = response.json()

    items: List[Dict[str, Any]] = []
    for item in data.get("items", [])[:top_n]:
        items.append(
            {
                "name": item.get("name"),
                "full_name": item.get("full_name"),
                "url": item.get("html_url"),
                "stars": item.get("stargazers_count"),
                "description": item.get("description"),
                "language": item.get("language"),
            }
        )

    return {
        "query": query,
        "total_count": data.get("total_count"),
        "items": items,
        "source_url": response.url,
    }
