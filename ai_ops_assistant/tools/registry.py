from typing import Any, Callable, Dict, List

from .github import search_repositories
from .weather import get_current_weather
from .llm_generate import generate_text

ToolFn = Callable[..., Dict[str, Any]]

TOOL_REGISTRY: Dict[str, ToolFn] = {
    "github_search": search_repositories,
    "weather_current": get_current_weather,
    "llm_generate": generate_text,
}

TOOL_SPECS: List[Dict[str, Any]] = [
    {
        "name": "github_search",
        "description": "Search GitHub repositories by keyword.",
        "args": {
            "query": "string search query",
            "top_n": "integer, number of repos to return",
        },
    },
    {
        "name": "weather_current",
        "description": "Get current weather by city name.",
        "args": {
            "city": "string city name, e.g. Mumbai",
        },
    },
    {
        "name": "llm_generate",
        "description": "Generate a text response for general writing tasks.",
        "args": {
            "instruction": "string instruction, e.g. Write a 60-second intro speech.",
        },
    },
]
