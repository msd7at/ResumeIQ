"""
DuckDuckGo web search tool for ResumeIQ agents.

Used by:
  - question_generator.py — recent {target_company} interview reports
  - salary_agent.py       — live salary data + active hiring listings

Free, no API key required. Returns plain Python dicts.
On failure (rate limit, network error, parse error) returns an empty list
so the calling agent gracefully falls back to LLM training knowledge.
"""

from duckduckgo_search import DDGS


def web_search(query: str, max_results: int = 5) -> list[dict]:
    """
    Run a DuckDuckGo text search.

    Returns: list of {"title": str, "url": str, "snippet": str}
    Returns: [] if the search call fails for any reason.
    """
    try:
        results: list[dict] = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append({
                    "title":   r.get("title", ""),
                    "url":     r.get("href", ""),
                    "snippet": r.get("body", ""),
                })
        return results
    except Exception:
        return []


def format_search_results(results: list[dict], max_chars: int = 1500) -> str:
    """
    Compress search results into a compact string for LLM prompt injection.

    Truncates the joined output at max_chars to keep prompts within token budget.
    Returns "(no web search results available)" when the list is empty,
    so prompts always have something safe to interpolate.
    """
    if not results:
        return "(no web search results available)"

    parts: list[str] = []
    total = 0
    for i, r in enumerate(results, 1):
        block = (
            f"[{i}] {r['title']}\n"
            f"    {r['url']}\n"
            f"    {r['snippet']}"
        )
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)

    return "\n".join(parts)


if __name__ == "__main__":
    print("Testing DuckDuckGo web search ...")
    sample = web_search(
        "Python FastAPI backend developer salary Bangalore 2026",
        max_results=5,
    )
    print(f"Got {len(sample)} results.\n")
    print(format_search_results(sample))
