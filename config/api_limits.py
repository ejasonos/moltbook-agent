from __future__ import annotations

import os


def max_api_requests_per_minute() -> int:
    """Return the shared, bounded Moltbook API request budget."""
    raw = os.getenv("MOLTBOOK_MAX_REQUESTS_PER_MINUTE", "100")
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError("MOLTBOOK_MAX_REQUESTS_PER_MINUTE must be an integer.") from exc
    if value < 1:
        raise ValueError("MOLTBOOK_MAX_REQUESTS_PER_MINUTE must be positive.")
    return min(value, 100)
