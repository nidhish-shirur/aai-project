from __future__ import annotations

from collections import defaultdict, deque
from threading import Lock
from time import monotonic

from fastapi import HTTPException, Request


_WINDOW_SECONDS = 60.0
_MAX_REQUESTS = 30

_requests: dict[str, deque[float]] = defaultdict(deque)
_lock = Lock()


def enforce_rate_limit(request: Request) -> None:
    client = request.client
    key = client.host if client else "unknown"
    now = monotonic()

    with _lock:
        timestamps = _requests[key]

        while timestamps and now - timestamps[0] > _WINDOW_SECONDS:
            timestamps.popleft()

        if len(timestamps) >= _MAX_REQUESTS:
            raise HTTPException(
                status_code=429,
                detail="Too many requests. Try again later.",
            )

        timestamps.append(now)