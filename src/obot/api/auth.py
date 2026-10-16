"""X-API-Key authentication."""

import hmac

from fastapi import HTTPException, Request


def require_api_key(request: Request) -> None:
    """Enabled only when OBOT_API_KEY is set."""
    expected = request.app.state.settings.secrets.obot_api_key
    if expected is None:
        return
    given = request.headers.get("X-API-Key", "")
    if not hmac.compare_digest(given.encode(), expected.encode()):
        raise HTTPException(
            401,
            {"code": "UNAUTHORIZED", "message": "missing or invalid X-API-Key header"},
        )
