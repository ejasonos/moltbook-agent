from __future__ import annotations

from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from moltbook_tools_http import get_memory, mcp
from runtime.health import RuntimeHealth


def _health_snapshot():
    try:
        memory = get_memory()
    except Exception:
        memory = None
    return RuntimeHealth(
        memory=memory,
        application_role="mcp_http_server",
    ).check()


def health(request: Request):
    snapshot = _health_snapshot()
    return JSONResponse(
        {"status": snapshot["status"], "checked_at": snapshot["checked_at"]},
        status_code=503 if snapshot["status"] == "unhealthy" else 200,
    )


def health_details(request: Request):
    snapshot = _health_snapshot()
    return JSONResponse(
        snapshot,
        status_code=503 if snapshot["status"] == "unhealthy" else 200,
    )


app = mcp.http_app(stateless_http=True)
app.router.routes.insert(0, Route("/health/details", health_details, methods=["GET"]))
app.router.routes.insert(0, Route("/health", health, methods=["GET"]))
