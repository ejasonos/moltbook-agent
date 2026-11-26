from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any


logger = logging.getLogger("local_agent.moltbook.health")
EXPECTED_MOLTBOOK_TOOLS = {
    "molt_get_my_profile",
    "molt_get_feed",
    "molt_get_posts",
    "molt_create_comment",
}


class RuntimeHealth:
    """Read-only runtime checks; failures are reported, never raised to worker."""

    def __init__(self, runtime):
        self.runtime = runtime

    def check(self) -> dict[str, Any]:
        components: dict[str, dict[str, Any]] = {}

        model = self.runtime.agent_loop.model
        health_check = getattr(model, "health_check", None)
        if callable(health_check):
            try:
                available, detail = health_check()
                components["nvidia_model"] = self._component(
                    "healthy" if available else "degraded", detail
                )
            except Exception:
                components["nvidia_model"] = self._component(
                    "degraded", "nvidia_health_check_failed"
                )
        else:
            configured = bool(getattr(model, "model", None))
            components["nvidia_model"] = self._component(
                "healthy" if configured else "unhealthy",
                "configured" if configured else "model_not_configured",
            )

        discovered = []
        try:
            discovered = self.runtime.registry.discover()
            names = {tool.name for tool in discovered}
            server_ok = bool(discovered)
            tools_ok = EXPECTED_MOLTBOOK_TOOLS.issubset(names)
            components["mcp_connectivity"] = self._component(
                "healthy" if server_ok else "degraded",
                "connected" if server_ok else "no_tools_discovered",
            )
            components["tool_discovery"] = self._component(
                "healthy" if tools_ok else "degraded",
                "required_moltbook_tools_available" if tools_ok else "required_moltbook_tools_missing",
            )
        except Exception as exc:
            components["mcp_connectivity"] = self._component(
                "degraded", self._safe_detail(exc)
            )
            components["tool_discovery"] = self._component(
                "degraded", "discovery_failed"
            )

        auth_status = "unavailable"
        if discovered:
            try:
                result = self.runtime.mcp_client.call_tool("molt_get_my_profile", {})
                payload = self._decode(result)
                error = self._payload_error(payload)
                if result.success and not error:
                    auth_status = "healthy"
                else:
                    http_status = self._find_http_status(payload)
                    auth_status = "unhealthy" if http_status in {401, 403} else "degraded"
                    detail = error or result.error or "profile_check_failed"
                    components["moltbook_authentication"] = self._component(
                        auth_status, self._safe_detail(detail)
                    )
            except Exception as exc:
                auth_status = "degraded"
                components["moltbook_authentication"] = self._component(
                    auth_status, self._safe_detail(exc)
                )
        if "moltbook_authentication" not in components:
            components["moltbook_authentication"] = self._component(
                auth_status,
                "authenticated_profile_read_succeeded"
                if auth_status == "healthy"
                else "profile_unavailable",
            )

        try:
            row = self.runtime.memory.database.fetch_one("SELECT 1 AS available")
            sqlite_ok = bool(row and row.get("available") == 1)
            components["sqlite"] = self._component(
                "healthy" if sqlite_ok else "unhealthy",
                "available" if sqlite_ok else "query_failed",
            )
        except Exception:
            components["sqlite"] = self._component("unhealthy", "database_unavailable")

        store = self.runtime.memory.store
        try:
            last_heartbeat = store.get_state("last_cycle_completion")
            last_successful_cycle = store.get_state("last_successful_cycle")
            last_error = store.get_state("last_error")
            failures = int(store.get_state("consecutive_failure_count") or 0)
            last_social = self.runtime.memory.retrieval.get_last_successful_social_action()
            components["last_heartbeat"] = self._component(
                "healthy" if last_heartbeat else "degraded",
                last_heartbeat or "no_cycle_recorded",
            )
            components["last_successful_cycle"] = self._component(
                "healthy" if last_successful_cycle else "degraded",
                last_successful_cycle or "no_successful_cycle_recorded",
            )
            components["last_successful_social_action"] = self._component(
                "healthy" if last_social else "degraded",
                last_social.get("updated_at") or last_social.get("created_at") if last_social else "no_successful_social_action_recorded",
            )
        except Exception as exc:
            last_heartbeat = last_successful_cycle = last_error = None
            failures = 0
            components["last_heartbeat"] = self._component("degraded", self._safe_detail(exc))
            components["last_successful_cycle"] = self._component("degraded", "state_unavailable")
            components["last_successful_social_action"] = self._component("degraded", "memory_unavailable")

        states = [component["status"] for component in components.values()]
        overall = (
            "unhealthy" if "unhealthy" in states
            else "degraded" if "degraded" in states
            else "healthy"
        )
        return {
            "status": overall,
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "components": components,
            "last_heartbeat": last_heartbeat,
            "last_successful_cycle": last_successful_cycle,
            "last_successful_social_action": components["last_successful_social_action"]["detail"],
            "last_error": self._safe_detail(last_error) if last_error else None,
            "consecutive_failures": failures,
        }

    @staticmethod
    def _component(status: str, detail: Any):
        return {"status": status, "detail": str(detail)[:500]}

    @staticmethod
    def _decode(result) -> Any:
        payload = result.structured_content
        if payload is None:
            payload = result.content
        if isinstance(payload, list):
            texts = [item.get("text") for item in payload if isinstance(item, dict) and isinstance(item.get("text"), str)]
            payload = texts[0] if len(texts) == 1 else "\n".join(texts) if texts else payload
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except json.JSONDecodeError:
                return {"success": result.success, "message": payload}
        return payload

    @classmethod
    def _payload_error(cls, payload: Any) -> str | None:
        if isinstance(payload, dict):
            if payload.get("success") is False:
                return str(payload.get("error") or payload.get("message") or "profile_read_failed")
            for value in payload.values():
                error = cls._payload_error(value)
                if error:
                    return error
        elif isinstance(payload, list):
            for value in payload:
                error = cls._payload_error(value)
                if error:
                    return error
        return None

    @classmethod
    def _find_http_status(cls, payload: Any) -> int | None:
        if isinstance(payload, dict):
            status = payload.get("http_status") or payload.get("status_code")
            if isinstance(status, int):
                return status
            for value in payload.values():
                found = cls._find_http_status(value)
                if found is not None:
                    return found
        elif isinstance(payload, list):
            for value in payload:
                found = cls._find_http_status(value)
                if found is not None:
                    return found
        return None

    @staticmethod
    def _safe_detail(value: Any) -> str:
        text = re.sub(r"(?i)bearer\s+[^\s,;]+", "Bearer [REDACTED]", str(value))
        return text[:500]
