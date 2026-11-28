from __future__ import annotations

import json
import logging
import os
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
    """Read-only health reporting backed by live checks or cached runtime state."""

    def __init__(self, runtime=None, *, memory=None, application_role: str = "worker"):
        self.runtime = runtime
        self.memory = memory or getattr(runtime, "memory", None)
        self.application_role = application_role

    def check(self) -> dict[str, Any]:
        components = self._dependency_components()
        components.update(self._state_components())
        statuses = [item["status"] for item in components.values()]
        overall = (
            "unhealthy" if "unhealthy" in statuses
            else "degraded" if "degraded" in statuses
            else "healthy"
        )
        last_heartbeat = self._get_state("last_cycle_completion")
        last_successful_cycle = self._get_state("last_successful_cycle")
        last_error = self._get_state("last_error")
        try:
            failures = int(self._get_state("consecutive_failure_count") or 0)
        except (TypeError, ValueError):
            failures = 0
        try:
            row = self.memory.database.fetch_one(
                "SELECT COUNT(*) AS count FROM interactions "
                "WHERE status IN ('unknown', 'verification_required', 'pending')"
            )
            uncertain_actions = int(row["count"] if row else 0)
        except Exception:
            uncertain_actions = None
        social = components.get("last_successful_social_action", {})
        return {
            "status": overall,
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "application_role": self.application_role,
            "components": components,
            "last_heartbeat": last_heartbeat,
            "last_successful_cycle": last_successful_cycle,
            "last_successful_social_action": social.get("detail"),
            "last_error": self._safe_detail(last_error) if last_error else None,
            "consecutive_failures": failures,
            "current_execution_state": self._get_state("cycle_status") or "not_started",
            "outstanding_uncertain_actions": uncertain_actions,
        }

    def _dependency_components(self) -> dict[str, dict[str, Any]]:
        cached = self._cached_components()
        if self.runtime is None:
            if self.application_role == "mcp_http_server":
                cached["mcp_connectivity"] = self._component("healthy", "mcp_http_server_serving")
            cached.setdefault("mcp_connectivity", self._component("degraded", "runtime_connectivity_not_recorded"))
            cached.setdefault("tool_discovery", self._component("degraded", "runtime_tool_discovery_not_recorded"))
            cached.setdefault("nvidia_model", self._component("degraded", "runtime_model_status_not_recorded"))
            return cached

        model = self.runtime.agent_loop.model
        health_check = getattr(model, "health_check", None)
        if callable(health_check):
            try:
                available, detail = health_check()
                cached["nvidia_model"] = self._component(
                    "healthy" if available else "degraded", detail
                )
            except Exception:
                logger.warning("Runtime health probe failed | component=nvidia_model")
                cached["nvidia_model"] = self._component("degraded", "nvidia_health_check_failed")
        else:
            configured = bool(getattr(model, "model", None))
            cached["nvidia_model"] = self._component(
                "healthy" if configured else "unhealthy",
                "configured" if configured else "model_not_configured",
            )

        try:
            discovered = self.runtime.registry.discover()
            names = {tool.name for tool in discovered}
            connected = bool(discovered)
            tools_ok = EXPECTED_MOLTBOOK_TOOLS.issubset(names)
            cached["mcp_connectivity"] = self._component(
                "healthy" if connected else "degraded",
                "connected" if connected else "no_tools_discovered",
            )
            cached["tool_discovery"] = self._component(
                "healthy" if tools_ok else "degraded",
                "required_moltbook_tools_available" if tools_ok else "required_moltbook_tools_missing",
            )
        except Exception as exc:
            cached["mcp_connectivity"] = self._component("degraded", self._safe_detail(exc))
            cached["tool_discovery"] = self._component("degraded", "discovery_failed")

        # Persist only safe component summaries so the HTTP health endpoint can
        # report status without making additional network requests.
        try:
            self.memory.store.set_state(
                "runtime_health_components",
                json.dumps(cached, ensure_ascii=False),
            )
        except Exception:
            logger.warning("Could not persist runtime health component snapshot.")
        return cached

    def _cached_components(self) -> dict[str, dict[str, Any]]:
        raw = self._get_state("runtime_health_components")
        if not raw:
            return {}
        try:
            value = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            return {}
        if not isinstance(value, dict):
            return {}
        return {
            str(name): item
            for name, item in value.items()
            if isinstance(item, dict)
            and item.get("status") in {"healthy", "degraded", "unhealthy"}
        }

    def _state_components(self) -> dict[str, dict[str, Any]]:
        components: dict[str, dict[str, Any]] = {}
        try:
            row = self.memory.database.fetch_one("SELECT 1 AS available")
            available = bool(row and row.get("available") == 1)
            components["sqlite"] = self._component(
                "healthy" if available else "unhealthy",
                "available" if available else "query_failed",
            )
        except Exception:
            components["sqlite"] = self._component("unhealthy", "database_unavailable")

        auth = self._get_state("moltbook_auth_status") or "unavailable"
        auth_status = {
            "authenticated": "healthy",
            "failed": "unhealthy",
        }.get(auth, "degraded")
        components["moltbook_authentication"] = self._component(auth_status, auth)

        heartbeat_status = self._get_state("heartbeat_status")
        cycle_status = self._get_state("cycle_status")
        if heartbeat_status == "running":
            process_id = self._get_state("heartbeat_process_id")
            try:
                if not process_id:
                    raise ProcessLookupError
                os.kill(int(process_id), 0)
                status, detail = "healthy", "running"
            except (ValueError, OSError, ProcessLookupError):
                status, detail = "degraded", "worker_process_not_running"
        elif heartbeat_status == "stopped":
            status, detail = "degraded", "stopped"
        elif cycle_status in {"success", "abstained"}:
            status, detail = "healthy", cycle_status
        elif cycle_status == "failed":
            status, detail = "unhealthy", "failed"
        else:
            status, detail = "degraded", "not_started"
        components["heartbeat"] = self._component(status, detail)

        last_heartbeat = self._get_state("last_cycle_completion")
        last_success = self._get_state("last_successful_cycle")
        components["last_heartbeat"] = self._component(
            "healthy" if last_heartbeat else "degraded",
            last_heartbeat or "no_cycle_recorded",
        )
        components["last_successful_cycle"] = self._component(
            "healthy" if last_success else "degraded",
            last_success or "no_successful_cycle_recorded",
        )
        try:
            last_social = self.memory.retrieval.get_last_successful_social_action()
        except Exception:
            last_social = None
        components["last_successful_social_action"] = self._component(
            "healthy" if last_social else "degraded",
            (
                last_social.get("updated_at") or last_social.get("created_at")
                if last_social else "no_successful_social_action_recorded"
            ),
        )
        return components

    def _get_state(self, key: str) -> str | None:
        if self.memory is None:
            return None
        try:
            return self.memory.store.get_state(key)
        except Exception:
            return None

    @staticmethod
    def _component(status: str, detail: Any):
        return {"status": status, "detail": str(detail)[:500]}

    @staticmethod
    def _safe_detail(value: Any) -> str:
        text = str(value)
        for variable in ("MOLTBOOK_API_KEY", "NVIDIA_API_KEY"):
            secret = os.getenv(variable)
            if secret:
                text = text.replace(secret, "[REDACTED]")
        text = re.sub(r"(?i)bearer\s+[^\s,;]+", "Bearer [REDACTED]", text)
        return text[:500]
