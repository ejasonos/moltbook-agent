from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv

from agent.loop import AgentLoop, AgentLoopConfig
from config.settings import load_settings
from memory import SocialMemory, resolve_database_path
from mcp_client.client import MCPClient
from mcp_client.registry import ToolRegistry
from model.factory import create_model_client
from moltbook.context_manager import MoltContextManager
from moltbook.policy import MoltPolicy
from runtime.health import RuntimeHealth
from workspace.manager import WorkspaceManager


@dataclass
class MoltRuntime:
    agent_loop: AgentLoop
    mcp_client: MCPClient
    registry: ToolRegistry
    memory: SocialMemory
    policy: MoltPolicy
    context_manager: MoltContextManager
    health: RuntimeHealth | None = None

    def close(self) -> None:
        try:
            self.memory.close()
        finally:
            self.mcp_client.http.close()


def create_runtime() -> MoltRuntime:
    """
    Construct all components required by Molt.

    The existing Agent Harness remains responsible for
    planning, execution, evaluation and tool orchestration.
    """

    project_root = Path(__file__).resolve().parent.parent

    # Load environment variables before constructing settings.
    load_dotenv(project_root / ".env")

    settings = load_settings()

    if settings.model_provider.strip().lower() != "nvidia":
        raise RuntimeError(
            "Molt is configured for NVIDIA. "
            "Set MODEL_PROVIDER=nvidia in .env."
        )

    if not settings.nvidia_api_key:
        raise RuntimeError(
            "NVIDIA_API_KEY is missing from your .env file."
        )

    if not settings.mcp_url:
        raise RuntimeError(
            "MCP_URL is not configured."
        )
    mcp_url = urlparse(settings.mcp_url)
    if mcp_url.scheme not in {"http", "https"} or not mcp_url.hostname or mcp_url.path != "/mcp":
        raise RuntimeError(
            "MCP_URL must point to the Moltbook MCP server's /mcp endpoint."
        )
    if not settings.nvidia_model.strip():
        raise RuntimeError("NVIDIA_MODEL must be configured.")
    nvidia_url = urlparse(settings.nvidia_endpoint)
    if nvidia_url.scheme != "https" or not nvidia_url.hostname:
        raise RuntimeError("NVIDIA_ENDPOINT must be an HTTPS endpoint.")

    # Model
    model = create_model_client(settings)

    # Moltbook MCP connection
    mcp_client = MCPClient(settings)
    registry = ToolRegistry(mcp_client)

    # Establish the MCP connection and verify tool discovery.
    discovered_tools = registry.discover()

    discovered_names = {tool.name for tool in discovered_tools}
    required_tools = {"molt_get_my_profile", "molt_get_feed", "molt_create_comment"}
    if not required_tools.issubset(discovered_names):
        mcp_client.http.close()
        raise RuntimeError(
            "The configured MCP endpoint did not expose the required Moltbook tools. "
            "Check MCP_URL and start moltbook_tools_http.py."
        )

    # Persistent social memory
    memory_path = resolve_database_path(project_root=project_root)

    memory = SocialMemory(
        database_path=memory_path,
    )

    policy = MoltPolicy(
        heartbeat_interval=settings.moltbook_heartbeat_interval_seconds,
    )

    context_manager = MoltContextManager(
        memory=memory,
        max_messages=settings.max_context_messages,
        max_tokens=settings.max_context_tokens,
    )

    # Keep the workspace isolated from destructive operations.
    workspace = WorkspaceManager(
        project_root / "workspace",
        allow_delete=False,
        allow_write=False,
        allow_git_write=False,
        allow_shell=False,
        allow_network=False,
    )

    agent_config = AgentLoopConfig(
        max_iterations=settings.max_iterations,
        max_tool_calls=settings.max_tool_calls,
        max_context_messages=settings.max_context_messages,
        max_context_tokens=settings.max_context_tokens,
        model_input_cost_per_million_tokens=(
            settings.model_input_cost_per_million_tokens
        ),
        model_output_cost_per_million_tokens=(
            settings.model_output_cost_per_million_tokens
        ),
        enable_subagents=settings.enable_subagents,
        max_subagents_per_run=settings.max_subagents_per_run,
        max_subagent_iterations=settings.max_subagent_iterations,
        max_subagent_tool_calls=settings.max_subagent_tool_calls,
        max_subagent_task_chars=settings.max_subagent_task_chars,
        max_runtime_seconds=settings.max_cycle_runtime_seconds,
        max_social_actions_per_run=settings.max_social_actions_per_cycle,
        max_candidate_posts_per_run=settings.max_candidate_posts,
    )

    agent_loop = AgentLoop(
        model=model,
        mcp_client=mcp_client,
        registry=registry,
        workspace=workspace,
        context=context_manager,
        config=agent_config,
    )

    runtime = MoltRuntime(
        agent_loop=agent_loop,
        mcp_client=mcp_client,
        registry=registry,
        memory=memory,
        policy=policy,
        context_manager=context_manager,
    )
    runtime.health = RuntimeHealth(runtime)
    return runtime
