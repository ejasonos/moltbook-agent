from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from agent.loop import AgentLoop, AgentLoopConfig
from config.settings import AgentSettings
from memory import SocialMemory
from mcp_client.client import MCPClient
from mcp_client.registry import ToolRegistry
from model.factory import create_model_client
from moltbook.context_manager import MoltContextManager
from moltbook.policy import MoltPolicy
from workspace.manager import WorkspaceManager


@dataclass
class MoltRuntime:
    agent_loop: AgentLoop
    mcp_client: MCPClient
    registry: ToolRegistry
    memory: SocialMemory
    policy: MoltPolicy
    context_manager: MoltContextManager

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

    os.environ.setdefault(
        "MCP_URL",
        "http://127.0.0.1:8001/mcp",
    )

    settings = AgentSettings()

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

    # Model
    model = create_model_client(settings)

    # Moltbook MCP connection
    mcp_client = MCPClient(settings)
    registry = ToolRegistry(mcp_client)

    # Establish the MCP connection and verify tool discovery.
    discovered_tools = registry.discover()

    if not discovered_tools:
        mcp_client.http.close()
        raise RuntimeError(
            "No MCP tools were discovered. "
            "Check whether the Moltbook MCP server is running."
        )

    # Persistent social memory
    memory_path = os.getenv(
        "MOLTBOOK_MEMORY_DB",
        str(project_root / "data" / "molt_memory.db"),
    )

    memory = SocialMemory(
        database_path=memory_path,
    )

    policy = MoltPolicy()

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
        enable_subagents=False,
        max_subagents_per_run=0,
    )

    agent_loop = AgentLoop(
        model=model,
        mcp_client=mcp_client,
        registry=registry,
        workspace=workspace,
        context=context_manager,
        config=agent_config,
    )

    return MoltRuntime(
        agent_loop=agent_loop,
        mcp_client=mcp_client,
        registry=registry,
        memory=memory,
        policy=policy,
        context_manager=context_manager,
    )
