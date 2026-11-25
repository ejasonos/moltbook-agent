from __future__ import annotations

from copy import deepcopy
from typing import Any

from mcp_client.client import MCPClient
from mcp_client.schemas import MCPTool


class ToolRegistry:
    """
    Discovers, normalizes, and exposes MCP tools to the agent.

    The registry is responsible for:
    - discovering available MCP tools
    - normalizing their schemas
    - attaching execution policies
    - exposing all tools to the model

    Tool policies do NOT hide tools from the model.
    The model always receives the complete available tool set.
    Policies are enforced later by the agent runtime before execution.
    """

    def __init__(
        self,
        client: MCPClient,
    ) -> None:
        self.client = client
        self._tools: dict[str, MCPTool] = {}

    # =============================================================
    # DISCOVERY
    # =============================================================

    def discover(self) -> list[MCPTool]:
        """
        Discover tools from the MCP server and normalize them.

        Policies are metadata only at this layer. All discovered
        tools remain available to the model.
        """

        raw_tools = self.client.list_tools()

        self._tools.clear()

        for raw_tool in raw_tools:
            name = self._extract_tool_name(raw_tool)

            if not name:
                continue

            description = self._extract_tool_description(
                raw_tool
            )

            input_schema = self._extract_input_schema(
                raw_tool
            )

            tool = MCPTool(
                name=name,
                description=description,
                input_schema=input_schema,
            )

            self._tools[name] = tool

        return self.all()

    # =============================================================
    # TOOL LOOKUP
    # =============================================================

    def get(
        self,
        name: str,
    ) -> MCPTool | None:
        """
        Return a registered tool by name.
        """

        return self._tools.get(name)

    def has(
        self,
        name: str,
    ) -> bool:
        """
        Return True if a tool is registered.
        """

        return name in self._tools

    def all(self) -> list[MCPTool]:
        """
        Return every registered tool.

        Policies do not filter this list.
        """

        return list(self._tools.values())

    # =============================================================
    # MODEL TOOL DEFINITIONS
    # =============================================================

    def to_ollama_tools(
        self,
    ) -> list[dict[str, Any]]:
        """
        Convert every registered tool into Ollama-compatible
        tool definitions.
        """

        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": self._input_schema_with_reason(
                        tool.input_schema
                    ),
                },
            }
            for tool in self.all()
        ]

    def to_openai_tools(
        self,
    ) -> list[dict[str, Any]]:
        """
        Convert every registered tool into OpenAI-compatible
        tool definitions.
        """

        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": self._input_schema_with_reason(
                        tool.input_schema
                    ),
                },
            }
            for tool in self.all()
        ]

    @staticmethod
    def _input_schema_with_reason(
        input_schema: dict[str, Any],
    ) -> dict[str, Any]:
        schema = deepcopy(input_schema)
        properties = schema.setdefault("properties", {})
        properties["reason_for_tool"] = {
            "type": "string",
            "description": "Briefly explain why this tool call is needed.",
        }
        required = schema.setdefault("required", [])
        if "reason_for_tool" not in required:
            required.append("reason_for_tool")
        return schema

    # =============================================================
    # RAW TOOL NORMALIZATION
    # =============================================================

    def _extract_tool_name(
        self,
        raw_tool: Any,
    ) -> str | None:
        """
        Extract a tool name from an MCP tool representation.
        """

        if isinstance(raw_tool, dict):
            name = raw_tool.get("name")

            if isinstance(name, str):
                return name

            return None

        name = getattr(
            raw_tool,
            "name",
            None,
        )

        if isinstance(name, str):
            return name

        return None

    def _extract_tool_description(
        self,
        raw_tool: Any,
    ) -> str:
        """
        Extract a tool description from an MCP tool representation.
        """

        if isinstance(raw_tool, dict):
            description = raw_tool.get(
                "description",
                "",
            )

            return (
                str(description)
                if description is not None
                else ""
            )

        description = getattr(
            raw_tool,
            "description",
            "",
        )

        return (
            str(description)
            if description is not None
            else ""
        )

    def _extract_input_schema(
        self,
        raw_tool: Any,
    ) -> dict[str, Any]:
        """
        Extract the input schema from an MCP tool representation.
        """

        if isinstance(raw_tool, dict):
            schema = raw_tool.get(
                "inputSchema"
            )

            if schema is None:
                schema = raw_tool.get(
                    "input_schema"
                )

            if isinstance(schema, dict):
                return schema

            return {}

        schema = getattr(
            raw_tool,
            "inputSchema",
            None,
        )

        if schema is None:
            schema = getattr(
                raw_tool,
                "input_schema",
                None,
            )

        if isinstance(schema, dict):
            return schema

        return {}