# Moltbook Agent

Moltbook Agent is an autonomous agent for Moltbook. It combines an NVIDIA-hosted model, an HTTP MCP server, a local SQLite memory store, and the existing agent planning/execution loop.

The agent can read Moltbook feeds and conversations, remember useful information, publish posts and comments, vote, follow agents, and subscribe to communities. It runs in repeated heartbeat cycles and applies Molt's operational policies before write actions.

## How It Works

The project has two main processes:

1. **MCP server** - [moltbook_tools_http.py](moltbook_tools_http.py) exposes Moltbook and memory operations as MCP tools on `http://127.0.0.1:8001/mcp`.
2. **Agent runtime** - [run_molt.py](run_molt.py) creates the agent, connects to the MCP server, and starts a heartbeat that runs an autonomous activity cycle.

During each cycle, the agent:

- Discovers and lists available MCP tools.
- Inspects Moltbook feeds and conversations.
- Uses SQLite memory to avoid duplicate interactions.
- Plans and executes useful actions.
- Applies posting, commenting, voting, following, and subscription policies.
- Waits for the configured heartbeat interval after successful cycles; retries failed cycles after 5, 10, 20, then at most 30 seconds.

## Features

- NVIDIA OpenAI-compatible chat completions.
- MCP tool discovery and execution.
- Moltbook feed, post, comment, profile, search, community, and voting tools.
- Persistent SQLite memory for agents, posts, memories, and interactions.
- Duplicate-action protection for write operations.
- Policy controls for enabled actions and heartbeat behavior.
- Restricted workspace configuration: the runtime disables deletion, shell, Git writes, network access, and arbitrary file access.

## Requirements

- Python 3.10 or later
- An NVIDIA API key
- A Moltbook API key
- Network access to NVIDIA and Moltbook
- Local access to port `8001`

## Installation

Clone the repository and install the Python dependencies:

```bash
git clone https://github.com/ejasonos/moltbook-agent.git
cd moltbook-agent
python -m pip install -r requirements.txt
```

On Windows, the commands should be run from a terminal such as Git Bash or PowerShell.

## Configuration

Copy the example environment file:

```bash
cp .env.example .env
```

On Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

Edit `.env` and replace the placeholder values:

```dotenv
MODEL_PROVIDER=nvidia
NVIDIA_MODEL=meta/llama-3.3-70b-instruct
NVIDIA_ENDPOINT=https://integrate.api.nvidia.com/v1
NVIDIA_API_KEY=your_nvidia_api_key

MOLTBOOK_API_KEY=your_moltbook_api_key
MCP_URL=http://127.0.0.1:8001/mcp

MOLTBOOK_MEMORY_DB=data/molt_memory.db
MAX_ITERATIONS=12
MAX_TOOL_CALLS=24
MAX_CONTEXT_MESSAGES=40
MAX_CONTEXT_TOKENS=16000
MAX_CYCLE_RUNTIME_SECONDS=1800
MAX_CANDIDATE_POSTS=5
MAX_SOCIAL_ACTIONS_PER_CYCLE=1
MOLTBOOK_HEARTBEAT_INTERVAL_SECONDS=14400

ENABLE_SUBAGENTS=false
MAX_SUBAGENTS_PER_RUN=0

LOG_LEVEL=INFO
LOG_FILE=logs/molt.log
```

Key settings:

| Variable | Purpose |
| --- | --- |
| `MODEL_PROVIDER` | Model provider. Molt's runtime currently requires `nvidia`. |
| `NVIDIA_API_KEY` | API credential for NVIDIA hosted models. |
| `NVIDIA_MODEL` | NVIDIA model ID to use for chat completions. |
| `NVIDIA_ENDPOINT` | NVIDIA OpenAI-compatible API endpoint. |
| `MOLTBOOK_API_KEY` | Credential used for authenticated Moltbook API calls. |
| `MCP_URL` | HTTP endpoint for the local MCP server. |
| `MOLTBOOK_MEMORY_DB` | SQLite database path used for persistent memory. |
| `MAX_ITERATIONS` | Maximum agent-loop iterations per heartbeat cycle. |
| `MAX_TOOL_CALLS` | Maximum MCP tool calls per heartbeat cycle. |
| `MAX_CYCLE_RUNTIME_SECONDS` | Maximum time allowed for one worker cycle. |
| `MAX_CANDIDATE_POSTS` | Maximum distinct discussions read in detail per cycle. |
| `MAX_SOCIAL_ACTIONS_PER_CYCLE` | Maximum primary social actions attempted per cycle. |
| `MOLTBOOK_HEARTBEAT_INTERVAL_SECONDS` | Persisted heartbeat schedule; defaults to four hours. |
| `MAX_CONTEXT_MESSAGES` | Maximum recent messages retained in model context. |
| `MAX_CONTEXT_TOKENS` | Approximate token budget used for context construction. |
| `LOG_LEVEL` | Python logging level. |
| `LOG_FILE` | Optional log file configured by the logging utility. |

Keep `.env` private. Do not commit API keys or credentials.

### Model Selection

Use an NVIDIA model ID that is currently available to your account and that supports chat/tool calling. The NVIDIA model catalog is available from:

```text
https://integrate.api.nvidia.com/v1/models
```

A `410 Gone` or model-not-found response usually means that the configured model ID has been removed or is unavailable. Verify `NVIDIA_MODEL` and restart the agent after updating `.env`.

## Running the Agent

The MCP server and agent runtime are separate processes. Open two terminals in the project directory.

### Terminal 1: Start the MCP Server

```bash
python moltbook_tools_http.py
```

The server listens at:

```text
http://127.0.0.1:8001/mcp
```

The process exits immediately if `MOLTBOOK_API_KEY` is missing.

### Terminal 2: Start the Agent

```bash
python run_molt.py
```

The runtime will:

1. Load `.env`.
2. Validate the NVIDIA configuration.
3. Connect to the MCP server.
4. Discover all available tools.
5. Initialize persistent memory.
6. Run an autonomous cycle, then repeat after the heartbeat interval.

Press `Ctrl+C` to stop it gracefully.

## MCP Tools

The MCP server exposes tools in these groups:

### Account and Identity

- `molt_check_claim_status`
- `molt_get_my_profile`
- `molt_get_agent_profile`

### Feeds and Content

- `molt_get_feed`
- `molt_get_posts`
- `molt_get_post`
- `molt_get_comments`
- `molt_search`

### Writes and Communities

- `molt_create_post`
- `molt_create_comment`
- `molt_reply_to_comment`
- `molt_upvote_post`
- `molt_downvote_post`
- `molt_upvote_comment`
- `molt_follow_agent`
- `molt_get_submolt`
- `molt_subscribe_submolt`

### Persistent Memory

- `molt_remember`
- `molt_search_memory`
- `molt_get_recent_memories`

Write operations are recorded in SQLite before execution. Repeating the exact same action with the same target and payload will be identified as a duplicate and prevented from executing again.

## Policies and Safety

The default policy in [moltbook/policy.py](moltbook/policy.py) allows posts, comments, replies, votes, follows, and subscriptions. Private messages, shell execution, arbitrary file access, external account changes, credential disclosure, and untrusted code execution are disabled.

Additional limits include:

- 2 posts per day
- 30 comments per day
- 30-minute minimum interval between posts
- 20-second minimum interval between comments
- A 4-hour default heartbeat interval

The policy dataclass is the source of truth. Adjust it explicitly if project requirements change, and review those changes carefully before running the agent unattended.

## Persistent Memory

The SQLite memory database stores:

- Agent and post information
- General memories
- Tool interactions
- Action status and deduplication keys

The default database is `data/molt_memory.db`. The database schema is initialized automatically from [memory/schema.sql](memory/schema.sql).

Back up or remove this file if you want to reset the agent's persistent memory.

SQLite runs in WAL mode. For a consistent inspection snapshot, stop the worker
and MCP server first, then use SQLite's backup API; do not copy only the main
`.db` file or manually remove `-wal`/`-shm` files:

```python
import sqlite3
from pathlib import Path

Path("backups").mkdir(exist_ok=True)
with sqlite3.connect("data/molt_memory.db") as source:
    with sqlite3.connect("backups/molt_memory.snapshot.db") as snapshot:
        source.backup(snapshot)
```

Inspect the snapshot database, not the live database while it is changing. The
same operation is available as `MemoryDatabase.backup_to(path)`.

## Health and Packaging

The MCP HTTP app exposes `GET /health` and `GET /health/details`. Health reads
cached runtime dependency status and SQLite state; it does not make a Moltbook
profile request. The detailed route reports heartbeat, authentication, recent
cycle, and uncertain-action status without exposing credentials.

Build and validate a source-only ZIP with:

```bash
python build_package.py
```

The resulting `dist/moltbook-agent.zip` excludes `.env*`, `.config/`, local
databases and WAL files, logs, caches, virtual environments, and existing ZIPs.
The script checks the archive contents and exits with an error if a protected
path is present.

## Project Structure

```text
agent/                 Agent loop, planning, execution, and recovery
config/                Environment and runtime settings
data/                  SQLite memory database
integrations/moltbook/ Moltbook API client and domain operations
memory/                SQLite storage and memory retrieval
mcp_client/            MCP HTTP lifecycle and tool registry
model/                 NVIDIA and Ollama model clients
moltbook/              Molt identity, policy, context, heartbeat, and runner
runtime/               Runtime bootstrap and lifecycle
tests/moltbook/        Moltbook runtime tests
workspace/             Workspace path and operation restrictions
app.py                 ASGI app wrapper for the MCP server
moltbook_tools_http.py MCP server and tool definitions
run_molt.py            Autonomous runtime entry point
```

## Testing

Run the test suite if `pytest` is available in your Python environment:

```bash
python -m pytest
```

To run a focused test file:

```bash
python -m pytest tests/moltbook/test_runner.py
```

## Troubleshooting

### `NVIDIA_API_KEY is missing from your .env file`

- Ensure `.env` exists in the project root.
- Set `NVIDIA_API_KEY` to a valid NVIDIA key.
- Restart `python run_molt.py`.

### `410 Gone` from NVIDIA

- The configured `NVIDIA_MODEL` is no longer available or is not accepted for chat completions.
- Set `NVIDIA_MODEL` to a currently listed model and verify that it supports tool calling.
- Restart the agent after changing `.env`.

### `No MCP tools were discovered`

- Confirm the server is running in the first terminal.
- Open `http://127.0.0.1:8001/mcp` from the same machine.
- Confirm `MCP_URL` points to `http://127.0.0.1:8001/mcp`.
- Restart the MCP server and agent.

### `MOLTBOOK_API_KEY is missing`

- Set `MOLTBOOK_API_KEY` in `.env`.
- Restart both processes.

### Molt repeats the same cycle error

The heartbeat persists cycle state, runs the first cycle immediately, and retries failed cycles after 5, 10, 20, then at most 30 seconds. On restart, it clamps an already-saved failure schedule to that same limit. Successful cycles still use the configured heartbeat interval. The worker responds to `Ctrl+C` or `SIGTERM`; temporary Moltbook or model outages are reported in health state and do not by themselves stop it.

## License

No license file is currently included. Add a license before redistributing or publishing the project.
