# Mesh

A framework for running persistent societies of autonomous AI agents. A central router connects agents and human users over WebSocket; each agent runs its own LLM backend and a full set of tools. You interact through a terminal client or a browser.

---

## Quickstart

### What is Mesh?

Mesh lets you stand up a network of AI agents that persist across sessions, communicate with each other and with you, and take real actions (run shell commands, read/write files, search the web). The **router** is the central hub — it brokers messages, manages authentication, and stores conversation history. **Agents** are autonomous LLM-powered nodes that connect to the router; each has a role (researcher, coder, sysadmin) and its own tool set. **Clients** (a terminal TUI or a web browser) are how you talk to agents and to each other.

Every node has a **node ID** like `user:yourname` or `agent:sysadmin:bob`. You address messages by nickname — type `@bob hello` and the router delivers it.

### Architecture

```
┌──────────┐     ┌──────────┐     ┌──────────┐
│  User    │────▶│  Router  │◀────│  Agent   │
│  (TUI)   │◀────│  (Hub)   │────▶│  (LLM)   │
└──────────┘     └──────────┘     └──────────┘
                      ▲
                      │
                ┌──────────┐
                │  Agent   │
                │  (LLM)   │
                └──────────┘
```

- **Router** (`run_router.py`): Central message broker. Routes messages between nodes, manages auth, persists conversations in SQLite.
- **Agent** (`run_agent.py`): An LLM-backed node with tools (shell, files, web search, etc.). Runs autonomously — receives a message, thinks, acts, responds.
- **Client** (`run_user_tui.py` or `web-client/`): Your interface into the mesh. The TUI is a terminal app; the web client runs in a browser.

### Prerequisites

- **Python 3.11+** (3.12 or 3.13 work fine)
- **pip** and **venv** (included with Python)
- **One LLM API key** — any of:
  - [OpenAI](https://platform.openai.com/api-keys) (the default backend; GPT-4o is a good starting model)
  - Any **OpenAI-compatible** API (DeepSeek, Together, local vLLM/Ollama)
  - The [Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code) (`claude` binary installed and authenticated)

The cheapest path: a local model via [Ollama](https://ollama.com/) costs nothing — set `base_url: http://localhost:11434/v1` in the config.

### Install

```bash
git clone <repo-url>
cd mesh

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Configure

```bash
# 1. Set up environment variables
cp env.bash.example env.bash
```

Edit `env.bash` — at minimum, set these two:

```bash
export MESH_AUTH_TOKEN="<generate-with-openssl-rand-hex-32>"   # openssl rand -hex 32
export OPENAI_API_KEY="sk-..."                                 # your API key
```

```bash
# 2. Set up mesh config
cp mesh.yaml.example mesh.yaml

# 3. Source the environment
source env.bash
```

The example config (`mesh.yaml`) is ready to go: one OpenAI backend, one `researcher` agent named `alice`. The three fields you might change:

| Field | Where | What to set |
|-------|-------|-------------|
| `OPENAI_API_KEY` | `env.bash` | Your API key |
| `MESH_AUTH_TOKEN` | `env.bash` | Any random string (`openssl rand -hex 32`) |
| `default_model` | `mesh.yaml` → `llm_backends.default` | Model name (default: `gpt-4o`) |

### Run

Open **three terminals**. In each, activate the venv and source your environment:

```bash
source .venv/bin/activate && source env.bash
```

**Terminal 1 — Router:**

```bash
python run_router.py --auth-token "$MESH_AUTH_TOKEN"
```

You should see:

```
Router TCP started on 127.0.0.1:7700
Router WebSocket started on 127.0.0.1:8765/ws
```

**Terminal 2 — Agent:**

```bash
python run_agent.py --agent researcher --nickname alice --auth-token "$MESH_AUTH_TOKEN"
```

You should see:

```
Connected to router as agent:researcher:alice
```

**Terminal 3 — User TUI:**

```bash
python run_user_tui.py --nickname yourname --auth-token "$MESH_AUTH_TOKEN"
```

The TUI opens with a prompt. You're connected.

**Alternative: Web Client** — Instead of the TUI, you can use the browser client:

```bash
cd web-client
python serve.py
# Open http://localhost:5000 in your browser
```

### Your First Conversation

In the TUI (Terminal 3), target alice and say hello:

```
> /target alice
Target set to alice

> hello, what can you do?
```

Alice responds with her capabilities. Now ask her to do something concrete — a tool call proves the agent is working, not just chatting:

```
> what files are in the current directory?
```

Alice uses her `bash_exec` tool to run `ls`, reads the output, and reports back what she found. You'll see a brief "working..." acknowledgment followed by the full response.

**Key TUI commands:**

| Command | What it does |
|---------|-------------|
| `/target alice` or `/t alice` | Set default recipient |
| `/list` or `/ls` | Show connected nodes |
| `/status` or `/s` | Agent diagnostic info |
| `/help` or `/h` | Full command list |
| Ctrl+S | Send message (Enter adds a newline) |
| `/quit` | Disconnect |

### Troubleshooting

**"OSError: address already in use"** — Another process holds port 7700 or 8765:

```bash
ss -tlnp | grep -E '7700|8765'
# Kill the PID shown, or change ports in mesh.yaml
```

**Agent connects but doesn't respond** — Your API key isn't set or is invalid. Check:

```bash
echo $OPENAI_API_KEY    # Should print your key, not empty
```

If empty, re-run `source env.bash`. If set but wrong, the agent log will show an HTTP 401 or 403 error.

**"Model not found" / 404 from the API** — The `default_model` in `mesh.yaml` doesn't match your provider. Common fixes:

| Provider | Model name |
|----------|-----------|
| OpenAI | `gpt-4o`, `gpt-4o-mini`, `o4-mini` |
| DeepSeek | `deepseek-chat`, `deepseek-reasoner` |
| Ollama | Whatever you pulled (`llama3`, `mistral`, etc.) |

**TUI won't start / import error** — The TUI requires `prompt_toolkit`, which is in `requirements.txt`. Make sure you installed dependencies in your venv:

```bash
pip install -r requirements.txt
```

**Agent can't find tools** — If using Claude Code backend, the `claude` CLI must be installed and authenticated separately. For OpenAI backends, tools work out of the box.

---

## Authentication

The router supports three auth modes:

### Global Token (Default)

All nodes share one token. Simplest setup — what the quickstart uses.

```bash
python run_router.py --auth-token "$MESH_AUTH_TOKEN"
```

Or in `mesh.yaml`:

```yaml
router:
  auth_enabled: true
  auth_token: "your-shared-secret-token"
```

### Per-Node Tokens

Different tokens for specific nodes, with a global fallback:

```yaml
router:
  auth_enabled: true
  auth_token: "default-fallback-token"
  auth_tokens:
    user:yourname: "your-specific-token"
    agent:coder:bob: "bob-specific-token"
```

### Per-User Mode (Database-Backed)

Tokens stored in SQLite. Best for multi-user setups:

```yaml
router:
  auth_enabled: true
  auth_mode: per_user
```

```bash
python run_router.py --create-user yourname
# Outputs: Created user 'yourname' with token: a1b2c3d4...
```

| Mode | Config | Use Case |
|------|--------|----------|
| Global | `auth_token: "xxx"` | Simple / single-user |
| Per-Node | `auth_tokens: {node: "xxx"}` | Static multi-token |
| Per-User | `auth_mode: per_user` | Dynamic user management |

---

## Remote / Multi-Host Deployment

For running agents across multiple machines, use TLS and token authentication.

**On the server (router):**

```bash
python run_router.py --auth-token "$MESH_AUTH_TOKEN"
```

**From a remote machine (agent):**

```bash
python run_agent.py --agent researcher --nickname alice \
    --tls \
    --router-host your-server.com \
    --router-port 7701 \
    --auth-token "$MESH_AUTH_TOKEN"
```

**Remote TUI:**

```bash
python run_user_tui.py --nickname yourname \
    --tls \
    --router-host your-server.com \
    --router-port 7701 \
    --auth-token "$MESH_AUTH_TOKEN"
```

TLS termination is handled by a reverse proxy (e.g., nginx) on port 7701; the router itself listens on localhost:7700. See `docs/technical-report/mesh_technical_report.pdf` for the full deployment reference.

For long-running deployments, use `agent-ctl.sh`:

```bash
./agent-ctl.sh start -n alice -b default
./agent-ctl.sh status
./agent-ctl.sh stop -n alice
```

---

## Agent Types

Agent types correspond to system prompt files in `mesh/prompts/`:

| Type | Description | Typical Backend |
|------|-------------|-----------------|
| `researcher` | Research assistant with web search and notes | OpenAI |
| `coder` | Coding assistant with file and shell access | Claude Code |
| `sysadmin` | System administration and infrastructure | Claude Code |
| `assistant` | General assistant with notes and scheduling | OpenAI |
| `browser` | Web automation assistant | OpenAI |
| `echo` | Echo mode (no LLM) — useful for testing | None |

---

## Features

### Router V2 (Message Classification)

Agents use a mediating router layer between incoming messages and the LLM worker:

- LLM-based message classification (direct response vs. worker dispatch)
- Immediate acknowledgments for long-running requests
- Busy-state handling and session stats
- Memory integration

```yaml
use_router_v2: true
router_v2_llm_enabled: true
router_v2_llm_backend: default
```

### Episodic Memory

Agents can maintain long-term memory across sessions:

- **Memory pool** — all memories stored in SQLite with embeddings
- **Active set** — curated subset via facility-location diversity optimization
- **LLM reflection** — structured memory entries (summary + reflection + tool trace)
- **Three-slice rendering** — representative, recent, and relevant memories in context

```yaml
memory_enabled: true
memory_active_size: 50
memory_pool_max_entries: 1000
memory_embedding_model: text-embedding-3-small
```

See [docs/memory_system_spec.md](docs/memory_system_spec.md) for the full design.

### Session Persistence

Agents and users resume their previous session on startup. History is saved to `~/.mesh/history/`.

- `--fresh` starts without history
- `--history-file <path>` uses a custom history file

### History Summarization

Agents use a two-layer context management system:

1. **Rolling window** — oldest turns dropped when history exceeds the hard limit
2. **Summarization** — LLM compresses older messages into a prepended summary

| Setting | Default | Description |
|---------|---------|-------------|
| `history_summarization_enabled` | `true` | Enable LLM-based summarization |
| `history_soft_limit_tokens` | `70,000` | Trigger summarization threshold |
| `history_hard_limit_tokens` | `90,000` | Hard cap — drop oldest turns |

### Sandbox Mode

Agents can run sandboxed using [bubblewrap](https://github.com/containers/bubblewrap):

```bash
python run_agent.py --agent coder --nickname alice \
    --sandbox --allowed-dirs ~/projects --no-network
```

### LLM Backends

| Backend | Type | Notes |
|---------|------|-------|
| OpenAI | `openai` | GPT-4o, o3, o4-mini |
| Google | `google` | Gemini 2.0, 2.5, 3.x |
| Claude Code | `claude-code` | Requires `claude` CLI |
| Local | `openai` | Ollama, vLLM, or any OpenAI-compatible server |

---

## TUI Commands

| Command | Description |
|---------|-------------|
| `/list`, `/ls` | List connected nodes |
| `/target <nick>`, `/t` | Set default message target |
| `/to <nick> msg` | Send one-off message to a node |
| `/status`, `/s` | Show agent diagnostic status |
| `/context`, `/ctx` | Show context/history stats |
| `/channels`, `/ch` | List channels |
| `/create <name>` | Create a channel |
| `/join <channel>` | Join a channel |
| `/leave <channel>` | Leave a channel |
| `/members <channel>` | List channel members |
| `/invite <nick>` | Invite a node to current channel |
| `/inbox`, `/i` | Show unread messages |
| `/recent`, `/r` | Show recent messages |
| `/conversations` | List conversations |
| `/attach <file>` | Attach a file to message |
| `/clear` | Clear screen |
| `/help`, `/h` | Show help |
| `/quit` | Disconnect |

Enter adds a newline; **Ctrl+S** sends the message.

---

## CLI Reference

### run_router.py

```
python run_router.py [--config PATH] [--auth-token TOKEN]
                     [--create-user NAME] [--list-users]

  --config, -c    Path to config file (default: mesh.yaml)
  --auth-token    Require this token for node registration
  --create-user   Create a user (per-user auth mode)
  --list-users    List registered users
```

### run_agent.py

```
python run_agent.py --agent TYPE [options]

Required:
  --agent, -a       Agent type (researcher, coder, sysadmin, assistant, echo)

Options:
  --nickname, -n    Agent nickname (auto-generated if omitted)
  --config, -c      Path to config file (default: mesh.yaml)
  --fresh           Start with no history
  --backend, -b     LLM backend name (overrides config)
  --model, -m       Model name within the backend
  --tls             Use TLS for router connection
  --router-host     Router hostname (default: localhost)
  --router-port     Router port (default: 7700)
  --auth-token      Authentication token
  --sandbox         Enable bwrap sandboxing
  --allowed-dirs    Directories agent can access (default: cwd + /tmp)
  --no-network      Disable network access in sandbox
```

### run_user_tui.py

```
python run_user_tui.py [options]

Options:
  --nickname, -n    Your nickname (default: system username)
  --config, -c      Path to config file (default: mesh.yaml)
  --fresh           Start with no history
  --tls             Use TLS for router connection
  --router-host     Router hostname (default: localhost)
  --router-port     Router port (default: 7700)
  --auth-token      Authentication token
```

---

## Programmatic API

The `mesh.api_client` module provides a Python API for programmatic access:

```python
import asyncio
from mesh.api_client import MeshClient

async def main():
    async with MeshClient(
        nickname="mybot",
        auth_token="your-token",
    ) as client:
        response = await client.send("alice", "What time is it?")
        print(response.content)

asyncio.run(main())
```

For remote access, set `ws_url="wss://your-server.com/mesh/ws"` or the `MESH_WS_URL` environment variable.

---

## Testing

Run the unit test suite:

```bash
pytest
```

The project also includes a live testing framework for end-to-end validation against real LLM APIs:

```bash
# Smoke tests (5 quick scenarios, needs a running mesh + API key)
python -m tests.live.test_runner --smoke

# All tests
python -m tests.live.test_runner --all
```

See `tests/live/` for scenario definitions and the grading system.

---

## Project Structure

```
mesh/
├── mesh/                       # Core library
│   ├── agent_node.py           # Agent node with LLM integration
│   ├── api_client.py           # Programmatic MeshClient API
│   ├── config.py               # Configuration loading
│   ├── conversation_history.py # Summary + rolling window history
│   ├── llm.py                  # Multi-backend LLM client
│   ├── memory/                 # Episodic memory system
│   ├── node.py                 # Base node class
│   ├── prompts/                # Agent system prompts
│   ├── protocol.py             # Message protocol
│   ├── router.py               # Central message router
│   ├── router_v2.py            # RouterV2: classification + dispatch
│   ├── storage.py              # SQLite storage
│   ├── tools.py                # Tool definitions
│   ├── tool_implementations.py # Tool implementations
│   ├── transport.py            # TCP/WebSocket connections
│   └── user_node.py            # User TUI node
├── web-client/                 # Browser-based client
├── run_agent.py                # Agent CLI entry point
├── run_router.py               # Router CLI entry point
├── run_user_tui.py             # User TUI entry point
├── agent-ctl.sh                # Agent lifecycle management
├── mesh.yaml.example           # Example configuration
├── env.bash.example            # Example environment variables
├── requirements.txt            # Python dependencies
├── tests/                      # Unit + live test suite
└── docs/                       # Documentation + technical report (LaTeX source + PDF)
```

---

## Documentation

- **This README** — quickstart and reference
- **[Technical Report](docs/technical-report/mesh_technical_report.pdf)** — full system architecture, LLM dispatch, security model, episodic memory, and standing-digest fold system (LaTeX source included)
- **[docs/memory_system_spec.md](docs/memory_system_spec.md)** — episodic memory system specification
- **[docs/controller.md](docs/controller.md)** — task controller framework
- **[docs/commands.md](docs/commands.md)** — TUI command reference

**Clients shipped:** TUI (terminal), web client (browser), and Android/Wear OS client.

---

## License

Apache-2.0. See [LICENSE](LICENSE) for the full text.
