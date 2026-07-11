#!/bin/bash
# Agent management script for the hello-world mesh
# Usage: ./agent-ctl.sh <command> [options]
#
# Commands:
#   start [options]                - Start an agent
#   stop <nick>                    - Stop an agent (local only)
#   remote-stop <nick> [reason]    - Stop an agent on any host via mesh
#   restart [options]              - Restart an agent
#   status [nick]                  - Show status of agent(s)
#   logs <nick> [lines]            - Tail agent logs
#
# Options (for start/restart):
#   -n, --nick <nickname>          - Agent nickname (required)
#   -b, --backend <backend>        - LLM backend
#   -t, --type <type>              - Agent type (researcher, coder, etc.)
#   -m, --model <model>            - Model override
#   -c, --conversation <name>      - Load a conversation from mesh storage
#   --list-conversations           - List available conversations
#
# Examples:
#   ./agent-ctl.sh start -n eve -b openai-reasoning-medium
#   ./agent-ctl.sh start --nick alice --backend claude-code --model opus
#   ./agent-ctl.sh start -n tron -t coder -b claude-code
#   ./agent-ctl.sh start -n alice -c research-notes     # Load conversation
#   ./agent-ctl.sh conversations                         # List conversations
#   ./agent-ctl.sh stop ada
#   ./agent-ctl.sh restart -n eve -b claude-code
#   ./agent-ctl.sh status
#   ./agent-ctl.sh logs eve 50
#
# Legacy positional syntax still works:
#   ./agent-ctl.sh start eve openai-reasoning-medium

set -e

# Resolve symlinks to get the real script directory
SCRIPT_SOURCE="${BASH_SOURCE[0]}"
while [[ -L "$SCRIPT_SOURCE" ]]; do
    SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_SOURCE")" && pwd)"
    SCRIPT_SOURCE="$(readlink "$SCRIPT_SOURCE")"
    [[ "$SCRIPT_SOURCE" != /* ]] && SCRIPT_SOURCE="$SCRIPT_DIR/$SCRIPT_SOURCE"
done
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_SOURCE")" && pwd)"

# Ensure HOME points to the real user home (not a CC fallback override)
# CC multi-account fallback can leave HOME set to ~/.claude-acctN, which
# breaks tool backends (gmail, calendar, etc.) that use ~/.config/mesh/
export HOME="$(getent passwd "$(whoami)" | cut -d: -f6)"

# Source environment
if [[ -f "$SCRIPT_DIR/env.bash" ]]; then
    source "$SCRIPT_DIR/env.bash"
fi

# Config file (absolute path)
CONFIG="${MESH_CONFIG:-$SCRIPT_DIR/mesh.yaml}"
LOG_DIR="${LOG_DIR:-/tmp}"

# Python executable (use venv if available)
# Try PYTHON env var first, then .venv, then any .venv-* pattern, then system python3
if [[ -n "${PYTHON:-}" && -x "$PYTHON" ]]; then
    : # Use caller-specified PYTHON
elif [[ -x "$SCRIPT_DIR/.venv/bin/python3" ]]; then
    PYTHON="$SCRIPT_DIR/.venv/bin/python3"
else
    # Search for any .venv-* directory (e.g., .venv-mira, .venv-sobek)
    for vdir in "$SCRIPT_DIR"/.venv-*/bin/python3; do
        if [[ -x "$vdir" ]]; then
            PYTHON="$vdir"
            break
        fi
    done
    PYTHON="${PYTHON:-python3}"
fi

# Remote router settings (can be overridden via env or command line)
ROUTER_HOST="${MESH_ROUTER_HOST:-localhost}"
ROUTER_PORT="${MESH_ROUTER_PORT:-7701}"
ROUTER_TLS="${MESH_ROUTER_TLS:-1}"  # 1 = use TLS, 0 = no TLS
AUTH_TOKEN="${MESH_AUTH_TOKEN:-}"

usage() {
    echo "Agent Control Script for Hello World Mesh"
    echo "=========================================="
    echo ""
    echo "Usage: $0 <command> [options]"
    echo ""
    echo "Commands:"
    echo "  start [options]                        - Start an agent"
    echo "  stop <nick>                            - Stop an agent (graceful, then force)"
    echo "  stop-all                               - Stop all agents on this host"
    echo "  restart [options]                      - Restart with optional new settings"
    echo "  restart-all                            - Restart all agents on this host"
    echo "  list                                   - List running agents on this host"
    echo "  status [nick]                          - Show status (config vs running)"
    echo "  remote-stop <nick> [reason]            - Stop an agent on any host via mesh"
    echo "  logs <nick> [lines]                    - Tail agent logs (default: 20 lines)"
    echo "  conversations                          - List available mesh conversations"
    echo "  help                                   - Show this detailed help"
    echo ""
    echo "Options (for start/restart - can be in any order):"
    echo "  -n, --nick <nickname>   Agent nickname (e.g., alice, bob, eve, tron)"
    echo "  -b, --backend <name>    LLM backend name (defaults to config or 'default')"
    echo "  -t, --type <type>       Agent type: researcher, coder, sysadmin, assistant"
    echo "  -m, --model <model>     Model name within the backend"
    echo "  -c, --conversation <n>  Load a conversation from mesh storage by name"
    echo "  --sandbox               Enable bwrap sandboxing (restricts file/bash access)"
    echo "  --allowed-dirs <dirs>   Directories writable in sandbox (space-separated)"
    echo "  --no-network            Block network access in sandbox"
    echo "  -f, --foreground        Run in foreground (for systemd, don't daemonize)"
    echo ""
    echo "Examples (new flag syntax):"
    echo "  $0 start -n alice                             # Start alice with default backend"
    echo "  $0 start -n alice -b openai-reasoning-medium  # Start with OpenAI reasoning"
    echo "  $0 start --nick alice --backend claude-code   # Long form flags"
    echo "  $0 start -n tron -b claude-code -t coder      # Start as coder type"
    echo "  $0 start -n eve -b claude-code -m opus        # With model override"
    echo "  $0 start -n alice -c research-notes           # Load a mesh conversation"
    echo "  $0 conversations                              # List available conversations"
    echo "  $0 restart -n eve -b openai-reasoning-medium  # Restart with new backend"
    echo ""
    echo "Examples (legacy positional syntax - still works):"
    echo "  $0 start alice                           # Start alice with default backend"
    echo "  $0 start alice openai-reasoning-medium   # Start with OpenAI reasoning"
    echo "  $0 start alice claude-code               # Start with Claude Code"
    echo "  $0 start tron claude-code-opus coder     # Start tron as coder with Opus"
    echo "  $0 start alice vllm-local               # Start with local vLLM (QwQ-32B)"
    echo "  $0 start eve claude-code - gpt-4o        # Start eve with claude-code, model gpt-4o"
    echo "  $0 start alice - - opus                  # Keep defaults but override model to opus"
    echo "  $0 start tron claude-code coder          # Start tron as coder"
    echo "  $0 stop alice                            # Stop alice (local process)"
    echo "  $0 remote-stop alice                     # Stop alice on any host via mesh"
    echo "  $0 status                                # Show all agents' status"
    echo "  $0 logs alice 100                        # Show last 100 lines of alice's log"
    echo ""
    echo "Available Backends:"
    echo "  default                  - OpenAI GPT-5.1 (standard)"
    echo "  openai-reasoning-low     - OpenAI GPT-5.1 with low reasoning effort"
    echo "  openai-reasoning-medium  - OpenAI GPT-5.1 with medium reasoning effort"
    echo "  openai-reasoning-high    - OpenAI GPT-5.2 with high reasoning effort"
    echo "  gemini                   - Google Gemini 3.5 Flash"
    echo "  gemini-thinking          - Google Gemini 3.5 with thinking"
    echo "  gemini-2.5               - Google Gemini 2.5"
    echo "  claude-code              - Claude Code (Sonnet via CLI)"
    echo "  claude-code-opus         - Claude Code (Opus via CLI)"
    echo "  zai                      - ZAI backend"
    echo "  synthetic                - Synthetic/test backend"
    echo "  synthetic-thinking       - Synthetic with thinking"
    echo "  synthetic-thinking-glm   - Synthetic thinking GLM"
    echo "  local                    - Local model via Ollama (localhost:11434)"
    echo "  vllm-local               - Local vLLM server (localhost:8080, QwQ-32B)"
    echo ""
    echo "Environment Variables:"
    echo "  MESH_ROUTER_HOST  - Router hostname (default: localhost)"
    echo "  MESH_ROUTER_PORT  - Router port (default: 7701)"
    echo "  MESH_ROUTER_TLS   - Use TLS (1=yes, 0=no, default: 1)"
    echo "  MESH_AUTH_TOKEN   - Auth token for router (required)"
    echo "  MESH_CONFIG       - Config file path (default: mesh.yaml)"
    echo "  LOG_DIR           - Log directory (default: /tmp)"
    echo "  PYTHON            - Python executable (default: .venv/bin/python3)"
    echo ""
    echo "Log Files:"
    echo "  Logs are written to: \$LOG_DIR/<nick>.log (default: /tmp/<nick>.log)"
    echo ""
    exit "${1:-1}"
}


# Parse flags for start/restart commands
# Sets: PARSED_NICK, PARSED_BACKEND, PARSED_TYPE, PARSED_MODEL, PARSED_CONVERSATION, PARSED_SANDBOX, PARSED_ALLOWED_DIRS, PARSED_NO_NETWORK
parse_start_args() {
    PARSED_NICK=""
    PARSED_BACKEND=""
    PARSED_TYPE=""
    PARSED_MODEL=""
    PARSED_CONVERSATION=""
    PARSED_SANDBOX=""
    PARSED_ALLOWED_DIRS=""
    PARSED_NO_NETWORK=""
    PARSED_FOREGROUND=""

    # Check if first arg looks like a flag
    local using_flags=false
    if [[ "$1" == -* ]]; then
        using_flags=true
    fi

    if $using_flags; then
        # Parse named flags
        while [[ $# -gt 0 ]]; do
            case "$1" in
                -n|--nick)
                    PARSED_NICK="$2"
                    shift 2
                    ;;
                -b|--backend)
                    PARSED_BACKEND="$2"
                    shift 2
                    ;;
                -t|--type)
                    PARSED_TYPE="$2"
                    shift 2
                    ;;
                -m|--model)
                    PARSED_MODEL="$2"
                    shift 2
                    ;;
                -c|--conversation)
                    PARSED_CONVERSATION="$2"
                    shift 2
                    ;;
                --sandbox)
                    PARSED_SANDBOX="1"
                    shift
                    ;;
                --allowed-dirs)
                    # Collect all following args until next flag or end
                    shift
                    while [[ $# -gt 0 && "$1" != -* ]]; do
                        PARSED_ALLOWED_DIRS="$PARSED_ALLOWED_DIRS $1"
                        shift
                    done
                    ;;
                --no-network)
                    PARSED_NO_NETWORK="1"
                    shift
                    ;;
                --foreground|-f)
                    PARSED_FOREGROUND="1"
                    shift
                    ;;
                *)
                    echo "Unknown option: $1"
                    usage
                    ;;
            esac
        done
    else
        # Legacy positional: nick [backend] [type] [model]
        PARSED_NICK="${1:-}"
        PARSED_BACKEND="${2:-}"
        PARSED_TYPE="${3:-}"
        PARSED_MODEL="${4:-}"

        # Handle "-" as placeholder for "use default"
        [[ "$PARSED_BACKEND" == "-" ]] && PARSED_BACKEND=""
        [[ "$PARSED_TYPE" == "-" ]] && PARSED_TYPE=""
        [[ "$PARSED_MODEL" == "-" ]] && PARSED_MODEL=""
    fi

    # Validate nick is provided
    if [[ -z "$PARSED_NICK" ]]; then
        echo "Error: nickname is required (-n/--nick or first positional argument)"
        usage
    fi
}

get_agent_type() {
    local nick="$1"
    # Look up agent type from mesh.yaml
    python3 -c "
import yaml
with open('$CONFIG') as f:
    cfg = yaml.safe_load(f)
nodes = cfg.get('nodes', {})
for node_id, node_cfg in nodes.items():
    if not node_cfg:
        continue
    # Match on explicit nickname field or node ID suffix (agent:TYPE:NICK)
    explicit_nick = node_cfg.get('nickname')
    id_nick = node_id.rsplit(':', 1)[-1] if ':' in node_id else None
    if explicit_nick == '$nick' or (explicit_nick is None and id_nick == '$nick'):
        if 'agent_type' in node_cfg:
            print(node_cfg['agent_type'])
        elif ':' in node_id:
            parts = node_id.split(':')
            if len(parts) >= 2:
                print(parts[1])
        exit(0)
# Default to researcher if not found
print('researcher')
"
}

get_agent_backend() {
    local nick="$1"
    python3 -c "
import yaml
with open('$CONFIG') as f:
    cfg = yaml.safe_load(f)
nodes = cfg.get('nodes', {})
for node_id, node_cfg in nodes.items():
    if not node_cfg:
        continue
    explicit_nick = node_cfg.get('nickname')
    id_nick = node_id.rsplit(':', 1)[-1] if ':' in node_id else None
    if explicit_nick == '$nick' or (explicit_nick is None and id_nick == '$nick'):
        print(node_cfg.get('llm_backend', 'default'))
        exit(0)
print('default')
"
}

start_agent() {
    # CC computes session paths from $PWD — always launch from the project dir
    # to prevent session resolution failures when invoked from elsewhere.
    cd "$SCRIPT_DIR"

    # Args already parsed into PARSED_* variables
    local nick="$PARSED_NICK"
    local backend="$PARSED_BACKEND"
    local type_override="$PARSED_TYPE"
    local model_override="$PARSED_MODEL"

    local agent_type
    if [[ -n "$type_override" ]]; then
        agent_type="$type_override"
    else
        agent_type=$(get_agent_type "$nick")
    fi

    # Check if already running
    if pgrep -f "run_agent.py.*--nickname $nick" > /dev/null 2>&1; then
        echo "Agent '$nick' is already running"
        pgrep -la -f "run_agent.py.*--nickname $nick"
        return 1
    fi

    # Build command (use absolute path to run_agent.py)
    local cmd="$PYTHON $SCRIPT_DIR/run_agent.py --agent $agent_type --nickname $nick --config $CONFIG"

    # Add remote router settings
    if [[ -n "$ROUTER_HOST" ]]; then
        cmd="$cmd --router-host $ROUTER_HOST"
    fi
    if [[ -n "$ROUTER_PORT" ]]; then
        cmd="$cmd --router-port $ROUTER_PORT"
    fi
    if [[ "$ROUTER_TLS" == "1" ]]; then
        cmd="$cmd --tls"
    fi
    if [[ -n "$AUTH_TOKEN" ]]; then
        cmd="$cmd --auth-token $AUTH_TOKEN"
    fi

    # Add backend override
    if [[ -n "$backend" ]]; then
        cmd="$cmd --backend $backend"
    fi

    # Add model override
    if [[ -n "$model_override" ]]; then
        cmd="$cmd --model $model_override"
    fi

    # Add conversation loading
    if [[ -n "$PARSED_CONVERSATION" ]]; then
        cmd="$cmd --load-conversation $PARSED_CONVERSATION"
    fi

    # Enable relevance router by default (LLM-based channel filtering)
    cmd="$cmd --relevance-router"

    # Add sandbox settings
    if [[ -n "$PARSED_SANDBOX" ]]; then
        cmd="$cmd --sandbox"
    fi
    if [[ -n "$PARSED_ALLOWED_DIRS" ]]; then
        cmd="$cmd --allowed-dirs$PARSED_ALLOWED_DIRS"
    fi
    if [[ -n "$PARSED_NO_NETWORK" ]]; then
        cmd="$cmd --no-network"
    fi

    echo "Starting $nick ($agent_type)..."
    if [[ -n "$backend" ]]; then
        echo "  Backend: $backend"
    fi
    if [[ -n "$model_override" ]]; then
        echo "  Model: $model_override"
    fi

    if [[ -n "$PARSED_FOREGROUND" ]]; then
        # Foreground mode for systemd - run directly, no nohup/background
        echo "Running in foreground mode..."
        exec $cmd
    else
        # Background mode - use nohup and return
        nohup $cmd > "$LOG_DIR/$nick.log" 2>&1 &
        local pid=$!

        # Wait a moment and check it's running
        sleep 2
        if kill -0 $pid 2>/dev/null; then
            echo "Started $nick (PID: $pid)"
            echo "Logs: $LOG_DIR/$nick.log"
        else
            echo "Failed to start $nick. Check logs:"
            tail -20 "$LOG_DIR/$nick.log"
            return 1
        fi
    fi
}

stop_agent() {
    local nick="$1"

    local pids=$(pgrep -f "run_agent.py.*--nickname $nick" 2>/dev/null || true)
    if [[ -z "$pids" ]]; then
        echo "Agent '$nick' is not running"
        return 0
    fi

    echo "Stopping $nick (PIDs: $pids)..."

    # Collect all child PIDs (recursively) before sending signals.
    # This catches claude -p subprocesses and any other children that
    # wouldn't match the run_agent.py pattern.
    local all_pids="$pids"
    for pid in $pids; do
        local children=$(pgrep -P "$pid" 2>/dev/null || true)
        if [[ -n "$children" ]]; then
            # Also get grandchildren (e.g., claude spawns bash/tail)
            for cpid in $children; do
                local grandchildren=$(pgrep -P "$cpid" 2>/dev/null || true)
                [[ -n "$grandchildren" ]] && all_pids="$all_pids $grandchildren"
            done
            all_pids="$all_pids $children"
        fi
    done

    # Try graceful shutdown with SIGTERM to the entire process tree
    kill -TERM $all_pids 2>/dev/null || true

    # Wait up to 5 seconds for graceful shutdown
    local waited=0
    while [[ $waited -lt 5 ]]; do
        if ! pgrep -f "run_agent.py.*--nickname $nick" > /dev/null 2>&1; then
            # Parent is gone — mop up any orphaned children still lingering
            for pid in $all_pids; do
                kill -0 "$pid" 2>/dev/null && kill -KILL "$pid" 2>/dev/null || true
            done
            echo "Stopped $nick (graceful)"
            return 0
        fi
        sleep 1
        waited=$((waited + 1))
    done

    # Still running after 5 seconds, force kill entire tree with SIGKILL
    echo "Agent didn't stop gracefully, forcing shutdown..."
    kill -KILL $all_pids 2>/dev/null || true
    sleep 1

    # Verify stopped
    if pgrep -f "run_agent.py.*--nickname $nick" > /dev/null 2>&1; then
        echo "Warning: $nick may still be running"
        return 1
    fi
    echo "Stopped $nick (forced)"
}

restart_agent() {
    # Args already parsed into PARSED_* variables
    local nick="$PARSED_NICK"

    stop_agent "$nick"
    sleep 1

    if [[ -z "$PARSED_BACKEND" ]]; then
        PARSED_BACKEND=$(get_agent_backend "$nick")
    fi

    start_agent
}

status_agent() {
    local nick="${1:-}"

    if [[ -n "$nick" ]]; then
        # Status for specific agent
        local pids=$(pgrep -f "run_agent.py.*--nickname $nick" 2>/dev/null || true)
        if [[ -n "$pids" ]]; then
            echo "$nick: RUNNING (PID: $pids)"
            local backend=$(get_agent_backend "$nick")
            echo "  Backend: $backend"
            echo "  Log: $LOG_DIR/$nick.log"
        else
            echo "$nick: STOPPED"
        fi
    else
        # Status for all configured agents
        echo "Agent Status:"
        echo "============="
        $PYTHON -c "
import yaml
with open('$CONFIG') as f:
    cfg = yaml.safe_load(f)
nodes = cfg.get('nodes', {})
for node_id, node_cfg in nodes.items():
    if not node_cfg:
        continue
    if node_id.startswith('agent:'):
        nick = node_cfg.get('nickname', node_id.split(':')[-1])
        print(nick)
" | while read nick; do
            local pids=$(pgrep -f "run_agent.py.*--nickname $nick" 2>/dev/null || true)
            local backend=$(get_agent_backend "$nick")
            if [[ -n "$pids" ]]; then
                printf "  %-12s RUNNING  (PID: %s, backend: %s)\n" "$nick" "$pids" "$backend"
            else
                printf "  %-12s STOPPED  (backend: %s)\n" "$nick" "$backend"
            fi
        done
    fi
}

logs_agent() {
    local nick="$1"
    local lines="${2:-20}"

    if [[ ! -f "$LOG_DIR/$nick.log" ]]; then
        echo "No log file for $nick at $LOG_DIR/$nick.log"
        return 1
    fi

    tail -n "$lines" "$LOG_DIR/$nick.log"
}

list_conversations() {
    # List conversations in mesh storage
    $PYTHON "$SCRIPT_DIR/run_agent.py" --list-conversations
}

remote_stop_agent() {
    local nick="$1"
    local reason="${2:-}"

    if [[ -z "$AUTH_TOKEN" ]]; then
        echo "Error: MESH_AUTH_TOKEN is required for remote-stop"
        echo "Set it in env.bash or export MESH_AUTH_TOKEN=..."
        return 1
    fi

    # Get agent type from config to build the full node ID
    local agent_type
    agent_type=$(get_agent_type "$nick")

    local target="agent:${agent_type}:${nick}"

    echo "Sending shutdown request to $target via mesh..."
    if [[ -n "$reason" ]]; then
        echo "  Reason: $reason"
    fi

    # Use Python to send the shutdown message
    $PYTHON -c "
import asyncio
import sys
sys.path.insert(0, '$SCRIPT_DIR')
from mesh.protocol import make_shutdown_request
from mesh.transport import connect

async def main():
    # Connect to router
    host = '${ROUTER_HOST}'
    port = ${ROUTER_PORT}
    use_tls = ${ROUTER_TLS} == 1
    auth_token = '${AUTH_TOKEN}'

    try:
        conn = await connect(host, port, use_tls=use_tls)

        # Register as a temporary control node
        from mesh.protocol import Message, MessageType, ControlAction
        reg_msg = Message(
            from_node='ctl:shutdown',
            to_node='router',
            type=MessageType.CONTROL,
            content={'action': ControlAction.REGISTER.value},
            metadata={'auth_token': auth_token},
        )
        await conn.send(reg_msg)

        # Wait for registration ACK
        ack = await asyncio.wait_for(conn.receive(), timeout=5.0)
        if not (ack and ack.type == MessageType.CONTROL):
            print('Failed to register with router')
            return 1

        # Send shutdown request
        shutdown_msg = make_shutdown_request(
            from_node='ctl:shutdown',
            target_node='$target',
            auth_token=auth_token,
            reason='$reason',
        )
        await conn.send(shutdown_msg)
        print(f'Shutdown request sent to $target')

        # Wait for ACK (with timeout)
        try:
            response = await asyncio.wait_for(conn.receive(), timeout=5.0)
            if response:
                content = response.content if isinstance(response.content, dict) else {}
                if content.get('action') == 'shutdown_ack':
                    print('Agent acknowledged shutdown and is stopping')
                    return 0
        except asyncio.TimeoutError:
            print('No acknowledgment received (agent may be offline or already stopped)')

        await conn.close()
        return 0

    except ConnectionRefusedError:
        print(f'Could not connect to router at {host}:{port}')
        return 1
    except Exception as e:
        print(f'Error: {e}')
        return 1

sys.exit(asyncio.run(main()))
"

    local exit_code=$?
    if [[ $exit_code -eq 0 ]]; then
        echo "Shutdown request completed"
    else
        echo "Shutdown request failed (exit code: $exit_code)"
        return 1
    fi
}

list_running() {
    # List all agents currently running on this host (by inspecting processes)
    local found=0
    echo "Running agents on $(hostname):"
    echo "=============================="
    while IFS= read -r line; do
        local pid=$(echo "$line" | awk '{print $1}')
        local nick=$(echo "$line" | grep -oP '(?<=--nickname )\S+')
        local agent_type=$(echo "$line" | grep -oP '(?<=--agent )\S+')
        local backend=$(echo "$line" | grep -oP '(?<=--backend )\S+')
        [[ -z "$backend" ]] && backend="(default)"
        if [[ -n "$nick" ]]; then
            printf "  %-12s %-12s %-28s PID: %s\n" "$nick" "$agent_type" "$backend" "$pid"
            found=$((found + 1))
        fi
    done < <(pgrep -af "run_agent.py" 2>/dev/null | grep -v grep || true)
    if [[ $found -eq 0 ]]; then
        echo "  (no agents running)"
    fi
    echo ""
    echo "Total: $found agent(s)"
}

stop_all() {
    # Deduplicate: an agent may have multiple PIDs (bash wrapper + python child)
    local -A seen
    local nicks=()
    while IFS= read -r line; do
        local nick=$(echo "$line" | grep -oP '(?<=--nickname )\S+')
        if [[ -n "$nick" && -z "${seen[$nick]:-}" ]]; then
            seen[$nick]=1
            nicks+=("$nick")
        fi
    done < <(pgrep -af "run_agent.py" 2>/dev/null | grep -v grep || true)

    if [[ ${#nicks[@]} -eq 0 ]]; then
        echo "No agents running on $(hostname)"
        return 0
    fi

    echo "Stopping ${#nicks[@]} agent(s) on $(hostname)..."
    for nick in "${nicks[@]}"; do
        stop_agent "$nick"
    done
    echo "All agents stopped."
}

restart_all() {
    # Deduplicate: an agent may have multiple PIDs (bash wrapper + python child)
    local -A seen
    local agents=()
    while IFS= read -r line; do
        local nick=$(echo "$line" | grep -oP '(?<=--nickname )\S+')
        local backend=$(echo "$line" | grep -oP '(?<=--backend )\S+')
        local agent_type=$(echo "$line" | grep -oP '(?<=--agent )\S+')
        if [[ -n "$nick" && -z "${seen[$nick]:-}" ]]; then
            seen[$nick]=1
            agents+=("$nick:$backend:$agent_type")
        fi
    done < <(pgrep -af "run_agent.py" 2>/dev/null | grep -v grep || true)

    if [[ ${#agents[@]} -eq 0 ]]; then
        echo "No agents running on $(hostname)"
        return 0
    fi

    echo "Restarting ${#agents[@]} agent(s) on $(hostname)..."
    for entry in "${agents[@]}"; do
        IFS=':' read -r nick backend agent_type <<< "$entry"
        echo ""
        echo "--- $nick ---"
        stop_agent "$nick"
        sleep 1
        PARSED_NICK="$nick"
        PARSED_BACKEND="${backend:-$(get_agent_backend "$nick")}"
        PARSED_TYPE="${agent_type:-}"
        PARSED_MODEL=""
        PARSED_CONVERSATION=""
        PARSED_SANDBOX=""
        PARSED_ALLOWED_DIRS=""
        PARSED_NO_NETWORK=""
        PARSED_FOREGROUND=""
        start_agent
    done
    echo ""
    echo "All agents restarted."
}

# Main
[[ $# -lt 1 ]] && usage

cmd="$1"
shift

case "$cmd" in
    start)
        [[ $# -lt 1 ]] && usage
        parse_start_args "$@"
        start_agent
        ;;
    stop)
        [[ $# -lt 1 ]] && usage
        # Support both: stop alice  OR  stop -n alice
        if [[ "$1" == "-n" || "$1" == "--nick" ]]; then
            stop_agent "$2"
        else
            stop_agent "$1"
        fi
        ;;
    remote-stop)
        [[ $# -lt 1 ]] && usage
        remote_stop_agent "$@"
        ;;
    restart)
        [[ $# -lt 1 ]] && usage
        parse_start_args "$@"
        restart_agent
        ;;
    status)
        status_agent "$@"
        ;;
    logs)
        [[ $# -lt 1 ]] && usage
        logs_agent "$@"
        ;;
    list|ls)
        list_running
        ;;
    stop-all)
        stop_all
        ;;
    restart-all)
        restart_all
        ;;
    conversations|convos|list-conversations)
        list_conversations
        ;;
    help|-h|--help)
        usage 0
        ;;
    *)
        echo "Unknown command: $cmd"
        usage
        ;;
esac
