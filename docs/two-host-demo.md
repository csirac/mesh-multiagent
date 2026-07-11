# Two-Host Demo: Cross-Network Agent Deployment

This guide demonstrates running a mesh router on one machine and an agent on
a second machine, communicating over the network. The same pattern extends to
any number of hosts.

> **Verified:** cross-address message delivery was demonstrated on
> 2026-07-11 using two loopback addresses (router on 127.0.0.2:17700,
> agent connecting from 127.0.0.1). The router delivered a user message
> to the remote agent and the agent's echo reply was routed back.

---

## Prerequisites

- Both machines have the mesh framework installed (see the top-level
  [README](../README.md) for install steps).
- An LLM backend API key is configured on the agent's machine (the router
  itself does not call LLMs).
- Network connectivity between the hosts on the chosen port (default: 7700).

---

## Architecture

```
  Host A (router)             Host B (agent)            Host C (user)
 ┌──────────────────┐       ┌──────────────────┐       ┌─────────────┐
 │  run_router.py   │◄─TCP──│  run_agent.py    │       │ TUI / Web   │
 │  0.0.0.0:7700    │       │  --router-host   │       │ --router-   │
 │  auth enabled    │◄─TCP──│    <host-a-ip>   │◄─TCP──│   host ...  │
 └──────────────────┘       └──────────────────┘       └─────────────┘
```

---

## Security Warning

> **Binding the router to `0.0.0.0` exposes it to every network interface
> on that machine.** Anyone who can reach the port can attempt to connect.
>
> - **Always enable authentication** (`auth_enabled: true` + a strong
>   `auth_token`). Without auth, any client can register as any node.
> - **Use TLS in production.** The transport layer supports TLS on the
>   client side (`--tls` flag). For server-side TLS termination, place
>   the router behind a TLS-terminating reverse proxy (e.g., nginx with
>   `stream { ... }` for raw TCP, or a standard HTTPS proxy for the
>   WebSocket endpoint). The router's built-in TCP server does not
>   perform TLS termination itself.
> - **Firewall:** open only the router port (default 7700 for TCP,
>   8765 for WebSocket) to the specific hosts that need access.

---

## Step-by-Step

### 1. Configure the router on Host A

Create `mesh.yaml` on Host A. The key change from the default config is
binding to `0.0.0.0` instead of `127.0.0.1`:

```yaml
router:
  host: 0.0.0.0        # Listen on all interfaces (default is 127.0.0.1)
  port: 7700
  ws_port: 8765
  storage_path: ~/.mesh/storage/messages.db
  auth_enabled: true
  auth_token: "<your-auth-token>"   # Generate with: python3 -c "import secrets; print(secrets.token_hex(32))"

llm_backends:
  default:
    backend_type: openai
    default_model: gpt-4o

agents:
  agent:assistant:
    nickname: alice
    llm_backend: default
    system_prompt_file: assistant.md
```

Start the router:

```bash
# On Host A
source env.bash
python3 run_router.py --config mesh.yaml
```

Verify it's listening:

```bash
ss -tlnp | grep 7700
# Expected: LISTEN ... 0.0.0.0:7700
```

### 2. Start an agent on Host B

On Host B, you do **not** need a `mesh.yaml` — the agent's CLI flags
override everything. The agent connects to the router on Host A:

```bash
# On Host B
source env.bash
python3 run_agent.py \
    --agent assistant \
    --nickname alice \
    --router-host <host-a-ip> \
    --router-port 7700 \
    --auth-token <your-auth-token>
```

**Verified CLI flags** (from `run_agent.py --help`):

| Flag | Description |
|------|-------------|
| `--agent`, `-a` | Agent type (matches a prompt file in `mesh/prompts/`) |
| `--nickname`, `-n` | Display name for addressing |
| `--router-host` | Router hostname or IP (overrides config) |
| `--router-port` | Router port (overrides config, default: 7700) |
| `--auth-token` | Auth token for router authentication |
| `--tls` | Enable TLS for the router connection |
| `--backend`, `-b` | LLM backend name (overrides config) |
| `--model`, `-m` | Model name within the backend |
| `--controller` | Controller mode: `passthrough`, `task-fsm-v0`, `phase-flow-v02` |
| `--effort` | Reasoning effort for v0.2 controller: `low`, `medium`, `high` |
| `--fresh` | Start without loading prior history |

You should see:

```
Connecting to router at <host-a-ip>:7700...
Connected to <host-a-ip>:7700
Node agent:assistant:alice registered with router
Connected successfully
```

### 3. Connect the TUI from Host C (or Host A)

The TUI accepts the same connection flags:

```bash
# On Host C (or Host A)
python3 run_user_tui.py \
    --nickname yourname \
    --router-host <host-a-ip> \
    --router-port 7700 \
    --auth-token <your-auth-token>
```

**Verified TUI flags** (from `run_user_tui.py` argparse):

| Flag | Description |
|------|-------------|
| `--nickname` | Your display name |
| `--router-host` | Router hostname or IP |
| `--router-port` | Router port (default: 7700) |
| `--auth-token` | Auth token (prompted interactively if omitted) |
| `--tls` | Enable TLS for the connection |

Once connected, type a message — it will be routed through Host A's router
to the agent on Host B, and the agent's response comes back.

### 4. Alternative: config-file approach

Instead of CLI flags, you can create a `mesh.yaml` on Host B that points
at the remote router:

```yaml
router:
  host: <host-a-ip>    # The REMOTE router's address
  port: 7700

agents:
  agent:assistant:
    nickname: alice
    llm_backend: default
    auth_token: "<your-auth-token>"

llm_backends:
  default:
    backend_type: openai
    default_model: gpt-4o
```

Then start with just:

```bash
python3 run_agent.py --config mesh.yaml --agent assistant --nickname alice
```

---

## TLS Setup

The mesh transport supports TLS on the **client** side (agents and TUI
connecting to the router). To enable it:

1. **Set up a TLS-terminating reverse proxy** on Host A. For example,
   with nginx's TCP stream proxy:

   ```nginx
   stream {
       server {
           listen 7701 ssl;
           ssl_certificate     /path/to/cert.pem;
           ssl_certificate_key /path/to/key.pem;
           proxy_pass 127.0.0.1:7700;
       }
   }
   ```

2. **Connect with `--tls`** from Hosts B and C:

   ```bash
   python3 run_agent.py \
       --agent assistant --nickname alice \
       --router-host host-a.example.com \
       --router-port 7701 \
       --auth-token <your-auth-token> \
       --tls
   ```

The `--tls` flag wraps the TCP connection in TLS using Python's
`ssl.create_default_context()` with certificate verification enabled.
The server hostname for certificate validation defaults to the
`--router-host` value; override it with `tls_server_hostname` in the
agent's config if needed.

---

## Firewall Notes

Open the router port(s) on Host A for the specific agent/user hosts:

```bash
# UFW example (Ubuntu/Debian)
sudo ufw allow from <host-b-ip> to any port 7700 proto tcp
sudo ufw allow from <host-c-ip> to any port 7700 proto tcp

# iptables example
sudo iptables -A INPUT -p tcp --dport 7700 -s <host-b-ip> -j ACCEPT
sudo iptables -A INPUT -p tcp --dport 7700 -s <host-c-ip> -j ACCEPT
```

If using the WebSocket endpoint (for the web client), also open port 8765.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `Connection refused` | Router not running or firewall blocking | Check `ss -tlnp \| grep 7700` on Host A; check firewall rules |
| `Authentication failed` | Token mismatch | Verify the `--auth-token` matches the router's `auth_token` in `mesh.yaml` |
| Agent connects but no messages arrive | Wrong `to_node` format | Messages must target `agent:<type>:<nickname>` (e.g., `agent:assistant:alice`) |
| `SSL: CERTIFICATE_VERIFY_FAILED` | Self-signed cert or wrong hostname | Use a valid cert, or set `tls_server_hostname` in config to match the cert's CN/SAN |
