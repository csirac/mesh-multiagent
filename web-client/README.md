# Mesh Web Client

A browser-based client for the mesh network.

## Quick Start

```bash
# Set auth token (required)
export MESH_AUTH_TOKEN=your_token_here

# Optional: customize settings
export MESH_ROUTER_URL=ws://localhost:8765
export MESH_NODE_ID=user:yourname

# Start the server
python serve.py

# Open http://localhost:5000 in your browser
```

## Direct Usage (without serve.py)

You can also open `index.html` directly in a browser:

1. Open `index.html` in your browser
2. Click the settings gear icon
3. Enter:
   - Server URL: `ws://your-router-host:8765`
   - Node ID: `user:yourname`
   - Auth Token: your auth token
4. Click Connect

## Features

- **Real-time messaging** via WebSocket
- **Markdown rendering** (Marked.js) with **math support** (MathJax)
- **Conversations, Channels, Roster** tabs
- **Message context menu** (right-click): Copy, Copy code, Delete
- **Conversation context menu**: Delete conversation
- **Dark/Light theme** toggle
- **Auto-reconnect** with exponential backoff
- **Local storage** for offline message caching

## Configuration Options

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `MESH_ROUTER_URL` | `ws://localhost:8765` | WebSocket URL of the mesh router |
| `MESH_NODE_ID` | `user:web` | Your node ID in the mesh |
| `MESH_AUTH_TOKEN` | (none) | Authentication token |

Command line arguments override environment variables:

```bash
python serve.py --router-url ws://mesh.example.com:8765 --node-id user:alice --auth-token xxx
```

## Development

The client is a single HTML file with embedded CSS and JavaScript. No build step required.

Key libraries (loaded from CDN):
- [Marked.js](https://marked.js.org/) - Markdown parsing
- [MathJax](https://www.mathjax.org/) - LaTeX math rendering
