#!/usr/bin/env python3
"""
Simple Flask server for the Mesh Web Client.

Serves the web client and can inject auth tokens for convenience.
Run this server, then open http://localhost:5000 in your browser.
"""

import os
import argparse
from flask import Flask, send_file, jsonify

app = Flask(__name__)

# Configuration
MESH_ROUTER_URL = os.environ.get("MESH_ROUTER_URL", "ws://localhost:8765")
MESH_AUTH_TOKEN = os.environ.get("MESH_AUTH_TOKEN", "")
# Node ID is now set by the user in the login screen, not by the server


@app.route("/")
def index():
    """Serve the main web client."""
    return send_file("index.html")


@app.route("/config")
def config():
    """Return client configuration.

    The web client can fetch this on load to auto-configure.
    Auth token is NOT returned - users must enter it manually for security.
    Node ID is set by the user via the login screen.
    """
    return jsonify({
        "serverUrl": MESH_ROUTER_URL,
        # Don't expose auth token - require manual entry
    })


def main():
    parser = argparse.ArgumentParser(description="Mesh Web Client Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=5000, help="Port to listen on")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--router-url", help="Mesh router WebSocket URL")
    parser.add_argument("--auth-token", help="Auth token for mesh")
    args = parser.parse_args()

    global MESH_ROUTER_URL, MESH_AUTH_TOKEN

    if args.router_url:
        MESH_ROUTER_URL = args.router_url
    if args.auth_token:
        MESH_AUTH_TOKEN = args.auth_token

    print(f"Starting Mesh Web Client Server")
    print(f"  Router URL: {MESH_ROUTER_URL}")
    print(f"  Auth Token: {'***' if MESH_AUTH_TOKEN else '(not set)'}")
    print(f"  Open http://{args.host if args.host != '0.0.0.0' else 'localhost'}:{args.port}")

    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
