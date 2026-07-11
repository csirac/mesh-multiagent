# SPDX-License-Identifier: Apache-2.0
"""
Central router/broker for the mesh.

Accepts connections from nodes, maintains a registry of connected nodes,
and routes messages between them. Persists undelivered messages.

Supports both:
- TCP with length-prefixed framing (for Python nodes)
- WebSocket (for browser/mobile clients)
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import hmac
import json
import logging
import mimetypes
import os
import time
import urllib.parse
from pathlib import Path
from typing import Any, Union

from .protocol import (
    Attachment,
    Message,
    MessageType,
    ControlAction,
    make_control,
    parse_node_id,
    make_presence,
    is_channel_address,
    parse_channel_name,
    make_history_response,
    make_scratchpad_response,
    make_todo_response,
    make_calendar_response,
)
from dataclasses import asdict
from .transport import Server, Connection, WebSocketConnection
from .storage import MessageStore
from .config import RouterConfig
from .fcm import FCMSender
from .cc_usage import fetch_all_usage_async, format_usage_summary, CCUsageResult

# Type alias for either connection type
AnyConnection = Union[Connection, WebSocketConnection]

logger = logging.getLogger(__name__)


class Router:
    """
    Central message broker for the mesh.

    Responsibilities:
    - Accept node connections
    - Maintain registry of connected nodes
    - Route messages between nodes
    - Persist messages for offline nodes
    - Handle control messages (register, list_nodes, etc.)

    Supports multiple simultaneous connections per node (e.g., same user on TUI + mobile).
    """

    def __init__(self, config: RouterConfig):
        self.config = config
        self.store = MessageStore(config.storage_path)
        # node_id -> list of connections (supports multiple connections per node)
        self._connections: dict[str, list[AnyConnection]] = {}
        self._node_metadata: dict[str, dict[str, Any]] = {}  # node_id -> {description, ...}
        self._node_status: dict[str, dict] = {}  # node_id -> heartbeat-lite status summary
        self._calendar_lock = asyncio.Lock()
        self._server: Server | None = None
        self._ws_runner: Any = None  # aiohttp.web.AppRunner
        self._running = False

        # Initialize FCM sender if enabled
        self._fcm: FCMSender | None = None
        if config.fcm_enabled and config.fcm_credentials_file:
            self._fcm = FCMSender(config.fcm_credentials_file)
            if self._fcm.is_available:
                logger.info("FCM push notifications enabled")
            else:
                logger.warning("FCM enabled in config but failed to initialize")
                self._fcm = None

        # CC Usage monitor (refreshes every 5 minutes)
        self._cc_usage_cache: list[CCUsageResult] = []
        self._cc_usage_cache_time: float = 0
        self._cc_usage_refresh_task: asyncio.Task | None = None
        self._cc_usage_home_dirs: list[tuple[str, str]] = []  # (label, home_dir)

    @property
    def connected_nodes(self) -> list[str]:
        """List of currently connected node IDs."""
        return list(self._connections.keys())

    async def start(self) -> None:
        """Start the router (TCP and optionally WebSocket)."""
        # Start TCP server
        self._server = Server(
            self.config.host,
            self.config.port,
            self._handle_connection,
        )
        await self._server.start()
        self._running = True
        logger.info(f"Router TCP started on {self.config.host}:{self.config.port}")

        # Start WebSocket server if enabled
        if self.config.ws_enabled:
            await self._start_websocket_server()
            logger.info(f"Router WebSocket started on {self.config.host}:{self.config.ws_port}/ws")

        # Start CC usage monitor
        self._init_cc_usage_paths()
        self._cc_usage_refresh_task = asyncio.create_task(self._cc_usage_refresh_loop())
        logger.info("CC usage monitor started (15 min refresh)")

    async def stop(self) -> None:
        """Stop the router."""
        self._running = False
        # Cancel CC usage monitor
        if self._cc_usage_refresh_task:
            self._cc_usage_refresh_task.cancel()
            try:
                await self._cc_usage_refresh_task
            except asyncio.CancelledError:
                pass
            self._cc_usage_refresh_task = None
        # Close all connections
        for conn_list in list(self._connections.values()):
            for conn in conn_list:
                await conn.close()
        self._connections.clear()
        if self._server:
            await self._server.stop()
        # Stop WebSocket server
        if self._ws_runner:
            await self._ws_runner.cleanup()
            self._ws_runner = None
        logger.info("Router stopped")

    async def run_forever(self) -> None:
        """Run the router until cancelled."""
        await self.start()
        if self._server:
            try:
                await self._server.serve_forever()
            except asyncio.CancelledError:
                pass
            finally:
                await self.stop()

    def _init_cc_usage_paths(self) -> None:
        """Initialize CC usage home dirs by auto-discovering account directories.

        Scans for ~/.claude-acct2, ~/.claude-acct3, etc. in addition to the
        default ~/.claude account.  Falls back to mesh.yaml cc_fallback_homes
        if auto-discovery finds nothing.
        """
        import os
        import pwd
        from pathlib import Path

        # Use /etc/passwd home, not $HOME (which may be overridden for CC multi-account)
        real_home = Path(pwd.getpwuid(os.getuid()).pw_dir)
        self._cc_usage_home_dirs = [("default", str(real_home))]

        # Auto-discover ~/.claude-acct2, ~/.claude-acct3, ...
        n = 2
        while True:
            acct_dir = real_home / f".claude-acct{n}"
            if not acct_dir.exists():
                break
            self._cc_usage_home_dirs.append((f"acct{n}", str(acct_dir)))
            n += 1

        # If auto-discovery found nothing, fall back to mesh.yaml config
        if len(self._cc_usage_home_dirs) == 1:
            try:
                from .config import MeshConfig
                config_path = Path(__file__).parent.parent / "mesh.yaml"
                if config_path.exists():
                    mesh_config = MeshConfig.load(config_path)
                    seen_homes = {str(real_home)}
                    account_num = 2
                    for backend in mesh_config.llm_backends.values():
                        for home in backend.cc_fallback_homes:
                            expanded = str(Path(home).expanduser())
                            if expanded not in seen_homes:
                                seen_homes.add(expanded)
                                self._cc_usage_home_dirs.append((f"acct{account_num}", expanded))
                                account_num += 1
            except Exception as e:
                logger.warning(f"Failed to load cc_fallback_homes from config: {e}")

    async def _cc_usage_refresh_loop(self) -> None:
        """Background task to refresh CC usage every 15 minutes."""
        while self._running:
            try:
                results = await fetch_all_usage_async(self._cc_usage_home_dirs)
                self._cc_usage_cache = results
                self._cc_usage_cache_time = time.time()
                logger.debug(f"CC usage refreshed for {len(results)} accounts")
            except Exception as e:
                logger.warning(f"CC usage refresh failed: {e}")

            # Wait 15 minutes before next refresh
            try:
                await asyncio.sleep(900)
            except asyncio.CancelledError:
                break

    def get_cc_usage_summary(self) -> str:
        """Get cached CC usage as a compact summary string."""
        if not self._cc_usage_cache:
            return "CC Usage: (refreshing...)"
        return "CC Usage: " + format_usage_summary(self._cc_usage_cache)

    async def _start_websocket_server(self) -> None:
        """Start the aiohttp WebSocket server."""
        import aiohttp.web

        app = aiohttp.web.Application()
        app.router.add_get("/ws", self._handle_websocket)
        # Simple health check endpoint
        app.router.add_get("/health", self._handle_health)
        # Plaid OAuth callback endpoint
        app.router.add_get("/plaid/callback", self._handle_plaid_callback)
        # Attachment upload/download endpoints
        app.router.add_post("/attachments", self._handle_attachment_upload)
        app.router.add_options("/attachments", self._handle_attachment_options)
        app.router.add_get("/attachments/{att_id}", self._handle_attachment_download)
        app.router.add_options("/attachments/{att_id}", self._handle_attachment_options)
        app.router.add_get("/attachments/{att_id}/url", self._handle_attachment_url_refresh)
        app.router.add_options("/attachments/{att_id}/url", self._handle_attachment_options)

        self._ws_runner = aiohttp.web.AppRunner(app)
        await self._ws_runner.setup()
        site = aiohttp.web.TCPSite(
            self._ws_runner,
            self.config.host,
            self.config.ws_port,
        )
        await site.start()

    async def _handle_health(self, request: Any) -> Any:
        """Health check endpoint."""
        import aiohttp.web

        return aiohttp.web.json_response({
            "status": "ok",
            "connected_nodes": len(self._connections),
        })

    async def _handle_plaid_callback(self, request: Any) -> Any:
        """
        Handle Plaid OAuth callback.

        Plaid redirects here after the user completes Link authentication.
        The URL includes public_token, institution metadata, etc.
        """
        import aiohttp.web

        # Extract query parameters from Plaid redirect
        public_token = request.query.get("public_token")
        institution_id = request.query.get("institution_id", "")
        institution_name = request.query.get("institution_name", "Unknown Bank")

        if not public_token:
            # Plaid Link sends the token in the URL fragment for browser-based flows
            # For server-side, it may come as a query param or need JS handling
            return aiohttp.web.Response(
                text="""
                <!DOCTYPE html>
                <html>
                <head><title>Plaid Link Complete</title></head>
                <body>
                    <h1>Bank Connection Status</h1>
                    <p id="status">Processing...</p>
                    <script>
                        // For OAuth flows, Plaid sends params in fragment
                        const params = new URLSearchParams(window.location.hash.substring(1));
                        const publicToken = params.get('public_token');

                        if (publicToken) {
                            // Redirect with token as query param for server processing
                            window.location.href = '/plaid/callback?public_token=' + encodeURIComponent(publicToken)
                                + '&institution_id=' + encodeURIComponent(params.get('institution_id') || '')
                                + '&institution_name=' + encodeURIComponent(params.get('institution_name') || 'Bank');
                        } else if (!window.location.search.includes('public_token')) {
                            document.getElementById('status').textContent = 'No public token received. Please try again.';
                        }
                    </script>
                </body>
                </html>
                """,
                content_type="text/html",
            )

        # Exchange public token for access token
        try:
            from .clients.plaid_client import PlaidClient

            client = PlaidClient(user_id=os.environ.get("PLAID_USER_ID", "default"))
            result = client.exchange_public_token(
                public_token=public_token,
                institution_id=institution_id,
                institution_name=institution_name,
            )

            if "error" in result:
                return aiohttp.web.Response(
                    text=f"""
                    <!DOCTYPE html>
                    <html>
                    <head><title>Connection Failed</title></head>
                    <body>
                        <h1>Bank Connection Failed</h1>
                        <p>Error: {result['error']}</p>
                        <p><a href="/">Return Home</a></p>
                    </body>
                    </html>
                    """,
                    content_type="text/html",
                )

            # Success!
            return aiohttp.web.Response(
                text=f"""
                <!DOCTYPE html>
                <html>
                <head><title>Bank Connected</title></head>
                <body>
                    <h1>Bank Account Connected!</h1>
                    <p>Successfully linked: <strong>{institution_name}</strong></p>
                    <p>You can now use <code>plaid_sync</code> and <code>plaid_transactions</code> to access your data.</p>
                    <p>This window can be closed.</p>
                </body>
                </html>
                """,
                content_type="text/html",
            )

        except Exception as e:
            logger.error(f"Plaid callback error: {e}")
            return aiohttp.web.Response(
                text=f"""
                <!DOCTYPE html>
                <html>
                <head><title>Error</title></head>
                <body>
                    <h1>Error Processing Callback</h1>
                    <p>{str(e)}</p>
                </body>
                </html>
                """,
                content_type="text/html",
                status=500,
            )

    # =========================================================================
    # Attachment HTTP endpoints
    # =========================================================================

    def _cors_headers(self, request: Any) -> dict[str, str]:
        origin = request.headers.get("Origin", "*")
        return {
            "Access-Control-Allow-Origin": origin,
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "Authorization, X-Node-ID, X-Filename, Content-Type",
            "Access-Control-Max-Age": "600",
            "Vary": "Origin",
        }

    async def _handle_attachment_options(self, request: Any) -> Any:
        import aiohttp.web

        return aiohttp.web.Response(status=200, headers=self._cors_headers(request))

    def _attachment_secret(self) -> bytes | None:
        secret = self.config.attachments_signing_secret
        if not self.config.attachments_enabled or not secret:
            return None
        return secret.encode("utf-8")

    def _attachment_http_base(self) -> str:
        host = self.config.host
        if host in ("0.0.0.0", "::"):
            host = "localhost"
        return f"http://{host}:{self.config.ws_port}"

    def _sign_attachment_url(
        self,
        att_id: str,
        for_node: str,
        exp: int | None = None,
    ) -> tuple[str, int]:
        secret = self._attachment_secret()
        if not secret:
            raise RuntimeError("attachments require MESH_ATTACHMENT_SECRET")
        exp = exp or int(time.time()) + int(self.config.attachments_url_ttl_secs)
        payload = f"{att_id}|{exp}|{for_node}".encode("utf-8")
        sig = hmac.new(secret, payload, hashlib.sha256).hexdigest()[:32]
        query = urllib.parse.urlencode({"exp": exp, "node": for_node, "sig": sig})
        return f"{self._attachment_http_base()}/attachments/{urllib.parse.quote(att_id)}?{query}", exp

    def _verify_attachment_signature(self, att_id: str, exp: str, node_id: str, sig: str) -> tuple[bool, int]:
        secret = self._attachment_secret()
        if not secret:
            return False, 0
        try:
            exp_i = int(exp)
        except (TypeError, ValueError):
            return False, 0
        if exp_i <= int(time.time()):
            return False, exp_i
        max_exp = int(time.time()) + int(self.config.attachments_url_ttl_secs) + 5
        if exp_i > max_exp:
            return False, exp_i
        payload = f"{att_id}|{exp_i}|{node_id}".encode("utf-8")
        expected = hmac.new(secret, payload, hashlib.sha256).hexdigest()[:32]
        return hmac.compare_digest(expected, sig or ""), exp_i

    @staticmethod
    def _sanitize_filename(name: str, fallback: str = "attachment.bin") -> str:
        import re
        import unicodedata

        name = urllib.parse.unquote(name or "")
        name = unicodedata.normalize("NFC", name).replace("\x00", "")
        name = name.replace("\\", "/").split("/")[-1]
        name = name.replace("..", "")
        name = re.sub(r"[\r\n\t]", " ", name).strip().strip(".")
        if not name:
            name = fallback
        return name[:255]

    @staticmethod
    def _attachment_id_for_sha(sha256: str) -> str:
        return f"att_{sha256[:32]}"

    def _mime_for_upload(self, path: str, filename: str, client_mime: str | None) -> str:
        guessed, _ = mimetypes.guess_type(filename)
        if guessed:
            return guessed
        if client_mime and "/" in client_mime:
            return client_mime.split(";", 1)[0].strip()
        return "application/octet-stream"

    def _canonicalize_message_attachments(self, msg: Message) -> Message:
        if not msg.attachments:
            return msg
        canonical: list[Attachment] = []
        for attachment in msg.attachments:
            blob = self.store.get_blob(attachment.id)
            if blob:
                canonical.append(Attachment(
                    id=blob.id,
                    name=self._sanitize_filename(attachment.name or blob.id, fallback=f"{blob.id}.bin"),
                    size=blob.size,
                    mime=attachment.mime or blob.mime_inferred,
                    sha256=blob.sha256,
                    url=None,
                ))
            else:
                canonical.append(attachment.canonical())
        msg.attachments = canonical
        return msg

    def _message_with_signed_attachments(self, msg: Message, recipient_node: str) -> Message:
        clone = copy.deepcopy(msg)
        signed = []
        for attachment in clone.attachments:
            canonical = attachment.canonical()
            try:
                url, _exp = self._sign_attachment_url(canonical.id, recipient_node)
                canonical.url = url
            except RuntimeError:
                canonical.url = None
            signed.append(canonical)
        clone.attachments = signed
        return clone

    def _attachment_error(self, request: Any, status: int, message: str) -> Any:
        import aiohttp.web

        return aiohttp.web.json_response(
            {"error": message},
            status=status,
            headers=self._cors_headers(request),
        )

    async def _handle_attachment_upload(self, request: Any) -> Any:
        import aiohttp.web
        from tempfile import NamedTemporaryFile

        if not self.config.attachments_enabled:
            return self._attachment_error(request, 404, "attachments disabled")
        if not self._attachment_secret():
            return self._attachment_error(request, 503, "attachments require MESH_ATTACHMENT_SECRET")

        authz = request.headers.get("Authorization", "")
        token = authz.removeprefix("Bearer ").strip() if authz.startswith("Bearer ") else ""
        node_id = request.headers.get("X-Node-ID", "").strip()
        if not node_id:
            return self._attachment_error(request, 400, "X-Node-ID required")
        ok, resolved_node = self._authenticate_token(token, node_id)
        if not ok or resolved_node != node_id:
            return self._attachment_error(request, 403, "invalid token for node")

        filename = self._sanitize_filename(
            request.headers.get("X-Filename", ""),
            fallback="attachment.bin",
        )
        attachments_dir = Path(self.config.attachments_dir)
        attachments_dir.mkdir(parents=True, exist_ok=True)

        max_bytes = int(self.config.attachments_max_file_bytes)
        quota = int(self.config.attachments_per_owner_quota_bytes)
        hasher = hashlib.sha256()
        total = 0
        temp_path = None
        try:
            with NamedTemporaryFile("wb", delete=False, dir=str(attachments_dir)) as tmp:
                temp_path = tmp.name
                async for chunk in request.content.iter_chunked(1024 * 1024):
                    total += len(chunk)
                    if total > max_bytes:
                        return self._attachment_error(request, 413, "attachment too large")
                    hasher.update(chunk)
                    tmp.write(chunk)
            sha256 = hasher.hexdigest()
            att_id = self._attachment_id_for_sha(sha256)
            existing_blob = self.store.get_blob(att_id)
            if not existing_blob and self.store.blob_owner_total_bytes(node_id) + total > quota:
                return self._attachment_error(request, 413, "attachment quota exceeded")
            shard_dir = attachments_dir / sha256[:2]
            shard_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
            final_path = shard_dir / sha256
            if not final_path.exists():
                os.replace(temp_path, final_path)
                temp_path = None
                try:
                    os.chmod(final_path, 0o600)
                except OSError:
                    pass
            mime = self._mime_for_upload(str(final_path), filename, request.headers.get("Content-Type"))
            meta_path = final_path.with_suffix(final_path.suffix + ".meta.json")
            if not meta_path.exists():
                meta_path.write_text(json.dumps({
                    "sha256": sha256,
                    "mime_inferred": mime,
                    "size": total,
                    "owner_node": node_id,
                    "refs": [],
                }))
            self.store.register_blob(att_id, sha256, total, str(final_path), mime, node_id)
            return aiohttp.web.json_response(
                {
                    "id": att_id,
                    "sha256": sha256,
                    "size": total,
                    "mime": mime,
                    "name": filename,
                },
                headers=self._cors_headers(request),
            )
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass

    async def _handle_attachment_download(self, request: Any) -> Any:
        import aiohttp.web

        att_id = request.match_info.get("att_id", "")
        exp = request.query.get("exp", "")
        node_id = request.query.get("node", "")
        sig = request.query.get("sig", "")
        ok, exp_i = self._verify_attachment_signature(att_id, exp, node_id, sig)
        if not ok:
            return self._attachment_error(request, 410 if exp_i and exp_i <= int(time.time()) else 403, "invalid or expired attachment URL")

        blob = self.store.get_blob(att_id)
        if not blob or not Path(blob.path).exists():
            return self._attachment_error(request, 404, "attachment not found")
        self.store.bump_blob_access(att_id)

        headers = self._cors_headers(request)
        headers["X-Content-Type-Options"] = "nosniff"
        safe_inline = blob.mime_inferred.startswith("image/") or blob.mime_inferred == "application/pdf"
        if not safe_inline:
            headers["Content-Disposition"] = f'attachment; filename="{att_id}.bin"'
        return aiohttp.web.FileResponse(
            blob.path,
            headers=headers,
            reason=None,
        )

    async def _handle_attachment_url_refresh(self, request: Any) -> Any:
        import aiohttp.web

        authz = request.headers.get("Authorization", "")
        token = authz.removeprefix("Bearer ").strip() if authz.startswith("Bearer ") else ""
        node_id = request.headers.get("X-Node-ID", "").strip()
        if not node_id:
            return self._attachment_error(request, 400, "X-Node-ID required")
        ok, resolved_node = self._authenticate_token(token, node_id)
        if not ok or resolved_node != node_id:
            return self._attachment_error(request, 401, "invalid token for node")
        att_id = request.match_info.get("att_id", "")
        if not self.store.node_can_access_blob(node_id, att_id):
            return self._attachment_error(request, 403, "attachment not visible to node")
        try:
            url, exp = self._sign_attachment_url(att_id, node_id)
        except RuntimeError:
            return self._attachment_error(request, 503, "attachments require MESH_ATTACHMENT_SECRET")
        return aiohttp.web.json_response(
            {"url": url, "exp": exp},
            headers=self._cors_headers(request),
        )

    async def _handle_websocket(self, request: Any) -> Any:
        """Handle a WebSocket connection."""
        import aiohttp.web

        ws = aiohttp.web.WebSocketResponse()
        await ws.prepare(request)

        # Get remote address
        peername = request.transport.get_extra_info("peername")
        remote_addr = f"{peername[0]}:{peername[1]}" if peername else "unknown"

        conn = WebSocketConnection(ws, remote_address=remote_addr)
        logger.info(f"New WebSocket connection from {remote_addr}")

        try:
            await self._handle_connection(conn)
        except Exception as e:
            logger.error(f"Error handling WebSocket connection: {e}")
        finally:
            await conn.close()

        return ws

    def _get_expected_token(self, node_id: str) -> str | None:
        """Get the expected auth token for a node (for global/per-node modes)."""
        # Check per-node tokens first
        if self.config.auth_tokens and node_id in self.config.auth_tokens:
            return self.config.auth_tokens[node_id]
        # Fall back to global token
        return self.config.auth_token

    def _authenticate_token(self, token: str | None, claimed_node: str | None) -> tuple[bool, str | None]:
        """Validate a bearer/auth token for a claimed node id."""
        claimed_node = claimed_node or ""
        if not self.config.auth_enabled:
            return True, claimed_node

        if self.config.auth_mode == "per_user":
            if not token:
                return False, None
            result = self._validate_per_user_token(token)
            if not result:
                return False, None
            _username, allowed_prefixes = result
            if allowed_prefixes and not any(claimed_node.startswith(p) for p in allowed_prefixes):
                return False, None
            return True, claimed_node

        expected_token = self._get_expected_token(claimed_node)
        if not expected_token or token != expected_token:
            return False, None
        return True, claimed_node

    def _validate_per_user_token(self, token: str) -> tuple[str, list[str] | None] | None:
        """
        Validate a token against the users table.

        Returns (username, allowed_prefixes) if valid, None otherwise.
        """
        return self.store.validate_user_token(token)

    async def _send_to_node(self, node_id: str, msg: Message) -> None:
        """Send a message to all connections of a node."""
        conn_list = self._connections.get(node_id, [])
        for conn in conn_list:
            try:
                await conn.send(msg)
            except Exception as e:
                logger.error(f"Failed to send to {node_id}: {e}")

    async def _send_auth_error(self, conn: AnyConnection, node_id: str, reason: str) -> None:
        """Send an authentication error response."""
        error_msg = Message(
            from_node="router",
            to_node=node_id,
            type=MessageType.CONTROL,
            content={
                "action": ControlAction.ACK.value,
                "status": "error",
                "error": f"authentication failed: {reason}",
            },
        )
        try:
            await conn.send(error_msg)
        except Exception:
            pass  # Best effort

    async def _send_current_roster(self, new_node_id: str, conn: AnyConnection) -> None:
        """
        Send PRESENCE (join) events for all currently connected nodes to a new joiner.

        This lets the new node populate its roster immediately without waiting
        for other nodes to join/leave.
        """
        for existing_id, conn_list in self._connections.items():
            if existing_id == new_node_id:
                continue  # Skip self
            if not conn_list:
                continue  # No active connections

            # Parse the existing node ID to extract info
            node_type, type_or_nick, nickname = parse_node_id(existing_id)

            if node_type == "user":
                nick = type_or_nick
                ntype = "user"
            elif node_type == "agent":
                nick = nickname if nickname else type_or_nick
                ntype = type_or_nick
            else:
                nick = existing_id
                ntype = "unknown"

            # Get metadata (description, backend info, hostname)
            metadata = self._node_metadata.get(existing_id, {})
            description = metadata.get("description", "")
            llm_backend = metadata.get("llm_backend", "")
            llm_model = metadata.get("llm_model", "")
            hostname = metadata.get("hostname", "")

            presence_msg = make_presence(
                from_node=existing_id,
                event="join",
                nickname=nick,
                node_type=ntype,
                description=description,
                llm_backend=llm_backend,
                llm_model=llm_model,
                hostname=hostname,
            )
            # Override to_node to send directly to the new joiner (not broadcast)
            presence_msg.to_node = new_node_id

            try:
                await conn.send(presence_msg)
            except Exception as e:
                logger.error(f"Failed to send roster entry {existing_id} to {new_node_id}: {e}")

    async def _handle_connection(self, conn: AnyConnection) -> None:
        """Handle a new node connection."""
        # First message must be a REGISTER control message
        msg = await conn.receive()
        if msg is None:
            return

        if msg.type != MessageType.CONTROL:
            logger.warning(f"Expected CONTROL message, got {msg.type}")
            await conn.close()
            return

        content = msg.content if isinstance(msg.content, dict) else {}
        action = content.get("action")

        if action != ControlAction.REGISTER.value:
            logger.warning(f"Expected REGISTER action, got {action}")
            await conn.close()
            return

        node_id = msg.from_node
        conn.node_id = node_id

        # Verify authentication if enabled
        if self.config.auth_enabled:
            provided_token = content.get("auth_token")
            ok, _resolved_node = self._authenticate_token(provided_token, node_id)
            if not ok:
                logger.warning(f"Invalid auth token for node {node_id}")
                await self._send_auth_error(conn, node_id, "invalid auth token")
                await conn.close()
                return
            logger.info(f"Node {node_id} authenticated successfully")

        # Extract optional metadata from registration
        description = content.get("description", "")
        llm_backend = content.get("llm_backend", "")
        llm_model = content.get("llm_model", "")
        hostname = content.get("hostname", "")
        router_v2_llm_backend = content.get("router_v2_llm_backend", "")
        router_v2_llm_model = content.get("router_v2_llm_model", "")
        harness_session_backend = content.get("harness_session_backend", "")
        cc_interactive_tools = content.get("cc_interactive_tools", False)
        cc_interactive_model = content.get("cc_interactive_model", "")
        cc_interactive_binary = content.get("cc_interactive_binary", "")
        cc_interactive_effort = content.get("cc_interactive_effort", "")

        # Enforce single-connection-per-nickname for agents
        # This prevents confusion when the same nickname (e.g., "alice") connects
        # with different node_ids (e.g., agent:researcher:alice vs agent:assistant:alice)
        node_type, type_or_nick, nickname = parse_node_id(node_id)
        if node_type == "agent":
            nick = nickname if nickname else type_or_nick
            # Check if another node with same nickname is already connected
            for existing_id, conn_list in list(self._connections.items()):
                if not conn_list:
                    continue
                if existing_id == node_id:
                    continue  # Same node_id is fine (multi-connection support)
                ex_type, ex_type_or_nick, ex_nickname = parse_node_id(existing_id)
                if ex_type == "agent":
                    ex_nick = ex_nickname if ex_nickname else ex_type_or_nick
                    if ex_nick == nick:
                        # Disconnect the old node
                        logger.warning(
                            f"Nickname conflict: {node_id} replacing {existing_id} "
                            f"(same nickname '{nick}')"
                        )
                        # Close all connections for the old node
                        for old_conn in conn_list:
                            await old_conn.close()
                        del self._connections[existing_id]
                        # Broadcast leave for old node
                        await self._broadcast_presence(existing_id, "leave")
                        self._node_metadata.pop(existing_id, None)
                        break

        # Track whether this is a new node (first connection) or additional connection
        is_new_node = node_id not in self._connections or not self._connections[node_id]

        # Add connection to the list (supports multiple connections per node)
        if node_id not in self._connections:
            self._connections[node_id] = []
        self._connections[node_id].append(conn)
        self._node_metadata[node_id] = {
            "description": description,
            "llm_backend": llm_backend,
            "llm_model": llm_model,
            "hostname": hostname,
            "router_v2_llm_backend": router_v2_llm_backend,
            "router_v2_llm_model": router_v2_llm_model,
            "harness_session_backend": harness_session_backend,
            "cc_interactive_tools": cc_interactive_tools,
            "cc_interactive_model": cc_interactive_model,
            "cc_interactive_binary": cc_interactive_binary,
            "cc_interactive_effort": cc_interactive_effort,
        }

        conn_count = len(self._connections[node_id])
        if is_new_node:
            logger.info(f"Node registered: {node_id}" + (f" ({description})" if description else ""))
        else:
            logger.info(f"Node {node_id} added connection #{conn_count}")

        # Send ACK
        ack = Message(
            from_node="router",
            to_node=node_id,
            type=MessageType.CONTROL,
            content={"action": ControlAction.ACK.value, "status": "registered"},
        )
        await conn.send(ack)

        # Send current roster to new node (PRESENCE for each existing node)
        await self._send_current_roster(node_id, conn)

        # Broadcast presence (join) only for the first connection of this node
        if is_new_node:
            await self._broadcast_presence(node_id, "join")

        # Deliver any pending messages
        await self._deliver_pending(node_id, conn)

        # Message loop
        await self._message_loop(conn)

    async def _message_loop(self, conn: AnyConnection) -> None:
        """Process messages from a connected node."""
        node_id = conn.node_id
        try:
            while self._running and not conn.is_closed:
                msg = await conn.receive()
                if msg is None:
                    break
                await self._route_message(msg, source_conn=conn)
        except Exception as e:
            logger.error(f"Error in message loop for {node_id}: {e}")
        finally:
            # Cleanup on disconnect - remove this specific connection
            if node_id and node_id in self._connections:
                conn_list = self._connections[node_id]
                if conn in conn_list:
                    conn_list.remove(conn)

                if conn_list:
                    # Still have other connections for this node
                    logger.info(f"Node {node_id} connection closed ({len(conn_list)} remaining)")
                else:
                    # Last connection closed - node is fully offline
                    del self._connections[node_id]
                    logger.info(f"Node disconnected: {node_id}")
                    # Broadcast presence (leave) before cleaning up metadata
                    await self._broadcast_presence(node_id, "leave")
                    # Clean up metadata and status
                    self._node_metadata.pop(node_id, None)
                    self._node_status.pop(node_id, None)

    async def _route_message(self, msg: Message, source_conn: AnyConnection | None = None) -> None:
        """Route a message to its destination."""
        msg = self._canonicalize_message_attachments(msg)
        # Handle control messages to router
        if msg.to_node == "router":
            await self._handle_control(msg)
            return

        # Handle broadcast messages
        if msg.to_node == "broadcast":
            await self._broadcast_to_all(msg, exclude=msg.from_node)
            return

        # Archive messages for sync (MESSAGE, TOOL_REQUEST, TOOL_RESULT)
        if msg.type in (MessageType.MESSAGE, MessageType.TOOL_REQUEST, MessageType.TOOL_RESULT):
            self.store.archive_message(msg)
            # Echo to sender's other connections (multi-device sync)
            if msg.type == MessageType.MESSAGE:
                await self._echo_to_sender(msg, source_conn=source_conn)

        # Handle channel messages
        if is_channel_address(msg.to_node):
            await self._route_to_channel(msg)
            return

        target = msg.to_node
        conn_list = self._connections.get(target, [])
        if conn_list:
            # Node is online, deliver to all connections
            delivered = False
            for conn in list(conn_list):  # Copy list in case of modification
                try:
                    await conn.send(self._message_with_signed_attachments(msg, target))
                    delivered = True
                except Exception as e:
                    logger.error(f"Failed to deliver to {target} (one connection): {e}")
            if delivered:
                logger.debug(f"Delivered {msg.id} to {target} ({len(conn_list)} connections)")
            else:
                # All connections failed
                self.store.store(msg)
                await self._send_push_notification(target, msg)
        else:
            # Node offline, persist for later
            logger.debug(f"Node {target} offline, storing message {msg.id}")
            self.store.store(msg)
            await self._send_push_notification(target, msg)

    async def _handle_control(self, msg: Message) -> None:
        """Handle control messages directed at the router."""
        content = msg.content if isinstance(msg.content, dict) else {}
        action = content.get("action")
        from_node = msg.from_node

        if action == ControlAction.LIST_NODES.value:
            # Include heartbeat-lite status and registration metadata
            node_status = {}
            node_meta = {}
            for node_id in self.connected_nodes:
                if node_id in self._node_status:
                    node_status[node_id] = self._node_status[node_id]
                if node_id in self._node_metadata:
                    node_meta[node_id] = self._node_metadata[node_id]
            response = Message(
                from_node="router",
                to_node=from_node,
                type=MessageType.CONTROL,
                content={
                    "action": ControlAction.LIST_NODES.value,
                    "nodes": self.connected_nodes,
                    "status": node_status,
                    "metadata": node_meta,
                    "cc_usage": self.get_cc_usage_summary(),
                },
                in_reply_to=msg.id,
            )
            await self._send_to_node(from_node, response)

        elif action == ControlAction.STATUS.value:
            response = Message(
                from_node="router",
                to_node=from_node,
                type=MessageType.CONTROL,
                content={
                    "action": ControlAction.STATUS.value,
                    "connected_nodes": len(self._connections),
                    "pending_messages": self.store.count_pending(),
                },
                in_reply_to=msg.id,
            )
            await self._send_to_node(from_node, response)

        elif action == ControlAction.REGISTER_PUSH_TOKEN.value:
            fcm_token = content.get("fcm_token")
            if fcm_token:
                self.store.set_fcm_token(from_node, fcm_token)
                logger.info(f"Registered FCM token for {from_node}")
                # Send ACK
                response = Message(
                    from_node="router",
                    to_node=from_node,
                    type=MessageType.CONTROL,
                    content={
                        "action": ControlAction.ACK.value,
                        "status": "push_token_registered",
                    },
                    in_reply_to=msg.id,
                )
                await self._send_to_node(from_node, response)
            else:
                logger.warning(f"REGISTER_PUSH_TOKEN from {from_node} missing fcm_token")

        # Channel operations
        elif action == ControlAction.CHANNEL_CREATE.value:
            await self._handle_channel_create(msg, content, from_node)

        elif action == ControlAction.CHANNEL_DELETE.value:
            await self._handle_channel_delete(msg, content, from_node)

        elif action == ControlAction.CHANNEL_JOIN.value:
            await self._handle_channel_join(msg, content, from_node)

        elif action == ControlAction.CHANNEL_LEAVE.value:
            await self._handle_channel_leave(msg, content, from_node)

        elif action == ControlAction.CHANNEL_LIST.value:
            await self._handle_channel_list(msg, from_node)

        elif action == ControlAction.CHANNEL_MEMBERS.value:
            await self._handle_channel_members(msg, content, from_node)

        elif action == ControlAction.CHANNEL_INVITE.value:
            await self._handle_channel_invite(msg, content, from_node)

        elif action == ControlAction.CHANNEL_REMOVE_MEMBER.value:
            await self._handle_channel_remove_member(msg, content, from_node)

        # Message sync operations
        elif action == ControlAction.HISTORY_SYNC.value:
            await self._handle_history_sync(msg, content, from_node)

        elif action == ControlAction.MARK_READ.value:
            await self._handle_mark_read(msg, content, from_node)

        elif action == ControlAction.PING.value:
            # Store heartbeat-lite status summary if present
            status_summary = content.get("status_summary")
            if status_summary and isinstance(status_summary, dict):
                self._node_status[from_node] = status_summary
            # Respond to heartbeat ping with pong
            response = Message(
                from_node="router",
                to_node=from_node,
                type=MessageType.CONTROL,
                content={"action": ControlAction.PONG.value},
                in_reply_to=msg.id,
            )
            await self._send_to_node(from_node, response)

        # Agent management
        elif action == ControlAction.LIST_AGENTS.value:
            await self._handle_list_agents(msg, from_node)

        elif action == ControlAction.START_AGENT.value:
            await self._handle_start_agent(msg, content, from_node)

        elif action == ControlAction.STOP_AGENT.value:
            await self._handle_stop_agent(msg, content, from_node)

        elif action == ControlAction.CC_USAGE.value:
            await self._handle_cc_usage(msg, from_node)

        # Scratchpad sync
        elif action == ControlAction.SCRATCHPAD_GET.value:
            await self._handle_scratchpad_get(msg, content, from_node)

        elif action == ControlAction.SCRATCHPAD_SET.value:
            await self._handle_scratchpad_set(msg, content, from_node)

        # Per-conversation todo sync
        elif action == ControlAction.TODO_GET.value:
            await self._handle_todo_get(msg, content, from_node)

        elif action == ControlAction.TODO_MUTATE.value:
            await self._handle_todo_mutate(msg, content, from_node)

        elif action == ControlAction.CALENDAR_GET.value:
            await self._handle_calendar_get(msg, content, from_node)

        else:
            logger.warning(f"Unknown control action: {action}")

    async def _deliver_pending(self, node_id: str, conn: AnyConnection) -> None:
        """Deliver any pending messages to a newly connected node."""
        pending = self.store.get_pending(node_id)
        if not pending:
            return

        logger.info(f"Delivering {len(pending)} pending messages to {node_id}")
        for msg in pending:
            try:
                await conn.send(self._message_with_signed_attachments(msg, node_id))
                self.store.remove(msg.id)
            except Exception as e:
                logger.error(f"Failed to deliver pending message {msg.id}: {e}")
                break  # Stop if connection fails

    async def _send_push_notification(self, target_node: str, msg: Message) -> None:
        """
        Send a push notification for a message stored for an offline node.

        Only sends if FCM is enabled and the target has a registered token.
        """
        if not self._fcm:
            return

        fcm_token = self.store.get_fcm_token(target_node)
        if not fcm_token:
            logger.debug(f"No FCM token for {target_node}, skipping push")
            return

        # Build notification content
        sender_name = parse_node_id(msg.from_node)[1]  # Get nickname/type
        if msg.type == MessageType.MESSAGE:
            title = f"Message from {sender_name}"
            body = str(msg.content)[:100]  # Truncate long messages
        else:
            title = f"Notification from {sender_name}"
            body = f"New {msg.type.value}"

        # Data payload for the app to handle
        data = {
            "message_id": msg.id,
            "from_node": msg.from_node,
            "message_type": msg.type.value,
        }

        await self._fcm.send_notification(fcm_token, title, body, data)

    async def _echo_to_sender(self, msg: Message, source_conn: AnyConnection | None = None) -> None:
        """
        Echo a sent message to the sender's other connections.

        This enables multi-device sync: when user sends a message from one device,
        their other connected devices receive the message immediately.

        Args:
            msg: The message that was sent
            source_conn: The connection that sent the message (to exclude from echo)
        """
        sender = msg.from_node
        conn_list = self._connections.get(sender, [])

        if len(conn_list) <= 1:
            return  # Only one connection, nothing to echo to

        for conn in conn_list:
            if source_conn and conn is source_conn:
                continue  # Don't echo back to the sender connection
            try:
                await conn.send(self._message_with_signed_attachments(msg, sender))
            except Exception as e:
                logger.error(f"Failed to echo message to {sender}: {e}")

    async def _broadcast_to_all(self, msg: Message, exclude: str | None = None) -> None:
        """
        Broadcast a message to all connected nodes.

        Args:
            msg: The message to broadcast
            exclude: Optional node ID to exclude from broadcast
        """
        for node_id, conn_list in list(self._connections.items()):
            if node_id == exclude:
                continue
            for conn in conn_list:
                try:
                    await conn.send(self._message_with_signed_attachments(msg, node_id))
                except Exception as e:
                    logger.error(f"Failed to broadcast to {node_id}: {e}")

    async def _broadcast_presence(self, node_id: str, event: str) -> None:
        """
        Broadcast a presence (join/leave) event to all connected nodes.

        Args:
            node_id: The node that joined or left
            event: "join" or "leave"
        """
        # Parse the node ID to extract nickname and type
        node_type, type_or_nick, nickname = parse_node_id(node_id)

        if node_type == "user":
            nick = type_or_nick
            ntype = "user"
        elif node_type == "agent":
            nick = nickname if nickname else type_or_nick
            ntype = type_or_nick  # Agent type (e.g., "coder")
        else:
            nick = node_id
            ntype = "unknown"

        # Get metadata (description, backend info)
        metadata = self._node_metadata.get(node_id, {})
        description = metadata.get("description", "")
        llm_backend = metadata.get("llm_backend", "")
        llm_model = metadata.get("llm_model", "")

        presence_msg = make_presence(
            from_node=node_id,
            event=event,
            nickname=nick,
            node_type=ntype,
            description=description,
            llm_backend=llm_backend,
            llm_model=llm_model,
        )

        await self._broadcast_to_all(presence_msg, exclude=node_id)

    # =========================================================================
    # Channel operations
    # =========================================================================

    async def _route_to_channel(self, msg: Message) -> None:
        """
        Route a message to all members of a channel.

        The message is delivered to all online members except the sender.
        Offline members do NOT receive the message (no persistence for channels).
        """
        channel_name = parse_channel_name(msg.to_node)
        if not channel_name:
            logger.warning(f"Invalid channel address: {msg.to_node}")
            return

        if not self.store.channel_exists(channel_name):
            logger.warning(f"Channel '{channel_name}' does not exist")
            await self._send_channel_error(
                msg.from_node, msg.id, f"channel '{channel_name}' does not exist"
            )
            return

        # Check if sender is a member
        if not self.store.is_channel_member(channel_name, msg.from_node):
            logger.warning(f"Node {msg.from_node} not a member of channel '{channel_name}'")
            await self._send_channel_error(
                msg.from_node, msg.id, f"you are not a member of channel '{channel_name}'"
            )
            return

        # Get all members and broadcast to online ones
        members = self.store.get_channel_members(channel_name)
        delivered_count = 0

        for member_id in members:
            if member_id == msg.from_node:
                continue  # Don't send back to sender

            conn_list = self._connections.get(member_id, [])
            for conn in conn_list:
                try:
                    await conn.send(self._message_with_signed_attachments(msg, member_id))
                    delivered_count += 1
                except Exception as e:
                    logger.error(f"Failed to deliver channel message to {member_id}: {e}")

        logger.debug(
            f"Channel '{channel_name}': delivered message from {msg.from_node} "
            f"to {delivered_count}/{len(members)-1} members"
        )

    async def _send_channel_error(
        self, to_node: str, in_reply_to: str, error: str
    ) -> None:
        """Send a channel operation error to a node."""
        error_msg = Message(
            from_node="router",
            to_node=to_node,
            type=MessageType.CONTROL,
            content={
                "action": ControlAction.ACK.value,
                "status": "error",
                "error": error,
            },
            in_reply_to=in_reply_to,
        )
        await self._send_to_node(to_node, error_msg)

    async def _handle_channel_create(
        self, msg: Message, content: dict, from_node: str
    ) -> None:
        """Handle CHANNEL_CREATE control message."""
        # Only users can create channels
        if not from_node.startswith("user:"):
            await self._send_channel_error(
                from_node, msg.id, "only users can create channels"
            )
            return

        channel_name = content.get("channel_name", "").strip()
        if not channel_name:
            await self._send_channel_error(from_node, msg.id, "channel_name required")
            return

        description = content.get("description", "")
        created = self.store.create_channel(channel_name, from_node, description)

        if created:
            # Auto-join the creator
            self.store.join_channel(channel_name, from_node)

            response = Message(
                from_node="router",
                to_node=from_node,
                type=MessageType.CONTROL,
                content={
                    "action": ControlAction.CHANNEL_CREATE.value,
                    "status": "created",
                    "channel_name": channel_name,
                },
                in_reply_to=msg.id,
            )
        else:
            response = Message(
                from_node="router",
                to_node=from_node,
                type=MessageType.CONTROL,
                content={
                    "action": ControlAction.ACK.value,
                    "status": "error",
                    "error": f"channel '{channel_name}' already exists",
                },
                in_reply_to=msg.id,
            )

        await self._send_to_node(from_node, response)

    async def _handle_channel_delete(
        self, msg: Message, content: dict, from_node: str
    ) -> None:
        """Handle CHANNEL_DELETE control message."""
        # Only users can delete channels
        if not from_node.startswith("user:"):
            await self._send_channel_error(
                from_node, msg.id, "only users can delete channels"
            )
            return

        channel_name = content.get("channel_name", "").strip()
        if not channel_name:
            await self._send_channel_error(from_node, msg.id, "channel_name required")
            return

        deleted = self.store.delete_channel(channel_name)

        if deleted:
            response = Message(
                from_node="router",
                to_node=from_node,
                type=MessageType.CONTROL,
                content={
                    "action": ControlAction.CHANNEL_DELETE.value,
                    "status": "deleted",
                    "channel_name": channel_name,
                },
                in_reply_to=msg.id,
            )
        else:
            response = Message(
                from_node="router",
                to_node=from_node,
                type=MessageType.CONTROL,
                content={
                    "action": ControlAction.ACK.value,
                    "status": "error",
                    "error": f"channel '{channel_name}' does not exist",
                },
                in_reply_to=msg.id,
            )

        await self._send_to_node(from_node, response)

    async def _handle_channel_join(
        self, msg: Message, content: dict, from_node: str
    ) -> None:
        """Handle CHANNEL_JOIN control message."""
        channel_name = content.get("channel_name", "").strip()
        if not channel_name:
            await self._send_channel_error(from_node, msg.id, "channel_name required")
            return

        if not self.store.channel_exists(channel_name):
            await self._send_channel_error(
                from_node, msg.id, f"channel '{channel_name}' does not exist"
            )
            return

        joined = self.store.join_channel(channel_name, from_node)

        if joined:
            # Send success response
            response = Message(
                from_node="router",
                to_node=from_node,
                type=MessageType.CONTROL,
                content={
                    "action": ControlAction.CHANNEL_JOIN.value,
                    "status": "joined",
                    "channel_name": channel_name,
                },
                in_reply_to=msg.id,
            )
            await self._send_to_node(from_node, response)

            # Broadcast join event to other channel members
            await self._broadcast_channel_presence(channel_name, from_node, "join")
        else:
            # Already a member
            response = Message(
                from_node="router",
                to_node=from_node,
                type=MessageType.CONTROL,
                content={
                    "action": ControlAction.CHANNEL_JOIN.value,
                    "status": "already_member",
                    "channel_name": channel_name,
                },
                in_reply_to=msg.id,
            )
            await self._send_to_node(from_node, response)

    async def _handle_channel_leave(
        self, msg: Message, content: dict, from_node: str
    ) -> None:
        """Handle CHANNEL_LEAVE control message."""
        channel_name = content.get("channel_name", "").strip()
        if not channel_name:
            await self._send_channel_error(from_node, msg.id, "channel_name required")
            return

        left = self.store.leave_channel(channel_name, from_node)

        if left:
            # Broadcast leave event to remaining channel members
            await self._broadcast_channel_presence(channel_name, from_node, "leave")

            response = Message(
                from_node="router",
                to_node=from_node,
                type=MessageType.CONTROL,
                content={
                    "action": ControlAction.CHANNEL_LEAVE.value,
                    "status": "left",
                    "channel_name": channel_name,
                },
                in_reply_to=msg.id,
            )
        else:
            response = Message(
                from_node="router",
                to_node=from_node,
                type=MessageType.CONTROL,
                content={
                    "action": ControlAction.ACK.value,
                    "status": "error",
                    "error": f"not a member of channel '{channel_name}'",
                },
                in_reply_to=msg.id,
            )

        await self._send_to_node(from_node, response)

    async def _handle_channel_list(self, msg: Message, from_node: str) -> None:
        """Handle CHANNEL_LIST control message."""
        channels = self.store.list_channels(for_node=from_node)

        response = Message(
            from_node="router",
            to_node=from_node,
            type=MessageType.CONTROL,
            content={
                "action": ControlAction.CHANNEL_LIST.value,
                "channels": channels,
            },
            in_reply_to=msg.id,
        )

        await self._send_to_node(from_node, response)

    async def _handle_channel_members(
        self, msg: Message, content: dict, from_node: str
    ) -> None:
        """Handle CHANNEL_MEMBERS control message."""
        channel_name = content.get("channel_name", "").strip()
        if not channel_name:
            await self._send_channel_error(from_node, msg.id, "channel_name required")
            return

        if not self.store.channel_exists(channel_name):
            await self._send_channel_error(
                from_node, msg.id, f"channel '{channel_name}' does not exist"
            )
            return

        members = self.store.get_channel_members(channel_name)

        # Include online status for each member
        members_with_status = [
            {"node_id": m, "online": bool(self._connections.get(m))}
            for m in members
        ]

        response = Message(
            from_node="router",
            to_node=from_node,
            type=MessageType.CONTROL,
            content={
                "action": ControlAction.CHANNEL_MEMBERS.value,
                "channel_name": channel_name,
                "members": members_with_status,
            },
            in_reply_to=msg.id,
        )

        await self._send_to_node(from_node, response)

    async def _handle_channel_invite(
        self, msg: Message, content: dict, from_node: str
    ) -> None:
        """Handle CHANNEL_INVITE control message - add a node to a channel."""
        # Only users can invite
        if not from_node.startswith("user:"):
            await self._send_channel_error(
                from_node, msg.id, "only users can invite members to channels"
            )
            return

        channel_name = content.get("channel_name", "").strip()
        node_id = content.get("node_id", "").strip()

        if not channel_name:
            await self._send_channel_error(from_node, msg.id, "channel_name required")
            return

        if not node_id:
            await self._send_channel_error(from_node, msg.id, "node_id required")
            return

        if not self.store.channel_exists(channel_name):
            await self._send_channel_error(
                from_node, msg.id, f"channel '{channel_name}' does not exist"
            )
            return

        # Check if already a member
        if self.store.is_channel_member(channel_name, node_id):
            await self._send_channel_error(
                from_node, msg.id, f"'{node_id}' is already a member of '{channel_name}'"
            )
            return

        # Add the node to the channel
        self.store.join_channel(channel_name, node_id)
        logger.info(f"User {from_node} invited {node_id} to channel '{channel_name}'")

        # Send ACK to inviter
        response = Message(
            from_node="router",
            to_node=from_node,
            type=MessageType.CONTROL,
            content={
                "action": ControlAction.CHANNEL_INVITE.value,
                "status": "invited",
                "channel_name": channel_name,
                "node_id": node_id,
            },
            in_reply_to=msg.id,
        )
        await self._send_to_node(from_node, response)

        # Broadcast join presence to channel members
        await self._broadcast_channel_presence(channel_name, node_id, "join")

    async def _handle_channel_remove_member(
        self, msg: Message, content: dict, from_node: str
    ) -> None:
        """Handle CHANNEL_REMOVE_MEMBER control message - remove a member from a channel."""
        # Only users can remove members
        if not from_node.startswith("user:"):
            await self._send_channel_error(
                from_node, msg.id, "only users can remove members from channels"
            )
            return

        channel_name = content.get("channel_name", "").strip()
        node_id = content.get("node_id", "").strip()

        if not channel_name:
            await self._send_channel_error(from_node, msg.id, "channel_name required")
            return

        if not node_id:
            await self._send_channel_error(from_node, msg.id, "node_id required")
            return

        if not self.store.channel_exists(channel_name):
            await self._send_channel_error(
                from_node, msg.id, f"channel '{channel_name}' does not exist"
            )
            return

        # Prevent self-removal (use leave for that)
        if node_id == from_node:
            await self._send_channel_error(
                from_node, msg.id, "use 'leave channel' to remove yourself"
            )
            return

        # Check if target is a member
        if not self.store.is_channel_member(channel_name, node_id):
            await self._send_channel_error(
                from_node, msg.id, f"'{node_id}' is not a member of '{channel_name}'"
            )
            return

        # Remove the node from the channel
        self.store.leave_channel(channel_name, node_id)
        logger.info(f"User {from_node} removed {node_id} from channel '{channel_name}'")

        # Send ACK to requester
        response = Message(
            from_node="router",
            to_node=from_node,
            type=MessageType.CONTROL,
            content={
                "action": ControlAction.CHANNEL_REMOVE_MEMBER.value,
                "status": "removed",
                "channel_name": channel_name,
                "node_id": node_id,
            },
            in_reply_to=msg.id,
        )
        await self._send_to_node(from_node, response)

        # Broadcast leave presence to channel members
        await self._broadcast_channel_presence(channel_name, node_id, "leave")

    async def _broadcast_channel_presence(
        self, channel_name: str, node_id: str, event: str
    ) -> None:
        """
        Broadcast a presence event (join/leave) to all members of a channel.

        Args:
            channel_name: The channel name
            node_id: The node that joined or left
            event: "join" or "leave"
        """
        # Parse the node ID for presence info
        node_type, type_or_nick, nickname = parse_node_id(node_id)

        if node_type == "user":
            nick = type_or_nick
            ntype = "user"
        elif node_type == "agent":
            nick = nickname if nickname else type_or_nick
            ntype = type_or_nick
        else:
            nick = node_id
            ntype = "unknown"

        metadata = self._node_metadata.get(node_id, {})
        description = metadata.get("description", "")

        presence_msg = make_presence(
            from_node=node_id,
            event=event,
            nickname=nick,
            node_type=ntype,
            description=description,
        )
        # Set the target to the channel
        presence_msg.to_node = f"channel:{channel_name}"
        # Add channel info to content so clients can read it
        presence_msg.content["channel"] = channel_name

        # Send to all online channel members except the one who joined/left
        members = self.store.get_channel_members(channel_name)
        for member_id in members:
            if member_id == node_id:
                continue
            conn_list = self._connections.get(member_id, [])
            for conn in conn_list:
                try:
                    await conn.send(presence_msg)
                except Exception as e:
                    logger.error(
                        f"Failed to send channel presence to {member_id}: {e}"
                    )

    # =========================================================================
    # Message history sync
    # =========================================================================

    async def _handle_history_sync(
        self, msg: Message, content: dict, from_node: str
    ) -> None:
        """Handle HISTORY_SYNC control message."""
        conversation_id = content.get("conversation_id")
        since = content.get("since")
        limit = content.get("limit", 500)

        # Clamp limit to a reasonable maximum
        limit = min(limit, 1000)

        logger.info(
            f"History sync request from {from_node}: "
            f"conversation={conversation_id}, since={since}, limit={limit}"
        )

        # Get messages
        if conversation_id:
            messages = self.store.get_conversation_history(
                conversation_id, since_timestamp=since, limit=limit
            )
        else:
            messages = self.store.get_all_history_for_node(
                from_node, since_timestamp=since, limit=limit
            )

        # Convert messages to dicts for JSON serialization
        # Filter to MESSAGE type only - clients may not handle tool_request/tool_result
        message_dicts = []
        for m in messages:
            if m.type != MessageType.MESSAGE:
                continue
            signed_msg = self._message_with_signed_attachments(m, from_node)
            d = asdict(signed_msg)
            d["type"] = signed_msg.type.value
            message_dicts.append(d)

        # Get read receipts for this node
        read_receipts = self.store.get_all_read_receipts(from_node)

        # Determine if there are more messages
        has_more = len(messages) >= limit

        response = make_history_response(
            to_node=from_node,
            messages=message_dicts,
            read_receipts=read_receipts,
            conversation_id=conversation_id,
            has_more=has_more,
        )
        response.in_reply_to = msg.id

        await self._send_to_node(from_node, response)
        logger.info(
            f"Sent {len(messages)} messages to {from_node} "
            f"(read receipts for {len(read_receipts)} conversations)"
        )

    async def _handle_mark_read(
        self, msg: Message, content: dict, from_node: str
    ) -> None:
        """Handle MARK_READ control message."""
        conversation_id = content.get("conversation_id")
        up_to_timestamp = content.get("up_to_timestamp")

        if not conversation_id or not up_to_timestamp:
            logger.warning(f"MARK_READ from {from_node} missing required fields")
            return

        self.store.mark_read(from_node, conversation_id, up_to_timestamp)

        # Send ACK
        response = Message(
            from_node="router",
            to_node=from_node,
            type=MessageType.CONTROL,
            content={
                "action": ControlAction.ACK.value,
                "status": "marked_read",
                "conversation_id": conversation_id,
            },
            in_reply_to=msg.id,
        )
        await self._send_to_node(from_node, response)

    # =========================================================================
    # Scratchpad Sync
    # =========================================================================

    def _is_conversation_participant(self, from_node: str, conversation_id: str) -> bool:
        """Check if from_node is a participant in the given conversation."""
        if conversation_id.startswith("channel:"):
            channel_name = conversation_id[len("channel:"):]
            return self.store.is_channel_member(channel_name, from_node)
        # DM: conversation_id is "nodeA,nodeB" (sorted pair)
        return from_node in conversation_id.split(",")

    async def _handle_scratchpad_get(
        self, msg: Message, content: dict, from_node: str
    ) -> None:
        """Handle SCRATCHPAD_GET control message."""
        conversation_ids = content.get("conversation_ids")

        if conversation_ids:
            authorized = [
                cid for cid in conversation_ids
                if self._is_conversation_participant(from_node, cid)
            ]
            notes = self.store.get_scratchpads(authorized) if authorized else {}
        else:
            all_notes = self.store.get_all_scratchpads()
            notes = {
                cid: note for cid, note in all_notes.items()
                if self._is_conversation_participant(from_node, cid)
            }

        response = make_scratchpad_response(
            to_node=from_node,
            notes=notes,
            in_reply_to=msg.id,
        )
        await self._send_to_node(from_node, response)
        logger.debug(f"Sent {len(notes)} scratchpad(s) to {from_node}")

    async def _handle_scratchpad_set(
        self, msg: Message, content: dict, from_node: str
    ) -> None:
        """Handle SCRATCHPAD_SET control message with optimistic concurrency."""
        conversation_id = content.get("conversation_id")
        text = content.get("text", "")
        client_timestamp = content.get("client_timestamp", "")

        if not conversation_id:
            logger.warning(f"SCRATCHPAD_SET from {from_node} missing conversation_id")
            return

        if not self._is_conversation_participant(from_node, conversation_id):
            logger.warning(
                f"SCRATCHPAD_SET rejected: {from_node} not a participant in {conversation_id}"
            )
            return

        accepted, current_state = self.store.set_scratchpad(
            conversation_id=conversation_id,
            content=text,
            updated_by=from_node,
            client_timestamp=client_timestamp,
        )

        response = Message(
            from_node="router",
            to_node=from_node,
            type=MessageType.CONTROL,
            content={
                "action": ControlAction.SCRATCHPAD_RESPONSE.value,
                "accepted": accepted,
                "conversation_id": conversation_id,
                "notes": {conversation_id: current_state},
            },
            in_reply_to=msg.id,
        )
        await self._send_to_node(from_node, response)

    # =========================================================================
    # Conversation Todo Sync
    # =========================================================================

    def _conversation_participants(self, conversation_id: str) -> list[str]:
        """Return node IDs allowed to receive shared state for a conversation."""
        if conversation_id.startswith("channel:"):
            channel_name = conversation_id[len("channel:"):]
            return self.store.get_channel_members(channel_name)
        return [p for p in conversation_id.split(",") if p]

    async def _send_todo_error(
        self,
        from_node: str,
        msg: Message,
        conversation_id: str | None,
        error: str,
    ) -> None:
        response = make_todo_response(
            to_node=from_node,
            todos={},
            section_order=(
                {conversation_id: self.store.get_todo_section_order(conversation_id)}
                if conversation_id else None
            ),
            accepted=False,
            conversation_id=conversation_id,
            error=error,
            in_reply_to=msg.id,
        )
        await self._send_to_node(from_node, response)

    async def _handle_todo_get(
        self, msg: Message, content: dict, from_node: str
    ) -> None:
        """Handle TODO_GET control message."""
        conversation_ids = content.get("conversation_ids")
        include_done = bool(content.get("include_done", True))

        if conversation_ids:
            authorized = [
                cid for cid in conversation_ids
                if self._is_conversation_participant(from_node, cid)
            ]
        else:
            # There is no cheap "all conversations this node participates in"
            # todo index today. Clients request specific conversations.
            authorized = []

        todos = {
            cid: self.store.list_todos(cid, include_done=include_done, limit=1000)
            for cid in authorized
        }
        section_order = {
            cid: self.store.get_todo_section_order(cid)
            for cid in authorized
        }
        response = make_todo_response(
            to_node=from_node,
            todos=todos,
            section_order=section_order,
            in_reply_to=msg.id,
        )
        await self._send_to_node(from_node, response)
        logger.debug(f"Sent todo lists for {len(todos)} conversation(s) to {from_node}")

    async def _broadcast_todo_response(
        self,
        conversation_id: str,
        todos: list[dict],
        source_msg: Message,
        accepted: bool = True,
        server_state: dict | None = None,
    ) -> None:
        """Broadcast fresh todo state to online participants."""
        payload = {conversation_id: todos}
        section_order = {conversation_id: self.store.get_todo_section_order(conversation_id)}
        for participant in self._conversation_participants(conversation_id):
            response = make_todo_response(
                to_node=participant,
                todos=payload,
                section_order=section_order,
                accepted=accepted,
                conversation_id=conversation_id,
                server_state=server_state,
                in_reply_to=source_msg.id if participant == source_msg.from_node else None,
            )
            await self._send_to_node(participant, response)

    def _todo_belongs_to_conversation(
        self, todo_id: str, conversation_id: str,
    ) -> tuple[bool, dict]:
        todo = self.store.get_todo(todo_id)
        if not todo:
            return False, {"error": "todo not found", "id": todo_id}
        if todo.get("conversation_id") != conversation_id:
            return False, {
                "error": "todo does not belong to this conversation",
                "id": todo_id,
                "conversation_id": todo.get("conversation_id"),
            }
        return True, todo

    async def _handle_todo_mutate(
        self, msg: Message, content: dict, from_node: str
    ) -> None:
        """Handle TODO_MUTATE control message."""
        conversation_id = content.get("conversation_id")
        op = str(content.get("op", "")).strip().lower()
        payload = content.get("payload") if isinstance(content.get("payload"), dict) else {}
        expected_version = content.get("expected_version")
        if expected_version is None:
            expected_version = payload.get("expected_version")

        if not conversation_id:
            await self._send_todo_error(from_node, msg, None, "conversation_id required")
            return
        if not self._is_conversation_participant(from_node, conversation_id):
            logger.warning(
                f"TODO_MUTATE rejected: {from_node} not a participant in {conversation_id}"
            )
            await self._send_todo_error(
                from_node, msg, conversation_id, "not a participant in conversation"
            )
            return

        accepted = True
        server_state: dict | None = None
        try:
            if op == "add":
                server_state = self.store.add_todo(
                    conversation_id=conversation_id,
                    text=payload.get("text", ""),
                    created_by=from_node,
                    priority=int(payload.get("priority", 0) or 0),
                    position=payload.get("position"),
                    section=payload.get("section"),
                )
            elif op in {"update", "toggle", "remove", "delete"}:
                todo_id = str(payload.get("todo_id") or payload.get("id") or "")
                if not todo_id:
                    await self._send_todo_error(from_node, msg, conversation_id, "todo_id required")
                    return
                belongs, current = self._todo_belongs_to_conversation(todo_id, conversation_id)
                if not belongs:
                    await self._send_todo_error(
                        from_node, msg, conversation_id, str(current.get("error", "invalid todo"))
                    )
                    return
                if op == "toggle":
                    done = payload.get("done")
                    status = "done" if done is not False else "open"
                    accepted, server_state = self.store.update_todo(
                        todo_id=todo_id,
                        updated_by=from_node,
                        status=status,
                        expected_version=expected_version,
                    )
                elif op in {"remove", "delete"}:
                    accepted, server_state = self.store.delete_todo(
                        todo_id=todo_id,
                        updated_by=from_node,
                        expected_version=expected_version,
                    )
                else:
                    accepted, server_state = self.store.update_todo(
                        todo_id=todo_id,
                        updated_by=from_node,
                        text=payload.get("text") if "text" in payload else None,
                        status=payload.get("status") if "status" in payload else None,
                        priority=payload.get("priority") if "priority" in payload else None,
                        position=payload.get("position") if "position" in payload else None,
                        section=payload.get("section"),
                        update_section="section" in payload,
                        expected_version=expected_version,
                    )
            elif op == "set_section_order":
                section_order_payload = payload.get("section_order")
                if section_order_payload is not None and not isinstance(section_order_payload, list):
                    await self._send_todo_error(
                        from_node, msg, conversation_id, "section_order must be a list or null"
                    )
                    return
                server_state = {
                    "section_order": self.store.set_todo_section_order(
                        conversation_id=conversation_id,
                        section_order=section_order_payload,
                        updated_by=from_node,
                    )
                }
            elif op == "reorder":
                ordered_ids = payload.get("ordered_ids", [])
                if not isinstance(ordered_ids, list):
                    await self._send_todo_error(
                        from_node, msg, conversation_id, "ordered_ids must be a list"
                    )
                    return
                for todo_id in ordered_ids:
                    belongs, current = self._todo_belongs_to_conversation(str(todo_id), conversation_id)
                    if not belongs:
                        await self._send_todo_error(
                            from_node, msg, conversation_id, str(current.get("error", "invalid todo"))
                        )
                        return
                server_state = {
                    "reordered": self.store.reorder_todos(
                        conversation_id=conversation_id,
                        ordered_ids=[str(todo_id) for todo_id in ordered_ids],
                        updated_by=from_node,
                    )
                }
            elif op == "clear_done":
                cleared = []
                todos = self.store.list_todos(conversation_id, include_done=True, limit=1000)
                for todo in todos:
                    if todo.get("status") == "done":
                        ok, state = self.store.delete_todo(todo["id"], updated_by=from_node)
                        if ok:
                            cleared.append(state)
                server_state = {"cleared": cleared}
            else:
                await self._send_todo_error(from_node, msg, conversation_id, f"unknown op '{op}'")
                return
        except Exception as e:
            logger.exception("TODO_MUTATE failed")
            await self._send_todo_error(from_node, msg, conversation_id, str(e))
            return

        todos = self.store.list_todos(conversation_id, include_done=True, limit=1000)
        if accepted:
            await self._broadcast_todo_response(
                conversation_id=conversation_id,
                todos=todos,
                source_msg=msg,
                accepted=True,
                server_state=server_state,
            )
        else:
            response = make_todo_response(
                to_node=from_node,
                todos={conversation_id: todos},
                section_order={
                    conversation_id: self.store.get_todo_section_order(conversation_id)
                },
                accepted=False,
                conversation_id=conversation_id,
                server_state=server_state,
                in_reply_to=msg.id,
            )
            await self._send_to_node(from_node, response)

    def _calendar_event_sort_key(self, event: dict) -> tuple[str, str]:
        start = event.get("start") if isinstance(event.get("start"), dict) else {}
        value = start.get("dateTime") or start.get("date") or ""
        return (str(value), str(event.get("summary", "")))

    def _fetch_calendar_events_for_account(
        self,
        account: str,
        date: str,
        timezone: str,
    ) -> tuple[list[dict], str | None]:
        from .tool_implementations import calendar_list_on_date

        raw = calendar_list_on_date(date=date, timezone=timezone, account=account)
        if isinstance(raw, str) and raw.startswith("Error:"):
            return [], f"{account}: {raw}"
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as e:
            return [], f"{account}: invalid calendar response ({e})"
        if not isinstance(parsed, list):
            return [], f"{account}: unexpected calendar response ({type(parsed).__name__})"

        events: list[dict] = []
        for event in parsed:
            if isinstance(event, dict):
                item = dict(event)
                item["account"] = account
                events.append(item)
        return events, None

    async def _handle_calendar_get(
        self, msg: Message, content: dict, from_node: str
    ) -> None:
        """Handle CALENDAR_GET control message."""
        date = str(content.get("date") or "").strip()
        timezone = str(content.get("timezone") or "America/Chicago").strip() or "America/Chicago"
        accounts = content.get("accounts")
        if not isinstance(accounts, list) or not accounts:
            accounts = ["work", "personal"]
        accounts = [str(account).strip() for account in accounts if str(account).strip()]

        if not date:
            response = make_calendar_response(
                to_node=from_node,
                date="",
                events=[],
                errors=["date required"],
                timezone=timezone,
                accounts=accounts,
                in_reply_to=msg.id,
            )
            await self._send_to_node(from_node, response)
            return

        events: list[dict] = []
        errors: list[str] = []
        async with self._calendar_lock:
            for account in accounts:
                account_events, error = await asyncio.to_thread(
                    self._fetch_calendar_events_for_account,
                    account,
                    date,
                    timezone,
                )
                events.extend(account_events)
                if error:
                    errors.append(error)

        events.sort(key=self._calendar_event_sort_key)
        response = make_calendar_response(
            to_node=from_node,
            date=date,
            events=events,
            errors=errors,
            timezone=timezone,
            accounts=accounts,
            in_reply_to=msg.id,
        )
        await self._send_to_node(from_node, response)

    # =========================================================================
    # Agent Management
    # =========================================================================

    async def _handle_list_agents(self, msg: Message, from_node: str) -> None:
        """
        Handle LIST_AGENTS control message.

        Returns:
            - configured: Agent types defined in mesh.yaml
            - connected: Currently running agents
        """
        from .config import load_config

        # Get connected agents
        connected_agents = [
            node_id for node_id in self._connections.keys()
            if node_id.startswith("agent:")
        ]

        # Get configured agent types from mesh.yaml
        configured = []
        try:
            config = load_config()
            for node_id, node_config in config.nodes.items():
                if node_id.startswith("agent:"):
                    # Extract agent type from node_id
                    parts = node_id.split(":")
                    agent_type = parts[1] if len(parts) >= 2 else node_id
                    # Get controller mode if configured
                    controller_mode = None
                    if node_config.controller:
                        controller_mode = node_config.controller.mode
                    configured.append({
                        "node_id": node_id,
                        "agent_type": agent_type,
                        "llm_model": node_config.llm_model,
                        "llm_backend": node_config.llm_backend,
                        "system_prompt_file": node_config.system_prompt_file,
                        "controller": controller_mode,
                    })
        except Exception as e:
            logger.warning(f"Failed to load config for LIST_AGENTS: {e}")

        # Build per-agent status from heartbeat-lite data
        agent_status = {}
        for node_id in connected_agents:
            if node_id in self._node_status:
                agent_status[node_id] = self._node_status[node_id]

        # Also include online users in status
        connected_users = [
            node_id for node_id in self._connections.keys()
            if node_id.startswith("user:")
        ]

        response = Message(
            from_node="router",
            to_node=from_node,
            type=MessageType.CONTROL,
            content={
                "action": ControlAction.LIST_AGENTS.value,
                "configured": configured,
                "connected": connected_agents,
                "connected_users": connected_users,
                "status": agent_status,
                "cc_usage": self.get_cc_usage_summary(),
            },
            in_reply_to=msg.id,
        )
        await self._send_to_node(from_node, response)

    async def _handle_start_agent(
        self, msg: Message, content: dict, from_node: str
    ) -> None:
        """
        Handle START_AGENT control message.

        Spawns a new agent process using run_agent.py.

        Required content fields:
            - agent_type: Agent type (e.g., "assistant", "coder")
            - nickname: Unique nickname for the agent

        Optional fields:
            - backend: LLM backend name
            - model: Model override
            - fresh: Start without history (default: False)
            - controller: Controller mode (passthrough, task-fsm-v0, phase-flow-v02)
            - effort: Effort level for v0.2 controller (low, medium, high)
        """
        import subprocess
        import sys

        agent_type = content.get("agent_type")
        nickname = content.get("nickname")
        backend = content.get("backend")
        model = content.get("model")
        fresh = content.get("fresh", False)
        controller = content.get("controller")
        effort = content.get("effort")

        if not agent_type:
            response = Message(
                from_node="router",
                to_node=from_node,
                type=MessageType.CONTROL,
                content={
                    "action": ControlAction.ACK.value,
                    "success": False,
                    "error": "agent_type is required",
                },
                in_reply_to=msg.id,
            )
            await self._send_to_node(from_node, response)
            return

        if not nickname:
            response = Message(
                from_node="router",
                to_node=from_node,
                type=MessageType.CONTROL,
                content={
                    "action": ControlAction.ACK.value,
                    "success": False,
                    "error": "nickname is required",
                },
                in_reply_to=msg.id,
            )
            await self._send_to_node(from_node, response)
            return

        # Build node ID to check if already running
        from .protocol import build_agent_node_id
        node_id = build_agent_node_id(agent_type, nickname)

        if node_id in self._connections:
            response = Message(
                from_node="router",
                to_node=from_node,
                type=MessageType.CONTROL,
                content={
                    "action": ControlAction.ACK.value,
                    "success": False,
                    "error": f"Agent {node_id} is already running",
                },
                in_reply_to=msg.id,
            )
            await self._send_to_node(from_node, response)
            return

        # Build command
        cmd = [
            sys.executable, "run_agent.py",
            "--agent", agent_type,
            "--nickname", nickname,
        ]
        if backend:
            cmd.extend(["--backend", backend])
        if model:
            cmd.extend(["--model", model])
        if fresh:
            cmd.append("--fresh")
        if controller:
            cmd.extend(["--controller", controller])
        if effort:
            cmd.extend(["--effort", effort])

        # Spawn the agent process
        try:
            # Use subprocess.Popen for non-blocking spawn
            import os
            env = os.environ.copy()
            # Pass auth token if configured
            if self.config.auth_enabled and self.config.auth_token:
                env["MESH_AUTH_TOKEN"] = self.config.auth_token

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,  # Detach from parent
                env=env,
            )
            logger.info(f"Started agent {node_id} with PID {process.pid}")

            response = Message(
                from_node="router",
                to_node=from_node,
                type=MessageType.CONTROL,
                content={
                    "action": ControlAction.ACK.value,
                    "success": True,
                    "node_id": node_id,
                    "pid": process.pid,
                },
                in_reply_to=msg.id,
            )
        except Exception as e:
            logger.error(f"Failed to start agent {node_id}: {e}")
            response = Message(
                from_node="router",
                to_node=from_node,
                type=MessageType.CONTROL,
                content={
                    "action": ControlAction.ACK.value,
                    "success": False,
                    "error": str(e),
                },
                in_reply_to=msg.id,
            )

        await self._send_to_node(from_node, response)

    async def _handle_stop_agent(
        self, msg: Message, content: dict, from_node: str
    ) -> None:
        """
        Handle STOP_AGENT control message.

        Sends a SHUTDOWN message to the target agent.

        Required content fields:
            - target: Agent node ID or nickname
        """
        from .protocol import make_shutdown_request

        target = content.get("target")
        reason = content.get("reason", f"Stopped by {from_node}")

        if not target:
            response = Message(
                from_node="router",
                to_node=from_node,
                type=MessageType.CONTROL,
                content={
                    "action": ControlAction.ACK.value,
                    "success": False,
                    "error": "target is required",
                },
                in_reply_to=msg.id,
            )
            await self._send_to_node(from_node, response)
            return

        # Resolve target to full node ID if it's a nickname
        target_id = target
        if not target.startswith("agent:"):
            # Search for agent with this nickname
            for node_id in self._connections.keys():
                if node_id.startswith("agent:") and node_id.endswith(f":{target}"):
                    target_id = node_id
                    break

        if target_id not in self._connections:
            response = Message(
                from_node="router",
                to_node=from_node,
                type=MessageType.CONTROL,
                content={
                    "action": ControlAction.ACK.value,
                    "success": False,
                    "error": f"Agent {target_id} is not connected",
                },
                in_reply_to=msg.id,
            )
            await self._send_to_node(from_node, response)
            return

        # Send shutdown request to the agent
        # The agent will handle the SHUTDOWN message and exit
        shutdown_msg = make_shutdown_request(
            from_node="router",
            target_node=target_id,
            auth_token=self.config.auth_token or "",
            reason=reason,
        )
        await self._send_to_node(target_id, shutdown_msg)

        logger.info(f"Sent shutdown request to {target_id}")

        response = Message(
            from_node="router",
            to_node=from_node,
            type=MessageType.CONTROL,
            content={
                "action": ControlAction.ACK.value,
                "success": True,
                "target": target_id,
                "message": f"Shutdown request sent to {target_id}",
            },
            in_reply_to=msg.id,
        )
        await self._send_to_node(from_node, response)

    async def _handle_cc_usage(self, msg: Message, from_node: str) -> None:
        """Handle CC_USAGE request — fresh fetch returning raw API data.

        Always does a fresh async fetch (user explicitly requested it).
        Returns raw API response shape so clients get ISO timestamps for
        resets_at (not pre-formatted deltas from CCUsageResult).
        """
        from .cc_usage import _derive_creds_path, _refresh_token_async, _read_account_email
        import httpx
        import json as _json
        from pathlib import Path

        async def _fetch_raw(label: str, home_dir: str) -> dict:
            """Fetch raw usage data for one account."""
            home = Path(home_dir).expanduser()
            creds_path = _derive_creds_path(home)
            email = _read_account_email(home)
            if not creds_path.exists():
                return {"label": label, "email": email, "error": "no credentials"}
            try:
                creds = _json.loads(creds_path.read_text())
            except Exception as e:
                return {"label": label, "email": email, "error": str(e)}

            oauth = creds.get("claudeAiOauth")
            if not oauth or not isinstance(oauth, dict):
                return {"label": label, "email": email, "error": "no OAuth"}

            sub_type = oauth.get("subscriptionType", "unknown")
            access_token = await _refresh_token_async(oauth, creds, creds_path)
            if not access_token:
                return {"label": label, "email": email, "error": "no token", "sub": sub_type}

            try:
                async with httpx.AsyncClient() as client:
                    r = await client.get(
                        "https://api.anthropic.com/api/oauth/usage",
                        headers={
                            "Authorization": f"Bearer {access_token}",
                            "anthropic-beta": "oauth-2025-04-20",
                        },
                        timeout=10,
                    )
                if r.status_code != 200:
                    return {"label": label, "email": email, "error": f"HTTP {r.status_code}", "sub": sub_type}
                data = r.json()
                data["label"] = label
                data["email"] = email
                data["sub"] = sub_type
                return data
            except Exception as e:
                return {"label": label, "email": email, "error": str(e), "sub": sub_type}

        # Fetch all accounts concurrently
        tasks = [_fetch_raw(lbl, hdir) for lbl, hdir in self._cc_usage_home_dirs]
        accounts = await asyncio.gather(*tasks)

        response = Message(
            from_node="router",
            to_node=from_node,
            type=MessageType.CONTROL,
            content={
                "action": ControlAction.CC_USAGE.value,
                "accounts": list(accounts),
            },
            in_reply_to=msg.id,
        )
        await self._send_to_node(from_node, response)
