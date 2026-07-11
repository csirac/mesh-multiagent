package com.mesh.client.data.remote

import android.util.Log
import com.mesh.client.data.remote.protocol.ControlAction
import com.mesh.client.data.remote.protocol.Message
import com.mesh.client.data.remote.protocol.MessageFactory
import com.mesh.client.data.remote.protocol.MessageType
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.receiveAsFlow
import kotlinx.coroutines.launch
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.Response
import okhttp3.WebSocket
import okhttp3.WebSocketListener
import okio.ByteString
import java.nio.ByteBuffer
import java.util.concurrent.TimeUnit
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Connection state for the mesh socket.
 */
sealed class ConnectionState {
    object Disconnected : ConnectionState()
    object Connecting : ConnectionState()
    object Connected : ConnectionState()
    data class Error(val message: String) : ConnectionState()
}

/**
 * WebSocket client for mesh communication.
 *
 * Uses OkHttp WebSocket with length-prefixed message framing.
 * Supports automatic reconnection with exponential backoff.
 */
@Singleton
class MeshSocketClient @Inject constructor() {
    private val tag = "MeshSocketClient"

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)

    private val client = OkHttpClient.Builder()
        .readTimeout(0, TimeUnit.MILLISECONDS) // No timeout for WebSocket
        .pingInterval(30, TimeUnit.SECONDS)    // Keep-alive pings
        .build()

    private var webSocket: WebSocket? = null
    private var _nodeId: String? = null
    val nodeId: String? get() = _nodeId
    private var nickname: String? = null

    // Connection state
    private val _connectionState = MutableStateFlow<ConnectionState>(ConnectionState.Disconnected)
    val connectionState: StateFlow<ConnectionState> = _connectionState.asStateFlow()

    // Whether REGISTER has been sent (for FCM token sequencing)
    private val _isRegistered = MutableStateFlow(false)
    val isRegistered: StateFlow<Boolean> = _isRegistered.asStateFlow()

    // Incoming messages channel
    private val _incomingMessages = Channel<Message>(Channel.BUFFERED)
    val incomingMessages = _incomingMessages.receiveAsFlow()

    // Buffer for partial messages (length-prefixed framing)
    private var receiveBuffer = ByteArray(0)

    // Reconnection settings
    private var autoReconnect = false
    private var reconnectAttempts = 0
    private val maxReconnectAttempts = 10
    private var lastHost: String? = null
    private var lastPort: Int? = null
    private var lastAuthToken: String? = null
    private var lastUseTls: Boolean = false

    /**
     * Connect to the mesh router.
     *
     * @param host Router hostname
     * @param port Router WebSocket port (default 8080)
     * @param nodeId This client's node ID (e.g., "user:yourname")
     * @param nickname Display nickname
     * @param authToken Optional authentication token
     * @param useTls Whether to use secure WebSocket (wss://)
     */
    fun connect(
        host: String,
        port: Int,
        nodeId: String,
        nickname: String,
        authToken: String? = null,
        useTls: Boolean = false
    ) {
        if (_connectionState.value == ConnectionState.Connecting) {
            Log.w(tag, "Already connecting, ignoring duplicate connect request")
            return
        }

        this._nodeId = nodeId
        this.nickname = nickname
        this.lastHost = host
        this.lastPort = port
        this.lastAuthToken = authToken
        this.lastUseTls = useTls
        this.autoReconnect = true

        _connectionState.value = ConnectionState.Connecting

        // Connect to WebSocket endpoint on the mesh router
        // nginx proxies /mesh/ws to the router's /ws endpoint
        val protocol = if (useTls) "wss" else "ws"
        // Omit port for default ports (443 for wss, 80 for ws)
        val url = if ((useTls && port == 443) || (!useTls && port == 80)) {
            "$protocol://$host/mesh/ws"
        } else {
            "$protocol://$host:$port/mesh/ws"
        }
        Log.i(tag, "Connecting to $url")

        val requestBuilder = Request.Builder().url(url)
        authToken?.let { requestBuilder.addHeader("Authorization", "Bearer $it") }
        requestBuilder.addHeader("X-Node-Id", nodeId)
        requestBuilder.addHeader("X-Nickname", nickname)

        val request = requestBuilder.build()

        webSocket = client.newWebSocket(request, object : WebSocketListener() {
            override fun onOpen(webSocket: WebSocket, response: Response) {
                Log.i(tag, "WebSocket connected")
                reconnectAttempts = 0

                // Send registration FIRST - router requires REGISTER as the first message
                sendRegistration()

                // Only set Connected after registration is sent, so that
                // MeshService observers (e.g. history sync) don't fire before register
                _connectionState.value = ConnectionState.Connected
            }

            override fun onMessage(webSocket: WebSocket, bytes: ByteString) {
                handleIncomingBytes(bytes.toByteArray())
            }

            override fun onMessage(webSocket: WebSocket, text: String) {
                // If server sends text JSON directly (without length prefix)
                handleIncomingText(text)
            }

            override fun onClosing(webSocket: WebSocket, code: Int, reason: String) {
                Log.i(tag, "WebSocket closing: $code $reason")
                webSocket.close(1000, null)
            }

            override fun onClosed(webSocket: WebSocket, code: Int, reason: String) {
                Log.i(tag, "WebSocket closed: $code $reason")
                handleDisconnect()
            }

            override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
                Log.e(tag, "WebSocket error: ${t.message}", t)
                _connectionState.value = ConnectionState.Error(t.message ?: "Unknown error")
                handleDisconnect()
            }
        })
    }

    /**
     * Disconnect from the mesh router.
     */
    fun disconnect() {
        autoReconnect = false
        webSocket?.close(1000, "Client disconnect")
        webSocket = null
        _isRegistered.value = false
        _connectionState.value = ConnectionState.Disconnected
    }

    /**
     * Send a message to the mesh.
     */
    fun send(message: Message): Boolean {
        val ws = webSocket ?: return false

        return try {
            // Send as plain JSON text (WebSocket handles framing)
            val json = Message.toJson(message)
            ws.send(json)
            Log.d(tag, "Sent message: ${message.type} to ${message.toNode}")
            true
        } catch (e: Exception) {
            Log.e(tag, "Failed to send message", e)
            false
        }
    }

    /**
     * Send a text message to a target node.
     */
    fun sendMessage(toNode: String, content: String): Boolean {
        val from = nodeId
        if (from == null) {
            Log.w(tag, "sendMessage: nodeId is null, cannot send")
            return false
        }
        Log.d(tag, "sendMessage: from=$from, toNode=$toNode, content='$content'")
        val message = MessageFactory.makeMessage(from, toNode, content)
        return send(message)
    }

    /**
     * Send a confirmation response.
     */
    fun sendConfirmResponse(toNode: String, inReplyTo: String, confirmed: Boolean): Boolean {
        val from = nodeId ?: return false
        val message = MessageFactory.makeConfirmResponse(from, toNode, inReplyTo, confirmed)
        return send(message)
    }

    /**
     * Request status from an agent.
     */
    fun sendStatusRequest(toNode: String, numMessages: Int = 5): Boolean {
        val from = nodeId ?: return false
        val message = MessageFactory.makeStatusRequest(from, toNode, numMessages)
        return send(message)
    }

    // --- Message deletion ---

    /**
     * Delete a message (sync to server).
     */
    fun deleteMessage(messageId: String, conversationId: String): Boolean {
        val from = nodeId ?: return false
        val message = MessageFactory.makeDeleteMessage(from, messageId, conversationId)
        return send(message)
    }

    // --- Channel operations ---

    /**
     * Create a new channel.
     */
    fun createChannel(channelName: String, description: String = ""): Boolean {
        val from = nodeId ?: return false
        val message = MessageFactory.makeChannelCreate(from, channelName, description)
        return send(message)
    }

    /**
     * Delete a channel (users only).
     */
    fun deleteChannel(channelName: String): Boolean {
        val from = nodeId ?: return false
        val message = MessageFactory.makeChannelDelete(from, channelName)
        return send(message)
    }

    /**
     * Join a channel.
     */
    fun joinChannel(channelName: String): Boolean {
        val from = nodeId ?: return false
        val message = MessageFactory.makeChannelJoin(from, channelName)
        return send(message)
    }

    /**
     * Leave a channel.
     */
    fun leaveChannel(channelName: String): Boolean {
        val from = nodeId ?: return false
        val message = MessageFactory.makeChannelLeave(from, channelName)
        return send(message)
    }

    /**
     * Request list of all channels.
     */
    fun listChannels(): Boolean {
        val from = nodeId ?: return false
        val message = MessageFactory.makeChannelList(from)
        return send(message)
    }

    /**
     * Request members of a channel.
     */
    fun getChannelMembers(channelName: String): Boolean {
        val from = nodeId ?: return false
        val message = MessageFactory.makeChannelMembers(from, channelName)
        return send(message)
    }

    /**
     * Invite a node to join a channel.
     */
    fun inviteToChannel(channelName: String, nodeId: String): Boolean {
        val from = this.nodeId ?: return false
        val message = MessageFactory.makeChannelInvite(from, channelName, nodeId)
        return send(message)
    }

    // --- Message sync operations ---

    /**
     * Request message history sync from the server.
     *
     * @param since Optional ISO timestamp - only return messages after this time
     * @param limit Maximum number of messages to return
     * @param conversationId Optional specific conversation to sync
     */
    fun requestHistorySync(since: String? = null, limit: Int = 500, conversationId: String? = null): Boolean {
        val from = nodeId ?: return false
        val message = MessageFactory.makeHistorySync(from, since, limit, conversationId)
        return send(message)
    }

    /**
     * Mark messages in a conversation as read.
     *
     * @param conversationId The conversation ID
     * @param upToTimestamp ISO timestamp - mark all messages up to this point as read
     */
    fun markRead(conversationId: String, upToTimestamp: String): Boolean {
        val from = nodeId ?: return false
        val message = MessageFactory.makeMarkRead(from, conversationId, upToTimestamp)
        return send(message)
    }

    /**
     * Request Claude Code account usage from the router.
     */
    fun requestCcUsage(): Boolean {
        val from = nodeId ?: return false
        val message = MessageFactory.makeCcUsageRequest(from)
        return send(message)
    }

    private fun sendRegistration() {
        val from = nodeId ?: return
        val nick = nickname ?: return

        val registerMsg = MessageFactory.makeControl(
            fromNode = from,
            action = ControlAction.REGISTER,
            authToken = lastAuthToken?.trim()
        )
        send(registerMsg)

        // Mark as registered so FCM token manager knows it can send
        _isRegistered.value = true
        Log.d(tag, "Registration sent, isRegistered=true")

        // Also send presence announcement
        val presenceMsg = MessageFactory.makePresence(
            fromNode = from,
            event = "join",
            nickname = nick,
            nodeType = "user"
        )
        send(presenceMsg)
    }

    private fun handleIncomingBytes(bytes: ByteArray) {
        // Append to buffer
        receiveBuffer += bytes

        // Process complete messages
        while (receiveBuffer.size >= 4) {
            val messageLength = ByteBuffer.wrap(receiveBuffer.copyOfRange(0, 4)).int

            if (receiveBuffer.size < 4 + messageLength) {
                // Incomplete message, wait for more data
                break
            }

            // Extract message
            val jsonBytes = receiveBuffer.copyOfRange(4, 4 + messageLength)
            receiveBuffer = receiveBuffer.copyOfRange(4 + messageLength, receiveBuffer.size)

            val jsonString = jsonBytes.toString(Charsets.UTF_8)
            processMessage(jsonString)
        }
    }

    private fun handleIncomingText(text: String) {
        // Direct JSON text (no length prefix)
        Log.d(tag, "handleIncomingText: received text frame, length=${text.length}")
        processMessage(text)
    }

    private fun processMessage(json: String) {
        try {
            val message = Message.fromJson(json)
            Log.d(tag, "Received: ${message.type} from ${message.fromNode}")

            // Check for auth errors - stop reconnection if auth fails
            if (message.type == MessageType.CONTROL && message.fromNode == "router") {
                val content = message.content
                if (content is kotlinx.serialization.json.JsonObject) {
                    val status = content["status"]?.toString()?.trim('"')
                    val error = content["error"]?.toString()?.trim('"')
                    if (status == "error" && error?.contains("authentication") == true) {
                        Log.e(tag, "Authentication failed: $error")
                        autoReconnect = false  // Don't retry on auth failure
                        _connectionState.value = ConnectionState.Error("Authentication failed: $error")
                        disconnect()
                        return
                    }
                }
            }

            scope.launch {
                _incomingMessages.send(message)
            }
        } catch (e: Exception) {
            Log.e(tag, "Failed to parse message: $json", e)
        }
    }

    private fun handleDisconnect() {
        webSocket = null

        if (autoReconnect && reconnectAttempts < maxReconnectAttempts) {
            scope.launch {
                val delayMs = minOf(1000L * (1 shl reconnectAttempts), 60000L) // Exponential backoff, max 60s
                Log.i(tag, "Reconnecting in ${delayMs}ms (attempt ${reconnectAttempts + 1})")
                delay(delayMs)

                reconnectAttempts++

                val host = lastHost ?: return@launch
                val port = lastPort ?: return@launch
                val node = nodeId ?: return@launch
                val nick = nickname ?: return@launch

                connect(host, port, node, nick, lastAuthToken, lastUseTls)
            }
        } else if (reconnectAttempts >= maxReconnectAttempts) {
            Log.e(tag, "Max reconnection attempts reached")
            _connectionState.value = ConnectionState.Error("Max reconnection attempts reached")
        }
    }
}
