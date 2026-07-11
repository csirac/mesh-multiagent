package com.mesh.client.service

import android.app.Service
import android.content.Intent
import android.os.Binder
import android.os.IBinder
import android.util.Log
import androidx.core.app.ServiceCompat
import com.mesh.client.data.local.db.MeshDatabase
import com.mesh.client.data.local.db.entities.ChannelEntity
import com.mesh.client.data.local.db.entities.ChannelMemberEntity
import com.mesh.client.data.local.db.entities.MessageEntity
import com.mesh.client.data.local.db.entities.MessageStatus
import com.mesh.client.data.local.db.entities.RosterEntry
import com.mesh.client.data.remote.ConfirmationManager
import com.mesh.client.data.remote.ConfirmationRequest
import com.mesh.client.data.remote.ConnectionState
import com.mesh.client.data.remote.MeshSocketClient
import com.mesh.client.data.remote.StatusContextEntry
import com.mesh.client.data.remote.StatusManager
import com.mesh.client.data.remote.StatusResponse
import com.mesh.client.data.remote.StatusSummary
import com.mesh.client.data.remote.DiagnosticReport
import com.mesh.client.data.remote.UsageManager
import com.mesh.client.data.remote.AccountUsage
import com.mesh.client.data.remote.CcUsageResponse
import com.mesh.client.data.remote.ExtraUsage
import com.mesh.client.data.remote.UsageWindow
import com.mesh.client.data.remote.protocol.ControlAction
import com.mesh.client.data.remote.protocol.Message
import com.mesh.client.data.remote.protocol.MessageFactory
import com.mesh.client.data.remote.protocol.MessageType
import com.mesh.client.data.remote.protocol.isChannelAddress
import com.mesh.client.data.remote.protocol.normalizeToUtc
import com.mesh.client.data.remote.protocol.nowIso
import com.mesh.client.data.remote.protocol.parseChannelName
import com.mesh.client.util.ImageUtils
import com.mesh.client.util.NotificationHelper
import android.net.Uri
import kotlinx.serialization.json.JsonObject
import com.mesh.client.wear.WearSyncManager
import dagger.hilt.android.AndroidEntryPoint
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.serialization.json.boolean
import kotlinx.serialization.json.float
import kotlinx.serialization.json.floatOrNull
import kotlinx.serialization.json.booleanOrNull
import kotlinx.serialization.json.double
import kotlinx.serialization.json.doubleOrNull
import kotlinx.serialization.json.int
import kotlinx.serialization.json.intOrNull
import kotlinx.serialization.json.long
import kotlinx.serialization.json.longOrNull
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import javax.inject.Inject

/**
 * Foreground service that maintains the mesh connection.
 *
 * This service:
 * - Keeps a persistent WebSocket connection to the mesh router
 * - Handles incoming messages and routes them appropriately
 * - Shows notifications for messages when the app is backgrounded
 * - Survives app process termination
 */
@AndroidEntryPoint
class MeshService : Service() {

    private val tag = "MeshService"

    @Inject lateinit var socketClient: MeshSocketClient
    @Inject lateinit var notificationHelper: NotificationHelper
    @Inject lateinit var database: MeshDatabase
    @Inject lateinit var confirmationManager: ConfirmationManager
    @Inject lateinit var statusManager: StatusManager
    @Inject lateinit var usageManager: UsageManager
    @Inject lateinit var wearSyncManager: WearSyncManager

    private val serviceScope = CoroutineScope(SupervisorJob() + Dispatchers.IO)

    // Binder for local binding
    private val binder = LocalBinder()

    // Current target for sending messages
    private val _currentTarget = MutableStateFlow<String?>(null)
    val currentTarget: StateFlow<String?> = _currentTarget.asStateFlow()

    // Whether the app is in foreground (affects notification behavior)
    private var isAppInForeground = false

    // Connection config
    private var nodeId: String? = null

    inner class LocalBinder : Binder() {
        fun getService(): MeshService = this@MeshService
    }

    override fun onBind(intent: Intent?): IBinder = binder

    override fun onCreate() {
        super.onCreate()
        Log.i(tag, "MeshService created")

        notificationHelper.createChannels()

        // Start as foreground service
        val notification = notificationHelper.buildServiceNotification(isConnected = false)
        ServiceCompat.startForeground(
            this,
            NotificationHelper.NOTIFICATION_ID_SERVICE,
            notification,
            android.content.pm.ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC
        )

        // Set up Wear OS callbacks (WearSyncListenerService handles message reception)
        wearSyncManager.setReplyCallback { _, replyTo, content ->
            serviceScope.launch {
                Log.d(tag, "Sending reply from watch: to=$replyTo, content=$content")
                sendMessageTo(replyTo, content)
            }
        }
        wearSyncManager.setSendMessageCallback { to, content ->
            serviceScope.launch {
                Log.d(tag, "Sending message from watch: to=$to, content=$content")
                sendMessageTo(to, content)
            }
        }

        // Observe connection state
        serviceScope.launch {
            socketClient.connectionState.collect { state ->
                updateServiceNotification(state)
                // Clear stale roster entries when we connect
                // The server will send PRESENCE join events for all currently online nodes
                if (state is ConnectionState.Connected) {
                    Log.i(tag, "Connected - marking all roster entries as offline")
                    database.rosterDao().markAllOffline()
                    // Request history sync on connect
                    requestHistorySync()
                    // Request node status after a brief delay (let PRESENCE messages arrive first)
                    kotlinx.coroutines.delay(2000)
                    requestNodeStatus()
                }
            }
        }

        // Handle incoming messages
        serviceScope.launch {
            socketClient.incomingMessages.collect { message ->
                handleIncomingMessage(message)
            }
        }
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        Log.i(tag, "MeshService onStartCommand")
        return START_STICKY
    }

    override fun onDestroy() {
        Log.i(tag, "MeshService destroyed")
        serviceScope.cancel()
        socketClient.disconnect()
        super.onDestroy()
    }

    /**
     * Connect to the mesh router.
     */
    fun connect(host: String, port: Int, nodeId: String, nickname: String, authToken: String? = null, useTls: Boolean = false) {
        this.nodeId = nodeId
        // Tell WearSyncManager our node ID so it can include it in syncs
        wearSyncManager.setUserNodeId(nodeId)
        socketClient.connect(host, port, nodeId, nickname, authToken, useTls)
    }

    /**
     * Disconnect from the mesh router.
     */
    fun disconnect() {
        socketClient.disconnect()
    }

    /**
     * Send a message to the current target.
     */
    fun sendMessage(content: String): Boolean {
        val target = _currentTarget.value
        Log.d(tag, "sendMessage: content='$content', target=$target")
        if (target == null) {
            Log.w(tag, "sendMessage: no current target set")
            return false
        }
        return sendMessageTo(target, content)
    }

    /**
     * Send a message to a specific target.
     */
    fun sendMessageTo(target: String, content: String): Boolean {
        Log.d(tag, "sendMessageTo: target=$target, content='$content'")

        val myNodeId = nodeId
        if (myNodeId == null) {
            Log.w(tag, "sendMessageTo: nodeId is null, cannot send message")
            return false
        }

        val messageId = "local-${System.currentTimeMillis()}"
        // For channels, use the channel address as conversation ID
        // For direct messages, use sorted pair of node IDs
        val convId = if (isChannelAddress(target)) {
            target
        } else {
            MessageEntity.computeConversationId(myNodeId, target)
        }

        // Store message immediately with PENDING status
        serviceScope.launch {
            Log.d(tag, "sendMessageTo: storing outgoing message as PENDING, fromNode=$myNodeId, toNode=$target, convId=$convId")
            val entity = MessageEntity(
                id = messageId,
                fromNode = myNodeId,
                toNode = target,
                type = MessageType.MESSAGE.value,
                content = content,
                timestamp = nowIso(),
                conversationId = convId,
                isOutgoing = true,
                isRead = true,
                status = MessageStatus.PENDING
            )
            database.messageDao().insert(entity)
        }

        // Send the message
        val success = socketClient.sendMessage(target, content)
        Log.d(tag, "sendMessageTo: socketClient.sendMessage returned $success")

        // Update status based on result
        serviceScope.launch {
            val newStatus = if (success) MessageStatus.SENT else MessageStatus.FAILED
            Log.d(tag, "sendMessageTo: updating message status to $newStatus")
            database.messageDao().updateStatus(messageId, newStatus)
        }

        return success
    }

    /**
     * Send an image to a specific target.
     *
     * @param target Target node ID
     * @param imageUri URI of the image to send
     * @param caption Optional text caption for the image
     * @return Result indicating success or failure with error message
     */
    suspend fun sendImageTo(target: String, imageUri: Uri, caption: String? = null): Result<String> {
        val myNodeId = nodeId
        if (myNodeId == null) {
            Log.w(tag, "sendImageTo: nodeId is null, cannot send image")
            return Result.failure(IllegalStateException("Not connected"))
        }

        return try {
            // Process image (resize, compress, generate thumbnail)
            Log.d(tag, "sendImageTo: processing image from $imageUri")
            val processed = ImageUtils.processImageForSending(applicationContext, imageUri)

            // Check size limit (2MB max)
            if (processed.data.size > 2 * 1024 * 1024) {
                Log.w(tag, "sendImageTo: image too large after compression (${processed.data.size} bytes)")
                return Result.failure(IllegalArgumentException("Image too large (${processed.data.size / 1024}KB), max 2MB"))
            }

            Log.d(tag, "sendImageTo: image processed - ${processed.width}x${processed.height}, ${processed.data.size} bytes")

            // Create image message
            val message = MessageFactory.makeImageMessage(
                fromNode = myNodeId,
                toNode = target,
                imageData = processed.data,
                mimeType = processed.mimeType,
                thumbnail = processed.thumbnail,
                width = processed.width,
                height = processed.height,
                caption = caption
            )

            // Compute conversation ID
            val convId = if (isChannelAddress(target)) {
                target
            } else {
                MessageEntity.computeConversationId(myNodeId, target)
            }

            // Store message locally with PENDING status
            val contentJson = Message.toJson(message).let { json ->
                // Extract just the content portion for storage
                try {
                    val parsed = kotlinx.serialization.json.Json.parseToJsonElement(json)
                    parsed.jsonObject["content"]?.toString() ?: "[Image]"
                } catch (e: Exception) {
                    "[Image]"
                }
            }

            val entity = MessageEntity(
                id = message.id,
                fromNode = myNodeId,
                toNode = target,
                type = MessageType.MESSAGE.value,
                content = contentJson,
                timestamp = message.timestamp,
                conversationId = convId,
                isOutgoing = true,
                isRead = true,
                status = MessageStatus.PENDING
            )
            database.messageDao().insert(entity)

            // Send via socket
            val success = socketClient.send(message)
            Log.d(tag, "sendImageTo: socketClient.send returned $success")

            // Update status based on result
            val newStatus = if (success) MessageStatus.SENT else MessageStatus.FAILED
            database.messageDao().updateStatus(message.id, newStatus)

            if (success) {
                Result.success(message.id)
            } else {
                Result.failure(Exception("Failed to send image"))
            }
        } catch (e: Exception) {
            Log.e(tag, "sendImageTo: failed", e)
            Result.failure(e)
        }
    }

    /**
     * Set the current target for messages.
     */
    fun setTarget(target: String?) {
        Log.d(tag, "setTarget: target=$target")
        _currentTarget.value = target

        // Clear any notifications from this node when we open the conversation
        if (target != null) {
            notificationHelper.cancelNotificationsForNode(target)
        }
    }

    /**
     * Send a confirmation response.
     */
    fun sendConfirmResponse(toNode: String, inReplyTo: String, confirmed: Boolean): Boolean {
        return socketClient.sendConfirmResponse(toNode, inReplyTo, confirmed)
    }

    /**
     * Request status from an agent.
     */
    fun requestStatus(target: String, numMessages: Int = 5): Boolean {
        statusManager.setLoading(true)
        return socketClient.sendStatusRequest(target, numMessages)
    }

    // --- Message deletion ---

    /**
     * Delete a message (locally and sync to server).
     */
    fun deleteMessage(messageId: String, conversationId: String): Boolean {
        return socketClient.deleteMessage(messageId, conversationId)
    }

    // --- Channel operations ---

    /**
     * Create a new channel.
     */
    fun createChannel(name: String, description: String = ""): Boolean {
        return socketClient.createChannel(name, description)
    }

    /**
     * Delete a channel.
     */
    fun deleteChannel(name: String): Boolean {
        return socketClient.deleteChannel(name)
    }

    /**
     * Join a channel.
     */
    fun joinChannel(name: String): Boolean {
        return socketClient.joinChannel(name)
    }

    /**
     * Leave a channel.
     */
    fun leaveChannel(name: String): Boolean {
        return socketClient.leaveChannel(name)
    }

    /**
     * Request list of all channels.
     */
    fun listChannels(): Boolean {
        return socketClient.listChannels()
    }

    /**
     * Request members of a channel.
     */
    fun getChannelMembers(name: String): Boolean {
        return socketClient.getChannelMembers(name)
    }

    /**
     * Set whether the app is in foreground.
     */
    fun setAppInForeground(inForeground: Boolean) {
        isAppInForeground = inForeground
    }

    private fun updateServiceNotification(state: ConnectionState) {
        val isConnected = state is ConnectionState.Connected
        val notification = notificationHelper.buildServiceNotification(isConnected)
        notificationHelper.cancel(NotificationHelper.NOTIFICATION_ID_SERVICE)

        try {
            val manager = getSystemService(NOTIFICATION_SERVICE) as android.app.NotificationManager
            manager.notify(NotificationHelper.NOTIFICATION_ID_SERVICE, notification)
        } catch (e: Exception) {
            Log.e(tag, "Failed to update notification", e)
        }
    }

    private suspend fun handleIncomingMessage(message: Message) {
        Log.d(tag, "Handling message: ${message.type} from ${message.fromNode}")

        when (message.type) {
            MessageType.MESSAGE -> handleChatMessage(message)
            MessageType.PRESENCE -> handlePresence(message)
            MessageType.CONFIRM_REQUEST -> handleConfirmRequest(message)
            MessageType.STATUS_RESPONSE -> handleStatusResponse(message)
            MessageType.CONTROL -> handleControlMessage(message)
            else -> Log.d(tag, "Unhandled message type: ${message.type}")
        }
    }

    private suspend fun handleChatMessage(message: Message) {
        Log.d(tag, "handleChatMessage: from=${message.fromNode}, to=${message.toNode}, content=${message.content}")
        val myNodeId = nodeId
        if (myNodeId == null) {
            Log.w(tag, "handleChatMessage: nodeId is null, ignoring message")
            return
        }
        val content = message.contentAsString()
        if (content == null) {
            Log.w(tag, "handleChatMessage: content is null or not a string")
            return
        }

        // Check if this is an echo of our own message (sent from another device)
        val isEcho = message.fromNode == myNodeId

        // Determine conversation ID - for channels, use the channel address
        val convId = if (isChannelAddress(message.toNode)) {
            // Channel message: conversation ID is the channel address
            message.toNode
        } else {
            // Direct message: sorted pair of node IDs
            // For echo messages, the "other" party is the toNode (recipient)
            // For received messages, the "other" party is the fromNode (sender)
            val otherParty = if (isEcho) message.toNode else message.fromNode
            MessageEntity.computeConversationId(myNodeId, otherParty)
        }
        Log.d(tag, "handleChatMessage: storing message, myNodeId=$myNodeId, fromNode=${message.fromNode}, isEcho=$isEcho, convId=$convId")

        // Normalize timestamp to UTC for consistent sorting
        val normalizedTimestamp = normalizeToUtc(message.timestamp)

        // Store in database
        val entity = MessageEntity(
            id = message.id,
            fromNode = message.fromNode,
            toNode = message.toNode,
            type = message.type.value,
            content = content,
            timestamp = normalizedTimestamp,
            inReplyTo = message.inReplyTo,
            conversationId = convId,
            isOutgoing = isEcho,  // Echo messages are outgoing (sent by us from another device)
            isRead = isAppInForeground || isEcho  // Echo messages are always read (we sent them)
        )
        database.messageDao().insert(entity)
        Log.d(tag, "handleChatMessage: message stored successfully")

        // Show notification if app is not in foreground and it's not our own message
        if (!isAppInForeground && !isEcho) {
            val channelName = parseChannelName(message.toNode)
            val notifFrom = if (channelName != null) {
                "#$channelName: ${message.fromNode}"
            } else {
                message.fromNode
            }
            // Use the conversation target as the key so cancelNotificationsForNode matches
            // For channels: "channel:name", for DMs: sender's node ID
            val conversationKey = if (channelName != null) message.toNode else message.fromNode
            notificationHelper.showMessageNotification(notifFrom, content, conversationKey)
        }
    }

    private suspend fun handlePresence(message: Message) {
        Log.d(tag, "handlePresence: from=${message.fromNode}, content=${message.content}")
        val content = message.contentAsObject()
        if (content == null) {
            Log.w(tag, "handlePresence: content is null or not an object")
            return
        }
        val event = content["event"]?.jsonPrimitive?.content
        if (event == null) {
            Log.w(tag, "handlePresence: event is null")
            return
        }

        // Check if this is a channel presence event
        val channelName = content["channel"]?.jsonPrimitive?.content
        if (channelName != null) {
            handleChannelPresence(event, channelName, message)
            return
        }

        // Node presence
        val nickname = content["nickname"]?.jsonPrimitive?.content
        if (nickname == null) {
            Log.w(tag, "handlePresence: nickname is null")
            return
        }
        val nodeType = content["node_type"]?.jsonPrimitive?.content
        if (nodeType == null) {
            Log.w(tag, "handlePresence: nodeType is null")
            return
        }
        val description = content["description"]?.jsonPrimitive?.content ?: ""
        val llmBackend = content["llm_backend"]?.jsonPrimitive?.content ?: ""
        val llmModel = content["llm_model"]?.jsonPrimitive?.content ?: ""
        val hostname = content["hostname"]?.jsonPrimitive?.content ?: ""

        Log.i(tag, "Presence: event=$event, node=${message.fromNode}, nickname=$nickname, type=$nodeType, backend=$llmBackend, model=$llmModel, host=$hostname")

        when (event) {
            "join" -> {
                val entry = RosterEntry(
                    nodeId = message.fromNode,
                    nickname = nickname,
                    nodeType = nodeType,
                    description = description,
                    isOnline = true,
                    lastSeen = message.timestamp,
                    llmBackend = llmBackend,
                    llmModel = llmModel,
                    hostname = hostname
                )
                database.rosterDao().insert(entry)
                Log.i(tag, "Roster: added/updated ${message.fromNode} as online")
            }
            "leave" -> {
                database.rosterDao().markOffline(message.fromNode)
                Log.i(tag, "Roster: marked ${message.fromNode} as offline")
            }
        }
    }

    private suspend fun handleChannelPresence(event: String, channelName: String, message: Message) {
        Log.i(tag, "Channel presence: event=$event, channel=$channelName, node=${message.fromNode}")

        when (event) {
            "join" -> {
                // A node joined the channel
                val member = ChannelMemberEntity(
                    channelName = channelName,
                    nodeId = message.fromNode,
                    joinedAt = message.timestamp
                )
                database.channelDao().insertMember(member)

                // Update member count
                val count = database.channelDao().getMemberCount(channelName)
                database.channelDao().updateMemberCount(channelName, count)
            }
            "leave" -> {
                // A node left the channel
                database.channelDao().removeMember(channelName, message.fromNode)

                // Update member count
                val count = database.channelDao().getMemberCount(channelName)
                database.channelDao().updateMemberCount(channelName, count)
            }
        }
    }

    private suspend fun handleControlMessage(message: Message) {
        if (message.fromNode != "router") {
            Log.d(tag, "handleControlMessage: ignoring non-router control message")
            return
        }

        val content = message.contentAsObject() ?: return
        val action = content["action"]?.jsonPrimitive?.content ?: return
        val status = content["status"]?.let {
            if (it is kotlinx.serialization.json.JsonPrimitive) it.content else null
        }

        Log.d(tag, "Control message: action=$action, status=$status")

        when (action) {
            ControlAction.CHANNEL_CREATE.value,
            ControlAction.CHANNEL_DELETE.value,
            ControlAction.CHANNEL_JOIN.value,
            ControlAction.CHANNEL_LEAVE.value -> handleChannelAck(action, status, content)
            ControlAction.CHANNEL_INVITE.value -> handleChannelInviteAck(action, status, content)
            ControlAction.CHANNEL_LIST.value -> handleChannelList(content)
            ControlAction.CHANNEL_MEMBERS.value -> handleChannelMembersList(content)
            ControlAction.HISTORY_RESPONSE.value -> handleHistoryResponse(content)
            ControlAction.LIST_NODES.value -> handleListNodes(content)
            ControlAction.CC_USAGE.value -> handleCcUsageResponse(content)
        }
    }

    /**
     * Handle LIST_NODES response from router.
     * Updates roster entries with heartbeat-lite status data and online/offline state.
     * The router includes a "nodes" list of currently connected node IDs — use it
     * to reconcile isOnline, fixing drift from missed PRESENCE events.
     */
    private suspend fun handleListNodes(content: kotlinx.serialization.json.JsonObject) {
        // Reconcile online/offline from the authoritative connected-nodes list
        val connectedNodes = content["nodes"]?.jsonArray
            ?.map { it.jsonPrimitive.content }
            ?.toSet()
        if (connectedNodes != null) {
            val roster = database.rosterDao().getAllNodesSnapshot()
            for (entry in roster) {
                val shouldBeOnline = entry.nodeId in connectedNodes
                if (entry.isOnline != shouldBeOnline) {
                    if (shouldBeOnline) {
                        database.rosterDao().markOnline(entry.nodeId, "")
                    } else {
                        database.rosterDao().markOffline(entry.nodeId)
                    }
                    Log.i(tag, "handleListNodes: corrected ${entry.nodeId} online=${shouldBeOnline}")
                }
            }
        }

        // Update heartbeat-lite status fields
        val statusObj = content["status"]?.jsonObject ?: return
        Log.d(tag, "handleListNodes: updating status for ${statusObj.keys.size} nodes")

        for ((nodeId, statusJson) in statusObj) {
            val s = statusJson.jsonObject
            val state = s["state"]?.jsonPrimitive?.content ?: ""
            val contextTokens = s["context_tokens"]?.jsonPrimitive?.intOrNull ?: 0
            val workerElapsed = s["worker_elapsed_s"]?.jsonPrimitive?.floatOrNull
            val historyPct = s["history_pct"]?.jsonPrimitive?.floatOrNull
            val memPool = s["memory_pool"]?.jsonPrimitive?.intOrNull
            val memActive = s["memory_active"]?.jsonPrimitive?.intOrNull
            val activeMap = s["active_map"]?.jsonPrimitive?.content ?: ""

            database.rosterDao().updateStatus(
                nodeId = nodeId,
                state = state,
                contextTokens = contextTokens,
                workerElapsed = workerElapsed,
                historyPct = historyPct,
                memPool = memPool,
                memActive = memActive,
                activeMap = activeMap,
            )
        }
    }

    private fun handleCcUsageResponse(content: kotlinx.serialization.json.JsonObject) {
        Log.d(tag, "handleCcUsageResponse: content=$content")
        try {
            val accountsJson = content["accounts"]?.jsonArray
            if (accountsJson == null) {
                Log.e(tag, "handleCcUsageResponse: accounts array is null")
                return
            }
            Log.d(tag, "handleCcUsageResponse: parsing ${accountsJson.size} accounts")
            val accounts = accountsJson.map { elem ->
                val obj = elem.jsonObject
                val label = obj["label"]?.jsonPrimitive?.content ?: "unknown"
                val error = obj["error"]?.jsonPrimitive?.content
                val sub = obj["sub"]?.jsonPrimitive?.content ?: "unknown"
                Log.d(tag, "handleCcUsageResponse: account $label, sub=$sub, error=$error")

                if (error != null) {
                    AccountUsage(label = label, subscriptionType = sub, error = error)
                } else {
                    val windows = mutableListOf<UsageWindow>()
                    for ((key, displayName) in listOf(
                        "five_hour" to "5-hour",
                        "seven_day" to "7-day",
                        "seven_day_opus" to "7d-opus",
                        "seven_day_sonnet" to "7d-sonnet"
                    )) {
                        val element = obj[key]
                        if (element == null || element is kotlinx.serialization.json.JsonNull) continue
                        val w = element.jsonObject
                        val util = w["utilization"]?.jsonPrimitive?.doubleOrNull ?: continue
                        val resetsAt = w["resets_at"]?.jsonPrimitive?.content
                        windows.add(UsageWindow(name = displayName, utilization = util, resetsAt = resetsAt))
                    }

                    val extraObj = obj["extra_usage"]?.jsonObject
                    val extra = extraObj?.let {
                        ExtraUsage(
                            isEnabled = it["is_enabled"]?.jsonPrimitive?.booleanOrNull ?: false,
                            usedCredits = it["used_credits"]?.jsonPrimitive?.doubleOrNull ?: 0.0,
                            monthlyLimit = it["monthly_limit"]?.jsonPrimitive?.doubleOrNull ?: 0.0
                        )
                    }

                    AccountUsage(
                        label = label,
                        subscriptionType = sub,
                        windows = windows,
                        extraUsage = extra
                    )
                }
            }
            Log.d(tag, "handleCcUsageResponse: setting usage with ${accounts.size} accounts")
            usageManager.setUsage(CcUsageResponse(accounts = accounts))
        } catch (e: Exception) {
            Log.e(tag, "handleCcUsageResponse: error parsing response", e)
        }
    }

    /**
     * Request CC usage from the router.
     */
    fun requestCcUsage() {
        Log.d(tag, "requestCcUsage: setting loading=true and sending request")
        usageManager.setLoading(true)
        val sent = socketClient.requestCcUsage()
        Log.d(tag, "requestCcUsage: request sent=$sent")
    }

    /**
     * Request node list with status from router.
     * The response is handled by handleListNodes().
     */
    fun requestNodeStatus() {
        val myNodeId = nodeId ?: return
        val msg = MessageFactory.makeControl(
            fromNode = myNodeId,
            action = ControlAction.LIST_NODES,
        )
        socketClient.send(msg)
    }

    private suspend fun handleChannelAck(action: String, status: String?, content: kotlinx.serialization.json.JsonObject) {
        val channelName = content["channel_name"]?.jsonPrimitive?.content ?: return

        if (status == "error") {
            val error = content["error"]?.jsonPrimitive?.content ?: "Unknown error"
            Log.e(tag, "Channel operation failed: $action - $error")
            return
        }

        val myNodeId = nodeId ?: return

        when (action) {
            ControlAction.CHANNEL_CREATE.value -> {
                // Channel created - also auto-joined
                val description = content["description"]?.jsonPrimitive?.content ?: ""
                val channel = ChannelEntity(
                    name = channelName,
                    description = description,
                    createdAt = nowIso(),
                    createdBy = myNodeId,
                    memberCount = 1
                )
                database.channelDao().insertChannel(channel)

                // Add self as member
                val member = ChannelMemberEntity(
                    channelName = channelName,
                    nodeId = myNodeId,
                    joinedAt = nowIso()
                )
                database.channelDao().insertMember(member)
                Log.i(tag, "Channel created and joined: $channelName")
            }
            ControlAction.CHANNEL_DELETE.value -> {
                // Channel deleted
                database.channelDao().deleteChannel(channelName)
                Log.i(tag, "Channel deleted: $channelName")
            }
            ControlAction.CHANNEL_JOIN.value -> {
                // Joined channel - need to fetch channel info if we don't have it
                if (!database.channelDao().channelExists(channelName)) {
                    // Channel doesn't exist locally, create placeholder
                    val channel = ChannelEntity(
                        name = channelName,
                        description = "",
                        createdAt = nowIso(),
                        createdBy = "",
                        memberCount = 1
                    )
                    database.channelDao().insertChannel(channel)
                }

                // Add self as member
                val member = ChannelMemberEntity(
                    channelName = channelName,
                    nodeId = myNodeId,
                    joinedAt = nowIso()
                )
                database.channelDao().insertMember(member)
                Log.i(tag, "Joined channel: $channelName")
            }
            ControlAction.CHANNEL_LEAVE.value -> {
                // Left channel
                database.channelDao().removeMember(channelName, myNodeId)
                Log.i(tag, "Left channel: $channelName")
            }
        }
    }

    private suspend fun handleChannelInviteAck(action: String, status: String?, content: kotlinx.serialization.json.JsonObject) {
        val channelName = content["channel_name"]?.jsonPrimitive?.content ?: return
        val invitedNodeId = content["node_id"]?.jsonPrimitive?.content ?: return

        if (status == "error") {
            val error = content["error"]?.jsonPrimitive?.content ?: "Unknown error"
            Log.e(tag, "Channel invite failed: $error")
            return
        }

        Log.i(tag, "Successfully invited $invitedNodeId to channel $channelName")

        // Refresh channel list to update member count
        socketClient.listChannels()
    }

    private suspend fun handleChannelList(content: kotlinx.serialization.json.JsonObject) {
        val channelsArray = content["channels"]?.jsonArray ?: return
        val myNodeId = nodeId ?: return

        channelsArray.forEach { channelJson ->
            val obj = channelJson.jsonObject
            val name = obj["name"]?.jsonPrimitive?.content ?: return@forEach
            val description = obj["description"]?.jsonPrimitive?.content ?: ""
            val memberCount = obj["member_count"]?.jsonPrimitive?.int ?: 0
            val createdAt = obj["created_at"]?.jsonPrimitive?.content ?: ""
            val createdBy = obj["created_by"]?.jsonPrimitive?.content ?: ""
            val isMember = try {
                obj["is_member"]?.jsonPrimitive?.boolean ?: false
            } catch (e: Exception) {
                false
            }

            val channel = ChannelEntity(
                name = name,
                description = description,
                createdAt = createdAt,
                createdBy = createdBy,
                memberCount = memberCount
            )
            database.channelDao().insertChannel(channel)

            // Update membership based on server response
            Log.d(tag, "Channel '$name': is_member=$isMember, myNodeId=$myNodeId")
            if (isMember) {
                val member = ChannelMemberEntity(
                    channelName = name,
                    nodeId = myNodeId,
                    joinedAt = nowIso()
                )
                database.channelDao().insertMember(member)
            } else {
                database.channelDao().removeMember(name, myNodeId)
            }
        }
        Log.i(tag, "Updated channel list: ${channelsArray.size} channels")
    }

    private suspend fun handleChannelMembersList(content: kotlinx.serialization.json.JsonObject) {
        val channelName = content["channel_name"]?.jsonPrimitive?.content ?: return
        val membersArray = content["members"]?.jsonArray ?: return

        // Clear existing members and re-add
        database.channelDao().deleteAllMembers(channelName)

        membersArray.forEach { memberJson ->
            val obj = memberJson.jsonObject
            val nodeId = obj["node_id"]?.jsonPrimitive?.content ?: return@forEach
            val joinedAt = obj["joined_at"]?.jsonPrimitive?.content ?: ""

            val member = ChannelMemberEntity(
                channelName = channelName,
                nodeId = nodeId,
                joinedAt = joinedAt
            )
            database.channelDao().insertMember(member)
        }

        // Update member count
        database.channelDao().updateMemberCount(channelName, membersArray.size)
        Log.i(tag, "Updated members for channel $channelName: ${membersArray.size} members")
    }

    private fun handleConfirmRequest(message: Message) {
        val content = message.contentAsObject() ?: return
        val toolName = content["tool_name"]?.jsonPrimitive?.content ?: "unknown"
        val preview = content["preview"]?.jsonPrimitive?.content ?: ""

        // Show notification (always, even if app is in foreground, as this needs action)
        val notificationId = notificationHelper.showConfirmNotification(
            fromNode = message.fromNode,
            messageId = message.id,
            toolName = toolName,
            preview = preview
        )

        // Add to confirmation manager for in-app dialog display
        val request = ConfirmationRequest(
            messageId = message.id,
            fromNode = message.fromNode,
            toolName = toolName,
            preview = preview,
            notificationId = notificationId
        )
        confirmationManager.addRequest(request)
    }

    /**
     * Handle a confirmation response (approve/reject).
     * Called from UI or notification action.
     */
    fun handleConfirmation(messageId: String, fromNode: String, approved: Boolean) {
        Log.i(tag, "handleConfirmation: messageId=$messageId, fromNode=$fromNode, approved=$approved")

        // Send response to agent
        sendConfirmResponse(fromNode, messageId, approved)

        // Remove from pending queue
        confirmationManager.removeRequest(messageId)
    }

    private fun handleStatusResponse(message: Message) {
        Log.d(tag, "Status response from ${message.fromNode}")
        val content = message.contentAsObject() ?: run {
            Log.w(tag, "handleStatusResponse: content is null or not an object")
            statusManager.setLoading(false)
            return
        }

        val contextArray = content["context"]?.jsonArray ?: run {
            Log.w(tag, "handleStatusResponse: context is null or not an array")
            statusManager.setLoading(false)
            return
        }

        val contextEntries = contextArray.mapNotNull { entry ->
            val obj = entry.jsonObject
            val from = obj["from"]?.jsonPrimitive?.content ?: return@mapNotNull null
            val msgContent = obj["content"]?.jsonPrimitive?.content ?: return@mapNotNull null
            val timestamp = obj["timestamp"]?.jsonPrimitive?.content ?: ""
            val type = obj["type"]?.jsonPrimitive?.content ?: "message"
            StatusContextEntry(from, msgContent, timestamp, type)
        }

        val summary = content["summary"]?.jsonPrimitive?.content
        val currentActivity = content["current_activity"]?.jsonPrimitive?.content
        val hostname = content["hostname"]?.jsonPrimitive?.content
        val model = content["model"]?.jsonPrimitive?.content
        val backend = content["backend"]?.jsonPrimitive?.content
        val workingDirectory = content["working_directory"]?.jsonPrimitive?.content

        // Parse heartbeat-lite status summary
        val statusSummary = content["status_summary"]?.jsonObject?.let { ss ->
            StatusSummary(
                state = ss["state"]?.jsonPrimitive?.content ?: "",
                workerElapsedS = ss["worker_elapsed_s"]?.jsonPrimitive?.doubleOrNull,
                contextTokens = ss["context_tokens"]?.jsonPrimitive?.intOrNull ?: 0,
                historyTurns = ss["history_turns"]?.jsonPrimitive?.intOrNull ?: 0,
                historyPct = ss["history_pct"]?.jsonPrimitive?.doubleOrNull ?: 0.0,
                memoryPool = ss["memory_pool"]?.jsonPrimitive?.intOrNull ?: 0,
                memoryActive = ss["memory_active"]?.jsonPrimitive?.intOrNull ?: 0,
                uptimeS = ss["uptime_s"]?.jsonPrimitive?.doubleOrNull ?: 0.0,
                activeMap = ss["active_map"]?.jsonPrimitive?.content
            )
        }

        // Parse full diagnostics report
        val diagnostics = content["diagnostics"]?.jsonObject?.let { diag ->
            fun jsonObjToMap(obj: kotlinx.serialization.json.JsonObject): Map<String, Any?> {
                return obj.entries.associate { (k, v) ->
                    k to when {
                        v is kotlinx.serialization.json.JsonNull -> null
                        v is kotlinx.serialization.json.JsonPrimitive -> {
                            v.intOrNull ?: v.longOrNull ?: v.doubleOrNull ?: v.booleanOrNull ?: v.content
                        }
                        v is kotlinx.serialization.json.JsonObject -> jsonObjToMap(v)
                        v is kotlinx.serialization.json.JsonArray -> v.map { elem ->
                            when (elem) {
                                is kotlinx.serialization.json.JsonObject -> jsonObjToMap(elem)
                                is kotlinx.serialization.json.JsonPrimitive -> elem.content
                                else -> elem.toString()
                            }
                        }
                        else -> v.toString()
                    }
                }
            }
            DiagnosticReport(
                identity = diag["identity"]?.jsonObject?.let { jsonObjToMap(it) },
                llm = diag["llm"]?.jsonObject?.let { jsonObjToMap(it) },
                router = diag["router"]?.jsonObject?.let { jsonObjToMap(it) },
                history = diag["history"]?.jsonObject?.let { jsonObjToMap(it) },
                memory = diag["memory"]?.jsonObject?.let { jsonObjToMap(it) },
                contextHealth = diag["context_health"]?.jsonObject?.let { jsonObjToMap(it) }
            )
        }

        val statusResponse = StatusResponse(
            fromNode = message.fromNode,
            context = contextEntries,
            summary = summary,
            currentActivity = currentActivity,
            hostname = hostname,
            model = model,
            backend = backend,
            workingDirectory = workingDirectory,
            statusSummary = statusSummary,
            diagnostics = diagnostics
        )

        statusManager.setStatus(statusResponse)
    }

    // --- Message History Sync ---

    private fun getLastSyncTimestamp(): String? {
        val prefs = getSharedPreferences("mesh_sync", MODE_PRIVATE)
        return prefs.getString("last_sync_timestamp", null)
    }

    private fun setLastSyncTimestamp(timestamp: String) {
        val prefs = getSharedPreferences("mesh_sync", MODE_PRIVATE)
        prefs.edit().putString("last_sync_timestamp", timestamp).apply()
    }

    /**
     * Request message history sync from the server.
     */
    fun requestHistorySync() {
        val since = getLastSyncTimestamp()
        Log.i(tag, "Requesting history sync since: $since")
        socketClient.requestHistorySync(since = since)
    }

    private suspend fun handleHistoryResponse(content: kotlinx.serialization.json.JsonObject) {
        val messagesArray = content["messages"]?.jsonArray ?: return
        val readReceipts = content["read_receipts"]?.jsonObject
        val hasMore = content["has_more"]?.jsonPrimitive?.boolean ?: false

        val myNodeId = nodeId ?: return

        Log.i(tag, "History sync received: ${messagesArray.size} messages")

        // Collect unique nodes we've communicated with to add to roster
        val seenNodes = mutableSetOf<String>()

        // Process and insert messages
        var latestTimestamp: String? = null
        for (msgJson in messagesArray) {
            val obj = msgJson.jsonObject
            val id = obj["id"]?.jsonPrimitive?.content ?: continue
            val fromNode = obj["from_node"]?.jsonPrimitive?.content ?: continue
            val toNode = obj["to_node"]?.jsonPrimitive?.content ?: continue
            val type = obj["type"]?.jsonPrimitive?.content ?: continue
            val msgContent = obj["content"]?.jsonPrimitive?.content ?: ""
            val timestamp = obj["timestamp"]?.jsonPrimitive?.content ?: continue
            val inReplyTo = obj["in_reply_to"]?.jsonPrimitive?.content

            // Only sync MESSAGE type
            if (type != MessageType.MESSAGE.value) continue

            // Track nodes for roster (skip channels and self)
            if (!isChannelAddress(fromNode) && fromNode != myNodeId) {
                seenNodes.add(fromNode)
            }
            if (!isChannelAddress(toNode) && toNode != myNodeId) {
                seenNodes.add(toNode)
            }

            // Compute conversation ID
            val convId = if (isChannelAddress(toNode)) {
                toNode
            } else {
                MessageEntity.computeConversationId(myNodeId, if (fromNode == myNodeId) toNode else fromNode)
            }

            // Normalize timestamp to UTC for consistent sorting
            val normalizedTimestamp = normalizeToUtc(timestamp)

            val entity = MessageEntity(
                id = id,
                fromNode = fromNode,
                toNode = toNode,
                type = type,
                content = msgContent,
                timestamp = normalizedTimestamp,
                inReplyTo = inReplyTo,
                conversationId = convId,
                isOutgoing = fromNode == myNodeId,
                isRead = true,  // Synced messages are considered read
                status = MessageStatus.SENT
            )

            // Insert if not already present (IGNORE conflict strategy)
            database.messageDao().insertIfNotExists(entity)

            // Track latest timestamp
            if (latestTimestamp == null || timestamp > latestTimestamp) {
                latestTimestamp = timestamp
            }
        }

        // Add seen nodes to roster (if not already present)
        for (nodeIdStr in seenNodes) {
            val existing = database.rosterDao().getByNodeId(nodeIdStr)
            if (existing == null) {
                // Parse node type from ID (e.g., "agent:researcher" -> "agent")
                val isAgent = nodeIdStr.startsWith("agent:")
                val nickname = nodeIdStr.substringAfter(":")
                val entry = RosterEntry(
                    nodeId = nodeIdStr,
                    nickname = nickname,
                    nodeType = if (isAgent) "agent" else "user",
                    description = "",
                    isOnline = false,  // Will be updated by PRESENCE events
                    lastSeen = ""
                )
                database.rosterDao().insert(entry)
                Log.i(tag, "Added $nodeIdStr to roster from history")
            }
        }

        // Update last sync timestamp
        if (latestTimestamp != null) {
            setLastSyncTimestamp(latestTimestamp)
            Log.i(tag, "Updated last sync timestamp to: $latestTimestamp")
        } else {
            // No messages, update to current time
            setLastSyncTimestamp(nowIso())
        }

        // Process read receipts
        readReceipts?.let { receipts ->
            for ((conversationId, timestampElement) in receipts) {
                val readTimestamp = timestampElement.jsonPrimitive.content
                Log.d(tag, "Read receipt: $conversationId up to $readTimestamp")
                // Could update local read state here if needed
            }
        }

        Log.i(tag, "History sync complete: ${messagesArray.size} messages processed")

        // TODO: If hasMore, request more history
    }

    /**
     * Mark messages as read and sync to server.
     */
    fun markReadAndSync(conversationId: String, upToTimestamp: String) {
        serviceScope.launch {
            database.messageDao().markConversationAsRead(conversationId)
        }
        socketClient.markRead(conversationId, upToTimestamp)
    }
}
