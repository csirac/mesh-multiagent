package com.mesh.client.wear

import android.content.Context
import android.util.Log
import com.google.android.gms.wearable.DataClient
import com.google.android.gms.wearable.MessageClient
import com.google.android.gms.wearable.MessageEvent
import com.google.android.gms.wearable.NodeClient
import com.google.android.gms.wearable.PutDataMapRequest
import com.google.android.gms.wearable.Wearable
import com.google.android.gms.wearable.WearableListenerService
import com.mesh.client.data.local.db.MessageDao
import dagger.hilt.EntryPoint
import dagger.hilt.InstallIn
import dagger.hilt.android.EntryPointAccessors
import dagger.hilt.android.qualifiers.ApplicationContext
import dagger.hilt.components.SingletonComponent
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.launch
import kotlinx.coroutines.tasks.await
import kotlinx.serialization.Serializable
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import java.time.Instant
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Service that listens for messages from the watch and syncs data.
 *
 * Note: WearableListenerService requires manual Hilt injection via EntryPoint
 * because it's instantiated by the system, not by Hilt.
 * Do NOT use @AndroidEntryPoint here - it doesn't work with WearableListenerService.
 */
class WearSyncListenerService : WearableListenerService() {

    companion object {
        private const val TAG = "WearSyncListener"
        const val PATH_SYNC_REQUEST = "/mesh/sync_request"
        const val PATH_REPLY = "/mesh/reply"
        const val PATH_MARK_READ = "/mesh/mark_read"
        const val PATH_SEND_MESSAGE = "/mesh/send_message"
    }

    // Use EntryPoint for injection since WearableListenerService is system-instantiated
    @EntryPoint
    @InstallIn(SingletonComponent::class)
    interface WearSyncEntryPoint {
        fun wearSyncManager(): WearSyncManager
    }

    private lateinit var wearSyncManager: WearSyncManager
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)

    override fun onCreate() {
        super.onCreate()
        // Manual injection via EntryPoint
        val entryPoint = EntryPointAccessors.fromApplication(
            applicationContext,
            WearSyncEntryPoint::class.java
        )
        wearSyncManager = entryPoint.wearSyncManager()
        Log.d(TAG, "WearSyncListenerService created, wearSyncManager injected")
    }

    override fun onMessageReceived(messageEvent: MessageEvent) {
        Log.d(TAG, "!!! Message received from watch: path=${messageEvent.path}")

        scope.launch {
            try {
                when (messageEvent.path) {
                    PATH_SYNC_REQUEST -> {
                        Log.d(TAG, "Handling sync request from watch")
                        wearSyncManager.syncMessagesToWatch()
                    }
                    PATH_REPLY -> {
                        val data = String(messageEvent.data, Charsets.UTF_8)
                        Log.d(TAG, "Handling reply from watch: $data")
                        wearSyncManager.handleWatchReply(data)
                    }
                    PATH_MARK_READ -> {
                        val messageId = String(messageEvent.data, Charsets.UTF_8)
                        Log.d(TAG, "Handling mark as read from watch: $messageId")
                        wearSyncManager.handleMarkAsRead(messageId)
                    }
                    PATH_SEND_MESSAGE -> {
                        val data = String(messageEvent.data, Charsets.UTF_8)
                        Log.d(TAG, "Handling send message from watch: $data")
                        wearSyncManager.handleWatchSendMessage(data)
                    }
                    else -> {
                        Log.w(TAG, "Unknown message path: ${messageEvent.path}")
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error handling watch message", e)
            }
        }
    }
}

/**
 * Manager for syncing data to the watch.
 */
@Singleton
class WearSyncManager @Inject constructor(
    @ApplicationContext private val context: Context,
    private val messageDao: MessageDao
) {
    companion object {
        private const val TAG = "WearSyncManager"
        private const val PATH_MESSAGES = "/mesh/messages"
        private const val PATH_CONNECTION = "/mesh/connection"
        private const val MESSAGES_PER_CONVERSATION = 20
    }

    private val json = Json { ignoreUnknownKeys = true }
    private val dataClient: DataClient = Wearable.getDataClient(context)
    private val messageClient: MessageClient = Wearable.getMessageClient(context)
    private val nodeClient: NodeClient = Wearable.getNodeClient(context)

    private var onReplyCallback: ((String, String, String) -> Unit)? = null
    private var onSendMessageCallback: ((String, String) -> Unit)? = null

    // The current user's node ID (e.g., "user:yourname")
    private var userNodeId: String? = null

    /**
     * Set the user's node ID for syncing to watch.
     */
    fun setUserNodeId(nodeId: String) {
        userNodeId = nodeId
        Log.d(TAG, "User node ID set to: $nodeId")
    }

    /**
     * Set callback for when a reply is received from the watch.
     * Callback params: originalMessageId, replyTo, content
     */
    fun setReplyCallback(callback: (String, String, String) -> Unit) {
        onReplyCallback = callback
    }

    /**
     * Set callback for when a new message is sent from the watch.
     * Callback params: to, content
     */
    fun setSendMessageCallback(callback: (String, String) -> Unit) {
        onSendMessageCallback = callback
    }

    /**
     * Sync recent messages to the watch.
     */
    suspend fun syncMessagesToWatch() {
        try {
            Log.d(TAG, ">>> syncMessagesToWatch called, userNodeId=$userNodeId")

            val connectedNodes = nodeClient.connectedNodes.await()
            Log.d(TAG, ">>> Found ${connectedNodes.size} connected nodes")
            if (connectedNodes.isEmpty()) {
                Log.d(TAG, ">>> No watch connected, returning")
                return
            }

            // Ensure we have the user's node ID - fall back to detecting from outgoing messages
            var effectiveUserNodeId = userNodeId
            if (effectiveUserNodeId == null) {
                Log.w(TAG, ">>> userNodeId not set, attempting to detect from DB")
                effectiveUserNodeId = messageDao.getCurrentUserNodeId()
                if (effectiveUserNodeId != null) {
                    Log.d(TAG, ">>> Detected userNodeId from DB: $effectiveUserNodeId")
                    userNodeId = effectiveUserNodeId  // Cache it for next time
                } else {
                    Log.w(TAG, ">>> Could not detect userNodeId from DB")
                }
            }

            // Get all conversations (one preview message per conversation)
            val conversationPreviews = messageDao.getConversationPreviews().first()
            Log.d(TAG, ">>> Found ${conversationPreviews.size} conversation previews")

            // For each conversation, get last N messages
            val allMessages = mutableListOf<WearMessageDto>()
            for (preview in conversationPreviews) {
                Log.d(TAG, ">>> Getting messages for conversation: ${preview.conversationId}")
                val conversationMessages = messageDao.getRecentConversationMessages(
                    preview.conversationId,
                    MESSAGES_PER_CONVERSATION
                )
                Log.d(TAG, ">>> Found ${conversationMessages.size} messages for ${preview.conversationId}")
                allMessages.addAll(conversationMessages.map { entity ->
                    WearMessageDto(
                        id = entity.id,
                        fromNode = entity.fromNode,
                        toNode = entity.toNode,
                        content = entity.content,
                        timestamp = parseTimestamp(entity.timestamp),
                        isRead = entity.isRead
                    )
                })
            }

            val wearMessages = allMessages
            Log.d(TAG, ">>> Total messages to sync: ${wearMessages.size}")

            // Log first few messages for debugging
            wearMessages.take(5).forEachIndexed { idx, msg ->
                Log.d(TAG, ">>> MSG[$idx]: from=${msg.fromNode}, to=${msg.toNode}, content=${msg.content.take(30)}")
            }

            val messagesJson = json.encodeToString(wearMessages)
            Log.d(TAG, ">>> JSON size: ${messagesJson.length} chars")

            // Send via Data Layer
            val dataRequest = PutDataMapRequest.create(PATH_MESSAGES).apply {
                dataMap.putString("messages", messagesJson)
                dataMap.putLong("lastSync", System.currentTimeMillis())
                // Include user's node ID so watch knows which messages are outgoing
                if (effectiveUserNodeId != null) {
                    dataMap.putString("userNodeId", effectiveUserNodeId)
                    Log.d(TAG, ">>> Including userNodeId in dataMap: $effectiveUserNodeId")
                } else {
                    Log.w(TAG, ">>> WARNING: userNodeId is NULL, watch won't know which messages are outgoing!")
                }
            }
            dataRequest.setUrgent()

            dataClient.putDataItem(dataRequest.asPutDataRequest()).await()
            Log.d(TAG, ">>> Synced ${wearMessages.size} messages to watch successfully")

            // Update connection status
            syncConnectionStatus(true)

        } catch (e: Exception) {
            Log.e(TAG, ">>> EXCEPTION in syncMessagesToWatch", e)
        }
    }

    /**
     * Sync connection status to watch.
     */
    suspend fun syncConnectionStatus(isConnected: Boolean) {
        try {
            val dataRequest = PutDataMapRequest.create(PATH_CONNECTION).apply {
                dataMap.putBoolean("connected", isConnected)
                dataMap.putLong("lastSync", System.currentTimeMillis())
            }
            dataRequest.setUrgent()

            dataClient.putDataItem(dataRequest.asPutDataRequest()).await()
        } catch (e: Exception) {
            Log.e(TAG, "Failed to sync connection status", e)
        }
    }

    /**
     * Send a new message notification to the watch.
     */
    suspend fun notifyNewMessage(message: WearMessageDto) {
        try {
            val connectedNodes = nodeClient.connectedNodes.await()
            val watchNode = connectedNodes.firstOrNull() ?: return

            val messageJson = json.encodeToString(message)

            messageClient.sendMessage(
                watchNode.id,
                PATH_MESSAGES,
                messageJson.toByteArray(Charsets.UTF_8)
            ).await()

            Log.d(TAG, "Notified watch of new message: ${message.id}")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to notify watch", e)
        }
    }

    /**
     * Handle a reply from the watch.
     */
    suspend fun handleWatchReply(replyJson: String) {
        try {
            val reply = json.decodeFromString<WatchReply>(replyJson)
            Log.d(TAG, "Reply from watch: ${reply.content} to ${reply.replyTo}")

            onReplyCallback?.invoke(reply.originalMessageId, reply.replyTo, reply.content)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to parse watch reply", e)
        }
    }

    /**
     * Handle mark as read from watch.
     */
    suspend fun handleMarkAsRead(messageId: String) {
        try {
            messageDao.markAsRead(messageId)
            Log.d(TAG, "Marked message as read: $messageId")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to mark as read", e)
        }
    }

    /**
     * Handle a new message sent from the watch.
     */
    suspend fun handleWatchSendMessage(messageJson: String) {
        try {
            val message = json.decodeFromString<WatchSendMessage>(messageJson)
            Log.d(TAG, "Send message from watch: ${message.content} to ${message.to}")

            onSendMessageCallback?.invoke(message.to, message.content)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to parse watch send message", e)
        }
    }

    /**
     * Parse ISO timestamp string to epoch millis.
     */
    private fun parseTimestamp(timestamp: String): Long {
        return try {
            Instant.parse(timestamp).toEpochMilli()
        } catch (e: Exception) {
            System.currentTimeMillis()
        }
    }
}

/**
 * Message DTO for watch sync.
 */
@Serializable
data class WearMessageDto(
    val id: String,
    val fromNode: String,
    val toNode: String,
    val content: String,
    val timestamp: Long,
    val isRead: Boolean = false
)

/**
 * Reply from watch.
 */
@Serializable
data class WatchReply(
    val originalMessageId: String,
    val replyTo: String,
    val content: String
)

/**
 * New message from watch.
 */
@Serializable
data class WatchSendMessage(
    val to: String,
    val content: String
)
