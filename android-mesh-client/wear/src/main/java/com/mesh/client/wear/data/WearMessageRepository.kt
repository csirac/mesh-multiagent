package com.mesh.client.wear.data

import android.content.Context
import android.content.SharedPreferences
import android.util.Log
import com.google.android.gms.wearable.DataClient
import com.google.android.gms.wearable.DataItem
import com.google.android.gms.wearable.DataMapItem
import com.google.android.gms.wearable.MessageClient
import com.google.android.gms.wearable.NodeClient
import com.google.android.gms.wearable.PutDataMapRequest
import com.google.android.gms.wearable.Wearable
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.tasks.await
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Repository for managing message data synced from the phone.
 */
@Singleton
class WearMessageRepository @Inject constructor(
    @ApplicationContext private val context: Context
) {
    companion object {
        private const val TAG = "WearMessageRepo"
        private const val MAX_MESSAGES = 20
        private const val PREFS_NAME = "mesh_wear_cache"
        private const val KEY_MESSAGES_JSON = "cached_messages"
        private const val KEY_MY_NODE_ID = "my_node_id"
    }

    private val json = Json { ignoreUnknownKeys = true }
    private val prefs: SharedPreferences = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)

    private val dataClient: DataClient = Wearable.getDataClient(context)
    private val messageClient: MessageClient = Wearable.getMessageClient(context)
    private val nodeClient: NodeClient = Wearable.getNodeClient(context)

    private val _messages = MutableStateFlow<List<WearMessage>>(emptyList())
    val messages: StateFlow<List<WearMessage>> = _messages.asStateFlow()

    // Conversations grouped by partner/channel
    private val _conversations = MutableStateFlow<List<WearConversation>>(emptyList())
    val conversations: StateFlow<List<WearConversation>> = _conversations.asStateFlow()

    // Full message list (not truncated) for conversation filtering
    private var allMessages: List<WearMessage> = emptyList()

    // Track the user's own node ID for proper message filtering
    private var myNodeId: String = "user:yourname"

    private val _connectionStatus = MutableStateFlow(ConnectionStatus(isConnected = false))
    val connectionStatus: StateFlow<ConnectionStatus> = _connectionStatus.asStateFlow()

    private val _isLoading = MutableStateFlow(true)
    val isLoading: StateFlow<Boolean> = _isLoading.asStateFlow()

    init {
        loadCachedData()
    }

    /**
     * Load cached messages from SharedPreferences on startup.
     * Shows cached data immediately while waiting for phone sync.
     */
    private fun loadCachedData() {
        try {
            // Load cached user node ID
            val cachedNodeId = prefs.getString(KEY_MY_NODE_ID, null)
            if (cachedNodeId != null) {
                myNodeId = cachedNodeId
                Log.d(TAG, ">>> Loaded cached myNodeId: $myNodeId")
            }

            // Load cached messages
            val cachedJson = prefs.getString(KEY_MESSAGES_JSON, null)
            if (cachedJson != null) {
                val messageList = json.decodeFromString<List<WearMessage>>(cachedJson)
                Log.d(TAG, ">>> Loaded ${messageList.size} cached messages")

                allMessages = messageList
                _messages.value = messageList.take(MAX_MESSAGES)
                _conversations.value = groupMessagesIntoConversations(messageList)

                // Show cached data immediately (not loading)
                _isLoading.value = false
                Log.d(TAG, ">>> Cache loaded: ${_conversations.value.size} conversations, isLoading=false")
            } else {
                Log.d(TAG, ">>> No cached data found")
            }
        } catch (e: Exception) {
            Log.e(TAG, ">>> Failed to load cached data", e)
            // Clear corrupted cache
            prefs.edit().remove(KEY_MESSAGES_JSON).apply()
        }
    }

    /**
     * Save messages to SharedPreferences cache.
     */
    private fun saveCacheData(messageList: List<WearMessage>) {
        try {
            val messagesJson = json.encodeToString(messageList)
            prefs.edit()
                .putString(KEY_MESSAGES_JSON, messagesJson)
                .putString(KEY_MY_NODE_ID, myNodeId)
                .apply()
            Log.d(TAG, ">>> Saved ${messageList.size} messages to cache")
        } catch (e: Exception) {
            Log.e(TAG, ">>> Failed to save cache", e)
        }
    }

    /**
     * Handle messages update from Data Layer.
     */
    suspend fun handleMessagesUpdate(dataItem: DataItem) {
        try {
            Log.d(TAG, ">>> handleMessagesUpdate called, dataItem uri=${dataItem.uri}")
            val dataMap = DataMapItem.fromDataItem(dataItem).dataMap
            val messagesJson = dataMap.getString("messages")
            Log.d(TAG, ">>> messagesJson is ${if (messagesJson == null) "NULL" else "${messagesJson.length} chars"}")

            if (messagesJson == null) {
                Log.w(TAG, ">>> No messages field in data map, returning")
                return
            }

            // Get user's node ID from phone if provided
            val phoneUserNodeId = dataMap.getString("userNodeId")
            Log.d(TAG, ">>> phoneUserNodeId from dataMap: $phoneUserNodeId")
            if (phoneUserNodeId != null) {
                myNodeId = phoneUserNodeId
                Log.d(TAG, ">>> Set myNodeId to: $myNodeId")
            } else {
                Log.w(TAG, ">>> userNodeId NOT provided by phone, keeping myNodeId=$myNodeId")
            }

            val messageList = json.decodeFromString<List<WearMessage>>(messagesJson)
            Log.d(TAG, ">>> Parsed ${messageList.size} messages from JSON")

            // Log first few messages for debugging
            messageList.take(5).forEachIndexed { idx, msg ->
                Log.d(TAG, ">>> MSG[$idx]: id=${msg.id}, from=${msg.fromNode}, to=${msg.toNode}, content=${msg.content.take(30)}")
            }

            // Store full message list for conversation filtering
            allMessages = messageList
            _messages.value = messageList.take(MAX_MESSAGES)
            Log.d(TAG, ">>> Set _messages.value with ${_messages.value.size} messages, allMessages has ${allMessages.size}")

            _conversations.value = groupMessagesIntoConversations(messageList)
            Log.d(TAG, ">>> Created ${_conversations.value.size} conversations")
            _conversations.value.forEach { conv ->
                Log.d(TAG, ">>> CONV: id=${conv.id}, name=${conv.displayName}, lastMsg=${conv.lastMessage?.content?.take(30)}")
            }

            // Cache the data for instant display on next launch
            saveCacheData(messageList)

            _isLoading.value = false
            Log.d(TAG, ">>> handleMessagesUpdate complete, isLoading=false")
        } catch (e: Exception) {
            Log.e(TAG, ">>> EXCEPTION in handleMessagesUpdate", e)
        }
    }

    /**
     * Handle channels update from Data Layer.
     */
    suspend fun handleChannelsUpdate(dataItem: DataItem) {
        // For now, we just log channel updates
        // Future: could show channel list on watch
        try {
            val dataMap = DataMapItem.fromDataItem(dataItem).dataMap
            val channelsJson = dataMap.getString("channels") ?: return
            Log.d(TAG, "Channels updated: $channelsJson")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to parse channels", e)
        }
    }

    /**
     * Handle connection status update from Data Layer.
     */
    suspend fun handleConnectionUpdate(dataItem: DataItem) {
        try {
            val dataMap = DataMapItem.fromDataItem(dataItem).dataMap
            val isConnected = dataMap.getBoolean("connected", false)
            val phoneNodeId = dataMap.getString("phoneNodeId")
            val lastSync = dataMap.getLong("lastSync", 0)

            _connectionStatus.value = ConnectionStatus(
                isConnected = isConnected,
                phoneNodeId = phoneNodeId,
                lastSyncTime = lastSync
            )

            Log.d(TAG, "Connection status: connected=$isConnected")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to parse connection status", e)
        }
    }

    /**
     * Handle a new message notification (direct message from phone).
     */
    suspend fun handleNewMessage(messageJson: String) {
        try {
            val message = json.decodeFromString<WearMessage>(messageJson)
            val currentList = _messages.value.toMutableList()

            // Add to front of list, remove duplicates
            currentList.removeAll { it.id == message.id }
            currentList.add(0, message)

            _messages.value = currentList.take(MAX_MESSAGES)
            Log.d(TAG, "New message added: ${message.id}")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to parse new message", e)
        }
    }

    /**
     * Send a quick reply through the phone.
     */
    suspend fun sendQuickReply(originalMessage: WearMessage, reply: String): Boolean {
        return try {
            val connectedNodes = nodeClient.connectedNodes.await()
            val phoneNode = connectedNodes.firstOrNull()

            if (phoneNode == null) {
                Log.w(TAG, "No connected phone found")
                return false
            }

            val replyData = json.encodeToString(
                mapOf(
                    "originalMessageId" to originalMessage.id,
                    "replyTo" to originalMessage.fromNode,
                    "content" to reply
                )
            )

            messageClient.sendMessage(
                phoneNode.id,
                WearDataLayerListenerService.PATH_REPLY,
                replyData.toByteArray(Charsets.UTF_8)
            ).await()

            Log.d(TAG, "Reply sent to ${originalMessage.fromNode}")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to send reply", e)
            false
        }
    }

    /**
     * Send a new message to a conversation (user or channel).
     */
    suspend fun sendMessageToConversation(conversationId: String, content: String): Boolean {
        return try {
            val connectedNodes = nodeClient.connectedNodes.await()
            val phoneNode = connectedNodes.firstOrNull()

            if (phoneNode == null) {
                Log.w(TAG, "No connected phone found")
                return false
            }

            val messageData = json.encodeToString(
                mapOf(
                    "to" to conversationId,
                    "content" to content
                )
            )

            messageClient.sendMessage(
                phoneNode.id,
                WearDataLayerListenerService.PATH_SEND_MESSAGE,
                messageData.toByteArray(Charsets.UTF_8)
            ).await()

            Log.d(TAG, "Message sent to $conversationId")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to send message", e)
            false
        }
    }

    /**
     * Mark a message as read.
     */
    suspend fun markAsRead(messageId: String) {
        try {
            val connectedNodes = nodeClient.connectedNodes.await()
            val phoneNode = connectedNodes.firstOrNull() ?: return

            messageClient.sendMessage(
                phoneNode.id,
                WearDataLayerListenerService.PATH_MARK_READ,
                messageId.toByteArray(Charsets.UTF_8)
            ).await()

            // Update local state
            _messages.value = _messages.value.map {
                if (it.id == messageId) it.copy(isRead = true) else it
            }

            Log.d(TAG, "Marked message as read: $messageId")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to mark as read", e)
        }
    }

    /**
     * Group messages into conversations by partner/channel.
     */
    private fun groupMessagesIntoConversations(messageList: List<WearMessage>): List<WearConversation> {
        // Try to detect the user's node ID if we don't have a reliable value
        // Priority:
        // 1. Already set by phone sync (myNodeId will be set in handleMessagesUpdate)
        // 2. Find a user: node that appears as both fromNode AND toNode in DMs
        //    (indicates this node both sends and receives, so it's "us")
        // 3. Find any user: node that appears as fromNode (we sent a message)

        val dmMessages = messageList.filter { !it.isChannelMessage }
        val fromNodes = dmMessages.map { it.fromNode }.toSet()
        val toNodes = dmMessages.map { it.toNode }.toSet()
        val userFromNodes = fromNodes.filter { it.startsWith("user:") }.toSet()
        val userToNodes = toNodes.filter { it.startsWith("user:") }.toSet()

        Log.d(TAG, ">>> Detection: userFromNodes=$userFromNodes, userToNodes=$userToNodes")

        // Only re-detect if myNodeId is the default or missing
        if (myNodeId == "user:yourname" || !myNodeId.startsWith("user:")) {
            // A user node that appears in both fromNodes and toNodes is likely us
            val userBothSides = userFromNodes.intersect(userToNodes)
            if (userBothSides.isNotEmpty()) {
                myNodeId = userBothSides.first()
                Log.d(TAG, ">>> Detected myNodeId (both send and receive): $myNodeId")
            } else if (userFromNodes.isNotEmpty()) {
                // Fall back to a user node that sent messages
                myNodeId = userFromNodes.first()
                Log.d(TAG, ">>> Detected myNodeId (sender only): $myNodeId")
            }
        }

        Log.d(TAG, ">>> groupMessagesIntoConversations: final myNodeId=$myNodeId (${dmMessages.size} DM messages)")

        // Group by conversation partner or channel
        val grouped = messageList.groupBy { msg ->
            if (msg.isChannelMessage) {
                // For channels, use just the channel name (strip any |user:xxx suffix from malformed data)
                val toNode = msg.toNode
                if (toNode.contains("|")) {
                    // Malformed channel address like "channel:test|user:yourname" - extract just channel part
                    toNode.substringBefore("|")
                } else {
                    toNode  // Normal channel:name
                }
            } else {
                // For DMs, use the other party (not self)
                if (msg.fromNode == myNodeId) {
                    msg.toNode  // outgoing message - partner is toNode
                } else {
                    msg.fromNode  // incoming message - partner is fromNode
                }
            }
        }

        Log.d(TAG, ">>> Grouped into ${grouped.size} conversations: ${grouped.keys.joinToString()}")

        return grouped.map { (partnerId, messages) ->
            val sortedMessages = messages.sortedByDescending { it.timestamp }
            val lastMessage = sortedMessages.firstOrNull()
            val isChannel = partnerId.startsWith("channel:")
            val displayName = if (isChannel) {
                "#${partnerId.removePrefix("channel:")}"
            } else {
                partnerId.substringAfterLast(":")
            }

            WearConversation(
                id = partnerId,
                displayName = displayName,
                isChannel = isChannel,
                lastMessage = lastMessage,
                unreadCount = messages.count { !it.isRead }
            )
        }.sortedByDescending { it.timestamp }
    }

    /**
     * Get messages for a specific conversation.
     * Uses the same logic as groupMessagesIntoConversations to correctly identify
     * messages that belong to a conversation (using the stored myNodeId).
     */
    fun getMessagesForConversation(conversationId: String): List<WearMessage> {
        Log.d(TAG, ">>> getMessagesForConversation($conversationId): myNodeId=$myNodeId, total messages=${allMessages.size}")

        // Debug: log all unique fromNode and toNode values
        val fromNodes = allMessages.map { it.fromNode }.toSet()
        val toNodes = allMessages.map { it.toNode }.toSet()
        Log.d(TAG, ">>> All fromNodes: $fromNodes")
        Log.d(TAG, ">>> All toNodes: $toNodes")

        val result = allMessages.filter { msg ->
            if (msg.isChannelMessage) {
                // Handle malformed channel addresses (channel:test|user:yourname)
                val effectiveToNode = if (msg.toNode.contains("|")) {
                    msg.toNode.substringBefore("|")
                } else {
                    msg.toNode
                }
                val matches = effectiveToNode == conversationId
                Log.d(TAG, ">>> Channel msg ${msg.id}: toNode=${msg.toNode} effectiveToNode=$effectiveToNode matches=$matches")
                matches
            } else {
                // DM: find the partner in this message
                val partnerId = if (msg.fromNode == myNodeId) msg.toNode else msg.fromNode
                val matches = partnerId == conversationId
                Log.d(TAG, ">>> DM msg ${msg.id}: from=${msg.fromNode}, to=${msg.toNode}, partnerId=$partnerId, matches=$matches")
                matches
            }
        }.sortedByDescending { it.timestamp }

        Log.d(TAG, ">>> getMessagesForConversation($conversationId): found ${result.size} messages")
        if (result.isEmpty() && allMessages.isNotEmpty()) {
            Log.w(TAG, ">>> WARNING: No messages found for $conversationId but we have ${allMessages.size} total messages")
            allMessages.take(5).forEach { msg ->
                Log.w(TAG, ">>>   Sample: from=${msg.fromNode}, to=${msg.toNode}")
            }
        }
        return result
    }

    /**
     * Request a sync from the phone.
     */
    suspend fun requestSync() {
        try {
            Log.d(TAG, ">>> requestSync called, current state: messages=${_messages.value.size}, convs=${_conversations.value.size}, myNodeId=$myNodeId")
            // Only show loading if we have no cached data to display
            if (_conversations.value.isEmpty()) {
                _isLoading.value = true
            }

            val connectedNodes = nodeClient.connectedNodes.await()
            Log.d(TAG, ">>> Found ${connectedNodes.size} connected nodes")
            val phoneNode = connectedNodes.firstOrNull()

            if (phoneNode != null) {
                Log.d(TAG, ">>> Sending sync request to phone: id=${phoneNode.id}, name=${phoneNode.displayName}")
                messageClient.sendMessage(
                    phoneNode.id,
                    "/mesh/sync_request",
                    ByteArray(0)
                ).await()

                _connectionStatus.value = _connectionStatus.value.copy(
                    isConnected = true,
                    phoneNodeId = phoneNode.id
                )

                Log.d(TAG, ">>> Sync requested from phone: ${phoneNode.displayName}")
            } else {
                _connectionStatus.value = ConnectionStatus(isConnected = false)
                _isLoading.value = false
                Log.w(TAG, ">>> No phone connected, setting isConnected=false")
            }
        } catch (e: Exception) {
            Log.e(TAG, ">>> EXCEPTION in requestSync", e)
            _isLoading.value = false
        }
    }
}
