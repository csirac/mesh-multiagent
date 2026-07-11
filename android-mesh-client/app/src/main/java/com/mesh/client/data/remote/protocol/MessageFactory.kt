package com.mesh.client.data.remote.protocol

import android.util.Base64
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put

/**
 * Convenience constructors for creating common message types.
 * Mirrors mesh/protocol.py factory functions.
 */
object MessageFactory {

    /**
     * Create a standard conversation message.
     */
    fun makeMessage(
        fromNode: String,
        toNode: String,
        content: String,
        inReplyTo: String? = null
    ): Message = Message(
        fromNode = fromNode,
        toNode = toNode,
        type = MessageType.MESSAGE,
        content = JsonPrimitive(content),
        inReplyTo = inReplyTo
    )

    /**
     * Create an image message.
     *
     * @param fromNode Sender node ID
     * @param toNode Recipient node ID
     * @param imageData Raw image bytes (already compressed)
     * @param mimeType MIME type (e.g., "image/jpeg")
     * @param thumbnail Thumbnail image bytes (optional)
     * @param width Image width in pixels
     * @param height Image height in pixels
     * @param caption Optional text caption for the image
     */
    fun makeImageMessage(
        fromNode: String,
        toNode: String,
        imageData: ByteArray,
        mimeType: String,
        thumbnail: ByteArray?,
        width: Int,
        height: Int,
        caption: String? = null
    ): Message {
        val base64Data = Base64.encodeToString(imageData, Base64.NO_WRAP)
        val base64Thumbnail = thumbnail?.let { Base64.encodeToString(it, Base64.NO_WRAP) }

        return Message(
            fromNode = fromNode,
            toNode = toNode,
            type = MessageType.MESSAGE,
            content = buildJsonObject {
                put("type", "image")
                put("data", base64Data)
                put("mime_type", mimeType)
                base64Thumbnail?.let { put("thumbnail", it) }
                put("width", width)
                put("height", height)
                put("size_bytes", imageData.size)
                caption?.let { put("caption", it) }
            }
        )
    }

    /**
     * Create a control message for the router.
     */
    fun makeControl(
        fromNode: String,
        action: ControlAction,
        targetNode: String? = null,
        config: JsonObject? = null,
        authToken: String? = null,
        fcmToken: String? = null
    ): Message = Message(
        fromNode = fromNode,
        toNode = "router",
        type = MessageType.CONTROL,
        content = buildJsonObject {
            put("action", action.value)
            targetNode?.let { put("target", it) }
            authToken?.let { put("auth_token", it) }
            fcmToken?.let { put("fcm_token", it) }
        },
        metadata = config?.let { buildJsonObject { put("config", it) } } ?: JsonObject(emptyMap())
    )

    /**
     * Create a presence message announcing a node joining or leaving.
     */
    fun makePresence(
        fromNode: String,
        event: String, // "join" or "leave"
        nickname: String,
        nodeType: String, // "user" or agent type like "coder"
        description: String = ""
    ): Message = Message(
        fromNode = fromNode,
        toNode = "broadcast",
        type = MessageType.PRESENCE,
        content = buildJsonObject {
            put("event", event)
            put("nickname", nickname)
            put("node_type", nodeType)
            if (description.isNotEmpty()) {
                put("description", description)
            }
        }
    )

    /**
     * Create a response to a confirmation request.
     */
    fun makeConfirmResponse(
        fromNode: String,
        toNode: String,
        inReplyTo: String,
        confirmed: Boolean
    ): Message = Message(
        fromNode = fromNode,
        toNode = toNode,
        type = MessageType.CONFIRM_RESPONSE,
        content = buildJsonObject {
            put("confirmed", confirmed)
        },
        inReplyTo = inReplyTo
    )

    /**
     * Request an agent's recent context (for supervision).
     */
    fun makeStatusRequest(
        fromNode: String,
        toNode: String,
        numMessages: Int = 5,
        diagnostics: Boolean = true
    ): Message = Message(
        fromNode = fromNode,
        toNode = toNode,
        type = MessageType.STATUS_REQUEST,
        content = buildJsonObject {
            put("num_messages", numMessages)
            if (diagnostics) put("diagnostics", true)
        }
    )

    // --- Message deletion ---

    /**
     * Delete a message.
     */
    fun makeDeleteMessage(
        fromNode: String,
        messageId: String,
        conversationId: String
    ): Message = Message(
        fromNode = fromNode,
        toNode = "router",
        type = MessageType.CONTROL,
        content = buildJsonObject {
            put("action", ControlAction.DELETE_MESSAGE.value)
            put("message_id", messageId)
            put("conversation_id", conversationId)
        }
    )

    // --- Channel operations ---

    /**
     * Create a new channel.
     */
    fun makeChannelCreate(
        fromNode: String,
        channelName: String,
        description: String = ""
    ): Message = Message(
        fromNode = fromNode,
        toNode = "router",
        type = MessageType.CONTROL,
        content = buildJsonObject {
            put("action", ControlAction.CHANNEL_CREATE.value)
            put("channel_name", channelName)
            if (description.isNotEmpty()) {
                put("description", description)
            }
        }
    )

    /**
     * Delete a channel (users only).
     */
    fun makeChannelDelete(
        fromNode: String,
        channelName: String
    ): Message = Message(
        fromNode = fromNode,
        toNode = "router",
        type = MessageType.CONTROL,
        content = buildJsonObject {
            put("action", ControlAction.CHANNEL_DELETE.value)
            put("channel_name", channelName)
        }
    )

    /**
     * Join a channel.
     */
    fun makeChannelJoin(
        fromNode: String,
        channelName: String
    ): Message = Message(
        fromNode = fromNode,
        toNode = "router",
        type = MessageType.CONTROL,
        content = buildJsonObject {
            put("action", ControlAction.CHANNEL_JOIN.value)
            put("channel_name", channelName)
        }
    )

    /**
     * Leave a channel.
     */
    fun makeChannelLeave(
        fromNode: String,
        channelName: String
    ): Message = Message(
        fromNode = fromNode,
        toNode = "router",
        type = MessageType.CONTROL,
        content = buildJsonObject {
            put("action", ControlAction.CHANNEL_LEAVE.value)
            put("channel_name", channelName)
        }
    )

    /**
     * List all channels.
     */
    fun makeChannelList(
        fromNode: String
    ): Message = Message(
        fromNode = fromNode,
        toNode = "router",
        type = MessageType.CONTROL,
        content = buildJsonObject {
            put("action", ControlAction.CHANNEL_LIST.value)
        }
    )

    /**
     * Get members of a channel.
     */
    fun makeChannelMembers(
        fromNode: String,
        channelName: String
    ): Message = Message(
        fromNode = fromNode,
        toNode = "router",
        type = MessageType.CONTROL,
        content = buildJsonObject {
            put("action", ControlAction.CHANNEL_MEMBERS.value)
            put("channel_name", channelName)
        }
    )

    /**
     * Create a request to invite a node to a channel.
     */
    fun makeChannelInvite(
        fromNode: String,
        channelName: String,
        nodeId: String
    ): Message = Message(
        fromNode = fromNode,
        toNode = "router",
        type = MessageType.CONTROL,
        content = buildJsonObject {
            put("action", ControlAction.CHANNEL_INVITE.value)
            put("channel_name", channelName)
            put("node_id", nodeId)
        }
    )

    // --- Message sync operations ---

    /**
     * Request message history sync from the server.
     *
     * @param fromNode The requesting node
     * @param since Optional ISO timestamp - only return messages after this time
     * @param limit Maximum number of messages to return
     * @param conversationId Optional specific conversation to sync
     */
    fun makeHistorySync(
        fromNode: String,
        since: String? = null,
        limit: Int = 500,
        conversationId: String? = null
    ): Message = Message(
        fromNode = fromNode,
        toNode = "router",
        type = MessageType.CONTROL,
        content = buildJsonObject {
            put("action", ControlAction.HISTORY_SYNC.value)
            put("limit", limit)
            since?.let { put("since", it) }
            conversationId?.let { put("conversation_id", it) }
        }
    )

    /**
     * Mark messages in a conversation as read.
     *
     * @param fromNode The node marking messages as read
     * @param conversationId The conversation ID
     * @param upToTimestamp ISO timestamp - mark all messages up to this point as read
     */
    fun makeMarkRead(
        fromNode: String,
        conversationId: String,
        upToTimestamp: String
    ): Message = Message(
        fromNode = fromNode,
        toNode = "router",
        type = MessageType.CONTROL,
        content = buildJsonObject {
            put("action", ControlAction.MARK_READ.value)
            put("conversation_id", conversationId)
            put("up_to_timestamp", upToTimestamp)
        }
    )

    /**
     * Request Claude Code account usage from the router.
     */
    fun makeCcUsageRequest(
        fromNode: String
    ): Message = Message(
        fromNode = fromNode,
        toNode = "router",
        type = MessageType.CONTROL,
        content = buildJsonObject {
            put("action", ControlAction.CC_USAGE.value)
        }
    )
}

/**
 * Parse a node ID into its components.
 *
 * Node ID formats:
 * - Users: "user:{nickname}" -> Triple("user", nickname, null)
 * - Agents: "agent:{type}:{nickname}" -> Triple("agent", type, nickname)
 * - Legacy agents: "agent:{type}" -> Triple("agent", type, null)
 */
fun parseNodeId(nodeId: String): Triple<String, String, String?> {
    val parts = nodeId.split(":", limit = 3)
    return when (parts.size) {
        2 -> Triple(parts[0], parts[1], null)
        3 -> Triple(parts[0], parts[1], parts[2])
        else -> Triple(parts[0], "", null)
    }
}

/**
 * Get a human-friendly display name for a node.
 *
 * For users: returns the nickname
 * For agents: returns the nickname (or type if no nickname)
 */
fun getDisplayName(nodeId: String): String {
    val (nodeType, typeOrNick, nickname) = parseNodeId(nodeId)
    return when (nodeType) {
        "user" -> typeOrNick.replaceFirstChar { it.uppercase() }
        "agent" -> (nickname ?: typeOrNick).replaceFirstChar { it.uppercase() }
        else -> nodeId
    }
}

/**
 * Build a full agent node ID from type and nickname.
 */
fun buildAgentNodeId(agentType: String, nickname: String): String =
    "agent:$agentType:$nickname"

/**
 * Build a user node ID from nickname.
 */
fun buildUserNodeId(nickname: String): String = "user:$nickname"

/**
 * Check if an address is a channel address.
 */
fun isChannelAddress(address: String): Boolean = address.startsWith("channel:")

/**
 * Parse channel name from a channel address.
 * Returns null if not a channel address.
 */
fun parseChannelName(address: String): String? =
    if (isChannelAddress(address)) address.removePrefix("channel:") else null

/**
 * Build a channel address from a channel name.
 */
fun buildChannelAddress(channelName: String): String = "channel:$channelName"
