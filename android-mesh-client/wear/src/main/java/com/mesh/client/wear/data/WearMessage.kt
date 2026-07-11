package com.mesh.client.wear.data

import kotlinx.serialization.Serializable

/**
 * Simplified message representation for watch display.
 * Synced from phone via Data Layer API.
 */
@Serializable
data class WearMessage(
    val id: String,
    val fromNode: String,
    val toNode: String,
    val content: String,
    val timestamp: Long,
    val isRead: Boolean = false
) {
    /**
     * Get a display-friendly sender name.
     * Converts "user:yourname" to "yourname", "agent:coder:claude" to "claude"
     */
    val senderDisplayName: String
        get() = fromNode.substringAfterLast(":")

    /**
     * Check if this is a channel message.
     */
    val isChannelMessage: Boolean
        get() = toNode.startsWith("channel:")

    /**
     * Get channel name if this is a channel message.
     */
    val channelName: String?
        get() = if (isChannelMessage) toNode.removePrefix("channel:") else null

    /**
     * Get a preview of the content (first 50 chars).
     */
    val preview: String
        get() = if (content.length > 50) content.take(50) + "…" else content
}

/**
 * Connection status with phone.
 */
@Serializable
data class ConnectionStatus(
    val isConnected: Boolean,
    val phoneNodeId: String? = null,
    val lastSyncTime: Long = 0
)

/**
 * Quick reply options.
 */
enum class QuickReply(val text: String) {
    OK("OK"),
    THANKS("Thanks"),
    ON_IT("On it"),
    LATER("Later")
}
