package com.mesh.client.wear.data

import kotlinx.serialization.Serializable

/**
 * Represents a conversation for the watch UI.
 * Groups messages by conversation partner or channel.
 */
@Serializable
data class WearConversation(
    val id: String,  // node ID or channel address
    val displayName: String,
    val isChannel: Boolean,
    val lastMessage: WearMessage?,
    val unreadCount: Int = 0
) {
    val preview: String
        get() = lastMessage?.preview ?: ""

    val timestamp: Long
        get() = lastMessage?.timestamp ?: 0
}
