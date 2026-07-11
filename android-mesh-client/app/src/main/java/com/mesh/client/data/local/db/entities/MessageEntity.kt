package com.mesh.client.data.local.db.entities

import androidx.room.ColumnInfo
import androidx.room.Entity
import androidx.room.Index
import androidx.room.PrimaryKey

/**
 * Status of an outgoing message.
 */
enum class MessageStatus {
    /** Message is being sent */
    PENDING,
    /** Message was sent successfully */
    SENT,
    /** Message send failed */
    FAILED,
    /** Not applicable (incoming messages) */
    NONE
}

/**
 * Room entity for storing mesh messages.
 */
@Entity(
    tableName = "messages",
    indices = [
        Index(value = ["from_node"]),
        Index(value = ["to_node"]),
        Index(value = ["timestamp"]),
        Index(value = ["conversation_id"])
    ]
)
data class MessageEntity(
    @PrimaryKey
    @ColumnInfo(name = "id")
    val id: String,

    @ColumnInfo(name = "from_node")
    val fromNode: String,

    @ColumnInfo(name = "to_node")
    val toNode: String,

    @ColumnInfo(name = "type")
    val type: String, // MessageType.value

    @ColumnInfo(name = "content")
    val content: String, // JSON serialized content

    @ColumnInfo(name = "timestamp")
    val timestamp: String,

    @ColumnInfo(name = "in_reply_to")
    val inReplyTo: String? = null,

    @ColumnInfo(name = "metadata")
    val metadata: String = "{}", // JSON serialized metadata

    // Computed conversation ID for grouping messages
    // For direct messages: sorted pair of node IDs
    @ColumnInfo(name = "conversation_id")
    val conversationId: String,

    // Local-only fields
    @ColumnInfo(name = "is_read")
    val isRead: Boolean = false,

    @ColumnInfo(name = "is_outgoing")
    val isOutgoing: Boolean = false,

    @ColumnInfo(name = "status")
    val status: MessageStatus = MessageStatus.NONE
) {
    companion object {
        /**
         * Compute conversation ID from two node IDs.
         * Returns a consistent ID regardless of message direction.
         */
        fun computeConversationId(nodeA: String, nodeB: String): String {
            return if (nodeA < nodeB) "$nodeA|$nodeB" else "$nodeB|$nodeA"
        }
    }
}
