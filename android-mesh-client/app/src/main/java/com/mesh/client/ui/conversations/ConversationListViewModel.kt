package com.mesh.client.ui.conversations

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.mesh.client.data.local.db.MeshDatabase
import com.mesh.client.data.local.db.entities.MessageEntity
import com.mesh.client.data.remote.MeshSocketClient
import com.mesh.client.data.remote.protocol.isChannelAddress
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.launch
import javax.inject.Inject

/**
 * Represents a conversation preview for the list.
 */
data class ConversationPreview(
    val conversationId: String,
    val otherNodeId: String,
    val lastMessage: String,
    val timestamp: String,
    val unreadCount: Int,
    val isOnline: Boolean,
    val isChannel: Boolean = false
)

@HiltViewModel
class ConversationListViewModel @Inject constructor(
    private val database: MeshDatabase,
    private val socketClient: MeshSocketClient
) : ViewModel() {

    private val _isLoading = MutableStateFlow(true)
    val isLoading: StateFlow<Boolean> = _isLoading.asStateFlow()

    init {
        // Clear loading state after a timeout to prevent indefinite loading
        viewModelScope.launch {
            kotlinx.coroutines.delay(2000)
            _isLoading.value = false
        }
    }

    // Get conversation previews from the database
    private val conversationPreviews: Flow<List<MessageEntity>> =
        database.messageDao().getConversationPreviews()

    // Get roster for online status
    private val rosterEntries = database.rosterDao().getAllRoster()

    // Combine conversation previews with roster info and unread counts
    val conversations: Flow<List<ConversationPreview>> = combine(
        conversationPreviews,
        rosterEntries,
        database.messageDao().getUnreadMessages()
    ) { previews, roster, unreadMessages ->
        _isLoading.value = false

        val myNodeId = socketClient.nodeId
        val rosterMap = roster.associateBy { it.nodeId }

        // Count unread per conversation
        val unreadByConversation = unreadMessages.groupBy { it.conversationId }
            .mapValues { it.value.size }

        previews.map { message ->
            // Determine the other party in the conversation
            // For channels, the conversation ID is the channel address
            val isChannel = isChannelAddress(message.conversationId)
            val otherNodeId = if (isChannel) {
                // Channel conversation - use the channel address as the identifier
                message.conversationId
            } else if (myNodeId != null && message.fromNode == myNodeId) {
                message.toNode
            } else if (myNodeId != null) {
                message.fromNode
            } else {
                // Fallback: use the other node from the conversation ID
                val parts = message.conversationId.split("|")
                if (parts.size == 2) parts[1] else message.fromNode
            }

            val isOnline = if (isChannel) true else rosterMap[otherNodeId]?.isOnline ?: false
            val unreadCount = unreadByConversation[message.conversationId] ?: 0

            ConversationPreview(
                conversationId = message.conversationId,
                otherNodeId = otherNodeId,
                lastMessage = message.content.take(100),
                timestamp = message.timestamp,
                unreadCount = unreadCount,
                isOnline = isOnline,
                isChannel = isChannel
            )
        }
    }

    fun deleteConversation(conversationId: String) {
        viewModelScope.launch {
            database.messageDao().deleteConversation(conversationId)
        }
    }
}
