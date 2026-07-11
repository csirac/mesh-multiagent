package com.mesh.client.wear.presentation

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.mesh.client.wear.data.ConnectionStatus
import com.mesh.client.wear.data.QuickReply
import com.mesh.client.wear.data.WearConversation
import com.mesh.client.wear.data.WearMessage
import com.mesh.client.wear.data.WearMessageRepository
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch
import javax.inject.Inject

@HiltViewModel
class WearViewModel @Inject constructor(
    private val repository: WearMessageRepository
) : ViewModel() {

    // Use WhileSubscribed with timeout to keep data warm during config changes
    // but allow cleanup when truly inactive
    val messages: StateFlow<List<WearMessage>> = repository.messages
        .stateIn(viewModelScope, SharingStarted.WhileSubscribed(5000), emptyList())

    val conversations: StateFlow<List<WearConversation>> = repository.conversations
        .stateIn(viewModelScope, SharingStarted.WhileSubscribed(5000), emptyList())

    val connectionStatus: StateFlow<ConnectionStatus> = repository.connectionStatus
        .stateIn(viewModelScope, SharingStarted.WhileSubscribed(5000), ConnectionStatus(false))

    val isLoading: StateFlow<Boolean> = repository.isLoading
        .stateIn(viewModelScope, SharingStarted.WhileSubscribed(5000), true)

    private val _replySendingState = MutableStateFlow<ReplySendingState>(ReplySendingState.Idle)
    val replySendingState: StateFlow<ReplySendingState> = _replySendingState.asStateFlow()

    init {
        // Request sync when ViewModel is created
        viewModelScope.launch {
            repository.requestSync()
        }
    }

    fun refresh() {
        viewModelScope.launch {
            repository.requestSync()
        }
    }

    fun getMessageById(messageId: String): WearMessage? {
        return messages.value.find { it.id == messageId }
    }

    fun getMessagesForConversation(conversationId: String): List<WearMessage> {
        return repository.getMessagesForConversation(conversationId)
    }

    fun sendQuickReply(message: WearMessage, reply: QuickReply) {
        viewModelScope.launch {
            _replySendingState.value = ReplySendingState.Sending

            val success = repository.sendQuickReply(message, reply.text)

            _replySendingState.value = if (success) {
                ReplySendingState.Sent
            } else {
                ReplySendingState.Error("Failed to send reply")
            }
        }
    }

    fun sendCustomReply(message: WearMessage, text: String) {
        viewModelScope.launch {
            _replySendingState.value = ReplySendingState.Sending

            val success = repository.sendQuickReply(message, text)

            _replySendingState.value = if (success) {
                ReplySendingState.Sent
            } else {
                ReplySendingState.Error("Failed to send reply")
            }
        }
    }

    fun sendMessageToConversation(conversationId: String, text: String) {
        viewModelScope.launch {
            _replySendingState.value = ReplySendingState.Sending

            val success = repository.sendMessageToConversation(conversationId, text)

            _replySendingState.value = if (success) {
                ReplySendingState.Sent
            } else {
                ReplySendingState.Error("Failed to send message")
            }
        }
    }

    fun markAsRead(messageId: String) {
        viewModelScope.launch {
            repository.markAsRead(messageId)
        }
    }

    fun resetReplyState() {
        _replySendingState.value = ReplySendingState.Idle
    }
}

sealed class ReplySendingState {
    data object Idle : ReplySendingState()
    data object Sending : ReplySendingState()
    data object Sent : ReplySendingState()
    data class Error(val message: String) : ReplySendingState()
}
