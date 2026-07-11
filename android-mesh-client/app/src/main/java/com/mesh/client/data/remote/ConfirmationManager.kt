package com.mesh.client.data.remote

import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Represents a pending confirmation request from an agent.
 */
data class ConfirmationRequest(
    val messageId: String,
    val fromNode: String,
    val toolName: String,
    val preview: String,
    val notificationId: Int
)

/**
 * Manages pending confirmation requests.
 *
 * This allows the UI to observe and display confirmation dialogs,
 * and ensures requests are handled in order.
 */
@Singleton
class ConfirmationManager @Inject constructor() {

    private val _pendingConfirmations = MutableStateFlow<List<ConfirmationRequest>>(emptyList())
    val pendingConfirmations: StateFlow<List<ConfirmationRequest>> = _pendingConfirmations.asStateFlow()

    /**
     * Add a new confirmation request to the queue.
     */
    fun addRequest(request: ConfirmationRequest) {
        _pendingConfirmations.value = _pendingConfirmations.value + request
    }

    /**
     * Remove a confirmation request (after it's been handled).
     */
    fun removeRequest(messageId: String) {
        _pendingConfirmations.value = _pendingConfirmations.value.filter { it.messageId != messageId }
    }

    /**
     * Get the next pending confirmation (if any).
     */
    fun getNextPending(): ConfirmationRequest? = _pendingConfirmations.value.firstOrNull()

    /**
     * Clear all pending confirmations.
     */
    fun clearAll() {
        _pendingConfirmations.value = emptyList()
    }
}
