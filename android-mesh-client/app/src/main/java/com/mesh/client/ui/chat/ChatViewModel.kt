package com.mesh.client.ui.chat

import android.app.Application
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.net.Uri
import android.os.IBinder
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.mesh.client.data.local.db.MeshDatabase
import com.mesh.client.data.local.db.entities.DraftEntity
import com.mesh.client.data.local.db.entities.MessageEntity
import com.mesh.client.data.remote.ConnectionState
import com.mesh.client.data.remote.protocol.isChannelAddress
import com.mesh.client.data.remote.MeshSocketClient
import com.mesh.client.data.remote.StatusManager
import com.mesh.client.data.remote.StatusResponse
import com.mesh.client.service.MeshService
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.emptyFlow
import kotlinx.coroutines.flow.flatMapLatest
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import javax.inject.Inject

data class ChatUiState(
    val currentTarget: String? = null,
    val isConnected: Boolean = false,
    val isSending: Boolean = false,
    val error: String? = null,
    val selectedImageUri: Uri? = null,
    val isProcessingImage: Boolean = false
)

@HiltViewModel
class ChatViewModel @Inject constructor(
    application: Application,
    private val database: MeshDatabase,
    private val socketClient: MeshSocketClient,
    private val statusManager: StatusManager
) : AndroidViewModel(application) {

    private val tag = "ChatViewModel"

    private val _uiState = MutableStateFlow(ChatUiState())
    val uiState: StateFlow<ChatUiState> = _uiState.asStateFlow()

    private val _currentConversationId = MutableStateFlow<String?>(null)

    // Our node ID from the socket client
    private var myNodeId: String? = null

    // Draft text for current conversation
    private val _draftText = MutableStateFlow("")
    val draftText: StateFlow<String> = _draftText.asStateFlow()
    private var draftSaveJob: Job? = null

    // Status response state
    val statusResponse: StateFlow<StatusResponse?> = statusManager.currentStatus
    val statusLoading: StateFlow<Boolean> = statusManager.isLoading

    // Messages for current conversation
    val messages: Flow<List<MessageEntity>> = _currentConversationId.flatMapLatest { convId ->
        Log.d(tag, "Querying messages for conversationId=$convId")
        if (convId != null) {
            database.messageDao().getConversationMessages(convId)
        } else {
            Log.d(tag, "No conversation ID set, returning empty flow")
            emptyFlow()
        }
    }

    private var meshService: MeshService? = null
    private var serviceBound = false

    private val serviceConnection = object : ServiceConnection {
        override fun onServiceConnected(name: ComponentName?, service: IBinder?) {
            val binder = service as MeshService.LocalBinder
            meshService = binder.getService()
            serviceBound = true

            // Observe service state
            viewModelScope.launch {
                meshService?.currentTarget?.collect { target ->
                    _uiState.value = _uiState.value.copy(currentTarget = target)
                    updateConversationId(target)
                }
            }
        }

        override fun onServiceDisconnected(name: ComponentName?) {
            meshService = null
            serviceBound = false
        }
    }

    init {
        // Bind to MeshService
        val intent = Intent(application, MeshService::class.java)
        application.bindService(intent, serviceConnection, Context.BIND_AUTO_CREATE)

        // Observe connection state
        viewModelScope.launch {
            socketClient.connectionState.collect { state ->
                _uiState.value = _uiState.value.copy(
                    isConnected = state is ConnectionState.Connected
                )
            }
        }
    }

    override fun onCleared() {
        // Persist current draft synchronously — viewModelScope is already cancelled
        val target = _uiState.value.currentTarget
        val text = _draftText.value
        if (target != null && text.isNotEmpty()) {
            runBlocking {
                database.messageDao().saveDraft(DraftEntity(conversationKey = target, text = text))
            }
        }
        if (serviceBound) {
            getApplication<Application>().unbindService(serviceConnection)
            serviceBound = false
        }
        super.onCleared()
    }

    fun sendMessage(content: String) {
        Log.d(tag, "sendMessage: content='$content'")
        if (content.isBlank()) {
            Log.w(tag, "sendMessage: content is blank, ignoring")
            return
        }

        viewModelScope.launch {
            _uiState.value = _uiState.value.copy(isSending = true)

            val service = meshService
            if (service == null) {
                Log.w(tag, "sendMessage: meshService is null")
                _uiState.value = _uiState.value.copy(isSending = false, error = "Service not connected")
                return@launch
            }

            Log.d(tag, "sendMessage: calling meshService.sendMessage")
            val success = service.sendMessage(content)
            Log.d(tag, "sendMessage: meshService.sendMessage returned $success")

            if (success) {
                // Clear draft on successful send
                _draftText.value = ""
                val target = _uiState.value.currentTarget
                if (target != null) {
                    database.messageDao().deleteDraft(target)
                }
            }

            _uiState.value = _uiState.value.copy(
                isSending = false,
                error = if (success) null else "Failed to send message"
            )
        }
    }

    fun setTarget(nodeId: String) {
        Log.d(tag, "setTarget: nodeId=$nodeId")
        meshService?.setTarget(nodeId)
    }

    private fun updateConversationId(target: String?) {
        Log.d(tag, "updateConversationId: target=$target, myNodeId=$myNodeId")

        // Save draft for the previous conversation before switching (cancel any pending debounce)
        draftSaveJob?.cancel()
        val prevTarget = _uiState.value.currentTarget
        val prevText = _draftText.value
        if (prevTarget != null && prevText.isNotEmpty()) {
            saveDraft(prevTarget, prevText)
        } else if (prevTarget != null) {
            // Clear empty draft from DB
            viewModelScope.launch { database.messageDao().deleteDraft(prevTarget) }
        }

        if (target == null) {
            _currentConversationId.value = null
            _draftText.value = ""
            return
        }

        val nodeId = myNodeId ?: socketClient.nodeId
        if (nodeId == null) {
            Log.w(tag, "No nodeId available, cannot compute conversationId")
            _currentConversationId.value = null
            return
        }
        myNodeId = nodeId
        // For channels, use the channel address directly as conversationId
        // For direct messages, use sorted pair of node IDs
        val convId = if (isChannelAddress(target)) {
            target
        } else {
            MessageEntity.computeConversationId(nodeId, target)
        }
        Log.d(tag, "Computed conversationId: $convId")
        _currentConversationId.value = convId

        // Load draft for the new conversation
        loadDraft(target)
    }

    fun updateDraftText(text: String) {
        _draftText.value = text
        // Debounce-save draft to DB after 500ms of idle typing
        val target = _uiState.value.currentTarget ?: return
        draftSaveJob?.cancel()
        draftSaveJob = viewModelScope.launch {
            delay(500)
            saveDraft(target, text)
        }
    }

    private fun saveDraft(target: String, text: String) {
        viewModelScope.launch {
            if (text.isBlank()) {
                database.messageDao().deleteDraft(target)
            } else {
                database.messageDao().saveDraft(DraftEntity(conversationKey = target, text = text))
            }
        }
    }

    private fun loadDraft(target: String) {
        viewModelScope.launch {
            val draft = database.messageDao().getDraft(target) ?: ""
            _draftText.value = draft
        }
    }

    fun markConversationAsRead() {
        val convId = _currentConversationId.value ?: return
        viewModelScope.launch {
            database.messageDao().markConversationAsRead(convId)
        }
    }

    fun requestStatus(numMessages: Int = 5) {
        val target = _uiState.value.currentTarget
        if (target == null) {
            Log.w(tag, "requestStatus: no current target set")
            return
        }
        meshService?.requestStatus(target, numMessages)
    }

    fun dismissStatus() {
        statusManager.clearStatus()
    }

    // --- Image handling ---

    /**
     * Set a selected image URI for preview before sending.
     */
    fun setSelectedImage(uri: Uri?) {
        Log.d(tag, "setSelectedImage: uri=$uri")
        _uiState.value = _uiState.value.copy(selectedImageUri = uri)
    }

    /**
     * Clear the selected image.
     */
    fun clearSelectedImage() {
        Log.d(tag, "clearSelectedImage")
        _uiState.value = _uiState.value.copy(selectedImageUri = null)
    }

    /**
     * Send the currently selected image with an optional caption.
     */
    fun sendImage(caption: String? = null) {
        val uri = _uiState.value.selectedImageUri
        if (uri == null) {
            Log.w(tag, "sendImage: no image selected")
            return
        }

        val target = _uiState.value.currentTarget
        if (target == null) {
            Log.w(tag, "sendImage: no current target set")
            _uiState.value = _uiState.value.copy(error = "No recipient selected")
            return
        }

        viewModelScope.launch {
            _uiState.value = _uiState.value.copy(isProcessingImage = true, isSending = true)

            val service = meshService
            if (service == null) {
                Log.w(tag, "sendImage: meshService is null")
                _uiState.value = _uiState.value.copy(
                    isProcessingImage = false,
                    isSending = false,
                    error = "Service not connected"
                )
                return@launch
            }

            Log.d(tag, "sendImage: calling meshService.sendImageTo target=$target, uri=$uri")
            val result = service.sendImageTo(target, uri, caption)

            result.fold(
                onSuccess = { messageId ->
                    Log.d(tag, "sendImage: success, messageId=$messageId")
                    _uiState.value = _uiState.value.copy(
                        isProcessingImage = false,
                        isSending = false,
                        selectedImageUri = null,  // Clear after successful send
                        error = null
                    )
                },
                onFailure = { error ->
                    Log.e(tag, "sendImage: failed", error)
                    _uiState.value = _uiState.value.copy(
                        isProcessingImage = false,
                        isSending = false,
                        error = error.message ?: "Failed to send image"
                    )
                }
            )
        }
    }

    /**
     * Clear any error message.
     */
    fun clearError() {
        _uiState.value = _uiState.value.copy(error = null)
    }

    /**
     * Delete a message (locally and sync to server).
     */
    fun deleteMessage(message: MessageEntity) {
        Log.d(tag, "deleteMessage: id=${message.id}")
        viewModelScope.launch {
            // Delete locally
            database.messageDao().deleteMessage(message.id)
            // Send delete to server
            meshService?.deleteMessage(message.id, message.conversationId)
        }
    }
}
