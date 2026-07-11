package com.mesh.client.ui.channels

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.mesh.client.data.local.db.MeshDatabase
import com.mesh.client.data.local.db.entities.ChannelEntity
import com.mesh.client.data.local.db.entities.RosterEntry
import com.mesh.client.data.remote.MeshSocketClient
import com.mesh.client.data.remote.protocol.buildChannelAddress
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.flow.flatMapLatest
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch
import javax.inject.Inject

data class ChannelListItem(
    val name: String,
    val description: String,
    val memberCount: Int,
    val isMember: Boolean,
    val address: String,
    val createdBy: String,
    val canDelete: Boolean
)

@OptIn(ExperimentalCoroutinesApi::class)
@HiltViewModel
class ChannelListViewModel @Inject constructor(
    private val database: MeshDatabase,
    private val socketClient: MeshSocketClient
) : ViewModel() {

    private val _isLoading = MutableStateFlow(true)
    val isLoading: StateFlow<Boolean> = _isLoading.asStateFlow()

    private val _showCreateDialog = MutableStateFlow(false)
    val showCreateDialog: StateFlow<Boolean> = _showCreateDialog.asStateFlow()

    private val _showInviteDialog = MutableStateFlow<String?>(null)
    val showInviteDialog: StateFlow<String?> = _showInviteDialog.asStateFlow()

    // All roster entries for invite dialog
    val rosterEntries: StateFlow<List<RosterEntry>> = database.rosterDao().getAllRoster()
        .stateIn(
            scope = viewModelScope,
            started = SharingStarted.WhileSubscribed(5000),
            initialValue = emptyList()
        )

    private val myNodeId: String?
        get() = socketClient.nodeId

    // Track node ID changes to refresh membership query
    private val nodeIdFlow = MutableStateFlow(socketClient.nodeId ?: "")

    // Combine all channels with my memberships (reactive to node ID changes)
    val channels: StateFlow<List<ChannelListItem>> = combine(
        database.channelDao().getAllChannels(),
        nodeIdFlow.flatMapLatest { nodeId ->
            database.channelDao().getNodeChannels(nodeId)
        }
    ) { allChannels, myMemberships ->
        _isLoading.value = false
        val myChannelNames = myMemberships.map { it.channelName }.toSet()

        val currentNodeId = myNodeId ?: ""
        allChannels.map { channel ->
            ChannelListItem(
                name = channel.name,
                description = channel.description,
                memberCount = channel.memberCount,
                isMember = channel.name in myChannelNames,
                address = buildChannelAddress(channel.name),
                createdBy = channel.createdBy,
                canDelete = channel.createdBy == currentNodeId
            )
        }
    }.stateIn(
        scope = viewModelScope,
        started = SharingStarted.WhileSubscribed(5000),
        initialValue = emptyList()
    )

    init {
        // Request channel list on init
        refreshChannels()
    }

    fun refreshChannels() {
        viewModelScope.launch {
            _isLoading.value = true
            // Update node ID in case it changed (e.g., after reconnect)
            nodeIdFlow.value = socketClient.nodeId ?: ""
            socketClient.listChannels()
            // Set a timeout to clear loading state in case the server doesn't respond
            kotlinx.coroutines.delay(3000)
            _isLoading.value = false
        }
    }

    fun showCreateDialog() {
        _showCreateDialog.value = true
    }

    fun hideCreateDialog() {
        _showCreateDialog.value = false
    }

    fun showInviteDialog(channelName: String) {
        _showInviteDialog.value = channelName
    }

    fun hideInviteDialog() {
        _showInviteDialog.value = null
    }

    fun createChannel(name: String, description: String) {
        viewModelScope.launch {
            socketClient.createChannel(name, description)
            _showCreateDialog.value = false
            // Refresh the list to pick up the new channel
            kotlinx.coroutines.delay(500)  // Give the ACK time to arrive
            refreshChannels()
        }
    }

    fun joinChannel(name: String) {
        viewModelScope.launch {
            socketClient.joinChannel(name)
            kotlinx.coroutines.delay(500)
            refreshChannels()
        }
    }

    fun leaveChannel(name: String) {
        viewModelScope.launch {
            socketClient.leaveChannel(name)
            kotlinx.coroutines.delay(500)
            refreshChannels()
        }
    }

    fun deleteChannel(name: String) {
        viewModelScope.launch {
            socketClient.deleteChannel(name)
            kotlinx.coroutines.delay(500)
            refreshChannels()
        }
    }

    fun inviteToChannel(channelName: String, nodeId: String) {
        android.util.Log.d("ChannelListVM", "inviteToChannel: channel=$channelName, nodeId=$nodeId")
        viewModelScope.launch {
            val result = socketClient.inviteToChannel(channelName, nodeId)
            android.util.Log.d("ChannelListVM", "inviteToChannel result: $result")
            kotlinx.coroutines.delay(500)
            refreshChannels()
        }
    }
}
