package com.mesh.client.ui.channels

import androidx.lifecycle.SavedStateHandle
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.mesh.client.data.local.db.MeshDatabase
import com.mesh.client.data.local.db.entities.ChannelMemberEntity
import com.mesh.client.data.local.db.entities.RosterEntry
import com.mesh.client.data.remote.MeshSocketClient
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch
import javax.inject.Inject

data class ChannelMemberItem(
    val nodeId: String,
    val nickname: String,
    val isOnline: Boolean,
    val joinedAt: String,
    val nodeType: String
)

@HiltViewModel
class ChannelMembersViewModel @Inject constructor(
    savedStateHandle: SavedStateHandle,
    private val database: MeshDatabase,
    private val socketClient: MeshSocketClient
) : ViewModel() {

    val channelName: String = savedStateHandle.get<String>("channelName") ?: ""

    private val _isLoading = MutableStateFlow(true)
    val isLoading: StateFlow<Boolean> = _isLoading.asStateFlow()

    // Combine channel members with roster to get online status and nicknames
    val members: StateFlow<List<ChannelMemberItem>> = combine(
        database.channelDao().getChannelMembers(channelName),
        database.rosterDao().getAllRoster()
    ) { channelMembers, roster ->
        _isLoading.value = false

        val rosterMap = roster.associateBy { it.nodeId }

        channelMembers.map { member ->
            val rosterEntry = rosterMap[member.nodeId]
            ChannelMemberItem(
                nodeId = member.nodeId,
                nickname = rosterEntry?.nickname ?: parseNickname(member.nodeId),
                isOnline = rosterEntry?.isOnline ?: false,
                joinedAt = member.joinedAt,
                nodeType = rosterEntry?.nodeType ?: parseNodeType(member.nodeId)
            )
        }.sortedWith(
            compareByDescending<ChannelMemberItem> { it.isOnline }
                .thenBy { it.nickname.lowercase() }
        )
    }.stateIn(
        scope = viewModelScope,
        started = SharingStarted.WhileSubscribed(5000),
        initialValue = emptyList()
    )

    init {
        refreshMembers()
    }

    fun refreshMembers() {
        viewModelScope.launch {
            _isLoading.value = true
            socketClient.getChannelMembers(channelName)
            // Set a timeout to clear loading state
            kotlinx.coroutines.delay(3000)
            _isLoading.value = false
        }
    }

    private fun parseNickname(nodeId: String): String {
        // Extract nickname from node ID like "user:yourname" or "agent:coder:alice"
        val parts = nodeId.split(":")
        return when {
            parts.size >= 2 && parts[0] == "user" -> parts[1]
            parts.size >= 3 && parts[0] == "agent" -> parts[2]
            parts.size >= 2 && parts[0] == "agent" -> parts[1]
            else -> nodeId
        }
    }

    private fun parseNodeType(nodeId: String): String {
        val parts = nodeId.split(":")
        return when {
            parts.isNotEmpty() && parts[0] == "user" -> "user"
            parts.size >= 2 && parts[0] == "agent" -> parts[1]
            else -> "unknown"
        }
    }
}
