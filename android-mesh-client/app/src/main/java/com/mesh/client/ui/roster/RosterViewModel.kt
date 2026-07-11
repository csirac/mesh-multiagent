package com.mesh.client.ui.roster

import android.app.Application
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.os.IBinder
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import com.mesh.client.data.local.db.MeshDatabase
import com.mesh.client.data.local.db.entities.RosterEntry
import com.mesh.client.service.MeshService
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.Flow
import javax.inject.Inject

@HiltViewModel
class RosterViewModel @Inject constructor(
    application: Application,
    private val database: MeshDatabase
) : AndroidViewModel(application) {

    private val tag = "RosterViewModel"

    val roster: Flow<List<RosterEntry>> = database.rosterDao().getAllNodes()

    val agents: Flow<List<RosterEntry>> = database.rosterDao().getAgents()

    val users: Flow<List<RosterEntry>> = database.rosterDao().getUsers()

    val onlineNodes: Flow<List<RosterEntry>> = database.rosterDao().getOnlineNodes()

    private var meshService: MeshService? = null

    private val serviceConnection = object : ServiceConnection {
        override fun onServiceConnected(name: ComponentName?, service: IBinder?) {
            val binder = service as MeshService.LocalBinder
            meshService = binder.getService()
            Log.d(tag, "Bound to MeshService")
        }

        override fun onServiceDisconnected(name: ComponentName?) {
            meshService = null
        }
    }

    init {
        val intent = Intent(application, MeshService::class.java)
        application.bindService(intent, serviceConnection, Context.BIND_AUTO_CREATE)
    }

    /**
     * Request fresh roster status from the router.
     * Call this when the roster tab becomes visible.
     */
    fun refreshStatus() {
        meshService?.requestNodeStatus()
            ?: Log.w(tag, "refreshStatus: meshService not bound yet")
    }

    suspend fun deleteEntry(nodeId: String) {
        database.rosterDao().delete(nodeId)
    }

    override fun onCleared() {
        super.onCleared()
        try {
            getApplication<Application>().unbindService(serviceConnection)
        } catch (e: Exception) {
            Log.w(tag, "Failed to unbind service: ${e.message}")
        }
    }
}
