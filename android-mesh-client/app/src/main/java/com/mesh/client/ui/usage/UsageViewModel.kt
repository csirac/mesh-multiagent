package com.mesh.client.ui.usage

import android.app.Application
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.os.IBinder
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.mesh.client.data.remote.CcUsageResponse
import com.mesh.client.data.remote.ConnectionState
import com.mesh.client.data.remote.MeshSocketClient
import com.mesh.client.data.remote.UsageManager
import com.mesh.client.service.MeshService
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.launch
import javax.inject.Inject

private const val TAG = "UsageViewModel"

@HiltViewModel
class UsageViewModel @Inject constructor(
    application: Application,
    private val socketClient: MeshSocketClient,
    private val usageManager: UsageManager
) : AndroidViewModel(application) {

    val usage: StateFlow<CcUsageResponse?> = usageManager.currentUsage
    val isLoading: StateFlow<Boolean> = usageManager.isLoading

    val isConnected: StateFlow<Boolean> = socketClient.connectionState
        .map { it is ConnectionState.Connected }
        .stateIn(viewModelScope, SharingStarted.WhileSubscribed(5000), false)

    private var meshService: MeshService? = null
    private var serviceBound = false

    private val serviceConnection = object : ServiceConnection {
        override fun onServiceConnected(name: ComponentName?, service: IBinder?) {
            Log.d(TAG, "onServiceConnected: service=$service")
            try {
                val binder = service as MeshService.LocalBinder
                meshService = binder.getService()
                serviceBound = true
                Log.d(TAG, "onServiceConnected: meshService=$meshService, auto-fetching usage")
                // Auto-fetch on connect
                fetchUsage()
            } catch (e: Exception) {
                Log.e(TAG, "onServiceConnected: error binding to service", e)
            }
        }

        override fun onServiceDisconnected(name: ComponentName?) {
            Log.d(TAG, "onServiceDisconnected")
            meshService = null
            serviceBound = false
        }
    }

    init {
        Log.i(TAG, "=== init: ViewModel created, binding to MeshService ===")
        try {
            val intent = Intent(application, MeshService::class.java)
            val bound = application.bindService(intent, serviceConnection, Context.BIND_AUTO_CREATE)
            Log.i(TAG, "init: bindService returned $bound")
        } catch (e: Exception) {
            Log.e(TAG, "init: bindService failed", e)
        }
    }

    override fun onCleared() {
        if (serviceBound) {
            getApplication<Application>().unbindService(serviceConnection)
            serviceBound = false
        }
        super.onCleared()
    }

    fun fetchUsage() {
        Log.d(TAG, "fetchUsage: meshService=$meshService, serviceBound=$serviceBound")
        if (meshService == null) {
            Log.w(TAG, "fetchUsage: meshService is null, cannot request usage")
            return
        }
        try {
            meshService?.requestCcUsage()
            Log.d(TAG, "fetchUsage: requestCcUsage called successfully")
        } catch (e: Exception) {
            Log.e(TAG, "fetchUsage: error calling requestCcUsage", e)
        }
    }
}
