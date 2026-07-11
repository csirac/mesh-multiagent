package com.mesh.client.ui.settings

import android.app.Application
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.os.IBinder
import androidx.datastore.core.DataStore
import androidx.datastore.preferences.core.Preferences
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.stringPreferencesKey
import androidx.datastore.preferences.preferencesDataStore
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.mesh.client.data.remote.ConnectionState
import com.mesh.client.data.remote.MeshSocketClient
import com.mesh.client.data.remote.protocol.buildUserNodeId
import com.mesh.client.service.MeshService
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.launch
import javax.inject.Inject

private val Context.dataStore: DataStore<Preferences> by preferencesDataStore(name = "settings")

private object PreferencesKeys {
    val HOST = stringPreferencesKey("host")
    val PORT = stringPreferencesKey("port")
    val USE_TLS = stringPreferencesKey("use_tls")
    val AUTH_TOKEN = stringPreferencesKey("auth_token")
    val NICKNAME = stringPreferencesKey("nickname")
}

data class SettingsUiState(
    val host: String = "192.168.1.100",
    val port: String = "443",
    val useTls: Boolean = true,
    val authToken: String = "",
    val nickname: String = "",
    val isConnected: Boolean = false,
    val error: String? = null
)

@HiltViewModel
class SettingsViewModel @Inject constructor(
    application: Application,
    private val socketClient: MeshSocketClient
) : AndroidViewModel(application) {

    private val dataStore = application.dataStore

    private val _uiState = MutableStateFlow(SettingsUiState())
    val uiState: StateFlow<SettingsUiState> = _uiState.asStateFlow()

    private var meshService: MeshService? = null
    private var serviceBound = false

    private val serviceConnection = object : ServiceConnection {
        override fun onServiceConnected(name: ComponentName?, service: IBinder?) {
            val binder = service as MeshService.LocalBinder
            meshService = binder.getService()
            serviceBound = true
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

        // Load saved settings
        viewModelScope.launch {
            loadSettings()
        }

        // Observe connection state
        viewModelScope.launch {
            socketClient.connectionState.collect { state ->
                _uiState.value = _uiState.value.copy(
                    isConnected = state is ConnectionState.Connected,
                    error = when (state) {
                        is ConnectionState.Error -> state.message
                        else -> null
                    }
                )
            }
        }
    }

    override fun onCleared() {
        if (serviceBound) {
            getApplication<Application>().unbindService(serviceConnection)
            serviceBound = false
        }
        super.onCleared()
    }

    private suspend fun loadSettings() {
        val prefs = dataStore.data.first()
        _uiState.value = SettingsUiState(
            host = prefs[PreferencesKeys.HOST] ?: "192.168.1.100",
            port = prefs[PreferencesKeys.PORT] ?: "443",
            useTls = prefs[PreferencesKeys.USE_TLS]?.toBoolean() ?: true,
            authToken = prefs[PreferencesKeys.AUTH_TOKEN] ?: "",
            nickname = prefs[PreferencesKeys.NICKNAME] ?: ""
        )
    }

    fun updateHost(value: String) {
        _uiState.value = _uiState.value.copy(host = value)
    }

    fun updatePort(value: String) {
        _uiState.value = _uiState.value.copy(port = value.filter { it.isDigit() })
    }

    fun updateAuthToken(value: String) {
        _uiState.value = _uiState.value.copy(authToken = value)
    }

    fun updateNickname(value: String) {
        _uiState.value = _uiState.value.copy(nickname = value.lowercase().filter { it.isLetterOrDigit() })
    }

    fun updateUseTls(value: Boolean) {
        _uiState.value = _uiState.value.copy(useTls = value)
    }

    fun saveSettings() {
        viewModelScope.launch {
            dataStore.edit { prefs ->
                prefs[PreferencesKeys.HOST] = _uiState.value.host
                prefs[PreferencesKeys.PORT] = _uiState.value.port
                prefs[PreferencesKeys.USE_TLS] = _uiState.value.useTls.toString()
                prefs[PreferencesKeys.AUTH_TOKEN] = _uiState.value.authToken
                prefs[PreferencesKeys.NICKNAME] = _uiState.value.nickname
            }
        }
    }

    fun connect() {
        val state = _uiState.value
        val port = state.port.toIntOrNull() ?: return
        val nodeId = buildUserNodeId(state.nickname)

        meshService?.connect(
            host = state.host,
            port = port,
            nodeId = nodeId,
            nickname = state.nickname,
            authToken = state.authToken.ifEmpty { null },
            useTls = state.useTls
        )
    }

    fun disconnect() {
        meshService?.disconnect()
    }
}
