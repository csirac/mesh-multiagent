package com.mesh.client.data.remote

import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Represents a context entry in an agent's status response.
 */
data class StatusContextEntry(
    val from: String,
    val content: String,
    val timestamp: String,
    val type: String  // "message", "tool_call", or "tool_result"
)

/**
 * Heartbeat-lite status summary from the agent.
 */
data class StatusSummary(
    val state: String = "",
    val workerElapsedS: Double? = null,
    val contextTokens: Int = 0,
    val historyTurns: Int = 0,
    val historyPct: Double = 0.0,
    val memoryPool: Int = 0,
    val memoryActive: Int = 0,
    val uptimeS: Double = 0.0,
    val activeMap: String? = null
)

/**
 * Full diagnostic report from agent_status (6 sections).
 */
data class DiagnosticReport(
    val identity: Map<String, Any?>? = null,
    val llm: Map<String, Any?>? = null,
    val router: Map<String, Any?>? = null,
    val history: Map<String, Any?>? = null,
    val memory: Map<String, Any?>? = null,
    val contextHealth: Map<String, Any?>? = null
)

/**
 * Represents a status response from an agent.
 */
data class StatusResponse(
    val fromNode: String,
    val context: List<StatusContextEntry>,
    val summary: String?,
    val currentActivity: String? = null,  // Real-time CC tool activity
    val hostname: String? = null,
    val model: String? = null,
    val backend: String? = null,
    val workingDirectory: String? = null,
    val statusSummary: StatusSummary? = null,  // Heartbeat-lite live status
    val diagnostics: DiagnosticReport? = null  // Full diagnostic report
)

/**
 * Manages agent status responses.
 *
 * This allows the UI to observe status responses and display them in dialogs.
 */
@Singleton
class StatusManager @Inject constructor() {

    private val _currentStatus = MutableStateFlow<StatusResponse?>(null)
    val currentStatus: StateFlow<StatusResponse?> = _currentStatus.asStateFlow()

    private val _isLoading = MutableStateFlow(false)
    val isLoading: StateFlow<Boolean> = _isLoading.asStateFlow()

    /**
     * Set loading state when requesting status.
     */
    fun setLoading(loading: Boolean) {
        _isLoading.value = loading
    }

    /**
     * Set the current status response.
     */
    fun setStatus(status: StatusResponse) {
        _isLoading.value = false
        _currentStatus.value = status
    }

    /**
     * Clear the current status (dismiss dialog).
     */
    fun clearStatus() {
        _currentStatus.value = null
        _isLoading.value = false
    }
}
