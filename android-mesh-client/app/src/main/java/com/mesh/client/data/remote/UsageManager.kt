package com.mesh.client.data.remote

import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Usage data for a single rate-limit window.
 */
data class UsageWindow(
    val name: String,        // e.g. "five_hour", "seven_day"
    val utilization: Double,  // 0.0 to 100.0
    val resetsAt: String?     // ISO timestamp
)

/**
 * Extra usage (overages) info.
 */
data class ExtraUsage(
    val isEnabled: Boolean = false,
    val usedCredits: Double = 0.0,
    val monthlyLimit: Double = 0.0
)

/**
 * Usage data for a single CC account.
 */
data class AccountUsage(
    val label: String,
    val subscriptionType: String = "unknown",
    val error: String? = null,
    val windows: List<UsageWindow> = emptyList(),
    val extraUsage: ExtraUsage? = null
)

/**
 * Full CC usage response from the router.
 */
data class CcUsageResponse(
    val accounts: List<AccountUsage>
)

/**
 * Manages CC usage responses from the router.
 */
@Singleton
class UsageManager @Inject constructor() {

    private val _currentUsage = MutableStateFlow<CcUsageResponse?>(null)
    val currentUsage: StateFlow<CcUsageResponse?> = _currentUsage.asStateFlow()

    private val _isLoading = MutableStateFlow(false)
    val isLoading: StateFlow<Boolean> = _isLoading.asStateFlow()

    fun setLoading(loading: Boolean) {
        _isLoading.value = loading
    }

    fun setUsage(usage: CcUsageResponse) {
        _isLoading.value = false
        _currentUsage.value = usage
    }

    fun clear() {
        _currentUsage.value = null
        _isLoading.value = false
    }
}
