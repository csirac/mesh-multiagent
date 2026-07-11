package com.mesh.client.data.remote

import android.util.Log
import com.google.firebase.messaging.FirebaseMessaging
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.tasks.await
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Manages FCM token lifecycle and registration with the mesh router.
 *
 * Responsibilities:
 * - Retrieve initial FCM token on startup
 * - Handle token refresh events
 * - Send token to mesh router when connected
 * - Track token registration state
 */
@Singleton
class FcmTokenManager @Inject constructor(
    private val socketClient: MeshSocketClient
) {
    private val tag = "FcmTokenManager"
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)

    // Current FCM token
    private val _currentToken = MutableStateFlow<String?>(null)
    val currentToken: StateFlow<String?> = _currentToken.asStateFlow()

    // Whether token has been sent to router since last connection
    private var tokenSentToRouter = false

    init {
        // Observe registration state to send token only after REGISTER is sent
        scope.launch {
            socketClient.isRegistered.collect { registered ->
                if (registered && !tokenSentToRouter) {
                    _currentToken.value?.let { token ->
                        sendTokenToRouter(token)
                    }
                } else if (!registered) {
                    // Reset flag on disconnect so we re-send on next connection
                    tokenSentToRouter = false
                }
            }
        }
    }

    /**
     * Initialize FCM and get the current token.
     * Call this on app startup.
     *
     * This method gracefully handles the case where Firebase is not configured
     * (i.e., google-services.json is missing).
     */
    fun initialize() {
        scope.launch {
            try {
                val token = FirebaseMessaging.getInstance().token.await()
                Log.i(tag, "FCM token retrieved successfully")
                _currentToken.value = token

                // Send to router if already registered
                if (socketClient.isRegistered.value) {
                    sendTokenToRouter(token)
                }
            } catch (e: IllegalStateException) {
                // Firebase not configured (google-services.json missing)
                Log.w(tag, "Firebase not configured - push notifications disabled")
            } catch (e: Exception) {
                Log.e(tag, "Failed to get FCM token", e)
            }
        }
    }

    /**
     * Called when FCM generates a new token.
     * This can happen when:
     * - App is installed fresh
     * - App data is cleared
     * - App is restored on a new device
     * - Token expires and Firebase refreshes it
     */
    fun onTokenRefreshed(token: String) {
        Log.i(tag, "FCM token refreshed")
        _currentToken.value = token
        tokenSentToRouter = false  // Force re-registration

        // Send immediately if registered
        if (socketClient.isRegistered.value) {
            scope.launch {
                sendTokenToRouter(token)
            }
        }
    }

    /**
     * Send the FCM token to the mesh router.
     * The router stores this token and uses it to send push notifications
     * when the client is disconnected.
     */
    private fun sendTokenToRouter(token: String) {
        val nodeId = socketClient.nodeId
        if (nodeId == null) {
            Log.w(tag, "Cannot send FCM token: not registered with router")
            return
        }

        val message = com.mesh.client.data.remote.protocol.MessageFactory.makeControl(
            fromNode = nodeId,
            action = com.mesh.client.data.remote.protocol.ControlAction.REGISTER_PUSH_TOKEN,
            fcmToken = token
        )

        val sent = socketClient.send(message)
        if (sent) {
            Log.i(tag, "FCM token sent to router")
            tokenSentToRouter = true
        } else {
            Log.w(tag, "Failed to send FCM token to router")
        }
    }
}
