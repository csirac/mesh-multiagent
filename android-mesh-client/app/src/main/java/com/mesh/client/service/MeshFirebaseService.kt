package com.mesh.client.service

import android.content.Intent
import android.util.Log
import com.google.firebase.messaging.FirebaseMessagingService
import com.google.firebase.messaging.RemoteMessage
import com.mesh.client.data.remote.FcmTokenManager
import com.mesh.client.util.NotificationHelper
import dagger.hilt.android.AndroidEntryPoint
import javax.inject.Inject

/**
 * Firebase Cloud Messaging service for push notifications.
 *
 * Handles:
 * - FCM token generation and refresh
 * - Incoming push notifications when app is in background
 * - Data messages from the mesh router
 */
@AndroidEntryPoint
class MeshFirebaseService : FirebaseMessagingService() {

    private val tag = "MeshFirebaseService"

    @Inject lateinit var notificationHelper: NotificationHelper
    @Inject lateinit var fcmTokenManager: FcmTokenManager

    override fun onCreate() {
        super.onCreate()
        Log.i(tag, "MeshFirebaseService created")
    }

    /**
     * Called when a new FCM token is generated.
     * This happens on first app start and when the token is refreshed.
     */
    override fun onNewToken(token: String) {
        super.onNewToken(token)
        Log.i(tag, "New FCM token received")
        fcmTokenManager.onTokenRefreshed(token)
    }

    /**
     * Called when a message is received from FCM.
     *
     * Message types from mesh router:
     * - notification: Display notification with title/body
     * - data: Contains mesh message details (from_node, content, etc.)
     */
    override fun onMessageReceived(message: RemoteMessage) {
        super.onMessageReceived(message)
        Log.d(tag, "FCM message received from: ${message.from}")

        // Handle notification payload (when app is in foreground)
        message.notification?.let { notification ->
            Log.d(tag, "Notification payload: ${notification.title} - ${notification.body}")
            notificationHelper.showMessageNotification(
                fromNode = notification.title ?: "Mesh",
                content = notification.body ?: ""
            )
        }

        // Handle data payload (custom mesh message data)
        if (message.data.isNotEmpty()) {
            Log.d(tag, "Data payload: ${message.data}")
            handleDataMessage(message.data)
        }
    }

    /**
     * Handle data-only messages from the mesh router.
     *
     * Expected data fields:
     * - from_node: The sender's node ID
     * - content: The message content
     * - message_type: The type of mesh message (message, confirm_request, etc.)
     * - message_id: The unique message ID
     */
    private fun handleDataMessage(data: Map<String, String>) {
        val fromNode = data["from_node"]
        val content = data["content"]
        val messageType = data["message_type"] ?: "message"

        when (messageType) {
            "message" -> {
                if (fromNode != null && content != null) {
                    notificationHelper.showMessageNotification(fromNode, content)
                }
            }
            "confirm_request" -> {
                val messageId = data["message_id"] ?: return
                val toolName = data["tool_name"] ?: "unknown"
                val preview = data["preview"] ?: ""
                if (fromNode != null) {
                    notificationHelper.showConfirmNotification(
                        fromNode = fromNode,
                        messageId = messageId,
                        toolName = toolName,
                        preview = preview
                    )
                }
            }
            else -> {
                Log.d(tag, "Unhandled message type: $messageType")
            }
        }
    }
}
