package com.mesh.client.util

import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import android.os.Build
import androidx.core.app.NotificationCompat
import androidx.core.app.NotificationManagerCompat
import com.mesh.client.R
import com.mesh.client.data.remote.protocol.getDisplayName
import com.mesh.client.ui.MainActivity
import dagger.hilt.android.qualifiers.ApplicationContext
import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class NotificationHelper @Inject constructor(
    @ApplicationContext private val context: Context
) {
    companion object {
        const val CHANNEL_MESSAGES = "messages"
        const val CHANNEL_SERVICE = "service"
        const val CHANNEL_CONFIRM = "confirm"

        const val NOTIFICATION_ID_SERVICE = 1
        const val NOTIFICATION_ID_MESSAGE_BASE = 1000
        const val NOTIFICATION_ID_CONFIRM_BASE = 2000

        private var messageNotificationId = NOTIFICATION_ID_MESSAGE_BASE
        private var confirmNotificationId = NOTIFICATION_ID_CONFIRM_BASE
    }

    private val notificationManager = NotificationManagerCompat.from(context)

    // Track notification IDs per node for clearing when conversation is opened
    private val nodeNotificationIds = mutableMapOf<String, MutableSet<Int>>()

    /**
     * Create all notification channels. Should be called on app startup.
     */
    fun createChannels() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val manager = context.getSystemService(NotificationManager::class.java)

            // Messages channel
            val messagesChannel = NotificationChannel(
                CHANNEL_MESSAGES,
                context.getString(R.string.notification_channel_messages),
                NotificationManager.IMPORTANCE_HIGH
            ).apply {
                description = context.getString(R.string.notification_channel_messages_desc)
                enableVibration(true)
                enableLights(true)
            }

            // Service channel (low priority, just shows connection status)
            val serviceChannel = NotificationChannel(
                CHANNEL_SERVICE,
                context.getString(R.string.notification_channel_service),
                NotificationManager.IMPORTANCE_LOW
            ).apply {
                description = context.getString(R.string.notification_channel_service_desc)
                setShowBadge(false)
            }

            // Confirmation channel (high priority, needs user action)
            val confirmChannel = NotificationChannel(
                CHANNEL_CONFIRM,
                context.getString(R.string.notification_channel_confirm),
                NotificationManager.IMPORTANCE_HIGH
            ).apply {
                description = context.getString(R.string.notification_channel_confirm_desc)
                enableVibration(true)
                enableLights(true)
            }

            manager.createNotificationChannels(listOf(messagesChannel, serviceChannel, confirmChannel))
        }
    }

    /**
     * Build the foreground service notification.
     */
    fun buildServiceNotification(isConnected: Boolean): android.app.Notification {
        val intent = Intent(context, MainActivity::class.java).apply {
            flags = Intent.FLAG_ACTIVITY_SINGLE_TOP
        }
        val pendingIntent = PendingIntent.getActivity(
            context,
            0,
            intent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )

        val statusText = if (isConnected) {
            context.getString(R.string.connection_connected)
        } else {
            context.getString(R.string.connection_disconnected)
        }

        return NotificationCompat.Builder(context, CHANNEL_SERVICE)
            .setContentTitle(context.getString(R.string.app_name))
            .setContentText(statusText)
            .setSmallIcon(android.R.drawable.ic_dialog_info) // TODO: custom icon
            .setOngoing(true)
            .setContentIntent(pendingIntent)
            .setPriority(NotificationCompat.PRIORITY_LOW)
            .build()
    }

    /**
     * Show a notification for an incoming message.
     *
     * @param fromNode Display name/label for the notification title
     * @param content Message content
     * @param conversationKey Key for tracking notifications per conversation.
     *   Used by cancelNotificationsForNode to dismiss notifications when the conversation is opened.
     *   For DMs this is the sender's node ID, for channels this is "channel:name".
     *   If null, defaults to fromNode.
     */
    fun showMessageNotification(fromNode: String, content: String, conversationKey: String? = null) {
        val trackingKey = conversationKey ?: fromNode

        val intent = Intent(context, MainActivity::class.java).apply {
            flags = Intent.FLAG_ACTIVITY_SINGLE_TOP
            putExtra("target", trackingKey)
        }
        val pendingIntent = PendingIntent.getActivity(
            context,
            trackingKey.hashCode(),
            intent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )

        val displayName = getDisplayName(fromNode)

        val notification = NotificationCompat.Builder(context, CHANNEL_MESSAGES)
            .setContentTitle(displayName)
            .setContentText(content.take(100))
            .setStyle(NotificationCompat.BigTextStyle().bigText(content.take(500)))
            .setSmallIcon(android.R.drawable.ic_dialog_email) // TODO: custom icon
            .setAutoCancel(true)
            .setContentIntent(pendingIntent)
            .setPriority(NotificationCompat.PRIORITY_HIGH)
            .build()

        try {
            val notifId = messageNotificationId++
            if (messageNotificationId > NOTIFICATION_ID_MESSAGE_BASE + 100) {
                messageNotificationId = NOTIFICATION_ID_MESSAGE_BASE
            }
            notificationManager.notify(notifId, notification)

            // Track notification ID by conversation key so we can clear when conversation opens
            nodeNotificationIds.getOrPut(trackingKey) { mutableSetOf() }.add(notifId)
        } catch (e: SecurityException) {
            // Notification permission not granted
        }
    }

    /**
     * Show a notification for a confirmation request.
     */
    fun showConfirmNotification(
        fromNode: String,
        messageId: String,
        toolName: String,
        preview: String
    ): Int {
        val id = confirmNotificationId++
        if (confirmNotificationId > NOTIFICATION_ID_CONFIRM_BASE + 100) {
            confirmNotificationId = NOTIFICATION_ID_CONFIRM_BASE
        }

        val intent = Intent(context, MainActivity::class.java).apply {
            flags = Intent.FLAG_ACTIVITY_SINGLE_TOP
            putExtra("confirm_message_id", messageId)
            putExtra("confirm_from", fromNode)
        }
        val pendingIntent = PendingIntent.getActivity(
            context,
            id,
            intent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )

        val displayName = getDisplayName(fromNode)

        val notification = NotificationCompat.Builder(context, CHANNEL_CONFIRM)
            .setContentTitle(context.getString(R.string.confirm_title))
            .setContentText("$displayName: ${context.getString(R.string.confirm_tool, toolName)}")
            .setStyle(NotificationCompat.BigTextStyle().bigText(preview.take(500)))
            .setSmallIcon(android.R.drawable.ic_dialog_alert) // TODO: custom icon
            .setAutoCancel(true)
            .setContentIntent(pendingIntent)
            .setPriority(NotificationCompat.PRIORITY_HIGH)
            // Action buttons
            .addAction(
                android.R.drawable.ic_menu_close_clear_cancel,
                context.getString(R.string.confirm_reject),
                createConfirmActionIntent(id, messageId, fromNode, false)
            )
            .addAction(
                android.R.drawable.ic_menu_send,
                context.getString(R.string.confirm_approve),
                createConfirmActionIntent(id, messageId, fromNode, true)
            )
            .build()

        try {
            notificationManager.notify(id, notification)
        } catch (e: SecurityException) {
            // Notification permission not granted
        }

        return id
    }

    private fun createConfirmActionIntent(
        notificationId: Int,
        messageId: String,
        fromNode: String,
        confirmed: Boolean
    ): PendingIntent {
        val intent = Intent(context, ConfirmActionReceiver::class.java).apply {
            action = if (confirmed) ConfirmActionReceiver.ACTION_APPROVE else ConfirmActionReceiver.ACTION_REJECT
            putExtra(ConfirmActionReceiver.EXTRA_NOTIFICATION_ID, notificationId)
            putExtra(ConfirmActionReceiver.EXTRA_MESSAGE_ID, messageId)
            putExtra(ConfirmActionReceiver.EXTRA_FROM_NODE, fromNode)
        }
        return PendingIntent.getBroadcast(
            context,
            notificationId * 2 + if (confirmed) 1 else 0,
            intent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )
    }

    /**
     * Cancel a notification by ID.
     */
    fun cancel(notificationId: Int) {
        notificationManager.cancel(notificationId)
    }

    /**
     * Cancel all message notifications.
     */
    fun cancelAllMessages() {
        for (i in NOTIFICATION_ID_MESSAGE_BASE until NOTIFICATION_ID_MESSAGE_BASE + 100) {
            notificationManager.cancel(i)
        }
        nodeNotificationIds.clear()
    }

    /**
     * Cancel all notifications for a specific node (e.g., when opening that conversation).
     */
    fun cancelNotificationsForNode(nodeId: String) {
        val ids = nodeNotificationIds.remove(nodeId) ?: return
        for (id in ids) {
            notificationManager.cancel(id)
        }
    }
}

/**
 * Broadcast receiver for handling confirmation actions from notifications.
 *
 * When the user taps Approve or Reject on a confirmation notification,
 * this receiver handles the action and communicates with MeshService.
 */
class ConfirmActionReceiver : android.content.BroadcastReceiver() {

    companion object {
        const val ACTION_APPROVE = "com.mesh.client.CONFIRM_APPROVE"
        const val ACTION_REJECT = "com.mesh.client.CONFIRM_REJECT"
        const val EXTRA_NOTIFICATION_ID = "notification_id"
        const val EXTRA_MESSAGE_ID = "message_id"
        const val EXTRA_FROM_NODE = "from_node"
    }

    override fun onReceive(context: Context?, intent: Intent?) {
        if (context == null || intent == null) return

        val notificationId = intent.getIntExtra(EXTRA_NOTIFICATION_ID, -1)
        val messageId = intent.getStringExtra(EXTRA_MESSAGE_ID) ?: return
        val fromNode = intent.getStringExtra(EXTRA_FROM_NODE) ?: return
        val approved = intent.action == ACTION_APPROVE

        android.util.Log.i("ConfirmActionReceiver", "Received: action=${intent.action}, messageId=$messageId, fromNode=$fromNode, approved=$approved")

        // Cancel the notification
        if (notificationId != -1) {
            val notificationManager = context.getSystemService(Context.NOTIFICATION_SERVICE) as android.app.NotificationManager
            notificationManager.cancel(notificationId)
        }

        // Send an intent to MainActivity to handle the confirmation
        // This is needed because we can't directly access the service from a manifest-registered receiver
        val handleIntent = Intent(context, com.mesh.client.ui.MainActivity::class.java).apply {
            flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_SINGLE_TOP
            action = if (approved) ACTION_APPROVE else ACTION_REJECT
            putExtra(EXTRA_MESSAGE_ID, messageId)
            putExtra(EXTRA_FROM_NODE, fromNode)
        }
        context.startActivity(handleIntent)
    }
}
