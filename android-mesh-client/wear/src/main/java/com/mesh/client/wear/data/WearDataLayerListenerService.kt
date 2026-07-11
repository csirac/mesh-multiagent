package com.mesh.client.wear.data

import android.util.Log
import com.google.android.gms.wearable.DataEventBuffer
import com.google.android.gms.wearable.MessageEvent
import com.google.android.gms.wearable.WearableListenerService
import dagger.hilt.EntryPoint
import dagger.hilt.InstallIn
import dagger.hilt.android.EntryPointAccessors
import dagger.hilt.components.SingletonComponent
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch

/**
 * Listens for data and message events from the phone app.
 * Runs as a background service to receive updates even when the app is not active.
 *
 * Note: WearableListenerService requires manual Hilt injection via EntryPoint
 * because it's instantiated by the system, not by Hilt.
 * Do NOT use @AndroidEntryPoint here - it doesn't work with WearableListenerService.
 */
class WearDataLayerListenerService : WearableListenerService() {

    companion object {
        private const val TAG = "WearDataListener"

        // Data paths
        const val PATH_MESSAGES = "/mesh/messages"
        const val PATH_CHANNELS = "/mesh/channels"
        const val PATH_CONNECTION_STATUS = "/mesh/connection"

        // Message paths
        const val PATH_REPLY = "/mesh/reply"
        const val PATH_MARK_READ = "/mesh/mark_read"
        const val PATH_SEND_MESSAGE = "/mesh/send_message"
    }

    // Use EntryPoint for injection since WearableListenerService is system-instantiated
    @EntryPoint
    @InstallIn(SingletonComponent::class)
    interface WearDataListenerEntryPoint {
        fun messageRepository(): WearMessageRepository
    }

    private lateinit var messageRepository: WearMessageRepository

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)

    override fun onCreate() {
        super.onCreate()
        // Manual injection via EntryPoint
        val entryPoint = EntryPointAccessors.fromApplication(
            applicationContext,
            WearDataListenerEntryPoint::class.java
        )
        messageRepository = entryPoint.messageRepository()
        Log.d(TAG, "WearDataLayerListenerService created, messageRepository injected")
    }

    override fun onDataChanged(dataEvents: DataEventBuffer) {
        Log.d(TAG, ">>> onDataChanged: ${dataEvents.count} events")

        // Must freeze/copy data before launching coroutine since buffer closes after this method returns
        val frozenEvents = dataEvents.map { event ->
            val path = event.dataItem.uri.path
            val dataItem = event.dataItem.freeze()  // Freeze to keep data after buffer closes
            Log.d(TAG, ">>> Event: type=${event.type}, path=$path, uri=${event.dataItem.uri}")
            path to dataItem
        }

        frozenEvents.forEach { (path, dataItem) ->
            if (path == null) {
                Log.w(TAG, ">>> Skipping event with null path")
                return@forEach
            }
            Log.d(TAG, ">>> Processing data at path: $path")

            scope.launch {
                when {
                    path.startsWith(PATH_MESSAGES) -> {
                        Log.d(TAG, ">>> Routing to handleMessagesUpdate")
                        messageRepository.handleMessagesUpdate(dataItem)
                    }
                    path.startsWith(PATH_CHANNELS) -> {
                        Log.d(TAG, ">>> Routing to handleChannelsUpdate")
                        messageRepository.handleChannelsUpdate(dataItem)
                    }
                    path.startsWith(PATH_CONNECTION_STATUS) -> {
                        Log.d(TAG, ">>> Routing to handleConnectionUpdate")
                        messageRepository.handleConnectionUpdate(dataItem)
                    }
                    else -> {
                        Log.w(TAG, ">>> Unknown path, not handled: $path")
                    }
                }
            }
        }
    }

    override fun onMessageReceived(messageEvent: MessageEvent) {
        Log.d(TAG, "onMessageReceived: ${messageEvent.path}")

        // Handle direct messages from phone (e.g., new message notification)
        scope.launch {
            when (messageEvent.path) {
                PATH_MESSAGES -> {
                    val data = String(messageEvent.data, Charsets.UTF_8)
                    messageRepository.handleNewMessage(data)
                }
            }
        }
    }
}
