package com.mesh.client.ui

import android.Manifest
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.os.IBinder
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.core.content.ContextCompat
import com.mesh.client.data.remote.ConfirmationManager
import com.mesh.client.service.MeshService
import com.mesh.client.ui.dialogs.ConfirmationDialog
import com.mesh.client.ui.navigation.MeshNavHost
import com.mesh.client.ui.theme.MeshClientTheme
import com.mesh.client.ui.theme.ThemeMode
import com.mesh.client.ui.theme.ThemePreference
import com.mesh.client.util.ConfirmActionReceiver
import dagger.hilt.android.AndroidEntryPoint
import javax.inject.Inject

@AndroidEntryPoint
class MainActivity : ComponentActivity() {

    private val tag = "MainActivity"

    @Inject lateinit var confirmationManager: ConfirmationManager

    private lateinit var themePreference: ThemePreference

    private var meshService: MeshService? = null
    private var serviceBound = false

    private val serviceConnection = object : ServiceConnection {
        override fun onServiceConnected(name: ComponentName?, service: IBinder?) {
            val binder = service as MeshService.LocalBinder
            meshService = binder.getService()
            serviceBound = true
            meshService?.setAppInForeground(true)
        }

        override fun onServiceDisconnected(name: ComponentName?) {
            meshService = null
            serviceBound = false
        }
    }

    private val notificationPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        // Notification permission result - proceed regardless
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Request notification permission on Android 13+
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.POST_NOTIFICATIONS)
                != PackageManager.PERMISSION_GRANTED
            ) {
                notificationPermissionLauncher.launch(Manifest.permission.POST_NOTIFICATIONS)
            }
        }

        // Start and bind to MeshService
        val serviceIntent = Intent(this, MeshService::class.java)
        ContextCompat.startForegroundService(this, serviceIntent)
        bindService(serviceIntent, serviceConnection, Context.BIND_AUTO_CREATE)

        // Handle intent extras (e.g., from notification tap)
        handleIntent(intent)

        themePreference = ThemePreference(applicationContext)

        setContent {
            val themeMode by themePreference.themeMode.collectAsState(initial = ThemeMode.SYSTEM)
            MeshClientTheme(themeMode = themeMode) {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    // Observe pending confirmations
                    val pendingConfirmations by confirmationManager.pendingConfirmations.collectAsState()
                    val nextConfirmation = pendingConfirmations.firstOrNull()

                    Box(modifier = Modifier.fillMaxSize()) {
                        MeshNavHost(
                            onTargetSelected = { nodeId ->
                                meshService?.setTarget(nodeId)
                            }
                        )

                        // Show confirmation dialog if there's a pending request
                        nextConfirmation?.let { request ->
                            ConfirmationDialog(
                                request = request,
                                onApprove = {
                                    handleConfirmationAction(request.messageId, request.fromNode, true, request.notificationId)
                                },
                                onReject = {
                                    handleConfirmationAction(request.messageId, request.fromNode, false, request.notificationId)
                                }
                            )
                        }
                    }
                }
            }
        }
    }

    private fun handleConfirmationAction(messageId: String, fromNode: String, approved: Boolean, notificationId: Int) {
        Log.i(tag, "handleConfirmationAction: messageId=$messageId, fromNode=$fromNode, approved=$approved")

        // Cancel the notification
        val notificationManager = getSystemService(NOTIFICATION_SERVICE) as android.app.NotificationManager
        notificationManager.cancel(notificationId)

        // Send the response via service
        meshService?.handleConfirmation(messageId, fromNode, approved)
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        handleIntent(intent)
    }

    override fun onResume() {
        super.onResume()
        meshService?.setAppInForeground(true)
    }

    override fun onPause() {
        super.onPause()
        meshService?.setAppInForeground(false)
    }

    override fun onDestroy() {
        if (serviceBound) {
            unbindService(serviceConnection)
            serviceBound = false
        }
        super.onDestroy()
    }

    private fun handleIntent(intent: Intent?) {
        intent ?: return

        Log.d(tag, "handleIntent: action=${intent.action}")

        // Handle tapping on a message notification
        intent.getStringExtra("target")?.let { target ->
            meshService?.setTarget(target)
        }

        // Handle confirmation actions from notification buttons (via ConfirmActionReceiver)
        when (intent.action) {
            ConfirmActionReceiver.ACTION_APPROVE, ConfirmActionReceiver.ACTION_REJECT -> {
                val messageId = intent.getStringExtra(ConfirmActionReceiver.EXTRA_MESSAGE_ID) ?: return
                val fromNode = intent.getStringExtra(ConfirmActionReceiver.EXTRA_FROM_NODE) ?: return
                val approved = intent.action == ConfirmActionReceiver.ACTION_APPROVE

                Log.i(tag, "handleIntent: confirmation action messageId=$messageId, fromNode=$fromNode, approved=$approved")

                // Get notification ID to look up the request
                val pendingRequest = confirmationManager.pendingConfirmations.value.find { it.messageId == messageId }
                val notificationId = pendingRequest?.notificationId ?: -1

                handleConfirmationAction(messageId, fromNode, approved, notificationId)
            }
        }

        // Handle confirmation request from tapping on notification body
        intent.getStringExtra("confirm_message_id")?.let { messageId ->
            val fromNode = intent.getStringExtra("confirm_from") ?: return@let
            // The dialog will be shown automatically since the request is in confirmationManager
            Log.d(tag, "handleIntent: confirmation tap for messageId=$messageId, fromNode=$fromNode")
        }
    }
}
