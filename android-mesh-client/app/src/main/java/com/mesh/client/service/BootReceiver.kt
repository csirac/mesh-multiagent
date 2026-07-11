package com.mesh.client.service

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.util.Log
import androidx.core.content.ContextCompat

/**
 * Broadcast receiver that starts MeshService on device boot.
 *
 * This allows the mesh connection to be maintained automatically
 * after the device restarts.
 */
class BootReceiver : BroadcastReceiver() {

    override fun onReceive(context: Context?, intent: Intent?) {
        if (intent?.action == Intent.ACTION_BOOT_COMPLETED) {
            Log.i("BootReceiver", "Device boot completed, starting MeshService")

            context?.let { ctx ->
                // TODO: Check if auto-start is enabled in preferences
                // For now, we don't auto-start - user must open the app first

                // val serviceIntent = Intent(ctx, MeshService::class.java)
                // ContextCompat.startForegroundService(ctx, serviceIntent)
            }
        }
    }
}
