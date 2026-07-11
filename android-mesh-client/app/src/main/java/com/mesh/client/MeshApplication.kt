package com.mesh.client

import android.app.Application
import com.mesh.client.data.remote.FcmTokenManager
import com.mesh.client.util.NotificationHelper
import dagger.hilt.android.HiltAndroidApp
import javax.inject.Inject

@HiltAndroidApp
class MeshApplication : Application() {

    @Inject lateinit var notificationHelper: NotificationHelper
    @Inject lateinit var fcmTokenManager: FcmTokenManager

    override fun onCreate() {
        super.onCreate()

        // Create notification channels
        notificationHelper.createChannels()

        // Initialize FCM token manager
        fcmTokenManager.initialize()
    }
}
