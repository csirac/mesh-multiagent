package com.mesh.client.wear.presentation

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.runtime.Composable
import androidx.wear.compose.navigation.SwipeDismissableNavHost
import androidx.wear.compose.navigation.composable
import androidx.wear.compose.navigation.rememberSwipeDismissableNavController
import dagger.hilt.android.AndroidEntryPoint

@AndroidEntryPoint
class MainActivity : ComponentActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        setContent {
            MeshWearNavHost()
        }
    }
}

@Composable
fun MeshWearNavHost() {
    val navController = rememberSwipeDismissableNavController()

    SwipeDismissableNavHost(
        navController = navController,
        startDestination = "conversations"
    ) {
        // Conversation list (main screen)
        composable("conversations") {
            ConversationListScreen(
                onConversationClick = { conversation ->
                    navController.navigate("conversation/${java.net.URLEncoder.encode(conversation.id, "UTF-8")}")
                }
            )
        }

        // Chat view within a conversation
        composable("conversation/{conversationId}") { backStackEntry ->
            val conversationId = backStackEntry.arguments?.getString("conversationId")?.let {
                java.net.URLDecoder.decode(it, "UTF-8")
            } ?: return@composable

            ConversationDetailScreen(
                conversationId = conversationId,
                onMessageSent = {
                    // Stay in conversation after sending
                }
            )
        }
    }
}
