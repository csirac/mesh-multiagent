package com.mesh.client.ui.navigation

import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.Chat
import androidx.compose.material.icons.automirrored.filled.Message
import androidx.compose.material.icons.filled.People
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material.icons.filled.Tag
import androidx.compose.material3.Icon
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.res.stringResource
import androidx.navigation.NavController
import androidx.navigation.NavDestination.Companion.hierarchy
import androidx.navigation.NavGraph.Companion.findStartDestination
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.currentBackStackEntryAsState
import androidx.navigation.compose.rememberNavController
import com.mesh.client.R
import com.mesh.client.ui.chat.ChatScreen
import com.mesh.client.ui.channels.ChannelListScreen
import com.mesh.client.ui.channels.ChannelMembersScreen
import com.mesh.client.ui.conversations.ConversationListScreen
import com.mesh.client.ui.roster.RosterScreen
import com.mesh.client.ui.settings.SettingsScreen
import com.mesh.client.ui.usage.UsageScreen
import com.mesh.client.ui.workspace.WorkspaceScreen

sealed class Screen(val route: String, val labelRes: Int, val icon: ImageVector) {
    object Conversations : Screen("conversations", R.string.nav_conversations, Icons.AutoMirrored.Filled.Message)
    object Chat : Screen("chat/{nodeId}", R.string.nav_chat, Icons.AutoMirrored.Filled.Chat) {
        fun createRoute(nodeId: String) = "chat/${java.net.URLEncoder.encode(nodeId, "UTF-8")}"
    }
    object Channels : Screen("channels", R.string.nav_channels, Icons.Default.Tag)
    object ChannelMembers : Screen("channel_members/{channelName}", R.string.channels_view_members, Icons.Default.People) {
        fun createRoute(channelName: String) = "channel_members/${java.net.URLEncoder.encode(channelName, "UTF-8")}"
    }
    object Roster : Screen("roster", R.string.nav_roster, Icons.Default.People)
    object Settings : Screen("settings", R.string.nav_settings, Icons.Default.Settings)
    object Usage : Screen("usage", R.string.usage_title, Icons.Default.Settings)
    object Workspace : Screen("workspace", R.string.workspace_title, Icons.Default.Settings)
}

// Only show these in the bottom nav (Chat is a detail screen, not a tab)
val bottomNavItems = listOf(Screen.Conversations, Screen.Channels, Screen.Roster, Screen.Settings)

@Composable
fun MeshNavHost(
    onTargetSelected: (String) -> Unit,
    modifier: Modifier = Modifier
) {
    val navController = rememberNavController()

    Scaffold(
        bottomBar = { MeshBottomBar(navController = navController) },
        modifier = modifier
    ) { innerPadding ->
        NavHost(
            navController = navController,
            startDestination = Screen.Conversations.route,
            modifier = Modifier.padding(innerPadding)
        ) {
            composable(Screen.Conversations.route) {
                ConversationListScreen(
                    onConversationSelected = { nodeId ->
                        onTargetSelected(nodeId)
                        navController.navigate(Screen.Chat.createRoute(nodeId))
                    }
                )
            }
            composable(Screen.Chat.route) { backStackEntry ->
                val nodeId = backStackEntry.arguments?.getString("nodeId")?.let {
                    java.net.URLDecoder.decode(it, "UTF-8")
                }
                // Set the target when navigating to chat
                nodeId?.let { onTargetSelected(it) }
                ChatScreen()
            }
            composable(Screen.Channels.route) {
                ChannelListScreen(
                    onChannelSelected = { channelAddress ->
                        onTargetSelected(channelAddress)
                        navController.navigate(Screen.Chat.createRoute(channelAddress))
                    },
                    onViewMembers = { channelName ->
                        navController.navigate(Screen.ChannelMembers.createRoute(channelName))
                    }
                )
            }
            composable(Screen.ChannelMembers.route) { backStackEntry ->
                val channelName = backStackEntry.arguments?.getString("channelName")?.let {
                    java.net.URLDecoder.decode(it, "UTF-8")
                } ?: ""
                ChannelMembersScreen(
                    channelName = channelName,
                    onBack = { navController.popBackStack() }
                )
            }
            composable(Screen.Roster.route) {
                RosterScreen(
                    onAgentSelected = { nodeId ->
                        onTargetSelected(nodeId)
                        navController.navigate(Screen.Chat.createRoute(nodeId))
                    }
                )
            }
            composable(Screen.Settings.route) {
                SettingsScreen(
                    onNavigateToUsage = {
                        navController.navigate(Screen.Usage.route)
                    },
                    onNavigateToWorkspace = {
                        navController.navigate(Screen.Workspace.route)
                    }
                )
            }
            composable(Screen.Usage.route) {
                UsageScreen(
                    onNavigateBack = { navController.popBackStack() }
                )
            }
            composable(Screen.Workspace.route) {
                WorkspaceScreen(
                    onBack = { navController.popBackStack() }
                )
            }
        }
    }
}

@Composable
fun MeshBottomBar(navController: NavController) {
    val navBackStackEntry by navController.currentBackStackEntryAsState()
    val currentDestination = navBackStackEntry?.destination
    val currentRoute = currentDestination?.route

    // Check if we're on the Chat detail screen (a child route, not a tab destination)
    val isOnChatScreen = currentRoute?.startsWith("chat/") == true

    NavigationBar {
        bottomNavItems.forEach { screen ->
            NavigationBarItem(
                icon = { Icon(screen.icon, contentDescription = null) },
                label = { Text(stringResource(screen.labelRes)) },
                selected = currentDestination?.hierarchy?.any { it.route == screen.route } == true,
                onClick = {
                    if (isOnChatScreen) {
                        // When on Chat screen, pop back to start and navigate fresh
                        // This avoids restoreState issues where Chat gets restored
                        navController.navigate(screen.route) {
                            popUpTo(navController.graph.findStartDestination().id) {
                                inclusive = false
                                saveState = false
                            }
                            launchSingleTop = true
                            restoreState = false
                        }
                    } else {
                        // Standard navigation between tab destinations
                        // Use simpler navigation without state restoration to prevent glitches
                        navController.navigate(screen.route) {
                            popUpTo(navController.graph.findStartDestination().id) {
                                saveState = false
                            }
                            launchSingleTop = true
                            restoreState = false
                        }
                    }
                }
            )
        }
    }
}
