package com.mesh.client.ui.roster

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.DismissDirection
import androidx.compose.material3.DismissValue
import androidx.compose.material3.SwipeToDismiss
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.rememberDismissState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.rememberCoroutineScope
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleEventObserver
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import com.mesh.client.R
import com.mesh.client.data.local.db.entities.RosterEntry
import com.mesh.client.ui.theme.AgentColor
import com.mesh.client.ui.theme.OfflineGray
import com.mesh.client.ui.theme.OnlineGreen
import com.mesh.client.ui.theme.UserColor
import com.mesh.client.ui.theme.WarningOrange
import kotlinx.coroutines.launch

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun RosterScreen(
    onAgentSelected: (String) -> Unit,
    viewModel: RosterViewModel = hiltViewModel()
) {
    val roster by viewModel.roster.collectAsState(initial = emptyList())

    // Refresh roster status every time this screen becomes visible (ON_RESUME)
    val lifecycleOwner = LocalLifecycleOwner.current
    DisposableEffect(lifecycleOwner) {
        val observer = LifecycleEventObserver { _, event ->
            if (event == Lifecycle.Event.ON_RESUME) {
                viewModel.refreshStatus()
            }
        }
        lifecycleOwner.lifecycle.addObserver(observer)
        onDispose {
            lifecycleOwner.lifecycle.removeObserver(observer)
        }
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text(stringResource(R.string.roster_title)) }
            )
        }
    ) { padding ->
        if (roster.isEmpty()) {
            Box(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(padding),
                contentAlignment = Alignment.Center
            ) {
                Text(
                    text = stringResource(R.string.roster_empty),
                    style = MaterialTheme.typography.bodyLarge,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
        } else {
            val scope = rememberCoroutineScope()
            LazyColumn(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(padding),
                contentPadding = PaddingValues(16.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                items(roster, key = { it.nodeId }) { entry ->
                    SwipeableRosterItem(
                        entry = entry,
                        onClick = { onAgentSelected(entry.nodeId) },
                        onDelete = {
                            scope.launch {
                                viewModel.deleteEntry(entry.nodeId)
                            }
                        }
                    )
                }
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SwipeableRosterItem(
    entry: RosterEntry,
    onClick: () -> Unit,
    onDelete: () -> Unit,
    modifier: Modifier = Modifier
) {
    val dismissState = rememberDismissState(
        confirmValueChange = { value ->
            if (value == DismissValue.DismissedToStart) {
                onDelete()
                true
            } else {
                false
            }
        }
    )

    SwipeToDismiss(
        state = dismissState,
        modifier = modifier,
        directions = setOf(DismissDirection.EndToStart),
        background = {
            Box(
                modifier = Modifier
                    .fillMaxSize()
                    .background(
                        color = MaterialTheme.colorScheme.error,
                        shape = RoundedCornerShape(12.dp)
                    )
                    .padding(horizontal = 20.dp),
                contentAlignment = Alignment.CenterEnd
            ) {
                Text(
                    text = "Delete",
                    color = MaterialTheme.colorScheme.onError,
                    style = MaterialTheme.typography.titleMedium
                )
            }
        },
        dismissContent = {
            RosterItem(
                entry = entry,
                onClick = onClick
            )
        }
    )
}

@Composable
fun RosterItem(
    entry: RosterEntry,
    onClick: () -> Unit,
    modifier: Modifier = Modifier
) {
    val nodeColor = if (entry.isAgent) AgentColor else UserColor
    // Brighter, more saturated green for online; darker gray for offline
    val statusColor = if (entry.isOnline) OnlineGreen else OfflineGray

    Card(
        modifier = modifier
            .fillMaxWidth()
            .clickable(onClick = onClick),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surfaceVariant
        )
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            // Status indicator - larger with border/glow for online
            Box(
                modifier = Modifier
                    .size(14.dp)
                    .then(
                        if (entry.isOnline) {
                            Modifier.border(2.dp, OnlineGreen.copy(alpha = 0.5f), CircleShape)
                        } else {
                            Modifier
                        }
                    ),
                contentAlignment = Alignment.Center
            ) {
                Surface(
                    modifier = Modifier.size(if (entry.isOnline) 10.dp else 12.dp),
                    shape = CircleShape,
                    color = statusColor
                ) {}
            }

            Spacer(modifier = Modifier.width(12.dp))

            // Info
            Column(modifier = Modifier.weight(1f)) {
                // First row: nickname and status
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text(
                        text = entry.nickname.replaceFirstChar { it.uppercase() },
                        style = MaterialTheme.typography.titleMedium,
                        color = nodeColor,
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis,
                        modifier = Modifier.weight(1f, fill = false)
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    // Online/Offline label for clarity
                    Text(
                        text = if (entry.isOnline) "online" else "offline",
                        style = MaterialTheme.typography.labelSmall,
                        color = if (entry.isOnline) OnlineGreen else MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.6f)
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Text(
                        text = if (entry.isUser) {
                            stringResource(R.string.roster_user)
                        } else {
                            entry.nodeType
                        },
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
                // Second row: LLM backend and hostname (if applicable)
                if ((entry.isAgent && entry.llmBackend.isNotEmpty()) || entry.hostname.isNotEmpty()) {
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        modifier = Modifier.padding(top = 2.dp)
                    ) {
                        // Show LLM backend info for agents
                        if (entry.isAgent && entry.llmBackend.isNotEmpty()) {
                            val backendDisplay = if (entry.llmModel.isNotEmpty() && entry.llmModel != entry.llmBackend) {
                                "[${entry.llmBackend}/${entry.llmModel}]"
                            } else {
                                "[${entry.llmBackend}]"
                            }
                            Text(
                                text = backendDisplay,
                                style = MaterialTheme.typography.labelSmall,
                                color = MaterialTheme.colorScheme.primary,
                                maxLines = 1,
                                overflow = TextOverflow.Ellipsis,
                                modifier = Modifier.weight(1f, fill = false)
                            )
                        }
                        // Show hostname if available
                        if (entry.hostname.isNotEmpty()) {
                            if (entry.isAgent && entry.llmBackend.isNotEmpty()) {
                                Spacer(modifier = Modifier.width(8.dp))
                            }
                            Text(
                                text = "@${entry.hostname}",
                                style = MaterialTheme.typography.labelSmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.7f),
                                maxLines = 1,
                                overflow = TextOverflow.Ellipsis
                            )
                        }
                    }
                }

                // Third row: Heartbeat-lite status (state, context tokens, history %)
                if (entry.isAgent && entry.agentState.isNotEmpty()) {
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        modifier = Modifier.padding(top = 2.dp)
                    ) {
                        val statusParts = mutableListOf<String>()
                        val stateDisplay = entry.agentState.uppercase()
                        if (entry.agentState == "busy" && entry.workerElapsedSecs != null) {
                            statusParts.add("$stateDisplay (${entry.workerElapsedSecs.toInt()}s)")
                        } else {
                            statusParts.add(stateDisplay)
                        }
                        if (entry.contextTokens > 0) {
                            statusParts.add("${entry.contextTokens / 1000}k ctx")
                        }
                        entry.historyUtilizationPct?.let {
                            statusParts.add("${it.toInt()}% hist")
                        }
                        if (entry.activeMap.isNotEmpty()) {
                            statusParts.add("map:${entry.activeMap}")
                        }
                        val stateColor = when (entry.agentState) {
                            "busy" -> WarningOrange
                            "planning" -> MaterialTheme.colorScheme.primary
                            else -> OnlineGreen
                        }
                        Text(
                            text = statusParts.joinToString(" · "),
                            style = MaterialTheme.typography.labelSmall,
                            color = stateColor,
                            maxLines = 1,
                            overflow = TextOverflow.Ellipsis
                        )
                    }
                }

                if (entry.description.isNotEmpty()) {
                    Text(
                        text = entry.description,
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis
                    )
                }
            }
        }
    }
}
