package com.mesh.client.wear.presentation

import android.app.RemoteInput
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.wear.compose.foundation.lazy.ScalingLazyColumn
import androidx.wear.compose.foundation.lazy.items
import androidx.wear.compose.foundation.lazy.rememberScalingLazyListState
import androidx.wear.compose.material.Button
import androidx.wear.compose.material.ButtonDefaults
import androidx.wear.compose.material.CircularProgressIndicator
import androidx.wear.compose.material.Icon
import androidx.wear.compose.material.MaterialTheme
import androidx.wear.compose.material.Text
import androidx.wear.input.RemoteInputIntentHelper
import com.mesh.client.wear.data.WearMessage
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

private const val COMPOSE_MESSAGE_KEY = "compose_message"

@Composable
fun ConversationDetailScreen(
    conversationId: String,
    onMessageSent: () -> Unit = {},
    viewModel: WearViewModel = hiltViewModel()
) {
    // Use ViewModel's method which correctly identifies messages for this conversation
    // by properly handling the user's own node ID
    val allMessages by viewModel.messages.collectAsState()
    val messages = remember(allMessages, conversationId) {
        viewModel.getMessagesForConversation(conversationId)
    }
    val conversations by viewModel.conversations.collectAsState()
    val conversation = conversations.find { it.id == conversationId }
    val replySendingState by viewModel.replySendingState.collectAsState()

    // Refresh when screen appears
    LaunchedEffect(Unit) {
        viewModel.refresh()
    }

    // Mark all messages as read when viewing conversation
    LaunchedEffect(conversationId, messages) {
        messages.filter { !it.isRead }.forEach { msg ->
            viewModel.markAsRead(msg.id)
        }
    }

    // Handle message sent
    LaunchedEffect(replySendingState) {
        if (replySendingState is ReplySendingState.Sent) {
            viewModel.resetReplyState()
            onMessageSent()
        }
    }

    // Launcher for keyboard input
    val inputLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { result ->
        val data = result.data ?: return@rememberLauncherForActivityResult
        val results = RemoteInput.getResultsFromIntent(data)
        val messageText = results?.getCharSequence(COMPOSE_MESSAGE_KEY)?.toString()
        if (!messageText.isNullOrBlank()) {
            viewModel.sendMessageToConversation(conversationId, messageText)
        }
    }

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(MaterialTheme.colors.background)
    ) {
        when {
            replySendingState is ReplySendingState.Sending -> {
                Box(
                    modifier = Modifier.fillMaxSize(),
                    contentAlignment = Alignment.Center
                ) {
                    CircularProgressIndicator()
                }
            }
            messages.isEmpty() -> {
                Column(
                    modifier = Modifier.fillMaxSize(),
                    horizontalAlignment = Alignment.CenterHorizontally,
                    verticalArrangement = Arrangement.Center
                ) {
                    Text(
                        text = "No messages",
                        style = MaterialTheme.typography.body1,
                        textAlign = TextAlign.Center,
                        color = MaterialTheme.colors.onBackground.copy(alpha = 0.7f)
                    )
                    Spacer(modifier = Modifier.height(12.dp))
                    ComposeButton(
                        conversationName = conversation?.displayName ?: conversationId,
                        onClick = {
                            launchCompose(inputLauncher, conversation?.displayName ?: conversationId)
                        }
                    )
                }
            }
            else -> {
                val listState = rememberScalingLazyListState()

                // Scroll to bottom (most recent message) when messages load
                LaunchedEffect(messages.size) {
                    if (messages.isNotEmpty()) {
                        // +2 accounts for header item and compose button
                        listState.scrollToItem(messages.size + 1)
                    }
                }

                ScalingLazyColumn(
                    modifier = Modifier.fillMaxSize(),
                    state = listState,
                    horizontalAlignment = Alignment.CenterHorizontally,
                    verticalArrangement = Arrangement.spacedBy(2.dp)
                ) {
                    // Conversation header
                    item {
                        Text(
                            text = conversation?.displayName ?: conversationId.substringAfterLast(":"),
                            style = MaterialTheme.typography.title3,
                            color = MaterialTheme.colors.primary,
                            modifier = Modifier.padding(bottom = 4.dp)
                        )
                    }

                    // Messages (oldest first for chat view, but we show newest at bottom)
                    items(messages.reversed(), key = { it.id }) { message ->
                        // Message is outgoing if fromNode does NOT match the conversationId
                        // (conversationId is the partner, so if fromNode != partner, we sent it)
                        val isOutgoing = message.fromNode != conversationId
                        ChatBubble(message = message, isOutgoing = isOutgoing)
                    }

                    // Compose button at bottom
                    item {
                        Spacer(modifier = Modifier.height(8.dp))
                        ComposeButton(
                            conversationName = conversation?.displayName ?: conversationId,
                            onClick = {
                                launchCompose(inputLauncher, conversation?.displayName ?: conversationId)
                            }
                        )
                    }
                }
            }
        }
    }
}

private fun launchCompose(
    launcher: androidx.activity.result.ActivityResultLauncher<android.content.Intent>,
    conversationName: String
) {
    val remoteInput = RemoteInput.Builder(COMPOSE_MESSAGE_KEY)
        .setLabel("Message $conversationName")
        .setAllowFreeFormInput(true)
        .build()
    val intent = RemoteInputIntentHelper.createActionRemoteInputIntent()
    RemoteInputIntentHelper.putRemoteInputsExtra(intent, listOf(remoteInput))
    launcher.launch(intent)
}

@Composable
private fun ComposeButton(
    conversationName: String,
    onClick: () -> Unit
) {
    Button(
        onClick = onClick,
        modifier = Modifier.size(ButtonDefaults.DefaultButtonSize),
        colors = ButtonDefaults.primaryButtonColors()
    ) {
        Text(
            text = "+",
            style = MaterialTheme.typography.title1
        )
    }
}

@Composable
private fun ChatBubble(message: WearMessage, isOutgoing: Boolean = false) {
    Column(
        modifier = Modifier
            .fillMaxWidth(0.95f)
            .padding(horizontal = 4.dp, vertical = 2.dp),
        horizontalAlignment = if (isOutgoing) Alignment.End else Alignment.Start
    ) {
        // Sender name (only show for incoming messages)
        if (!isOutgoing) {
            Text(
                text = message.senderDisplayName,
                style = MaterialTheme.typography.caption2,
                color = MaterialTheme.colors.primary,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis
            )
        }

        // Message bubble
        Box(
            modifier = Modifier
                .fillMaxWidth(0.85f)
                .clip(RoundedCornerShape(12.dp))
                .background(
                    if (isOutgoing) MaterialTheme.colors.primary.copy(alpha = 0.3f)
                    else MaterialTheme.colors.surface
                )
                .padding(8.dp)
        ) {
            Column {
                MarkdownText(
                    text = message.content,
                    color = MaterialTheme.colors.onSurface
                )

                // Timestamp (small, right-aligned)
                Text(
                    text = formatTime(message.timestamp),
                    style = MaterialTheme.typography.caption2,
                    color = MaterialTheme.colors.onSurface.copy(alpha = 0.5f),
                    modifier = Modifier.align(Alignment.End)
                )
            }
        }
    }
}

private fun formatTime(timestamp: Long): String {
    val now = System.currentTimeMillis()
    val diff = now - timestamp

    return when {
        diff < 60_000 -> "now"
        diff < 3600_000 -> "${diff / 60_000}m"
        diff < 86400_000 -> SimpleDateFormat("h:mm a", Locale.getDefault()).format(Date(timestamp))
        else -> SimpleDateFormat("MMM d", Locale.getDefault()).format(Date(timestamp))
    }
}
