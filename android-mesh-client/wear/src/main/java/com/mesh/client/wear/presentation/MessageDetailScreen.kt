package com.mesh.client.wear.presentation

import android.app.RemoteInput
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.wear.compose.foundation.lazy.ScalingLazyColumn
import androidx.wear.compose.foundation.lazy.rememberScalingLazyListState
import androidx.wear.compose.material.Chip
import androidx.wear.compose.material.ChipDefaults
import androidx.wear.compose.material.CircularProgressIndicator
import androidx.wear.compose.material.MaterialTheme
import androidx.wear.compose.material.Text
import androidx.wear.input.RemoteInputIntentHelper
import com.mesh.client.wear.R
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

private const val REPLY_MESSAGE_KEY = "reply_message"

@Composable
fun MessageDetailScreen(
    messageId: String,
    onReplyClick: () -> Unit,
    onReplySent: () -> Unit,
    onBack: () -> Unit,
    viewModel: WearViewModel = hiltViewModel()
) {
    val messages by viewModel.messages.collectAsState()
    val replySendingState by viewModel.replySendingState.collectAsState()
    val message = messages.find { it.id == messageId }

    // Mark as read when viewing
    LaunchedEffect(messageId) {
        viewModel.markAsRead(messageId)
    }

    // Handle reply sent
    LaunchedEffect(replySendingState) {
        if (replySendingState is ReplySendingState.Sent) {
            viewModel.resetReplyState()
            onReplySent()
        }
    }

    // Launcher for keyboard input
    val inputLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { result ->
        val data = result.data ?: return@rememberLauncherForActivityResult
        val results = RemoteInput.getResultsFromIntent(data)
        val replyText = results?.getCharSequence(REPLY_MESSAGE_KEY)?.toString()
        if (!replyText.isNullOrBlank() && message != null) {
            viewModel.sendCustomReply(message, replyText)
        }
    }

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(MaterialTheme.colors.background)
    ) {
        when {
            message == null -> {
                Text(
                    text = "Message not found",
                    modifier = Modifier.align(Alignment.Center),
                    style = MaterialTheme.typography.body1,
                    color = MaterialTheme.colors.onBackground.copy(alpha = 0.7f)
                )
            }
            replySendingState is ReplySendingState.Sending -> {
                Box(
                    modifier = Modifier.fillMaxSize(),
                    contentAlignment = Alignment.Center
                ) {
                    CircularProgressIndicator()
                }
            }
            else -> {
                MessageDetailContent(
                    senderName = message.senderDisplayName,
                    channelName = message.channelName,
                    content = message.content,
                    timestamp = message.timestamp,
                    onReplyClick = {
                        val remoteInput = RemoteInput.Builder(REPLY_MESSAGE_KEY)
                            .setLabel("Reply to ${message.senderDisplayName}")
                            .setAllowFreeFormInput(true)
                            .build()
                        val intent = RemoteInputIntentHelper.createActionRemoteInputIntent()
                        RemoteInputIntentHelper.putRemoteInputsExtra(intent, listOf(remoteInput))
                        inputLauncher.launch(intent)
                    }
                )
            }
        }
    }
}

@Composable
private fun MessageDetailContent(
    senderName: String,
    channelName: String?,
    content: String,
    timestamp: Long,
    onReplyClick: () -> Unit
) {
    val listState = rememberScalingLazyListState()

    ScalingLazyColumn(
        modifier = Modifier.fillMaxSize(),
        state = listState,
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        // Header with sender info
        item {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 16.dp),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                Text(
                    text = buildString {
                        append(senderName)
                        channelName?.let { append(" → #$it") }
                    },
                    style = MaterialTheme.typography.title3,
                    color = MaterialTheme.colors.primary,
                    textAlign = TextAlign.Center
                )

                Text(
                    text = formatFullTimestamp(timestamp),
                    style = MaterialTheme.typography.caption2,
                    color = MaterialTheme.colors.onBackground.copy(alpha = 0.6f)
                )
            }
        }

        // Message content
        item {
            Text(
                text = content,
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 12.dp),
                style = MaterialTheme.typography.body1,
                color = MaterialTheme.colors.onBackground,
                textAlign = TextAlign.Start
            )
        }

        // Reply button
        item {
            Chip(
                onClick = onReplyClick,
                label = {
                    Text(stringResource(R.string.reply))
                },
                modifier = Modifier.fillMaxWidth(0.8f),
                colors = ChipDefaults.primaryChipColors()
            )
        }
    }
}

private fun formatFullTimestamp(timestamp: Long): String {
    val now = System.currentTimeMillis()
    val isToday = SimpleDateFormat("yyyyMMdd", Locale.getDefault())
        .format(Date(timestamp)) == SimpleDateFormat("yyyyMMdd", Locale.getDefault())
        .format(Date(now))

    return if (isToday) {
        SimpleDateFormat("h:mm a", Locale.getDefault()).format(Date(timestamp))
    } else {
        SimpleDateFormat("MMM d, h:mm a", Locale.getDefault()).format(Date(timestamp))
    }
}
