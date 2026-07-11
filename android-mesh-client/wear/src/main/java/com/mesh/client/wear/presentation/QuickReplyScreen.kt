package com.mesh.client.wear.presentation

import android.app.RemoteInput
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
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
import androidx.wear.compose.foundation.lazy.items
import androidx.wear.compose.foundation.lazy.rememberScalingLazyListState
import androidx.wear.compose.material.Chip
import androidx.wear.compose.material.ChipDefaults
import androidx.wear.compose.material.CircularProgressIndicator
import androidx.wear.compose.material.MaterialTheme
import androidx.wear.compose.material.Text
import androidx.wear.input.RemoteInputIntentHelper
import com.mesh.client.wear.R
import com.mesh.client.wear.data.QuickReply

@Composable
fun QuickReplyScreen(
    messageId: String,
    onReplySent: () -> Unit,
    onBack: () -> Unit,
    viewModel: WearViewModel = hiltViewModel()
) {
    val messages by viewModel.messages.collectAsState()
    val replySendingState by viewModel.replySendingState.collectAsState()
    val message = messages.find { it.id == messageId }

    // Handle reply sent
    LaunchedEffect(replySendingState) {
        if (replySendingState is ReplySendingState.Sent) {
            viewModel.resetReplyState()
            onReplySent()
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
                SendingState()
            }
            replySendingState is ReplySendingState.Error -> {
                ErrorState(
                    message = (replySendingState as ReplySendingState.Error).message,
                    onRetry = { viewModel.resetReplyState() }
                )
            }
            else -> {
                QuickReplyContent(
                    replyingTo = message.senderDisplayName,
                    onReplySelected = { reply ->
                        viewModel.sendQuickReply(message, reply)
                    },
                    onCustomMessage = { customText ->
                        viewModel.sendCustomReply(message, customText)
                    }
                )
            }
        }
    }
}

@Composable
private fun SendingState() {
    Box(
        modifier = Modifier.fillMaxSize(),
        contentAlignment = Alignment.Center
    ) {
        CircularProgressIndicator()
    }
}

@Composable
private fun ErrorState(
    message: String,
    onRetry: () -> Unit
) {
    Box(
        modifier = Modifier.fillMaxSize(),
        contentAlignment = Alignment.Center
    ) {
        Text(
            text = message,
            style = MaterialTheme.typography.body1,
            color = MaterialTheme.colors.error,
            textAlign = TextAlign.Center,
            modifier = Modifier.padding(16.dp)
        )
    }
}

private const val CUSTOM_MESSAGE_KEY = "custom_message"

@Composable
private fun QuickReplyContent(
    replyingTo: String,
    onReplySelected: (QuickReply) -> Unit,
    onCustomMessage: (String) -> Unit
) {
    val listState = rememberScalingLazyListState()
    val quickReplies = QuickReply.entries

    // Launcher for system text input
    val inputLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { result ->
        val results = RemoteInput.getResultsFromIntent(result.data)
        val customText = results?.getCharSequence(CUSTOM_MESSAGE_KEY)?.toString()
        if (!customText.isNullOrBlank()) {
            onCustomMessage(customText)
        }
    }

    ScalingLazyColumn(
        modifier = Modifier.fillMaxSize(),
        state = listState,
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(4.dp)
    ) {
        // Header
        item {
            Text(
                text = stringResource(R.string.quick_replies),
                style = MaterialTheme.typography.title3,
                color = MaterialTheme.colors.primary,
                modifier = Modifier.padding(bottom = 8.dp)
            )
        }

        item {
            Text(
                text = "Reply to $replyingTo",
                style = MaterialTheme.typography.caption1,
                color = MaterialTheme.colors.onBackground.copy(alpha = 0.6f),
                modifier = Modifier.padding(bottom = 4.dp)
            )
        }

        // Custom message input chip (at top)
        item {
            Chip(
                onClick = {
                    val remoteInput = RemoteInput.Builder(CUSTOM_MESSAGE_KEY)
                        .setLabel("Type a message")
                        .setAllowFreeFormInput(true)
                        .build()
                    val intent = RemoteInputIntentHelper.createActionRemoteInputIntent()
                    RemoteInputIntentHelper.putRemoteInputsExtra(intent, listOf(remoteInput))
                    inputLauncher.launch(intent)
                },
                label = {
                    Text(
                        text = "Custom message...",
                        textAlign = TextAlign.Center,
                        modifier = Modifier.fillMaxWidth()
                    )
                },
                modifier = Modifier.fillMaxWidth(0.85f),
                colors = ChipDefaults.primaryChipColors()
            )
        }

        // Quick reply buttons
        items(quickReplies) { reply ->
            Chip(
                onClick = { onReplySelected(reply) },
                label = {
                    Text(
                        text = reply.text,
                        textAlign = TextAlign.Center,
                        modifier = Modifier.fillMaxWidth()
                    )
                },
                modifier = Modifier.fillMaxWidth(0.85f),
                colors = ChipDefaults.secondaryChipColors()
            )
        }
    }
}
