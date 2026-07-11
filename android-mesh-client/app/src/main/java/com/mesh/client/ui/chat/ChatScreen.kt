package com.mesh.client.ui.chat

import android.graphics.BitmapFactory
import android.net.Uri
import android.util.Base64
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.ime
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.Send
import androidx.compose.material.icons.filled.AttachFile
import androidx.compose.material.icons.filled.Check
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.Image
import androidx.compose.material.icons.filled.Info
import androidx.compose.material.icons.filled.Schedule
import androidx.compose.material.icons.filled.Warning
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Divider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalDensity
import coil.compose.AsyncImage
import coil.request.ImageRequest
import kotlinx.coroutines.launch
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import java.time.Instant
import java.time.ZoneId
import java.time.format.DateTimeFormatter
import com.mesh.client.R
import com.mesh.client.data.local.db.entities.MessageEntity
import com.mesh.client.data.local.db.entities.MessageStatus
import com.mesh.client.data.remote.DiagnosticReport
import com.mesh.client.data.remote.StatusContextEntry
import com.mesh.client.data.remote.StatusResponse
import com.mesh.client.data.remote.StatusSummary
import com.mesh.client.data.remote.protocol.getDisplayName
import com.mesh.client.ui.components.MarkdownText
import com.mesh.client.ui.theme.AgentColor
import com.mesh.client.ui.theme.UserColor
import com.mesh.client.util.ImageUtils
import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.widget.Toast
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ChatScreen(
    viewModel: ChatViewModel = hiltViewModel()
) {
    val uiState by viewModel.uiState.collectAsState()
    val messages by viewModel.messages.collectAsState(initial = emptyList())
    val statusResponse by viewModel.statusResponse.collectAsState()
    val statusLoading by viewModel.statusLoading.collectAsState()
    val draftText by viewModel.draftText.collectAsState()

    // Key the list state on the current target so scroll position resets on conversation switch
    val conversationKey = uiState.currentTarget ?: ""
    val listState = rememberLazyListState()
    val coroutineScope = rememberCoroutineScope()

    // Image picker launcher
    val imagePickerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        viewModel.setSelectedImage(uri)
    }

    // Track IME (keyboard) visibility
    val imeVisible = WindowInsets.ime.getBottom(LocalDensity.current) > 0

    // Track if we've done initial scroll to bottom - keyed on conversation
    var hasScrolledToBottom by remember(conversationKey) { mutableStateOf(false) }

    // Check if user is near bottom (within 3 items) - only auto-scroll if true
    fun isNearBottom(): Boolean {
        val layoutInfo = listState.layoutInfo
        val visibleItems = layoutInfo.visibleItemsInfo
        if (visibleItems.isEmpty()) return true
        val lastVisibleIndex = visibleItems.last().index
        return lastVisibleIndex >= layoutInfo.totalItemsCount - 3
    }

    // Mark conversation as read when entering chat
    LaunchedEffect(Unit) {
        viewModel.markConversationAsRead()
    }

    // Mark as read again if new messages arrive while viewing
    LaunchedEffect(messages.size) {
        if (messages.isNotEmpty()) {
            viewModel.markConversationAsRead()
        }
    }

    // Scroll to bottom: instantly on first load, only auto-scroll for new messages if near bottom
    LaunchedEffect(messages.size) {
        if (messages.isNotEmpty()) {
            if (!hasScrolledToBottom) {
                // First load - scroll instantly without animation
                listState.scrollToItem(messages.size - 1)
                hasScrolledToBottom = true
            } else if (isNearBottom()) {
                // New message arrived and user is near bottom - animate scroll
                listState.animateScrollToItem(messages.size - 1)
            }
        }
    }

    // Scroll to bottom when keyboard opens (only if near bottom)
    LaunchedEffect(imeVisible) {
        if (imeVisible && messages.isNotEmpty() && isNearBottom()) {
            listState.animateScrollToItem(messages.size - 1)
        }
    }

    // Show status dialog if we have a response or loading
    if (statusResponse != null || statusLoading) {
        StatusDialog(
            status = statusResponse,
            isLoading = statusLoading,
            onDismiss = { viewModel.dismissStatus() }
        )
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    val target = uiState.currentTarget
                    if (target != null) {
                        Text(stringResource(R.string.chat_target_label, getDisplayName(target)))
                    } else {
                        Text(stringResource(R.string.chat_no_target))
                    }
                },
                actions = {
                    // Status button - only show when target is set
                    if (uiState.currentTarget != null) {
                        IconButton(
                            onClick = { viewModel.requestStatus() },
                            enabled = uiState.isConnected && !statusLoading
                        ) {
                            Icon(
                                imageVector = Icons.Default.Info,
                                contentDescription = stringResource(R.string.status_button)
                            )
                        }
                    }
                }
            )
        }
    ) { padding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
                .imePadding()
        ) {
            // Message list
            LazyColumn(
                state = listState,
                modifier = Modifier
                    .weight(1f)
                    .fillMaxWidth(),
                contentPadding = PaddingValues(horizontal = 16.dp, vertical = 8.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                items(messages, key = { it.id }) { message ->
                    val context = LocalContext.current
                    MessageBubble(
                        message = message,
                        onCopy = { text ->
                            val clipboard = context.getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
                            clipboard.setPrimaryClip(ClipData.newPlainText("message", text))
                            Toast.makeText(context, "Copied to clipboard", Toast.LENGTH_SHORT).show()
                        },
                        onDelete = { msg ->
                            viewModel.deleteMessage(msg)
                        }
                    )
                }
            }

            // Image preview (when image is selected)
            uiState.selectedImageUri?.let { uri ->
                ImagePreview(
                    uri = uri,
                    isProcessing = uiState.isProcessingImage,
                    onSend = { caption ->
                        viewModel.sendImage(caption)
                        // Scroll after a brief delay to let DB update propagate
                        coroutineScope.launch {
                            kotlinx.coroutines.delay(150)
                            if (messages.isNotEmpty()) {
                                listState.animateScrollToItem(messages.size - 1)
                            }
                        }
                    },
                    onCancel = { viewModel.clearSelectedImage() }
                )
            }

            // Input bar
            MessageInput(
                text = draftText,
                onTextChange = { viewModel.updateDraftText(it) },
                enabled = uiState.currentTarget != null && uiState.isConnected,
                showAttachButton = uiState.selectedImageUri == null,
                onAttach = { imagePickerLauncher.launch("image/*") },
                onSend = { content ->
                    viewModel.sendMessage(content)
                    // Scroll after a brief delay to let DB update propagate
                    coroutineScope.launch {
                        kotlinx.coroutines.delay(150)
                        if (messages.isNotEmpty()) {
                            listState.animateScrollToItem(messages.size - 1)
                        }
                    }
                }
            )
        }
    }
}

@Composable
fun MessageBubble(
    message: MessageEntity,
    onCopy: (String) -> Unit,
    onDelete: (MessageEntity) -> Unit,
    modifier: Modifier = Modifier
) {
    val isOutgoing = message.isOutgoing
    val alignment = if (isOutgoing) Alignment.CenterEnd else Alignment.CenterStart
    val bubbleColor = if (isOutgoing) {
        MaterialTheme.colorScheme.primary
    } else {
        MaterialTheme.colorScheme.surfaceVariant
    }
    val textColor = if (isOutgoing) {
        MaterialTheme.colorScheme.onPrimary
    } else {
        MaterialTheme.colorScheme.onSurfaceVariant
    }

    // Parse node type for label color
    val senderColor = if (message.fromNode.startsWith("agent:")) AgentColor else UserColor

    // Check if this is an image message
    val isImage = isImageMessageContent(message.content)

    // Context menu state
    var showMenu by remember { mutableStateOf(false) }
    var showDeleteDialog by remember { mutableStateOf(false) }

    // Check for code blocks
    val codeBlockRegex = Regex("```[\\s\\S]*?```")
    val hasCodeBlocks = codeBlockRegex.containsMatchIn(message.content)

    // Extract code from code blocks
    fun extractCode(content: String): String {
        return codeBlockRegex.findAll(content)
            .map { match ->
                match.value
                    .removePrefix("```")
                    .let { it.substringAfter("\n", it) }
                    .removeSuffix("```")
                    .trim()
            }
            .joinToString("\n\n")
    }

    Box(
        modifier = modifier.fillMaxWidth(),
        contentAlignment = alignment
    ) {
        Column(
            modifier = Modifier.widthIn(max = 300.dp)
        ) {
            // Sender label (only for incoming)
            if (!isOutgoing) {
                Text(
                    text = getDisplayName(message.fromNode),
                    style = MaterialTheme.typography.labelSmall,
                    color = senderColor,
                    modifier = Modifier.padding(start = 8.dp, bottom = 2.dp)
                )
            }

            // Render image or text bubble
            if (isImage) {
                ImageMessageBubble(
                    content = message.content,
                    isOutgoing = isOutgoing
                )
            } else {
                Box {
                    Card(
                        colors = CardDefaults.cardColors(containerColor = bubbleColor),
                        shape = RoundedCornerShape(
                            topStart = 16.dp,
                            topEnd = 16.dp,
                            bottomStart = if (isOutgoing) 16.dp else 4.dp,
                            bottomEnd = if (isOutgoing) 4.dp else 16.dp
                        ),
                        modifier = Modifier.clickable { showMenu = true }
                    ) {
                        MarkdownText(
                            text = message.content,
                            textColor = textColor,
                            modifier = Modifier.padding(12.dp),
                            onLongClick = { showMenu = true }
                        )
                    }

                    // Context menu
                    DropdownMenu(
                        expanded = showMenu,
                        onDismissRequest = { showMenu = false }
                    ) {
                        DropdownMenuItem(
                            text = { Text("Copy") },
                            onClick = {
                                onCopy(message.content)
                                showMenu = false
                            }
                        )
                        if (hasCodeBlocks) {
                            DropdownMenuItem(
                                text = { Text("Copy code") },
                                onClick = {
                                    onCopy(extractCode(message.content))
                                    showMenu = false
                                }
                            )
                        }
                        DropdownMenuItem(
                            text = { Text("Delete") },
                            onClick = {
                                showMenu = false
                                showDeleteDialog = true
                            }
                        )
                    }
                }
            }

            // Timestamp and status row
            Row(
                modifier = Modifier
                    .padding(
                        start = if (isOutgoing) 0.dp else 8.dp,
                        end = if (isOutgoing) 8.dp else 0.dp,
                        top = 2.dp
                    )
                    .align(if (isOutgoing) Alignment.End else Alignment.Start),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(4.dp)
            ) {
                // Timestamp
                Text(
                    text = formatTimestamp(message.timestamp),
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.6f)
                )

                // Status indicator (only for outgoing messages)
                if (isOutgoing) {
                    MessageStatusIcon(status = message.status)
                }
            }
        }
    }

    // Delete confirmation dialog
    if (showDeleteDialog) {
        AlertDialog(
            onDismissRequest = { showDeleteDialog = false },
            title = { Text("Delete message?") },
            text = { Text("This will delete the message for everyone.") },
            confirmButton = {
                TextButton(
                    onClick = {
                        onDelete(message)
                        showDeleteDialog = false
                    }
                ) {
                    Text("Delete")
                }
            },
            dismissButton = {
                TextButton(onClick = { showDeleteDialog = false }) {
                    Text("Cancel")
                }
            }
        )
    }
}

@Composable
private fun MessageStatusIcon(status: MessageStatus) {
    val (icon, tint, contentDescription) = when (status) {
        MessageStatus.PENDING -> Triple(
            Icons.Default.Schedule,
            MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.5f),
            "Sending"
        )
        MessageStatus.SENT -> Triple(
            Icons.Default.Check,
            MaterialTheme.colorScheme.primary,
            "Sent"
        )
        MessageStatus.FAILED -> Triple(
            Icons.Default.Warning,
            MaterialTheme.colorScheme.error,
            "Failed"
        )
        MessageStatus.NONE -> return // No icon for incoming messages
    }

    Icon(
        imageVector = icon,
        contentDescription = contentDescription,
        tint = tint,
        modifier = Modifier.height(14.dp)
    )
}

@Composable
fun MessageInput(
    enabled: Boolean,
    onSend: (String) -> Unit,
    modifier: Modifier = Modifier,
    text: String = "",
    onTextChange: ((String) -> Unit)? = null,
    showAttachButton: Boolean = true,
    onAttach: (() -> Unit)? = null
) {
    // Use external state if provided, otherwise fall back to local state
    var localText by remember { mutableStateOf("") }
    val currentText = if (onTextChange != null) text else localText
    val updateText: (String) -> Unit = onTextChange ?: { localText = it }

    Row(
        modifier = modifier
            .fillMaxWidth()
            .background(MaterialTheme.colorScheme.surface)
            .padding(horizontal = 8.dp, vertical = 8.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        // Attach button
        if (showAttachButton && onAttach != null) {
            IconButton(
                onClick = onAttach,
                enabled = enabled
            ) {
                Icon(
                    Icons.Default.AttachFile,
                    contentDescription = "Attach image"
                )
            }
        }

        OutlinedTextField(
            value = currentText,
            onValueChange = updateText,
            modifier = Modifier.weight(1f),
            placeholder = { Text(stringResource(R.string.chat_hint)) },
            enabled = enabled,
            maxLines = 4,
            keyboardOptions = KeyboardOptions(imeAction = ImeAction.Default)
        )

        IconButton(
            onClick = {
                if (currentText.isNotBlank()) {
                    onSend(currentText)
                    updateText("")
                }
            },
            enabled = enabled && currentText.isNotBlank()
        ) {
            Icon(
                Icons.AutoMirrored.Filled.Send,
                contentDescription = stringResource(R.string.chat_send)
            )
        }
    }
}

private fun formatTimestamp(isoTimestamp: String): String {
    // Parse UTC ISO timestamp and convert to local time
    return try {
        val instant = Instant.parse(isoTimestamp)
        val localTime = instant.atZone(ZoneId.systemDefault())
        DateTimeFormatter.ofPattern("HH:mm").format(localTime)
    } catch (e: Exception) {
        // Fallback: extract time portion directly
        isoTimestamp.substringAfter("T").substringBefore(".").take(5)
    }
}

@Composable
fun StatusDialog(
    status: StatusResponse?,
    isLoading: Boolean,
    onDismiss: () -> Unit
) {
    AlertDialog(
        onDismissRequest = onDismiss,
        title = {
            if (status != null) {
                Text(stringResource(R.string.status_title, getDisplayName(status.fromNode)))
            } else {
                Text(stringResource(R.string.status_loading))
            }
        },
        text = {
            if (isLoading && status == null) {
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(32.dp),
                    contentAlignment = Alignment.Center
                ) {
                    // Use simple text instead of CircularProgressIndicator
                    // to avoid Compose animation version mismatch crash
                    Text(
                        text = "Loading...",
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
            } else if (status != null) {
                StatusContent(status = status)
            }
        },
        confirmButton = {
            TextButton(onClick = onDismiss) {
                Text(stringResource(R.string.status_dismiss))
            }
        }
    )
}

@Composable
private fun StatusContent(status: StatusResponse) {
    val scrollState = rememberScrollState()

    Column(
        modifier = Modifier
            .fillMaxWidth()
            .verticalScroll(scrollState),
        verticalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        // Live status section (heartbeat-lite: state, tokens, memory, uptime)
        if (status.statusSummary != null) {
            val ss = status.statusSummary
            val stateColor = if (ss.state == "busy") MaterialTheme.colorScheme.tertiary
                            else MaterialTheme.colorScheme.primary

            // State line
            val stateText = buildString {
                append(ss.state.uppercase())
                if (ss.state == "busy" && ss.workerElapsedS != null) {
                    append(" (${ss.workerElapsedS.toInt()}s)")
                }
                if (ss.contextTokens > 0) {
                    append("  ${ss.contextTokens / 1000}k ctx")
                }
            }
            Text(
                text = stateText,
                style = MaterialTheme.typography.titleSmall,
                fontWeight = FontWeight.Bold,
                fontFamily = FontFamily.Monospace,
                color = stateColor
            )

            // Detail line: history, memory, uptime
            val detailParts = mutableListOf<String>()
            detailParts.add("hist: ${ss.historyTurns} turns (${ss.historyPct.toInt()}%)")
            if (ss.memoryPool > 0 || ss.memoryActive > 0) {
                detailParts.add("mem: ${ss.memoryPool}/${ss.memoryActive}")
            }
            if (ss.uptimeS > 0) {
                val hours = (ss.uptimeS / 3600).toInt()
                val mins = ((ss.uptimeS % 3600) / 60).toInt()
                detailParts.add(if (hours > 0) "up: ${hours}h${mins}m" else "up: ${mins}m")
            }
            Text(
                text = detailParts.joinToString(" · "),
                style = MaterialTheme.typography.bodySmall,
                fontFamily = FontFamily.Monospace,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
            Divider(modifier = Modifier.padding(vertical = 8.dp))
        }

        // Full diagnostics (if available) or fallback to basic agent info
        if (status.diagnostics != null) {
            DiagnosticsContent(diag = status.diagnostics)
        } else if (status.hostname != null || status.model != null || status.backend != null) {
            Text(
                text = "Agent Info",
                style = MaterialTheme.typography.labelMedium,
                fontWeight = FontWeight.Bold
            )
            Column(
                modifier = Modifier.padding(start = 8.dp),
                verticalArrangement = Arrangement.spacedBy(2.dp)
            ) {
                if (status.hostname != null) {
                    DiagLine("Host", status.hostname)
                }
                if (status.backend != null || status.model != null) {
                    DiagLine("LLM", listOfNotNull(status.backend, status.model).joinToString("/"))
                }
                if (status.workingDirectory != null) {
                    DiagLine("Dir", status.workingDirectory)
                }
            }
            Divider(modifier = Modifier.padding(vertical = 8.dp))
        }

        // Real-time activity section (if agent is currently processing)
        if (!status.currentActivity.isNullOrEmpty()) {
            Text(
                text = "In-Progress Activity",
                style = MaterialTheme.typography.labelMedium,
                fontWeight = FontWeight.Bold,
                color = MaterialTheme.colorScheme.primary
            )
            Text(
                text = status.currentActivity,
                style = MaterialTheme.typography.bodySmall,
                fontFamily = FontFamily.Monospace,
                color = MaterialTheme.colorScheme.primary
            )
            Divider(modifier = Modifier.padding(vertical = 8.dp))
        }

        // Summary section (if available)
        if (!status.summary.isNullOrEmpty()) {
            Text(
                text = stringResource(R.string.status_summary_label),
                style = MaterialTheme.typography.labelMedium,
                fontWeight = FontWeight.Bold
            )
            Text(
                text = status.summary.take(300) + if (status.summary.length > 300) "..." else "",
                style = MaterialTheme.typography.bodySmall,
                fontStyle = FontStyle.Italic,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
            Divider(modifier = Modifier.padding(vertical = 8.dp))
        }

        // Context entries
        if (status.context.isEmpty()) {
            Text(
                text = stringResource(R.string.status_no_messages),
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        } else {
            Text(
                text = stringResource(R.string.status_context_label, status.context.size),
                style = MaterialTheme.typography.labelMedium,
                fontWeight = FontWeight.Bold
            )
            Spacer(modifier = Modifier.height(4.dp))

            status.context.forEach { entry ->
                StatusContextItem(entry = entry, agentNodeId = status.fromNode)
            }
        }
    }
}

@Composable
private fun StatusContextItem(
    entry: StatusContextEntry,
    agentNodeId: String
) {
    val isFromAgent = entry.from == agentNodeId
    val senderColor = when {
        entry.type == "tool_call" -> MaterialTheme.colorScheme.primary
        entry.type == "tool_result" -> MaterialTheme.colorScheme.tertiary
        isFromAgent -> AgentColor
        else -> UserColor
    }

    val typeLabel = when (entry.type) {
        "tool_call" -> "[tool]"
        "tool_result" -> "[result]"
        else -> ""
    }

    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 4.dp)
    ) {
        Row(
            horizontalArrangement = Arrangement.spacedBy(8.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text(
                text = getDisplayName(entry.from) + if (typeLabel.isNotEmpty()) " $typeLabel" else "",
                style = MaterialTheme.typography.labelSmall,
                fontWeight = FontWeight.Bold,
                color = senderColor
            )
            Text(
                text = formatTimestamp(entry.timestamp),
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.6f)
            )
        }

        // Truncate long messages
        val maxLen = if (entry.type == "tool_result") 150 else 300
        val displayContent = if (entry.content.length > maxLen) {
            entry.content.take(maxLen) + "... (${entry.content.length} chars)"
        } else {
            entry.content
        }

        Text(
            text = displayContent,
            style = MaterialTheme.typography.bodySmall,
            maxLines = 8,
            overflow = TextOverflow.Ellipsis
        )
    }
}

@Composable
private fun DiagLine(label: String, value: String) {
    Text(
        text = "$label: $value",
        style = MaterialTheme.typography.bodySmall,
        fontFamily = FontFamily.Monospace
    )
}

@Composable
private fun DiagSection(title: String, content: @Composable () -> Unit) {
    Text(
        text = title,
        style = MaterialTheme.typography.labelMedium,
        fontWeight = FontWeight.Bold
    )
    Column(
        modifier = Modifier.padding(start = 8.dp),
        verticalArrangement = Arrangement.spacedBy(2.dp)
    ) { content() }
    Divider(modifier = Modifier.padding(vertical = 4.dp))
}

@Composable
private fun DiagnosticsContent(diag: DiagnosticReport) {
    // Identity
    diag.identity?.let { i ->
        DiagSection("Identity") {
            DiagLine("Node", i["node_id"]?.toString() ?: "?")
            val host = i["hostname"]?.toString() ?: "?"
            val pid = i["pid"]?.toString() ?: "?"
            DiagLine("Host", "$host (PID $pid)")
            val uptimeS = (i["uptime_seconds"] as? Number)?.toDouble() ?: 0.0
            if (uptimeS > 0) {
                val hours = (uptimeS / 3600).toInt()
                val mins = ((uptimeS % 3600) / 60).toInt()
                DiagLine("Uptime", if (hours > 0) "${hours}h ${mins}m" else "${mins}m")
            }
            i["working_directory"]?.toString()?.let { DiagLine("Dir", it) }
        }
    }

    // LLM
    diag.llm?.let { ll ->
        DiagSection("LLM") {
            DiagLine("Worker", "${ll["backend"] ?: "?"} / ${ll["model"] ?: "?"}")
            DiagLine("Router", "${ll["router_llm_backend"] ?: "?"} / ${ll["router_llm_model"] ?: "?"}")
        }
    }

    // Router
    diag.router?.let { r ->
        DiagSection("Router") {
            val state = r["state"]?.toString()?.uppercase() ?: "?"
            val stateColor = if (state == "BUSY") MaterialTheme.colorScheme.tertiary
                             else MaterialTheme.colorScheme.primary
            Text(
                text = "State: $state",
                style = MaterialTheme.typography.bodySmall,
                fontFamily = FontFamily.Monospace,
                color = stateColor
            )
            if (r["worker_active"] == true) {
                val wid = r["worker_id"]?.toString() ?: "?"
                val elapsed = (r["worker_elapsed_seconds"] as? Number)?.toDouble()
                val elapsedStr = if (elapsed != null) ", ${elapsed.toInt()}s" else ""
                DiagLine("Worker", "active ($wid$elapsedStr)")
            } else {
                DiagLine("Worker", "inactive")
            }
            @Suppress("UNCHECKED_CAST")
            (r["session_stats"] as? Map<String, Any?>)?.let { ss ->
                DiagLine("Session", "${ss["user_turns"] ?: 0} user turns, ${ss["tool_calls"] ?: 0} tool calls")
            }
        }
    }

    // History
    diag.history?.let { h ->
        DiagSection("History") {
            if (h["detail"] != null) {
                DiagLine("", h["detail"].toString())
            } else {
                val turns = (h["window_turns"] as? Number)?.toInt() ?: 0
                val tokens = (h["estimated_tokens"] as? Number)?.toInt() ?: 0
                val soft = (h["soft_limit_tokens"] as? Number)?.toInt() ?: 0
                val hard = (h["hard_limit_tokens"] as? Number)?.toInt() ?: 0
                val pct = (h["utilization_pct"] as? Number)?.toDouble() ?: 0.0
                DiagLine("Window", "$turns turns (~${"%,d".format(tokens)} tokens)")
                DiagLine("Limits", "${"%,d".format(soft)} soft / ${"%,d".format(hard)} hard (${pct.toInt()}%)")
                val summ = if (h["summarization_enabled"] != true) "none (rolling window)" else "active"
                DiagLine("Summary", summ)
            }
        }
    }

    // Memory
    diag.memory?.let { m ->
        DiagSection("Memory") {
            if (m["enabled"] != true) {
                DiagLine("", m["detail"]?.toString() ?: "disabled")
            } else {
                DiagLine("Pool", "${m["pool_size"] ?: 0} entries (max ${m["pool_max_entries"] ?: "?"})")
                DiagLine("Active", "${m["active_set_size"] ?: 0} / ${m["active_set_target"] ?: "?"} target")
                val ago = (m["last_reflection_ago_seconds"] as? Number)?.toDouble()
                if (ago != null) {
                    val hours = (ago / 3600).toInt()
                    val mins = ((ago % 3600) / 60).toInt()
                    DiagLine("Last reflection", if (hours > 0) "${hours}h ${mins}m ago" else "${mins}m ago")
                }
            }
        }
    }

    // Health Checks
    diag.contextHealth?.let { ch ->
        @Suppress("UNCHECKED_CAST")
        val checks = ch["checks"] as? List<Map<String, Any?>> ?: return@let
        if (checks.isNotEmpty()) {
            DiagSection("Health Checks") {
                checks.forEach { check ->
                    val ok = check["ok"] == true
                    val icon = if (ok) "+" else "!"
                    val color = if (ok) MaterialTheme.colorScheme.primary
                                else MaterialTheme.colorScheme.error
                    Text(
                        text = "$icon ${check["name"] ?: "?"} (${check["detail"] ?: ""})",
                        style = MaterialTheme.typography.bodySmall,
                        fontFamily = FontFamily.Monospace,
                        color = color
                    )
                }
            }
        }
    }
}

/**
 * Image preview component shown when user has selected an image to send.
 */
@Composable
fun ImagePreview(
    uri: Uri,
    isProcessing: Boolean,
    onSend: (caption: String?) -> Unit,
    onCancel: () -> Unit,
    modifier: Modifier = Modifier
) {
    var caption by remember { mutableStateOf("") }
    val context = LocalContext.current

    Surface(
        modifier = modifier.fillMaxWidth(),
        color = MaterialTheme.colorScheme.surfaceVariant,
        tonalElevation = 2.dp
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(12.dp)
        ) {
            // Header with cancel button
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = "Send Image",
                    style = MaterialTheme.typography.titleSmall,
                    fontWeight = FontWeight.Bold
                )
                IconButton(
                    onClick = onCancel,
                    enabled = !isProcessing
                ) {
                    Icon(Icons.Default.Close, contentDescription = "Cancel")
                }
            }

            Spacer(modifier = Modifier.height(8.dp))

            // Image preview
            AsyncImage(
                model = ImageRequest.Builder(context)
                    .data(uri)
                    .crossfade(true)
                    .build(),
                contentDescription = "Selected image",
                modifier = Modifier
                    .fillMaxWidth()
                    .height(200.dp)
                    .clip(RoundedCornerShape(8.dp)),
                contentScale = ContentScale.Fit
            )

            Spacer(modifier = Modifier.height(8.dp))

            // Caption input and send button
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically
            ) {
                OutlinedTextField(
                    value = caption,
                    onValueChange = { caption = it },
                    modifier = Modifier.weight(1f),
                    placeholder = { Text("Add a caption (optional)") },
                    enabled = !isProcessing,
                    maxLines = 2,
                    singleLine = false
                )

                Spacer(modifier = Modifier.width(8.dp))

                IconButton(
                    onClick = { onSend(caption.ifBlank { null }) },
                    enabled = !isProcessing
                ) {
                    if (isProcessing) {
                        Text("...", style = MaterialTheme.typography.bodySmall)
                    } else {
                        Icon(Icons.AutoMirrored.Filled.Send, contentDescription = "Send image")
                    }
                }
            }
        }
    }
}

/**
 * Check if a message content string represents an image message.
 */
private fun isImageMessageContent(content: String): Boolean {
    return try {
        val json = Json.parseToJsonElement(content).jsonObject
        json["type"]?.jsonPrimitive?.content == "image"
    } catch (e: Exception) {
        false
    }
}

/**
 * Extract thumbnail data from image message content.
 */
private fun extractImageThumbnail(content: String): String? {
    return try {
        val json = Json.parseToJsonElement(content).jsonObject
        json["thumbnail"]?.jsonPrimitive?.content
    } catch (e: Exception) {
        null
    }
}

/**
 * Extract full image data from image message content.
 */
private fun extractImageData(content: String): String? {
    return try {
        val json = Json.parseToJsonElement(content).jsonObject
        json["data"]?.jsonPrimitive?.content
    } catch (e: Exception) {
        null
    }
}

/**
 * Extract caption from image message content.
 */
private fun extractImageCaption(content: String): String? {
    return try {
        val json = Json.parseToJsonElement(content).jsonObject
        json["caption"]?.jsonPrimitive?.content
    } catch (e: Exception) {
        null
    }
}

/**
 * Composable for rendering an image message bubble.
 */
@Composable
fun ImageMessageBubble(
    content: String,
    isOutgoing: Boolean,
    modifier: Modifier = Modifier
) {
    val thumbnail = extractImageThumbnail(content)
    val caption = extractImageCaption(content)
    var showFullImage by remember { mutableStateOf(false) }

    val bubbleColor = if (isOutgoing) {
        MaterialTheme.colorScheme.primary
    } else {
        MaterialTheme.colorScheme.surfaceVariant
    }
    val textColor = if (isOutgoing) {
        MaterialTheme.colorScheme.onPrimary
    } else {
        MaterialTheme.colorScheme.onSurfaceVariant
    }

    Card(
        colors = CardDefaults.cardColors(containerColor = bubbleColor),
        shape = RoundedCornerShape(
            topStart = 16.dp,
            topEnd = 16.dp,
            bottomStart = if (isOutgoing) 16.dp else 4.dp,
            bottomEnd = if (isOutgoing) 4.dp else 16.dp
        ),
        modifier = modifier.clickable { showFullImage = true }
    ) {
        Column(modifier = Modifier.padding(8.dp)) {
            // Render thumbnail
            if (thumbnail != null) {
                val bitmap = remember(thumbnail) {
                    ImageUtils.decodeBase64Image(thumbnail)
                }
                if (bitmap != null) {
                    Image(
                        bitmap = bitmap.asImageBitmap(),
                        contentDescription = "Image",
                        modifier = Modifier
                            .fillMaxWidth()
                            .clip(RoundedCornerShape(8.dp)),
                        contentScale = ContentScale.FillWidth
                    )
                } else {
                    // Fallback placeholder
                    Box(
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(100.dp)
                            .background(
                                MaterialTheme.colorScheme.surfaceVariant,
                                RoundedCornerShape(8.dp)
                            ),
                        contentAlignment = Alignment.Center
                    ) {
                        Icon(
                            Icons.Default.Image,
                            contentDescription = "Image",
                            modifier = Modifier.size(48.dp),
                            tint = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.5f)
                        )
                    }
                }
            } else {
                // No thumbnail - show placeholder
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(100.dp)
                        .background(
                            MaterialTheme.colorScheme.surfaceVariant,
                            RoundedCornerShape(8.dp)
                        ),
                    contentAlignment = Alignment.Center
                ) {
                    Icon(
                        Icons.Default.Image,
                        contentDescription = "Image",
                        modifier = Modifier.size(48.dp),
                        tint = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.5f)
                    )
                }
            }

            // Caption
            if (!caption.isNullOrBlank()) {
                Spacer(modifier = Modifier.height(4.dp))
                Text(
                    text = caption,
                    style = MaterialTheme.typography.bodyMedium,
                    color = textColor
                )
            }
        }
    }

    // Full image dialog
    if (showFullImage) {
        val fullImageData = extractImageData(content)
        AlertDialog(
            onDismissRequest = { showFullImage = false },
            title = { Text("Image") },
            text = {
                if (fullImageData != null) {
                    val bitmap = remember(fullImageData) {
                        ImageUtils.decodeBase64Image(fullImageData)
                    }
                    if (bitmap != null) {
                        Image(
                            bitmap = bitmap.asImageBitmap(),
                            contentDescription = "Full size image",
                            modifier = Modifier
                                .fillMaxWidth()
                                .clip(RoundedCornerShape(8.dp)),
                            contentScale = ContentScale.FillWidth
                        )
                    } else {
                        Text("Could not load image")
                    }
                } else {
                    Text("Image data not available")
                }
            },
            confirmButton = {
                TextButton(onClick = { showFullImage = false }) {
                    Text("Close")
                }
            }
        )
    }
}
