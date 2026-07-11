package com.mesh.client.ui.usage

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import com.mesh.client.R
import com.mesh.client.data.remote.AccountUsage
import com.mesh.client.data.remote.UsageWindow
import java.time.Duration
import java.time.Instant

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun UsageScreen(
    onNavigateBack: () -> Unit,
    viewModel: UsageViewModel = hiltViewModel()
) {
    val usage by viewModel.usage.collectAsState()
    val isLoading by viewModel.isLoading.collectAsState()
    val isConnected by viewModel.isConnected.collectAsState()

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text(stringResource(R.string.usage_title)) },
                navigationIcon = {
                    IconButton(onClick = onNavigateBack) {
                        Icon(Icons.AutoMirrored.Filled.ArrowBack, contentDescription = "Back")
                    }
                }
            )
        }
    ) { innerPadding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(innerPadding)
                .padding(horizontal = 16.dp)
                .verticalScroll(rememberScrollState())
        ) {
            // Refresh button
            OutlinedButton(
                onClick = { viewModel.fetchUsage() },
                enabled = isConnected && !isLoading,
                modifier = Modifier.fillMaxWidth()
            ) {
                Text(if (isLoading) "Loading..." else "Refresh Usage")
            }

            if (!isConnected) {
                Spacer(modifier = Modifier.height(16.dp))
                Text(
                    text = "Not connected to mesh",
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.error
                )
            }

            Spacer(modifier = Modifier.height(16.dp))

            if (isLoading && usage == null) {
                Box(
                    modifier = Modifier.fillMaxWidth(),
                    contentAlignment = Alignment.Center
                ) {
                    // Use simple text instead of CircularProgressIndicator
                    // to avoid Compose animation version mismatch crash
                    Text(
                        text = "Loading usage data...",
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
            }

            usage?.accounts?.forEach { account ->
                AccountCard(account)
                Spacer(modifier = Modifier.height(12.dp))
            }
        }
    }
}

@Composable
private fun AccountCard(account: AccountUsage) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        elevation = CardDefaults.cardElevation(defaultElevation = 2.dp)
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = account.label,
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.Bold
                )
                Text(
                    text = account.subscriptionType,
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }

            Spacer(modifier = Modifier.height(8.dp))

            if (account.error != null) {
                Text(
                    text = "Error: ${account.error}",
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.error
                )
            } else {
                account.windows.forEach { window ->
                    WindowRow(window)
                    Spacer(modifier = Modifier.height(6.dp))
                }

                account.extraUsage?.let { extra ->
                    if (extra.isEnabled && extra.monthlyLimit > 0) {
                        Spacer(modifier = Modifier.height(4.dp))
                        Text(
                            text = "Extra: $${String.format("%.2f", extra.usedCredits)} / $${String.format("%.2f", extra.monthlyLimit)}",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }
                }
            }
        }
    }
}

@Composable
private fun WindowRow(window: UsageWindow) {
    val pct = window.utilization.toFloat() / 100f
    val barColor = when {
        window.utilization >= 90 -> Color(0xFFE53935) // red
        window.utilization >= 70 -> Color(0xFFFB8C00) // orange
        else -> Color(0xFF43A047)                      // green
    }

    Column {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            Text(
                text = window.name,
                style = MaterialTheme.typography.bodyMedium
            )
            Row {
                Text(
                    text = "${String.format("%.1f", window.utilization)}%",
                    style = MaterialTheme.typography.bodyMedium,
                    fontWeight = FontWeight.Medium,
                    color = barColor
                )
                window.resetsAt?.let { resetStr ->
                    val resetText = formatResetTime(resetStr)
                    if (resetText.isNotEmpty()) {
                        Spacer(modifier = Modifier.width(8.dp))
                        Text(
                            text = resetText,
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }
                }
            }
        }
        Spacer(modifier = Modifier.height(2.dp))
        LinearProgressIndicator(
            progress = pct.coerceIn(0f, 1f),
            modifier = Modifier.fillMaxWidth().height(6.dp),
            color = barColor,
            trackColor = MaterialTheme.colorScheme.surfaceVariant,
        )
    }
}

private fun formatResetTime(isoTimestamp: String): String {
    return try {
        val resetInstant = Instant.parse(isoTimestamp)
        val now = Instant.now()
        val duration = Duration.between(now, resetInstant)
        if (duration.isNegative) return "now"
        val totalSecs = duration.seconds
        val days = totalSecs / 86400
        val hours = (totalSecs % 86400) / 3600
        val mins = (totalSecs % 3600) / 60
        val parts = mutableListOf<String>()
        if (days > 0) parts.add("${days}d")
        if (hours > 0) parts.add("${hours}h")
        if (mins > 0) parts.add("${mins}m")
        parts.joinToString(" ").ifEmpty { "<1m" }
    } catch (_: Exception) {
        ""
    }
}
