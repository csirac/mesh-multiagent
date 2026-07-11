package com.mesh.client.ui.dialogs

import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.dp
import com.mesh.client.R
import com.mesh.client.data.remote.ConfirmationRequest
import com.mesh.client.data.remote.protocol.getDisplayName

/**
 * Dialog for approving or rejecting tool confirmation requests from agents.
 */
@Composable
fun ConfirmationDialog(
    request: ConfirmationRequest,
    onApprove: () -> Unit,
    onReject: () -> Unit
) {
    val displayName = getDisplayName(request.fromNode)

    AlertDialog(
        onDismissRequest = { /* Don't allow dismissing without action */ },
        title = {
            Text(text = stringResource(R.string.confirm_title))
        },
        text = {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .verticalScroll(rememberScrollState())
            ) {
                // Agent name
                Text(
                    text = stringResource(R.string.confirm_from_label, displayName),
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )

                Spacer(modifier = Modifier.height(8.dp))

                // Tool name
                Text(
                    text = stringResource(R.string.confirm_tool, request.toolName),
                    style = MaterialTheme.typography.titleMedium
                )

                Spacer(modifier = Modifier.height(12.dp))

                // Preview (code/content)
                if (request.preview.isNotBlank()) {
                    Surface(
                        color = MaterialTheme.colorScheme.surfaceVariant,
                        shape = MaterialTheme.shapes.small,
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Text(
                            text = request.preview,
                            style = MaterialTheme.typography.bodySmall.copy(
                                fontFamily = FontFamily.Monospace
                            ),
                            modifier = Modifier.padding(12.dp)
                        )
                    }
                }
            }
        },
        confirmButton = {
            TextButton(onClick = onApprove) {
                Text(stringResource(R.string.confirm_approve))
            }
        },
        dismissButton = {
            TextButton(onClick = onReject) {
                Text(stringResource(R.string.confirm_reject))
            }
        }
    )
}
