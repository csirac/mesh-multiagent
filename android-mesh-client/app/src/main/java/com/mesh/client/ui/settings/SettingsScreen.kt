package com.mesh.client.ui.settings

import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.foundation.layout.Row
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.Checkbox
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.ui.Alignment
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import com.mesh.client.R
import com.mesh.client.ui.theme.ErrorRed
import com.mesh.client.ui.theme.OnlineGreen
import com.mesh.client.ui.theme.ThemeMode
import com.mesh.client.ui.theme.ThemePreference
import kotlinx.coroutines.launch

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SettingsScreen(
    onNavigateToUsage: () -> Unit = {},
    onNavigateToWorkspace: () -> Unit = {},
    viewModel: SettingsViewModel = hiltViewModel()
) {
    val uiState by viewModel.uiState.collectAsState()
    val context = LocalContext.current
    val themePreference = remember { ThemePreference(context.applicationContext) }
    val currentThemeMode by themePreference.themeMode.collectAsState(initial = ThemeMode.SYSTEM)
    val scope = rememberCoroutineScope()

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text(stringResource(R.string.settings_title)) }
            )
        }
    ) { padding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
                .padding(16.dp)
                .verticalScroll(rememberScrollState())
        ) {
            // Connection status
            Card(
                modifier = Modifier.fillMaxWidth()
            ) {
                Column(modifier = Modifier.padding(16.dp)) {
                    Text(
                        text = if (uiState.isConnected) {
                            stringResource(R.string.connection_connected)
                        } else {
                            stringResource(R.string.connection_disconnected)
                        },
                        style = MaterialTheme.typography.titleMedium,
                        color = if (uiState.isConnected) OnlineGreen else ErrorRed
                    )

                    uiState.error?.let { error ->
                        Spacer(modifier = Modifier.height(8.dp))
                        Text(
                            text = error,
                            style = MaterialTheme.typography.bodySmall,
                            color = ErrorRed
                        )
                    }
                }
            }

            Spacer(modifier = Modifier.height(24.dp))

            // Appearance
            Text(
                text = "Appearance",
                style = MaterialTheme.typography.titleMedium
            )

            Spacer(modifier = Modifier.height(8.dp))

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = androidx.compose.foundation.layout.Arrangement.spacedBy(8.dp)
            ) {
                ThemeMode.values().forEach { mode ->
                    val selected = currentThemeMode == mode
                    OutlinedButton(
                        onClick = {
                            scope.launch { themePreference.setThemeMode(mode) }
                        },
                        modifier = Modifier.weight(1f),
                        colors = if (selected) {
                            ButtonDefaults.outlinedButtonColors(
                                containerColor = MaterialTheme.colorScheme.primaryContainer
                            )
                        } else {
                            ButtonDefaults.outlinedButtonColors()
                        },
                        border = if (selected) {
                            BorderStroke(2.dp, MaterialTheme.colorScheme.primary)
                        } else {
                            ButtonDefaults.outlinedButtonBorder
                        }
                    ) {
                        Text(
                            text = when (mode) {
                                ThemeMode.SYSTEM -> "System"
                                ThemeMode.LIGHT -> "Light"
                                ThemeMode.DARK -> "Dark"
                            },
                            style = MaterialTheme.typography.labelMedium
                        )
                    }
                }
            }

            Spacer(modifier = Modifier.height(24.dp))

            // Server settings
            Text(
                text = stringResource(R.string.settings_server),
                style = MaterialTheme.typography.titleMedium
            )

            Spacer(modifier = Modifier.height(8.dp))

            OutlinedTextField(
                value = uiState.host,
                onValueChange = { viewModel.updateHost(it) },
                label = { Text(stringResource(R.string.settings_host)) },
                modifier = Modifier.fillMaxWidth(),
                singleLine = true
            )

            Spacer(modifier = Modifier.height(8.dp))

            OutlinedTextField(
                value = uiState.port,
                onValueChange = { viewModel.updatePort(it) },
                label = { Text(stringResource(R.string.settings_port)) },
                modifier = Modifier.fillMaxWidth(),
                singleLine = true,
                keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number)
            )

            Spacer(modifier = Modifier.height(8.dp))

            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Checkbox(
                    checked = uiState.useTls,
                    onCheckedChange = { viewModel.updateUseTls(it) }
                )
                Text(
                    text = "Use TLS (wss://)",
                    style = MaterialTheme.typography.bodyMedium
                )
            }

            Spacer(modifier = Modifier.height(8.dp))

            OutlinedTextField(
                value = uiState.authToken,
                onValueChange = { viewModel.updateAuthToken(it) },
                label = { Text(stringResource(R.string.settings_auth_token)) },
                modifier = Modifier.fillMaxWidth(),
                singleLine = true,
                visualTransformation = PasswordVisualTransformation()
            )

            Spacer(modifier = Modifier.height(8.dp))

            OutlinedTextField(
                value = uiState.nickname,
                onValueChange = { viewModel.updateNickname(it) },
                label = { Text(stringResource(R.string.settings_nickname)) },
                modifier = Modifier.fillMaxWidth(),
                singleLine = true
            )

            Spacer(modifier = Modifier.height(24.dp))

            // Action buttons
            if (uiState.isConnected) {
                OutlinedButton(
                    onClick = { viewModel.disconnect() },
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Text(stringResource(R.string.settings_disconnect))
                }
            } else {
                Button(
                    onClick = { viewModel.connect() },
                    modifier = Modifier.fillMaxWidth(),
                    enabled = uiState.host.isNotBlank() &&
                            uiState.port.isNotBlank() &&
                            uiState.nickname.isNotBlank()
                ) {
                    Text(stringResource(R.string.settings_connect))
                }
            }

            Spacer(modifier = Modifier.height(8.dp))

            Button(
                onClick = { viewModel.saveSettings() },
                modifier = Modifier.fillMaxWidth()
            ) {
                Text(stringResource(R.string.settings_save))
            }

            Spacer(modifier = Modifier.height(24.dp))

            // Usage section
            Text(
                text = "Claude Code",
                style = MaterialTheme.typography.titleMedium
            )

            Spacer(modifier = Modifier.height(8.dp))

            OutlinedButton(
                onClick = onNavigateToUsage,
                modifier = Modifier.fillMaxWidth()
            ) {
                Text(stringResource(R.string.usage_button))
            }

            Spacer(modifier = Modifier.height(24.dp))

            // Workspace section
            Text(
                text = stringResource(R.string.workspace_title),
                style = MaterialTheme.typography.titleMedium
            )

            Spacer(modifier = Modifier.height(8.dp))

            OutlinedButton(
                onClick = onNavigateToWorkspace,
                modifier = Modifier.fillMaxWidth()
            ) {
                Text(stringResource(R.string.workspace_button))
            }
        }
    }
}
