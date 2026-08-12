package com.firstt175.deepdrop.ui

import android.content.Intent
import android.widget.Toast
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ContentCopy
import androidx.compose.material.icons.filled.DeleteOutline
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.Share
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalClipboardManager
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.unit.dp
import androidx.navigation.NavHostController
import com.firstt175.deepdrop.session.CrashReporter
import com.firstt175.deepdrop.ui.components.LsfgCard
import com.firstt175.deepdrop.ui.components.LsfgSecondaryButton
import com.firstt175.deepdrop.ui.components.LsfgTopBar
import com.firstt175.deepdrop.ui.theme.LsfgPrimary
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * Max bytes read from `filesDir/lsfg.log` for on-screen display. The full
 * file is still attached whole by [CrashReporter.buildShareIntent] — this
 * cap only bounds what gets rendered as Compose text (a multi-MB log would
 * otherwise make this screen janky to scroll).
 */
private const val LOG_DISPLAY_TAIL_BYTES = 256 * 1024

@Composable
fun LogScreen(nav: NavHostController) {
    val ctx = LocalContext.current
    val clipboard = LocalClipboardManager.current

    var logText by remember { mutableStateOf("") }
    var loading by remember { mutableStateOf(true) }
    var refreshTick by remember { mutableStateOf(0) }
    var showClearConfirm by remember { mutableStateOf(false) }

    LaunchedEffect(refreshTick) {
        loading = true
        logText = withContext(Dispatchers.IO) {
            runCatching {
                val f = CrashReporter.logFile(ctx)
                if (!f.exists()) return@runCatching ""
                val bytes = f.readBytes()
                val start = (bytes.size - LOG_DISPLAY_TAIL_BYTES).coerceAtLeast(0)
                String(bytes, start, bytes.size - start, Charsets.UTF_8)
            }.getOrDefault("")
        }
        loading = false
    }

    if (showClearConfirm) {
        AlertDialog(
            onDismissRequest = { showClearConfirm = false },
            title = { Text("Clear log?") },
            text = { Text("This deletes lsfg.log on this device. It can't be undone.") },
            confirmButton = {
                TextButton(onClick = {
                    showClearConfirm = false
                    runCatching { CrashReporter.logFile(ctx).delete() }
                    refreshTick++
                }) { Text("Clear") }
            },
            dismissButton = {
                TextButton(onClick = { showClearConfirm = false }) { Text("Cancel") }
            },
        )
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .statusBarsPadding()
            .padding(horizontal = 20.dp)
            .padding(bottom = 16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp),
    ) {
        LsfgTopBar(
            title = "Log",
            onBack = { nav.popBackStack() },
            trailing = {
                IconTextButton(
                    icon = Icons.Filled.Refresh,
                    onClick = { refreshTick++ },
                )
            },
        )

        LsfgCard {
            Text(
                text = "LSFG.LOG",
                style = MaterialTheme.typography.labelSmall,
                color = LsfgPrimary,
            )
            Spacer(Modifier.height(4.dp))
            Text(
                text = when {
                    loading -> "Loading…"
                    logText.isEmpty() -> "Nothing logged yet."
                    else -> "Showing the most recent entries. Copy or share to send the full picture with a bug report."
                },
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }

        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            LsfgSecondaryButton(
                text = "Copy",
                onClick = {
                    clipboard.setText(AnnotatedString(logText))
                    Toast.makeText(ctx, "Log copied", Toast.LENGTH_SHORT).show()
                },
                leadingIcon = Icons.Filled.ContentCopy,
                enabled = logText.isNotEmpty(),
                modifier = Modifier.weight(1f),
            )
            LsfgSecondaryButton(
                text = "Share",
                onClick = {
                    val intent = CrashReporter.buildShareIntent(ctx)
                    if (intent == null) {
                        Toast.makeText(ctx, "Nothing to share yet", Toast.LENGTH_SHORT).show()
                    } else {
                        ctx.startActivity(
                            Intent.createChooser(intent, "Share log")
                                .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK),
                        )
                    }
                },
                leadingIcon = Icons.Filled.Share,
                modifier = Modifier.weight(1f),
            )
            LsfgSecondaryButton(
                text = "Clear",
                onClick = { showClearConfirm = true },
                leadingIcon = Icons.Filled.DeleteOutline,
                enabled = logText.isNotEmpty(),
                modifier = Modifier.weight(1f),
            )
        }

        LsfgCard(
            modifier = Modifier.fillMaxSize(),
            contentPadding = androidx.compose.foundation.layout.PaddingValues(12.dp),
        ) {
            val lines = remember(logText) { logText.split("\n") }
            LazyColumn(modifier = Modifier.fillMaxSize()) {
                items(lines) { line ->
                    Text(
                        text = line,
                        style = MaterialTheme.typography.bodySmall.copy(
                            fontFamily = androidx.compose.ui.text.font.FontFamily.Monospace,
                        ),
                        color = MaterialTheme.colorScheme.onSurface,
                    )
                }
            }
        }
    }
}

@Composable
private fun IconTextButton(icon: androidx.compose.ui.graphics.vector.ImageVector, onClick: () -> Unit) {
    androidx.compose.material3.IconButton(onClick = onClick) {
        Icon(icon, contentDescription = "Refresh", tint = MaterialTheme.colorScheme.onSurface)
    }
}
