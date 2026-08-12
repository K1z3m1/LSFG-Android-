package com.firstt175.deepdrop.ui

import android.content.Context
import android.content.Intent
import android.net.Uri
import android.provider.Settings
import android.widget.Toast
import androidx.compose.foundation.clickable
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
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Accessibility
import androidx.compose.material.icons.filled.BatteryFull
import androidx.compose.material.icons.filled.BugReport
import androidx.compose.material.icons.filled.ContentCopy
import androidx.compose.material.icons.filled.Gavel
import androidx.compose.material.icons.filled.Info
import androidx.compose.material.icons.filled.Language
import androidx.compose.material.icons.filled.MoreVert
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.Stop
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.RadioButton
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalClipboardManager
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.navigation.NavHostController
import com.firstt175.deepdrop.R
import com.firstt175.deepdrop.prefs.AppLanguage
import com.firstt175.deepdrop.prefs.AppLanguagePrefs
import com.firstt175.deepdrop.prefs.CaptureSource
import com.firstt175.deepdrop.prefs.LsfgPreferences
import com.firstt175.deepdrop.session.CrashReporter
import com.firstt175.deepdrop.session.LsfgForegroundService
import com.firstt175.deepdrop.session.PermissionsHelper
import com.firstt175.deepdrop.ui.components.IconBadge
import com.firstt175.deepdrop.ui.components.LsfgCard
import com.firstt175.deepdrop.ui.components.LsfgLogoMark
import com.firstt175.deepdrop.ui.components.StatusTone
import com.firstt175.deepdrop.ui.components.StepCard
import com.firstt175.deepdrop.ui.theme.LsfgPrimary
import com.firstt175.deepdrop.ui.theme.LsfgStatusGood
import com.firstt175.deepdrop.ui.theme.LsfgStatusWarn

@Composable
fun HomeScreen(nav: NavHostController) {
    val ctx = LocalContext.current
    val prefs = remember { LsfgPreferences(ctx) }
    val state by produceConfigState(prefs).collectAsState()
    val sessionRunning by LsfgForegroundService.isRunning.collectAsState()
    val clipboard = LocalClipboardManager.current

    var lastError by remember { mutableStateOf<String?>(null) }
    var showMoreMenu by remember { mutableStateOf(false) }
    var showCrashDialog by remember { mutableStateOf(false) }
    var crashPreview by remember { mutableStateOf("") }
    var showCrashDetail by remember { mutableStateOf(false) }
    var showLanguageDialog by remember { mutableStateOf(false) }

    LaunchedEffect(Unit) {
        if (CrashReporter.hasPendingCrash(ctx)) {
            crashPreview = CrashReporter.readCrashSummary(ctx)
            // Move the crash file aside immediately so the dialog never
            // reappears on the next launch (e.g. if the user swipes the app
            // away instead of tapping a button). The file is still kept
            // under last_crash_seen.txt so the share chip can attach it.
            CrashReporter.markPendingCrashSeen(ctx)
            showCrashDialog = true
        }
    }

    if (showCrashDialog) {
        AlertDialog(
            onDismissRequest = { showCrashDialog = false },
            icon = {
                IconBadge(
                    icon = Icons.Filled.BugReport,
                    tint = MaterialTheme.colorScheme.tertiary,
                    size = 44.dp,
                )
            },
            title = { Text(stringResource(R.string.crash_dialog_title)) },
            text = {
                Column {
                    Text(stringResource(R.string.crash_dialog_body))
                    if (showCrashDetail) {
                        Spacer(Modifier.height(8.dp))
                        Text(
                            text = crashPreview.take(4000),
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                        Spacer(Modifier.height(4.dp))
                        TextButton(onClick = {
                            clipboard.setText(AnnotatedString(crashPreview))
                            Toast.makeText(ctx, "Copied to clipboard", Toast.LENGTH_SHORT).show()
                        }) {
                            Icon(
                                imageVector = Icons.Filled.ContentCopy,
                                contentDescription = null,
                                modifier = Modifier.size(16.dp),
                            )
                            Spacer(Modifier.size(6.dp))
                            Text("Copy")
                        }
                    }
                }
            },
            confirmButton = {
                TextButton(onClick = {
                    val intent = CrashReporter.buildShareIntent(ctx)
                    if (intent != null) {
                        ctx.startActivity(
                            Intent.createChooser(intent, "Share crash report")
                                .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK),
                        )
                    }
                    CrashReporter.clearPendingCrash(ctx)
                    showCrashDialog = false
                }) { Text(stringResource(R.string.crash_dialog_share)) }
            },
            dismissButton = {
                Row {
                    TextButton(onClick = { showCrashDetail = !showCrashDetail }) {
                        Text(stringResource(R.string.crash_dialog_view))
                    }
                    TextButton(onClick = {
                        CrashReporter.clearPendingCrash(ctx)
                        showCrashDialog = false
                    }) { Text(stringResource(R.string.crash_dialog_dismiss)) }
                }
            },
        )
    }

    val a11yEnabled = PermissionsHelper.isAccessibilityServiceEnabled(ctx)

    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .statusBarsPadding()
            .padding(horizontal = 20.dp)
            .padding(top = 12.dp, bottom = 32.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp),
    ) {
        // Top bar: logo + title + quick actions
        Row(
            verticalAlignment = Alignment.CenterVertically,
            modifier = Modifier.fillMaxWidth().padding(vertical = 4.dp),
        ) {
            LsfgLogoMark(size = 36.dp)
            Spacer(Modifier.size(12.dp))
            Column(modifier = Modifier.weight(1f)) {
                Text(
                    text = stringResource(R.string.app_name),
                    style = MaterialTheme.typography.titleLarge,
                    color = MaterialTheme.colorScheme.onSurface,
                )
                Text(
                    text = "Frame generation via Vulkan",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
            IconChip(
                icon = Icons.Filled.Accessibility,
                tint = if (a11yEnabled) LsfgStatusGood else LsfgStatusWarn,
                onClick = {
                    ctx.startActivity(
                        Intent(Settings.ACTION_ACCESSIBILITY_SETTINGS)
                            .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK),
                    )
                },
            )
            Spacer(Modifier.size(8.dp))
            IconChip(
                icon = Icons.Filled.BugReport,
                tint = MaterialTheme.colorScheme.onSurfaceVariant,
                onClick = {
                    val intent = CrashReporter.buildShareIntent(ctx)
                    if (intent == null) {
                        Toast.makeText(ctx, R.string.crash_export_none, Toast.LENGTH_SHORT).show()
                    } else {
                        ctx.startActivity(
                            Intent.createChooser(intent, "Export diagnostic log")
                                .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK),
                        )
                    }
                },
            )
            Spacer(Modifier.size(8.dp))
            Box {
                IconChip(
                    icon = Icons.Filled.MoreVert,
                    tint = MaterialTheme.colorScheme.onSurfaceVariant,
                    onClick = { showMoreMenu = true },
                )
                DropdownMenu(expanded = showMoreMenu, onDismissRequest = { showMoreMenu = false }) {
                    if (!PermissionsHelper.isIgnoringBatteryOptimizations(ctx)) {
                        DropdownMenuItem(
                            text = { Text(stringResource(R.string.battery_exempt_menu_item)) },
                            leadingIcon = { Icon(Icons.Filled.BatteryFull, contentDescription = null) },
                            onClick = {
                                showMoreMenu = false
                                ctx.startActivity(
                                    PermissionsHelper.buildIgnoreBatteryOptimizationsIntent(ctx)
                                        .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK),
                                )
                            },
                        )
                    }
                    DropdownMenuItem(
                        text = { Text("Re-read legal notice") },
                        leadingIcon = { Icon(Icons.Filled.Gavel, contentDescription = null) },
                        onClick = {
                            showMoreMenu = false
                            nav.navigate(Routes.LEGAL)
                        },
                    )
                    DropdownMenuItem(
                        text = { Text("View log") },
                        leadingIcon = { Icon(Icons.Filled.ContentCopy, contentDescription = null) },
                        onClick = {
                            showMoreMenu = false
                            nav.navigate(Routes.LOG)
                        },
                    )
                    DropdownMenuItem(
                        text = { Text(stringResource(R.string.crash_export_log)) },
                        leadingIcon = { Icon(Icons.Filled.BugReport, contentDescription = null) },
                        onClick = {
                            showMoreMenu = false
                            val intent = CrashReporter.buildShareIntent(ctx)
                            if (intent == null) {
                                Toast.makeText(ctx, R.string.crash_export_none, Toast.LENGTH_SHORT).show()
                            } else {
                                ctx.startActivity(
                                    Intent.createChooser(intent, "Export diagnostic log")
                                        .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK),
                                )
                            }
                        },
                    )
                    DropdownMenuItem(
                        text = { Text(stringResource(R.string.language_menu_item)) },
                        leadingIcon = { Icon(Icons.Filled.Language, contentDescription = null) },
                        onClick = {
                            showMoreMenu = false
                            showLanguageDialog = true
                        },
                    )
                    DropdownMenuItem(
                        text = { Text(stringResource(R.string.credits_title)) },
                        leadingIcon = { Icon(Icons.Filled.Info, contentDescription = null) },
                        onClick = {
                            showMoreMenu = false
                            nav.navigate(Routes.CREDITS)
                        },
                    )
                }
            }
        }

        if (showLanguageDialog) {
            LanguagePickerDialog(onDismiss = { showLanguageDialog = false })
        }

        // Session card: sessions are launched exclusively via the Automatic
        // Overlay handle now (see step 4 below) — there is no manual "pick an
        // app, then start" flow. This card just surfaces overlay-permission
        // guidance and a manual stop, in case the handle/dot is unreachable.
        LsfgCard(accent = true) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Icon(
                    imageVector = Icons.Filled.PlayArrow,
                    contentDescription = null,
                    tint = MaterialTheme.colorScheme.onSurfaceVariant,
                    modifier = Modifier.size(18.dp),
                )
                Spacer(Modifier.size(8.dp))
                Text(
                    text = "Sessions start from the Automatic Overlay handle — add apps in step 4 below, then tap the handle when that app is open.",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    modifier = Modifier.weight(1f),
                )
            }
            Spacer(Modifier.height(12.dp))
            TextButton(
                onClick = {
                    lastError = null
                    if (!Settings.canDrawOverlays(ctx)) {
                        val uri = Uri.parse("package:${ctx.packageName}")
                        val intent = Intent(Settings.ACTION_MANAGE_OVERLAY_PERMISSION, uri)
                            .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                        ctx.startActivity(intent)
                        lastError = ctx.getString(R.string.perm_overlay_missing)
                        return@TextButton
                    }
                    if (sessionRunning) {
                        LsfgForegroundService.stop(ctx)
                    } else {
                        // No manual start here — sessions launch only from the
                        // Automatic Overlay handle (step 4). Surface that instead
                        // of silently doing nothing when the button reads START.
                        lastError = ctx.getString(R.string.session_start_hint)
                    }
                },
            ) {
                Icon(
                    imageVector = if (sessionRunning) Icons.Filled.Stop else Icons.Filled.PlayArrow,
                    contentDescription = null,
                    modifier = Modifier.size(16.dp),
                )
                Spacer(Modifier.size(6.dp))
                Text(
                    if (sessionRunning) {
                        stringResource(R.string.session_button_stop)
                    } else {
                        stringResource(R.string.session_button_start)
                    },
                )
            }
            if (lastError != null) {
                Spacer(Modifier.height(8.dp))
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text(
                        text = lastError!!,
                        color = MaterialTheme.colorScheme.error,
                        style = MaterialTheme.typography.bodySmall,
                        modifier = Modifier.weight(1f),
                    )
                    Spacer(Modifier.size(8.dp))
                    Box(
                        modifier = Modifier
                            .size(28.dp)
                            .clip(CircleShape)
                            .clickable {
                                clipboard.setText(AnnotatedString(lastError!!))
                                Toast.makeText(ctx, "Copied to clipboard", Toast.LENGTH_SHORT).show()
                            },
                        contentAlignment = Alignment.Center,
                    ) {
                        Icon(
                            imageVector = Icons.Filled.ContentCopy,
                            contentDescription = "Copy error",
                            tint = MaterialTheme.colorScheme.error,
                            modifier = Modifier.size(16.dp),
                        )
                    }
                }
            }

        }

        // Limitation notice
        LsfgCard {
            Row(verticalAlignment = Alignment.Top) {
                Box(
                    modifier = Modifier
                        .size(6.dp)
                        .clip(CircleShape)
                        .padding(top = 8.dp),
                )
                Column {
                    Text(
                        text = stringResource(R.string.limitation_title).uppercase(),
                        style = MaterialTheme.typography.labelSmall,
                        color = LsfgPrimary,
                    )
                    Spacer(Modifier.height(4.dp))
                    Text(
                        text = stringResource(R.string.limitation_body),
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
            }
        }

        if (state.captureSource == CaptureSource.SHIZUKU) {
            LsfgCard {
                Column {
                    Text(
                        text = "SHIZUKU CAPTURE",
                        style = MaterialTheme.typography.labelSmall,
                        color = LsfgPrimary,
                    )
                    Spacer(Modifier.height(4.dp))
                    Text(
                        text = "Uses Shizuku's privileged UID-filtered capture for the target app, avoiding MediaProjection overlay feedback. Requires the Shizuku app, Shizuku permission, and ADB or wireless debugging active before starting.",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
            }
        }

        if (state.captureSource == CaptureSource.ROOT) {
            LsfgCard {
                Column {
                    Text(
                        text = "ROOT CAPTURE",
                        style = MaterialTheme.typography.labelSmall,
                        color = LsfgPrimary,
                    )
                    Spacer(Modifier.height(4.dp))
                    Text(
                        text = "Uses root UID-filtered capture for the target app — no MediaProjection consent dialog. Requires root access granted to LLS in your root manager (Magisk / KernelSU / APatch).",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
            }
        }

        // Steps — cards grouped by concept (prerequisite → target → frame gen/pacing → display).
        val dllStatus = if (state.shadersReady) StatusTone.Good
        else if (state.dllDisplayName != null) StatusTone.Warn
        else StatusTone.Neutral
        val dllLabel = if (state.shadersReady) "Ready"
        else if (state.dllDisplayName != null) "Pending"
        else "Required"

        StepCard(
            number = 1,
            title = stringResource(R.string.nav_dll),
            subtitle = if (state.dllDisplayName != null)
                state.dllDisplayName!!
            else stringResource(R.string.dll_status_none),
            status = dllStatus,
            statusLabel = dllLabel,
            onClick = {
                if (!state.legalAccepted) nav.navigate(Routes.LEGAL) else nav.navigate(Routes.DLL)
            },
        )

        val frameGenSummary = if (state.lsfgEnabled) {
            buildString {
                append("Multiplier ${state.multiplier}× · flow ${"%.2f".format(state.flowScale)}")
                if (state.performanceMode) append(" · perf")
                if (state.hdrMode) append(" · HDR")
            }
        } else {
            "Off — raw capture passthrough"
        }
        StepCard(
            number = 2,
            title = stringResource(R.string.nav_framegen_pacing),
            subtitle = frameGenSummary,
            status = if (state.lsfgEnabled) StatusTone.Good else StatusTone.Neutral,
            statusLabel = if (state.lsfgEnabled) "On" else "Off",
            onClick = { nav.navigate(Routes.PARAMS_FRAMEGEN_PACING) },
        )

        val overlayDisplaySummary = buildString {
            append(
                when (state.captureSource) {
                    CaptureSource.SHIZUKU -> "Shizuku capture"
                    CaptureSource.ROOT -> "Root capture"
                    else -> "MediaProjection"
                },
            )
            append(" · ")
            append(
                when (state.drawerEdge) {
                    com.firstt175.deepdrop.prefs.DrawerEdge.LEFT -> "left handle"
                    com.firstt175.deepdrop.prefs.DrawerEdge.RIGHT -> "right handle"
                    com.firstt175.deepdrop.prefs.DrawerEdge.TOP -> "top handle"
                    com.firstt175.deepdrop.prefs.DrawerEdge.BOTTOM -> "bottom handle"
                },
            )
            if (state.fpsCounterEnabled) append(" · FPS")
        }
        StepCard(
            number = 3,
            title = stringResource(R.string.nav_overlay_display),
            subtitle = overlayDisplaySummary,
            status = StatusTone.Neutral,
            statusLabel = when (state.captureSource) {
                CaptureSource.SHIZUKU -> "Shizuku"
                CaptureSource.ROOT -> "Root"
                else -> "MP"
            },
            onClick = { nav.navigate(Routes.OVERLAY_DISPLAY) },
        )

        val autoCount = state.autoEnabledApps.size
        val autoSubtitle = if (autoCount == 0) {
            stringResource(R.string.automatic_overlay_count_zero)
        } else {
            stringResource(R.string.automatic_overlay_count_n, autoCount)
        }
        StepCard(
            number = 4,
            title = stringResource(R.string.nav_automatic_overlay),
            subtitle = autoSubtitle,
            status = if (autoCount > 0) StatusTone.Good else StatusTone.Neutral,
            statusLabel = if (autoCount > 0) "On" else "Off",
            onClick = { nav.navigate(Routes.AUTOMATIC_OVERLAY) },
        )

        Spacer(Modifier.height(8.dp))
        Text(
            text = "v${versionName(ctx)}",
            style = MaterialTheme.typography.labelSmall.copy(
                fontWeight = FontWeight.Normal,
                fontSize = 11.sp,
            ),
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            modifier = Modifier.align(Alignment.CenterHorizontally),
        )
    }
}

/** System / English / Thai picker. Saves to [AppLanguagePrefs] and recreates the
 *  Activity so [MainActivity.attachBaseContext] re-wraps with the new locale —
 *  Compose can't hot-swap resource-derived strings otherwise. */
@Composable
private fun LanguagePickerDialog(onDismiss: () -> Unit) {
    val ctx = LocalContext.current
    val activity = ctx as? android.app.Activity
    var selected by remember { mutableStateOf(AppLanguagePrefs.get(ctx)) }

    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text(stringResource(R.string.language_menu_item)) },
        text = {
            Column {
                val options = listOf(
                    AppLanguage.SYSTEM to stringResource(R.string.language_system),
                    AppLanguage.ENGLISH to stringResource(R.string.language_english),
                    AppLanguage.THAI to stringResource(R.string.language_thai),
                )
                options.forEach { (lang, label) ->
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        modifier = Modifier
                            .fillMaxWidth()
                            .clickable { selected = lang },
                    ) {
                        RadioButton(selected = selected == lang, onClick = { selected = lang })
                        Spacer(Modifier.size(4.dp))
                        Text(label, style = MaterialTheme.typography.bodyMedium)
                    }
                }
            }
        },
        confirmButton = {
            TextButton(onClick = {
                AppLanguagePrefs.set(ctx, selected)
                onDismiss()
                activity?.recreate()
            }) { Text(stringResource(R.string.language_apply)) }
        },
        dismissButton = {
            TextButton(onClick = onDismiss) { Text(stringResource(R.string.crash_dialog_dismiss)) }
        },
    )
}

@Composable
private fun IconChip(
    icon: androidx.compose.ui.graphics.vector.ImageVector,
    tint: Color,
    onClick: () -> Unit,
) {
    Box(
        modifier = Modifier
            .size(40.dp)
            .clip(CircleShape)
            .clickable { onClick() },
        contentAlignment = Alignment.Center,
    ) {
        Icon(
            imageVector = icon,
            contentDescription = null,
            tint = tint,
            modifier = Modifier.size(20.dp),
        )
    }
}

private fun versionName(ctx: Context): String {
    return runCatching {
        ctx.packageManager.getPackageInfo(ctx.packageName, 0).versionName ?: "—"
    }.getOrDefault("—")
}

