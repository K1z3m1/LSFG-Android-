package com.firstt175.deepdrop.ui

import androidx.compose.animation.core.tween
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.RowScope
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ShowChart
import androidx.compose.material.icons.automirrored.filled.ViewSidebar
import androidx.compose.material.icons.filled.AspectRatio
import androidx.compose.material.icons.filled.FlashOn
import androidx.compose.material.icons.filled.HdrOn
import androidx.compose.material.icons.filled.Memory
import androidx.compose.material.icons.filled.OpenInFull
import androidx.compose.material.icons.filled.Speed
import androidx.compose.material.icons.filled.Timeline
import androidx.compose.material.icons.filled.Tune
import androidx.compose.material.icons.filled.TouchApp
import androidx.compose.material.icons.filled.Warning
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.FilterChip
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
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
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.unit.dp
import androidx.navigation.NavHostController
import com.firstt175.deepdrop.R
import com.firstt175.deepdrop.prefs.CaptureSource
import com.firstt175.deepdrop.prefs.DrawerEdge
import com.firstt175.deepdrop.prefs.FramegenBackend
import com.firstt175.deepdrop.prefs.LsfgPreferences
import com.firstt175.deepdrop.prefs.OverlayMode
import com.firstt175.deepdrop.session.AutoOverlayController
import com.firstt175.deepdrop.session.NativeBridge
import com.firstt175.deepdrop.ui.components.IconBadge
import com.firstt175.deepdrop.ui.components.LsfgCard
import com.firstt175.deepdrop.ui.components.LsfgTopBar
import com.firstt175.deepdrop.ui.components.SectionHeader
import com.firstt175.deepdrop.ui.components.ToggleRow
import com.firstt175.deepdrop.ui.components.ValueSlider
import com.firstt175.deepdrop.ui.theme.LsfgPrimary

// ----------------------------------------------------------------------------------------
// Frame generation & pacing — consolidated screen
// ----------------------------------------------------------------------------------------

@Composable
fun ParamsFrameGenPacingScreen(nav: NavHostController) {
    val ctx = LocalContext.current
    val prefs = remember { LsfgPreferences(ctx) }
    val state by produceConfigState(prefs).collectAsState()
    // The FP16 toggle is only meaningful when (a) the GPU advertises shaderFloat16
    // and (b) the FP16 SPIR-V cache from Lossless.dll has been populated. The
    // native probe ANDs both. Cached for the screen's lifetime — recomputed on
    // re-entry, which covers the post-DLL-pick / post-extract case naturally.
    val fp16Available = remember {
        val cacheDir = java.io.File(ctx.filesDir, "spirv").absolutePath
        runCatching { NativeBridge.isFramegenFp16Supported(cacheDir) }.getOrDefault(false)
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .statusBarsPadding()
            .padding(horizontal = 20.dp)
            .padding(bottom = 32.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp),
    ) {
        LsfgTopBar(
            title = stringResource(R.string.nav_framegen_pacing),
            onBack = { nav.popBackStack() },
        )

        // ---- Frame generation -------------------------------------------------------
        LsfgCard {
            SectionHeader(eyebrow = stringResource(R.string.section_frame_generation), title = null)
            Spacer(Modifier.height(4.dp))
            Text(
                text = if (state.framegenBackend == FramegenBackend.NCNN_AI) {
                    "Backend: AI (ncnn). Multiplier, flow scale and render resolution below apply " +
                        "to it too — the LSFG-shader-only options (performance mode, HDR, FP16) are " +
                        "hidden since they don't do anything on this backend. Switch backend and " +
                        "pick the compute path on the Frame-gen source screen."
                } else {
                    "Backend: Lossless.dll (LSFG). Switch backend on the Frame-gen source screen."
                },
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            Spacer(Modifier.height(8.dp))

            ToggleRow(
                icon = Icons.Filled.FlashOn,
                title = "LSFG-Android+ Frame Gen",
                description = "Master toggle for frame generation. Off = raw capture passthrough.",
                checked = state.lsfgEnabled,
                onCheckedChange = {
                    prefs.setLsfgEnabled(it)
                    refreshConfigState(prefs)
                },
            )
            HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant)
            Spacer(Modifier.height(8.dp))

            // The AI (ncnn) backend is locked to ×2. RIFE/IFRNet were only ever exported
            // and validated for the single-midpoint (k=1, multiplier=2) case — nothing
            // above ×2 has been trained/tested for these models, so the slider is
            // disabled and forced back to 2 rather than letting it silently run an
            // unvalidated higher multiplier. Only the LSFG_DLL shader backend supports
            // the full 2..8 range.
            if (state.framegenBackend == FramegenBackend.NCNN_AI) {
                if (state.multiplier != 2) {
                    LaunchedEffect(Unit) {
                        prefs.setMultiplier(2)
                        refreshConfigState(prefs)
                    }
                }
                ValueSlider(
                    title = stringResource(R.string.param_multiplier),
                    valueDisplay = "2× (locked)",
                    description = "AI backend is locked to ×2 — RIFE/IFRNet are only " +
                        "validated at the single midpoint frame. Switch to the Lossless.dll " +
                        "backend for ×3–×8.",
                    value = 2f,
                    range = 2f..2f,
                    steps = 0,
                    leadingIcon = Icons.Filled.Timeline,
                    enabled = false,
                    onValueChange = {},
                )
            } else {
                ValueSlider(
                    title = stringResource(R.string.param_multiplier),
                    valueDisplay = "${state.multiplier}×",
                    description = stringResource(R.string.param_multiplier_desc),
                    value = state.multiplier.toFloat(),
                    range = 2f..8f,
                    steps = 5,
                    leadingIcon = Icons.Filled.Timeline,
                    onValueChange = {
                        prefs.setMultiplier(it.toInt().coerceIn(2, 8))
                        refreshConfigState(prefs)
                    },
                )
            }

            // Like the multiplier slider above, flow scale only affects the LSFG_DLL
            // shader chain's motion-estimation pass (see resourcepool.cpp). The AI
            // (ncnn) backend's RIFE/IFRNet nets are single-pass with no separate
            // low-res flow stage to downscale — flowScale is accepted by
            // NcnnInterpolator::interpolate()/IfrnetInterpolator::interpolate() only
            // for call-site compatibility and is explicitly unused (see the
            // `/*flowScale — unused*/` parameter comment in both .cpp files).
            // Leaving the slider live here made it look broken: dragging it changed
            // the stored pref but had no visible effect on AI-backend output.
            if (state.framegenBackend == FramegenBackend.NCNN_AI) {
                ValueSlider(
                    title = stringResource(R.string.param_flow_scale),
                    valueDisplay = "N/A",
                    description = "AI backend's RIFE/IFRNet nets have no separate flow " +
                        "stage to scale — this only affects the Lossless.dll shader backend.",
                    value = 1.0f,
                    range = 1.0f..1.0f,
                    steps = 0,
                    leadingIcon = Icons.AutoMirrored.Filled.ShowChart,
                    enabled = false,
                    onValueChange = {},
                )
            } else {
                ValueSlider(
                    title = stringResource(R.string.param_flow_scale),
                    // Keep the original 0.1.3 representation: Flow scale is the
                    // raw 0.25..1.0 value, not a percentage and not a preset.
                    valueDisplay = "%.2f".format(state.flowScale),
                    description = stringResource(R.string.param_flow_scale_desc),
                    value = state.flowScale,
                    range = 0.25f..1.0f,
                    steps = 0,
                    leadingIcon = Icons.AutoMirrored.Filled.ShowChart,
                    onValueChange = {
                        prefs.setFlowScale(it)
                        refreshConfigState(prefs)
                    },
                )
            }

            ValueSlider(
                title = stringResource(R.string.param_render_resolution_scale),
                valueDisplay = "${(state.renderResolutionScale * 100f).toInt()}%",
                description = stringResource(R.string.param_render_resolution_scale_desc),
                value = state.renderResolutionScale,
                range = 0f..1f,
                steps = 19,
                leadingIcon = Icons.Filled.AspectRatio,
                onValueChange = {
                    prefs.setRenderResolutionScale(it)
                    refreshConfigState(prefs)
                },
            )

            // Everything below this point only takes effect on the LSFG_3_1/3_1P
            // shader chain — the AI (ncnn) backend never reads performanceMode,
            // hdrMode or framegenFp16 at all (see lsfg_render_loop.cpp's
            // initRenderLoop: when cfg.aiBackend loads successfully,
            // initFramegen/createFramegenContext — the only code that consults
            // these three — is skipped entirely). Showing them while the AI
            // backend is selected would suggest they do something they don't.
            if (state.framegenBackend == FramegenBackend.LSFG_DLL) {
                Spacer(Modifier.height(8.dp))
                HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant)
                ToggleRow(
                    icon = Icons.Filled.Speed,
                    title = stringResource(R.string.param_performance_mode),
                    description = stringResource(R.string.param_performance_mode_desc),
                    checked = state.performanceMode,
                    onCheckedChange = {
                        prefs.setPerformance(it)
                        refreshConfigState(prefs)
                    },
                )
                HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant)
                ToggleRow(
                    icon = Icons.Filled.HdrOn,
                    title = stringResource(R.string.param_hdr),
                    description = stringResource(R.string.param_hdr_desc),
                    checked = state.hdrMode,
                    onCheckedChange = {
                        prefs.setHdr(it)
                        refreshConfigState(prefs)
                    },
                )
                if (fp16Available) {
                    HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant)
                    ToggleRow(
                        icon = Icons.Filled.Memory,
                        title = stringResource(R.string.param_framegen_fp16),
                        description = stringResource(R.string.param_framegen_fp16_desc),
                        checked = state.framegenFp16,
                        onCheckedChange = {
                            prefs.setFramegenFp16(it)
                            refreshConfigState(prefs)
                        },
                    )
                }
            }
        }

        // ---- Post-processing -------------------------------------------------------
        LsfgCard {
            SectionHeader(eyebrow = "POST-PROCESSING", title = null)
            Spacer(Modifier.height(4.dp))
            Text(
                text = "Optional display-stage pipeline. It runs after frame generation and does not replace or alter the frame-generation loop.",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            Spacer(Modifier.height(8.dp))

            ToggleRow(
                icon = Icons.Filled.Tune,
                title = "GPU upscale (FSR1 EASU + RCAS)",
                description = "Real Vulkan compute pass. Off by default; when enabled it runs immediately before display.",
                checked = state.gpuPostProcessing,
                onCheckedChange = {
                    prefs.setGpuPostProcessing(it)
                    refreshConfigState(prefs)
                },
            )
            if (state.gpuPostProcessing) {
                ValueSlider(
                    title = "GPU upscale factor",
                    valueDisplay = "%.1fx".format(state.gpuUpscaleFactor),
                    description = "1.0x keeps the render size; 1.5x or 2.0x increases the final post-process image.",
                    value = state.gpuUpscaleFactor,
                    range = 1.0f..2.0f,
                    steps = 1,
                    leadingIcon = Icons.Filled.AspectRatio,
                    onValueChange = {
                        prefs.setGpuUpscaleFactor(it)
                        refreshConfigState(prefs)
                    },
                )
            }

            val npuAvailable = remember {
                runCatching { NativeBridge.isNpuPostProcessAvailable() }.getOrDefault(false)
            }
            if (npuAvailable) {
                HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant)
                ToggleRow(
                    icon = Icons.Filled.Memory,
                    title = "NPU enhance (NNAPI)",
                    description = "Dedicated NNAPI accelerator only. No CPU/GPU fallback if the NPU graph is rejected.",
                    checked = state.npuPostProcessing,
                    onCheckedChange = {
                        prefs.setNpuPostProcessing(it)
                        refreshConfigState(prefs)
                    },
                )
            }
            HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant)
            ToggleRow(
                icon = Icons.Filled.Memory,
                title = "CPU final sharpen",
                description = "Final CPU 3×3 unsharp-mask pass. Off by default and only runs when enabled.",
                checked = state.cpuPostProcessing,
                onCheckedChange = {
                    prefs.setCpuPostProcessing(it)
                    refreshConfigState(prefs)
                },
            )
        }

        // ---- Pacing -----------------------------------------------------------------
        LsfgCard {
            SectionHeader(eyebrow = stringResource(R.string.section_pacing), title = null)
            Spacer(Modifier.height(4.dp))

            Text(
                text = "Pacing controls timing of generated frames — preset and VSYNC alignment. Full pacing tuning is only available in the in-game overlay (advanced EMA/slack/queue sliders).",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            Spacer(Modifier.height(8.dp))
            Text(
                text = "Current preset: ${state.pacingPreset.name.lowercase().replaceFirstChar { it.uppercase() }}" +
                    " · VSYNC-aligned: ${if (state.vsyncAlignmentEnabled) "on" else "off"}",
                style = MaterialTheme.typography.bodySmall,
                color = LsfgPrimary,
            )
        }

        TailNote()
    }
}

// ----------------------------------------------------------------------------------------
// Overlay & Display — new screen consolidating overlay handle, HUD, capture mode
// ----------------------------------------------------------------------------------------

@Composable
fun OverlayDisplayScreen(nav: NavHostController) {
    val ctx = LocalContext.current
    val prefs = remember { LsfgPreferences(ctx) }
    val state by produceConfigState(prefs).collectAsState()

    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .statusBarsPadding()
            .padding(horizontal = 20.dp)
            .padding(bottom = 32.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp),
    ) {
        LsfgTopBar(
            title = stringResource(R.string.nav_overlay_display),
            onBack = { nav.popBackStack() },
        )

        // ---- Overlay entry mode (icon button vs drawer) -----------------------------
        LsfgCard {
            SectionHeader(eyebrow = stringResource(R.string.section_overlay_mode), title = null)
            Spacer(Modifier.height(4.dp))
            OverlayModeSelector(
                selected = state.overlayMode,
                onSelected = {
                    prefs.setOverlayMode(it)
                    refreshConfigState(prefs)
                    // Live-update the Automatic Overlay launcher: if the dot is
                    // currently shown for a target app it is recreated with the
                    // new affordance, otherwise the change is picked up the next
                    // time it appears.
                    AutoOverlayController.onOverlayModeChanged(ctx)
                },
            )
        }

        // ---- Overlay handle ---------------------------------------------------------
        LsfgCard {
            SectionHeader(eyebrow = stringResource(R.string.section_overlay_handle), title = null)
            Spacer(Modifier.height(4.dp))
            DrawerEdgeSelector(
                selected = state.drawerEdge,
                onSelected = {
                    prefs.setDrawerEdge(it)
                    refreshConfigState(prefs)
                },
            )
        }

        // ---- HUD --------------------------------------------------------------------
        LsfgCard {
            SectionHeader(eyebrow = stringResource(R.string.section_hud), title = null)
            Spacer(Modifier.height(4.dp))
            ToggleRow(
                icon = Icons.Filled.FlashOn,
                title = stringResource(R.string.param_fps_counter),
                description = stringResource(R.string.param_fps_counter_desc),
                checked = state.fpsCounterEnabled,
                onCheckedChange = {
                    prefs.setFpsCounterEnabled(it)
                    refreshConfigState(prefs)
                },
            )
            HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant)
            ToggleRow(
                icon = Icons.AutoMirrored.Filled.ShowChart,
                title = stringResource(R.string.param_frame_graph),
                description = stringResource(R.string.param_frame_graph_desc),
                checked = state.frameGraphEnabled,
                onCheckedChange = {
                    prefs.setFrameGraphEnabled(it)
                    refreshConfigState(prefs)
                },
            )
        }

        // ---- Trusted overlay (accessibility) ---------------------------------------
        LsfgCard {
            SectionHeader(eyebrow = stringResource(R.string.section_trusted_overlay), title = null)
            Spacer(Modifier.height(4.dp))
            ToggleRow(
                icon = Icons.Filled.TouchApp,
                title = stringResource(R.string.param_trusted_overlay),
                description = stringResource(R.string.param_trusted_overlay_desc),
                checked = state.trustedOverlay,
                onCheckedChange = {
                    prefs.setTrustedOverlay(it)
                    if (!it) prefs.setGestureForwardingEnabled(false)
                    refreshConfigState(prefs)
                },
            )
            // Gesture forwarding only makes sense as a companion to trusted overlay
            // (see LsfgAccessibilityService's class doc) — hide it until that's on,
            // and force it back off if the user later turns trusted overlay off, so
            // the preference can never be left dangling in an inconsistent state.
            if (state.trustedOverlay) {
                HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant)
                ToggleRow(
                    icon = Icons.Filled.TouchApp,
                    title = stringResource(R.string.param_gesture_forwarding),
                    description = stringResource(R.string.param_gesture_forwarding_desc),
                    checked = state.gestureForwardingEnabled,
                    onCheckedChange = {
                        prefs.setGestureForwardingEnabled(it)
                        refreshConfigState(prefs)
                    },
                )
            }
        }

        // ---- Capture mode ----------------------------------------------------------
        LsfgCard {
            SectionHeader(eyebrow = stringResource(R.string.section_capture_mode), title = null)
            Spacer(Modifier.height(4.dp))
            Text(
                text = stringResource(R.string.capture_mode_desc),
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            Spacer(Modifier.height(10.dp))
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                FilterChip(
                    selected = state.captureSource == CaptureSource.MEDIA_PROJECTION,
                    onClick = {
                        prefs.setCaptureSource(CaptureSource.MEDIA_PROJECTION)
                        refreshConfigState(prefs)
                    },
                    label = { Text(stringResource(R.string.capture_mode_mediaprojection)) },
                    modifier = Modifier.weight(1f),
                )
                FilterChip(
                    selected = state.captureSource == CaptureSource.SHIZUKU,
                    onClick = {
                        prefs.setCaptureSource(CaptureSource.SHIZUKU)
                        refreshConfigState(prefs)
                    },
                    label = { Text(stringResource(R.string.capture_mode_shizuku)) },
                    modifier = Modifier.weight(1f),
                )
                FilterChip(
                    selected = state.captureSource == CaptureSource.ROOT,
                    onClick = {
                        prefs.setCaptureSource(CaptureSource.ROOT)
                        refreshConfigState(prefs)
                    },
                    label = { Text(stringResource(R.string.capture_mode_root)) },
                    modifier = Modifier.weight(1f),
                )
            }
        }

        TailNote()
    }
}

@Composable
private fun TailNote() {
    Text(
        text = "Changes apply on the next session start. During an active session, open the in-game drawer to tweak values live.",
        style = MaterialTheme.typography.bodySmall,
        color = MaterialTheme.colorScheme.onSurfaceVariant,
        modifier = Modifier.fillMaxWidth(),
    )
}

@Composable
private fun OverlayModeSelector(
    selected: OverlayMode,
    onSelected: (OverlayMode) -> Unit,
) {
    var pendingDrawerConfirm by remember { mutableStateOf(false) }

    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 10.dp, horizontal = 4.dp),
    ) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            IconBadge(icon = Icons.Filled.TouchApp, size = 36.dp)
            Spacer(Modifier.size(12.dp))
            Column(modifier = Modifier.weight(1f)) {
                Text(
                    text = stringResource(R.string.param_overlay_mode),
                    style = MaterialTheme.typography.titleMedium,
                    color = MaterialTheme.colorScheme.onSurface,
                )
                Spacer(Modifier.height(2.dp))
                Text(
                    text = stringResource(R.string.param_overlay_mode_desc),
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        }
        Spacer(Modifier.height(10.dp))
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            FilterChip(
                selected = selected == OverlayMode.ICON_BUTTON,
                onClick = {
                    if (selected != OverlayMode.ICON_BUTTON) onSelected(OverlayMode.ICON_BUTTON)
                },
                label = { Text(stringResource(R.string.overlay_mode_icon_button)) },
                leadingIcon = {
                    Icon(
                        imageVector = Icons.Filled.TouchApp,
                        contentDescription = null,
                        modifier = Modifier.size(18.dp),
                    )
                },
                modifier = Modifier.weight(1f),
            )
            FilterChip(
                selected = selected == OverlayMode.DRAWER,
                onClick = {
                    if (selected != OverlayMode.DRAWER) {
                        pendingDrawerConfirm = true
                    }
                },
                label = { Text(stringResource(R.string.overlay_mode_drawer)) },
                leadingIcon = {
                    Icon(
                        imageVector = Icons.AutoMirrored.Filled.ViewSidebar,
                        contentDescription = null,
                        modifier = Modifier.size(18.dp),
                    )
                },
                modifier = Modifier.weight(1f),
            )
        }
    }

    if (pendingDrawerConfirm) {
        AlertDialog(
            onDismissRequest = { pendingDrawerConfirm = false },
            icon = {
                Icon(
                    imageVector = Icons.Filled.Warning,
                    contentDescription = null,
                    tint = MaterialTheme.colorScheme.error,
                )
            },
            title = { Text(stringResource(R.string.overlay_mode_drawer_warning_title)) },
            text = { Text(stringResource(R.string.overlay_mode_drawer_warning_body)) },
            confirmButton = {
                TextButton(onClick = {
                    pendingDrawerConfirm = false
                    onSelected(OverlayMode.DRAWER)
                }) { Text(stringResource(R.string.overlay_mode_drawer_warning_confirm)) }
            },
            dismissButton = {
                TextButton(onClick = { pendingDrawerConfirm = false }) {
                    Text(stringResource(R.string.overlay_mode_drawer_warning_cancel))
                }
            },
        )
    }
}

@Composable
private fun DrawerEdgeSelector(
    selected: DrawerEdge,
    onSelected: (DrawerEdge) -> Unit,
) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 10.dp, horizontal = 4.dp),
    ) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            IconBadge(icon = Icons.Filled.OpenInFull, size = 36.dp)
            Spacer(Modifier.size(12.dp))
            Column(modifier = Modifier.weight(1f)) {
                Text(
                    text = stringResource(R.string.param_drawer_edge),
                    style = MaterialTheme.typography.titleMedium,
                    color = MaterialTheme.colorScheme.onSurface,
                )
                Spacer(Modifier.height(2.dp))
                Text(
                    text = stringResource(R.string.param_drawer_edge_desc),
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        }
        Spacer(Modifier.height(10.dp))
        val options = listOf(
            DrawerEdge.LEFT to stringResource(R.string.drawer_edge_left),
            DrawerEdge.RIGHT to stringResource(R.string.drawer_edge_right),
            DrawerEdge.TOP to stringResource(R.string.drawer_edge_top),
            DrawerEdge.BOTTOM to stringResource(R.string.drawer_edge_bottom),
        )
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            options.take(2).forEach { (edge, label) ->
                FilterChip(
                    selected = selected == edge,
                    onClick = { onSelected(edge) },
                    label = { Text(label) },
                    modifier = Modifier.weight(1f),
                )
            }
        }
        Spacer(Modifier.height(6.dp))
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            options.drop(2).forEach { (edge, label) ->
                FilterChip(
                    selected = selected == edge,
                    onClick = { onSelected(edge) },
                    label = { Text(label) },
                    modifier = Modifier.weight(1f),
                )
            }
        }
    }
}
