package com.firstt175.deepdrop.ui

import android.net.Uri
import android.provider.OpenableColumns
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
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
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.Error
import androidx.compose.material.icons.filled.FileOpen
import androidx.compose.material.icons.automirrored.filled.InsertDriveFile
import androidx.compose.material.icons.filled.Psychology
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.FilterChip
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.unit.dp
import androidx.navigation.NavHostController
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import com.firstt175.deepdrop.R
import com.firstt175.deepdrop.prefs.AiEngine
import com.firstt175.deepdrop.prefs.FramegenBackend
import com.firstt175.deepdrop.prefs.RifeModel
import com.firstt175.deepdrop.prefs.IfrnetModel
import com.firstt175.deepdrop.prefs.LsfgPreferences
import com.firstt175.deepdrop.session.BundledIfrnetModel
import com.firstt175.deepdrop.session.BundledRifeModel
import com.firstt175.deepdrop.session.ExtractResult
import com.firstt175.deepdrop.session.NativeBridge
import com.firstt175.deepdrop.session.ShaderExtractor
import com.firstt175.deepdrop.ui.components.IconBadge
import com.firstt175.deepdrop.ui.components.LsfgCard
import com.firstt175.deepdrop.ui.components.LsfgSecondaryButton
import com.firstt175.deepdrop.ui.components.LsfgTopBar
import com.firstt175.deepdrop.ui.components.SectionHeader
import com.firstt175.deepdrop.ui.theme.LsfgPrimary
import com.firstt175.deepdrop.ui.theme.LsfgStatusGood
import com.firstt175.deepdrop.ui.theme.LsfgStatusWarn

private sealed class ExtractionState {
    data object Idle : ExtractionState()
    data object Running : ExtractionState()
    data class Done(val success: Boolean, val message: String?) : ExtractionState()
}

private sealed class ImportState {
    data object Idle : ImportState()
    data object Running : ImportState()
    data class Done(val success: Boolean, val message: String?) : ImportState()
}

/** Mirrors lsfg_android::kNcnnErr* in NcnnInterpolator.hpp. */
private fun describeNcnnError(code: Int, engine: AiEngine = AiEngine.RIFE): String = when (code) {
    0 -> "ok"
    -1 -> "This build doesn't include ncnn (see the LSFG_HAVE_NCNN block in CMakeLists.txt — " +
        "download the ncnn Android Vulkan SDK and rebuild)."
    -2 -> "Bundled model files are missing from storage. Try \"Test model load\" again — " +
        "if it keeps failing, the app install may be corrupt."
    -3 -> if (engine == AiEngine.IFRNET) {
        "ncnn rejected ifrnet.param/ifrnet.bin, or the ifrnet.Warp custom layer failed to " +
            "register — check logcat tag lsfg-ncnn."
    } else {
        "ncnn rejected flownet.param/flownet.bin, or the rife.Warp custom layer failed to " +
            "register — check logcat tag lsfg-ncnn."
    }
    -4 -> "Model isn't loaded."
    -5 -> "Invalid arguments passed to the native interpolator."
    else -> "Unknown ncnn error ($code)"
}

@Composable
fun DllPickerScreen(nav: NavHostController) {
    val ctx = LocalContext.current
    val prefs = remember { LsfgPreferences(ctx) }
    val state by produceConfigState(prefs).collectAsState()

    var pickError by remember { mutableStateOf<String?>(null) }
    var extractionState by remember { mutableStateOf<ExtractionState>(ExtractionState.Idle) }
    var pendingUri by remember { mutableStateOf<Uri?>(null) }

    var importState by remember { mutableStateOf<ImportState>(ImportState.Idle) }
    // Bumped by the "Test model load" button to force the LaunchedEffect below to
    // re-run even when none of its other keys (backend/useVulkan/cpuThreads) changed.
    var testLoadNonce by remember { mutableStateOf(0) }

    // Frees whichever test-loaded ncnn interpolator (RIFE or IFRNet) is
    // currently resident once the user leaves this screen. Without this,
    // a model loaded here purely to show "does it load on this device"
    // stayed loaded (GPU buffers and all) for the rest of the process's
    // life, even after the settings screen was gone. See the LaunchedEffect
    // below for the switch-time half of this fix.
    DisposableEffect(Unit) {
        onDispose { NativeBridge.releaseAiInterpolator() }
    }

    val picker = rememberLauncherForActivityResult(
        ActivityResultContracts.OpenDocument(),
    ) { uri: Uri? ->
        if (uri == null) return@rememberLauncherForActivityResult
        val resolver = ctx.contentResolver
        runCatching {
            resolver.takePersistableUriPermission(
                uri,
                android.content.Intent.FLAG_GRANT_READ_URI_PERMISSION,
            )
        }
        val name = resolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)?.use {
            if (it.moveToFirst()) it.getString(0) else null
        } ?: uri.lastPathSegment ?: "Lossless.dll"
        if (!name.equals("Lossless.dll", ignoreCase = true)) {
            pickError = "Selected file is \"$name\", expected \"Lossless.dll\". Pick the correct file."
            return@rememberLauncherForActivityResult
        }
        pickError = null
        prefs.setDll(uri.toString(), name)
        refreshConfigState(prefs)
        pendingUri = uri
        extractionState = ExtractionState.Running
    }

    LaunchedEffect(pendingUri, extractionState) {
        val uri = pendingUri
        if (uri != null && extractionState is ExtractionState.Running) {
            val result = withContext(Dispatchers.IO) { ShaderExtractor.extract(ctx, uri) }
            when (result) {
                is ExtractResult.Success -> {
                    prefs.setShadersReady(true)
                    refreshConfigState(prefs)
                    extractionState = ExtractionState.Done(success = true, message = null)
                }
                is ExtractResult.Failure -> {
                    prefs.setShadersReady(false)
                    refreshConfigState(prefs)
                    extractionState = ExtractionState.Done(success = false, message = result.message)
                }
            }
            pendingUri = null
        }
    }

    // Both engines' models ship as bundled APK assets (see BundledRifeModel /
    // BundledIfrnetModel) — there's nothing to pick anymore. This
    // LaunchedEffect just extracts-if-needed and does a real
    // NativeBridge.initAiInterpolator() call so the status card reflects
    // whether the model actually loads on this device, not just whether the
    // asset exists. Re-runs whenever the compute settings OR the selected
    // engine change, since both affect whether load succeeds, and once on
    // first entering this backend's settings. rifeUseVulkan/ifrnetUseVulkan
    // are no longer user-editable (see the COMPUTE card below — GPU/CPU is
    // now always hybrid, not a toggle) but stay in this key list since
    // they're still the allowGpu flag threaded through to
    // NativeBridge.initAiInterpolator() below.
    LaunchedEffect(
        state.framegenBackend, state.aiEngine, state.rifeModel,
        testLoadNonce,
    ) {
        if (state.framegenBackend != FramegenBackend.NCNN_AI) return@LaunchedEffect
        val engine = state.aiEngine
        importState = ImportState.Running
        val (success, message) = withContext(Dispatchers.IO) {
            // Free whatever the previous run of this effect loaded before
            // loading the new one. Without this, switching RIFE <-> IFRNet
            // in this screen left BOTH engines' ncnn::Net instances resident
            // in native memory (GPU weight buffers included) indefinitely —
            // NativeBridge.releaseAiInterpolator() existed but nothing ever
            // called it, so every engine switch only ever added memory,
            // never freed it. Re-testing the SAME engine again was already
            // safe (NcnnInterpolator::load()/IfrnetInterpolator::load() both
            // call unload() internally first) — this closes the gap for the
            // *other* engine's leftover instance.
            NativeBridge.releaseAiInterpolator()
            val modelDir = when (engine) {
                AiEngine.IFRNET -> {
                    if (!BundledIfrnetModel.ensureExtracted(ctx, state.ifrnetModel)) {
                        return@withContext false to "Couldn't extract the bundled model from the APK's assets."
                    }
                    BundledIfrnetModel.modelDir(ctx, state.ifrnetModel).absolutePath
                }
                AiEngine.RIFE -> {
                    if (!BundledRifeModel.ensureExtracted(ctx, state.rifeModel)) {
                        return@withContext false to "Couldn't extract the bundled model from the APK's assets."
                    }
                    BundledRifeModel.modelDir(ctx, state.rifeModel).absolutePath
                }
            }
            val loadCode = NativeBridge.initAiInterpolator(
                modelDir,
                true,
                -1,
                // GPU-only AI. CPU thread tuning is intentionally ignored.
                1,
                engine.nativeValue,
            )
            if (loadCode == 0) {
                // Both engines run with the same ncnn Option (use_fp16_packed/
                // storage/arithmetic = true — see IfrnetInterpolator.cpp /
                // NcnnInterpolator.cpp's load()), so both report fp16 here.
                true to (if (engine == AiEngine.IFRNET) "ifrnet (fp16)" else "flownet (fp16)")
            } else {
                false to describeNcnnError(loadCode, engine)
            }
        }
        prefs.setAiModelReady(
            success,
            if (success) "fp16" else null,
            if (success) listOf(if (engine == AiEngine.IFRNET) "ifrnet" else "flownet") else emptyList(),
        )
        refreshConfigState(prefs)
        importState = ImportState.Done(success = success, message = message)
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
            title = stringResource(R.string.nav_dll),
            onBack = { nav.popBackStack() },
        )

        LsfgCard {
            SectionHeader(eyebrow = stringResource(R.string.section_framegen_backend), title = null)
            Spacer(Modifier.height(4.dp))
            Text(
                text = stringResource(R.string.backend_desc),
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            Spacer(Modifier.height(10.dp))
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                FilterChip(
                    selected = state.framegenBackend == FramegenBackend.LSFG_DLL,
                    onClick = {
                        prefs.setFramegenBackend(FramegenBackend.LSFG_DLL)
                        refreshConfigState(prefs)
                    },
                    label = { Text(stringResource(R.string.backend_lsfg_dll)) },
                    modifier = Modifier.weight(1f),
                )
                FilterChip(
                    selected = state.framegenBackend == FramegenBackend.NCNN_AI,
                    onClick = {
                        prefs.setFramegenBackend(FramegenBackend.NCNN_AI)
                        refreshConfigState(prefs)
                    },
                    label = { Text(stringResource(R.string.backend_ncnn_ai)) },
                    modifier = Modifier.weight(1f),
                )
            }
        }

        // Only the settings for the currently selected backend are shown below —
        // the LSFG_DLL card (Lossless.dll / shader extraction) and the NCNN_AI
        // cards (model bundle + compute path) used to render together
        // regardless of which pipeline was active, which made it impossible to
        // tell which values were actually in effect. Whichever backend isn't
        // selected keeps its saved state untouched; it's just not shown.
        if (state.framegenBackend == FramegenBackend.LSFG_DLL) {

        val statusIcon: ImageVector
        val statusTint = when {
            extractionState is ExtractionState.Done && !(extractionState as ExtractionState.Done).success -> {
                statusIcon = Icons.Filled.Error
                MaterialTheme.colorScheme.error
            }
            state.shadersReady -> {
                statusIcon = Icons.Filled.CheckCircle
                LsfgStatusGood
            }
            state.dllDisplayName != null -> {
                statusIcon = Icons.AutoMirrored.Filled.InsertDriveFile
                LsfgStatusWarn
            }
            else -> {
                statusIcon = Icons.AutoMirrored.Filled.InsertDriveFile
                MaterialTheme.colorScheme.onSurfaceVariant
            }
        }

        LsfgCard(accent = state.shadersReady) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                IconBadge(icon = statusIcon, tint = statusTint, size = 48.dp)
                Spacer(Modifier.size(14.dp))
                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        text = state.dllDisplayName ?: "No file selected",
                        style = MaterialTheme.typography.titleMedium,
                        color = MaterialTheme.colorScheme.onSurface,
                    )
                    Spacer(Modifier.height(2.dp))
                    Text(
                        text = when {
                            state.shadersReady -> "Shaders extracted and cached."
                            state.dllDisplayName != null -> "DLL selected. Shaders not extracted yet."
                            else -> stringResource(R.string.dll_status_none)
                        },
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
            }

            if (extractionState is ExtractionState.Running) {
                Spacer(Modifier.height(16.dp))
                Text(
                    text = "Extracting and translating shaders…",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
                Spacer(Modifier.height(8.dp))
                LinearProgressIndicator(
                    color = LsfgPrimary,
                    trackColor = MaterialTheme.colorScheme.surfaceContainerHighest,
                    modifier = Modifier.fillMaxWidth(),
                )
            }

            val s = extractionState
            if (s is ExtractionState.Done) {
                Spacer(Modifier.height(12.dp))
                Text(
                    text = if (s.success) "Extraction succeeded. SPIR-V cached."
                    else "Extraction failed: ${s.message}",
                    color = if (s.success) LsfgStatusGood else MaterialTheme.colorScheme.error,
                    style = MaterialTheme.typography.bodySmall,
                )
            }

            if (pickError != null) {
                Spacer(Modifier.height(8.dp))
                Text(
                    text = pickError!!,
                    color = MaterialTheme.colorScheme.error,
                    style = MaterialTheme.typography.bodySmall,
                )
            }
        }

        LsfgCard {
            Text(
                text = "SOURCE",
                style = MaterialTheme.typography.labelSmall,
                color = LsfgPrimary,
            )
            Spacer(Modifier.height(8.dp))
            Text(
                text = "Pick Lossless.dll from your own legally purchased copy of Lossless Scaling on Steam.",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            Spacer(Modifier.height(16.dp))
            Button(
                onClick = { picker.launch(arrayOf("*/*")) },
                enabled = extractionState !is ExtractionState.Running,
                shape = MaterialTheme.shapes.small,
                colors = ButtonDefaults.buttonColors(
                    containerColor = LsfgPrimary,
                    contentColor = MaterialTheme.colorScheme.onPrimary,
                ),
                modifier = Modifier.fillMaxWidth(),
            ) {
                androidx.compose.material3.Icon(
                    Icons.Filled.FileOpen,
                    contentDescription = null,
                    modifier = Modifier.size(18.dp),
                )
                Spacer(Modifier.size(8.dp))
                Text(stringResource(R.string.dll_pick_button))
            }
            Spacer(Modifier.height(8.dp))
            LsfgSecondaryButton(
                text = stringResource(R.string.dll_reextract_button),
                onClick = {
                    val uri = state.dllUri?.let(Uri::parse) ?: return@LsfgSecondaryButton
                    prefs.setShadersReady(false)
                    refreshConfigState(prefs)
                    pendingUri = uri
                    extractionState = ExtractionState.Running
                },
                enabled = state.dllUri != null && extractionState !is ExtractionState.Running,
                leadingIcon = Icons.Filled.Refresh,
                modifier = Modifier.fillMaxWidth(),
            )
        }

        } // framegenBackend == LSFG_DLL

        if (state.framegenBackend == FramegenBackend.NCNN_AI) {

        val aiStatusIcon: ImageVector
        val aiStatusTint = when {
            importState is ImportState.Done && !(importState as ImportState.Done).success -> {
                aiStatusIcon = Icons.Filled.Error
                MaterialTheme.colorScheme.error
            }
            state.aiModelReady -> {
                aiStatusIcon = Icons.Filled.CheckCircle
                LsfgStatusGood
            }
            else -> {
                aiStatusIcon = Icons.Filled.Psychology
                MaterialTheme.colorScheme.onSurfaceVariant
            }
        }

        LsfgCard(accent = state.aiModelReady) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                IconBadge(icon = aiStatusIcon, tint = aiStatusTint, size = 48.dp)
                Spacer(Modifier.size(14.dp))
                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        text = if (state.aiEngine == AiEngine.IFRNET) {
                            "${state.ifrnetModel.label} (${state.ifrnetModel.sizeMb} MB)"
                        } else {
                            "${state.rifeModel.label} (${state.rifeModel.sizeMb} MB)"
                        },
                        style = MaterialTheme.typography.titleMedium,
                        color = MaterialTheme.colorScheme.onSurface,
                    )
                    Spacer(Modifier.height(2.dp))
                    Text(
                        text = when {
                            state.aiModelReady -> stringResource(
                                R.string.ai_model_status_ready,
                                state.aiModelGraphs.joinToString(", "),
                                state.aiModelPrecision ?: "fp16",
                            )
                            importState is ImportState.Done -> stringResource(
                                R.string.ai_model_status_failed,
                                (importState as ImportState.Done).message ?: "unknown error",
                            )
                            else -> stringResource(R.string.ai_model_status_none)
                        },
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
            }

            if (importState is ImportState.Running) {
                Spacer(Modifier.height(16.dp))
                Text(
                    text = "Loading bundled model…",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
                Spacer(Modifier.height(8.dp))
                LinearProgressIndicator(
                    color = LsfgPrimary,
                    trackColor = MaterialTheme.colorScheme.surfaceContainerHighest,
                    modifier = Modifier.fillMaxWidth(),
                )
            }
        }

        LsfgCard {
            Text(
                text = "ENGINE",
                style = MaterialTheme.typography.labelSmall,
                color = LsfgPrimary,
            )
            Spacer(Modifier.height(8.dp))
            Text(
                text = "Which bundled ncnn model runs the interpolation. Both are single-pass " +
                    "networks with the same call shape — switching re-tests the load " +
                    "automatically below.",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            Spacer(Modifier.height(10.dp))
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                FilterChip(
                    selected = state.aiEngine == AiEngine.RIFE,
                    onClick = {
                        prefs.setAiEngine(AiEngine.RIFE)
                        refreshConfigState(prefs)
                    },
                    label = { Text("RIFE") },
                    modifier = Modifier.weight(1f),
                )
                FilterChip(
                    selected = state.aiEngine == AiEngine.IFRNET,
                    onClick = {
                        prefs.setAiEngine(AiEngine.IFRNET)
                        refreshConfigState(prefs)
                    },
                    label = { Text("IFRNet") },
                    modifier = Modifier.weight(1f),
                )
            }

            if (state.aiEngine == AiEngine.RIFE) {
                Spacer(Modifier.height(10.dp))
                Text(
                    text = "RIFE MODEL — lightweight only",
                    style = MaterialTheme.typography.labelSmall,
                    color = LsfgPrimary,
                )
                Spacer(Modifier.height(8.dp))
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    RifeModel.values().forEach { model ->
                        FilterChip(
                            selected = state.rifeModel == model,
                            onClick = {
                                prefs.setAiEngine(AiEngine.RIFE)
                                prefs.setRifeModel(model)
                                refreshConfigState(prefs)
                            },
                            label = { Text("${model.label}\n${model.sizeMb} MB") },
                            modifier = Modifier.weight(1f),
                        )
                    }
                }
                Spacer(Modifier.height(4.dp))
                Text(
                    text = "Only the small Vulkan RIFE graphs are included. Larger HD/UHD/anime models are intentionally omitted.",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
            if (state.aiEngine == AiEngine.IFRNET) {
                Spacer(Modifier.height(10.dp))
                Text(
                    text = "IFRNet MODEL — lightweight only",
                    style = MaterialTheme.typography.labelSmall,
                    color = LsfgPrimary,
                )
                Spacer(Modifier.height(8.dp))
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    IfrnetModel.values().forEach { model ->
                        FilterChip(
                            selected = state.ifrnetModel == model,
                            onClick = {
                                prefs.setAiEngine(AiEngine.IFRNET)
                                prefs.setIfrnetModel(model)
                                refreshConfigState(prefs)
                            },
                            label = { Text("${model.label}\n${model.sizeMb} MB") },
                            modifier = Modifier.weight(1f),
                        )
                    }
                }
                Spacer(Modifier.height(4.dp))
                Text(
                    text = "Only IFRNet_S models are included. The larger IFRNet_L and full models are intentionally omitted.",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        }

        LsfgCard {
            Text(
                text = "AI MODEL (NCNN)",
                style = MaterialTheme.typography.labelSmall,
                color = LsfgPrimary,
            )
            Spacer(Modifier.height(8.dp))
            Text(
                text = stringResource(R.string.ai_model_source_desc),
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            Spacer(Modifier.height(12.dp))
            LsfgSecondaryButton(
                text = stringResource(R.string.ai_model_test_load_button),
                onClick = { testLoadNonce++ },
                enabled = importState !is ImportState.Running,
                leadingIcon = Icons.Filled.Refresh,
                modifier = Modifier.fillMaxWidth(),
            )
        }

        LsfgCard {
            Text(
                text = "COMPUTE — ${if (state.aiEngine == AiEngine.IFRNET) "IFRNET" else "RIFE"}",
                style = MaterialTheme.typography.labelSmall,
                color = LsfgPrimary,
            )
            Spacer(Modifier.height(8.dp))
            Text(
                text = "The ncnn interpolator is Vulkan-GPU-only on this build — there's no " +
                    "NPU or CPU compute path to pick, so nothing here is user-editable anymore. " +
                    "This has no effect on the Lossless.dll pipeline above, which always runs " +
                    "on the GPU too. RIFE and IFRNet load onto the GPU independently, so " +
                    "switching the engine above doesn't change the status below for the other one.",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            Spacer(Modifier.height(12.dp))

            val activeEngineLabel = if (state.aiEngine == AiEngine.IFRNET) "IFRNet" else "RIFE"
            val vulkanGpuCount = remember { runCatching { NativeBridge.getVulkanGpuCount() }.getOrDefault(0) }
            val vulkanGpuName = remember(vulkanGpuCount) {
                if (vulkanGpuCount > 0) {
                    runCatching { NativeBridge.getVulkanGpuName(0) }.getOrDefault(null)?.takeIf { it.isNotBlank() }
                } else {
                    null
                }
            }
            // "Actually used" = the live test-load below succeeded for this engine, not just
            // that a GPU exists — a device can have a Vulkan GPU that still rejects the model.
            val gpuActuallyUsed = state.aiModelReady

            Row(verticalAlignment = Alignment.CenterVertically) {
                IconBadge(
                    icon = if (gpuActuallyUsed) Icons.Filled.CheckCircle else Icons.Filled.Error,
                    tint = if (gpuActuallyUsed) {
                        LsfgStatusGood
                    } else if (vulkanGpuCount > 0) {
                        LsfgStatusWarn
                    } else {
                        MaterialTheme.colorScheme.error
                    },
                    size = 40.dp,
                )
                Spacer(Modifier.size(12.dp))
                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        text = vulkanGpuName
                            ?: if (vulkanGpuCount > 0) "Vulkan GPU (name unavailable)" else "No Vulkan GPU detected",
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurface,
                    )
                    Spacer(Modifier.height(2.dp))
                    Text(
                        text = when {
                            gpuActuallyUsed -> "In use — $activeEngineLabel is running on this GPU right now."
                            vulkanGpuCount > 0 -> "Not in use — GPU detected but $activeEngineLabel hasn't " +
                                "loaded onto it (see the model status card above)."
                            else -> "Not in use — $activeEngineLabel has nowhere to run without a Vulkan GPU."
                        },
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
            }
        }

        } // framegenBackend == NCNN_AI

        DeviceInfoCard()
    }
}

/**
 * GPU-only device card. Deliberately contains no CPU telemetry or polling:
 * reading /proc/stat/sysfs is removed from the runtime path.
 */
@Composable
private fun DeviceInfoCard() {
    val vulkanApiVersion = remember {
        runCatching { NativeBridge.getVulkanApiVersion() }.getOrDefault("unknown")
    }
    val gpuVendor = remember { runCatching { NativeBridge.getGpuVendor() }.getOrDefault("unknown") }
    val gpuDeviceType = remember { runCatching { NativeBridge.getGpuDeviceType() }.getOrDefault("unknown") }
    val gpuDriverVersion = remember { runCatching { NativeBridge.getGpuDriverVersion() }.getOrDefault("unknown") }
    val gpuVramMb = remember { runCatching { NativeBridge.getGpuVramMb() }.getOrDefault(-1L) }
    val gpuName = remember {
        runCatching { NativeBridge.getVulkanGpuName(0) }.getOrDefault(null)?.takeIf { it.isNotBlank() }
    }

    LsfgCard {
        Text(text = "DEVICE / GPU", style = MaterialTheme.typography.labelSmall, color = LsfgPrimary)
        Spacer(Modifier.height(8.dp))
        Text(
            text = "Android ${android.os.Build.VERSION.RELEASE} (API ${android.os.Build.VERSION.SDK_INT}) · Vulkan $vulkanApiVersion",
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurface,
        )
        Spacer(Modifier.height(8.dp))
        Text(
            text = "GPU: ${gpuName ?: "unknown"} ($gpuDeviceType, $gpuVendor)",
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurface,
        )
        Spacer(Modifier.height(2.dp))
        Text(
            text = "Driver $gpuDriverVersion · GPU memory ~" +
                (if (gpuVramMb >= 0) "$gpuVramMb MiB" else "unknown"),
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        Spacer(Modifier.height(8.dp))
        Text(
            text = "AI compute is hard-locked to Vulkan GPU. CPU telemetry, CPU affinity and CPU inference paths are not part of this build.",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
    }
}
