package com.firstt175.deepdrop.ui

import android.content.Intent
import android.net.Uri
import android.os.Build
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
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
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ChevronRight
import androidx.compose.material.icons.filled.Code
import androidx.compose.material.icons.filled.Gavel
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.unit.dp
import androidx.navigation.NavHostController
import coil.ImageLoader
import coil.compose.AsyncImage
import coil.decode.GifDecoder
import coil.decode.ImageDecoderDecoder
import coil.request.ImageRequest
import com.firstt175.deepdrop.BuildConfig
import com.firstt175.deepdrop.R
import com.firstt175.deepdrop.ui.components.IconBadge
import com.firstt175.deepdrop.ui.components.LsfgCard
import com.firstt175.deepdrop.ui.components.LsfgTopBar
import com.firstt175.deepdrop.ui.components.SectionHeader
import com.firstt175.deepdrop.ui.theme.LsfgPrimary

private const val YOUTUBE_URL = "https://youtu.be/ENVY8OXfzis?si=1Dw6sHZQASr_i7ri"
private const val YOUTUBE_LOGO_ASSET = "file:///android_asset/credits/youtube_logo.gif"
private const val BASE_APK_URL = "https://github.com/FrankBarretta/LSFG-Android"
private const val MY_PROJECT_URL = "https://github.com/K1z3m1/LSFG-Android-"
private const val LICENSE_URL = "https://www.gnu.org/licenses/gpl-3.0.html"
private const val NCNN_AUTHOR_URL = "https://github.com/nihui"
private const val PE_PARSE_URL = "https://github.com/trailofbits/pe-parse/tree/31ac5966503689d5693cd9fb520bd525a8710e17"
private const val VOLK_URL = "https://github.com/zeux/volk/tree/be3dbd49bf77052665e96b6c7484af855e7e5f67"

@Composable
fun CreditsScreen(nav: NavHostController) {
    val ctx = LocalContext.current

    fun openUrl(url: String) {
        ctx.startActivity(Intent(Intent.ACTION_VIEW, Uri.parse(url)))
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .statusBarsPadding()
            .padding(horizontal = 20.dp)
            .padding(bottom = 20.dp),
    ) {
        LsfgTopBar(
            title = stringResource(R.string.credits_title),
            onBack = { nav.popBackStack() },
        )

        Column(
            modifier = Modifier
                .fillMaxWidth()
                .verticalScroll(rememberScrollState()),
            verticalArrangement = Arrangement.spacedBy(16.dp),
        ) {
            Spacer(Modifier.height(4.dp))

            Column(horizontalAlignment = Alignment.CenterHorizontally, modifier = Modifier.fillMaxWidth()) {
                Image(
                    painter = painterResource(R.drawable.lsfg_app_icon),
                    contentDescription = null,
                    contentScale = ContentScale.Crop,
                    modifier = Modifier
                        .size(56.dp)
                        .clip(RoundedCornerShape(14.dp)),
                )
                Spacer(Modifier.height(8.dp))
                Text(
                    text = stringResource(R.string.app_name),
                    style = MaterialTheme.typography.titleLarge,
                    color = MaterialTheme.colorScheme.onSurface,
                )
                Text(
                    text = "v${BuildConfig.VERSION_NAME}",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }

            // YouTube — shown first, above the base-apk credit.
            SectionHeader(eyebrow = stringResource(R.string.credits_section_channel))
            CreditRow(
                gifUrl = YOUTUBE_LOGO_ASSET,
                title = "First",
                subtitle = "@firstT175 — YouTube",
                onClick = { openUrl(YOUTUBE_URL) },
            )

            // Base APK credit — shown below the YouTube entry.
            SectionHeader(eyebrow = stringResource(R.string.credits_section_base))
            CreditRow(
                imageRes = R.drawable.base_apk_logo,
                title = "LSFG-Android",
                subtitle = "by FrankBarretta — original base project",
                onClick = { openUrl(BASE_APK_URL) },
            )

            // This fork.
            SectionHeader(eyebrow = stringResource(R.string.credits_section_myproject))
            CreditRow(
                icon = Icons.Filled.Code,
                title = "LSFG-Android+",
                subtitle = "K1z3m1 — this fork on GitHub",
                onClick = { openUrl(MY_PROJECT_URL) },
            )

            // Vendored native dependencies (see framegen-native-lib/thirdparty/).
            SectionHeader(eyebrow = stringResource(R.string.credits_section_libraries))
            CreditRow(
                icon = Icons.Filled.Code,
                title = "ncnn",
                subtitle = "nihui — neural network inference framework",
                onClick = { openUrl(NCNN_AUTHOR_URL) },
            )
            CreditRow(
                icon = Icons.Filled.Code,
                title = "pe-parse",
                subtitle = "Trail of Bits — PE/DLL parsing library",
                onClick = { openUrl(PE_PARSE_URL) },
            )
            CreditRow(
                icon = Icons.Filled.Code,
                title = "volk",
                subtitle = "Arseny Kapoulkine (zeux) — Vulkan meta-loader",
                onClick = { openUrl(VOLK_URL) },
            )

            // License, folded into the same page.
            SectionHeader(eyebrow = stringResource(R.string.credits_section_license))
            LsfgCard(onClick = { openUrl(LICENSE_URL) }) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Icon(Icons.Filled.Gavel, contentDescription = null, tint = LsfgPrimary)
                    Spacer(Modifier.width(12.dp))
                    Column(Modifier.weight(1f)) {
                        Text(
                            text = "GNU GPL v3.0",
                            style = MaterialTheme.typography.bodyMedium,
                            color = MaterialTheme.colorScheme.onSurface,
                        )
                        Text(
                            text = stringResource(R.string.credits_license_body),
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    }
                }
            }

            Spacer(Modifier.height(8.dp))
        }
    }
}

/**
 * Credit row. Pass exactly one of [icon] (vector glyph), [imageRes] (static bitmap), or
 * [gifUrl] (animated GIF, played via Coil).
 */
@Composable
private fun CreditRow(
    title: String,
    subtitle: String,
    onClick: () -> Unit,
    icon: androidx.compose.ui.graphics.vector.ImageVector? = null,
    imageRes: Int? = null,
    gifUrl: String? = null,
) {
    val ctx = LocalContext.current
    val gifLoader = remember(gifUrl) {
        if (gifUrl == null) return@remember null
        ImageLoader.Builder(ctx)
            .components {
                if (Build.VERSION.SDK_INT >= 28) {
                    add(ImageDecoderDecoder.Factory())
                } else {
                    add(GifDecoder.Factory())
                }
            }
            .build()
    }

    LsfgCard(onClick = onClick) {
        Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.fillMaxWidth()) {
            when {
                gifUrl != null && gifLoader != null -> {
                    AsyncImage(
                        model = ImageRequest.Builder(ctx).data(gifUrl).build(),
                        imageLoader = gifLoader,
                        contentDescription = null,
                        contentScale = ContentScale.Crop,
                        modifier = Modifier
                            .size(40.dp)
                            .clip(CircleShape)
                            .background(MaterialTheme.colorScheme.surfaceContainerHigh),
                    )
                }
                imageRes != null -> {
                    Image(
                        painter = painterResource(imageRes),
                        contentDescription = null,
                        contentScale = ContentScale.Crop,
                        modifier = Modifier
                            .size(40.dp)
                            .clip(CircleShape)
                            .background(MaterialTheme.colorScheme.surfaceContainerHigh),
                    )
                }
                icon != null -> {
                    IconBadge(icon = icon, size = 40.dp)
                }
            }
            Spacer(Modifier.width(12.dp))
            Column(Modifier.weight(1f)) {
                Text(
                    text = title,
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurface,
                )
                Text(
                    text = subtitle,
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
            Icon(
                Icons.Filled.ChevronRight,
                contentDescription = null,
                tint = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }
}
