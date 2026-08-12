package com.firstt175.deepdrop.ui.components

// ── StatMeter ─────────────────────────────────────────────────────────────────
// Ported from DeepDrop, adapted to the LLS palette.
// Provides animated horizontal stat bars, big monospaced FPS numbers, and a
// flat status-dot indicator for the in-session HUD.

import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.size
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.firstt175.deepdrop.ui.theme.LsfgPrimary
import com.firstt175.deepdrop.ui.theme.LsfgStatusGood

/**
 * Animated horizontal bar meter with a flat solid fill.
 *
 * Use for FPS, latency, pacing, or any 0–1 ratio metric.
 *
 * @param value       Normalised progress in 0..1.
 * @param displayText Formatted value shown at the right.
 * @param accentColor Bar fill colour — default orange (LsfgPrimary).
 */
@Composable
fun StatBar(
    label: String,
    value: Float,
    displayText: String,
    modifier: Modifier = Modifier,
    accentColor: Color = LsfgPrimary,
    height: Dp = 4.dp,
) {
    val animValue by animateFloatAsState(
        targetValue  = value.coerceIn(0f, 1f),
        animationSpec = tween(400),
        label        = "statBar",
    )
    val trackColor = MaterialTheme.colorScheme.outlineVariant

    Column(modifier = modifier) {
        Row(
            modifier              = Modifier.fillMaxWidth(),
            horizontalArrangement = androidx.compose.foundation.layout.Arrangement.SpaceBetween,
        ) {
            Text(
                text  = label,
                style = MaterialTheme.typography.labelMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            Text(
                text         = displayText,
                fontFamily   = FontFamily.Monospace,
                fontWeight   = FontWeight.Bold,
                fontSize     = 11.sp,
                color        = accentColor,
            )
        }
        Spacer(Modifier.height(4.dp))
        Canvas(
            modifier = Modifier
                .fillMaxWidth()
                .height(height),
        ) {
            val y = size.height / 2
            // Track
            drawLine(
                color       = trackColor,
                start       = Offset(0f, y),
                end         = Offset(size.width, y),
                strokeWidth = size.height,
                cap         = StrokeCap.Round,
            )
            // Fill
            if (animValue > 0f) {
                drawLine(
                    color       = accentColor,
                    start       = Offset(0f, y),
                    end         = Offset(size.width * animValue, y),
                    strokeWidth = size.height,
                    cap         = StrokeCap.Round,
                )
            }
        }
    }
}

/**
 * Large monospaced FPS / latency number with a label above and unit below.
 * Centrepiece stat widget for the session panel.
 */
@Composable
fun BigStatNumber(
    label: String,
    value: String,
    unit: String,
    modifier: Modifier = Modifier,
    valueColor: Color = MaterialTheme.colorScheme.onSurface,
) {
    Column(
        modifier           = modifier,
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Text(
            text  = label.uppercase(),
            style = MaterialTheme.typography.labelSmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        Spacer(Modifier.height(2.dp))
        Text(
            text          = value,
            fontFamily    = FontFamily.Monospace,
            fontWeight    = FontWeight.Black,
            fontSize      = 38.sp,
            color         = valueColor,
            letterSpacing  = (-1).sp,
        )
        Text(
            text  = unit,
            style = MaterialTheme.typography.labelMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
    }
}

/**
 * Flat status-dot indicator.
 *
 * When [active] is true the dot is [LsfgStatusGood] green; when false it uses
 * the muted surface variant colour, matching the RedMagic "ACTIVE / STANDBY"
 * pattern from DeepDrop.
 */
@Composable
fun StatusDot(
    active: Boolean,
    modifier: Modifier = Modifier,
    size: Dp = 8.dp,
    activeColor: Color = LsfgStatusGood,
) {
    val color = if (active) activeColor else MaterialTheme.colorScheme.onSurfaceVariant
    Canvas(modifier = modifier.size(size)) {
        drawCircle(color = color, radius = this.size.width * 0.45f)
    }
}
