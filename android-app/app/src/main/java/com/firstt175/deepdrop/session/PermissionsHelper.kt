package com.firstt175.deepdrop.session

import android.content.Context
import android.content.Intent
import android.net.Uri
import android.os.PowerManager
import android.provider.Settings
import android.text.TextUtils

/** Utilities to query the permission/accessibility-service state the session needs. */
object PermissionsHelper {

    fun canDrawOverlays(ctx: Context): Boolean = Settings.canDrawOverlays(ctx)

    /**
     * True once the OS has exempted this app from Doze/App Standby battery
     * optimizations — the same exemption system apps get by default. Needed
     * here because the foreground capture/frame-gen session must keep running
     * at full scheduling priority even while the screen is off or the app is
     * backgrounded; without it Doze can throttle or suspend the session.
     */
    fun isIgnoringBatteryOptimizations(ctx: Context): Boolean {
        val pm = ctx.getSystemService(Context.POWER_SERVICE) as PowerManager
        return pm.isIgnoringBatteryOptimizations(ctx.packageName)
    }

    /**
     * Builds the system dialog that lets the user grant the exemption above.
     * This is the standard, user-visible Android API for it (the app's real
     * identity and permission request are shown to the user by the OS) —
     * there is no way to silently acquire system-app battery treatment.
     */
    fun buildIgnoreBatteryOptimizationsIntent(ctx: Context): Intent =
        Intent(
            Settings.ACTION_REQUEST_IGNORE_BATTERY_OPTIMIZATIONS,
            Uri.parse("package:${ctx.packageName}"),
        )

    fun isAccessibilityServiceEnabled(ctx: Context): Boolean {
        val expected = "${ctx.packageName}/${LsfgAccessibilityService::class.java.name}"
        val setting = Settings.Secure.getString(
            ctx.contentResolver,
            Settings.Secure.ENABLED_ACCESSIBILITY_SERVICES,
        ) ?: return false
        val splitter = TextUtils.SimpleStringSplitter(':').apply { setString(setting) }
        for (entry in splitter) {
            if (entry.equals(expected, ignoreCase = true)) return true
        }
        return false
    }
}
