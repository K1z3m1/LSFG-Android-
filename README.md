# LSFG-Android+

A rebrand/fork of [LSFG-Android](https://github.com/FrankBarretta/LSFG-Android)
by FrankBarretta — an Android frame-generation overlay that runs Lossless
Scaling-style frame interpolation on top of another app via screen capture.

## Channel

Maintained by **First** — [@firstT175 on YouTube](https://m.youtube.com/channel/UCujpG08a_M_JYYiALCtLQ9Q)

## Credits

| | |
|---|---|
| **Base project** | [LSFG-Android](https://github.com/FrankBarretta/LSFG-Android) by FrankBarretta |
| **Channel** | [First — @firstT175](https://m.youtube.com/channel/UCujpG08a_M_JYYiALCtLQ9Q) |
| **License** | [GNU GPL v3.0](https://www.gnu.org/licenses/gpl-3.0.html), inherited from the base project |

The same credits and license are also shown in-app, under **Credits** in the
app's overflow menu.

## Changes from the base APK

- Renamed `applicationId`/package from `com.lsfg.android` → `com.firstt175.deepdrop`
  (Kotlin sources, AIDL, JNI native method bindings all updated to match).
- App label changed to **LSFG-Android+**; internal branding strings
  (notifications, accessibility service label, overlay status text) updated
  to match.
- Version bumped **0.1.2 → 0.1.3** (`versionCode` 1 → 2).
- New in-app **Credits** screen (accessible from the ⋮ menu on the home
  screen): YouTube channel listed first, base-APK credit below it, and the
  GPL-3.0 license folded into the same page.
- New app icon.
- Animated YouTube-channel avatar on the Credits screen (GIF, played via
  Coil's `coil-gif` decoder from a bundled asset — no network calls).

## GPU-only AI build

The bundled RIFE/IFRNet ncnn AI backend is hard-locked to Vulkan GPU compute.
It does not create a CPU inference network and does not fall back to CPU when
Vulkan initialization fails. Both custom Warp layers are Vulkan compute
pipelines, so the ncnn graph no longer performs a CPU readback for Warp.

Lossless.dll compatibility is pinned to the exact 3.2.2.0 ProductVersion. The
shader extractor rejects other Lossless.dll versions instead of silently
applying the 3.2.2 resource map to a different DLL layout.

The Android/OS orchestration itself still necessarily runs on CPU (JNI,
threads, SurfaceFlinger/BufferQueue and, with the currently bundled ncnn
prebuilt, the RGBA8 AHB bridge used by the AI backend). This build removes
CPU *inference* and CPU Warp; it does not falsely claim that an Android app
can execute zero CPU instructions.
