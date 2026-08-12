#pragma once

// Glue between MediaProjection (AHardwareBuffer frames pushed from Kotlin),
// the framegen LSFG_3_1 pipeline (consumes opaque FDs), and the overlay
// SurfaceView (an ANativeWindow we blit final frames to).
//
// Lifecycle:
//   init()        — create VulkanSession, allocate input/output AhbImages,
//                   call LSFG_3_1::initialize and createContext
//   setOutputSurface(window, w, h) — attach the overlay surface for blit
//   pushFrame(ahb)— enqueue a fresh capture frame; render thread will copy
//                   into the next ping-pong slot, present, and blit outputs
//   shutdown()    — flush, deleteContext, finalize, destroy session
//
// Counter:
//   getGeneratedFrameCount() — atomic counter incremented per generated
//   frame (multiplier-many per pushFrame). Drives the overlay's "total fps".

#include <cstdint>
#include <string>

struct ANativeWindow;
struct AHardwareBuffer;

namespace lsfg_android {

constexpr int kRenderLoopAlreadyInit = -40;
constexpr int kRenderLoopSessionFailed = -41;
constexpr int kRenderLoopFramegenFailed = -42;
constexpr int kRenderLoopBufferAlloc = -43;

// Positive "soft" status: init succeeded but framegen is disabled (missing
// extensions, shader load failure, FD export refused, …). The caller should
// fall back to mirror mode — keep the overlay up, feed it the raw capture,
// and surface a notice to the user. The render loop itself stays initialised
// and must be torn down normally via shutdownRenderLoop() when the session ends.
constexpr int kRenderLoopFramegenDisabled = 1;

struct RenderLoopConfig {
    uint32_t width;
    uint32_t height;
    int multiplier;       // generationCount passed to LSFG: total = capture * multiplier
    float flowScale;      // 0.25 .. 1.0
    bool performance;     // selects LSFG_3_1P vs LSFG_3_1
    bool hdr;
    // Use the precompiled SPIR-V FP16 shader variants (Lossless.dll resource
    // IDs 304..351) instead of the default precompiled FP32 SPIR-V set
    // (IDs 353..400). The FP16 variants enable OpCapability Float16 and use
    // mixed FP16/FP32 ops. Requires the GPU to support
    // VK_KHR_shader_float16_int8 + shaderFloat16, and requires the FP16
    // SPIR-V cache to have been populated by the DLL extraction step. The
    // render loop transparently falls back to the FP32 path when either
    // prerequisite is missing.
    bool framegenFp16;
    // Pacing tunables (0/negative values fall back to defaults inside the loop).
    float emaAlpha;        // 0.05 .. 0.5 (default 0.125)
    float outlierRatio;    // 2.0 .. 8.0 (default 4.0)
    float vsyncSlackMs;    // 1.0 .. 5.0 (default 2.0)
    // Display-stage post-processing. These run only when enabled and are
    // deliberately downstream of frame generation: the LSFG/AI interpolation
    // loop and its timing/pacing are not changed by these switches.
    bool gpuPostProcessing;
    float gpuUpscaleFactor; // 1.0 or 1.5 or 2.0
    int gpuMethod;          // 0 = FSR1 EASU+RCAS (only production default)
    float gpuSharpness;
    float gpuStrength;
    bool npuPostProcessing; // NNAPI dedicated accelerator only; no fallback
    int npuPreset;          // 4 = GAME_CRISP
    bool cpuPostProcessing;
    int cpuPreset;          // 5 = GAMER_SHARP
    float cpuStrength;
    float cpuSaturation;
    float cpuVibrance;
    float cpuVignette;

    // Selects the interpolation backend. When true, the render loop skips
    // LSFG_3_1::initialize/createContext entirely and instead loads
    // NcnnInterpolator from `aiModelDir` (the directory AiModelBundleReader.kt
    // copies from the bundled model asset), running the selected RIFE/IFRNet graph on Vulkan GPU only. Ignored (falls back to the
    // LSFG shader path) when this .so was built without ncnn (LSFG_HAVE_NCNN
    // not defined) or when the GPU-only model load fails — check logcat tag
    // "lsfg-render" for the reason. `multiplier`/`flowScale` above are shared
    // between both backends.
    bool aiBackend = false;
    std::string aiModelDir;   // must contain the selected GPU model files
    // Which ncnn graph aiModelDir is expected to contain: 0 = RIFE
    // (NcnnInterpolator, flownet.param/.bin), 1 = IFRNet (IfrnetInterpolator,
    // ifrnet.param/.bin). Mirrors com.firstt175.deepdrop.prefs.AiEngine's prefValue
    // ordering — keep them in sync. Any other value falls back to RIFE.
    // Ignored when aiBackend is false.
    int aiEngine = 0;
};

// Initialise render loop: create Vulkan session, allocate ping-pong inputs +
// (multiplier-1) outputs, initialize framegen, create context. Returns kOk
// or one of kRenderLoop* on failure.
//
// `cacheDir` is forwarded to the framegen shader-loader callback.
int initRenderLoop(const char *cacheDir, const RenderLoopConfig &cfg);

// Attach (or replace) the output ANativeWindow. Caller transfers ownership;
// the render loop will release the previous one (if any) and acquire `win`.
// Pass nullptr to detach.
void setOutputSurface(ANativeWindow *win, uint32_t w, uint32_t h);

// Push a capture frame. The render loop acquires a reference to `ahb`
// (so it stays valid past the caller's Image.close()) and processes it
// asynchronously. `timestampNs` should come from Image.getTimestamp() when
// available so output pacing can follow the real capture cadence.
// Safe to call from any thread.
void pushFrame(AHardwareBuffer *ahb, int64_t timestampNs);

// Shut down everything. Idempotent.
void shutdownRenderLoop();

// Counter for the FPS overlay. Returns 0 before init / after shutdown.
uint64_t getGeneratedFrameCount();

// Total frames actually posted to the overlay surface (CPU blit or WSI
// present path). Unlike getGeneratedFrameCount this includes the real
// capture post on each cycle, giving the HUD a ground-truth "total fps".
uint64_t getPostedFrameCount();

// Number of capture frames whose content differs from the previous capture.
// MediaProjection delivers at the display refresh rate, which is usually
// higher than the target app's render rate — duplicates are common. This
// counter is the target app's TRUE render rate (what the HUD should show
// as "real fps"). Computed via a cheap 8×8 luma hash in pushFrame.
uint64_t getUniqueCaptureCount();

// Copies up to `cap` nanosecond-intervals between consecutive overlay posts
// into `outIntervalsNs`. Returns the number of intervals actually written.
// Newest-first order. Used by the HUD frame-pacing graph to show real
// jitter instead of rolling counts.
uint32_t getRecentPostIntervalsNs(int64_t *outIntervalsNs, uint32_t cap);

// Interval-based FPS snapshot — no polling window, no counter+delta math.
// Computed from the average spacing between the last few recorded capture/
// post timestamps (steady_clock, recorded at the moment each event happens).
// Writes [realFps, totalFps] into `out` (must be at least 2 floats) and
// returns true once at least 2 samples exist in both rings; returns false
// (out left untouched) before enough data has accumulated, e.g. right after
// init. realFps tracks capture arrivals (getUniqueCaptureCount's source),
// totalFps tracks every overlay post (getPostedFrameCount's source) —
// same ground truth as the existing counters, just measured as elapsed time
// between events instead of events-per-fixed-window, so there's no
// window-aliasing to smooth out on the caller side.
bool getFpsSnapshot(float *out, uint32_t cap);

// Snapshot of the most recently completed profiling window. `out` must be
// at least 6 longs; on success populates [copyNs, presentNs, waitIdleNs,
// blitNs, totalNs, samples] (each segment is the SUM over the window —
// divide by samples for per-frame averages) and returns 6. Returns 0 when
// no window has completed yet or `cap` < 6.
uint32_t getProfileWindowNs(int64_t *out, uint32_t cap);

// Toggle frame-gen bypass. When true, the worker skips framegen entirely and
// blits the latest captured frame straight to the output surface — useful for
// A/B comparisons against the generated output. Safe to call from any thread.
void setBypass(bool bypass);

// Report the display's vsync period in nanoseconds (e.g. 16_666_666 for a
// 60 Hz display, 8_333_333 for 120 Hz). When set to a positive value the
// pacing loop aligns its sleep targets to vsync boundaries so consecutive
// ANativeWindow_unlockAndPost calls land on distinct vsync slots, avoiding
// the double-post → SurfaceFlinger-stall pattern that shows as periodic
// stutter. Passing 0 falls back to raw sleep_until.
void setVsyncPeriodNs(int64_t periodNs);

// Hot-apply pacing tunables without tearing down the render loop. Parameters
// fall back to their defaults when outside a sane range. Safe to call from
// any thread.
void setPacingParams(float emaAlpha,
                     float outlierRatio,
                     float vsyncSlackMs);

// Enable/disable the Shizuku timing side channel. When enabled, the pacing
// loop may prefer the externally reported frame time over the raw capture
// timestamps, and may suppress generated frames when the external pacing
// jitter indicates the target stream is unstable.
void setShizukuTimingEnabled(bool enabled);

// Report one external timing sample from the Shizuku metrics side channel.
// `frameTimeNs` is the delta between two target frames; `pacingJitterNs` is
// the absolute deviation from the target cadence. Safe to call from any thread.
void reportShizukuTiming(int64_t timestampNs,
                         int64_t frameTimeNs,
                         int64_t pacingJitterNs);

double getAverageQueueMs();
double getAverageLatencyMs();

} // namespace lsfg_android
