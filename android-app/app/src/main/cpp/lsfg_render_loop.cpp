#include "lsfg_render_loop.hpp"
#include "android_vk_session.hpp"
#include "android_vk_probe.hpp"
#include "ahb_image_bridge.hpp"
#include "gpu_postprocess.hpp"
#include "nnapi_npu.hpp"
#include "nnapi_postprocess.hpp"
#include "cpu_postprocess.hpp"
#include "android_shader_loader.hpp"
#ifdef LSFG_HAVE_NCNN
#include "NcnnInterpolator.hpp"
#include "IfrnetInterpolator.hpp"
#endif

#include "lsfg_3_1.hpp"
#include "lsfg_3_1p.hpp"

#include <volk.h>

#include <android/log.h>
#include <android/native_window.h>
#include <android/hardware_buffer.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <cmath>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <vector>
#include <unordered_map>
#include <sys/resource.h>
#include <sys/types.h>
#include <unistd.h>

// android/thread_defs.h is a platform-private header not shipped in the NDK
// sysroot, so it never resolves in app builds. Define the constants we need
// (match AOSP's system/core/libutils/include/utils/ThreadDefs.h).
#ifndef ANDROID_PRIORITY_URGENT_DISPLAY
#define ANDROID_PRIORITY_URGENT_DISPLAY (-8)
#endif
#ifndef ANDROID_PRIORITY_URGENT_AUDIO
#define ANDROID_PRIORITY_URGENT_AUDIO (-19)
#endif

#include "crash_reporter.hpp"
#include "cpu_core_policy.hpp"
#ifdef LSFG_HAVE_NCNN
#include "ncnn_cpu_policy.hpp"
#endif

#define LOG_TAG "lsfg-vk-loop"
#define LOGE(...) ::lsfg_android::ring_logf(LOG_TAG, ANDROID_LOG_ERROR, __VA_ARGS__)
#define LOGW(...) ::lsfg_android::ring_logf(LOG_TAG, ANDROID_LOG_WARN,  __VA_ARGS__)
#define LOGI(...) ::lsfg_android::ring_logf(LOG_TAG, ANDROID_LOG_INFO,  __VA_ARGS__)

namespace lsfg_android {

namespace {

struct State {
    using Clock = std::chrono::steady_clock;

    std::mutex mu;
    std::atomic<bool> initialized{false};
    bool performanceMode = false;
    bool framegenInitOk = false;  // tracks whether LSFG_3_1::initialize succeeded
    bool framegenFp16 = false;    // load IDs 304..351 (FP16 SPIR-V) instead of 353..400 (FP32 SPIR-V)
    bool hdr = false;
    float flowScale = 1.0f;
    int32_t framegenCtxId = -1;
    int multiplier = 2;          // generationCount

    VulkanSession vk{};

    AhbImage inSlot[2]{};        // ping-pong inputs
    uint64_t framesCopied = 0;   // total inputs we've copied into a slot
    uint64_t presentsDone = 0;   // mirrors framegen's internal frameIdx

    // Vulkan swapchain state. When live, blitOutputToWindow takes the
    // GPU-only fast path: vkAcquireNextImageKHR → vkCmdBlitImage from the
    // framegen output AHB → vkQueuePresentKHR. This eliminates the CPU
    // memcpy (AHardwareBuffer_lock(CPU_READ) + ANativeWindow_lock +
    // per-row memcpy) that was the single biggest cost at multiplier ≥ 2.
    //
    // The WSI path is disabled when the extension chain or surface support is
    // unavailable. The presentation pipeline stays GPU-only; a failed WSI path
    // drops the frame rather than copying pixels through the CPU.
    struct SwapchainState {
        VkSurfaceKHR surface = VK_NULL_HANDLE;
        VkSwapchainKHR swapchain = VK_NULL_HANDLE;
        VkExtent2D extent{};
        VkFormat format = VK_FORMAT_UNDEFINED;
        std::vector<VkImage> images;
        // One acquire semaphore per slot, cycled round-robin. Vulkan spec
        // forbids reusing a semaphore until the corresponding acquire has
        // completed; N+1 is the safe lower bound.
        std::vector<VkSemaphore> acquireSems;
        // One render-done semaphore per swapchain image (vkQueuePresentKHR
        // waits on the image's semaphore before presenting).
        std::vector<VkSemaphore> renderSems;
        uint32_t acquireCursor = 0;
        bool outOfDate = false;
        // Set when an attempt to build the swapchain failed (e.g. the compute
        // queue family cannot present, or the surface rejects TRANSFER_DST
        // usage). Once set we stop retrying for the rest of the session and
        // use the CPU blit path exclusively. Cleared by destroySwapchain()
        // so a surface re-attach gets a fresh attempt.
        bool disabledForSession = false;
    } swap;
    // ANativeWindow width/height the swapchain was built for. When the
    // overlay reshapes (rare, mostly on orientation change), we recreate.
    uint32_t swapWinW = 0;
    uint32_t swapWinH = 0;
    // AI backend (ncnn RIFE flownet) state. `aiRequested` mirrors
    // cfg.aiBackend from the most recent initRenderLoop call; `aiLoaded`
    // reflects whether NcnnInterpolator::load() actually succeeded. Kept
    // outside the #ifdef so runAi's condition below compiles identically
    // whether or not this .so was built with ncnn — it just stays false.
    bool aiRequested = false;
    bool aiLoaded = false;
    // Mirrors cfg.aiEngine from the most recent initRenderLoop call (0 =
    // RIFE/NcnnInterpolator, 1 = IFRNet/IfrnetInterpolator). Only one of
    // `ai`/`aiIfrnet` below is ever non-null at a time — runAiInterpolate()
    // branches on this to know which one. Kept outside the #ifdef for the
    // same reason as aiRequested/aiLoaded.
    int aiEngine = 0;
#ifdef LSFG_HAVE_NCNN
    NcnnInterpolator *ai = nullptr;
    IfrnetInterpolator *aiIfrnet = nullptr;
#endif
    std::atomic<bool> bypass{false}; // skip framegen, blit raw input
    // Auto-bypass triggered when framegen returns VK_ERROR_DEVICE_LOST during
    // presentContext. Distinct from the user-controlled `bypass` so the user
    // toggle isn't silently flipped by a recoverable driver event. Cleared on
    // every initRenderLoop / context recreation — if the next session
    // succeeds, framegen runs again. Stuck-on means the next presentContext
    // would error too, so passthrough is the right behaviour anyway.
    std::atomic<bool> framegenAutoDisabled{false};
    std::atomic<uint64_t> cacheHits{0};
    std::atomic<uint64_t> cacheMisses{0};
    std::vector<AhbImage> outputs; // multiplier-many outputs
    // Reusable AI host staging buffers. These are allocated once per resolution
    // instead of creating/destroying several large vectors every frame. The
    // actual model math remains Vulkan-only; this buffer is only the legacy
    // RGBA8 bridge required by the bundled ncnn build.
    std::vector<uint8_t> aiInputAStaging;
    std::vector<uint8_t> aiInputCStaging;
    std::vector<std::vector<uint8_t>> aiOutputStaging;
    std::vector<uint8_t*> aiOutputPtrs;

    // Optional display-stage post-processing. These buffers are never fed back
    // into frame generation; they exist only between the completed frame and
    // blitOutputToSwapchain().
    bool gpuPostProcessing = false;
    int gpuMethod = 0;
    float gpuUpscaleFactor = 1.0f;
    float gpuSharpness = 0.55f;
    float gpuStrength = 0.75f;
    bool npuPostProcessing = false;
    bool npuStageFailed = false;
    int npuPreset = 4;
    bool cpuPostProcessing = false;
    int cpuPreset = 5;
    float cpuStrength = 0.45f;
    float cpuSaturation = 0.5f;
    float cpuVibrance = 0.0f;
    float cpuVignette = 0.0f;
    GpuPostProcessor gpuPost{};
    NnapiPostProcessor npuPost{};
    CpuPostProcessor cpuPost{};
    AhbImage postImage{};
    std::vector<uint8_t> postInputStaging;
    std::vector<uint8_t> postOutputStaging;
    std::vector<uint8_t> postCpuStaging;

    // AHB → VkImage import cache. MediaProjection rotates a small pool (2-4)
    // of AHBs; re-importing on every frame wastes vkCreateImage +
    // vkAllocateMemory. The cache holds its own AHardwareBuffer_acquire ref
    // so entries outlive the per-frame release. Evicted FIFO beyond kAhbCacheMax.
    // Capped at 4 (not 8): MediaProjection rotates a pool of 2-4 buffers, so a
    // cache of 8 pins up to 4 stale/duplicate AHBs' worth of GPU-shared memory
    // (each ~10 MB at 1080x2408 RGBA8) with zero cache-hit benefit — pure waste
    // that pushes the process toward the low-memory killer on constrained
    // devices when a second heavy app (e.g. video playback) shares memory.
    // Must be fully cleared (destroyAhbImage + AHardwareBuffer_release for each
    // entry) before vkDestroyDevice in the cleanup path.
    static constexpr size_t kAhbCacheMax = 4;
    std::unordered_map<AHardwareBuffer*, AhbImage> ahbImportCache;

    // Output surface for the final blit. Owned (acquired from JNI).
    ANativeWindow *outWindow = nullptr;
    uint32_t outWidth = 0;
    uint32_t outHeight = 0;
    // Once we have produced a CPU buffer on this ANativeWindow (via
    // ANativeWindow_lock / setBuffersGeometry), the BufferQueue is bound to a
    // CPU producer. Mali (Bifrost/Valhall) and many other drivers will then
    // return VK_ERROR_NATIVE_WINDOW_IN_USE_KHR (-1000000001) for any subsequent
    // vkCreateAndroidSurfaceKHR on the same native window — the producer slot
    // can't be retargeted live. We track this taint per-attached window and
    // skip the WSI swapchain attempt entirely, avoiding the spammy retries seen
    // on Mali-G57 and the cascade where a failed surface creation correlates
    // with a DEVICE_LOST on framegen's compute queue. Reset on each
    // setOutputSurface(win) call, since a freshly-acquired ANativeWindow has
    // no producer yet.
    bool windowCpuProducerLocked = false;
    // Feedback-loop gate: suppresses blits when framegen output luma is very
    // dark, indicating setSkipScreenshot failed and MediaProjection is capturing
    // the overlay instead of the game.  Keeping the overlay transparent lets the
    // next capture see the real game and breaks the loop.
    bool     lumaGateOpen      = false;
    uint32_t lumaGateDarkCount = 0;
    int64_t  lumaGateStartNs   = 0; // steady_clock ns at first suppressed dark frame



    // No queue, no worker thread. pushFrame() runs the whole capture →
    // framegen → blit pipeline synchronously, on whichever thread the
    // capture engine calls it from. frameMu only serializes that work
    // against shutdownRenderLoop() (and against a second pushFrame() in
    // the rare case two calls overlap) — it never buffers a frame. If a
    // call arrives while a previous one is still running it simply blocks
    // until that one finishes, then processes exactly the frame it was
    // given; nothing is ever dropped, coalesced, or queued up on our side.
    std::mutex frameMu;

    // CPU-side work starts on the performance cluster. If the measured CPU
    // portion of a frame misses its budget, the worker widens its affinity to
    // all online cores so little cores can help instead of remaining unused.
    CpuCorePolicy cpuPolicy{};
    bool cpuLittleAssist = false;
    std::atomic<uint64_t> generatedFrames{0};
    // Counts every successful post (WSI present) to the overlay.
    // This is the ground-truth "frames on screen" metric — includes real
    // captures AND LSFG-generated frames. Used by the HUD total-fps counter
    // instead of the old `capturedFps + genFps` which double-counted.
    std::atomic<uint64_t> postedFrames{0};
    // Counts every capture frame that arrives (regardless of content).
    // Used as the HUD's "real fps" / unique-capture metric. Previously this
    // used a per-pixel luma hash to detect duplicate frames from MediaProjection,
    // but the CPU cost of reading back AHB pixels was not worth the metric
    // accuracy gain. We now count every arriving frame as unique — the number
    // is the true capture rate from the OS capture path, which closely tracks
    // the target app's render rate in practice.
    std::atomic<uint64_t> uniqueCaptures{0};

    // Ring buffer of recent post timestamps (ns from CLOCK_MONOTONIC, via
    // steady_clock). Consumed by the HUD frame-pacing graph to show real
    // frame-to-frame intervals instead of rolling counts.
    static constexpr size_t kPostRingSize = 128;
    std::atomic<uint64_t> postRingTimestamps[kPostRingSize]{};
    std::atomic<uint64_t> postRingHead{0};

    // Same idea as postRingTimestamps, but for capture arrivals (pushFrame).
    // Lets getFpsSnapshot derive "real fps" from actual elapsed time between
    // captures instead of a counter+delta over a fixed polling window.
    std::atomic<uint64_t> captureRingTimestamps[kPostRingSize]{};
    std::atomic<uint64_t> captureRingHead{0};

    std::atomic<uint32_t> pushLogCount{0};
    std::atomic<uint32_t> blitLogCount{0};

    // Snapshot of the last completed kProfileWindow rolling window. Updated
    // atomically by processFrame() when a window closes (see ProfileAccum).
    // Layout: [copyNs, presentNs, waitIdleNs, blitNs, totalNs,
    // samples]. Each value is the SUM over the window — divide by samples to
    // get the per-frame average. samples == 0 means no window has completed
    // since session start.
    std::atomic<int64_t> profileSnapshotCopyNs{0};
    std::atomic<int64_t> profileSnapshotPresentNs{0};
    std::atomic<int64_t> profileSnapshotWaitIdleNs{0};
    std::atomic<int64_t> profileSnapshotBlitNs{0};
    std::atomic<int64_t> profileSnapshotTotalNs{0};
    std::atomic<int64_t> profileSnapshotQueueNs{0};
    std::atomic<int64_t> profileSnapshotLatencyNs{0};
    std::atomic<int64_t> profileSnapshotBlitCount{0};
    std::atomic<int64_t> profileSnapshotSamples{0};
    std::atomic<bool> shizukuTimingEnabled{false};
    std::atomic<int64_t> shizukuSampleTimestampNs{0};
    std::atomic<int64_t> shizukuFrameTimeNs{0};
    std::atomic<int64_t> shizukuPacingJitterNs{0};

};

State g{};

void handleFramegenException(const char *callSite, const std::exception &e) {
    const char *what = e.what() != nullptr ? e.what() : "(null)";
    LOGE("%s threw: %s", callSite, what);
    const bool isDeviceLost = std::strstr(what, "error -4") != nullptr ||
                              std::strstr(what, "DEVICE_LOST")  != nullptr;
    if (isDeviceLost && !g.framegenAutoDisabled.load(std::memory_order_relaxed)) {
        LOGE("%s: VK_ERROR_DEVICE_LOST — auto-disabling framegen for this session (passthrough until next context reinit)",
             callSite);
        g.framegenAutoDisabled.store(true, std::memory_order_relaxed);
    }
}

// Record a successful post (WSI present) for HUD metrics.
// Increments postedFrames and pushes the current steady-clock timestamp into
// the ring buffer so the pacing graph can compute real inter-frame intervals.
void recordOverlayPost() {
    g.postedFrames.fetch_add(1, std::memory_order_relaxed);
    const uint64_t nowNs = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            State::Clock::now().time_since_epoch()).count());
    const uint64_t slot = g.postRingHead.fetch_add(1, std::memory_order_relaxed)
                          % State::kPostRingSize;
    g.postRingTimestamps[slot].store(nowNs, std::memory_order_relaxed);
}

// ---- Shared Vulkan one-shot helpers -------------------------------------------
//
// Every AHB-side image transition in this file (copy, GPU blit, swapchain
// blit, initial layout transition) builds VkImageMemoryBarrier literals that
// differ only in image/access/layout/family — factored here so each call site
// is one line instead of a ~9-line struct.
VkImageMemoryBarrier makeImageBarrier(VkImage image, VkAccessFlags srcAccess, VkAccessFlags dstAccess,
                                       VkImageLayout oldLayout, VkImageLayout newLayout,
                                       uint32_t srcFamily, uint32_t dstFamily) {
    return VkImageMemoryBarrier{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask = srcAccess,
        .dstAccessMask = dstAccess,
        .oldLayout = oldLayout,
        .newLayout = newLayout,
        .srcQueueFamilyIndex = srcFamily,
        .dstQueueFamilyIndex = dstFamily,
        .image = image,
        .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
    };
}

// Allocate a one-shot primary command buffer from the session pool, run
// `record` to fill it, submit on the compute queue and block until the GPU
// has finished. Uses a per-submission VkFence (not vkQueueWaitIdle) so we
// only stall on this submission, not the whole queue. Frees the fence and
// command buffer before returning either way.
//
// Used by every transient (allocate → record → submit → wait → free) Vulkan
// op in this file: initial layout transitions, the CPU-side AHB copy, and
// the linear-blit fallback. The session's persistent command ring
// (acquireCommandRing) is a separate, non-transient path used by the
// steady-state swapchain blit.
bool runTransientCommands(const std::function<void(VkCommandBuffer)> &record) {
    const VkCommandBufferAllocateInfo cbai{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = g.vk.commandPool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    VkCommandBuffer cb = VK_NULL_HANDLE;
    if (g.vk.fn.vkAllocateCommandBuffers(g.vk.device, &cbai, &cb) != VK_SUCCESS) return false;

    const VkCommandBufferBeginInfo bi{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    g.vk.fn.vkBeginCommandBuffer(cb, &bi);
    record(cb);
    g.vk.fn.vkEndCommandBuffer(cb);

    const VkFenceCreateInfo fci{ .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    VkFence fence = VK_NULL_HANDLE;
    g.vk.fn.vkCreateFence(g.vk.device, &fci, nullptr, &fence);

    const VkSubmitInfo si{
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &cb,
    };
    const bool submitted = g.vk.fn.vkQueueSubmit(g.vk.computeQueue, 1, &si, fence) == VK_SUCCESS;
    if (submitted) g.vk.fn.vkWaitForFences(g.vk.device, 1, &fence, VK_TRUE, UINT64_MAX);
    if (fence != VK_NULL_HANDLE) g.vk.fn.vkDestroyFence(g.vk.device, fence, nullptr);
    g.vk.fn.vkFreeCommandBuffers(g.vk.device, g.vk.commandPool, 1, &cb);
    return submitted;
}

// ---- Vulkan swapchain helpers ------------------------------------------------
//
// The swapchain lives on top of the overlay's ANativeWindow and provides the
// GPU-only output path: generated frames are vkCmdBlitImage'd from their AHB
// storage directly into the swapchain image and presented via vkQueuePresentKHR.
// No CPU touch of the pixel data.
//
// DEFAULT OFF. An earlier run on Adreno 840 / Android 14 crashed in a driver
// call during createSwapchain before any of the per-step LOGIs could fire —
// the Adreno compute queue reports `vkGetPhysicalDeviceSurfaceSupportKHR =
// VK_TRUE` but then crashes inside `vkQueuePresentKHR` (known quirk on some
// Qualcomm revisions). Keeping the code path in-tree with aggressive logging
// so future device revisions / validation work can flip this flag without
// another refactor. To enable for testing, set `kEnableWsiSwapchain` to true.
constexpr bool kEnableWsiSwapchain = true;

void destroySwapchain() {
    // Must drain every in-flight ring submission before touching the images
    // — the swapchain images are referenced by recorded CBs via barriers and
    // destroying them while those CBs are still executing = SIGSEGV on some
    // drivers (observed on Qualcomm). vkDeviceWaitIdle is the sledgehammer.
    if (g.vk.device != VK_NULL_HANDLE && g.vk.fn.vkDeviceWaitIdle != nullptr) {
        g.vk.fn.vkDeviceWaitIdle(g.vk.device);
    }

    if (g.vk.fn.vkDestroySemaphore != nullptr) {
        for (VkSemaphore s : g.swap.acquireSems) {
            if (s != VK_NULL_HANDLE) g.vk.fn.vkDestroySemaphore(g.vk.device, s, nullptr);
        }
        for (VkSemaphore s : g.swap.renderSems) {
            if (s != VK_NULL_HANDLE) g.vk.fn.vkDestroySemaphore(g.vk.device, s, nullptr);
        }
    }
    g.swap.acquireSems.clear();
    g.swap.renderSems.clear();
    g.swap.images.clear();
    g.swap.acquireCursor = 0;
    g.swap.outOfDate = false;
    g.swap.disabledForSession = false;

    if (g.swap.swapchain != VK_NULL_HANDLE && g.vk.fn.vkDestroySwapchainKHR != nullptr) {
        g.vk.fn.vkDestroySwapchainKHR(g.vk.device, g.swap.swapchain, nullptr);
        g.swap.swapchain = VK_NULL_HANDLE;
    }
    if (g.swap.surface != VK_NULL_HANDLE && g.vk.instance != VK_NULL_HANDLE) {
        // Surface destruction uses the instance-level function. volk populates
        // it globally after volkLoadInstance.
        if (g.vk.pfnDestroySurfaceKHR != nullptr) {
            g.vk.pfnDestroySurfaceKHR(g.vk.instance, g.swap.surface, nullptr);
        } else {
            vkDestroySurfaceKHR(g.vk.instance, g.swap.surface, nullptr);
        }
        g.swap.surface = VK_NULL_HANDLE;
    }
    g.swap.extent = {0, 0};
    g.swap.format = VK_FORMAT_UNDEFINED;
}

// Build (or rebuild) the swapchain on the current outWindow. Returns true if
// the WSI path is live after this call; false means the caller must fall back
// to the CPU blit path for this session.
//
// Safe to call multiple times; previous swapchain is torn down first.
bool createSwapchain() {
    // GPU-only presentation. A failure disables WSI for this session; there is
    // deliberately no CPU pixel fallback in the hot path.
    LOGI("createSwapchain: enter outWindow=%p hasSwapchain=%d enable=%d cpuTainted=%d",
         static_cast<void *>(g.outWindow), (int)g.vk.hasSwapchain,
         (int)kEnableWsiSwapchain, (int)g.windowCpuProducerLocked);
    destroySwapchain();
    if (!kEnableWsiSwapchain) return false;
    if (!g.vk.hasSwapchain) return false;
    if (g.outWindow == nullptr) return false;
    if (g.vk.instance == VK_NULL_HANDLE) return false;
    // Once the worker has produced any CPU buffer on this ANativeWindow
    // (overlay clear, CPU blit fallback, etc.) the BufferQueue is locked to
    // a CPU producer and vkCreateAndroidSurfaceKHR will return
    // VK_ERROR_NATIVE_WINDOW_IN_USE_KHR (-1000000001) on Mali / many other
    // drivers. Don't even attempt — failed surface creation has been observed
    // to correlate with framegen-side DEVICE_LOST on Mali-G57 (likely an
    // instance-level state corruption when the WSI driver path bails late).
    if (g.windowCpuProducerLocked) {
        LOGI("createSwapchain: skipping — window already bound to CPU producer this session");
        return false;
    }
    // Use the session-cached function pointer instead of volk's global. The
    // global gets clobbered to NULL when framegen's Instance::Instance()
    // calls volkLoadInstance() against its own surface-less VkInstance,
    // because that instance can't resolve vkCreateAndroidSurfaceKHR. The
    // session pointer was resolved against OUR instance at session-init time
    // and survives that overwrite.
    PFN_vkCreateAndroidSurfaceKHR pfnCreateSurface = g.vk.pfnCreateAndroidSurfaceKHR;
    if (pfnCreateSurface == nullptr) pfnCreateSurface = vkCreateAndroidSurfaceKHR;
    if (pfnCreateSurface == nullptr) {
        LOGW("vkCreateAndroidSurfaceKHR fn ptr is NULL — volk didn't load it");
        return false;
    }

    LOGI("createSwapchain: calling vkCreateAndroidSurfaceKHR");
    const VkAndroidSurfaceCreateInfoKHR sci{
        .sType = VK_STRUCTURE_TYPE_ANDROID_SURFACE_CREATE_INFO_KHR,
        .window = g.outWindow,
    };
    const VkResult sres = pfnCreateSurface(g.vk.instance, &sci, nullptr, &g.swap.surface);
    if (sres != VK_SUCCESS) {
        // VK_ERROR_NATIVE_WINDOW_IN_USE_KHR means the ANativeWindow is locked
        // to a CPU producer (or another consumer). The state is sticky — no
        // amount of retrying on the same window will recover. Pin the taint
        // so subsequent createSwapchain calls bail at the early gate above
        // instead of repeating the same failed driver call (which on some
        // Mali / Adreno revisions can leave the instance in a tainted state
        // and induce a DEVICE_LOST on framegen's separate device).
        if (sres == VK_ERROR_NATIVE_WINDOW_IN_USE_KHR) {
            g.windowCpuProducerLocked = true;
            LOGW("vkCreateAndroidSurfaceKHR: window in use by CPU producer (rc=%d) — WSI disabled for this surface",
                 (int)sres);
        } else {
            LOGW("vkCreateAndroidSurfaceKHR failed (rc=%d) — WSI disabled", (int)sres);
        }
        g.swap.surface = VK_NULL_HANDLE;
        return false;
    }
    LOGI("createSwapchain: surface=%p", static_cast<void *>(g.swap.surface));

    // Can our compute queue actually present on this surface?
    VkBool32 canPresent = VK_FALSE;
    const VkResult suppr = g.vk.pfnGetPhysicalDeviceSurfaceSupportKHR(g.vk.physicalDevice,
            g.vk.computeFamilyIdx, g.swap.surface, &canPresent);
    LOGI("createSwapchain: surfaceSupport rc=%d canPresent=%d", (int)suppr, (int)canPresent);
    if (suppr != VK_SUCCESS || canPresent != VK_TRUE) {
        LOGW("compute queue family %u cannot present on this surface — fall back to CPU blit",
             g.vk.computeFamilyIdx);
        destroySwapchain();
        return false;
    }

    VkSurfaceCapabilitiesKHR caps{};
    if (g.vk.pfnGetPhysicalDeviceSurfaceCapabilitiesKHR(g.vk.physicalDevice,
            g.swap.surface, &caps) != VK_SUCCESS) {
        LOGW("vkGetPhysicalDeviceSurfaceCapabilitiesKHR failed");
        destroySwapchain();
        return false;
    }

    // Keep the presentation path on one simple, low-latency format: RGBA8
    // sRGB. HDR is still available through the separate toggle later, but the
    // default runtime should stay on the narrowest possible path so the swapchain
    // can stay fully GPU-native and avoid any extra format conversion work.
    uint32_t fmtCount = 0;
    g.vk.pfnGetPhysicalDeviceSurfaceFormatsKHR(g.vk.physicalDevice, g.swap.surface,
                                         &fmtCount, nullptr);
    std::vector<VkSurfaceFormatKHR> fmts(fmtCount);
    if (fmtCount > 0) {
        g.vk.pfnGetPhysicalDeviceSurfaceFormatsKHR(g.vk.physicalDevice, g.swap.surface,
                                             &fmtCount, fmts.data());
    }
    VkSurfaceFormatKHR chosen{VK_FORMAT_R8G8B8A8_UNORM, VK_COLORSPACE_SRGB_NONLINEAR_KHR};
    for (const auto &f : fmts) {
        if (f.format == VK_FORMAT_R8G8B8A8_UNORM &&
            f.colorSpace == VK_COLORSPACE_SRGB_NONLINEAR_KHR) {
            chosen = f;
            break;
        }
    }
    if (fmts.empty()) {
        chosen = {VK_FORMAT_R8G8B8A8_UNORM, VK_COLORSPACE_SRGB_NONLINEAR_KHR};
    }

    // MAILBOX is the only presentation mode used by this low-latency path.
    // Do not silently fall back to FIFO: FIFO introduces producer blocking and
    // defeats the uncapped/low-latency render path. If the Android surface does
    // not advertise MAILBOX, WSI creation is rejected and the caller can use
    // the non-WSI overlay path instead.
    VkPresentModeKHR presentMode = VK_PRESENT_MODE_MAILBOX_KHR;
    uint32_t presentModeCount = 0;
    if (g.vk.pfnGetPhysicalDeviceSurfacePresentModesKHR != nullptr &&
        g.vk.pfnGetPhysicalDeviceSurfacePresentModesKHR(
            g.vk.physicalDevice, g.swap.surface, &presentModeCount, nullptr) == VK_SUCCESS &&
        presentModeCount > 0) {
        std::vector<VkPresentModeKHR> modes(presentModeCount);
        if (g.vk.pfnGetPhysicalDeviceSurfacePresentModesKHR(
                g.vk.physicalDevice, g.swap.surface, &presentModeCount, modes.data()) == VK_SUCCESS) {
            bool mailboxAvailable = false;
            for (const auto mode : modes) {
                if (mode == VK_PRESENT_MODE_MAILBOX_KHR) {
                    mailboxAvailable = true;
                    break;
                }
            }
            if (!mailboxAvailable) {
                LOGW("swapchain surface does not advertise VK_PRESENT_MODE_MAILBOX_KHR — WSI disabled (FIFO fallback disabled)");
                destroySwapchain();
                return false;
            }
        } else {
            LOGW("failed to enumerate present modes — WSI disabled (MAILBOX required)");
            destroySwapchain();
            return false;
        }
    } else {
        LOGW("present-mode query unavailable — WSI disabled (MAILBOX required)");
        destroySwapchain();
        return false;
    }

    // MAILBOX normally benefits from at least 3 swapchain images so the
    // compositor can replace an older queued image without blocking the producer.
    // ahead by one while SurfaceFlinger holds the currently-displayed image).
    uint32_t imageCount = caps.minImageCount + 1;
    if (caps.maxImageCount > 0 && imageCount > caps.maxImageCount) {
        imageCount = caps.maxImageCount;
    }
    if (imageCount < 2) imageCount = 2;

    VkExtent2D extent = caps.currentExtent;
    if (extent.width == 0 || extent.width == UINT32_MAX) {
        extent.width = g.outWidth > 0 ? g.outWidth : 1920;
    }
    if (extent.height == 0 || extent.height == UINT32_MAX) {
        extent.height = g.outHeight > 0 ? g.outHeight : 1080;
    }

    // TRANSFER_DST is mandatory for the GPU-native swapchain copy.
    // Some drivers require explicit opt-in via capabilities.supportedUsageFlags.
    if ((caps.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_DST_BIT) == 0) {
        LOGW("swapchain surface does not advertise TRANSFER_DST usage — WSI disabled");
        destroySwapchain();
        return false;
    }

    const VkSwapchainCreateInfoKHR sci2{
        .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
        .surface = g.swap.surface,
        .minImageCount = imageCount,
        .imageFormat = chosen.format,
        .imageColorSpace = chosen.colorSpace,
        .imageExtent = extent,
        .imageArrayLayers = 1,
        .imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT,
        .imageSharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .preTransform = caps.currentTransform,
        .compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
        .presentMode = presentMode,
        .clipped = VK_TRUE,
    };
    LOGI("createSwapchain: vkCreateSwapchainKHR fmt=%d extent=%ux%u imageCount=%u",
         (int)chosen.format, extent.width, extent.height, imageCount);
    if (g.vk.fn.vkCreateSwapchainKHR == nullptr) {
        LOGW("vkCreateSwapchainKHR fn ptr is NULL");
        destroySwapchain();
        return false;
    }
    if (g.vk.fn.vkCreateSwapchainKHR(g.vk.device, &sci2, nullptr,
            &g.swap.swapchain) != VK_SUCCESS) {
        LOGW("vkCreateSwapchainKHR failed");
        destroySwapchain();
        return false;
    }
    LOGI("createSwapchain: swapchain=%p", static_cast<void *>(g.swap.swapchain));

    uint32_t realCount = 0;
    g.vk.fn.vkGetSwapchainImagesKHR(g.vk.device, g.swap.swapchain, &realCount, nullptr);
    g.swap.images.resize(realCount);
    g.vk.fn.vkGetSwapchainImagesKHR(g.vk.device, g.swap.swapchain, &realCount,
                                    g.swap.images.data());

    // Semaphore pools:
    //   acquireSems: one per "in-flight frame" slot (we use realCount + 1)
    //   renderSems:  one per swapchain image (vkQueuePresent waits on this
    //                semaphore for the specific image we rendered to)
    g.swap.acquireSems.resize(realCount + 1, VK_NULL_HANDLE);
    g.swap.renderSems.resize(realCount, VK_NULL_HANDLE);
    const VkSemaphoreCreateInfo semInfo{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
    };
    for (auto &s : g.swap.acquireSems) {
        if (g.vk.fn.vkCreateSemaphore(g.vk.device, &semInfo, nullptr, &s) != VK_SUCCESS) {
            LOGW("vkCreateSemaphore(acquire) failed");
            destroySwapchain();
            return false;
        }
    }
    for (auto &s : g.swap.renderSems) {
        if (g.vk.fn.vkCreateSemaphore(g.vk.device, &semInfo, nullptr, &s) != VK_SUCCESS) {
            LOGW("vkCreateSemaphore(render) failed");
            destroySwapchain();
            return false;
        }
    }

    g.swap.extent = extent;
    g.swap.format = chosen.format;
    g.swap.acquireCursor = 0;
    g.swap.outOfDate = false;
    LOGI("Swapchain ready: %ux%u fmt=%d images=%u mode=%s",
         extent.width, extent.height, (int)chosen.format, realCount,
         "MAILBOX");
    return true;
}

// Blit `src` AHB-backed VkImage to the next swapchain image and present.
// Returns true on success. On VK_ERROR_OUT_OF_DATE_KHR or _SUBOPTIMAL_KHR the
// swapchain is marked dirty; the caller's next blit will recreate it.
bool blitOutputToSwapchain(const AhbImage &src) {
    // Note: Called with g.mu held from blitOutputToWindow.
    if (g.swap.swapchain == VK_NULL_HANDLE) return false;
    if (src.image == VK_NULL_HANDLE) return false;
    if (g.swap.outOfDate) {
        if (!createSwapchain()) return false;
    }

    if (g.swap.acquireSems.empty()) return false;

    // Pick an acquire semaphore from the round-robin pool. Using a single
    // semaphore risks "semaphore already has a pending wait" on fast backs.
    VkSemaphore acquireSem = g.swap.acquireSems[g.swap.acquireCursor];
    g.swap.acquireCursor = (g.swap.acquireCursor + 1) % g.swap.acquireSems.size();

    uint32_t imageIdx = 0;
    const VkResult ar = g.vk.fn.vkAcquireNextImageKHR(g.vk.device, g.swap.swapchain,
        500ULL * 1'000'000ULL,  // 500 ms: generous — SurfaceFlinger normally returns in <16ms
        acquireSem, VK_NULL_HANDLE, &imageIdx);
    if (ar == VK_ERROR_OUT_OF_DATE_KHR) {
        g.swap.outOfDate = true;
        return false;
    }
    if (ar != VK_SUCCESS && ar != VK_SUBOPTIMAL_KHR) {
        LOGW("vkAcquireNextImageKHR returned %d", (int)ar);
        return false;
    }

    if (imageIdx >= g.swap.images.size()) {
        LOGW("vkAcquireNextImageKHR returned OOB index %u (size %zu)", imageIdx, g.swap.images.size());
        return false;
    }

    VkCommandBuffer cb = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    if (!acquireCommandRing(g.vk, cb, fence)) return false;
    const VkCommandBufferBeginInfo bi{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    g.vk.fn.vkBeginCommandBuffer(cb, &bi);

    // Source: AHB-backed image owned by framegen's device between uses.
    // Acquire for TRANSFER_READ, release back to EXTERNAL after blit.
    const uint32_t foreign = VK_QUEUE_FAMILY_EXTERNAL;
    VkImageMemoryBarrier srcAcquire = makeImageBarrier(src.image,
        0, VK_ACCESS_TRANSFER_READ_BIT,
        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        foreign, g.vk.computeFamilyIdx);
    // Destination: swapchain image. Start UNDEFINED → TRANSFER_DST, blit,
    // then → PRESENT_SRC so SurfaceFlinger can pick it up.
    VkImageMemoryBarrier dstToTransfer = makeImageBarrier(g.swap.images[imageIdx],
        0, VK_ACCESS_TRANSFER_WRITE_BIT,
        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED);
    VkImageMemoryBarrier pre[2] = {srcAcquire, dstToTransfer};
    g.vk.fn.vkCmdPipelineBarrier(cb,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 0, nullptr, 0, nullptr, 2, pre);

    static std::atomic<bool> loggedBlitMode{false};

    if (src.extent.width == g.swap.extent.width &&
        src.extent.height == g.swap.extent.height) {
        if (!loggedBlitMode.exchange(true, std::memory_order_relaxed)) {
            LOGW("blitOutputToSwapchain: 1:1 scale detected. Using FAST-PATH vkCmdCopyImage.");
        }
        const VkImageCopy region{
            .srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
            .srcOffset = {0, 0, 0},
            .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
            .dstOffset = {0, 0, 0},
            .extent = {src.extent.width, src.extent.height, 1},
        };
        g.vk.fn.vkCmdCopyImage(cb,
            src.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            g.swap.images[imageIdx], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1, &region);
    } else {
        if (!loggedBlitMode.exchange(true, std::memory_order_relaxed)) {
            LOGW("blitOutputToSwapchain: Scaled output detected (%ux%u -> %ux%u). Using VK_FILTER_LINEAR vkCmdBlitImage.",
                 src.extent.width, src.extent.height, g.swap.extent.width, g.swap.extent.height);
        }
        const VkImageBlit region{
            .srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
            .srcOffsets = {{0, 0, 0}, {
                static_cast<int32_t>(src.extent.width),
                static_cast<int32_t>(src.extent.height), 1,
            }},
            .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
            .dstOffsets = {{0, 0, 0}, {
                static_cast<int32_t>(g.swap.extent.width),
                static_cast<int32_t>(g.swap.extent.height), 1,
            }},
        };
        g.vk.fn.vkCmdBlitImage(cb,
            src.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            g.swap.images[imageIdx], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1, &region, VK_FILTER_LINEAR);
    }

    VkImageMemoryBarrier srcRelease = makeImageBarrier(src.image,
        VK_ACCESS_TRANSFER_READ_BIT, 0,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
        g.vk.computeFamilyIdx, foreign);
    VkImageMemoryBarrier dstToPresent = makeImageBarrier(g.swap.images[imageIdx],
        VK_ACCESS_TRANSFER_WRITE_BIT, 0,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
        VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED);
    VkImageMemoryBarrier post[2] = {srcRelease, dstToPresent};
    g.vk.fn.vkCmdPipelineBarrier(cb,
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
        0, 0, nullptr, 0, nullptr, 2, post);

    g.vk.fn.vkEndCommandBuffer(cb);

    VkSemaphore renderSem = g.swap.renderSems[imageIdx];
    const VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    const VkSubmitInfo si{
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &acquireSem,
        .pWaitDstStageMask = &waitStage,
        .commandBufferCount = 1,
        .pCommandBuffers = &cb,
        .signalSemaphoreCount = 1,
        .pSignalSemaphores = &renderSem,
    };
    if (g.vk.fn.vkQueueSubmit(g.vk.computeQueue, 1, &si, fence) != VK_SUCCESS) {
        LOGW("vkQueueSubmit(swapchain blit) failed");
        return false;
    }
    // Arm the ring fence manually — submitCommandRing would have done this
    // but we bypassed it to add the semaphore wait/signal pair.
    for (uint32_t i = 0; i < kCommandRingSize; ++i) {
        if (g.vk.ringFences[i] == fence) {
            g.vk.ringFenceArmed[i] = true;
            break;
        }
    }

    const VkPresentInfoKHR pi{
        .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &renderSem,
        .swapchainCount = 1,
        .pSwapchains = &g.swap.swapchain,
        .pImageIndices = &imageIdx,
    };
    const VkResult pr = g.vk.fn.vkQueuePresentKHR(g.vk.computeQueue, &pi);
    if (pr == VK_ERROR_OUT_OF_DATE_KHR || pr == VK_SUBOPTIMAL_KHR) {
        // Note for next pass — don't fall back this frame (we already posted).
        g.swap.outOfDate = true;
    } else if (pr != VK_SUCCESS) {
        LOGW("vkQueuePresentKHR returned %d", (int)pr);
    }
    return true;
}

// Transition a freshly-created AHB-backed inSlot image UNDEFINED→GENERAL
// and release ownership to VK_QUEUE_FAMILY_EXTERNAL. This must be called once
// per inSlot after createAhbImage so that every subsequent copyAhbImage can
// acquire with oldLayout=GENERAL, preserving Mali AFRC state across frames.
void initInSlotImageLayout(VkImage image) {
    // Transition UNDEFINED→GENERAL and release to EXTERNAL in one barrier.
    const bool ok = runTransientCommands([&](VkCommandBuffer cb) {
        const VkImageMemoryBarrier barrier = makeImageBarrier(image, 0, 0,
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
            g.vk.computeFamilyIdx, VK_QUEUE_FAMILY_EXTERNAL);
        g.vk.fn.vkCmdPipelineBarrier(cb,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
            0, 0, nullptr, 0, nullptr, 1, &barrier);
    });
    if (!ok) {
        LOGE("initInSlotImageLayout: transient command submission failed");
        return;
    }
    LOGI("initInSlotImageLayout: image=%p initialized to GENERAL/EXTERNAL", (void*)image);
}

// Copy `src` AhbImage into `dst` AhbImage on the compute queue using a
// transient command buffer. Both images are AHB-backed so layouts are
// EXTERNAL initially; we transition to TRANSFER_DST/SRC, copy, transition
// back to GENERAL so framegen can read them.
//
// Uses transient alloc+free per call rather than the session's CB ring:
// on Adreno the per-frame vkAllocateCommandBuffers/vkFreeCommandBuffers
// pair runs in single-digit microseconds, while vkResetCommandBuffer +
// vkWaitForFences (the ring path) was measurably slower in field testing.
bool copyAhbImage(const AhbImage &src, const AhbImage &dst) {
    // Per VK_ANDROID_external_memory_android_hardware_buffer: AHB-backed
    // images are conceptually owned by the FOREIGN_EXT queue family between
    // uses. Both src (just received from MediaProjection / ImageReader) and
    // dst (last touched by framegen on its own device) must be acquired from
    // FOREIGN_EXT before our compute queue can touch them — this is what
    // tells the driver "the foreign side is done writing, copy what's there".
    // Keep the Android-side session aligned with framegen's AHB import path:
    // both devices transfer ownership through the generic EXTERNAL family.
    const uint32_t foreign = VK_QUEUE_FAMILY_EXTERNAL;
    const uint32_t w = std::min(src.extent.width,  dst.extent.width);
    const uint32_t h = std::min(src.extent.height, dst.extent.height);

    const bool ok = runTransientCommands([&](VkCommandBuffer cb) {
        VkImageMemoryBarrier toSrc = makeImageBarrier(src.image,
            0, VK_ACCESS_TRANSFER_READ_BIT,
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            foreign, g.vk.computeFamilyIdx);
        // dst.oldLayout is always GENERAL: set by initInSlotImageLayout and
        // preserved by releaseDst below.
        VkImageMemoryBarrier toDst = makeImageBarrier(dst.image,
            0, VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            foreign, g.vk.computeFamilyIdx);
        VkImageMemoryBarrier preBarriers[2] = {toSrc, toDst};
        g.vk.fn.vkCmdPipelineBarrier(cb,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
            0, 0, nullptr, 0, nullptr, 2, preBarriers);

        const VkImageCopy region{
            .srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
            .srcOffset = {0, 0, 0},
            .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
            .dstOffset = {0, 0, 0},
            .extent = {w, h, 1},
        };
        g.vk.fn.vkCmdCopyImage(cb,
            src.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            dst.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1, &region);

        // Release dst back to FOREIGN so framegen (on its own device) can
        // acquire it cleanly via its own image-memory-barrier. We also
        // release src (we don't need it anymore — destroyAhbImage tears
        // down our VkImage wrapper, but the AHB itself stays alive in the
        // ImageReader's pool).
        VkImageMemoryBarrier releaseDst = makeImageBarrier(dst.image,
            VK_ACCESS_TRANSFER_WRITE_BIT, 0,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
            g.vk.computeFamilyIdx, foreign);
        VkImageMemoryBarrier releaseSrc = makeImageBarrier(src.image,
            VK_ACCESS_TRANSFER_READ_BIT, 0,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
            g.vk.computeFamilyIdx, foreign);
        VkImageMemoryBarrier postBarriers[2] = {releaseDst, releaseSrc};
        g.vk.fn.vkCmdPipelineBarrier(cb,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
            0, 0, nullptr, 0, nullptr, 2, postBarriers);

        // Flush GPU L2 to DRAM before the CPU reads via AHardwareBuffer_lock.
        // Mali-G77 does not automatically flush the L2 on vkQueueWaitIdle, so
        // AHardwareBuffer_lock(fence=-1) reads stale zeroes without this barrier.
        const VkMemoryBarrier hostFlush{
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
        };
        g.vk.fn.vkCmdPipelineBarrier(cb,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
            0, 1, &hostFlush, 0, nullptr, 0, nullptr);
    });
    if (!ok) {
        LOGE("copyAhbImage: transient command submission failed dst=%ux%u",
             dst.extent.width, dst.extent.height);
    }
    return ok;
}

bool blitAhbImageGpu(const AhbImage &src, const AhbImage &dst) {
    if (src.image == VK_NULL_HANDLE || dst.image == VK_NULL_HANDLE) return false;
    const uint32_t foreign = VK_QUEUE_FAMILY_EXTERNAL;

    return runTransientCommands([&](VkCommandBuffer cb) {
        VkImageMemoryBarrier toSrc = makeImageBarrier(src.image,
            0, VK_ACCESS_TRANSFER_READ_BIT,
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            foreign, g.vk.computeFamilyIdx);
        VkImageMemoryBarrier toDst = makeImageBarrier(dst.image,
            0, VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            foreign, g.vk.computeFamilyIdx);
        VkImageMemoryBarrier preBarriers[2] = {toSrc, toDst};
        g.vk.fn.vkCmdPipelineBarrier(cb,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
            0, 0, nullptr, 0, nullptr, 2, preBarriers);

        const VkImageBlit region{
            .srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
            .srcOffsets = {{0, 0, 0}, {
                static_cast<int32_t>(src.extent.width),
                static_cast<int32_t>(src.extent.height),
                1,
            }},
            .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
            .dstOffsets = {{0, 0, 0}, {
                static_cast<int32_t>(dst.extent.width),
                static_cast<int32_t>(dst.extent.height),
                1,
            }},
        };
        g.vk.fn.vkCmdBlitImage(cb,
            src.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            dst.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1, &region, VK_FILTER_LINEAR);

        VkImageMemoryBarrier releaseSrc = makeImageBarrier(src.image,
            VK_ACCESS_TRANSFER_READ_BIT, 0,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
            g.vk.computeFamilyIdx, foreign);
        VkImageMemoryBarrier releaseDst = makeImageBarrier(dst.image,
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_MEMORY_READ_BIT,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
            g.vk.computeFamilyIdx, foreign);
        VkImageMemoryBarrier postBarriers[2] = {releaseSrc, releaseDst};
        g.vk.fn.vkCmdPipelineBarrier(cb,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
            0, 0, nullptr, 0, nullptr, 2, postBarriers);
    });
}

bool processRealFrameIntoSlot(const AhbImage &src, const AhbImage &dst) {
    if (copyAhbImage(src, dst)) {
        return true;
    }
    LOGW("Raw copy failed; falling back to Vulkan linear blit");
    return blitAhbImageGpu(src, dst);
}

bool initFramegen(const char *cacheDir) {
    const std::string cache(cacheDir ? cacheDir : "");

    // There are two shader sources, in order of preference for a given
    // session:
    //
    //   1. FP16 SPIR-V (Lossless.dll IDs 304..351): user-requested half-precision.
    //   2. FP32 SPIR-V (Lossless.dll IDs 353..400): default for a normal session.
    //
    // FP32 SPIR-V is the default: it uses `OpMemoryModel Logical GLSL450`
    // with NO VulkanMemoryModel capability — verified across the entire
    // 353..400 range via _analysis/dump_fp32_spv.py / _analysis/fp32/. That
    // makes it work on devices that lack vulkanMemoryModel (Mali Bifrost/
    // Valhall G57/G68/G77), which a VMM-requiring shader would DEVICE_LOST
    // on at first compute dispatch (see Mali-G57 field log "presentContext
    // threw: Unable to submit command buffer (error -4)").
    const bool fp16Available = fp16_shaders_available(cache);
    const bool fp32SpirvAvailable = fp32_spirv_shaders_available(cache);

    bool useFp16 = g.framegenFp16 && fp16Available;
    if (g.framegenFp16 && !useFp16) {
        LOGW("FP16 framegen requested but SPIR-V FP16 cache is incomplete in %s — falling back",
             cache.c_str());
    }

    const bool useFp32Spirv = !useFp16 && fp32SpirvAvailable;
    if (useFp16) {
        LOGI("Loading framegen shaders from FP16 SPIR-V cache (Lossless.dll resource IDs 304..351)");
    } else if (useFp32Spirv) {
        LOGI("Loading framegen shaders from FP32 SPIR-V cache (Lossless.dll resource IDs 353..400)");
    } else {
        LOGE("Neither FP16 nor FP32 SPIR-V cache is available in %s — "
             "is Lossless Scaling up to date? (requires 3.2.2.0 or newer)", cache.c_str());
        return false;
    }

    auto loader = [cache, useFp16](const std::string &name) -> std::vector<uint8_t> {
        // Framegen requests shaders by symbolic name (e.g. "p_mipmaps");
        // we cache them on disk by numeric resource ID. Each of the two
        // sources has its own ID range and on-disk subdirectory, both keyed
        // off the shared base id via constant offsets.
        uint32_t id = 0;
        ShaderCache source = ShaderCache::Fp32Spirv;
        if (useFp16) {
            id = shader_name_to_resource_id_fp16(name);
            source = ShaderCache::Fp16Spirv;
        } else {
            id = shader_name_to_resource_id_fp32_spirv(name);
            source = ShaderCache::Fp32Spirv;
        }
        if (id == 0) {
            LOGE("Unknown shader name '%s' from framegen", name.c_str());
            return {};
        }
        auto spirv = load_cached_spirv(cache, id, source);

        // Per-shader fallback: FP16 → FP32 SPIR-V. A single missing blob
        // shouldn't kill the session if the other tier can serve it.
        if (spirv.empty() && source == ShaderCache::Fp16Spirv) {
            const uint32_t fp32Id = shader_name_to_resource_id_fp32_spirv(name);
            if (fp32Id != 0) {
                LOGW("FP16 shader '%s' (id %u) missing — falling back to FP32 SPIR-V (id %u)",
                     name.c_str(), id, fp32Id);
                spirv = load_cached_spirv(cache, fp32Id, ShaderCache::Fp32Spirv);
            }
        }
        if (spirv.empty()) {
            LOGE("Shader '%s' (id %u) missing from cache (%s)",
                 name.c_str(), id, cache.c_str());
        }
        return spirv;
    };

    try {
        if (g.performanceMode) {
            LSFG_3_1P::initialize(g.vk.deviceUuid, g.hdr, g.flowScale,
                                  static_cast<uint64_t>(g.multiplier), loader);
        } else {
            LSFG_3_1::initialize(g.vk.deviceUuid, g.hdr, g.flowScale,
                                 static_cast<uint64_t>(g.multiplier), loader);
        }
        return true;
    } catch (const std::exception &e) {
        LOGE("LSFG_3_1::initialize threw: %s — likely missing extension or shader. Continuing in capture-only mode.", e.what());
        return false;
    } catch (...) {
        LOGE("LSFG_3_1::initialize threw unknown exception");
        return false;
    }
}

bool createFramegenContext() {
    // Pass AHardwareBuffer pointers to framegen's Android variant. Framegen
    // imports them in its own VkDevice and shares pixel storage with us via
    // the AHB itself — we keep ownership and refcount on the Android side.
    if (g.inSlot[0].ahb == nullptr || g.inSlot[1].ahb == nullptr) {
        LOGE("Input AhbImages have no AHB pointer");
        return false;
    }
    std::vector<AHardwareBuffer*> outAhbs;
    outAhbs.reserve(g.outputs.size());
    for (auto &o : g.outputs) {
        if (o.ahb == nullptr) {
            LOGE("Output AhbImage missing AHB pointer");
            return false;
        }
        outAhbs.push_back(o.ahb);
    }

    try {
        if (g.performanceMode) {
            g.framegenCtxId = LSFG_3_1P::createContextFromAHB(
                g.inSlot[0].ahb, g.inSlot[1].ahb, outAhbs,
                g.inSlot[0].extent, g.inSlot[0].format);
        } else {
            g.framegenCtxId = LSFG_3_1::createContextFromAHB(
                g.inSlot[0].ahb, g.inSlot[1].ahb, outAhbs,
                g.inSlot[0].extent, g.inSlot[0].format);
        }
    } catch (const std::exception &e) {
        LOGE("createContextFromAHB threw: %s", e.what());
        return false;
    }
    return true;
}

// Output blit. Has two paths:
//
//  1. GPU fast path (WSI): vkCmdBlitImage from the output AHB's VkImage
//     straight into the next swapchain image, then vkQueuePresentKHR. Zero
//     CPU touch of the pixel data. Saves ~3-5 ms/blit at 1080p.
//
//  The CPU/software blit fallback (AHardwareBuffer_lock(CPU_READ) +
//  ANativeWindow_lock + per-row memcpy) has been REMOVED. This is now a
//  GPU-only presentation path: if the WSI swapchain isn't available on the
//  current surface (missing extension, driver rejects the surface, compute
//  queue can't present, etc.) the frame is dropped instead of falling back
//  to CPU. This trades away compatibility with devices/drivers that reject
//  the WSI path (e.g. the Mali-G57 window-in-use / Adreno present quirks
//  documented around createSwapchain()) for guaranteeing every posted frame
//  went through the GPU. On an affected device this means no frames get
//  posted at all rather than a degraded CPU blit — check logcat for
//  "blit dropped" warnings if the overlay goes blank.
//
// Returns true iff the frame was actually posted (recordOverlayPost() ran).
// Callers use this to keep generatedFrames/postedFrames in lockstep instead
// of assuming every attempted blit succeeds — see the comment at the
// generatedFrames.fetch_add() call site in processFrame().
bool blitOutputToWindow(const AhbImage &out) {
    if (g.outWindow == nullptr || out.ahb == nullptr) return false;

    if (!kEnableWsiSwapchain || !g.vk.hasSwapchain || g.swap.disabledForSession) {
        LOGW("blit dropped: WSI swapchain unavailable on this surface (no fallback)");
        return false;
    }

    std::lock_guard<std::mutex> lock(g.mu);
    if (g.swap.swapchain == VK_NULL_HANDLE) {
        if (!createSwapchain()) {
            g.swap.disabledForSession = true;
            LOGW("blit dropped: createSwapchain failed on this surface (no fallback)");
            return false;
        }
    }
    if (blitOutputToSwapchain(out)) {
        recordOverlayPost();
        return true;
    }
    LOGW("blit dropped: blitOutputToSwapchain failed (no fallback)");
    return false;
}

#ifdef LSFG_HAVE_NCNN
// Runs the ncnn AI backend for one frame pair. The model inference is GPU-only;
// input slots (oldSlot = previous capture, newSlot = just-copied current
// capture — matching the same chronological order LSFG's own frameIdx
// tracking uses), calls NcnnInterpolator::interpolate(), and CPU-writes each
// resulting frame straight into g.outputs[i]'s AHB so the existing
// timedBlit()/blitOutputToWindow() loop in processFrame() can present them
// exactly like LSFG-generated output — no other code downstream needs to
// know which backend produced the pixels.
//
// AHardwareBuffer row stride can exceed width*4 bytes (GPU alignment
// padding), so every lock goes through a tightly-packed staging buffer —
// NcnnInterpolator's interpolate() contract assumes no stride.
//
// Input-side host visibility: no extra barrier is needed here.
// processRealFrameIntoSlot() (via copyAhbImage) already issues a
// TRANSFER_WRITE -> HOST_READ barrier before returning, so by the time this
// runs, whichever slot was just written this iteration is safe to
// CPU-read, and the other slot was already flushed on a prior iteration and
// hasn't been GPU-written since.
bool runAiInterpolate(int oldSlot, int newSlot, uint32_t w, uint32_t h) {
    const bool useIfrnet = (g.aiEngine == 1);
    if (useIfrnet) {
        if (g.aiIfrnet == nullptr || !g.aiIfrnet->isLoaded()) return false;
    } else {
        if (g.ai == nullptr || !g.ai->isLoaded()) return false;
    }

    auto lockRead = [](const AhbImage &img, std::vector<uint8_t> &staging) -> bool {
        void *ptr = nullptr;
        if (AHardwareBuffer_lock(img.ahb, AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN,
                                  -1, nullptr, &ptr) != 0 || ptr == nullptr) {
            return false;
        }
        AHardwareBuffer_Desc desc{};
        AHardwareBuffer_describe(img.ahb, &desc);
        const size_t strideBytes = static_cast<size_t>(desc.stride) * 4; // stride is in pixels
        const size_t rowBytes = static_cast<size_t>(desc.width) * 4;
        staging.resize(static_cast<size_t>(desc.height) * rowBytes);
        const uint8_t *src = static_cast<const uint8_t *>(ptr);
        for (uint32_t row = 0; row < desc.height; ++row) {
            std::memcpy(staging.data() + static_cast<size_t>(row) * rowBytes,
                        src + static_cast<size_t>(row) * strideBytes, rowBytes);
        }
        AHardwareBuffer_unlock(img.ahb, nullptr);
        return true;
    };

    auto lockWrite = [](const AhbImage &img, const uint8_t *staging) -> bool {
        void *ptr = nullptr;
        if (AHardwareBuffer_lock(img.ahb, AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN,
                                  -1, nullptr, &ptr) != 0 || ptr == nullptr) {
            return false;
        }
        AHardwareBuffer_Desc desc{};
        AHardwareBuffer_describe(img.ahb, &desc);
        const size_t strideBytes = static_cast<size_t>(desc.stride) * 4;
        const size_t rowBytes = static_cast<size_t>(desc.width) * 4;
        uint8_t *dst = static_cast<uint8_t *>(ptr);
        for (uint32_t row = 0; row < desc.height; ++row) {
            std::memcpy(dst + static_cast<size_t>(row) * strideBytes,
                        staging + static_cast<size_t>(row) * rowBytes, rowBytes);
        }
        // fence == nullptr: block until the write is visible to other
        // consumers (the WSI swapchain's vkCmdBlitImage right after this).
        AHardwareBuffer_unlock(img.ahb, nullptr);
        return true;
    };

    const size_t frameBytes = static_cast<size_t>(w) * h * 4u;
    if (g.aiInputAStaging.size() != frameBytes) g.aiInputAStaging.resize(frameBytes);
    if (g.aiInputCStaging.size() != frameBytes) g.aiInputCStaging.resize(frameBytes);

    if (!lockRead(g.inSlot[oldSlot], g.aiInputAStaging) ||
        !lockRead(g.inSlot[newSlot], g.aiInputCStaging)) {
        LOGE("AI backend: failed to bridge input AHBs");
        return false;
    }

    // Reuse output staging allocations across frames. This removes a large
    // per-frame allocation/free cycle at 1080p+ and prevents heap fragmentation.
    const size_t outputCount = g.outputs.size();
    if (g.aiOutputStaging.size() != outputCount) {
        g.aiOutputStaging.resize(outputCount);
        g.aiOutputPtrs.resize(outputCount);
    }
    for (size_t i = 0; i < outputCount; ++i) {
        if (g.aiOutputStaging[i].size() != frameBytes)
            g.aiOutputStaging[i].resize(frameBytes);
        g.aiOutputPtrs[i] = g.aiOutputStaging[i].data();
    }

    // g.outputs.size() == g.multiplier (extra frames per pair); NcnnInterpolator
    // wants the total segment count (extra + 1).
    const int totalMult = g.multiplier + 1;

    // g.flowScale was inverted at init time for LSFG's convention
    // (g.flowScale = 1/userFlow); both interpolators want the plain
    // user-facing 0..1 fraction back, so invert it again here. RIFE and
    // IFRNet share the exact same interpolate() call shape (see the
    // "intentionally interchangeable" note in IfrnetInterpolator.hpp), so
    // only the object the call is made on differs.
    const int rc = useIfrnet
        ? g.aiIfrnet->interpolate(g.aiInputAStaging.data(), g.aiInputCStaging.data(),
                                   static_cast<int>(w), static_cast<int>(h),
                                   g.aiOutputPtrs.data(), totalMult, 1.0f / g.flowScale)
        : g.ai->interpolate(g.aiInputAStaging.data(), g.aiInputCStaging.data(),
                             static_cast<int>(w), static_cast<int>(h),
                             g.aiOutputPtrs.data(), totalMult, 1.0f / g.flowScale);
    if (rc != kNcnnOk) {
        LOGE("AI backend interpolate() failed rc=%d", rc);
        return false;
    }

    for (size_t i = 0; i < g.outputs.size(); ++i) {
        if (!lockWrite(g.outputs[i], g.aiOutputStaging[i].data())) {
            LOGE("AI backend: failed to CPU-lock output AHB %zu for write", i);
            return false;
        }
    }
    return true;
}
#endif // LSFG_HAVE_NCNN


bool ensurePostImage(uint32_t width, uint32_t height) {
    if (g.postImage.ahb != nullptr &&
            g.postImage.extent.width == width &&
            g.postImage.extent.height == height) {
        return true;
    }
    destroyAhbImage(g.vk, g.postImage);
    const int rc = createAhbImage(g.vk, width, height, VK_FORMAT_R8G8B8A8_UNORM,
                                  g.postImage);
    if (rc != kOk) {
        LOGW("post-process AHB allocation failed rc=%d size=%ux%u", rc, width, height);
        return false;
    }
    return true;
}

bool lockAhbTight(const AhbImage &img, std::vector<uint8_t> &dst) {
    if (img.ahb == nullptr) return false;
    AHardwareBuffer_Desc desc{};
    AHardwareBuffer_describe(img.ahb, &desc);
    if (desc.format != AHARDWAREBUFFER_FORMAT_R8G8B8A8_UNORM) return false;
    const size_t rowBytes = static_cast<size_t>(desc.width) * 4u;
    const size_t strideBytes = static_cast<size_t>(desc.stride) * 4u;
    dst.resize(static_cast<size_t>(desc.height) * rowBytes);
    void *ptr = nullptr;
    if (AHardwareBuffer_lock(img.ahb, AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN,
            -1, nullptr, &ptr) != 0 || ptr == nullptr) {
        return false;
    }
    const auto *src = static_cast<const uint8_t *>(ptr);
    for (uint32_t y = 0; y < desc.height; ++y) {
        std::memcpy(dst.data() + static_cast<size_t>(y) * rowBytes,
                    src + static_cast<size_t>(y) * strideBytes, rowBytes);
    }
    AHardwareBuffer_unlock(img.ahb, nullptr);
    return true;
}

bool writeAhbTight(const AhbImage &img, const uint8_t *src, size_t srcBytes) {
    if (img.ahb == nullptr) return false;
    AHardwareBuffer_Desc desc{};
    AHardwareBuffer_describe(img.ahb, &desc);
    if (desc.format != AHARDWAREBUFFER_FORMAT_R8G8B8A8_UNORM) return false;
    const size_t rowBytes = static_cast<size_t>(desc.width) * 4u;
    const size_t strideBytes = static_cast<size_t>(desc.stride) * 4u;
    if (src == nullptr || srcBytes < static_cast<size_t>(desc.height) * rowBytes) return false;
    void *ptr = nullptr;
    if (AHardwareBuffer_lock(img.ahb, AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN,
            -1, nullptr, &ptr) != 0 || ptr == nullptr) {
        return false;
    }
    auto *dst = static_cast<uint8_t *>(ptr);
    for (uint32_t y = 0; y < desc.height; ++y) {
        std::memcpy(dst + static_cast<size_t>(y) * strideBytes,
                    src + static_cast<size_t>(y) * rowBytes, rowBytes);
    }
    AHardwareBuffer_unlock(img.ahb, nullptr);
    return true;
}

// Final display-stage pipeline only. Framegen has already completed before this
// function is called. The returned AHB is the only object handed to the WSI
// presenter. No result is ever copied back into inSlot[] or outputs[].
const AhbImage *prepareDisplayFrame(const AhbImage &src) {
    const bool anyPost = g.gpuPostProcessing || g.npuPostProcessing || g.cpuPostProcessing;
    if (!anyPost) return &src;

    uint32_t dstW = src.extent.width;
    uint32_t dstH = src.extent.height;
    if (g.gpuPostProcessing) {
        const float scale = std::clamp(g.gpuUpscaleFactor, 1.0f, 2.0f);
        dstW = std::max(1u, static_cast<uint32_t>(std::lround(dstW * scale)));
        dstH = std::max(1u, static_cast<uint32_t>(std::lround(dstH * scale)));
    }

    if (!ensurePostImage(dstW, dstH)) {
        return &src;
    }

    const AhbImage *current = &src;
    if (g.gpuPostProcessing) {
        const GpuPostProcessConfig cfg{
            .method = std::clamp(g.gpuMethod, 0, 15),
            .sharpness = std::clamp(g.gpuSharpness, 0.0f, 1.0f),
            .strength = std::clamp(g.gpuStrength, 0.0f, 1.0f),
        };
        if (g.gpuPost.process(g.vk, src, g.postImage, cfg)) {
            current = &g.postImage;
        } else {
            LOGW("GPU display post-process failed; using unprocessed frame for this stage");
            if (!g.npuPostProcessing && !g.cpuPostProcessing) return &src;
        }
    }

    if (!g.npuPostProcessing && !g.cpuPostProcessing) return current;

    if (!lockAhbTight(*current, g.postInputStaging)) {
        LOGW("display post-process: CPU readback failed; using current frame");
        return current;
    }

    const uint32_t w = current->extent.width;
    const uint32_t h = current->extent.height;
    const uint32_t stride = w * 4u;
    const uint8_t *pipelineSrc = g.postInputStaging.data();
    bool npuDone = false;

    if (g.npuPostProcessing && !g.npuStageFailed) {
        const NnapiPostProcessConfig npuCfg{
            .preset = static_cast<NpuPreset>(std::clamp(g.npuPreset, 0, 4)),
            .upscaleFactor = 1, // GPU owns the optional upscale stage.
            .amount = 0.65f,
            .radius = 1.0f,
            .threshold = 0.0f,
            .fp16 = true,
        };
        if (g.npuPost.configure(w, h, npuCfg) &&
                g.npuPost.outputWidth() == w &&
                g.npuPost.outputHeight() == h) {
            g.postOutputStaging.resize(static_cast<size_t>(w) * h * 4u);
            npuDone = g.npuPost.processRgba8888(
                pipelineSrc, stride, g.postOutputStaging.data(), stride);
            if (npuDone) pipelineSrc = g.postOutputStaging.data();
        }
        if (!npuDone) {
            LOGW("NPU display post-process unavailable/rejected for %ux%u; skipping NPU stage for this session",
                 w, h);
            g.npuStageFailed = true;
        }
    }

    if (g.cpuPostProcessing) {
        g.postCpuStaging.resize(static_cast<size_t>(w) * h * 4u);
        const CpuPostProcessConfig cpuCfg{
            .preset = static_cast<CpuPreset>(std::clamp(g.cpuPreset, 0, 6)),
            .strength = std::clamp(g.cpuStrength, 0.0f, 1.0f),
            .saturation = std::clamp(g.cpuSaturation, 0.0f, 1.0f),
            .vibrance = std::clamp(g.cpuVibrance, 0.0f, 1.0f),
            .vignette = std::clamp(g.cpuVignette, 0.0f, 1.0f),
        };
        g.cpuPost.configure(w, h, cpuCfg);
        if (g.cpuPost.process(pipelineSrc, stride, g.postCpuStaging.data(), stride,
                              w, h)) {
            pipelineSrc = g.postCpuStaging.data();
        } else {
            LOGW("CPU display post-process returned no output; keeping prior stage");
        }
    }

    const size_t pipelineBytes = static_cast<size_t>(w) * h * 4u;
    if (!writeAhbTight(g.postImage, pipelineSrc, pipelineBytes)) {
        LOGW("display post-process: CPU writeback failed; using current frame");
        return current;
    }
    return &g.postImage;
}

// Applies big-core affinity + elevated scheduling priority to whichever
// thread is currently calling into processFrame(). Runs once per OS thread
// (thread_local latch) rather than once per process, since pushFrame() is no
// longer funneled through a single dedicated worker thread — it now runs
// directly on the calling capture thread (the LSFG ImageReader callback
// thread, or a Shizuku/root binder thread), whichever that turns out to be.
void applyHotPathThreadTuning() {
    thread_local bool tuned = false;
    if (tuned) return;
    tuned = true;

    // Prefer big/performance cores for every CPU-side operation in this hot
    // path. This is thread-local affinity; it does not require root and does
    // not force unrelated application threads onto the big cluster.
    if (!g.cpuPolicy.useBigCores()) {
        LOGW("applyHotPathThreadTuning: big-core affinity unavailable; using all online CPUs");
        g.cpuPolicy.useAllCores();
    }
    // Elevate scheduling priority: this thread runs the whole capture→framegen
    // submit→wait→blit pipeline and must wake up promptly. At the default
    // priority (SCHED_OTHER, nice 0) it competes evenly with every other
    // thread on the device — the captured app's own render thread, GC,
    // system services — any of which can delay its wakeup and cause a
    // late/dropped present regardless of how fast the pipeline itself is.
    //
    // Pushed to ANDROID_PRIORITY_URGENT_AUDIO (-19) rather than the previous
    // URGENT_DISPLAY (-8) — the strongest nice value an app can self-assign
    // without root — on an explicit "give this everything, thermal/scheduling
    // fairness be damned" request. This is a deliberate tradeoff: it can starve
    // less latency-sensitive background threads (this device's, and other
    // apps') and burns more power/heat for it. Requires no special permission —
    // any process may renice its own threads into this range.
    if (setpriority(PRIO_PROCESS, gettid(), ANDROID_PRIORITY_URGENT_AUDIO) != 0) {
        // Some OEM kernels cap the range unprivileged apps can self-renice
        // into; if the aggressive value is rejected, fall back one step
        // rather than silently staying at the SCHED_OTHER default.
        if (setpriority(PRIO_PROCESS, gettid(), ANDROID_PRIORITY_URGENT_DISPLAY) != 0) {
            LOGW("applyHotPathThreadTuning: setpriority(URGENT_AUDIO/URGENT_DISPLAY) both failed, "
                 "staying at default priority");
        }
    }
}

// ---- Frame-time profiling ------------------------------------------------
//
// Rolling accumulators over a 60-frame window. The 4 segments are:
//   copy     — importAhbImage + processRealFrameIntoSlot (input prep)
//   present  — LSFG_3_1::presentContext (framegen submit; usually <1 ms)
//   waitIdle — LSFG_3_1::waitIdle (cross-device sync; biggest single cost)
//   blit     — output blit loop (CPU memcpy + ANativeWindow_unlockAndPost,
//              one per generated + real frame)
//   total    — frameWorkStartedAt → end of blit loop
// There is no pacing sleep in the render path, and — since pushFrame() now
// calls straight into this function with no queue in between — queueNs is
// always ~0. Kept (rather than deleted) so the existing HUD/diagnostics
// readout (getAverageQueueMs) honestly reports "no queue" instead of the
// field just vanishing.
struct ProfileAccum {
    int64_t copyNs    = 0;
    int64_t presentNs = 0;
    int64_t waitIdleNs= 0;
    int64_t blitNs    = 0;
    int64_t totalNs   = 0;
    int64_t queueNs   = 0;
    int64_t captureToDisplayNs = 0;
    uint32_t blitCount = 0;
    uint32_t samples  = 0;
};
constexpr uint32_t kProfileWindow = 60;

// Processes exactly one capture frame, synchronously, on the calling
// thread: import → copy into the ping-pong slot → framegen (or AI) →
// blit every output. This is the entire former workerThread loop body,
// called directly from pushFrame() — there is no queue, mailbox, or
// separate thread between capture and this. Must be called with
// g.frameMu held.
void processFrame(AHardwareBuffer *ahb, int64_t captureTimestampNs) {
    // Persist the rolling profiling window and the reusable empty-semaphore
    // vector across calls (this function may be called thousands of times a
    // second) without resurrecting a dedicated long-lived thread for it.
    static ProfileAccum prof{};
    static const std::vector<int> noOutputSemaphores{};

    applyHotPathThreadTuning();

    const auto frameWorkStartedAt = State::Clock::now();

    // Wrap the imported AHB (read-only from our perspective) and copy
    // into the oldest input slot on-the-fly.
    // AHB import cache: MediaProjection rotates a small pool (2-4) of AHBs.
    // Reuse the cached VkImage+VkDeviceMemory instead of allocating per frame.
    AhbImage src{};
    bool srcFromCache = false;
    {
        auto it = g.ahbImportCache.find(ahb);
        if (it != g.ahbImportCache.end()) {
            src = it->second;
            srcFromCache = true;
        }
    }
    if (!srcFromCache) {
        const int rc = importAhbImage(g.vk, ahb, src);
        if (rc != kOk) {
            LOGW("importAhbImage failed rc=%d", rc);
            return;
        }
        // Cache the import; acquire an extra ref so the cache outlives the frame.
        AHardwareBuffer_acquire(ahb);
        if (g.ahbImportCache.size() >= State::kAhbCacheMax) {
            auto oldest = g.ahbImportCache.begin();
            destroyAhbImage(g.vk, oldest->second);
            AHardwareBuffer_release(oldest->first);
            g.ahbImportCache.erase(oldest);
        }
        g.ahbImportCache[ahb] = src;
    }

    // Framegen tracks an internal frameIdx and treats inImg_0 as the
    // "current" frame when frameIdx % 2 == 0, inImg_1 when % 2 == 1
    // (see lsfg-vk-android/framegen/v3.1_include/v3_1/context.hpp:61).
    // We must write the new capture into the slot framegen will consider
    // "current" at the upcoming present — otherwise it computes the optical
    // flow backwards (treating yesterday's frame as "now"), which collapses
    // moving objects like a head or torso.
    const int newSlot = (g.presentsDone % 2 == 0) ? 0 : 1;

    // Bootstrap: the very first capture has no predecessor, so seed BOTH
    // slots with the same pixels. That makes the optical flow for the first
    // present a no-op (same image on both inputs) and the output equals the
    // input — clean instead of ghosted.
    if (g.framesCopied == 0) {
        if (!processRealFrameIntoSlot(src, g.inSlot[0]) ||
                !processRealFrameIntoSlot(src, g.inSlot[1])) {
            LOGW("bootstrap frame input processing failed");
            // Evict from cache on failure — state may be corrupt.
            if (g.ahbImportCache.erase(ahb)) AHardwareBuffer_release(ahb);
            destroyAhbImage(g.vk, src);
            return;
        }
    } else {
        if (!processRealFrameIntoSlot(src, g.inSlot[newSlot])) {
            LOGW("frame input processing failed");
            if (g.ahbImportCache.erase(ahb)) AHardwareBuffer_release(ahb);
            destroyAhbImage(g.vk, src);
            return;
        }
    }

    g.framesCopied++;

    // Cache owns the AhbImage (both hit and miss paths). There is no
    // extra "queue-enqueue" ref to release here any more — pushFrame()
    // no longer acquires one, since ahb is used synchronously for the
    // duration of this call and stays valid for as long as the caller
    // (Kotlin's HardwareBuffer / the JNI frame) keeps it open.
    if (srcFromCache) {
        g.cacheHits.fetch_add(1, std::memory_order_relaxed);
    } else {
        g.cacheMisses.fetch_add(1, std::memory_order_relaxed);
    }

    // PROFILE: input copy phase done.
    const auto tCopyDone = State::Clock::now();

    // AI backend takes priority over the LSFG shader path when it's
    // loaded and active — they're mutually exclusive per session
    // (initRenderLoop only stands up one or the other; see below).
    const bool runAi = g.aiLoaded
                       && !g.bypass.load(std::memory_order_relaxed);
    const bool runFramegen = !runAi
                             && g.framegenCtxId >= 0
                             && !g.bypass.load(std::memory_order_relaxed)
                             && !g.framegenAutoDisabled.load(std::memory_order_relaxed);
    if (runFramegen || runAi) {
        if (runFramegen) {
            try {
                if (g.performanceMode)
                    LSFG_3_1P::presentContext(g.framegenCtxId, /*inSem*/ -1, noOutputSemaphores);
                else
                    LSFG_3_1::presentContext(g.framegenCtxId, /*inSem*/ -1, noOutputSemaphores);
            } catch (const std::exception &e) {
                handleFramegenException("presentContext", e);
                return;
            }
        }
        // PROFILE: presentContext returned (CPU-side; the GPU work is
        // still pending on framegen's queue). For the AI backend this
        // timestamp doesn't mean much on its own — the actual ncnn
        // compute happens below and lands in the waitIdle bucket — but
        // splitting it out isn't worth a second profiling code path.
        const auto tPresentDone = State::Clock::now();
        // Wait for framegen's GPU work to actually finish before we (a)
        // overwrite the input AHB on the next pushFrame and (b) read the
        // output AHB for the blit. Framegen and our session use different
        // VkDevices with no shared queue, so a CPU-side wait is still
        // required — but waitIdle() now waits only on this frame's own
        // completion fences inside framegen's Context (see
        // Context::waitForCompletion), not vkDeviceWaitIdle(). That
        // previously drained every queue and every context on framegen's
        // device and was the single biggest per-frame cost in profiling
        // (see waitIdleNs below).
        if (runFramegen) {
            try {
                if (g.performanceMode) LSFG_3_1P::waitIdle();
                else                   LSFG_3_1::waitIdle();
            } catch (const std::exception &e) {
                // Same failure class as presentContext() above. This call was
                // previously unguarded, so on devices where the heavier
                // FP32/normal-mode workload (roughly 2x the per-frame
                // image/descriptor count vs performance mode) pushed a weaker
                // GPU into a fence timeout, the exception escaped this worker
                // thread uncaught and took down the whole app via
                // std::terminate(). Degrade the same way presentContext does.
                handleFramegenException("waitIdle", e);
                return;
            }
        } else {
#ifdef LSFG_HAVE_NCNN
            // AI path: the model inference itself is Vulkan/GPU-only.
            // The bundled ncnn bridge still uses host staging buffers to
            // exchange RGBA8 frames with AHardwareBuffer.
            // oldSlot/newSlot mirror the ping-pong bookkeeping just above
            // (newSlot = the capture just copied in this iteration).
            const int oldSlot = 1 - newSlot;
            if (!runAiInterpolate(oldSlot, newSlot,
                                   g.inSlot[0].extent.width, g.inSlot[0].extent.height)) {
                LOGE("AI backend: runAiInterpolate failed for this frame — skipping");
                return;
            }
#else
            // Unreachable: runAi is always false when built without ncnn.
            return;
#endif
        }
        // PROFILE: cross-device sync complete; outputs ready to read.
        const auto tWaitIdleDone = State::Clock::now();

        // Note: a previous revision auto-disabled framegen here when the
        // GPU pipeline was slower than the source cadence. Removed by
        // design — if the user picks heavy settings (flowScale=1.0 +
        // high multiplier on a weak device) it's their call to live with
        // the resulting frame rate. Silently flipping to passthrough
        // looked like a bug ("framegen stopped after 5s"). The bottom-of-
        // pipeline DEVICE_LOST fallback in presentContext above still
        // protects against unrecoverable driver errors.
        // Accumulator for the actual time spent inside blitOutputToWindow
        // calls. Reset each iteration for the lightweight diagnostics window.
        int64_t blitWorkNsThisFrame = 0;
        // Tracks how many of THIS iteration's blits actually posted (vs.
        // were silently dropped by blitOutputToWindow — no swapchain,
        // device lost, OEM present hiccup, etc). generatedFrames must be
        // incremented by this, not by g.outputs.size(): the old code
        // unconditionally added outputs.size() regardless of whether the
        // blits succeeded, so a run of dropped frames still counted as
        // "generated" on the HUD even though nothing new hit the screen —
        // the displayed "total fps" (driven by postedFrames, which only
        // counts real successes) would fall behind real+generated, and
        // the two numbers stopped matching.
        int postedGeneratedThisIter = 0;
        auto timedBlit = [&blitWorkNsThisFrame, captureTimestampNs, &prof](const AhbImage &out) {
            const auto t0 = State::Clock::now();
            const AhbImage *displayOut = prepareDisplayFrame(out);
            const bool posted = blitOutputToWindow(*displayOut);
            // The post-process image is a reusable AHB that is not part of
            // framegen's own ring. Drain this queue before reusing it on the
            // next output so an asynchronous WSI transfer can never race
            // the next GPU/CPU post-process write. This is intentionally
            // confined to the optional display-stage path.
            if (posted && (g.gpuPostProcessing || g.npuPostProcessing || g.cpuPostProcessing) &&
                    g.vk.fn.vkQueueWaitIdle != nullptr) {
                g.vk.fn.vkQueueWaitIdle(g.vk.computeQueue);
            }
            const auto t1 = State::Clock::now();
            blitWorkNsThisFrame += std::chrono::duration_cast<
                std::chrono::nanoseconds>(t1 - t0).count();

            if (posted) {
                const uint64_t postTimeNs = std::chrono::duration_cast<std::chrono::nanoseconds>(
                    t1.time_since_epoch()).count();
                if (postTimeNs > static_cast<uint64_t>(captureTimestampNs)) {
                    prof.captureToDisplayNs += (postTimeNs - static_cast<uint64_t>(captureTimestampNs));
                }
                prof.blitCount++;
            }
            return posted;
        };

        // Uncapped output path: once the GPU/AI work for this capture is
        // complete, publish every generated frame followed immediately by
        // the real frame. There is no software pacing
        // in the hot path, so the worker stays continuously productive.
        for (auto &o : g.outputs) {
            if (timedBlit(o)) postedGeneratedThisIter++;
        }
        timedBlit(g.inSlot[newSlot]);

        if (postedGeneratedThisIter > 0) {
            g.generatedFrames.fetch_add(static_cast<uint64_t>(postedGeneratedThisIter),
                std::memory_order_relaxed);
        }
        g.presentsDone++;  // keep our slot indexing in sync with framegen's frameIdx

        // PROFILE: accumulate this frame's segments and emit a summary
        // every kProfileWindow frames. Numbers are averages over the
        // completed window.
        const auto tFrameEnd = State::Clock::now();
        using ns = std::chrono::nanoseconds;

        // Big cores are the first choice. If the CPU-side portion of the
        // frame is already consuming most of a 60 Hz budget, widen this
        // worker to all online CPUs for the next frame so little cores can
        // contribute. The GPU wait itself is excluded: it is not CPU work.
        const int64_t cpuHotPathNs =
            std::chrono::duration_cast<ns>(tCopyDone - frameWorkStartedAt).count() +
            std::chrono::duration_cast<ns>(tPresentDone - tCopyDone).count() +
            blitWorkNsThisFrame;
        constexpr int64_t kCpuBudgetNs = 4'000'000;
        if (cpuHotPathNs > kCpuBudgetNs && !g.cpuLittleAssist &&
            g.cpuPolicy.allCpuCount() > g.cpuPolicy.bigCpuCount()) {
            if (g.cpuPolicy.useAllCores()) {
                g.cpuLittleAssist = true;
#ifdef LSFG_HAVE_NCNN
                ncnnCpuEnableLittleAssist(true);
#endif
                LOGI("CPU scheduler: big cores insufficient (%.2f ms CPU work); enabling little-core assist",
                     cpuHotPathNs / 1'000'000.0);
            }
        } else if (g.cpuLittleAssist && cpuHotPathNs < kCpuBudgetNs / 2) {
            if (g.cpuPolicy.useBigCores()) {
                g.cpuLittleAssist = false;
#ifdef LSFG_HAVE_NCNN
                ncnnCpuEnableLittleAssist(false);
#endif
                LOGI("CPU scheduler: CPU load recovered (%.2f ms); returning to big cores only",
                     cpuHotPathNs / 1'000'000.0);
            }
        }

        prof.copyNs     += std::chrono::duration_cast<ns>(tCopyDone     - frameWorkStartedAt).count();
        prof.presentNs  += std::chrono::duration_cast<ns>(tPresentDone  - tCopyDone).count();
        prof.waitIdleNs += std::chrono::duration_cast<ns>(tWaitIdleDone - tPresentDone).count();
        prof.blitNs     += blitWorkNsThisFrame;
        prof.totalNs    += std::chrono::duration_cast<ns>(tFrameEnd - frameWorkStartedAt).count();
        prof.samples++;
        if (prof.samples >= kProfileWindow) {
            const double n = static_cast<double>(prof.samples);
            const double avgWaitIdleMs = (prof.waitIdleNs / n) / 1'000'000.0;
            const double avgWallEndMs  = (prof.totalNs    / n) / 1'000'000.0;
            const double avgQueueMs    = (prof.queueNs    / n) / 1'000'000.0;
            const double avgLatencyMs  = prof.blitCount > 0 ? (prof.captureToDisplayNs / static_cast<double>(prof.blitCount)) / 1'000'000.0 : 0.0;
            const uint64_t hits = g.cacheHits.exchange(0, std::memory_order_relaxed);
            const uint64_t misses = g.cacheMisses.exchange(0, std::memory_order_relaxed);
            LOGW("frame profile (avg over %u): copy=%.2fms present=%.2fms waitIdle=%.2fms blitWork=%.2fms wallEnd=%.2fms queue=%.2fms latency=%.2fms cache_hits=%llu cache_misses=%llu (gen=%d extra)",
                 prof.samples,
                 (prof.copyNs / n)     / 1'000'000.0,
                 (prof.presentNs / n)  / 1'000'000.0,
                 avgWaitIdleMs,
                 (prof.blitNs / n)     / 1'000'000.0,
                 avgWallEndMs,
                 avgQueueMs,
                 avgLatencyMs,
                 static_cast<unsigned long long>(hits),
                 static_cast<unsigned long long>(misses),
                 g.multiplier);
            // Publish the closed profiling window for the UI / diagnostics.
            g.profileSnapshotCopyNs.store(prof.copyNs, std::memory_order_relaxed);
            g.profileSnapshotPresentNs.store(prof.presentNs, std::memory_order_relaxed);
            g.profileSnapshotWaitIdleNs.store(prof.waitIdleNs, std::memory_order_relaxed);
            g.profileSnapshotBlitNs.store(prof.blitNs, std::memory_order_relaxed);
            g.profileSnapshotTotalNs.store(prof.totalNs, std::memory_order_relaxed);
            g.profileSnapshotQueueNs.store(prof.queueNs, std::memory_order_relaxed);
            g.profileSnapshotLatencyNs.store(prof.captureToDisplayNs, std::memory_order_relaxed);
            g.profileSnapshotBlitCount.store(prof.blitCount, std::memory_order_relaxed);
            g.profileSnapshotSamples.store(prof.samples, std::memory_order_relaxed);
            prof = ProfileAccum{};
        }
    } else {
        // Bypass mode (user toggled the switch) OR framegen unavailable
        // (createContext failed): blit the raw capture straight through so
        // the overlay still shows live frames at capture rate. The total-FPS
        // counter stays equal to the real-FPS counter because we don't
        // increment generatedFrames here.
        blitOutputToWindow(g.inSlot[newSlot]);
    }
}

} // namespace

int initRenderLoop(const char *cacheDir, const RenderLoopConfig &cfg) {
    std::lock_guard<std::mutex> lock(g.mu);
    if (g.initialized) return kRenderLoopAlreadyInit;

    // multiplier in the prefs is the *total* output rate factor (2x = 60fps from 30fps);
    // framegen's generationCount is how many *extra* frames to interpolate per input
    // pair. So generationCount = multiplier - 1, and we allocate that many output AHBs.
    // (Mirrors lsfg-vk-android/src/context.cpp:80-81,101.)
    const int totalMult = cfg.multiplier > 1 ? cfg.multiplier : 2;
    g.multiplier = totalMult - 1;  // generationCount = N extra frames per pair
    g.performanceMode = cfg.performance;
    g.framegenFp16 = cfg.framegenFp16;
    g.hdr = cfg.hdr;
    g.gpuPostProcessing = cfg.gpuPostProcessing;
    g.gpuUpscaleFactor = std::clamp(cfg.gpuUpscaleFactor, 1.0f, 2.0f);
    g.gpuMethod = std::clamp(cfg.gpuMethod, 0, 15);
    g.gpuSharpness = std::clamp(cfg.gpuSharpness, 0.0f, 1.0f);
    g.gpuStrength = std::clamp(cfg.gpuStrength, 0.0f, 1.0f);
    g.npuPostProcessing = cfg.npuPostProcessing && nnapi_has_npu_accelerator();
    g.npuStageFailed = false;
    g.npuPreset = std::clamp(cfg.npuPreset, 0, 4);
    g.cpuPostProcessing = cfg.cpuPostProcessing;
    g.cpuPreset = std::clamp(cfg.cpuPreset, 0, 6);
    g.cpuStrength = std::clamp(cfg.cpuStrength, 0.0f, 1.0f);
    g.cpuSaturation = std::clamp(cfg.cpuSaturation, 0.0f, 1.0f);
    g.cpuVibrance = std::clamp(cfg.cpuVibrance, 0.0f, 1.0f);
    g.cpuVignette = std::clamp(cfg.cpuVignette, 0.0f, 1.0f);
    g.gpuPost.reset(g.vk);
    g.npuPost.reset();
    g.cpuPost.reset();
    destroyAhbImage(g.vk, g.postImage);
    g.postInputStaging.clear();
    g.postOutputStaging.clear();
    g.postCpuStaging.clear();
    if (cfg.npuPostProcessing && !g.npuPostProcessing) {
        LOGW("NPU post-process requested but no dedicated NNAPI accelerator is available; NPU stage disabled");
    }
    // flowScale on the prefs slider is "0.25..1.0" in user-friendly form, but
    // framegen wants the reciprocal (Linux passes 1.0f / conf.flowScale at
    // src/context.cpp:101). Larger user value = finer flow grid = better quality.
    const float userFlow = (cfg.flowScale >= 0.25f && cfg.flowScale <= 1.0f) ? cfg.flowScale : 1.0f;
    g.flowScale = 1.0f / userFlow;
    g.generatedFrames.store(0, std::memory_order_relaxed);
    g.postedFrames.store(0, std::memory_order_relaxed);
    g.uniqueCaptures.store(0, std::memory_order_relaxed);
    g.postRingHead.store(0, std::memory_order_relaxed);
    for (auto &slot : g.postRingTimestamps) {
        slot.store(0, std::memory_order_relaxed);
    }
    g.captureRingHead.store(0, std::memory_order_relaxed);
    for (auto &slot : g.captureRingTimestamps) {
        slot.store(0, std::memory_order_relaxed);
    }
    g.pushLogCount.store(0, std::memory_order_relaxed);
    g.blitLogCount.store(0, std::memory_order_relaxed);
    g.lumaGateOpen      = false;
    g.lumaGateDarkCount = 0;
    g.lumaGateStartNs   = 0;
    g.shizukuSampleTimestampNs.store(0, std::memory_order_relaxed);
    g.shizukuFrameTimeNs.store(0, std::memory_order_relaxed);
    g.shizukuPacingJitterNs.store(0, std::memory_order_relaxed);
    g.framesCopied = 0;
    g.presentsDone = 0;
    // Don't reset g.bypass — the user toggle should persist across re-inits
    // (e.g. when they change multiplier while bypass is on, the new context
    // should also start in bypass).
    // DO reset the framegen auto-disable latch — that flag tracks a driver
    // error tied to the previous device handle, which is being recreated here.
    // Carrying it forward would silently keep framegen off forever after one
    // bad submit, even on a healthy new device.
    g.framegenAutoDisabled.store(false, std::memory_order_relaxed);

    int rc = create_session(g.vk);
    if (rc != kOk) {
        LOGE("create_session failed rc=%d", rc);
        destroy_session(g.vk);
        return kRenderLoopSessionFailed;
    }

    const uint32_t renderW = cfg.width;
    const uint32_t renderH = cfg.height;

    const VkFormat fmt = VK_FORMAT_R8G8B8A8_UNORM;
    for (int i = 0; i < 2; ++i) {
        rc = createAhbImage(g.vk, renderW, renderH, fmt, g.inSlot[i]);
        if (rc != kOk) {
            LOGE("createAhbImage(input %d) failed rc=%d", i, rc);
            shutdownRenderLoop();
            return kRenderLoopBufferAlloc;
        }
        initInSlotImageLayout(g.inSlot[i].image);
    }
    // Number of outputs = generationCount, matching framegen "level".
    g.outputs.resize(g.multiplier);
    for (int i = 0; i < g.multiplier; ++i) {
        rc = createAhbImage(g.vk, renderW, renderH, fmt, g.outputs[i]);
        if (rc != kOk) {
            LOGE("createAhbImage(output %d) failed rc=%d", i, rc);
            shutdownRenderLoop();
            return kRenderLoopBufferAlloc;
        }
    }

    g.aiRequested = cfg.aiBackend;
    g.aiLoaded = false;
    g.aiEngine = cfg.aiEngine;
    if (cfg.aiBackend) {
#ifdef LSFG_HAVE_NCNN
        // AI backend replaces the LSFG shader chain entirely for this
        // session — skip initFramegen/createFramegenContext so we don't pay
        // for shader compilation we won't use, and so g.framegenCtxId stays
        // -1 (processFrame's runFramegen check already excludes runAi, but
        // this also keeps the "framegen disabled" return code below honest
        // if AI loading fails and there's no LSFG context to fall back to).
        const char *engineName = (cfg.aiEngine == 1) ? "IFRNet" : "RIFE";
        int rc;
        if (cfg.aiEngine == 1) {
            g.aiIfrnet = new IfrnetInterpolator();
            rc = g.aiIfrnet->load(cfg.aiModelDir, true, /*vulkanDeviceIndex*/ -1, 1);
        } else {
            g.ai = new NcnnInterpolator();
            rc = g.ai->load(cfg.aiModelDir, true, /*vulkanDeviceIndex*/ -1, 1);
        }
        if (rc == kNcnnOk) {
            g.aiLoaded = true;
            LOGI("AI backend (%s) loaded from %s (GPU-only Vulkan)", engineName, cfg.aiModelDir.c_str());
        } else {
            LOGE("AI backend (%s) GPU/Vulkan load() failed rc=%d (modelDir=%s) — "
                 "AI inference will NOT use CPU; falling back to the LSFG Vulkan shader path",
                 engineName, rc, cfg.aiModelDir.c_str());
            if (cfg.aiEngine == 1) {
                delete g.aiIfrnet;
                g.aiIfrnet = nullptr;
            } else {
                delete g.ai;
                g.ai = nullptr;
            }
        }
#else
        LOGE("AI backend requested but this .so was built without ncnn (LSFG_HAVE_NCNN) — "
             "falling back to LSFG shader path");
#endif
    }

    if (!g.aiLoaded) {
        g.framegenInitOk = initFramegen(cacheDir);
        if (g.framegenInitOk) {
            if (!createFramegenContext()) {
                LOGE("createFramegenContext failed — running in capture-only mode (counter will stay at 0)");
                g.framegenCtxId = -1;
            }
        } else {
            g.framegenCtxId = -1;
        }
    } else {
        g.framegenInitOk = false;
        g.framegenCtxId = -1;
    }

    g.initialized = true;

    // No worker thread to spin up any more — pushFrame() drives the
    // pipeline directly on the calling thread from here on.
    // Do NOT build the swapchain here. At initContext time the overlay's
    // Surface is still owned by the mirror VirtualDisplay producer — it's
    // only detached when setLsfgMode() runs (which retargets the VD to the
    // ImageReader). Creating the swapchain before that point races against
    // ANativeWindow's single-producer rule. The first blitOutputToWindow
    // call after LSFG mode is live builds it lazily.
    LOGW("Render loop initialised: capture=%ux%u render=%ux%u totalMult=%dx (gen=%d extra) flowScale=%.2f(internal=%.2f) hdr=%d perf=%d ctxId=%d",
         cfg.width, cfg.height, renderW, renderH, totalMult, g.multiplier,
         userFlow, g.flowScale,
         (int)g.hdr, (int)g.performanceMode, g.framegenCtxId);
    // Tell the caller whether framegen is actually running (either backend).
    // If not, Kotlin will keep the overlay up in mirror mode instead of
    // routing the capture through a dead context.
    return (g.framegenCtxId >= 0 || g.aiLoaded) ? kOk : kRenderLoopFramegenDisabled;
}

void setOutputSurface(ANativeWindow *win, uint32_t w, uint32_t h) {
    std::lock_guard<std::mutex> lock(g.mu);
    // Always tear down any prior swapchain first — it holds a VkSurfaceKHR
    // which holds an ANativeWindow reference, and the spec requires the
    // surface outlive the swapchain but be destroyed before the window.
    destroySwapchain();
    if (g.outWindow != nullptr) {
        ANativeWindow_release(g.outWindow);
        g.outWindow = nullptr;
    }
    if (win != nullptr) {
        ANativeWindow_acquire(win);
        g.outWindow = win;
        g.outWidth = w;
        g.outHeight = h;
        g.swapWinW = w;
        g.swapWinH = h;
        // Fresh ANativeWindow — its BufferQueue producer slot is empty until
        // the first ANativeWindow_lock or vkCreateAndroidSurfaceKHR succeeds.
        // Reset the taint so a new window can take the WSI fast path even if
        // a previous one was poisoned for CPU.
        g.windowCpuProducerLocked = false;
        if (kEnableWsiSwapchain && g.initialized) {
            if (createSwapchain()) {
                LOGW("Output surface attached %ux%u native=%p path=WSI", w, h, static_cast<void *>(win));
            } else {
                LOGW("Output surface attached %ux%u native=%p path=GPU/WSI unavailable", w, h, static_cast<void *>(win));
            }
        } else {
            const char *why = !kEnableWsiSwapchain ? "WSI disabled"
                            : !g.initialized        ? "pre-init"
                                                    : "unknown";
            LOGW("Output surface attached %ux%u native=%p path=GPU/WSI unavailable (%s)",
                 w, h, static_cast<void *>(win), why);
        }
    } else {
        g.outWidth = 0;
        g.outHeight = 0;
        g.swapWinW = 0;
        g.swapWinH = 0;
        LOGW("Output surface detached");
    }
}

void pushFrame(AHardwareBuffer *ahb, int64_t timestampNs) {
    if (ahb == nullptr) return;
    const uint32_t pushLogIndex = g.pushLogCount.load(std::memory_order_relaxed);
    if (pushLogIndex < 30) {
        AHardwareBuffer_Desc desc{};
        AHardwareBuffer_describe(ahb, &desc);
        if (g.pushLogCount.fetch_add(1, std::memory_order_relaxed) < 30) {
            LOGW("pushFrame #%u ahb=%ux%u stride=%u fmt=%u usage=0x%llx ts=%lld",
                 pushLogIndex + 1, desc.width, desc.height, desc.stride, desc.format,
                 static_cast<unsigned long long>(desc.usage),
                 static_cast<long long>(timestampNs));
        }
    }

    // Count every arriving capture frame as unique for the HUD's "real fps" metric.
    // We no longer hash pixel content to detect duplicates — the CPU cost of
    // reading back AHardwareBuffer pixels on every frame outweighed the benefit.
    // The OS capture path already throttles delivery to the target app's render
    // rate in practice, so the raw arrival count is a good proxy for real fps.
    g.uniqueCaptures.fetch_add(1, std::memory_order_relaxed);
    {
        const uint64_t nowNs = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                State::Clock::now().time_since_epoch()).count());
        const uint64_t slot = g.captureRingHead.fetch_add(1, std::memory_order_relaxed)
                              % State::kPostRingSize;
        g.captureRingTimestamps[slot].store(nowNs, std::memory_order_relaxed);
    }

    // Direct, synchronous hand-off: no queue, no mailbox, no separate
    // worker thread. Whatever thread called pushFrame() (the LSFG
    // ImageReader callback thread, or a Shizuku/root binder thread) runs
    // the entire import → framegen → blit pipeline itself, right here,
    // before this function returns. frameMu exists only to serialize this
    // against shutdownRenderLoop() (and against a second overlapping
    // pushFrame() call, if that ever happens) — it never buffers a frame.
    // If the pipeline is still busy with a previous frame when this one
    // arrives, this call simply blocks on frameMu until it's free, then
    // processes exactly the frame it was given.
    std::lock_guard<std::mutex> lock(g.frameMu);
    if (!g.initialized) return;
    processFrame(ahb, timestampNs);
}

void shutdownRenderLoop() {
    // Wait out any pushFrame() currently mid-pipeline, and block any new one
    // from starting, before we tear down the Vulkan/framegen state below.
    // There's no worker thread to join any more — pushFrame() runs
    // synchronously on the calling thread, so once we hold frameMu we know
    // nothing is touching this state concurrently.
    std::lock_guard<std::mutex> frameLock(g.frameMu);
    {
        std::lock_guard<std::mutex> lock(g.mu);
        if (!g.initialized) return; // idempotent: already shut down

        if (g.framegenCtxId >= 0) {
            try {
                if (g.performanceMode) LSFG_3_1P::deleteContext(g.framegenCtxId);
                else                   LSFG_3_1::deleteContext(g.framegenCtxId);
            } catch (...) {}
            g.framegenCtxId = -1;
        }
        if (g.framegenInitOk) {
            try {
                if (g.performanceMode) LSFG_3_1P::finalize();
                else                   LSFG_3_1::finalize();
            } catch (...) {}
            g.framegenInitOk = false;
        }

#ifdef LSFG_HAVE_NCNN
        if (g.ai != nullptr) {
            g.ai->unload();
            delete g.ai;
            g.ai = nullptr;
        }
        if (g.aiIfrnet != nullptr) {
            g.aiIfrnet->unload();
            delete g.aiIfrnet;
            g.aiIfrnet = nullptr;
        }
#endif
        g.aiLoaded = false;
        g.aiRequested = false;
        g.aiEngine = 0;

        // Clear AHB import cache before any Vulkan teardown.
        for (auto &[cachedAhb, img] : g.ahbImportCache) {
            destroyAhbImage(g.vk, img);
            AHardwareBuffer_release(cachedAhb);
        }
        g.ahbImportCache.clear();

        for (auto &o : g.outputs) destroyAhbImage(g.vk, o);
        g.outputs.clear();
        g.aiInputAStaging.clear();
        g.aiInputCStaging.clear();
        g.aiOutputStaging.clear();
        g.aiOutputPtrs.clear();
        g.gpuPost.reset(g.vk);
        g.npuPost.reset();
        g.cpuPost.reset();
        destroyAhbImage(g.vk, g.postImage);
        g.postInputStaging.clear();
        g.postOutputStaging.clear();
        g.postCpuStaging.clear();
        for (int i = 0; i < 2; ++i) destroyAhbImage(g.vk, g.inSlot[i]);


        // Destroy swapchain (and its surface) before the underlying ANativeWindow
        // is touched — surface destruction drops its internal window ref.
        // g.outWindow is intentionally kept live so pushFrame() can still
        // blit during the next session's shader-compilation window (before
        // setOutputSurface is called).  setOutputSurface always releases the
        // old reference before acquiring the new one, so there is no leak.
        destroySwapchain();
        g.swapWinW = 0;
        g.swapWinH = 0;

        destroy_session(g.vk);
        g.initialized = false;
        g.generatedFrames.store(0, std::memory_order_relaxed);
        g.postedFrames.store(0, std::memory_order_relaxed);
        g.uniqueCaptures.store(0, std::memory_order_relaxed);
        g.postRingHead.store(0, std::memory_order_relaxed);
        for (auto &slot : g.postRingTimestamps) {
            slot.store(0, std::memory_order_relaxed);
        }
        g.captureRingHead.store(0, std::memory_order_relaxed);
        for (auto &slot : g.captureRingTimestamps) {
            slot.store(0, std::memory_order_relaxed);
        }
        g.shizukuSampleTimestampNs.store(0, std::memory_order_relaxed);
        g.shizukuFrameTimeNs.store(0, std::memory_order_relaxed);
        g.shizukuPacingJitterNs.store(0, std::memory_order_relaxed);
    }
    LOGI("Render loop shut down");
}

uint64_t getGeneratedFrameCount() {
    return g.generatedFrames.load(std::memory_order_relaxed);
}

uint64_t getPostedFrameCount() {
    return g.postedFrames.load(std::memory_order_relaxed);
}

uint64_t getUniqueCaptureCount() {
    return g.uniqueCaptures.load(std::memory_order_relaxed);
}

double getAverageQueueMs() {
    const int64_t samples = g.profileSnapshotSamples.load(std::memory_order_relaxed);
    if (samples <= 0) return 0.0;
    const int64_t queueNs = g.profileSnapshotQueueNs.load(std::memory_order_relaxed);
    return (static_cast<double>(queueNs) / samples) / 1'000'000.0;
}

double getAverageLatencyMs() {
    const int64_t blitCount = g.profileSnapshotBlitCount.load(std::memory_order_relaxed);
    if (blitCount <= 0) return 0.0;
    const int64_t latencyNs = g.profileSnapshotLatencyNs.load(std::memory_order_relaxed);
    return (static_cast<double>(latencyNs) / blitCount) / 1'000'000.0;
}

uint32_t getProfileWindowNs(int64_t *out, uint32_t cap) {
    // out layout: [copyNs, presentNs, waitIdleNs, blitNs, totalNs, samples].
    // Each value is the SUM over the last completed kProfileWindow window;
    // divide by samples for per-frame averages. Returns 6 when populated, 0
    // when no window has closed yet (samples==0) or when out is too small.
    if (out == nullptr || cap < 6) return 0;
    const int64_t samples = g.profileSnapshotSamples.load(std::memory_order_relaxed);
    if (samples <= 0) return 0;
    out[0] = g.profileSnapshotCopyNs.load(std::memory_order_relaxed);
    out[1] = g.profileSnapshotPresentNs.load(std::memory_order_relaxed);
    out[2] = g.profileSnapshotWaitIdleNs.load(std::memory_order_relaxed);
    out[3] = g.profileSnapshotBlitNs.load(std::memory_order_relaxed);
    out[4] = g.profileSnapshotTotalNs.load(std::memory_order_relaxed);
    out[5] = samples;
    return 6;
}

uint32_t getRecentPostIntervalsNs(int64_t *outIntervalsNs, uint32_t cap) {
    if (outIntervalsNs == nullptr || cap == 0) return 0;
    // Snapshot the ring head. Entries from (head - kPostRingSize) to (head-1)
    // are populated (older ones are overwritten by wrap-around). For intervals
    // we need consecutive pairs, so we can produce at most min(cap, N-1)
    // where N is how many valid entries are present.
    const uint64_t head = g.postRingHead.load(std::memory_order_acquire);
    if (head < 2) return 0;  // need at least 2 timestamps for one interval
    const uint64_t validEntries = std::min<uint64_t>(head, State::kPostRingSize);
    const uint64_t available = validEntries - 1;
    const uint32_t want = static_cast<uint32_t>(std::min<uint64_t>(cap, available));
    // Walk the ring newest-first: slot (head-1), (head-2), ... subtracting
    // successive pairs to produce intervals. Skip the pair if either half
    // is zero (race with concurrent write during startup).
    uint32_t written = 0;
    uint64_t prevTs = g.postRingTimestamps[(head - 1) % State::kPostRingSize]
                          .load(std::memory_order_relaxed);
    for (uint32_t i = 1; i <= want && written < cap; ++i) {
        const uint64_t slot = (head - 1 - i) % State::kPostRingSize;
        const uint64_t ts = g.postRingTimestamps[slot].load(std::memory_order_relaxed);
        if (ts == 0 || prevTs == 0 || prevTs < ts) {
            // Either a torn read (zero) or non-monotonic (wraparound race):
            // stop and return what we have.
            break;
        }
        outIntervalsNs[written++] = static_cast<int64_t>(prevTs - ts);
        prevTs = ts;
    }
    return written;
}

// Average interval spanning the last `kSampleCount` ring entries (or fewer
// while the ring is still filling), converted to fps. Anchored to actual
// event timestamps rather than a fixed wall-clock polling window, so there's
// no window-aliasing for the caller to smooth out — a short averaging span
// is enough to stay stable frame-to-frame.
float fpsFromRing(const std::atomic<uint64_t> (&ring)[State::kPostRingSize],
                   const std::atomic<uint64_t> &ringHead) {
    constexpr uint64_t kSampleCount = 16;
    const uint64_t head = ringHead.load(std::memory_order_acquire);
    if (head < 2) return 0.0f;
    const uint64_t validEntries = std::min<uint64_t>(head, State::kPostRingSize);
    const uint64_t span = std::min<uint64_t>(kSampleCount, validEntries - 1);
    if (span == 0) return 0.0f;
    const uint64_t newestTs = ring[(head - 1) % State::kPostRingSize]
                                  .load(std::memory_order_relaxed);
    const uint64_t oldestTs = ring[(head - 1 - span) % State::kPostRingSize]
                                  .load(std::memory_order_relaxed);
    // Zero means a torn read during startup; oldest >= newest means a
    // wraparound race. Either way, no usable sample this call — the next
    // poll will have fresh data.
    if (newestTs == 0 || oldestTs == 0 || oldestTs >= newestTs) return 0.0f;
    const double elapsedNs = static_cast<double>(newestTs - oldestTs);
    return static_cast<float>(static_cast<double>(span) * 1'000'000'000.0 / elapsedNs);
}

bool getFpsSnapshot(float *out, uint32_t cap) {
    if (out == nullptr || cap < 2) return false;
    const float realFps = fpsFromRing(g.captureRingTimestamps, g.captureRingHead);
    const float totalFps = fpsFromRing(g.postRingTimestamps, g.postRingHead);
    if (realFps <= 0.0f && totalFps <= 0.0f) return false;
    out[0] = realFps;
    out[1] = totalFps;
    return true;
}

void setBypass(bool bypass) {
    g.bypass.store(bypass, std::memory_order_relaxed);
}

void setVsyncPeriodNs(int64_t /*periodNs*/) {
    // Kept as a binary/API compatibility shim. The render loop is intentionally
    // uncapped and no longer aligns work to display-vsync deadlines.
}

void setPacingParams(float /*emaAlpha*/, float /*outlierRatio*/,
                     float /*vsyncSlackMs*/) {
    // Kept as a binary/API compatibility shim. No software frame limiter is
    // applied in the native hot path.
}

void setShizukuTimingEnabled(bool enabled) {
    g.shizukuTimingEnabled.store(enabled, std::memory_order_relaxed);
    if (!enabled) {
        g.shizukuSampleTimestampNs.store(0, std::memory_order_relaxed);
        g.shizukuFrameTimeNs.store(0, std::memory_order_relaxed);
        g.shizukuPacingJitterNs.store(0, std::memory_order_relaxed);
    }
    LOGI("Shizuku timing %s", enabled ? "enabled" : "disabled");
}

void reportShizukuTiming(int64_t timestampNs,
                         int64_t frameTimeNs,
                         int64_t pacingJitterNs) {
    if (!g.shizukuTimingEnabled.load(std::memory_order_relaxed)) return;
    g.shizukuSampleTimestampNs.store(timestampNs, std::memory_order_relaxed);
    g.shizukuFrameTimeNs.store(frameTimeNs, std::memory_order_relaxed);
    g.shizukuPacingJitterNs.store(
        pacingJitterNs >= 0 ? pacingJitterNs : 0,
        std::memory_order_relaxed);
}

} // namespace lsfg_android
