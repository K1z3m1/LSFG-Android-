// JNI entry points for com.firstt175.deepdrop.session.NativeBridge.
//
// Phase 3: extractShaders extracts precompiled FP16/FP32 SPIR-V resources
// (Lossless Scaling 3.2.2.0) and writes them to disk.
// Phase 4: probeShaders validates that every cached SPIR-V blob is accepted by
// the device driver via vkCreateShaderModule.
// Phase 5: initContext / pushFrame / setOutputSurface / destroyContext wire
// the LSFG_3_1 Vulkan pipeline through the AHB bridge.

#include "android_shader_loader.hpp"
#include "android_vk_probe.hpp"
#include "crash_reporter.hpp"
#include "lsfg_render_loop.hpp"
#include "nnapi_npu.hpp"
#ifdef LSFG_HAVE_NCNN
#include "NcnnInterpolator.hpp"
#include "IfrnetInterpolator.hpp"
#endif

#include <android/hardware_buffer_jni.h>
#include <android/native_window_jni.h>
#include <jni.h>

#include <string>
#include <vector>
#include <cstring>

namespace {

constexpr const char *kVersion = "lsfg-android 0.1.4-gpu";

std::string jstring_to_std(JNIEnv *env, jstring s) {
    if (s == nullptr) {
        return {};
    }
    const char *chars = env->GetStringUTFChars(s, nullptr);
    std::string out = chars ? chars : "";
    if (chars) {
        env->ReleaseStringUTFChars(s, chars);
    }
    return out;
}

#ifdef LSFG_HAVE_NCNN
lsfg_android::NcnnInterpolator *g_ncnnInterpolator = nullptr;
lsfg_android::IfrnetInterpolator *g_ifrnetInterpolator = nullptr;
#endif

} // namespace

extern "C" JNIEXPORT jstring JNICALL
Java_com_firstt175_deepdrop_session_NativeBridge_nativeVersion(JNIEnv *env, jobject /*thiz*/) {
    return env->NewStringUTF(kVersion);
}

extern "C" JNIEXPORT void JNICALL
Java_com_firstt175_deepdrop_session_NativeBridge_initCrashReporter(
        JNIEnv *env, jobject /*thiz*/, jstring crashPath, jstring logPath) {
    lsfg_android::init_crash_reporter(
        jstring_to_std(env, crashPath),
        jstring_to_std(env, logPath));
}

extern "C" JNIEXPORT jint JNICALL
Java_com_firstt175_deepdrop_session_NativeBridge_extractShaders(
        JNIEnv *env, jobject /*thiz*/,
        jstring dllPath, jstring /*dllSha256*/, jstring cacheDir) {
    const std::string path = jstring_to_std(env, dllPath);
    const std::string cache = jstring_to_std(env, cacheDir);
    if (path.empty() || cache.empty()) {
        return lsfg_android::kErrDllUnreadable;
    }
    return lsfg_android::extract_dll_to_spirv(path, cache);
}

extern "C" JNIEXPORT jint JNICALL
Java_com_firstt175_deepdrop_session_NativeBridge_probeShaders(
        JNIEnv *env, jobject /*thiz*/, jstring cacheDir) {
    return lsfg_android::probe_shaders_on_device(jstring_to_std(env, cacheDir));
}

extern "C" JNIEXPORT jint JNICALL
Java_com_firstt175_deepdrop_session_NativeBridge_initContext(
        JNIEnv *env, jobject /*thiz*/,
        jstring cacheDir, jint width, jint height,
        jint multiplier, jfloat flowScale,
        jboolean performance, jboolean hdr,
        jboolean framegenFp16,
        jfloat emaAlpha,
        jfloat outlierRatio, jfloat vsyncSlackMs,
        jboolean gpuPostProcessing, jfloat gpuUpscaleFactor,
        jint gpuMethod, jfloat gpuSharpness, jfloat gpuStrength,
        jboolean npuPostProcessing, jint npuPreset,
        jboolean cpuPostProcessing, jint cpuPreset,
        jfloat cpuStrength, jfloat cpuSaturation,
        jfloat cpuVibrance, jfloat cpuVignette,
        jboolean aiBackend, jstring aiModelDir,
        jint aiEngine) {
    const std::string cache = jstring_to_std(env, cacheDir);
    if (cache.empty() || width <= 0 || height <= 0) {
        return lsfg_android::kErrDllUnreadable;
    }
    const bool wantAi = aiBackend == JNI_TRUE;
    if (!wantAi) {
        // Sanity check: at least one cached SPIR-V file must exist. Without shaders,
        // LSFG_3_1::initialize() will throw inside framegen (we catch it but the
        // pipeline will be useless anyway). Failing fast here gives a cleaner error
        // path back to Kotlin so the service can stay in mirror mode.
        // id 255 is "mipmaps"'s base id; +kFp32SpirvIdOffset gives its FP32
        // SPIR-V counterpart (353), the default cache populated by extraction.
        // Skipped for the AI backend, which never touches the LSFG shader cache.
        auto probeShader = lsfg_android::load_cached_spirv(
            cache, 255 + lsfg_android::kFp32SpirvIdOffset, lsfg_android::ShaderCache::Fp32Spirv);
        if (probeShader.empty()) {
            return lsfg_android::kErrMissingResource;
        }
    }
    const lsfg_android::RenderLoopConfig cfg{
        .width = static_cast<uint32_t>(width),
        .height = static_cast<uint32_t>(height),
        .multiplier = static_cast<int>(multiplier),
        .flowScale = static_cast<float>(flowScale),
        .performance = performance == JNI_TRUE,
        .hdr = hdr == JNI_TRUE,
        .framegenFp16 = framegenFp16 == JNI_TRUE,
        .emaAlpha = static_cast<float>(emaAlpha),
        .outlierRatio = static_cast<float>(outlierRatio),
        .vsyncSlackMs = static_cast<float>(vsyncSlackMs),
        .gpuPostProcessing = gpuPostProcessing == JNI_TRUE,
        .gpuUpscaleFactor = static_cast<float>(gpuUpscaleFactor),
        .gpuMethod = static_cast<int>(gpuMethod),
        .gpuSharpness = static_cast<float>(gpuSharpness),
        .gpuStrength = static_cast<float>(gpuStrength),
        .npuPostProcessing = npuPostProcessing == JNI_TRUE,
        .npuPreset = static_cast<int>(npuPreset),
        .cpuPostProcessing = cpuPostProcessing == JNI_TRUE,
        .cpuPreset = static_cast<int>(cpuPreset),
        .cpuStrength = static_cast<float>(cpuStrength),
        .cpuSaturation = static_cast<float>(cpuSaturation),
        .cpuVibrance = static_cast<float>(cpuVibrance),
        .cpuVignette = static_cast<float>(cpuVignette),
        .aiBackend = wantAi,
        .aiModelDir = jstring_to_std(env, aiModelDir),
        .aiEngine = static_cast<int>(aiEngine),
    };
    return lsfg_android::initRenderLoop(cache.c_str(), cfg);
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_firstt175_deepdrop_session_NativeBridge_isNpuPostProcessAvailable(
        JNIEnv * /*env*/, jobject /*thiz*/) {
    return lsfg_android::nnapi_has_npu_accelerator() ? JNI_TRUE : JNI_FALSE;
}

// Reports the number of Vulkan-capable GPU devices ncnn's gpu.h can see on
// this device (usually 0 or 1 on Android). Used by the COMPUTE card in
// DllPickerScreen to show which Vulkan GPU the AI (ncnn) backend actually
// runs on, since that backend is GPU-only — see NcnnInterpolator.hpp's
// class doc comment.
extern "C" JNIEXPORT jint JNICALL
Java_com_firstt175_deepdrop_session_NativeBridge_getVulkanGpuCount(JNIEnv * /*env*/, jobject /*thiz*/) {
#ifdef LSFG_HAVE_NCNN
    return static_cast<jint>(lsfg_android::ncnnGpuCount());
#else
    return 0;
#endif
}

// Human-readable Vulkan device name (e.g. "Adreno (TM) 740") for gpu index
// [index], matching the device index NcnnInterpolator::load()/
// IfrnetInterpolator::load() default to (index -1 from Kotlin resolves to
// device 0 there). Returns an empty string if index is out of range or this
// .so was built without ncnn.
extern "C" JNIEXPORT jstring JNICALL
Java_com_firstt175_deepdrop_session_NativeBridge_getVulkanGpuName(
        JNIEnv *env, jobject /*thiz*/, jint index) {
#ifdef LSFG_HAVE_NCNN
    const std::string name = lsfg_android::ncnnGpuName(static_cast<int>(index));
    return env->NewStringUTF(name.c_str());
#else
    (void) index;
    return env->NewStringUTF("");
#endif
}

// Vulkan API version this device's GPU driver actually supports (e.g.
// "1.3.106"), used by the Device Info card. See
// android_vk_probe.hpp::get_vulkan_device_api_version_string() — this
// negotiates via vkEnumerateInstanceVersion rather than assuming 1.1, so
// the render loop and this readout always agree on what the driver speaks.
extern "C" JNIEXPORT jstring JNICALL
Java_com_firstt175_deepdrop_session_NativeBridge_getVulkanApiVersion(
        JNIEnv *env, jobject /*thiz*/) {
    const std::string version = lsfg_android::get_vulkan_device_api_version_string();
    return env->NewStringUTF(version.c_str());
}

// GPU vendor, integrated/discrete, driver version, and an approximate VRAM
// figure (largest DEVICE_LOCAL heap — on mobile's near-universal unified
// memory this is a GPU-addressable slice of shared RAM, not dedicated
// video RAM; see GpuInfo's doc comment in android_vk_probe.hpp). Exposed
// as four separate getters rather than one packed call to match the
// existing getVulkanGpuCount/getVulkanGpuName pattern above.
extern "C" JNIEXPORT jstring JNICALL
Java_com_firstt175_deepdrop_session_NativeBridge_getGpuVendor(JNIEnv *env, jobject /*thiz*/) {
    return env->NewStringUTF(lsfg_android::get_vulkan_gpu_info().vendor.c_str());
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_firstt175_deepdrop_session_NativeBridge_getGpuDeviceType(JNIEnv *env, jobject /*thiz*/) {
    return env->NewStringUTF(lsfg_android::get_vulkan_gpu_info().deviceType.c_str());
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_firstt175_deepdrop_session_NativeBridge_getGpuDriverVersion(JNIEnv *env, jobject /*thiz*/) {
    return env->NewStringUTF(lsfg_android::get_vulkan_gpu_info().driverVersion.c_str());
}

// -1 if the probe failed or the driver reported no DEVICE_LOCAL heap.
extern "C" JNIEXPORT jlong JNICALL
Java_com_firstt175_deepdrop_session_NativeBridge_getGpuVramMb(JNIEnv * /*env*/, jobject /*thiz*/) {
    return static_cast<jlong>(lsfg_android::get_vulkan_gpu_info().deviceLocalHeapMb);
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_firstt175_deepdrop_session_NativeBridge_isFramegenFp16Supported(
        JNIEnv *env, jobject /*thiz*/, jstring cacheDir) {
    // Two prerequisites: the GPU has to expose shaderFloat16, AND the FP16
    // SPIR-V cache must already be populated by the DLL extraction step.
    // The UI ANDs them so the toggle appears only when the user can actually
    // enable it without surprises.
    if (!lsfg_android::device_supports_float16()) {
        return JNI_FALSE;
    }
    if (cacheDir == nullptr) {
        return JNI_FALSE;
    }
    const std::string cache = jstring_to_std(env, cacheDir);
    if (cache.empty()) {
        return JNI_FALSE;
    }
    return lsfg_android::fp16_shaders_available(cache) ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT void JNICALL
Java_com_firstt175_deepdrop_session_NativeBridge_setOutputSurface(
        JNIEnv *env, jobject /*thiz*/, jobject surface, jint w, jint h) {
    ANativeWindow *win = (surface != nullptr)
        ? ANativeWindow_fromSurface(env, surface)
        : nullptr;
    lsfg_android::setOutputSurface(win, static_cast<uint32_t>(w), static_cast<uint32_t>(h));
    if (win != nullptr) {
        // setOutputSurface has acquired its own reference; release ours.
        ANativeWindow_release(win);
    }
}

extern "C" JNIEXPORT void JNICALL
Java_com_firstt175_deepdrop_session_NativeBridge_pushFrame(
        JNIEnv *env, jobject /*thiz*/, jobject hardwareBuffer, jlong timestampNs) {
    if (hardwareBuffer == nullptr) return;
    AHardwareBuffer *ahb = AHardwareBuffer_fromHardwareBuffer(env, hardwareBuffer);
    lsfg_android::pushFrame(ahb, static_cast<int64_t>(timestampNs));
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_firstt175_deepdrop_session_NativeBridge_getGeneratedFrameCount(
        JNIEnv * /*env*/, jobject /*thiz*/) {
    return static_cast<jlong>(lsfg_android::getGeneratedFrameCount());
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_firstt175_deepdrop_session_NativeBridge_getPostedFrameCount(
        JNIEnv * /*env*/, jobject /*thiz*/) {
    return static_cast<jlong>(lsfg_android::getPostedFrameCount());
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_firstt175_deepdrop_session_NativeBridge_getUniqueCaptureCount(
        JNIEnv * /*env*/, jobject /*thiz*/) {
    return static_cast<jlong>(lsfg_android::getUniqueCaptureCount());
}

extern "C" JNIEXPORT jdouble JNICALL
Java_com_firstt175_deepdrop_session_NativeBridge_getAverageQueueMs(
        JNIEnv * /*env*/, jobject /*thiz*/) {
    return static_cast<jdouble>(lsfg_android::getAverageQueueMs());
}

extern "C" JNIEXPORT jdouble JNICALL
Java_com_firstt175_deepdrop_session_NativeBridge_getAverageLatencyMs(
        JNIEnv * /*env*/, jobject /*thiz*/) {
    return static_cast<jdouble>(lsfg_android::getAverageLatencyMs());
}

extern "C" JNIEXPORT jint JNICALL
Java_com_firstt175_deepdrop_session_NativeBridge_getRecentPostIntervalsNs(
        JNIEnv *env, jobject /*thiz*/, jlongArray outArray) {
    if (outArray == nullptr) return 0;
    const jsize cap = env->GetArrayLength(outArray);
    if (cap <= 0) return 0;
    jlong *buf = env->GetLongArrayElements(outArray, nullptr);
    if (buf == nullptr) return 0;
    static_assert(sizeof(jlong) == sizeof(int64_t), "jlong must be int64_t");
    const uint32_t written = lsfg_android::getRecentPostIntervalsNs(
        reinterpret_cast<int64_t *>(buf), static_cast<uint32_t>(cap));
    env->ReleaseLongArrayElements(outArray, buf, 0);
    return static_cast<jint>(written);
}

extern "C" JNIEXPORT jint JNICALL
Java_com_firstt175_deepdrop_session_NativeBridge_getProfileWindowNs(
        JNIEnv *env, jobject /*thiz*/, jlongArray outArray) {
    if (outArray == nullptr) return 0;
    const jsize cap = env->GetArrayLength(outArray);
    if (cap < 6) return 0;
    jlong *buf = env->GetLongArrayElements(outArray, nullptr);
    if (buf == nullptr) return 0;
    static_assert(sizeof(jlong) == sizeof(int64_t), "jlong must be int64_t");
    const uint32_t written = lsfg_android::getProfileWindowNs(
        reinterpret_cast<int64_t *>(buf), static_cast<uint32_t>(cap));
    env->ReleaseLongArrayElements(outArray, buf, 0);
    return static_cast<jint>(written);
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_firstt175_deepdrop_session_NativeBridge_getFpsSnapshot(
        JNIEnv *env, jobject /*thiz*/, jfloatArray outArray) {
    if (outArray == nullptr) return JNI_FALSE;
    const jsize cap = env->GetArrayLength(outArray);
    if (cap < 2) return JNI_FALSE;
    jfloat *buf = env->GetFloatArrayElements(outArray, nullptr);
    if (buf == nullptr) return JNI_FALSE;
    static_assert(sizeof(jfloat) == sizeof(float), "jfloat must be float");
    const bool ok = lsfg_android::getFpsSnapshot(
        reinterpret_cast<float *>(buf), static_cast<uint32_t>(cap));
    env->ReleaseFloatArrayElements(outArray, buf, 0);
    return ok ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT void JNICALL
Java_com_firstt175_deepdrop_session_NativeBridge_setBypass(
        JNIEnv * /*env*/, jobject /*thiz*/, jboolean bypass) {
    lsfg_android::setBypass(bypass == JNI_TRUE);
}

extern "C" JNIEXPORT void JNICALL
Java_com_firstt175_deepdrop_session_NativeBridge_setVsyncPeriodNs(
        JNIEnv * /*env*/, jobject /*thiz*/, jlong periodNs) {
    lsfg_android::setVsyncPeriodNs(static_cast<int64_t>(periodNs));
}

extern "C" JNIEXPORT void JNICALL
Java_com_firstt175_deepdrop_session_NativeBridge_setPacingParams(
        JNIEnv * /*env*/, jobject /*thiz*/,
        jfloat emaAlpha, jfloat outlierRatio,
        jfloat vsyncSlackMs) {
    lsfg_android::setPacingParams(
        static_cast<float>(emaAlpha),
        static_cast<float>(outlierRatio),
        static_cast<float>(vsyncSlackMs));
}

extern "C" JNIEXPORT void JNICALL
Java_com_firstt175_deepdrop_session_NativeBridge_setShizukuTimingEnabled(
        JNIEnv * /*env*/, jobject /*thiz*/, jboolean enabled) {
    lsfg_android::setShizukuTimingEnabled(enabled == JNI_TRUE);
}

extern "C" JNIEXPORT void JNICALL
Java_com_firstt175_deepdrop_session_NativeBridge_reportShizukuTiming(
        JNIEnv * /*env*/, jobject /*thiz*/,
        jlong timestampNs, jlong frameTimeNs, jlong pacingJitterNs) {
    lsfg_android::reportShizukuTiming(
        static_cast<int64_t>(timestampNs),
        static_cast<int64_t>(frameTimeNs),
        static_cast<int64_t>(pacingJitterNs));
}

extern "C" JNIEXPORT void JNICALL
Java_com_firstt175_deepdrop_session_NativeBridge_destroyContext(JNIEnv * /*env*/, jobject /*thiz*/) {
    lsfg_android::shutdownRenderLoop();
}

// -----------------------------------------------------------------------------
// AI (ncnn) frame-gen backend. Only functional when this .so was built with
// ncnn found (see the LSFG_HAVE_NCNN block in CMakeLists.txt) — otherwise
// these all return lsfg_android::kNcnnErrNotBuilt / false / -1 without
// touching any global state, so calling them is always safe.
//
// initAiInterpolator/aiInterpolatePreview below are the standalone
// load()/interpolate() pair used by the settings-screen "test this model"
// action. The live per-frame path is separate: lsfg_render_loop.cpp owns
// its own NcnnInterpolator instance (g.ai, populated by initContext's
// aiBackend/aiModelDir args) and drives it directly from workerThread via
// runAiInterpolate() — see the LSFG_HAVE_NCNN block there for the real
// per-frame AHB readback/upload stage.
extern "C" JNIEXPORT jint JNICALL
Java_com_firstt175_deepdrop_session_NativeBridge_initAiInterpolator(
        JNIEnv *env, jobject /*thiz*/,
        jstring modelDir, jboolean useVulkan, jint vulkanDeviceIndex, jint numThreads, jint engine) {
#ifdef LSFG_HAVE_NCNN
    const std::string dir = jstring_to_std(env, modelDir);
    if (dir.empty()) {
        return lsfg_android::kNcnnErrModelMissing;
    }
    if (engine == 1) {
        if (g_ifrnetInterpolator == nullptr) {
            g_ifrnetInterpolator = new lsfg_android::IfrnetInterpolator();
        }
        return g_ifrnetInterpolator->load(
            dir, useVulkan == JNI_TRUE, static_cast<int>(vulkanDeviceIndex), static_cast<int>(numThreads));
    }
    if (g_ncnnInterpolator == nullptr) {
        g_ncnnInterpolator = new lsfg_android::NcnnInterpolator();
    }
    return g_ncnnInterpolator->load(
        dir, useVulkan == JNI_TRUE, static_cast<int>(vulkanDeviceIndex), static_cast<int>(numThreads));
#else
    (void) env; (void) modelDir; (void) useVulkan; (void) vulkanDeviceIndex; (void) numThreads; (void) engine;
    return -1; // lsfg_android::kNcnnErrNotBuilt — kept as a literal since the header isn't included
#endif
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_firstt175_deepdrop_session_NativeBridge_isAiInterpolatorLoaded(JNIEnv * /*env*/, jobject /*thiz*/, jint engine) {
#ifdef LSFG_HAVE_NCNN
    if (engine == 1) {
        return (g_ifrnetInterpolator != nullptr && g_ifrnetInterpolator->isLoaded()) ? JNI_TRUE : JNI_FALSE;
    }
    return (g_ncnnInterpolator != nullptr && g_ncnnInterpolator->isLoaded()) ? JNI_TRUE : JNI_FALSE;
#else
    (void) engine;
    return JNI_FALSE;
#endif
}

extern "C" JNIEXPORT void JNICALL
Java_com_firstt175_deepdrop_session_NativeBridge_releaseAiInterpolator(JNIEnv * /*env*/, jobject /*thiz*/) {
#ifdef LSFG_HAVE_NCNN
    if (g_ncnnInterpolator != nullptr) {
        g_ncnnInterpolator->unload();
        delete g_ncnnInterpolator;
        g_ncnnInterpolator = nullptr;
    }
    if (g_ifrnetInterpolator != nullptr) {
        g_ifrnetInterpolator->unload();
        delete g_ifrnetInterpolator;
        g_ifrnetInterpolator = nullptr;
    }
#endif
}

// frameA/frameC: direct RGBA8 byte buffers (width*height*4 each) — pass
// android.graphics.Bitmap.copyPixelsToBuffer()'d direct ByteBuffers from
// Kotlin. outFrame: a single pre-allocated direct ByteBuffer, width*height*4
// bytes, that receives the interpolated frame at outIndex/multiplier (e.g.
// outIndex=1, multiplier=2 for the classic midpoint). Returns 0
// (lsfg_android::kNcnnOk) on success, a negative NcnnInterpolator error code
// otherwise. Intended for a settings-screen "test this model" preview, not
// per-frame session use (see the comment above initAiInterpolator). `engine`
// selects which loaded interpolator to run against: 0 = RIFE
// (g_ncnnInterpolator), 1 = IFRNet (g_ifrnetInterpolator) — must match
// whichever engine was passed to the initAiInterpolator() call that loaded it.
extern "C" JNIEXPORT jint JNICALL
Java_com_firstt175_deepdrop_session_NativeBridge_aiInterpolatePreview(
        JNIEnv *env, jobject /*thiz*/,
        jobject frameA, jobject frameC, jint width, jint height,
        jobject outFrame, jint outIndex, jint multiplier, jfloat flowScale, jint engine) {
#ifdef LSFG_HAVE_NCNN
    const bool useIfrnet = (engine == 1);
    if (useIfrnet) {
        if (g_ifrnetInterpolator == nullptr || !g_ifrnetInterpolator->isLoaded()) {
            return lsfg_android::kNcnnErrNotLoaded;
        }
    } else {
        if (g_ncnnInterpolator == nullptr || !g_ncnnInterpolator->isLoaded()) {
            return lsfg_android::kNcnnErrNotLoaded;
        }
    }
    auto *a = static_cast<uint8_t *>(env->GetDirectBufferAddress(frameA));
    auto *c = static_cast<uint8_t *>(env->GetDirectBufferAddress(frameC));
    auto *out = static_cast<uint8_t *>(env->GetDirectBufferAddress(outFrame));
    if (a == nullptr || c == nullptr || out == nullptr) {
        return lsfg_android::kNcnnErrBadArgs;
    }
    if (outIndex < 1 || outIndex >= multiplier) {
        return lsfg_android::kNcnnErrBadArgs;
    }

    const size_t frameBytes = static_cast<size_t>(width) * static_cast<size_t>(height) * 4;
    std::vector<uint8_t> scratch(frameBytes * static_cast<size_t>(multiplier - 1));
    std::vector<uint8_t *> outPtrs(multiplier - 1);
    for (int i = 0; i < multiplier - 1; ++i) {
        outPtrs[i] = scratch.data() + static_cast<size_t>(i) * frameBytes;
    }

    const int rc = useIfrnet
        ? g_ifrnetInterpolator->interpolate(
              a, c, static_cast<int>(width), static_cast<int>(height),
              outPtrs.data(), static_cast<int>(multiplier), static_cast<float>(flowScale))
        : g_ncnnInterpolator->interpolate(
              a, c, static_cast<int>(width), static_cast<int>(height),
              outPtrs.data(), static_cast<int>(multiplier), static_cast<float>(flowScale));
    if (rc != lsfg_android::kNcnnOk) {
        return rc;
    }

    std::memcpy(out, outPtrs[outIndex - 1], frameBytes);
    return lsfg_android::kNcnnOk;
#else
    (void) env; (void) frameA; (void) frameC; (void) width; (void) height;
    (void) outFrame; (void) outIndex; (void) multiplier; (void) flowScale; (void) engine;
    return -1; // lsfg_android::kNcnnErrNotBuilt
#endif
}
