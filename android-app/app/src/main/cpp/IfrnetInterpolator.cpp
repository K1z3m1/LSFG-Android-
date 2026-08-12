// Only compiled when CMakeLists.txt found the ncnn Android prebuilt SDK
// (LSFG_HAVE_NCNN) — see that file. Built into the same lsfg-ncnn-interpolator
// static lib as NcnnInterpolator.cpp/RifeWarp.cpp, for the same -fno-rtti/
// -fno-exceptions isolation reasons documented in CMakeLists.txt.

#include "IfrnetInterpolator.hpp"
#include "IfrnetWarp.hpp"
#include "crash_reporter.hpp"
#include "ncnn_cpu_policy.hpp"

#include <net.h>
#include <mat.h>
#include <gpu.h>
#include <layer.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>
#include <thread>

#define LOG_TAG "lsfg-ifrnet"
#define LOGE(...) ::lsfg_android::ring_logf(LOG_TAG, ANDROID_LOG_ERROR, __VA_ARGS__)
#define LOGI(...) ::lsfg_android::ring_logf(LOG_TAG, ANDROID_LOG_INFO,  __VA_ARGS__)

namespace lsfg_android {

namespace {

bool file_exists(const std::string &path) {
    if (FILE *f = fopen(path.c_str(), "rb")) {
        fclose(f);
        return true;
    }
    return false;
}

int round_up_to_multiple_of_32(int v) {
    return ((v + 31) / 32) * 32;
}

// GPU-only AI backend. No CPU ncnn::Net is constructed and no CPU/GPU
// work splitting or CPU affinity selection is performed here. ncnn's
// num_threads value is kept at 1 only for unavoidable CPU-side utility/custom
// layer work; model convolution/compute is Vulkan-backed.
::ncnn::Layer *ifrnet_warp_layer_creator(void * /*userdata*/) {
    return new IfrnetWarp();
}

} // namespace

struct IfrnetInterpolator::Impl {
    // Only built (and only used) when load() was asked for GPU and a
    // Vulkan device was actually available.
    ncnn::Net ifrnetGpu;

    // Tracks whether ifrnetGpu currently holds a successfully loaded
    // model. Set true at the end of a successful load(), cleared by
    // unload() and checked by isLoaded()/interpolate().
    bool gpuLoaded = false;


    // Single IFRNet forward pass on a specific network: predicts the frame
    // at `timestep` (in [0,1], 0 = exactly frame a, 1 = exactly frame c)
    // between two already-padded, normalized-to-[0,1], full-res RGB Mats.
    // Same in0/in1/in2 -> out0 blob contract as NcnnInterpolator::Impl.
    // Takes the net explicitly so the hybrid CPU and GPU halves can call
    // this from two different threads at once without sharing an
    // Extractor.
    static int predict(ncnn::Net &net, const ncnn::Mat &aPadded, const ncnn::Mat &cPadded,
                        int wPadded, int hPadded, float timestep, ncnn::Mat &outPadded) {
        ncnn::Mat timestepMap;
        timestepMap.create(wPadded, hPadded, 1);
        if (timestepMap.empty()) return kNcnnErrLoadFailed;
        timestepMap.fill(timestep);

        ncnn::Extractor ex = net.create_extractor();
        if (ex.input("in0", aPadded) != 0) return kNcnnErrLoadFailed;
        if (ex.input("in1", cPadded) != 0) return kNcnnErrLoadFailed;
        if (ex.input("in2", timestepMap) != 0) return kNcnnErrLoadFailed;
        if (ex.extract("out0", outPadded) != 0) return kNcnnErrLoadFailed;
        return kNcnnOk;
    }

    // Runs predict() for output index k (1-based, matching interpolate()'s
    // loop) and crops/denormalizes straight into outFrames[k - 1]. Shared
    // by the single-path and hybrid loops below so the pad/crop math only
    // lives in one place.
    static int predictOne(ncnn::Net &net, const ncnn::Mat &aPadded, const ncnn::Mat &cPadded,
                           int wPadded, int hPadded, int width, int height,
                           int k, int multiplier, uint8_t **outFrames) {
        const float timestep = static_cast<float>(k) / static_cast<float>(multiplier);
        ncnn::Mat outPadded;
        int rc = predict(net, aPadded, cPadded, wPadded, hPadded, timestep, outPadded);
        if (rc != kNcnnOk) return rc;

        ncnn::Mat outCropped(width, height, 3);
        if (outCropped.empty()) return kNcnnErrLoadFailed;
        for (int q = 0; q < 3; q++) {
            float *outptr = outCropped.channel(q);
            for (int y = 0; y < height; y++) {
                const float *inptr = outPadded.channel(q).row(y);
                for (int x = 0; x < width; x++) {
                    *outptr++ = std::min(std::max(inptr[x] * 255.f + 0.5f, 0.f), 255.f);
                }
            }
        }
        outCropped.to_pixels(outFrames[k - 1], ncnn::Mat::PIXEL_RGB2RGBA);
        return kNcnnOk;
    }
};

IfrnetInterpolator::IfrnetInterpolator() : impl_(new Impl()) {}

IfrnetInterpolator::~IfrnetInterpolator() {
    unload();
    delete impl_;
}

int IfrnetInterpolator::load(const std::string &modelDir, bool allowGpu, int vulkanDeviceIndex, int numThreads) {
    unload();

    const std::string ifrnetParam = modelDir + "/ifrnet.param";
    const std::string ifrnetBin   = modelDir + "/ifrnet.bin";

    for (const auto &p : {ifrnetParam, ifrnetBin}) {
        if (!file_exists(p)) {
            LOGE("model file missing: %s (expected ifrnet.param/ifrnet.bin in modelDir)", p.c_str());
            return kNcnnErrModelMissing;
        }
    }

    (void)numThreads; // CPU thread count is controlled by the adaptive big-core policy.
    ncnnCpuSetBigFirst();
    const int kUtilityThreads = ncnnCpuBigThreadCount();

    ncnn::Option baseOpt;
    baseOpt.num_threads = kUtilityThreads;
    baseOpt.use_fp16_packed = true;
    baseOpt.use_fp16_storage = true;
    baseOpt.use_fp16_arithmetic = true;
    baseOpt.use_winograd_convolution = false;
    baseOpt.use_sgemm_convolution = true;

    // GPU-only path. Do NOT construct a CPU ncnn::Net: the AI backend
    // is intentionally Vulkan-only on Android so inference cannot silently
    // fall back to the CPU when Vulkan initialization fails.
    if (!allowGpu) {
        LOGE("GPU-only IFRNet backend requested but Vulkan compute is disabled");
        return kNcnnErrLoadFailed;
    }

    const int gpuCount = ncnn::get_gpu_count();
    if (gpuCount <= 0) {
        LOGE("GPU-only IFRNet backend: ncnn reports no Vulkan GPU devices");
        return kNcnnErrLoadFailed;
    }

    ncnn::Option gpuOpt = baseOpt;
    gpuOpt.use_vulkan_compute = true;
    impl_->ifrnetGpu.opt = gpuOpt;
    const int dev = vulkanDeviceIndex >= 0 ? vulkanDeviceIndex : 0;
    if (dev >= gpuCount) {
        LOGE("GPU-only IFRNet backend: Vulkan device index %d out of range (count=%d)", dev, gpuCount);
        impl_->ifrnetGpu.clear();
        return kNcnnErrLoadFailed;
    }
    impl_->ifrnetGpu.set_vulkan_device(dev);
    impl_->ifrnetGpu.register_custom_layer("ifrnet.Warp", ifrnet_warp_layer_creator);

    if (impl_->ifrnetGpu.load_param(ifrnetParam.c_str()) != 0 ||
        impl_->ifrnetGpu.load_model(ifrnetBin.c_str()) != 0) {
        LOGE("GPU-only IFRNet model load failed from %s", modelDir.c_str());
        impl_->ifrnetGpu.clear();
        return kNcnnErrLoadFailed;
    }
    impl_->gpuLoaded = true;

    LOGI("ncnn IFRNet model loaded from %s (mode=%s, bigCoreThreads=%d, allCoreThreads=%d, winograd=%s)", modelDir.c_str(),
         "GPU-only", ncnnCpuBigThreadCount(), ncnnCpuAllThreadCount(), "off");
    return kNcnnOk;
}

void IfrnetInterpolator::unload() {
    if (impl_->gpuLoaded) {
        impl_->ifrnetGpu.clear();
        impl_->gpuLoaded = false;
    }
}

bool IfrnetInterpolator::isLoaded() const {
    return impl_->gpuLoaded;
}


int IfrnetInterpolator::interpolate(const uint8_t *frameA, const uint8_t *frameC,
                                     int width, int height,
                                     uint8_t **outFrames, int multiplier,
                                     float /*flowScale — unused, see header*/) {
    if (!impl_->gpuLoaded) {
        return kNcnnErrNotLoaded;
    }
    if (width <= 0 || height <= 0 || multiplier < 2 || multiplier > 8 ||
        frameA == nullptr || frameC == nullptr || outFrames == nullptr) {
        return kNcnnErrBadArgs;
    }

    ncnn::Mat fullA = ncnn::Mat::from_pixels(frameA, ncnn::Mat::PIXEL_RGBA2RGB, width, height);
    ncnn::Mat fullC = ncnn::Mat::from_pixels(frameC, ncnn::Mat::PIXEL_RGBA2RGB, width, height);

    const int wPadded = round_up_to_multiple_of_32(width);
    const int hPadded = round_up_to_multiple_of_32(height);

    // Normalize to [0,1] and zero-pad up to (wPadded, hPadded), matching
    // ifrnet-ncnn-vulkan's own preproc exactly (constant zero padding, not
    // edge replication) — same rule as NcnnInterpolator, confirmed against
    // upstream src/ifrnet.cpp's "pad to 32n" comment.
    ncnn::Mat aPadded(wPadded, hPadded, 3);
    ncnn::Mat cPadded(wPadded, hPadded, 3);
    if (aPadded.empty() || cPadded.empty()) {
        return kNcnnErrLoadFailed;
    }
    auto normalizeAndPad = [&](const ncnn::Mat &src, ncnn::Mat &dst) {
        for (int q = 0; q < 3; q++) {
            float *outptr = dst.channel(q);
            for (int y = 0; y < hPadded; y++) {
                const float *inptr = (y < height) ? src.channel(q).row(y) : nullptr;
                for (int x = 0; x < wPadded; x++) {
                    *outptr++ = (inptr != nullptr && x < width) ? inptr[x] * (1.f / 255.f) : 0.f;
                }
            }
        }
    };
    normalizeAndPad(fullA, aPadded);
    normalizeAndPad(fullC, cPadded);

    // GPU-only inference. Every interpolation pass goes through the
    // Vulkan-backed ncnn network; there is deliberately no CPU fallback.
    if (!impl_->gpuLoaded) {
        return kNcnnErrNotLoaded;
    }
    for (int k = 1; k < multiplier; k++) {
        int rc = Impl::predictOne(impl_->ifrnetGpu, aPadded, cPadded,
                                   wPadded, hPadded, width, height,
                                   k, multiplier, outFrames);
        if (rc != kNcnnOk) {
            LOGE("GPU-only IFRNet inference failed rc=%d", rc);
            return rc;
        }
    }
    return kNcnnOk;

}

} // namespace lsfg_android
