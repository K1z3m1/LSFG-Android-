#pragma once

#include <cstdint>
#include <string>

// Only ever compiled/linked when the ncnn Android prebuilt SDK was found at
// configure time — see the LSFG_HAVE_NCNN block in CMakeLists.txt. Callers in
// lsfg_jni.cpp must guard every use of this header behind `#ifdef LSFG_HAVE_NCNN`.

namespace lsfg_android {

// Error codes returned to Kotlin via JNI (mirrors the style of
// android_shader_loader.hpp's kErr* constants — keep 0 == ok).
constexpr int kNcnnOk = 0;
constexpr int kNcnnErrNotBuilt = -1;       // this .so was built without ncnn (see CMakeLists.txt)
constexpr int kNcnnErrModelMissing = -2;   // flownet.param/.bin missing in modelDir
constexpr int kNcnnErrLoadFailed = -3;     // ncnn::Net::load_param/load_model rejected a file
constexpr int kNcnnErrNotLoaded = -4;      // interpolate() called before a successful load()
constexpr int kNcnnErrBadArgs = -5;        // bad width/height/multiplier

// Runs a single-network RIFE (IFNet, e.g. rife-v4.25-lite) ncnn graph
// against a pair of RGBA8 frames, producing the frame(s) between them.
//
// This replaces the previous two-stage FlowNetLite + RefineNetLite pipeline.
// A RIFE v4.x-style net takes the two source frames plus a per-pixel
// timestep map and predicts the frame at that timestep directly in one
// forward pass — no separate refine network, and (unlike the old
// architecture) arbitrary timesteps are natively supported, so
// interpolate() computes each output frame's timestep directly
// (k / multiplier) instead of recursing through fixed t=0.5 midpoints.
//
// Model shape contract (verified against a rife-v4.25-lite_ensembleFalse
// flownet.param exported the standard rife-ncnn-vulkan way):
//   in0  = frame A, RGB, normalized to [0,1], padded up to a multiple of 32
//   in1  = frame C, same shape as in0
//   in2  = timestep map: 1-channel, same H/W as the padded frames, every
//          pixel filled with the requested timestep in [0,1]
//   out0 = predicted frame at that timestep, RGB [0,1], same padded shape
//
// The .param also references a "rife.Warp" layer type that isn't part of
// stock ncnn — RifeWarp.cpp/.hpp implement and register it as a Vulkan compute layer before load_param() is called.
//
// GPU-only policy: load() creates only a Vulkan ncnn::Net. If Vulkan is
// unavailable, load fails; there is no CPU network and no CPU scheduling/
// benchmarking fallback. The custom rife.Warp layer is also Vulkan-only.
class NcnnInterpolator {
public:
    NcnnInterpolator();
    ~NcnnInterpolator();

    NcnnInterpolator(const NcnnInterpolator &) = delete;
    NcnnInterpolator &operator=(const NcnnInterpolator &) = delete;

    // modelDir must contain flownet.param and flownet.bin (the standard
    // rife-ncnn-vulkan file names — rename an exported "flownet_param.txt"
    // to "flownet.param" if that's how it was handed to you).
    //
    // GPU-only load. allowGpu is retained for JNI compatibility and must be true;
    // numThreads is retained for ABI compatibility and is ignored.
    int load(const std::string &modelDir, bool allowGpu, int vulkanDeviceIndex, int numThreads);

    void unload();
    bool isLoaded() const;


    // frameA/frameC: interleaved RGBA8 buffers, width*height*4 bytes each —
    // the two real captured frames to interpolate between.
    // outFrames: array of (multiplier - 1) pointers, each already allocated
    // to width*height*4 bytes, that receive the interpolated frames in
    // chronological order between A and C.
    // flowScale: accepted for call-site compatibility with the old
    // FlowNetLite-based signature, but unused — a single-pass RIFE net has
    // no separate low-res flow stage to downscale. Pass anything in (0,1].
    int interpolate(const uint8_t *frameA, const uint8_t *frameC,
                     int width, int height,
                     uint8_t **outFrames, int multiplier,
                     float flowScale);

private:
    struct Impl;
    Impl *impl_;
};

// Free functions wrapping ncnn::get_gpu_count()/get_gpu_info() so callers
// (lsfg_jni.cpp's getVulkanGpuCount/getVulkanGpuName JNI exports) never need
// to include ncnn's <gpu.h> or link the `ncnn` CMake target directly. That
// target's INTERFACE_COMPILE_OPTIONS (-fno-rtti/-fno-exceptions) apply to
// every source file of any target that links it directly — which previously
// broke the real try/catch handling in lsfg_render_loop.cpp once lsfg-android
// itself linked `ncnn`. Keeping this pair of functions here, compiled into
// the isolated lsfg-ncnn-interpolator static lib alongside NcnnInterpolator,
// keeps that isolation intact while still exposing GPU probing to JNI.
int ncnnGpuCount();
std::string ncnnGpuName(int index);

} // namespace lsfg_android
