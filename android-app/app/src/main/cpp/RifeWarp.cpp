#include "RifeWarp.hpp"

#include <gpu.h>
#include <pipeline.h>

namespace lsfg_android {
namespace {

static constexpr char kWarpComp[] = R"glsl(
#version 450
#if NCNN_fp16_storage
#extension GL_EXT_shader_16bit_storage: require
#endif
#if NCNN_fp16_arithmetic
#extension GL_EXT_shader_explicit_arithmetic_types_float16: require
#endif
layout (binding = 0) readonly buffer image_blob { sfp image_blob_data[]; };
layout (binding = 1) readonly buffer flow_blob { sfp flow_blob_data[]; };
layout (binding = 2) writeonly buffer top_blob { sfp top_blob_data[]; };
layout (push_constant) uniform parameter { int w; int h; int c; int cstep; } p;
void main()
{
    int gx = int(gl_GlobalInvocationID.x);
    int gy = int(gl_GlobalInvocationID.y);
    int gz = int(gl_GlobalInvocationID.z);
    if (gx >= p.w || gy >= p.h || gz >= p.c) return;

    afp flow_x = buffer_ld1(flow_blob_data, gy * p.w + gx);
    afp flow_y = buffer_ld1(flow_blob_data, p.cstep + gy * p.w + gx);
    afp sample_x = afp(gx) + flow_x;
    afp sample_y = afp(gy) + flow_y;

    int x0 = int(floor(sample_x));
    int y0 = int(floor(sample_y));
    int x1 = x0 + 1;
    int y1 = y0 + 1;
    x0 = clamp(x0, 0, p.w - 1);
    y0 = clamp(y0, 0, p.h - 1);
    x1 = clamp(x1, 0, p.w - 1);
    y1 = clamp(y1, 0, p.h - 1);

    afp alpha = sample_x - afp(x0);
    afp beta = sample_y - afp(y0);
    afp v0 = buffer_ld1(image_blob_data, gz * p.cstep + y0 * p.w + x0);
    afp v1 = buffer_ld1(image_blob_data, gz * p.cstep + y0 * p.w + x1);
    afp v2 = buffer_ld1(image_blob_data, gz * p.cstep + y1 * p.w + x0);
    afp v3 = buffer_ld1(image_blob_data, gz * p.cstep + y1 * p.w + x1);
    afp v4 = v0 * (afp(1.f) - alpha) + v1 * alpha;
    afp v5 = v2 * (afp(1.f) - alpha) + v3 * alpha;
    afp v = v4 * (afp(1.f) - beta) + v5 * beta;

    const int gi = gz * p.cstep + gy * p.w + gx;
    buffer_st1(top_blob_data, gi, v);
}
)glsl";

} // namespace

RifeWarp::RifeWarp()
{
    one_blob_only = false;
    support_inplace = false;
    support_vulkan = true;
    // Keep the custom layer in pack1. ncnn inserts GPU packing conversion
    // around the layer when the surrounding graph uses pack4/pack8.
    support_vulkan_packing = false;
    support_vulkan_any_packing = false;
}

int RifeWarp::create_pipeline(const ncnn::Option& opt)
{
#if NCNN_VULKAN
    if (!vkdev) return -1;
    std::vector<uint32_t> spirv;
    if (ncnn::compile_spirv_module(kWarpComp, (int)sizeof(kWarpComp) - 1, opt, spirv) != 0)
        return -1;
    pipeline_warp = new ncnn::Pipeline(vkdev);
    pipeline_warp->set_optimal_local_size_xyz();
    std::vector<ncnn::vk_specialization_type> specs;
    if (pipeline_warp->create(spirv.data(), spirv.size() * sizeof(uint32_t), specs) != 0) {
        delete pipeline_warp;
        pipeline_warp = nullptr;
        return -1;
    }
    return 0;
#else
    (void)opt;
    return -1;
#endif
}

int RifeWarp::destroy_pipeline(const ncnn::Option& opt)
{
    (void)opt;
#if NCNN_VULKAN
    delete pipeline_warp;
    pipeline_warp = nullptr;
#endif
    return 0;
}

int RifeWarp::forward(const std::vector<ncnn::Mat>&,
                      std::vector<ncnn::Mat>&,
                      const ncnn::Option&) const
{
    // Hard GPU-only policy: never execute this layer on CPU.
    return -1001;
}

#if NCNN_VULKAN
int RifeWarp::forward(const std::vector<ncnn::VkMat>& bottom_blobs,
                      std::vector<ncnn::VkMat>& top_blobs,
                      ncnn::VkCompute& cmd,
                      const ncnn::Option& opt) const
{
    if (!pipeline_warp || bottom_blobs.size() < 2 || top_blobs.empty())
        return -1001;

    const ncnn::VkMat& image_blob = bottom_blobs[0];
    const ncnn::VkMat& flow_blob = bottom_blobs[1];
    ncnn::VkMat& top_blob = top_blobs[0];

    top_blob.create(image_blob.w, image_blob.h, image_blob.c,
                    image_blob.elemsize, 1, opt.blob_vkallocator);
    if (top_blob.empty()) return -100;

    std::vector<ncnn::VkMat> bindings(3);
    bindings[0] = image_blob;
    bindings[1] = flow_blob;
    bindings[2] = top_blob;

    std::vector<ncnn::vk_constant_type> constants(4);
    constants[0].i = top_blob.w;
    constants[1].i = top_blob.h;
    constants[2].i = top_blob.c;
    constants[3].i = top_blob.cstep;
    cmd.record_pipeline(pipeline_warp, bindings, constants, top_blob);
    return 0;
}
#endif

} // namespace lsfg_android
