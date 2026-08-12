#pragma once

namespace lsfg_android {

// ncnn's Android CPU scheduler. Vulkan model layers remain GPU-only; these
// functions only control unavoidable host-side preprocessing/OpenMP/custom
// layer work. Start on big cores and widen to all cores on overload.
void ncnnCpuSetBigFirst();
void ncnnCpuEnableLittleAssist(bool enable);
int ncnnCpuBigThreadCount();
int ncnnCpuAllThreadCount();

} // namespace lsfg_android
