#include "ncnn_cpu_policy.hpp"

#include <algorithm>
#include <cpu.h>

namespace lsfg_android {

void ncnnCpuSetBigFirst() {
    if (ncnn::get_big_cpu_count() > 0) {
        ncnn::set_cpu_powersave(2); // big cluster only
        ncnn::set_omp_num_threads(std::max(1, ncnn::get_big_cpu_count()));
    } else {
        ncnn::set_cpu_powersave(0);
        ncnn::set_omp_num_threads(std::max(1, ncnn::get_cpu_count()));
    }
}

void ncnnCpuEnableLittleAssist(bool enable) {
    if (enable) {
        // 0 = all clusters. This lets little cores contribute only after the
        // render-loop CPU budget says the performance cluster is insufficient.
        ncnn::set_cpu_powersave(0);
        ncnn::set_omp_num_threads(std::max(1, ncnn::get_cpu_count()));
    } else {
        ncnnCpuSetBigFirst();
    }
}

int ncnnCpuBigThreadCount() {
    return std::max(1, ncnn::get_big_cpu_count());
}

int ncnnCpuAllThreadCount() {
    return std::max(1, ncnn::get_cpu_count());
}

} // namespace lsfg_android
