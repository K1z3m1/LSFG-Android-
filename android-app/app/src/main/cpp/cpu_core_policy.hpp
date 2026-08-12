#pragma once

#include <cstdint>
#include <vector>
#include <sched.h>

namespace lsfg_android {

// Android CPU policy for latency-sensitive native work.
// Starts on the highest-capacity CPU cluster and can widen to all online CPUs
// when the measured CPU-side work misses its budget. This is per-thread affinity
// and does not require root.
class CpuCorePolicy {
public:
    CpuCorePolicy();

    bool valid() const { return !allCpus_.empty(); }
    int bigCpuCount() const { return static_cast<int>(bigCpus_.size()); }
    int allCpuCount() const { return static_cast<int>(allCpus_.size()); }

    // Prefer the highest-capacity/frequency cluster. If the device has no
    // distinguishable little cluster, this simply selects all CPUs.
    bool useBigCores();

    // Widen this thread to all online CPUs. The scheduler may then use little
    // cores alongside big cores rather than forcing the workload to stay on
    // the performance cluster.
    bool useAllCores();

private:
    std::vector<int> allCpus_;
    std::vector<int> bigCpus_;

    static bool readCpuMetric(int cpu, uint64_t &metric);
    static bool apply(const std::vector<int> &cpus);
    static void appendCpuSet(cpu_set_t &set, const std::vector<int> &cpus);
};

} // namespace lsfg_android
