#include "cpu_core_policy.hpp"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <unistd.h>

namespace lsfg_android {

namespace {

bool readUintFile(const char *path, uint64_t &value) {
    std::ifstream in(path);
    if (!in) return false;
    return static_cast<bool>(in >> value);
}

bool cpuOnline(int cpu) {
    if (cpu == 0) return true;
    char path[128];
    std::snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/online", cpu);
    uint64_t online = 1;
    return !readUintFile(path, online) || online != 0;
}

} // namespace

CpuCorePolicy::CpuCorePolicy() {
    const long configured = sysconf(_SC_NPROCESSORS_CONF);
    const int count = configured > 0 ? static_cast<int>(configured) : 1;

    uint64_t maxMetric = 0;
    for (int cpu = 0; cpu < count; ++cpu) {
        if (!cpuOnline(cpu)) continue;
        allCpus_.push_back(cpu);
        uint64_t metric = 0;
        if (readCpuMetric(cpu, metric)) maxMetric = std::max(maxMetric, metric);
    }

    if (allCpus_.empty()) return;

    // Prefer sched capacity where available; otherwise use cpufreq max frequency.
    // CPUs at the top capacity/frequency tier are treated as the performance
    // cluster. A 90% threshold avoids accidentally classifying a slightly
    // slower big core as a little core on heterogeneous devices.
    for (int cpu : allCpus_) {
        uint64_t metric = 0;
        if (readCpuMetric(cpu, metric) && maxMetric > 0 && metric * 100 >= maxMetric * 90) {
            bigCpus_.push_back(cpu);
        }
    }

    // If topology information is unavailable or all CPUs are effectively equal,
    // there is no little/big distinction to enforce.
    if (bigCpus_.empty()) bigCpus_ = allCpus_;
}

bool CpuCorePolicy::readCpuMetric(int cpu, uint64_t &metric) {
    char path[160];

    std::snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/cpu_capacity", cpu);
    if (readUintFile(path, metric)) return true;

    std::snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq", cpu);
    if (readUintFile(path, metric)) return true;

    std::snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/cpufreq/scaling_max_freq", cpu);
    return readUintFile(path, metric);
}

void CpuCorePolicy::appendCpuSet(cpu_set_t &set, const std::vector<int> &cpus) {
    CPU_ZERO(&set);
    for (int cpu : cpus) {
        if (cpu >= 0 && cpu < CPU_SETSIZE) CPU_SET(cpu, &set);
    }
}

bool CpuCorePolicy::apply(const std::vector<int> &cpus) {
    if (cpus.empty()) return false;
    cpu_set_t set;
    appendCpuSet(set, cpus);
    return sched_setaffinity(0, sizeof(set), &set) == 0;
}

bool CpuCorePolicy::useBigCores() {
    return apply(bigCpus_);
}

bool CpuCorePolicy::useAllCores() {
    return apply(allCpus_);
}

} // namespace lsfg_android
