//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/performance_metrics.hpp"
#include "vpux/compiler/dialect/config/IR/ops.hpp"
#include "vpux/compiler/dialect/config/constraints.hpp"
#include "vpux/compiler/dialect/net/IR/ops.hpp"
#include "vpux/compiler/dialect/net/utils/network_info_utils.hpp"

#include <cstdint>

namespace vpux {
namespace VPU {

// Base of bandwidth values used in tables (in MB/s).
static constexpr uint32_t BW_BASE = 2000;
// Step of bandwidth values used in tables (in MB/s).
static constexpr uint32_t BW_STEP = 100;
// Num entries in table, each entry contains set of values for particular frequency
static constexpr uint32_t NUM_ENTRIES = 5;
// Profiling timer block fixed frequency by MHz
// For MTL it runs at 38.4 MHz
// TODO: it should be provided by vpunn API per arch
static constexpr double PROF_CLK = 38.4;
// Recommended default activity factor by VPUNN team.
static constexpr double DEFAULT_ACTIVITY_FACTOR = 0.5;

uint32_t getBWBase() {
    return BW_BASE;
}
uint32_t getBWStep() {
    return BW_STEP;
}
uint32_t getNumEntries() {
    return NUM_ENTRIES;
}

double getProfClk() {
    return PROF_CLK;
}

const SmallVector<float>& getBWScales() {
    // value in [0.0..1.0] range indicating scalability of network for a given DDR bandwidth.
    static const SmallVector<float> byBWScales({0.0F, 0.2F, 0.4F, 0.6F, 0.8F});
    return byBWScales;
}

SmallVector<SmallVector<uint64_t>> getBWTicks(mlir::ModuleOp module) {
    SmallVector<SmallVector<uint64_t>> ret;
    ret.reserve(VPU::getNumEntries());
    // expected ticks (based on FRC @38.4MHz) an inference should take for a given DDR bandwidth.
    SmallVector<uint64_t> byBWTicks({0UL, 0UL, 0UL, 0UL, 0UL});
    // inferenceTime table will be zero defaultly when InferenceExecutionAnalysisPass disabled
    // In this way the runtime will be able to use measured ticks instead
    size_t inferenceTimebyDPUCycle = 0;
    auto netInfo = net::getNetworkInfo(module);
    if (netInfo.getInferenceTiming().has_value()) {
        inferenceTimebyDPUCycle = netInfo.getInferenceTiming().value();
    } else {
        vpux::Logger::global().warning("NetworkInfoOp {0} doesn't have value in InferenceTiming attribute. Will use "
                                       "default InferenceTiming and activity factor value instead",
                                       netInfo);
    }

    const auto& constraints = config::getNPUConstraints(module->getContext());
    const auto perfClk = constraints.perfClock.defaultFreq;
    const auto& freqTable = constraints.frequencyTable;
    for (uint32_t i = 0; i < VPU::getNumEntries(); ++i) {
        auto dpuFreq = freqTable.base + i * freqTable.step;
        auto duration = static_cast<double>(inferenceTimebyDPUCycle) / static_cast<double>(dpuFreq);
        auto ticksByDPUFreq = duration * perfClk;
        // TODO: Scale ticks by dma bandwidth
        // Currently ignore bandwidth scaling, put same ticks for all bw steps
        for (uint32_t j = 0; j < VPU::getNumEntries(); ++j) {
            byBWTicks[j] = static_cast<uint64_t>(ticksByDPUFreq);
        }
        ret.push_back(byBWTicks);
    }
    Logger::global().debug("BWTicks table: {0}, total cycle {1}, freq base {2}, step {3}, perf clock {4}", ret,
                           inferenceTimebyDPUCycle, freqTable.base, freqTable.step, perfClk);
    return ret;
}

double getActivityFactor(config::ExecutorKind execKind, config::ComputeResourceOpInterface res) {
    // 0.5 is a recommended default value for AF by VPUNN team
    double activityFactor = DEFAULT_ACTIVITY_FACTOR;
    if (execKind != config::ExecutorKind::NCE && execKind != config::ExecutorKind::SHAVE_NN) {
        return INVALID_AF;
    }

    // Here we must get AF from NCE res (a ResourcesOp) as the AF attribute is attached to tile op
    if (execKind == config::ExecutorKind::NCE) {
        // nceRes is a ResourcesOp, which corresponds to the tile-level container.
        auto nceRes = mlir::cast<config::ResourcesOp>(res.getOperation());
        if (auto factorAttr = nceRes.getActivityFactorAttr()) {
            activityFactor = factorAttr.getValue().convertToDouble();
        }
    } else if (execKind == config::ExecutorKind::SHAVE_NN) {
        auto shaveRes = mlir::cast<config::ExecutorResourceOp>(res.getOperation());
        if (auto factorAttr = shaveRes.getActivityFactorAttr()) {
            activityFactor = factorAttr.getValue().convertToDouble();
        }
    }

    // In below situation, activityFactor may to be >1
    // 1) when the energy reference is not the maximum powervirus. Eg: the powerVirus for INT is smaller
    // than powerVirus for FLOAT. Now we are using INT8 powervirus (for NPU2.7 /w v1.5.9 VPUNN releases) as
    // max power reference so that AF>1 is possible 2) If inferenceTime estimation is smaller than the
    // Energy estimated in powervirusDPUCycles their ratio will be >1. This is transitory because in the
    // real world the measured time will be bigger and the RuntimeNN will normalize the numbers considering
    // real execution time. Eg: NewAF = (OldAF * CompledInferedTimeSmall) / measuredTimeBig. This scenario
    // might happen in some extreme cases.
    if (activityFactor < 0 || activityFactor > 1) {
        vpux::Logger::global().warning("Activity factor value should be in range [0, 1] for general situation "
                                       "but got {0}. Some unexpected cases may happen",
                                       activityFactor);
    }

    return activityFactor;
}

}  // namespace VPU
}  // namespace vpux
