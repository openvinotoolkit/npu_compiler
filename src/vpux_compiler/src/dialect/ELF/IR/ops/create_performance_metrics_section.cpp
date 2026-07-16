//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpux_elf/writer.hpp>
#include "vpux/compiler/dialect/ELF/utils/utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/performance_metrics.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/dialect/config/constraints.hpp"

#include <algorithm>
#include <cstring>

#include <npu_40xx_nnrt.hpp>

using namespace vpux;
using namespace npu40xx;

void vpux::ELF::PerformanceMetricsOp::serialize(elf::writer::BinaryDataSection<uint8_t>& binDataSection) {
    VpuPerformanceMetrics perf{};
    auto operation = getOperation();
    auto mainModule = operation->getParentOfType<mlir::ModuleOp>();

    const auto& freqTable = config::getNPUConstraints(mainModule->getContext()).frequencyTable;
    perf.freq_base = freqTable.base;
    perf.freq_step = freqTable.step;
    perf.bw_base = VPU::getBWBase();
    perf.bw_step = VPU::getBWStep();

    // Here we must get AF from NCE res (a config::ResourcesOp) as the AF attribute is attached to tile op
    auto tileResources = config::getTileExecutor(mainModule);
    const auto execKind = config::getKindValue<config::ExecutorKind>(tileResources);
    if (config::ExecutorKind::NCE == execKind) {
        perf.activity_factor = static_cast<float>(VPU::getActivityFactor(execKind, tileResources));
    }
    VPUX_THROW_WHEN(perf.activity_factor == VPU::INVALID_AF, "Invalid activity factor {0}!", perf.activity_factor);

    const auto numEntries = VPU::getNumEntries();
    auto& byBWScales = VPU::getBWScales();
    auto byBWTicks = VPU::getBWTicks(mainModule);
    for (size_t row = 0; row < numEntries; ++row) {
        std::copy_n(&byBWScales[0], numEntries, &perf.scalability[row][0]);
        std::copy_n(&byBWTicks[row][0], numEntries, &perf.ticks[row][0]);
    }

    const auto ptrCharTmp = reinterpret_cast<uint8_t*>(&perf);
    binDataSection.appendData(ptrCharTmp, getBinarySize(config::ArchKind::UNKNOWN));
}

size_t vpux::ELF::PerformanceMetricsOp::getBinarySize(config::ArchKind) {
    return sizeof(VpuPerformanceMetrics);
}

size_t vpux::ELF::PerformanceMetricsOp::getAlignmentRequirements(config::ArchKind) {
    return alignof(VpuPerformanceMetrics);
}

std::optional<ELF::SectionSignature> vpux::ELF::PerformanceMetricsOp::getSectionSignature() {
    return ELF::SectionSignature(vpux::ELF::generateSignature("perf", "metrics"), ELF::SectionFlagsAttr::SHF_NONE,
                                 ELF::SectionTypeAttr::VPU_SHT_PERF_METRICS);
}

bool vpux::ELF::PerformanceMetricsOp::hasMemoryFootprint() {
    return true;
}

void vpux::ELF::PerformanceMetricsOp::build(mlir::OpBuilder& builder, mlir::OperationState& state) {
    build(builder, state, "PerfMetrics");
}
