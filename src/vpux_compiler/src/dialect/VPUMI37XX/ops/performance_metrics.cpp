//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUASM/performance_metrics_utils.hpp"
#include "vpux/compiler/dialect/VPUMI37XX/ops.hpp"

#include <npu_37xx_nnrt.hpp>

using namespace vpux;
using namespace npu37xx;

//
// PerformanceMetrics
//

void vpux::VPUMI37XX::PerformanceMetricsOp::serialize(elf::writer::BinaryDataSection<uint8_t>& binDataSection) {
    VpuPerformanceMetrics perf{};
    auto operation = getOperation();
    auto mainModule = operation->getParentOfType<mlir::ModuleOp>();
    VPUASM::populatePerformanceMetrics(perf, mainModule);

    const auto ptrCharTmp = reinterpret_cast<uint8_t*>(&perf);
    binDataSection.appendData(ptrCharTmp, getBinarySize());
}

size_t vpux::VPUMI37XX::PerformanceMetricsOp::getBinarySize() {
    return sizeof(VpuPerformanceMetrics);
}

size_t vpux::VPUMI37XX::PerformanceMetricsOp::getAlignmentRequirements() {
    return alignof(VpuPerformanceMetrics);
}

vpux::ELFNPU37XX::SectionFlagsAttr vpux::VPUMI37XX::PerformanceMetricsOp::getAccessingProcs() {
    return (ELFNPU37XX::SectionFlagsAttr::SHF_NONE);
}

vpux::ELFNPU37XX::SectionFlagsAttr vpux::VPUMI37XX::PerformanceMetricsOp::getUserProcs() {
    return (ELFNPU37XX::SectionFlagsAttr::SHF_NONE);
}

vpux::VPURT::BufferSection vpux::VPUMI37XX::PerformanceMetricsOp::getMemorySpace() {
    return vpux::VPURT::BufferSection::DDR;
}
