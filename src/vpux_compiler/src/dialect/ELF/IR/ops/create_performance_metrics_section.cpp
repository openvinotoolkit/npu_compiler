//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpux_elf/writer.hpp>
#include "vpux/compiler/dialect/ELF/utils/utils.hpp"
#include "vpux/compiler/dialect/VPUASM/performance_metrics_utils.hpp"

#include <npu_40xx_nnrt.hpp>

using namespace vpux;
using namespace npu40xx;

void vpux::ELF::PerformanceMetricsOp::serialize(elf::writer::BinaryDataSection<uint8_t>& binDataSection) {
    VpuPerformanceMetrics perf{};
    auto operation = getOperation();
    auto mainModule = operation->getParentOfType<mlir::ModuleOp>();
    VPUASM::populatePerformanceMetrics(perf, mainModule);

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
