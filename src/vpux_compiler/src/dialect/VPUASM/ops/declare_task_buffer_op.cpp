//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUASM/ops.hpp"
#include "vpux/compiler/utils/ELF/utils.hpp"

#include <npu_40xx_nnrt.hpp>

using namespace vpux;

//
// DeclareTaskBufferOp
//

size_t VPUASM::DeclareTaskBufferOp::getBinarySize([[maybe_unused]] VPU::ArchKind arch) {
    switch (getTaskType()) {
    case VPURegMapped::TaskType::DMA:
        return sizeof(npu40xx::nn_public::VpuDMATask);
    case VPURegMapped::TaskType::ActKernelInvocation:
        return sizeof(npu40xx::nn_public::VpuActKernelInvocation);
    case VPURegMapped::TaskType::ActKernelRange:
        return sizeof(npu40xx::nn_public::VpuActKernelRange);
    case VPURegMapped::TaskType::DPUInvariant:
        return sizeof(npu40xx::nn_public::VpuDPUInvariant);
    case VPURegMapped::TaskType::DPUVariant:
        return sizeof(npu40xx::nn_public::VpuDPUVariant);
    case VPURegMapped::TaskType::M2I:
        return sizeof(npu40xx::nn_public::VpuMediaTask);
    default:
        VPUX_THROW("Invalid task type for DeclareTaskBufferOp {0}", *this);
    }
}

size_t VPUASM::DeclareTaskBufferOp::getAlignmentRequirements(VPU::ArchKind) {
    return ELF::VPUX_NO_ALIGNMENT;
}

std::optional<ELF::SectionSignature> vpux::VPUASM::DeclareTaskBufferOp::getSectionSignature() {
    return ELF::SectionSignature(vpux::ELF::generateSignature("program", "metadata", "cmx"),
                                 ELF::SectionFlagsAttr::SHF_NONE, ELF::SectionTypeAttr::VPU_SHT_CMX_METADATA);
}

bool vpux::VPUASM::DeclareTaskBufferOp::hasMemoryFootprint() {
    return false;
}

void VPUASM::DeclareTaskBufferOp::setMemoryOffset(mlir::IntegerAttr offset) {
    setOffsetAttr(offset);
}

uint64_t VPUASM::DeclareTaskBufferOp::getMemoryOffset() {
    return getOffset().value_or(0);
}
