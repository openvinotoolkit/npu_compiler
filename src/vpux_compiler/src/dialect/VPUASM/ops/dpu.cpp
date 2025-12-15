//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/ELF/utils/utils.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"

#include <npu_40xx_nnrt.hpp>

using namespace vpux;

vpux::ELF::SectionFlagsAttr vpux::VPUASM::DPUInvariantOp::getPredefinedMemoryAccessors() {
    // DPU can't access DDR, therefore DPU descriptors are copied from DDR to metadata in CMX by DMA
    return ELF::SectionFlagsAttr::SHF_EXECINSTR | ELF::SectionFlagsAttr::VPU_SHF_PROC_DMA;
}

std::optional<ELF::SectionSignature> vpux::VPUASM::DPUInvariantOp::getSectionSignature() {
    return ELF::SectionSignature(vpux::ELF::generateSignature("task", "dpu", "invariant", getTaskIndex()),
                                 ELF::SectionFlagsAttr::SHF_ALLOC);
}

bool vpux::VPUASM::DPUInvariantOp::hasMemoryFootprint() {
    return true;
}

vpux::ELF::SectionFlagsAttr vpux::VPUASM::DPUVariantOp::getPredefinedMemoryAccessors() {
    return ELF::SectionFlagsAttr::SHF_EXECINSTR | ELF::SectionFlagsAttr::VPU_SHF_PROC_DMA;
}

std::optional<ELF::SectionSignature> vpux::VPUASM::DPUVariantOp::getSectionSignature() {
    return ELF::SectionSignature(vpux::ELF::generateSignature("task", "dpu", "variant", getTaskIndex()),
                                 ELF::SectionFlagsAttr::SHF_ALLOC);
}

bool vpux::VPUASM::DPUVariantOp::hasMemoryFootprint() {
    return true;
}

size_t vpux::VPUASM::DPUInvariantOp_37XX::getBinarySize(config::ArchKind) {
    return sizeof(npu40xx::nn_public::VpuDPUInvariant);
}

size_t vpux::VPUASM::DPUInvariantOp_37XX::getAlignmentRequirements(config::ArchKind) {
    return alignof(npu40xx::nn_public::VpuDPUInvariant);
}

vpux::ELF::SectionFlagsAttr vpux::VPUASM::DPUInvariantOp_37XX::getPredefinedMemoryAccessors() {
    return ELF::SectionFlagsAttr::SHF_EXECINSTR | ELF::SectionFlagsAttr::VPU_SHF_PROC_DMA;
}

std::optional<ELF::SectionSignature> vpux::VPUASM::DPUInvariantOp_37XX::getSectionSignature() {
    return ELF::SectionSignature("text.invariants", ELF::SectionFlagsAttr::SHF_ALLOC);
}

bool vpux::VPUASM::DPUInvariantOp_37XX::hasMemoryFootprint() {
    return true;
}

size_t vpux::VPUASM::DPUVariantOp_37XX::getBinarySize(config::ArchKind) {
    return sizeof(npu40xx::nn_public::VpuDPUVariant);
}

size_t vpux::VPUASM::DPUVariantOp_37XX::getAlignmentRequirements(config::ArchKind) {
    return alignof(npu40xx::nn_public::VpuDPUVariant);
}

vpux::ELF::SectionFlagsAttr vpux::VPUASM::DPUVariantOp_37XX::getPredefinedMemoryAccessors() {
    return ELF::SectionFlagsAttr::SHF_EXECINSTR | ELF::SectionFlagsAttr::VPU_SHF_PROC_DMA;
}

std::optional<ELF::SectionSignature> vpux::VPUASM::DPUVariantOp_37XX::getSectionSignature() {
    return ELF::SectionSignature("text.variants", ELF::SectionFlagsAttr::SHF_ALLOC);
}

bool vpux::VPUASM::DPUVariantOp_37XX::hasMemoryFootprint() {
    return true;
}
