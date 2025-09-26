//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUASM/ops.hpp"
#include "vpux/compiler/utils/ELF/utils.hpp"

using namespace vpux;

//
// ShaveStackFrameOp
//

size_t vpux::VPUASM::ShaveStackFrameOp::getBinarySizeCached(ELF::SymbolReferenceMap&, config::ArchKind) {
    return getStackSize();
}

size_t vpux::VPUASM::ShaveStackFrameOp::getAlignmentRequirements(config::ArchKind) {
    return ELF::VPUX_SHAVE_ALIGNMENT;
}

vpux::ELF::SectionFlagsAttr vpux::VPUASM::ShaveStackFrameOp::getPredefinedMemoryAccessors() {
    return (ELF::SectionFlagsAttr::VPU_SHF_PROC_SHAVE);
}

std::optional<ELF::SectionSignature> vpux::VPUASM::ShaveStackFrameOp::getSectionSignature() {
    return ELF::SectionSignature(vpux::ELF::generateSignature("shave", "stack"),
                                 ELF::SectionFlagsAttr::SHF_ALLOC | ELF::SectionFlagsAttr::SHF_WRITE,
                                 ELF::SectionTypeAttr::SHT_NOBITS);
}

bool vpux::VPUASM::ShaveStackFrameOp::hasMemoryFootprint() {
    return false;
}

vpux::VPURT::BufferSection VPUASM::ShaveStackFrameOp::getMemorySection() {
    return VPURT::BufferSection::DDR;
}
