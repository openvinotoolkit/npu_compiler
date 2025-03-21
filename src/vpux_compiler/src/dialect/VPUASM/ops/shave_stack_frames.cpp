//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUASM/ops.hpp"
#include "vpux/compiler/utils/ELF/utils.hpp"

using namespace vpux;

//
// ShaveStackFrameOp
//

size_t vpux::VPUASM::ShaveStackFrameOp::getBinarySizeCached(ELF::SymbolReferenceMap&, VPU::ArchKind) {
    return getStackSize();
}

size_t vpux::VPUASM::ShaveStackFrameOp::getAlignmentRequirements(VPU::ArchKind) {
    return ELF::VPUX_DEFAULT_ALIGNMENT;
}

vpux::ELF::SectionFlagsAttr vpux::VPUASM::ShaveStackFrameOp::getPredefinedMemoryAccessors() {
    return (ELF::SectionFlagsAttr::VPU_SHF_PROC_SHAVE);
}

std::optional<ELF::SectionSignature> vpux::VPUASM::ShaveStackFrameOp::getSectionSignature() {
    return ELF::SectionSignature(vpux::ELF::generateSignature("shave", "stack"), ELF::SectionFlagsAttr::SHF_ALLOC,
                                 ELF::SectionTypeAttr::SHT_NOBITS);
}

bool vpux::VPUASM::ShaveStackFrameOp::hasMemoryFootprint() {
    return false;
}
