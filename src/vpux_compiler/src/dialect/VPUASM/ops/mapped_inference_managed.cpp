//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUASM/ops.hpp"
#include "vpux/compiler/utils/ELF/utils.hpp"

using namespace vpux;

//
// ManagedMappedInferenceOp
//

vpux::ELF::SectionFlagsAttr vpux::VPUASM::ManagedMappedInferenceOp::getPredefinedMemoryAccessors() {
    return (ELF::SectionFlagsAttr::SHF_EXECINSTR);
}

std::optional<ELF::SectionSignature> vpux::VPUASM::ManagedMappedInferenceOp::getSectionSignature() {
    return ELF::SectionSignature(vpux::ELF::generateSignature("program", "mapped_inference"),
                                 ELF::SectionFlagsAttr::SHF_ALLOC);
}

bool vpux::VPUASM::ManagedMappedInferenceOp::hasMemoryFootprint() {
    return true;
}
