//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/ELF/utils/utils.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"

using namespace vpux;

//
// nnrtConfigOp
//

vpux::ELF::SectionFlagsAttr vpux::VPUASM::NNrtConfigOp::getPredefinedMemoryAccessors() {
    return (ELF::SectionFlagsAttr::SHF_EXECINSTR);
}

std::optional<ELF::SectionSignature> vpux::VPUASM::NNrtConfigOp::getSectionSignature() {
    return ELF::SectionSignature(vpux::ELF::generateSignature("program", "nnrt_config"),
                                 ELF::SectionFlagsAttr::SHF_ALLOC);
}

bool vpux::VPUASM::NNrtConfigOp::hasMemoryFootprint() {
    return true;
}
