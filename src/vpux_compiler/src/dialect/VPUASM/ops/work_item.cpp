//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/ELF/utils/utils.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"

using namespace vpux;

//
// WorkItemOp
//

vpux::ELF::SectionFlagsAttr vpux::VPUASM::WorkItemOp::getPredefinedMemoryAccessors() {
    return (ELF::SectionFlagsAttr::SHF_EXECINSTR);
}

std::optional<ELF::SectionSignature> vpux::VPUASM::WorkItemOp::getSectionSignature() {
    return ELF::SectionSignature(vpux::ELF::generateSignature("program", "workItem"), ELF::SectionFlagsAttr::SHF_ALLOC);
}

bool vpux::VPUASM::WorkItemOp::hasMemoryFootprint() {
    return true;
}
