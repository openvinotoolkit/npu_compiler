//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUASM/ops.hpp"
#include "vpux/compiler/utils/ELF/utils.hpp"

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
