//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/ELF/utils/utils.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"

using namespace vpux;

//
// DpuPVPOp
//

vpux::ELF::SectionFlagsAttr vpux::VPUASM::DpuPVPOp::getPredefinedMemoryAccessors() {
    return ELF::SectionFlagsAttr::SHF_NONE;
}

std::optional<ELF::SectionSignature> vpux::VPUASM::DpuPVPOp::getSectionSignature() {
    return ELF::SectionSignature(vpux::ELF::generateSignature("dpu", "pvp"), ELF::SectionFlagsAttr::SHF_ALLOC);
}

bool vpux::VPUASM::DpuPVPOp::hasMemoryFootprint() {
    return true;
}
