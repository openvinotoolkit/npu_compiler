//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/ELF/utils/utils.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"

using namespace vpux;

//
// ActShaveRtOp
//

vpux::ELF::SectionFlagsAttr vpux::VPUASM::ActShaveRtOp::getPredefinedMemoryAccessors() {
    return (ELF::SectionFlagsAttr::VPU_SHF_PROC_SHAVE);
}

std::optional<ELF::SectionSignature> vpux::VPUASM::ActShaveRtOp::getSectionSignature() {
    return ELF::SectionSignature(vpux::ELF::generateSignature("shave", "runtime"), ELF::SectionFlagsAttr::SHF_ALLOC);
}

bool vpux::VPUASM::ActShaveRtOp::hasMemoryFootprint() {
    return true;
}
