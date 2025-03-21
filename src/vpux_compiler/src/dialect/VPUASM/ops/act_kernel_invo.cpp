//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUASM/ops.hpp"
#include "vpux/compiler/utils/ELF/utils.hpp"

using namespace vpux;

//
// ActKernelInvocationOp
//

vpux::ELF::SectionFlagsAttr vpux::VPUASM::ActKernelInvocationOp::getPredefinedMemoryAccessors() {
    return ELF::SectionFlagsAttr::SHF_EXECINSTR | ELF::SectionFlagsAttr::VPU_SHF_PROC_DMA;
}

std::optional<ELF::SectionSignature> vpux::VPUASM::ActKernelInvocationOp::getSectionSignature() {
    return ELF::SectionSignature(vpux::ELF::generateSignature("task", "shave", "invocation", getTaskIndex()),
                                 ELF::SectionFlagsAttr::SHF_ALLOC);
}

bool vpux::VPUASM::ActKernelInvocationOp::hasMemoryFootprint() {
    return true;
}
