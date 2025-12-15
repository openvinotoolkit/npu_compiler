//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include <mlir/IR/BuiltinTypes.h>
#include "vpux/compiler/dialect/ELF/utils/utils.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"

using namespace vpux;

//
// ActKernelRangeOp
//

vpux::ELF::SectionFlagsAttr vpux::VPUASM::ActKernelRangeOp::getPredefinedMemoryAccessors() {
    return ELF::SectionFlagsAttr::SHF_EXECINSTR | ELF::SectionFlagsAttr::VPU_SHF_PROC_DMA;
}

std::optional<ELF::SectionSignature> vpux::VPUASM::ActKernelRangeOp::getSectionSignature() {
    return ELF::SectionSignature(vpux::ELF::generateSignature("task", "shave", "range", getTaskIndex()),
                                 ELF::SectionFlagsAttr::SHF_ALLOC);
}

bool vpux::VPUASM::ActKernelRangeOp::hasMemoryFootprint() {
    return true;
}
