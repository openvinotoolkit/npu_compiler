//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <mlir/IR/BuiltinTypes.h>
#include "vpux/compiler/dialect/VPUASM/ops.hpp"
#include "vpux/compiler/utils/ELF/utils.hpp"

using namespace vpux;

//
// M2IOp
//

ELF::SectionFlagsAttr VPUASM::M2IOp::getPredefinedMemoryAccessors() {
    return ELF::SectionFlagsAttr::SHF_EXECINSTR | ELF::SectionFlagsAttr::VPU_SHF_PROC_DMA;
}

std::optional<ELF::SectionSignature> VPUASM::M2IOp::getSectionSignature() {
    return ELF::SectionSignature(ELF::generateSignature("task", "m2i", getTaskIndex()),
                                 ELF::SectionFlagsAttr::SHF_ALLOC);
}

bool VPUASM::M2IOp::hasMemoryFootprint() {
    return true;
}
