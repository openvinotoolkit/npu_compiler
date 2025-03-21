//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUASM/ops.hpp"
#include "vpux/compiler/utils/ELF/utils.hpp"

using namespace vpux;

//
// TAskSinkOp
//

size_t VPUASM::TaskSinkOp::getAlignmentRequirements(VPU::ArchKind) {
    return sizeof(uint32_t);
}

ELF::SectionFlagsAttr VPUASM::TaskSinkOp::getPredefinedMemoryAccessors() {
    return ELF::SectionFlagsAttr::VPU_SHF_PROC_DMA;
}

void VPUASM::TaskSinkOp::setMemoryOffset(mlir::IntegerAttr) {
    return;
}

uint64_t VPUASM::TaskSinkOp::getMemoryOffset() {
    // TODO: E110144
    const uint64_t FIFO_OFFSET = 31415;
    return FIFO_OFFSET;
}

std::optional<ELF::SectionSignature> vpux::VPUASM::TaskSinkOp::getSectionSignature() {
    return ELF::SectionSignature(vpux::ELF::generateSignature("buffer", VPURT::BufferSection::CMX_NN, getTile()),
                                 ELF::SectionFlagsAttr::SHF_ALLOC);
}

bool vpux::VPUASM::TaskSinkOp::hasMemoryFootprint() {
    return false;
}
