//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUASM/ops.hpp"
#include "vpux/compiler/utils/ELF/utils.hpp"

using namespace vpux;

//
// DeclareBufferOp
//

void VPUASM::DeclareTaskAddrBufOp::serialize(elf::writer::BinaryDataSection<uint8_t>& binDataSection) {
    auto size = getBinarySize(VPU::ArchKind::UNKNOWN);
    std::vector<uint8_t> tmpBuf(size, 0);
    binDataSection.appendData(tmpBuf.data(), size);
    return;
}

size_t VPUASM::DeclareTaskAddrBufOp::getBinarySize(VPU::ArchKind) {
    const auto type = getBufferType().getMemref().cast<vpux::NDTypeInterface>();
    return type.getTotalAllocSize().count();
}

size_t VPUASM::DeclareTaskAddrBufOp::getAlignmentRequirements(VPU::ArchKind) {
    return sizeof(uint32_t);
}

ELF::SectionFlagsAttr VPUASM::DeclareTaskAddrBufOp::getPredefinedMemoryAccessors() {
    return ELF::SectionFlagsAttr::VPU_SHF_PROC_DMA;
}

void VPUASM::DeclareTaskAddrBufOp::setMemoryOffset(mlir::IntegerAttr offset) {
    if (getBufferType().getLocation().getSection() == VPURT::BufferSection::DDR) {
        getOperation()->setAttr(ELF::BinaryOpInterface::elfMemOffsetAttrName(), offset);
    }
    return;
}

uint64_t VPUASM::DeclareTaskAddrBufOp::getMemoryOffset() {
    auto location = getBufferType().getLocation();
    if (location.getSection() == VPURT::BufferSection::DDR) {
        auto op = getOperation();
        auto memOffsetAttrName = ELF::BinaryOpInterface::elfMemOffsetAttrName();
        if (op->hasAttr(memOffsetAttrName)) {
            auto attr = op->getAttrOfType<mlir::IntegerAttr>(memOffsetAttrName);
            return attr.getUInt();
        } else {
            return 0;
        }
    } else {
        return location.getByteOffset();
    }
}

std::optional<ELF::SectionSignature> vpux::VPUASM::DeclareTaskAddrBufOp::getSectionSignature() {
    auto location = getBufferType().getLocation();
    if (location.getSection() == VPURT::BufferSection::DDR) {
        return ELF::SectionSignature(ELF::generateSignature("buffer", getBufferType(), "WLMPtrs"),
                                     ELF::SectionFlagsAttr::SHF_ALLOC);
    } else if (location.getSection() == VPURT::BufferSection::CMX_NN) {
        return ELF::SectionSignature(vpux::ELF::generateSignature("buffer", getBufferType()),
                                     ELF::SectionFlagsAttr::SHF_NONE);
    }
    VPUX_THROW("unexpected data location");

    return std::nullopt;
}

bool vpux::VPUASM::DeclareTaskAddrBufOp::hasMemoryFootprint() {
    return getBufferType().getLocation().getSection() == VPURT::BufferSection::DDR;
}
