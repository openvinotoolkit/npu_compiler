//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/ELF/utils/utils.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"

using namespace vpux;

//
// BootstrapOp
//

void vpux::VPUASM::BootstrapOp::serialize(elf::writer::BinaryDataSection<uint8_t>& binaryDataSection) {
    uint32_t barId = getBarrierId();
    auto ptrCharTmp = reinterpret_cast<uint8_t*>(&barId);
    binaryDataSection.appendData(ptrCharTmp, getBinarySize(config::ArchKind::UNKNOWN));
}

size_t vpux::VPUASM::BootstrapOp::getBinarySize(config::ArchKind) {
    return sizeof(uint32_t);
}

size_t vpux::VPUASM::BootstrapOp::getAlignmentRequirements(config::ArchKind) {
    return alignof(uint32_t);
}

vpux::ELF::SectionFlagsAttr vpux::VPUASM::BootstrapOp::getPredefinedMemoryAccessors() {
    return (ELF::SectionFlagsAttr::SHF_EXECINSTR);
}

std::optional<ELF::SectionSignature> vpux::VPUASM::BootstrapOp::getSectionSignature() {
    return ELF::SectionSignature(vpux::ELF::generateSignature("program", "bootstrap"),
                                 ELF::SectionFlagsAttr::SHF_ALLOC);
}

bool vpux::VPUASM::BootstrapOp::hasMemoryFootprint() {
    return true;
}
