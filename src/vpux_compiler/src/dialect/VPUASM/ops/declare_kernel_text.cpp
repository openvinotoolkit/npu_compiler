//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include <vpux_elf/reader.hpp>

#include "vpux/compiler/dialect/ELF/utils/utils.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"

using namespace vpux;

//
// DeclareKernelTextOp
//

void vpux::VPUASM::DeclareKernelTextOp::serialize(elf::writer::BinaryDataSection<uint8_t>& binDataSection) {
    const auto text = vpux::ELF::getKernelELF(getOperation(), getKernelPath(), {".text"});
    binDataSection.appendData(text.data(), text.size());
}

size_t vpux::VPUASM::DeclareKernelTextOp::getBinarySize(config::ArchKind) {
    return vpux::ELF::getKernelELF(getOperation(), getKernelPath(), {".text"}).size();
}

// The .text sections for the sw layers must be 1kB aligned as an ActShave requirement
size_t vpux::VPUASM::DeclareKernelTextOp::getAlignmentRequirements(config::ArchKind) {
    return ELF::VPUX_SHAVE_ALIGNMENT;
}

vpux::ELF::SectionFlagsAttr vpux::VPUASM::DeclareKernelTextOp::getPredefinedMemoryAccessors() {
    return ELF::SectionFlagsAttr::VPU_SHF_PROC_SHAVE;
}

std::optional<ELF::SectionSignature> vpux::VPUASM::DeclareKernelTextOp::getSectionSignature() {
    return ELF::SectionSignature(vpux::ELF::generateSignature("shave", "text"), ELF::SectionFlagsAttr::SHF_ALLOC);
}

bool vpux::VPUASM::DeclareKernelTextOp::hasMemoryFootprint() {
    return true;
}
