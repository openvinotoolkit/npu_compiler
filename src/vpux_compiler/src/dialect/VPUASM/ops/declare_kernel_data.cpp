//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include <vpux_elf/reader.hpp>

#include "vpux/compiler/act_kernels/shave_binary_resources.h"
#include "vpux/compiler/dialect/ELF/utils/utils.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"

using namespace vpux;

//
// DeclareKernelDataOp
//

void vpux::VPUASM::DeclareKernelDataOp::serialize(elf::writer::BinaryDataSection<uint8_t>& binDataSection) {
    const auto data = vpux::ELF::getKernelELF(getOperation(), getKernelPath(), {".data", ".arg.data"});
    binDataSection.appendData(data.data(), data.size());
}

size_t vpux::VPUASM::DeclareKernelDataOp::getBinarySize(config::ArchKind) {
    return vpux::ELF::getKernelELF(getOperation(), getKernelPath(), {".data", ".arg.data"}).size();
}

// The .data sections for the sw layers must be 1kB aligned as an ActShave requirement
size_t vpux::VPUASM::DeclareKernelDataOp::getAlignmentRequirements(config::ArchKind) {
    return ELF::VPUX_SHAVE_ALIGNMENT;
}

vpux::ELF::SectionFlagsAttr vpux::VPUASM::DeclareKernelDataOp::getPredefinedMemoryAccessors() {
    return ELF::SectionFlagsAttr::VPU_SHF_PROC_SHAVE;
}

std::optional<ELF::SectionSignature> vpux::VPUASM::DeclareKernelDataOp::getSectionSignature() {
    return ELF::SectionSignature(vpux::ELF::generateSignature("shave", "data"),
                                 ELF::SectionFlagsAttr::SHF_WRITE | ELF::SectionFlagsAttr::SHF_ALLOC);
}

bool vpux::VPUASM::DeclareKernelDataOp::hasMemoryFootprint() {
    return true;
}
