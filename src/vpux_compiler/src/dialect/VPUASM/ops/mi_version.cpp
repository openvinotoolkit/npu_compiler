//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include <cstring>
#include <vpux_elf/writer.hpp>
#include "vpux/compiler/dialect/ELF/utils/utils.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"

using namespace vpux;
using MIVersionNote = elf::elf_note::VersionNote;

std::optional<ELF::SectionSignature> vpux::VPUASM::MappedInferenceVersionOp::getSectionSignature() {
    return ELF::SectionSignature(vpux::ELF::generateSignature("note", "MappedInferenceVersion"),
                                 ELF::SectionFlagsAttr::SHF_NONE, ELF::SectionTypeAttr::SHT_NOTE);
}

bool vpux::VPUASM::MappedInferenceVersionOp::hasMemoryFootprint() {
    return true;
}
