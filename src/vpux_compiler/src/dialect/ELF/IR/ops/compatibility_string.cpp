//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/ELF/IR/ops.hpp"
#include "vpux/compiler/dialect/ELF/utils/utils.hpp"

using namespace vpux;

void vpux::ELF::CompatibilityStringOp::serialize(elf::writer::BinaryDataSection<uint8_t>& binDataSection) {
    const auto& compatibilityString = getCompatibilityString();
    VPUX_THROW_WHEN(compatibilityString.empty(), "Trying to serialize empty compatibility string");
    binDataSection.appendData(reinterpret_cast<const uint8_t*>(compatibilityString.data()), compatibilityString.size());

    constexpr uint8_t nullTerminator = 0;
    binDataSection.appendData(&nullTerminator, sizeof(nullTerminator));
}

size_t vpux::ELF::CompatibilityStringOp::getBinarySize(config::ArchKind) {
    return getCompatibilityString().size() + 1;  // +1 for null terminator byte
}

std::optional<ELF::SectionSignature> vpux::ELF::CompatibilityStringOp::getSectionSignature() {
    return ELF::SectionSignature("compatibility_string", ELF::SectionFlagsAttr::SHF_NONE,
                                 ELF::SectionTypeAttr::VPU_SHT_COMPAT_STR);
}

bool vpux::ELF::CompatibilityStringOp::hasMemoryFootprint() {
    return true;
}
