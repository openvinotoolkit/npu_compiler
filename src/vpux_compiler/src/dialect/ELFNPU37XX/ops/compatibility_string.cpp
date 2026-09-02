//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpux_elf/writer.hpp>

#include "vpux/compiler/dialect/ELFNPU37XX/ops.hpp"
#include "vpux/compiler/dialect/ELFNPU37XX/utils.hpp"

void vpux::ELFNPU37XX::CompatibilityStringOp::serialize(elf::writer::BinaryDataSection<uint8_t>& binDataSection) {
    const auto& compatibilityString = getCompatibilityString();
    VPUX_THROW_WHEN(compatibilityString.empty(), "Trying to serialize empty compatibility string");
    binDataSection.appendData(reinterpret_cast<const uint8_t*>(compatibilityString.data()), compatibilityString.size());

    constexpr uint8_t nullTerminator = 0;
    binDataSection.appendData(&nullTerminator, sizeof(nullTerminator));
}

size_t vpux::ELFNPU37XX::CompatibilityStringOp::getBinarySize() {
    return getCompatibilityString().size() + 1;  // +1 for null terminator byte
}

vpux::VPURT::BufferSection vpux::ELFNPU37XX::CompatibilityStringOp::getMemorySpace() {
    return vpux::VPURT::BufferSection::DDR;
}

vpux::ELFNPU37XX::SectionFlagsAttr vpux::ELFNPU37XX::CompatibilityStringOp::getAccessingProcs() {
    return ELFNPU37XX::SectionFlagsAttr::SHF_NONE;
}

vpux::ELFNPU37XX::SectionFlagsAttr vpux::ELFNPU37XX::CompatibilityStringOp::getUserProcs() {
    return ELFNPU37XX::SectionFlagsAttr::SHF_NONE;
}

size_t vpux::ELFNPU37XX::CompatibilityStringOp::getAlignmentRequirements() {
    return ELFNPU37XX::VPUX_NO_ALIGNMENT;
}
