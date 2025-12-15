//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include <cstring>
#include <vpux_elf/writer.hpp>
#include "vpux/compiler/dialect/ELFNPU37XX/ops.hpp"

namespace {
// Loader ABI version (HPI3720 specific) - To be updated on Loader ABI changes
static constexpr uint32_t VERSION_MAJOR = 1;
static constexpr uint32_t VERSION_MINOR = 2;
static constexpr uint32_t VERSION_PATCH = 0;
}  // namespace

using LoaderAbiVersionNote = elf::elf_note::VersionNote;

void vpux::ELFNPU37XX::ABIVersionOp::serialize(elf::writer::BinaryDataSection<uint8_t>& binDataSection) {
    LoaderAbiVersionNote abiVersionStruct;
    constexpr uint8_t nameSize = 4;
    constexpr uint8_t descSize = 16;
    abiVersionStruct.n_namesz = nameSize;
    abiVersionStruct.n_descz = descSize;
    abiVersionStruct.n_type = elf::elf_note::NT_GNU_ABI_TAG;

    const uint8_t name[4] = {0x47, 0x4e, 0x55, 0};  // 'G' 'N' 'U' '\0' as required by standard
    static_assert(sizeof(name) == nameSize);
    std::memcpy(abiVersionStruct.n_name, name, nameSize);

    const uint32_t desc[4] = {0, VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH};
    static_assert(sizeof(desc) == descSize);
    std::memcpy(abiVersionStruct.n_desc, desc, descSize);

    auto ptrCharTmp = reinterpret_cast<uint8_t*>(&abiVersionStruct);
    binDataSection.appendData(ptrCharTmp, getBinarySize());
}

size_t vpux::ELFNPU37XX::ABIVersionOp::getBinarySize() {
    return sizeof(LoaderAbiVersionNote);
}

size_t vpux::ELFNPU37XX::ABIVersionOp::getAlignmentRequirements() {
    return alignof(LoaderAbiVersionNote);
}

vpux::VPURT::BufferSection vpux::ELFNPU37XX::ABIVersionOp::getMemorySpace() {
    return vpux::VPURT::BufferSection::DDR;
}

vpux::ELFNPU37XX::SectionFlagsAttr vpux::ELFNPU37XX::ABIVersionOp::getAccessingProcs() {
    return ELFNPU37XX::SectionFlagsAttr::SHF_NONE;
}

vpux::ELFNPU37XX::SectionFlagsAttr vpux::ELFNPU37XX::ABIVersionOp::getUserProcs() {
    return ELFNPU37XX::SectionFlagsAttr::SHF_NONE;
}

void vpux::ELFNPU37XX::ABIVersionOp::build(::mlir::OpBuilder& odsBuilder, ::mlir::OperationState& odsState) {
    build(odsBuilder, odsState, VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH);
}
