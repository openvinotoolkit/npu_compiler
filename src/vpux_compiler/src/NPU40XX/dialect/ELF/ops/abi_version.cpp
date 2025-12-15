//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include <cstdint>
#include <cstring>
#include <vpux_elf/writer.hpp>
#include "vpux/compiler/NPU40XX/dialect/ELF/ops.hpp"
#include "vpux/compiler/dialect/ELF/utils/utils.hpp"

using LoaderAbiVersionNote = elf::elf_note::VersionNote;

void vpux::ELF::ABIVersionOp::serialize(elf::writer::BinaryDataSection<uint8_t>& binDataSection) {
    LoaderAbiVersionNote abiVersionStruct;
    constexpr uint8_t nameSize = 4;
    constexpr uint8_t descSize = 16;
    abiVersionStruct.n_namesz = nameSize;
    abiVersionStruct.n_descz = descSize;
    abiVersionStruct.n_type = elf::elf_note::NT_GNU_ABI_TAG;

    const uint8_t name[4] = {0x47, 0x4e, 0x55, 0};  // 'G' 'N' 'U' '\0' as required by standard
    static_assert(sizeof(name) == nameSize);
    std::memcpy(abiVersionStruct.n_name, name, nameSize);

    const uint32_t desc[4] = {0, getMajor(), getMinor(), getPatch()};
    static_assert(sizeof(desc) == descSize);
    std::memcpy(abiVersionStruct.n_desc, desc, descSize);

    auto ptrCharTmp = reinterpret_cast<uint8_t*>(&abiVersionStruct);
    binDataSection.appendData(ptrCharTmp, getBinarySize(config::ArchKind::UNKNOWN));
}

size_t vpux::ELF::ABIVersionOp::getBinarySize(config::ArchKind) {
    return sizeof(LoaderAbiVersionNote);
}

size_t vpux::ELF::ABIVersionOp::getAlignmentRequirements(config::ArchKind) {
    return alignof(LoaderAbiVersionNote);
}

std::optional<ELF::SectionSignature> vpux::ELF::ABIVersionOp::getSectionSignature() {
    return ELF::SectionSignature(vpux::ELF::generateSignature("note", "LoaderABIVersion"),
                                 ELF::SectionFlagsAttr::SHF_NONE, ELF::SectionTypeAttr::SHT_NOTE);
}

bool vpux::ELF::ABIVersionOp::hasMemoryFootprint() {
    return true;
}

void vpux::ELF::ABIVersionOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, uint32_t verMajor,
                                    uint32_t verMinor, uint32_t verPatch) {
    build(builder, state, "LoaderABIVersion", verMajor, verMinor, verPatch);
}
