//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include <cstring>
#include <vpux_elf/types/vpu_extensions.hpp>
#include <vpux_elf/writer.hpp>
#include "vpux/compiler/NPU40XX/dialect/NPUReg40XX/ops.hpp"

#include <npu_40xx_nnrt.hpp>

using namespace vpux;
using MIVersionNote = elf::elf_note::VersionNote;

void vpux::NPUReg40XX::MappedInferenceVersionOp::serialize(elf::writer::BinaryDataSection<uint8_t>& binDataSection) {
    MIVersionNote MIVersionStruct;
    constexpr uint8_t nameSize = 4;
    constexpr uint8_t descSize = 16;
    MIVersionStruct.n_namesz = nameSize;
    MIVersionStruct.n_descz = descSize;
    MIVersionStruct.n_type = elf::elf_note::NT_NPU_MPI_VERSION;

    // As we don't have the readelf constraints of standard NOTE section types, we can here choose custom names for the
    // notes
    constexpr uint8_t name[nameSize] = {0x4d, 0x49, 0x56, 0};  // 'M'(apped) 'I'(nference) 'V'(ersion) '\0'
    static_assert(sizeof(name) == 4);
    std::memcpy(MIVersionStruct.n_name, name, nameSize);
    uint32_t desc[descSize] = {0, getMajor(), getMinor(), getPatch()};
    static_assert(sizeof(desc) == 64);
    std::memcpy(MIVersionStruct.n_desc, desc, descSize);

    auto ptrCharTmp = reinterpret_cast<uint8_t*>(&MIVersionStruct);
    binDataSection.appendData(ptrCharTmp, getBinarySize(config::ArchKind::NPU40XX));
}

size_t vpux::NPUReg40XX::MappedInferenceVersionOp::getBinarySize(config::ArchKind) {
    return sizeof(MIVersionNote);
}

size_t vpux::NPUReg40XX::MappedInferenceVersionOp::getAlignmentRequirements(config::ArchKind) {
    return alignof(MIVersionNote);
}

void NPUReg40XX::MappedInferenceVersionOp::setVersion(const elf::Version& version) {
    setMajor(version.getMajor());
    setMinor(version.getMinor());
    setPatch(version.getPatch());
}
