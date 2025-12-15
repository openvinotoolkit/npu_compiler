//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include <mlir/IR/BuiltinTypes.h>
#include <cstring>
#include <vpux_elf/types/vpu_extensions.hpp>
#include <vpux_elf/writer.hpp>
#include "vpux/compiler/dialect/VPUMI37XX/ops.hpp"
#include "vpux/compiler/dialect/VPURegMapped/types.hpp"

#include <npu_37xx_nnrt.hpp>

using MIVersionNote = elf::elf_note::VersionNote;

void vpux::VPUMI37XX::MappedInferenceVersionOp::serialize(elf::writer::BinaryDataSection<uint8_t>& binDataSection) {
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

    constexpr uint32_t desc[descSize] = {0, VPU_NNRT_37XX_API_VER_MAJOR, VPU_NNRT_37XX_API_VER_MINOR,
                                         VPU_NNRT_37XX_API_VER_PATCH};
    static_assert(sizeof(desc) == 64);
    std::memcpy(MIVersionStruct.n_desc, desc, descSize);

    auto ptrCharTmp = reinterpret_cast<uint8_t*>(&MIVersionStruct);
    binDataSection.appendData(ptrCharTmp, getBinarySize());
}

size_t vpux::VPUMI37XX::MappedInferenceVersionOp::getBinarySize() {
    return sizeof(MIVersionNote);
}

size_t vpux::VPUMI37XX::MappedInferenceVersionOp::getAlignmentRequirements() {
    return alignof(MIVersionNote);
}

vpux::VPURT::BufferSection vpux::VPUMI37XX::MappedInferenceVersionOp::getMemorySpace() {
    return vpux::VPURT::BufferSection::DDR;
}

vpux::ELFNPU37XX::SectionFlagsAttr vpux::VPUMI37XX::MappedInferenceVersionOp::getAccessingProcs() {
    return ELFNPU37XX::SectionFlagsAttr::SHF_NONE;
}

vpux::ELFNPU37XX::SectionFlagsAttr vpux::VPUMI37XX::MappedInferenceVersionOp::getUserProcs() {
    return ELFNPU37XX::SectionFlagsAttr::SHF_NONE;
}

void vpux::VPUMI37XX::MappedInferenceVersionOp::build(::mlir::OpBuilder& odsBuilder, ::mlir::OperationState& odsState) {
    build(odsBuilder, odsState, VPU_NNRT_37XX_API_VER_MAJOR, VPU_NNRT_37XX_API_VER_MINOR, VPU_NNRT_37XX_API_VER_PATCH);
}
