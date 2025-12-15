//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include <mlir/IR/BuiltinTypes.h>
#include <cstring>
#include <vpux_elf/writer.hpp>
#include <vpux_headers/platform.hpp>
#include "vpux/compiler/dialect/ELFNPU37XX/utils.hpp"
#include "vpux/compiler/dialect/VPUMI37XX/ops.hpp"
#include "vpux/compiler/dialect/VPURegMapped/types.hpp"

void vpux::VPUMI37XX::PlatformInfoOp::serialize(elf::writer::BinaryDataSection<uint8_t>& binDataSection) {
    elf::platform::PlatformInfo platformInfo;

    platformInfo.mArchKind = ELFNPU37XX::mapVpuArchKindToElfArchKind(getArchKind());

    auto serializedPlatformInfo = elf::platform::PlatformInfoSerialization::serialize(platformInfo);
    binDataSection.appendData(&serializedPlatformInfo[0], serializedPlatformInfo.size());
}

size_t vpux::VPUMI37XX::PlatformInfoOp::getBinarySize() {
    // calculate size based on serialized form, instead of just sizeof(PlatformInfoOp)
    // serialization uses metadata that also gets stored in the blob and must be accounted for
    // also for non-POD types (e.g. have vector as member) account for all data to be serialized
    // (data owned by vector, instead of just pointer)
    elf::platform::PlatformInfo platformInfo;
    platformInfo.mArchKind = ELFNPU37XX::mapVpuArchKindToElfArchKind(getArchKind());
    return elf::platform::PlatformInfoSerialization::serialize(platformInfo).size();
}

size_t vpux::VPUMI37XX::PlatformInfoOp::getAlignmentRequirements() {
    return alignof(elf::platform::PlatformInfo);
}

vpux::VPURT::BufferSection vpux::VPUMI37XX::PlatformInfoOp::getMemorySpace() {
    return vpux::VPURT::BufferSection::DDR;
}

vpux::ELFNPU37XX::SectionFlagsAttr vpux::VPUMI37XX::PlatformInfoOp::getAccessingProcs() {
    return ELFNPU37XX::SectionFlagsAttr::SHF_NONE;
}

vpux::ELFNPU37XX::SectionFlagsAttr vpux::VPUMI37XX::PlatformInfoOp::getUserProcs() {
    return ELFNPU37XX::SectionFlagsAttr::SHF_NONE;
}

void vpux::VPUMI37XX::PlatformInfoOp::build(::mlir::OpBuilder& odsBuilder, ::mlir::OperationState& odsState) {
    build(odsBuilder, odsState, config::ArchKind::NPU37XX);
}
