//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include <cstring>
#include <vpux_elf/writer.hpp>
#include "vpux/compiler/dialect/ELF/utils/utils.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux_headers/platform.hpp"

using namespace vpux;

void vpux::VPUASM::PlatformInfoOp::serialize(elf::writer::BinaryDataSection<uint8_t>& binDataSection) {
    elf::platform::PlatformInfo platformInfo;

    platformInfo.mArchKind = ELF::getElfArchKind(getOperation());

    auto serializedPlatformInfo = elf::platform::PlatformInfoSerialization::serialize(platformInfo);
    binDataSection.appendData(&serializedPlatformInfo[0], serializedPlatformInfo.size());
}

size_t vpux::VPUASM::PlatformInfoOp::getBinarySize(config::ArchKind) {
    // calculate size based on serialized form, instead of just sizeof(PlatformInfo)
    // serialization uses metadata that also gets stored in the blob and must be accounted for
    // also for non-POD types (e.g. have vector as member) account for all data to be serialized
    // (data owned by vector, instead of just pointer)
    elf::platform::PlatformInfo platformInfo;
    platformInfo.mArchKind = ELF::getElfArchKind(getOperation());
    return elf::platform::PlatformInfoSerialization::serialize(platformInfo).size();
}

size_t vpux::VPUASM::PlatformInfoOp::getAlignmentRequirements(config::ArchKind) {
    return alignof(elf::platform::PlatformInfo);
}

std::optional<ELF::SectionSignature> vpux::VPUASM::PlatformInfoOp::getSectionSignature() {
    return ELF::SectionSignature(vpux::ELF::generateSignature("meta", "PlatformInfo"), ELF::SectionFlagsAttr::SHF_NONE,
                                 ELF::SectionTypeAttr::VPU_SHT_PLATFORM_INFO);
}

bool vpux::VPUASM::PlatformInfoOp::hasMemoryFootprint() {
    return true;
}

void vpux::VPUASM::PlatformInfoOp::build(mlir::OpBuilder& builder, mlir::OperationState& state) {
    build(builder, state, "PlatformInfo");
}
