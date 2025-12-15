//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux_headers/compiler_hash.hpp"
#include "vpux/compiler/compiler_version.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"

std::vector<uint8_t> vpux::VPUASM::CompilerHashOp::getSerializedCompilerHash() {
    auto compilerHashString = getCompilerHash();
    elf::CompilerHashInfo compilerHashStruct(compilerHashString.str());
    return elf::CompilerHashSerialization::serialize(compilerHashStruct);
}

void vpux::VPUASM::CompilerHashOp::serialize(elf::writer::BinaryDataSection<uint8_t>& binDataSection) {
    auto serializedCompilerHash = getSerializedCompilerHash();
    binDataSection.appendData(serializedCompilerHash.data(), serializedCompilerHash.size());
}

size_t vpux::VPUASM::CompilerHashOp::getBinarySize(config::ArchKind) {
    return getSerializedCompilerHash().size();
}

size_t vpux::VPUASM::CompilerHashOp::getAlignmentRequirements(config::ArchKind) {
    return alignof(elf::CompilerHashInfo);
}

std::optional<ELF::SectionSignature> vpux::VPUASM::CompilerHashOp::getSectionSignature() {
    return ELF::SectionSignature(vpux::ELF::generateSignature("info", "compiler", "hash"),
                                 ELF::SectionFlagsAttr::SHF_NONE, ELF::SectionTypeAttr::VPU_SHT_COMPILER_HASH);
}

bool vpux::VPUASM::CompilerHashOp::hasMemoryFootprint() {
    return true;
}
