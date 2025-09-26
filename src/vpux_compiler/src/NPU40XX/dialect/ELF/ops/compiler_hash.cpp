//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include <cstdint>
#include <cstring>
#include <vpux_elf/writer.hpp>
#include <vpux_headers/compiler_hash.hpp>
#include "vpux/compiler/NPU40XX/dialect/ELF/ops.hpp"
#include "vpux/compiler/compiler_version.hpp"
#include "vpux/compiler/utils/ELF/utils.hpp"

using CompilerHashInfo = elf::CompilerHashInfo;

std::vector<uint8_t> vpux::ELF::CompilerHashOp::getSerializedCompilerHash() {
    auto compilerHashString = getCompilerHash();
    CompilerHashInfo compilerHashStruct(compilerHashString.str());
    return elf::CompilerHashSerialization::serialize(compilerHashStruct);
}

void vpux::ELF::CompilerHashOp::serialize(elf::writer::BinaryDataSection<uint8_t>& binDataSection) {
    auto serializedCompilerHash = getSerializedCompilerHash();
    binDataSection.appendData(serializedCompilerHash.data(), serializedCompilerHash.size());
}

size_t vpux::ELF::CompilerHashOp::getBinarySize(config::ArchKind) {
    return getSerializedCompilerHash().size();
}

size_t vpux::ELF::CompilerHashOp::getAlignmentRequirements(config::ArchKind) {
    return alignof(CompilerHashInfo);
}

std::optional<ELF::SectionSignature> vpux::ELF::CompilerHashOp::getSectionSignature() {
    return ELF::SectionSignature(vpux::ELF::generateSignature("info", "compiler", "hash"),
                                 ELF::SectionFlagsAttr::SHF_NONE, ELF::SectionTypeAttr::VPU_SHT_COMPILER_HASH);
}

bool vpux::ELF::CompilerHashOp::hasMemoryFootprint() {
    return true;
}
