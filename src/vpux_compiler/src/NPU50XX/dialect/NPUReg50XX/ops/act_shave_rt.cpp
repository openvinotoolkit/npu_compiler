//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include <mlir/IR/BuiltinTypes.h>
#include <vpux_elf/accessor.hpp>
#include <vpux_elf/reader.hpp>
#include "vpux/compiler/NPU50XX/dialect/NPUReg50XX/ops.hpp"
#include "vpux/compiler/dialect/ELF/utils/utils.hpp"

using namespace vpux;

//
// ActShaveRtOp
//

void vpux::NPUReg50XX::ActShaveRtOp::serialize(elf::writer::BinaryDataSection<uint8_t>& binDataSection) {
    const auto kernelText = vpux::ELF::getKernelELF(getOperation(), getKernelPath(), {".text"});

    binDataSection.appendData(kernelText.data(), kernelText.size());
}

size_t vpux::NPUReg50XX::ActShaveRtOp::getBinarySize(config::ArchKind) {
    const auto kernelText = vpux::ELF::getKernelELF(getOperation(), getKernelPath(), {".text"});

    return kernelText.size();
}

// The management kernel code must be 1kB aligned as an ActShave requirement
size_t vpux::NPUReg50XX::ActShaveRtOp::getAlignmentRequirements(config::ArchKind) {
    return ELF::VPUX_SHAVE_ALIGNMENT;
}

uint32_t vpux::NPUReg50XX::ActShaveRtOp::getKernelEntry() {
    const auto elfBlob = ELF::getKernelELF(getOperation(), getKernelPath());

    auto accessor = elf::DDRAccessManager<elf::DDRAlwaysEmplace>(elfBlob.data(), elfBlob.size());
    auto elf_reader = elf::Reader<elf::ELF_Bitness::Elf32>(&accessor);

    auto actKernelHeader = elf_reader.getHeader();
    return actKernelHeader->e_entry;
}

uint32_t vpux::NPUReg50XX::ActShaveRtOp::getVersion() {
    const auto elfBlob = ELF::getKernelELF(getOperation(), getKernelPath());

    auto secDataSizePair = ELF::getDataAndSizeOfElfSection(elfBlob, {".versiondata"});

    auto nnActEntryRtVersion = reinterpret_cast<const uint32_t*>(secDataSizePair.data());

    return *nnActEntryRtVersion;
}
