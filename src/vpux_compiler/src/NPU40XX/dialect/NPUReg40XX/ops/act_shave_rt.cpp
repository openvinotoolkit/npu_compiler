//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <mlir/IR/BuiltinTypes.h>
#include <vpux_elf/accessor.hpp>
#include <vpux_elf/reader.hpp>
#include "vpux/compiler/NPU40XX/dialect/NPUReg40XX/ops.hpp"
#include "vpux/compiler/utils/ELF/utils.hpp"

using namespace vpux;

//
// ActShaveRtOp
//

void vpux::NPUReg40XX::ActShaveRtOp::serialize(elf::writer::BinaryDataSection<uint8_t>& binDataSection) {
    const auto kernelText = vpux::ELF::getKernelELF(getOperation(), getKernelPath(), {".text"});

    binDataSection.appendData(kernelText.data(), kernelText.size());
}

size_t vpux::NPUReg40XX::ActShaveRtOp::getBinarySize(VPU::ArchKind) {
    const auto kernelText = vpux::ELF::getKernelELF(getOperation(), getKernelPath(), {".text"});

    return kernelText.size();
}

// The management kernel code must be 1kB aligned as an ActShave requirement
size_t vpux::NPUReg40XX::ActShaveRtOp::getAlignmentRequirements(VPU::ArchKind) {
    return ELF::VPUX_SHAVE_ALIGNMENT;
}

uint32_t vpux::NPUReg40XX::ActShaveRtOp::getKernelEntry() {
    const auto elfBlob = ELF::getKernelELF(getOperation(), getKernelPath());

    auto accessor = elf::DDRAccessManager<elf::DDRAlwaysEmplace>(elfBlob.data(), elfBlob.size());
    auto elf_reader = elf::Reader<elf::ELF_Bitness::Elf32>(&accessor);

    auto actKernelHeader = elf_reader.getHeader();
    return actKernelHeader->e_entry;
}

uint32_t vpux::NPUReg40XX::ActShaveRtOp::getVersion() {
    const auto elfBlob = ELF::getKernelELF(getOperation(), getKernelPath());

    auto secDataSizePair = ELF::getDataAndSizeOfElfSection(elfBlob, {".versiondata"});

    auto nnActEntryRtVersion = reinterpret_cast<const uint32_t*>(secDataSizePair.data());

    return *nnActEntryRtVersion;
}
