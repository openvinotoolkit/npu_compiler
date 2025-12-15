//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include <llvm/Support/FileSystem.h>
#include <mlir/IR/BuiltinTypes.h>
#include <vector>
#include <vpux_elf/accessor.hpp>
#include <vpux_elf/reader.hpp>
#include "vpux/compiler/act_kernels/shave_binary_resources.h"
#include "vpux/compiler/dialect/ELFNPU37XX/utils.hpp"
#include "vpux/compiler/dialect/VPUMI37XX/ops.hpp"
#include "vpux/compiler/dialect/VPURegMapped/types.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"

using namespace vpux;

//
// ActShaveRtOp
//

void vpux::VPUMI37XX::ActShaveRtOp::serialize(elf::writer::BinaryDataSection<uint8_t>& binDataSection) {
    auto kernel = getKernelPath();

    const auto& kernelInfo = ShaveBinaryResourcesCache::getCache(getContext());
    const SmallString arch = ELFNPU37XX::getSwKernelArchString(config::getArch(this->getOperation()));

    const auto elfBlob = kernelInfo.getElf(kernel, arch);

    auto secDataSizePair = vpux::ELFNPU37XX::getDataAndSizeOfElfSection(elfBlob, {".text"});

    binDataSection.appendData(secDataSizePair.first, secDataSizePair.second);
}

size_t vpux::VPUMI37XX::ActShaveRtOp::getBinarySize() {
    auto kernel = getKernelPath();

    const auto& kernelInfo = ShaveBinaryResourcesCache::getCache(getContext());
    const SmallString arch = ELFNPU37XX::getSwKernelArchString(config::getArch(this->getOperation()));

    const auto elfBlob = kernelInfo.getElf(kernel, arch);

    auto secDataSizePair = vpux::ELFNPU37XX::getDataAndSizeOfElfSection(elfBlob, {".text"});

    return secDataSizePair.second;
}

uint32_t vpux::VPUMI37XX::ActShaveRtOp::getKernelEntry() {
    auto kernel = getKernelPath();

    const auto& kernelInfo = ShaveBinaryResourcesCache::getCache(getContext());
    const SmallString arch = ELFNPU37XX::getSwKernelArchString(config::getArch(this->getOperation()));

    const auto elfBlob = kernelInfo.getElf(kernel, arch);

    auto accessor = elf::DDRAccessManager<elf::DDRAlwaysEmplace>(elfBlob.data(), elfBlob.size());
    auto elf_reader = elf::Reader<elf::ELF_Bitness::Elf32>(&accessor);

    auto actKernelHeader = elf_reader.getHeader();
    return actKernelHeader->e_entry;
}

uint32_t vpux::VPUMI37XX::ActShaveRtOp::getVersion() {
    auto kernel = getKernelPath();

    const auto& kernelInfo = ShaveBinaryResourcesCache::getCache(getContext());
    const SmallString arch = ELFNPU37XX::getSwKernelArchString(config::getArch(this->getOperation()));

    const auto elfBlob = kernelInfo.getElf(kernel, arch);

    auto secDataSizePair = vpux::ELFNPU37XX::getDataAndSizeOfElfSection(elfBlob, {".versiondata"});

    auto nnActEntryRtVersion = reinterpret_cast<const uint32_t*>(secDataSizePair.first);

    return *nnActEntryRtVersion;
}

// The management kernel code must be 1kB aligned as an ActShave requirement
size_t vpux::VPUMI37XX::ActShaveRtOp::getAlignmentRequirements() {
    return ELFNPU37XX::VPUX_SHAVE_ALIGNMENT;
}

vpux::VPURT::BufferSection vpux::VPUMI37XX::ActShaveRtOp::getMemorySpace() {
    return vpux::VPURT::BufferSection::SW_KernelText;
}

vpux::ELFNPU37XX::SectionFlagsAttr vpux::VPUMI37XX::ActShaveRtOp::getAccessingProcs() {
    return (ELFNPU37XX::SectionFlagsAttr::VPU_SHF_PROC_SHAVE);
}

vpux::ELFNPU37XX::SectionFlagsAttr vpux::VPUMI37XX::ActShaveRtOp::getUserProcs() {
    return (ELFNPU37XX::SectionFlagsAttr::SHF_NONE);
}
