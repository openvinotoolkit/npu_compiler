//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include <llvm/Support/FileSystem.h>
#include <mlir/IR/BuiltinTypes.h>
#include <fstream>
#include <vector>
#include <vpux_elf/reader.hpp>
#include "vpux/compiler/act_kernels/shave_binary_resources.h"
#include "vpux/compiler/dialect/ELFNPU37XX/utils.hpp"
#include "vpux/compiler/dialect/VPUMI37XX/ops.hpp"
#include "vpux/compiler/dialect/VPURegMapped/types.hpp"
#include "vpux/utils/core/scope_exit.hpp"

using namespace vpux;

//
// DeclareKernelTextOp
//

void vpux::VPUMI37XX::DeclareKernelTextOp::serialize(elf::writer::BinaryDataSection<uint8_t>& binDataSection) {
    auto kernel = getKernelPath();

    const auto& kernelInfo = ShaveBinaryResourcesCache::getCache(getContext());
    const auto elfBlob = kernelInfo.getElf(kernel);

    auto secDataSizePair = vpux::ELFNPU37XX::getDataAndSizeOfElfSection(elfBlob, {".text"});

    binDataSection.appendData(secDataSizePair.first, secDataSizePair.second);
}

size_t vpux::VPUMI37XX::DeclareKernelTextOp::getBinarySize() {
    auto kernel = getKernelPath();

    const auto& kernelInfo = ShaveBinaryResourcesCache::getCache(getContext());
    const auto elfBlob = kernelInfo.getElf(kernel);

    auto secDataSizePair = vpux::ELFNPU37XX::getDataAndSizeOfElfSection(elfBlob, {".text"});

    return secDataSizePair.second;
}

// The .text sections for the sw layers must be 1kB aligned as an ActShave requirement
size_t vpux::VPUMI37XX::DeclareKernelTextOp::getAlignmentRequirements() {
    return ELFNPU37XX::VPUX_SHAVE_ALIGNMENT;
}

vpux::VPURT::BufferSection vpux::VPUMI37XX::DeclareKernelTextOp::getMemorySpace() {
    return vpux::VPURT::BufferSection::SW_KernelText;
}

vpux::ELFNPU37XX::SectionFlagsAttr vpux::VPUMI37XX::DeclareKernelTextOp::getAccessingProcs() {
    return (ELFNPU37XX::SectionFlagsAttr::VPU_SHF_PROC_SHAVE);
}

vpux::ELFNPU37XX::SectionFlagsAttr vpux::VPUMI37XX::DeclareKernelTextOp::getUserProcs() {
    return (ELFNPU37XX::SectionFlagsAttr::SHF_NONE);
}
