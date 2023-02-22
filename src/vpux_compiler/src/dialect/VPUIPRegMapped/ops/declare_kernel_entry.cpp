//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include <llvm/Support/FileSystem.h>
#include <mlir/IR/BuiltinTypes.h>
#include <fstream>
#include <vector>
#include <vpux_elf/accessor.hpp>
#include <vpux_elf/reader.hpp>
#include "vpux/compiler/act_kernels/shave_binary_resources.h"
#include "vpux/compiler/dialect/ELF/utils.hpp"
#include "vpux/compiler/dialect/VPUIPRegMapped/ops.hpp"
#include "vpux/utils/core/scope_exit.hpp"

using namespace vpux;

//
// DeclareKernelEntryOp
//

uint32_t vpux::VPUIPRegMapped::DeclareKernelEntryOp::getKernelEntry() {
    auto kernel = kernel_path();

    const auto& kernelInfo = ShaveBinaryResources::getInstance();
    const SmallString arch = ELF::getSwKernelArchString(VPU::getArch(this->getOperation()));

    const auto elfBlob = kernelInfo.getElf(kernel, arch).vec();

    auto accessor = elf::ElfDDRAccessManager(elfBlob.data(), elfBlob.size());
    auto elf_reader = elf::Reader<elf::ELF_Bitness::Elf32>(&accessor);

    auto actKernelHeader = elf_reader.getHeader();
    return actKernelHeader->e_entry;
}
