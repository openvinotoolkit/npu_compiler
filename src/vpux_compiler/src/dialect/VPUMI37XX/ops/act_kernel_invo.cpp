//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include <mlir/IR/BuiltinTypes.h>
#include "vpux/compiler/dialect/ELFNPU37XX/utils.hpp"
#include "vpux/compiler/dialect/VPUMI37XX/ops.hpp"
#include "vpux/compiler/dialect/VPURegMapped/types.hpp"

#include "vpux/compiler/dialect/VPUMI37XX/utils.hpp"

#include <npu_37xx_nnrt.hpp>

using namespace vpux;
using namespace npu37xx;

//
// ActKernelInvocationOp
//

void vpux::VPUMI37XX::ActKernelInvocationOp::serialize(elf::writer::BinaryDataSection<uint8_t>& binDataSection) {
    nn_public::VpuActKernelInvocation actKernelInvocation;
    memset(reinterpret_cast<void*>(&actKernelInvocation), 0, sizeof(actKernelInvocation));

    auto rangeIndex = mlir::cast<vpux::VPURegMapped::IndexType>(getRangeIndex().getType());
    auto invoIndex = mlir::cast<vpux::VPURegMapped::IndexType>(getType());

    actKernelInvocation.range = rangeIndex.getValue();
    actKernelInvocation.kernel_range_index = rangeIndex.getValue();
    actKernelInvocation.invo_tile = checked_cast<uint32_t>(getTile());
    actKernelInvocation.barriers_sched.start_after_ = checked_cast<uint32_t>(getStartAfter());
    actKernelInvocation.barriers_sched.clean_after_ = checked_cast<uint32_t>(getCleanAfter());
    actKernelInvocation.invo_index = invoIndex.getValue();
    actKernelInvocation.barriers.wait_mask_ = VPUMI37XX::computeMask(getWaitBarriers());
    actKernelInvocation.barriers.post_mask_ = VPUMI37XX::computeMask(getUpdateBarriers());

    actKernelInvocation.barriers.group_ = 0;
    actKernelInvocation.barriers.mask_ = 0;

    for (uint64_t mask = actKernelInvocation.barriers.wait_mask_, group = 1; mask > 0; mask >>= 8, ++group) {
        if (mask & 0xff) {
            if (actKernelInvocation.barriers.group_ == 0) {
                actKernelInvocation.barriers.group_ = static_cast<unsigned char>(group);
                actKernelInvocation.barriers.mask_ = mask & 0xff;
            } else {
                actKernelInvocation.barriers.group_ = 0;
                actKernelInvocation.barriers.mask_ = 0;
                break;
            }
        }
    }

    uint8_t* ptrCharTmp = reinterpret_cast<uint8_t*>(&actKernelInvocation);
    binDataSection.appendData(ptrCharTmp, getBinarySize());
}

size_t vpux::VPUMI37XX::ActKernelInvocationOp::getBinarySize() {
    return sizeof(nn_public::VpuActKernelInvocation);
}

size_t vpux::VPUMI37XX::ActKernelInvocationOp::getAlignmentRequirements() {
    return alignof(nn_public::VpuActKernelInvocation);
}

vpux::VPURT::BufferSection vpux::VPUMI37XX::ActKernelInvocationOp::getMemorySpace() {
    return vpux::VPURT::BufferSection::DDR;
}

vpux::ELFNPU37XX::SectionFlagsAttr vpux::VPUMI37XX::ActKernelInvocationOp::getAccessingProcs() {
    return (ELFNPU37XX::SectionFlagsAttr::SHF_EXECINSTR | ELFNPU37XX::SectionFlagsAttr::VPU_SHF_PROC_DMA);
}

vpux::ELFNPU37XX::SectionFlagsAttr vpux::VPUMI37XX::ActKernelInvocationOp::getUserProcs() {
    return (ELFNPU37XX::SectionFlagsAttr::VPU_SHF_PROC_SHAVE);
}

size_t vpux::VPUMI37XX::ActKernelInvocationOp::getOffsetOfWithinOperation(mlir::Value val) {
    if (val == getRangeIndex()) {
        return offsetof(nn_public::VpuActKernelInvocation, range);
    }

    VPUX_THROW("Provided Value is not linked to the ActKernelInvocation Op or getOffset does not support it");
}
