//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/ELFNPU37XX/utils.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/sw_utils.hpp"
#include "vpux/compiler/dialect/VPUMI37XX/ops.hpp"
#include "vpux/compiler/dialect/VPURegMapped/types.hpp"
#include "vpux/compiler/utils/error.hpp"

#include <npu_37xx_nnrt.hpp>

#include <mlir/IR/BuiltinTypes.h>

using namespace vpux;
using namespace npu37xx;

//
// ActKernelRangeOp
//

void vpux::VPUMI37XX::ActKernelRangeOp::serialize(elf::writer::BinaryDataSection<uint8_t>& binDataSection) {
    auto kernel_task_type = getKernelTaskType();
    bool isCacheOp = VPUIP::isCacheOpTaskType(kernel_task_type);

    size_t kernel_text_size;
    size_t kernel_args_size;
    uint32_t kernel_entry;
    if (!isCacheOp) {
        auto kernel_text_value = getKernelTextIndex();
        VPUX_THROW_UNLESS(kernel_text_value != nullptr, "ActKernelRange has no kernel text index");
        auto kernel_args_value = getKernelArgsIndex();
        VPUX_THROW_UNLESS(kernel_args_value != nullptr, "ActKernelRange has no kernel args index");
        auto kernel_entry_value = getKernelEntryIndex();
        VPUX_THROW_UNLESS(kernel_entry_value != nullptr, "ActKernelRange has no kernel entry index");

        auto kernel_text_op = kernel_text_value.getDefiningOp<VPUMI37XX::DeclareKernelTextOp>();
        auto kernel_args_op = kernel_args_value.getDefiningOp<VPUMI37XX::DeclareKernelArgsOp>();
        auto kernel_entry_op = kernel_entry_value.getDefiningOp<VPUMI37XX::DeclareKernelEntryOp>();

        kernel_text_size = kernel_text_op.getBinarySize();
        kernel_args_size = kernel_args_op.getBinarySize();
        kernel_entry = kernel_entry_op.getKernelEntry();
    } else {
        kernel_text_size = 0;
        kernel_args_size = 0;
        kernel_entry = 0;
    }

    auto op = this;
    auto uses = op->getResult().getUses();

    uint32_t invo_count = 0;

    for (auto use = uses.begin(); use != uses.end(); use++) {
        if (mlir::isa<VPUMI37XX::ActKernelInvocationOp>(use.getUser())) {
            invo_count++;
        }
    }

    nn_public::VpuActKernelRange actKernelRange;

    memset(reinterpret_cast<void*>(&actKernelRange), 0, sizeof(actKernelRange));

    if (!isCacheOp) {
        actKernelRange.type = nn_public::VpuActWLType::WL_KERNEL;
    } else {
        auto taskType = VPU::symbolizeActShaveTaskType(kernel_task_type.getLeafReference().strref());
        switch (taskType.value()) {
        case VPU::ActShaveTaskType::CACHE_FLUSH:
            actKernelRange.type = nn_public::VpuActWLType::WL_CACHE_OP_FLUSH;
            break;
        case VPU::ActShaveTaskType::CACHE_INVALIDATE:
            actKernelRange.type = nn_public::VpuActWLType::WL_CACHE_OP_INVALIDATE;
            break;
        case VPU::ActShaveTaskType::CACHE_FLUSH_INVALIDATE:
            actKernelRange.type = nn_public::VpuActWLType::WL_CACHE_OP_FLUSHINV;
            break;
        default:
            VPUX_THROW("Unrecognized Kernel Task Type '{0}'", kernel_task_type.getLeafReference());
            break;
        }
    }
    actKernelRange.kernel_entry = kernel_entry;
    // text_window_base not used
    actKernelRange.code_size = checked_cast<uint32_t>(kernel_text_size);
    actKernelRange.data_sec_size = checked_cast<uint32_t>(kernel_args_size);
    actKernelRange.kernel_invo_count = invo_count;

    uint8_t* ptrCharTmp = reinterpret_cast<uint8_t*>(&actKernelRange);
    binDataSection.appendData(ptrCharTmp, getBinarySize());
}

size_t vpux::VPUMI37XX::ActKernelRangeOp::getBinarySize() {
    return sizeof(nn_public::VpuActKernelRange);
}

size_t vpux::VPUMI37XX::ActKernelRangeOp::getAlignmentRequirements() {
    return alignof(nn_public::VpuActKernelRange);
}

vpux::ELFNPU37XX::SectionFlagsAttr vpux::VPUMI37XX::ActKernelRangeOp::getAccessingProcs() {
    return (ELFNPU37XX::SectionFlagsAttr::SHF_EXECINSTR | ELFNPU37XX::SectionFlagsAttr::VPU_SHF_PROC_DMA);
}

vpux::ELFNPU37XX::SectionFlagsAttr vpux::VPUMI37XX::ActKernelRangeOp::getUserProcs() {
    return (ELFNPU37XX::SectionFlagsAttr::VPU_SHF_PROC_SHAVE);
}

vpux::VPURT::BufferSection vpux::VPUMI37XX::ActKernelRangeOp::getMemorySpace() {
    return vpux::VPURT::BufferSection::DDR;
}

size_t vpux::VPUMI37XX::ActKernelRangeOp::getOffsetOfWithinOperation(mlir::Value val) {
    if (val == getKernelTextIndex()) {
        return offsetof(nn_public::VpuActKernelRange, text_window_base);
    }
    VPUX_THROW("Provided Value is not linked to the ActKernelRange Op or getOffset does not support it");
}

mlir::LogicalResult vpux::VPUMI37XX::ActKernelRangeOp::verify() {
    auto taskType = getKernelTaskType();
    auto taskTypeVal = VPU::symbolizeActShaveTaskType(taskType.getLeafReference().strref());
    VPUX_THROW_UNLESS(taskTypeVal.has_value(), "VPU::ActShaveTaskType has no value.");

    if (taskTypeVal == VPU::ActShaveTaskType::COMPUTE &&
        !(getKernelTextIndex() && getKernelArgsIndex() && getKernelEntryIndex())) {
        return errorAt(getOperation(),
                       "ActKernelRange with COMPUTE taskType should have KernelText, KernelArgs, KernelEntry.");
    }

    return mlir::success();
}
