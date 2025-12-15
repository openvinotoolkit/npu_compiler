//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/conversion/rewriters/VPUASM2NPUReg50XX/work_item_rewriter.hpp"

#include "vpux/compiler/NPU50XX/dialect/NPUReg50XX/ops.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/utils.hpp"
#include "vpux/compiler/dialect/config/utils/config_option_utils.hpp"

using namespace NPUReg50XX;
using namespace NPUReg50XX::Descriptors;

namespace vpux {
namespace vpuasm2npureg50xx {

mlir::LogicalResult WorkItemRewriter::matchAndRewrite(VPUASM::WorkItemOp origOp,
                                                      mlir::PatternRewriter& rewriter) const {
    enum TaskType : uint8_t { DPU = 0, DMA, KERNEL, SYSTEM_MANAGEMENT, UNKNOWN = 255 };

    auto realTaskIndex = origOp.getRealTaskIndex();
    auto nextWorkItemIdx = origOp.getNextWorkitemIdx();

    uint32_t nextWorkItemIdxValue = 0;

    if (nextWorkItemIdx.has_value()) {
        nextWorkItemIdxValue = nextWorkItemIdx.value().getValue();
    }

    uint64_t descPtrOffset = 0;
    TaskType workItemType;
    auto listIndex = realTaskIndex.getListIdx();

    switch (origOp.getTaskType()) {
    case VPURegMapped::TaskType::DPUVariant:
        workItemType = TaskType::DPU;
        descPtrOffset = static_cast<uint64_t>(VPUMI40XX::generateTileMask({realTaskIndex.getTileIdx()}));
        break;
    case VPURegMapped::TaskType::DMA:
        workItemType = TaskType::DMA;
        break;
    case VPURegMapped::TaskType::ActKernelInvocation:
        workItemType = TaskType::KERNEL;
        descPtrOffset = static_cast<uint64_t>(VPUMI40XX::generateTileMask({realTaskIndex.getTileIdx()}));
        // value 0 for Fields::wi_sub_unit is reserved for indicating to the RT that no dedicated Shave FIFO algorithm
        // was used (it is also useful for backward compatibility reasons - old blobs). Therefore in case dedicated
        // Shave FIFOs were assigned, increment the list index (and RT will decrement it on its side).
        if (config::isFifoPerShaveEngineEnabled(origOp)) {
            listIndex++;
        }
        break;
    default:
        return origOp.emitOpError("Invalid workItem task type");
        ;
    }

    WorkItem workItemDescriptor;
    workItemDescriptor.write<Fields::desc_ptr>(descPtrOffset);
    workItemDescriptor.write<Fields::wi_type>(workItemType);
    workItemDescriptor.write<Fields::wi_unit>(realTaskIndex.getTileIdx());
    workItemDescriptor.write<Fields::wi_sub_unit>(listIndex);
    workItemDescriptor.write<Fields::next_workitem_idx>(nextWorkItemIdxValue);

    rewriter.create<NPUReg50XX::WorkItemOp>(origOp.getLoc(), origOp.getSymNameAttr(), origOp.getTaskTypeAttr(),
                                            origOp.getFirstTaskAttr(), std::move(workItemDescriptor));
    rewriter.eraseOp(origOp);
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());
    return mlir::success();
}
}  // namespace vpuasm2npureg50xx
}  // namespace vpux
