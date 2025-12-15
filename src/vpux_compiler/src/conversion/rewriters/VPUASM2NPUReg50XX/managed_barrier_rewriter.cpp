//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/conversion/rewriters/VPUASM2NPUReg50XX/managed_barrier_rewriter.hpp"

#include "vpux/compiler/NPU50XX/dialect/NPUReg50XX/ops.hpp"
#include "vpux/compiler/NPU50XX/dialect/NPUReg50XX/types.hpp"
#include "vpux/compiler/dialect/VPURegMapped/utils.hpp"

using namespace NPUReg50XX;
using namespace NPUReg50XX::Descriptors;
using namespace vpux::VPURegMapped;

namespace vpux {
namespace vpuasm2npureg50xx {

mlir::LogicalResult ManagedBarrierRewriter::matchAndRewrite(VPUASM::ManagedBarrierOp origOp,
                                                            mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    auto workItemIdx = origOp.getWorkItemIdx();

    uint32_t workItemRegVal = 0;
    uint32_t enqueueCount = 0;

    if (workItemIdx.has_value()) {
        enqueueCount = origOp.getWorkItemCount();
        workItemRegVal = workItemIdx.value().getValue();
    }

    VpuTaskBarrierMap taskBarrierMapDescriptor;
    taskBarrierMapDescriptor.write<Fields::tb_next_same_id>(
            checked_cast_reg<NPUReg50XX::RegField_next_same_id_Type>(static_cast<uint32_t>(origOp.getNextSameId())));
    taskBarrierMapDescriptor.write<Fields::tb_producer_count>(origOp.getProducerCount());
    taskBarrierMapDescriptor.write<Fields::tb_consumer_count>(origOp.getConsumerCount());
    taskBarrierMapDescriptor.write<Fields::tb_real_id>(origOp.getId());
    taskBarrierMapDescriptor.write<Fields::tb_work_item_idx>(workItemRegVal);
    taskBarrierMapDescriptor.write<Fields::tb_enqueue_count>(enqueueCount);

    rewriter.create<NPUReg50XX::ManagedBarrierOp>(origOp.getLoc(), origOp.getSymNameAttr(),
                                                  std::move(taskBarrierMapDescriptor));
    rewriter.eraseOp(origOp);

    return mlir::success();
}
}  // namespace vpuasm2npureg50xx
}  // namespace vpux
