//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/conversion/rewriters/VPUASM2NPUReg50XX/barrier_configure_rewriter.hpp"

#include "vpux/compiler/NPU50XX/dialect/NPUReg50XX/ops.hpp"
#include "vpux/compiler/NPU50XX/dialect/NPUReg50XX/types.hpp"
#include "vpux/compiler/dialect/VPURegMapped/utils.hpp"

using namespace NPUReg50XX;
using namespace NPUReg50XX::Descriptors;
using namespace vpux::VPURegMapped;

namespace vpux {
namespace vpuasm2npureg50xx {

mlir::LogicalResult BarrierRewriter::matchAndRewrite(VPUASM::ConfigureBarrierOp origOp,
                                                     mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());
    // origOp.getNextSameId() is int64 with invalid barrier represented by -1 and a max
    // value of numeric_limits<uint32_t>::max() - 1
    // At this point it is cast to uint32 as required by the NNRuntime with invalid barrier
    // represented by numeric_limits<uint32_t>::max()
    VpuBarrierCountConfig barrierConfigDescriptor;
    barrierConfigDescriptor.write<Fields::next_same_id_>(
            checked_cast_reg<NPUReg50XX::RegField_next_same_id_Type>(static_cast<uint32_t>(origOp.getNextSameId())));
    barrierConfigDescriptor.write<Fields::producer_count_>(origOp.getProducerCount());
    barrierConfigDescriptor.write<Fields::consumer_count_>(origOp.getConsumerCount());
    barrierConfigDescriptor.write<Fields::real_id_>(origOp.getId());

    rewriter.create<NPUReg50XX::ConfigureBarrierOp>(origOp->getLoc(), origOp.getSymNameAttr(),
                                                    std::move(barrierConfigDescriptor));

    rewriter.eraseOp(origOp);

    return mlir::success();
}
}  // namespace vpuasm2npureg50xx
}  // namespace vpux
