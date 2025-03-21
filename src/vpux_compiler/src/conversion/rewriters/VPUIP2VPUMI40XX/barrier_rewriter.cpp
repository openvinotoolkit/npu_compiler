//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/conversion/rewriters/VPUIP2VPUMI40XX/barrier_rewriter.hpp"

#include "vpux/compiler/dialect/VPUMI40XX/ops.hpp"
#include "vpux/compiler/dialect/VPURegMapped/types.hpp"
#include "vpux/compiler/utils/types.hpp"

namespace vpux::vpuip2vpumi40xx {

mlir::LogicalResult BarrierRewriter::matchAndRewrite(VPURT::ConfigureBarrierOp origOp, OpAdaptor,
                                                     mlir::ConversionPatternRewriter& rewriter) const {
    auto ctx = origOp.getContext();
    auto zeroByteAttr = mlir::IntegerAttr::get(getUInt8Type(ctx), 0);
    rewriter.replaceOpWithNewOp<VPUMI40XX::ConfigureBarrierOp>(
            origOp, VPURegMapped::IndexType::get(ctx, 0),  // setup all barriers with the trivial index (0)
            checked_cast<uint8_t>(origOp.getId()),         // realId
            -1,                                            // nextSameId
            zeroByteAttr,                                  // producerCount,
            zeroByteAttr,                                  // consumerCount,
            origOp.getIsFinalBarrier(), origOp.getIsStartBarrier());
    return mlir::success();
}

}  // namespace vpux::vpuip2vpumi40xx
