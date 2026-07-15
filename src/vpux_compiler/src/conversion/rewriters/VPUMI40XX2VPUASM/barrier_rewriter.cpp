//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/conversion/rewriters/VPUMI40XX2VPUASM/barrier_rewriter.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"
#include "vpux/compiler/dialect/VPURegMapped/ops.hpp"

namespace vpux {
namespace vpumi40xx2vpuasm {

mlir::FailureOr<SymbolizationResult> BarrierRewriter::symbolize(VPUMI40XX::ConfigureBarrierOp op, SymbolMapper&,
                                                                mlir::ConversionPatternRewriter& rewriter) const {
    static constexpr auto countBitWidth = 16;

    mlir::Operation* newOperation = nullptr;

    if (!_skipBarrierLowering) {
        auto result = op.getResult();
        auto symName = findSym(result).getRootReference();
        auto taskIdx = mlir::TypeAttr::get(op.getType());

        mlir::TypeAttr workItemIndex = nullptr;
        uint32_t enqueueUsersCount = 0;
        auto minWorkItemIdx = std::numeric_limits<uint64_t>::max();
        for (auto user : result.getUsers()) {
            if (auto enqu = mlir::dyn_cast<VPURegMapped::EnqueueOp>(user)) {
                ++enqueueUsersCount;

                auto workItemIdx = enqu.getType().getValue();
                if (workItemIdx < minWorkItemIdx) {
                    minWorkItemIdx = workItemIdx;
                    workItemIndex = mlir::TypeAttr::get(enqu.getType());
                }
            }
        }

        uint16_t producerCount = checked_cast<uint8_t>(op.getProducerCount().value_or(0));
        uint16_t consumerCount = checked_cast<uint8_t>(op.getConsumerCount().value_or(0));
        uint16_t barrierId = checked_cast<uint8_t>(op.getId());

        auto countType = rewriter.getIntegerType(countBitWidth, /*isSigned*/ false);

        auto newOp = rewriter.create<VPUASM::ManagedBarrierOp>(
                op.getLoc(), symName, taskIdx, workItemIndex, rewriter.getUI32IntegerAttr(enqueueUsersCount),
                rewriter.getIntegerAttr(countType, barrierId), op.getNextSameIdAttr(),
                rewriter.getIntegerAttr(countType, producerCount), rewriter.getIntegerAttr(countType, consumerCount));

        newOperation = newOp.getOperation();
    }

    rewriter.eraseOp(op);

    return SymbolizationResult(newOperation);
}

}  // namespace vpumi40xx2vpuasm
}  // namespace vpux
