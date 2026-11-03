//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/utils/dynamic_shape_utils.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/reify_shape.hpp"
#include "vpux/compiler/dialect/const/dialect.hpp"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Bufferization/IR/Bufferization.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>

namespace vpux::VPU {
#define GEN_PASS_DECL_POPULATEDYNAMICDIMENSIONSHW
#define GEN_PASS_DEF_POPULATEDYNAMICDIMENSIONSHW
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;

namespace {

// Adapter for mlir::bufferization::populateDynamicDimSizes
// StridedSlice is required to crop the output (which will eventually become static) to dynamic sizes.
// Since populateDynamicDimSizes returns only dynamic dimensions, the pass needs to concatenate them
// with static dimensions and then provide the result to StridedSlice.
// StridedSlice infers its output as tensor<?x?x?x?xf16> when any of begins, ends or strides are
// unknown at compile time. DynamicReshape eliminates the discrepancy between the output of
// StridedSlice and the type expected by the consumer.
void populateDynamicOperand(mlir::Operation* op, const unsigned operandIdx, Logger log) {
    mlir::Value operand{op->getOperand(operandIdx)};
    if (mlir::isa<mlir::BlockArgument>(operand)) {
        log.trace("Operand is a BlockArg.");
        return;
    }

    auto operandType = mlir::cast<NDTypeInterface>(operand.getType());
    const auto operandShape = operandType.getShape();
    if (operandShape.isStatic()) {
        log.trace("Operand has static shape.");
        return;
    }
    // VPU.DynamicReshape supports only the default dims order.
    if (operandType.getDimsOrder() != DimsOrder::fromNumDims(operandType.getRank())) {
        log.trace("Operand has non-default dims order {0}.", operandType.getDimsOrder());
        return;
    }
    auto producer{operand.getDefiningOp()};
    if (!mlir::isa<mlir::ReifyRankedShapedTypeOpInterface>(producer)) {
        log.trace("Operand producer is not a ReifyRankedShapedTypeOpInterface.");
        return;
    }
    // VPU::repackDynamicTensor always repacks the first result of the producer.
    if (operand != producer->getResult(0)) {
        log.trace("Operand is not the first result of its producer.");
        return;
    }

    SmallVector<mlir::Value> dynamicOperands{};
    IE::DynamicDimOpBuilder builder(op);
    mlir::bufferization::populateDynamicDimSizes(builder, producer->getLoc(), operand, dynamicOperands);
    auto newShapeValue = VPU::buildConcat(producer->getLoc(), builder, operandShape, dynamicOperands);
    auto newResult = VPU::repackDynamicTensor(builder, producer, operandType, newShapeValue);

    op->setOperand(operandIdx, newResult);
}

void populateDynamicSizes(mlir::Operation* op, Logger log) {
    log.debug("Got '{0}' at '{1}'", op->getName(), op->getLoc());

    auto nestedLog = log.nest();
    if (!IE::needsStaticShape(op)) {
        nestedLog.trace("Op does not need static shape.");
        return;
    }
    mlir::Value output = op->getResult(0);
    if (!output.hasOneUse()) {
        nestedLog.trace("Op's output tensor has more than one use.");
        return;
    }
    mlir::Operation* consumer = *output.getUsers().begin();
    // Skip NCE.Convolution -> NCE.Convolution, NCE.Convolution -> DepthToSpace and other combinations.
    // Populate dynamic dimensions only when a consumer does not require static shape.
    // For example: Convert -> SoftMax becomes Convert -> StridedSlice -> DynamicReshape -> SoftMax
    if (IE::needsStaticShape(consumer)) {
        nestedLog.trace("Op consumer of type {0} at loc {1} needs static shape.", consumer->getName(),
                        consumer->getLoc());
        return;
    }

    nestedLog.debug("Crop dynamic output from the static one.");
    for (const unsigned idx : irange(consumer->getNumOperands())) {
        populateDynamicOperand(consumer, idx, nestedLog);
    }
}

class PopulateDynamicDimensionsHWPass final :
        public VPU::impl::PopulateDynamicDimensionsHWBase<PopulateDynamicDimensionsHWPass> {
public:
    explicit PopulateDynamicDimensionsHWPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void PopulateDynamicDimensionsHWPass::safeRunOnFunc() {
    auto func = getOperation();
    func->walk([&](mlir::Operation* op) {
        populateDynamicSizes(op, _log);
    });
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::VPU::createPopulateDynamicDimensionsHWPass(Logger log) {
    return std::make_unique<PopulateDynamicDimensionsHWPass>(log);
}
