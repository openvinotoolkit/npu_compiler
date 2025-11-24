//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/dynamic_shape_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/reify_shape.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Dialect/Bufferization/IR/Bufferization.h>

namespace vpux::IE {
#define GEN_PASS_DECL_POPULATEDYNAMICDIMENSIONSHW
#define GEN_PASS_DEF_POPULATEDYNAMICDIMENSIONSHW
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

// Adapter for mlir::bufferization::populateDynamicDimSizes
// StridedSlice is required to crop the output (which will eventually become static) to dynamic sizes.
// Since populateDynamicDimSizes returns only dynamic dimensions, the pass needs to concatenate them
// with static dimensions and then provide the result to StridedSlice.
// StridedSlice infers its output as tensor<?x?x?x?xf16> when any of begins, ends or strides are
// unknown at compile time. DynamicReshape eliminates the discrepancy between the output of
// StridedSlice and the input of mlir.return (which is not necessarily set to tensor<?x?x?x?xf16>)
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
    auto producer{operand.getDefiningOp()};
    if (!mlir::isa<mlir::ReifyRankedShapedTypeOpInterface>(producer)) {
        log.trace("Operand producer is not a ReifyRankedShapedTypeOpInterface.");
        return;
    }

    SmallVector<mlir::Value> dynamicOperands{};
    IE::DynamicDimOpBuilder builder(op);
    mlir::bufferization::populateDynamicDimSizes(builder, producer->getLoc(), operand, dynamicOperands);
    auto newShapeValue = buildConcat(producer->getLoc(), builder, getShape(producer->getResult(0)), dynamicOperands);
    auto newResult = repackDynamicTensor(builder, producer, operandType, newShapeValue);

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
    // Skip Convolution -> Add, MaxPool -> ReLU and other combinations.
    // Populate dynamic dimensions only when a consumer is not from the list.
    // For example: ReLU -> Reshape is a good candidate to become ReLU -> StridedSlice -> Reshape
    // In this example reshape kernel can handle dynamic shapes properly.
    // Convolution, MaxPool, Add and ReLU cannot.
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
}  // namespace

namespace {
class PopulateDynamicDimensionsHWPass final :
        public IE::impl::PopulateDynamicDimensionsHWBase<PopulateDynamicDimensionsHWPass> {
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
};  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createPopulateDynamicDimensionsHWPass(Logger log) {
    return std::make_unique<PopulateDynamicDimensionsHWPass>(log);
}
