//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/transforms/passes/insert_identity_pool_before_op.hpp"
#include "vpux/compiler/dialect/IE/IR/ops.hpp"

#include <mlir/IR/IRMapping.h>

using namespace vpux;

bool vpux::IE::isEligiblePostOp(mlir::Operation* op, Logger log) {
    auto postOpInterface = op->getOperand(0).getDefiningOp<IE::LayerWithPostOpInterface>();
    if (postOpInterface == nullptr || postOpInterface.getPostOp().has_value() ||
        !postOpInterface->getResult(0).hasOneUse()) {
        return true;
    }

    const auto inElemType = postOpInterface->getOperand(0).getType().cast<vpux::NDTypeInterface>().getElementType();
    const auto outElemType = postOpInterface->getResult(0).getType().cast<vpux::NDTypeInterface>().getElementType();
    // Because of the convert to float, the prelu shift will be bypassed. Check PPE diagram
    if (inElemType.isa<mlir::quant::QuantizedType>() && !outElemType.isa<mlir::quant::QuantizedType>() &&
        mlir::isa<IE::PReluOp, IE::LeakyReluOp>(op)) {
        log.trace("A PRelu or LeakyRely at {0} has mixed precision producer, and because of this the prelu shift will "
                  "be skiped",
                  op->getLoc());
        return true;
    }

    // Insert AvgPool between MaxPool and Clamp since MaxPool fused with Clamp is not fully supported
    // by firmware. Tracking Number: E#-145636
    auto parentMaxPoolOp = op->getOperand(0).getDefiningOp<IE::MaxPoolOp>();
    if (parentMaxPoolOp != nullptr && mlir::isa<IE::ClampOp>(op)) {
        log.trace("A MaxPool Op followed by a Clamp Op at {0} will not trigger a fusion, insert an AvgPool Op then",
                  op->getLoc());
        return true;
    }

    log.trace("A PostOp at {0} has already got a suitable producer", op->getLoc());
    return false;
}

mlir::LogicalResult vpux::IE::genericIdInserter(mlir::Operation* concreteOp, const InsertIdFunctor& insertId,
                                                mlir::PatternRewriter& rewriter, Logger log) {
    mlir::Operation* identityOp = insertId(concreteOp, rewriter, log);
    if (identityOp == nullptr) {
        return mlir::failure();
    }

    mlir::IRMapping mapper;
    const SmallVector<mlir::Value> inputsToMap = {identityOp->getResult(0)};
    mapper.map(concreteOp->getOperands(), ArrayRef(inputsToMap));
    auto* newLayerOp = rewriter.clone(*concreteOp, mapper);
    rewriter.replaceOp(concreteOp, newLayerOp->getResult(0));

    return mlir::success();
}
