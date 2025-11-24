//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/dynamic_shape_utils.hpp"
#include "vpux/compiler/dialect/const/dialect.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/dialect/core/types.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/error.hpp"

#include <mlir/IR/Builders.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/DialectConversion.h>

namespace vpux::IE {
#define GEN_PASS_DECL_OPTDYNAMICELTWISEWITHSHAPEOF
#define GEN_PASS_DEF_OPTDYNAMICELTWISEWITHSHAPEOF
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

bool isDynamicShape(mlir::Value value) {
    return getShape(value).isDynamic();
};

int findOperandMatchingOutput(mlir::Operation* origOp) {
    const auto output = origOp->getResult(0);
    const auto outputShape = getShape(output);
    const auto numOperands = static_cast<int>(origOp->getNumOperands());

    for (auto i = 0; i < numOperands; ++i) {
        const auto operand = origOp->getOperand(i);
        const auto operandShape = getShape(operand);
        if (operandShape == outputShape) {
            return i;
        }
    }
    return -1;
}

mlir::Value getDynamicOperand(mlir::Operation* origOp) {
    for (auto operand : origOp->getOperands()) {
        if (isDynamicShape(operand)) {
            return operand;
        }
    }
    return nullptr;
}

//
// OptDynamicEltwiseWithShapeOf
//

class OptDynamicEltwiseWithShapeOf final : public mlir::OpRewritePattern<IE::ShapeOfOp> {
public:
    OptDynamicEltwiseWithShapeOf(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::ShapeOfOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ShapeOfOp shapeOfOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

//
// isEltwiseWithoutBroadcast: foldDynamicEltwiseBeforeShapeOf
//

/*
                  input                                   Input   ----------------
                    |                                       |                    |
                    v                                       v                    V
            +----------------+                     +----------------+       +-----------+
            | DynamicEltwise |                     | DynamicEltwise |       |  ShapeOf  |
            +----------------+                     +----------------+       +-----------+
                    |                =====>                                       |
                    v                                                             v
               +-----------+
               |  ShapeOf  |
               +-----------+
                     |
                     v
*/

//
// isEltwiseWithBroadcast and isNonDynamicDimsAllOnes: convertDynamicEltwiseToDynamicReshape
//

/*
```
        dynamicInput1  input2                    input2   dynamicInput  -----
              |          |                          |         |              |
              v          v                          v         v              v
            +--------------+                     +--------------+     +--------------+
            |DynamicEltwise|                     |DynamicEltwise|     |DynamicReshape|
            +--------------+                     +--------------+     +--------------+
                   |                =====>                                    |
                   v                                                          v
             +-----------+                                              +-----------+
             |  ShapeOf  |                                              |  ShapeOf  |
             +-----------+                                              +-----------+
                   |                                                          |
                   v                                                          v
```
*/

mlir::LogicalResult OptDynamicEltwiseWithShapeOf::matchAndRewrite(IE::ShapeOfOp shapeOfOp,
                                                                  mlir::PatternRewriter& rewriter) const {
    auto definingOp = shapeOfOp.getInput().getDefiningOp();
    const auto outElemType = mlir::cast<vpux::NDTypeInterface>(shapeOfOp.getOutput().getType()).getElementType();
    const auto output = definingOp->getResult(0);

    if (findOperandMatchingOutput(definingOp) != -1) {
        int matchingIndex = findOperandMatchingOutput(definingOp);
        auto operand = definingOp->getOperand(matchingIndex);
        auto newResult = rewriter.create<IE::ShapeOfOp>(takeOpLoc(shapeOfOp, "new_result"), operand, outElemType);
        rewriter.replaceOp(shapeOfOp, newResult);
        return mlir::success();
    } else {
        const auto outShape = getShape(output);
        const auto outputRank = checked_cast<int64_t>(outShape.size());
        auto inputOperand = getDynamicOperand(definingOp);
        if (inputOperand == nullptr) {
            return mlir::failure();
        }
        const auto inElemType = mlir::cast<vpux::NDTypeInterface>(inputOperand.getType()).getElementType();
        auto shapedType = mlir::RankedTensorType::get({outputRank}, inElemType);

        mlir::Value shapeTensor;
        if (inElemType.isSignedInteger(32)) {
            auto shapeValues = IE::replaceDynamicDimsWithValue<int32_t>(to_small_vector(outShape), -1);
            shapeTensor = Const::createConst(rewriter, shapeOfOp->getLoc(), shapedType, ArrayRef(shapeValues));
        } else {
            auto shapeValues = IE::replaceDynamicDimsWithValue<int64_t>(to_small_vector(outShape), -1);
            shapeTensor = Const::createConst(rewriter, shapeOfOp->getLoc(), shapedType, ArrayRef(shapeValues));
        }

        auto outputboundedType = mlir::dyn_cast<Core::BoundedTensorType>(output.getType());
        VPUX_THROW_UNLESS(outputboundedType != nullptr, "Expected to get bounded output type, got {0}",
                          output.getType());
        auto outBounds = outputboundedType.getBounds();

        const auto reshapeLoc = appendLoc(shapeOfOp->getLoc(), "dynamic_reshape");
        auto reshapeResult = rewriter.create<IE::DynamicReshapeOp>(reshapeLoc, inputOperand, shapeTensor,
                                                                   getIntArrayAttr(getContext(), outShape),
                                                                   getIntArrayAttr(getContext(), outBounds));
        auto newResult =
                rewriter.create<IE::ShapeOfOp>(takeOpLoc(shapeOfOp, "dyn_reshape_result"), reshapeResult, outElemType);

        rewriter.replaceOp(shapeOfOp, newResult);
        return mlir::success();
    }
}

//
// OptDynamicEltwiseWithShapeOfPass
//

class OptDynamicEltwiseWithShapeOfPass final :
        public IE::impl::OptDynamicEltwiseWithShapeOfBase<OptDynamicEltwiseWithShapeOfPass> {
public:
    explicit OptDynamicEltwiseWithShapeOfPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void OptDynamicEltwiseWithShapeOfPass::safeRunOnFunc() {
    auto& ctx = getContext();
    const auto isOptimizableShapeOf = [](IE::ShapeOfOp op) {
        const auto isNonDynamicDimsAllOnes = [](mlir::Value value) {
            auto shape = getShape(value);
            return std::all_of(shape.begin(), shape.end(), [](int64_t dim) {
                return dim == mlir::ShapedType::kDynamic || dim == 1;
            });
        };

        auto definingOp = op->getOperand(0).getDefiningOp();
        if (definingOp == nullptr || !definingOp->hasTrait<IE::EltwiseOp>()) {
            return true;
        } else if (findOperandMatchingOutput(definingOp) != -1) {
            return false;
        }
        auto output = definingOp->getResult(0);
        return !(isDynamicShape(output) && isNonDynamicDimsAllOnes(output));
    };

    mlir::ConversionTarget target(ctx);
    target.addLegalDialect<Const::ConstDialect>();
    target.addDynamicallyLegalOp<IE::ShapeOfOp>(isOptimizableShapeOf);
    target.addLegalOp<IE::DynamicReshapeOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<OptDynamicEltwiseWithShapeOf>(&ctx, _log);

    auto func = getOperation();
    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createOptDynamicEltwiseWithShapeOfPass(Logger log) {
    return std::make_unique<OptDynamicEltwiseWithShapeOfPass>(log);
}
