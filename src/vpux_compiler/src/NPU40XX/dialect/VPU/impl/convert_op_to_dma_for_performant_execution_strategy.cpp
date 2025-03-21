//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU40XX/dialect/VPU/impl/convert_ops_to_dma_for_performant_execution_strategy.hpp"

#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"

#include <llvm/ADT/TypeSwitch.h>

using namespace vpux;

namespace {
//
// MovetoDMAGather
//

class MovetoDMAGather final : public mlir::OpRewritePattern<VPU::GatherOp> {
public:
    MovetoDMAGather(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<VPU::GatherOp>(ctx), _log(log) {
        setDebugName("MovetoDMAGather");
    }

    mlir::LogicalResult matchAndRewrite(VPU::GatherOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

// GatherDMA indices only support positive values
// - If the indices is constant, iterate through the values and convert any negatives to positives
// - If the indices is dynamic, TODO: E#149660
mlir::Value handleNegativeIndices(mlir::Value indices, ShapeRef dataShape, const Dim axis,
                                  mlir::PatternRewriter& rewriter) {
    if (auto indicesCst = mlir::dyn_cast_or_null<Const::DeclareOp>(indices.getDefiningOp())) {
        const auto indicesContent = indicesCst.getContent();
        auto indicesVals = to_small_vector(indicesContent.getValues<int64_t>());
        auto firstNegativeIt = std::find_if(indicesVals.begin(), indicesVals.end(), [](int64_t val) {
            return val < 0;
        });

        if (firstNegativeIt != indicesVals.end()) {
            for (auto it = firstNegativeIt; it != indicesVals.end(); ++it) {
                if (*it < 0) {
                    *it += dataShape[axis];
                }
            }

            auto indicesType = mlir::cast<NDTypeInterface>(indicesCst.getOutput().getType());
            auto indicesStorageType = mlir::cast<mlir::RankedTensorType>(
                    indicesType.changeElemType(mlir::IntegerType::get(indicesCst.getContext(), 64)));
            auto indicesStorageAttr = Const::createConstContent(indicesStorageType, ArrayRef(indicesVals));

            return rewriter
                    .create<Const::DeclareOp>(indicesCst.getLoc(), indicesStorageType,
                                              Const::ContentAttr::get(indicesStorageAttr))
                    .getOutput();
        }
    }
    return indices;
}

mlir::LogicalResult MovetoDMAGather::matchAndRewrite(VPU::GatherOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", this->getDebugName(), origOp->getName(), origOp.getLoc());

    auto inputType = mlir::cast<NDTypeInterface>(origOp.getInput().getType());
    auto axis = Dim(origOp.getAxisValue().value());

    auto indices = handleNegativeIndices(origOp.getIndices(), inputType.getShape(), axis, rewriter);

    auto reshapeOperand = [&](mlir::Value operand, ShapeRef newShape, const mlir::Location& location) {
        auto newShapeAttr = getIntArrayAttr(operand.getContext(), newShape);
        return rewriter.createOrFold<VPU::ReshapeOp>(location, operand, nullptr, false, newShapeAttr);
    };

    // Ensure Indices tensor has the same rank as the Input tensor for GatherDMA
    //  - Fuse Indices into one dimension and align it with the axis dimension of Input
    //  - Fill other dimensions with 1
    // Example:                          Reshape To:
    //   Input:   [1, 16, 32, 32]          Input:   [1, 16, 32, 32]
    //   Indices: [2, 5]                   Indices: [1, 10, 1, 1]
    //   Axis:    1                        Axis:    1
    //   Output:  [1, 2, 5, 32, 32]        Output:  [1, 10, 32, 32]

    auto indicesType = mlir::cast<NDTypeInterface>(indices.getType());
    auto outputType = mlir::cast<NDTypeInterface>(origOp.getOutput().getType());
    Shape newIndicesShape(inputType.getRank(), 1);
    newIndicesShape[axis] = indicesType.getShape().totalSize();
    auto reshapeIndicesOp = reshapeOperand(indices, newIndicesShape, takeOpLoc(origOp, "reshape_indices"));

    // HW requirement: each list entry must be 64 bits
    auto requiredType64 = mlir::IntegerType::get(origOp.getContext(), 64);
    auto convertIndicesOp = rewriter.createOrFold<VPU::ConvertOp>(origOp->getLoc(), reshapeIndicesOp,
                                                                  mlir::TypeAttr::get(requiredType64));

    auto gatherDMAOp =
            rewriter.create<VPU::GatherDMAOp>(origOp.getLoc(), origOp.getInput(), convertIndicesOp, origOp.getAxis(),
                                              origOp.getAxisValueAttr(), origOp.getBatchDims());

    auto reshapeOutOp =
            reshapeOperand(gatherDMAOp.getOutput(), outputType.getShape(), takeOpLoc(origOp, "reshape_output"));

    origOp.getOutput().replaceAllUsesWith(reshapeOutOp);
    rewriter.eraseOp(origOp);

    return mlir::success();
}

}  // namespace

//
// ConvertOpToDMAForPerformantExecutionStrategy
//

void VPU::arch40xx::ConvertOpToDMAForPerformantExecutionStrategy::addPatterns(mlir::RewritePatternSet& patterns,
                                                                              Logger& log) const {
    auto ctx = patterns.getContext();
    patterns.insert<MovetoDMAGather>(ctx, log);
}

void VPU::arch40xx::ConvertOpToDMAForPerformantExecutionStrategy::markOpLegality(mlir::ConversionTarget& target,
                                                                                 Logger& log) const {
    target.addDynamicallyLegalOp<VPU::GatherOp>([&](VPU::GatherOp op) {
        if (!VPU::isLegalConvertToGatherDMA(op, /*isElementTile*/ false, /*isIndicesTile*/ false, log)) {
            return true;
        }
        return false;
    });

    target.addLegalOp<Const::DeclareOp>();
    target.addLegalOp<VPU::GatherDMAOp>();
    target.addLegalOp<VPU::ReshapeOp>();
    target.addLegalOp<VPU::ConvertOp>();
}
