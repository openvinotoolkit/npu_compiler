//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/attributes.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/activation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/arithmetic.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/pooling.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/walk_utils.hpp"

#include <mlir/IR/OperationSupport.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace vpux::IE {
#define GEN_PASS_DECL_FUSEOUTSTANDINGDEQUANT
#define GEN_PASS_DEF_FUSEOUTSTANDINGDEQUANT
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

// Helper function to check if operation is allowed to be walked through during backward search from Dequantize.
// These are view-like/layout transformation operations that reorganize data without computation.
// They're transparent to quantization - they don't change numerical values or quantization parameters,
// only how the data is organized in memory.
bool isAllowedElemTypeOp(mlir::Operation* op) {
    return mlir::isa<IE::AffineReshapeOp, IE::ExpandOp, IE::ExpandDilatedOp, IE::ReorderOp, IE::ReshapeOp,
                     IE::TransposeOp, IE::SliceOp>(op);
}

//
// FuseOutstandingDequantPass
//

class FuseOutstandingDequantPass final : public IE::impl::FuseOutstandingDequantBase<FuseOutstandingDequantPass> {
public:
    explicit FuseOutstandingDequantPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final;

private:
    void safeRunOnFunc() final;
};

mlir::LogicalResult FuseOutstandingDequantPass::initialize(mlir::MLIRContext* ctx) {
    if (mlir::failed(Base::initialize(ctx))) {
        return mlir::failure();
    }

    return mlir::success();
}

class DequantizeWithNCERewriter final : public mlir::OpRewritePattern<IE::DequantizeOp> {
public:
    DequantizeWithNCERewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::DequantizeOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::DequantizeOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult DequantizeWithNCERewriter::matchAndRewrite(IE::DequantizeOp origOp,
                                                               mlir::PatternRewriter& rewriter) const {
    const auto logCb = [&](const formatv_object_base& msg) {
        _log.trace("{0}", msg.str());
    };

    // Get the Dequantize Op input and output types
    const auto dequantInputType = origOp.getInput().getType();
    const auto elemType = mlir::cast<mlir::ShapedType>(dequantInputType).getElementType();
    const auto dequantUniformType = mlir::dyn_cast<mlir::quant::UniformQuantizedType>(elemType);
    const bool isPerChannel = mlir::isa<mlir::quant::UniformQuantizedPerAxisType>(elemType);

    const auto dequantOutputType = origOp.getOutput().getType();

    auto maybeQuantizedLayerOp = origOp.getInput().getDefiningOp();
    if (maybeQuantizedLayerOp == nullptr) {
        return matchFailed(rewriter, origOp, "Producer is a block argument");
    }
    if (!maybeQuantizedLayerOp->getResult(0).hasOneUse()) {
        return matchFailed(rewriter, origOp, "Producer has more than one consumer");
    }

    if (!mlir::isa<IE::QuantizedLayerOpInterface>(maybeQuantizedLayerOp)) {
        SmallVector<mlir::Operation*> targetOps;
        mlir::Operation* operation = origOp;
        _log.trace("[{0}] Search quantized NCE task for {1} at {2}", this->getDebugName(), origOp->getName(),
                   origOp->getLoc());
        while (operation) {
            auto input = (*operation->getOperands().begin()).getDefiningOp();

            // Input is a block argument - no NCE producer to find
            if (input == nullptr) {
                return matchFailed(rewriter, origOp, "Reached block argument while searching for NCE producer");
            }

            // Check if input is either an allowed elem-type operation or NCE task
            const bool isAllowedElemOp = isAllowedElemTypeOp(input);
            const bool isQuantizedLayerOp = mlir::isa<IE::QuantizedLayerOpInterface>(input);

            if (!isAllowedElemOp && !isQuantizedLayerOp) {
                return matchFailed(rewriter, origOp,
                                   "Ancestor {0} at {1} is neither allowed operation nor NCE operation",
                                   input->getName(), input->getLoc());
            }

            if (!input->hasOneUse()) {
                return matchFailed(rewriter, origOp, "Ancestor {0} at {1} has more than one consumer", input->getName(),
                                   input->getLoc());
            }

            if (isAllowedElemOp) {
                if (input->getNumOperands() > 1) {
                    return matchFailed(rewriter, origOp, "Ancestor {0} at {1} has more than one ancestors",
                                       input->getName(), input->getLoc());
                }

                // This is an allowed memory operation - continue walking
                _log.trace("[{0}] Push allowed operation {1} at {2}", this->getDebugName(), input->getName(),
                           input->getLoc());
                targetOps.push_back(input);
                operation = input;
                continue;
            }

            if (isQuantizedLayerOp) {
                _log.trace("[{0}] Found quantized layer {1} at {2}, stop pattern searching", this->getDebugName(),
                           input->getName(), input->getLoc());
                maybeQuantizedLayerOp = input;
                break;
            }
        }

        _log.trace("[{0}] Capture the pattern for {1} at {2}", this->getDebugName(), origOp->getName(),
                   origOp->getLoc());

        auto quantizedLayerOp = mlir::dyn_cast_or_null<IE::QuantizedLayerOpInterface>(maybeQuantizedLayerOp);
        if (quantizedLayerOp == nullptr) {
            return matchFailed(rewriter, origOp, "Producer is not a quantized layer operation");
        }
        if (!quantizedLayerOp.isMixPrecisionSupported(!isPerChannel)) {
            return matchFailed(rewriter, origOp, "Producer {0} is not supported", maybeQuantizedLayerOp->getName());
        }

        // Check if we have intermediate operations
        if (targetOps.empty()) {
            return matchFailed(rewriter, origOp, "No intermediate operations found");
        }

        auto* newQuantizedLayerOp = rewriter.clone(*maybeQuantizedLayerOp);
        vpux::NDTypeInterface newType = newQuantizedLayerOp->getResult(0).getType();
        newType = newType.changeElemType(dequantOutputType.getElementType());
        newQuantizedLayerOp->getResult(0).setType(newType);
        newQuantizedLayerOp->moveBefore(targetOps.back());

        _log.trace("[{0}] Replace {1} {2} at {3} with {4} {5} at {6}", this->getDebugName(),
                   maybeQuantizedLayerOp->getName(), maybeQuantizedLayerOp->getResult(0).getType(),
                   maybeQuantizedLayerOp->getLoc(), newQuantizedLayerOp->getName(),
                   newQuantizedLayerOp->getResult(0).getType(), newQuantizedLayerOp->getLoc());
        rewriter.replaceOp(maybeQuantizedLayerOp, newQuantizedLayerOp->getResult(0));

        // [NCE with quantized output]->[Allowed operations] ... ->[Dequantize] pattern is captured
        // Rewrite the sub-graph.
        for (auto iterator = targetOps.rbegin(); iterator != targetOps.rend(); ++iterator) {
            _log.trace("[{0}] Change {1} at {2} to {3}", this->getDebugName(), (*iterator)->getName(),
                       (*iterator)->getLoc(), (*iterator)->getResult(0).getType());
            inferReturnTypes(*iterator, InferShapedTypeMode::ELEM_TYPE);
        }

        // Remove old Dequantize ops.
        _log.trace("[{0}] Replace {1} at {2} with {3} at {4}", this->getDebugName(), origOp->getName(),
                   origOp->getLoc(), targetOps.front()->getName(), targetOps.front()->getLoc());
        rewriter.replaceOp(origOp, targetOps.front()->getResult(0));
    } else {
        auto quantizedLayerOp = mlir::dyn_cast_or_null<IE::QuantizedLayerOpInterface>(maybeQuantizedLayerOp);
        if (quantizedLayerOp == nullptr) {
            return matchFailed(rewriter, origOp, "Producer is not a quantized layer operation");
        }
        if (!quantizedLayerOp.isMixPrecisionSupported(!isPerChannel)) {
            return matchFailed(rewriter, origOp, "Producer {0} is not supported", maybeQuantizedLayerOp->getName());
        }
        auto layerWithPostOp = mlir::dyn_cast<IE::LayerWithPostOpInterface>(maybeQuantizedLayerOp);

        if (layerWithPostOp != nullptr) {
            // Check if postOp exists and is NOT ReLU - preserve Dequantize in that case
            // postOpAttr can be IE::ReluAttr (#IE.Relu<>) or other post-op attributes
            auto postOpAttr = layerWithPostOp.getPostOp();
            if (postOpAttr != nullptr && !mlir::isa<IE::ReluAttr>(postOpAttr)) {
                return matchFailed(rewriter, origOp, "{0} has non-ReLU postOp, preserving Dequantize",
                                   maybeQuantizedLayerOp->getName().getStringRef());
            }

            // Check if clamp attribute exists and is supported if fusing were to occur
            auto clampAttr = layerWithPostOp.getClampAttr();
            if (clampAttr != nullptr) {
                auto minValue = clampAttr.getAs<mlir::FloatAttr>("min").getValueAsDouble();
                auto maxValue = clampAttr.getAs<mlir::FloatAttr>("max").getValueAsDouble();
                auto type = origOp->getResult(0).getType();
                if (!layerWithPostOp.isSupportedClampProperties(minValue, maxValue, type, logCb)) {
                    return matchFailed(_log, rewriter, origOp, "Layer with new clamp is not supported");
                }
            }
        }

        auto* newQuantizedLayerOp = rewriter.clone(*maybeQuantizedLayerOp);
        mlir::Value result = newQuantizedLayerOp->getResult(0);
        result.setType(dequantOutputType);
        if (dequantUniformType && !isPerChannel && !dequantUniformType.isSigned() &&
            dequantUniformType.getZeroPoint() == 0) {
            // Replace implicit ReLU in the Dequantize Op with an actual ReLU Op
            result = rewriter.create<IE::ReLUOp>(takeOpLoc(origOp, "as_relu"), dequantOutputType, result);
        }
        rewriter.replaceOp(origOp, result);
        rewriter.eraseOp(maybeQuantizedLayerOp);
    }

    return mlir::success();
}

void FuseOutstandingDequantPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<DequantizeWithNCERewriter>(&ctx, _log);

    collectOpsAndApplyPatterns(func, std::move(patterns));
}

}  // namespace

//
// createFuseOutstandingDequant
//

std::unique_ptr<mlir::Pass> vpux::IE::createFuseOutstandingDequant(Logger log) {
    return std::make_unique<FuseOutstandingDequantPass>(log);
}
