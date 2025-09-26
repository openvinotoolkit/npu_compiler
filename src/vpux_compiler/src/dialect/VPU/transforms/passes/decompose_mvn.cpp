//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/tiling.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>

namespace vpux::VPU {
#define GEN_PASS_DECL_DECOMPOSEMVN
#define GEN_PASS_DEF_DECOMPOSEMVN
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;

namespace {

// This computes the buffers for the most granular tiling possible for the original MVN op.
// The decomposition happens if not even this tiling scheme fits in CMX.
SmallVector<vpux::NDTypeInterface> getTiledBuffers(vpux::NDTypeInterface input, vpux::NDTypeInterface output,
                                                   DimArr nonNormDims) {
    auto inputShape = to_small_vector(input.getShape());
    auto outputShape = to_small_vector(output.getShape());
    for (auto& dim : nonNormDims) {
        inputShape[dim.ind()] = 1;
        outputShape[dim.ind()] = 1;
    }
    return SmallVector<vpux::NDTypeInterface>{input.changeShape(ShapeRef(inputShape)),
                                              output.changeShape(ShapeRef(outputShape))};
}

bool checkInsertReshapeDimOrder(DimsOrder dimOrder, bool acrossChannel) {
    if (dimOrder == DimsOrder::HCNW || dimOrder == DimsOrder::HNWC || dimOrder == DimsOrder::CWNH) {
        return false;
    }

    if (acrossChannel == true) {
        return true;
    }

    return dimOrder != DimsOrder::NHCW && dimOrder != DimsOrder::NWCH && dimOrder != DimsOrder::WCHN;
}

//
// DecomposeMVNPass
//

class DecomposeMVNPass final : public VPU::impl::DecomposeMVNBase<DecomposeMVNPass> {
public:
    explicit DecomposeMVNPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

public:
    class MVNConverter;

private:
    void safeRunOnFunc() final;
};

//
// MVNConverter
//

class DecomposeMVNPass::MVNConverter final : public mlir::OpRewritePattern<VPU::MVNOp> {
public:
    MVNConverter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<VPU::MVNOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::MVNOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult DecomposeMVNPass::MVNConverter::matchAndRewrite(VPU::MVNOp origOp,
                                                                    mlir::PatternRewriter& rewriter) const {
    _log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    const auto& ctx = origOp.getContext();
    auto module = origOp.getOperation()->getParentOfType<mlir::ModuleOp>();
    auto inputType = mlir::cast<vpux::NDTypeInterface>(origOp.getInput().getType());
    auto inputDimOrder = inputType.getDimsOrder();
    auto numClusters = config::getTileExecutor(module).getCount();
    const auto accrossChannels = origOp.getAcrossChannels();

    mlir::Value lastOp = origOp.getInput();
    if (checkInsertReshapeDimOrder(inputDimOrder, accrossChannels)) {
        const auto inputShape = inputType.getShape();
        const auto inputBatch = inputShape[Dims4D::Act::N];
        const auto inputChannel = accrossChannels ? 1 : inputShape[Dims4D::Act::C];
        const auto inputHeight =
                accrossChannels ? inputShape[Dims4D::Act::H] * inputShape[Dims4D::Act::W] * inputShape[Dims4D::Act::C]
                                : inputShape[Dims4D::Act::H] * inputShape[Dims4D::Act::W];

        auto newShape = Shape{inputBatch, inputChannel, inputHeight, 1};

        lastOp = rewriter.create<VPU::ShapeCastOp>(origOp.getLoc(), inputType.changeShape(newShape), origOp.getInput(),
                                                   getIntArrayAttr(ctx, newShape));
    }

    auto tileMVN1SumOp = rewriter.create<VPU::MVN1SumOp>(appendLoc(origOp.getLoc(), "mvn1Sum"), lastOp, accrossChannels,
                                                         origOp.getNormalizeVariance(), numClusters);

    const auto internalReshape =
            origOp.getInternalReshape().has_value() ? origOp.getInternalReshape().value() : nullptr;
    auto tileMVN1MeanVarOp = rewriter.create<VPU::MVN1MeanVarOp>(
            appendLoc(origOp.getLoc(), "mvn1MeanVar"), tileMVN1SumOp->getResult(0),
            getIntArrayAttr(rewriter, inputType.getShape().raw()), accrossChannels, origOp.getNormalizeVariance(),
            origOp.getEps(), inputType.getElementType(), internalReshape);

    auto tileMVN1NormalizeOp = rewriter.create<VPU::MVN1NormalizeOp>(
            appendLoc(origOp.getLoc(), "mvn1Normalize"), lastOp, tileMVN1MeanVarOp.getResult(),
            origOp.getAcrossChannelsAttr(), origOp.getNormalizeVarianceAttr(), origOp.getHighPrecisionNormalizeAttr());

    auto origOpOutType = mlir::cast<vpux::NDTypeInterface>(origOp.getOutput().getType());
    auto reshapeOutOp =
            rewriter.createOrFold<VPU::ShapeCastOp>(origOp.getLoc(), origOpOutType, tileMVN1NormalizeOp.getOutput(),
                                                    getIntArrayAttr(ctx, origOpOutType.getShape()));

    rewriter.replaceOp(origOp, reshapeOutOp);
    return mlir::success();
}

//
// safeRunOnFunc
//

void DecomposeMVNPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::ConversionTarget target(ctx);

    target.addDynamicallyLegalOp<VPU::MVNOp>([&](VPU::MVNOp op) {
        const auto inputType = mlir::cast<vpux::NDTypeInterface>(op.getInput().getType());
        const auto outputType = mlir::cast<vpux::NDTypeInterface>(op.getOutput().getType());
        if (inputType.getRank() != 4) {
            _log.nest(1).trace("Support for decompose MVN is limited to 4D tensors only");
            return true;
        }

        if (op.getInternalReshape().has_value()) {
            _log.nest(1).trace("Real 'internal_reshape' does not fit into CMX");
            return false;
        }

        // Can't get feasible tiling strategy for MVNOp because it will not fit into CMX.
        if (!op.fitIntoCMX(getTiledBuffers(inputType, outputType, op.getNonNormDims()))) {
            _log.nest(1).trace("Can't still fit into CMX after tiling. The pass is used to decompose MVNOp.");
            return false;
        }
        return true;
    });

    target.addLegalOp<VPU::MVN1SumOp>();
    target.addLegalOp<VPU::MVN1MeanVarOp>();
    target.addLegalOp<VPU::MVN1NormalizeOp>();
    target.addLegalOp<VPU::ShapeCastOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<MVNConverter>(&ctx, _log);

    auto func = getOperation();
    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        _log.debug("Failed to decompose MVNOp into 3 separete functions.");
        signalPassFailure();
    }
}

}  // namespace

//
// createDecomposeMVNPass
//

std::unique_ptr<mlir::Pass> vpux::VPU::createDecomposeMVNPass(Logger log) {
    return std::make_unique<DecomposeMVNPass>(log);
}
