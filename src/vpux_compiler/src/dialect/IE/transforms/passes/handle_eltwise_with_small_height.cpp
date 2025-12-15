//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/reshape_utils.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/factors.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/DialectConversion.h>
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Support/LLVM.h"

namespace vpux::IE {
#define GEN_PASS_DECL_HANDLEELTWISEWITHSMALLHEIGHT
#define GEN_PASS_DEF_HANDLEELTWISEWITHSMALLHEIGHT
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;
namespace {

//
// HandleEltwiseWithSmallHeight
//

class HandleEltwiseWithSmallHeight final : public mlir::OpRewritePattern<IE::AddOp> {
public:
    HandleEltwiseWithSmallHeight(mlir::MLIRContext* ctx, int64_t numClusters, Logger log)
            : mlir::OpRewritePattern<IE::AddOp>(ctx), _log(log), _numClusters(numClusters) {
        this->setDebugName("HandleEltwiseWithSmallHeight");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::AddOp addOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
    int64_t _numClusters;
};

/*
Eltwise-like ops with small height can be adjusted by:
1. Reshaping width, e.g. 1x1920x3x1080 -> 1x1920x810x4
2. Reshaping channel, e.g. 1x200000x1x1 -> 1x16x3125x4
*/

mlir::LogicalResult HandleEltwiseWithSmallHeight::matchAndRewrite(IE::AddOp addOp,
                                                                  mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), addOp->getName(), addOp->getLoc());

    auto isLastOp = addOp->getResult(0).getUsers().empty();
    if (isLastOp) {
        return matchFailed(_log, rewriter, addOp, "No users for AddOp found");
    }

    auto input1 = addOp.getInput1();
    auto input2 = addOp.getInput2();
    auto output = addOp.getResult();

    if (mlir::isa_and_nonnull<IE::AffineReshapeOp>(input1.getDefiningOp()) ||
        mlir::isa_and_nonnull<IE::AffineReshapeOp>(input2.getDefiningOp())) {
        return matchFailed(_log, rewriter, addOp, "Input Operands are null or already an AffineReshape op");
    }

    auto input1Type = mlir::dyn_cast<vpux::NDTypeInterface>(input1.getType());
    auto input2Type = mlir::dyn_cast<vpux::NDTypeInterface>(input2.getType());
    auto outputType = mlir::dyn_cast<vpux::NDTypeInterface>(output.getType());

    if (!input1Type || !input2Type) {
        return matchFailed(_log, rewriter, addOp, "Operands do not have NDTypeInterface");
    }
    if (input1Type != input2Type) {
        return matchFailed(_log, rewriter, addOp, "Inputs to AddOp are not of the same type");
    }

    auto input1Shape = input1Type.getShape();
    if (input1Shape.size() != 4) {
        return matchFailed(_log, rewriter, addOp, "Input to AddOp is not 4D");
    }
    if (input1Shape.isDynamic()) {
        return matchFailed(_log, rewriter, addOp, "Input shape is dynamic");
    }

    // This is a conservative constraint to make sure that
    // the benefit of faster op calculation is larger than the overhead of more workloads
    // TO-DO: loose the size constraint - Track E#161180
    const auto totalCMXSize = VPU::getTotalCMXSize(addOp);
    const auto totalOperandsSize =
            input1Type.getTotalAllocSize() * 2 + outputType.getTotalAllocSize();  // 2 inputs + 1 output
    if (totalOperandsSize <= totalCMXSize) {
        return matchFailed(_log, rewriter, addOp, "Size is not large enough for reshape benefit");
    }

    // Check if the height dimension is greater than the number of clusters
    if (input1Shape[Dims4D::Act::H] > _numClusters) {
        return matchFailed(_log, rewriter, addOp, "Small H dim not found");
    }

    VPUX_THROW_UNLESS(input1Shape[Dims4D::Act::C] % VPU::NCEInvariant::VPU_CHANNEL_ALIGNMENT == 0,
                      "Input channels '{0}' is not aligned by '{1}'", input1Shape[Dims4D::Act::C],
                      VPU::NCEInvariant::VPU_CHANNEL_ALIGNMENT);
    auto channelAlignmentFactor = input1Shape[Dims4D::Act::C] / VPU::NCEInvariant::VPU_CHANNEL_ALIGNMENT;
    bool canReshapeC = allowsChannelsReshape(addOp) && channelAlignmentFactor > 1;

    Shape newInputShape(input1Shape.size());
    newInputShape[Dims4D::Act::N] = input1Shape[Dims4D::Act::N];
    newInputShape[Dims4D::Act::C] = input1Shape[Dims4D::Act::C];

    // Make H as large as possible for better tiling while keeping W >= VPU_SPATIAL_ALIGNMENT
    auto totalHW = input1Shape[Dims4D::Act::H] * input1Shape[Dims4D::Act::W];
    auto factors = vpux::getFactorsListWithMinLimit(totalHW, VPU::NCEInvariant::VPU_SPATIAL_ALIGNMENT);
    if (factors.empty()) {
        // Cannot meet the minimum alignment requirement when only reshaping H and W
        // Try also reshaping C if possible
        if (!canReshapeC) {
            return matchFailed(_log, rewriter, addOp, "No factors found for Input Shape H*W with min limit {0}",
                               VPU::NCEInvariant::VPU_SPATIAL_ALIGNMENT);
        }

        newInputShape[Dims4D::Act::C] = VPU::NCEInvariant::VPU_CHANNEL_ALIGNMENT;
        totalHW = input1Shape[Dims4D::Act::H] * input1Shape[Dims4D::Act::W] * channelAlignmentFactor;
        factors = vpux::getFactorsListWithMinLimit(totalHW, VPU::NCEInvariant::VPU_SPATIAL_ALIGNMENT);
        if (factors.empty()) {
            return matchFailed(_log, rewriter, addOp,
                               "No factors found for Input Shape H*W*channelAlignmentFactor with min limit {0}",
                               VPU::NCEInvariant::VPU_SPATIAL_ALIGNMENT);
        }
    }
    newInputShape[Dims4D::Act::H] = factors[0].second;
    newInputShape[Dims4D::Act::W] = factors[0].first;

    auto newInputShapeAttr = getIntArrayAttr(getContext(), newInputShape);

    // Reshape Inputs
    auto input1ShapeCastOp = rewriter.create<IE::ShapeCastOp>(addOp.getLoc(), input1, newInputShapeAttr);
    auto input2ShapeCastOp = rewriter.create<IE::ShapeCastOp>(addOp.getLoc(), input2, newInputShapeAttr);

    auto newOutputType = outputType.changeShape(newInputShape);

    // Create a new AddOp with the reshaped inputs
    auto newAddOp = rewriter.create<IE::AddOp>(
            addOp->getLoc(), newOutputType, input1ShapeCastOp, input2ShapeCastOp, addOp.getAutoBroadcastAttr(),
            addOp.getPostOpAttr(), addOp.getClampAttr(), addOp.getOutputPaddingAttr(), addOp.getInputPaddingAttr());

    _log.trace("Found AddOp with small H dim: Reshaped to new shape at location '{0}'", newAddOp->getLoc());
    _log.nest().trace("Original Input shape: {0}", input1Shape);
    _log.nest().trace("New Input shape: {0}", newInputShape);

    auto restoreShapeOp =
            rewriter.create<IE::ShapeCastOp>(newAddOp.getLoc(), newAddOp, getIntArrayAttr(getContext(), input1Shape));
    rewriter.replaceOp(addOp, restoreShapeOp.getResult());

    return mlir::success();
}

class HandleEltwiseWithSmallHeightPass final :
        public IE::impl::HandleEltwiseWithSmallHeightBase<HandleEltwiseWithSmallHeightPass> {
public:
    explicit HandleEltwiseWithSmallHeightPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};
void HandleEltwiseWithSmallHeightPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();
    auto numClusters = config::getTileExecutor(func).getCount();
    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<HandleEltwiseWithSmallHeight>(&ctx, numClusters, _log);
    if (mlir::failed(mlir::applyPatternsGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}
}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createHandleEltwiseWithSmallHeightPass(Logger log) {
    return std::make_unique<HandleEltwiseWithSmallHeightPass>(log);
}
