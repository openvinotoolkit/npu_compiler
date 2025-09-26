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
Eltwise-like ops only support SOH and Clustering for MC strategy split.

Convert Subgraph
1x3x1080x1920     1x3x1080x1920
        \           /
        Add (Clustering)
            |
          Convert

To -
AFReshape         AFReshape
1x192x1080x30     1x192x1080x30
        \           /
           Add (SOH)
             |
         AFReshape
        1x3x1080x1920
             |
          Convert
*/

mlir::LogicalResult HandleEltwiseWithSmallHeight::matchAndRewrite(IE::AddOp addOp,
                                                                  mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), addOp->getName(), addOp->getLoc());

    auto isLastOp = addOp->getResult(0).getUsers().empty();
    if (isLastOp) {
        return matchFailed(_log, rewriter, addOp, "No users for AddOp found");
    }

    // TO-DO remove subgraph constrain - Track E#161180
    const auto outputConvertOp = mlir::dyn_cast<IE::ConvertOp>(*(addOp->getResult(0).user_begin()));
    if (outputConvertOp == nullptr) {
        return matchFailed(_log, rewriter, addOp, "Required subgraph not found");
    }

    auto input1 = addOp.getInput1();
    auto input2 = addOp.getInput2();

    if (mlir::isa_and_nonnull<IE::AffineReshapeOp>(input1.getDefiningOp()) ||
        mlir::isa_and_nonnull<IE::AffineReshapeOp>(input2.getDefiningOp())) {
        return matchFailed(_log, rewriter, addOp, "Input Operands are null or already an AffineReshape op");
    }

    auto input1Type = mlir::dyn_cast<vpux::NDTypeInterface>(input1.getType());
    auto input2Type = mlir::dyn_cast<vpux::NDTypeInterface>(input2.getType());

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

    // Check if the height dimension is greater than the number of clusters
    if (input1Shape[Dims4D::Act::H] > _numClusters) {
        return matchFailed(_log, rewriter, addOp, "Small H dim not found");
    }

    Shape newInputShape(input1Shape.size());

    auto channelAlignmentFactor = _numClusters * VPU::NCEInvariant::VPU_CHANNEL_ALIGNMENT;
    auto newInputChannel = input1Shape[Dims4D::Act::C] / channelAlignmentFactor;

    auto isChannelAlignedInput = input1Shape[Dims4D::Act::C] % channelAlignmentFactor == 0;
    auto isChannelAlignedOutput =
            newInputChannel > 0 && newInputChannel % VPU::NCEInvariant::VPU_CHANNEL_ALIGNMENT == 0;

    if (isChannelAlignedInput && isChannelAlignedOutput) {
        newInputShape[Dims4D::Act::N] = input1Shape[Dims4D::Act::N];
        newInputShape[Dims4D::Act::C] = newInputChannel;
        newInputShape[Dims4D::Act::H] = input1Shape[Dims4D::Act::H] * channelAlignmentFactor;
        newInputShape[Dims4D::Act::W] = input1Shape[Dims4D::Act::W];

    } else if ((input1Shape[Dims4D::Act::H] * input1Shape[Dims4D::Act::W]) % _numClusters == 0) {
        const auto factors = vpux::getFactorsList(input1Shape[Dims4D::Act::H] * input1Shape[Dims4D::Act::W]);
        if (factors.empty()) {
            return matchFailed(_log, rewriter, addOp, "No factors found for Input Shape H*W");
        }

        auto newHeight = factors.back().first;
        auto newWidth = factors.back().second;

        newInputShape[Dims4D::Act::N] = input1Shape[Dims4D::Act::N];
        newInputShape[Dims4D::Act::C] = input1Shape[Dims4D::Act::C];
        newInputShape[Dims4D::Act::H] = newHeight;
        newInputShape[Dims4D::Act::W] = newWidth;

    } else {
        return matchFailed(_log, rewriter, addOp, "Dimensions not aligned for Reshape Ops");
    }

    auto newInputShapeAttr = getIntArrayAttr(getContext(), newInputShape);

    SmallVector<SmallVector<int64_t>> reassociationMap(newInputShape.size());

    for (size_t dimIdx = 0; dimIdx < newInputShape.size(); dimIdx++) {
        reassociationMap[dimIdx].push_back(dimIdx);
    }

    auto newInputDimAttr = getIntArrayOfArray(getContext(), reassociationMap);
    // Reshape Inputs
    auto input1ShapeCastOp =
            rewriter.create<IE::AffineReshapeOp>(addOp.getLoc(), input1, newInputDimAttr, newInputShapeAttr);
    auto input2ShapeCastOp =
            rewriter.create<IE::AffineReshapeOp>(addOp.getLoc(), input2, newInputDimAttr, newInputShapeAttr);

    auto newOutputType = mlir::cast<vpux::NDTypeInterface>(addOp.getResult().getType()).changeShape(newInputShape);

    // Create a new AddOp with the reshaped inputs
    auto newAddOp = rewriter.create<IE::AddOp>(
            addOp->getLoc(), newOutputType, input1ShapeCastOp, input2ShapeCastOp, addOp.getAutoBroadcastAttr(),
            addOp.getPostOpAttr(), addOp.getClampAttr(), addOp.getOutputPaddingAttr(), addOp.getInputPaddingAttr());

    _log.trace("Found AddOp with small H dim: Reshaped to new shape at location '{0}'", newAddOp->getLoc());

    auto restoreShapeOp = rewriter.create<IE::AffineReshapeOp>(newAddOp.getLoc(), newAddOp, newInputDimAttr,
                                                               getIntArrayAttr(getContext(), input1Shape));
    rewriter.replaceOp(addOp, restoreShapeOp.getResult());

    return mlir::success();
}

class HandleEltwiseWithSmallHeightPass final :
        public IE::impl::HandleEltwiseWithSmallHeightBase<HandleEltwiseWithSmallHeightPass> {
public:
    explicit HandleEltwiseWithSmallHeightPass(Logger log): _log(log) {
        _log.setName(Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};
void HandleEltwiseWithSmallHeightPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();
    auto numClusters = config::getTileExecutor(func).getCount();
    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<HandleEltwiseWithSmallHeight>(&ctx, numClusters, _log);
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}
}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createHandleEltwiseWithSmallHeightPass(Logger log) {
    return std::make_unique<HandleEltwiseWithSmallHeightPass>(log);
}
