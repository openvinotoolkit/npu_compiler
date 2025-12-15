//
// Copyright (C) 2024-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/dpu.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"

#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/walk_utils.hpp"

#include <llvm/ADT/SetOperations.h>
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace vpux::VPU {
#define GEN_PASS_DECL_ADJUSTDISTRIBUTEDTENSORAROUNDOPS
#define GEN_PASS_DEF_ADJUSTDISTRIBUTEDTENSORAROUNDOPS
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;
using namespace VPU;

namespace {

//
// DistributedInputTypeRewriter
//
class DistributedInputTypeRewriter final : public mlir::OpInterfaceRewritePattern<VPU::NCEOpInterface> {
public:
    DistributedInputTypeRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpInterfaceRewritePattern<VPU::NCEOpInterface>(ctx), _log(log) {
        this->setDebugName("DistributedInputTypeRewriter");
    }

private:
    mlir::LogicalResult matchAndRewrite(VPU::NCEOpInterface, mlir::PatternRewriter& rewriter) const final;

    bool fitIntoCMX(VPU::NCEOpInterface origOp, VPU::DistributedTensorType newInType) const;

private:
    Logger _log;
};

mlir::LogicalResult DistributedInputTypeRewriter::matchAndRewrite(VPU::NCEOpInterface origOp,
                                                                  mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp.getLoc());
    /*
       Convert subgraph below when percluster memory shape of DistributedType1 is included in percluster memory shape
       of DistributedType0

            DistributedType0       DistributedType0
                  |                       |
                Copy                    Copy
                  |                       |
                Copy            =>      Copy
                  |                       |
            DistributedType1       DistributedType0
                  |                       |
                 NCE                     NCE
    */
    if (auto eltwiseOp = mlir::dyn_cast<VPU::NCEEltwiseOp>(origOp.getOperation())) {
        if (eltwiseOp.getIsInplace().value_or(false)) {
            _log.trace("Skip for inplace case since the change will affect the output type");
            return mlir::failure();
        }
    }

    auto input = origOp->getOperand(0);
    auto distributedInType = mlir::dyn_cast<vpux::VPU::DistributedTensorType>(input.getType());
    if (distributedInType == nullptr) {
        return matchFailed(_log, rewriter, origOp, "Input is not distributed tensor type at '{0}'", origOp->getLoc());
    }
    auto inMode = distributedInType.getDistribution().getMode().getValue();
    if (inMode != VPU::DistributionMode::OVERLAPPED) {
        return matchFailed(_log, rewriter, origOp, "Input distributed tensor type is not OVERLAPPED at '{0}'",
                           origOp->getLoc());
    }

    auto inCopy = input.getDefiningOp<VPU::CopyOp>();
    if (inCopy == nullptr) {
        return matchFailed(_log, rewriter, origOp, "Input is not from copy op at '{0}'", origOp->getLoc());
    }

    const auto tilingScheme = vpux::parseIntArrayAttr<int64_t>(distributedInType.getDistribution().getNumTiles());

    auto parentCopy = inCopy.getInput().getDefiningOp<VPU::CopyOp>();
    if (parentCopy == nullptr) {
        return matchFailed(_log, rewriter, inCopy, "parent is not copy op at '{0}'", inCopy->getLoc());
    }
    auto parentDistributedInType = mlir::dyn_cast<vpux::VPU::DistributedTensorType>(parentCopy.getInput().getType());
    if (parentDistributedInType == nullptr) {
        return matchFailed(_log, rewriter, parentCopy,
                           "Input type of parent copy op is not distributed tensor type at '{0}'",
                           parentCopy->getLoc());
    }

    if (mlir::succeeded(VPU::isDistributedCastCompatible(parentDistributedInType, distributedInType))) {
        return matchFailed(_log, rewriter, inCopy, "Copy op types are compatible for optimization at '{0}'",
                           inCopy->getLoc());
    }
    auto parentMode = parentDistributedInType.getDistribution().getMode().getValue();
    if (parentMode != inMode) {
        return matchFailed(_log, rewriter, parentCopy, "Input distributed tensor type is not OVERLAPPED at '{0}'",
                           parentCopy->getLoc());
    }

    const auto parentTilingScheme =
            vpux::parseIntArrayAttr<int64_t>(parentDistributedInType.getDistribution().getNumTiles());

    if (tilingScheme != parentTilingScheme) {
        return matchFailed(_log, rewriter, origOp, "Tiling Scheme are different for {0} output and {1} input",
                           inCopy->getLoc(), parentCopy->getLoc());
    }

    auto perClusterMemShapes = distributedInType.getPerClusterMemoryShapes();
    auto perClusterMemShapeOffsets = distributedInType.getPerClusterMemoryShapeOffsets();
    auto parentPerClusterMemShapes = parentDistributedInType.getPerClusterMemoryShapes();
    auto parentPerClusterMemShapeOffsets = parentDistributedInType.getPerClusterMemoryShapeOffsets();

    // Check if the memory shapes are included in parent's memory shapes
    for (auto idx : irange(perClusterMemShapes.size())) {
        const auto currentMemShape = to_small_vector(perClusterMemShapes[idx]);
        const auto parentMemShape = to_small_vector(parentPerClusterMemShapes[idx]);
        const auto currentMemShapeOffset = to_small_vector(perClusterMemShapeOffsets[idx]);
        const auto parentMemShapeOffset = to_small_vector(parentPerClusterMemShapeOffsets[idx]);

        for (size_t dim = 0; dim < perClusterMemShapes.front().size(); dim++) {
            if (tilingScheme[dim] != 1) {
                if (currentMemShapeOffset[dim] < parentMemShapeOffset[dim] ||
                    currentMemShapeOffset[dim] + currentMemShape[dim] >
                            parentMemShapeOffset[dim] + parentMemShape[dim]) {
                    _log.trace("Memory shape {0} is not included in parent memory shape {1} at '{2}'", currentMemShape,
                               parentMemShape, origOp->getLoc());
                    return mlir::failure();
                }
            } else {
                if (currentMemShapeOffset[dim] != parentMemShapeOffset[dim] ||
                    currentMemShape[dim] != parentMemShape[dim]) {
                    _log.trace("Memory shape {0} is not included in parent memory shape {1} at '{2}'", currentMemShape,
                               parentMemShape, origOp->getLoc());
                    return mlir::failure();
                }
            }
        }
    }
    if (!fitIntoCMX(origOp, parentDistributedInType)) {
        _log.trace("Can not fit into cmx with new input type");
        return mlir::failure();
    }

    _log.trace("Update distributed type {0} to {1} at '{2}'", distributedInType, parentDistributedInType,
               origOp->getLoc());

    rewriter.startOpModification(inCopy);
    inCopy.getResult().setType(parentDistributedInType);
    rewriter.finalizeOpModification(inCopy);

    rewriter.startOpModification(origOp);
    input.setType(parentDistributedInType);
    rewriter.finalizeOpModification(origOp);

    return mlir::success();
}

bool DistributedInputTypeRewriter::fitIntoCMX(VPU::NCEOpInterface origOp, VPU::DistributedTensorType newInType) const {
    return llvm::TypeSwitch<mlir::Operation*, bool>(origOp.getOperation())
            .Case<VPU::NCEConvolutionOp, VPU::NCECompressConvolutionOp, VPU::NCEDepthConvolutionOp>([&](auto convOp) {
                auto filterType = convOp.getFilter().getType();
                auto outputType = convOp.getOutput().getType();
                return convOp.fitIntoCMX(newInType, filterType, outputType);
            })
            .Case<VPU::NCEInterpolateOp, VPU::NCEMatMulOp>([&](auto op) {
                auto filterType = op.getWeights().getType();
                auto outputType = op.getOutput().getType();
                return op.fitIntoCMX(newInType, filterType, outputType);
            })
            .Case<VPU::NCEMaxPoolOp, VPU::NCEAveragePoolOp, VPU::NCEPermuteOp>([&](auto op) {
                auto outputType = op.getOutput().getType();
                return op.fitIntoCMX(newInType, outputType);
            })
            .Case<VPU::NCEEltwiseOp>([&](auto eltwiseOp) {
                auto input2Type = eltwiseOp.getInput2().getType();
                auto outputType = eltwiseOp.getOutput().getType();
                return eltwiseOp.fitIntoCMX(newInType, input2Type, outputType);
            })
            .Default([&](mlir::Operation* op) {
                _log.trace("Unsupported op type at {0}", op->getLoc());
                return false;
            });
}

//
// AdjustDistributedTensorAroundOpsPass
//

class AdjustDistributedTensorAroundOpsPass final :
        public VPU::impl::AdjustDistributedTensorAroundOpsBase<AdjustDistributedTensorAroundOpsPass> {
public:
    explicit AdjustDistributedTensorAroundOpsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() override;

private:
};

void AdjustDistributedTensorAroundOpsPass::safeRunOnFunc() {
    auto func = getOperation();
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<DistributedInputTypeRewriter>(&ctx, _log);
    collectOpsAndApplyPatterns(func, std::move(patterns));
}

}  // namespace

//
// createAdjustDistributedTensorAroundOpsPass
//

std::unique_ptr<mlir::Pass> vpux::VPU::createAdjustDistributedTensorAroundOpsPass(Logger log) {
    return std::make_unique<AdjustDistributedTensorAroundOpsPass>(log);
}
