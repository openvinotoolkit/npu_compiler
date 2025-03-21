//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU40XX/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/NPU40XX/utils.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/IR/PatternMatch.h>

namespace vpux::VPU::arch40xx {
#define GEN_PASS_DECL_MOVECONVERTAROUNDVIEWLIKEOPS
#define GEN_PASS_DEF_MOVECONVERTAROUNDVIEWLIKEOPS
#include "vpux/compiler/NPU40XX/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU::arch40xx

using namespace vpux;

namespace {

template <class ViewLikeOp>
class MoveConvertAfterOperation : public mlir::OpRewritePattern<ViewLikeOp> {
public:
    MoveConvertAfterOperation(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<ViewLikeOp>(ctx), _log(log) {
    }

    mlir::LogicalResult matchAndRewrite(ViewLikeOp originOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

template <class ViewLikeOp>
mlir::LogicalResult MoveConvertAfterOperation<ViewLikeOp>::matchAndRewrite(ViewLikeOp originOp,
                                                                           mlir::PatternRewriter& rewriter) const {
    _log.trace("Got '{0}' at '{1}'", originOp->getName(), originOp->getLoc());
    auto nestedLogger = _log.nest();
    auto convertOp = originOp->getOperand(0).template getDefiningOp<VPU::ConvertOp>();
    if (convertOp == nullptr) {
        nestedLogger.trace("Did not find input to be ConvertOp", originOp->getLoc());
        return mlir::failure();
    }

    if (!convertOp->hasOneUse()) {
        nestedLogger.trace("ConvertOp has more than 1 users", convertOp->getLoc());
        return mlir::failure();
    }

    if (!isConvertSupportedOnDMA<VPU::ConvertOp>(convertOp)) {
        nestedLogger.trace("ConvertOp not supported on DMA only FP32->BF16/F16 is supported", originOp->getLoc());
        return mlir::failure();
    }

    // Move ConvertOp after ViewLikeOp so we can later fuse Copy and convertDMAOp
    auto newViewLikeOp = rewriter.clone(*originOp);
    auto result = newViewLikeOp->getResult(0);
    newViewLikeOp->setOperand(0, convertOp.getInput());

    auto newOpResultType = result.getType().template cast<vpux::NDTypeInterface>();
    auto inputType = convertOp.getInput().getType().template cast<vpux::NDTypeInterface>();
    result.setType(newOpResultType.changeElemType(inputType.getElementType()));

    auto originOpType = originOp->getResult(0).getType().template cast<vpux::NDTypeInterface>();
    auto newConvert = rewriter.replaceOpWithNewOp<VPU::ConvertOp>(originOp, result, convertOp.getDstElemTypeAttr());
    newConvert->getResult(0).setType(originOpType);

    return mlir::success();
}

//
// MoveConvertBeforeAffineReshape
//
// Move the ConvertOp before AffineReshape
// AffineReshape              ConvertOp
//      |                         |
//   ConvertOp       ->      Affine Reshape
class MoveConvertBeforeAffineReshape final : public mlir::OpRewritePattern<VPU::ConvertOp> {
public:
    MoveConvertBeforeAffineReshape(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPU::ConvertOp>(ctx), _log(log) {
        this->setDebugName("MoveConvertAroundViewLikeOpsPass::MoveConvertBeforeAffineReshape");
    }

private:
    mlir::LogicalResult matchAndRewrite(VPU::ConvertOp originOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult MoveConvertBeforeAffineReshape::matchAndRewrite(VPU::ConvertOp originOp,
                                                                    mlir::PatternRewriter& rewriter) const {
    _log.trace("Got '{0}' at '{1}'", originOp->getName(), originOp->getLoc());
    auto nestedLogger = _log.nest();
    auto affineReshapeOp = originOp.getInput().getDefiningOp<VPU::AffineReshapeOp>();
    if (affineReshapeOp == nullptr) {
        nestedLogger.trace("ConvertOp does not have AffineReshape input {0}", originOp->getName());
        return mlir::failure();
    }

    auto affineReshapeOutputType = affineReshapeOp.getOutput().getType().cast<vpux::NDTypeInterface>();
    if (affineReshapeOutputType.getShape().size() == 4) {
        nestedLogger.trace("AffineReshape output is already 4D {0}", affineReshapeOp->getName());
        return mlir::failure();
    }

    // If the AffineReshape input is not 4D then this movement is useless
    auto affineReshapeInputType = affineReshapeOp.getInput().getType().cast<vpux::NDTypeInterface>();
    if (affineReshapeInputType.getShape().size() != 4) {
        nestedLogger.trace("AffineReshape input is not 4D {0}", affineReshapeOp->getName());
        return mlir::failure();
    }
    // Move ConvertOp before AffineReshape so we can wrap ConvertOp in NCEClusterTiling with all 4 Dims
    auto originOpType = originOp->getResult(0).getType().cast<vpux::NDTypeInterface>();
    auto newConvertOp = rewriter.create<VPU::ConvertOp>(originOp.getLoc(), affineReshapeOp.getInput(),
                                                        originOp.getDstElemTypeAttr());
    auto newAffineReshape = rewriter.replaceOpWithNewOp<VPU::AffineReshapeOp>(originOp, newConvertOp->getResult(0),
                                                                              affineReshapeOp.getDimMappingAttr(),
                                                                              affineReshapeOp.getShapeValueAttr());

    newAffineReshape->getResult(0).setType(originOpType);

    return mlir::success();
}

//
// MoveConvertAroundViewLikeOpsPass
//
class MoveConvertAroundViewLikeOpsPass final :
        public VPU::arch40xx::impl::MoveConvertAroundViewLikeOpsBase<MoveConvertAroundViewLikeOpsPass> {
public:
    explicit MoveConvertAroundViewLikeOpsPass(Logger log): _log(log) {
        _log.setName(Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

void MoveConvertAroundViewLikeOpsPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<MoveConvertAfterOperation<VPU::PermuteCastOp>>(&ctx, _log.nest());
    patterns.add<MoveConvertAfterOperation<VPU::ShapeCastOp>>(&ctx, _log.nest());
    patterns.add<MoveConvertBeforeAffineReshape>(&ctx, _log.nest());

    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createMoveConvertAroundViewLikeOpsPass
//
std::unique_ptr<mlir::Pass> vpux::VPU::arch40xx::createMoveConvertAroundViewLikeOpsPass(Logger log) {
    return std::make_unique<MoveConvertAroundViewLikeOpsPass>(log);
}
