//
// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU40XX/utils.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/IR/PatternMatch.h>

namespace vpux::VPU {
#define GEN_PASS_DECL_MOVECONVERTAROUNDVIEWLIKEOPS
#define GEN_PASS_DEF_MOVECONVERTAROUNDVIEWLIKEOPS
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

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

    auto newOpResultType = mlir::cast<vpux::NDTypeInterface>(result.getType());
    auto inputType = mlir::cast<vpux::NDTypeInterface>(convertOp.getInput().getType());
    result.setType(newOpResultType.changeElemType(inputType.getElementType()));

    auto originOpType = mlir::cast<vpux::NDTypeInterface>(originOp->getResult(0).getType());
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

    auto affineReshapeOutputType = mlir::cast<vpux::NDTypeInterface>(affineReshapeOp.getOutput().getType());
    if (affineReshapeOutputType.getShape().size() == 4) {
        nestedLogger.trace("AffineReshape output is already 4D {0}", affineReshapeOp->getName());
        return mlir::failure();
    }

    // If the AffineReshape input is not 4D then this movement is useless
    auto affineReshapeInputType = mlir::cast<vpux::NDTypeInterface>(affineReshapeOp.getInput().getType());
    if (affineReshapeInputType.getShape().size() != 4) {
        nestedLogger.trace("AffineReshape input is not 4D {0}", affineReshapeOp->getName());
        return mlir::failure();
    }
    auto originOpType = mlir::cast<vpux::NDTypeInterface>(originOp->getResult(0).getType());
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
        public VPU::impl::MoveConvertAroundViewLikeOpsBase<MoveConvertAroundViewLikeOpsPass> {
public:
    explicit MoveConvertAroundViewLikeOpsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
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
std::unique_ptr<mlir::Pass> vpux::VPU::createMoveConvertAroundViewLikeOpsPass(Logger log) {
    return std::make_unique<MoveConvertAroundViewLikeOpsPass>(log);
}
