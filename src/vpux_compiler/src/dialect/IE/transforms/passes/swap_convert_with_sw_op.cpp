//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/image.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/interpolate_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"

#include "vpux/compiler/dialect/IE/utils/const_attributes.hpp"

#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/IR/Operation.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/DialectConversion.h>

namespace vpux::IE {
#define GEN_PASS_DECL_SWAPCONVERTWITHSWOP
#define GEN_PASS_DEF_SWAPCONVERTWITHSWOP
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

// TODO: Adding limitation, noting that small tensors are not that optimal to fuse in NCE
constexpr int64_t EXPERIMENTAL_F32_FUSION_THRESHOLD = 36000;

//
// SwapSWOpWithConvert
//

class SwapSWOpWithConvert final : public mlir::OpRewritePattern<IE::ConvertOp> {
public:
    SwapSWOpWithConvert(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::ConvertOp>(ctx), _log(log) {
        this->setDebugName("SwapSWOpWithConvert");
    }

private:
    mlir::LogicalResult matchAndRewrite(IE::ConvertOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

bool isReshapeKindOp(mlir::Operation* op) {
    return mlir::isa_and_nonnull<IE::TransposeOp, IE::ReshapeOp, IE::AffineReshapeOp>(op);
}

mlir::LogicalResult SwapSWOpWithConvert::matchAndRewrite(IE::ConvertOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    const auto convertInput = origOp.getInput();

    mlir::Operation* nceOp = convertInput.getDefiningOp();
    while (isReshapeKindOp(nceOp)) {
        nceOp = nceOp->getOperand(0).getDefiningOp();
    }

    rewriter.setInsertionPointAfter(nceOp);
    auto newConvert = rewriter.create<IE::ConvertOp>(nceOp->getLoc(), nceOp->getResult(0), origOp.getDstElemType());

    nceOp->getResult(0).replaceAllUsesExcept(newConvert.getOutput(),
                                             llvm::SmallPtrSet<mlir::Operation*, 1>{newConvert});

    origOp->replaceAllUsesWith(mlir::ValueRange(convertInput));
    rewriter.eraseOp(origOp);

    mlir::Operation* lastOp = *newConvert.getOutput().getUsers().begin();
    while (isReshapeKindOp(lastOp)) {
        vpux::inferReturnTypes(lastOp, vpux::InferShapedTypeMode::ALL);
        lastOp = *lastOp->getResult(0).getUsers().begin();
    }

    return mlir::success();
}

//
// SwapConvertWithEltwiseOp
//

template <typename EltwiseOp>
class SwapConvertWithEltwiseOp final : public mlir::OpRewritePattern<EltwiseOp> {
public:
    SwapConvertWithEltwiseOp(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<EltwiseOp>(ctx), _log(log) {
        this->setDebugName("SwapConvertWithEltwiseOp");
    }

public:
    mlir::LogicalResult matchAndRewrite(EltwiseOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

// If pattern like EltwiseOp[si32] -> ConvertOp[si32, fp16], since EltwiseOp with IntegerType cannot convert to DPU
// task, if we swap EltwiseOp with ConvertOp, the EltwiseOp will be converted to DPU task.

/* Rewrite the pattern from:
   Input       Const
     |          |
      \        /
        Eltwise (IntegerType, will not be converted to DPU task)
           |
        ConvertOp (IntegerType to Float16Type)

    to:
   Input                                Const (CastElemType to Float16Type)
      |                                   |
  ConvertOp (IntegerType to Float16Type)  |
       \                                 /
                EltwiseOp (Float16Type, will be converted to DPU task)
 */
template <typename EltwiseOp>
mlir::LogicalResult SwapConvertWithEltwiseOp<EltwiseOp>::matchAndRewrite(EltwiseOp origOp,
                                                                         mlir::PatternRewriter& rewriter) const {
    _log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    if (!origOp.getOutput().hasOneUse()) {
        return mlir::failure();
    }

    auto convertOp = mlir::dyn_cast<IE::ConvertOp>(*origOp.getOutput().getUsers().begin());
    if (convertOp == nullptr) {
        return mlir::failure();
    }

    auto convertInElemType = mlir::cast<NDTypeInterface>(convertOp.getInput().getType()).getElementType();
    auto convertOutElemType = mlir::cast<NDTypeInterface>(convertOp.getOutput().getType()).getElementType();
    if (!mlir::isa<mlir::IntegerType>(convertInElemType) || !mlir::isa<mlir::Float16Type>(convertOutElemType)) {
        return mlir::failure();
    }

    if (mlir::failed(IE::getConstParentOp(origOp.getInput2()))) {
        return mlir::failure();
    }

    // Experimental number to determine if swapping ConvertOp with EltwiseOp is beneficial.
    constexpr int BENEFICIAL_SIZE = 1024;
    auto shapeSize = vpux::details::calcTotalShapeSize(getShape(origOp.getOutput()));
    if (shapeSize < BENEFICIAL_SIZE) {
        return mlir::failure();
    }

    auto newConvert = rewriter.create<IE::ConvertOp>(convertOp->getLoc(), origOp.getInput1(), convertOutElemType);

    auto constInput = origOp.getInput2().template getDefiningOp<Const::DeclareOp>();
    auto biasContentAttr = constInput.transformContentAttr().castElemType(convertOutElemType).get();
    auto newBiasValue =
            rewriter.create<Const::DeclareOp>(origOp.getLoc(), biasContentAttr.getType(), std::move(biasContentAttr))
                    .getResult();
    mlir::IRMapping mapper;
    mapper.map(origOp->getOperands(), SmallVector<mlir::Value>{newConvert.getOutput(), newBiasValue});
    auto* newOp = rewriter.clone(*origOp, mapper);
    vpux::inferReturnTypes(newOp, vpux::InferShapedTypeMode::ALL);

    convertOp.replaceAllUsesWith(newOp->getResult(0));

    return mlir::success();
}

//
// SwapConvertWithSWOp
//

class SwapConvertWithSWOp final : public IE::impl::SwapConvertWithSWOpBase<SwapConvertWithSWOp> {
public:
    explicit SwapConvertWithSWOp(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void SwapConvertWithSWOp::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();
    const auto isLegalOp = [](IE::ConvertOp op) -> bool {
        auto inputElemType = mlir::cast<NDTypeInterface>(op.getInput().getType()).getElementType();
        auto outputElemType = mlir::cast<NDTypeInterface>(op.getOutput().getType()).getElementType();

        auto outShape = getBoundedShape(op.getOutput());
        if (outShape.totalSize() < EXPERIMENTAL_F32_FUSION_THRESHOLD) {
            return true;
        }

        if (!mlir::isa<mlir::Float16Type>(inputElemType) || !mlir::isa<mlir::Float32Type>(outputElemType)) {
            return true;
        }

        if (!op->hasOneUse()) {
            return true;
        }

        mlir::Operation* parentOp = op.getInput().getDefiningOp();
        if (!isReshapeKindOp(parentOp)) {
            return true;
        }
        while (isReshapeKindOp(parentOp)) {
            if (!parentOp->getResult(0).hasOneUse()) {
                return true;
            }
            parentOp = parentOp->getOperand(0).getDefiningOp();
        }

        if (parentOp == nullptr || !parentOp->getResult(0).hasOneUse()) {
            return true;
        }

        auto convertOutType = op.getOutput().getType().getElementType();
        if (auto interpolateOp = mlir::dyn_cast<IE::InterpolateOp>(parentOp)) {
            // Check if it's beneficial to fuse our ConvertOp into InterpolateOp
            return !IE::isFusingConvertIntoBilinearInterpolateOnDpuBeneficial(interpolateOp, convertOutType);
        }

        if (mlir::failed(VPU::NCEInvariant::isSupported(parentOp))) {
            return true;
        }

        const auto inputShape = getBoundedShape(parentOp->getOperand(0));
        // This will cause an error, because of EnsureNCEOpsSizeRequirementsPass.
        return inputShape[Dims4D::Act::C] > VPU::NCEInvariant::VPU_DIMENSION_LIMIT;
    };

    mlir::ConversionTarget target(ctx);
    target.addDynamicallyLegalOp<IE::ConvertOp>(isLegalOp);

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<SwapSWOpWithConvert>(&ctx, _log);
    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }

    mlir::RewritePatternSet eltwisePatterns(&ctx);
    eltwisePatterns.add<SwapConvertWithEltwiseOp<IE::MultiplyOp>>(&ctx, _log);
    eltwisePatterns.add<SwapConvertWithEltwiseOp<IE::AddOp>>(&ctx, _log);
    eltwisePatterns.add<SwapConvertWithEltwiseOp<IE::SubtractOp>>(&ctx, _log);
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(eltwisePatterns),
                                                        getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
        return;
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createSwapConvertWithSWOpPass(Logger log) {
    return std::make_unique<SwapConvertWithSWOp>(log);
}
