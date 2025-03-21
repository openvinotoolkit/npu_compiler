//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

namespace vpux::IE {
#define GEN_PASS_DECL_PROPAGATESHAPECAST
#define GEN_PASS_DEF_PROPAGATESHAPECAST
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
// MoveThroughActivationOp
//

template <class ConcreteOp>
class MoveThroughActivationOp final : public mlir::OpRewritePattern<IE::ShapeCastOp> {
public:
    MoveThroughActivationOp(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::ShapeCastOp>(ctx), _log(log) {
        this->setDebugName("MoveThroughActivationOp");
    }

private:
    mlir::LogicalResult matchAndRewrite(IE::ShapeCastOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

template <class ConcreteOp>
mlir::LogicalResult MoveThroughActivationOp<ConcreteOp>::matchAndRewrite(IE::ShapeCastOp origOp,
                                                                         mlir::PatternRewriter& rewriter) const {
    _log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());
    auto concreteOp = origOp.getOperand().getDefiningOp<ConcreteOp>();

    if (concreteOp == nullptr || !concreteOp->hasOneUse() || concreteOp->getNumOperands() != 1) {
        return matchFailed(_log, rewriter, origOp, "ConcreteOp not found or has multiple producers or uses");
    }

    auto preShapeCastOp = concreteOp->getOperand(0).template getDefiningOp<IE::ShapeCastOp>();
    if (preShapeCastOp == nullptr) {
        return matchFailed(_log, rewriter, origOp, "Previous ShapeCastOp not found");
    }

    auto newShapeCastOp =
            rewriter.create<IE::ShapeCastOp>(origOp.getLoc(), preShapeCastOp.getResult(), origOp.getShapeAttr());

    mlir::IRMapping mapper;
    mapper.map(concreteOp->getOperand(0), newShapeCastOp.getResult());
    auto newOp = rewriter.clone(*concreteOp, mapper);
    vpux::inferReturnTypes(newOp, vpux::InferShapedTypeMode::SHAPE);

    rewriter.replaceOp(origOp, newOp->getResults());

    _log.trace("[{0}] Replaced with '{1}'", getDebugName(), concreteOp->getLoc());
    return mlir::success();
}

//
// MoveThroughPoolOp
//

template <class PoolingOp>
class MoveThroughPoolOp final : public mlir::OpRewritePattern<IE::ShapeCastOp> {
public:
    MoveThroughPoolOp(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::ShapeCastOp>(ctx), _log(log) {
        this->setDebugName("MoveThroughPoolOp");
    }

private:
    mlir::LogicalResult matchAndRewrite(IE::ShapeCastOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

template <class PoolingOp>
mlir::LogicalResult MoveThroughPoolOp<PoolingOp>::matchAndRewrite(IE::ShapeCastOp origOp,
                                                                  mlir::PatternRewriter& rewriter) const {
    _log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());
    auto poolingOp = origOp.getOperand().getDefiningOp<PoolingOp>();

    if (poolingOp == nullptr || !poolingOp->hasOneUse() || poolingOp->getNumOperands() != 1) {
        return matchFailed(_log, rewriter, origOp, "poolingOp not found or has multiple producers or uses");
    }

    const auto supportedPooling = [&](PoolingOp layerOp) {
        const auto kernels = parseIntArrayAttr<int64_t>(layerOp.getKernelSize());
        const auto padStart = parseIntArrayAttr<int64_t>(layerOp.getPadsBegin());
        const auto padEnd = parseIntArrayAttr<int64_t>(layerOp.getPadsEnd());
        const auto strides = parseIntArrayAttr<int64_t>(layerOp.getStrides());

        mlir::Value input = layerOp.getInput();
        mlir::Value output = layerOp.getOutput();
        auto inputLayout = input.getType().cast<vpux::NDTypeInterface>().getDimsOrder();
        auto outputLayout = output.getType().cast<vpux::NDTypeInterface>().getDimsOrder();
        // input and output layer need to be same
        if (inputLayout != outputLayout) {
            return false;
        }

        auto hasValidKernels = llvm::all_of(kernels, [&](const auto& kernel) {
            return kernel == 1;
        });
        auto hasValidPadStart = llvm::all_of(padStart, [&](const auto& pad) {
            return pad == 0;
        });
        auto hasValidPadEnd = llvm::all_of(padEnd, [&](const auto& pad) {
            return pad == 0;
        });
        auto hasValidStrides = llvm::all_of(strides, [&](const auto& stride) {
            return stride == 1;
        });

        const auto sizeToAlign = std::max(
                VPU::NCEInvariant::getAlignment(input.getType().cast<vpux::NDTypeInterface>().getElementType()),
                VPU::NCEInvariant::getAlignment(output.getType().cast<vpux::NDTypeInterface>().getElementType()));
        const auto origOutShape = getShape(origOp.getResult());

        return hasValidKernels && hasValidPadStart && hasValidPadEnd && hasValidStrides &&
               (origOutShape[Dims4D::Act::C] % sizeToAlign == 0);
    };

    if (!supportedPooling(poolingOp)) {
        return matchFailed(_log, rewriter, origOp, "poolingOp is not eltwise");
    }

    auto preShapeCastOp = poolingOp->getOperand(0).template getDefiningOp<IE::ShapeCastOp>();
    if (preShapeCastOp == nullptr) {
        return matchFailed(_log, rewriter, origOp, "Previous ShapeCastOp not found");
    }

    auto newShapeCastOp =
            rewriter.create<IE::ShapeCastOp>(origOp.getLoc(), preShapeCastOp.getResult(), origOp.getShapeAttr());

    mlir::IRMapping mapper;
    mapper.map(poolingOp->getOperand(0), newShapeCastOp.getResult());
    auto newOp = rewriter.clone(*poolingOp, mapper);
    vpux::inferReturnTypes(newOp, vpux::InferShapedTypeMode::SHAPE);

    rewriter.replaceOp(origOp, newOp->getResults());

    _log.trace("[{0}] Replaced with '{1}'", getDebugName(), poolingOp->getLoc());
    return mlir::success();
}

//
// MoveThroughEltwiseOp
//

template <class EltwiseOp>
class MoveThroughEltwiseOp final : public mlir::OpRewritePattern<IE::ShapeCastOp> {
public:
    MoveThroughEltwiseOp(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::ShapeCastOp>(ctx), _log(log) {
        this->setDebugName("MoveThroughEltwiseOp");
    }

private:
    mlir::LogicalResult matchAndRewrite(IE::ShapeCastOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

template <class EltwiseOp>
mlir::LogicalResult MoveThroughEltwiseOp<EltwiseOp>::matchAndRewrite(IE::ShapeCastOp origOp,
                                                                     mlir::PatternRewriter& rewriter) const {
    _log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());
    auto eltwiseOp = origOp.getOperand().getDefiningOp<EltwiseOp>();

    if (eltwiseOp == nullptr) {
        return matchFailed(_log, rewriter, origOp, "eltwiseOp not found");
    }

    // Get eltwiseOp's operands number and index with Shape{1, 1, 1, 1}
    const auto getOpWithShapeOnes = [&](EltwiseOp layerOp) {
        int64_t getOpNum = 0;
        int64_t getLastOpIdx = 0;
        for (auto operand : layerOp->getOperands() | indexed) {
            const auto index = operand.index();
            const auto operandOutType = layerOp->getOperand(index).getType().template cast<vpux::NDTypeInterface>();
            const auto operandOutShape = operandOutType.getShape();
            const auto expectedShape = Shape{1, 1, 1, 1};
            if (operandOutShape == expectedShape) {
                getOpNum++;
                getLastOpIdx = index;
            }
        }
        return std::make_tuple(getOpNum, getLastOpIdx);
    };

    const auto isOneInput = eltwiseOp->getOperand(0) == eltwiseOp->getOperand(1);
    int64_t getOpNum = 0;
    int64_t getLastOpIdx = 0;
    std::tie(getOpNum, getLastOpIdx) = getOpWithShapeOnes(eltwiseOp);

    // Check for supported eltwise
    const auto supportedEltwise = [&](EltwiseOp layerOp) {
        if (!layerOp->hasOneUse()) {
            return false;
        }

        if (!isOneInput && getOpNum != 1) {
            return false;
        }

        if (VPU::NCEInvariant::isSupported(layerOp).succeeded()) {
            return false;
        }

        // Could not support per channel quantize type
        const auto input1ElementType =
                layerOp->getOperand(0).getType().template cast<vpux::NDTypeInterface>().getElementType();
        const auto input2ElementType =
                layerOp->getOperand(1).getType().template cast<vpux::NDTypeInterface>().getElementType();
        const auto outputElementType =
                layerOp->getResult(0).getType().template cast<vpux::NDTypeInterface>().getElementType();
        if (input1ElementType.template isa<mlir::quant::UniformQuantizedPerAxisType>() ||
            input2ElementType.template isa<mlir::quant::UniformQuantizedPerAxisType>() ||
            outputElementType.template isa<mlir::quant::UniformQuantizedPerAxisType>()) {
            return false;
        }

        return true;
    };

    if (!supportedEltwise(eltwiseOp)) {
        return matchFailed(_log, rewriter, origOp, "eltwiseOp not supported");
    }

    // Found previous shapeCast
    const auto opIdxForShapeCast = (getOpNum == 1 && getLastOpIdx == 0) ? 1 : 0;
    auto preShapeCastOp = eltwiseOp->getOperand(opIdxForShapeCast).template getDefiningOp<IE::ShapeCastOp>();
    if (preShapeCastOp == nullptr) {
        return matchFailed(_log, rewriter, origOp, "Previous ShapeCastOp not found");
    }

    // Create new shapeCast
    auto newShapeCastOp =
            rewriter.create<IE::ShapeCastOp>(origOp.getLoc(), preShapeCastOp.getResult(), origOp.getShapeAttr());

    SmallVector<mlir::Value> newInputValues;
    newInputValues.push_back(newShapeCastOp->getResult(0));
    if (!isOneInput) {
        newInputValues.push_back(eltwiseOp->getOperand(getLastOpIdx));
    } else {
        newInputValues.push_back(newShapeCastOp->getResult(0));
    }

    // Create new eltwiseOp
    mlir::IRMapping mapper;
    mapper.map(eltwiseOp->getOperands(), newInputValues);
    auto newOp = rewriter.clone(*eltwiseOp, mapper);
    vpux::inferReturnTypes(newOp, vpux::InferShapedTypeMode::SHAPE);

    rewriter.replaceOp(origOp, newOp->getResults());

    _log.trace("[{0}] Replaced with '{1}'", getDebugName(), eltwiseOp->getLoc());
    return mlir::success();
}

//
// MoveThroughMVNOp
//

class MoveThroughMVNOp final : public mlir::OpRewritePattern<IE::ShapeCastOp> {
public:
    MoveThroughMVNOp(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::ShapeCastOp>(ctx), _log(log) {
        this->setDebugName("MoveThroughMVNOp");
    }

private:
    mlir::LogicalResult matchAndRewrite(IE::ShapeCastOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult MoveThroughMVNOp::matchAndRewrite(IE::ShapeCastOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    auto mvnOp = origOp.getOperand().getDefiningOp<IE::MVNOp>();
    if (mvnOp == nullptr || !mvnOp->hasOneUse()) {
        return matchFailed(_log, rewriter, origOp, "MvnOp not found or has multiple uses");
    }

    auto preShapeCastOp = mvnOp->getOperand(0).getDefiningOp<IE::ShapeCastOp>();
    if (preShapeCastOp == nullptr) {
        return matchFailed(_log, rewriter, origOp, "Previous ShapeCastOp not found");
    }

    auto isAcrossChannels = mvnOp.getAcrossChannels();
    const auto mvnInShape = getShape(mvnOp.getInput());
    const auto shapeCastOutShape = getShape(origOp.getResult());
    const int64_t rank4D = 4;
    if (shapeCastOutShape.size() != rank4D) {
        return matchFailed(_log, rewriter, origOp, "User ShapeCastOp output shape size is not 4");
    }

    auto inChSize = mvnInShape[Dims4D::Act::H] * mvnInShape[Dims4D::Act::W] *
                    (isAcrossChannels ? mvnInShape[Dims4D::Act::C] : 1);
    auto outChSize = shapeCastOutShape[Dims4D::Act::H] * shapeCastOutShape[Dims4D::Act::W] *
                     (isAcrossChannels ? shapeCastOutShape[Dims4D::Act::C] : 1);
    if (inChSize != outChSize) {
        if (!isAcrossChannels && mvnInShape[Dims4D::Act::C] == 1 &&
            mvnInShape[Dims4D::Act::N] == shapeCastOutShape[Dims4D::Act::N]) {
            // For example, it is equivalent to propagate
            // across_channels = false, 1x1x1280x1 -> across_channels = true, 1x1280x1x1
            isAcrossChannels = true;
        } else {
            return matchFailed(_log, rewriter, origOp, "ShapeCastOp not suitable to propagate");
        }
    }

    // Create new ShapeCastOp
    auto newShapeCastOp =
            rewriter.create<IE::ShapeCastOp>(origOp.getLoc(), preShapeCastOp.getResult(), origOp.getShapeAttr());

    // Create new MVNOp
    auto newMvnOp = rewriter.create<IE::MVNOp>(mvnOp->getLoc(), newShapeCastOp.getResult(),
                                               mlir::BoolAttr::get(getContext(), isAcrossChannels),
                                               mvnOp.getNormalizeVarianceAttr(), mvnOp.getEpsAttr());

    origOp.replaceAllUsesWith(newMvnOp.getOutput());

    return mlir::success();
}

//
// MoveThroughConvBasedOp
//

template <class ConvBasedOp>
class MoveThroughConvBasedOp final : public mlir::OpRewritePattern<IE::ShapeCastOp> {
public:
    MoveThroughConvBasedOp(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::ShapeCastOp>(ctx), _log(log) {
        this->setDebugName("MoveThroughConvBasedOp");
    }

private:
    mlir::LogicalResult matchAndRewrite(IE::ShapeCastOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

template <class ConvBasedOp>
mlir::LogicalResult MoveThroughConvBasedOp<ConvBasedOp>::matchAndRewrite(IE::ShapeCastOp origOp,
                                                                         mlir::PatternRewriter& rewriter) const {
    _log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    auto inputOp = origOp->getOperand(0).getDefiningOp();
    while (mlir::isa_and_nonnull<IE::ShapeCastOp, IE::AffineReshapeOp, IE::PermuteCastOp>(inputOp) &&
           inputOp->hasOneUse()) {
        inputOp = inputOp->getOperand(0).getDefiningOp();
    }

    if (inputOp == origOp->getOperand(0).getDefiningOp()) {
        return matchFailed(_log, rewriter, origOp, "ViewLike ops before ShapeCast not found or has multi uses");
    }

    if (!mlir::isa_and_nonnull<ConvBasedOp>(inputOp) || !inputOp->hasOneUse()) {
        return matchFailed(_log, rewriter, origOp, "ConvBasedOp not found or has multi uses");
    }

    if (DimsOrder::fromValue(inputOp->getResult(0)) != DimsOrder::fromValue(origOp->getOperand(0))) {
        return matchFailed(_log, rewriter, origOp, "ConvBasedOp's layout is different from origOp");
    }

    auto userEltwiseOp = mlir::dyn_cast_or_null<IE::AddOp>(*origOp->getUsers().begin());

    auto checkLegalEltwiseOp = [](IE::AddOp addOp) -> bool {
        if (!addOp->hasOneUse()) {
            return false;
        }

        if (!mlir::isa_and_nonnull<IE::ShapeCastOp>(*addOp->getUsers().begin())) {
            return false;
        }

        auto outShape = getShape(addOp.getResult());
        if (getShape(addOp.getInput1()) != outShape || getShape(addOp.getInput2()) != outShape) {
            return false;
        }

        return true;
    };

    if (origOp->hasOneUse() && userEltwiseOp != nullptr && checkLegalEltwiseOp(userEltwiseOp)) {
        /*
            Remove viewLike ops between conv and add
            Conv/GroupConv
                |
            viewLike1
                |
               ...
                |
            viewLikeN
                |
            ShapeCast  AnotherOperand
                |        /
               Add
                |
            ShapeCast

            will be converted into:
            Conv/GroupConv  AnotherOperand
                |              /
                |           ShapeCast
                |          /
               Add
                |
            ShapeCast
        */
        SmallVector<mlir::Value> inputs;
        for (auto input : userEltwiseOp->getOperands()) {
            if (input == origOp.getResult()) {
                inputs.push_back(inputOp->getResult(0));
            } else {
                auto newShapeCastOp = rewriter.create<IE::ShapeCastOp>(
                        origOp->getLoc(), input,
                        getIntArrayAttr(rewriter.getContext(), getShape(inputOp->getResult(0))));
                inputs.push_back(newShapeCastOp.getResult());
            }
        }

        const auto outType = mlir::cast<vpux::NDTypeInterface>(userEltwiseOp.getType());
        const auto newType = outType.changeShape(getShape(inputs[0]));
        auto newAddOp = rewriter.create<IE::AddOp>(origOp->getLoc(), newType, inputs[0], inputs[1],
                                                   userEltwiseOp.getAutoBroadcastAttr(), userEltwiseOp.getPostOpAttr(),
                                                   userEltwiseOp.getClampAttr(), nullptr, nullptr);

        auto newOutShapeCastOp = rewriter.create<IE::ShapeCastOp>(
                origOp->getLoc(), newAddOp.getResult(),
                getIntArrayAttr(rewriter.getContext(), getShape(userEltwiseOp.getResult())));
        rewriter.replaceOp(userEltwiseOp, newOutShapeCastOp.getResult());

        _log.trace("[{0}] Successfully applied ConvLikeOp -> Add -> ShapeCast optimization", getDebugName());
        return mlir::success();
    }

    /*
        Remove viewLike ops between conv and shapeCast
        Conv/GroupConv
            |
        viewLike1
            |
           ...
            |
        viewLikeN
            |
        ShapeCast
            |
        NonViewLikeOp

        will be converted into:
        Conv/GroupConv
            |
        ShapeCast
            |
        NonViewLikeOp
    */
    auto newShapeCastOp =
            rewriter.create<IE::ShapeCastOp>(origOp.getLoc(), inputOp->getResult(0), origOp.getShapeAttr());
    rewriter.replaceOp(origOp, newShapeCastOp.getResult());

    _log.trace("[{0}] Successfully remove viewLike ops before '{1}'", getDebugName(), origOp->getLoc());
    return mlir::success();
}

//
// PropagateShapeCast
//

class PropagateShapeCast final : public IE::impl::PropagateShapeCastBase<PropagateShapeCast> {
public:
    explicit PropagateShapeCast(Logger log): _log(log) {
        _log.setName(Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

/**
 * ShapeCast operations are insert before and after ops when they call adjust passes, such as GroupConv and Eltwise.
 * And ShapeCast blocked the link relationship for original adjacent ops. Sometimes we can directly move some ops post
 * ShapeCast to reconstruct their link relationship. Such as
 *
 *          ShapeCast                            ShapeCast
 *              |                                    |
 *             Op                                ShapeCast
 *              |                --->                |
 *          ShapeCast                               Op
 *              |                                    |
 *            NceOp                                NceOp
 */
void PropagateShapeCast::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<MoveThroughActivationOp<IE::AbsOp>>(&ctx, _log);
    patterns.add<MoveThroughActivationOp<IE::GeluOp>>(&ctx, _log);
    patterns.add<MoveThroughActivationOp<IE::SwishOp>>(&ctx, _log);
    patterns.add<MoveThroughActivationOp<IE::HSwishOp>>(&ctx, _log);
    patterns.add<MoveThroughActivationOp<IE::SigmoidOp>>(&ctx, _log);
    patterns.add<MoveThroughActivationOp<IE::TanhOp>>(&ctx, _log);
    patterns.add<MoveThroughPoolOp<IE::AvgPoolOp>>(&ctx, _log);
    patterns.add<MoveThroughPoolOp<IE::MaxPoolOp>>(&ctx, _log);
    patterns.add<MoveThroughEltwiseOp<IE::SubtractOp>>(&ctx, _log);
    patterns.add<MoveThroughEltwiseOp<IE::AddOp>>(&ctx, _log);
    patterns.add<MoveThroughEltwiseOp<IE::MultiplyOp>>(&ctx, _log);
    patterns.add<MoveThroughMVNOp>(&ctx, _log);
    patterns.add<MoveThroughConvBasedOp<IE::GroupConvolutionOp>>(&ctx, _log);
    IE::ShapeCastOp::getCanonicalizationPatterns(patterns, &ctx);

    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createPropagateShapeCastPass(Logger log) {
    return std::make_unique<PropagateShapeCast>(log);
}
