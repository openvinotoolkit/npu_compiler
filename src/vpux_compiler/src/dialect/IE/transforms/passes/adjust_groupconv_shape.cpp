//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/reshape_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/shape_infer.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/dialect/const/attributes/attributes.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

namespace vpux::IE {
#define GEN_PASS_DECL_ADJUSTGROUPCONVSHAPE
#define GEN_PASS_DEF_ADJUSTGROUPCONVSHAPE
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {
const uint32_t levelCount = 2;
SmallVector<mlir::PatternBenefit> benefitLevels = getBenefitLevels(levelCount);

bool isConstAndSameAsGroup(mlir::Value value, int64_t groups) {
    auto cst = value.getDefiningOp<Const::DeclareOp>();
    if (!cst) {
        return false;
    }
    // Only handle below scenario:
    // kernel size 1x1 and Group == Output Channel
    auto shapeSize = mlir::cast<vpux::NDTypeInterface>(value.getType()).getNumElements();
    if (shapeSize != groups) {
        return false;
    }
    return true;
};

//
// ReshapeGroupConvInput
//

class ReshapeGroupConvInput final : public mlir::OpRewritePattern<IE::GroupConvolutionOp> {
public:
    ReshapeGroupConvInput(mlir::MLIRContext* ctx, mlir::PatternBenefit benefit, Logger log)
            : mlir::OpRewritePattern<IE::GroupConvolutionOp>(ctx, benefit), _log(log) {
        setDebugName("ReshapeGroupConvInput");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::GroupConvolutionOp convOp, mlir::PatternRewriter& rewriter) const final;

private:
    IE::ShapeCastOp reshapeOutput(IE::GroupConvolutionOp origOp, mlir::Value convOutput,
                                  mlir::PatternRewriter& rewriter) const;
    Const::DeclareOp broadcastConst(mlir::Value activation, int64_t factor, Dim onDim,
                                    mlir::PatternRewriter& rewriter) const;

private:
    Logger _log;
};

Const::DeclareOp ReshapeGroupConvInput::broadcastConst(mlir::Value activation, int64_t dimVal, Dim onDim,
                                                       mlir::PatternRewriter& rewriter) const {
    const auto origInShape = getShape(activation);
    const auto KX = 1;
    const auto KY = 1;
    const auto IC = (onDim == Dims4D::Filter::IC) ? dimVal : 1;
    const auto OC = (onDim == Dims4D::Filter::OC) ? dimVal : 1;

    const Shape weightShape = {OC, IC, KY, KX};

    auto cst = activation.getDefiningOp<Const::DeclareOp>();

    Const::ContentAttr newContentAttr;
    if (cst.getContentAttr().isSplat()) {
        const auto& contentAttr = cst.getContentAttr();
        const auto& baseContent = contentAttr.getBaseContent();
        Const::ContentSetup setup(baseContent.getType());  // content-only copy
        Shape shape(origInShape.size(), int64_t(1));
        setup = setup.reshape(shape);

        for (auto attr : contentAttr.getTransformations()) {
            if (Const::canChangeShape(attr)) {
                // The const's shape will fully handled by this pass, the broadcast will be added,
                //   so ignore the origin broadcast, reshape and pad transformation
                continue;
            }
            setup = setup.addTransformation(attr);
        }

        newContentAttr = Const::ContentAttr::get(baseContent, setup.broadcast(onDim, weightShape[onDim]));
    } else {
        auto broadcastDim = (OC > 1 ? Dims4D::Filter::OC : Dims4D::Filter::IC);
        newContentAttr = cst.getContentAttr()
                                 .transform()  // Note: a *complete* attribute copy (with transformations)
                                 .broadcast(broadcastDim, weightShape[broadcastDim])
                                 .reshape(weightShape)
                                 .get();
    }
    return rewriter.create<Const::DeclareOp>(activation.getLoc(), newContentAttr.getType(), std::move(newContentAttr));
}

IE::ShapeCastOp ReshapeGroupConvInput::reshapeOutput(IE::GroupConvolutionOp origOp, mlir::Value convOutput,
                                                     mlir::PatternRewriter& rewriter) const {
    const auto origOutShape = getShape(origOp.getOutput());
    const SmallVector<int64_t> targetShape = to_small_vector(origOutShape.raw());

    const auto reshapedLoc = appendLoc(origOp.getLoc(), "_output_reshape");
    return vpux::IE::buildShapeCast(reshapedLoc, convOutput, ArrayRef(targetShape), rewriter);
}

mlir::LogicalResult ReshapeGroupConvInput::matchAndRewrite(IE::GroupConvolutionOp convOp,
                                                           mlir::PatternRewriter& rewriter) const {
    const auto alignedChannel = VPU::NCEInvariant::VPU_CHANNEL_ALIGNMENT;
    auto groups = convOp.getGroups().value();
    auto inNDInterface = mlir::dyn_cast<vpux::NDTypeInterface>(convOp.getInput().getType());
    auto inDimOrder = inNDInterface.getDimsOrder();
    SmallVector<int64_t> newInputShape;
    if (DimsOrder::NHWC != inDimOrder) {
        return mlir::failure();
    }

    auto isSplat = [](mlir::Value val) {
        auto cst = val.getDefiningOp<Const::DeclareOp>();
        return cst.getContentAttr().isSplat();
    };

    auto isElementWised = [isSplat](IE::GroupConvolutionOp op) {
        return isSplat(op.getFilter()) && (!op.getBias() || isSplat(op.getBias()));
    };

    const auto supportedGroupConv = [&](IE::GroupConvolutionOp layerOp) {
        auto outNDInterface = mlir::cast<vpux::NDTypeInterface>(layerOp.getOutput().getType());
        auto outDimOrder = outNDInterface.getDimsOrder();
        if (inDimOrder != outDimOrder) {
            return false;
        }

        if (outNDInterface.getElementType() != inNDInterface.getElementType()) {
            return false;
        }

        auto strides = parseIntArrayAttr<int64_t>(layerOp.getStrides());
        auto withOutStrides = std::all_of(strides.begin(), strides.end(), [](auto val) {
            return val == 1;
        });
        if (!withOutStrides) {
            return false;
        }

        // Already aligned
        if (!(groups % alignedChannel)) {
            return false;
        }
        // check kernel size and is a depthwise convolution or not
        if (!isConstAndSameAsGroup(layerOp.getFilter(), groups)) {
            return false;
        }

        // check filter and bias const is same
        if (layerOp.getBias() != nullptr) {
            if (!isConstAndSameAsGroup(layerOp.getBias(), groups)) {
                return false;
            }
        }

        auto input = layerOp.getInput().getDefiningOp<IE::PermuteQuantizeOp>();
        if (input) {
            // Expand can be fused to PermuteQuantize, ignore
            if (!layerOp.getOutput().hasOneUse()) {
                return false;
            }
        }

        // Input can be reshaped
        auto newExpandedShapeResult =
                getShapeCastExpandedShapeKeepDimC(layerOp, getShape(layerOp.getInput()), _log.nest());
        // Borrowing from weight or height is better than adjust channel to alignment value when it's bigger for all
        // cases, since the other nce ops using the same method when adjust input shape. So it would not insert
        // ShapeCast between these ops which would benefit the other optimizations such as VF.
        if (!mlir::failed(newExpandedShapeResult)) {
            newInputShape = newExpandedShapeResult.value().raw();
            return true;
        } else if (isElementWised(layerOp)) {
            newExpandedShapeResult =
                    getShapeCastExpandedShapeInDimC(layerOp, getShape(layerOp.getInput()), _log.nest());
            if (mlir::failed(newExpandedShapeResult)) {
                newExpandedShapeResult =
                        getShapeCastExpandedShapeCanNotAlign(layerOp, getShape(layerOp.getInput()), _log.nest());
                if (mlir::failed(newExpandedShapeResult)) {
                    return false;
                }
            }
            newInputShape = newExpandedShapeResult.value().raw();
            return true;
        }

        return false;
    };
    if (!supportedGroupConv(convOp)) {
        return mlir::failure();
    }
    _log.trace("Adjust input/filter/bias shape for group convolution at '{0}'", convOp->getLoc());
    auto input = vpux::IE::buildShapeCast(convOp.getLoc(), convOp.getInput(), newInputShape, rewriter);
    auto weights = broadcastConst(convOp.getFilter(), newInputShape[Dims4D::Act::C.ind()], Dims4D::Filter::OC, rewriter)
                           .getOutput();
    auto bias = convOp.getBias();
    if (bias) {
        bias = mlir::cast<mlir::TypedValue<mlir::RankedTensorType>>(
                broadcastConst(bias, newInputShape[Dims4D::Act::C.ind()], Dims4D::Filter::IC, rewriter).getOutput());
    }
    auto newGroupConv = rewriter.create<IE::GroupConvolutionOp>(
            convOp.getLoc(), input, weights, bias, convOp.getStrides(), convOp.getPadsBegin(), convOp.getPadsEnd(),
            convOp.getDilations(), getIntAttr(rewriter.getContext(), newInputShape[Dims4D::Act::C.ind()]),
            convOp.getPostOpAttr(), convOp.getClampAttr(), convOp.getOutputPaddingAttr(), convOp.getInputPaddingAttr());
    auto newInShape = getShape(input);
    auto origOutputType = mlir::cast<vpux::NDTypeInterface>(convOp.getType());
    newGroupConv.getOutput().setType(mlir::cast<mlir::RankedTensorType>(origOutputType.changeShape(newInShape)));
    auto output = reshapeOutput(convOp, newGroupConv, rewriter);
    rewriter.replaceOp(convOp, output.getResult());
    return mlir::success();
}

//
// SliceGroupConvInput
//

class SliceGroupConvInput final : public mlir::OpRewritePattern<IE::GroupConvolutionOp> {
public:
    SliceGroupConvInput(mlir::MLIRContext* ctx, mlir::PatternBenefit benefit, Logger log)
            : mlir::OpRewritePattern<IE::GroupConvolutionOp>(ctx, benefit), _log(log) {
        setDebugName("SliceGroupConvInput");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::GroupConvolutionOp convOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult SliceGroupConvInput::matchAndRewrite(IE::GroupConvolutionOp origOp,
                                                         mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());
    auto ctx = rewriter.getContext();
    auto groups = origOp.getGroups().value();
    auto inNDInterface = mlir::dyn_cast<vpux::NDTypeInterface>(origOp.getInput().getType());
    auto outNDInterface = mlir::dyn_cast<vpux::NDTypeInterface>(origOp.getOutput().getType());
    auto inDimOrder = inNDInterface.getDimsOrder();
    auto outDimOrder = outNDInterface.getDimsOrder();
    auto inputShape = inNDInterface.getShape();
    auto channel = inputShape[Dims4D::Act::C];

    IE::ReorderOp preReorder = nullptr;

    const auto supportedGroupConv = [&](IE::GroupConvolutionOp layerOp) {
        if (DimsOrder::NHWC != inDimOrder) {
            return false;
        }

        if (inDimOrder != outDimOrder) {
            return false;
        }

        if (outNDInterface.getElementType() != inNDInterface.getElementType()) {
            return false;
        }

        auto strides = parseIntArrayAttr<int64_t>(layerOp.getStrides());
        auto withOutStrides = std::all_of(strides.begin(), strides.end(), [](auto val) {
            return val == 1;
        });
        if (!withOutStrides) {
            return false;
        }

        if (channel >= 4) {
            return false;
        }

        if (groups == 1) {
            return false;
        }

        // check kernel size and is a depthwise convolution or not
        if (!isConstAndSameAsGroup(layerOp.getFilter(), groups)) {
            return false;
        }

        // check filter and bias const is same
        if (layerOp.getBias() != nullptr) {
            if (!isConstAndSameAsGroup(layerOp.getBias(), groups)) {
                return false;
            }
        }

        preReorder = layerOp.getInput().getDefiningOp<IE::ReorderOp>();
        if (preReorder == nullptr || !preReorder.getOutput().hasOneUse()) {
            return false;
        }

        auto postReorder = mlir::dyn_cast<IE::ReorderOp>(*layerOp->getUsers().begin());
        if (postReorder == nullptr || !layerOp->hasOneUse() ||
            DimsOrder::fromValue(postReorder.getOutput()) != DimsOrder::NCHW) {
            return false;
        }

        return true;
    };

    if (!supportedGroupConv(origOp)) {
        return mlir::failure();
    }

    SmallVector<mlir::Value> values;
    auto weightShape = getShape(origOp.getFilter());
    mlir::Value biasValue = nullptr;

    for (auto i : irange(groups)) {
        auto staticOffsets = SmallVector<int64_t>(inputShape.size(), 0);
        staticOffsets[Dims4D::Act::C.ind()] = i;

        SmallVector<int64_t> staticSizes(inputShape.begin(), inputShape.end());
        staticSizes[Dims4D::Act::C.ind()] = 1;
        auto sliceInput =
                rewriter.create<IE::SliceOp>(takeOpLoc(origOp, llvm::formatv("_input_{0}", i)), preReorder.getInput(),
                                             getIntArrayAttr(ctx, staticOffsets), getIntArrayAttr(ctx, staticSizes));

        auto preInOrder = DimsOrder::fromValue(preReorder.getInput());
        const auto inMemShape = preInOrder.toMemoryOrder(getShape(sliceInput.getResult()));
        auto permutation = getPermutationFromOrders(preInOrder, DimsOrder::fromValue(preReorder.getOutput()), ctx);

        if (!isTrivialPermute(inMemShape, permutation)) {
            return mlir::failure();
        }

        auto permuteCast = rewriter.create<IE::PermuteCastOp>(
                origOp->getLoc(), sliceInput.getResult(),
                mlir::AffineMapAttr::get(DimsOrder::fromValue(preReorder.getOutput()).toAffineMap(ctx)),
                mlir::AffineMapAttr::get(permutation));

        auto weightStaticOffsets = SmallVector<int64_t>(weightShape.size(), 0);
        weightStaticOffsets[Dims4D::Filter::OC.ind()] = i;

        SmallVector<int64_t> weightStaticSizes(weightShape.begin(), weightShape.end());
        weightStaticSizes[Dims4D::Filter::OC.ind()] = 1;
        auto sliceWeight = rewriter.create<IE::SliceOp>(takeOpLoc(origOp, llvm::formatv("_weight_{0}", i)),
                                                        origOp.getFilter(), getIntArrayAttr(ctx, weightStaticOffsets),
                                                        getIntArrayAttr(ctx, weightStaticSizes));

        if (origOp.getBias()) {
            auto biasShape = getShape(origOp.getBias());
            auto biasStaticOffsets = SmallVector<int64_t>(biasShape.size(), 0);
            biasStaticOffsets[Dims4D::Filter::IC.ind()] = i;

            SmallVector<int64_t> biasStaticSizes(biasShape.begin(), biasShape.end());
            biasStaticSizes[Dims4D::Filter::IC.ind()] = 1;
            auto sliceBias = rewriter.create<IE::SliceOp>(takeOpLoc(origOp, llvm::formatv("_bias_{0}", i)),
                                                          origOp.getBias(), getIntArrayAttr(ctx, biasStaticOffsets),
                                                          getIntArrayAttr(ctx, biasStaticSizes));
            biasValue = sliceBias.getResult();
        }

        auto newGroupConv = rewriter.create<IE::GroupConvolutionOp>(
                takeOpLoc(origOp, llvm::formatv("_{0}", i)), permuteCast.getOutput(), sliceWeight.getResult(),
                biasValue, origOp.getStrides(), origOp.getPadsBegin(), origOp.getPadsEnd(), origOp.getDilations(),
                getIntAttr(ctx, 1), origOp.getPostOpAttr(), origOp.getClampAttr(), origOp.getOutputPaddingAttr(),
                origOp.getInputPaddingAttr());
        auto groupOutputType = mlir::cast<vpux::NDTypeInterface>(newGroupConv.getOutput().getType());
        newGroupConv.getOutput().setType(mlir::cast<mlir::RankedTensorType>(
                groupOutputType.changeDimsOrder(DimsOrder::fromValue(origOp.getOutput()))));

        values.push_back(newGroupConv.getOutput());
    }

    auto newConcat = rewriter.create<IE::ConcatOp>(origOp->getLoc(), values, Dim(Dims4D::Act::C.ind()));
    rewriter.replaceOp(origOp, newConcat.getOutput());

    return mlir::success();
}

//
// AdjustGroupConvShape
//

class AdjustGroupConvShapePass final : public IE::impl::AdjustGroupConvShapeBase<AdjustGroupConvShapePass> {
public:
    explicit AdjustGroupConvShapePass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void AdjustGroupConvShapePass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<SliceGroupConvInput>(&ctx, benefitLevels[0], _log);
    patterns.add<ReshapeGroupConvInput>(&ctx, benefitLevels[1], _log);

    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}
}  // namespace

//
// createAdjustGroupConvShapePass
//

std::unique_ptr<mlir::Pass> vpux::IE::createAdjustGroupConvShapePass(Logger log) {
    return std::make_unique<AdjustGroupConvShapePass>(log);
}
