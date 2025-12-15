//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/reshape_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/slice_utils.hpp"
#include "vpux/compiler/dialect/const/utils/affine_reshape.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/range.hpp"

namespace vpux::IE {
#define GEN_PASS_DECL_FUSEMEMPERMUTEPASS
#define GEN_PASS_DEF_FUSEMEMPERMUTEPASS
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
// MemPermuteRewriter
//

class MemPermuteRewriter final : public mlir::OpRewritePattern<IE::MemPermuteOp> {
public:
    MemPermuteRewriter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::MemPermuteOp>(ctx), _log(log) {
        this->setDebugName("MemPermuteRewriter");
    }

private:
    mlir::LogicalResult matchAndRewrite(IE::MemPermuteOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult MemPermuteRewriter::matchAndRewrite(IE::MemPermuteOp origOp,
                                                        mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    const auto inOrder = DimsOrder::fromValue(origOp.getInput());
    const auto inShape = getShape(origOp.getInput());
    const auto inMemShape = inOrder.toMemoryOrder(inShape);
    if (isTrivialPermute(inMemShape, origOp.getMemPerm())) {
        return matchFailed(_log.nest(), rewriter, origOp, "MemPermuteOp is actually a permute cast");
    }

    auto layerWithPermute = getFusableLayerWithPermuteInterface(origOp.getOperation());
    if (layerWithPermute == nullptr) {
        return matchFailed(_log.nest(), rewriter, origOp, "MemPermuteRewriter applies for NCE tasks");
    }

    if (!layerWithPermute.isSupportedPermutation(origOp)) {
        return matchFailed(_log.nest(), rewriter, origOp, "ODU permutation does not support {0} at {1}",
                           origOp->getName(), origOp->getLoc());
    }

    if (!layerWithPermute->getResult(0).hasOneUse()) {
        return matchFailed(_log.nest(), rewriter, origOp,
                           "ReorderRewriter applies only for NCE tasks with one consumer");
    }

    auto output = layerWithPermute->getResult(0);
    const auto origType = mlir::cast<vpux::NDTypeInterface>(output.getType());
    if (origType == nullptr) {
        return matchFailed(_log.nest(), rewriter, origOp, "NCE task does not implement vpux::NDTypeInterface");
    }

    _log.trace("Fuse {0} to {1}", origOp->getLoc(), layerWithPermute->getLoc());

    auto maybeQuantizeCastOp = mlir::dyn_cast_or_null<IE::QuantizeCastOp>(*(layerWithPermute->getUsers().begin()));

    const auto targetOrder = applyPermutation(inOrder, DimsOrder::fromAffineMap(origOp.getMemPerm()));
    const auto adjustedOrder = moveD0ToTheFront(targetOrder);
    const auto newType = origType.changeDimsOrder(adjustedOrder);
    layerWithPermute->getResult(0).setType(newType);

    auto ctx = rewriter.getContext();
    const auto dstOrderMap = origOp.getDstOrder();
    const auto trivialMemPerm = getPermutationFromOrders(adjustedOrder, targetOrder, ctx);

    auto getDimMappingAttrValue = [&](auto inShape, auto outShape,
                                      auto inOrder) -> std::optional<SmallVector<SmallVector<int64_t>>> {
        const auto reassociationMap = vpux::IE::getReassociationMap(inShape, outShape);
        if (mlir::failed(reassociationMap)) {
            return std::nullopt;
        }

        const auto outputLayout = Const::inferAffineReshapeOutputLayout(
                inOrder.toPermutation(), getIntArrayOfArray(ctx, reassociationMap.value()));

        if (!outputLayout.has_value() || outputLayout.value() != DimsOrder::fromValue(origOp.getOutput())) {
            return std::nullopt;
        }

        return reassociationMap.value();
    };

    mlir::Value newOutput;
    auto ODULayerOutShape = getShape(layerWithPermute->getResult(0));
    auto ODUOutOrder = DimsOrder::fromValue(layerWithPermute->getResult(0));
    auto outputShape = getShape(origOp.getOutput());
    auto dimMappingAttrValue = getDimMappingAttrValue(ODULayerOutShape, outputShape, ODUOutOrder);
    // Cases like [1, 0, X, X] ->[0, 1, X, X] which changed by adjustedOrder can be affineReshaped to original outShape
    // E#163862: runtime issue of maxpool+permuteCast than maxpool+affineReshape when DimN changed for some pattern
    if (targetOrder.toPermutation()[1] == Dim(0) && dimMappingAttrValue.has_value()) {
        newOutput = rewriter.createOrFold<IE::AffineReshapeOp>(origOp.getLoc(), layerWithPermute->getResult(0),
                                                               getIntArrayOfArray(ctx, dimMappingAttrValue.value()),
                                                               getIntArrayAttr(ctx, outputShape));
    } else {
        newOutput = rewriter.createOrFold<IE::PermuteCastOp>(origOp.getLoc(), layerWithPermute->getResult(0),
                                                             dstOrderMap, trivialMemPerm);
    }

    if (maybeQuantizeCastOp != nullptr) {
        newOutput = rewriter.createOrFold<IE::QuantizeCastOp>(
                maybeQuantizeCastOp->getLoc(), origOp.getType(), newOutput,
                mlir::cast<vpux::NDTypeInterface>(maybeQuantizeCastOp.getOutput().getType()).getElementType());
    }

    rewriter.replaceOp(origOp, newOutput);

    return mlir::success();
}

//
// FuseMemPermuteThroughViewOps
//

// In the following scenario, it is not feasible to move MemPermute through AffineReshape because it swaps the data on
// the split axes. The initial graph structure is as follows:
// 1x16x262144x4@NHWC  16x16x1x1@NHWC
//          \               /
//             Convolution
//                  |
//          1x16x262144x4@NHWC
//                  |
//                Slice
//                  |
//          1x7x262144x4@NHWC
//                  |
//             AffineReshape
//                  |
//          1x7x1048576x1@NHWC
//                  |
//              PermuteCast
//                  |
//            1048576x7x1x1
//                  |
//             AffineReshape
//                  |
//             1x4096x256x7
//                  |
//              MemPermute
//                  |
//             1x4096x7x256
//
// Given that the innermost dimension is never modified, we can reshape Convolution input shape between height and
// width dimensions.
// The sub-graph obtained after transformation is as follows, and finally the MemPermute can be fused with Convolution
// for a free ODU permutation.
// 1x16x4096x256@NHWC	16x16x1x1@NHWC
//          \               /
//             Convolution
//                  |
//          1x16x4096x256@NHWC
//                  |
//              MemPermute
//                  |
//             1x4096x16x256
//                  |
//                Slice
//                  |
//             1x4096x7x256
//
struct ConvOpResult {
    IE::ConvolutionOp convOp;
    IE::SliceOp sliceOp;
    int64_t viewOpsCount{};
};

class FuseMemPermuteThroughViewOps final : public mlir::OpRewritePattern<IE::MemPermuteOp> {
public:
    FuseMemPermuteThroughViewOps(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::MemPermuteOp>(ctx), _log(log) {
        this->setDebugName("FuseMemPermuteThroughViewOps");
    }

private:
    mlir::LogicalResult matchAndRewrite(IE::MemPermuteOp origOp, mlir::PatternRewriter& rewriter) const final;

    std::optional<ConvOpResult> retrieveConvOpThroughViewOps(IE::MemPermuteOp origOp) const;
    bool isValidMemPermuteOp(IE::MemPermuteOp memPermuteOp) const;
    bool isValidConvOp(IE::ConvolutionOp convOp) const;

private:
    const size_t SUPPORTED_RANK = 4;
    Logger _log;
};

std::optional<ConvOpResult> FuseMemPermuteThroughViewOps::retrieveConvOpThroughViewOps(IE::MemPermuteOp origOp) const {
    IE::SliceOp sliceOp = nullptr;
    IE::ConvolutionOp convOp = nullptr;
    int64_t viewOpsCnt = 0;
    auto parentOp = origOp.getInput().getDefiningOp();

    while (parentOp != nullptr && parentOp->hasOneUse()) {
        if (mlir::isa<IE::SliceOp>(parentOp)) {
            sliceOp = mlir::cast<IE::SliceOp>(parentOp);
            convOp = mlir::dyn_cast_or_null<IE::ConvolutionOp>(sliceOp.getSource().getDefiningOp());
            if (convOp == nullptr || !convOp->hasOneUse()) {
                return std::nullopt;
            }

            // Matched ConvolutionOp - SliceOp - [ViewOps] - MemPermuteOp
            return ConvOpResult{convOp, sliceOp, viewOpsCnt};
        }

        if (mlir::isa<IE::ConvolutionOp>(parentOp)) {
            // Matched ConvolutionOp  - [ViewOps] - MemPermuteOp
            return ConvOpResult{mlir::cast<IE::ConvolutionOp>(parentOp), nullptr, viewOpsCnt};
        }

        if (IE::isPureViewOp(parentOp) && !mlir::isa<IE::QuantizeCastOp>(parentOp)) {
            parentOp = parentOp->getOperand(0).getDefiningOp();
            viewOpsCnt++;
        } else {
            return std::nullopt;
        }
    }

    return std::nullopt;
}

bool FuseMemPermuteThroughViewOps::isValidMemPermuteOp(IE::MemPermuteOp memPermuteOp) const {
    const auto inMemShape = getMemShape(memPermuteOp.getInput());
    if (isTrivialPermute(inMemShape, memPermuteOp.getMemPerm())) {
        return false;
    }

    // Ensure MemPermuteOp is 4D and batch size is 1
    if (inMemShape.size() != SUPPORTED_RANK || inMemShape.front() != 1) {
        return false;
    }

    return true;
}

bool FuseMemPermuteThroughViewOps::isValidConvOp(IE::ConvolutionOp convOp) const {
    const auto inputShape = getShape(convOp.getInput());
    const auto filterShape = getShape(convOp.getFilter());

    // TODO: support Convolution with Bias
    if (convOp.getBias() != nullptr) {
        return false;
    }

    // Current implementation only supports input and filter shape with 4 dimensions
    if (inputShape.size() != SUPPORTED_RANK || filterShape.size() != SUPPORTED_RANK) {
        return false;
    }

    // Check suitable 1x1 convolution: strides = [1, 1], kernel = [1, 1] and no padding
    if (filterShape[Dims4D::Filter::KX] != 1 || filterShape[Dims4D::Filter::KY] != 1) {
        return false;
    }

    auto checkAllElementsEqualToValue = [](const SmallVector<int64_t>& shape, const int64_t value) {
        return std::all_of(shape.begin(), shape.end(), [&](auto size) {
            return size == value;
        });
    };

    const auto strides = parseIntArrayAttr<int64_t>(convOp.getStrides());
    if (!checkAllElementsEqualToValue(strides, 1)) {
        return false;
    }

    auto origPadsBegin = parseIntArrayAttr<int64_t>(convOp.getPadsBegin());
    auto origPadsEnd = parseIntArrayAttr<int64_t>(convOp.getPadsEnd());
    if (!checkAllElementsEqualToValue(origPadsBegin, 0) || !checkAllElementsEqualToValue(origPadsEnd, 0)) {
        return false;
    }

    // When the convolution output order is NHWC and MemShape[DimC] is unchanged by viewOps,
    // the original 1x1 convolution input shape can be freely reshaped on dimensions H and W.
    auto convOutType = mlir::cast<NDTypeInterface>(convOp.getOutput().getType());
    return convOutType.getDimsOrder() == DimsOrder::NHWC;
}

mlir::LogicalResult FuseMemPermuteThroughViewOps::matchAndRewrite(IE::MemPermuteOp origOp,
                                                                  mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    if (!isValidMemPermuteOp(origOp)) {
        _log.nest().trace("Transformation aborted: MemPermuteOp validation failed.");
        return mlir::failure();
    }

    auto result = retrieveConvOpThroughViewOps(origOp);
    if (!result.has_value()) {
        _log.nest().trace("Transformation skipped: Pattern match failed.");
        return mlir::failure();
    }

    auto viewOpsCnt = result.value().viewOpsCount;
    if (viewOpsCnt == 0) {
        _log.nest().trace("Transformation skipped: MemPermuteOp follows ConvOp, making conversion redundant.");
        return mlir::failure();
    }

    auto convOp = result.value().convOp;
    if (convOp == nullptr) {
        _log.nest().trace("Failure: Unable to locate ConvOp via View operations.");
        return mlir::failure();
    }

    if (!isValidConvOp(convOp)) {
        _log.nest().trace("Transformation aborted: ConvOp validation failed.");
        return mlir::failure();
    }

    auto areMemShapesCompatible = [this](mlir::Operation* parentOp, mlir::Operation* childOp) {
        auto parentOutMemShape = getMemShape(parentOp->getResult(0));
        auto childInMemShape = getMemShape(childOp->getOperand(0));
        if (parentOutMemShape.back() != childInMemShape.back()) {
            _log.nest().trace("Transformation skipped: Parent operation output MemShape {0} and child operation input "
                              "MemShape {1} are not compatible.",
                              parentOutMemShape, childInMemShape);
            return false;
        }
        return true;
    };
    auto sliceOp = result.value().sliceOp;
    if (sliceOp != nullptr) {
        // Check Slice axis is on C to ensure original SliceOp offsets[W] and offsets[H] are 0.
        // Otherwise it's difficult to determine the offsets[W] and offsets[H] of new SliceOp after shape adjustment.
        auto sliceAxes = getSliceAxes(sliceOp);
        if (sliceAxes.size() != 1) {
            return mlir::failure();
        }
        auto sliceAxis = sliceAxes.front();
        if (sliceAxis != checked_cast<uint64_t>(Dims4D::Act::C.ind())) {
            return mlir::failure();
        }

        if (!areMemShapesCompatible(sliceOp, origOp)) {
            return mlir::failure();
        }
    } else {
        if (!areMemShapesCompatible(convOp, origOp)) {
            return mlir::failure();
        }
    }

    auto permuteInterface = mlir::dyn_cast<IE::LayerWithPermuteInterface>(convOp.getOperation());
    if (permuteInterface == nullptr) {
        return mlir::failure();
    }
    if (!permuteInterface.isSupportedPermutation(origOp)) {
        _log.nest().trace("Transformation skipped: Unsupported ODU permute.");
        return mlir::failure();
    }

    auto ctx = rewriter.getContext();

    // To ensure that the new convolution output MemShape matches the input MemShape of the original MemPermute
    // operation, adjustments are made to the input shape specifically on dimensions H and W, the size of DimC
    // remains unchanged.
    auto convOutOrder = mlir::cast<NDTypeInterface>(convOp.getOutput().getType()).getDimsOrder();
    auto permuteInMemShape = getMemShape(origOp.getInput());
    auto newConvOutShapeVec = to_small_vector(convOutOrder.toLogicalOrder(permuteInMemShape));
    newConvOutShapeVec[Dims4D::Act::C.ind()] = getShape(convOp.getOutput())[Dims4D::Act::C];
    auto newConvInShapeVec = newConvOutShapeVec;
    auto origConvInShape = getShape(convOp.getInput());
    newConvInShapeVec[Dims4D::Act::C.ind()] = origConvInShape[Dims4D::Act::C];

    auto newInputType =
            mlir::cast<NDTypeInterface>(convOp.getInput().getType()).changeShape(ShapeRef(newConvInShapeVec));
    auto inShapeCast = rewriter.create<IE::ShapeCastOp>(appendLoc(convOp.getLoc(), "_in_reshape"), newInputType,
                                                        convOp.getInput(), getIntArrayAttr(ctx, newConvInShapeVec));

    auto convType = mlir::cast<NDTypeInterface>(convOp.getOutput().getType()).changeShape(ShapeRef(newConvOutShapeVec));
    auto newConv = rewriter.create<IE::ConvolutionOp>(
            convOp->getLoc(), convType, inShapeCast, convOp.getFilter(), convOp.getBias(), convOp.getStrides(),
            convOp.getPadsBegin(), convOp.getPadsEnd(), convOp.getDilations(), convOp.getPostOpAttr(),
            convOp.getClampAttr(), convOp.getStaticScaleAttr(), convOp.getOutputPaddingAttr(),
            convOp.getInputPaddingAttr());

    // New MemPermute has the same dstOrder and memPerm attributes
    auto newPermute = rewriter.create<IE::MemPermuteOp>(origOp->getLoc(), newConv.getOutput(), origOp.getDstOrderAttr(),
                                                        origOp.getMemPermAttr());

    if (sliceOp == nullptr) {
        rewriter.replaceOp(origOp, newPermute.getOutput());
    } else {
        auto srcOrder = convOutOrder;
        auto dstOrder = DimsOrder::fromAffineMap(origOp.getDstOrder());
        auto perm = origOp.getMemPerm();

        auto sliceAxis = getSliceAxes(sliceOp).front();
        auto newSliceAxis = inferDimAfterPermutation(Dim(sliceAxis), srcOrder, dstOrder, perm);

        const auto origOffsets = parseIntArrayAttr<int64_t>(sliceOp.getStaticOffsets());
        const auto origShape = parseIntArrayAttr<int64_t>(sliceOp.getStaticSizes());
        auto newOffsets = std::move(origOffsets);
        auto newSizes = to_small_vector(getShape(newPermute.getOutput()));
        newSizes[newSliceAxis.ind()] = origShape[sliceAxis];

        auto newSlice = rewriter.create<IE::SliceOp>(sliceOp.getLoc(), newPermute.getOutput(),
                                                     getIntArrayAttr(ctx, newOffsets), getIntArrayAttr(ctx, newSizes));

        rewriter.replaceOp(origOp, newSlice.getResult());
    }

    return mlir::success();
}

//
// FuseMemPermutePass
//

class FuseMemPermutePass final : public IE::impl::FuseMemPermutePassBase<FuseMemPermutePass> {
public:
    explicit FuseMemPermutePass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void FuseMemPermutePass::safeRunOnFunc() {
    auto& ctx = getContext();
    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<MemPermuteRewriter>(&ctx, _log);
    patterns.add<FuseMemPermuteThroughViewOps>(&ctx, _log);

    auto func = getOperation();
    if (mlir::failed(applyPatternsGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createFuseMemPermutePass(Logger log) {
    return std::make_unique<FuseMemPermutePass>(log);
}
