//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/IE/transforms/rewriters.hpp"
#include "vpux/compiler/dialect/IE/utils/convolution_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/permute_quantize_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/permute_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/reshape_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/slice_utils.hpp"
#include "vpux/compiler/dialect/const/utils/affine_reshape.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/range.hpp"

using namespace vpux;

namespace {

// Returns the size of the innermost non-trivial memory dimension of val
// (the last dim in memory order whose extent is > 1).
std::optional<int64_t> innermostNonTrivialMemDimSize(mlir::Value val) {
    const auto shape = getShape(val);
    const auto order = DimsOrder::fromValue(val);
    if (const auto dim = vpux::getInnermostNonTrivialDim(shape, order)) {
        return shape[*dim];
    }
    return std::nullopt;
}

// Returns true if every element of values equals expected.
bool allEqualTo(ArrayRef<int64_t> values, int64_t expected) {
    return llvm::all_of(values, [expected](int64_t v) {
        return v == expected;
    });
}

//
// MemPermuteRewriter
//

class MemPermuteRewriter final : public mlir::OpRewritePattern<IE::MemPermuteOp> {
public:
    MemPermuteRewriter(mlir::MLIRContext* ctx, Logger log, mlir::PatternBenefit benefit = 1)
            : mlir::OpRewritePattern<IE::MemPermuteOp>(ctx, benefit), _log(log) {
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

    auto layerWithPermute = IE::getFusableLayerWithPermuteInterface(origOp.getOperation());
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
                                      const DimsOrder& inOrder) -> std::optional<SmallVector<SmallVector<int64_t>>> {
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
    FuseMemPermuteThroughViewOps(mlir::MLIRContext* ctx, Logger log, mlir::PatternBenefit benefit = 1)
            : mlir::OpRewritePattern<IE::MemPermuteOp>(ctx, benefit), _log(log) {
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
            // Every view op must preserve the innermost non-trivial memory dimension so the
            // C dimension remains constant throughout the chain, allowing the H×W reshape.
            const auto sizeIn = innermostNonTrivialMemDimSize(parentOp->getOperand(0));
            const auto sizeOut = innermostNonTrivialMemDimSize(parentOp->getResult(0));
            if (!sizeIn.has_value() || !sizeOut.has_value() || *sizeIn != *sizeOut) {
                return std::nullopt;
            }
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

    if (!allEqualTo(parseIntArrayAttr<int64_t>(convOp.getStrides()), 1)) {
        return false;
    }

    if (!allEqualTo(parseIntArrayAttr<int64_t>(convOp.getPadsBegin()), 0) ||
        !allEqualTo(parseIntArrayAttr<int64_t>(convOp.getPadsEnd()), 0)) {
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

    // The per-step innermostNonTrivialDim check in retrieveConvOpThroughViewOps guarantees
    // by transitivity that the innermost non-trivial memory dimension is preserved from the
    // Conv/Slice output through all view ops to the MemPermute input.
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
    auto inShapeCast = rewriter.create<IE::ShapeCastOp>(appendLoc(convOp.getLoc(), "in_reshape"), newInputType,
                                                        convOp.getInput(), getIntArrayAttr(ctx, newConvInShapeVec));

    auto convType = mlir::cast<NDTypeInterface>(convOp.getOutput().getType()).changeShape(ShapeRef(newConvOutShapeVec));

    auto newConv = cloneConvolutionOp(rewriter, convOp, convType, inShapeCast, convOp.getFilter());
    // New MemPermute has the same dstOrder and memPerm attributes
    auto newPermute = rewriter.create<IE::MemPermuteOp>(origOp->getLoc(), newConv.getOutput(), origOp.getDstOrderAttr(),
                                                        origOp.getMemPermAttr());

    if (sliceOp == nullptr) {
        rewriter.replaceOp(origOp, newPermute.getOutput());
    } else {
        auto dstOrder = DimsOrder::fromAffineMap(origOp.getDstOrder());
        auto perm = origOp.getMemPerm();

        auto sliceAxis = getSliceAxes(sliceOp).front();
        auto newSliceAxis = inferDimAfterPermutation(Dim(sliceAxis), convOutOrder, dstOrder, perm);

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
// FusePermuteQuantizeThroughViewOps
//

// When IE::PermuteQuantizeOp acts as a pure transpose (same element type, no padding), and is
// separated from an IE::ConvolutionOp or IE::GroupConvolutionOp only by view ops that
// preserve the innermost memory dimension (C in NHWC), the PermuteQuantize can be replaced by a
// sequence of zero-cost operations placed immediately after the NCE op. The NCE op's input shape
// is kept unchanged, avoiding the H×W reshape that would otherwise produce an HW-unfriendly
// shape such as [128×1].
//
// Strategy: A MemPermute (NHWC→NCHW) placed adjacent to a cloned NCE op is fused into the NCE
// ODU as a free permutation by MemPermuteRewriter. A PermuteCast then reinterprets the NCHW
// output as NHWC with permuted spatial dims, and a final ShapeCast reinterprets the flat buffer
// to match the original PermuteQuantize output shape and layout.
//
// Correctness: Both the original PermuteQuantize and the new NHWC→NCHW MemPermute move the
// same innermost C=C_out NCE channels to the outer position. The view ops (PermuteCast +
// ShapeCast) do not move data; they only relabel the flat buffer. The required condition is
// C_out == W_pq (the innermost-dimension check below), which ensures the W dimension of the
// PermuteQuantize output directly indexes NCE channels, making the relabeling exact.
//
// Initial graph:
// 1x2048x32x4@NHWC   2048x1x1x1@NHWC
//         \               /
//    GroupConvolution
//               |
//       1x2048x32x4@NHWC
//               |
//        AffineReshape          [ViewOps preserving innermost C=2048]
//               |
//       1x2048x128x1@NHWC
//               |
//          PermuteCast
//               |
//       1x128x2048x1@NCHW
//               |
//        AffineReshape
//               |
//       1x128x1x2048@NCHW
//               |
// PermuteQuantize (f16→f16, pads=[0,0,0,0])
//               |
//       1x128x1x2048@NHWC
//
// After transformation:
// 1x2048x32x4@NHWC   2048x1x1x1@NHWC
//         \               /
//    GroupConvolution   ← input shape unchanged
//               |
//       1x2048x32x4@NHWC
//               |
//          MemPermute (NHWC→NCHW)       ← fused into NCE ODU by MemPermuteRewriter
//               |
//       1x2048x32x4@NCHW
//               |
//          PermuteCast (trivial)         ← NCHW[1,2048,32,4] → NHWC[1,4,2048,32]
//               |
//       1x4x2048x32@NHWC
//               |
//          ShapeCast                     ← flat buffer reinterpretation
//               |
//       1x128x1x2048@NHWC
//

// Holds the matched NCE convolution-like op (ConvolutionOp or GroupConvolutionOp) and the
// number of intervening pure view ops between it and the triggering PermuteQuantizeOp.
struct NceConvOpResult {
    mlir::Operation* nceOp{};
    int64_t viewOpsCount{};
};

class FusePermuteQuantizeThroughViewOps final : public mlir::OpRewritePattern<IE::PermuteQuantizeOp> {
public:
    FusePermuteQuantizeThroughViewOps(mlir::MLIRContext* ctx, Logger log, mlir::PatternBenefit benefit = 1)
            : mlir::OpRewritePattern<IE::PermuteQuantizeOp>(ctx, benefit), _log(log) {
        this->setDebugName("FusePermuteQuantizeThroughViewOps");
    }

private:
    mlir::LogicalResult matchAndRewrite(IE::PermuteQuantizeOp origOp, mlir::PatternRewriter& rewriter) const final;

    std::optional<NceConvOpResult> retrieveNceConvOpThroughViewOps(IE::PermuteQuantizeOp origOp) const;
    bool isValidPermuteQuantizeOp(IE::PermuteQuantizeOp permuteQuantize) const;

private:
    const size_t SUPPORTED_RANK = 4;
    Logger _log;
};

std::optional<NceConvOpResult> FusePermuteQuantizeThroughViewOps::retrieveNceConvOpThroughViewOps(
        IE::PermuteQuantizeOp origOp) const {
    int64_t viewOpsCnt = 0;
    auto parentOp = origOp.getInput().getDefiningOp();

    while (parentOp != nullptr && parentOp->hasOneUse()) {
        if (mlir::isa<IE::ConvolutionOp, IE::GroupConvolutionOp>(parentOp)) {
            // Matched ConvolutionOp or GroupConvolutionOp - [ViewOps] - PermuteQuantizeOp
            return NceConvOpResult{parentOp, viewOpsCnt};
        }

        if (!IE::isPureViewOp(parentOp) || mlir::isa<IE::QuantizeCastOp>(parentOp)) {
            return std::nullopt;
        }

        // Every view op must preserve the innermost non-trivial memory dimension.
        // Any op that changes it (e.g. a ShapeCast that alters C in NHWC) breaks the
        // C_out == W_pq equivalence required for the fusion and must stop the traversal.
        const auto sizeIn = innermostNonTrivialMemDimSize(parentOp->getOperand(0));
        const auto sizeOut = innermostNonTrivialMemDimSize(parentOp->getResult(0));
        if (!sizeIn.has_value() || !sizeOut.has_value() || *sizeIn != *sizeOut) {
            return std::nullopt;
        }

        parentOp = parentOp->getOperand(0).getDefiningOp();
        viewOpsCnt++;
    }

    return std::nullopt;
}

bool FusePermuteQuantizeThroughViewOps::isValidPermuteQuantizeOp(IE::PermuteQuantizeOp permuteQuantize) const {
    // Must act as a pure transpose: same element type and no padding
    const auto inElemType = mlir::cast<vpux::NDTypeInterface>(permuteQuantize.getInput().getType()).getElementType();
    if (!IE::isPurePermuteCompatiblePrecision(inElemType, permuteQuantize.getDstElemType())) {
        return false;
    }

    if (!allEqualTo(parseIntArrayAttr<int64_t>(permuteQuantize.getPadsBegin()), 0) ||
        !allEqualTo(parseIntArrayAttr<int64_t>(permuteQuantize.getPadsEnd()), 0)) {
        return false;
    }

    // Ensure 4D with batch size 1
    const auto inMemShape = getMemShape(permuteQuantize.getInput());
    if (inMemShape.size() != SUPPORTED_RANK || inMemShape.front() != 1) {
        return false;
    }

    // Exclude trivial permutations (no actual data rearrangement)
    if (isTrivialPermute(inMemShape, permuteQuantize.getMemPerm())) {
        return false;
    }

    // Keep split axes intact
    if (auto affineReshape = permuteQuantize.getInput().getDefiningOp<IE::AffineReshapeOp>()) {
        const auto dimMapping = parseIntArrayOfArrayAttr<int64_t>(affineReshape.getDimMapping());
        const auto perm = DimsOrder::fromAffineMap(permuteQuantize.getMemPerm()).toPermutation();
        SmallVector<int64_t> permAxis;
        for (const auto dim : perm) {
            permAxis.push_back(checked_cast<int64_t>(dim.ind()));
        }

        const auto reshapeOutputType = mlir::cast<NDTypeInterface>(affineReshape.getOutput().getType());
        const MemShape reshapeOutputMemShape(reshapeOutputType.getMemShape());
        if (!areReshapedAxesPermutedIntegratedly(dimMapping, permAxis, reshapeOutputType.getDimsOrder(),
                                                 reshapeOutputMemShape)) {
            return false;
        }
    }

    return true;
}

mlir::LogicalResult FusePermuteQuantizeThroughViewOps::matchAndRewrite(IE::PermuteQuantizeOp origOp,
                                                                       mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    if (!isValidPermuteQuantizeOp(origOp)) {
        _log.nest().trace("Transformation aborted: PermuteQuantizeOp validation failed.");
        return mlir::failure();
    }

    const auto result = retrieveNceConvOpThroughViewOps(origOp);
    if (!result.has_value()) {
        _log.nest().trace("Transformation skipped: Pattern match failed.");
        return mlir::failure();
    }

    if (result.value().viewOpsCount == 0) {
        _log.nest().trace("Transformation skipped: PermuteQuantizeOp follows NCE conv op directly.");
        return mlir::failure();
    }

    auto* nceOp = result.value().nceOp;
    // Require NHWC output so the NHWC→NCHW MemPermute can be placed directly after the NCE op.
    if (!mlir::isa<IE::ConvolutionOp, IE::GroupConvolutionOp>(nceOp) ||
        mlir::cast<NDTypeInterface>(nceOp->getResult(0).getType()).getDimsOrder() != DimsOrder::NHWC) {
        _log.nest().trace("Transformation aborted: NCE op output is not NHWC.");
        return mlir::failure();
    }

    auto permuteInterface = mlir::dyn_cast<IE::LayerWithPermuteInterface>(nceOp);
    if (permuteInterface == nullptr) {
        return mlir::failure();
    }

    auto ctx = rewriter.getContext();

    // The MemPermute (NHWC→NCHW) placed adjacent to the NCE op is the operation that
    // MemPermuteRewriter will fuse into the NCE ODU as a free permutation.
    // mem_perm (d0,d3,d1,d2) maps NHWC input memory dims [N,H,W,C] to NCHW output [N,C,H,W].
    const auto nhwcToNchwPerm = mlir::AffineMap::getPermutationMap(SmallVector<unsigned>{0, 3, 1, 2}, ctx);
    const auto nchwDstOrder = mlir::AffineMap::getPermutationMap(SmallVector<unsigned>{0, 1, 2, 3}, ctx);

    // Verify ODU support using a temporary MemPermute with the NHWC→NCHW attributes.
    auto tempMemPermute = rewriter.create<IE::MemPermuteOp>(origOp->getLoc(), nceOp->getResult(0),
                                                            mlir::AffineMapAttr::get(nchwDstOrder),
                                                            mlir::AffineMapAttr::get(nhwcToNchwPerm));
    const bool isPermutationSupported = permuteInterface.isSupportedPermutation(tempMemPermute);
    rewriter.eraseOp(tempMemPermute);
    if (!isPermutationSupported) {
        _log.nest().trace("Transformation skipped: Unsupported ODU permutation.");
        return mlir::failure();
    }

    // Clone the NCE op with the same inputs and output type (no spatial reshape). The clone
    // becomes the exclusive producer consumed by the new MemPermute, satisfying the single-use
    // requirement of MemPermuteRewriter. The original NCE→view-ops→PermuteQuantize chain
    // becomes dead code and is eliminated by DCE.
    mlir::Value newNceOut;
    if (auto convOp = mlir::dyn_cast<IE::ConvolutionOp>(nceOp)) {
        newNceOut = cloneConvolutionOp(rewriter, convOp, convOp.getOutput().getType(), convOp.getInput(),
                                       convOp.getFilter())
                            .getOutput();
    } else {
        auto groupConvOp = mlir::cast<IE::GroupConvolutionOp>(nceOp);
        newNceOut =
                rewriter.create<IE::GroupConvolutionOp>(
                                groupConvOp.getLoc(), groupConvOp.getOutput().getType(), groupConvOp.getInput(),
                                groupConvOp.getFilter(), groupConvOp.getBias(), groupConvOp.getStridesAttr(),
                                groupConvOp.getPadsBegin(), groupConvOp.getPadsEnd(), groupConvOp.getDilations(),
                                groupConvOp.getGroupsAttr(), groupConvOp.getPostOpAttr(), groupConvOp.getClampAttr(),
                                groupConvOp.getOutputPaddingAttr(), groupConvOp.getInputPaddingAttr())
                        .getOutput();
    }

    // MemPermute: NHWC[1,C_out,H,W] → NCHW[1,C_out,H,W]
    // Fused into the NCE ODU by MemPermuteRewriter in a subsequent iteration.
    auto newMemPermute =
            rewriter.create<IE::MemPermuteOp>(origOp->getLoc(), newNceOut, mlir::AffineMapAttr::get(nchwDstOrder),
                                              mlir::AffineMapAttr::get(nhwcToNchwPerm));

    // PermuteCast + ShapeCast: restore the original PermuteQuantize output shape and layout.
    const auto nhwcDstOrder = mlir::AffineMap::getPermutationMap(SmallVector<unsigned>{0, 2, 3, 1}, ctx);
    const auto identityPerm = mlir::AffineMap::getPermutationMap(SmallVector<unsigned>{0, 1, 2, 3}, ctx);
    auto newPermuteCast = rewriter.create<IE::PermuteCastOp>(origOp->getLoc(), newMemPermute.getOutput(),
                                                             mlir::AffineMapAttr::get(nhwcDstOrder),
                                                             mlir::AffineMapAttr::get(identityPerm));

    const auto pqOutShape = to_small_vector(getShape(origOp.getOutput()));
    auto newShapeCast = rewriter.create<IE::ShapeCastOp>(origOp->getLoc(), origOp.getOutput().getType(),
                                                         newPermuteCast.getOutput(), getIntArrayAttr(ctx, pqOutShape));

    rewriter.replaceOp(origOp, newShapeCast.getOutput());
    return mlir::success();
}

}  // namespace

void vpux::IE::registerFuseMemPermuteRewriters(RewriterRegistry& registry, ArrayRef<mlir::PatternBenefit> benefitLevels,
                                               size_t index, Logger log) {
    registry.registerRewriterSet("fuse-mem-permute-set", [&registry, log, benefitLevels, index]() {
        registry.registerRewriter<MemPermuteRewriter>("mem-permute-rewriter", log, benefitLevels[index]);
        registry.registerRewriter<FuseMemPermuteThroughViewOps>("fuse-mem-permute-through-view-ops", log,
                                                                benefitLevels[index]);
        registry.registerRewriter<FusePermuteQuantizeThroughViewOps>("fuse-permute-quantize-through-view-ops", log,
                                                                     benefitLevels[index]);
    });
}
