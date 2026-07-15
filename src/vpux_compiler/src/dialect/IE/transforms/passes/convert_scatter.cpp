//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <optional>

#include <mlir/Transforms/DialectConversion.h>
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

namespace vpux::IE {
#define GEN_PASS_DECL_CONVERTSCATTER
#define GEN_PASS_DEF_CONVERTSCATTER
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

bool isIdentityMapBetweenInputAndUpdates(Const::details::ContentRange<int64_t>& indicesData, mlir::Value updates,
                                         ArrayRef<int64_t> dimsToScale, ArrayRef<int64_t> scaleFactors) {
    // check elementwise indices equal to stride operation.
    // e.g. input shape 1x3x40x40x15, indices 1x3x40x40x5x5, updates shape 1x3x40x40x5
    // check indices last dim 5 values could meet offset and stride operation.

    const auto isDimToUpdate = [&](auto dim) {
        return llvm::count_if(dimsToScale, [&](auto dimToScale) {
                   return dimToScale == dim;
               }) != 0;
    };

    const auto updatesType = mlir::cast<vpux::NDTypeInterface>(updates.getType());
    const auto elemSize = updatesType.getElemTypeSize().count();
    const auto updateStride = to_small_vector(getStrides(updates) | transformed([&](Bit stride) {
                                                  return stride.count() / elemSize;
                                              }));

    const int64_t inRank = updatesType.getRank();
    for (int64_t indiceIndex = 0; indiceIndex < static_cast<int64_t>(indicesData.size()); indiceIndex += inRank) {
        int64_t locationInUpdate = 0;
        for (int64_t dim = 0; dim < inRank; dim++) {
            auto scaleAtDim = 1;
            auto offsetAtDim = 0;
            if (isDimToUpdate(dim)) {
                scaleAtDim = scaleFactors[dim];
                offsetAtDim = indicesData[dim];
            }

            locationInUpdate =
                    updateStride[dim] * (indicesData[indiceIndex + dim] - offsetAtDim) / scaleAtDim + locationInUpdate;
        }
        if (locationInUpdate != indiceIndex / inRank) {
            return false;
        }
    }
    return true;
}

class ConvertScatterPass final : public IE::impl::ConvertScatterBase<ConvertScatterPass> {
public:
    explicit ConvertScatterPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

public:
    class ConvertScatterNDUpdateToStridedConcat;
    class ConvertScatterNDUpdateToSliceConcat;
    class ConvertScatterNDUpdateArithmeticToConcat;
    class ConvertNDUpdateDataToSliceConcat;
    class ConvertScatterNDUpdateColumnSelectToConcat;
    class SplitToMultiScatterNDUpdateOp;
    class ConvertScatterElementsUpdateToAddMultiply;
    class MergeChainedScatterElementsUpdate;

private:
    void safeRunOnFunc() final;
};

//
// ConvertScatterNDUpdateToStridedConcat
//

class ConvertScatterPass::ConvertScatterNDUpdateToStridedConcat final :
        public mlir::OpRewritePattern<IE::ScatterNDUpdateOp> {
public:
    ConvertScatterNDUpdateToStridedConcat(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::ScatterNDUpdateOp>(ctx), _log(log) {
        setDebugName("ConvertScatterNDUpdateToStridedConcat");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ScatterNDUpdateOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

// For example, if there is a 1x15 tensor: [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
// The Indices to update data is [0,3,6,9,12] . The data to update is [fx0,fx1,fx2,fx3,fx4]
// The results is [fx0,2,3,fx1,5,6,fx2,8,9,fx3,11,12,fx4,14,15].
// It equals to offset 0, stride 3, strided concat.

// For example that 2 dimensions to update data
// input data @shape<4x4>:
//     [[1, 2, 3, 4],
//      [5, 6, 7, 8]]
//     [[9, A, B, C],
//      [D, E, F, G]]
//
// Indices @shape<2x2x2>:
//     [[(0, 0), (0, 2)],
//      [(2, 0), (2, 2)]]
//
// updates @shape<2x2>:
//     [[u1, u2],
//      [u3, u4]]
//
// Steps:
// #1. upsampling updates at dim 1.
//     [[u1, 0, u2, 0],
//      [u3, 0, u4, 0]]
// #2. strided slice input at dim 0
//      [[5, 6, 7, 8]]
//       [D, E, F, G]]
// #3. perAxis concat #1 and #2 at dim 0
//     [[u1, 0, u2, 0],
//      [ 5, 6,  7, 8]]
//     [[u3, 0, u4, 0],
//      [ D, E,  F, G]]
// #4. strided slice #3 at dim 1
//     [[u1, u2],
//      [ 5,  7]]
//     [[u3, u4],
//      [ D,  F]]
// #5. strided slice input data at dim 1
//     [[2, 4],
//      [6, 8]]
//     [[A, C],
//      [E, G]]
//
// #6. perAxis concat #4 and #5 at dim 1
//     [[u1, 2, u2, 4],
//      [ 5, 6,  7, 8]]
//     [[u3, A, u4, C],
//      [ D, E,  F, G]]
//
// Got the final result.
//
// The sub-graph looks like something below:
//
//         input-data           updates
//         /        \              |
//  StridedSlice StridedSlice   Upsampling
//        |               \       /
//        |            Strided-Concat
//        |                   |
//        |             StridedSlice
//         \                /
//          \              /
//           Strided-Concat
//

mlir::LogicalResult ConvertScatterPass::ConvertScatterNDUpdateToStridedConcat::matchAndRewrite(
        IE::ScatterNDUpdateOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp.getLoc());

    const auto origInType = mlir::cast<vpux::NDTypeInterface>(origOp.getInput().getType());
    const auto inputShape = origInType.getShape();
    const auto origInRank = origInType.getRank();
    const auto indices = origOp.getIndices();
    const auto indicesShape = getShape(indices);
    const auto indicesRank = checked_cast<int64_t>(indicesShape.size());

    if (indicesShape.back() != origInRank || indicesRank - 1 != origInRank) {
        _log.trace("only elementwise case supported");
        return mlir::failure();
    }

    auto indicesConst = indices.getDefiningOp<Const::DeclareOp>();
    if (indicesConst == nullptr) {
        _log.trace("only constant indices case supported");
        return mlir::failure();
    }

    SmallVector<int64_t> scaleFactors;
    SmallVector<int64_t> dimsToScale;
    for (int64_t i = 0; i < origInRank; i++) {
        // check potential stride.
        // if not integer stride return
        if (inputShape[Dim(i)] % indicesShape[Dim(i)] != 0) {
            _log.trace("the scale factor is not integer");
            return mlir::failure();
        }

        auto factor = inputShape[Dim(i)] / indicesShape[Dim(i)];
        scaleFactors.push_back(factor);
        if (factor > 1) {
            dimsToScale.push_back(i);
        }
    }

    if (dimsToScale.size() > 2) {
        _log.trace("only 1D or 2D dimensions to update supported");
        return mlir::failure();
    }

    // For example, Input shape 1x10x1, indices shape 1x1x1x3
    // indices data [[[[0, 0, 0]]]]
    // This case will be handled by ConvertToSliceConcat Rewriter with 1 Slice
    // Otherwise it will need 10 Slice by ConvertToStridedConcat Rewriter
    if (dimsToScale.size() == 1 && indicesShape[Dim(dimsToScale.front())] == 1) {
        _log.trace("it's beneficial to handle this case by ConvertToSliceConcat");
        return mlir::failure();
    }

    if (dimsToScale.size() == 2 && origInRank != 4) {
        _log.trace("only 4D input supported due to UpsamplingOp needed");
        return mlir::failure();
    }

    const auto indicesConstValue = indicesConst.getContent();
    auto indicesData = indicesConstValue.getValues<int64_t>();

    if (!isIdentityMapBetweenInputAndUpdates(indicesData, origOp.getUpdates(), dimsToScale, scaleFactors)) {
        _log.trace("It's not the identity map between input and updates");
        return mlir::failure();
    }

    auto ctx = rewriter.getContext();

    const auto getUpsamplingOpAtDim = [&](int64_t dim, int64_t factor, bool padAtEnd) -> mlir::Value {
        const auto numToPad = factor - 1;
        auto padAtDim = SmallVector<int64_t>{numToPad, 0};
        if (padAtEnd) {
            padAtDim = SmallVector<int64_t>{0, numToPad};
        }

        const auto padAtDimAttr = getIntArrayAttr(ctx, padAtDim);
        const auto padZeroAttr = getIntArrayAttr(ctx, SmallVector<int64_t>{0, 0});

        IE::UpsamplingPadAttr padAttr;
        SmallVector<int64_t> upsamplingFactor;
        if (dim == Dims4D::Act::H.ind()) {
            padAttr = IE::UpsamplingPadAttr::get(ctx, padZeroAttr, padAtDimAttr, padZeroAttr);
            upsamplingFactor = SmallVector<int64_t>{1, factor, 1};
        } else if (dim == Dims4D::Act::W.ind()) {
            padAttr = IE::UpsamplingPadAttr::get(ctx, padZeroAttr, padZeroAttr, padAtDimAttr);
            upsamplingFactor = SmallVector<int64_t>{factor, 1, 1};
        } else if (dim == Dims4D::Act::C.ind()) {
            padAttr = IE::UpsamplingPadAttr::get(ctx, padAtDimAttr, padZeroAttr, padZeroAttr);
            upsamplingFactor = SmallVector<int64_t>{1, 1, factor};
        } else {
            _log.error("illegal dimension {0}", dim);
        }

        const auto upsamplingFactorAttr = getIntArrayAttr(ctx, upsamplingFactor);
        return rewriter
                .create<IE::UpsamplingOp>(takeOpLoc(origOp, "upsample"), origOp.getUpdates(), upsamplingFactorAttr,
                                          padAttr, nullptr, nullptr)
                .getOutput();
    };

    const auto createSlice = [&](mlir::Value input, int64_t offset, int64_t stride, int64_t dim) {
        auto offsetValues = SmallVector<int64_t>(origInRank, 0);
        offsetValues[dim] = offset;
        auto strideValues = SmallVector<int64_t>(origInRank, 1);
        strideValues[dim] = stride;

        const auto zeros = SmallVector<int64_t>(origInRank, 0);
        const auto stridesAttr = getIntArrayAttr(ctx, ArrayRef(strideValues));
        const auto beginsAttr = getIntArrayAttr(ctx, ArrayRef(offsetValues));
        const auto endsAttr = getIntArrayAttr(ctx, inputShape);
        const auto zeroMask = getIntArrayAttr(ctx, ArrayRef(zeros));

        return rewriter.create<IE::StridedSliceOp>(takeOpLoc(origOp, StringLiteral("slice_{0}_{1}"), offset, dim),
                                                   input, nullptr, nullptr, nullptr, beginsAttr, endsAttr, stridesAttr,
                                                   /*beginMask =*/zeroMask, /*endMask =*/zeroMask,
                                                   /*newAxisMask =*/zeroMask,
                                                   /*shrinkAxisMask =*/zeroMask, /*ellipsisMask = */ zeroMask);
    };

    mlir::Value inputValue = origOp.getUpdates();
    int64_t dimToUpsampling = -1;
    if (dimsToScale.size() > 1) {
        // dimToUpsampling must be one of C/H/W
        dimToUpsampling = dimsToScale.back();
        bool isLegalDimForUpsampling =
                llvm::count_if(SmallVector<Dim>{Dims4D::Act::C, Dims4D::Act::H, Dims4D::Act::W}, [&](const auto dim) {
                    return Dim(dimToUpsampling) == dim;
                }) != 0;
        if (!isLegalDimForUpsampling) {
            _log.trace("illegal dimension {0} for upsampling found", dimToUpsampling);
            return mlir::failure();
        }
        const auto offsetAtUpsamplingDim = indicesData[dimToUpsampling];
        inputValue = getUpsamplingOpAtDim(dimToUpsampling, scaleFactors[dimToUpsampling], offsetAtUpsamplingDim == 0);
    }

    for (const auto axisIndex : dimsToScale) {
        const auto offsetAtAxis = indicesData[axisIndex];
        const auto factorAtAxis = scaleFactors[axisIndex];

        SmallVector<mlir::Value> subSlices;
        for (const auto ind : irange(factorAtAxis)) {
            if (ind == offsetAtAxis) {
                mlir::Value updatesData = inputValue;
                if (axisIndex == dimToUpsampling) {
                    updatesData = createSlice(inputValue, ind, factorAtAxis, axisIndex);
                }
                subSlices.push_back(updatesData);
            } else {
                auto stridedSliceOp = createSlice(origOp.getInput(), ind, factorAtAxis, axisIndex);
                subSlices.push_back(stridedSliceOp);
            }
        }
        inputValue = rewriter.create<IE::ConcatOp>(takeOpLoc(origOp, StringLiteral("concat_over_{0}"), axisIndex),
                                                   subSlices, axisIndex, 1, scaleFactors[axisIndex])
                             .getOutput();
    }

    _log.trace("{0} is replaced with Upsampling+Slice+Concat", origOp.getLoc());
    rewriter.replaceOp(origOp, inputValue);

    return mlir::success();
}

//
// ConvertScatterNDUpdateToSliceConcat
//

class ConvertScatterPass::ConvertScatterNDUpdateToSliceConcat final :
        public mlir::OpRewritePattern<IE::ScatterNDUpdateOp> {
public:
    ConvertScatterNDUpdateToSliceConcat(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::ScatterNDUpdateOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ScatterNDUpdateOp origOp, mlir::PatternRewriter& rewriter) const final;

    std::optional<Dim> getUpdateDim(ShapeRef inputShape, ShapeRef indicesShape, Const::DeclareOp indicesConst) const;

private:
    Logger _log;
};

std::optional<Dim> ConvertScatterPass::ConvertScatterNDUpdateToSliceConcat::getUpdateDim(
        ShapeRef inputShape, ShapeRef indicesShape, Const::DeclareOp indicesConst) const {
    const auto indicesConstValue = indicesConst.getContent();
    const auto indicesData = indicesConstValue.getValues<int64_t>();

    const auto greaterThanOne = [](auto dimSize) {
        return dimSize > 1;
    };

    // Scenario 1: Elements Update
    // For example, Input shape 1x32x1, indices shape 1x3x1x3
    // indices data [[[[0, 5, 0], [0, 6, 0], [0, 7, 0]]]]
    // The updateDim will be Dim(1)
    const auto inputRank = checked_cast<int64_t>(inputShape.size());
    const auto indicesRank = checked_cast<int64_t>(indicesShape.size());
    if (indicesShape.back() == inputRank && indicesRank - 1 == inputRank) {
        const auto inputShapeGreaterThanOne = llvm::count_if(inputShape, greaterThanOne);
        const auto indicesShapeGreaterThanOne =
                std::count_if(indicesShape.begin(), indicesShape.end() - 1, greaterThanOne);
        if (inputShapeGreaterThanOne > 1 || indicesShapeGreaterThanOne > 1) {
            _log.trace("Elements Update: Only support ScatterNDUpdate Op update at one axis");
            return std::nullopt;
        }

        // Input shape 1x1x1, indices shape 1x1x1x3
        // indices data [[[[0, 0, 0]]]]
        // The updateDim will be Dim(0)
        if (inputShapeGreaterThanOne == 0 && indicesShapeGreaterThanOne == 0) {
            return Dim(0);
        }

        auto axis = llvm::find_if(inputShape, greaterThanOne);
        VPUX_THROW_UNLESS(axis != inputShape.end(), "Can not get correct Axis");
        auto updateDim = std::distance(inputShape.begin(), axis);

        const auto beginOffset = indicesData[updateDim];
        for (auto idx = 1; idx < indicesShape[Dim(updateDim)]; idx++) {
            if (indicesData[updateDim + inputRank * idx] != beginOffset + idx) {
                _log.trace("Elements Update: The data in indices and at the updateDim should be increase with step 1");
                return std::nullopt;
            }
        }

        return Dim(updateDim);
    }

    // Scenario 2: Tensor Update
    // For example, Input shape 16x32x64, indices shape 3x1
    // indices data [[5], [6], [7]]
    // The updateDim will be Dim(0)
    if (indicesShape.back() == 1) {
        const auto beginOffset = indicesData.front();
        for (auto idx = 1; idx < indicesShape.totalSize(); idx++) {
            if (indicesData[idx] != beginOffset + idx) {
                _log.trace("Tensor Update: The data in indices and at the updateDim should be increase with step 1");
                return std::nullopt;
            }
        }
        return Dim(0);
    }

    return std::nullopt;
}

// There are two possible patterns can be converted
// Scenario 1: Elements Update, it has the following limitations:
// - indices.shape[-1] = input.shape.rank
// - Only has one updateDim
// - All dim size for indices shape[:-1] should be 1 except the updateDim
// - All dim size for input shape should be 1 except the updateDim
// - The data in indices and at the updateDim should be increase with step 1
// For example, Input shape 1x32x1, indices shape 1x3x1x3
// indices data [[[[0, 5, 0], [0, 6, 0], [0, 7, 0]]]]

// Scenario 2: Tensor Update, it has the following limitations:
// - indices.shape[-1] = 1, if not the update data shape rank will not same with input
// - The data in indices should be increase with step 1
// For example, Input shape 16x32x64, indices shape 3x1
// indices data [[5], [6], [7]]
mlir::LogicalResult ConvertScatterPass::ConvertScatterNDUpdateToSliceConcat::matchAndRewrite(
        IE::ScatterNDUpdateOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    const auto inputShape = getShape(origOp.getInput());
    const auto indices = origOp.getIndices();
    const auto indicesShape = getShape(indices);
    auto indicesConst = indices.getDefiningOp<Const::DeclareOp>();
    if (indicesConst == nullptr) {
        _log.trace("ScatterNDUpdate Op should with constant indices");
        return mlir::failure();
    }

    auto dimValue = getUpdateDim(inputShape, indicesShape, indicesConst);
    if (!dimValue.has_value()) {
        _log.trace("ScatterNDUpdate Op can not convert to Slice and Concat");
        return mlir::failure();
    }
    auto updateDim = dimValue.value().ind();

    const auto indicesConstValue = indicesConst.getContent();
    const auto indicesData = indicesConstValue.getValues<int64_t>();
    auto beginOffset = indicesData[updateDim];

    SmallVector<mlir::Value> concatInputs;
    // Create the left Slice Op
    auto leftSliceOffset = SmallVector<int64_t>(inputShape.size(), 0);
    auto leftSliceShape = to_small_vector(inputShape.raw());
    leftSliceShape[updateDim] = beginOffset;

    if (beginOffset != 0) {
        concatInputs.push_back(rewriter.create<IE::SliceOp>(takeOpLoc(origOp, "slice_left"), origOp.getInput(),
                                                            leftSliceOffset, leftSliceShape)
                                       .getResult());
    }

    // Update data value
    concatInputs.push_back(origOp.getUpdates());

    // Create the right Slice Op
    auto endOffset = beginOffset + indicesShape[Dim(updateDim)];
    auto rightSliceOffset = SmallVector<int64_t>(inputShape.size(), 0);
    rightSliceOffset[updateDim] = endOffset;
    auto rightSliceShape = to_small_vector(inputShape.raw());
    rightSliceShape[updateDim] = rightSliceShape[updateDim] - endOffset;

    if (rightSliceShape[updateDim] != 0) {
        concatInputs.push_back(rewriter.create<IE::SliceOp>(takeOpLoc(origOp, "slice_right"), origOp.getInput(),
                                                            rightSliceOffset, rightSliceShape)
                                       .getResult());
    }

    _log.trace("Replace '{0}' at '{1}' with Slice and Concat Op", origOp->getName(), origOp->getLoc());
    rewriter.replaceOpWithNewOp<IE::ConcatOp>(origOp, concatInputs, updateDim);

    return mlir::success();
}

//
// ConvertScatterNDUpdateArithmeticToConcat
//
// Rewrites a rank-1 ScatterNDUpdate whose constant indices form a regular strided grid
//
//     indices[i*M + j] = base + i*outerStride + j*innerStride   for i in [0, N), j in [0, M)
//
// into Slice + Reshape + Concat. A 1D arithmetic progression is the special case N == 1.
// Requires innerStride >= 2 and outerStride >= M*innerStride (non-overlapping rows) - exactly the
// gap left by the existing patterns: ConvertScatterNDUpdateToSliceConcat handles stride 1 only, and
// ConvertScatterNDUpdateToStridedConcat requires input_dim % indices_count == 0.
//
// Example : input tensor<8192xf32>, indices base=1, innerStride=3, M=20,
// outerStride=64, N=128, updates tensor<2560xf32>.
//

class ConvertScatterPass::ConvertScatterNDUpdateArithmeticToConcat final :
        public mlir::OpRewritePattern<IE::ScatterNDUpdateOp> {
public:
    ConvertScatterNDUpdateArithmeticToConcat(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::ScatterNDUpdateOp>(ctx), _log(log) {
        setDebugName("ConvertScatterNDUpdateArithmeticToConcat");
    }

    mlir::LogicalResult matchAndRewrite(IE::ScatterNDUpdateOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

namespace {

// Description of a 1D / 2D regular strided index pattern.
struct StridedIndexLayout {
    int64_t base = 0;
    int64_t innerStride = 0;
    int64_t innerCount = 0;
    int64_t outerStride = 0;
    int64_t outerCount = 1;
};

// Detect that `indicesData` is laid out as:
//     indices[i*innerCount + j] = base + i*outerStride + j*innerStride
// Returns std::nullopt if the data does not match.
std::optional<StridedIndexLayout> detectStridedLayout(ArrayRef<int64_t> indicesData) {
    const auto total = checked_cast<int64_t>(indicesData.size());
    if (total < 2) {
        return std::nullopt;
    }

    StridedIndexLayout layout;
    layout.base = indicesData[0];
    layout.innerStride = indicesData[1] - indicesData[0];
    if (layout.innerStride < 2) {
        return std::nullopt;
    }

    // Walk along the inner dimension while the diff stays equal to innerStride.
    int64_t innerLen = 1;
    while (innerLen < total && indicesData[innerLen] - indicesData[innerLen - 1] == layout.innerStride) {
        ++innerLen;
    }
    layout.innerCount = innerLen;

    if (total % layout.innerCount != 0) {
        return std::nullopt;
    }
    layout.outerCount = total / layout.innerCount;

    if (layout.outerCount == 1) {
        layout.outerStride = 0;
    } else {
        layout.outerStride = indicesData[layout.innerCount] - indicesData[0];
        if (layout.outerStride < layout.innerCount * layout.innerStride) {
            // Rows would overlap.
            return std::nullopt;
        }
    }

    // Validate every element matches the predicted formula.
    for (int64_t i = 0; i < layout.outerCount; ++i) {
        const int64_t rowBase = layout.base + i * layout.outerStride;
        for (int64_t j = 0; j < layout.innerCount; ++j) {
            if (indicesData[i * layout.innerCount + j] != rowBase + j * layout.innerStride) {
                return std::nullopt;
            }
        }
    }

    return layout;
}

}  // namespace

mlir::LogicalResult ConvertScatterPass::ConvertScatterNDUpdateArithmeticToConcat::matchAndRewrite(
        IE::ScatterNDUpdateOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp.getLoc());

    const auto ctx = rewriter.getContext();
    const auto inputType = mlir::cast<vpux::NDTypeInterface>(origOp.getInput().getType());
    const auto inputShape = inputType.getShape();
    const auto indicesShape = getShape(origOp.getIndices());
    const auto updatesShape = getShape(origOp.getUpdates());

    if (inputType.getRank() != 1) {
        _log.trace("Only rank-1 input is supported");
        return mlir::failure();
    }

    if (indicesShape.size() != 2 || indicesShape.back() != 1) {
        _log.trace("Indices shape must be [N, 1]");
        return mlir::failure();
    }

    auto indicesConstOp = origOp.getIndices().getDefiningOp<Const::DeclareOp>();
    if (indicesConstOp == nullptr) {
        _log.trace("Indices must be constant");
        return mlir::failure();
    }

    const auto numIndices = indicesShape[Dim(0)];
    if (numIndices < 2) {
        _log.trace("Need at least 2 indices to determine a strided pattern");
        return mlir::failure();
    }

    if (updatesShape.size() != 1 || updatesShape[Dim(0)] != numIndices) {
        _log.trace("Updates must be rank-1 with size {0}", numIndices);
        return mlir::failure();
    }

    const auto indicesContent = indicesConstOp.getContent();
    const auto indicesData = indicesContent.getValues<int64_t>();
    if (checked_cast<int64_t>(indicesData.size()) != numIndices) {
        _log.trace("Indices content size {0} does not match shape size {1}", indicesData.size(), numIndices);
        return mlir::failure();
    }

    SmallVector<int64_t> indicesBuffer(indicesData.begin(), indicesData.end());
    auto layoutOpt = detectStridedLayout(indicesBuffer);
    if (!layoutOpt.has_value()) {
        _log.trace("Indices are not a 1D or 2D regular strided grid");
        return mlir::failure();
    }
    const auto layout = layoutOpt.value();

    if (layout.base < 0) {
        _log.trace("First index must be non-negative");
        return mlir::failure();
    }

    // The minimum input length the layout requires is the end of the last stride block:
    //     [base, base + (N-1)*outerStride + M*innerStride)
    // Anything beyond that point goes into the tail and stays unchanged. The last row's
    // keep region is therefore allowed to be partial or even absent.
    const int64_t strideBlockWidth = layout.innerCount * layout.innerStride;
    const int64_t bodyEnd = layout.base + (layout.outerCount - 1) * layout.outerStride + strideBlockWidth;
    const int64_t inputSize = inputShape[Dim(0)];
    if (bodyEnd > inputSize) {
        _log.trace("Required body region end {0} exceeds input size {1}", bodyEnd, inputSize);
        return mlir::failure();
    }

    const auto loc = origOp->getLoc();
    SmallVector<mlir::Value> outerConcatInputs;

    // 1D Slice of `size` elements starting at `offset`.
    auto sliceFlat = [&](mlir::Value src, int64_t offset, int64_t size, StringRef name) -> mlir::Value {
        return rewriter
                .create<IE::SliceOp>(takeOpLoc(origOp, name), src, SmallVector<int64_t>{offset},
                                     SmallVector<int64_t>{size})
                .getResult();
    };

    // Rewrite a batch of `rows` rows, each `rowWidth` wide and laid out contiguously in `bodyFlat`.
    // Within every row the first M*innerStride columns are M groups of `innerStride`; column 0 of
    // each group is replaced by the matching update and the other innerStride-1 columns are kept.
    // The trailing (rowWidth - M*innerStride) columns (the inter-row gap) are kept unchanged.
    // Returns the rewritten rows flattened back to [rows*rowWidth]. The 1D progression is rows == 1.
    auto rewriteRowBatch = [&](mlir::Value bodyFlat, mlir::Value updatesFlat, int64_t rows, int64_t rowWidth,
                               StringRef suffix) -> mlir::Value {
        auto reshape = [&](mlir::Value src, ArrayRef<int64_t> shape, StringRef tag) -> mlir::Value {
            return rewriter
                    .create<IE::ReshapeOp>(takeOpLoc(origOp, StringLiteral("arith_{0}_{1}"), tag, suffix), src,
                                           getIntArrayAttr(ctx, shape))
                    .getOutput();
        };

        auto body2D = reshape(bodyFlat, {rows, rowWidth}, "body");
        auto strideRegion =
                rewriter.create<IE::SliceOp>(takeOpLoc(origOp, StringLiteral("arith_stride_{0}"), suffix), body2D,
                                             SmallVector<int64_t>{0, 0}, SmallVector<int64_t>{rows, strideBlockWidth});
        // Merge the row and group dims: the per-group interleave is independent of the row, so all
        // (rows * M) groups are handled by a single 2D op - keep columns 1..innerStride-1 and take
        // column 0 from the updates. (1D progression is rows == 1, i.e. just M groups.)
        const int64_t numGroups = rows * layout.innerCount;
        auto groups2D = reshape(strideRegion.getResult(), {numGroups, layout.innerStride}, "groups");
        auto groupKeep = rewriter.create<IE::SliceOp>(takeOpLoc(origOp, StringLiteral("arith_keep_{0}"), suffix),
                                                      groups2D, SmallVector<int64_t>{0, 1},
                                                      SmallVector<int64_t>{numGroups, layout.innerStride - 1});
        auto upd2D = reshape(updatesFlat, {numGroups, 1}, "upd");
        auto newGroups2D =
                rewriter.create<IE::ConcatOp>(takeOpLoc(origOp, StringLiteral("arith_newgroups_{0}"), suffix),
                                              SmallVector<mlir::Value>{upd2D, groupKeep.getResult()}, /*axis=*/1);

        mlir::Value rowResult = reshape(newGroups2D.getOutput(), {rows, strideBlockWidth}, "newstride2d");
        if (rowWidth > strideBlockWidth) {
            auto gap = rewriter.create<IE::SliceOp>(takeOpLoc(origOp, StringLiteral("arith_gap_{0}"), suffix), body2D,
                                                    SmallVector<int64_t>{0, strideBlockWidth},
                                                    SmallVector<int64_t>{rows, rowWidth - strideBlockWidth});
            rowResult = rewriter.create<IE::ConcatOp>(takeOpLoc(origOp, StringLiteral("arith_row_{0}"), suffix),
                                                      SmallVector<mlir::Value>{rowResult, gap.getResult()}, /*axis=*/1)
                                .getOutput();
        }
        return reshape(rowResult, {rows * rowWidth}, "rowflat");
    };

    if (layout.base > 0) {
        outerConcatInputs.push_back(sliceFlat(origOp.getInput(), 0, layout.base, "arith_head"));
    }

    // The first (N-1) rows have the full outerStride width (the gap is kept). Skipped when N == 1.
    const int64_t headRows = layout.outerCount - 1;
    if (headRows > 0) {
        auto headBody = sliceFlat(origOp.getInput(), layout.base, headRows * layout.outerStride, "arith_head_body");
        auto headUpd = sliceFlat(origOp.getUpdates(), 0, headRows * layout.innerCount, "arith_head_upd");
        outerConcatInputs.push_back(rewriteRowBatch(headBody, headUpd, headRows, layout.outerStride, "head"));
    }

    // The last row is rewritten as a single stride block; its gap (if any) stays in the unchanged
    // tail, so we never read past the end of the input.
    const int64_t lastRowStart = layout.base + headRows * layout.outerStride;
    auto lastBody = sliceFlat(origOp.getInput(), lastRowStart, strideBlockWidth, "arith_last_body");
    auto lastUpd = sliceFlat(origOp.getUpdates(), headRows * layout.innerCount, layout.innerCount, "arith_last_upd");
    outerConcatInputs.push_back(rewriteRowBatch(lastBody, lastUpd, 1, strideBlockWidth, "last"));

    const int64_t tailSize = inputSize - bodyEnd;
    if (tailSize > 0) {
        auto tailOp = rewriter.create<IE::SliceOp>(takeOpLoc(origOp, "arith_tail"), origOp.getInput(),
                                                   SmallVector<int64_t>{bodyEnd}, SmallVector<int64_t>{tailSize});
        outerConcatInputs.push_back(tailOp.getResult());
    }

    _log.trace("Replaced ScatterNDUpdate at {0} with Slice+Reshape+Concat "
               "(base={1}, innerStride={2}, M={3}, outerStride={4}, N={5}, L={6})",
               loc, layout.base, layout.innerStride, layout.innerCount, layout.outerStride, layout.outerCount,
               inputSize);
    rewriter.replaceOpWithNewOp<IE::ConcatOp>(origOp, outerConcatInputs, /*axis=*/0);
    return mlir::success();
}

// ConvertNDUpdateDataToSliceConcat

class ConvertScatterPass::ConvertNDUpdateDataToSliceConcat final :
        public mlir::OpRewritePattern<IE::ScatterNDUpdateOp> {
public:
    ConvertNDUpdateDataToSliceConcat(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::ScatterNDUpdateOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ScatterNDUpdateOp origOp, mlir::PatternRewriter& rewriter) const final;
    std::optional<std::pair<Shape, Shape>> getUpdateRange(Const::DeclareOp indicesConst, ShapeRef inputShape) const;

private:
    Logger _log;
};

std::optional<std::pair<Shape, Shape>> ConvertScatterPass::ConvertNDUpdateDataToSliceConcat::getUpdateRange(
        Const::DeclareOp indicesConst, ShapeRef inputShape) const {
    const auto indicesConstValue = indicesConst.getContent();
    const auto indicesData = indicesConstValue.getValues<int64_t>();
    const auto indicesDataSize = indicesData.size();

    auto indicesShape = getShape(indicesConst);
    const auto coordSize = indicesShape.back();

    if (coordSize >= checked_cast<int64_t>(indicesShape.size())) {
        return std::nullopt;
    }
    // Verifies whether the update range forms a continuous block
    for (auto dimIdx = 0; dimIdx < coordSize; dimIdx++) {
        auto patternRepeatCount = std::accumulate(indicesShape.begin(), indicesShape.begin() + dimIdx, int64_t(1),
                                                  std::multiplies<int64_t>());
        auto increaseCount = indicesShape[Dim(dimIdx)];
        auto equalDataRepeatCount = std::accumulate(indicesShape.begin() + dimIdx + 1, indicesShape.end() - 1,
                                                    int64_t(1), std::multiplies<int64_t>());
        for (auto patternIdx = 0; patternIdx < patternRepeatCount; patternIdx++) {
            auto startVal = indicesData[dimIdx];
            auto patternLevelStrides = patternIdx * equalDataRepeatCount * increaseCount;
            for (auto increaseIdx = 0; increaseIdx < increaseCount; increaseIdx++) {
                auto increaseLevelStrides = increaseIdx * equalDataRepeatCount;
                for (auto equalDataIdx = 0; equalDataIdx < equalDataRepeatCount; equalDataIdx++) {
                    int64_t dataIdx = dimIdx + (patternLevelStrides + increaseLevelStrides + equalDataIdx) * coordSize;
                    VPUX_THROW_UNLESS(dataIdx < checked_cast<int64_t>(indicesDataSize),
                                      "Index '{0}' is out of the range of indicesData, which has a size of '{1}'",
                                      dataIdx, indicesDataSize);
                    if (indicesData[dataIdx] != startVal) {
                        return std::nullopt;
                    }
                }
                startVal += 1;
            }
        }
    }

    auto minRange = Shape(indicesData.begin(), indicesData.begin() + coordSize);
    auto maxRange = Shape(indicesData.begin() + (indicesShape.totalSize() - coordSize), indicesData.end());

    // Normalize negative indices
    for (auto dimIdx = 0; dimIdx < coordSize; ++dimIdx) {
        const int64_t dimSize = inputShape[Dim(dimIdx)];
        if (minRange[Dim(dimIdx)] < 0) {
            minRange[Dim(dimIdx)] += dimSize;
        }
        if (maxRange[Dim(dimIdx)] < 0) {
            maxRange[Dim(dimIdx)] += dimSize;
        }

        if (minRange[Dim(dimIdx)] > maxRange[Dim(dimIdx)]) {
            return std::nullopt;
        }
    }

    return std::pair(minRange, maxRange);
}

// Handles scenarios where the data update range forms a continuous block
// This can be transformed into a sequence of Slice and Concat operations across various dimensions
mlir::LogicalResult ConvertScatterPass::ConvertNDUpdateDataToSliceConcat::matchAndRewrite(
        IE::ScatterNDUpdateOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    const auto inputShape = getShape(origOp.getInput());
    const auto inputTensorRank = inputShape.size();
    const auto indices = origOp.getIndices();
    auto indicesConst = indices.getDefiningOp<Const::DeclareOp>();
    if (indicesConst == nullptr) {
        _log.trace("ScatterNDUpdate Op should with constant indices");
        return mlir::failure();
    }

    const auto indicesShape = getShape(indices);
    if (indicesShape.size() - 1 > inputTensorRank || indicesShape.back() > checked_cast<int64_t>(inputTensorRank)) {
        _log.trace("ScatterNDUpdate Op requires element wise updates");
        return mlir::failure();
    }

    auto updateRange = getUpdateRange(indicesConst, inputShape);
    if (!updateRange.has_value()) {
        _log.trace("Updates range is not continuous");
        return mlir::failure();
    }

    auto minRange = to_small_vector(updateRange.value().first);
    auto maxRange = to_small_vector(updateRange.value().second);
    // If indicesShape.back() < inputTensorRank, we need to update range data. Indices are starting from the outer most
    // dim, so for inner most dim that hasn't specified in indices, slicing from the beginning to the end.
    if (indicesShape.back() < checked_cast<int64_t>(inputTensorRank)) {
        auto origInShape = to_small_vector(inputShape);
        minRange.insert(minRange.end(), inputTensorRank - indicesShape.back(), 0);
        maxRange.insert(maxRange.end(), origInShape.end() + indicesShape.back() - inputTensorRank, origInShape.end());
    }

    auto updateIn = origOp.getUpdates();
    for (int64_t dimIdx = inputTensorRank - 1; dimIdx >= 0; dimIdx--) {
        SmallVector<mlir::Value> concatInputs;
        auto updatesShape = getShape(updateIn);
        // Slice data at begin
        auto beginSliceOffset = minRange;
        std::fill(beginSliceOffset.begin() + dimIdx, beginSliceOffset.end(), 0);
        auto beginSliceShape = to_small_vector(updatesShape.raw());
        beginSliceShape[dimIdx] = minRange[dimIdx];
        if (beginSliceShape[dimIdx] > 0) {
            auto beginSliceLoc = takeOpLoc(origOp, StringLiteral("slice_begin_at_Dim{0}"), dimIdx);
            concatInputs.push_back(
                    rewriter.create<IE::SliceOp>(beginSliceLoc, origOp.getInput(), beginSliceOffset, beginSliceShape)
                            .getResult());
        }
        // Update data
        concatInputs.push_back(updateIn);
        // Slice data at end
        auto endSliceOffset = std::move(beginSliceOffset);
        endSliceOffset[dimIdx] = maxRange[dimIdx] + 1;
        auto endSliceShape = to_small_vector(updatesShape.raw());
        endSliceShape[dimIdx] = inputShape[Dim(dimIdx)] - endSliceOffset[dimIdx];
        if (endSliceShape[dimIdx] > 0) {
            auto endSliceLoc = takeOpLoc(origOp, StringLiteral("slice_end_at_Dim{0}"), dimIdx);
            concatInputs.push_back(
                    rewriter.create<IE::SliceOp>(endSliceLoc, origOp.getInput(), endSliceOffset, endSliceShape)
                            .getResult());
        }
        // Create Concat Op
        auto concatLoc = takeOpLoc(origOp, StringLiteral("concat_at_Dim{0}"), dimIdx);
        updateIn = rewriter.create<IE::ConcatOp>(concatLoc, concatInputs, dimIdx).getResult();
    }

    _log.trace("Replace '{0}' at '{1}' with Slice and Concat Op", origOp->getName(), origOp->getLoc());
    origOp.replaceAllUsesWith(updateIn);

    return mlir::success();
}

//
// ConvertScatterNDUpdateSelectToConcat
//
// Handles ScatterNDUpdate with constant indices that have an "identity prefix" pattern:
// the first (coordSize-1) elements of each index tuple equal their own coordinate, and
// the last element selects a position in the "select dimension" (dim at coordSize-1).
//
// Covers both:
// - Point-level scatter (coordSize == inputRank): selects positions in the last dim
// - Partial scatter (coordSize < inputRank): selects rows, remaining dims bulk-copied
//
// Example 1 (point-level): data [1,4,34,3], indices [1,4,34,2,4], updates [1,4,34,2]
//   coordSize=4=rank, selectDim=3, selects columns [0,2] from dim3 (size 3)
//
// Example 2 (partial): data [1,4,34,3], indices [1,4,4,3], updates [1,4,4,3]
//   coordSize=3<rank, selectDim=2, selects rows [7,14,25,33] from dim2 (size 34)
//
// Conversion: Slice data into segments between selected positions, interleave with
// slices from updates, then Concat along the select dimension.

class ConvertScatterPass::ConvertScatterNDUpdateColumnSelectToConcat final :
        public mlir::OpRewritePattern<IE::ScatterNDUpdateOp> {
public:
    ConvertScatterNDUpdateColumnSelectToConcat(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::ScatterNDUpdateOp>(ctx), _log(log) {
        setDebugName("ConvertScatterNDUpdateSelectToConcat");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ScatterNDUpdateOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvertScatterPass::ConvertScatterNDUpdateColumnSelectToConcat::matchAndRewrite(
        IE::ScatterNDUpdateOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp.getLoc());

    const auto inputType = mlir::cast<vpux::NDTypeInterface>(origOp.getInput().getType());
    const auto inputShape = inputType.getShape();
    const auto inputRank = inputType.getRank();
    const auto indices = origOp.getIndices();
    const auto indicesShape = getShape(indices);
    const auto indicesRank = checked_cast<int64_t>(indicesShape.size());

    const auto coordSize = indicesShape.back();

    // coordSize must be in [1, inputRank]
    if (coordSize < 1 || coordSize > inputRank) {
        return mlir::failure();
    }

    // indices shape: [d0, ..., d_{coordSize-2}, numUpdates, coordSize]
    // For identity prefix: d_i == inputShape[i] for i < coordSize-1
    // and numUpdates < inputShape[coordSize-1]
    if (indicesRank - 1 < coordSize) {
        return mlir::failure();
    }

    for (int64_t i = 0; i < coordSize - 1; i++) {
        if (indicesShape[Dim(i)] != inputShape[Dim(i)]) {
            return mlir::failure();
        }
    }

    const auto numUpdates = indicesShape[Dim(coordSize - 1)];
    const auto selectDimSize = inputShape[Dim(coordSize - 1)];
    if (numUpdates >= selectDimSize) {
        return mlir::failure();
    }

    auto indicesConst = indices.getDefiningOp<Const::DeclareOp>();
    if (indicesConst == nullptr) {
        return mlir::failure();
    }

    const auto indicesConstValue = indicesConst.getContent();
    auto indicesData = indicesConstValue.getValues<int64_t>();

    // Verify identity prefix and extract selected positions
    const int64_t numTuples = indicesShape.totalSize() / coordSize;
    SmallVector<int64_t> selectedPositions(numUpdates, -1);

    for (int64_t tupleIdx = 0; tupleIdx < numTuples; tupleIdx++) {
        const int64_t baseIdx = tupleIdx * coordSize;

        // Compute expected spatial coordinates and update index
        int64_t remaining = tupleIdx;
        SmallVector<int64_t> expectedCoords(coordSize - 1);
        const int64_t updateIdx = remaining % numUpdates;
        remaining /= numUpdates;
        for (int64_t d = coordSize - 2; d >= 0; d--) {
            expectedCoords[d] = remaining % inputShape[Dim(d)];
            remaining /= inputShape[Dim(d)];
        }

        // Check identity prefix
        for (int64_t d = 0; d < coordSize - 1; d++) {
            if (indicesData[baseIdx + d] != expectedCoords[d]) {
                return mlir::failure();
            }
        }

        // Check position consistency across all spatial locations
        const int64_t pos = indicesData[baseIdx + coordSize - 1];
        if (pos < 0 || pos >= selectDimSize) {
            return mlir::failure();
        }
        if (selectedPositions[updateIdx] == -1) {
            selectedPositions[updateIdx] = pos;
        } else if (selectedPositions[updateIdx] != pos) {
            return mlir::failure();
        }
    }

    // Sort by position and check for duplicates
    SmallVector<std::pair<int64_t, int64_t>> sortedSel;  // (position, updateIdx)
    for (int64_t i = 0; i < numUpdates; i++) {
        if (selectedPositions[i] < 0) {
            return mlir::failure();
        }
        sortedSel.push_back({selectedPositions[i], i});
    }
    llvm::sort(sortedSel, [](const auto& a, const auto& b) {
        return a.first < b.first;
    });

    for (int64_t i = 1; i < numUpdates; i++) {
        if (sortedSel[i].first == sortedSel[i - 1].first) {
            return mlir::failure();
        }
    }

    // Build Slice+Concat along the select dimension
    const int64_t selectDim = coordSize - 1;
    SmallVector<mlir::Value> concatInputs;

    int64_t currentPos = 0;
    for (int64_t i = 0; i < numUpdates; i++) {
        const int64_t pos = sortedSel[i].first;
        const int64_t updateIdx = sortedSel[i].second;

        // Slice unchanged segment from input: [currentPos, pos)
        if (pos > currentPos) {
            SmallVector<int64_t> sliceOffset(inputRank, 0);
            SmallVector<int64_t> sliceSize(inputShape.raw().begin(), inputShape.raw().end());
            sliceOffset[selectDim] = currentPos;
            sliceSize[selectDim] = pos - currentPos;
            auto sliceLoc = takeOpLoc(origOp, StringLiteral("slice_input_{0}_{1}"), currentPos, pos);
            concatInputs.push_back(
                    rewriter.create<IE::SliceOp>(sliceLoc, origOp.getInput(), sliceOffset, sliceSize).getResult());
        }

        // Slice from updates at updateIdx
        SmallVector<int64_t> updOffset(inputRank, 0);
        SmallVector<int64_t> updSize(inputShape.raw().begin(), inputShape.raw().end());
        updOffset[selectDim] = updateIdx;
        updSize[selectDim] = 1;
        auto updLoc = takeOpLoc(origOp, StringLiteral("slice_updates_{0}"), pos);
        concatInputs.push_back(
                rewriter.create<IE::SliceOp>(updLoc, origOp.getUpdates(), updOffset, updSize).getResult());

        currentPos = pos + 1;
    }

    // Remaining tail from input
    if (currentPos < selectDimSize) {
        SmallVector<int64_t> sliceOffset(inputRank, 0);
        SmallVector<int64_t> sliceSize(inputShape.raw().begin(), inputShape.raw().end());
        sliceOffset[selectDim] = currentPos;
        sliceSize[selectDim] = selectDimSize - currentPos;
        auto sliceLoc = takeOpLoc(origOp, StringLiteral("slice_input_{0}_{1}"), currentPos, selectDimSize);
        concatInputs.push_back(
                rewriter.create<IE::SliceOp>(sliceLoc, origOp.getInput(), sliceOffset, sliceSize).getResult());
    }

    _log.trace("Replaced ScatterNDUpdate with select Slice+Concat (selectDim={0})", selectDim);
    rewriter.replaceOpWithNewOp<IE::ConcatOp>(origOp, concatInputs, selectDim);

    return mlir::success();
}

//
// SplitToMultiScatterNDUpdateOp
//

struct SplitParams {
    Dim tileDim;
    int64_t splitOffset;
    int64_t splitSize;
};

class ConvertScatterPass::SplitToMultiScatterNDUpdateOp final : public mlir::OpRewritePattern<IE::ScatterNDUpdateOp> {
public:
    SplitToMultiScatterNDUpdateOp(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::ScatterNDUpdateOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ScatterNDUpdateOp origOp, mlir::PatternRewriter& rewriter) const final;

    std::optional<SmallVector<SplitParams>> getSplitInfo(ShapeRef inputShape, ShapeRef indicesShape,
                                                         Const::DeclareOp indicesConst) const;

private:
    Logger _log;
};

std::optional<SmallVector<SplitParams>> ConvertScatterPass::SplitToMultiScatterNDUpdateOp::getSplitInfo(
        ShapeRef inputShape, ShapeRef indicesShape, Const::DeclareOp indicesConst) const {
    const auto greaterThanOne = [](auto dimSize) {
        return dimSize > 1;
    };

    const auto indicesShapeGreaterThanOne = std::count_if(indicesShape.begin(), indicesShape.end() - 1, greaterThanOne);
    if (indicesShapeGreaterThanOne != 1) {
        _log.trace("Elements Update: Only support ScatterNDUpdate Op update at one axis");
        return std::nullopt;
    }

    auto inAxis = llvm::find_if(inputShape, greaterThanOne);
    auto indicesAxis = llvm::find_if(indicesShape, greaterThanOne);
    if (inAxis == inputShape.end() || indicesAxis == indicesShape.end() - 1) {
        _log.trace("Cannot determine update axis for split");
        return std::nullopt;
    }

    auto inAxisDim = std::distance(inputShape.begin(), inAxis);
    auto indicesAxisDim = std::distance(indicesShape.begin(), indicesAxis);
    if (inAxisDim != indicesAxisDim) {
        _log.trace("Input and indices update axes do not match");
        return std::nullopt;
    }
    const auto updateDim = inAxisDim;

    const auto indicesConstValue = indicesConst.getContent();
    const auto indicesData = indicesConstValue.getValues<int64_t>();
    const auto indicesDataRank = checked_cast<int64_t>(indicesShape.size() - 1);

    SmallVector<SplitParams> splitInfos;
    int64_t currentStart = indicesData[updateDim];
    int64_t currentLength = 0;
    int64_t currentOffset = 0;

    constexpr size_t SPLIT_NUMBER_THRESHOLD = 4;
    for (int64_t idx = 1; idx < indicesShape[Dim(updateDim)]; ++idx) {
        const int64_t dataIdx = updateDim + indicesDataRank * idx;
        VPUX_THROW_UNLESS(dataIdx < checked_cast<int64_t>(indicesData.size()), "Indices data access out of bounds");

        if (indicesData[dataIdx] != currentStart + currentLength + 1) {
            splitInfos.emplace_back(SplitParams{Dim(updateDim), currentOffset, currentLength + 1});
            if (splitInfos.size() > SPLIT_NUMBER_THRESHOLD) {
                _log.trace("Split number is larger than {0}", SPLIT_NUMBER_THRESHOLD);
                return std::nullopt;
            }
            currentStart = indicesData[dataIdx];
            currentLength = 0;
            currentOffset = idx;
        } else {
            currentLength++;
        }

        if (idx == indicesShape[Dim(updateDim)] - 1) {
            splitInfos.emplace_back(SplitParams{Dim(updateDim), currentOffset, currentLength + 1});
        }
    }

    const auto splitSizeEqualOne = [](SplitParams splitInfo) {
        return splitInfo.splitSize == 1;
    };

    if (llvm::all_of(splitInfos, splitSizeEqualOne)) {
        _log.trace("All of split size is one");
        return std::nullopt;
    }

    return splitInfos;
}

// Handle ScatterNDUpdate with indices containing several contiguous blocks
// For example: Indices [24, 25, 26, 31, 32, 41, 42, 43]
// can be split into three ScatterNDUpdate operations with indices: [24, 25, 26], [31, 32], [41, 42, 43]
// Each ScatterNDUpdate can then be converted to a slice and concat operation
// To avoid splitting into too many small blocks, a threshold of 5 is used
// The pattern converted by this rewriter looks like:

//  Input  Indices[B1, B2]  Update[B1, B2]       ->       Input  Indices[B1]  Update[B1]
//     |          |             |                           |          |         |
//     \          |             /                           \          |         /
//         ScatterNDUpdate                                      ScatterNDUpdate
//                                                                  |
//                                                                  |         Indices[B2]  Update[B2]
//                                                                  |            |             |
//                                                                  \            |             /
//                                                                        ScatterNDUpdate

mlir::LogicalResult ConvertScatterPass::SplitToMultiScatterNDUpdateOp::matchAndRewrite(
        IE::ScatterNDUpdateOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    const auto inputShape = getShape(origOp.getInput());
    const auto indices = origOp.getIndices();
    const auto indicesShape = getShape(indices);
    const auto updates = origOp.getUpdates();
    const auto updatesShape = getShape(updates);
    auto indicesConst = indices.getDefiningOp<Const::DeclareOp>();
    if (!indicesConst) {
        _log.trace("ScatterNDUpdate Op should have constant indices");
        return mlir::failure();
    }

    auto splitInfo = getSplitInfo(inputShape, indicesShape, indicesConst);
    if (!splitInfo.has_value() || splitInfo.value().size() <= 1) {
        _log.trace("ScatterNDUpdate Op cannot be split");
        return mlir::failure();
    }

    auto nextSliceInput = origOp.getInput();
    const auto& splits = splitInfo.value();
    for (const auto& split : splits) {
        auto offset = split.splitOffset;
        auto size = split.splitSize;
        auto tileDim = split.tileDim;

        // Create indices slice
        auto indicesOffset = Shape(indicesShape.size(), 0);
        auto indicesSize = Shape(indicesShape.raw());
        indicesOffset[tileDim] = offset;
        indicesSize[tileDim] = size;
        auto indicesSlice = rewriter.createOrFold<IE::SliceOp>(
                takeOpLoc(origOp, StringLiteral("slice_indices_{0}"), offset), indices, indicesOffset, indicesSize);

        // Create update slice
        auto updateOffset = Shape(updatesShape.size(), 0);
        auto updateSize = Shape(updatesShape.raw());
        updateOffset[tileDim] = offset;
        updateSize[tileDim] = size;
        auto updateSlice = rewriter.createOrFold<IE::SliceOp>(
                takeOpLoc(origOp, StringLiteral("slice_update_{0}"), offset), updates, updateOffset, updateSize);

        // Create new ScatterNDUpdateOp
        auto newScatter = rewriter.create<IE::ScatterNDUpdateOp>(
                takeOpLoc(origOp, StringLiteral("slice_ND_{0}"), offset), nextSliceInput, indicesSlice, updateSlice);

        nextSliceInput = newScatter.getOutput();
    }

    _log.trace("Replace '{0}' at '{1}' with multi ScatterNDUpdate", origOp->getName(), origOp->getLoc());
    origOp.replaceAllUsesWith(nextSliceInput);

    return mlir::success();
}

//
// ConvertScatterElementsUpdateToAddMultiply
//

// Convert ScatterElementsUpdate with reduction type SUM/PROD to Add/Multiply.
// Special scatterElement like below, inputShape[Axis] = 1, inputShape is same as updateShape.
// And all the indices should be 0 in this case if it is a valid IR, but we still check it here.
// Like inputShape, indicesShape, updateShape = [1x1024], axis = 0, according to OV defination:
// output[indices[i][j]][j] = reduction(updates[i][j], output[indices[i][j]][j]), axis = 0.
// They are actually performing element-wise operation between input and update.

class ConvertScatterPass::ConvertScatterElementsUpdateToAddMultiply final :
        public mlir::OpRewritePattern<IE::ScatterElementsUpdateOp> {
public:
    ConvertScatterElementsUpdateToAddMultiply(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::ScatterElementsUpdateOp>(ctx), _log(log) {
        setDebugName("ConvertScatterElementsUpdateToAddMultiply");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ScatterElementsUpdateOp origOp,
                                        mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvertScatterPass::ConvertScatterElementsUpdateToAddMultiply::matchAndRewrite(
        IE::ScatterElementsUpdateOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());
    const auto ctx = origOp.getContext();
    const auto reduction = origOp.getReduction();
    if (reduction != IE::ScatterElementsUpdateReductionType::SUM &&
        reduction != IE::ScatterElementsUpdateReductionType::PROD) {
        return mlir::failure();
    }

    auto hasAxis = origOp.getAxisValue();
    if (!hasAxis.has_value()) {
        return mlir::failure();
    }

    // input need to have same data type as update.
    const auto inputType = mlir::cast<vpux::NDTypeInterface>(origOp.getInput().getType());
    const auto updatesType = mlir::cast<vpux::NDTypeInterface>(origOp.getUpdates().getType());
    if (inputType != updatesType) {
        return mlir::failure();
    }

    auto axis = hasAxis.value();
    if (axis < 0) {
        axis += inputType.getRank();
    }
    if (inputType.getShape()[Dim(axis)] != 1) {
        return mlir::failure();
    }

    // all indices data = 0
    auto indicesOp = origOp.getIndices().getDefiningOp<Const::DeclareOp>();
    if (!indicesOp) {
        return mlir::failure();
    }
    auto indicesContent = indicesOp.getContent();
    auto indicesVals = indicesContent.getValues<int64_t>();
    for (auto val : indicesVals) {
        if (val != 0) {
            return mlir::failure();
        }
    }

    auto numpyBroadcastTypeAttr = IE::AutoBroadcastTypeAttr::get(ctx, IE::AutoBroadcastType::NUMPY);
    if (reduction == IE::ScatterElementsUpdateReductionType::SUM) {
        rewriter.replaceOpWithNewOp<IE::AddOp>(origOp, origOp.getInput(), origOp.getUpdates(), numpyBroadcastTypeAttr,
                                               nullptr, nullptr, nullptr, nullptr);
    }

    if (reduction == IE::ScatterElementsUpdateReductionType::PROD) {
        rewriter.replaceOpWithNewOp<IE::MultiplyOp>(origOp, origOp.getInput(), origOp.getUpdates(),
                                                    numpyBroadcastTypeAttr, nullptr, nullptr, nullptr, nullptr);
    }

    _log.trace("Replace '{0}' at '{1}' with add/multiply", origOp->getName(), origOp->getLoc());
    return mlir::success();
}

//
// MergeChainedScatterElementsUpdate
//

// Merge a chain of ScatterElementsUpdate ops that write to the same buffer.
// Assumes indices across ops do NOT overlap; overlapping indices would have
// undefined element ordering in the merged op.
//
// Pattern:
//   %0 = ScatterElementsUpdate(%data, %indices_0, %updates_0, axis=A)
//   %1 = ScatterElementsUpdate(%0,    %indices_1, %updates_1, axis=A)
//   ...
//   %N = ScatterElementsUpdate(%N-1,  %indices_N, %updates_N, axis=A)
//
// Merged into:
//   %merged_indices = Concat(%indices_0, %indices_1, ..., %indices_N, axis=A)
//   %merged_updates = Concat(%updates_0, %updates_1, ..., %updates_N, axis=A)
//   %result = ScatterElementsUpdate(%data, %merged_indices, %merged_updates, axis=A)

static std::optional<int64_t> getScatterAxis(IE::ScatterElementsUpdateOp op) {
    auto axis = op.getAxisValue();
    if (axis.has_value()) {
        return axis.value();
    }
    auto axisOp = op.getAxis();
    if (!axisOp) {
        return std::nullopt;
    }
    auto axisConst = axisOp.getDefiningOp<Const::DeclareOp>();
    if (!axisConst) {
        return std::nullopt;
    }
    auto axisContent = axisConst.getContent();
    auto axisValues = axisContent.getValues<int64_t>();
    return axisValues[0];
}

class ConvertScatterPass::MergeChainedScatterElementsUpdate final :
        public mlir::OpRewritePattern<IE::ScatterElementsUpdateOp> {
public:
    MergeChainedScatterElementsUpdate(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::ScatterElementsUpdateOp>(ctx, /*benefit=*/2), _log(log) {
        setDebugName("MergeChainedScatterElementsUpdate");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ScatterElementsUpdateOp origOp,
                                        mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvertScatterPass::MergeChainedScatterElementsUpdate::matchAndRewrite(
        IE::ScatterElementsUpdateOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    // Only match on the LAST op in a chain (no ScatterElementsUpdate user on the output)
    for (auto user : origOp.getOutput().getUsers()) {
        if (mlir::isa<IE::ScatterElementsUpdateOp>(user)) {
            return mlir::failure();
        }
    }

    if (origOp.getReduction() != IE::ScatterElementsUpdateReductionType::NONE) {
        return mlir::failure();
    }

    auto axis = getScatterAxis(origOp);
    if (!axis.has_value()) {
        return mlir::failure();
    }
    const auto scatterAxis = axis.value();
    const auto useInitVal = origOp.getUseInitVal();

    // Walk backward to collect the chain
    SmallVector<IE::ScatterElementsUpdateOp> chain;
    chain.push_back(origOp);

    auto currentOp = origOp;
    while (true) {
        auto prevScatter = currentOp.getInput().getDefiningOp<IE::ScatterElementsUpdateOp>();
        if (!prevScatter) {
            break;
        }
        if (!prevScatter.getOutput().hasOneUse()) {
            break;
        }
        if (prevScatter.getReduction() != IE::ScatterElementsUpdateReductionType::NONE) {
            break;
        }
        if (prevScatter.getUseInitVal() != useInitVal) {
            break;
        }
        auto prevAxis = getScatterAxis(prevScatter);
        if (!prevAxis.has_value() || prevAxis.value() != scatterAxis) {
            break;
        }
        chain.push_back(prevScatter);
        currentOp = prevScatter;
    }

    if (chain.size() < 2) {
        return mlir::failure();
    }

    // Reverse so chain[0] is the first (topmost) scatter
    std::reverse(chain.begin(), chain.end());

    // Verify all updates and indices have compatible shapes (same except along scatter axis)
    const auto refUpdatesType = mlir::cast<vpux::NDTypeInterface>(chain[0].getUpdates().getType());
    const auto refIndicesType = mlir::cast<vpux::NDTypeInterface>(chain[0].getIndices().getType());
    const auto rank = refUpdatesType.getRank();

    const auto normalizedAxis = scatterAxis >= 0 ? scatterAxis : scatterAxis + rank;

    for (size_t i = 1; i < chain.size(); ++i) {
        const auto updatesType = mlir::cast<vpux::NDTypeInterface>(chain[i].getUpdates().getType());
        const auto indicesType = mlir::cast<vpux::NDTypeInterface>(chain[i].getIndices().getType());

        if (updatesType.getRank() != rank || indicesType.getRank() != rank) {
            return mlir::failure();
        }
        for (int64_t d = 0; d < rank; ++d) {
            if (d == normalizedAxis) {
                continue;
            }
            if (updatesType.getShape()[Dim(d)] != refUpdatesType.getShape()[Dim(d)]) {
                return mlir::failure();
            }
            if (indicesType.getShape()[Dim(d)] != refIndicesType.getShape()[Dim(d)]) {
                return mlir::failure();
            }
        }
    }

    _log.trace("Merging chain of {0} ScatterElementsUpdate ops at '{1}'", chain.size(), origOp->getLoc());

    SmallVector<mlir::Value> updatesList;
    SmallVector<mlir::Value> indicesList;
    for (auto& op : chain) {
        updatesList.push_back(op.getUpdates());
        indicesList.push_back(op.getIndices());
    }

    auto mergedUpdates =
            rewriter.create<IE::ConcatOp>(takeOpLoc(origOp, "merged_updates"), updatesList, normalizedAxis);
    auto mergedIndices =
            rewriter.create<IE::ConcatOp>(takeOpLoc(origOp, "merged_indices"), indicesList, normalizedAxis);

    auto originalData = chain[0].getInput();

    rewriter.replaceOpWithNewOp<IE::ScatterElementsUpdateOp>(
            origOp, originalData, mergedIndices.getOutput(), mergedUpdates.getOutput(), /*axis=*/nullptr,
            rewriter.getI64IntegerAttr(normalizedAxis), chain[0].getReductionAttr(), chain[0].getUseInitValAttr());

    for (int64_t i = static_cast<int64_t>(chain.size()) - 2; i >= 0; --i) {
        rewriter.eraseOp(chain[i]);
    }

    return mlir::success();
}

//
// safeRunOnFunc
//

void ConvertScatterPass::safeRunOnFunc() {
    auto& ctx = getContext();
    mlir::ConversionTarget target(ctx);
    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<ConvertScatterNDUpdateToStridedConcat>(&ctx, _log);
    patterns.add<ConvertScatterNDUpdateToSliceConcat>(&ctx, _log);
    patterns.add<ConvertScatterNDUpdateArithmeticToConcat>(&ctx, _log);
    patterns.add<ConvertNDUpdateDataToSliceConcat>(&ctx, _log);
    patterns.add<ConvertScatterNDUpdateColumnSelectToConcat>(&ctx, _log);
    patterns.add<SplitToMultiScatterNDUpdateOp>(&ctx, _log);
    patterns.add<MergeChainedScatterElementsUpdate>(&ctx, _log);
    patterns.add<ConvertScatterElementsUpdateToAddMultiply>(&ctx, _log);

    auto func = getOperation();
    if (mlir::failed(applyPatternsGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertScatterPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertScatterPass(Logger log) {
    return std::make_unique<ConvertScatterPass>(log);
}
