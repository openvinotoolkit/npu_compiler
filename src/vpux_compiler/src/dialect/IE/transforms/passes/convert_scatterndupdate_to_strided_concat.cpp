//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Transforms/DialectConversion.h>

namespace vpux::IE {
#define GEN_PASS_DECL_CONVERTSCATTERNDUPDATETOSTRIDEDCONCAT
#define GEN_PASS_DEF_CONVERTSCATTERNDUPDATETOSTRIDEDCONCAT
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

class ConvertScatterNDUpdateToStridedConcatPass final :
        public IE::impl::ConvertScatterNDUpdateToStridedConcatBase<ConvertScatterNDUpdateToStridedConcatPass> {
public:
    explicit ConvertScatterNDUpdateToStridedConcatPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

public:
    class ConvertToStridedConcat;
    class ConvertToSliceConcat;
    class ConvertNDUpdateDataToSliceConcat;
    class SplitToMultiScatterNDUpdateOp;

private:
    void safeRunOnFunc() final;
};

//
// ConvertToStridedConcat
//

class ConvertScatterNDUpdateToStridedConcatPass::ConvertToStridedConcat final :
        public mlir::OpRewritePattern<IE::ScatterNDUpdateOp> {
public:
    ConvertToStridedConcat(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::ScatterNDUpdateOp>(ctx), _log(log) {
        setDebugName("ConvertToStridedConcat");
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

mlir::LogicalResult ConvertScatterNDUpdateToStridedConcatPass::ConvertToStridedConcat::matchAndRewrite(
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
// ConvertToSliceConcat
//

class ConvertScatterNDUpdateToStridedConcatPass::ConvertToSliceConcat final :
        public mlir::OpRewritePattern<IE::ScatterNDUpdateOp> {
public:
    ConvertToSliceConcat(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::ScatterNDUpdateOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ScatterNDUpdateOp origOp, mlir::PatternRewriter& rewriter) const final;

    std::optional<Dim> getUpdateDim(ShapeRef inputShape, ShapeRef indicesShape, Const::DeclareOp indicesConst) const;

private:
    Logger _log;
};

std::optional<Dim> ConvertScatterNDUpdateToStridedConcatPass::ConvertToSliceConcat::getUpdateDim(
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
mlir::LogicalResult ConvertScatterNDUpdateToStridedConcatPass::ConvertToSliceConcat::matchAndRewrite(
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

// ConvertNDUpdateDataToSliceConcat

class ConvertScatterNDUpdateToStridedConcatPass::ConvertNDUpdateDataToSliceConcat final :
        public mlir::OpRewritePattern<IE::ScatterNDUpdateOp> {
public:
    ConvertNDUpdateDataToSliceConcat(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::ScatterNDUpdateOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ScatterNDUpdateOp origOp, mlir::PatternRewriter& rewriter) const final;
    std::optional<std::pair<Shape, Shape>> getUpdateRange(Const::DeclareOp indicesConst) const;

private:
    Logger _log;
};

std::optional<std::pair<Shape, Shape>>
ConvertScatterNDUpdateToStridedConcatPass::ConvertNDUpdateDataToSliceConcat::getUpdateRange(
        Const::DeclareOp indicesConst) const {
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
    return std::pair(minRange, maxRange);
}

// Handles scenarios where the data update range forms a continuous block
// This can be transformed into a sequence of Slice and Concat operations across various dimensions
mlir::LogicalResult ConvertScatterNDUpdateToStridedConcatPass::ConvertNDUpdateDataToSliceConcat::matchAndRewrite(
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

    auto updateRange = getUpdateRange(indicesConst);
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
// SplitToMultiScatterNDUpdateOp
//

struct SplitParams {
    Dim tileDim;
    int64_t splitOffset;
    int64_t splitSize;
};

class ConvertScatterNDUpdateToStridedConcatPass::SplitToMultiScatterNDUpdateOp final :
        public mlir::OpRewritePattern<IE::ScatterNDUpdateOp> {
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

std::optional<SmallVector<SplitParams>>
ConvertScatterNDUpdateToStridedConcatPass::SplitToMultiScatterNDUpdateOp::getSplitInfo(
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

    VPUX_THROW_UNLESS(inAxis != inputShape.end() && indicesAxis != indicesShape.end() - 1, "Can not get correct Axis");
    auto inAxisDim = std::distance(inputShape.begin(), inAxis);
    auto indicesAxisDim = std::distance(indicesShape.begin(), indicesAxis);
    VPUX_THROW_UNLESS(inAxisDim == indicesAxisDim, "Can not get same Axis");
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

mlir::LogicalResult ConvertScatterNDUpdateToStridedConcatPass::SplitToMultiScatterNDUpdateOp::matchAndRewrite(
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
// safeRunOnFunc
//

void ConvertScatterNDUpdateToStridedConcatPass::safeRunOnFunc() {
    auto& ctx = getContext();
    mlir::ConversionTarget target(ctx);
    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<ConvertToStridedConcat>(&ctx, _log);
    patterns.add<ConvertToSliceConcat>(&ctx, _log);
    patterns.add<ConvertNDUpdateDataToSliceConcat>(&ctx, _log);
    patterns.add<SplitToMultiScatterNDUpdateOp>(&ctx, _log);

    auto func = getOperation();
    if (mlir::failed(applyPatternsGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertScatterNDUpdateToStridedConcatPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertScatterNDUpdateToStridedConcatPass(Logger log) {
    return std::make_unique<ConvertScatterNDUpdateToStridedConcatPass>(log);
}
