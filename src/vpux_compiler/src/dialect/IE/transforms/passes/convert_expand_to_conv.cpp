//
// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/expand_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/pooling_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/quantization.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/IR/PatternMatch.h>

namespace vpux::IE {
#define GEN_PASS_DECL_CONVERTEXPANDTOCONVPASS
#define GEN_PASS_DEF_CONVERTEXPANDTOCONVPASS
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {
Shape calculateWeightsShape(ShapeRef expandInShape, ShapeRef expandOutShape, const int64_t alignment) {
    const int64_t kernelOutputChannels = expandOutShape[Dims4D::Act::C] * alignment;
    const int64_t kernelInputChannels = expandInShape[Dims4D::Act::C] * alignment;
    const int64_t kernelY = 1;
    const int64_t kernelX = 1;
    return Shape{kernelOutputChannels, kernelInputChannels, kernelY, kernelX};
}

IE::AffineReshapeOp buildReshape(mlir::Location loc, mlir::Value input, ArrayRef<int64_t> targetShape,
                                 mlir::PatternRewriter& rewriter) {
    const auto ctx = rewriter.getContext();
    const auto srcType = mlir::cast<vpux::NDTypeInterface>(input.getType());
    const auto dstType = srcType.changeShape(ShapeRef(targetShape));
    SmallVector<SmallVector<int64_t>> reassociationMap(targetShape.size());
    for (const auto& dimIdx : irange(reassociationMap.size())) {
        reassociationMap[dimIdx].push_back(dimIdx);
    }
    const auto reassociationMapAttr = getIntArrayOfArray(ctx, reassociationMap);
    const auto targetShapeAttr = getIntArrayAttr(ctx, targetShape);
    auto reshapeOp = rewriter.create<IE::AffineReshapeOp>(loc, dstType, input, reassociationMapAttr, targetShapeAttr);

    return reshapeOp;
}

IE::AffineReshapeOp reshapeInput(mlir::Location loc, mlir::Value input, ShapeRef origShape,
                                 const int64_t channelAlignment, mlir::PatternRewriter& rewriter) {
    const auto batch = origShape[Dims4D::Act::N];
    const auto channels = origShape[Dims4D::Act::C] * channelAlignment;
    const auto height = origShape[Dims4D::Act::H];
    const auto width = origShape[Dims4D::Act::W] / channelAlignment;

    const SmallVector<int64_t> targetShape = {batch, channels, height, width};
    const auto reshapedLoc = appendLoc(loc, "reshape input for DPU expand");
    return buildReshape(reshapedLoc, input, ArrayRef(targetShape), rewriter);
}

IE::AffineReshapeOp reshapeInputFromHeightToChannel(mlir::Location loc, mlir::Value input, ShapeRef origShape,
                                                    const int64_t channelAlignment, mlir::PatternRewriter& rewriter) {
    const auto batch = 1;
    const auto channels = origShape[Dims4D::Act::C] * channelAlignment;
    const auto height = origShape[Dims4D::Act::N] * origShape[Dims4D::Act::H] / channelAlignment;
    const auto width = origShape[Dims4D::Act::W];

    const SmallVector<int64_t> targetShape = {batch, channels, height, width};
    const auto reshapedLoc = appendLoc(loc, "reshape input for DPU expand");
    return buildReshape(reshapedLoc, input, ArrayRef(targetShape), rewriter);
}

IE::AffineReshapeOp padHReshapeInput(mlir::Location loc, mlir::Value input, ShapeRef origShape,
                                     const int64_t channelAlignment, mlir::PatternRewriter& rewriter) {
    const auto origWidth = origShape[Dims4D::Act::W];
    if (origWidth % channelAlignment == 0) {
        return reshapeInput(loc, input, origShape, channelAlignment, rewriter);
    }

    const auto origBatch = origShape[Dims4D::Act::N];
    const auto origChannels = origShape[Dims4D::Act::C];
    const auto origHeight = origShape[Dims4D::Act::H];
    // reshape as a one dimensional vector
    const auto totalElemSize = origShape.totalSize();
    const SmallVector<int64_t> targetShapeLinear = {1, 1, totalElemSize, 1};
    auto linearReshapeOp = buildReshape(loc, input, ArrayRef(targetShapeLinear), rewriter);

    const auto origWidthPadded = origWidth + channelAlignment - (origWidth % channelAlignment);
    const auto totalPaddedElemSize = origBatch * origChannels * origHeight * origWidthPadded;

    // expand to the padded dimension
    auto padBegin = mlir::SmallVector<int64_t>(origShape.size(), 0);
    auto padEnd = mlir::SmallVector<int64_t>(origShape.size(), 0);
    padEnd[vpux::Dims4D::Act::H.ind()] = totalPaddedElemSize - totalElemSize;
    const auto ctx = rewriter.getContext();
    auto padExpand = rewriter.create<IE::ExpandOp>(appendLoc(loc, "_pad"), linearReshapeOp.getOutput(),
                                                   getIntArrayAttr(ctx, padBegin), getIntArrayAttr(ctx, padEnd));

    const SmallVector<int64_t> targetPaddedShape = {origBatch, origChannels, origHeight, origWidthPadded};
    return reshapeInput(loc, padExpand.getOutput(), ShapeRef(targetPaddedShape), channelAlignment, rewriter);
}

IE::AffineReshapeOp padWReshapeInput(mlir::Location loc, mlir::Value input, ShapeRef origShape,
                                     const int64_t channelAlignment, mlir::PatternRewriter& rewriter) {
    // Expand to the padded dimension
    SmallVector<int64_t> padBegin(origShape.size(), 0);
    SmallVector<int64_t> padEnd(origShape.size(), 0);
    padEnd[Dims4D::Act::W.ind()] = channelAlignment - origShape[Dims4D::Act::W] % channelAlignment;

    auto expandOp = rewriter.create<IE::ExpandOp>(appendLoc(loc, "_input_pad"), input,
                                                  getIntArrayAttr(rewriter.getContext(), ArrayRef(padBegin)),
                                                  getIntArrayAttr(rewriter.getContext(), ArrayRef(padEnd)));

    return reshapeInput(loc, expandOp, getShape(expandOp), channelAlignment, rewriter);
}

IE::AffineReshapeOp reshapeOutput(mlir::Location loc, mlir::Value convOutput, ShapeRef outShape,
                                  mlir::PatternRewriter& rewriter) {
    const auto reshapedLoc = appendLoc(loc, "reshape output for DPU expand");
    return buildReshape(reshapedLoc, convOutput, ArrayRef(outShape.raw()), rewriter);
}

IE::AffineReshapeOp unpadHReshapeOutput(mlir::Location loc, ShapeRef origOutShape, mlir::Value convOutput,
                                        const int64_t channelAlignment, mlir::PatternRewriter& rewriter) {
    const auto origWidth = origOutShape[Dims4D::Act::W];

    if (origWidth % channelAlignment == 0) {
        return reshapeOutput(loc, convOutput, origOutShape, rewriter);
    }

    const auto origBatch = origOutShape[Dims4D::Act::N];
    const auto origChannels = origOutShape[Dims4D::Act::C];
    const auto origHeight = origOutShape[Dims4D::Act::H];
    const auto paddedWidth = origWidth + channelAlignment - (origWidth % channelAlignment);

    // reshape as a one dimensional vector
    const auto totalPaddedElemSize = origBatch * origChannels * origHeight * paddedWidth;
    const SmallVector<int64_t> targetShapeLinear = {1, 1, totalPaddedElemSize, 1};
    auto linearReshapeOp = buildReshape(loc, convOutput, ArrayRef(targetShapeLinear), rewriter);

    // remove the padded memory allocated
    const auto totalElemSize = origBatch * origChannels * origHeight * origWidth;
    auto staticOffsets = mlir::SmallVector<int64_t>(origOutShape.size(), 0);
    auto staticSizes = mlir::SmallVector<int64_t>(origOutShape.size(), 1);
    staticSizes[vpux::Dims4D::Act::H.ind()] = totalElemSize;
    const auto ctx = rewriter.getContext();
    auto sliceOp = rewriter.create<IE::SliceOp>(appendLoc(loc, "_slice_out"), linearReshapeOp.getOutput(),
                                                getIntArrayAttr(ctx, staticOffsets), getIntArrayAttr(ctx, staticSizes));

    return reshapeOutput(loc, sliceOp.getResult(), origOutShape, rewriter);
}

IE::AffineReshapeOp unpadWReshapeOutput(mlir::Location loc, ShapeRef origOutShape, mlir::Value convOutput,
                                        mlir::PatternRewriter& rewriter) {
    auto outShape = origOutShape.toValues();
    const auto convShape = getShape(convOutput);
    outShape[Dims4D::Act::W] = convShape[Dims4D::Act::C] * convShape[Dims4D::Act::W] / outShape[Dims4D::Act::C];

    return reshapeOutput(loc, convOutput, outShape, rewriter);
}

IE::AffineReshapeOp channelToHeightReshapeOutput(mlir::Location loc, ShapeRef origOutShape, mlir::Value convOutput,
                                                 mlir::PatternRewriter& rewriter) {
    return reshapeOutput(loc, convOutput, origOutShape, rewriter);
}

// The idea is to create the following structure:
// Filter #0
// 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
// Filter #1
// 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
// Filter #2
// 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
// Filter #3
// 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
// ...
// Filter #15
// 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
// Filter #16
// 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
// Filter #17
// 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
// Filter #18
// 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
// Filter #19
// 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
// ...
std::vector<vpux::type::float16> populateWeights(ShapeRef weightShape, const int64_t numInputChannels,
                                                 const int64_t numOutputChannels, const int64_t convolutionAlignment) {
    const int64_t kernelInputChannels = weightShape[Dims4D::Filter::IC];
    const int64_t kernelY = weightShape[Dims4D::Filter::KY];
    const int64_t kernelX = weightShape[Dims4D::Filter::KX];

    const auto strideIC = kernelY * kernelX;
    const auto strideOC = strideIC * kernelInputChannels;
    std::vector<vpux::type::float16> weightValues(weightShape.totalSize(), checked_cast<vpux::type::float16>(0.f));
    // Resulting weights can be split into the number of blocks defined by convolutionAlignment.
    // convolutionAlignment is usually 16. E.g. for 64x48x1x1 weights there are 4 blocks with 16x48x1x1 geometry.
    // Block offset iterates over these chunks. Each block must contain a diagonal matrix padded with zeros.
    // outChan iterates inside each chunk. The idea is to fill main diagonals with ones.
    // However, in order to maintain continuation, diagonal must be shifted by prefixSize.
    // We want to do this:
    // 1 0 0 0 0 0 0 0
    // 0 1 0 0 0 0 0 0
    // 0 0 1 0 0 0 0 0
    // 0 0 0 0 0 0 0 0
    // 0 0 0 1 0 0 0 0
    // Not this:
    // 1 0 0 0 0 0 0 0
    // 0 1 0 0 0 0 0 0
    // 0 0 1 0 0 0 0 0
    // 0 0 0 0 0 0 0 0
    // 1 0 0 0 0 0 0 0
    // Note that outChan iterates only over origInShape[Dims4D::Act::C] because we want to skip padded rows.
    // For the simple example counters go like that:
    //
    //           prefixSize (3)
    //                   |
    //             1 0 0 0 0 0 0 0
    //             0 1 0 0 0 0 0 0
    //             0 0 1 0 0 0 0 0
    //             0 0 0 0 0 0 0 0
    // blockRow -> 0 0 0 1 0 0 0 0
    //             ^     ^
    //       outChan     inChan = prefixSize + outChan (0 + 3)
    for (int64_t blockOffset = 0; blockOffset < convolutionAlignment; blockOffset++) {
        const auto blockRow = blockOffset * numOutputChannels;
        const auto prefixSize = blockOffset * numInputChannels;
        for (int64_t outChan = 0; outChan < numInputChannels; outChan++) {
            const auto inChan = prefixSize + outChan;
            const auto pos = (blockRow + outChan) * strideOC + inChan * strideIC;
            weightValues.at(pos) = 1.f;
        }
    }

    return weightValues;
}

mlir::Value composeWeights(mlir::Location loc, ShapeRef origInShape, ShapeRef origOutShape,
                           const mlir::Type convolutionInputType, const int64_t convolutionAlignment,
                           mlir::PatternRewriter& rewriter) {
    const auto weightShape = calculateWeightsShape(origInShape, origOutShape, convolutionAlignment);
    const auto weightValues = populateWeights(weightShape, origInShape[Dims4D::Act::C], origOutShape[Dims4D::Act::C],
                                              convolutionAlignment);

    const auto ctx = rewriter.getContext();
    const auto weightStorageType = mlir::RankedTensorType::get(weightShape.raw(), mlir::Float16Type::get(ctx));
    const auto weightStorageAttr = Const::createConstContent(weightStorageType, ArrayRef(weightValues));
    const auto weightContentAttr = Const::ContentAttr::get(weightStorageAttr);
    const auto declLoc = appendLoc(loc, "weights for DPU expand");

    const auto weightExpressedElemType = IE::composeWeightsExpressedType(convolutionInputType);
    const auto weightExpressedType = mlir::RankedTensorType::get(weightShape.raw(), weightExpressedElemType);
    auto targetContentAttr = weightContentAttr.transform().castElemType(weightExpressedElemType).get();
    auto declOp = rewriter.create<Const::DeclareOp>(declLoc, weightExpressedType, std::move(targetContentAttr));

    const auto reorderLoc = appendLoc(loc, "reorder weights for DPU expand");
    const auto weightTypeNCHW = mlir::cast<vpux::NDTypeInterface>(declOp.getOutput().getType());
    const auto reorderType = weightTypeNCHW.changeDimsOrder(DimsOrder::NHWC);
    const auto orderMap = DimsOrder::NHWC.toAffineMap(ctx);
    auto reorderOut = rewriter.createOrFold<IE::ReorderOp>(reorderLoc, reorderType, declOp.getOutput(), orderMap);

    return reorderOut;
}

IE::ConvolutionOp buildConvolution(IE::ExpandOp expandOp, mlir::Value activation, mlir::Value weights,
                                   mlir::Type convOutElemType, mlir::PatternRewriter& rewriter) {
    const auto ctx = rewriter.getContext();
    const auto strides = getIntArrayAttr(ctx, SmallVector<int64_t>{1, 1});
    const auto kernelPadsBegin = getIntArrayAttr(ctx, SmallVector<int64_t>{0, 0});
    const auto kernelPadsEnd = getIntArrayAttr(ctx, SmallVector<int64_t>{0, 0});
    const auto dilations = getIntArrayAttr(ctx, SmallVector<int64_t>{1, 1});

    // IE::ConvolutionOp output type inference sets NCHW output order.
    // Specify convolution output type explicitly.
    const auto origOutType = mlir::cast<vpux::NDTypeInterface>(expandOp.getOutput().getType());
    const auto weightsShape = getShape(weights);
    const auto outChannels = weightsShape[Dims4D::Filter::OC];
    const Shape convInShape = getShape(activation).toValues();
    const Shape convOutShape = {convInShape[Dims4D::Act::N], outChannels, convInShape[Dims4D::Act::H],
                                convInShape[Dims4D::Act::W]};
    const auto convOutType = origOutType.changeShape(convOutShape).changeElemType(convOutElemType);
    return rewriter.create<IE::ConvolutionOp>(takeOpLoc(expandOp, "_as_convolution"), convOutType, activation, weights,
                                              /*bias=*/nullptr, strides, kernelPadsBegin, kernelPadsEnd, dilations,
                                              /*postOp=*/nullptr, /*clamp=*/nullptr, /*staticScale=*/nullptr,
                                              /*outputPadding=*/nullptr, /*inputPadding=*/nullptr);
}

bool isDerivedFromQuantize(mlir::Operation* op) {
    if (op->getNumOperands() != 2) {
        return false;
    }
    if (op->getOperand(0) != op->getOperand(1)) {
        return false;
    }
    const auto quantInputType = mlir::cast<vpux::NDTypeInterface>(op->getOperand(0).getType()).getElementType();
    const auto quantOutputType = mlir::cast<vpux::NDTypeInterface>(op->getResult(0).getType()).getElementType();
    return quantInputType.isF16() && mlir::isa<mlir::quant::QuantizedType>(quantOutputType);
}

// IE.Add -> IE.QuantizeCast -> IE.Expand is a rare case.
// The absence of IE.ShapeCast usually means that no further IE.Expand is required.
// For example IE.Add (1x16xHxW) -> IE.QuantizeCast -> IE.Convolution
// Such convolution does not need 1x16xHxW tensor expansion.
mlir::Value tryExpandWithAdd(mlir::Value in) {
    auto quantize = in.getDefiningOp<IE::AddOp>();
    if (quantize == nullptr) {
        return nullptr;
    }
    if (!isDerivedFromQuantize(quantize)) {
        return nullptr;
    }
    return quantize.getInput1();
}

mlir::Value tryExpandWithAddShapeCast(mlir::Value in) {
    auto outputShapeCast = in.getDefiningOp<IE::ShapeCastOp>();
    if (outputShapeCast == nullptr || mlir::isa<mlir::BlockArgument>(outputShapeCast.getSource())) {
        return nullptr;
    }
    auto quantize = outputShapeCast.getSource().getDefiningOp<IE::AddOp>();
    if (quantize == nullptr || mlir::isa<mlir::BlockArgument>(quantize.getInput1())) {
        return nullptr;
    }
    if (!isDerivedFromQuantize(quantize)) {
        return nullptr;
    }
    auto inputShapeCast = quantize.getInput1().getDefiningOp<IE::ShapeCastOp>();
    if (inputShapeCast == nullptr) {
        return nullptr;
    }
    return inputShapeCast.getSource();
}

mlir::Value tryExpandWithAvgPool(mlir::Value in) {
    auto avgPool = in.getDefiningOp<IE::AvgPoolOp>();
    if (avgPool == nullptr || !vpux::IE::isQuantizedPurposeAvgPool(avgPool)) {
        return nullptr;
    }
    if (!avgPool.getOutput().hasOneUse()) {
        return nullptr;
    }

    return avgPool.getInput();
}

mlir::Value tryExpandWithAvgPoolShapeCast(mlir::Value in) {
    auto shapeCast = in.getDefiningOp<IE::ShapeCastOp>();
    if (shapeCast == nullptr) {
        return nullptr;
    }
    if (!shapeCast.getResult().hasOneUse()) {
        return nullptr;
    }

    auto avgPool = shapeCast.getSource().getDefiningOp<IE::AvgPoolOp>();
    if (avgPool == nullptr || !vpux::IE::isQuantizedPurposeAvgPool(avgPool)) {
        return nullptr;
    }
    if (!avgPool.getOutput().hasOneUse()) {
        return nullptr;
    }

    auto inputShapeCast = avgPool.getInput().getDefiningOp<IE::ShapeCastOp>();
    if (inputShapeCast == nullptr) {
        return nullptr;
    }
    if (!inputShapeCast.getResult().hasOneUse()) {
        return nullptr;
    }

    return inputShapeCast.getSource();
}

// Search for any of these patterns:
// 1. IE.Add -> IE.QuantizeCast -> IE.Expand
// 2. IE.ShapeCast -> IE.Add -> IE.ShapeCast -> IE.QuantizeCast -> IE.Expand
// 3. IE.AvgPool -> IE.Expand
// 4. IE.ShapeCast -> IE.AvgPool -> IE.ShapeCast -> IE.Expand
// For Add case:
// These chains of operations are usually derived from IE.Quantize.
// We want to replace this entire subgraph with a single DPU expand operation with appropriate target element type.
// Note that IE.And match is skipped on purpose.
// IE.And implies that the target architecture does not fully support mixed precision convolutions.
mlir::Value getExpandInput(mlir::Value origInput) {
    if (mlir::isa<mlir::BlockArgument>(origInput)) {
        return nullptr;
    }

    // For Add case
    if (auto quantizeCast = origInput.getDefiningOp<IE::QuantizeCastOp>()) {
        if (mlir::isa<mlir::BlockArgument>(quantizeCast.getInput())) {
            return nullptr;
        }
        if (const auto input = tryExpandWithAdd(quantizeCast.getInput())) {
            return input;
        }
        if (const auto input = tryExpandWithAddShapeCast(quantizeCast.getInput())) {
            return input;
        }
    } else {
        // For avgPool case
        if (const auto input = tryExpandWithAvgPool(origInput)) {
            return input;
        }
        if (const auto input = tryExpandWithAvgPoolShapeCast(origInput)) {
            return input;
        }
    }
    return nullptr;
}

// Search for any of these patterns:
// 1. IE.Expand -> IE.Add -> IE.QuantizeCast
// 2. IE.Expand -> IE.Add -> IE.Slice -> IE.QuantizeCast
// 3. IE.Expand -> IE.AvgPool
// For Add case:
// These chains of operations are usually derived from IE.Quantize.
// We want to replace this entire subgraph with a single DPU expand operation with appropriate target element type.
// Note that IE.And match is skipped on purpose.
// IE.And implies that the target architecture does not fully support mixed precision convolutions.
mlir::Operation* getExpandOutput(mlir::Value origOutput) {
    const auto* expandOutputOp = *origOutput.getUsers().begin();

    // AvgPoolOp case
    if (auto avgPoolOp = mlir::dyn_cast_or_null<IE::AvgPoolOp>(expandOutputOp)) {
        if (!origOutput.hasOneUse()) {
            return nullptr;
        }
        if (vpux::IE::isQuantizedPurposeAvgPool(avgPoolOp)) {
            return avgPoolOp;
        }
        return nullptr;
    }

    // AddOp -> [SliceOp] -> QuantizeCastOp case
    if (auto addOp = mlir::dyn_cast_or_null<IE::AddOp>(expandOutputOp)) {
        const size_t expandUsersNum = std::distance(origOutput.getUsers().begin(), origOutput.getUsers().end());
        if (expandUsersNum != 2) {
            // AddOp has to be the only user of ExpandOp (using twice the same ExpandOp output value)
            return nullptr;
        }

        if (!isDerivedFromQuantize(addOp) || !addOp.getOutput().hasOneUse()) {
            return nullptr;
        }

        const auto* addChildOp = *addOp.getOutput().getUsers().begin();
        auto sliceOp = mlir::dyn_cast_or_null<IE::SliceOp>(addChildOp);
        if (sliceOp != nullptr) {
            if (!sliceOp.getResult().hasOneUse()) {
                return nullptr;
            }

            auto staticOffsetsAttr = parseIntArrayAttr<int64_t>(sliceOp.getStaticOffsetsAttr());
            auto staticSizesAttr = parseIntArrayAttr<int64_t>(sliceOp.getStaticSizesAttr());
            if (staticOffsetsAttr.size() != 4 || staticSizesAttr.size() != 4) {
                return nullptr;
            }

            auto sliceOpInputShape = getShape(sliceOp.getSource());
            // only slice over C is included in the pattern
            if (staticOffsetsAttr[Dims4D::Act::N.ind()] != 0 || staticOffsetsAttr[Dims4D::Act::C.ind()] != 0 ||
                staticOffsetsAttr[Dims4D::Act::H.ind()] != 0 || staticOffsetsAttr[Dims4D::Act::W.ind()] != 0 ||
                staticSizesAttr[Dims4D::Act::N.ind()] != 1 ||
                staticSizesAttr[Dims4D::Act::H.ind()] != sliceOpInputShape[Dims4D::Act::H] ||
                staticSizesAttr[Dims4D::Act::W.ind()] != sliceOpInputShape[Dims4D::Act::W]) {
                return nullptr;
            }
        }

        auto quantizeCastOp =
                (sliceOp != nullptr)
                        ? mlir::dyn_cast_or_null<IE::QuantizeCastOp>(*sliceOp.getResult().getUsers().begin())
                        : mlir::dyn_cast_or_null<IE::QuantizeCastOp>(*addOp.getOutput().getUsers().begin());

        return quantizeCastOp;
    }

    return nullptr;
}

// IE.Expand may have IE.Add -> IE.ShapeCast -> IE.QuantizeCast -> IE.Expand input chain.
// In that case it may be incorrect to fetch the element type type from IE.Expand output.
// For example: Input (zp = 64) -> IE.QuantizeCast (zp = 128) -> IE.QuantizeCast (zp = 0) -> IE.Expand (zp = 0)
// Canonicalizer folds such subgraphs into Input (zp = 64) -> IE.QuantizeCast (zp = 0) -> IE.Expand (zp = 0)
// Thus, when zp for convolution output element type is derived from IE.Expand, results may degrade in accuracy.
// In order to avoid the accuracy degradation, fetch the quantization parameters from IE.QuantizeCast input.
// Also, the scale must be divided by 2 because it comes from IE.Add that adds its input with itself.
mlir::Type getConvolutionOutputType(IE::ExpandOp expandOp) {
    const auto origOutType = mlir::cast<vpux::NDTypeInterface>(expandOp.getOutput().getType());
    if (getExpandInput(expandOp.getInput()) == nullptr) {
        return origOutType.getElementType();
    }
    auto origInputQuantCast = expandOp.getInput().getDefiningOp<IE::QuantizeCastOp>();
    if (origInputQuantCast == nullptr) {
        return origOutType.getElementType();
    }

    const auto quantCastInType = mlir::dyn_cast<mlir::quant::UniformQuantizedType>(
            mlir::cast<vpux::NDTypeInterface>(origInputQuantCast.getInput().getType()).getElementType());
    if (quantCastInType == nullptr) {
        return origOutType.getElementType();
    }
    const auto quantType = mlir::quant::UniformQuantizedType::get(
            quantCastInType.getFlags(), quantCastInType.getStorageType(), quantCastInType.getExpressedType(),
            quantCastInType.getScale() / 2.0, quantCastInType.getZeroPoint(), quantCastInType.getStorageTypeMin(),
            quantCastInType.getStorageTypeMax());
    return quantType;
}

// This method is required to cast the output of quantized convolution back to IE.Expand output element type.
// It usually has scale * 2 and may have different zero point, depending on how many QuantizeCast operations
// have been fused together.
mlir::Value quantCastOutput(IE::ExpandOp expandOp, mlir::Value reshapeOutput, mlir::PatternRewriter& rewriter) {
    auto origInputQuantCast = expandOp.getInput().getDefiningOp<IE::QuantizeCastOp>();
    if (origInputQuantCast == nullptr) {
        return reshapeOutput;
    }

    const auto quantCastLoc = appendLoc(expandOp.getLoc(), "quantize cast for DPU expand");
    auto quantCast = rewriter.create<IE::QuantizeCastOp>(
            quantCastLoc, reshapeOutput,
            mlir::cast<vpux::NDTypeInterface>(origInputQuantCast.getOutput().getType()).getElementType());
    return quantCast.getOutput();
}

class ExpandQuantizeSliceRewriter final : public mlir::OpRewritePattern<IE::ExpandOp> {
public:
    ExpandQuantizeSliceRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::ExpandOp>(ctx, benefitHigh), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ExpandOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ExpandQuantizeSliceRewriter::matchAndRewrite(IE::ExpandOp origOp,
                                                                 mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got IE.ExpandOp at '{1}'", getDebugName(), origOp->getLoc());
    if (!IE::isEligibleConvertToConv(origOp, _log, getDebugName()) ||
        (IE::isEligibleConvertToConv(origOp, _log, getDebugName()) && !IE::beneficialToPadHeight(origOp))) {
        return matchFailed(rewriter, origOp, "[{0}] Cannot convert IE.ExpandOp at '{1}'", getDebugName(),
                           origOp->getLoc());
    }

    const auto expandInput = origOp.getInput();
    const auto expandInShape = getShape(expandInput);
    auto* patternLastOp = getExpandOutput(origOp.getOutput());
    if (patternLastOp == nullptr) {
        return matchFailed(rewriter, origOp, "[{0}] IE.ExpandOp at '{1}' does not match ExpandQuantizeSlice pattern",
                           getDebugName(), origOp->getLoc());
    }

    auto* patternOutChildOp = *patternLastOp->getResult(0).getUsers().begin();
    Shape expandPatternOutShape = getShape(patternLastOp->getResult(0)).toValues();

    auto sliceOp = mlir::dyn_cast_or_null<IE::SliceOp>(patternOutChildOp);
    bool expandOutChannelsReduction = false;
    if (patternLastOp->getResult(0).hasOneUse() && sliceOp != nullptr) {
        const auto sliceOutShape = getShape(sliceOp.getResult());
        if (sliceOutShape[Dims4D::Act::C] <= 4 && expandInShape[Dims4D::Act::C] <= 4) {
            // the extension to multiple of 16 for AvgPool channels becomes unnecessary, so force the convolution to
            // only expand to 4
            expandOutChannelsReduction = true;
            expandPatternOutShape[Dims4D::Act::C] = 4;
        }
    }

    const auto expandInType = mlir::cast<vpux::NDTypeInterface>(expandInput.getType());
    const auto convolutionAlignment = IE::calculateAlignmentRequirementForExpandOpConversion(expandInType);
    auto reshapeIn = padHReshapeInput(origOp.getLoc(), expandInput, expandInShape, convolutionAlignment, rewriter);
    auto weights = composeWeights(origOp.getLoc(), expandInShape, expandPatternOutShape, expandInType.getElementType(),
                                  convolutionAlignment, rewriter);

    auto convOutElemType = mlir::cast<vpux::NDTypeInterface>(patternLastOp->getResult(0).getType()).getElementType();
    auto convOp = buildConvolution(origOp, reshapeIn.getOutput(), weights, convOutElemType, rewriter);
    auto reshapeOut = unpadHReshapeOutput(origOp.getLoc(), expandPatternOutShape, convOp.getOutput(),
                                          convolutionAlignment, rewriter);

    if (!expandOutChannelsReduction) {
        // only substitute the identified pattern
        rewriter.replaceOp(patternLastOp, reshapeOut.getOutput());
    } else {
        // Apply the reduction of expanded channels to 4 and therefore also substitute the final sliceOp
        auto staticOffsetsAttr = sliceOp.getStaticOffsetsAttr();
        auto staticSizesAttr = sliceOp.getStaticSizesAttr();
        const auto sliceLoc = appendLoc(sliceOp.getLoc(), "replace output slice op");
        auto newSliceOp =
                rewriter.create<IE::SliceOp>(sliceLoc, reshapeOut.getOutput(), staticOffsetsAttr, staticSizesAttr);
        rewriter.replaceOp(sliceOp, newSliceOp.getResult());
    }

    return mlir::success();
}

class QuantizedExpandRewriter final : public mlir::OpRewritePattern<IE::ExpandOp> {
public:
    QuantizedExpandRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::ExpandOp>(ctx, benefitHigh), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ExpandOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult QuantizedExpandRewriter::matchAndRewrite(IE::ExpandOp origOp,
                                                             mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got IE.ExpandOp at '{1}'", getDebugName(), origOp->getLoc());
    if (!IE::isEligibleConvertToConv(origOp, _log, getDebugName()) ||
        (IE::isEligibleConvertToConv(origOp, _log, getDebugName()) && !IE::beneficialToPadHeight(origOp))) {
        return matchFailed(rewriter, origOp, "[{0}] Cannot convert IE.ExpandOp at '{1}'", getDebugName(),
                           origOp->getLoc());
    }
    const auto expandInput = getExpandInput(origOp.getInput());
    if (expandInput == nullptr) {
        return matchFailed(rewriter, origOp, "[{0}] IE.ExpandOp at '{1}' does not have IE.Add producer", getDebugName(),
                           origOp->getLoc());
    }
    const auto expandInShape = getShape(origOp.getInput());
    const auto expandOutShape = getShape(origOp.getOutput());
    const auto expandInType = mlir::cast<vpux::NDTypeInterface>(expandInput.getType()).changeShape(expandInShape);
    const auto convolutionAlignment = IE::calculateAlignmentRequirementForExpandOpConversion(expandInType);
    auto reshapeIn = padHReshapeInput(origOp.getLoc(), expandInput, expandInShape, convolutionAlignment, rewriter);
    auto weights = composeWeights(origOp.getLoc(), expandInShape, expandOutShape, expandInType.getElementType(),
                                  convolutionAlignment, rewriter);
    auto convOutElemType = getConvolutionOutputType(origOp);
    auto convOp = buildConvolution(origOp, reshapeIn.getOutput(), weights, convOutElemType, rewriter);
    auto reshapeOut =
            unpadHReshapeOutput(origOp.getLoc(), expandOutShape, convOp.getOutput(), convolutionAlignment, rewriter);
    const auto quantCastOut = quantCastOutput(origOp, reshapeOut, rewriter);
    rewriter.replaceOp(origOp, quantCastOut);
    return mlir::success();
}

class DPUExpandRewriter final : public mlir::OpRewritePattern<IE::ExpandOp> {
public:
    DPUExpandRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::ExpandOp>(ctx, benefitLow), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ExpandOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult DPUExpandRewriter::matchAndRewrite(IE::ExpandOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got IE.ExpandOp at '{1}'", getDebugName(), origOp->getLoc());

    if (!IE::isEligibleConvertToConv(origOp, _log, getDebugName())) {
        return matchFailed(rewriter, origOp, "[{0}] Cannot convert IE.ExpandOp at '{1}'", getDebugName(),
                           origOp->getLoc());
    }

    const auto expandInput = origOp.getInput();
    const auto expandInType = mlir::cast<vpux::NDTypeInterface>(expandInput.getType());
    const auto convolutionAlignment = IE::calculateAlignmentRequirementForExpandOpConversion(expandInType);
    const auto expandInShape = expandInType.getShape();
    const auto expandOutShape = getShape(origOp.getOutput());

    enum ReshapeMode { RESHAPE_H_TO_C, PAD_W_RESHAPE, PAD_H_RESHAPE };
    ReshapeMode reshapeMode = IE::beneficialToReshapeHeightToChannel(origOp) ? RESHAPE_H_TO_C
                              : IE::beneficialToPadWidth(origOp)             ? PAD_W_RESHAPE
                                                                             : PAD_H_RESHAPE;

    auto reshapeIn = (reshapeMode == RESHAPE_H_TO_C)
                             ? reshapeInputFromHeightToChannel(origOp.getLoc(), expandInput, getShape(expandInput),
                                                               convolutionAlignment, rewriter)
                     : (reshapeMode == PAD_W_RESHAPE)
                             ? padWReshapeInput(origOp.getLoc(), expandInput, getShape(expandInput),
                                                convolutionAlignment, rewriter)
                             : padHReshapeInput(origOp.getLoc(), expandInput, getShape(expandInput),
                                                convolutionAlignment, rewriter);

    auto weights = composeWeights(origOp.getLoc(), expandInShape, expandOutShape, expandInType.getElementType(),
                                  convolutionAlignment, rewriter);
    const auto expandOutType = mlir::cast<vpux::NDTypeInterface>(origOp.getOutput().getType());
    const auto convOutElemType = expandOutType.getElementType();
    auto convOp = buildConvolution(origOp, reshapeIn.getOutput(), weights, convOutElemType, rewriter);

    auto reshapeOut = (reshapeMode == RESHAPE_H_TO_C) ? channelToHeightReshapeOutput(origOp.getLoc(), expandOutShape,
                                                                                     convOp.getOutput(), rewriter)
                      : (reshapeMode == PAD_W_RESHAPE)
                              ? unpadWReshapeOutput(origOp.getLoc(), expandOutShape, convOp.getOutput(), rewriter)
                              : unpadHReshapeOutput(origOp.getLoc(), expandOutShape, convOp.getOutput(),
                                                    convolutionAlignment, rewriter);

    rewriter.replaceOp(origOp, reshapeOut.getOutput());

    return mlir::success();
}

//
// ConvertExpandToConvPass
//

class ConvertExpandToConvPass final : public IE::impl::ConvertExpandToConvPassBase<ConvertExpandToConvPass> {
public:
    explicit ConvertExpandToConvPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void ConvertExpandToConvPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<ExpandQuantizeSliceRewriter>(&ctx, _log);
    patterns.add<QuantizedExpandRewriter>(&ctx, _log);
    patterns.add<DPUExpandRewriter>(&ctx, _log);

    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertExpandToConvPass
//
std::unique_ptr<mlir::Pass> vpux::IE::createConvertExpandToConvPass(Logger log) {
    return std::make_unique<ConvertExpandToConvPass>(log);
}
