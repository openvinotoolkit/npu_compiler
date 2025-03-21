//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/transforms/passes/map_bilinear_interpolate_on_DPU.hpp"
#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/utils/core/numeric.hpp"

using namespace vpux;

namespace {

//
// Functions for operation generation
//

// Expand the number of input channels to be aligned to VPU::NCEInvariant::VPU_CHANNEL_ALIGNMENT
mlir::Value alignInputChannels(mlir::PatternRewriter& rewriter, mlir::Location loc, mlir::Value input,
                               ArrayRef<int64_t> inputShape) {
    const auto alignedInputC = alignValUp(inputShape[Dims4D::Act::C.ind()], VPU::NCEInvariant::VPU_CHANNEL_ALIGNMENT);
    auto padBegin = mlir::SmallVector<int64_t>(inputShape.size(), 0);
    auto padEnd = mlir::SmallVector<int64_t>(inputShape.size(), 0);
    padEnd[vpux::Dims4D::Act::C.ind()] = alignedInputC - inputShape[Dims4D::Act::C.ind()];
    auto inputExpandOp = rewriter.create<IE::ExpandOp>(loc, input, getIntArrayAttr(rewriter, ArrayRef(padBegin)),
                                                       getIntArrayAttr(rewriter, ArrayRef(padEnd)));
    return inputExpandOp.getOutput();
}

// Generates the weights for GroupConvolution tasks by
// duplicating the fraction coefficients to the number of input/output channels
mlir::Value createGroupConvWeightsForBilinearInterp(mlir::PatternRewriter& rewriter, mlir::Location loc,
                                                    mlir::Value input, std::vector<double>& bilinearCoeffs,
                                                    size_t index, ShapeRef weightShape, int64_t outputSize) {
    // OC is equal with IC
    auto inputC = weightShape[Dims4D::Filter::OC];
    std::vector<vpux::type::float16> duplicatedWeights(inputC * 2);
    for (size_t i = 0; i < static_cast<size_t>(inputC); i++) {
        duplicatedWeights[2 * i] = static_cast<vpux::type::float16>(bilinearCoeffs[index]);
        duplicatedWeights[2 * i + 1] = static_cast<vpux::type::float16>(bilinearCoeffs[outputSize + index]);
    }
    const auto elemType = input.getType().cast<vpux::NDTypeInterface>().getElementType();
    const auto weightsStorageType = mlir::RankedTensorType::get(weightShape.raw(), elemType);
    return Const::createConst(rewriter, loc, weightsStorageType, ArrayRef(duplicatedWeights));
}

// Create the GroupConvolution operations that does the vertical and horizontal scaling
mlir::Value createGenericGroupConv(mlir::PatternRewriter& rewriter, mlir::Location loc, mlir::Value input,
                                   std::vector<double>& bilinearCoeffs, size_t index, ShapeRef weightShape,
                                   int64_t outputSize) {
    auto inShape = getShape(input);
    auto dilationsAttr = getIntArrayAttr(rewriter, SmallVector<int32_t>{1, 1});
    auto stridesAttr = getIntArrayAttr(rewriter, SmallVector<int32_t>{1, 1});
    auto padBeginAttr = getIntArrayAttr(rewriter, SmallVector<int32_t>{0, 0});
    auto padEndAttr = getIntArrayAttr(rewriter, SmallVector<int32_t>{0, 0});
    auto groupAttr = getIntAttr(rewriter, inShape[Dims4D::Act::C]);

    auto weights = createGroupConvWeightsForBilinearInterp(rewriter, loc, input, bilinearCoeffs, index, weightShape,
                                                           outputSize);
    auto groupConvOp = rewriter.create<IE::GroupConvolutionOp>(
            loc, input, weights, /*bias=*/nullptr, stridesAttr, padBeginAttr, padEndAttr, dilationsAttr, groupAttr,
            /*post_opAttr=*/nullptr, /*clampAttr*/ nullptr, /*outputChannels=*/nullptr, /*inputChannels=*/nullptr);
    return groupConvOp.getOutput();
}

//
// Scale on one axis
//

// Compute all fraction coefficients and input offsets
// The fraction coefficients are used to generate the GroupConvolution weights
// The input offsets are used to get the correct slices from input for generating a tile of width/height 1
void computeCoefficientsAndIndexes(std::vector<double>& allFractionCoefficients,
                                   std::vector<int32_t>& allInputSlicesOffsets, int32_t inputSize, int32_t outputSize,
                                   IE::MapCoordFuncT mapCoord) {
    double scaleValue = static_cast<double>(inputSize) / outputSize;
    for (int32_t i = 0; i < outputSize; i++) {
        const auto localMapCoord = mapCoord(i, scaleValue, outputSize, inputSize);
        const auto localComputeFractionCoefficients = IE::computeFractionCoefficients(localMapCoord.second);
        allFractionCoefficients[i] = localComputeFractionCoefficients.first;
        allFractionCoefficients[i + outputSize] = localComputeFractionCoefficients.second;
        allInputSlicesOffsets[i] = localMapCoord.first;
    }
}

}  // namespace

// Creates identify pooling operations which to ensure that all the inputs of the Concat operations that
// compose the scaled results on each axis has only NCE inputs
// This ensures that the compiler will map more optimal the generated operations on the available HW resources
mlir::Value IE::MapBilinearInterpolateOnDPUBaseRewriter::createIdentityPooling(mlir::PatternRewriter& rewriter,
                                                                               mlir::Location loc,
                                                                               mlir::Value input) const {
    const SmallVector<int64_t> poolStrides = {1, 1};
    const SmallVector<int64_t> poolKernels = {1, 1};
    const SmallVector<int64_t> pads = {0, 0};
    const auto padsAttr = getIntArrayAttr(rewriter, pads);

    auto avgPoolOp = rewriter.create<IE::AvgPoolOp>(
            loc, input, getIntArrayAttr(rewriter, poolKernels), getIntArrayAttr(rewriter, poolStrides), padsAttr,
            padsAttr, vpux::IE::RoundingTypeAttr::get(rewriter.getContext(), vpux::IE::RoundingType::FLOOR),
            mlir::UnitAttr::get(rewriter.getContext()), nullptr, nullptr, nullptr, nullptr, nullptr);

    return avgPoolOp.getOutput();
}

// Function for performing the scaling on one axis
// On each axis the processing is split in three main regions BEGIN, MIDDLE and END
//
// BEGIN region:
//            Input
//              |
//           Slice
//       first line/column
//        |    ...    |
//  Identity        Identity
// Max/AvgPool      Max/AvgPool
//
// MIDDLE region
//                 Input
//          ---------|---------
//         |                   |
//     Slice        ...       Slice
// two lines/colums       two lines/colums
//       |                        |
//   GroupConv               GroupConv
// one output line/colum   one output line/colum
//
// END region:
//            Input
//              |
//           Slice
//       last line/column
//        |    ...     |
//  Identity        Identity
// Max/AvgPool      Max/AvgPool
//
// After all the results from the three parts are concatenated on specified axis
mlir::Value IE::MapBilinearInterpolateOnDPUBaseRewriter::scaleOnAxis(mlir::PatternRewriter& rewriter,
                                                                     mlir::Location loc, mlir::Value input,
                                                                     int64_t inputSize, int64_t outputSize,
                                                                     vpux::Dim axis, IE::MapCoordFuncT mapCoord) const {
    auto scaleInputType = input.getType().cast<vpux::NDTypeInterface>();
    auto scaleInputShape = scaleInputType.getShape().raw();
    mlir::SmallVector<mlir::Value> gatheredConcatInputs;
    // To generate one line/column from the output it is needed to take two consecutive lines/columns from the input
    // Fraction coefficients refers to the weight each of the two lines/columns has in computing the output line/column
    std::vector<double> allFractionCoefficients(2 * outputSize);
    // Integer indexes of the first line/column from input used to compute one line/column from output
    std::vector<int32_t> allInputSlicesOffsets(outputSize);
    computeCoefficientsAndIndexes(allFractionCoefficients, allInputSlicesOffsets, checked_cast<int32_t>(inputSize),
                                  checked_cast<int32_t>(outputSize), mapCoord);

    //
    // Begin region
    //
    // Is represented by the region for which the allInputSlicesOffsets[i] < 0
    // For this region there is needed to take the first line/column from the input and duplicate it for all
    // allInputSlicesOffsets[i] < 0
    size_t outputSliceIndex = 0;
    mlir::SmallVector<int64_t> groupConvWeightsShapeVector = {scaleInputShape[vpux::Dims4D::Act::C.ind()], 1, 1, 1};
    // On the interpolation axis the kernel size is 2
    groupConvWeightsShapeVector[axis.ind()] = 2;
    auto groupConvWeightsShape = Shape{groupConvWeightsShapeVector};
    if (allInputSlicesOffsets[outputSliceIndex] < 0) {
        auto staticOffsets = mlir::SmallVector<int64_t>(scaleInputShape.size(), 0);
        mlir::SmallVector<int64_t> staticSizes = to_small_vector(scaleInputShape);
        staticSizes[axis.ind()] = 1;
        // Create Slice op with first line/column
        auto beginSliceOp = rewriter.create<IE::SliceOp>(loc, input, getIntArrayAttr(rewriter, ArrayRef(staticOffsets)),
                                                         getIntArrayAttr(rewriter, ArrayRef(staticSizes)));
        while (allInputSlicesOffsets[outputSliceIndex] < 0 && outputSliceIndex < checked_cast<size_t>(outputSize)) {
            // In order to benefit from some further optimizations it is needed that all the concat inputs to be NCE
            // operations. So some identity pooling operations are artificially inserted in order to make this
            // optimizations happen
            auto identityOpResult = createIdentityPooling(rewriter, loc, beginSliceOp.getResult());
            gatheredConcatInputs.push_back(identityOpResult);
            outputSliceIndex++;
        }
    }

    //
    // Middle region
    //
    // Is represented by the region for which the allInputSlicesOffsets[i] >= 0 && < outputSize
    // For this region there is needed to take two consecutive lines/columns first from the input starting with
    // allInputSlicesOffsets[i] and generate one line/column from the output with them
    while (allInputSlicesOffsets[outputSliceIndex] < (inputSize - 1) &&
           outputSliceIndex < checked_cast<size_t>(outputSize)) {
        auto staticOffsets = mlir::SmallVector<int64_t>(scaleInputShape.size(), 0);
        staticOffsets[axis.ind()] = allInputSlicesOffsets[outputSliceIndex];
        mlir::SmallVector<int64_t> staticSizes = to_small_vector(scaleInputShape);
        staticSizes[axis.ind()] = 2;
        // Create Slice op with two consecutive lines/columns
        auto middleSliceOp =
                rewriter.create<IE::SliceOp>(loc, input, getIntArrayAttr(rewriter, ArrayRef(staticOffsets)),
                                             getIntArrayAttr(rewriter, ArrayRef(staticSizes)));
        const auto currentOffset = allInputSlicesOffsets[outputSliceIndex];
        // Create all GroupConv that uses the uses the current slice
        while (allInputSlicesOffsets[outputSliceIndex] == currentOffset &&
               outputSliceIndex < checked_cast<size_t>(outputSize)) {
            auto groupConvResult =
                    createGenericGroupConv(rewriter, loc, middleSliceOp.getResult(), allFractionCoefficients,
                                           outputSliceIndex, groupConvWeightsShape, outputSize);
            gatheredConcatInputs.push_back(groupConvResult);
            outputSliceIndex++;
        }
    }

    //
    // End region
    //
    // Is represented by the region where the allInputSlicesOffsets[i] >= outputSize
    // For this region there is needed to take the last line/column from the input and duplicate it
    if (outputSliceIndex < checked_cast<size_t>(outputSize)) {
        auto staticOffsets = mlir::SmallVector<int64_t>(scaleInputShape.size(), 0);
        staticOffsets[axis.ind()] = allInputSlicesOffsets[outputSliceIndex];
        mlir::SmallVector<int64_t> staticSizes = to_small_vector(scaleInputShape);
        staticSizes[axis.ind()] = 1;
        // Create Slice op with last line/column
        auto endSliceOp = rewriter.create<IE::SliceOp>(loc, input, getIntArrayAttr(rewriter, ArrayRef(staticOffsets)),
                                                       getIntArrayAttr(rewriter, ArrayRef(staticSizes)));
        while (outputSliceIndex < checked_cast<size_t>(outputSize)) {
            // In order to benefit from some further optimizations implemented for Concat->Slice optimizations
            // it is needed that all the concat inputs to be NCE operations
            // So some identity pooling operations are artificially inserted in order to make this optimization happen
            auto identityOpResult = createIdentityPooling(rewriter, loc, endSliceOp.getResult());
            gatheredConcatInputs.push_back(identityOpResult);
            outputSliceIndex++;
        }
    }

    //
    // Final Concat of all operations that do the scale
    //
    auto outputConcatOp = rewriter.create<IE::ConcatOp>(loc, gatheredConcatInputs, axis);
    return outputConcatOp.getOutput();
}

mlir::LogicalResult IE::MapBilinearInterpolateOnDPUBaseRewriter::matchAndRewrite(
        IE::InterpolateOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Got '{1}' at '{2}'", origOp->getName(), origOp->getLoc());

    const auto attrs = origOp.getAttr();
    const auto axesValue = parseIntArrayAttr<int64_t>(origOp.getAxesAttrAttr());
    auto mapCoord = IE::getMapCoordMethod(attrs.getCoordMode().getValue());

    // Get input shape info
    auto inputType = origOp.getInput().getType().cast<vpux::NDTypeInterface>();
    const auto inputShape = inputType.getShape().raw();
    auto inputW = inputShape[Dims4D::Act::W.ind()];
    auto inputH = inputShape[Dims4D::Act::H.ind()];
    auto inputC = inputShape[Dims4D::Act::C.ind()];

    // Get output shape info
    auto outputType = origOp.getOutput().getType().cast<vpux::NDTypeInterface>();
    const auto outputShape = outputType.getShape().raw();
    auto outputW = outputShape[Dims4D::Act::W.ind()];
    auto outputH = outputShape[Dims4D::Act::H.ind()];

    //
    // Alignment of input channels
    //
    mlir::Value scaleInput = origOp.getInput();
    if (inputC % VPU::NCEInvariant::VPU_CHANNEL_ALIGNMENT != 0) {
        auto alignedInput = alignInputChannels(rewriter, origOp->getLoc(), origOp.getInput(), inputShape);
        scaleInput = alignedInput;
        _log.trace("The input channels needed alignment");
    }

    //
    // Vertical scale
    //
    if (llvm::find(axesValue, Dims4D::Act::H.ind()) != axesValue.end() && inputH != outputH) {
        scaleInput =
                scaleOnAxis(rewriter, origOp->getLoc(), scaleInput, inputH, outputH, vpux::Dims4D::Act::H, mapCoord);
        _log.nest().trace("The vertical scaling is done.");
    }

    //
    // Horizontal scale
    //
    if (llvm::find(axesValue, Dims4D::Act::W.ind()) != axesValue.end() && inputW != outputW) {
        scaleInput =
                scaleOnAxis(rewriter, origOp->getLoc(), scaleInput, inputW, outputW, vpux::Dims4D::Act::W, mapCoord);
        _log.nest().trace("The horizontal scaling is done.");
    }

    //
    // Slice back to the initial input channels
    //
    if (inputC % VPU::NCEInvariant::VPU_CHANNEL_ALIGNMENT != 0) {
        auto staticOffsets = mlir::SmallVector<int64_t>(inputShape.size(), 0);
        mlir::SmallVector<int64_t> staticSizes = to_small_vector(outputShape);
        rewriter.replaceOpWithNewOp<IE::SliceOp>(origOp, scaleInput, getIntArrayAttr(rewriter, ArrayRef(staticOffsets)),
                                                 getIntArrayAttr(rewriter, ArrayRef(staticSizes)));
        _log.trace("Sliced back to original input channels.");
    } else {
        rewriter.replaceOp(origOp, scaleInput);
        _log.trace("Replaced the Interpolate with final Concat");
    }

    return mlir::success();
}

bool vpux::IE::isLegalInterpolateOp(IE::InterpolateOp op, bool interpolateAsSEOp, LogCb logCb) {
    if (interpolateAsSEOp) {
        if (VPU::NCEInterpolateOp::isSupported(op, logCb, /*checkLayout=*/false, /*checkChannelAlignment=*/false,
                                               /*checkBatch=*/false)) {
            return true;
        }
    }

    const auto attrs = op.getAttr();
    const auto interpMode = attrs.getMode().getValue();
    const auto antiAlias = attrs.getAntialias().getValue();
    const auto inputShape = getShape(op.getInput());
    const auto outputShape = getShape(op.getOutput());

    if ((interpMode != IE::InterpolateMode::LINEAR_ONNX && interpMode != IE::InterpolateMode::LINEAR) || antiAlias) {
        return true;
    }

    // Only support 4D Input shape
    if (inputShape.size() != 4) {
        return true;
    }

    // Only support interpolation on W and H axes
    const auto axesValue = parseIntArrayAttr<int64_t>(op.getAxesAttrAttr());
    for (size_t i = 0; i < axesValue.size(); i++) {
        if (axesValue[i] <= 1 && outputShape[Dim(axesValue[i])] != inputShape[Dim(axesValue[i])]) {
            return true;
        }
    }

    // Is more efficient to execute interpolates with one input channel on SHAVE
    if (inputShape[Dims4D::Act::C] == 1) {
        return true;
    }

    // If the input and output of the interpolate fits in CMX then use run interpolate on Shave
    auto inputType = op.getInput().getType().cast<vpux::NDTypeInterface>();
    auto outputType = op.getOutput().getType().cast<vpux::NDTypeInterface>();
    Byte elemSizeBytes = inputType.getElemTypeSize().to<Byte>();
    Byte requiredCMXSize = inputType.getTotalAllocSize() + outputType.getTotalAllocSize();

    Byte quantizedCMXSize = requiredCMXSize / elemSizeBytes.count();
    Byte totalCMXSize = VPU::getTotalCMXSize(op);

    if (quantizedCMXSize <= totalCMXSize) {
        return true;
    }

    return false;
}
