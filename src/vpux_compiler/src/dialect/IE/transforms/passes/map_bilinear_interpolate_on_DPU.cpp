//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/transforms/passes/map_bilinear_interpolate_on_DPU.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/pooling.hpp"
#include "vpux/compiler/dialect/IE/transforms/factories/map_bilinear_interpolate_on_dpu_strategy_getter.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/dpu.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/numeric.hpp"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LLVM.h>

namespace vpux::IE {
#define GEN_PASS_DECL_MAPBILINEARINTERPOLATEONDPU
#define GEN_PASS_DEF_MAPBILINEARINTERPOLATEONDPU
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

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
    auto inputExpandOp = rewriter.create<IE::ExpandOp>(appendLoc(loc, "_expand"), input,
                                                       getIntArrayAttr(rewriter, ArrayRef(padBegin)),
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
    const auto elemType = mlir::cast<vpux::NDTypeInterface>(input.getType()).getElementType();
    const auto weightsStorageType = mlir::RankedTensorType::get(weightShape.raw(), elemType);
    return Const::createConst(rewriter, loc, weightsStorageType, ArrayRef(duplicatedWeights));
}

// Create the GroupConvolution operations that does the vertical and horizontal scaling
mlir::Value createGenericGroupConv(mlir::PatternRewriter& rewriter, mlir::Location loc, mlir::Value input,
                                   vpux::NDTypeInterface outType, std::vector<double>& bilinearCoeffs, size_t index,
                                   ShapeRef weightShape, int64_t outputSize) {
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
            /*post_opAttr=*/nullptr, /*clampAttr*/ nullptr, /*outputPadding=*/nullptr, /*inputPadding=*/nullptr);
    auto oldOutType = mlir::cast<vpux::NDTypeInterface>(groupConvOp.getOutput().getType());
    VPUX_THROW_WHEN(oldOutType == nullptr, "Expected NDTypeInterface");
    auto newOutType = oldOutType.changeElemType(outType.getElementType());
    groupConvOp->getResult(0).setType(newOutType);
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
                                                                               mlir::Location loc, mlir::Value input,
                                                                               vpux::NDTypeInterface outType) const {
    auto nestedLogger = this->_log.nest();
    nestedLogger.trace("Creating identity pooling");
    const SmallVector<int64_t> poolStrides = {1, 1};
    const SmallVector<int64_t> poolKernels = {1, 1};
    const SmallVector<int64_t> pads = {0, 0};
    const auto padsAttr = getIntArrayAttr(rewriter, pads);

    auto avgPoolOp = rewriter.create<IE::AvgPoolOp>(
            loc, input, getIntArrayAttr(rewriter, poolKernels), getIntArrayAttr(rewriter, poolStrides), padsAttr,
            padsAttr, vpux::IE::RoundingTypeAttr::get(rewriter.getContext(), vpux::IE::RoundingType::FLOOR),
            mlir::UnitAttr::get(rewriter.getContext()), nullptr, nullptr, nullptr, nullptr, nullptr);
    auto oldOutType = mlir::cast<vpux::NDTypeInterface>(avgPoolOp.getOutput().getType());
    VPUX_THROW_WHEN(oldOutType == nullptr, "Expected NDTypeInterface");
    auto newOutType = oldOutType.changeElemType(outType.getElementType());
    avgPoolOp->getResult(0).setType(newOutType);

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
                                                                     vpux::NDTypeInterface outType, int64_t inputSize,
                                                                     int64_t outputSize, vpux::Dim axis,
                                                                     IE::MapCoordFuncT mapCoord) const {
    auto scaleInputType = mlir::cast<vpux::NDTypeInterface>(input.getType());
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
    if (outputSliceIndex < checked_cast<size_t>(outputSize) && allInputSlicesOffsets[outputSliceIndex] < 0) {
        auto staticOffsets = mlir::SmallVector<int64_t>(scaleInputShape.size(), 0);
        mlir::SmallVector<int64_t> staticSizes = to_small_vector(scaleInputShape);
        staticSizes[axis.ind()] = 1;
        // Create Slice op with first line/column
        auto newLoc = appendLoc(loc, llvm::formatv("_begin_{0}_{1}", staticOffsets, staticSizes));
        auto beginSliceOp =
                rewriter.create<IE::SliceOp>(newLoc, input, getIntArrayAttr(rewriter, ArrayRef(staticOffsets)),
                                             getIntArrayAttr(rewriter, ArrayRef(staticSizes)));
        while (outputSliceIndex < checked_cast<size_t>(outputSize) && allInputSlicesOffsets[outputSliceIndex] < 0) {
            // In order to benefit from some further optimizations it is needed that all the concat inputs to be NCE
            // operations. So some identity pooling operations are artificially inserted in order to make this
            // optimizations happen
            auto identityOpResult =
                    createIdentityPooling(rewriter, appendLoc(newLoc, llvm::formatv("_{0}", outputSliceIndex)),
                                          beginSliceOp.getResult(), outType);
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
    while (outputSliceIndex < checked_cast<size_t>(outputSize) &&
           allInputSlicesOffsets[outputSliceIndex] < (inputSize - 1)) {
        auto staticOffsets = mlir::SmallVector<int64_t>(scaleInputShape.size(), 0);
        staticOffsets[axis.ind()] = allInputSlicesOffsets[outputSliceIndex];
        mlir::SmallVector<int64_t> staticSizes = to_small_vector(scaleInputShape);
        staticSizes[axis.ind()] = 2;
        // Create Slice op with two consecutive lines/columns
        auto newLoc = appendLoc(loc, llvm::formatv("_middle_{0}_{1}", staticOffsets, staticSizes));
        auto middleSliceOp =
                rewriter.create<IE::SliceOp>(newLoc, input, getIntArrayAttr(rewriter, ArrayRef(staticOffsets)),
                                             getIntArrayAttr(rewriter, ArrayRef(staticSizes)));
        const auto currentOffset = allInputSlicesOffsets[outputSliceIndex];
        // Create all GroupConv that uses the uses the current slice
        while (outputSliceIndex < checked_cast<size_t>(outputSize) &&
               allInputSlicesOffsets[outputSliceIndex] == currentOffset) {
            auto groupConvResult = createGenericGroupConv(
                    rewriter, appendLoc(newLoc, llvm::formatv("_{0}", outputSliceIndex)), middleSliceOp.getResult(),
                    outType, allFractionCoefficients, outputSliceIndex, groupConvWeightsShape, outputSize);
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
        auto newLoc = appendLoc(loc, llvm::formatv("_end_{0}_{1}", staticOffsets, staticSizes));
        auto endSliceOp =
                rewriter.create<IE::SliceOp>(newLoc, input, getIntArrayAttr(rewriter, ArrayRef(staticOffsets)),
                                             getIntArrayAttr(rewriter, ArrayRef(staticSizes)));
        while (outputSliceIndex < checked_cast<size_t>(outputSize)) {
            // In order to benefit from some further optimizations implemented for Concat->Slice optimizations
            // it is needed that all the concat inputs to be NCE operations
            // So some identity pooling operations are artificially inserted in order to make this optimization happen
            auto identityOpResult =
                    createIdentityPooling(rewriter, appendLoc(newLoc, llvm::formatv("_{0}", outputSliceIndex)),
                                          endSliceOp.getResult(), outType);
            gatheredConcatInputs.push_back(identityOpResult);
            outputSliceIndex++;
        }
    }

    //
    // Final Concat of all operations that do the scale
    //
    auto outputConcatOp = rewriter.create<IE::ConcatOp>(appendLoc(loc, "_output_concat"), gatheredConcatInputs, axis);
    return outputConcatOp.getOutput();
}

mlir::LogicalResult IE::MapBilinearInterpolateOnDPUBaseRewriter::matchAndRewrite(
        IE::InterpolateOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    const auto attrs = origOp.getAttr();
    const auto axesValue = parseIntArrayAttr<int64_t>(origOp.getAxesAttrAttr());
    auto mapCoord = IE::getMapCoordMethod(attrs.getCoordMode().getValue());

    // Get input shape info
    auto inputType = mlir::cast<vpux::NDTypeInterface>(origOp.getInput().getType());
    const auto inputShape = inputType.getShape().raw();
    auto inputW = inputShape[Dims4D::Act::W.ind()];
    auto inputH = inputShape[Dims4D::Act::H.ind()];
    auto inputC = inputShape[Dims4D::Act::C.ind()];

    // Get output shape info
    auto outputType = mlir::cast<vpux::NDTypeInterface>(origOp.getOutput().getType());
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
        scaleInput = scaleOnAxis(rewriter, takeOpLoc(origOp, "_scale_h"), scaleInput, inputType, inputH, outputH,
                                 vpux::Dims4D::Act::H, mapCoord);
        _log.nest().trace("The vertical scaling is done.");
    }

    //
    // Horizontal scale
    //
    if (llvm::find(axesValue, Dims4D::Act::W.ind()) != axesValue.end() && inputW != outputW) {
        scaleInput = scaleOnAxis(rewriter, takeOpLoc(origOp, "_scale_w"), scaleInput, outputType, inputW, outputW,
                                 vpux::Dims4D::Act::W, mapCoord);
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
    auto inputType = mlir::cast<vpux::NDTypeInterface>(op.getInput().getType());
    auto outputType = mlir::cast<vpux::NDTypeInterface>(op.getOutput().getType());

    if (interpolateAsSEOp) {
        if (VPU::NCEInterpolateOp::isSupported(op, logCb, /*checkLayout=*/false, /*checkChannelAlignment=*/false,
                                               /*checkBatch=*/false)) {
            auto convert = mlir::dyn_cast_or_null<IE::ConvertOp>(*(op.getOutput().getUsers().begin()));
            return convert == nullptr;
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

    // Use ExecutorOpInterface to determine if SHAVE is preferred
    if (auto iface = mlir::dyn_cast<IE::ExecutorOpInterface>(op.getOperation())) {
        auto execs = iface.getPreferredExecutors();
        if (!execs.empty() && execs[0] == VPU::ExecutorKind::SHAVE_ACT) {
            return true;
        }
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
    Byte elemSizeBytes = inputType.getElemTypeSize().to<Byte>();
    Byte requiredCMXSize = inputType.getTotalAllocSize() + outputType.getTotalAllocSize();

    Byte quantizedCMXSize = requiredCMXSize / elemSizeBytes.count();
    Byte totalCMXSize = VPU::getTotalCMXSize(op);

    if (quantizedCMXSize <= totalCMXSize) {
        return true;
    }

    return false;
}

namespace {

//
// MapBilinearInterpolateOnDPUPass
//

class MapBilinearInterpolateOnDPUPass final :
        public IE::impl::MapBilinearInterpolateOnDPUBase<MapBilinearInterpolateOnDPUPass> {
public:
    explicit MapBilinearInterpolateOnDPUPass(const bool interpolateAsSEOp, Logger& log)
            : _interpolateAsSEOp(interpolateAsSEOp) {
        Base::initLogger(log, Base::getArgumentName());
    }

    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) override;

private:
    void safeRunOnFunc() override;

    bool _interpolateAsSEOp = false;
};

mlir::LogicalResult MapBilinearInterpolateOnDPUPass::initialize(mlir::MLIRContext* ctx) {
    if (mlir::failed(Base::initialize(ctx))) {
        return mlir::failure();
    }

    // When this parameter has a value, it probably comes from LIT test.
    // Override the default
    if (interpolateAsSEOp.hasValue()) {
        _interpolateAsSEOp = interpolateAsSEOp.getValue();
    }

    return mlir::success();
}

void MapBilinearInterpolateOnDPUPass::safeRunOnFunc() {
    auto& ctx = getContext();
    const auto func = getOperation();
    const auto logCb = [&](const formatv_object_base& msg) {
        _log.trace("{0}", msg.str());
    };

    auto strategy = IE::createMapBilinearInterpolateOnDPUStrategy(func, _interpolateAsSEOp);

    mlir::ConversionTarget target(ctx);
    target.addLegalOp<IE::ExpandOp>();
    target.addLegalOp<IE::AvgPoolOp>();
    target.addLegalOp<IE::SliceOp>();
    target.addLegalOp<IE::ConcatOp>();
    target.addLegalOp<Const::DeclareOp>();
    target.addLegalOp<IE::GroupConvolutionOp>();
    strategy->prepareInterpolate(target, logCb);

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<IE::MapBilinearInterpolateOnDPUBaseRewriter>(&ctx, _log);

    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createMapBilinearInterpolateOnDPUPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createMapBilinearInterpolateOnDPUPass(const bool interpolateAsSEOp, Logger log) {
    return std::make_unique<MapBilinearInterpolateOnDPUPass>(interpolateAsSEOp, log);
}
