//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/IE/utils/transposed_convolution_utils.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/dpu.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/utils/auto_padding_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/clustered_op_interface_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/conv_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/generate_tiling.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_reduce_output_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/sparsity_support.hpp"
#include "vpux/compiler/dialect/VPU/utils/sprlut_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/tile_utils.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/infer_output_shape.hpp"
#include "vpux/compiler/utils/verifier_utils.hpp"

#include <mlir/IR/Matchers.h>
#include <openvino/op/convolution.hpp>

using namespace vpux;

//
// fitIntoCMX
//

bool vpux::VPU::NCEConvolutionOp::fitIntoCMX(vpux::NDTypeInterface input, vpux::NDTypeInterface filter,
                                             vpux::NDTypeInterface output) {
    return fitIntoCMX(input, filter, output, Byte(0));
}

bool vpux::VPU::NCEConvolutionOp::fitIntoCMX(vpux::NDTypeInterface input, vpux::NDTypeInterface filter,
                                             vpux::NDTypeInterface output, Byte reservedMem) {
    // These depend on a particular tile
    const auto OC = output.getShape()[Dims4D::Act::C];

    SmallVector<Byte> buffers = {input.getTotalAllocSize(), filter.getTotalAllocSize(), output.getTotalAllocSize()};

    const auto op = getOperation();
    auto ppeAttr = getPpe();
    addSprLutBufferIfPresent(ppeAttr, buffers);

    if (mlir::failed(NCEInvariant::getWeightTableBuffers(op, buffers, OC))) {
        VPUX_THROW("getWeightTableBuffers function failed");
    }
    if (mlir::failed(getReduceOutputBuffers(op, buffers, output))) {
        VPUX_THROW("getReduceOutputBuffers failed at '{0}' for op '{1}', outputTileShape '{2}', axes_value '{3}'",
                   getLoc(), op->getName(), output.getShape(), op->getAttr("axes_value"));
    }

    auto totalAvailableCMXSize =
            reservedMem.count() == 0 ? getTotalCMXSize(op).count() : getTotalCMXFragmentationAwareSize(op).count();

    return vpux::VPU::calculateAlignedBuffersMemoryRequirement(config::getArch(op), buffers).count() +
                   reservedMem.count() <=
           totalAvailableCMXSize;
}

//
// ShapeInfoOpInterface
//

mlir::LogicalResult vpux::VPU::NCEConvolutionOp::verifyShapeInfo() {
    if (mlir::failed(vpux::VPU::verifyInputIs4D(getInput()))) {
        return mlir::failure();
    }

    return vpux::VPU::verifyInputIs4D(getFilter());
}

//
// isSupported
//

bool vpux::VPU::NCEConvolutionOp::isSupported(IE::ConvolutionOp op, LogCb logCb, bool checkLayout,
                                              bool checkChannelAlignment) {
    return VPU::isSupportedConv(op, logCb, checkLayout, checkChannelAlignment);
}

//
// verify
//

static mlir::LogicalResult verifyConv(mlir::Location loc, mlir::Operation* op, VPU::NCEConvolutionOpAdaptor opAdaptor,
                                      mlir::Value output) {
    const auto filterShape = Shape(opAdaptor.getStaticRawFilterShape());
    const auto KY = filterShape[Dims4D::Filter::KY];
    const auto KX = filterShape[Dims4D::Filter::KX];

    VPUX_THROW_WHEN(mlir::ShapedType::isDynamic(KY) || mlir::ShapedType::isDynamic(KX),
                    "Dynamic kernel size is not supported for NCE operations");

    const auto kernelStrides = Shape(parseIntArrayAttr<int64_t>(opAdaptor.getStrides()));
    const auto padAttr = opAdaptor.getPad();
    const auto weightsTableShape = opAdaptor.getWeightsTable() == nullptr
                                           ? std::nullopt
                                           : std::optional<vpux::ShapeRef>(getShape(opAdaptor.getWeightsTable()));

    return VPU::verifyConvUtil(loc, op, filterShape, kernelStrides, padAttr, weightsTableShape, output);
}

mlir::LogicalResult vpux::VPU::NCEConvolutionOp::verify() {
    auto op = getOperation();
    const auto arch = config::getArch(op);

    // Skip checks if architecture is unknown since all of them depend on the architecture used
    if (arch == config::ArchKind::UNKNOWN) {
        return mlir::success();
    }

    if (mlir::failed(isTypeSignedOrUnsigned(*this, getFilter()))) {
        return mlir::failure();
    }

    if (mlir::failed(VPU::NCEInvariant::verifyWeightTables(op))) {
        return mlir::failure();
    }

    if (mlir::failed(vpux::VPU::verifyNCEOp(op))) {
        return mlir::failure();
    }

    const NCEConvolutionOpAdaptor convAdaptor(op->getOperands(), op->getAttrDictionary(), op->getPropertiesStorage(),
                                              op->getRegions());
    if (mlir::failed(verifyConv(getOperation()->getLoc(), op, convAdaptor, getOutput()))) {
        return mlir::failure();
    }

    const auto inputType = mlir::cast<vpux::NDTypeInterface>(getInput().getType());
    const auto outputType = mlir::cast<vpux::NDTypeInterface>(getOutput().getType());
    const auto filterType = mlir::cast<vpux::NDTypeInterface>(getFilter().getType());

    const auto alignedFilterShape = getBoundedShape(filterType);
    const auto expectedAlignedFilterShape = inferAlignedFilterShape(inputType, outputType, filterType);

    if (alignedFilterShape != expectedAlignedFilterShape) {
        return errorAt(op, "Got wrong shape for NCE Convolution 'filter' '{0}', expected '{1}'", alignedFilterShape,
                       expectedAlignedFilterShape);
    }

    return mlir::success();
}

Shape vpux::VPU::NCEConvolutionOp::inferAlignedFilterShape(NDTypeInterface input, NDTypeInterface output,
                                                           NDTypeInterface filter) {
    const auto rawFilterShape = Shape(this->getStaticRawFilterShape());
    const auto KY = rawFilterShape[Dims4D::Filter::KY];
    const auto KX = rawFilterShape[Dims4D::Filter::KX];
    const auto inputShape = getBoundedShape(input);
    const auto outputShape = getBoundedShape(output);

    // When IDU autopad is used and the weight pointers are computed during the inference by the DPU, the weight set
    // must have the input channels aligned to 16 as a hardware requirement. For this reason, the filter shape is larger
    // than normally expected, even though there are fewer than 16 channels for the input (due to IDU autopad)
    const auto usesIDUAutopad = inputShape[Dims4D::Act::C] < VPU::NCEInvariant::VPU_CHANNEL_ALIGNMENT;
    const auto weightSetsNeedPaddedIC = usesIDUAutopad && getWeightsTable() == nullptr;
    const auto IC = weightSetsNeedPaddedIC ? VPU::NCEInvariant::VPU_CHANNEL_ALIGNMENT : inputShape[Dims4D::Act::C];
    const auto OC = outputShape[Dims4D::Act::C];

    // Without HW weight set packing the per-OC weight set must be aligned to 16 bytes (in elements,
    // that is `getAlignment()`). When packing is supported, the constraint is relaxed to byte alignment,
    // which corresponds to `getAlignment() / 16` elements (clamped to at least 1 for >=byte-sized types).
    const auto alignment = VPU::NCEInvariant::getWeightSetAlignment(getOperation(), filter.getElementType());

    const auto remainder = (IC * KY * KX) % alignment;

    // In case IDU autopad is used (i.e. IC<16), the filter shape is always flattened and aligned
    if (remainder == 0 && !usesIDUAutopad) {
        return Shape{OC, IC, KY, KX};
    }

    const auto padding = (remainder > 0) ? (alignment - remainder) : 0;
    return Shape{OC, 1, 1, IC * KY * KX + padding};
}

//
// InferTypeOpInterface
//

mlir::LogicalResult vpux::VPU::NCEConvolutionOp::inferReturnTypes(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange /*regions*/,
        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    NCEConvolutionOpAdaptor op(operands, attrs, prop);
    if (mlir::failed(op.verify(loc))) {
        return mlir::failure();
    }

    const auto inShape = getShape(op.getInput());
    // RawFilterShape can have either static or dynamic dimensions, so we need to resolve it before inferring the output
    // shape.
    const auto resolvedFilterShape = VPU::resolveRawFilterShape(op.getStaticRawFilterShape(), op.getRawFilterShape());
    const auto filterShape = Shape(resolvedFilterShape);

    const auto inputChannels = inShape[Dims4D::Act::C];
    const auto filterInputChannels = filterShape[Dims4D::Filter::IC];

    if (!mlir::ShapedType::isDynamic(inputChannels) && !mlir::ShapedType::isDynamic(filterInputChannels) &&
        inputChannels != filterInputChannels) {
        return errorAt(loc, "Input tensor channels and filter shape must be the same");
    }

    const auto windowStrides = parseIntArrayAttr<int64_t>(op.getStrides());
    const auto windowDilations = SmallVector<int64_t>({1, 1});

    const auto padTop = op.getPad().getTop().getValue().getSExtValue();
    const auto padBottom = op.getPad().getBottom().getValue().getSExtValue();
    const auto padLeft = op.getPad().getLeft().getValue().getSExtValue();
    const auto padRight = op.getPad().getRight().getValue().getSExtValue();

    const auto dataPaddingBelow = ov::CoordinateDiff({padTop, padLeft});
    const auto dataPaddingAbove = ov::CoordinateDiff({padBottom, padRight});

    auto inputType = mlir::cast<vpux::NDTypeInterface>(op.getInput().getType());
    auto filterType = mlir::cast<vpux::NDTypeInterface>(op.getFilter().getType());

    const auto inShapeInfo = ShapeInfo::fromNDType(inputType);
    auto filterShapeInfo = ShapeInfo::fromNDType(filterType);
    filterShapeInfo.shape = filterShape.raw();

    // When the actual filter OC is static and differs from rawFilterShape OC (e.g. after loop unrolling
    // a channel-tiled convolution), use the actual filter OC so the inferred output channels are correct.
    const auto actualFilterOC = filterType.getShape()[Dims4D::Filter::OC];
    if (!mlir::ShapedType::isDynamic(actualFilterOC) &&
        filterShapeInfo.shape[Dims4D::Filter::OC.ind()] != actualFilterOC) {
        filterShapeInfo.shape[Dims4D::Filter::OC.ind()] = actualFilterOC;
    }

    auto shapeInfo = inferConvolutionOutputShapeInfo(inShapeInfo, filterShapeInfo, filterType, windowStrides,
                                                     dataPaddingBelow, dataPaddingAbove, windowDilations);
    const auto outType =
            vpux::getTensorType(ShapeRef(shapeInfo.shape), inputType.getElementType(), inputType.getDimsOrder(),
                                /*memSpace=*/nullptr, BoundsRef(shapeInfo.bounds), /*DynamicDimsMask=*/{});
    inferredReturnTypes.push_back(outType);

    // Infer the extra NCE output types if any
    auto resultSegmentSizes = op.getProperties().getResultSegmentSizes();

    return inferReduceExtraNCETypes(loc, outType, op.getAxesValue(), resultSegmentSizes, inferredReturnTypes);
}

//
// TilingBuilderOpInterface
//

vpux::InputTiling vpux::VPU::NCEConvolutionOp::backInferTileInfo(const vpux::TileInfo& outputTile, vpux::Logger log) {
    const auto origInputShape = getBoundedShape(getInput());
    const auto origFilterShape = Shape(getStaticRawFilterShape());
    const auto origPadding = toPadInfo(getPad());

    // This op incorporates bias values in WeightsTable
    const auto origBiasShape = ShapeRef();

    auto inputTiling =
            backInferConvTile(outputTile, origInputShape, origFilterShape, origBiasShape, getStrides(), origPadding);
    VPUX_THROW_UNLESS(mlir::succeeded(checkAndAlignActInputTiling(
                              mlir::cast<VPU::NCEOpInterface>(*this->getOperation()), inputTiling, log)),
                      "Failed to get an aligned act input tiling");

    // Remove bias input tile if present
    if (inputTiling.tiles.size() > 2) {
        // Drop the bias tile
        inputTiling.tiles.pop_back();
    }

    // For conv with strides > 1, the input tile should be the same as the original input shape on the not tiled axis
    const auto windowStrides = parseIntArrayAttr<int64_t>(getStrides());
    const auto hasNonOneStrides = llvm::any_of(windowStrides, [](auto stride) {
        return stride > 1;
    });
    const auto& tilingAxis = inputTiling.tiles.front().axis;
    const auto hasAxisConfigured = llvm::any_of(tilingAxis, [](auto axis) {
        return axis > 1;
    });
    if (hasNonOneStrides && hasAxisConfigured) {
        for (auto item : tilingAxis | indexed) {
            const auto dim = Dim(item.index());
            const auto axis = item.value();
            auto& tiledInShape = inputTiling.tiles.front().shape;
            if (axis == 1 && origInputShape[dim] != tiledInShape[dim]) {
                tiledInShape[dim] = origInputShape[dim];
            }
        }
    }

    // Adjust filter tile for the aligned filter
    inputTiling.tiles[1].shape = getShape(getFilter()).toValues();
    inputTiling.tiles[1].shape[Dims4D::Filter::OC] = outputTile.shape[Dims4D::Act::C];

    auto nceOp = mlir::cast<VPU::NCEConvolutionOp>(getOperation());
    if (nceOp.getWeightsTable()) {
        inputTiling.tiles.push_back(
                VPU::getWeightsTableTile(this, outputTile, VPU::getWeightsChannelsAutopad(getOperation())));
    }
    if (nceOp.getWeightTableScale()) {
        inputTiling.tiles.push_back(
                VPU::getScaleTableTile(this, outputTile, VPU::getWeightsChannelsAutopad(getOperation())));
    }
    if (nceOp.getWeightTableBias()) {
        inputTiling.tiles.push_back(
                VPU::getBiasTableTile(this, outputTile, VPU::getWeightsChannelsAutopad(getOperation())));
    }
    if (nceOp.getWeightZeroPoints()) {
        inputTiling.tiles.push_back(
                VPU::getZeroPointTableTile(this, outputTile, VPU::getWeightsChannelsAutopad(getOperation())));
    }

    return inputTiling;
}

void vpux::VPU::NCEConvolutionOp::adjustAttrs(const TilingInfo& inputTiling, const TileInfo& outputTile) {
    VPU::adjustPaddings(this, inputTiling);
    VPU::adjustRawFilterShape(this, outputTile);
}

vpux::OutputTiling vpux::VPU::NCEConvolutionOp::getOutputTiling(const vpux::TileInfo& firstOutputTile,
                                                                vpux::Logger /*log*/) {
    OutputTiling outputTiling;
    outputTiling.push_back(firstOutputTile);
    const auto reduceOutputTiles = VPU::getReduceOutputTiling(getOperation(), firstOutputTile);
    outputTiling.append(reduceOutputTiles.begin(), reduceOutputTiles.end());
    return outputTiling;
}

vpux::TileInfo vpux::VPU::NCEConvolutionOp::getMainOutputTile(mlir::OpResult secondaryOutput,
                                                              const vpux::TileInfo& secondaryOutputTile,
                                                              vpux::Logger /*log*/) {
    return VPU::getMainTileFromReduceOutputTiling(getOperation(), {secondaryOutput, secondaryOutputTile});
}

mlir::FailureOr<OutputTiling> vpux::VPU::NCEConvolutionOp::getTilingStrategy(TilingMode tilingMode, Logger log) {
    return vpux::getHWLayerTilingStrategy(this->getOperation(), tilingMode, log);
}

//
// ClusteredOpInterface
//

bool vpux::VPU::NCEConvolutionOp::checkStrategyCompatibility(VPU::MultiClusterStrategy strategy, size_t) {
    auto nceOp = mlir::cast<VPU::NCEConvolutionOp>(getOperation());
    const auto isCompatible = VPU::isSEPConvCompatibleWithClusterStrategy(nceOp, strategy);
    if (isCompatible.has_value()) {
        return isCompatible.value();
    }

    const auto outputType = mlir::cast<vpux::NDTypeInterface>(getOutput().getType());
    const auto outputDimsOrder = outputType.getDimsOrder();
    // Unsupported to broadcast the lowest dimension
    // Track E#120804
    if (outputDimsOrder.dimAt(outputDimsOrder.numDims() - 1) == Dims4D::Act::H) {
        // SplitOverKernel tiles over C; reduce outputs have C=1 and cannot be partitioned per-cluster.
        if (VPU::hasReduceOutputs(getOperation())) {
            return strategy == VPU::MultiClusterStrategy::Clustering ||
                   strategy == VPU::MultiClusterStrategy::SplitOverHeight;
        }
        return strategy == VPU::MultiClusterStrategy::Clustering ||
               strategy == VPU::MultiClusterStrategy::SplitOverHeight ||
               strategy == VPU::MultiClusterStrategy::SplitOverKernel;
    }

    const auto batchSize = outputType.getShape()[Dims4D::Act::N];
    const auto enabledTileNum = config::getNumOfTiles(getOperation());

    if (batchSize > 1 && batchSize <= enabledTileNum) {
        return strategy == VPU::MultiClusterStrategy::SplitOverBatch;
    }

    // SplitOverKernel uses num_tiles over C. For ops with active reduce outputs the reduce result
    // has C=1, so per-cluster partials cannot be distributed across clusters.
    if (VPU::hasReduceOutputs(getOperation())) {
        return strategy == VPU::MultiClusterStrategy::Clustering ||
               strategy == VPU::MultiClusterStrategy::SplitOverHeight ||
               strategy == VPU::MultiClusterStrategy::HKSwitch;
    }

    return strategy == VPU::MultiClusterStrategy::Clustering ||
           strategy == VPU::MultiClusterStrategy::SplitOverHeight ||
           strategy == VPU::MultiClusterStrategy::SplitOverKernel || strategy == VPU::MultiClusterStrategy::HKSwitch;
}

vpux::VPU::DistributionInfo vpux::VPU::NCEConvolutionOp::getExplicitDistributionInfoAttr(
        vpux::ShapeRef shape, vpux::VPU::DistributionMode distributionMode, ArrayRef<int64_t> numTiles,
        const int64_t numClusters, ArrayRef<int64_t> alignment, const bool uniformDistributedSegments,
        const vpux::VPU::OverlapDistributionParams& overlapParams,
        const std::optional<ArrayRef<int64_t>> memoryNumTiles) {
    return VPU::getNCEExplicitDistributionInfo(mlir::dyn_cast<VPU::NCEOpInterface>(getOperation()), shape,
                                               distributionMode, numTiles, numClusters, alignment,
                                               uniformDistributedSegments, overlapParams, memoryNumTiles);
}

// Each cluster should compute at least one output line. Therefore in order for a layer to be SOH
// compatible it must have an output height of at least the number of clusters
// specified for compilation.
// For example for 4 cluster compilation the output height must be a minimum of 4.
bool VPU::NCEConvolutionOp::isOperationSplitOverHeightCompatible(const vpux::TileInfo& oriOutputTile) {
    return VPU::isNCEOpSplitOverHeightCompatible(getOperation(), getInput(), getBoundedShape(getOutput()),
                                                 oriOutputTile, false);
}

bool VPU::NCEConvolutionOp::isOperationSplitOverWidthCompatible(ShapeRef outputShape, ShapeRef offset, ShapeRef axis) {
    return VPU::isOperationSplitOverWidthCompatible(getOperation(), outputShape, offset, axis);
}

bool VPU::NCEConvolutionOp::isOperationSplitOverKernelCompatible(ShapeRef outputShape, ShapeRef offset, ShapeRef axis) {
    // SplitOverKernel tiles over C. Reduce outputs have C=1, so SOK is incompatible.
    if (VPU::hasReduceOutputs(getOperation())) {
        return false;
    }
    return VPU::isOperationSplitOverKernelCompatible(getOperation(), outputShape, offset, axis);
}

bool VPU::NCEConvolutionOp::isOperationSplitOverBatchCompatible(vpux::ShapeRef outputShape) {
    return VPU::isOperationSplitOverBatchCompatible(getOperation(), outputShape);
}

bool VPU::NCEConvolutionOp::doesLayerFitIntoCMX(VPU::MultiClusterStrategy strategy,
                                                SiblingOpsAnalysis& siblingsAnalysis, Byte reservedMem) {
    auto nceOp = mlir::cast<VPU::NCEConvolutionOp>(getOperation());
    auto nceOpInterface = mlir::cast<VPU::NCEOpInterface>(getOperation());
    auto output = mlir::cast<vpux::NDTypeInterface>(getOutput().getType());
    auto numClusters = VPU::getOptimalNumClusters(nceOp, output.getShape(), strategy);

    // These depend on a particular tile
    const auto OC = output.getShape()[Dims4D::Act::C];

    const auto outputDistributionMap = std::make_pair(
            output, getOutputDistributionAttrFromOp(nceOp, output, numClusters, strategy, siblingsAnalysis));

    SmallVector<Byte> buffers = {
            VPU::getTotalAllocSizeWithDistribution(
                    getInput().getType(), getActivationDistributionAttrFromOp(nceOp, getInput(), getInput().getType(),
                                                                              numClusters, strategy, siblingsAnalysis)),
            VPU::getTotalAllocSizeWithDistribution(
                    getFilter().getType(),
                    getFilterDistributionAttrFromOp(nceOpInterface, getFilter().getType(), numClusters, strategy)),
            VPU::getTotalAllocSizeWithDistribution(outputDistributionMap.first, outputDistributionMap.second)};

    auto ppeAttr = getPpe();
    addSprLutBufferIfPresent(ppeAttr, buffers);

    const auto op = getOperation();
    if (mlir::failed(NCEInvariant::getWeightTableBuffers(op, buffers, OC))) {
        VPUX_THROW("getWeightTableBuffers function failed");
    }
    if (mlir::failed(getReduceOutputBuffers(op, buffers, outputDistributionMap))) {
        VPUX_THROW("getReduceOutputBuffers function failed");
    }

    auto totalAvailableCMXSize = reservedMem.count() == 0 ? VPU::getTotalCMXSize(op).count()
                                                          : VPU::getTotalCMXFragmentationAwareSize(op).count();

    return vpux::VPU::calculateAlignedBuffersMemoryRequirement(config::getArch(op), buffers).count() +
                   reservedMem.count() <=
           totalAvailableCMXSize;
}

bool VPU::NCEConvolutionOp::doesLayerChangeOutputAlignmentFitIntoCMX(
        VPU::MultiClusterStrategy strategy, VPU::DistributedTypeInterface newDistributedTensorType) {
    auto nceOp = mlir::cast<VPU::NCEConvolutionOp>(getOperation());
    auto nceOpInterface = mlir::cast<VPU::NCEOpInterface>(getOperation());
    auto numClusters = VPU::getOptimalNumClusters(
            nceOp, mlir::cast<vpux::NDTypeInterface>(nceOp.getOutput().getType()).getShape(), strategy);
    auto distributedInputType = getDistributedActivationTypeFromOp(nceOp, nceOp.getInput(), nceOp.getInput().getType(),
                                                                   numClusters, strategy);
    auto distributedFilterType =
            getDistributedFilterTypeFromOp(nceOpInterface, nceOp.getFilter().getType(), numClusters, strategy);
    return fitIntoCMX(distributedInputType, distributedFilterType, newDistributedTensorType);
}

/*
 * Return the mixed raw filter shape by combining the static and dynamic raw filter shape values into a single
 * SmallVector of OpFoldResults.
 */
SmallVector<mlir::OpFoldResult> vpux::VPU::NCEConvolutionOp::getMixedRawFilterShape() {
    mlir::Builder builder(getContext());
    return mlir::getMixedValues(getStaticRawFilterShape(), getRawFilterShape(), builder);
}

/*
 * Return the constant raw filter shape by extracting the constant values from the mixed raw filter shape.
 */
SmallVector<int64_t> vpux::VPU::NCEConvolutionOp::getConstRawFilterShape() {
    auto vals = mlir::getConstantIntValues(getMixedRawFilterShape());
    VPUX_THROW_WHEN(!vals.has_value(), "Cannot get constant raw filter shape from NCEConvolutionOp '{0}'", getLoc());
    return vals.value();
}

DimArr vpux::VPU::NCEConvolutionOp::restrictedFusionAxes() {
    return {Dims4D::Act::C};
}

bool vpux::VPU::NCEConvolutionOp::isVFSupported() {
    return getReduceTensorMinMax() == nullptr;
}

vpux::NDTypeInterface vpux::VPU::NCEConvolutionOp::getDistributedTypeForOpOperand(
        mlir::OpOperand& operand, bool hasExplicitDistributedAttr, SiblingOpsAnalysis& siblingsAnalysis) {
    auto clusteredOp = mlir::cast<VPU::ClusteredOpInterface>(getOperation());
    auto origOp = mlir::cast<VPU::NCEConvolutionOp>(getOperation());
    const auto strategy = clusteredOp.getMultiClusterStrategy().value();

    if (operand.get() == origOp.getInput()) {
        return VPU::getDistributedActivationTypeForOpOperand(clusteredOp, origOp.getInput(), strategy,
                                                             hasExplicitDistributedAttr, siblingsAnalysis);
    } else if (operand.get() == origOp.getFilter()) {
        return VPU::getDistributedWeightsTypeForOpOperand(clusteredOp, origOp.getFilter(), strategy,
                                                          hasExplicitDistributedAttr, siblingsAnalysis);
    } else if (operand.get() == origOp.getWeightsTable() || operand.get() == origOp.getWeightTableScale() ||
               operand.get() == origOp.getWeightTableBias() || operand.get() == origOp.getWeightZeroPoints()) {
        return VPU::getDistributedWeightsTypeForOpOperand(clusteredOp, operand.get(), strategy,
                                                          hasExplicitDistributedAttr, siblingsAnalysis);
    }
    VPUX_THROW("Failed to compute distributed type for op {0}", clusteredOp);
    return nullptr;
}

//
// sparsitySupport
//

vpux::VPU::SparsitySupport vpux::VPU::NCEConvolutionOp::sparsitySupport() {
    // Super-dense mode does not support ODU sparsity
    const auto outputType = mlir::cast<vpux::NDTypeInterface>(getOutput().getType());
    auto excludeMode = VPU::NCESparsity::bitwiseNot(VPU::SparsitySupport::NONE);
    if (VPU::NCESparsity::isSuperdenseRequired(outputType.getDimsOrder(), outputType.getShape(),
                                               outputType.getElementType())) {
        excludeMode = VPU::NCESparsity::bitwiseNot(VPU::SparsitySupport::SPARSE_OUTPUTS);
    }

    return NCESparsity::FULLY_SUPPORTED_SPARSITY_MODE & excludeMode;
}

mlir::LogicalResult vpux::VPU::NCEConvolutionOp::verifyKernel(IE::ConvolutionOp origOp, Logger log) {
    log.setName("NCEInvariant");

    if (mlir::cast<vpux::NDTypeInterface>(origOp.getInput().getType()).getRank() != 4) {
        return mlir::failure();
    }

    const auto dilations = parseIntArrayAttr<int64_t>(origOp.getDilations());
    if (dilations[0] != 1 || dilations[1] != 1) {
        log.trace("[{0}] Unsupported kernel dilations '{1}'", origOp->getLoc(), dilations);
        return mlir::failure();
    }

    const auto filterShape = getShape(origOp.getFilter());
    const auto KY = filterShape[Dims4D::Filter::KY];
    const auto KX = filterShape[Dims4D::Filter::KX];

    const auto kernelStrides = parseIntArrayAttr<int64_t>(origOp.getStrides());
    const auto SY = kernelStrides[0];
    const auto SX = kernelStrides[1];

    const auto padsBegin = parseIntArrayAttr<int64_t>(origOp.getPadsBegin());
    const auto padsEnd = parseIntArrayAttr<int64_t>(origOp.getPadsEnd());
    const auto padTop = padsBegin[0];
    const auto padBottom = padsEnd[0];
    const auto padLeft = padsBegin[1];
    const auto padRight = padsEnd[1];

    return NCEInvariant::verifyKernel(origOp, KY, KX, SY, SX, padTop, padBottom, padLeft, padRight, log);
}

mlir::LogicalResult vpux::VPU::NCEConvolutionOp::verifyKernel(IE::TransposedConvolutionOp origOp, Logger log) {
    log.setName("NCEInvariant");

    if (mlir::failed(IE::canConvertTransposedConvToConv(origOp))) {
        return mlir::failure();
    }

    const auto filterShape = getShape(origOp.getFilter());
    const auto KY = filterShape[Dims4D::Filter::KY];
    const auto KX = filterShape[Dims4D::Filter::KX];

    const auto SY = 1;
    const auto SX = 1;

    const auto padTop = 0;
    const auto padBottom = 0;
    const auto padLeft = 0;
    const auto padRight = 0;

    return NCEInvariant::verifyKernel(origOp, KY, KX, SY, SX, padTop, padBottom, padLeft, padRight, log);
}

mlir::LogicalResult vpux::VPU::NCEConvolutionOp::verifyConvCMX(mlir::Location loc, mlir::ModuleOp module,
                                                               vpux::NDTypeInterface inputType,
                                                               vpux::NDTypeInterface filterType,
                                                               vpux::NDTypeInterface outputType,
                                                               mlir::ArrayAttr /*kernelStrides*/, Logger log) {
    VPUX_THROW_UNLESS(mlir::isa<VPU::NCEConvolutionOp>(module.getOperation()),
                      "The operation has to be a NCEConvolutionOp");
    log.setName("NCEInvariant");

    const auto filterShape = filterType.getShape();
    const auto OC = filterShape[Dims4D::Filter::OC];
    const auto IC = filterShape[Dims4D::Filter::IC];
    const auto KY = filterShape[Dims4D::Filter::KY];
    const auto KX = filterShape[Dims4D::Filter::KX];

    const auto alignment = VPU::NCEInvariant::getAlignment(outputType.getElementType());

    if (OC % alignment != 0) {
        log.debug("[{0}] Output channels count of depthwise convolution must be a multiple of {1}, got {2}", loc,
                  alignment, OC);
        return mlir::failure();
    }

    const auto inOrder = inputType.getDimsOrder();

    auto convOp = mlir::cast<VPU::NCEConvolutionOp>(module.getOperation());
    Byte requiredCMX;
    if (inOrder == DimsOrder::NHWC) {
        requiredCMX = VPU::getRequiredCMXSizeForNCEOps(convOp, {inputType, filterType, outputType}, OC);
    } else if (inOrder == DimsOrder::NCHW) {
        const auto remainder = (IC * KY * KX) % alignment;
        VPUX_THROW_UNLESS(remainder >= 0, "Channel alignment cannot be negative: {0}", remainder);

        const auto padding = (remainder > 0) ? (alignment - remainder) : 0;

        const auto alignedWeightShape = SmallVector<int64_t>{OC, 1, 1, IC * KY * KX + padding};
        const auto alignedFilterType = mlir::RankedTensorType::get(alignedWeightShape, filterType.getElementType());

        requiredCMX = VPU::getRequiredCMXSizeForNCEOps({inputType, alignedFilterType, outputType}, OC,
                                                       VPU::countElementsPerOutputChannelInWeightTable(
                                                               mlir::cast<VPU::NCEOpInterface>(convOp.getOperation())));
    } else {
        log.debug("[{0}] Unsupported input layout '{1}'", loc, inOrder);
        return mlir::failure();
    }

    if (convOp.getWeightZeroPoints()) {
        const auto weightsElemType = mlir::cast<vpux::NDTypeInterface>(convOp.getFilter().getType()).getElementType();
        requiredCMX += getRequiredCMXSizeForZeroPointTable(convOp, OC, weightsElemType);
    }

    const auto cmxSize = vpux::VPU::getTotalCMXSize(module);
    if (requiredCMX > cmxSize) {
        log.trace("[{0}] CMX memory is not enough for Convolution, available '{1}', required '{2}'", loc, cmxSize,
                  requiredCMX);
        return mlir::failure();
    }

    return mlir::success();
}

mlir::LogicalResult vpux::VPU::NCEConvolutionOp::reifyResultShapes(
        mlir::OpBuilder& builder, mlir::ReifiedRankedShapedTypeDims& reifiedReturnShapes) {
    // Parse attributes
    const auto strides = parseIntArrayAttr<int64_t>(getStrides());

    const auto padTop = getPad().getTop().getValue().getSExtValue();
    const auto padBottom = getPad().getBottom().getValue().getSExtValue();
    const auto padLeft = getPad().getLeft().getValue().getSExtValue();
    const auto padRight = getPad().getRight().getValue().getSExtValue();

    const auto dataPaddingAbove = SmallVector<int64_t>({padTop, padLeft});
    const auto dataPaddingBelow = SmallVector<int64_t>({padBottom, padRight});

    const auto rawFilterShape = Shape(getStaticRawFilterShape());
    SmallVector<int64_t> kernelSize{rawFilterShape[Dims4D::Filter::KY], rawFilterShape[Dims4D::Filter::KX]};

    // Compute output shape using utility
    auto outShape = reifyConvPoolTensors(builder, getInput(), getOutput(), getFilter(), kernelSize, strides,
                                         dataPaddingAbove, dataPaddingBelow, getLoc());
    if (mlir::failed(outShape)) {
        return outShape;
    }

    reifiedReturnShapes.emplace_back(std::move(outShape.value()));
    return mlir::success();
}
