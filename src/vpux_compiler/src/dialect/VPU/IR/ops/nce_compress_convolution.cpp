//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/layers.hpp"
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
#include "vpux/compiler/dialect/VPU/utils/sparsity_support.hpp"
#include "vpux/compiler/dialect/VPU/utils/sprlut_utils.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/dialect/config/utils/config_option_utils.hpp"
#include "vpux/compiler/utils/infer_output_shape.hpp"

#include "vpux/compiler/utils/error.hpp"

#include <mlir/IR/Matchers.h>
#include <openvino/op/convolution.hpp>
#include <openvino/op/parameter.hpp>

using namespace vpux;

//
// fitIntoCMX
//

bool vpux::VPU::NCECompressConvolutionOp::fitIntoCMX(vpux::NDTypeInterface input, vpux::NDTypeInterface filter,
                                                     vpux::NDTypeInterface output) {
    return fitIntoCMX(input, filter, output, Byte(0));
}

bool vpux::VPU::NCECompressConvolutionOp::fitIntoCMX(vpux::NDTypeInterface input, vpux::NDTypeInterface filter,
                                                     vpux::NDTypeInterface output, Byte reservedMem) {
    // These depend on a particular tile
    const auto OC = output.getShape()[Dims4D::Act::C];

    const auto inOrder = input.getDimsOrder();

    SmallVector<Byte> buffers = {input.getTotalAllocSize(), filter.getTotalAllocSize(), output.getTotalAllocSize(),
                                 NCEInvariant::getWeightsTableSize(OC)};

    auto ppeAttr = getPpe();
    addSprLutBufferIfPresent(ppeAttr, buffers);

    VPUX_THROW_UNLESS(inOrder == DimsOrder::NHWC, "[{0}] Unsupported input layout '{1}'", getLoc(), inOrder);

    auto totalAvailableCMXSize = reservedMem.count() == 0 ? getTotalCMXSize(getOperation()).count()
                                                          : getTotalCMXFragmentationAwareSize(getOperation()).count();

    return vpux::VPU::calculateAlignedBuffersMemoryRequirement(config::getArch(getOperation()), buffers).count() +
                   reservedMem.count() <=
           totalAvailableCMXSize;
}

/*
 * Return the mixed raw filter shape by combining the static and dynamic raw filter shape values into a single
 * SmallVector of OpFoldResults.
 */
SmallVector<mlir::OpFoldResult> vpux::VPU::NCECompressConvolutionOp::getMixedRawFilterShape() {
    mlir::Builder builder(getContext());
    return mlir::getMixedValues(getStaticRawFilterShape(), getRawFilterShape(), builder);
}

/*
 * Return the constant raw filter shape by extracting the constant values from the mixed raw filter shape.
 */
SmallVector<int64_t> vpux::VPU::NCECompressConvolutionOp::getConstRawFilterShape() {
    auto vals = mlir::getConstantIntValues(getMixedRawFilterShape());
    VPUX_THROW_WHEN(!vals.has_value(), "Cannot get constant raw filter shape from NCECompressConvolutionOp '{0}'",
                    getLoc());
    return vals.value();
}

//
// isDeprecated
//

bool vpux::VPU::NCECompressConvolutionOp::isDeprecated([[maybe_unused]] mlir::Operation* op) {
    return false;
}

//
// isSupported
//

bool vpux::VPU::NCECompressConvolutionOp::isSupported(IE::ConvolutionOp op, LogCb logCb, bool checkLayout,
                                                      bool checkChannelAlignment) {
    if (isDeprecated(op)) {
        return false;
    }

    auto inputType = mlir::cast<NDTypeInterface>(op.getInput().getType());
    auto canUseIDUAutopad =
            VPU::inputCompatibleWithAutoPad(inputType) && config::hasAutoPaddingIDU(getModuleOp(op.getOperation()));
    // IDU autopad can support the IC=4 configuration represented by NCECompressConvolution, along with multiple others.
    // Therefore, we should avoid using NCECompressConvolution when IDU autopad can be used
    if (canUseIDUAutopad) {
        return false;
    }
    return VPU::isSupportedConv(op, logCb, checkLayout, checkChannelAlignment, /*supportsInputActCompression*/ true);
}

//
// verifyOp
//

static mlir::LogicalResult verifyConv(mlir::Location loc, mlir::Operation* op,
                                      VPU::NCECompressConvolutionOpAdaptor& opAdaptor, mlir::Value output) {
    const auto filterShape = Shape(opAdaptor.getStaticRawFilterShape());
    const auto KY = filterShape[Dims4D::Filter::KY];
    const auto KX = filterShape[Dims4D::Filter::KX];

    VPUX_THROW_WHEN(mlir::ShapedType::isDynamic(KY) || mlir::ShapedType::isDynamic(KX),
                    "Dynamic kernel size is not supported for NCE operations");

    const auto kernelStrides = Shape(parseIntArrayAttr<int64_t>(opAdaptor.getStrides()));
    const auto padAttr = opAdaptor.getPad();
    const auto weightsTableShape = getShape(opAdaptor.getWeightsTable());

    return VPU::verifyConvUtil(loc, op, filterShape, kernelStrides, padAttr, weightsTableShape, output);
}

mlir::LogicalResult vpux::VPU::NCECompressConvolutionOp::verify() {
    auto op = getOperation();
    const auto arch = config::getArch(op);

    // Skip checks if architecture is unknown since all of them depend on the architecture used
    if (arch == config::ArchKind::UNKNOWN) {
        return mlir::success();
    }

    if (mlir::failed(vpux::VPU::verifyNCEOp(op))) {
        return mlir::failure();
    }

    NCECompressConvolutionOpAdaptor convAdaptor(op->getOperands(), op->getAttrDictionary(), op->getPropertiesStorage(),
                                                op->getRegions());
    if (mlir::failed(verifyConv(getOperation()->getLoc(), op, convAdaptor, getOutput()))) {
        return mlir::failure();
    }

    const auto inputOrder = DimsOrder::fromValue(getInput());
    const auto filterOrder = DimsOrder::fromValue(getFilter());

    const auto mixedRawFilterShape = getMixedRawFilterShape();
    const auto maybeIC = mlir::getConstantIntValue(mixedRawFilterShape[checked_cast<size_t>(Dims4D::Filter::IC.ind())]);
    if (!maybeIC.has_value()) {
        return errorAt(op, "Dynamic raw kernel shape IC is not supported for NCECompressConvolutionOp");
    }
    VPUX_THROW_UNLESS(maybeIC.value() <= vpux::VPU::NCEInvariant::VPU_COMPRESSED_INPUT_CHANNEL_NUM,
                      "Filter input channels : [{0}] must be less than [{1}]", maybeIC.value(),
                      vpux::VPU::NCEInvariant::VPU_COMPRESSED_INPUT_CHANNEL_NUM);

    VPUX_THROW_UNLESS(inputOrder == DimsOrder::NHWC, "[{0}] Unsupported input layout [{1}], expected NHWC", getLoc(),
                      inputOrder);
    if (filterOrder != DimsOrder::OYXI) {
        return errorAt(op, "Unsupported 'filter' layout '{0}', expected OYXI", filterOrder);
    }

    return mlir::success();
}

//
// InferTypeOpInterface
//

mlir::LogicalResult vpux::VPU::NCECompressConvolutionOp::inferReturnTypes(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange /*regions*/,
        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    NCECompressConvolutionOpAdaptor op(operands, attrs, prop);
    if (mlir::failed(op.verify(loc))) {
        return mlir::failure();
    }

    const auto inShape = getShape(op.getInput());
    // RawFilterShape can have either static or dynamic dimensions, so we need to resolve it before inferring the output
    // shape. Raw filter shape for compress convolution has 3 IC actually used for weights. In order to infer return
    // type we change the IC to aligned values of 4 which is same as activation Channel value.
    const auto resolvedFilterShape = VPU::resolveRawFilterShape(op.getStaticRawFilterShape(), op.getRawFilterShape());
    const auto filterShape =
            Shape({resolvedFilterShape[Dims4D::Filter::OC.ind()], inShape[Dims4D::Act::C],
                   resolvedFilterShape[Dims4D::Filter::KY.ind()], resolvedFilterShape[Dims4D::Filter::KX.ind()]});

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
    auto filterShapeInfo = ShapeInfo{/*shape*/ filterShape.raw(), /*bounds*/ {}};
    auto shapeInfo = inferConvolutionOutputShapeInfo(inShapeInfo, filterShapeInfo, filterType, windowStrides,
                                                     dataPaddingBelow, dataPaddingAbove, windowDilations);

    const auto outType =
            vpux::getTensorType(ShapeRef(shapeInfo.shape), inputType.getElementType(), inputType.getDimsOrder(),
                                /*memSpace=*/nullptr, BoundsRef(shapeInfo.bounds), /*DynamicDimsMask=*/{});
    inferredReturnTypes.push_back(outType);
    return mlir::success();
}

//
// TilingBuilderOpInterface
//

vpux::InputTiling vpux::VPU::NCECompressConvolutionOp::backInferTileInfo(const vpux::TileInfo& outputTile,
                                                                         vpux::Logger log) {
    const auto origInputShape = getBoundedShape(getInput());
    const auto origFilterShape = Shape(getConstRawFilterShape());
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

    // Adjust filter tile for the aligned filter
    inputTiling.tiles[1].shape = getShape(getFilter()).toValues();
    inputTiling.tiles[1].shape[Dims4D::Filter::OC] = outputTile.shape[Dims4D::Act::C];

    inputTiling.tiles.push_back(
            VPU::getWeightsTableTile(this, outputTile, VPU::getWeightsChannelsAutopad(getOperation())));

    return inputTiling;
}

void vpux::VPU::NCECompressConvolutionOp::adjustAttrs(const TilingInfo& inputTiling, const TileInfo& outputTile) {
    VPU::adjustPaddings(this, inputTiling);
    VPU::adjustRawFilterShape(this, outputTile);
}

mlir::FailureOr<OutputTiling> vpux::VPU::NCECompressConvolutionOp::getTilingStrategy(TilingMode tilingMode,
                                                                                     Logger log) {
    return vpux::getHWLayerTilingStrategy(this->getOperation(), tilingMode, log);
}

//
// ClusteredOpInterface
//

bool vpux::VPU::NCECompressConvolutionOp::checkStrategyCompatibility(VPU::MultiClusterStrategy strategy, size_t) {
    const auto outputType = mlir::cast<vpux::NDTypeInterface>(getOutput().getType());
    const auto batchSize = outputType.getShape()[Dims4D::Act::N];
    const auto enabledTileNum = config::getNumOfTiles(getOperation());

    if (batchSize > 1 && batchSize <= enabledTileNum) {
        return strategy == VPU::MultiClusterStrategy::SplitOverBatch;
    }

    return strategy == VPU::MultiClusterStrategy::SplitOverHeightOverlapped ||
           strategy == VPU::MultiClusterStrategy::Clustering;
}

vpux::VPU::DistributionInfo vpux::VPU::NCECompressConvolutionOp::getExplicitDistributionInfoAttr(
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
bool VPU::NCECompressConvolutionOp::isOperationSplitOverHeightCompatible(const vpux::TileInfo& oriOutputTile) {
    return VPU::isNCEOpSplitOverHeightCompatible(getOperation(), getInput(), getShape(getOutput()), oriOutputTile,
                                                 false);
}

bool VPU::NCECompressConvolutionOp::isOperationSplitOverWidthCompatible(ShapeRef outputShape, ShapeRef offset,
                                                                        ShapeRef axis) {
    return VPU::isOperationSplitOverWidthCompatible(getOperation(), outputShape, offset, axis);
}

bool VPU::NCECompressConvolutionOp::isOperationSplitOverKernelCompatible(ShapeRef outputShape, ShapeRef offset,
                                                                         ShapeRef axis) {
    return VPU::isOperationSplitOverKernelCompatible(getOperation(), outputShape, offset, axis);
}

bool VPU::NCECompressConvolutionOp::isOperationSplitOverBatchCompatible(vpux::ShapeRef outputShape) {
    return VPU::isOperationSplitOverBatchCompatible(getOperation(), outputShape);
}

bool VPU::NCECompressConvolutionOp::doesLayerFitIntoCMX(VPU::MultiClusterStrategy strategy,
                                                        SiblingOpsAnalysis& siblingsAnalysis, Byte reservedMem) {
    auto nceOp = mlir::cast<VPU::NCECompressConvolutionOp>(getOperation());
    auto nceOpInterface = mlir::cast<VPU::NCEOpInterface>(getOperation());
    const auto outputType = mlir::cast<vpux::NDTypeInterface>(nceOp->getResult(0).getType());
    auto numClusters = VPU::getOptimalNumClusters(nceOp, outputType.getShape(), strategy);
    auto input = mlir::cast<vpux::NDTypeInterface>(getInput().getType());
    auto output = mlir::cast<vpux::NDTypeInterface>(getOutput().getType());
    // These depend on a particular tile
    const auto OC = output.getShape()[Dims4D::Act::C];

    const auto inOrder = input.getDimsOrder();

    SmallVector<Byte> buffers = {
            VPU::getTotalAllocSizeWithDistribution(
                    getInput().getType(), getActivationDistributionAttrFromOp(nceOp, getInput(), getInput().getType(),
                                                                              numClusters, strategy, siblingsAnalysis)),
            VPU::getTotalAllocSizeWithDistribution(
                    getFilter().getType(),
                    getFilterDistributionAttrFromOp(nceOpInterface, getFilter().getType(), numClusters, strategy)),
            VPU::getTotalAllocSizeWithDistribution(
                    getOutput().getType(), getOutputDistributionAttrFromOp(nceOp, getOutput().getType(), numClusters,
                                                                           strategy, siblingsAnalysis)),
            NCEInvariant::getWeightsTableSize(OC)};

    auto ppeAttr = getPpe();
    addSprLutBufferIfPresent(ppeAttr, buffers);

    VPUX_THROW_UNLESS(inOrder == DimsOrder::NHWC, "[{0}] Unsupported input layout '{1}'", getLoc(), inOrder);

    auto totalAvailableCMXSize = reservedMem.count() == 0 ? getTotalCMXSize(getOperation()).count()
                                                          : getTotalCMXFragmentationAwareSize(getOperation()).count();

    return vpux::VPU::calculateAlignedBuffersMemoryRequirement(config::getArch(getOperation()), buffers).count() +
                   reservedMem.count() <=
           totalAvailableCMXSize;
}

bool VPU::NCECompressConvolutionOp::doesLayerChangeOutputAlignmentFitIntoCMX(
        VPU::MultiClusterStrategy strategy, VPU::DistributedTypeInterface newDistributedTensorType) {
    auto nceOp = mlir::cast<NCECompressConvolutionOp>(getOperation());
    auto nceOpInterface = mlir::cast<VPU::NCEOpInterface>(getOperation());
    auto numClusters = VPU::getOptimalNumClusters(
            nceOp, mlir::cast<vpux::NDTypeInterface>(nceOp.getOutput().getType()).getShape(), strategy);
    auto distributedInputType = getDistributedActivationTypeFromOp(nceOp, nceOp.getInput(), nceOp.getInput().getType(),
                                                                   numClusters, strategy);
    auto distributedFilterType =
            getDistributedFilterTypeFromOp(nceOpInterface, nceOp.getFilter().getType(), numClusters, strategy);
    return fitIntoCMX(distributedInputType, distributedFilterType, newDistributedTensorType);
}

vpux::NDTypeInterface vpux::VPU::NCECompressConvolutionOp::getDistributedTypeForOpOperand(
        mlir::OpOperand& operand, bool hasExplicitDistributedAttr, SiblingOpsAnalysis& siblingsAnalysis) {
    auto clusteredOp = mlir::cast<VPU::ClusteredOpInterface>(getOperation());
    auto origOp = mlir::cast<NCECompressConvolutionOp>(getOperation());
    const auto strategy = clusteredOp.getMultiClusterStrategy().value();

    if (operand.get() == origOp.getInput()) {
        return VPU::getDistributedActivationTypeForOpOperand(clusteredOp, origOp.getInput(), strategy,
                                                             hasExplicitDistributedAttr, siblingsAnalysis);
    } else if (operand.get() == origOp.getFilter()) {
        return VPU::getDistributedWeightsTypeForOpOperand(clusteredOp, origOp.getFilter(), strategy,
                                                          hasExplicitDistributedAttr, siblingsAnalysis);
    } else if (operand.get() == origOp.getWeightsTable()) {
        return VPU::getDistributedWeightsTypeForOpOperand(clusteredOp, operand.get(), strategy,
                                                          hasExplicitDistributedAttr, siblingsAnalysis);
    }
    VPUX_THROW("Failed to compute distributed type for op {0}", clusteredOp);
    return nullptr;
}

//
// sparsitySupport
//

vpux::VPU::SparsitySupport vpux::VPU::NCECompressConvolutionOp::sparsitySupport() {
    // Super-dense mode does not support ODU sparsity
    const auto outputType = mlir::cast<vpux::NDTypeInterface>(getOutput().getType());
    auto excludeMode = VPU::NCESparsity::bitwiseNot(VPU::SparsitySupport::NONE);
    if (VPU::NCESparsity::isSuperdenseRequired(outputType.getDimsOrder(), outputType.getShape(),
                                               outputType.getElementType())) {
        excludeMode = VPU::NCESparsity::bitwiseNot(VPU::SparsitySupport::SPARSE_OUTPUTS);
    }
    return VPU::SparsitySupport::SPARSE_OUTPUTS & excludeMode;
}

mlir::LogicalResult vpux::VPU::NCECompressConvolutionOp::reifyResultShapes(
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
