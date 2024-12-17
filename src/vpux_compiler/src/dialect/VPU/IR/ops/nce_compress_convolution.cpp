//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/VPU/utils/conv_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/generate_tiling.hpp"
#include "vpux/compiler/dialect/VPU/utils/max_kernel_size_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/utils/empty_node.hpp"
#include "vpux/compiler/utils/error.hpp"

#include <openvino/op/convolution.hpp>

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

    VPUX_THROW_UNLESS(inOrder == DimsOrder::NHWC, "[{0}] Unsupported input layout '{1}'", getLoc(), inOrder);

    auto totalAvailableCMXSize = reservedMem.count() == 0 ? getTotalCMXSize(getOperation()).count()
                                                          : getTotalCMXFragmentationAwareSize(getOperation()).count();

    return vpux::VPU::calculateAlignedBuffersMemoryRequirement(getArch(getOperation()), buffers).count() +
                   reservedMem.count() <=
           totalAvailableCMXSize;
}

//
// isSupported
//

bool vpux::VPU::NCECompressConvolutionOp::isSupported(IE::ConvolutionOp op, LogCb logCb, bool checkLayout,
                                                      bool checkChannelAlignment) {
    return VPU::isSupportedConv(op, logCb, checkLayout, checkChannelAlignment, /*supportsInputActCompression*/ true);
}

//
// verifyOp
//

static mlir::LogicalResult verifyConv(mlir::Location loc, mlir::Operation* op,
                                      VPU::NCECompressConvolutionOpAdaptor opAdaptor, mlir::Value output) {
    const auto filterShape = Shape(parseIntArrayAttr<int64_t>(opAdaptor.getRawFilterShape()));
    const auto kernelStrides = Shape(parseIntArrayAttr<int64_t>(opAdaptor.getStrides()));
    const auto padAttr = opAdaptor.getPad();
    const auto weightsTableShape = getShape(opAdaptor.getWeightsTable());

    return VPU::verifyConvUtil(loc, op, filterShape, kernelStrides, padAttr, weightsTableShape, output);
}

mlir::LogicalResult vpux::VPU::NCECompressConvolutionOp::verify() {
    auto op = getOperation();
    const auto arch = getArch(op);

    // Skip checks if architecture is unknown since all of them depend on the architecture used
    if (arch == VPU::ArchKind::UNKNOWN) {
        return mlir::success();
    }

    if (mlir::failed(vpux::VPU::verifyNCEOp(op))) {
        return mlir::failure();
    }

    const NCECompressConvolutionOpAdaptor convAdaptor(op->getOperands(), op->getAttrDictionary(),
                                                      op->getPropertiesStorage(), op->getRegions());
    if (mlir::failed(verifyConv(getOperation()->getLoc(), op, convAdaptor, getOutput()))) {
        return mlir::failure();
    }

    const auto inputOrder = DimsOrder::fromValue(getInput());
    const auto filterOrder = DimsOrder::fromValue(getFilter());
    const auto outputOrder = DimsOrder::fromValue(getOutput());

    const auto filterShape = Shape(parseIntArrayAttr<int64_t>(getRawFilterShape()));
    VPUX_THROW_UNLESS(filterShape[Dims4D::Filter::IC] <= vpux::VPU::NCEInvariant::VPU_COMPRESSED_INPUT_CHANNEL_NUM,
                      "Filter input channels : [{0}] must be less than [{1}]", filterShape[Dims4D::Filter::IC],
                      vpux::VPU::NCEInvariant::VPU_COMPRESSED_INPUT_CHANNEL_NUM);

    VPUX_THROW_UNLESS(inputOrder == DimsOrder::NHWC, "[{0}] Unsupported input layout [{1}], expected NHWC", getLoc(),
                      inputOrder);
    if (filterOrder != DimsOrder::OYXI) {
        return errorAt(op, "Unsupported 'filter' layout '{0}', expected OYXI", filterOrder);
    }
    const std::set<VPU::ArchKind> compatibleTargets = {
            VPU::ArchKind::NPU37XX,
            VPU::ArchKind::NPU40XX,
    };
    if (compatibleTargets.count(arch) <= 0 && outputOrder != DimsOrder::NHWC) {
        return errorAt(op, "Unsupported 'output' layout '{0}', expected NHWC", outputOrder);
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
    // Raw filter shape for compress convolution has 3 IC actually used for weights.
    // In order to infer return type we change the IC to aligned values of 4
    // which is same as activation Channel value .
    const auto filterShapeVect = parseIntArrayAttr<int64_t>(op.getRawFilterShape());
    const auto filterShape =
            Shape({filterShapeVect[Dims4D::Filter::OC.ind()], inShape[Dims4D::Act::C],
                   filterShapeVect[Dims4D::Filter::KY.ind()], filterShapeVect[Dims4D::Filter::KX.ind()]});

    const auto windowStrides = parseIntArrayAttr<int64_t>(op.getStrides());
    const auto windowDilations = ov::Strides({1, 1});

    const auto padTop = op.getPad().getTop().getValue().getSExtValue();
    const auto padBottom = op.getPad().getBottom().getValue().getSExtValue();
    const auto padLeft = op.getPad().getLeft().getValue().getSExtValue();
    const auto padRight = op.getPad().getRight().getValue().getSExtValue();

    const auto dataPaddingBelow = ov::CoordinateDiff({padTop, padLeft});
    const auto dataPaddingAbove = ov::CoordinateDiff({padBottom, padRight});

    const auto conv = ov::op::v1::Convolution(
            std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape(inShape.begin(), inShape.end())),
            std::make_shared<ov::op::v0::Parameter>(ov::element::i32,
                                                    ov::Shape(filterShape.begin(), filterShape.end())),
            ov::Strides(windowStrides.begin(), windowStrides.end()), dataPaddingBelow, dataPaddingAbove,
            ov::Strides(windowDilations.begin(), windowDilations.end()));

    const auto& outputShapeNG = conv.get_output_partial_shape(0);

    const auto outputShape = to_small_vector(outputShapeNG.get_shape() | transformed([](size_t val) {
                                                 return checked_cast<int64_t>(val);
                                             }));

    auto inputType = mlir::cast<vpux::NDTypeInterface>(op.getInput().getType());
    auto outputType =
            mlir::RankedTensorType::get(outputShape, inputType.getElementType(), createTensorAttrFromType(inputType));

    inferredReturnTypes.push_back(outputType);
    return mlir::success();
}

//
// TilingBuilderOpInterface
//

vpux::InputTiling vpux::VPU::NCECompressConvolutionOp::backInferTileInfo(const vpux::TileInfo& outputTile,
                                                                         vpux::Logger log) {
    const auto origInputShape = getShape(getInput());
    const auto origFilterShape = Shape(parseIntArrayAttr<int64_t>(getRawFilterShape()));
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

    inputTiling.tiles.push_back(VPU::getWeightsTableTile(this, outputTile));

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
    const auto arch = VPU::getArch(getOperation());
    const auto outputType = getOutput().getType().cast<vpux::NDTypeInterface>();
    const auto batchSize = outputType.getShape()[Dims4D::Act::N];
    if (batchSize > 1 && batchSize <= VPU::getMaxArchDPUClusterNum(arch)) {
        return strategy == VPU::MultiClusterStrategy::SplitOverBatch;
    }

    return strategy == VPU::MultiClusterStrategy::SplitOverHeightOverlapped ||
           strategy == VPU::MultiClusterStrategy::Clustering;
}

vpux::VPU::DistributionInfo vpux::VPU::NCECompressConvolutionOp::getExplicitDistributionInfoAttr(
        vpux::ShapeRef shape, vpux::VPU::DistributionMode distributionMode, ArrayRef<int64_t> numTiles,
        const int64_t numClusters, ArrayRef<int64_t> alignment, const bool uniformDistributedSegments,
        const vpux::VPU::OverlapDistributionParams& overlapParams) {
    return VPU::getNCEExplicitDistributionInfo(mlir::dyn_cast<VPU::NCEOpInterface>(getOperation()), shape,
                                               distributionMode, numTiles, numClusters, alignment,
                                               uniformDistributedSegments, overlapParams);
}

// Each cluster should compute at least one output line. Therefore in order for a layer to be SOH
// compatible it must have an output height of at least the number of clusters
// specified for compilation.
// For example for 4 cluster compilation the output height must be a minimum of 4.
bool VPU::NCECompressConvolutionOp::isOperationSplitOverHeightCompatible(const vpux::TileInfo& oriOutputTile) {
    auto outputShape = ShapeRef(oriOutputTile.shape);
    auto offset = ShapeRef(oriOutputTile.offsets);
    auto axis = ShapeRef(oriOutputTile.axis);
    if (outputShape == ShapeRef()) {
        outputShape = getShape(getOutput());
    }
    vpux::TileInfo outputTile{outputShape, offset, axis, oriOutputTile.isCompletedTile};
    if (!VPU::isOperationSplitOverHeightCompatible(getOperation(), outputTile)) {
        return false;
    }

    auto nceOp = mlir::cast<NCECompressConvolutionOp>(getOperation());
    Shape inputShape = getShape(nceOp.getInput()).toValues();
    auto inputType = nceOp.getInput().getType().cast<NDTypeInterface>();
    // If has custom output shape, infer the input shape
    if (outputShape != getShape(nceOp->getResult(0))) {
        VPUX_THROW_UNLESS(offset != ShapeRef() && axis != ShapeRef(),
                          "Offsets and axis must have value when create TileInfo. Loc: {0}", nceOp->getLoc());
        outputTile.isCompletedTile = true;
        auto computerShape = nceOp.backInferTileInfo(outputTile, Logger::global());
        inputShape = computerShape.tiles.front().shape;
        auto inputOffset = computerShape.tiles.front().offsets;
        inputType = inputType.extractDenseTile(inputOffset, inputShape);
    }

    auto moduleOp = nceOp->getParentOfType<mlir::ModuleOp>();
    auto tileOp = IE::getTileExecutor(moduleOp);
    const auto numTiles = tileOp.getCount();

    return isSOHSupportedByDPU(inputType, inputShape, numTiles, false, VPU::getArch(nceOp.getOperation()));
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
    const auto outputType = nceOp->getResult(0).getType().cast<vpux::NDTypeInterface>();
    auto numClusters = VPU::getOptimalNumClusters(nceOp, outputType.getShape(), strategy);
    auto input = getInput().getType().cast<vpux::NDTypeInterface>();
    auto output = getOutput().getType().cast<vpux::NDTypeInterface>();
    // These depend on a particular tile
    const auto OC = output.getShape()[Dims4D::Act::C];

    const auto inOrder = input.getDimsOrder();

    SmallVector<Byte> buffers = {
            VPU::getTotalAllocSizeWithDistribution(
                    getInput().getType(), getActivationDistributionAttrFromOp(nceOp, getInput().getType(), numClusters,
                                                                              strategy, siblingsAnalysis)),
            VPU::getTotalAllocSizeWithDistribution(
                    getFilter().getType(),
                    getFilterDistributionAttrFromOp(nceOpInterface, getFilter().getType(), numClusters, strategy)),
            VPU::getTotalAllocSizeWithDistribution(
                    getOutput().getType(), getOutputDistributionAttrFromOp(nceOp, getOutput().getType(), numClusters,
                                                                           strategy, siblingsAnalysis)),
            NCEInvariant::getWeightsTableSize(OC)};

    VPUX_THROW_UNLESS(inOrder == DimsOrder::NHWC, "[{0}] Unsupported input layout '{1}'", getLoc(), inOrder);

    auto totalAvailableCMXSize = reservedMem.count() == 0 ? getTotalCMXSize(getOperation()).count()
                                                          : getTotalCMXFragmentationAwareSize(getOperation()).count();

    return vpux::VPU::calculateAlignedBuffersMemoryRequirement(getArch(getOperation()), buffers).count() +
                   reservedMem.count() <=
           totalAvailableCMXSize;
}

bool VPU::NCECompressConvolutionOp::doesLayerChangeOutputAlignmentFitIntoCMX(
        VPU::MultiClusterStrategy strategy, VPU::DistributedTypeInterface newDistributedTensorType) {
    auto nceOp = mlir::cast<NCECompressConvolutionOp>(getOperation());
    auto nceOpInterface = mlir::cast<VPU::NCEOpInterface>(getOperation());
    auto numClusters = VPU::getOptimalNumClusters(
            nceOp, nceOp.getOutput().getType().cast<vpux::NDTypeInterface>().getShape(), strategy);
    auto distributedInputType =
            getDistributedActivationTypeFromOp(nceOp, nceOp.getInput().getType(), numClusters, strategy);
    auto distributedFilterType =
            getDistributedFilterTypeFromOp(nceOpInterface, nceOp.getFilter().getType(), numClusters, strategy);
    return fitIntoCMX(distributedInputType, distributedFilterType, newDistributedTensorType);
}

bool vpux::VPU::NCECompressConvolutionOp::isVFSupported() {
    return vpux::VPU::isVFNCESupported(mlir::cast<NCEOpInterface>(getOperation()));
}

vpux::NDTypeInterface vpux::VPU::NCECompressConvolutionOp::getDistributedTypeForOpOperand(
        mlir::OpOperand& operand, bool hasExplicitDistributedAttr, SiblingOpsAnalysis& siblingsAnalysis) {
    auto clusteredOp = mlir::cast<VPU::ClusteredOpInterface>(getOperation());
    auto origOp = mlir::cast<NCECompressConvolutionOp>(getOperation());
    const auto strategy = clusteredOp.getMultiClusterStrategy().value();
    auto outputTensorType = origOp.getOutput().getType().cast<vpux::NDTypeInterface>();
    auto numClusters = VPU::getOptimalNumClusters(clusteredOp, outputTensorType.getShape(), strategy);
    auto* ctx = clusteredOp->getContext();

    if (operand.get() == origOp.getInput()) {
        mlir::ArrayAttr activationAlignmentAttr = nullptr;
        const auto activationTensorDistributionMode = getActivationTensorDistributionMode(clusteredOp, strategy);
        const auto activationTensorNumTiles =
                getIntArrayAttr(ctx, getActivationTensorNumTiles(clusteredOp, numClusters, strategy));
        return getDistributedTypeFromInput(clusteredOp, origOp.getInput(), activationTensorDistributionMode,
                                           activationTensorNumTiles, activationAlignmentAttr, strategy,
                                           hasExplicitDistributedAttr, siblingsAnalysis);
    } else if (operand.get() == origOp.getFilter()) {
        mlir::ArrayAttr weightAlignmentAttr = nullptr;
        auto filterType = origOp.getFilter().getType().cast<vpux::NDTypeInterface>();
        const auto weightsTensorDistributionMode = getWeightsTensorDistributionMode(strategy);
        const auto weightsTensorNumTiles =
                getIntArrayAttr(ctx, getWeightsTensorNumTiles(clusteredOp, filterType, numClusters, strategy));
        const auto weightAlignment = getWeightsTensorAlignment(strategy);

        if (weightAlignment.has_value()) {
            weightAlignmentAttr = getIntArrayAttr(ctx, weightAlignment.value());
        }
        return getDistributedTypeFromInput(clusteredOp, origOp.getFilter(), weightsTensorDistributionMode,
                                           weightsTensorNumTiles, weightAlignmentAttr, strategy,
                                           hasExplicitDistributedAttr, siblingsAnalysis);
    } else if (operand.get() == origOp.getWeightsTable()) {
        mlir::ArrayAttr weightAlignmentAttr = nullptr;
        auto outputType = origOp.getOutput().getType().cast<vpux::NDTypeInterface>();
        const auto weightsTableTensorDistributionMode = getWeightsTensorDistributionMode(strategy);
        const auto weightsTableTensorNumTiles =
                getIntArrayAttr(ctx, getWeightsTableTensorNumTiles(clusteredOp, outputType, numClusters, strategy));
        const auto weightAlignment = getWeightsTensorAlignment(strategy);

        if (weightAlignment.has_value()) {
            weightAlignmentAttr = getIntArrayAttr(ctx, weightAlignment.value());
        }
        return getDistributedTypeFromInput(clusteredOp, origOp.getWeightsTable(), weightsTableTensorDistributionMode,
                                           weightsTableTensorNumTiles, weightAlignmentAttr, strategy,
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
    const auto arch = getArch(getOperation());
    const auto outputType = getOutput().getType().cast<vpux::NDTypeInterface>();
    auto excludeMode = VPU::NCESparsity::bitwiseNot(VPU::SparsitySupport::NONE);
    if (VPU::NCESparsity::isSuperdenseRequired(arch, outputType.getDimsOrder(), outputType.getShape(),
                                               outputType.getElementType())) {
        excludeMode = VPU::NCESparsity::bitwiseNot(VPU::SparsitySupport::SPARSE_OUTPUTS);
    }

    switch (arch) {
    case VPU::ArchKind::NPU37XX:
    case VPU::ArchKind::NPU40XX:
        return VPU::SparsitySupport::SPARSE_OUTPUTS & excludeMode;
    default:
        VPUX_THROW("Unknown sparsity support mode for {0}", arch);
    }
}