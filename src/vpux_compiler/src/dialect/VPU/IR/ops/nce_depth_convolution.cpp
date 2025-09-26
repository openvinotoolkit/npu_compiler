//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/IR/types.hpp"
#include "vpux/compiler/dialect/VPU/utils/auto_padding_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/conv_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/generate_tiling.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_sparsity.hpp"
#include "vpux/compiler/dialect/VPU/utils/sparsity_support.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/utils/VPU/tile_utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/dilated_utils.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/infer_output_shape.hpp"

#include <openvino/op/convolution.hpp>

using namespace vpux;

//
// fitIntoCMX
//

bool vpux::VPU::NCEDepthConvolutionOp::fitIntoCMX(vpux::NDTypeInterface input, vpux::NDTypeInterface filter,
                                                  vpux::NDTypeInterface output, Byte reservedMem) {
    SmallVector<Byte> buffers = {input.getTotalAllocSize(), filter.getTotalAllocSize(), output.getTotalAllocSize()};
    const auto OC = output.getShape()[Dims4D::Act::C];

    const auto op = getOperation();
    if (mlir::failed(NCEInvariant::getWeightTableBuffers(op, buffers, OC))) {
        VPUX_THROW("getWeightTableBuffers function failed");
    }

    auto totalAvailableCMXSize =
            reservedMem.count() == 0 ? getTotalCMXSize(op).count() : getTotalCMXFragmentationAwareSize(op).count();
    auto arch = config::getArch(op);
    return vpux::VPU::calculateAlignedBuffersMemoryRequirement(arch, buffers).count() + reservedMem.count() <=
           totalAvailableCMXSize;
}

bool vpux::VPU::NCEDepthConvolutionOp::fitIntoCMX(vpux::NDTypeInterface input, vpux::NDTypeInterface filter,
                                                  vpux::NDTypeInterface output) {
    return fitIntoCMX(input, filter, output, Byte(0));
}

//
// isSupported
//

bool vpux::VPU::NCEDepthConvolutionOp::isSupported(IE::GroupConvolutionOp op, LogCb logCb, bool checkLayout,
                                                   bool checkChannelAlignment) {
    if (op.getType().getRank() != 4) {
        logCb(formatv("Only 4D tensors are supported"));
        return false;
    }
    if (getShape(op.getFilter()).size() != 4) {
        logCb(formatv("Only 4D tensors are supported"));
        return false;
    }

    const auto filterShape = getShape(op.getFilter());
    const auto fIC = filterShape[Dims4D::Filter::IC];
    const auto OC = filterShape[Dims4D::Filter::OC];
    const auto KY = filterShape[Dims4D::Filter::KY];
    const auto KX = filterShape[Dims4D::Filter::KX];

    if (!op.getGroups().has_value()) {
        logCb(formatv("Grouped convolution does not have groups attribute"));
        return false;
    }
    if (op.getGroups().value() != OC) {
        if (op.getOutputPaddingAttr() != nullptr) {
            const auto outputPadding = parseIntArrayAttr<int64_t>(op.getOutputPaddingAttr());
            if (op.getGroups().value() != (OC - outputPadding[Dims4D::Act::C.ind()])) {
                logCb(formatv("Unsupported group size: '{0}' expected '{1}' (output channels '{2}', padding '{3}')",
                              op.getGroups(), OC - outputPadding[Dims4D::Act::C.ind()], OC,
                              outputPadding[Dims4D::Act::C.ind()]));
                return false;
            }
        } else {
            logCb(formatv("Unsupported group size: '{0}' expected '{1}'", op.getGroups(), OC));
            return false;
        }
    }
    if (fIC != 1) {
        logCb(formatv("Group Convolution with more than one filter per input channel is not supported"));
        return false;
    }

    const auto inputType = mlir::cast<vpux::NDTypeInterface>(op.getInput().getType());
    const auto outputType = mlir::cast<vpux::NDTypeInterface>(op.getOutput().getType());

    const auto kernelStrides = Shape(parseIntArrayAttr<int64_t>(op.getStrides()));
    const auto SY = kernelStrides[Dims4D::Strides::Y];
    const auto SX = kernelStrides[Dims4D::Strides::X];

    auto pads = PadInfo(op.getPadsBegin(), op.getPadsEnd());
    const auto dilations = parseIntArrayAttr<int64_t>(op.getDilations());
    pads = VPU::shrinkPadsForDilatedConvolution(pads, dilations);

    if (!NCEInvariant::isAttrsSupported(op, KY, KX, SY, SX, pads.top, pads.bottom, pads.left, pads.right, logCb)) {
        return false;
    }

    if (checkChannelAlignment) {
        auto iface = mlir::cast<IE::AlignedChannelsOpInterface>(op.getOperation());
        if (!NCEInvariant::isInputActTypeSupported(inputType, iface.getInputChannelAlignment(), false) ||
            !NCEInvariant::isOutputActTypeSupported(outputType, iface.getOutputChannelAlignment())) {
            logCb(formatv("Misaligned tensor shape"));
            return false;
        }
    }

    if (checkLayout) {
        const auto arch = config::getArch(op);
        if (!NCEInvariant::checkLayouts(op->getOperandTypes(), op->getResultTypes(), arch, 2, logCb)) {
            return false;
        }
    }

    return true;
}

//
// verify
//

mlir::LogicalResult verifyDepthConv(mlir::Location loc, mlir::Operation* op,
                                    VPU::NCEDepthConvolutionOpAdaptor& opAdaptor, mlir::Value output) {
    const auto logCb = [loc](const llvm::formatv_object_base& msg) {
        std::ignore = errorAt(loc, "{0}", msg.str());
    };

    const auto outputShape = getShape(output);

    const auto filterShape = Shape(parseIntArrayAttr<int64_t>(opAdaptor.getRawFilterShape()));
    const auto KY = filterShape[Dims4D::Filter::KY];
    const auto KX = filterShape[Dims4D::Filter::KX];

    const auto kernelStrides = Shape(parseIntArrayAttr<int64_t>(opAdaptor.getStrides()));
    const auto SY = kernelStrides[Dims4D::Strides::Y];
    const auto SX = kernelStrides[Dims4D::Strides::X];

    const auto padTop = opAdaptor.getPad().getTop().getValue().getSExtValue();
    const auto padBottom = opAdaptor.getPad().getBottom().getValue().getSExtValue();
    const auto padLeft = opAdaptor.getPad().getLeft().getValue().getSExtValue();
    const auto padRight = opAdaptor.getPad().getRight().getValue().getSExtValue();

    if (!VPU::NCEInvariant::isAttrsSupported(op, KY, KX, SY, SX, padTop, padBottom, padLeft, padRight, logCb)) {
        return mlir::failure();
    }

    const auto OC =
            VPU::canAutopadOutput(op) ? vpux::VPU::NCEInvariant::VPU_CHANNEL_ALIGNMENT : outputShape[Dims4D::Act::C];

    const auto weightsTableShape = opAdaptor.getWeightsTable() == nullptr
                                           ? std::nullopt
                                           : std::optional<vpux::ShapeRef>(getShape(opAdaptor.getWeightsTable()));
    const auto expectedWeightsTableShape = VPU::NCESparsity::inferWeightsTableShape(OC);

    if (weightsTableShape.has_value() && weightsTableShape.value() != expectedWeightsTableShape) {
        return errorAt(loc, "Got wrong shape for 'weightsTable' '{0}', expected '{1}'", weightsTableShape.value(),
                       expectedWeightsTableShape);
    }

    return mlir::success();
}

mlir::LogicalResult vpux::VPU::NCEDepthConvolutionOp::verify() {
    const auto op = getOperation();
    const auto arch = config::getArch(op);

    // Skip checks if architecture is unknown since all of them depend on the architecture used
    if (arch == config::ArchKind::UNKNOWN) {
        return mlir::success();
    }

    if (mlir::failed(VPU::NCEInvariant::verifyWeightTables(op))) {
        return mlir::failure();
    }

    if (mlir::failed(vpux::VPU::verifyNCEOp(op))) {
        return mlir::failure();
    }

    VPU::NCEDepthConvolutionOpAdaptor convAdaptor(op->getOperands(), op->getAttrDictionary(),
                                                  op->getPropertiesStorage(), op->getRegions());
    if (mlir::failed(verifyDepthConv(op->getLoc(), op, convAdaptor, getOutput()))) {
        return mlir::failure();
    }

    const auto outputType = mlir::cast<vpux::NDTypeInterface>(getOutput().getType());
    const auto filterType = mlir::cast<vpux::NDTypeInterface>(getFilter().getType());

    const auto alignedFilterShape = filterType.getShape();
    const auto expectedAlignedFilterShape = inferAlignedFilterShape(outputType, filterType);

    if (alignedFilterShape != expectedAlignedFilterShape) {
        return errorAt(op, "Got wrong shape for 'filter' '{0}', expected '{1}'", alignedFilterShape,
                       expectedAlignedFilterShape);
    }

    return mlir::success();
}

Shape vpux::VPU::NCEDepthConvolutionOp::inferAlignedFilterShape(NDTypeInterface output, NDTypeInterface filter) {
    const auto rawFilterShape = Shape(parseIntArrayAttr<int64_t>(this->getRawFilterShape()));
    const auto KY = rawFilterShape[Dims4D::Filter::KY];
    const auto KX = rawFilterShape[Dims4D::Filter::KX];

    const auto OC = output.getShape()[Dims4D::Act::C];

    const auto alignment = NCEInvariant::getAlignment(filter.getElementType());

    const auto remainder = (KY * KX) % alignment;

    if (remainder == 0) {
        return Shape{OC, 1, KY, KX};
    }

    const auto padding = (remainder > 0) ? (alignment - remainder) : 0;

    return Shape{OC, KY * KX + padding, 1, 1};
}

//
// InferTypeOpInterface
//

mlir::LogicalResult vpux::VPU::NCEDepthConvolutionOp::inferReturnTypes(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange /*regions*/,
        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    NCEDepthConvolutionOpAdaptor op(operands, attrs, prop);
    if (mlir::failed(op.verify(loc))) {
        return mlir::failure();
    }

    const auto filterShape = Shape(parseIntArrayAttr<int64_t>(op.getRawFilterShape()));
    const auto fIC = filterShape[Dims4D::Filter::IC];

    if (fIC != 1) {
        return errorAt(loc, "Non depthwise convolution case");
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

    auto inShapeInfo = ShapeInfo::fromNDType(inputType);
    auto filterShapeInfo = ShapeInfo::fromNDType(filterType);

    // Adjust input shape to reuse helpers for standard convolution
    inShapeInfo.shape[Dims4D::Act::C.ind()] = 1;
    filterShapeInfo.shape = parseIntArrayAttr<int64_t>(op.getRawFilterShape());

    auto shapeInfo = inferConvolutionOutputShapeInfo(inShapeInfo, filterShapeInfo, windowStrides, dataPaddingBelow,
                                                     dataPaddingAbove, windowDilations);

    const auto outDesc =
            vpux::getTensorAttr(ctx, inputType.getDimsOrder(), /*memSpace=*/nullptr, BoundsRef(shapeInfo.bounds));
    const auto outputType = mlir::RankedTensorType::get(shapeInfo.shape, inputType.getElementType(), outDesc);

    inferredReturnTypes.push_back(outputType);
    return mlir::success();
}

//
// TilingBuilderOpInterface
//

vpux::InputTiling vpux::VPU::NCEDepthConvolutionOp::backInferTileInfo(const vpux::TileInfo& outputTile,
                                                                      vpux::Logger log) {
    const auto origInputShape = getBoundedShape(getInput());
    const auto origPadding = toPadInfo(getPad());
    const auto origFilterShape = Shape(parseIntArrayAttr<int64_t>(getRawFilterShape()));

    // This op incorporates bias values in WeightsTable
    const auto origBiasShape = ShapeRef();

    // The NCE depthwise convolution only supports one channel per group, therefore the group can be determined
    // based on the number of channels produced by the operation
    // Note: the IDU and ODU padding requirements are independent for the DPU (e.g. input could be padded, while the
    // output not), therefore the number of groups here is determined by the output channels here
    const auto groups = origFilterShape[Dims4D::Filter::OC];

    auto inputTiling = backInferGroupConvTile(outputTile, origInputShape, origFilterShape, origBiasShape, getStrides(),
                                              origPadding, groups);
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

    auto nceOp = mlir::cast<VPU::NCEDepthConvolutionOp>(getOperation());
    if (nceOp.getWeightsTable()) {
        inputTiling.tiles.push_back(
                VPU::getWeightsTableTile(this, outputTile, VPU::getWeightsChannelsAutopad(getOperation())));
    }
    if (nceOp.getWeightTableScale()) {
        inputTiling.tiles.push_back(VPU::getScaleTableTile(this, outputTile));
    }
    if (nceOp.getWeightTableBias()) {
        inputTiling.tiles.push_back(VPU::getBiasTableTile(this, outputTile));
    }

    return inputTiling;
}

void vpux::VPU::NCEDepthConvolutionOp::adjustAttrs(const TilingInfo& inputTiling, const TileInfo& outputTile) {
    VPU::adjustPaddings(this, inputTiling);
    VPU::adjustRawFilterShape(this, outputTile);
}

mlir::FailureOr<OutputTiling> vpux::VPU::NCEDepthConvolutionOp::getTilingStrategy(TilingMode tilingMode, Logger log) {
    return vpux::getHWLayerTilingStrategy(this->getOperation(), tilingMode, log);
}

//
// ClusteredOpInterface
//

bool vpux::VPU::NCEDepthConvolutionOp::checkStrategyCompatibility(VPU::MultiClusterStrategy strategy, size_t) {
    return strategy == VPU::MultiClusterStrategy::Clustering ||
           strategy == VPU::MultiClusterStrategy::SplitOverHeight ||
           strategy == VPU::MultiClusterStrategy::SplitOverKernel || strategy == VPU::MultiClusterStrategy::HKSwitch;
}

vpux::VPU::DistributionInfo vpux::VPU::NCEDepthConvolutionOp::getExplicitDistributionInfoAttr(
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
bool VPU::NCEDepthConvolutionOp::isOperationSplitOverHeightCompatible(const vpux::TileInfo& oriOutputTile) {
    if (isSEPDWConv(getOperation())) {
        // [E#154046] SOH Dilated SEP DWConv is inaccurate for now
        const auto sparseInputTensor = mlir::cast<VPU::SparseTensorType>(getOperand(0).getType());
        auto seAttr = sparseInputTensor.getSeAttr();
        if (mlir::isa<VPU::SEDilatedConvAttr>(seAttr)) {
            return false;
        }
    }

    auto outputShape = ShapeRef(oriOutputTile.shape);
    auto offset = ShapeRef(oriOutputTile.offsets);
    auto axis = ShapeRef(oriOutputTile.axis);
    if (outputShape.empty()) {
        outputShape = getBoundedShape(getOutput());
    }
    vpux::TileInfo outputTile{outputShape, offset, axis, oriOutputTile.isCompletedTile};
    if (!VPU::isOperationSplitOverHeightCompatible(getOperation(), outputTile)) {
        return false;
    }

    auto nceOp = mlir::cast<VPU::NCEDepthConvolutionOp>(getOperation());
    Shape inputShape = getShape(nceOp.getInput()).toValues();
    auto inputType = mlir::cast<vpux::NDTypeInterface>(nceOp.getInput().getType());
    // If has custom output shape, infer the input shape
    if (outputShape != getBoundedShape(nceOp->getResult(0))) {
        VPUX_THROW_WHEN(offset.empty() || axis.empty(),
                        "Offsets and axis must have value when create TileInfo. Loc: {0}", nceOp->getLoc());
        outputTile.isCompletedTile = true;
        auto computerShape = nceOp.backInferTileInfo(outputTile, Logger::global());
        inputShape = computerShape.tiles.front().shape;
        auto inputOffset = computerShape.tiles.front().offsets;
        inputType = inputType.extractDenseTile(inputOffset, inputShape);
    }

    auto moduleOp = nceOp->getParentOfType<mlir::ModuleOp>();
    auto tileOp = config::getTileExecutor(moduleOp);
    const auto numTiles = tileOp.getCount();

    return isSOHSupportedByDPU(inputType, inputShape, numTiles, true, config::getArch(nceOp.getOperation()));
}

bool VPU::NCEDepthConvolutionOp::isOperationSplitOverWidthCompatible(ShapeRef outputShape, ShapeRef offset,
                                                                     ShapeRef axis) {
    return VPU::isOperationSplitOverWidthCompatible(getOperation(), outputShape, offset, axis);
}

bool VPU::NCEDepthConvolutionOp::isOperationSplitOverKernelCompatible(ShapeRef outputShape, ShapeRef offset,
                                                                      ShapeRef axis) {
    return VPU::isOperationSplitOverKernelCompatible(getOperation(), outputShape, offset, axis);
}

bool VPU::NCEDepthConvolutionOp::doesLayerFitIntoCMX(VPU::MultiClusterStrategy strategy,
                                                     SiblingOpsAnalysis& siblingsAnalysis, Byte reservedMem) {
    auto nceOp = mlir::cast<VPU::NCEDepthConvolutionOp>(getOperation());
    auto nceOpInterface = mlir::cast<VPU::NCEOpInterface>(getOperation());
    const auto outputType = mlir::cast<vpux::NDTypeInterface>(nceOp->getResult(0).getType());
    auto numClusters = VPU::getOptimalNumClusters(nceOp, outputType.getShape(), strategy);

    auto output = mlir::cast<vpux::NDTypeInterface>(getOutput().getType());

    const auto OC = output.getShape()[Dims4D::Act::C];

    SmallVector<Byte> buffers = {
            VPU::getTotalAllocSizeWithDistribution(
                    getInput().getType(), getActivationDistributionAttrFromOp(nceOp, getInput(), getInput().getType(),
                                                                              numClusters, strategy, siblingsAnalysis)),
            VPU::getTotalAllocSizeWithDistribution(
                    getFilter().getType(),
                    getFilterDistributionAttrFromOp(nceOpInterface, getFilter().getType(), numClusters, strategy)),
            VPU::getTotalAllocSizeWithDistribution(
                    getOutput().getType(), getOutputDistributionAttrFromOp(nceOp, getOutput().getType(), numClusters,
                                                                           strategy, siblingsAnalysis))};
    const auto op = getOperation();
    if (mlir::failed(NCEInvariant::getWeightTableBuffers(op, buffers, OC))) {
        VPUX_THROW("getWeightTableBuffers function failed");
    }

    auto totalAvailableCMXSize = reservedMem.count() == 0 ? VPU::getTotalCMXSize(op).count()
                                                          : VPU::getTotalCMXFragmentationAwareSize(op).count();

    auto arch = config::getArch(op);
    return vpux::VPU::calculateAlignedBuffersMemoryRequirement(arch, buffers).count() + reservedMem.count() <=
           totalAvailableCMXSize;
}

bool VPU::NCEDepthConvolutionOp::doesLayerChangeOutputAlignmentFitIntoCMX(
        VPU::MultiClusterStrategy strategy, VPU::DistributedTypeInterface newDistributedTensorType) {
    auto nceOp = mlir::cast<VPU::NCEDepthConvolutionOp>(getOperation());
    auto nceOpInterface = mlir::cast<VPU::NCEOpInterface>(getOperation());
    auto numClusters = VPU::getOptimalNumClusters(
            nceOp, mlir::cast<vpux::NDTypeInterface>(nceOp.getOutput().getType()).getShape(), strategy);
    auto distributedInputType = getDistributedActivationTypeFromOp(nceOp, nceOp.getInput(), nceOp.getInput().getType(),
                                                                   numClusters, strategy);
    auto distributedFilterType =
            getDistributedFilterTypeFromOp(nceOpInterface, nceOp.getFilter().getType(), numClusters, strategy);
    return fitIntoCMX(distributedInputType, distributedFilterType, newDistributedTensorType);
}

vpux::NDTypeInterface vpux::VPU::NCEDepthConvolutionOp::getDistributedTypeForOpOperand(
        mlir::OpOperand& operand, bool hasExplicitDistributedAttr, SiblingOpsAnalysis& siblingsAnalysis) {
    auto clusteredOp = mlir::cast<VPU::ClusteredOpInterface>(getOperation());
    auto origOp = mlir::cast<VPU::NCEDepthConvolutionOp>(getOperation());
    const auto strategy = clusteredOp.getMultiClusterStrategy().value();
    auto outputTensorType = mlir::cast<vpux::NDTypeInterface>(origOp.getOutput().getType());
    auto numClusters = VPU::getOptimalNumClusters(clusteredOp, outputTensorType.getShape(), strategy);
    auto* ctx = clusteredOp->getContext();

    if (operand.get() == origOp.getInput()) {
        mlir::ArrayAttr activationAlignmentAttr = nullptr;
        const auto activationTensorDistributionMode = getActivationTensorDistributionMode(clusteredOp, strategy);
        const auto activationTensorNumTiles =
                getIntArrayAttr(ctx, getActivationTensorNumTiles(clusteredOp, numClusters, strategy));
        const auto activationAlignment = getActivationTensorAlignment(clusteredOp, numClusters, strategy);
        if (activationAlignment.has_value()) {
            activationAlignmentAttr = getIntArrayAttr(ctx, activationAlignment.value());
        }
        return getDistributedTypeFromInput(clusteredOp, origOp.getInput(), activationTensorDistributionMode,
                                           activationTensorNumTiles, activationAlignmentAttr, strategy,
                                           hasExplicitDistributedAttr, siblingsAnalysis);
    } else if (operand.get() == origOp.getFilter()) {
        auto filterType = mlir::cast<vpux::NDTypeInterface>(origOp.getFilter().getType());
        mlir::ArrayAttr weightAlignmentAttr = nullptr;
        const auto weightAlignment = getWeightsTensorAlignment(strategy);
        if (weightAlignment.has_value()) {
            weightAlignmentAttr = getIntArrayAttr(ctx, weightAlignment.value());
        }
        const auto weightsTensorDistributionMode = getWeightsTensorDistributionMode(strategy);
        const auto weightsTensorNumTiles =
                getIntArrayAttr(ctx, getWeightsTensorNumTiles(clusteredOp, filterType, numClusters, strategy));
        return getDistributedTypeFromInput(clusteredOp, origOp.getFilter(), weightsTensorDistributionMode,
                                           weightsTensorNumTiles, weightAlignmentAttr, strategy,
                                           hasExplicitDistributedAttr, siblingsAnalysis);
    } else if (operand.get() == origOp.getWeightsTable() || operand.get() == origOp.getWeightTableScale() ||
               operand.get() == origOp.getWeightTableBias()) {
        auto outputType = mlir::cast<vpux::NDTypeInterface>(origOp.getOutput().getType());
        mlir::ArrayAttr weightAlignmentAttr = nullptr;
        const auto weightsTableTensorDistributionMode = getWeightsTensorDistributionMode(strategy);
        const auto weightsTableTensorNumTiles =
                getIntArrayAttr(ctx, getWeightsTableTensorNumTiles(clusteredOp, outputType, numClusters, strategy));
        const auto weightAlignment = getWeightsTensorAlignment(strategy);
        if (weightAlignment.has_value()) {
            weightAlignmentAttr = getIntArrayAttr(ctx, weightAlignment.value());
        }
        return getDistributedTypeFromInput(clusteredOp, operand.get(), weightsTableTensorDistributionMode,
                                           weightsTableTensorNumTiles, weightAlignmentAttr, strategy,
                                           hasExplicitDistributedAttr, siblingsAnalysis);
    }
    VPUX_THROW("Failed to compute distributed type for op {0}", clusteredOp);
    return nullptr;
}

//
// sparsitySupport
//

vpux::VPU::SparsitySupport vpux::VPU::NCEDepthConvolutionOp::sparsitySupport() {
    // Super-dense mode does not support ODU sparsity
    const auto outputType = mlir::cast<vpux::NDTypeInterface>(getOutput().getType());
    auto excludeMode = VPU::NCESparsity::bitwiseNot(VPU::SparsitySupport::NONE);
    if (VPU::NCESparsity::isSuperdenseRequired(outputType.getDimsOrder(), outputType.getShape(),
                                               outputType.getElementType())) {
        excludeMode = VPU::NCESparsity::bitwiseNot(VPU::SparsitySupport::SPARSE_OUTPUTS);
    }

    return VPU::SparsitySupport::SPARSE_OUTPUTS & excludeMode;
}

mlir::LogicalResult vpux::VPU::NCEDepthConvolutionOp::verifyKernel(IE::GroupConvolutionOp origOp, Logger log) {
    log.setName("NCEInvariant");

    if (mlir::cast<vpux::NDTypeInterface>(origOp.getInput().getType()).getRank() != 4) {
        return mlir::failure();
    }
    if (mlir::cast<vpux::NDTypeInterface>(origOp.getFilter().getType()).getRank() != 4) {
        return mlir::failure();
    }

    const auto dilations = parseIntArrayAttr<int64_t>(origOp.getDilations());

    const auto filterShape = getShape(origOp.getFilter());
    const auto filtersPerInChan = filterShape[Dims4D::Filter::IC];
    const auto OC = filterShape[Dims4D::Filter::OC];
    const auto KY = filterShape[Dims4D::Filter::KY];
    const auto KX = filterShape[Dims4D::Filter::KX];

    if (!origOp.getGroups().has_value()) {
        log.trace("[{0}] Grouped convolution does not have groups", origOp->getLoc());
        return mlir::failure();
    }
    if (origOp.getGroups().value() != OC) {
        log.trace("[{0}] Unsupported group size: '{1}' expected '{2}'", origOp->getLoc(), origOp.getGroups(), OC);
        return mlir::failure();
    }
    if (filtersPerInChan != 1) {
        log.trace("[{0}] Group Convolution with more than one filter per channel is not supported", origOp->getLoc());
        return mlir::failure();
    }

    const auto inputShape = getShape(origOp.getInput());
    const auto IC = inputShape[Dims4D::Act::C];
    if (OC != IC) {
        log.trace("[{0}] Group Convolution has {1} groups, expected {2}", origOp->getLoc(), OC, IC);
        return mlir::failure();
    }

    const auto kernelStrides = parseIntArrayAttr<int64_t>(origOp.getStrides());
    const auto SY = kernelStrides[0];
    const auto SX = kernelStrides[1];
    auto pads = PadInfo(origOp.getPadsBegin(), origOp.getPadsEnd());
    pads = VPU::shrinkPadsForDilatedConvolution(pads, dilations);

    return NCEInvariant::verifyKernel(origOp, KY, KX, SY, SX, pads.top, pads.bottom, pads.left, pads.right, log);
}

mlir::LogicalResult vpux::VPU::NCEDepthConvolutionOp::verifyGroupConvCMX(mlir::Location loc, mlir::ModuleOp module,
                                                                         vpux::NDTypeInterface inputType,
                                                                         vpux::NDTypeInterface filterType,
                                                                         vpux::NDTypeInterface outputType,
                                                                         mlir::ArrayAttr kernelStrides, Logger log) {
    log.setName("NCEInvariant");

    VPUX_THROW_UNLESS(kernelStrides.size() == 2, "Unsupported strides size: {0}", kernelStrides.size());

    const auto filterShape = filterType.getShape();
    const auto OC = filterShape[Dims4D::Filter::OC];
    const auto filtersPerInChan = filterShape[Dims4D::Filter::IC];
    const auto KY = filterShape[Dims4D::Filter::KY];
    const auto KX = filterShape[Dims4D::Filter::KX];

    const auto alignment = VPU::NCEInvariant::getAlignment(outputType.getElementType());
    if (OC % alignment != 0) {
        log.debug("[{0}] Output channels count of depthwise convolution must be a multiple of {1}, got {2}", loc,
                  alignment, OC);
        return mlir::failure();
    }
    const auto remainder = (filtersPerInChan * KY * KX) % alignment;
    VPUX_THROW_UNLESS(remainder >= 0, "Channel alignment cannot be negative: {0}", remainder);

    const auto padding = (remainder > 0) ? (alignment - remainder) : 0;
    const auto alignedWeightShape = SmallVector<int64_t>{OC, 1, 1, filtersPerInChan * KY * KX + padding};
    const auto alignedFilterType = mlir::RankedTensorType::get(alignedWeightShape, filterType.getElementType());
    auto dwConvOp = mlir::cast<VPU::NCEDepthConvolutionOp>(module.getOperation());
    const auto requiredCMX = VPU::getRequiredCMXSizeForNCEOps(
            {inputType, alignedFilterType, outputType}, OC, VPU::countElementsPerOutputChannelInWeightTable(dwConvOp));

    const auto cmxSize = vpux::VPU::getTotalCMXSize(module);
    if (requiredCMX > cmxSize) {
        log.trace("[{0}] CMX memory is not enough for Depthwise Convolution, available '{1}', required '{2}'", loc,
                  cmxSize, requiredCMX);
        return mlir::failure();
    }

    return mlir::success();
}

mlir::LogicalResult vpux::VPU::NCEDepthConvolutionOp::reifyResultShapes(
        mlir::OpBuilder& builder, mlir::ReifiedRankedShapedTypeDims& reifiedReturnShapes) {
    // Parse attributes
    const auto strides = parseIntArrayAttr<int64_t>(getStrides());

    const auto padTop = getPad().getTop().getValue().getSExtValue();
    const auto padBottom = getPad().getBottom().getValue().getSExtValue();
    const auto padLeft = getPad().getLeft().getValue().getSExtValue();
    const auto padRight = getPad().getRight().getValue().getSExtValue();

    const auto dataPaddingBelow = SmallVector<int64_t>({padTop, padLeft});
    const auto dataPaddingAbove = SmallVector<int64_t>({padBottom, padRight});

    auto kernelShape = mlir::cast<vpux::NDTypeInterface>(getFilter().getType()).getShape();
    SmallVector<int64_t> kernelSize{kernelShape[Dims4D::Filter::KY], kernelShape[Dims4D::Filter::KX]};

    // Compute output shape using utility
    auto outShape = reifyConvPoolTensors(builder, getInput(), getOutput(), getFilter(), kernelSize, strides,
                                         dataPaddingBelow, dataPaddingAbove, getLoc());

    if (mlir::failed(outShape)) {
        return outShape;
    }

    reifiedReturnShapes.emplace_back(std::move(outShape.value()));
    return mlir::success();
}
