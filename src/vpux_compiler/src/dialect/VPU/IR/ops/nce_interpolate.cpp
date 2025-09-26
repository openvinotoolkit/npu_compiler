//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/image.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/generate_tiling.hpp"
#include "vpux/compiler/dialect/VPU/utils/max_kernel_size_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_interpolate_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_sparsity.hpp"
#include "vpux/compiler/dialect/VPU/utils/sparsity_support.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"

#include <openvino/op/convolution.hpp>
#include <openvino/op/parameter.hpp>

using namespace vpux;

//
// InferTypeOpInterface
//

mlir::LogicalResult vpux::VPU::NCEInterpolateOp::inferReturnTypes(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    NCEInterpolateOpAdaptor op(operands, attrs, prop);
    if (mlir::failed(op.verify(loc))) {
        return mlir::failure();
    }

    auto inShape = getShape(op.getInput());

    const auto dataPaddingBelow = ov::CoordinateDiff({0, 0});
    const auto dataPaddingAbove = ov::CoordinateDiff({0, 0});
    const auto filterShape = Shape(parseIntArrayAttr<int64_t>(op.getRawFilterShape()));
    const auto filterStrides = Shape(parseIntArrayAttr<int64_t>(op.getStrides()));
    const auto filterDilations = ov::Strides({1, 1});

    const auto conv = ov::op::v1::Convolution(
            std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape(inShape.begin(), inShape.end())),
            std::make_shared<ov::op::v0::Parameter>(ov::element::i32,
                                                    ov::Shape(filterShape.begin(), filterShape.end())),
            ov::Strides(filterStrides.begin(), filterStrides.end()), dataPaddingBelow, dataPaddingAbove,
            filterDilations);

    const auto& outputShapeNG = conv.get_output_partial_shape(0);

    const auto outShape = to_small_vector(outputShapeNG.get_shape() | transformed([](size_t val) {
                                              return checked_cast<int64_t>(val);
                                          }));

    auto inputType = mlir::cast<vpux::NDTypeInterface>(op.getInput().getType());
    auto outputType =
            mlir::RankedTensorType::get(outShape, inputType.getElementType(), createTensorAttrFromType(inputType));

    inferredReturnTypes.push_back(outputType);
    return mlir::success();
}

//
// Verifier
//

mlir::LogicalResult vpux::VPU::NCEInterpolateOp::verify() {
    const auto op = getOperation();
    if (mlir::failed(vpux::VPU::verifyNCEOp(op))) {
        return mlir::failure();
    }

    auto sparseInput = mlir::dyn_cast<vpux::VPU::SparseTensorType>(getInput().getType());
    if (sparseInput == nullptr) {
        return mlir::failure();
    }

    auto seAttr = mlir::dyn_cast_or_null<VPU::SEInterpolateAttr>(sparseInput.getSeAttr());
    if (seAttr == nullptr) {
        return mlir::failure();
    }

    return mlir::success();
}

bool isNCEInterpolateSupported(vpux::NDTypeInterface inputType, vpux::NDTypeInterface outputType,
                               IE::InterpolateAttr attr, config::ArchKind arch, bool checkLayout,
                               bool checkChannelAlignment, bool checkBatch, mlir::Operation* op, vpux::LogCb logCb) {
    // TODO E#71403: remove dimension check
    auto dimOver8K = [](ShapeRef shape) {
        for (auto dim : shape) {
            if (dim > VPU::NCEInvariant::VPU_DIMENSION_LIMIT) {
                return true;
            }
        }
        return false;
    };
    auto inputShape = inputType.getShape();
    auto outputShape = outputType.getShape();

    if (checkBatch && inputShape[Dims4D::Act::N] != 1) {
        return false;
    }

    if (dimOver8K(inputShape) || dimOver8K(outputShape)) {
        logCb(formatv("Dimension sizes over 8192 are not supported. Input shape {0}, output shape {1}", inputShape,
                      outputShape));
        return false;
    }

    if (attr == nullptr) {
        logCb(formatv("Missing Interpolate configuration information"));
        return false;
    }

    // Antialias is not supported
    if (attr.getAntialias() != nullptr && attr.getAntialias().getValue() == true) {
        logCb(formatv("Antialias is not supported"));
        return false;
    }

    // Only 4D interpolates are supported and the interpolation axes must be H and/or W
    auto potentialScales = VPU::getNCEInterpolateScales(inputType, outputType, attr.getCoordMode());
    if (!potentialScales.has_value()) {
        return false;
    }
    const auto scales = potentialScales.value();

    if (inputShape[Dims4D::Act::C] < 8) {
        // Interpolate layers with fewer than 8 channels may perform better on SHAVE than on DPU #E100988.
        // More experiments in #E156089 validated that:
        // 1) for nearest mode with total spatial size >= 1320720 (e.g. 512x512, scale=2)
        // DPU solution always has better performance even when channels < 8;
        // 2) for other modes, spatial size hasn't show signficant impact.
        // A better cost model can be introduced in the future to clearly identify which scenarios
        // receive a hit in performance when executed on DPU
        logCb(formatv("Interpolate has less than than 8 channels: {0}", inputShape[Dims4D::Act::C]));
        if (attr.getMode().getValue() == IE::InterpolateMode::NEAREST) {
            // For Nearest mode, check the total spatial size to decide if it is supported
            const auto totalSpatialSize = inputShape[Dims4D::Act::H] * inputShape[Dims4D::Act::W] +
                                          outputShape[Dims4D::Act::H] * outputShape[Dims4D::Act::W];
            if (totalSpatialSize < 1320720) {
                return false;
            }
        } else {
            // For other modes, directly return false
            return false;
        }
    }

    // Check for the supported modes
    SmallVector<IE::InterpolateMode> supportedModes = {IE::InterpolateMode::NEAREST, IE::InterpolateMode::LINEAR,
                                                       IE::InterpolateMode::LINEAR_ONNX};
    if (llvm::find(supportedModes, attr.getMode().getValue()) == supportedModes.end()) {
        logCb(formatv("Mode {0} is not supported", attr.getMode().getValue()));
        return false;
    }

    // TODO E#107568: Add support for LINEAR TF_HALF_PIXEL_FOR_NN mode
    if (attr.getMode().getValue() == IE::InterpolateMode::LINEAR ||
        attr.getMode().getValue() == IE::InterpolateMode::LINEAR_ONNX) {
        if (attr.getCoordMode().getValue() == IE::InterpolateCoordMode::TF_HALF_PIXEL_FOR_NN) {
            logCb(formatv("Bilinear InterpolateOp with coordinate transformation mode {0} is not yet supported",
                          attr.getCoordMode().getValue()));
            return false;
        }
    }

    // TODO E#83681: Add support for NEAREST ALIGN_CORNERS mode
    if (attr.getMode().getValue() == IE::InterpolateMode::NEAREST) {
        if (attr.getCoordMode().getValue() == IE::InterpolateCoordMode::ALIGN_CORNERS) {
            logCb(formatv("Coordinate transformation mode {0} is not yet supported", attr.getCoordMode().getValue()));
            return false;
        }
    }

    // Only interpolate ops without padding are supported
    auto hasNonZeroPads = [&](mlir::ArrayAttr padsAttr) -> bool {
        if (padsAttr == nullptr) {
            return false;
        }
        auto pads = parseIntArrayAttr<int64_t>(padsAttr);
        return llvm::any_of(pads, [](int64_t pad) {
            return pad != 0;
        });
    };
    if (hasNonZeroPads(attr.getPadsBegin()) || hasNonZeroPads(attr.getPadsEnd())) {
        logCb(formatv("Padding is not supported"));
        return false;
    }

    if (VPU::hasMaxKernelSize(op)) {
        // kernelSize must be in range [1:MAX_KERNEL_SIZE]
        const auto kernelSize = VPU::getNCEInterpolateKernelSize(scales, VPU::getNCEInterpolateModeAttr(attr.getMode()),
                                                                 attr.getCoordMode());
        auto maxKernelSize = VPU::getMaxKernelSize(op);
        for (auto kernel : kernelSize) {
            if (kernel > maxKernelSize || kernel <= 0) {
                logCb(formatv("Only kernel size less than {0} are supported for nce interpolate. Got kernel Size {1}",
                              maxKernelSize, kernel));
                return false;
            }
        }
    }

    if (checkChannelAlignment) {
        if (!VPU::NCEInvariant::isInputActTypeSupported(
                    inputType, vpux::VPU::NCEInvariant::getAlignment(inputType.getElementType()),
                    /*supportsInputActCompression=*/false) ||
            !VPU::NCEInvariant::isOutputActTypeSupported(
                    outputType, vpux::VPU::NCEInvariant::getAlignment(outputType.getElementType()))) {
            logCb(formatv("Misaligned tensor shape"));
            return false;
        }
    }

    if (checkLayout) {
        if (!VPU::NCEInvariant::checkLayouts({inputType}, {outputType}, arch, 1, logCb)) {
            return false;
        }
    }

    return true;
}

bool VPU::NCEInterpolateOp::isSupported(IE::InterpolateOp op, vpux::LogCb logCb, bool checkLayout,
                                        bool checkChannelAlignment, bool checkBatch) {
    auto inputType = mlir::cast<vpux::NDTypeInterface>(op.getInput().getType());
    auto outputType = mlir::cast<vpux::NDTypeInterface>(op.getOutput().getType());

    return isNCEInterpolateSupported(inputType, outputType, op.getAttr(), config::getArch(op), checkChannelAlignment,
                                     checkLayout, checkBatch, op, logCb);
}

bool VPU::NCEInterpolateOp::isSupported(VPU::InterpolateOp op, vpux::LogCb logCb, bool checkLayout,
                                        bool checkChannelAlignment, bool checkBatch) {
    auto inputType = mlir::cast<vpux::NDTypeInterface>(op.getInput().getType());
    auto outputType = mlir::cast<vpux::NDTypeInterface>(op.getOutput().getType());

    return isNCEInterpolateSupported(inputType, outputType, op.getAttr(), config::getArch(op), checkChannelAlignment,
                                     checkLayout, checkBatch, op, logCb);
}

//
// TilingBuilderOpInterace
//

TilingInfo vpux::VPU::NCEInterpolateOp::backInferTileInfo(const vpux::TileInfo& outputTile, vpux::Logger log) {
    const auto origInputShape = getShape(getInput());
    const auto origFilterShape = Shape(parseIntArrayAttr<int64_t>(getRawFilterShape()));

    // This op incorporates bias values in WeightsTable
    const auto origBiasShape = ShapeRef();
    auto nceOpInterface = mlir::cast<VPU::NCEOpInterface>(getOperation());
    const auto strides = getIntArrayAttr(getContext(), parseIntArrayAttr<int64_t>(getStrides()));
    const auto padding = VPU::toPadInfo(nceOpInterface.getPad());

    auto inputTiling = backInferConvTile(outputTile, origInputShape, origFilterShape, origBiasShape, strides, padding);
    VPUX_THROW_UNLESS(mlir::succeeded(checkAndAlignActInputTiling(
                              mlir::cast<VPU::NCEOpInterface>(*this->getOperation()), inputTiling, log)),
                      "Failed to get an aligned act input tiling");

    // Adjust filter tile for the aligned filter
    inputTiling.tiles[1].shape = getShape(getWeights()).toValues();
    inputTiling.tiles[1].shape[Dims4D::Filter::OC] = outputTile.shape[Dims4D::Act::C];

    inputTiling.tiles.push_back(
            VPU::getWeightsTableTile(this, outputTile, VPU::getWeightsChannelsAutopad(getOperation())));

    return inputTiling;
}

void vpux::VPU::NCEInterpolateOp::adjustAttrs(const vpux::TilingInfo&, const vpux::TileInfo& outputTile) {
    // Same as NCEConvolution, but without padding
    VPU::adjustRawFilterShape(this, outputTile);
}

mlir::FailureOr<OutputTiling> vpux::VPU::NCEInterpolateOp::getTilingStrategy(TilingMode tilingMode, Logger log) {
    return vpux::getHWLayerTilingStrategy(this->getOperation(), tilingMode, log);
}

//
// ClusteredOpInterface
//

bool vpux::VPU::NCEInterpolateOp::checkStrategyCompatibility(vpux::VPU::MultiClusterStrategy strategy, size_t) {
    return strategy == VPU::MultiClusterStrategy::Clustering ||
           strategy == VPU::MultiClusterStrategy::SplitOverHeight ||
           strategy == VPU::MultiClusterStrategy::SplitOverKernel || strategy == VPU::MultiClusterStrategy::HKSwitch;
}

vpux::VPU::DistributionInfo vpux::VPU::NCEInterpolateOp::getExplicitDistributionInfoAttr(
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
bool VPU::NCEInterpolateOp::isOperationSplitOverHeightCompatible(const vpux::TileInfo& oriOutputTile) {
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

    auto nceOp = mlir::cast<NCEInterpolateOp>(getOperation());
    Shape inputShape = getShape(nceOp.getInput()).toValues();
    auto inputType = mlir::cast<vpux::NDTypeInterface>(nceOp.getInput().getType());
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
    auto tileOp = config::getTileExecutor(moduleOp);
    const auto numTiles = tileOp.getCount();

    return isSOHSupportedByDPU(inputType, inputShape, numTiles, false, config::getArch(nceOp.getOperation()));
}

bool VPU::NCEInterpolateOp::isOperationSplitOverWidthCompatible(ShapeRef outputShape, ShapeRef offset, ShapeRef axis) {
    return VPU::isOperationSplitOverWidthCompatible(getOperation(), outputShape, offset, axis);
}

bool VPU::NCEInterpolateOp::isOperationSplitOverKernelCompatible(ShapeRef outputShape, ShapeRef offset, ShapeRef axis) {
    return VPU::isOperationSplitOverKernelCompatible(getOperation(), outputShape, offset, axis);
}

bool VPU::NCEInterpolateOp::doesLayerFitIntoCMX(VPU::MultiClusterStrategy strategy,
                                                SiblingOpsAnalysis& siblingsAnalysis, Byte reservedMem) {
    auto nceOp = mlir::cast<VPU::NCEInterpolateOp>(getOperation());
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
                    getOutput().getType(), getOutputDistributionAttrFromOp(nceOp, getOutput().getType(), numClusters,
                                                                           strategy, siblingsAnalysis)),
            NCEInvariant::getWeightsTableSize(OC)};

    if (getWeights() != nullptr) {
        buffers.push_back(VPU::getTotalAllocSizeWithDistribution(
                getWeights().getType(),
                getFilterDistributionAttrFromOp(nceOpInterface, getWeights().getType(), numClusters, strategy)));
    }

    auto totalAvailableCMXSize = reservedMem.count() == 0 ? getTotalCMXSize(getOperation()).count()
                                                          : getTotalCMXFragmentationAwareSize(getOperation()).count();

    return vpux::VPU::calculateAlignedBuffersMemoryRequirement(config::getArch(getOperation()), buffers).count() +
                   reservedMem.count() <=
           totalAvailableCMXSize;
}

bool VPU::NCEInterpolateOp::doesLayerChangeOutputAlignmentFitIntoCMX(
        VPU::MultiClusterStrategy strategy, VPU::DistributedTypeInterface newDistributedTensorType) {
    auto nceOp = mlir::cast<NCEInterpolateOp>(getOperation());
    auto nceOpInterface = mlir::cast<VPU::NCEOpInterface>(getOperation());
    auto numClusters = VPU::getOptimalNumClusters(
            nceOp, mlir::cast<vpux::NDTypeInterface>(nceOp.getOutput().getType()).getShape(), strategy);
    auto distributedInputType = getDistributedActivationTypeFromOp(nceOp, nceOp.getInput(), nceOp.getInput().getType(),
                                                                   numClusters, strategy);
    auto distributedFilterType = (nceOp.getWeights() != nullptr)
                                         ? getDistributedFilterTypeFromOp(nceOpInterface, nceOp.getWeights().getType(),
                                                                          numClusters, strategy)
                                         : nullptr;
    return fitIntoCMX(distributedInputType, distributedFilterType, newDistributedTensorType);
}

vpux::NDTypeInterface vpux::VPU::NCEInterpolateOp::getDistributedTypeForOpOperand(
        mlir::OpOperand& operand, bool hasExplicitDistributedAttr, SiblingOpsAnalysis& siblingsAnalysis) {
    auto clusteredOp = mlir::cast<VPU::ClusteredOpInterface>(getOperation());
    auto origOp = mlir::cast<NCEInterpolateOp>(getOperation());
    const auto strategy = clusteredOp.getMultiClusterStrategy().value();
    auto outputTensorType = mlir::cast<vpux::NDTypeInterface>(origOp.getOutput().getType());
    auto numClusters = VPU::getOptimalNumClusters(clusteredOp, outputTensorType.getShape(), strategy);
    auto* ctx = clusteredOp->getContext();

    if (operand.get() == origOp.getInput()) {
        const auto activationTensorDistributionMode = getActivationTensorDistributionMode(clusteredOp, strategy);
        const auto activationTensorNumTiles =
                getIntArrayAttr(ctx, getActivationTensorNumTiles(clusteredOp, numClusters, strategy));
        mlir::ArrayAttr activationAlignmentAttr = nullptr;
        const auto activationAlignment = getActivationTensorAlignment(clusteredOp, numClusters, strategy);
        if (activationAlignment.has_value()) {
            activationAlignmentAttr = getIntArrayAttr(ctx, activationAlignment.value());
        }
        return getDistributedTypeFromInput(clusteredOp, origOp.getInput(), activationTensorDistributionMode,
                                           activationTensorNumTiles, activationAlignmentAttr, strategy,
                                           hasExplicitDistributedAttr, siblingsAnalysis);
    } else if (operand.get() == origOp.getWeights()) {
        auto weightsType = mlir::cast<vpux::NDTypeInterface>(origOp.getWeights().getType());
        const auto weightsTensorDistributionMode = getWeightsTensorDistributionMode(strategy);
        const auto weightsTensorNumTiles =
                getIntArrayAttr(ctx, getWeightsTensorNumTiles(clusteredOp, weightsType, numClusters, strategy));
        mlir::ArrayAttr weightAlignmentAttr = nullptr;
        const auto weightAlignment = getWeightsTensorAlignment(strategy);
        if (weightAlignment.has_value()) {
            weightAlignmentAttr = getIntArrayAttr(ctx, weightAlignment.value());
        }
        return getDistributedTypeFromInput(clusteredOp, origOp.getWeights(), weightsTensorDistributionMode,
                                           weightsTensorNumTiles, weightAlignmentAttr, strategy,
                                           hasExplicitDistributedAttr, siblingsAnalysis);
    } else if (operand.get() == origOp.getWeightsTable()) {
        auto outputType = mlir::cast<vpux::NDTypeInterface>(origOp.getOutput().getType());

        const auto weightsTableTensorDistributionMode = getWeightsTensorDistributionMode(strategy);
        const auto weightsTableTensorNumTiles =
                getIntArrayAttr(ctx, getWeightsTableTensorNumTiles(clusteredOp, outputType, numClusters, strategy));
        mlir::ArrayAttr weightAlignmentAttr = nullptr;
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
// fitIntoCMX
//

bool vpux::VPU::NCEInterpolateOp::fitIntoCMX(vpux::NDTypeInterface input, vpux::NDTypeInterface filter,
                                             vpux::NDTypeInterface output) {
    return fitIntoCMX(input, filter, output, Byte(0));
}

bool vpux::VPU::NCEInterpolateOp::fitIntoCMX(vpux::NDTypeInterface input, vpux::NDTypeInterface filter,
                                             vpux::NDTypeInterface output, Byte reservedMem) {
    const auto OC = output.getShape()[Dims4D::Act::C];
    SmallVector<Byte> buffers = {input.getTotalAllocSize(), filter.getTotalAllocSize(), output.getTotalAllocSize(),
                                 NCEInvariant::getWeightsTableSize(OC)};

    auto totalAvailableCMXSize = reservedMem.count() == 0 ? getTotalCMXSize(getOperation()).count()
                                                          : getTotalCMXFragmentationAwareSize(getOperation()).count();

    return vpux::VPU::calculateAlignedBuffersMemoryRequirement(config::getArch(getOperation()), buffers).count() +
                   reservedMem.count() <=
           totalAvailableCMXSize;
}

//
// SparseOpInterface
//

vpux::VPU::SparsitySupport vpux::VPU::NCEInterpolateOp::sparsitySupport() {
    // Super-dense mode does not support ODU sparsity
    const auto outputType = mlir::cast<vpux::NDTypeInterface>(getOutput().getType());

    auto excludeMode = VPU::NCESparsity::bitwiseNot(VPU::SparsitySupport::NONE);

    if (VPU::NCESparsity::isSuperdenseRequired(outputType.getDimsOrder(), outputType.getShape(),
                                               outputType.getElementType())) {
        excludeMode = VPU::NCESparsity::bitwiseNot(VPU::SparsitySupport::SPARSE_OUTPUTS);
    }

    return NCESparsity::FULLY_SUPPORTED_SPARSITY_MODE & excludeMode;
}
