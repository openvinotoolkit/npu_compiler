//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <mlir/Support/LLVM.h>
#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/image.hpp"
#include "vpux/compiler/dialect/IE/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/dpu.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/image.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/utils/auto_padding_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/clustered_op_interface_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/generate_tiling.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_interpolate_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_sparsity.hpp"
#include "vpux/compiler/dialect/VPU/utils/sparsity_support.hpp"
#include "vpux/compiler/dialect/VPU/utils/sprlut_utils.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/dialect/config/constraints.hpp"

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
    const auto filterShape = Shape(op.getStaticRawFilterShape());
    const auto KY = filterShape[Dims4D::Filter::KY];
    const auto KX = filterShape[Dims4D::Filter::KX];

    VPUX_THROW_WHEN(mlir::ShapedType::isDynamic(KY) || mlir::ShapedType::isDynamic(KX),
                    "Dynamic kernel size is not supported for NCE operations");

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

/*
 * Return the mixed raw filter shape by combining the static and dynamic raw filter shape values into a single
 * SmallVector of OpFoldResults.
 */
SmallVector<mlir::OpFoldResult> vpux::VPU::NCEInterpolateOp::getMixedRawFilterShape() {
    mlir::Builder builder(getContext());
    return mlir::getMixedValues(getStaticRawFilterShape(), getRawFilterShape(), builder);
}

/*
 * Return the constant raw filter shape by extracting the constant values from the mixed raw filter shape.
 */
SmallVector<int64_t> vpux::VPU::NCEInterpolateOp::getConstRawFilterShape() {
    auto vals = mlir::getConstantIntValues(getMixedRawFilterShape());
    VPUX_THROW_WHEN(!vals.has_value(), "Cannot get constant raw filter shape from NCEInterpolateOp '{0}'", getLoc());
    return vals.value();
}

mlir::LogicalResult vpux::VPU::NCEInterpolateOp::verifyKernel(IE::InterpolateOp origOp, Logger log) {
    log.setName("NCEInvariant");
    const auto logCb = [&](const formatv_object_base& msg) {
        log.trace("{0}", msg.str());
    };

    auto seOp = mlir::dyn_cast<IE::SEOpInterface>(origOp.getOperation());
    if (!seOp || !seOp.isSupported(logCb, /*checkLayout=*/true, /*checkChannelAlignment=*/true, /*checkBatch=*/true)) {
        return mlir::failure();
    }

    return mlir::success();
}

//
// TilingBuilderOpInterace
//

TilingInfo vpux::VPU::NCEInterpolateOp::backInferTileInfo(const vpux::TileInfo& outputTile, vpux::Logger log) {
    const auto origInputShape = getShape(getInput());
    const auto origFilterShape = Shape(getConstRawFilterShape());

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

    auto nceOp = mlir::cast<VPU::NCEInterpolateOp>(getOperation());
    if (nceOp.getWeightsTable()) {
        inputTiling.tiles.push_back(
                VPU::getWeightsTableTile(this, outputTile, VPU::getWeightsChannelsAutopad(getOperation())));
    }
    if (nceOp.getWeightTableDataPtr()) {
        inputTiling.tiles.push_back(
                VPU::getDataPointerTableTile(this, outputTile, VPU::getWeightsChannelsAutopad(getOperation())));
    }
    if (nceOp.getWeightTableScale()) {
        inputTiling.tiles.push_back(VPU::getScaleTableTile(this, outputTile));
    }
    if (nceOp.getWeightTableBias()) {
        inputTiling.tiles.push_back(VPU::getBiasTableTile(this, outputTile));
    }

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
bool VPU::NCEInterpolateOp::isOperationSplitOverHeightCompatible(const vpux::TileInfo& oriOutputTile) {
    return VPU::isNCEOpSplitOverHeightCompatible(getOperation(), getInput(), getShape(getOutput()), oriOutputTile,
                                                 false);
}

bool VPU::NCEInterpolateOp::isOperationSplitOverWidthCompatible(ShapeRef outputShape, ShapeRef offset, ShapeRef axis) {
    return VPU::isOperationSplitOverWidthCompatible(getOperation(), outputShape, offset, axis);
}

bool VPU::NCEInterpolateOp::isOperationSplitOverKernelCompatible(ShapeRef outputShape, ShapeRef offset, ShapeRef axis) {
    return VPU::isOperationSplitOverKernelCompatible(getOperation(), outputShape, offset, axis);
}

bool VPU::NCEInterpolateOp::doesLayerFitIntoCMX(VPU::MultiClusterStrategy strategy,
                                                SiblingOpsAnalysis& siblingsAnalysis, Byte reservedMem) {
    const auto op = getOperation();
    auto nceOp = mlir::cast<VPU::NCEInterpolateOp>(op);
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
    auto ppeAttr = getPpe();
    addSprLutBufferIfPresent(ppeAttr, buffers);

    if (getWeights() != nullptr) {
        buffers.push_back(VPU::getTotalAllocSizeWithDistribution(
                getWeights().getType(),
                getFilterDistributionAttrFromOp(mlir::dyn_cast<VPU::NCEOpInterface>(op), getWeights().getType(),
                                                numClusters, strategy)));
    }

    if (mlir::failed(NCEInvariant::getWeightTableBuffers(op, buffers, OC))) {
        VPUX_THROW("getWeightTableBuffers function failed");
    }

    auto totalAvailableCMXSize = reservedMem.count() == 0 ? VPU::getTotalCMXSize(op).count()
                                                          : VPU::getTotalCMXFragmentationAwareSize(op).count();

    auto arch = config::getArch(op);
    return vpux::VPU::calculateAlignedBuffersMemoryRequirement(arch, buffers).count() + reservedMem.count() <=
           totalAvailableCMXSize;
}

bool VPU::NCEInterpolateOp::doesLayerChangeOutputAlignmentFitIntoCMX(
        VPU::MultiClusterStrategy strategy, VPU::DistributedTypeInterface newDistributedTensorType) {
    auto nceOp = mlir::cast<VPU::NCEInterpolateOp>(getOperation());
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
    auto origOp = mlir::cast<VPU::NCEInterpolateOp>(getOperation());
    const auto strategy = clusteredOp.getMultiClusterStrategy().value();

    if (operand.get() == origOp.getInput()) {
        return VPU::getDistributedActivationTypeForOpOperand(clusteredOp, origOp.getInput(), strategy,
                                                             hasExplicitDistributedAttr, siblingsAnalysis);
    } else if (operand.get() == origOp.getWeights()) {
        return VPU::getDistributedWeightsTypeForOpOperand(clusteredOp, origOp.getWeights(), strategy,
                                                          hasExplicitDistributedAttr, siblingsAnalysis);
    } else if (operand.get() == origOp.getWeightsTable() || operand.get() == origOp.getWeightTableDataPtr() ||
               operand.get() == origOp.getWeightTableScale() || operand.get() == origOp.getWeightTableBias()) {
        return VPU::getDistributedWeightsTypeForOpOperand(clusteredOp, operand.get(), strategy,
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
    SmallVector<Byte> buffers = {input.getTotalAllocSize(), filter.getTotalAllocSize(), output.getTotalAllocSize()};
    const auto OC = output.getShape()[Dims4D::Act::C];
    const auto op = getOperation();
    if (mlir::failed(NCEInvariant::getWeightTableBuffers(op, buffers, OC))) {
        VPUX_THROW("getWeightTableBuffers function failed");
    }
    auto ppeAttr = getPpe();
    addSprLutBufferIfPresent(ppeAttr, buffers);
    auto totalAvailableCMXSize =
            reservedMem.count() == 0 ? getTotalCMXSize(op).count() : getTotalCMXFragmentationAwareSize(op).count();
    auto arch = config::getArch(op);
    return vpux::VPU::calculateAlignedBuffersMemoryRequirement(arch, buffers).count() + reservedMem.count() <=
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
