//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/IE/utils/matmul.hpp"
#include "vpux/compiler/dialect/VPU/IR/native_attributes/distribution_info.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/generate_tiling.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <mlir/Support/LogicalResult.h>

using namespace vpux;

//
// InferTypeOpInterface
//

mlir::LogicalResult vpux::VPU::NCEMatMulOp::inferReturnTypes(mlir::MLIRContext* ctx,
                                                             std::optional<mlir::Location> optLoc,
                                                             mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                             mlir::OpaqueProperties props,
                                                             [[maybe_unused]] mlir::RegionRange regions,
                                                             mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    NCEMatMulOpAdaptor op(operands, attrs, props);

    if (mlir::failed(op.verify(loc))) {
        return mlir::failure();
    }

    auto inputType = mlir::cast<vpux::NDTypeInterface>(op.getInput().getType());
    auto weightsType = mlir::cast<vpux::NDTypeInterface>(op.getWeights().getType());

    const auto inputShape = inputType.getShape();
    const auto weightsShape = weightsType.getShape();

    SmallVector<int64_t> outputShape{inputShape[Dim(0)], inputShape[Dim(1)], weightsShape[Dim(1)], inputShape[Dim(3)],
                                     inputShape[Dim(4)]};

    auto outputType =
            mlir::RankedTensorType::get(outputShape, inputType.getElementType(), createTensorAttrFromType(inputType));

    inferredReturnTypes.push_back(outputType);
    return mlir::success();
}

//
// Verifier
//

mlir::LogicalResult vpux::VPU::NCEMatMulOp::verify() {
    const auto op = getOperation();

    if (mlir::failed(vpux::VPU::verifyNCEOp(op))) {
        return mlir::failure();
    }

    return mlir::success();
}

mlir::LogicalResult vpux::VPU::NCEMatMulOp::verifyKernel(IE::MatMulOp) {
    return mlir::success();
}

//
// fitIntoCMX
//

bool doesNCEMatMulFitIntoCMX(vpux::NDTypeInterface inputType, vpux::NDTypeInterface filterType,
                             vpux::NDTypeInterface outputType, mlir::ModuleOp moduleOp, Byte reservedMem) {
    auto arch = config::getArch(moduleOp);

    auto largestGroupsNumPerCluster = filterType.getShape()[DimsGroups5D::Act::G];
    if (auto distType = mlir::dyn_cast<VPU::DistributedTensorType>(filterType)) {
        largestGroupsNumPerCluster = distType.getLargestCompactShape()[DimsGroups5D::Act::G];
    }

    const auto weightsTableSize = vpux::VPU::NCEInvariant::getWeightsTableSize(
            outputType.getShape()[DimsGroups5D::Act::C] * largestGroupsNumPerCluster);

    SmallVector<Byte> buffers = {
            inputType.getTotalAllocSize(),
            filterType.getTotalAllocSize(),
            outputType.getTotalAllocSize(),
            weightsTableSize,
    };

    const auto totalAvailableCMXSize = reservedMem.count() == 0
                                               ? vpux::VPU::getTotalCMXSize(moduleOp).count()
                                               : vpux::VPU::getTotalCMXFragmentationAwareSize(moduleOp).count();

    const auto requiredMemoryAligned = vpux::VPU::calculateAlignedBuffersMemoryRequirement(arch, buffers).count();

    return requiredMemoryAligned + reservedMem.count() <= totalAvailableCMXSize;
}

bool vpux::VPU::NCEMatMulOp::fitIntoCMX(vpux::NDTypeInterface inputType, vpux::NDTypeInterface filterType,
                                        vpux::NDTypeInterface outputType, Byte reservedMem) {
    auto mod = getModuleOp(getOperation());
    return doesNCEMatMulFitIntoCMX(inputType, filterType, outputType, mod, reservedMem);
}

bool vpux::VPU::NCEMatMulOp::fitIntoCMX(vpux::NDTypeInterface inputType, vpux::NDTypeInterface filterType,
                                        vpux::NDTypeInterface outputType) {
    return fitIntoCMX(inputType, filterType, outputType, Byte(0));
}

//
// isSupported
//

bool isNCEMatMulSupported(vpux::NDTypeInterface inputType, [[maybe_unused]] vpux::NDTypeInterface filterType,
                          vpux::NDTypeInterface outputType, mlir::ModuleOp moduleOp, vpux::LogCb logCb,
                          bool checkLayout, [[maybe_unused]] bool checkChannelAlignment) {
    if (auto inOrder = inputType.getDimsOrder(); checkLayout && inOrder != DimsOrder::GNHWC) {
        logCb(llvm::formatv("VPU::NCEMatMulOp input has unsupported layout '{0}'", inOrder));
        return false;
    }

    // If we have less groups than clusters, it doesn't make sense to try split-over-group optimisation.
    const auto groups = outputType.getShape()[DimsGroups5D::Act::G];
    const auto clusters = config::getTileExecutor(moduleOp).getCount();

    if (groups < clusters) {
        logCb(llvm::formatv("VPU::NCEMatMulOp input has fewer groups than there are available clusters"));
        return false;
    }

    return true;
}

bool VPU::NCEMatMulOp::isSupported(IE::MatMulOp op, vpux::LogCb logCb, bool checkLayout, bool checkChannelAlignment) {
    auto mod = getModuleOp(op);

    const auto inputType = mlir::cast<vpux::NDTypeInterface>(op.getInput1().getType());
    const auto filterType = mlir::cast<vpux::NDTypeInterface>(op.getInput2().getType());
    const auto outputType = mlir::cast<vpux::NDTypeInterface>(op.getOutput().getType());

    const auto inputShape = inputType.getShape();
    const auto filterShape = filterType.getShape();
    const auto outputShape = outputType.getShape();

    bool isSupported = isNCEMatMulSupported(
            inputType.changeShape(ShapeRef({inputShape[Dims4D::Act::C] * inputShape[Dims4D::Act::N], 1,
                                            inputShape[Dims4D::Act::W], inputShape[Dims4D::Act::H], 1})),
            // Filter shape 2nd and 3rd can be incorrect depending on transposeB option, however filter is not used in
            // checks
            filterType.changeShape(ShapeRef({filterShape[Dims4D::Act::C] * filterShape[Dims4D::Act::N],
                                             filterShape[Dims4D::Act::H], filterShape[Dims4D::Act::W], 1, 1})),
            outputType.changeShape(ShapeRef({outputShape[Dims4D::Act::C] * outputShape[Dims4D::Act::N], 1,
                                             outputShape[Dims4D::Act::W], outputShape[Dims4D::Act::H], 1})),
            mod, logCb, checkLayout, checkChannelAlignment);
    return isSupported;
}

bool VPU::NCEMatMulOp::isSupported(VPU::NCEMatMulOp op, vpux::LogCb logCb, bool checkLayout,
                                   bool checkChannelAlignment) {
    auto mod = getModuleOp(op);

    const auto inputType = mlir::cast<vpux::NDTypeInterface>(op.getInput().getType());
    const auto filterType = mlir::cast<vpux::NDTypeInterface>(op.getWeights().getType());
    const auto outputType = mlir::cast<vpux::NDTypeInterface>(op.getOutput().getType());

    return isNCEMatMulSupported(inputType, filterType, outputType, mod, logCb, checkLayout, checkChannelAlignment);
}

//
// TilingBuilderOpInterace
//

// Returns a WeightsTable tile required to produce the specific output tile
TileInfo getWeightsTableTile5D(VPU::NCEMatMulOp origOp, const vpux::TileInfo& outputTile) {
    const auto origWeightsTable = origOp.getWeightsTable();
    VPUX_THROW_UNLESS(origWeightsTable != nullptr, "The operation {0} doesn't have a WeightsTable", *origOp);

    const auto origWeightsTableShape = getShape(origWeightsTable);

    VPUX_THROW_UNLESS(
            origWeightsTableShape[DimsGroups5D::Filter::G] == getShape(origOp.getOutput())[DimsGroups5D::Act::G] &&
                    origWeightsTableShape[DimsGroups5D::Filter::OC] ==
                            getShape(origOp.getOutput())[DimsGroups5D::Act::C] &&
                    origWeightsTableShape[DimsGroups5D::Filter::IC] == 1 &&
                    origWeightsTableShape[DimsGroups5D::Filter::KY] == 1 &&
                    origWeightsTableShape[DimsGroups5D::Filter::KX] ==
                            VPU::NCEInvariant::WEIGHT_TABLE_NUM_ELEMENTS_PER_OC,
            "Unexpected WeightsTable shape notation or order: {0} with output shape of {1}"
            "\nProbably, we need to update this logic",
            origWeightsTableShape, getShape(origOp.getOutput()));

    TileInfo weightsTableTile(origWeightsTableShape);
    weightsTableTile.offsets[DimsGroups5D::Filter::OC] = outputTile.offsets[DimsGroups5D::Act::C];
    weightsTableTile.shape[DimsGroups5D::Filter::OC] = outputTile.shape[DimsGroups5D::Act::C];
    weightsTableTile.offsets[DimsGroups5D::Filter::G] = outputTile.offsets[DimsGroups5D::Act::G];
    weightsTableTile.shape[DimsGroups5D::Filter::G] = outputTile.shape[DimsGroups5D::Act::G];
    return weightsTableTile;
}

TilingInfo vpux::VPU::NCEMatMulOp::backInferTileInfo(const vpux::TileInfo& outputTile, vpux::Logger log) {
    const auto origInputShape = getShape(getInput());
    const auto origFilterShape = Shape(parseIntArrayAttr<int64_t>(getRawFilterShape()));
    const auto origPadding = toPadInfo(getPad());

    auto inputTiling =
            vpux::backInferMatMulTile(outputTile, origInputShape, origFilterShape, getStrides(), origPadding);
    VPUX_THROW_UNLESS(mlir::succeeded(checkAndAlignActInputTiling(
                              mlir::cast<VPU::NCEOpInterface>(*this->getOperation()), inputTiling, log)),
                      "Failed to get an aligned act input tiling");

    inputTiling.tiles.push_back(getWeightsTableTile5D(*this, outputTile));

    return inputTiling;
}

void vpux::VPU::NCEMatMulOp::adjustAttrs(const vpux::TilingInfo& inputTiling, const vpux::TileInfo& outputTile) {
    VPU::adjustPaddings(this, inputTiling);
    auto newRawFilterShape = Shape(parseIntArrayAttr<int64_t>(getRawFilterShape()));
    newRawFilterShape[DimsGroups5D::Filter::OC] = outputTile.shape[DimsGroups5D::Act::C];
    newRawFilterShape[DimsGroups5D::Filter::G] = outputTile.shape[DimsGroups5D::Act::G];
    setRawFilterShapeAttr(getIntArrayAttr(getContext(), newRawFilterShape));
}

mlir::FailureOr<OutputTiling> vpux::VPU::NCEMatMulOp::getTilingStrategy(TilingMode tilingMode, Logger log) {
    return vpux::getHWLayerTilingStrategy(this->getOperation(), tilingMode, log);
}

//
// ClusteredOpInterface
//

bool vpux::VPU::NCEMatMulOp::checkStrategyCompatibility(vpux::VPU::MultiClusterStrategy strategy, size_t) {
    return strategy == VPU::MultiClusterStrategy::SplitOverGroup;
}

vpux::VPU::DistributionInfo vpux::VPU::NCEMatMulOp::getExplicitDistributionInfoAttr(
        vpux::ShapeRef shape, vpux::VPU::DistributionMode distributionMode, ArrayRef<int64_t> numTiles,
        const int64_t numClusters, ArrayRef<int64_t> alignment, const bool uniformDistributedSegments,
        const vpux::VPU::OverlapDistributionParams& overlapParams) {
    return VPU::getNCEExplicitDistributionInfo(mlir::dyn_cast<VPU::NCEOpInterface>(getOperation()), shape,
                                               distributionMode, numTiles, numClusters, alignment,
                                               uniformDistributedSegments, overlapParams);
}

bool VPU::NCEMatMulOp::isOperationSplitOverHeightCompatible([[maybe_unused]] const vpux::TileInfo& oriOutputTile) {
    return false;
}

bool VPU::NCEMatMulOp::isOperationSplitOverWidthCompatible([[maybe_unused]] ShapeRef outputShape,
                                                           [[maybe_unused]] ShapeRef offset,
                                                           [[maybe_unused]] ShapeRef axis) {
    return false;
}

bool VPU::NCEMatMulOp::isOperationSplitOverKernelCompatible([[maybe_unused]] ShapeRef outputShape,
                                                            [[maybe_unused]] ShapeRef offset,
                                                            [[maybe_unused]] ShapeRef axis) {
    return false;
}

bool VPU::NCEMatMulOp::doesLayerFitIntoCMX(VPU::MultiClusterStrategy strategy, SiblingOpsAnalysis& siblingsAnalysis,
                                           Byte reservedMem) {
    auto nceOp = mlir::cast<VPU::NCEMatMulOp>(getOperation());
    auto nceOpInterface = mlir::cast<VPU::NCEOpInterface>(getOperation());
    const auto outputType = mlir::cast<vpux::NDTypeInterface>(nceOp->getResult(0).getType());
    auto numClusters = VPU::getOptimalNumClusters(nceOp, outputType.getShape(), strategy);
    auto mod = getModuleOp(getOperation());
    auto arch = config::getArch(mod);

    auto filterType = mlir::cast<vpux::NDTypeInterface>(getWeights().getType());
    auto largestGroupsNumPerCluster = filterType.getShape()[DimsGroups5D::Act::G];
    if (auto distType = mlir::dyn_cast<VPU::DistributedTensorType>(filterType)) {
        largestGroupsNumPerCluster = distType.getLargestCompactShape()[DimsGroups5D::Act::G];
    }

    const auto weightsTableSize = vpux::VPU::NCEInvariant::getWeightsTableSize(
            outputType.getShape()[DimsGroups5D::Act::C] * largestGroupsNumPerCluster);

    SmallVector<Byte> buffers = {
            VPU::getTotalAllocSizeWithDistribution(
                    getInput().getType(), getActivationDistributionAttrFromOp(nceOp, getInput(), getInput().getType(),
                                                                              numClusters, strategy, siblingsAnalysis)),
            VPU::getTotalAllocSizeWithDistribution(
                    getWeights().getType(),
                    getFilterDistributionAttrFromOp(nceOpInterface, getWeights().getType(), numClusters, strategy)),
            VPU::getTotalAllocSizeWithDistribution(
                    getOutput().getType(), getOutputDistributionAttrFromOp(nceOp, getOutput().getType(), numClusters,
                                                                           strategy, siblingsAnalysis)),
            weightsTableSize};

    const auto totalAvailableCMXSize = reservedMem.count() == 0
                                               ? vpux::VPU::getTotalCMXSize(mod).count()
                                               : vpux::VPU::getTotalCMXFragmentationAwareSize(mod).count();

    const auto requiredMemoryAligned = vpux::VPU::calculateAlignedBuffersMemoryRequirement(arch, buffers).count();

    return requiredMemoryAligned + reservedMem.count() <= totalAvailableCMXSize;
}

bool VPU::NCEMatMulOp::doesLayerChangeOutputAlignmentFitIntoCMX(
        VPU::MultiClusterStrategy strategy, VPU::DistributedTypeInterface newDistributedTensorType) {
    auto nceOp = mlir::cast<VPU::NCEMatMulOp>(getOperation());
    auto nceOpInterface = mlir::cast<VPU::NCEOpInterface>(getOperation());
    auto numClusters = VPU::getOptimalNumClusters(
            nceOp, mlir::cast<vpux::NDTypeInterface>(nceOp.getOutput().getType()).getShape(), strategy);
    auto distributedInputType = getDistributedActivationTypeFromOp(nceOp, nceOp.getInput(), nceOp.getInput().getType(),
                                                                   numClusters, strategy);
    auto distributedFilterType =
            getDistributedFilterTypeFromOp(nceOpInterface, nceOp.getWeights().getType(), numClusters, strategy);
    return fitIntoCMX(distributedInputType, distributedFilterType, newDistributedTensorType);
}

vpux::NDTypeInterface vpux::VPU::NCEMatMulOp::getDistributedTypeForOpOperand(mlir::OpOperand& operand,
                                                                             bool hasExplicitDistributedAttr,
                                                                             SiblingOpsAnalysis& siblingsAnalysis) {
    auto clusteredOp = mlir::cast<VPU::ClusteredOpInterface>(getOperation());
    auto origOp = mlir::cast<NCEMatMulOp>(getOperation());
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
    } else if (operand.get() == origOp.getWeights()) {
        auto filterType = mlir::cast<vpux::NDTypeInterface>(origOp.getWeights().getType());
        const auto weightsTensorDistributionMode = getWeightsTensorDistributionMode(strategy);
        const auto weightsTensorNumTiles =
                getIntArrayAttr(ctx, getWeightsTensorNumTiles(clusteredOp, filterType, numClusters, strategy));
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
