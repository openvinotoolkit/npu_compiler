//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/IE/utils/matmul.hpp"
#include "vpux/compiler/dialect/VPU/IR/native_attributes/distribution_info.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/dpu.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/generate_tiling.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_matmul_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/sprlut_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/tile_utils.hpp"
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
    // Infer the extra NCE outputs types if any
    auto resultSegmentSizes = op.getProperties().getResultSegmentSizes();

    return inferReduceExtraNCETypes(loc, outputType, op.getAxesValue(), resultSegmentSizes, inferredReturnTypes);
}

//
// Verifier
//

mlir::LogicalResult vpux::VPU::NCEMatMulOp::verify() {
    const auto op = getOperation();
    const auto arch = config::getArch(op);

    // Skip checks if architecture is unknown
    if (arch == config::ArchKind::UNKNOWN) {
        return mlir::success();
    }

    if (mlir::failed(VPU::NCEInvariant::verifyWeightTables(op))) {
        return mlir::failure();
    }

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
                             vpux::NDTypeInterface outputType, mlir::Operation* op, Byte reservedMem) {
    auto moduleOp = getModuleOp(op);
    auto arch = config::getArch(moduleOp);

    auto largestGroupsNumPerCluster = filterType.getShape()[DimsGroups5D::Act::G];
    if (auto distType = mlir::dyn_cast<VPU::DistributedTensorType>(filterType)) {
        largestGroupsNumPerCluster = distType.getLargestCompactShape()[DimsGroups5D::Act::G];
    }

    SmallVector<Byte> buffers = {
            inputType.getTotalAllocSize(),
            filterType.getTotalAllocSize(),
            outputType.getTotalAllocSize(),
    };

    auto nceOpInterface = mlir::dyn_cast<VPU::NCEOpInterface>(op);
    if (nceOpInterface != nullptr) {
        auto ppeAttr = nceOpInterface.getPPE();
        addSprLutBufferIfPresent(ppeAttr, buffers);
    }

    // Legacy weights table is laid out per-group (shape includes G), so the size scales with OC*G.
    // New-format tables are created with G=1 and shared across groups (see BatchMatMulToMatMul).
    const auto OC = outputType.getShape()[DimsGroups5D::Act::C];
    const auto weightTableOC =
            nceOpInterface.getWeightsTableOperand() == nullptr ? OC : OC * largestGroupsNumPerCluster;
    if (mlir::failed(VPU::NCEInvariant::getWeightTableBuffers(op, buffers, weightTableOC))) {
        VPUX_THROW("getWeightTableBuffers function failed");
    }
    if (mlir::failed(vpux::VPU::getReduceOutputBuffers(op, buffers, outputType))) {
        VPUX_THROW("getReduceOutputBuffers function failed");
    }

    const auto totalAvailableCMXSize = reservedMem.count() == 0
                                               ? vpux::VPU::getTotalCMXSize(moduleOp).count()
                                               : vpux::VPU::getTotalCMXFragmentationAwareSize(moduleOp).count();

    const auto requiredMemoryAligned = vpux::VPU::calculateAlignedBuffersMemoryRequirement(arch, buffers).count();

    return requiredMemoryAligned + reservedMem.count() <= totalAvailableCMXSize;
}

bool vpux::VPU::NCEMatMulOp::fitIntoCMX(vpux::NDTypeInterface inputType, vpux::NDTypeInterface filterType,
                                        vpux::NDTypeInterface outputType, Byte reservedMem) {
    return doesNCEMatMulFitIntoCMX(inputType, filterType, outputType, getOperation(), reservedMem);
}

bool vpux::VPU::NCEMatMulOp::fitIntoCMX(vpux::NDTypeInterface inputType, vpux::NDTypeInterface filterType,
                                        vpux::NDTypeInterface outputType) {
    return fitIntoCMX(inputType, filterType, outputType, Byte(0));
}

/*
 * Return the mixed raw filter shape by combining the static and dynamic raw filter shape values into a single
 * SmallVector of OpFoldResults.
 */
SmallVector<mlir::OpFoldResult> vpux::VPU::NCEMatMulOp::getMixedRawFilterShape() {
    mlir::Builder builder(getContext());
    return mlir::getMixedValues(getStaticRawFilterShape(), getRawFilterShape(), builder);
}

/*
 * Return the constant raw filter shape by extracting the constant values from the mixed raw filter shape.
 */
SmallVector<int64_t> vpux::VPU::NCEMatMulOp::getConstRawFilterShape() {
    auto vals = mlir::getConstantIntValues(getMixedRawFilterShape());
    VPUX_THROW_WHEN(!vals.has_value(), "Cannot get constant raw filter shape from NCEMatMulOp '{0}'", getLoc());
    return vals.value();
}

//
// isSupported
//

bool VPU::NCEMatMulOp::isSupported(IE::MatMulOp op, vpux::LogCb logCb, bool checkLayout, bool checkChannelAlignment) {
    auto mod = getModuleOp(op);

    const auto inputType = mlir::cast<vpux::NDTypeInterface>(op.getInput1().getType());
    const auto filterType = mlir::cast<vpux::NDTypeInterface>(op.getInput2().getType());
    const auto outputType = mlir::cast<vpux::NDTypeInterface>(op.getOutput().getType());

    const auto inputShape = inputType.getShape();
    const auto filterShape = filterType.getShape();
    const auto outputShape = outputType.getShape();

    // NCEMatMulOp requires equal G dimensions across all inputs (no broadcast across groups).
    if (inputShape[Dims4D::Act::C] != filterShape[Dims4D::Act::C]) {
        logCb(llvm::formatv("NCEMatMulOp does not support broadcast MatMul (input1 G={0} != input2 G={1})",
                            inputShape[Dims4D::Act::C], filterShape[Dims4D::Act::C]));
        return false;
    }

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

namespace {
enum class WeightTableType : uint64_t {
    UNKNOWN = 0,
    LEGACY = 1,
    SCALE = 2,
    BIAS = 3,
    ZERO_POINTS = 4,
};
}

// Returns a WeightsTable tile required to produce the specific output tile
// WeightsTable is a generic name and it can be a legacy weight table, a scale table or a bias table, depending on the
// weightTableType argument passed
TileInfo getWeightsTableTile5D(VPU::NCEMatMulOp origOp, const vpux::TileInfo& outputTile,
                               const WeightTableType& weightTableType) {
    VPUX_THROW_WHEN(weightTableType == WeightTableType::UNKNOWN, "Please provide a valid weight table type");

    const auto isLegacyWeightTableType = weightTableType == WeightTableType::LEGACY;
    const auto origWeightsTable = isLegacyWeightTableType                     ? origOp.getWeightsTable()
                                  : weightTableType == WeightTableType::SCALE ? origOp.getWeightTableScale()
                                  : weightTableType == WeightTableType::BIAS  ? origOp.getWeightTableBias()
                                                                              : origOp.getWeightZeroPoints();
    VPUX_THROW_UNLESS(origWeightsTable != nullptr, "The operation {0} doesn't have the required type of weight table",
                      *origOp);

    const auto origWeightsTableShape = getShape(origWeightsTable);
    const auto expectedKX = isLegacyWeightTableType ? VPU::NCEInvariant::WEIGHT_TABLE_NUM_ELEMENTS_PER_OC
                                                    : VPU::NCEInvariant::NEW_WEIGHT_TABLE_NUM_ELEMENTS_PER_OC;

    // Scale/bias/zero-point tables are always created with G=1 and shared across all groups (see BatchMatMulToMatMul).
    const auto expectedG = isLegacyWeightTableType ? getShape(origOp.getOutput())[DimsGroups5D::Act::G] : 1;
    VPUX_THROW_UNLESS(origWeightsTableShape[DimsGroups5D::Filter::G] == expectedG &&
                              origWeightsTableShape[DimsGroups5D::Filter::OC] ==
                                      getShape(origOp.getOutput())[DimsGroups5D::Act::C] &&
                              origWeightsTableShape[DimsGroups5D::Filter::IC] == 1 &&
                              origWeightsTableShape[DimsGroups5D::Filter::KY] == 1 &&
                              origWeightsTableShape[DimsGroups5D::Filter::KX] == expectedKX,
                      "Unexpected WeightsTable shape notation or order: {0} with output shape of {1}"
                      "\nProbably, we need to update this logic",
                      origWeightsTableShape, getShape(origOp.getOutput()));

    TileInfo weightsTableTile(origWeightsTableShape);
    weightsTableTile.offsets[DimsGroups5D::Filter::OC] = outputTile.offsets[DimsGroups5D::Act::C];
    weightsTableTile.shape[DimsGroups5D::Filter::OC] = outputTile.shape[DimsGroups5D::Act::C];
    if (isLegacyWeightTableType) {
        weightsTableTile.offsets[DimsGroups5D::Filter::G] = outputTile.offsets[DimsGroups5D::Act::G];
        weightsTableTile.shape[DimsGroups5D::Filter::G] = outputTile.shape[DimsGroups5D::Act::G];
    } else {
        // Scale/bias/zero-point tables are always created with G=1 and are shared across all groups
        // (see BatchMatMulToMatMul pass). Keep the table tile at G=1 regardless of how the output is tiled over G.
        weightsTableTile.offsets[DimsGroups5D::Filter::G] = 0;
        weightsTableTile.shape[DimsGroups5D::Filter::G] = 1;
    }
    return weightsTableTile;
}

TilingInfo vpux::VPU::NCEMatMulOp::backInferTileInfo(const vpux::TileInfo& outputTile, vpux::Logger log) {
    const auto origInputShape = getShape(getInput());
    const auto origFilterShape = Shape(getConstRawFilterShape());
    const auto origPadding = toPadInfo(getPad());

    auto inputTiling =
            vpux::backInferMatMulTile(outputTile, origInputShape, origFilterShape, getStrides(), origPadding);
    VPUX_THROW_UNLESS(mlir::succeeded(checkAndAlignActInputTiling(
                              mlir::cast<VPU::NCEOpInterface>(*this->getOperation()), inputTiling, log)),
                      "Failed to get an aligned act input tiling");

    if (this->getWeightsTable()) {
        inputTiling.tiles.push_back(getWeightsTableTile5D(*this, outputTile, WeightTableType::LEGACY));
    }
    if (this->getWeightTableScale()) {
        inputTiling.tiles.push_back(getWeightsTableTile5D(*this, outputTile, WeightTableType::SCALE));
    }
    if (this->getWeightTableBias()) {
        inputTiling.tiles.push_back(getWeightsTableTile5D(*this, outputTile, WeightTableType::BIAS));
    }
    if (this->getWeightZeroPoints()) {
        inputTiling.tiles.push_back(getWeightsTableTile5D(*this, outputTile, WeightTableType::ZERO_POINTS));
    }

    return inputTiling;
}

void vpux::VPU::NCEMatMulOp::adjustAttrs(const vpux::TilingInfo& inputTiling, const vpux::TileInfo& outputTile) {
    VPU::adjustPaddings(this, inputTiling);
    // Use the mixed representation so that any dynamic operands are kept consistent
    auto mixedRawFilterShape = getMixedRawFilterShape();
    mixedRawFilterShape[DimsGroups5D::Filter::OC.ind()] =
            mlir::getAsIndexOpFoldResult(getContext(), outputTile.shape[DimsGroups5D::Act::C]);
    mixedRawFilterShape[DimsGroups5D::Filter::G.ind()] =
            mlir::getAsIndexOpFoldResult(getContext(), outputTile.shape[DimsGroups5D::Act::G]);

    SmallVector<int64_t> staticValues;
    SmallVector<mlir::Value> dynamicValues;
    mlir::dispatchIndexOpFoldResults(mixedRawFilterShape, dynamicValues, staticValues);

    setStaticRawFilterShape(staticValues);
    getRawFilterShapeMutable().assign(dynamicValues);
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
        const vpux::VPU::OverlapDistributionParams& overlapParams,
        const std::optional<ArrayRef<int64_t>> memoryNumTiles) {
    return VPU::getNCEExplicitDistributionInfo(mlir::dyn_cast<VPU::NCEOpInterface>(getOperation()), shape,
                                               distributionMode, numTiles, numClusters, alignment,
                                               uniformDistributedSegments, overlapParams, memoryNumTiles);
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

    const auto outputDistributionMap = std::make_pair(
            outputType, getOutputDistributionAttrFromOp(nceOp, outputType, numClusters, strategy, siblingsAnalysis));

    SmallVector<Byte> buffers = {
            VPU::getTotalAllocSizeWithDistribution(
                    getInput().getType(), getActivationDistributionAttrFromOp(nceOp, getInput(), getInput().getType(),
                                                                              numClusters, strategy, siblingsAnalysis)),
            VPU::getTotalAllocSizeWithDistribution(
                    getWeights().getType(),
                    getFilterDistributionAttrFromOp(nceOpInterface, getWeights().getType(), numClusters, strategy)),
            VPU::getTotalAllocSizeWithDistribution(outputDistributionMap.first, outputDistributionMap.second)};

    auto ppeAttr = getPpe();
    addSprLutBufferIfPresent(ppeAttr, buffers);

    // Legacy weights table is laid out per-group (shape includes G), so the size scales with OC*G.
    // New-format tables are created with G=1 and shared across groups (see BatchMatMulToMatMul).
    const auto OC = outputType.getShape()[DimsGroups5D::Act::C];
    const auto weightTableOC =
            nceOpInterface.getWeightsTableOperand() == nullptr ? OC : OC * largestGroupsNumPerCluster;
    if (mlir::failed(NCEInvariant::getWeightTableBuffers(getOperation(), buffers, weightTableOC))) {
        VPUX_THROW("getWeightTableBuffers function failed");
    }
    if (mlir::failed(vpux::VPU::getReduceOutputBuffers(getOperation(), buffers, outputDistributionMap))) {
        VPUX_THROW("getReduceOutputBuffers function failed");
    }

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

    if (operand.get() == origOp.getInput()) {
        return VPU::getDistributedActivationTypeForOpOperand(clusteredOp, origOp.getInput(), strategy,
                                                             hasExplicitDistributedAttr, siblingsAnalysis);
    } else if (operand.get() == origOp.getWeights()) {
        return VPU::getDistributedWeightsTypeForOpOperand(clusteredOp, origOp.getWeights(), strategy,
                                                          hasExplicitDistributedAttr, siblingsAnalysis);
    } else if (operand.get() == origOp.getWeightTableScale() || operand.get() == origOp.getWeightTableBias() ||
               operand.get() == origOp.getWeightZeroPoints()) {
        // Scale/bias/zero-point tables are created with G=1 (see BatchMatMulToMatMul pass). Use SEGMENTED mode only
        // when tiling over OC (SplitOverKernel); otherwise the mode is DUPLICATED.
        if (strategy == VPU::MultiClusterStrategy::SplitOverKernel) {
            return VPU::getDistributedWeightsTypeForOpOperand(clusteredOp, operand.get(), strategy,
                                                              hasExplicitDistributedAttr, siblingsAnalysis);
        }
        return getDistributedTypeFromInput(clusteredOp, operand.get(), VPU::DistributionMode::DUPLICATED, {}, {},
                                           strategy, hasExplicitDistributedAttr, siblingsAnalysis);
    } else if (operand.get() == origOp.getWeightsTable()) {
        return VPU::getDistributedWeightsTypeForOpOperand(clusteredOp, operand.get(), strategy,
                                                          hasExplicitDistributedAttr, siblingsAnalysis);
    }
    VPUX_THROW("Failed to compute distributed type for op {0}", clusteredOp);
    return nullptr;
}
