//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/tiling.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/VPU/IR/types.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"

using namespace vpux;

void vpux::VPU::OneHotOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                mlir::IntegerAttr depth, mlir::FloatAttr onValue, mlir::FloatAttr offValue,
                                mlir::IntegerAttr axis, vpux::IE::OneHotModeAttr mode, mlir::BoolAttr rankExpanded,
                                mlir::TypeAttr outputType) {
    build(builder, state, input, depth, onValue, offValue, axis, mode, rankExpanded, outputType,
          /*multiClusterStrategy=*/nullptr);
}

mlir::LogicalResult vpux::VPU::OneHotOp::inferReturnTypes(mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc,
                                                          mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                          mlir::OpaqueProperties prop, mlir::RegionRange,
                                                          mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::OneHotOpAdaptor oneHot(operands, attrs, prop);
    if (mlir::failed(oneHot.verify(loc))) {
        return mlir::failure();
    }

    auto inType = mlir::cast<vpux::NDTypeInterface>(oneHot.getInput().getType());
    mlir::Type outElemType = oneHot.getOutputType();

    SmallVector<int64_t> outShape = to_small_vector(inType.getShape());
    const auto axis = oneHot.getAxis();
    int64_t depth = oneHot.getDepthAttr().getInt();

    if (oneHot.getRankExpanded().value_or(false)) {
        // rank_expanded mode: input rank == output rank.
        // The axis dim of the input is a placeholder 1; replace it with depth.
        outShape[axis] = depth;
    } else {
        outShape.insert(outShape.begin() + axis, depth);
    }

    const auto typeComponents = TypeComponents().setShape(ShapeRef(outShape)).setElementType(outElemType);

    auto outputType = inType;
    if (auto distributedType = mlir::dyn_cast<VPU::DistributedTensorType>(inType);
        distributedType != nullptr &&
        VPU::isDistributedAttrWithExplicitShapesAndOffsets(distributedType.getDistribution())) {
        // Explicit per-cluster shapes and offsets must be regenerated for the new OneHot shape.
        const auto distribution = distributedType.getDistribution();
        const auto newDistribution = VPU::getNonOverlappedDistributedAttr(
                ShapeRef(outShape), distribution.getMode(), distribution.getNumTiles(), distribution.getNumClusters(),
                distribution.getAlignment(), distribution.getUniformDistributedSegments(), ctx);
        outputType = distributedType.changeTypeComponentsForExplicitDistribution(typeComponents, newDistribution);
    } else {
        outputType = inType.changeTypeComponents(typeComponents);
    }

    inferredReturnTypes.push_back(outputType);

    return mlir::success();
}

//
// TilingBuilderOpInterface
//

vpux::InputTiling vpux::VPU::OneHotOp::backInferTileInfo(const vpux::TileInfo& outputTile, vpux::Logger /*log*/) {
    const auto axis = getAxis();
    VPUX_THROW_UNLESS(outputTile.axis[Dim(axis)] == 1, "OneHot depth dimension cannot be tiled, got {0} tiles",
                      outputTile.axis[Dim(axis)]);

    if (!getRankExpanded().value_or(false)) {
        // In the regular form, the output contains the depth dimension inserted
        // at `axis`, so it has one more dimension than the input. The depth
        // dimension is not tileable; remove it while mapping the output tile
        // back to the input.
        const auto inputRank = getShape(getInput()).size();
        const auto outputRank = outputTile.shape.size();
        VPUX_THROW_UNLESS(outputRank == inputRank + 1,
                          "OneHot input/output rank mismatch: input rank {0}, output rank {1}", inputRank, outputRank);

        TileInfo inputTile(inputRank);
        for (size_t outputDim = 0, inputDim = 0; outputDim < outputRank; ++outputDim) {
            if (outputDim == checked_cast<size_t>(axis)) {
                continue;
            }
            inputTile.shape[Dim(inputDim)] = outputTile.shape[Dim(outputDim)];
            inputTile.offsets[Dim(inputDim)] = outputTile.offsets[Dim(outputDim)];
            inputTile.axis[Dim(inputDim)] = outputTile.axis[Dim(outputDim)];
            ++inputDim;
        }
        inputTile.isCompletedTile = outputTile.isCompletedTile;
        return InputTiling{{std::move(inputTile)}};
    }

    // In rank_expanded mode, input rank == output rank. The axis dim of input
    // has size 1 (placeholder), while output has size depth. Since the axis
    // dimension is not tileable, just copy outputTile and shrink it to 1.
    TileInfo inputTile = outputTile;
    inputTile.shape[Dim(axis)] = 1;
    inputTile.offsets[Dim(axis)] = 0;
    inputTile.axis[Dim(axis)] = 1;
    return InputTiling{{std::move(inputTile)}};
}

void vpux::VPU::OneHotOp::adjustAttrs(const TilingInfo& /*inputTiling*/, const TileInfo& /*outputTile*/) {
}

mlir::FailureOr<OutputTiling> vpux::VPU::OneHotOp::getTilingStrategy(TilingMode tilingMode, Logger log) {
    return vpux::getSWLayerTilingStrategy(this->getOperation(), tilingMode, log);
}

SmallVector<int64_t> vpux::VPU::OneHotOp::getMaxNumTiles() {
    const auto op = getOperation();
    // axis dim is the depth dimension; it cannot be tiled (depth cannot be split).
    const auto axis = getAxis();

    SmallVector<int64_t> excludedAxes{axis};
    SmallVector<int64_t> maxNumTiles = getMaxNumTilesWithAxesExclusion(op, excludedAxes);
    return vpux::getMaxNumTiles(op, false, false, maxNumTiles);
}

//
// ClusteredOpInterface
//

bool vpux::VPU::OneHotOp::checkStrategyCompatibility(VPU::MultiClusterStrategy strategy, size_t numTiles) {
    const auto inputShape = getShape(getInput());
    if (inputShape.isDynamic() || inputShape.size() != 4) {
        return false;
    }

    const auto canSplitOnDim = [&](Dim dim) {
        return numTiles > 1 && inputShape[dim] > 1;
    };

    switch (strategy) {
    case VPU::MultiClusterStrategy::Clustering:
        return true;
    case VPU::MultiClusterStrategy::SplitOverHeight:
        return getAxis() != Dims4D::Act::H.ind() && canSplitOnDim(Dims4D::Act::H);
    case VPU::MultiClusterStrategy::SplitOverWidth:
        return getAxis() != Dims4D::Act::W.ind() && canSplitOnDim(Dims4D::Act::W);
    case VPU::MultiClusterStrategy::SplitOverBatch:
        return getAxis() != Dims4D::Act::N.ind() && canSplitOnDim(Dims4D::Act::N);
    case VPU::MultiClusterStrategy::SplitOverKernel:
        return getAxis() != Dims4D::Act::C.ind() && canSplitOnDim(Dims4D::Act::C);
    default:
        return false;
    }
}

vpux::VPU::DistributionInfo vpux::VPU::OneHotOp::getExplicitDistributionInfoAttr(
        vpux::ShapeRef shape, vpux::VPU::DistributionMode distributionMode, ArrayRef<int64_t> numTiles,
        const int64_t numClusters, ArrayRef<int64_t> alignment, const bool uniformDistributedSegments,
        const vpux::VPU::OverlapDistributionParams& overlapParams,
        const std::optional<ArrayRef<int64_t>> /* memoryNumTiles */) {
    return VPU::getSWExplicitDistributionInfo(mlir::cast<VPU::SWOpInterface>(getOperation()), shape, distributionMode,
                                              numTiles, numClusters, alignment, uniformDistributedSegments,
                                              overlapParams);
}

//
// SWOpInterface
//

bool vpux::VPU::OneHotOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem) {
    VPUX_THROW_UNLESS(buffers.size() == 2, "OneHotOp requires 1 input and 1 output, but the number of buffers is {0}",
                      buffers.size());

    SmallVector<Byte> buffersSize;
    std::transform(buffers.begin(), buffers.end(), std::back_inserter(buffersSize), [](const auto buffer) {
        return buffer.getTotalAllocSize();
    });

    auto totalAvailableCMXSize = reservedMem.count() == 0 ? getTotalCMXSize(getOperation()).count()
                                                          : getTotalCMXFragmentationAwareSize(getOperation()).count();

    return vpux::VPU::calculateAlignedBuffersMemoryRequirement(config::getArch(getOperation()), buffersSize).count() +
                   reservedMem.count() <=
           totalAvailableCMXSize;
}

bool vpux::VPU::OneHotOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers) {
    return fitIntoCMX(buffers, Byte(0));
}

bool vpux::VPU::OneHotOp::supportCycleCostCalculation() {
    return false;
}
