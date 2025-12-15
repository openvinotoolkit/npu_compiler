//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/gather_dma_utils.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"

using namespace vpux;

void vpux::VPU::GatherElementsOp::build(::mlir::OpBuilder& odsBuilder, ::mlir::OperationState& odsState,
                                        ::mlir::Value input, ::mlir::Value indices, mlir::IntegerAttr axis) {
    build(odsBuilder, odsState, input, indices, axis, /*multiClusterStrategy*/ nullptr);
}

mlir::LogicalResult vpux::VPU::GatherElementsOp::inferReturnTypes(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange /*regions*/,
        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::GatherElementsOpAdaptor gatherElements(operands, attrs, prop);
    if (mlir::failed(gatherElements.verify(loc))) {
        return mlir::failure();
    }

    const auto inIndicesType = mlir::cast<vpux::NDTypeInterface>(gatherElements.getIndices().getType());
    const auto inInputType = mlir::cast<vpux::NDTypeInterface>(gatherElements.getInput().getType());

    const auto outType = inIndicesType.changeElemType(inInputType.getElementType());
    inferredReturnTypes.push_back(outType);

    return mlir::success();
}

//
// TilingBuilderOpInterface
//

vpux::InputTiling vpux::VPU::GatherElementsOp::backInferTileInfo(const vpux::TileInfo& outputTile, vpux::Logger log) {
    const auto origInputShape = getShape(getInput());
    const auto origIndicesShape = getShape(getIndices());

    return vpux::backInferGatherElementsTile(outputTile, origInputShape, origIndicesShape, getAxis(),
                                             origInputShape.size(), log);
}

void vpux::VPU::GatherElementsOp::adjustAttrs(const TilingInfo& /*inputTiling*/, const TileInfo& /*outputTile*/) {
}

mlir::FailureOr<OutputTiling> vpux::VPU::GatherElementsOp::getTilingStrategy(TilingMode tilingMode, Logger log) {
    auto baseOp = this->getOperation();
    VPUX_THROW_WHEN(tilingMode != TilingMode::ISOLATED,
                    "Only supporting isolated tiling for GatherElements currently, for op {0} at '{1}'",
                    baseOp->getName(), getLoc());
    const auto axis = getAxis();

    const auto outputType = mlir::cast<vpux::NDTypeInterface>(baseOp->getResult(0).getType());
    const auto outputShape = outputType.getShape();

    DimArr tileDimOrder;

    auto dimOrder = DimsOrder::fromValue(getOutput());
    for (auto idx : irange(dimOrder.numDims())) {
        auto dim = dimOrder.dimAt(idx);
        if (dim != Dim(axis)) {
            tileDimOrder.push_back(dim);
        }
    }

    auto nTilesOnDimforGather = VPU::getSupportedNTilesOnDimforGatherElements(tileDimOrder, baseOp, tilingMode, log);

    log.trace("Isolated tiling strategy: {0}", nTilesOnDimforGather);
    return fillDividedTiles(baseOp, nTilesOnDimforGather, outputShape);
}

//
// ClusteredOpInterface
//

bool vpux::VPU::GatherElementsOp::checkStrategyCompatibility(VPU::MultiClusterStrategy strategy, size_t) {
    const auto is4DInputs = std::all_of(getOperands().begin(), getOperands().end(), [](auto input) {
        return getShape(input).size() == 4;
    });
    const auto is4DOutput = (getShape(getOutput()).size() == 4);
    if (!is4DInputs || !is4DOutput) {
        return false;
    }

    // Check the shape whether follows the rule below:
    // 1.shape meets [1, DataBeforeAxisRange, AxisRange, DataAfterAxisRange],
    // 2 axis = 2
    const auto axis = getAxis();
    if (axis != 2) {
        return false;
    }
    return strategy == VPU::MultiClusterStrategy::Clustering ||
           strategy == VPU::MultiClusterStrategy::SplitOverKernel ||
           strategy == VPU::MultiClusterStrategy::SplitOverWidth;
}

vpux::VPU::DistributionInfo vpux::VPU::GatherElementsOp::getExplicitDistributionInfoAttr(
        vpux::ShapeRef shape, vpux::VPU::DistributionMode distributionMode, ArrayRef<int64_t> numTiles,
        const int64_t numClusters, ArrayRef<int64_t> alignment, const bool uniformDistributedSegments,
        const vpux::VPU::OverlapDistributionParams& overlapParams) {
    return VPU::getSWExplicitDistributionInfo(mlir::dyn_cast<VPU::SWOpInterface>(getOperation()), shape,
                                              distributionMode, numTiles, numClusters, alignment,
                                              uniformDistributedSegments, overlapParams);
}

//
// SWOpInterface
//

bool vpux::VPU::GatherElementsOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem) {
    VPUX_THROW_UNLESS(buffers.size() == 3, "GatherOp requires 2 inputs and 1 outputs, but the number of buffer is {0}",
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

bool vpux::VPU::GatherElementsOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers) {
    return fitIntoCMX(buffers, Byte(0));
}

bool vpux::VPU::GatherElementsOp::supportCycleCostCalculation() {
    return false;
}
