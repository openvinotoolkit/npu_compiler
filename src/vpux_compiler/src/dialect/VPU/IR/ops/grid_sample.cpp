//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops/image.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::GridSampleOp::inferReturnTypes(mlir::MLIRContext* ctx,
                                                              std::optional<mlir::Location> optLoc,
                                                              mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                              mlir::OpaqueProperties prop, mlir::RegionRange,
                                                              mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::GridSampleOpAdaptor gridSample(operands, attrs, prop);

    if (mlir::failed(gridSample.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = mlir::cast<vpux::NDTypeInterface>(gridSample.getInput().getType());
    const auto inputShape = inType.getShape().raw();

    const auto gridType = mlir::cast<vpux::NDTypeInterface>(gridSample.getGrid().getType());
    const auto gridShape = gridType.getShape().raw();

    SmallVector<int64_t> outShape = {inputShape[0], inputShape[1], gridShape[1], gridShape[2]};

    auto outType = mlir::RankedTensorType::get(outShape, inType.getElementType(), createTensorAttrFromType(inType));
    inferredReturnTypes.push_back(outType);

    return mlir::success();
}

//
// TilingBuilderOpInterface
//

vpux::InputTiling vpux::VPU::GridSampleOp::backInferTileInfo(const vpux::TileInfo& outputTile, vpux::Logger) {
    const auto origInputShape = getShape(getInput());
    const auto origGridShape = getShape(getGrid());

    TileInfo inputTile(origInputShape);
    TileInfo gridTile(origGridShape);

    inputTile.shape[Dims4D::Act::N] = outputTile.shape[Dims4D::Act::N];
    inputTile.shape[Dims4D::Act::C] = outputTile.shape[Dims4D::Act::C];

    inputTile.offsets[Dims4D::Act::N] = outputTile.offsets[Dims4D::Act::N];
    inputTile.offsets[Dims4D::Act::C] = outputTile.offsets[Dims4D::Act::C];

    gridTile.shape[Dims4D::Act::N] = outputTile.shape[Dims4D::Act::N];
    gridTile.shape[Dim(1)] = outputTile.shape[Dims4D::Act::H];
    gridTile.shape[Dim(2)] = outputTile.shape[Dims4D::Act::W];
    gridTile.offsets[Dims4D::Act::N] = outputTile.offsets[Dims4D::Act::N];
    gridTile.offsets[Dim(1)] = outputTile.offsets[Dims4D::Act::H];
    gridTile.offsets[Dim(2)] = outputTile.offsets[Dims4D::Act::W];

    return InputTiling{{inputTile, gridTile}};
}

void vpux::VPU::GridSampleOp::adjustAttrs(const TilingInfo& /*inputTiling*/, const TileInfo& /*outputTile*/) {
}

mlir::FailureOr<OutputTiling> vpux::VPU::GridSampleOp::getTilingStrategy(TilingMode tilingMode, Logger log) {
    auto op = this->getOperation();
    auto tilingInfo = mlir::dyn_cast<VPU::TilingInfoOpInterface>(op);

    const auto outputType = mlir::cast<vpux::NDTypeInterface>(op->getResult(0).getType());
    const auto outputShape = outputType.getShape();

    Shape nTilesOnDimforGridSample(outputShape.size(), 1);
    tilingMode = TilingMode::ISOLATED;
    const auto tilingModeToCheck = tilingMode;

    // Prioritize outer-most dimensions
    auto tileDimPriority = outputType.getDimsOrder().toPermutation();

    auto tileDimIter = tileDimPriority.begin();
    auto dimToTile = *tileDimIter;

    const auto isSupportedTileSize = [op, &tilingInfo, outputShape, log](ShapeRef nTilesOnDim,
                                                                         TilingMode tilingMode) -> bool {
        const auto tiles = fillDividedTiles(op, nTilesOnDim, outputShape);
        if (mlir::failed(tiles)) {
            return false;
        }
        return tilingInfo.isSupportedTiling(tiles.value(), tilingMode, log);
    };

    while (!isSupportedTileSize(nTilesOnDimforGridSample, tilingModeToCheck)) {
        if (nTilesOnDimforGridSample[dimToTile] >= outputShape[dimToTile]) {
            dimToTile = *(++tileDimIter);
            if (tileDimIter == tileDimPriority.end()) {
                VPUX_THROW("Unsupported dim to tile: {0}", dimToTile);
            }
        } else {
            ++nTilesOnDimforGridSample[dimToTile];
        }
    }

    auto origTiles = fillDividedTiles(op, nTilesOnDimforGridSample, outputShape);
    return origTiles;
}

void vpux::VPU::GridSampleOp::build(::mlir::OpBuilder& odsBuilder, ::mlir::OperationState& odsState,
                                    ::mlir::Value input, ::mlir::Value grid, ::mlir::UnitAttr align_corners,
                                    vpux::IE::GridSampleModeAttr mode,
                                    vpux::IE::GridSamplePaddingModeAttr padding_mode) {
    build(odsBuilder, odsState, input, grid, align_corners, mode, padding_mode, nullptr);
}

//
// ClusteredOpInterface
//

bool vpux::VPU::GridSampleOp::checkStrategyCompatibility(VPU::MultiClusterStrategy strategy, size_t) {
    auto ddrAccessOp = mlir::dyn_cast<VPU::DDRAccessOpInterface>(getOperation());
    if (ddrAccessOp != nullptr && ddrAccessOp.isDDRAccessNecessaryOrBeneficial(Logger::global())) {
        return false;
    }

    return strategy == VPU::MultiClusterStrategy::Clustering ||
           strategy == VPU::MultiClusterStrategy::SplitOverKernel ||
           strategy == VPU::MultiClusterStrategy::SplitOverHeight ||
           strategy == VPU::MultiClusterStrategy::SplitOverWidth ||
           strategy == VPU::MultiClusterStrategy::SplitOverBatch;
}

bool vpux::VPU::GridSampleOp::isOperationSplitOverBatchCompatible(ShapeRef) {
    const auto inputShape = getShape(getInput());
    const auto gridShape = getShape(getGrid());
    const auto inputBatchSize = inputShape[Dims4D::Act::N];
    const auto gridBatchSize = gridShape[Dims4D::Act::N];

    return inputBatchSize > 1 && inputBatchSize == gridBatchSize;
}

vpux::VPU::DistributionInfo vpux::VPU::GridSampleOp::getExplicitDistributionInfoAttr(
        vpux::ShapeRef shape, vpux::VPU::DistributionMode distributionMode, ArrayRef<int64_t> numTiles,
        const int64_t numClusters, ArrayRef<int64_t> alignment, const bool uniformDistributedSegments,
        const vpux::VPU::OverlapDistributionParams& overlapParams) {
    return VPU::getSWExplicitDistributionInfo(mlir::cast<VPU::SWOpInterface>(getOperation()), shape, distributionMode,
                                              numTiles, numClusters, alignment, uniformDistributedSegments,
                                              overlapParams);
}

//
// SWOpInterface
//

bool vpux::VPU::GridSampleOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem) {
    VPUX_THROW_UNLESS(buffers.size() == 3,
                      "GridSampleOp requires 2 inputs and 1 output, but the number of buffer is {0}", buffers.size());

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

bool vpux::VPU::GridSampleOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers) {
    return fitIntoCMX(buffers, Byte(0));
}

bool vpux::VPU::GridSampleOp::supportCycleCostCalculation() {
    return false;
}
