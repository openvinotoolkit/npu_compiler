//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops.hpp"

#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"
#include "vpux/compiler/utils/attributes_utils.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::TopKOp::inferReturnTypes(mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc,
                                                        mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                        mlir::OpaqueProperties prop, mlir::RegionRange /*regions*/,
                                                        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::TopKOpAdaptor topK(operands, attrs, prop);
    if (mlir::failed(topK.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = topK.getInput().getType().cast<vpux::NDTypeInterface>();
    const auto inputShape = inType.getShape().raw();

    const auto kValue = getConstOrAttrValue(topK.getK(), topK.getKValueAttr());

    if (mlir::failed(kValue)) {
        return mlir::failure();
    }

    SmallVector<int64_t> outShape;
    for (size_t i = 0; i < inputShape.size(); ++i) {
        outShape.push_back(inputShape[i]);
    }
    int64_t axis = topK.getAxis();
    const auto inRank = inType.getRank();
    if (axis < 0) {
        axis += inRank;
    }
    outShape[axis] = kValue.value();

    auto outputType = mlir::RankedTensorType::get(outShape, inType.getElementType(), createTensorAttrFromType(inType));

    inferredReturnTypes.push_back(outputType);

    auto outType1 = mlir::RankedTensorType::get(outShape, topK.getElementType(), createTensorAttrFromType(inType));
    inferredReturnTypes.push_back(outType1);

    return mlir::success();
}

//
// TilingBuilderOpInterface
//

InputTiling vpux::VPU::TopKOp::backInferTileInfo(const vpux::TileInfo& outputTile, vpux::Logger) {
    SmallVector<TileInfo> inputTiles;
    auto curTile = outputTile;
    const auto inShape = getShape(getInput());
    const auto kAxis = Dim(getAxis());
    curTile.shape[kAxis] = inShape[kAxis];
    inputTiles.push_back(curTile);

    if (getK()) {
        const auto kShape = getShape(getK());
        auto kTile = TileInfo(kShape);
        inputTiles.push_back(kTile);
    }

    if (getLineBuffer()) {
        const auto topKBufferShape = getShape(getLineBuffer());
        auto topKBufferTile = TileInfo(topKBufferShape);
        inputTiles.push_back(topKBufferTile);
    }

    return TilingInfo{inputTiles};
}

vpux::OutputTiling vpux::VPU::TopKOp::getOutputTiling(const vpux::TileInfo& firstOutputTile, vpux::Logger /*log*/) {
    return OutputTiling{firstOutputTile, firstOutputTile};
}

void vpux::VPU::TopKOp::adjustAttrs(const TilingInfo& /*inputTiling*/, const TileInfo& /*outputTile*/) {
    // Do nothing
}

mlir::FailureOr<OutputTiling> vpux::VPU::TopKOp::getTilingStrategy(TilingMode tilingMode, Logger log) {
    auto baseOp = this->getOperation();
    VPUX_THROW_WHEN(tilingMode != TilingMode::ISOLATED,
                    "Only supporting isolated tiling for TopK currently, for op {0} at '{1}'", baseOp->getName(),
                    getLoc());
    auto axis = this->getAxis();
    auto tileDim = 0;
    auto tilingInfo = mlir::dyn_cast<VPU::TilingInfoOpInterface>(baseOp);
    const auto outputType = baseOp->getResult(0).getType().cast<vpux::NDTypeInterface>();
    const auto outputShape = outputType.getShape();
    Shape nTilesOnDim(outputShape.size(), 1);
    const auto isSupportedTileSize = [baseOp, &tilingInfo, outputShape, log](ShapeRef nTilesOnDim,
                                                                             TilingMode tilingMode) -> bool {
        const auto tiles = fillDividedTiles(baseOp, nTilesOnDim, outputShape);
        if (mlir::failed(tiles)) {
            return false;
        }
        return tilingInfo.isSupportedTiling(tiles.value(), tilingMode, log);
    };

    while (!isSupportedTileSize(nTilesOnDim, tilingMode)) {
        VPUX_THROW_WHEN(tileDim >= static_cast<int>(outputShape.size()), "Failed to tile {0} at '{1}'",
                        baseOp->getName(), baseOp->getLoc());

        if (tileDim == axis) {
            ++tileDim;
        } else {
            if (nTilesOnDim[Dim(tileDim)] >= outputShape[Dim(tileDim)]) {
                ++tileDim;
            } else {
                ++nTilesOnDim[Dim(tileDim)];
            }
        }
    }

    log.trace("Isolated tiling strategy: {0}", nTilesOnDim);
    auto origTiles = fillDividedTiles(baseOp, nTilesOnDim, outputShape);
    return origTiles;
}

//
// ClusteredOpInterface
//

bool vpux::VPU::TopKOp::checkStrategyCompatibility(VPU::MultiClusterStrategy strategy, size_t) {
    const auto inputType = getInput().getType().cast<vpux::NDTypeInterface>();
    const auto inShape = inputType.getShape();
    int64_t axis = getAxisAttr().getValue().getSExtValue();

    if (getK()) {
        return false;
    }

    if (strategy == VPU::MultiClusterStrategy::Clustering) {
        return true;
    }

    // Split input/output by H dim when axis is not point to H
    if (strategy == VPU::MultiClusterStrategy::SplitOverHeight && axis != Dims4D::Act::H.ind() &&
        inShape[Dims4D::Act::H] > 1) {
        return true;
    }

    // Split input/output by C dim when axis is not point to C
    if (strategy == VPU::MultiClusterStrategy::SplitOverKernel && axis != Dims4D::Act::C.ind() &&
        inShape[Dims4D::Act::C] > 1) {
        return true;
    }

    return false;
}

vpux::VPU::DistributionInfo vpux::VPU::TopKOp::getExplicitDistributionInfoAttr(
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

bool vpux::VPU::TopKOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem) {
    SmallVector<Byte> buffersSize;
    std::transform(buffers.begin(), buffers.end(), std::back_inserter(buffersSize), [](const auto buffer) {
        return buffer.getTotalAllocSize();
    });

    auto totalAvailableCMXSize = reservedMem.count() == 0 ? getTotalCMXSize(getOperation()).count()
                                                          : getTotalCMXFragmentationAwareSize(getOperation()).count();

    return vpux::VPU::calculateAlignedBuffersMemoryRequirement(getArch(getOperation()), buffersSize).count() +
                   reservedMem.count() <=
           totalAvailableCMXSize;
}

bool vpux::VPU::TopKOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers) {
    return fitIntoCMX(buffers, Byte(0));
}

bool vpux::VPU::TopKOp::supportCycleCostCalculation() {
    return false;
}
