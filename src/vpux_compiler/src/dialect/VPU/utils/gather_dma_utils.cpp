//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/gather_dma_utils.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"

namespace vpux::VPU {

bool isLegalConvertToGatherDMA(VPU::GatherOp op, bool isElementTile, bool isIndicesTile, vpux::Logger log) {
    log.trace("Got Gather Op at {0}.", op->getLoc());

    const auto outShape = getShape(op.getOutput());
    if (outShape.isDynamic()) {
        return false;
    }

    const auto outputType = mlir::cast<vpux::NDTypeInterface>(op.getOutput().getType());
    const auto indicesType = mlir::cast<vpux::NDTypeInterface>(op.getIndices().getType());
    const auto inputType = mlir::cast<vpux::NDTypeInterface>(op.getInput().getType());
    auto arch = config::getArch(op);

    if (!op.getAxisValue().has_value()) {
        return false;
    }

    // For GatherDMA all dimensions before axis dimension must be 1
    size_t axis = op.getAxisValue().value();
    const auto inputShape = inputType.getShape();

    for (size_t idx = 0; idx < axis; ++idx) {
        if (inputShape[vpux::Dim(idx)] != 1) {
            return false;
        }
    }

    const size_t numberOfIndices = indicesType.getNumElements();

    const size_t DMA_MAX_INDICES_LIST_LENGTH_ARCH_BASED = VPU::getGatherDMAMaxIndicesListLength(arch);

    if (numberOfIndices > DMA_MAX_INDICES_LIST_LENGTH_ARCH_BASED) {
        return isIndicesTile;
    }

    const Bit elemOutSize = vpux::getElemTypeSize(outputType);
    const size_t dma_element_size_in_bit =
            (outputType.getNumElements() / indicesType.getNumElements()) * elemOutSize.count();

    const size_t GATHER_DMA_MAX_ELEMENT_SIZE_ARCH_BASED = VPU::getGatherDMAMaxElementSize(arch);

    if (dma_element_size_in_bit > GATHER_DMA_MAX_ELEMENT_SIZE_ARCH_BASED * CHAR_BIT) {
        return isElementTile;
    }

    return (!isElementTile) && (!isIndicesTile);
}

Shape getSupportedNTilesOnDimforGather(ArrayRef<int64_t> tileDimOrder, mlir::Operation* baseOp, TilingMode tilingMode,
                                       Logger log) {
    const auto outputType = mlir::cast<vpux::NDTypeInterface>(baseOp->getResult(0).getType());
    const auto outputShape = outputType.getShape();
    Shape nTilesOnDimforGather(outputShape.size(), 1);
    auto tilingInfo = mlir::dyn_cast<VPU::TilingInfoOpInterface>(baseOp);
    VPUX_THROW_WHEN(tilingInfo == nullptr, "Operation '{0}' doesn't implement TilingInfoOpInterface",
                    baseOp->getName());

    const auto isSupportedTileSize = [baseOp, &tilingInfo, outputShape, log](ShapeRef nTilesOnDim,
                                                                             TilingMode tilingMode) -> bool {
        const auto tiles = fillDividedTiles(baseOp, nTilesOnDim, outputShape);
        if (mlir::failed(tiles)) {
            return false;
        }
        return tilingInfo.isSupportedTiling(tiles.value(), tilingMode, log);
    };

    auto tileDimIter = tileDimOrder.begin();
    auto dimToTile = *tileDimIter;

    while (tileDimIter < tileDimOrder.end() && !isSupportedTileSize(nTilesOnDimforGather, tilingMode)) {
        if (nTilesOnDimforGather[Dim(dimToTile)] >= outputShape[Dim(dimToTile)]) {
            dimToTile = *(++tileDimIter);
        } else {
            ++nTilesOnDimforGather[Dim(dimToTile)];
        }
    }

    return nTilesOnDimforGather;
}

Shape getSupportedNTilesOnDimforGatherElements(DimArrRef tileDimOrder, mlir::Operation* baseOp, TilingMode tilingMode,
                                               Logger log) {
    const auto outputType = mlir::cast<vpux::NDTypeInterface>(baseOp->getResult(0).getType());
    const auto outputShape = outputType.getShape();
    Shape nTilesOnDimforGatherElements(outputShape.size(), 1);
    auto tilingInfo = mlir::dyn_cast<VPU::TilingInfoOpInterface>(baseOp);
    VPUX_THROW_WHEN(tilingInfo == nullptr, "Operation '{0}' doesn't implement TilingInfoOpInterface",
                    baseOp->getName());

    const auto isSupportedTileSize = [baseOp, &tilingInfo, outputShape, log](ShapeRef nTilesOnDim,
                                                                             TilingMode tilingMode) -> bool {
        const auto tiles = fillDividedTiles(baseOp, nTilesOnDim, outputShape);
        if (mlir::failed(tiles)) {
            return false;
        }
        return tilingInfo.isSupportedTiling(tiles.value(), tilingMode, log);
    };

    auto tileDimIter = tileDimOrder.begin();
    auto dimToTile = *tileDimIter;

    while (tileDimIter < tileDimOrder.end() && !isSupportedTileSize(nTilesOnDimforGatherElements, tilingMode)) {
        if (nTilesOnDimforGatherElements[Dim(dimToTile)] >= outputShape[Dim(dimToTile)]) {
            dimToTile = *(++tileDimIter);
        } else {
            ++nTilesOnDimforGatherElements[Dim(dimToTile)];
        }
    }

    return nTilesOnDimforGatherElements;
}

}  // namespace vpux::VPU
