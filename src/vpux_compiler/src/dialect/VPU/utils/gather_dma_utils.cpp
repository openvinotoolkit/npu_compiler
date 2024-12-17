//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/utils/gather_dma_utils.hpp"

namespace vpux::VPU {

bool isLegalConvertToGatherDMA(VPU::GatherOp op, bool isElementTile, bool isIndicesTile, vpux::Logger log) {
    log.trace("Got Gather Op at {0}.", op->getLoc());

    const auto outputType = op.getOutput().getType().cast<vpux::NDTypeInterface>();
    const auto indicesType = op.getIndices().getType().cast<vpux::NDTypeInterface>();
    const auto inputType = op.getInput().getType().cast<vpux::NDTypeInterface>();

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
    if (numberOfIndices > VPUIP::arch40xx::DMA_MAX_INDICES_LIST_LENGTH) {
        return isIndicesTile;
    }
    const Bit elemOutSize = vpux::getElemTypeSize(outputType);
    const size_t dma_element_size =
            (outputType.getNumElements() / indicesType.getNumElements()) * elemOutSize.to<Byte>().count();

    if (dma_element_size > VPUIP::arch40xx::GATHER_DMA_MAX_ELEMENT_SIZE) {
        return isElementTile;
    }

    return (!isElementTile) && (!isIndicesTile);
}

Shape getSupportedNTilesOnDimforGather(ArrayRef<int64_t> tileDimOrder, mlir::Operation* baseOp, TilingMode tilingMode,
                                       Logger log) {
    const auto outputType = baseOp->getResult(0).getType().cast<vpux::NDTypeInterface>();
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

}  // namespace vpux::VPU