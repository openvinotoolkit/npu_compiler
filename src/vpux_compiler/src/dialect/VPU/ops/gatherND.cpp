
//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/ops.hpp"

#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/error.hpp"

#include "vpux/utils/core/checked_cast.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::GatherNDOp::inferReturnTypes(mlir::MLIRContext* ctx,
                                                            mlir::Optional<mlir::Location> optLoc,
                                                            mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                            mlir::RegionRange /*regions*/,
                                                            mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    VPU::GatherNDOpAdaptor gatherND(operands, attrs);
    if (mlir::failed(gatherND.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = gatherND.input().getType().cast<vpux::NDTypeInterface>();
    const auto inputShape = inType.getShape().raw();
    const auto indicesShape = gatherND.indices().getType().cast<vpux::NDTypeInterface>().getShape().raw();

    const auto batchDims = gatherND.batch_dims();
    const auto lastIndices = indicesShape.back();
    const auto inputRank = static_cast<int64_t>(inputShape.size());

    SmallVector<int64_t> outShape;
    outShape.append(indicesShape.begin(), indicesShape.end() - 1);
    if (batchDims + lastIndices != inputRank) {
        outShape.append(inputShape.begin() + batchDims + lastIndices, inputShape.end());
    }

    const auto outType = inType.changeShape(Shape(outShape));
    inferredReturnTypes.emplace_back(outType);

    return mlir::success();
}

void vpux::VPU::GatherNDOp::inferLayoutInfo(mlir::Operation*, IE::LayerLayoutInfo& info) {
    IE::fillDefaultLayoutInfo(info);
}

//
// serialize
//

EMU::BlobWriter::SpecificTask vpux::VPU::GatherNDOp::serialize(EMU::BlobWriter& writer) {
    MVCNN::GatherNDParamsBuilder builder(writer);
    builder.add_batch_dims(checked_cast<uint32_t>(batch_dims()));
    const auto paramsOff = builder.Finish();
    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_GatherNDParams});
}

//
// verify
//

mlir::LogicalResult vpux::VPU::verifyOp(GatherNDOp op) {
    const auto inType = op.input().getType().cast<vpux::NDTypeInterface>();
    const auto inputShape = inType.getShape().raw();
    const auto indicesShape = op.indices().getType().cast<vpux::NDTypeInterface>().getShape().raw();

    const auto batchDims = op.batch_dims();
    const auto lastIndices = indicesShape.back();
    const auto inputRank = static_cast<int64_t>(inputShape.size());
    const auto indicesRank = static_cast<int64_t>(indicesShape.size());

    if (batchDims >= inputRank) {
        return errorAt(op, "batch_dims {0} exceeds input rank {1}", batchDims, inputRank);
    }

    if (batchDims >= indicesRank) {
        return errorAt(op, "batch_dims {0} exceeds indices rank {1}", batchDims, inputRank);
    }

    if (batchDims + lastIndices > inputRank) {
        return errorAt(op, "Slice index is out of bound");
    }

    for (size_t i = 0; i < static_cast<size_t>(batchDims); i++) {
        if (inputShape[i] != indicesShape[i]) {
            return errorAt(op, "Batch dimensions of data and indices must be the same");
        }
    }

    return mlir::success();
}

//
// TilingBuilderOpInterface
//

vpux::InputTiling vpux::VPU::GatherNDOp::backInferTileInfo(const vpux::TileInfo& outputTile, vpux::Logger) {
    const auto origInputShape = getShape(input());
    const auto origIndicesShape = getShape(indices());

    TileInfo indicesTile(origIndicesShape);

    const int64_t inputRank = origInputShape.size();
    const int64_t indicesRank = origIndicesShape.size();
    const int64_t outputRank = outputTile.shape.size();

    const auto lastIndices = origIndicesShape.back();
    const auto batchDims = batch_dims();

    TileInfo inputTile(origInputShape);

    for (int64_t i = 0; i < batchDims; i++) {
        inputTile.shape[Dim(i)] = outputTile.shape[Dim(i)];
        inputTile.offsets[Dim(i)] = outputTile.offsets[Dim(i)];
    }

    const int64_t sliceSize = inputRank - (batchDims + lastIndices);
    for (int64_t i = 0; i < sliceSize; i++) {
        inputTile.shape[Dim(inputRank - 1 - i)] = outputTile.shape[Dim(outputRank - 1 - i)];
        inputTile.offsets[Dim(inputRank - 1 - i)] = outputTile.offsets[Dim(outputRank - 1 - i)];
    }

    for (int64_t i = 0; i < indicesRank - 1; i++) {
        indicesTile.shape[Dim(i)] = outputTile.shape[Dim(i)];
        indicesTile.offsets[Dim(i)] = outputTile.offsets[Dim(i)];
    }

    return InputTiling{{inputTile, indicesTile}};
}

void vpux::VPU::GatherNDOp::adjustAttrs(const TilingInfo& /*inputTiling*/, const TileInfo& /*outputTile*/) {
}

OutputTiling vpux::VPU::GatherNDOp::getTilingStrategy(TilingMode tilingMode, Logger log) {
    auto baseOp = this->getOperation();
    VPUX_THROW_WHEN(tilingMode != TilingMode::ISOLATED,
                    "Only supporting isolated tiling for GatherND currently, for op {0} at '{1}'", baseOp->getName(),
                    getLoc());

    auto tilingInfo = mlir::dyn_cast<VPU::TilingInfoOpInterface>(baseOp);
    VPUX_THROW_WHEN(tilingInfo == nullptr, "Operation '{0}' doesn't implement TilingInfoOpInterface",
                    baseOp->getName());
    const auto outputType = baseOp->getResult(0).getType().cast<vpux::NDTypeInterface>();
    const auto outputShape = outputType.getShape();
    const auto outputRank = outputShape.size();
    Shape nTilesOnDimforGatherND(outputRank, 1);
    const auto isSupportedTileSize = [baseOp, &tilingInfo, outputShape, log](ShapeRef nTilesOnDim,
                                                                             TilingMode tilingMode) -> bool {
        const auto tiles = fillDividedTiles(baseOp, nTilesOnDim, outputShape);
        return tilingInfo.isSupportedTiling(tiles, tilingMode, log);
    };

    const auto tileDimRange = [isSupportedTileSize, &nTilesOnDimforGatherND, tilingMode, outputShape](
                                      const int64_t rangeBegin, const int64_t rangeEnd) -> void {
        auto tileDim = rangeBegin;
        while (!isSupportedTileSize(nTilesOnDimforGatherND, tilingMode)) {
            if (tileDim >= rangeEnd) {
                break;
            } else if (nTilesOnDimforGatherND[Dim(tileDim)] >= outputShape[Dim(tileDim)]) {
                ++tileDim;
            } else {
                ++nTilesOnDimforGatherND[Dim(tileDim)];
            }
        }
    };

    const auto inputType = input().getType().cast<vpux::NDTypeInterface>();
    const auto inputSize = inputType.getTotalAllocSize();

    const auto indicesType = indices().getType().cast<vpux::NDTypeInterface>();
    const auto indicesSize = indicesType.getTotalAllocSize();
    const auto indicesRank = indicesType.getShape().size();

    const auto batchDims = batch_dims();

    // The number of tiles on different dimensions impacts the total size of the data differently,
    // to reduce the number of kernel invocations, we first prioritize tiling on dimensions that reduce the most data:
    // 1) outputShape[:batchDims]               - output, input and indices sizes are reduced
    // 2) outputShape[batchDims:indicesRank-1]  - output and indices sizes are reduced
    // 3) outputShape[indicesRank-1:]           - output and input sizes are reduced

    tileDimRange(0, batchDims);
    if (inputSize > indicesSize) {
        tileDimRange(indicesRank - 1, outputRank);
        tileDimRange(batchDims, indicesRank);
    } else {
        tileDimRange(batchDims, indicesRank);
        tileDimRange(indicesRank - 1, outputRank);
    }

    VPUX_THROW_UNLESS(isSupportedTileSize(nTilesOnDimforGatherND, tilingMode), "Operation `GatherND` cannot be tiled");

    log.trace("Isolated tiling strategy: {0}", nTilesOnDimforGatherND);
    return fillDividedTiles(baseOp, nTilesOnDimforGatherND, outputShape);
}
