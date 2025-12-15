//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/tiling.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/image.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::YuvToRgbOp::inferReturnTypes(mlir::MLIRContext* ctx,
                                                            std::optional<mlir::Location> optLoc,
                                                            mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                            mlir::OpaqueProperties prop, mlir::RegionRange /*regions*/,
                                                            mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::YuvToRgbOpAdaptor colorConv(operands, attrs, prop);
    if (mlir::failed(colorConv.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = mlir::cast<vpux::NDTypeInterface>(colorConv.getInput1().getType());
    const auto shape = inType.getShape().raw();
    if (shape[3] != 1) {
        return errorAt(loc, "Incorrect input shape format: '{0}'", shape);
    }

    SmallVector<int64_t> outShape{shape[0], shape[1], shape[2], 3};

    if (colorConv.getInput2() == nullptr) {
        VPUX_THROW_UNLESS(colorConv.getInput3() == nullptr, "1xPlane config error");
        VPUX_THROW_UNLESS(((outShape[1] * 2) % 3) == 0, "Invalid height");
        outShape[1] = outShape[1] * 2 / 3;
    }

    const auto outType =
            mlir::RankedTensorType::get(outShape, inType.getElementType(), createTensorAttrFromType(inType));
    inferredReturnTypes.push_back(outType);

    return mlir::success();
}

//
// TilingBuilderOpInterface
//

vpux::InputTiling vpux::VPU::YuvToRgbOp::backInferTileInfo(const vpux::TileInfo& outputTile, vpux::Logger) {
    auto H = Dim(1), W = Dim(2), C = Dim(3);  // N = Dim(0)

    VPUX_THROW_UNLESS(outputTile.shape[H] % 2 == 0 && outputTile.shape[W] % 2 == 0,
                      "Invalid YuvToRgbOp outputTile, output C,H channels are not even");
    auto singlePlane = (getInput2() == nullptr);
    if (!singlePlane) {
        if (getInFmt() == IE::ColorFmt::NV12) {
            TileInfo input1Tile = outputTile;
            TileInfo input2Tile = outputTile;

            input1Tile.shape[C] = 1;
            input2Tile.shape[C] = 2;
            input2Tile.shape[H] = outputTile.shape[H] / 2;
            input2Tile.shape[W] = outputTile.shape[W] / 2;
            input2Tile.offsets[H] = outputTile.offsets[H] / 2;
            input2Tile.offsets[W] = outputTile.offsets[W] / 2;

            return TilingInfo{{input1Tile, input2Tile}};
        } else {
            TileInfo input1Tile = outputTile;
            TileInfo input2Tile = outputTile;
            TileInfo input3Tile = outputTile;

            input1Tile.shape[C] = 1;
            input2Tile.shape[C] = 1;
            input2Tile.shape[H] = outputTile.shape[H] / 2;
            input2Tile.shape[W] = outputTile.shape[W] / 2;
            input2Tile.offsets[H] = outputTile.offsets[H] / 2;
            input2Tile.offsets[W] = outputTile.offsets[W] / 2;

            input3Tile.shape[C] = 1;
            input3Tile.shape[H] = outputTile.shape[H] / 2;
            input3Tile.shape[W] = outputTile.shape[W] / 2;
            input3Tile.offsets[H] = outputTile.offsets[H] / 2;
            input3Tile.offsets[W] = outputTile.offsets[W] / 2;

            return TilingInfo{{input1Tile, input2Tile, input3Tile}};
        }
    } else {
        TileInfo input1Tile(getShape(getInput1()));
        input1Tile = outputTile;

        input1Tile.shape[C] = 1;
        input1Tile.shape[H] = outputTile.shape[H] / 2 * 3;

        return TilingInfo{input1Tile};
    }
}

void vpux::VPU::YuvToRgbOp::adjustAttrs(const TilingInfo& /*inputTiling*/, const TileInfo& /*outputTile*/) {
}

mlir::FailureOr<OutputTiling> vpux::VPU::YuvToRgbOp::getTilingStrategy(TilingMode tilingMode, Logger log) {
    auto op = this->getOperation();
    VPUX_THROW_WHEN(tilingMode != TilingMode::ISOLATED,
                    "Only supporting isolated tiling for YuvToRgbOp currently, for op {0} at '{1}'", op->getName(),
                    getLoc());

    auto tilingInfo = mlir::dyn_cast<VPU::TilingInfoOpInterface>(op);
    const auto outputType = mlir::cast<vpux::NDTypeInterface>(op->getResult(0).getType());
    const auto outputShape = outputType.getShape();
    Shape nTilesOnDimforYuv2RGB(outputShape.size(), 1);

    const auto isSupportedTileSize = [op, &tilingInfo, outputShape, log](ShapeRef nTilesOnDim,
                                                                         TilingMode tilingMode) -> bool {
        const auto tiles = fillDividedTiles(op, nTilesOnDim, outputShape);
        if (mlir::failed(tiles)) {
            return false;
        }
        return tilingInfo.isSupportedTiling(tiles.value(), tilingMode, log);
    };

    while (!isSupportedTileSize(nTilesOnDimforYuv2RGB, tilingMode)) {
        if (2 * nTilesOnDimforYuv2RGB[Dims4D::Act::C] < outputShape[Dims4D::Act::C]) {
            nTilesOnDimforYuv2RGB[Dims4D::Act::C]++;
            continue;
        }

        if (2 * nTilesOnDimforYuv2RGB[Dims4D::Act::H] < outputShape[Dims4D::Act::H]) {
            nTilesOnDimforYuv2RGB[Dims4D::Act::H]++;
            continue;
        }

        VPUX_THROW("Operation 'Yuv2RGB' cannot be tiled");
    }

    return vpux::fillDividedTiles(op, nTilesOnDimforYuv2RGB, outputShape);
}

//
// ClusteredOpInterface
//

// For SoK and SoH we make sure the UV plane (NV12) or U/V planes (I420) can be split up more if needed,
// since the input planes are half the size the output dimensions.
// This helps us tile even small tensors and keep things parallel.
bool vpux::VPU::YuvToRgbOp::checkStrategyCompatibility(VPU::MultiClusterStrategy strategy, size_t numTiles) {
    const auto outputShape = getShape(getOutput());
    // singlePlane MC disabled
    if (getInput2() == nullptr) {
        return false;
    }

    if (strategy == VPU::MultiClusterStrategy::Clustering) {
        return true;
    }

    if (strategy == VPU::MultiClusterStrategy::SplitOverKernel) {
        return outputShape[Dims4D::Act::C] / 2 >= checked_cast<int64_t>(numTiles) * 2;
    }

    if (strategy == VPU::MultiClusterStrategy::SplitOverHeight) {
        return outputShape[Dims4D::Act::H] / 2 >= checked_cast<int64_t>(numTiles) * 2;
    }

    return false;
}

vpux::VPU::DistributionInfo vpux::VPU::YuvToRgbOp::getExplicitDistributionInfoAttr(
        vpux::ShapeRef shape, vpux::VPU::DistributionMode distributionMode, ArrayRef<int64_t> numTiles,
        const int64_t numClusters, ArrayRef<int64_t> /* alignment */, const bool uniformDistributedSegments,
        const vpux::VPU::OverlapDistributionParams& overlapParams) {
    SmallVector<int64_t> yuvAlignment(shape.size(), 1);

    // Set alignment to 2 only on the dimension that is being tiled AND is even
    for (size_t i = 0; i < numTiles.size() && i < yuvAlignment.size(); ++i) {
        if (numTiles[i] > 1 && shape[Dim(i)] % 2 == 0) {
            yuvAlignment[i] = 2;
        }
    }

    return VPU::getSWExplicitDistributionInfo(mlir::cast<VPU::SWOpInterface>(getOperation()), shape, distributionMode,
                                              numTiles, numClusters, yuvAlignment, uniformDistributedSegments,
                                              overlapParams);
}

//
// SWOpInterface
//

bool vpux::VPU::YuvToRgbOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers) {
    return fitIntoCMX(buffers, Byte(0));
}

bool vpux::VPU::YuvToRgbOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem) {
    auto singlePlane = (getInput2() == nullptr);

    if (singlePlane) {
        VPUX_THROW_UNLESS(buffers.size() == 2,
                          "YuvToRgbOp (single plane) requires 1 input and 1 output, but the number of buffers is {0}",
                          buffers.size());
    } else {
        if (getInFmt() == IE::ColorFmt::NV12) {
            VPUX_THROW_UNLESS(buffers.size() == 3,
                              "YuvToRgbOp (NV12) requires 2 inputs and 1 output, but the number of buffers is {0}",
                              buffers.size());
        } else {
            VPUX_THROW_UNLESS(buffers.size() == 4,
                              "YuvToRgbOp (I420) requires 3 inputs and 1 output, but the number of buffers is {0}",
                              buffers.size());
        }
    }

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

bool vpux::VPU::YuvToRgbOp::supportCycleCostCalculation() {
    return false;
}

//
// build
//

void vpux::VPU::YuvToRgbOp::build(::mlir::OpBuilder& builder, ::mlir::OperationState& state, ::mlir::Value input1,
                                  vpux::IE::ColorFmtAttr inFmt, vpux::IE::ColorFmtAttr outFmt) {
    build(builder, state, input1, nullptr, nullptr, inFmt, outFmt, {});
}

void vpux::VPU::YuvToRgbOp::build(::mlir::OpBuilder& builder, ::mlir::OperationState& state, ::mlir::Value input1,
                                  ::mlir::Value input2, vpux::IE::ColorFmtAttr inFmt, vpux::IE::ColorFmtAttr outFmt) {
    build(builder, state, input1, input2, nullptr, inFmt, outFmt, {});
}

void vpux::VPU::YuvToRgbOp::build(::mlir::OpBuilder& builder, ::mlir::OperationState& state, ::mlir::Value input1,
                                  ::mlir::Value input2, ::mlir::Value input3, vpux::IE::ColorFmtAttr inFmt,
                                  vpux::IE::ColorFmtAttr outFmt) {
    build(builder, state, input1, input2, input3, inFmt, outFmt, {});
}
