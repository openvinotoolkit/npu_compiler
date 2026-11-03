//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/core/tiling.hpp"
#include "vpux/compiler/dialect/IE/utils/dynamic_shape_utils.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/image.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/dialect/core/types.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/infer_output_shape.hpp"
#include "vpux/utils/core/numeric.hpp"

using namespace vpux;
using namespace Dims4D::Act;

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

    const auto input = colorConv.getInput1();
    const auto inType = mlir::cast<NDTypeInterface>(input.getType());
    const auto shape = inType.getShape();

    // YuvToRgb follows NV12 docs semantics for 2-plane input:
    //   Y  must be [N, H,   W,   1]
    //   UV must be [N, H/2, W/2, 2]
    // with output [N, H, W, 3].
    //
    // In this compiler, YuvToRgb follows the common internal convention: tensor shapes
    // are interpreted in logical NCHW form. External NV12/I420 docs often describe these
    // tensors in NHWC terms, but inside the compiler semantic channel is always C and
    // semantic height is always H. For a combined single-plane input (Y and UV stacked
    // along semantic height), that stacked height shrinks by 2/3 to recover luma height.

    if (shape[C] != 1) {
        return errorAt(loc, "Incorrect input shape format. Expected Y input to have channels == '1', got '{0}'", shape);
    }

    auto [outStaticShape, outBounds, outDimMask] = callOnShapeOf(inType, [&](const auto& inShape) {
        auto outShape = copyShape(inShape);
        outShape[C] = 3;

        if (colorConv.getInput2() == nullptr) {
            outShape[H] = inShape[H] * 2 / 3;
        }

        return splitShapeAndRepresentation(outShape);
    });

    auto outDesc = vpux::getTensorAttr(ctx, DimsOrder::fromValue(input), inType.getMemSpace(), outBounds, outDimMask);

    const auto outType = mlir::RankedTensorType::get(outStaticShape.raw(), inType.getElementType(), outDesc);
    inferredReturnTypes.push_back(outType);

    return mlir::success();
}

//
// TilingBuilderOpInterface
//

vpux::InputTiling vpux::VPU::YuvToRgbOp::backInferTileInfo(const vpux::TileInfo& outputTile, vpux::Logger) {
    VPUX_THROW_UNLESS(outputTile.shape[H] % 2 == 0 && outputTile.shape[W] % 2 == 0,
                      "Invalid YuvToRgbOp outputTile, output H, W dimensions are not even");
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
    const auto outputBoundedShape = getBoundedShape(outputType);
    Shape tilingStrategy(outputBoundedShape.size(), 1);

    const auto isSupportedTileSize = [op, &tilingInfo, outputBoundedShape, log](ShapeRef nTilesOnDim,
                                                                                TilingMode tilingMode) -> bool {
        const auto tiles = fillDividedTiles(op, nTilesOnDim, outputBoundedShape);
        if (mlir::failed(tiles)) {
            return false;
        }
        return tilingInfo.isSupportedTiling(tiles.value(), tilingMode, log);
    };

    if (mlir::isa<Core::BoundedTensorType>(outputType)) {
        auto tilingDims = getTileDimOrder(op, tilingMode, log.nest());
        auto alignment = Shape(getAlignment(op, {}, {}));

        auto tile = Shape(outputBoundedShape);

        // HACK: force scf.for tiling by strarting with "divide by 2" on all dynamic dimensions
        // The problem is that if initially upper bounds are so small that the operation fits the CMX,
        // the getTilingStrategy would not be called and operation will not have scf.for loops.
        auto dynamicOutputShape = outputType.getShape();
        for (auto i : irange(dynamicOutputShape.size())) {
            auto dimSize = dynamicOutputShape[Dim(i)];
            if (mlir::ShapedType::isStatic(dimSize)) {
                continue;
            }

            auto dimBound = outputBoundedShape[Dim(i)];
            VPUX_THROW_UNLESS(dimBound >= (2 * alignment[Dim(i)]),
                              "Output upper bound for YuvToRgb layer is too small for tiling strategy algorithm at {0}",
                              op->getLoc());
        }

        while (!isSupportedTileSize(tilingStrategy, tilingMode)) {
            auto maxTileDim = Dim(0);
            auto canSplitTile = false;
            for (auto dimIndex : irange(tile.size())) {
                const auto dim = Dim(dimIndex);
                if (tile[dim] <= alignment[dim]) {
                    continue;
                }
                if (!canSplitTile || tile[dim] > tile[maxTileDim]) {
                    maxTileDim = dim;
                    canSplitTile = true;
                }
            }

            VPUX_THROW_UNLESS(canSplitTile, "Not enough CMX to tile '{0}' at {1}", op->getName(), op->getLoc());

            tilingStrategy[maxTileDim]++;
            auto newTileDimSize = divUp(outputBoundedShape[maxTileDim], tilingStrategy[maxTileDim]);
            tile[maxTileDim] = alignValUp(newTileDimSize, alignment[maxTileDim]);
        }
    } else {
        while (!isSupportedTileSize(tilingStrategy, tilingMode)) {
            if (2 * tilingStrategy[H] < outputBoundedShape[H]) {
                tilingStrategy[H]++;
                continue;
            }

            if (2 * tilingStrategy[W] < outputBoundedShape[W]) {
                tilingStrategy[W]++;
                continue;
            }

            VPUX_THROW("Operation 'Yuv2RGB' cannot be tiled");
        }
    }

    return vpux::fillDividedTiles(op, tilingStrategy, outputBoundedShape);
}

//
// ClusteredOpInterface
//

// For SoK and SoH we make sure the UV plane (NV12) or U/V planes (I420) can be split up more if needed,
// since the input planes are half the size the output dimensions.
// This helps us tile even small tensors and keep things parallel.
bool vpux::VPU::YuvToRgbOp::checkStrategyCompatibility(VPU::MultiClusterStrategy strategy, size_t numTiles) {
    const auto outputShape = getBoundedShape(getOutput());

    // singlePlane MC disabled
    if (getInput2() == nullptr) {
        return false;
    }

    if (strategy == VPU::MultiClusterStrategy::Clustering) {
        return true;
    }

    if (strategy == VPU::MultiClusterStrategy::SplitOverKernel) {
        return outputShape[C] / 2 >= checked_cast<int64_t>(numTiles) * 2;
    }

    if (strategy == VPU::MultiClusterStrategy::SplitOverHeight) {
        return outputShape[H] / 2 >= checked_cast<int64_t>(numTiles) * 2;
    }

    return false;
}

vpux::VPU::DistributionInfo vpux::VPU::YuvToRgbOp::getExplicitDistributionInfoAttr(
        vpux::ShapeRef shape, vpux::VPU::DistributionMode distributionMode, ArrayRef<int64_t> numTiles,
        const int64_t numClusters, ArrayRef<int64_t> /* alignment */, const bool uniformDistributedSegments,
        const vpux::VPU::OverlapDistributionParams& overlapParams,
        const std::optional<ArrayRef<int64_t>> /* memoryNumTiles */) {
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
// ReifyRankedShapedTypeOpInterface
//

mlir::LogicalResult vpux::VPU::YuvToRgbOp::reifyResultShapes(mlir::OpBuilder& builder,
                                                             mlir::ReifiedRankedShapedTypeDims& reifiedReturnShapes) {
    return reifyYuvToRgbTensors(getOperation(), builder, reifiedReturnShapes);
}

//
// build
//

void vpux::VPU::YuvToRgbOp::build(::mlir::OpBuilder& builder, ::mlir::OperationState& state, ::mlir::Value input1,
                                  vpux::IE::ColorFmtAttr inFmt, vpux::IE::ColorFmtAttr outFmt,
                                  ::mlir::FloatAttr scale) {
    build(builder, state, input1, nullptr, nullptr, inFmt, outFmt, scale, {});
}

void vpux::VPU::YuvToRgbOp::build(::mlir::OpBuilder& builder, ::mlir::OperationState& state, ::mlir::Value input1,
                                  ::mlir::Value input2, vpux::IE::ColorFmtAttr inFmt, vpux::IE::ColorFmtAttr outFmt,
                                  ::mlir::FloatAttr scale) {
    build(builder, state, input1, input2, nullptr, inFmt, outFmt, scale, {});
}

void vpux::VPU::YuvToRgbOp::build(::mlir::OpBuilder& builder, ::mlir::OperationState& state, ::mlir::Value input1,
                                  ::mlir::Value input2, ::mlir::Value input3, vpux::IE::ColorFmtAttr inFmt,
                                  vpux::IE::ColorFmtAttr outFmt, ::mlir::FloatAttr scale) {
    build(builder, state, input1, input2, input3, inFmt, outFmt, scale, {});
}
