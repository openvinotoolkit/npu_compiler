//
// Copyright (C) 2022-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/ThreadPool.h>
#include <mlir/Support/LogicalResult.h>
#include <algorithm>
#include <cmath>
#include <optional>

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/core/tiling.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/utils/core/numeric.hpp"

#include "vpux/compiler/dialect/IE/utils/resources.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/generate_tiling.hpp"
#include "vpux/compiler/dialect/VPU/utils/multi_cluster_strategy_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/op_tiling_cache.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/convert_to_dma_utils.hpp"
#include "vpux/compiler/utils/VPU/tile_utils.hpp"

using namespace vpux;

//
// TileInfo
//

// Imagine shape [8, 8, 9] and divisor [2, 3, 2].
// We'll end up with the following shapes and offsets.

// Shapes   {[4, 3, 5], [4, 3, 5], [4, 3, 5], [4, 3, 5], [4, 2, 5], [4, 2, 5],
//           [4, 3, 4], [4, 3, 4], [4, 3, 4], [4, 3, 4], [4, 2, 4], [4, 2, 4]}
// Offsets  {[0, 0, 0], [4, 0, 0], [0, 3, 0], [4, 3, 0], [0, 6, 0], [4, 6, 0],
//           [0, 0, 5], [4, 0, 5], [0, 3, 5], [4, 3, 5], [0, 6, 0], [4, 6, 5]}

//
// Divide tiles and return the size and interval per current dimension.
//
// The size of the tiles can contain at most two distinct values.
// As for the above case, take the dimension 1 for example:
// shape 8 is divided into 3 tiles with tile size [3, 3, 2]
// we return a tuple containing two values of tile sizes (3 and 2) and the interval between them (which is 2)
//
// Returned tuple contains three values:
// tileSize - the first kind of value of divided tile sizes (3 in the example above)
// remainderTileSize - the second kind of value of divided tile sizes (2 in the example above)
// tileSizeInterval - the numbers of tiles are divided as tileSize (2 in the example above)
std::optional<std::tuple<int64_t, int64_t, size_t>> divideTileSizeAndInterval(Dim dimension, ShapeRef divisors,
                                                                              ShapeRef shape,
                                                                              ArrayRef<int64_t> alignment,
                                                                              Logger log = Logger::global()) {
    const auto shapeVal = shape[dimension];
    const auto divisorVal = divisors[dimension];
    const auto alignmentVal = alignment[dimension.ind()];

    if (shapeVal < divisorVal) {
        // Indivisible when the shape size is smaller than the divisor
        return std::nullopt;
    }

    int64_t tileSize, remainderTileSize;
    size_t tileSizeInterval;
    if (alignmentVal > 1) {
        // Whenever there is alignment, all N-1 tiles need to be multiple
        // of said align value.
        // The remainder shape is admitted to not be a mutiple of align value,
        // since this code is tasked to simply tile the original shape, not also align it.
        tileSize = alignValUp(divUp(shapeVal, divisorVal), alignmentVal);
        remainderTileSize = shapeVal - tileSize * (divisorVal - 1);

        if (remainderTileSize <= 0) {
            log.trace("DivideTiles can't meet the request: ShapeVal = {0}, divisorVal = {1}, alignmentTileSize = {2}",
                      shapeVal, divisorVal, tileSize);
            return std::nullopt;
        }

        tileSizeInterval = divisorVal - 1;

    } else {
        // When there is no alignment needed, we prefer to distribute the remainder in an
        // equal way across the first tiles.
        // For example 17 tiled 4 ways can be done as:
        // A) [5, 5, 5, 2] when we take the ceil value of the division
        // and leave the remainder as the last tile.
        // B) [5, 4, 4, 4] when we take the floor of the division and distribute
        // the remainder across the first tiles.
        // In any of the two cases, we'll have just 2 distinct values in the shape array.
        tileSize = shapeVal / divisorVal;
        remainderTileSize = shapeVal % divisorVal;

        if (remainderTileSize) {
            tileSizeInterval = remainderTileSize;
            remainderTileSize = tileSize;
            tileSize++;
        } else {
            tileSizeInterval = divisorVal;
        }
    }

    return std::make_tuple(tileSize, remainderTileSize, tileSizeInterval);
}

// Arguments:
// dividedTiles - final array of computed tiles
// divisors - array with the tile divisors for each dimension
// shape - original shape to tile
// alignment - array with alignments for each dimension
// dimensionIndex - current dimension index to be processed
// ongoingTile - individual tile solution which we construct and push to dividedTiles array
// unrollSpatialFirst - unroll in the order of NHWC if it is true
mlir::LogicalResult divideTiles(OutputTiling& dividedTiles, ShapeRef divisors, ShapeRef shape,
                                ArrayRef<int64_t> alignment, size_t dimensionIndex, vpux::TileInfo& ongoingTile,
                                bool unrollSpatialFirst) {
    // If spatial first, unroll in order of NHWC
    // else, follow the default order NCHW
    // If we got here for GNHWC then probably grouped MatMul is not efficient
    const auto spatialFirstOrder = shape.size() == DimsGroups5D::Act::numDims ? DimsOrder::GNHWC : DimsOrder::NHWC;
    const auto dimension = unrollSpatialFirst ? spatialFirstOrder.dimAt(dimensionIndex) : Dim(dimensionIndex);

    auto tileSizeIntervalResult = divideTileSizeAndInterval(dimension, divisors, shape, alignment);
    if (!tileSizeIntervalResult) {
        return mlir::failure();
    }
    int64_t tileSize = std::get<0>(*tileSizeIntervalResult);
    int64_t remainderTileSize = std::get<1>(*tileSizeIntervalResult);
    size_t tileSizeInterval = std::get<2>(*tileSizeIntervalResult);

    // Iterate and backtrack on the current list of shapes and offsets
    const size_t totalTileSize = divisors[dimension];
    int64_t tileOffset = 0;
    for (auto tileIndex : irange(totalTileSize)) {
        int64_t tileShape = tileIndex < tileSizeInterval ? tileSize : remainderTileSize;
        ongoingTile.shape[dimension] = tileShape;
        ongoingTile.offsets[dimension] = tileOffset;
        ongoingTile.axis[dimension] = totalTileSize;
        tileOffset += tileShape;

        // Full dividedTile is created so need to register the solution
        if (dimensionIndex == (divisors.size() - 1)) {
            dividedTiles.push_back(ongoingTile);
        } else {
            auto isSuccessful = divideTiles(dividedTiles, divisors, shape, alignment, dimensionIndex + 1, ongoingTile,
                                            unrollSpatialFirst);
            if (mlir::failed(isSuccessful)) {
                return mlir::failure();
            }
        }
    }

    return mlir::success();
}

mlir::LogicalResult divideTilesYuvToRgbOp(OutputTiling& dividedTiles, ShapeRef divisors, ShapeRef shape,
                                          vpux::TileInfo& ongoingTile) {
    // N C H W. Tile on C and H dimensions, minimum granularity is 2
    const auto dimC = Dim(Dims4D::Act::C);
    const auto dimH = Dim(Dims4D::Act::H);
    ongoingTile.shape[Dim(Dims4D::Act::N)] = shape[Dim(Dims4D::Act::N)];
    ongoingTile.shape[Dim(Dims4D::Act::W)] = shape[Dim(Dims4D::Act::W)];

    ongoingTile.axis[Dim(Dims4D::Act::N)] = divisors[Dim(Dims4D::Act::N)];
    ongoingTile.axis[Dim(Dims4D::Act::C)] = divisors[Dim(Dims4D::Act::C)];
    ongoingTile.axis[Dim(Dims4D::Act::H)] = divisors[Dim(Dims4D::Act::H)];
    ongoingTile.axis[Dim(Dims4D::Act::W)] = divisors[Dim(Dims4D::Act::W)];

    const auto shapeValC = shape[dimC];
    auto divisorValC = divisors[dimC];

    size_t tileSizeInitC, tileSizeC, remainderTileSizeC;

    tileSizeInitC = shapeValC / divisorValC;
    tileSizeC = tileSizeInitC + (tileSizeInitC % 2);
    divisorValC = shapeValC / tileSizeC;
    remainderTileSizeC = shapeValC % tileSizeC;

    ongoingTile.shape[dimC] = tileSizeC;
    for (int i = 0; i < divisorValC; ++i) {
        ongoingTile.offsets[dimC] = tileSizeC * i;

        const auto shapeValH = shape[dimH];
        auto divisorValH = divisors[dimH];
        size_t tileSizeInitH, tileSizeH, remainderTileSizeH;

        tileSizeInitH = shapeValH / divisorValH;
        tileSizeH = tileSizeInitH + (tileSizeInitH % 2);
        divisorValH = shapeValH / tileSizeH;
        remainderTileSizeH = shapeValH % tileSizeH;
        ongoingTile.shape[dimH] = tileSizeH;

        for (int j = 0; j < divisorValH; ++j) {
            ongoingTile.offsets[dimH] = tileSizeH * j;
            dividedTiles.push_back(ongoingTile);
        }

        if (remainderTileSizeH) {
            ongoingTile.shape[dimH] = remainderTileSizeH;
            ongoingTile.offsets[dimH] = tileSizeH * divisorValH;
            dividedTiles.push_back(ongoingTile);
        }
    }

    if (remainderTileSizeC) {
        ongoingTile.shape[dimC] = remainderTileSizeC;
        ongoingTile.offsets[dimC] = tileSizeC * divisorValC;

        const auto shapeValH = shape[dimH];
        auto divisorValH = divisors[dimH];
        size_t tileSizeInitH, tileSizeH, remainderTileSizeH;

        tileSizeInitH = shapeValH / divisorValH;
        tileSizeH = tileSizeInitH + (tileSizeInitH % 2);
        divisorValH = shapeValH / tileSizeH;
        remainderTileSizeH = shapeValH % tileSizeH;
        ongoingTile.shape[dimH] = tileSizeH;

        for (int j = 0; j < divisorValH; ++j) {
            ongoingTile.offsets[dimH] = tileSizeH * j;
            dividedTiles.push_back(ongoingTile);
        }

        if (remainderTileSizeH) {
            ongoingTile.shape[dimH] = remainderTileSizeH;
            ongoingTile.offsets[dimH] = tileSizeH * divisorValH;
            dividedTiles.push_back(ongoingTile);
        }
    }

    return mlir::success();
}

mlir::FailureOr<OutputTiling> fillDividedTilesYuvToRgbOp(ShapeRef divisors, ShapeRef shape) {
    OutputTiling dividedTiles;
    size_t totalTileNum = 1;
    for (auto divVal : divisors) {
        totalTileNum *= divVal;
    }
    dividedTiles.reserve(totalTileNum);

    auto ongoingTile = vpux::TileInfo(divisors.size());
    ongoingTile.isCompletedTile = true;

    auto isSuccessful = divideTilesYuvToRgbOp(dividedTiles, divisors, shape, ongoingTile);
    if (mlir::failed(isSuccessful)) {
        return mlir::failure();
    }

    return dividedTiles;
}

mlir::FailureOr<OutputTiling> fillDividedTilesMVN1MeanVarOp(mlir::Operation* op, ShapeRef divisors, ShapeRef shape) {
    auto mvn1MeanVarOp = mlir::dyn_cast<VPU::MVN1MeanVarOp>(op);
    VPUX_THROW_UNLESS(mvn1MeanVarOp != nullptr, "Only support MVN1MeanVarOp, but got {0}", op->getName());

    std::optional<SmallVector<int64_t>> optionalAlignment = std::nullopt;
    int64_t groupC = 1;
    if (mvn1MeanVarOp.getInternalReshape().has_value()) {
        const auto internalReshape = parseIntArrayAttr<int64_t>(mvn1MeanVarOp.getInternalReshape().value());
        const auto origShape = parseIntArrayAttr<int64_t>(mvn1MeanVarOp.getOrigShape());
        groupC = origShape[Dims4D::Act::C.ind()] / internalReshape[Dims4D::Act::C.ind()];

        auto alignment = SmallVector<int64_t>(shape.size(), 1);
        alignment[Dims4D::Act::C.ind()] = groupC;
        optionalAlignment = std::move(alignment);
    }

    return vpux::fillDividedTiles(divisors, shape, optionalAlignment, /*unrollSpatialFirst = */ false);
}

mlir::FailureOr<OutputTiling> vpux::fillDividedTiles(ShapeRef divisors, ShapeRef shape,
                                                     std::optional<ArrayRef<int64_t>> alignment,
                                                     bool unrollSpatialFirst) {
    OutputTiling dividedTiles;
    size_t totalTileNum = 1;
    for (auto divVal : divisors) {
        totalTileNum *= divVal;
    }
    dividedTiles.reserve(totalTileNum);

    auto ongoingTile = vpux::TileInfo(divisors.size());
    ongoingTile.isCompletedTile = true;

    auto alignmentShape = SmallVector<int64_t>(shape.size(), 1);
    auto alignmentShapeRef = ArrayRef(alignmentShape);
    if (alignment.has_value()) {
        alignmentShapeRef = alignment.value();
    }

    auto isSuccessful =
            divideTiles(dividedTiles, divisors, shape, alignmentShapeRef, 0, ongoingTile, unrollSpatialFirst);
    if (mlir::failed(isSuccessful)) {
        return mlir::failure();
    }

    return dividedTiles;
}

/*
 * Consider memory sharing and compare the DMA cost of different loop orders
 * Spatial first - Unroll the op in the order of NHWC
 * Weights first - Unroll the op in the order of NCHW
 * If the inputMemSize * divisors[C] > filterMemSize * divisors[H] * divisors[W], spatial first
 * else weights first
 */
bool vpux::isSpatialFirstNestedTiling(mlir::Operation* op, ShapeRef divisors) {
    auto nceOp = mlir::dyn_cast<VPU::NCEOpInterface>(op);
    if (nceOp == nullptr || nceOp.getWeightsOperand() == nullptr) {
        // Only use spatial first for operations with weights
        // The memory sharing has no difference if the op has no weights
        return false;
    }
    if (getNonOneDim(divisors).size() <= 1) {
        return false;
    }
    auto inputMemSize = getTotalSize(nceOp->getOperand(0));
    auto filterMemSize = getTotalSize(nceOp.getWeightsOperand());

    // #E152765 - generic support for GNCHW
    if (mlir::isa<VPU::NCEMatMulOp>(op)) {
        return inputMemSize * divisors[DimsGroups5D::Act::C] >
               filterMemSize * divisors[DimsGroups5D::Act::H] * divisors[DimsGroups5D::Act::W];
    }

    return inputMemSize * divisors[Dims4D::Act::C] >
           filterMemSize * divisors[Dims4D::Act::H] * divisors[Dims4D::Act::W];
}

bool vpux::isWeightsFirstNestedTiling(mlir::Operation* op, ShapeRef divisors) {
    auto nceOp = mlir::dyn_cast<VPU::NCEOpInterface>(op);
    if (nceOp == nullptr || nceOp.getWeightsOperand() == nullptr) {
        // Only use weights first for operations with weights
        // The memory sharing has no difference if the op has no weights
        return false;
    }
    if (getNonOneDim(divisors).size() <= 1) {
        return false;
    }
    auto inputMemSize = getTotalSize(nceOp->getOperand(0));
    auto filterMemSize = getTotalSize(nceOp.getWeightsOperand());
    return inputMemSize * divisors[Dims4D::Act::C] <=
           filterMemSize * divisors[Dims4D::Act::H] * divisors[Dims4D::Act::W];
}

std::optional<SmallVector<int64_t>> getAlignment(mlir::Operation* op, ShapeRef divisors, ShapeRef shape) {
    std::optional<SmallVector<int64_t>> optionalAlignment = std::nullopt;

    const auto getSubByteAlignmentFactor = [&op] {
        int64_t alignmentFactor = 1;

        auto setFactor = [&alignmentFactor](mlir::Value value) {
            const auto elemSize = vpux::getElemTypeSize(value.getType());
            if (elemSize.count() < CHAR_BIT) {
                alignmentFactor = std::max(alignmentFactor, CHAR_BIT / elemSize.count());
            }
        };

        // check all operands
        for (auto operand : op->getOperands()) {
            setFactor(operand);
        }

        // check output
        setFactor(op->getResult(0));

        return alignmentFactor;
    };

    auto alignment = SmallVector<int64_t>(shape.size(), 1);

    if (op->hasTrait<VPU::EltwiseOp>() || mlir::isa<VPU::MemPermuteOp>(op)) {
        if (auto factor = getSubByteAlignmentFactor(); factor > 1) {
            std::transform(divisors.begin(), divisors.end(), alignment.begin(), [&factor](int x) {
                return x > 1 ? factor : x;
            });
            optionalAlignment = alignment;
        }
    }

    if (mlir::isa<VPU::NCEPermuteOp>(op)) {
        const auto outputType = op->getResult(0).getType().cast<vpux::NDTypeInterface>();
        alignment[vpux::Dims4D::Act::W.ind()] = VPU::NCEInvariant::getAlignment(outputType.getElementType());
        optionalAlignment = std::move(alignment);
    } else if (auto tilingIface = mlir::dyn_cast<IE::AlignedChannelsOpInterface>(op)) {
        // #E152765 - generic support for GNCHW
        const auto C = mlir::isa<VPU::NCEMatMulOp>(op) ? DimsGroups5D::Act::C.ind() : Dims4D::Act::C.ind();
        alignment[C] = std::lcm(tilingIface.getOutputChannelAlignment(), alignment[C]);
        optionalAlignment = std::move(alignment);
    }

    return optionalAlignment;
}

mlir::FailureOr<std::optional<SmallVector<int64_t>>> calculateAlignmentLCM(ArrayRef<SmallVector<int64_t>> alignments) {
    if (alignments.empty()) {
        return std::optional<SmallVector<int64_t>>(std::nullopt);
    }

    const auto numDimensions = alignments[0].size();
    if (std::any_of(alignments.begin(), alignments.end(), [&](auto alignment) {
            return alignment.size() != numDimensions;
        })) {
        return mlir::failure();
    }

    SmallVector<int64_t> alignmentLCM(numDimensions, 1);
    for (size_t dim = 0; dim < numDimensions; ++dim) {
        for (auto alignment : alignments) {
            alignmentLCM[dim] = std::lcm(alignmentLCM[dim], alignment[dim]);
        }
    }

    return std::optional<SmallVector<int64_t>>(alignmentLCM);
}

mlir::FailureOr<OutputTiling> vpux::fillDividedTiles(mlir::Operation* op, ShapeRef divisors, ShapeRef shape) {
    if (mlir::isa<VPU::YuvToRgbOp>(op)) {
        return fillDividedTilesYuvToRgbOp(divisors, shape);
    }

    if (mlir::isa<VPU::MVN1MeanVarOp>(op)) {
        return fillDividedTilesMVN1MeanVarOp(op, divisors, shape);
    }

    std::optional<SmallVector<int64_t>> optionalAlignment = std::nullopt;
    if (auto vfOp = mlir::dyn_cast<VPU::VerticalFusionOp>(op)) {
        SmallVector<SmallVector<int64_t>> alignments;
        for (auto& innerOp : vfOp.getBody()->without_terminator()) {
            const auto outShape = getShape(innerOp.getResult(0));
            auto currentAlignment = getAlignment(&innerOp, divisors, outShape);
            if (currentAlignment.has_value()) {
                alignments.emplace_back(currentAlignment.value());
            }
        }
        auto alignmentLCMResult = calculateAlignmentLCM(alignments);
        if (mlir::failed(alignmentLCMResult)) {
            return mlir::failure();
        }
        optionalAlignment = alignmentLCMResult.value();
    } else {
        optionalAlignment = getAlignment(op, divisors, shape);
    }

    auto unrollSpatialFirst = isSpatialFirstNestedTiling(op, divisors);
    if (VPU::isWeightsDequant(op)) {
        optionalAlignment = VPU::DISTRIBUTED_N_ALIGNMENT;
    }
    return vpux::fillDividedTiles(divisors, shape, optionalAlignment, unrollSpatialFirst);
}

//
// PadInfo
//

PadInfo vpux::backInferPadsTile(const TileInfo& outputTile, ShapeRef inShape, const PadInfo& origPads,
                                ArrayRef<int64_t> kernel, ArrayRef<int64_t> strides) {
    const std::array<int64_t, Dims4D::Act::numSpatialDims> origPadsBegin = {origPads.top, origPads.left};
    const std::array<int64_t, Dims4D::Act::numSpatialDims> origPadsEnd = {origPads.bottom, origPads.right};

    SmallVector<int64_t> tilePadsBegin(Dims4D::Act::numSpatialDims);
    SmallVector<int64_t> tilePadsEnd(Dims4D::Act::numSpatialDims);

    for (auto ind : irange(Dims4D::Act::numSpatialDims)) {
        const auto spatialDim = Dims4D::Act::getSpatialDim(ind);

        const auto outSize = outputTile.shape[spatialDim];
        const auto outOffset = outputTile.offsets[spatialDim];

        const DimRange inputRange(0, inShape[spatialDim]);
        const DimRange tileRange(outOffset, outOffset + outSize);

        std::tie(std::ignore, tilePadsBegin[ind], tilePadsEnd[ind]) = inputForOutputDim(
                tileRange, kernel[ind], strides[ind], inputRange, origPadsBegin[ind], origPadsEnd[ind]);
    }

    return PadInfo(tilePadsBegin[1], tilePadsEnd[1], tilePadsBegin[0], tilePadsEnd[0]);
}

//
// Common tiling utilities
//

namespace {

struct PlaneTile final {
    DimRange width;
    DimRange height;

    int64_t area() const {
        return width.length() * height.length();
    }

    // Checks if rhs located completely in this.
    bool contains(const PlaneTile& other) const {
        return width.contains(other.width) && height.contains(other.height);
    }

    // Returns new `PlaneTile` which represents `other` as ROI of this.
    PlaneTile asROI(const PlaneTile& other) const {
        return {width.asROI(other.width), height.asROI(other.height)};
    }

    bool operator==(const PlaneTile& other) const {
        return width == other.width && height == other.height;
    }
    bool operator!=(const PlaneTile& other) const {
        return !(*this == other);
    }

    void printFormat(llvm::raw_ostream& stream) const {
        printTo(stream, "PlaneTile [width tile = {0}, height tile = {1}]", width, height);
    }
};

struct PlaneTileSolution final {
    // Input tile which meets HW requirements in terms of alignment.
    PlaneTile inputTile;

    // Padding which should be applied to input tile in order to calculate output tile.
    // Meets HW requirements in terms of size and symmetry.
    PadInfo inputPad;

    void printFormat(llvm::raw_ostream& stream) const {
        printTo(stream, "PlaneTileSolution [inputTile = {0}, inputPad = {1}]", inputTile, inputPad);
    }
};

// Return input tile and padding required to calculate the output tile.
// Padding should be applied to the input tile. It could be asymmetric, or doesn't meet HW requirements in terms of its
// size.
// * initialInputDims - Dims of the whole input tensor (not of specific tile).
// * initialPad - padding which should be applied to the whole input tensor (not to specific tile).
template <typename Dims>
std::tuple<PlaneTile, PadInfo> inputForOutputTile(const PlaneTile& output, int64_t kernelX, int64_t kernelY,
                                                  int64_t strideX, int64_t strideY, ShapeRef initialInputDims,
                                                  const PadInfo& initialPad) {
    PlaneTile inputTile = {{0, 0}, {0, 0}};
    PadInfo pad = {0, 0, 0, 0};
    std::tie(inputTile.height, pad.top, pad.bottom) = inputForOutputDim(
            output.height, kernelY, strideY, {0, initialInputDims[Dims::Act::H]}, initialPad.top, initialPad.bottom);

    std::tie(inputTile.width, pad.left, pad.right) = inputForOutputDim(
            output.width, kernelX, strideX, {0, initialInputDims[Dims::Act::W]}, initialPad.left, initialPad.right);

    return std::make_tuple(inputTile, pad);
}

template <typename Dims>
PlaneTileSolution solutionForOutputTile(const PlaneTile& output, int64_t kernelX, int64_t kernelY, int64_t strideX,
                                        int64_t strideY, ShapeRef initialInputDims, const PadInfo& initialPad) {
    PlaneTileSolution solution;
    std::tie(solution.inputTile, solution.inputPad) =
            inputForOutputTile<Dims>(output, kernelX, kernelY, strideX, strideY, initialInputDims, initialPad);

    return solution;
}

}  // namespace

// inputTile planar H/W size should keep the same with original input H/W when no tiling over those axis.
// However the back inferring size may become smaller, e.g., OutputTile 7x7, Kernel 1x1, Stride 2x2.
// The inferring inputTile planar shape is 13x13 however original planar input shape may be 14x14, which will cause
// a redundant data slice from input. Here is to restore original input planar shape to avoid extra copies.
template <typename Dims>
void restorePlanarShapeForInputTile(TileInfo& inputTile, ShapeRef origInputShape, vpux::Dim planarDim) {
    if (planarDim != Dims::Act::H && planarDim != Dims::Act::W) {
        VPUX_THROW("Invalid planar dim {0}", planarDim);
    }
    if (inputTile.shape[planarDim] > origInputShape[planarDim]) {
        VPUX_THROW("Invalid back inferring size {0} over dim {1}", inputTile.shape[planarDim], planarDim);
    }

    inputTile.shape[planarDim] = origInputShape[planarDim];
}

//
// Convolution tiling
//

InputTiling vpux::backInferConvTile(const TileInfo& outputTile, ShapeRef origInputShape, ShapeRef origFilterShape,
                                    ShapeRef origBiasShape, mlir::ArrayAttr strides, const PadInfo& origPadding) {
    PlaneTile output;
    output.height.begin = outputTile.offsets[Dims4D::Act::H];
    output.height.end = outputTile.offsets[Dims4D::Act::H] + outputTile.shape[Dims4D::Act::H];
    output.width.begin = outputTile.offsets[Dims4D::Act::W];
    output.width.end = outputTile.offsets[Dims4D::Act::W] + outputTile.shape[Dims4D::Act::W];

    const auto strideY = strides[Dims4D::Strides::Y.ind()].cast<mlir::IntegerAttr>().getValue().getSExtValue();
    const auto strideX = strides[Dims4D::Strides::X.ind()].cast<mlir::IntegerAttr>().getValue().getSExtValue();

    const auto solution = solutionForOutputTile<Dims4D>(output, origFilterShape[Dims4D::Filter::KX],
                                                        origFilterShape[Dims4D::Filter::KY], strideX, strideY,
                                                        origInputShape, origPadding);

    TileInfo inputTile(origInputShape);
    TileInfo filterTile(origFilterShape);
    TileInfo biasTile(origBiasShape);

    inputTile.shape[Dims4D::Act::N] = outputTile.shape[Dims4D::Act::N];
    inputTile.offsets[Dims4D::Act::N] = outputTile.offsets[Dims4D::Act::N];
    inputTile.axis = outputTile.axis;

    inputTile.offsets[Dims4D::Act::H] = solution.inputTile.height.begin;
    inputTile.shape[Dims4D::Act::H] = solution.inputTile.height.length();

    inputTile.offsets[Dims4D::Act::W] = solution.inputTile.width.begin;
    inputTile.shape[Dims4D::Act::W] = solution.inputTile.width.length();

    if (outputTile.isCompletedTile && outputTile.axis[Dims4D::Act::H] == 1) {
        restorePlanarShapeForInputTile<Dims4D>(inputTile, origInputShape, Dims4D::Act::H);
    }
    if (outputTile.isCompletedTile && outputTile.axis[Dims4D::Act::W] == 1) {
        restorePlanarShapeForInputTile<Dims4D>(inputTile, origInputShape, Dims4D::Act::W);
    }

    filterTile.shape[Dims4D::Filter::OC] = outputTile.shape[Dims4D::Act::C];
    filterTile.offsets[Dims4D::Filter::OC] = outputTile.offsets[Dims4D::Act::C];

    if (!biasTile.shape.empty()) {
        biasTile.shape[Dims4D::Act::C] = outputTile.shape[Dims4D::Act::C];
        biasTile.offsets[Dims4D::Act::C] = outputTile.offsets[Dims4D::Act::C];
        return TilingInfo{{inputTile, filterTile, biasTile}, solution.inputPad};
    }
    return TilingInfo{{inputTile, filterTile}, solution.inputPad};
}

InputTiling vpux::backInferGroupConvTile(const TileInfo& outputTile, ShapeRef origInputShape, ShapeRef origFilterShape,
                                         ShapeRef origBiasShape, mlir::ArrayAttr strides, const PadInfo& origPadding,
                                         int64_t groups) {
    auto res = backInferConvTile(outputTile, origInputShape, origFilterShape, origBiasShape, strides, origPadding);

    const auto inputTileIdx = 0;
    auto& inputTiles = res.tiles[inputTileIdx];

    // For GroupConv, the weights' OC dim is the product of num_group * num_channels_per_group
    const auto numOutChannelsPerGroup = origFilterShape[Dims4D::Filter::OC] / groups;

    // To correctly compute input tile when tiling is done over out channels, we need to determine
    // the start group for the tile and the number of groups it spans.
    // Based on them, we can back-infer the necessary input tile.
    // E.g. GroupConv groups = 6; in channels = 12; out channels = 18; filter = (groups * 3 out ch) x 2 in ch
    //      w/ tiling = [1, 3, 1, 1]
    // The resulting tiled GroupConvs are:
    //      Tile 0: GC w/ groups = 2 (group 0 & 1 of orig GC): out channels 0 - 5, in channels 0 - 3
    //      Tile 1: GC w/ groups = 2 (group 2 & 3 of orig GC): out channels 6 - 11, in channels 4 - 7
    //      Tile 2: GC w/ groups = 2 (group 4 & 5 of orig GC): out channels 12 - 17, in channels 8 - 11
    const auto startGroupForTile = outputTile.offsets[Dims4D::Act::C] / numOutChannelsPerGroup;
    const auto numGroupsForTile = divUp(outputTile.shape[Dims4D::Act::C], numOutChannelsPerGroup);

    inputTiles.offsets[Dims4D::Act::C] = startGroupForTile * origFilterShape[Dims4D::Filter::IC];
    inputTiles.shape[Dims4D::Act::C] = numGroupsForTile * origFilterShape[Dims4D::Filter::IC];

    return res;
}

//
// NCEMatMul tiling
//

InputTiling vpux::backInferMatMulTile(const TileInfo& outputTile, ShapeRef origInputShape, ShapeRef origFilterShape,
                                      mlir::ArrayAttr strides, const PadInfo& origPadding) {
    const auto stridesY = DimsGroups5D::Strides::Y;
    const auto stridesX = DimsGroups5D::Strides::X;

    const auto actH = DimsGroups5D::Act::H;
    const auto actW = DimsGroups5D::Act::W;
    const auto actN = DimsGroups5D::Act::N;
    const auto actC = DimsGroups5D::Act::C;
    const auto actG = DimsGroups5D::Act::G;

    const auto filterY = DimsGroups5D::Filter::KY;
    const auto filterX = DimsGroups5D::Filter::KX;
    const auto filterOC = DimsGroups5D::Filter::OC;
    const auto filterG = DimsGroups5D::Filter::G;

    PlaneTile output;
    output.height.begin = outputTile.offsets[actH];
    output.height.end = outputTile.offsets[actH] + outputTile.shape[actH];
    output.width.begin = outputTile.offsets[actW];
    output.width.end = outputTile.offsets[actW] + outputTile.shape[actW];

    const auto strideY = strides[stridesY.ind()].cast<mlir::IntegerAttr>().getValue().getSExtValue();
    const auto strideX = strides[stridesX.ind()].cast<mlir::IntegerAttr>().getValue().getSExtValue();

    const auto solution = solutionForOutputTile<DimsGroups5D>(
            output, origFilterShape[filterX], origFilterShape[filterY], strideX, strideY, origInputShape, origPadding);

    TileInfo inputTile(origInputShape);
    TileInfo filterTile(origFilterShape);

    inputTile.shape[actN] = outputTile.shape[actN];
    inputTile.offsets[actN] = outputTile.offsets[actN];
    inputTile.axis = outputTile.axis;

    inputTile.offsets[actH] = solution.inputTile.height.begin;
    inputTile.shape[actH] = solution.inputTile.height.length();

    inputTile.offsets[actW] = solution.inputTile.width.begin;
    inputTile.shape[actW] = solution.inputTile.width.length();
    inputTile.shape[actG] = outputTile.shape[actG];
    inputTile.offsets[actG] = outputTile.offsets[actG];

    if (outputTile.isCompletedTile && outputTile.axis[actH] == 1) {
        restorePlanarShapeForInputTile<DimsGroups5D>(inputTile, origInputShape, actH);
    }
    if (outputTile.isCompletedTile && outputTile.axis[actW] == 1) {
        restorePlanarShapeForInputTile<DimsGroups5D>(inputTile, origInputShape, actW);
    }

    filterTile.shape[filterOC] = outputTile.shape[actC];
    filterTile.offsets[filterOC] = outputTile.offsets[actC];
    filterTile.shape[filterOC] = outputTile.shape[actC];
    filterTile.shape[filterG] = outputTile.shape[actG];
    filterTile.offsets[filterG] = outputTile.offsets[actG];

    return TilingInfo{{std::move(inputTile), std::move(filterTile)}, solution.inputPad};
}

//
// Pooling tiling
//

InputTiling vpux::backInferPoolTile(const TileInfo& outputTile, ShapeRef origInputShape, mlir::ArrayAttr kernel_size,
                                    mlir::ArrayAttr strides, const PadInfo& origPadding) {
    PlaneTile output;
    output.height.begin = outputTile.offsets[Dims4D::Act::H];
    output.height.end = outputTile.offsets[Dims4D::Act::H] + outputTile.shape[Dims4D::Act::H];
    output.width.begin = outputTile.offsets[Dims4D::Act::W];
    output.width.end = outputTile.offsets[Dims4D::Act::W] + outputTile.shape[Dims4D::Act::W];

    const auto kernelY = kernel_size[Dims4D::Kernel::Y.ind()].cast<mlir::IntegerAttr>().getValue().getSExtValue();
    const auto kernelX = kernel_size[Dims4D::Kernel::X.ind()].cast<mlir::IntegerAttr>().getValue().getSExtValue();

    const auto strideY = strides[Dims4D::Strides::Y.ind()].cast<mlir::IntegerAttr>().getValue().getSExtValue();
    const auto strideX = strides[Dims4D::Strides::X.ind()].cast<mlir::IntegerAttr>().getValue().getSExtValue();

    const auto solution =
            solutionForOutputTile<Dims4D>(output, kernelX, kernelY, strideX, strideY, origInputShape, origPadding);

    TileInfo inputTile(origInputShape);

    inputTile.shape[Dims4D::Act::N] = outputTile.shape[Dims4D::Act::N];
    inputTile.offsets[Dims4D::Act::N] = outputTile.offsets[Dims4D::Act::N];

    inputTile.shape[Dims4D::Act::C] = outputTile.shape[Dims4D::Act::C];
    inputTile.offsets[Dims4D::Act::C] = outputTile.offsets[Dims4D::Act::C];

    inputTile.offsets[Dims4D::Act::H] = solution.inputTile.height.begin;
    inputTile.shape[Dims4D::Act::H] = solution.inputTile.height.length();

    inputTile.offsets[Dims4D::Act::W] = solution.inputTile.width.begin;
    inputTile.shape[Dims4D::Act::W] = solution.inputTile.width.length();

    if (outputTile.isCompletedTile && outputTile.axis[Dims4D::Act::H] == 1) {
        restorePlanarShapeForInputTile<Dims4D>(inputTile, origInputShape, Dims4D::Act::H);
    }
    if (outputTile.isCompletedTile && outputTile.axis[Dims4D::Act::W] == 1) {
        restorePlanarShapeForInputTile<Dims4D>(inputTile, origInputShape, Dims4D::Act::W);
    }

    return TilingInfo{{inputTile}, solution.inputPad};
}

InputTiling vpux::backInferReduceTile(const vpux::TileInfo& outputTile, ShapeRef inShape, mlir::ArrayAttr axesAttr,
                                      bool keepDims) {
    SmallVector<TileInfo> inputTiles;

    const auto axesValue = parseIntArrayAttr<int64_t>(axesAttr);
    const auto tiledOutputAxis = outputTile.axis.raw();
    const auto tiledOutputShape = outputTile.shape.raw();
    const auto tiledOutputOffsets = outputTile.offsets.raw();

    // Adding tiling case when keep dims is false and the axes are reduced from outputShape
    if (keepDims == false) {
        Shape newInput, newAxis, newOffset;
        std::copy(tiledOutputShape.begin(), tiledOutputShape.end(), std::back_inserter(newInput));
        std::copy(tiledOutputAxis.begin(), tiledOutputAxis.end(), std::back_inserter(newAxis));
        std::copy(tiledOutputOffsets.begin(), tiledOutputOffsets.end(), std::back_inserter(newOffset));

        for (auto axesInd : axesValue) {
            // Adjusting the new input based on tiled output
            newInput.insert(newInput.begin() + axesInd, inShape[Dim(axesInd)]);
            newAxis.insert(newAxis.begin() + axesInd, 1);
            newOffset.insert(newOffset.begin() + axesInd, 0);
        }

        TileInfo inTile(newInput, newOffset, newAxis);

        return TilingInfo{{std::move(inTile)}};
    }

    auto inTile = outputTile;
    for (auto axesInd : axesValue) {
        inTile.shape[Dim(axesInd)] = inShape[Dim(axesInd)];
    }

    return TilingInfo{{std::move(inTile)}};
}

namespace {

// Transform the coordinate in the resized tensor to the coordinate in the original tensor.
// It is from Interpolate-4 document at OpenVINO.
// scale = input_shape / output_shape
double inferInCoord(IE::InterpolateCoordMode coordMode, int64_t outCoord, int64_t origInSize, int64_t origOutSize,
                    double scale) {
    double inCoord = 0;
    switch (coordMode) {
    case IE::InterpolateCoordMode::HALF_PIXEL:
        inCoord = scale * (outCoord + 0.5) - 0.5;
        break;
    case IE::InterpolateCoordMode::PYTORCH_HALF_PIXEL:
        inCoord = origOutSize == 1 ? 0.0f : scale * (outCoord + 0.5) - 0.5;
        break;
    case IE::InterpolateCoordMode::ASYMMETRIC:
        inCoord = outCoord * scale;
        break;
    case IE::InterpolateCoordMode::TF_HALF_PIXEL_FOR_NN:
        inCoord = (outCoord + 0.5) * scale;
        break;
    case IE::InterpolateCoordMode::ALIGN_CORNERS:
        inCoord = origOutSize == 1 ? 0.0 : outCoord * (origInSize - 1.0) / (origOutSize - 1.0);
        break;
    default:
        VPUX_THROW("Doesn't support coordMode: {0}", coordMode);
    }
    return inCoord;
};

// Get the integer input coordinate from the float input coordinate according to the interpolate attributes
int64_t getNearestCoord(IE::InterpolateMode interpolateMode, IE::InterpolateNearestMode nearestMode, double inCoord,
                        double scale, bool roundUp) {
    int64_t nearestDim = 0;
    if (interpolateMode == IE::InterpolateMode::LINEAR || interpolateMode == IE::InterpolateMode::LINEAR_ONNX) {
        nearestDim = roundUp ? std::ceil(inCoord) : std::floor(inCoord);
    } else if (interpolateMode == IE::InterpolateMode::CUBIC) {
        nearestDim = roundUp ? std::floor(inCoord) + 2 : std::floor(inCoord) - 1;
    } else if (interpolateMode == IE::InterpolateMode::NEAREST) {
        switch (nearestMode) {
        case IE::InterpolateNearestMode::ROUND_PREFER_FLOOR:
            if (isDoubleEqual(inCoord, std::floor(inCoord) + 0.5)) {
                nearestDim = std::floor(inCoord);
            } else {
                nearestDim = std::round(inCoord);
            }
            break;
        case IE::InterpolateNearestMode::ROUND_PREFER_CEIL:
            nearestDim = std::round(inCoord);
            break;
        case IE::InterpolateNearestMode::FLOOR:
            nearestDim = std::floor(inCoord);
            break;
        case IE::InterpolateNearestMode::CEIL:
            nearestDim = std::ceil(inCoord);
            break;
        case IE::InterpolateNearestMode::SIMPLE:
            if (scale > 1.0) {
                nearestDim = std::ceil(inCoord);
            } else {
                nearestDim = std::floor(inCoord);
            }
            break;
        default:
            VPUX_THROW("Doesn't support nearestMode: {0}", nearestMode);
        }
    } else {
        VPUX_THROW("Doesn't support interpolateMode: {0}", interpolateMode);
    }

    return nearestDim;
};

SmallVector<int64_t> propagateOffsetForInterpolate(
        ArrayRef<int64_t> axes, ArrayRef<int64_t> offset, ArrayRef<int64_t> initialInputDims,
        ArrayRef<int64_t> initialOutputDims, ArrayRef<int64_t> initialInputOffsets,
        ArrayRef<int64_t> initialOutputOffsets, ArrayRef<int64_t> currentInputDims,
        vpux::IE::InterpolateCalcMode calcMode, vpux::IE::InterpolateMode interpolateMode,
        vpux::IE::InterpolateCoordMode coordMode, vpux::IE::InterpolateNearestMode nearestMode, ArrayRef<int64_t> sizes,
        ArrayRef<double> scales, bool roundUp, SmallVector<int64_t>&& tiledIndices, vpux::Logger log) {
    log.trace("Interp propagate offset: input = {0}", offset);

    SmallVector<int64_t> inferedOffset(offset.begin(), offset.end());
    if (calcMode == IE::InterpolateCalcMode::SIZES) {
        VPUX_THROW_WHEN(sizes.size() != axes.size(),
                        "Num of elements in sizes tensor: {0} should be equal to number of indices in axes: {1}",
                        sizes.size(), axes.size());
        auto sizesIter = sizes.begin();
        for (const auto& i : axes) {
            log.trace("Interp sizes - axis: {0}", i);
            inferedOffset[i] = *sizesIter++;
        }
    } else if (calcMode == IE::InterpolateCalcMode::SCALES) {
        VPUX_THROW_WHEN(scales.size() != axes.size(),
                        "Num of elements in scales tensor: {0} should be equal to number of indices in axes: {1}",
                        scales.size(), axes.size());
        auto scalesIter = scales.begin();
        for (const auto& i : axes) {
            log.trace("Interp scales - axis: {0}", i);

            if (std::find(tiledIndices.begin(), tiledIndices.end(), i) == tiledIndices.end()) {
                inferedOffset[i] = roundUp ? currentInputDims[i] - 1 : 0;
                scalesIter++;
            } else {
                double inCoord = inferInCoord(coordMode, offset[i] + initialOutputOffsets[i], initialInputDims[i],
                                              initialOutputDims[i], *scalesIter) -
                                 initialInputOffsets[i];
                int64_t inCoordInt = getNearestCoord(interpolateMode, nearestMode, inCoord, *scalesIter, roundUp);

                inferedOffset[i] = std::clamp(inCoordInt, static_cast<int64_t>(0), currentInputDims[i] - 1);
                scalesIter++;
            }
        }
    } else {
        VPUX_THROW("Doesn't support shape_calculation_mode: {0}", calcMode);
    }

    log.trace("Interp propagate offset: output = {0}", inferedOffset);
    return inferedOffset;
}

SmallVector<int64_t> backInferOffsetForInterpolate(
        ArrayRef<int64_t> offset, IE::InterpolateMode interpolateMode, IE::InterpolateCoordMode coordMode,
        IE::InterpolateNearestMode nearestMode, ArrayRef<int64_t> initialInputDims, ArrayRef<int64_t> initialOutputDims,
        ArrayRef<int64_t> initialInputOffsets, ArrayRef<int64_t> initialOutputOffsets,
        ArrayRef<int64_t> currentInputDims, bool roundUp, SmallVector<int64_t>&& tiledIndices, Logger log) {
    SmallVector<int64_t> axes;
    for (auto i : irange(initialInputDims.size())) {
        if (initialInputDims[i] != initialOutputDims[i]) {
            axes.push_back(i);
        }
    }

    // Compute scale-factors based on full I/O resolution ratio
    SmallVector<int64_t> fullOutSize;
    SmallVector<double> backwardScale;
    for (size_t i = 0; i < axes.size(); i++) {
        backwardScale.push_back(static_cast<double>(initialInputDims[axes[i]]) / initialOutputDims[axes[i]]);
        fullOutSize.push_back(initialOutputDims[axes[i]]);
    }

    // TODO: E#36318 how to deal with calc-mode = size if scales missed - recalc them somewhere:
    auto shapeCalcMode = IE::InterpolateCalcMode::SCALES;
    return propagateOffsetForInterpolate(axes, offset, initialInputDims, initialOutputDims, initialInputOffsets,
                                         initialOutputOffsets, currentInputDims, shapeCalcMode, interpolateMode,
                                         coordMode, nearestMode, fullOutSize, backwardScale, roundUp,
                                         std::move(tiledIndices), log);
}
}  // namespace

//
// Interpolate tiling
//

InputTiling vpux::backInferInterpolateTile(const vpux::TileInfo& outputTile, ArrayRef<int64_t> initialInputDims,
                                           ArrayRef<int64_t> initialOutputDims, ArrayRef<int64_t> initialInputOffsets,
                                           ArrayRef<int64_t> initialOutputOffsets, ArrayRef<int64_t> currentInputDims,
                                           std::optional<ArrayRef<int64_t>> coordinatesDims,
                                           std::optional<ArrayRef<int64_t>> lambdasDims,
                                           vpux::IE::InterpolateMode interpolateMode,
                                           vpux::IE::InterpolateCoordMode coordMode,
                                           vpux::IE::InterpolateNearestMode nearestMode, vpux::Logger log) {
    log.trace("Try to back infer input tiling for Interpolate, output tile: {0}", outputTile);

    auto outputOffsetBegin = to_small_vector(outputTile.offsets);
    SmallVector<int64_t> outputOffsetEnd(outputOffsetBegin.size());
    for (size_t ind = 0; ind < outputOffsetEnd.size(); ind++) {
        outputOffsetEnd[ind] = outputOffsetBegin[ind] + outputTile.shape[Dim(ind)] - 1;
    }

    SmallVector<int64_t> tiledIndices;
    for (auto i : irange(outputTile.axis.size())) {
        if (outputTile.axis[Dim(i)] > 1) {
            tiledIndices.push_back(i);
        }
    }

    auto inferedInputOffsetBegin = backInferOffsetForInterpolate(
            outputOffsetBegin, interpolateMode, coordMode, nearestMode, initialInputDims, initialOutputDims,
            initialInputOffsets, initialOutputOffsets, currentInputDims, false, std::move(tiledIndices), log);
    auto inferedInputOffsetEnd = backInferOffsetForInterpolate(
            outputOffsetEnd, interpolateMode, coordMode, nearestMode, initialInputDims, initialOutputDims,
            initialInputOffsets, initialOutputOffsets, currentInputDims, true, std::move(tiledIndices), log);

    SmallVector<int64_t> inferedInputShape(inferedInputOffsetEnd.size(), 0);
    for (size_t ind = 0; ind < inferedInputOffsetEnd.size(); ind++) {
        inferedInputShape[ind] = inferedInputOffsetEnd[ind] - inferedInputOffsetBegin[ind] + 1;
    }

    TileInfo inputTile(Shape(inferedInputShape), Shape(inferedInputOffsetBegin), outputTile.axis);
    SmallVector<TileInfo> tiles({std::move(inputTile)});
    if (coordinatesDims.has_value()) {
        tiles.emplace_back(Shape(coordinatesDims.value()));
    }
    if (lambdasDims.has_value()) {
        tiles.emplace_back(Shape(lambdasDims.value()));
    }
    return InputTiling{tiles};
}

//
// Gather tiling
//

InputTiling vpux::backInferGatherTile(const vpux::TileInfo& outputTile, const ShapeRef& origInputShape,
                                      const ShapeRef& origIndicesShape, int64_t axisValue, int64_t batchDims,
                                      bool hasAxisTensor, const int64_t indicesRank, vpux::Logger log) {
    log.trace("Try to back infer input tiling for Gather, output tile: {0}", outputTile);
    TileInfo inputTile(origInputShape);
    TileInfo indicesTile(origIndicesShape);

    auto inputRank = origInputShape.size();

    for (int64_t i = 0; i < static_cast<int64_t>(inputRank); ++i) {
        if (i < axisValue) {
            inputTile.shape[Dim(i)] = outputTile.shape[Dim(i)];
            inputTile.offsets[Dim(i)] = outputTile.offsets[Dim(i)];
        } else if (i == axisValue) {
            continue;
        } else {
            inputTile.shape[Dim(i)] = outputTile.shape[Dim(i + indicesRank - batchDims - 1)];
            inputTile.offsets[Dim(i)] = outputTile.offsets[Dim(i + indicesRank - batchDims - 1)];
        }
    }

    for (int64_t i = 0; i < static_cast<int64_t>(indicesRank); ++i) {
        if (i < batchDims) {
            indicesTile.shape[Dim(i)] = outputTile.shape[Dim(i)];
            indicesTile.offsets[Dim(i)] = outputTile.offsets[Dim(i)];
        } else {
            indicesTile.shape[Dim(i)] = outputTile.shape[Dim(i + axisValue - batchDims)];
            indicesTile.offsets[Dim(i)] = outputTile.offsets[Dim(i + axisValue - batchDims)];
        }
    }

    if (hasAxisTensor) {
        return InputTiling{{std::move(inputTile), std::move(indicesTile), TileInfo(ShapeRef({1}))}};
    }

    return InputTiling{{std::move(inputTile), std::move(indicesTile)}};
}

//
// GatherDMA tiling
//

InputTiling vpux::backInferGatherDMATile(const vpux::TileInfo& outputTile, ShapeRef origInputShape,
                                         ShapeRef origIndicesShape, int64_t axisValue, bool hasAxisTensor,
                                         vpux::Logger log) {
    log.trace("Try to back infer input tiling for Gather-DMA, output tile: {0}", outputTile);
    TileInfo inputTile(origInputShape);
    TileInfo indicesTile(origIndicesShape);

    auto inputRank = origInputShape.size();

    for (int64_t i = 0; i < static_cast<int64_t>(inputRank); ++i) {
        if (i != axisValue) {
            inputTile.shape[Dim(i)] = outputTile.shape[Dim(i)];
            inputTile.offsets[Dim(i)] = outputTile.offsets[Dim(i)];
        }
    }

    indicesTile.shape[Dim(axisValue)] = outputTile.shape[Dim(axisValue)];
    indicesTile.offsets[Dim(axisValue)] = outputTile.offsets[Dim(axisValue)];

    if (hasAxisTensor) {
        return InputTiling{{std::move(inputTile), std::move(indicesTile), TileInfo(ShapeRef({1}))}};
    }

    return InputTiling{{std::move(inputTile), std::move(indicesTile)}};
}

//
// GatherElements tiling
//

InputTiling vpux::backInferGatherElementsTile(const vpux::TileInfo& outputTile, const ShapeRef& origInputShape,
                                              const ShapeRef& origIndicesShape, int64_t axisValue,
                                              const int64_t indicesRank, vpux::Logger log) {
    log.trace("Try to back infer input tiling for GatherElements, output tile: {0}", outputTile);
    TileInfo inputTile(origInputShape);
    TileInfo indicesTile(origIndicesShape);

    auto inputRank = origInputShape.size();

    for (int64_t i = 0; i < static_cast<int64_t>(inputRank); ++i) {
        if (i != axisValue) {
            inputTile.shape[Dim(i)] = outputTile.shape[Dim(i)];
            inputTile.offsets[Dim(i)] = outputTile.offsets[Dim(i)];
        }
    }

    for (int64_t i = 0; i < static_cast<int64_t>(indicesRank); ++i) {
        indicesTile.shape[Dim(i)] = outputTile.shape[Dim(i)];
        indicesTile.offsets[Dim(i)] = outputTile.offsets[Dim(i)];
    }
    return InputTiling{{std::move(inputTile), std::move(indicesTile)}};
}

//
// Pad tiling
//

InputTiling vpux::backInferPadTile(const vpux::TileInfo& outputTile, const ShapeRef inShape, const ShapeRef outShape,
                                   const ShapeRef origPadsBegin, const ShapeRef origPadsEnd, vpux::Logger log) {
    log.trace("Try to back infer input tiling for Pad, output tile: {0}", outputTile);
    const auto padBegins = origPadsBegin;
    const auto padEnds = origPadsEnd;
    auto curTile = outputTile;

    for (auto ind : irange(inShape.size())) {
        auto idx = Dim(ind);

        if (curTile.axis[idx] == 1) {
            curTile.shape[idx] = inShape[idx];
        } else {
            curTile.shape[idx] = outputTile.shape[idx];
            if (outputTile.offsets[idx] == 0) {
                curTile.shape[idx] -= padBegins[idx];
            }
            if (outputTile.offsets[idx] + outputTile.shape[idx] == outShape[idx]) {
                curTile.shape[idx] -= padEnds[idx];
            }
        }
        VPUX_THROW_UNLESS(curTile.shape[idx] > 0, "Unsupported tile shape : '{0}'. Must be grater than 0.",
                          curTile.shape[idx]);

        if (outputTile.offsets[idx] != 0) {
            curTile.offsets[idx] = outputTile.offsets[idx] - padBegins[idx];
        } else {
            curTile.offsets[idx] = outputTile.offsets[idx];
        }
        curTile.axis[idx] = outputTile.axis[idx];
        VPUX_THROW_UNLESS(curTile.offsets[idx] < inShape[idx],
                          "Tile offset '{0}' must be smaller than input shape '{1}'.", curTile.offsets[idx],
                          inShape[idx]);
    }

    return TilingInfo{curTile};
}

void vpux::updatePadOpAttrsAfterTiling(const ShapeRef outShape, const TileInfo& outTile,
                                       SmallVector<int64_t>& padsBegin, SmallVector<int64_t>& padsEnd) {
    for (auto ind : irange(outShape.size())) {
        if (outTile.axis[Dim(ind)] == 1) {
            continue;
        }
        if (outTile.offsets[Dim(ind)] < padsBegin[ind]) {
            padsBegin[ind] = padsBegin[ind] - outTile.offsets[Dim(ind)];
        } else {
            padsBegin[ind] = 0;
        }
        if (outTile.offsets[Dim(ind)] + outTile.shape[Dim(ind)] != outShape[Dim(ind)]) {
            padsEnd[ind] = 0;
        }
    }
}

//
// DepthToSpace tiling
//

InputTiling vpux::backInferDepthToSpaceTile(const vpux::TileInfo& outputTile, ShapeRef origInputShape,
                                            int64_t blockSize, vpux::Logger) {
    VPUX_THROW_WHEN(blockSize == 0, "BlockSize is zero and used as a divisor");
    VPUX_THROW_WHEN(origInputShape.size() != 4, "Unsupported shape rank: {0}", origInputShape.size());

    TileInfo inputTile(origInputShape);
    inputTile.shape[Dims4D::Act::N] = outputTile.shape[Dims4D::Act::N];
    inputTile.shape[Dims4D::Act::C] = outputTile.shape[Dims4D::Act::C] * (blockSize * blockSize);
    inputTile.shape[Dims4D::Act::W] = outputTile.shape[Dims4D::Act::W] / blockSize;
    inputTile.shape[Dims4D::Act::H] = outputTile.shape[Dims4D::Act::H] / blockSize;

    inputTile.offsets[Dims4D::Act::N] = outputTile.offsets[Dims4D::Act::N];
    inputTile.offsets[Dims4D::Act::C] = outputTile.offsets[Dims4D::Act::C] * (blockSize * blockSize);
    inputTile.offsets[Dims4D::Act::W] = outputTile.offsets[Dims4D::Act::W] / blockSize;
    inputTile.offsets[Dims4D::Act::H] = outputTile.offsets[Dims4D::Act::H] / blockSize;

    return InputTiling{inputTile};
}

/// @brief Infer output window size OH X OW from input window size IH X IW
std::optional<std::pair<int64_t, int64_t>> vpux::spatialOutputForInputWindowSize(
        const std::pair<int64_t, int64_t>& inputHW, ArrayRef<int64_t> kernel, ArrayRef<int64_t> strides,
        const PadInfo& pads) {
    VPUX_THROW_WHEN(kernel.size() != 2, "Expected kernel size to be 2. Got '{0}'", kernel.size());
    const auto KY = kernel[Dims4D::Kernel::Y.ind()];
    const auto KX = kernel[Dims4D::Kernel::X.ind()];

    VPUX_THROW_WHEN(strides.size() != 2, "Expected strides size to be 2. Got '{0}'", strides.size());
    const auto SY = strides[Dims4D::Strides::Y.ind()];
    const auto SX = strides[Dims4D::Strides::X.ind()];

    const auto padTop = pads.top;
    const auto padBottom = pads.bottom;
    const auto padLeft = pads.left;
    const auto padRight = pads.right;
    if (padTop < 0 || padBottom < 0 || padLeft < 0 || padRight < 0) {
        VPUX_THROW("Invalid pads: top '{0}', bottom '{1}', left '{2}', right '{3}'", padTop, padBottom, padLeft,
                   padRight);
    }

    const auto outputHeight = (inputHW.first - KY + padTop + padBottom) / SY + 1;
    const auto outputWidth = (inputHW.second - KX + padLeft + padRight) / SX + 1;

    if (outputHeight <= 0 || outputWidth <= 0) {
        return std::nullopt;
    }

    return std::make_pair(outputHeight, outputWidth);
}

//
// Tiling utils
//

std::tuple<DimRange, int64_t, int64_t> vpux::inputForOutputDim(const DimRange& output, int64_t kernel, int64_t stride,
                                                               const DimRange& initialInputRange, int64_t padBefore,
                                                               int64_t padAfter) {
    VPUX_THROW_UNLESS(output.length() > 0, "Wrong output tile '{0}'", output);
    VPUX_THROW_UNLESS(initialInputRange.length() > 0, "Wrong initial input range '{0}'", initialInputRange);
    VPUX_THROW_UNLESS(kernel > 0, "Wrong kernel '{0}'", kernel);
    VPUX_THROW_UNLESS(stride > 0, "Wrong stride '{0}'", stride);
    VPUX_THROW_UNLESS(padBefore >= 0, "Wrong padBefore '{0}'", padBefore);
    VPUX_THROW_UNLESS(padAfter >= 0, "Wrong padAfter '{0}'", padAfter);

    DimRange input = {0, 0};
    int64_t before = 0;
    int64_t after = 0;

    input.begin = output.begin * stride - padBefore;

    if (input.begin < initialInputRange.begin) {
        VPUX_THROW_UNLESS(initialInputRange.begin - input.begin <= padBefore,
                          "Input tile '{0}' and padBefore '{1}' doesn't match to initial range '{2}'", input, padBefore,
                          initialInputRange);

        before = std::min(initialInputRange.begin - input.begin, padBefore);
        input.begin = initialInputRange.begin;
    }

    VPUX_THROW_UNLESS(input.begin < initialInputRange.end, "Input tile '{0}' doesn't match to initial range '{1}'",
                      input, initialInputRange);

    input.end = (output.end - 1) * stride + kernel - padBefore;

    if (input.end > initialInputRange.end) {
        VPUX_THROW_UNLESS(input.end - initialInputRange.end <= padAfter,
                          "Input tile '{0}' and padAfter '{1}' doesn't match to initial range '{2}'", input, padAfter,
                          initialInputRange);

        after = std::min(input.end - initialInputRange.end, padAfter);
        input.end = initialInputRange.end;
    }

    VPUX_THROW_UNLESS(input.end > initialInputRange.begin, "Input tile '{0}' doesn't match to initial range '{1}'",
                      input, initialInputRange);
    VPUX_THROW_UNLESS(input.length() > 0, "Input tile '{0}' doesn't match to initial range '{1}'", input,
                      initialInputRange);

    return std::make_tuple(input, before, after);
}

// @brief Following function computes new strides based on the new tensor shape.
// @warning The new shape can be a result of tiling or aligning or something else.
SmallVector<Strides> vpux::adaptStrides(ShapeRef origShape, StridesRef origStrides, ArrayRef<Shape> adaptedShapes,
                                        DimsOrder dimsOrder) {
    auto adaptedStrides = SmallVector<Strides>();
    const auto memShape = dimsOrder.toMemoryOrder(origShape);
    const auto memStrides = dimsOrder.toMemoryOrder(origStrides);

    for (const auto& adaptedShape : adaptedShapes) {
        const auto adaptedMemShape = dimsOrder.toMemoryOrder(Shape(adaptedShape));

        SmallVector<Bit> adaptedMemStrides(memStrides.raw());
        // Automatically adaptedMemStrides.back() is equal to the element type size
        for (int i = static_cast<int>(memStrides.size()) - 2; i >= 0; --i) {
            // Compute the ration between consecutive strides.
            // This tells us how many elements were accounted for in the original
            // strides and by using this, we incrementally construct the new adapted strides.
            const auto currStride = memStrides[MemDim(i)].count();
            const auto prevStride = memStrides[MemDim(i + 1)].count();
            const auto prevAdaptedStride = adaptedMemStrides[i + 1].count();

            auto adaptedStride = prevAdaptedStride * currStride / prevStride;

            if (adaptedStride != (int)adaptedStride) {
                vpux::Logger log("VPUX Adapt Strides Tiling method", vpux::LogLevel::Error);
                log.error("Adapted strides has decimals and may cause problems");
            }

            const auto strideRatio = currStride / prevStride;
            // If there is a change between the original and the new shape,
            // we favor striding with the new shape size instead of the previous stride ratio.
            if (memShape[MemDim(i + 1)] != adaptedMemShape[MemDim(i + 1)]) {
                // In the case of multiclustering, all such scenarios like H|K cluster tiling
                // with H|K prefetch tiling should be concatenated in DDR as simple tensors.

                // Long story to why we don't allow strides and tiling on same axis:
                // Mostly it's unclear how we should handle correctly such a case, because the nature
                // of strides can be very multifaceted, and we don't have explicit knowledge of the
                // scope for that stride.
                //
                // Let's take a case like 24 dimension strided to 32.
                // You may do this to either stride to 32 to fit in a concat over the specific axis
                // Or you may do this for alignment reasons, such that each pixel starts at a 16 byte
                // aligned address.
                //
                // So if we tile 24 by 2, and have 12. How should the strides be adapted?
                // Should we keep them as 32 to satisfy the concat or should we readjust them and align
                // to next value multiple of 16, which will be 16.
                // It's this lack of information and very context dependent reason why we avoid to
                // tackle this case.
                //
                // Without having a solid and functional infrastructure, to do everything in full knowledge
                // of context it can easily lead to a lot of problems and instabilities in the future.

                VPUX_THROW_WHEN(strideRatio != memShape[MemDim(i + 1)],
                                "Can't have both stride ratio '{0}' != shape '{1}' and also adapted shape '{2}' on "
                                "same axis '{3}'.",
                                strideRatio, memShape[MemDim(i + 1)], adaptedMemShape[MemDim(i + 1)], i + 1);
                adaptedStride = adaptedMemShape[MemDim(i + 1)] * prevAdaptedStride;
            }

            adaptedMemStrides[i] = Bit(adaptedStride);
        }
        adaptedStrides.emplace_back(dimsOrder.toLogicalOrder(MemStrides(adaptedMemStrides)));
    }

    return adaptedStrides;
}

DimArr vpux::getTileDimOrderND(MemShape memShape, DimsOrder dimOrder) {
    // Function calculates tile dim order from memory shape and dimOrder
    // It prioritize dim order depending on dim size and dimsOrder
    // Ex: MemShape: 3x80x80x40x80  DimOrder: NCDHW (0x12345)
    //      the return will be {1, 2, 4, 3, 0}
    //            equivalent:  {C, D, W, H, N}
    auto outputMemShape = memShape.raw();
    auto outputSortShape = memShape.raw();
    const auto outputDimOrderVec = dimOrder.toPermutation();

    std::sort(outputSortShape.begin(), outputSortShape.end(), std::greater<int64_t>());

    DimArr returntileDimOrder;

    for (auto it : outputSortShape) {
        if (it > 1) {
            // find the first value that match
            auto dimIt = std::find(outputMemShape.begin(), outputMemShape.end(), it);
            // extract the DimOrder
            returntileDimOrder.push_back(outputDimOrderVec[dimIt - outputMemShape.begin()]);
            // set the value to 0 to avoid geting the same index if more values are equals
            *dimIt = 0;
        }
    }

    return returntileDimOrder;
}

DimArr getTileDimOrderByShape(mlir::Operation* op, Dim filterDimToCompare, Dim actInputDimToCompare) {
    const auto preferTilingOrder = VPU::getSEPConvTilingOrder(op);
    if (preferTilingOrder.has_value()) {
        return preferTilingOrder.value();
    }
    VPUX_THROW_WHEN(op->getOperands().size() < 2,
                    "Only support multi-operand ops to get tile dim order by shape, but got '{0}'",
                    op->getOperands().size());
    const auto activationType = op->getOperand(0).getType().cast<vpux::NDTypeInterface>();
    const auto filterType = op->getOperand(1).getType().cast<vpux::NDTypeInterface>();
    const auto outputShape = getShape(op->getResult(0));
    const auto isChannelValid = VPU::doesNCEOpChannelSatisfyWorkload(op, TileInfo(outputShape));
    const auto isFilterLargerToTile =
            (filterType.getShape()[filterDimToCompare] > activationType.getShape()[actInputDimToCompare]) ||
            !isChannelValid;

    // #E152765 - generic support for GNCHW
    if (mlir::isa<VPU::NCEMatMulOp>(op)) {
        return isFilterLargerToTile
                       ? DimArr{DimsGroups5D::Act::G, DimsGroups5D::Act::C, DimsGroups5D::Act::H, DimsGroups5D::Act::W}
                       : DimArr{DimsGroups5D::Act::G, DimsGroups5D::Act::H, DimsGroups5D::Act::C, DimsGroups5D::Act::W};
    }
    return isFilterLargerToTile ? DimArr{Dims4D::Act::C, Dims4D::Act::H, Dims4D::Act::W}
                                : DimArr{Dims4D::Act::H, Dims4D::Act::C, Dims4D::Act::W};
}

DimArr vpux::getTileDimOrder(mlir::Operation* op, TilingMode tilingMode, Logger log) {
    // Compare the Activation and Filter channels
    // if filter channels > activation channels
    //      First tile at C
    // else tile at H

    auto tileDimOrder =
            llvm::TypeSwitch<mlir::Operation*, DimArr>(op)
                    .Case<VPU::NCEConvolutionOp, VPU::NCECompressConvolutionOp>([&](mlir::Operation* op) {
                        log.nest(2).trace("Check tile Dim order for Op at {0}", op->getLoc());
                        // This can be removed when VPUNN is upgraded to support INT4 data type, tracked in E#113316.
                        if (VPU::isNCEWithInt4Weights(op)) {
                            return getTileDimOrderByShape(op, Dims4D::Filter::OC, Dims4D::Act::H);
                        }
                        return getTileDimOrderByShape(op, Dims4D::Filter::IC, Dims4D::Act::C);
                    })
                    .Case<VPU::NCEDepthConvolutionOp>([&](mlir::Operation* op) {
                        log.nest(2).trace("Check tile Dim order for Op at {0}", op->getLoc());
                        return getTileDimOrderByShape(op, Dims4D::Filter::OC, Dims4D::Act::H);
                    })
                    .Case<VPU::NCEMaxPoolOp, VPU::NCEAveragePoolOp>([&](mlir::Operation* op) {
                        log.nest(2).trace("Check tile Dim order for Op at {0}", op->getLoc());
                        const auto outputShape = getShape(op->getResult(0));
                        const auto isChannelValid = VPU::doesNCEOpChannelSatisfyWorkload(op, TileInfo(outputShape));
                        if (isChannelValid) {
                            return DimArr{Dims4D::Act::H, Dims4D::Act::C, Dims4D::Act::W};
                        } else {
                            return DimArr{Dims4D::Act::C, Dims4D::Act::H, Dims4D::Act::W};
                        }
                    })
                    .Case<VPU::NCEMatMulOp>([&](mlir::Operation* op) {
                        log.nest(2).trace("Check tile Dim order for Op at {0}", op->getLoc());
                        const auto outputShape = getShape(op->getResult(0));
                        const auto isChannelValid = VPU::doesNCEOpChannelSatisfyWorkload(op, TileInfo(outputShape));
                        if (isChannelValid) {
                            return DimArr{DimsGroups5D::Act::G, DimsGroups5D::Act::H, DimsGroups5D::Act::C,
                                          DimsGroups5D::Act::W};
                        } else {
                            return DimArr{DimsGroups5D::Act::G, DimsGroups5D::Act::C, DimsGroups5D::Act::H,
                                          DimsGroups5D::Act::W};
                        }
                    })
                    .Case<VPU::MVNOp>([&](mlir::Operation* op) {
                        log.nest(2).trace("Check tile Dim order for Op at {0}", op->getLoc());
                        auto mvn1 = mlir::dyn_cast<VPU::MVNOp>(op);
                        auto dims = mvn1.getNonNormDims();
                        VPUX_THROW_UNLESS(dims.size(), "Could not find non-norm axes");
                        return dims;
                    })
                    .Case<VPU::MVN6Op>([&](mlir::Operation* op) {
                        auto mvn6 = mlir::dyn_cast<VPU::MVN6Op>(op);
                        auto dims = mvn6.getNonNormDims();
                        VPUX_THROW_UNLESS(dims.size(), "Could not find non-norm axes");
                        return dims;
                    })
                    .Case<VPU::MVN1NormalizeOp>([&](mlir::Operation* op) {
                        const auto outType = op->getResult(0).getType().cast<vpux::NDTypeInterface>();
                        const auto order = outType.getDimsOrder();
                        auto retDims = getTileDimOrderND(outType.getMemShape(), order);

                        if (order.toMemDim(Dims4D::Act::C).ind() == (outType.getRank() - 1)) {
                            // Avoid C-tiling in C-minor layout as may lead to Shave suboptimal configs (e.g. C=21)
                            auto dimIt = std::find(retDims.begin(), retDims.end(), Dims4D::Act::C);
                            if (dimIt != retDims.end()) {
                                retDims.erase(dimIt);
                            }
                        }
                        return retDims;
                    })
                    .Case<VPU::QuantizeOp>([&](mlir::Operation*) {
                        // Not splitting over C, to keep aligned with number of Scales in qType
                        // and so avoid 'validateQuantElemType' fail
                        return DimArr{Dims4D::Act::H, Dims4D::Act::W};
                    })
                    .Case<VPU::DequantizeOp>([&](mlir::Operation*) {
                        return DimArr{Dims4D::Act::N, Dims4D::Act::C, Dims4D::Act::H, Dims4D::Act::W};
                    })
                    .Case<VPU::DetectionOutputDecodeBoxesOp>([&](mlir::Operation*) {
                        return DimArr{Dims4D::Act::C, Dims4D::Act::H};  // [1, numLocClasses, numPriors, 4]
                    })
                    .Case<VPU::DetectionOutputSortOp>([&](mlir::Operation*) {
                        return DimArr{Dims4D::Act::H};  // [1, 1, numClasses, numPriors]
                    })
                    .Case<VPU::DetectionOutputNmsCaffeOp>([&](mlir::Operation*) {
                        return DimArr{Dims4D::Act::H};  // [1, 1, numClasses, topK]
                    })
                    .Case<VPU::NCEEltwiseOp>([&](mlir::Operation* op) {
                        const auto outputShape = getShape(op->getResult(0));
                        return outputShape[Dims4D::Act::C] / VPU::NCEInvariant::VPU_CHANNEL_ALIGNMENT <
                                               outputShape[Dims4D::Act::H]
                                       ? DimArr{Dims4D::Act::H, Dims4D::Act::C, Dims4D::Act::W}
                                       : DimArr{Dims4D::Act::C, Dims4D::Act::H, Dims4D::Act::W};
                    })
                    .Case<VPU::SoftMaxOp>([&](mlir::Operation* op) {
                        const auto outputType = op->getResult(0).getType().cast<vpux::NDTypeInterface>();
                        auto tileDimOrder = getTileDimOrderND(outputType.getMemShape(), outputType.getDimsOrder());
                        auto softMaxOp = mlir::cast<VPU::SoftMaxOp>(op);
                        auto axis = softMaxOp.getAxisIndAttr().getValue().getSExtValue();
                        auto dimIt = std::find(tileDimOrder.begin(), tileDimOrder.end(), Dim(axis));
                        if (dimIt != tileDimOrder.end()) {
                            // Tiling along SoftMax operation axis is not supported
                            log.nest(2).trace("Removing axis dim {0} for SoftMax {1}", *dimIt, tileDimOrder);
                            tileDimOrder.erase(dimIt);
                        }
                        return tileDimOrder;
                    })
                    .Case<VPU::PReluOp>([&](mlir::Operation* op) {
                        log.nest(2).trace("Check tile Dim order for Op at {0}", op->getLoc());
                        auto preluOp = mlir::dyn_cast<VPU::PReluOp>(op);
                        auto inputShape = getShape(preluOp.getInput());
                        auto slopeShape = getShape(preluOp.getNegativeSlope());
                        const auto outType = mlir::cast<vpux::NDTypeInterface>(preluOp.getOutput().getType());
                        const auto order = outType.getDimsOrder();
                        auto retDims = getTileDimOrderND(outType.getMemShape(), order);

                        if (slopeShape[Dims4D::Act::C] == inputShape[Dims4D::Act::C]) {
                            auto dimIt = std::find(retDims.begin(), retDims.end(),
                                                   Dim(order.toMemDim(Dims4D::Act::C).ind()));
                            if (dimIt != retDims.end()) {
                                retDims.erase(dimIt);
                            }
                        }
                        return retDims;
                    })
                    .Case<VPU::DepthToSpaceOp>([&](mlir::Operation* op) {
                        auto origOp = mlir::dyn_cast<VPU::DepthToSpaceOp>(op);
                        const auto outputType = op->getResult(0).getType().cast<vpux::NDTypeInterface>();
                        VPUX_THROW_UNLESS(outputType.getDimsOrder() == DimsOrder::NCHW ||
                                                  outputType.getDimsOrder() == DimsOrder::NHWC,
                                          "DepthToSpace Op only support NCHW and NHWC layout, but got '{0}'",
                                          outputType.getDimsOrder());

                        // It is better to tile DepthToSpace Op at the highest dimension
                        // to avoid stride concat that is inefficient
                        if (origOp.getMode() == IE::DepthToSpaceMode::DEPTH_FIRST) {
                            return outputType.getDimsOrder() == DimsOrder::NHWC
                                           ? SmallVector<Dim>{Dims4D::Act::H, Dims4D::Act::W, Dims4D::Act::C}
                                           : SmallVector<Dim>{Dims4D::Act::C, Dims4D::Act::H, Dims4D::Act::W};
                        }

                        // It is illegal to tile DepthToSpace Op at channel when it is the BLOCKS_FIRST mode
                        // If that, the output will be a discontinuous memory buffer and will cause accuracy issue
                        if (origOp.getMode() == IE::DepthToSpaceMode::BLOCKS_FIRST) {
                            return SmallVector<Dim>{Dims4D::Act::H, Dims4D::Act::W};
                        }

                        VPUX_THROW("Unknown DepthToSpaceMode. BLOCKS_FIRST and DEPTH_FIRST methods are supported only");
                    })
                    .Case<VPU::NCEPermuteOp>([&](mlir::Operation* op) {
                        log.nest(2).trace("Check tile Dim order for Op at {0}", op->getLoc());
                        return DimArr{Dims4D::Act::H, Dims4D::Act::W, Dims4D::Act::C};
                    })
                    .Case<VPU::NormalizeL2Op>([&](VPU::NormalizeL2Op op) {
                        const auto outputType = mlir::cast<vpux::NDTypeInterface>(op->getResult(0).getType());
                        auto tileDimOrder = getTileDimOrderND(outputType.getMemShape(), outputType.getDimsOrder());
                        auto axes = parseIntArrayAttr<int64_t>(op.getAxesValue());

                        for (auto axis : axes) {
                            llvm::erase(tileDimOrder, Dim(axis));
                        }

                        return tileDimOrder;
                    })
                    .Case<VPU::DynamicDequantizeOp>([&](VPU::DynamicDequantizeOp) {
                        log.nest(2).trace("Check tile Dim order for Op at {0}", op->getLoc());
                        const auto outputType = op->getResult(0).getType().cast<vpux::NDTypeInterface>();
                        auto tileDimOrder = getTileDimOrderND(outputType.getMemShape(), outputType.getDimsOrder());
                        // Ensure tile less the W dim to avoid using slow C algo
                        auto dimIt = std::find(tileDimOrder.begin(), tileDimOrder.end(), Dims4D::Act::W);
                        if (dimIt != tileDimOrder.end()) {
                            tileDimOrder.erase(dimIt);
                            tileDimOrder.push_back(Dims4D::Act::W);
                        }

                        return tileDimOrder;
                    })
                    .Default([&](mlir::Operation* op) -> DimArr {
                        log.nest(2).trace("Check tile Dim order for Op at {0}", op->getLoc());
                        const auto outputType = op->getResult(0).getType().cast<vpux::NDTypeInterface>();

                        return getTileDimOrderND(outputType.getMemShape(), outputType.getDimsOrder());
                    });

    // For prefetching mode, only weights can be pre-fetched to the parent op
    if (tilingMode == TilingMode::PREFETCHING) {
        tileDimOrder = SmallVector<Dim>({Dims4D::Act::C});
    }
    return tileDimOrder;
}

bool vpux::isMultiClusterCompatibleForTiling(mlir::Operation* op, const OutputTiling& tiles, Logger log) {
    auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(op);
    VPUX_THROW_WHEN(op == nullptr, "Operation '{0}' doesn't implement ClusteredOpInterface", op->getName());
    if (!clusteredOp->hasAttr(VPU::multiClusterStrategy)) {
        return true;
    }

    // Instead of checking strategy compatible shapes for all tiles, we only check the tiles have unique shapes
    // to reduce compilation time as isStrategyCompatibleShape only cares about tiled shape
    auto tileCandidates = VPU::getUniqueShapeTilingCandidates(op, tiles, log);

    auto isStrategyCompatibleWithTile = [&](const TileInfo& outputTile) {
        return VPU::isStrategyCompatibleShape(clusteredOp, outputTile, clusteredOp.getMultiClusterStrategy().value(),
                                              log);
    };

    return llvm::all_of(tileCandidates, isStrategyCompatibleWithTile);
}

// Compute the maximum number of tiles for each dimension to ensure:
// 1. Tiling numbers are compatible for each dimension.
// 2. (Height) DPUs are fully utilized - at least one line per DPU.
// 3. checkMinimalWidthAndHeight ensures each DPU processes at least 4 lines for efficiency.
// 4. (Channel) No extra channel alignment - the output channel for each cluster should be larger than minChannelSize.
SmallVector<int64_t> vpux::getMaxNumTiles(mlir::Operation* op, bool checkMinimalWidthAndHeight) {
    const auto outputShape = getShape(op->getResult(0));
    // #E152765 - generic support for GNCHW
    const auto dimH = mlir::isa<VPU::NCEMatMulOp>(op) ? DimsGroups5D::Act::H : Dims4D::Act::H;
    const auto dimW = mlir::isa<VPU::NCEMatMulOp>(op) ? DimsGroups5D::Act::W : Dims4D::Act::W;
    const auto dimC = mlir::isa<VPU::NCEMatMulOp>(op) ? DimsGroups5D::Act::C : Dims4D::Act::C;
    auto maxNumTiles = SmallVector<int64_t>(outputShape.begin(), outputShape.end());

    if (outputShape.size() != 4 && outputShape.size() != DimsGroups5D::Act::numDims) {
        if (mlir::isa<VPU::MemPermuteOp>(op)) {
            return maxNumTiles;
        }

        VPUX_THROW("Unsupported shape rank: {0}", outputShape.size());
    }

    int64_t minChannelSize = 1;
    int64_t minHeightSize = 1;
    int64_t minWidthSize = 1;
    if (mlir::isa<VPU::NCEOpInterface>(op) && checkMinimalWidthAndHeight) {
        // Stencils are using 4x4x16 tile configuration
        // NCE is more efficient when height and width are larger than 4 lines
        minHeightSize = 4;
        minWidthSize = 4;
    }
    // NCEPermute operation requires alignment only for width
    if (mlir::isa<VPU::NCEPermuteOp>(op)) {
        VPUX_THROW_UNLESS(outputShape.size() == 4, "Unsupported shape rank: {0}", outputShape.size());

        const auto outputType = op->getResult(0).getType().cast<vpux::NDTypeInterface>();
        minWidthSize = VPU::NCEInvariant::getAlignment(outputType.getElementType());
    } else {
        if (auto channelsInfo = mlir::dyn_cast<IE::AlignedChannelsOpInterface>(op)) {
            VPUX_THROW_UNLESS(outputShape.size() == 4 || outputShape.size() == DimsGroups5D::Act::numDims,
                              "Unsupported shape rank: {0}", outputShape.size());
            minChannelSize = channelsInfo.getOutputChannelAlignment();
        }

        // Consider supported channels for DW ops
        if (auto channelAlignedIface = mlir::dyn_cast<VPU::AlignedWorkloadChannelsOpInterface>(op)) {
            const auto supportedChannels = channelAlignedIface.getSupportedWorkLoadChannels();
            const auto minSupportedChannel = supportedChannels.back();
            if (minChannelSize < minSupportedChannel) {
                minChannelSize = minSupportedChannel;
            }
        }

        const auto maxChannelTiles = outputShape[dimC] / minChannelSize;
        maxNumTiles[dimC.ind()] = maxChannelTiles;
    }

    if (op->hasAttr(VPU::multiClusterStrategy)) {
        VPUX_THROW_UNLESS(outputShape.size() == 4 || outputShape.size() == DimsGroups5D::Act::numDims,
                          "Unsupported shape rank: {0}", outputShape.size());

        auto strategy = op->getAttrOfType<VPU::MultiClusterStrategyAttr>(VPU::multiClusterStrategy).getValue();
        auto module = op->getParentOfType<mlir::ModuleOp>();
        auto tileOp = IE::getTileExecutor(module);
        if (strategy == VPU::MultiClusterStrategy::SplitOverHeight ||
            strategy == VPU::MultiClusterStrategy::SplitOverHeightOverlapped) {
            // To ensure the SOH MultiCluster strategy remains compatible and maintains NCE efficiency after tiling:
            // 1. Each cluster should compute at least minHeightSize output lines.
            //    - For example, in a 4-cluster compilation, each NCE tile should have a height of at least 4x4 = 16.
            //    - When tiling an NCE layer with an output height of 64, the number of tiles in the height dimension
            //    should be <= 64/16 = 4.
            // 2. For sw layers, ensure that each cluster computes at least 1 output line.
            // Allow tiling down to a single line if the tiling dimension and the MultiCluster segmented dimension are
            // the same.
            minHeightSize = 1;
            minHeightSize *= tileOp.getCount();
        } else if (strategy == VPU::MultiClusterStrategy::SplitOverWidth) {
            minWidthSize = 1;
            minWidthSize *= tileOp.getCount();
        } else if (strategy == VPU::MultiClusterStrategy::SplitOverKernel) {
            // To make sure the SOK MultiCluster strategy still compatible after tiling,
            // each cluster should compute at least minChannelSize(=16) output channels.
            // For SOK, we can use less than the specified number of clusters, to avoid the requirement to align output
            int64_t minNumClustersForSOK = tileOp.getCount();
            while (minNumClustersForSOK > 0 && outputShape[dimC] % (minChannelSize * minNumClustersForSOK) != 0) {
                --minNumClustersForSOK;
            }
            if (minNumClustersForSOK <= 1) {
                minNumClustersForSOK = tileOp.getCount();
            }
            maxNumTiles[dimC.ind()] = outputShape[dimC] / (minChannelSize * minNumClustersForSOK);
        } else if (strategy == VPU::MultiClusterStrategy::SplitOverGroup) {
            maxNumTiles[DimsGroups5D::Act::G.ind()] = divUp(outputShape[DimsGroups5D::Act::G], tileOp.getCount());
        }
    }

    if ((mlir::isa<VPU::NCEOpInterface>(op) && checkMinimalWidthAndHeight) ||
        mlir::isa<VPU::DetectionOutputSortOp>(op)) {
        // For DetectionOutputSortOp, there are some cases where the height dimension of the tiles
        // is smaller than the number of available clusters.
        // TODO: E#-150388 generic solution for all sw layers
        maxNumTiles[dimH.ind()] = divUp(outputShape[dimH], minHeightSize);
        maxNumTiles[dimW.ind()] = divUp(outputShape[dimW], minWidthSize);
    } else {
        // For other sw layers, ensure at least one line per cluster
        maxNumTiles[dimH.ind()] = outputShape[dimH] / minHeightSize;
        maxNumTiles[dimW.ind()] = outputShape[dimW] / minWidthSize;
    }

    return maxNumTiles;
}

InputTiling vpux::backInferEltwiseTile(mlir::Operation* op, const vpux::TileInfo& outputTile) {
    SmallVector<TileInfo> inputTiles;
    for (auto& origInput : op->getOpOperands()) {
        const auto curShape = getShape(origInput.get());
        VPUX_THROW_UNLESS(curShape.size() == outputTile.shape.size(),
                          "Can't tile eltwise operation '{0}' at '{1}', which has operands with different rank",
                          op->getName(), op->getLoc());

        // Handle broadcasted inputs
        auto curTile = outputTile;
        for (auto ind : irange(curShape.size())) {
            const auto d = Dim(ind);
            if (curShape[d] == 1) {
                curTile.shape[d] = 1;
                curTile.offsets[d] = 0;
            }
        }

        inputTiles.push_back(curTile);
    }
    return TilingInfo{inputTiles};
}

// SWLayer

mlir::FailureOr<OutputTiling> vpux::getSWLayerTilingStrategyWithTileDimOrder(mlir::Operation* op, TilingMode tilingMode,
                                                                             DimArrRef tileDimOrder, Logger log,
                                                                             ArrayRef<int64_t> maxTilesPerDim) {
    auto tilingInfo = mlir::dyn_cast<VPU::TilingInfoOpInterface>(op);
    VPUX_THROW_WHEN(tilingInfo == nullptr, "Operation '{0}' doesn't implement TilingInfoOpInterface", op->getName());
    auto tilingBuilder = mlir::dyn_cast<VPU::TilingBuilderOpInterface>(op);
    VPUX_THROW_WHEN(tilingBuilder == nullptr, "Operation '{0}' doesn't implement TilingBuilderOpInterface",
                    op->getName());
    VPUX_THROW_WHEN(tilingMode != TilingMode::ISOLATED && tilingMode != TilingMode::PIPELINING,
                    "Only supporting isolated and pipelining tiling for SW currently, for op {0} at '{1}'",
                    op->getName(), op->getLoc());

    const auto outputType = op->getResult(0).getType().cast<vpux::NDTypeInterface>();
    const auto outputShape = outputType.getShape();

    const auto isSupportedTileSize = [op, &tilingInfo, outputShape, log](ShapeRef nTilesOnDim,
                                                                         TilingMode tilingMode) -> bool {
        const auto tiles = fillDividedTiles(op, nTilesOnDim, outputShape);
        if (mlir::failed(tiles)) {
            return false;
        }
        return tilingInfo.isSupportedTiling(tiles.value(), tilingMode, log);
    };

    SmallVector<int64_t> maxNumTiles(maxTilesPerDim.begin(), maxTilesPerDim.end());
    if (maxTilesPerDim.empty()) {
        maxNumTiles = tilingBuilder.getMaxNumTiles();
    }

    // Step1. get an feasible isolated tiling strategy
    auto nTilesOnDim = [&]() {
        auto tilingStrategy = Shape(outputShape.size(), 1);

        for (auto tileDim : tileDimOrder) {
            while (true) {
                if (isSupportedTileSize(tilingStrategy, TilingMode::ISOLATED)) {
                    return tilingStrategy;
                }

                if (isDimLeftToTile(tilingStrategy, maxNumTiles, tileDim)) {
                    ++tilingStrategy[tileDim];
                } else {
                    break;
                }
            }
        }

        // Empty tileDimOrder case
        VPUX_THROW_UNLESS(isSupportedTileSize(tilingStrategy, TilingMode::ISOLATED), "Failed to tile {0} at '{1}'",
                          op->getName(), op->getLoc());

        return tilingStrategy;
    }();

    auto resultTiles = fillDividedTiles(op, nTilesOnDim, outputShape);

    auto tilingDims = getNonOneDim(nTilesOnDim);
    if (VPUIP::isLegalConvertToDMA(op, log, /*checkCMXSize*/ false) || tilingDims.size() != 1) {
        log.trace("Sw-DMA Isolated tiling strategy: {0}", nTilesOnDim);
        return resultTiles;
    }

    // Step2. For pipelining, continue to increase on the dimension of isolated tiling
    const auto targetDim = *tilingDims.begin();
    if (tilingMode == TilingMode::PIPELINING) {
        Shape prefetchableTilesOnDim = nTilesOnDim;
        log.trace("Sw attempting to generate tiling strategy for pipelining");
        while (!isSupportedTileSize(prefetchableTilesOnDim, TilingMode::PIPELINING)) {
            if (prefetchableTilesOnDim[targetDim] >= MAX_PREFETCH_TILING_TIME * nTilesOnDim[targetDim] ||
                !isDimLeftToTile(prefetchableTilesOnDim, maxNumTiles, targetDim)) {
                log.nest(3).trace("Sw fallback to isolated strategy: {0}", nTilesOnDim);
                tilingMode = TilingMode::ISOLATED;
                break;
            }
            ++prefetchableTilesOnDim[targetDim];
        }

        // Found the pipeline tiling
        if (tilingMode == TilingMode::PIPELINING) {
            nTilesOnDim = std::move(prefetchableTilesOnDim);
            log.trace("Sw Pipelining tiling strategy: {0}", nTilesOnDim);
            resultTiles = fillDividedTiles(op, nTilesOnDim, outputShape);
        }
    }

    if (vpux::VPU::canSWLayerBeEvenlyUnrolled(op, resultTiles.value(), targetDim, log)) {
        log.trace("Sw {0} tiling strategy: {1}", getTilingModeStr(tilingMode), nTilesOnDim);
        return resultTiles;
    }

    // Step3. continue to increase on the dimension of isolated tiling to get a output shape
    // that can be evenly distributed on ACT SHAVEs
    Shape evenUnrollingTilesOnDim = nTilesOnDim;
    log.trace("Sw attempting to generate tiling strategy for even unrolling");
    while (isDimLeftToTile(evenUnrollingTilesOnDim, maxNumTiles, targetDim) &&
           // Prevent long compilation time caused by excessive tiling
           evenUnrollingTilesOnDim[targetDim] <= MAX_EXCESSIVE_TILING_TIME * nTilesOnDim[targetDim]) {
        ++evenUnrollingTilesOnDim[targetDim];

        auto evenUnrollingTiles = fillDividedTiles(op, evenUnrollingTilesOnDim, outputShape);
        if (mlir::succeeded(evenUnrollingTiles) &&
            vpux::VPU::canSWLayerBeEvenlyUnrolled(op, evenUnrollingTiles.value(), targetDim, log)) {
            log.nest(3).trace("Sw found better {0} tiling strategy: {1} for even unrolling",
                              getTilingModeStr(tilingMode), evenUnrollingTilesOnDim);
            return evenUnrollingTiles;
        }
    }

    log.nest(3).trace("Sw fallback to {0} tiling strategy: {1}", getTilingModeStr(tilingMode), nTilesOnDim);
    return resultTiles;
}

mlir::FailureOr<OutputTiling> vpux::getSWLayerTilingStrategy(mlir::Operation* op, TilingMode tilingMode, Logger log,
                                                             ArrayRef<int64_t> maxTilesPerDim) {
    const auto tileDimOrder = getTileDimOrder(op, tilingMode, log);
    log.nest(2).trace("Tile Dim order is {0}", tileDimOrder);
    return getSWLayerTilingStrategyWithTileDimOrder(op, tilingMode, tileDimOrder, log, maxTilesPerDim);
}

// Compute the maximum of tile number for each dimension with respect of not tile specified axes. No other restriction
// apply, rest of maximum tiles reflect output shape.
SmallVector<int64_t> vpux::getMaxNumTilesWithAxesExclusion(mlir::Operation* op, ArrayRef<int64_t> axes) {
    const auto outputType = op->getResult(0).getType().cast<vpux::NDTypeInterface>();
    const auto outputShape = outputType.getShape();
    SmallVector<int64_t> maxNumTiles(outputShape.begin(), outputShape.end());
    const auto tileDimOrder = getTileDimOrderND(outputType.getMemShape(), outputType.getDimsOrder());
    for (const auto dimVal : tileDimOrder) {
        if (std::find(axes.begin(), axes.end(), dimVal.ind()) != axes.end()) {
            // not tile over axis, not alowed
            maxNumTiles[dimVal.ind()] = 1;
        }
    }
    return maxNumTiles;
}

namespace {

inline bool compareShape(ShapeRef smallShape, ShapeRef bigShape, Dim dimToTile) {
    VPUX_THROW_UNLESS(smallShape.size() == bigShape.size(), "Can't compare two shapes with different ranks");
    VPUX_THROW_UNLESS(smallShape.size() >= static_cast<size_t>(dimToTile.ind()), "Dim to tile exceeds shape size");
    return smallShape[dimToTile] < bigShape[dimToTile];
}

// Allow uneven tiling over OC, such as OC = 80 can be tiled as three tiles [32, 32, 16]
bool isSupportedAlignedDivision(int64_t dimSize, int64_t tiles, int64_t alignment) {
    auto base = vpux::divUp(dimSize, tiles);
    auto alignedBase = alignValUp(base, alignment);
    auto remainder = dimSize - alignedBase * (tiles - 1);
    return remainder > 0;
}

void dimPlus(Shape& nTilesOnDim, Dim dimToTile, Dim dimToAlign, int64_t dimAlignment, ShapeRef outputShape,
             ArrayRef<int64_t> maxNumTiles, const Logger& log) {
    if (dimToTile == dimToAlign && dimAlignment != 1) {
        do {
            ++nTilesOnDim[dimToTile];
        } while (!isSupportedAlignedDivision(outputShape[dimToTile], nTilesOnDim[dimToTile], dimAlignment) &&
                 isDimLeftToTile(nTilesOnDim, maxNumTiles, dimToTile));
    } else {
        ++nTilesOnDim[dimToTile];
    }
    log.nest().trace("dimPlus: nTilesOnDim - {0}", nTilesOnDim);
}

void dimMinus(Shape& nTilesOnDim, Dim dimToTile, Dim dimToAlign, int64_t dimAlignment, ShapeRef outputShape,
              const Logger& log) {
    --nTilesOnDim[dimToTile];
    // Skip the tiling numbers which are not aligned
    while (dimToTile == dimToAlign && dimAlignment != 1 && nTilesOnDim[dimToTile] > 1 &&
           !isSupportedAlignedDivision(outputShape[dimToTile], nTilesOnDim[dimToTile], dimAlignment)) {
        --nTilesOnDim[dimToTile];
    }
    log.nest().trace("dimMinus: nTilesOnDim - {0}", nTilesOnDim);
}

void ensureNTilesIsCompatibleWithMultiCluster(mlir::Operation* op, Shape& nTilesOnDim, Dim dimToTile,
                                              ShapeRef outputShape, TilingMode tilingModeToCheck, const Logger& log) {
    const auto dimAlignInfo = getAlignDimAndSize(op);
    auto dimToAlign = dimAlignInfo.first;
    auto dimAlignment = dimAlignInfo.second;
    auto tiles = fillDividedTiles(op, nTilesOnDim, outputShape);
    while (nTilesOnDim[dimToTile] > 1) {
        if (!mlir::failed(tiles)) {
            auto isMCCompatible = isMultiClusterCompatibleForTiling(op, tiles.value(), log);
            if (isMCCompatible) {
                if (tilingModeToCheck == TilingMode::ISOLATED) {
                    break;
                }
                const auto nceOp = mlir::dyn_cast<VPU::NCEOpInterface>(op);
                const auto isChannelDivisible = nceOp == nullptr || dimToTile != Dims4D::Act::C ||
                                                VPU::isDivisibleTile(op, tiles.value()[0].axis, dimToTile);
                if (isChannelDivisible) {
                    break;
                }
            }
        }
        dimMinus(nTilesOnDim, dimToTile, dimToAlign, dimAlignment, outputShape, log);
        tiles = fillDividedTiles(op, nTilesOnDim, outputShape);
    }
}

}  // namespace

// HWLayer
mlir::FailureOr<OutputTiling> vpux::getHWLayerTilingStrategyWithTileDimOrderForIsolatedOrPrefetch(
        mlir::Operation* op, TilingMode tilingMode, DimArrRef tileDimOrder, ShapeRef outputShape, Logger log) {
    auto tilingInfo = mlir::dyn_cast<VPU::TilingInfoOpInterface>(op);
    VPUX_THROW_WHEN(tilingInfo == nullptr, "Operation '{0}' doesn't implement TilingInfoOpInterface", op->getName());
    auto tilingBuilder = mlir::dyn_cast<VPU::TilingBuilderOpInterface>(op);
    VPUX_THROW_WHEN(tilingBuilder == nullptr, "Operation '{0}' doesn't implement TilingBuilderOpInterface",
                    op->getName());
    VPUX_THROW_UNLESS(outputShape.size() == 4 || outputShape.size() == DimsGroups5D::Act::numDims,
                      "Unsupported operation '{0}' at '{1}', it has non 4D/5D result", op->getName(), op->getLoc());

    const auto dimAlignInfo = getAlignDimAndSize(op);
    auto dimToAlign = dimAlignInfo.first;
    auto dimAlignment = dimAlignInfo.second;

    Shape nTilesOnDim(outputShape.size(), 1);

    auto tileDimIter = tileDimOrder.begin();

    const auto maxNumTiles = tileDimOrder.size() <= 1 ? tilingBuilder.getMaxNumTiles() : getMaxNumTiles(op, true);

    // In case of pipelining, an isolated tiling strategy is first created
    // Then the tiling number would be increased to get a pipelining tiling strategy
    // If no feasible pipelining tiling could be found, fallback to isolated tiling strategy

    auto findMiddleTile = [&](Shape& small, Shape& big, Dim dimToTile) {
        auto rank = small.size();
        VPUX_THROW_UNLESS(rank == big.size(), "Can't get middle for two shapes with different ranks");
        auto middle = Shape(small);
        VPUX_THROW_UNLESS(small[dimToTile] < big[dimToTile], "small {0} and big {1} illegal", small, big);
        middle[dimToTile] = (small[dimToTile] + big[dimToTile] + 1) / 2;
        ensureNTilesIsCompatibleWithMultiCluster(op, middle, dimToTile, outputShape, tilingMode, log);
        return middle;
    };

    auto storeAndReturnFailure = [&]() -> mlir::FailureOr<OutputTiling> {
        if (tilingMode == TilingMode::ISOLATED) {
            log.nest(1).trace("Failed to tile {0} at '{1}'", op->getName(), op->getLoc());
            return mlir::failure();
        }
        // If still not find the tiling strategy in PREFETCHING, fall back to neutral tiling
        auto neutralTiling = Shape(outputShape.size(), 1);
        auto tiles = fillDividedTiles(op, neutralTiling, outputShape);
        if (mlir::failed(tiles)) {
            return mlir::failure();
        }
        log.nest(1).trace("Fallback to neutral tiling while attempting prefetching: {0}", neutralTiling);
        return tiles;
    };

    auto adjustTilingNumberOnDim = [&](Shape& left, Shape& right, Dim dimToTile) -> bool {
        while (compareShape(left, right, dimToTile)) {
            // Left not supported; right supported
            // Now search for a smaller supported than right
            auto middle = findMiddleTile(left, right, dimToTile);
            if (!compareShape(left, middle, dimToTile) || !compareShape(middle, right, dimToTile)) {
                log.trace("Search finished, left {0}, middle {1}, right {2}", left, middle, right);
                return true;
            }
            log.trace("Iter left {0}, right {1}, middle {2}", left, right, middle);
            if (mlir::succeeded(isSupportedTileSize(op, middle, tilingMode, log))) {
                right = middle;
                nTilesOnDim = std::move(middle);
            } else {
                left = std::move(middle);
            }
        }
        return false;
    };

    // For small layers, binary search is less efficient than linear search
    // Try the first small tiling numbers first to fix the compile time increases
    auto tryFirstNSmallTiles = [&](Shape& smallTiling, Dim curDimToTile) -> bool {
        auto cnt = 0;
        while (cnt <= LINEAR_SEARCH_TIMES) {
            cnt++;
            dimPlus(smallTiling, curDimToTile, dimToAlign, dimAlignment, outputShape, maxNumTiles, log);
            if (mlir::succeeded(isSupportedTileSize(op, smallTiling, tilingMode, log))) {
                return true;
            }
        }
        return false;
    };

    auto left = nTilesOnDim;
    auto right = nTilesOnDim;
    // Binary search the minimum tiling number to meet the tiling mode requirement
    while (tileDimIter < tileDimOrder.end()) {
        log.trace("searching tileDim {0}", *tileDimIter);
        right[*tileDimIter] = maxNumTiles[(*tileDimIter).ind()];
        ensureNTilesIsCompatibleWithMultiCluster(op, left, *tileDimIter, outputShape, tilingMode, log);
        ensureNTilesIsCompatibleWithMultiCluster(op, right, *tileDimIter, outputShape, tilingMode, log);
        log.trace("original left {0} and right {1}", left, right);
        // Check edge to make sure the left bound is unsupported and right bound is supported
        if (mlir::succeeded(isSupportedTileSize(op, left, tilingMode, log))) {
            nTilesOnDim = left;
            log.trace("Find left tileDim supported {0}", left);
            break;
        }
        if (tryFirstNSmallTiles(left, *tileDimIter)) {
            nTilesOnDim = left;
            log.trace("Find small tiling supported {0}", left);
            break;
        }
        if (mlir::failed(isSupportedTileSize(op, right, tilingMode, log)) || !compareShape(left, right, *tileDimIter)) {
            log.trace("right {0} unsupported, goto next dim {1}", right, *tileDimIter);
            left[*tileDimIter] = right[*tileDimIter];
            tileDimIter++;
            if (tileDimIter == tileDimOrder.end()) {
                log.trace("dim end but tiling not found");
                return storeAndReturnFailure();
            }
            continue;
        }
        nTilesOnDim = right;
        if (!adjustTilingNumberOnDim(left, right, *tileDimIter)) {
            left[*tileDimIter] = right[*tileDimIter];
            tileDimIter++;
            if (tileDimIter == tileDimOrder.end()) {
                return storeAndReturnFailure();
            }
        } else {
            log.trace("Converged to {0}", nTilesOnDim);
            break;
        }
    }

    // Decrease the first dim until the tiling is just supported
    auto firstDimIter = tileDimOrder.begin();
    left = nTilesOnDim;
    left[*firstDimIter] = 1;
    right = nTilesOnDim;

    if (mlir::succeeded(isSupportedTileSize(op, left, tilingMode, log))) {
        nTilesOnDim = std::move(left);
    } else {
        adjustTilingNumberOnDim(left, right, *firstDimIter);
    }
    log.trace("Decreased first dim to {0}", nTilesOnDim);

    auto dimsToTile = getNonOneDim(nTilesOnDim);
    return fillDividedTiles(op, nTilesOnDim, outputShape);
}

OutputTiling vpux::getHWLayerTilingStrategyWithTileDimOrderForPipelining(mlir::Operation* op, ShapeRef outputShape,
                                                                         const OutputTiling& isolatedTiles,
                                                                         Logger log) {
    // For pipelining, continue to increase on the dimension of isolated tiling
    // or on the channel dimension in case of neutral tiling to cover cases with large constants

    VPUX_THROW_WHEN(isolatedTiles.empty(), "Empty tiles for op '{0}'", op->getLoc());
    auto nTilesOnDim = isolatedTiles.front().axis;
    auto dimsToTile = getNonOneDim(nTilesOnDim);
    if (dimsToTile.size() > 1) {
        log.nest(1).trace("Fallback to isolated strategy due to nested tiling: {0}", nTilesOnDim);
        return isolatedTiles;
    }

    auto tilingBuilder = mlir::cast<VPU::TilingBuilderOpInterface>(op);
    const auto& maxNumTiles = tilingBuilder.getMaxNumTiles();
    const auto dimAlignInfo = getAlignDimAndSize(op);
    auto dimToAlign = dimAlignInfo.first;
    auto dimAlignment = dimAlignInfo.second;

    // For pipelining, continue to increase on the dimension of isolated tiling or on the channel dimension in case of
    // neutral tiling to cover cases with large constants
    // #E152765 - generic support for GNCHW
    auto dimActC = mlir::isa<VPU::NCEMatMulOp>(op) ? DimsGroups5D::Act::C : Dims4D::Act::C;
    const auto targetDim = dimsToTile.size() == 0 ? dimActC : dimsToTile[0];
    Shape prefetchableTilesOnDim = nTilesOnDim;
    log.trace("Attempting to generate tiling strategy for pipelining");
    while (mlir::failed(isSupportedTileSize(op, prefetchableTilesOnDim, TilingMode::PIPELINING, log))) {
        if (prefetchableTilesOnDim[targetDim] >= MAX_PREFETCH_TILING_TIME * nTilesOnDim[targetDim] ||
            !isDimLeftToTile(prefetchableTilesOnDim, maxNumTiles, targetDim)) {
            log.nest(3).trace("Fallback to isolated strategy: {0}", nTilesOnDim);
            return isolatedTiles;
        }
        if (targetDim == dimToAlign && dimAlignment != 1) {
            do {
                ++prefetchableTilesOnDim[targetDim];
                if (!isDimLeftToTile(prefetchableTilesOnDim, maxNumTiles, targetDim)) {
                    return isolatedTiles;
                }
            } while (!isSupportedAlignedDivision(outputShape[targetDim], prefetchableTilesOnDim[targetDim],
                                                 dimAlignment));
        } else {
            ++prefetchableTilesOnDim[targetDim];
        }
    }

    log.trace("Pipelining tiling strategy: {0}", prefetchableTilesOnDim);
    auto pipeliningTiles = fillDividedTiles(op, prefetchableTilesOnDim, outputShape);
    if (mlir::failed(pipeliningTiles)) {
        return isolatedTiles;
    }
    return pipeliningTiles.value();
}

mlir::FailureOr<HWTilingStrategies> vpux::getHWLayerTilingStrategiesWithTileDimOrder(mlir::Operation* op,
                                                                                     TilingMode tilingMode,
                                                                                     DimArrRef tileDimOrder,
                                                                                     Logger log) {
    const auto outputType = op->getResult(0).getType().cast<vpux::NDTypeInterface>();

    // NB: bounds information is used for tiling when dynamic shape is passed
    // TODO: E#113258 this logic can be put inside getShape impelementation
    const auto outputShape = [&outputType] {
        const auto origOutputShape = outputType.getShape();
        if (origOutputShape.isDynamic()) {
            const auto bounds =
                    parseIntArrayAttr<int64_t>(outputType.dyn_cast<vpux::BoundedTypeInterface>().getBounds());
            return vpux::Shape(bounds.begin(), bounds.end());
        }
        return vpux::Shape(origOutputShape.begin(), origOutputShape.end());
    }();

    // In case of pipelining, an isolated tiling strategy is first created
    // Then the tiling number would be increased to get a pipelining tiling strategy
    // If no feasible pipelining tiling could be found, fallback to isolated tiling strategy
    const auto tilingModeToCheck = tilingMode == TilingMode::PIPELINING ? TilingMode::ISOLATED : tilingMode;
    auto& cache = VPU::OpTilingCache::instance();
    const auto opHash = cache.calculateOpHash(op, tileDimOrder);
    auto isolatedTiles = cache.getHWLayerTilingStrategyWithTileDimOrder(op, opHash, tilingModeToCheck, tileDimOrder,
                                                                        outputShape, std::nullopt, log);
    if (mlir::failed(isolatedTiles)) {
        return mlir::failure();
    }

    if (tilingMode != TilingMode::PIPELINING) {
        return HWTilingStrategies{isolatedTiles.value(), isolatedTiles.value()};
    }
    auto pipelineTiles = cache.getHWLayerTilingStrategyWithTileDimOrder(op, opHash, TilingMode::PIPELINING,
                                                                        tileDimOrder, outputShape, isolatedTiles, log);
    if (mlir::failed(pipelineTiles)) {
        return HWTilingStrategies{isolatedTiles.value(), isolatedTiles.value()};
    }

    return HWTilingStrategies{isolatedTiles.value(), pipelineTiles.value()};
}

mlir::FailureOr<OutputTiling> vpux::getHWLayerTilingStrategyWithTileDimOrder(mlir::Operation* op, TilingMode tilingMode,
                                                                             DimArrRef tileDimOrder, Logger log) {
    auto tilingResult = vpux::getHWLayerTilingStrategiesWithTileDimOrder(op, tilingMode, tileDimOrder, log);
    if (mlir::failed(tilingResult)) {
        return mlir::failure();
    }
    return tilingResult->pipeliningStrategy;
}

mlir::FailureOr<OutputTiling> vpux::getHWLayerTilingStrategy(mlir::Operation* op, TilingMode tilingMode, Logger log) {
    const auto tileDimOrder = getTileDimOrder(op, tilingMode, log);
    log.nest(2).trace("Tile Dim order is {0}", tileDimOrder);
    return getHWLayerTilingStrategyWithTileDimOrder(op, tilingMode, tileDimOrder, log);
}

bool vpux::isDimLeftToTile(ShapeRef curNumTiles, ArrayRef<int64_t> maxNumTiles, Dim testTileDim) {
    return curNumTiles[testTileDim] < maxNumTiles[testTileDim.ind()];
}

mlir::FailureOr<OutputTiling> vpux::isSupportedTileSize(mlir::Operation* op, ShapeRef nTilesOnDim,
                                                        TilingMode tilingMode, Logger log) {
    const auto outputShape = getShape(op->getResult(0));
    const auto tiles = fillDividedTiles(op, nTilesOnDim, outputShape);
    if (mlir::failed(tiles)) {
        return mlir::failure();
    }

    auto tilingInfo = mlir::dyn_cast<VPU::TilingInfoOpInterface>(op);
    if (tilingInfo == nullptr) {
        return mlir::failure();
    }

    // For isolated tiling isSupportedTiling will check all of the tiles passed to it
    // which results in a lot of time being spent checking identical tiles.
    // Limiting the tiles to only those that have unique shape and will result
    // in unique input tile speeds up compilation significantly.
    // For prefetch and pipelining tiling isSupportedTiling will only check last tile
    // so there is no need to limit number of tiles.
    auto tilesToCheck = tilingMode == TilingMode::ISOLATED ? VPU::getUniqueShapeTilingCandidates(op, tiles.value(), log)
                                                           : tiles.value();

    if (isMultiClusterCompatibleForTiling(op, tilesToCheck, log) &&
        tilingInfo.isSupportedTiling(tilesToCheck, tilingMode, log)) {
        return tiles;
    }

    return mlir::failure();
}

std::pair<Dim, int64_t> vpux::getAlignDimAndSize(mlir::Operation* op) {
    int64_t dimAlignment = 1;
    // #E152765 - generic support for GNCHW
    auto dimToAlign = mlir::isa<VPU::NCEMatMulOp>(op) ? DimsGroups5D::Act::C : Dims4D::Act::C;

    if (auto channelsInfo = mlir::dyn_cast<IE::AlignedChannelsOpInterface>(op)) {
        dimAlignment = channelsInfo.getOutputChannelAlignment();
    }

    // For NCE Permute operation we must have alignment over width because
    // in following passes a Reorder layer will be added that will generate NWCH order
    if (mlir::isa<VPU::NCEPermuteOp>(op)) {
        const auto outputType = op->getResult(0).getType().cast<vpux::NDTypeInterface>();
        dimToAlign = Dims4D::Act::W;
        dimAlignment = VPU::NCEInvariant::getAlignment(outputType.getElementType());
    }

    return std::make_pair(dimToAlign, dimAlignment);
}

bool vpux::isNewTileWithSameCostHasPotentialDMABenefits(mlir::Operation* op, ShapeRef currentTileAxis,
                                                        ShapeRef newTileAxis) {
    auto inputDimOrder = DimsOrder::fromValue(op->getOperand(0));

    VPUX_THROW_UNLESS(currentTileAxis.size() == newTileAxis.size(),
                      "currentTileAxis {0} should have the same dimension size as newTileAxis {1}", currentTileAxis,
                      newTileAxis);

    auto currentMemoryTile = inputDimOrder.toMemoryOrder(currentTileAxis);
    auto newMemoryTile = inputDimOrder.toMemoryOrder(newTileAxis);

    // Choose the tiling strategy with fewer tiles in the lower dimensions
    // Scenario 1: Same tile number, NHWC, Tile [1, 4, 8, 1] is better than Tile [1, 8, 4, 1]
    // Scenario 2: Different tile number, NHWC, Tile [1, 4, 8, 1] is better than Tile [1, 6, 4, 1]
    // Considering the same 'DPU + DMA' cost, there are more possible benefits from the higher continuous data movement
    // with a smaller tiling number in the lower dimensions
    return std::lexicographical_compare(newMemoryTile.rbegin(), newMemoryTile.rend(), currentMemoryTile.rbegin(),
                                        currentMemoryTile.rend());
}

// Allow uneven tiling over OC, such as OC = 80 can be tiled as three tiles [32, 32, 16]
bool vpux::isSupportedAlignedDivision(int64_t dimSize, int64_t tiles, int64_t alignment) {
    auto base = vpux::divUp(dimSize, tiles);
    auto alignedBase = alignValUp(base, alignment);
    auto remainder = dimSize - alignedBase * (tiles - 1);
    return remainder > 0;
}

SmallVector<Dim> vpux::getNonOneDim(ShapeRef inputShape) {
    SmallVector<Dim> nonOneDims;
    for (auto index : irange(inputShape.size())) {
        if (inputShape[Dim(index)] != 1) {
            nonOneDims.push_back(Dim(index));
        }
    }
    return nonOneDims;
}

mlir::FailureOr<Shape> vpux::getNextTiling(Dim targetDim, Dim dimToAlign, int64_t dimAlignment, Shape nTilesOnDim,
                                           ArrayRef<int64_t> maxNumTiles, ShapeRef outputShape) {
    if (!isDimLeftToTile(nTilesOnDim, maxNumTiles, targetDim)) {
        return mlir::failure();
    }
    if (targetDim == dimToAlign && dimAlignment != 1) {
        do {
            ++nTilesOnDim[targetDim];
            if (!isDimLeftToTile(nTilesOnDim, maxNumTiles, targetDim)) {
                return mlir::failure();
            }
        } while (!isSupportedAlignedDivision(outputShape[targetDim], nTilesOnDim[targetDim], dimAlignment));
    } else {
        ++nTilesOnDim[targetDim];
    }
    return nTilesOnDim;
}

std::optional<Dim> vpux::getMaxNonOneDim(ShapeRef inputShape) {
    Dim maxNonOneDim;
    int64_t maxShape = 0;
    for (auto index : irange(inputShape.size())) {
        if (inputShape[Dim(index)] != 1 && inputShape[Dim(index)] > maxShape) {
            maxNonOneDim = Dim(index);
            maxShape = inputShape[Dim(index)];
        }
    }
    if (maxShape <= 1) {
        return std::nullopt;
    }
    return maxNonOneDim;
}

// Optimize the compilation time by optimizing heuristic search permutations before calculating tiling strategy
SmallVector<SmallVector<Dim>> getValidPermutations(mlir::Operation* op, TilingMode tilingMode,
                                                   SmallVector<Dim>& dimensions, Logger log) {
    auto tilingBuilder = mlir::dyn_cast<VPU::TilingBuilderOpInterface>(op);
    VPUX_THROW_WHEN(tilingBuilder == nullptr, "Operation '{0}' doesn't implement TilingBuilderOpInterface",
                    op->getName());
    // Generate 1-dim and 2-dim permutations, skip 3-dim permutations
    // dimensions = [C, H, W]
    // oneDimPermutations = [C], [H], [W]
    // twoDimPermutations = [C, H], [C, W], [H, W], [H, C], [W, C], [W, H]
    SmallVector<SmallVector<Dim>> oneDimPermutations;
    SmallVector<SmallVector<Dim>> twoDimPermutations;
    SmallVector<SmallVector<Dim>> validPermutationsRes;
    for (const auto& dim : dimensions) {
        oneDimPermutations.push_back({dim});
    }
    for (size_t i = 0; i < dimensions.size(); ++i) {
        for (size_t j = 0; j < dimensions.size(); ++j) {
            if (i != j) {
                twoDimPermutations.push_back({dimensions[i], dimensions[j]});
            }
        }
    }

    const auto outputShape = getShape(op->getResult(0));
    const auto& maxNumTiles = tilingBuilder.getMaxNumTiles();
    const auto tilingModeToCheck = tilingMode == TilingMode::PIPELINING ? TilingMode::ISOLATED : tilingMode;

    Shape maxTilesOnDim(outputShape.size(), 1);
    for (auto dimToTile : dimensions) {
        Shape newTilesOnDim(outputShape.size(), 1);
        newTilesOnDim[dimToTile] = maxNumTiles[dimToTile.ind()];
        // If the number of tiles on current dimension is not supported because of incompatible multicluster
        // strategy decrease the number of tiles until the multicluster strategy is compatible again
        ensureNTilesIsCompatibleWithMultiCluster(op, newTilesOnDim, dimToTile, outputShape, tilingModeToCheck, log);
        maxTilesOnDim[dimToTile] = newTilesOnDim[dimToTile];
    }

    const auto isValidPermutation = [&](ArrayRef<Dim> permutation) -> bool {
        Shape tilesOnDim(outputShape.size(), 1);
        for (auto dimToTile : permutation) {
            tilesOnDim[dimToTile] = maxTilesOnDim[dimToTile];
        }
        // If operation tiling with maxTilesOnDim on current dimension
        // can not fit into CMX, this permutation is invalid
        if (mlir::failed(isSupportedTileSize(op, tilesOnDim, tilingModeToCheck, log))) {
            return false;
        }
        return true;
    };

    // Check 1-dim permutations
    for (const auto& oneDimPermutation : oneDimPermutations) {
        if (isValidPermutation(oneDimPermutation)) {
            validPermutationsRes.push_back(oneDimPermutation);
        }
    }

    // Stop checking 2-dim permutations, if any 1-dim permutations are valid
    if (!validPermutationsRes.empty()) {
        return validPermutationsRes;
    }

    // Check 2-dim permutations
    std::set<std::set<Dim>> invalidPermutationSet;
    std::set<std::set<Dim>> validPermutationSet;
    for (const auto& twoDimPermutation : twoDimPermutations) {
        std::set<Dim> permutationSet(twoDimPermutation.begin(), twoDimPermutation.end());
        // If [C, H] is invalid, [H, C] is also invalid
        if (invalidPermutationSet.find(permutationSet) != invalidPermutationSet.end()) {
            continue;
        }
        // If [C, H] is valid, [H, C] is also valid
        if (validPermutationSet.find(permutationSet) != validPermutationSet.end()) {
            validPermutationsRes.push_back(twoDimPermutation);
            continue;
        }
        if (!isValidPermutation(twoDimPermutation)) {
            invalidPermutationSet.insert(permutationSet);
        } else {
            validPermutationSet.insert(permutationSet);
            validPermutationsRes.push_back(twoDimPermutation);
        }
    }
    return validPermutationsRes;
}

SmallVector<Dim> getValidNonOneDim(ShapeRef inputShape, DimArrRef tileDimOrder) {
    SmallVector<Dim> nonOneDims;
    for (auto dim : tileDimOrder) {
        if (inputShape[dim] != 1) {
            nonOneDims.push_back(dim);
        }
    }
    return nonOneDims;
}

SmallVector<OutputTiling> vpux::getAllHWLayerTilingStrategies(mlir::Operation* op, TilingMode tilingMode,
                                                              DimArrRef tileDimOrder, Logger log) {
    SmallVector<OutputTiling> feasibleStrategies;
    const auto outputShape = getShape(op->getResult(0));
    auto dimensions = getValidNonOneDim(outputShape, tileDimOrder);
    auto validPermutations = getValidPermutations(op, tilingMode, dimensions, log);

    auto insertStrategy = [&](const OutputTiling& strategy) {
        if (std::find(feasibleStrategies.begin(), feasibleStrategies.end(), strategy) == feasibleStrategies.end() &&
            getNonOneDim(strategy[0].axis).size() >= 1) {
            feasibleStrategies.push_back(strategy);
        }
    };

    for (const auto& permutation : validPermutations) {
        auto tileStrategy = vpux::getHWLayerTilingStrategiesWithTileDimOrder(op, tilingMode, permutation, log);
        if (mlir::succeeded(tileStrategy)) {
            insertStrategy(tileStrategy->isolatedStrategy);
            insertStrategy(tileStrategy->pipeliningStrategy);
        }
    }

    return feasibleStrategies;
}
