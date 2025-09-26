//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/attributes/strides.hpp"
#include "vpux/compiler/core/layers.hpp"
#include "vpux/utils/core/format.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/BuiltinAttributes.h>

namespace vpux::IE {
enum class InterpolateMode : uint64_t;
enum class InterpolateCoordMode : uint64_t;
enum class InterpolateNearestMode : uint64_t;
}  // namespace vpux::IE

namespace vpux {

// Experimental number to avoid memory fragmentation when generating tiling.
// This one is also used in memory check of long-term spilling.
static constexpr double FRAGMENTATION_AVOID_RATIO = 0.9;

// Experimental number to avoid memory fragmentation when pipelining
static constexpr double FRAGMENTATION_AVOID_RATIO_PIPELINING = 0.85;

// Experimental number to avoid memory fragmentation caused by large activations (input & output) when pipelining
static constexpr double FRAGMENTATION_AVOID_RATIO_PIPELINING_LARGE_ACTIVATION = 0.27;
static constexpr double FRAGMENTATION_AVOID_RATIO_PIPELINING_LARGE_ACTIVATION_MAXPOOL = 0.2;

// Experimental number to define large constant size
// The constant filter is considered as large constant value
// when its size is bigger than CMXSize*LARGE_CONST_THRESHOLD_RATIO
static constexpr double LARGE_CONST_THRESHOLD_RATIO = 0.25;

// An experimental number from activation prefetch pass.
// The purpose is to avoid excessive tiling.
static constexpr int MAX_PREFETCH_TILING_TIME = 3;

// An experimental number from large data (weights & input & output) pipelining.
// The purpose is to avoid excessive tiling.
static constexpr int MAX_PIPELINE_TILING_TIME_FOR_LARGE_DATA = 2;

// Experimental number to avoid excessive tiling in output pipeline tiling
static constexpr int MAX_OUTPUT_PIPELINE_TILING_TIME = 10;

// Experimental number to avoid long compilation time caused by excessive tiling.
static constexpr int MAX_EXCESSIVE_TILING_TIME = 3;

// Track [E#87286]
// Experimental number to avoid spilling in vertical fusion
static constexpr double VF_CONST_RATIO = 0.55;

// Experimental number to avoid spilling in vertical fusion
static constexpr double VF_LARGEST_OP_MEM_RATIO = 0.6;

// Experimental number to avoid fragmentation for vertical fusion pipelining
static constexpr double FRAGMENTATION_AVOID_RATIO_VF_PIPELINING = 0.37;

// Experimental number to get accurate NCEEltwise VPUNN cost
// Track [E#98656]
static constexpr double NCEELTWISE_DPU_COST_RATIO = 2.5;

// Experimental number to get accurate NCEDWCONV VPUNN cost
// Track [E#117314]
static constexpr double NCEDWCONV_DPU_COST_RATIO = 2;
// Track [E#144661]
static constexpr double NCEDWCONV_HK_DPU_COST_RATIO = 1.6;

// Track [E#148159]
static constexpr double SOK_NO_BROADCAST_DPU_COST_RATIO = 1.2;

// Experimental number to get accurate ACT-SPARSITY VPUNN cost
// Track [E#117195]
static constexpr double ACTSPARSE_DPU_COST_RATIO = 2;

// Experimental number for reducemin to get better DPU performance than SHAVE
// Track [E#126141]
static constexpr double REDUCEMIN_DPU_THRESHOLD = 96 * 1024;

// Experimental numbers to correct Convolution SOK cost
static constexpr double NCECONV_DPU_SOK_COST_RATIO = 1.1;
static constexpr double NCECONV_DPU_SOK_OC_TO_SPATIAL_RATIO = 1.1;

// An experimental number for tiling strategy searching algorithms choice
// Linear search for the first several times
// If tilling strategy is not found, then binary search
static constexpr int LINEAR_SEARCH_TIMES = 3;

// An experimental number to control if we parallelize the loop calculating tiling costs
// Avoids compile time regression from multi-threading on small numbers of ops
static constexpr int MIN_OPS_FOR_PARALLEL_TILING = 100;

//
// Tiling Mode
//

enum class TilingMode {
    ISOLATED,    // (default) Split each original layer isolated with no heuristics or tweaks
    PIPELINING,  // Create more tiles to enable DMA/DPU overlapping between sub-tiles of one operation
    PREFETCHING  // Create more tiles to enable DMA/DPU overlapping between child and parent operations
};

inline StringRef getTilingModeStr(TilingMode mode) {
    switch (mode) {
    case TilingMode::ISOLATED:
        return StringRef("ISOLATED");
    case TilingMode::PIPELINING:
        return StringRef("PIPELINING");
    case TilingMode::PREFETCHING:
        return StringRef("PREFETCHING");
    default:
        VPUX_THROW("Tiling mode name is not defined");
    }
}

//
// TileInfo
//

struct TileInfo final {
    Shape shape;
    Shape offsets;
    Shape axis;
    // This flag represents a real tile by a tiling process and offsets & axis are meaningful
    bool isCompletedTile = false;

    TileInfo() = delete;

    explicit TileInfo(size_t rank): shape(rank), offsets(rank), axis(rank) {
    }

    explicit TileInfo(ShapeRef shape): shape(shape.raw()), offsets(shape.size(), 0), axis(shape.size(), 1) {
    }

    explicit TileInfo(ShapeRef shape, ShapeRef offsets, ShapeRef axis)
            : shape(shape.raw()), offsets(offsets.raw()), axis(axis.raw()) {
    }

    explicit TileInfo(ShapeRef shape, ShapeRef offsets, ShapeRef axis, bool isCompletedTile)
            : shape(shape.raw()), offsets(offsets.raw()), axis(axis.raw()), isCompletedTile(isCompletedTile) {
    }

    void printFormat(llvm::raw_ostream& stream) const {
        printTo(stream, "Tile [shape = {0}, offsets = {1}, axis = {2}]", shape, offsets, axis);
    }

    bool operator==(const TileInfo& other) const {
        return shape == other.shape && offsets == other.offsets && axis == other.axis;
    }

    bool operator!=(const TileInfo& other) const {
        return !(*this == other);
    }

    bool operator<(const TileInfo& other) const {
        if (shape != other.shape) {
            return shape < other.shape;
        } else if (offsets != other.offsets) {
            return offsets < other.offsets;
        }
        return axis < other.axis;
    }
};

// Operation output tiles information
using OutputTiling = SmallVector<TileInfo>;

// helper function to generate a set of tiles from dividing a shape. A shape divided across multiple
// dimensions will generate a set of tiles, each having its own size and offsets. Additionally an alignment
// can be specified per each dimension.
mlir::FailureOr<OutputTiling> fillDividedTiles(ShapeRef divisors, ShapeRef orig,
                                               std::optional<ArrayRef<int64_t>> alignment = std::nullopt,
                                               bool unrollSpatialFirst = false);
mlir::FailureOr<OutputTiling> fillDividedTiles(mlir::Operation* op, ShapeRef divisors, ShapeRef shape);

bool isSpatialFirstNestedTiling(mlir::Operation* op, ShapeRef divisors);
bool isWeightsFirstNestedTiling(mlir::Operation* op, ShapeRef divisors);
//
// PadInfo
//

struct PadInfo final {
    int32_t left = 0;
    int32_t right = 0;
    int32_t top = 0;
    int32_t bottom = 0;
    int32_t front = 0;  // 5D depth padding
    int32_t back = 0;   // 5D depth padding
    bool is5D = false;  // Flag to indicate 5D vs 4D usage

    PadInfo() = default;

    // 4D constructor
    PadInfo(int64_t left, int64_t right, int64_t top, int64_t bottom)
            : left(left), right(right), top(top), bottom(bottom), is5D(false) {
    }

    // 5D constructor
    PadInfo(int64_t left, int64_t right, int64_t top, int64_t bottom, int64_t front, int64_t back)
            : left(left), right(right), top(top), bottom(bottom), front(front), back(back), is5D(true) {
    }

    PadInfo(mlir::ArrayAttr pads_begin, mlir::ArrayAttr pads_end)
            : is5D(pads_begin.size() == 3 && pads_end.size() == 3) {
        if (is5D) {
            top = mlir::cast<mlir::IntegerAttr>(pads_begin[Dims5D::PadsBegin::Top.ind()]).getValue().getSExtValue();
            bottom = mlir::cast<mlir::IntegerAttr>(pads_end[Dims5D::PadsEnd::Bottom.ind()]).getValue().getSExtValue();
            left = mlir::cast<mlir::IntegerAttr>(pads_begin[Dims5D::PadsBegin::Left.ind()]).getValue().getSExtValue();
            right = mlir::cast<mlir::IntegerAttr>(pads_end[Dims5D::PadsEnd::Right.ind()]).getValue().getSExtValue();
            front = mlir::cast<mlir::IntegerAttr>(pads_begin[Dims5D::PadsBegin::Front.ind()]).getValue().getSExtValue();
            back = mlir::cast<mlir::IntegerAttr>(pads_end[Dims5D::PadsEnd::Back.ind()]).getValue().getSExtValue();
        } else {
            top = mlir::cast<mlir::IntegerAttr>(pads_begin[Dims4D::PadsBegin::Top.ind()]).getValue().getSExtValue();
            bottom = mlir::cast<mlir::IntegerAttr>(pads_end[Dims4D::PadsEnd::Bottom.ind()]).getValue().getSExtValue();
            left = mlir::cast<mlir::IntegerAttr>(pads_begin[Dims4D::PadsBegin::Left.ind()]).getValue().getSExtValue();
            right = mlir::cast<mlir::IntegerAttr>(pads_end[Dims4D::PadsEnd::Right.ind()]).getValue().getSExtValue();
        }
    }

    mlir::DenseMap<int64_t, std::pair<int64_t, int64_t>> toPadByDims() const {
        if (is5D) {
            return {{Dims5D::Act::H.ind(), {top, bottom}},
                    {Dims5D::Act::W.ind(), {left, right}},
                    {Dims5D::Act::D.ind(), {front, back}}};
        } else {
            return {{Dims4D::Act::H.ind(), {top, bottom}}, {Dims4D::Act::W.ind(), {left, right}}};
        }
    }

    bool enabled() const {
        if (is5D) {
            return left != 0 || right != 0 || top != 0 || bottom != 0 || front != 0 || back != 0;
        } else {
            return left != 0 || right != 0 || top != 0 || bottom != 0;
        }
    }

    bool operator==(const PadInfo& other) const {
        if (is5D != other.is5D) {
            return false;
        }
        bool base = left == other.left && right == other.right && top == other.top && bottom == other.bottom;
        return is5D ? base && front == other.front && back == other.back : base;
    }

    bool operator!=(const PadInfo& other) const {
        return !(*this == other);
    }

    void printFormat(llvm::raw_ostream& stream) const {
        if (is5D) {
            printTo(stream, "PadInfo5D [left = {0}, right = {1}, top = {2}, bottom = {3}, front = {4}, back = {5}]",
                    left, right, top, bottom, front, back);
        } else {
            printTo(stream, "PadInfo [left = {0}, right = {1}, top = {2}, bottom = {3}]", left, right, top, bottom);
        }
    }
};

PadInfo backInferPadsTile(const TileInfo& outputTile, ShapeRef inShape, const PadInfo& origPads,
                          ArrayRef<int64_t> kernel, ArrayRef<int64_t> strides);

//
// TilingInfo
//

struct TilingInfo final {
    SmallVector<TileInfo> tiles;
    std::optional<PadInfo> pads;

    explicit TilingInfo(ArrayRef<TileInfo> tiles): tiles(tiles.begin(), tiles.end()) {
    }

    explicit TilingInfo(ArrayRef<TileInfo> tiles, PadInfo pads): tiles(tiles.begin(), tiles.end()), pads(pads) {
    }
};

// Operation inputs tiling information
using InputTiling = TilingInfo;

//
// Convolution tiling
//

InputTiling backInferConvTile(const TileInfo& outputTile, ShapeRef origInputShape, ShapeRef origFilterShape,
                              ShapeRef origBiasShape, mlir::ArrayAttr strides, const PadInfo& origPadding);

InputTiling backInferGroupConvTile(const TileInfo& outputTile, ShapeRef origInputShape, ShapeRef origFilterShape,
                                   ShapeRef origBiasShape, mlir::ArrayAttr strides, const PadInfo& origPadding,
                                   int64_t groups);

//
// NCEMatMul tiling
//

InputTiling backInferMatMulTile(const TileInfo& outputTile, ShapeRef origInputShape, ShapeRef origFilterShape,
                                mlir::ArrayAttr strides, const PadInfo& origPadding);

//
// 5D Pooling tiling
//

InputTiling backInfer5DPoolTile(const TileInfo& outputTile, ShapeRef origInputShape, mlir::ArrayAttr kernel_size,
                                mlir::ArrayAttr strides, const PadInfo& origPadding);

//
// Pooling tiling
//

InputTiling backInferPoolTile(const TileInfo& outputTile, ShapeRef origInputShape, mlir::ArrayAttr kernel_size,
                              mlir::ArrayAttr strides, const PadInfo& origPadding);

//
// Reduce tiling
//

InputTiling backInferReduceTile(const vpux::TileInfo& outputTile, ShapeRef inShape, mlir::ArrayAttr axesAttr,
                                bool keepDims);

//
// Interpolate tiling
//

InputTiling backInferInterpolateTile(const vpux::TileInfo& outputTile, ArrayRef<int64_t> initialInputDims,
                                     ArrayRef<int64_t> initialOutputDims, ArrayRef<int64_t> initialInputOffsets,
                                     ArrayRef<int64_t> initialOutputOffsets, ArrayRef<int64_t> currentInputDims,
                                     std::optional<ArrayRef<int64_t>> coordinatesDims,
                                     std::optional<ArrayRef<int64_t>> lambdasDims,
                                     vpux::IE::InterpolateMode interpolateMode,
                                     vpux::IE::InterpolateCoordMode coordMode,
                                     vpux::IE::InterpolateNearestMode nearestMode, vpux::Logger log);

//
// Gather tiling
//

InputTiling backInferGatherTile(const vpux::TileInfo& outputTile, const ShapeRef& origInputShape,
                                const ShapeRef& origIndicesShape, int64_t axisValue, int64_t batchDims,
                                bool hasAxisTensor, const int64_t indicesRank, vpux::Logger log);

//
// GatherND tiling
//

mlir::ArrayAttr packOriginalShapeAttrForGatherNDSwOp(mlir::ArrayAttr originalShapeAttr, mlir::MLIRContext* ctx);
std::optional<Shape> extractOriginalShapeAttrFromGatherNDSwOp(mlir::ArrayAttr originalShapeAttr);

InputTiling backInferGatherNDTile(const vpux::TileInfo& outputTile, ShapeRef origInputShape, ShapeRef origIndicesShape,
                                  const int64_t batchDims, ShapeRef originalShapeAttrVal, vpux::Logger log);

//
// GatherDMA tiling
//

InputTiling backInferGatherDMATile(const vpux::TileInfo& outputTile, ShapeRef origInputShape, ShapeRef origIndicesShape,
                                   int64_t axisValue, bool hasAxisTensor, vpux::Logger log);

//
// GatherElements tiling
//
InputTiling backInferGatherElementsTile(const vpux::TileInfo& outputTile, const ShapeRef& origInputShape,
                                        const ShapeRef& origIndicesShape, int64_t axisValue, const int64_t indicesRank,
                                        vpux::Logger log);

//
// GridSample tiling
//
InputTiling backInferGridSampleTile(const vpux::TileInfo& outputTile, ShapeRef origInputShape, ShapeRef origGridShape,
                                    vpux::Logger);

//
// Pad tiling
//

InputTiling backInferPadTile(const vpux::TileInfo& outputTile, const ShapeRef origInputShape,
                             const ShapeRef origOutputShape, const ShapeRef origPadsBegin, const ShapeRef origPadsEnd,
                             vpux::Logger log);

void updatePadOpAttrsAfterTiling(const ShapeRef outShape, const TileInfo& outTile, SmallVector<int64_t>& padsBegin,
                                 SmallVector<int64_t>& padsEnd);

//
// DepthToSpace tiling
//

InputTiling backInferDepthToSpaceTile(const vpux::TileInfo& outputTile, ShapeRef origInputShape, int64_t blockSize,
                                      vpux::Logger);

//
// DimRange
//

struct DimRange final {
    int64_t begin = 0;
    int64_t end = 0;

    DimRange() = default;
    DimRange(int64_t begin, int64_t end): begin(begin), end(end) {
        VPUX_THROW_UNLESS(end >= begin, "Got wrong dimension range [{0}, {1})", begin, end);
    }

    int64_t length() const {
        return end - begin;
    }

    bool intersects(const DimRange& other) const {
        return (begin < other.end) && (other.begin < end);
    }

    bool contains(const DimRange& other) const {
        return (begin <= other.begin) && (end >= other.end);
    }

    // Represents `other` range to ROI of the current one.
    DimRange asROI(const DimRange& other) const {
        VPUX_THROW_UNLESS(contains(other), "DimRange '{0}' is not contained in '{1}'", other, *this);
        return {other.begin - begin, other.end - begin};
    }

    bool operator==(const DimRange& other) const {
        return begin == other.begin && end == other.end;
    }
    bool operator!=(const DimRange& other) const {
        return !(*this == other);
    }

    void printFormat(llvm::raw_ostream& stream) const {
        printTo(stream, "DimRange [{0}, {1})", begin, end);
    }
};

std::optional<std::pair<int64_t, int64_t>> spatialOutputForInputWindowSize(const std::pair<int64_t, int64_t>& inputHW,
                                                                           ArrayRef<int64_t> kernel,
                                                                           ArrayRef<int64_t> strides,
                                                                           const PadInfo& pads);

//
// Tiling utils
//

std::tuple<DimRange, int64_t, int64_t> inputForOutputDim(const DimRange& output, int64_t kernel, int64_t stride,
                                                         const DimRange& initialInputRange, int64_t padBefore,
                                                         int64_t padAfter);

template <typename AlignFunc>
SmallVector<int64_t> alignShape(ArrayRef<int64_t> shape, std::optional<ArrayRef<int64_t>> alignment,
                                AlignFunc alignFunc) {
    auto alignedShape = to_small_vector(shape);
    if (!alignment.has_value()) {
        return alignedShape;
    }
    std::transform(shape.begin(), shape.end(), alignment.value().begin(), alignedShape.begin(), alignFunc);
    return alignedShape;
}
SmallVector<Strides> adaptStrides(ShapeRef origShape, StridesRef origStrides, ArrayRef<Shape> adaptedShapes,
                                  DimsOrder dimsOrder);

//
// EltwiseOp
//

SmallVector<int64_t> getMaxNumTiles(mlir::Operation* op, bool checkMinimalWidthAndHeight = false,
                                    bool checkWorkloadEfficiency = false);
InputTiling backInferEltwiseTile(mlir::Operation* op, const vpux::TileInfo& outputTile);

// SWLayer

mlir::FailureOr<OutputTiling> getSWLayerTilingStrategyWithTileDimOrder(mlir::Operation* op, TilingMode& tilingMode,
                                                                       DimArrRef tileDimOrder, Logger log,
                                                                       ArrayRef<int64_t> maxTilesPerDim = {});
mlir::FailureOr<OutputTiling> getSWLayerTilingStrategy(mlir::Operation* op, TilingMode tilingMode, Logger log,
                                                       ArrayRef<int64_t> maxTilesPerDim = {});

InputTiling getSWLayerInputTiles(mlir::Operation* op, const vpux::TileInfo& outputTile);
SmallVector<int64_t> getMaxNumTilesWithAxesExclusion(mlir::Operation* op, ArrayRef<int64_t> axes);

// HWLayer
struct HWTilingStrategies {
    OutputTiling isolatedStrategy;
    OutputTiling pipeliningStrategy;
};

mlir::FailureOr<OutputTiling> getHWLayerTilingStrategyWithTileDimOrderForIsolatedOrPrefetch(
        mlir::Operation* op, TilingMode tilingMode, DimArrRef tileDimOrder, ShapeRef outputShape, Logger log);

mlir::FailureOr<OutputTiling> getHWLayerTilingStrategyWithTileDimOrderForPipelining(mlir::Operation* op,
                                                                                    ShapeRef outputShape,
                                                                                    const OutputTiling& isolatedTiles,
                                                                                    Logger log);

mlir::FailureOr<HWTilingStrategies> getHWLayerTilingStrategiesWithTileDimOrder(mlir::Operation* op,
                                                                               TilingMode tilingMode,
                                                                               DimArrRef tileDimOrder, Logger log);
mlir::FailureOr<OutputTiling> getHWLayerTilingStrategyWithTileDimOrder(mlir::Operation* op, TilingMode tilingMode,
                                                                       DimArrRef tileDimOrder, Logger log);
mlir::FailureOr<OutputTiling> getHWLayerTilingStrategy(mlir::Operation* op, TilingMode tilingMode, Logger log);

DimArr getTileDimOrder(mlir::Operation* op, TilingMode tilingMode, Logger log);
DimArr getTileDimOrderND(MemShape memShape, DimsOrder dimOrder);

// utils for tiling strategy

/*
 * Check if the operation's existing MultiCluster strategy still compatible with the tiling strategy
 * if the operation does not have MultiCluster strategy, return true
 */
bool isMultiClusterCompatibleForTiling(mlir::Operation* op, const OutputTiling& tiles, Logger log);

/*
 * Check if the shape can be split on the specific dimension
 */
bool isDimLeftToTile(ShapeRef curNumTiles, ArrayRef<int64_t> maxNumTiles, Dim testTileDim);

/*
 * Check if the tile size is suitable for large activation operations
 * to avoid memory fragmentation
 */
bool isSupportedTileSizeForLargeActivation(mlir::Operation* origOp, ShapeRef nTilesOnDim, double fragmentRatio,
                                           Logger log);

/*
 * Check if the tile size is suitable for large activation operations
 * to avoid memory fragmentation
 */
bool isSupportedTileSizeForLargeActivation(mlir::Operation* origOp, ShapeRef nTilesOnDim, Logger log);

/*
 * Check if the tiling strategy is supported
 * Consider alignment, multi-cluster strategy and memory size
 */
mlir::FailureOr<OutputTiling> isSupportedTileSize(mlir::Operation* op, ShapeRef nTilesOnDim, TilingMode tilingMode,
                                                  Logger log);

/*
 * Get the required alignment information for the op
 * @returns {dimension to align, alignment size}
 */
std::pair<Dim, int64_t> getAlignDimAndSize(mlir::Operation* op);

/*
 * Gets alignment for operation based on tiling
 */
std::optional<SmallVector<int64_t>> getAlignment(mlir::Operation* op, ShapeRef divisors, ShapeRef shape);

/*
 * Check if the shape size is divisible with alignment
 */
bool isSupportedAlignedDivision(int64_t dimSize, int64_t tiles, int64_t alignment);

/*
 * Check if the new tile has the same cost as the historical one and has benefits for DMA
 * It is possible that the tiling strategy cost calculated by VPUNN is the same
 * However, due to the stride, the DMA cost model may not be very accurate
 * Choose the tiling strategy at higher dimensions that has potential benefits for DMA
 */
bool isNewTileWithSameCostHasPotentialDMABenefits(mlir::Operation* op, ShapeRef currentTileAxis, ShapeRef newTileAxis);

/*
 * Get the dimensions greater than 1
 */
SmallVector<Dim> getNonOneDim(ShapeRef inputShape);

/*
 * Get the dimensions greater than 1 with the identity (NCHW for 4D) tiling order
 */
SmallVector<Dim> getSCFTilingOrderedDims(mlir::Operation* operation, ShapeRef tiling);

/*
 * Get the dimension with the maximum size in all non-one dimensions
 */
std::optional<Dim> getMaxNonOneDim(ShapeRef inputShape);

/*
 * Get the next supported tiling number
 * Consider alignment
 */
mlir::FailureOr<Shape> getNextTiling(Dim targetDim, Dim dimToAlign, int64_t dimAlignment, Shape nTilesOnDim,
                                     ArrayRef<int64_t> maxNumTiles, ShapeRef outputShape);

/*
Get all feasible tiling strategies for all dim orders for an op
*/
SmallVector<OutputTiling> getAllHWLayerTilingStrategies(mlir::Operation* op, TilingMode tilingMode,
                                                        DimArrRef tileDimOrder, Logger log);

SmallVector<int64_t> divideChannelForSEPDWConv(mlir::Operation* op, int64_t channelSize, int64_t channelDivisor);
}  // namespace vpux
