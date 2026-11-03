//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/NPU40XX/dialect/VPU/IR/ops_interfaces.hpp"
#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/tiling.hpp"
#include "vpux/compiler/dialect/VPU/utils/manual_strategy_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/scf/scf_analysis_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/vf_merge_configuration.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/vertical_fusion_utils.hpp"
#include "vpux/compiler/dialect/core/IR/dynamic_attrs.hpp"
#include "vpux/compiler/dialect/core/IR/tensor_attr.hpp"
#include "vpux/compiler/dialect/core/types.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include <llvm/ADT/StringRef.h>
#include <mlir/Dialect/Affine/Utils.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/Utils/StaticValueUtils.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/Interfaces/TilingInterface.h>
#include <mlir/Support/LLVM.h>

#include <algorithm>
#include <functional>
#include <optional>
#include <utility>

namespace vpux::VPU {

enum class TilePosition { MIDDLE = 0, END = 1, START = 2, FULLBLK = 3 };
constexpr size_t NUMBITS = 2;

// Low resolution (2-tile) data of a dim: a VPU::LowResAttr set by AdjustBlockSizeForScfTiling on the scf.for of the
// dim, and the id of that data on the pre-loop low resolution predicate (an i64 attribute)
constexpr llvm::StringLiteral LOW_RES_ATTR_NAME = "vpux.lowres";
constexpr llvm::StringLiteral LOW_RES_PRED_ATTR_NAME = "vpux.lowres_pred";
// Id of the low resolution data on the single-tile assert of the dim. The assert is removed once the 2-tile range is
// handled in every loop that can run a low resolution tile
constexpr llvm::StringLiteral LOW_RES_ASSERT_ATTR_NAME = "vpux.lowres_assert";
// Ids of the low resolution dims on which an input slice belongs to the FULLBLK case (a dense i64 array). Its offset on
// those dims is set later to the output tile offset, in the loops that can run a low resolution tile
constexpr llvm::StringLiteral LOW_RES_SLICE_ATTR_NAME = "vpux.lowres_slice";

int64_t getTilePositionShift(vpux::Dim dim);

mlir::Value encodePointPosition(mlir::OpBuilder& builder, mlir::Location loc, ArrayRef<mlir::Value> values,
                                ArrayRef<mlir::Value> shiftAmounts);
int64_t getEncodedPointPosition(ArrayRef<std::pair<vpux::Dim, TilePosition>> dimPositions);
TilePosition getDimPosition(int64_t pointEncoded, vpux::Dim dim);
int64_t setDimPosition(int64_t pointEncoded, vpux::Dim dim, TilePosition pos);
SmallVector<TilePosition> decodePointPosition(int64_t encoded, ArrayRef<vpux::Dim> dims);

// Range encoding: each dim slot expands to 2*NUMBITS bits {end[1:0], start[1:0]}.
// rangeStartShift(dim) = getTilePositionShift(dim) * 2
// rangeEndShift(dim)   = getTilePositionShift(dim) * 2 + NUMBITS
int64_t encodeRangePosition(vpux::Dim dim, int64_t currentValue, TilePosition start, TilePosition end);
std::pair<TilePosition, TilePosition> decodeRangePosition(int64_t rangeEncoded, vpux::Dim dim);

// TilePosition bit layout: bit1 = keepStart, bit0 = keepEnd.
inline std::pair<bool, bool> getTilePaddingMask(TilePosition pos) {
    auto v = static_cast<int>(pos);
    return {(v >> 1) & 1, v & 1};
}

/** @brief Information about a tile.

    The structure encapsulates data of offsets, shape and tile axis
    for a tensor represented as mlir::OpFoldResult
*/

using SCFShape = SmallVector<mlir::OpFoldResult>;
using SCFShapeRef = ArrayRef<mlir::OpFoldResult>;

const llvm::StringRef SKIP_CONNECTION_SLICE_MARKER_ATTR_NAME = "skip_connection_slice";

struct SCFTileInfo {
    SCFShape shape;
    SCFShape offsets;
    SCFShape axis;
    Bounds bounds;

    SCFTileInfo() = delete;

    explicit SCFTileInfo(SCFShapeRef shape, SCFShapeRef offsets, SCFShapeRef axis, BoundsRef bounds = {})
            : shape(shape), offsets(offsets), axis(axis), bounds(bounds) {
    }

    explicit SCFTileInfo(ArrayRef<int64_t> shapeInt, mlir::OpBuilder& builder)
            : shape(mlir::getAsIndexOpFoldResult(builder.getContext(), shapeInt)),
              offsets(SCFShape(shapeInt.size(), builder.getIndexAttr(0))),
              axis(SCFShape(shapeInt.size(), builder.getIndexAttr(1))) {
    }

    explicit SCFTileInfo(SCFShapeRef shape, mlir::OpBuilder& builder)
            : shape(shape),
              offsets(SCFShape(shape.size(), builder.getIndexAttr(0))),
              axis(SCFShape(shape.size(), builder.getIndexAttr(1))) {
    }

    void printFormat(llvm::raw_ostream& stream) const {
        printTo(stream, "SCFTile [shape = {0}, offsets = {1}, axis = {2}, bounds = {3}]", shape, offsets, axis, bounds);
    }
};

// SCFTilingInfo owns its tiles/pads, so ArrayRef (non-owning view) shouldn't be used here. Each ctor
// comes as a const&/&& pair rather than a single by-value parameter: SCFTileInfo/SmallVector<SCFTileInfo>
// are large (Coverity PASS_BY_VALUE), and a reference stays cheap to pass regardless of the pointee size,
// while the && overload still moves for real.
struct SCFTilingInfo {
    explicit SCFTilingInfo(const SCFTileInfo& tileValue) {
        tiles.push_back(tileValue);
    }
    explicit SCFTilingInfo(SCFTileInfo&& tileValue) {
        tiles.push_back(std::move(tileValue));
    }
    explicit SCFTilingInfo(const SCFTileInfo& tileValue, const SCFShape& padsValue): pads(padsValue) {
        tiles.push_back(tileValue);
    }
    explicit SCFTilingInfo(SCFTileInfo&& tileValue, SCFShape&& padsValue): pads(std::move(padsValue)) {
        tiles.push_back(std::move(tileValue));
    }
    explicit SCFTilingInfo(const SmallVector<SCFTileInfo>& tilesValue): tiles(tilesValue) {
    }
    explicit SCFTilingInfo(SmallVector<SCFTileInfo>&& tilesValue): tiles(std::move(tilesValue)) {
    }
    explicit SCFTilingInfo(const SmallVector<SCFTileInfo>& tilesValue, const SCFShape& padsValue)
            : tiles(tilesValue), pads(padsValue) {
    }
    explicit SCFTilingInfo(SmallVector<SCFTileInfo>&& tilesValue, SCFShape&& padsValue)
            : tiles(std::move(tilesValue)), pads(std::move(padsValue)) {
    }

    SmallVector<SCFTileInfo> tiles;
    std::optional<SCFShape> pads;
};

// Tracks deferred ExtractSlice replacements for skip-connections during SCF tile+fuse.
struct PendingSliceReplacement {
    mlir::Operation* biggestUserOp{};          // The user op that requires the biggest tile for this skip connection
    mlir::Value tiledValue;                    // The value from the biggest tile
    bool biggestUserTiled = false;             // Flag to indicate if the biggest user has been tiled
    bool allUsersWithTheSameTileSize = false;  // Flag to indicate if all users have the same tile size
    mlir::SetVector<mlir::tensor::ExtractSliceOp>
            relatedExtractSlices;  // ExtractSliceOps related to this skip connection that need replacement
    mlir::tensor::ExtractSliceOp
            biggestTileExtractSlice;  // The ExtractSliceOp that corresponds to the biggest tile, used for replacement
};

using OpTilingOperandsFunc = std::function<void(SCFTilingInfo&)>;
using OpGeneratorFunc = std::function<mlir::Operation*()>;

// @brief Dim value of input/output/weights shape
mlir::OpFoldResult getDimValue(mlir::OpBuilder& builder, mlir::Operation* operation, int64_t dim);

// @brief Calculates tile for weights table based on output tile
SCFTileInfo getWeightsTableSCFTile(mlir::Type origWeightsTableType, mlir::OpBuilder& builder,
                                   const SCFTileInfo& outputTile);

// @brief Builds AffineMap that computes alignValUp(dim, alignment) for a single dimension.
mlir::AffineMap getAlignValUpMap(mlir::OpBuilder& builder, int64_t alignment);

// @brief Const-folds an OpFoldResult, else falls back to a ValueBoundsConstraintSet upper bound.
// Returns std::nullopt for a non-positive extent (not a usable bound for sizing tensors/tiles).
std::optional<int64_t> getConstantOrUpperBound(mlir::OpFoldResult ofr);

/** @brief Restores input tiling from output tile data

    The function calculates input shape, offset and bounds based on
    parameters and shape and offset of output tile
*/
std::pair<std::optional<mlir::Range>, std::optional<int64_t>> solutionForOutputRange(
        mlir::Location loc, mlir::OpBuilder& builder, const SCFTileInfo& outputTile, Dim dim, int64_t kernel,
        int64_t stride, mlir::OpFoldResult origInputSize, int64_t origOutputSize,
        const std::pair<int64_t, int64_t>& origPadding, mlir::OpFoldResult& padBefore, mlir::OpFoldResult& padAfter);

/** @brief Generate slice based on tiling information

    The function generates ExtractSliceOp based on offset and size in tile info
*/
mlir::Value generateTile(mlir::Location loc, mlir::OpBuilder& builder, mlir::Value origInput,
                         const SCFTileInfo& inputTileInfo, SmallVector<mlir::Operation*>& generatedSlices);

/** @brief Return result type after tiling to new shape

    The function extracts result type of operation
    after changing shape
*/
mlir::Type extractResultType(mlir::Type origType, SCFShapeRef newShape, BoundsRef bounds);

/** @brief Return padding value for quantized type

    The function creates padding value for quantized type based on zero point
*/
inline mlir::Value createQuantizedPaddingValue(mlir::Location loc, mlir::OpBuilder& builder,
                                               mlir::quant::QuantizedType qType, vpux::NDTypeInterface tiledType) {
    auto storageType = qType.getStorageType();
    mlir::Type paddingType = storageType;
    if (auto intType = mlir::dyn_cast<mlir::IntegerType>(storageType)) {
        paddingType = builder.getIntegerType(intType.getWidth());
    }

    // A single padding value cannot represent a per-axis type whose zero points differ between channels.
    const auto zeroPoint = vpux::extractSingleZeroPoint(qType);
    VPUX_THROW_UNLESS(zeroPoint.has_value(), "Per-channel quantization with non-uniform zero points is not supported");

    auto constantOp = mlir::arith::ConstantOp::create(builder, loc, builder.getIntegerAttr(paddingType, *zeroPoint));
    return mlir::UnrealizedConversionCastOp::create(builder, loc, tiledType.getElementType(), constantOp.getResult())
            .getResult(0);
}

namespace {

/** @brief Casts a freshly generated tiled operation's result to the statically known
    tile shape when padding left some spatial dims dynamic in the inferred type.

    Uses the static tile sizes from @p outputTile where available and keeps the inferred
    dims otherwise; preserves the inferred bounds for the remaining dynamic dims.
*/
template <class ConcreteOp>
mlir::Operation* castCreatedOperation(mlir::OpBuilder& builder, const SCFTileInfo& outputTile, ConcreteOp generatedOp) {
    auto firstResult = generatedOp->getResult(0);
    auto generatedType = mlir::cast<vpux::NDTypeInterface>(firstResult.getType());

    // Build target shape: use static tile sizes where available, keep inferred dims otherwise.
    // This handles the case where padding makes spatial dims dynamic in the inferred type,
    // but the tile size computation guarantees a known static size.
    SmallVector<int64_t> staticOutputShape;
    bool needsCast = false;
    for (auto idx : irange(outputTile.shape.size())) {
        auto constVal = mlir::getConstantIntValue(outputTile.shape[idx]);
        auto inferredDim = generatedType.getShape()[Dim(idx)];
        if (constVal.has_value()) {
            staticOutputShape.push_back(*constVal);
            if (inferredDim == mlir::ShapedType::kDynamic) {
                needsCast = true;
            }
        } else {
            staticOutputShape.push_back(inferredDim);
        }
    }

    if (!needsCast) {
        return generatedOp.getOperation();
    }

    // Preserve bounds from the inferred type for remaining dynamic dims
    auto bounds = Bounds();
    if (auto boundedType = mlir::dyn_cast<vpux::Core::BoundedTensorType>(firstResult.getType())) {
        auto origBounds = boundedType.getBounds().toValues();
        SmallVector<int64_t> boundsVec;
        for (auto idx : irange(staticOutputShape.size())) {
            boundsVec.push_back(staticOutputShape[idx] != mlir::ShapedType::kDynamic ? staticOutputShape[idx]
                                                                                     : origBounds[Dim(idx)]);
        }
        bounds = Bounds(ArrayRef<int64_t>(boundsVec));
    }

    bool hasDynamicDim = llvm::any_of(staticOutputShape, [](int64_t d) {
        return d == mlir::ShapedType::kDynamic;
    });
    auto correctedTensorDesc = (hasDynamicDim && !bounds.raw().empty())
                                       ? vpux::getTensorAttr(generatedType.getContext(), generatedType.getDimsOrder(),
                                                             generatedType.getMemSpace(), bounds)
                                       : vpux::getTensorAttr(generatedType.getContext(), generatedType.getDimsOrder(),
                                                             generatedType.getMemSpace());

    mlir::Type correctedTiledOutputType =
            mlir::RankedTensorType::get(staticOutputShape, generatedType.getElementType(), correctedTensorDesc);
    return mlir::tensor::CastOp::create(builder, generatedOp.getLoc(), correctedTiledOutputType,
                                        generatedOp->getResult(0))
            .getOperation();
}

// @brief Pad's contribution to a spatial dim's bound: untiled dims always sum (both sides apply every
// tile); tiled dims use max, since a given tile's low and high pad can't both apply at once -- unless
// the tiling could collapse to a single tile along this dim, in which case sum is the only sound choice.
inline int64_t getPadBoundDelta(Dim spatialDim, bool canCollapseToSingleTile, DimArrRef dims,
                                const mlir::DenseMap<int64_t, std::pair<int64_t, int64_t>>& padsByDims,
                                ArrayRef<mlir::OpFoldResult> lows, ArrayRef<mlir::OpFoldResult> highs) {
    const auto& pad = padsByDims.lookup(spatialDim.ind());
    if (!is_contained(dims, spatialDim)) {
        return pad.first + pad.second;
    }
    const auto foldedLow = mlir::getConstantIntValue(lows[spatialDim.ind()]).value_or(pad.first);
    const auto foldedHigh = mlir::getConstantIntValue(highs[spatialDim.ind()]).value_or(pad.second);
    return canCollapseToSingleTile ? foldedLow + foldedHigh : std::max(foldedLow, foldedHigh);
};

// @brief Recomposes sourceSize + low + high into a single affine expression and returns its own true
// maximum (constant-folding it, else evaluating it via OpChainAnalysis). Returns std::nullopt if the
// composed expression can't be resolved either way.
inline std::optional<int64_t> tryComposePaddedBound(mlir::OpBuilder& builder, mlir::Location loc,
                                                    mlir::OpFoldResult sourceSize, mlir::OpFoldResult low,
                                                    mlir::OpFoldResult high, OpChainAnalysis& analysis,
                                                    ValueRangeMap& valueMap) {
    mlir::AffineExpr e0, e1, e2;
    bindDims(builder.getContext(), e0, e1, e2);
    auto paddedSizeMap = mlir::AffineMap::get(3, 0, {e0 + e1 + e2}, builder.getContext());
    auto paddedSize = mlir::affine::makeComposedFoldedAffineApply(builder, appendLoc(loc, "paddedSize"), paddedSizeMap,
                                                                  {sourceSize, low, high});
    if (const auto constVal = mlir::getConstantIntValue(paddedSize)) {
        return *constVal;
    }
    const auto evaluated = analysis.getOpFoldResultValue(paddedSize, valueMap, OpChainAnalysis::MODE::MAX_VALUE);
    if (evaluated.has_value() && !evaluated->empty()) {
        return evaluated->front();
    }
    return std::nullopt;
};

}  // namespace

/** @brief create operation with padding adjustment

    @note If operation has paddings which are not 0, they have to be corrected based on
    position of tile. Unfortunately, in OV based operations paddings have to be known integer attributes,
    they cannot be calculated or created as constant for each case. That's why there must be the structure
    with if-else which identifies how paddings are set.
*/
template <class ConcreteOp>
mlir::Operation* createTiledPaddedOperation(OpGeneratorFunc opGenerator, OpTilingOperandsFunc operandsGenerator,
                                            mlir::OpBuilder& builder, SCFTilingInfo& inputTiling,
                                            const SCFTileInfo& outputTile, DimArrRef dims,
                                            SmallVector<mlir::Value>& tiledOperands, mlir::Operation* operation) {
    const auto isSpatialDim = [](auto dim) {
        return dim.ind() >= static_cast<int32_t>(Dims4D::Act::numSpatialDims);
    };
    if (llvm::none_of(dims, isSpatialDim) || !inputTiling.pads.has_value()) {
        operandsGenerator(inputTiling);
        return opGenerator();
    }

    auto padInfo = toPadInfo(mlir::cast<ConcreteOp>(operation).getPad());
    if (!padInfo.enabled()) {
        operandsGenerator(inputTiling);
        return opGenerator();
    }
    auto loc = operation->getLoc();

    operandsGenerator(inputTiling);
    VPUX_THROW_WHEN(tiledOperands.empty(), "Empty tiled operation for operation");
    auto tiledInput = tiledOperands.front();
    auto tiledType = mlir::cast<vpux::NDTypeInterface>(tiledInput.getType());
    auto qType = mlir::dyn_cast<mlir::quant::QuantizedType>(tiledType.getElementType());

    SmallVector<mlir::OpFoldResult> lows(tiledType.getRank(), builder.getIndexAttr(0));
    SmallVector<mlir::OpFoldResult> highs(tiledType.getRank(), builder.getIndexAttr(0));

    auto padsByDims = padInfo.toPadByDims();
    for (auto index : irange(Dims4D::Act::numSpatialDims)) {
        const auto spatialDim = Dims4D::Act::getSpatialDim(index);
        if (is_contained(dims, spatialDim)) {
            lows[spatialDim.ind()] = inputTiling.pads.value()[index];
            // the order of pads is "left, top, right, bottom"
            // so, to get padding of other side, get +2 to current index
            highs[spatialDim.ind()] = inputTiling.pads.value()[index + 2];
        } else {
            lows[spatialDim.ind()] = builder.getIndexAttr(padsByDims[spatialDim.ind()].first);
            highs[spatialDim.ind()] = builder.getIndexAttr(padsByDims[spatialDim.ind()].second);
        }
    }

    mlir::Value paddingValue;
    if (qType != nullptr) {
        paddingValue = createQuantizedPaddingValue(loc, builder, qType, tiledType);
    } else {
        paddingValue = mlir::arith::ConstantOp::create(builder, loc, builder.getZeroAttr(tiledType.getElementType()));
    }

    tiledOperands[0] = mlir::tensor::PadOp::create(builder, loc, /*result=*/mlir::Type(), tiledInput, lows, highs,
                                                   paddingValue, /*nofold=*/false);

    auto adjustedBounds = vpux::getBounds(tiledType).toValues();
    // It can be checked with getShape(tiledInput).isDynamic but there is a collision with DynamicDim Type, which is not
    // supported here
    const bool tileWasDynamicBeforePad = !adjustedBounds.empty();
    // Padding a dynamic operand can't make it static again, but a static one can become dynamic
    // here if its pad amounts are loop-IV expressions -- so this can't be inferred from
    // tileWasDynamicBeforePad alone and needs its own check on the actual padded result.
    const bool tileIsDynamicAfterPad = !mlir::cast<mlir::RankedTensorType>(tiledOperands[0].getType()).hasStaticShape();

    if (tileIsDynamicAfterPad && !tileWasDynamicBeforePad) {
        // No bound to build on yet: derive one from the tiling machinery's own upper bound for
        // this tile when it computed one, else fall back to this tile's current (pre-pad) shape.
        if (!inputTiling.tiles.empty() && !inputTiling.tiles.front().bounds.empty()) {
            adjustedBounds = inputTiling.tiles.front().bounds;
        } else if (tiledType.getShape().isStatic()) {
            adjustedBounds = vpux::Bounds(tiledType.getShape());
        } else {
            assert(false && "Cannot derive bounds for a dynamic pre-pad shape with no existing bounds (passed "
                            "something like DynamicDimsMaskTensorType)");
        }
    }

    // Add the pad's own contribution to whichever bound we now have (already present, or just derived above).
    // Exception: when the tile already had a pre-pad bound, its shape is a composable expression that may
    // itself already depend on low/high (e.g. "requiredSize - padBefore - padAfter"), in which case adding
    // getPadBoundDelta on top would double-count. tryComposePaddedBound recomposes "sourceSize + low + high"
    // into one expression instead and takes its own max -- correct whether or not that cancellation applies.
    if (tileIsDynamicAfterPad) {
        OpChainAnalysis padOpChainAnalysis;
        ValueRangeMap padValueMap;
        for (auto index : irange(Dims4D::Act::numSpatialDims)) {
            const auto spatialDim = Dims4D::Act::getSpatialDim(index);
            if (tileWasDynamicBeforePad && is_contained(dims, spatialDim) && !inputTiling.tiles.empty()) {
                const auto composed = tryComposePaddedBound(
                        builder, loc, inputTiling.tiles.front().shape[spatialDim.ind()], lows[spatialDim.ind()],
                        highs[spatialDim.ind()], padOpChainAnalysis, padValueMap);
                if (composed.has_value()) {
                    adjustedBounds[spatialDim] = *composed;
                    continue;
                }
                // Could not evaluate the composed expression: fall through to the decomposed heuristic below.
            }
            // Non-Constants are not supported. Theoretically if we can't fold number of tiles into constant we should
            // estimate MIN possible amount of tiles with presburger/OpChainAnalysis
            const auto numTilesAlongDim = mlir::getConstantIntValue(outputTile.axis[spatialDim.ind()]);
            const bool canCollapseToSingleTile = !numTilesAlongDim.has_value() || *numTilesAlongDim <= 1;
            adjustedBounds[spatialDim] +=
                    getPadBoundDelta(spatialDim, canCollapseToSingleTile, dims, padsByDims, lows, highs);
        }
    }

    const auto tensorDesc = vpux::getTensorAttr(tiledType.getContext(), tiledType.getDimsOrder(),
                                                tiledType.getMemSpace(), adjustedBounds);
    SmallVector<int64_t> staticDims;
    auto rankedType = mlir::cast<mlir::RankedTensorType>(tiledOperands[0].getType());
    staticDims.reserve(rankedType.getRank());
    llvm::transform(llvm::seq<size_t>(0, rankedType.getRank()), std::back_inserter(staticDims), [&](auto i) {
        if (rankedType.isDynamicDim(checked_cast<unsigned>(i))) {
            return mlir::ShapedType::kDynamic;
        }
        return rankedType.getDimSize(checked_cast<unsigned>(i));
    });

    tiledOperands[0].setType(mlir::RankedTensorType::get(staticDims, tiledType.getElementType(), tensorDesc));
    const auto createOperation = [&]() {
        auto generatedOp = mlir::cast<ConcreteOp>(opGenerator());
        generatedOp.setPadAttr(getPaddingAttr(builder.getContext(), 0, 0, 0, 0));
        vpux::inferReturnTypes(generatedOp, vpux::InferShapedTypeMode::SHAPE);
        return generatedOp;
    };

    auto nextOperationIsNotLastOperationInFusion = llvm::any_of(operation->getUsers(), [](mlir::Operation* userOp) {
        return !mlir::isa<mlir::tensor::InsertSliceOp>(userOp);
    });

    if (nextOperationIsNotLastOperationInFusion) {
        return castCreatedOperation(builder, outputTile, createOperation());
    }

    return createOperation();
}

/** @brief Adds cast ops before tile insertion for each result of the tiled operation.

    For each result, if the original operation has dynamic dims that are not introduced by the current tiling
    (i.e. there are one or more dynamic dims that are not tiling dims), a tensor.cast operation is created to
    align the tiled result shape with the insertion point in the full tensor.

    @param outputTiles  Per-result output tile information (one entry per operation result).
    @return A vector of (possibly casted) values, one per result.
*/
SmallVector<mlir::Value> castOutputForInsertion(mlir::OpBuilder& builder, ArrayRef<SCFTileInfo> outputTiles,
                                                DimArrRef dims, mlir::Operation* origOperation,
                                                mlir::Operation* tiledOperation);

/** @brief adjust padded output
 * In case the PadOp has been added, but operation used to be with static output
 * generated by scf tiling functions, result sizes has to be corrected to be dynamic too
 * It doesn't affect the result type of the loop, only result size in InsertSliceOp
 */
template <class ConcreteOp>
void correctPaddedOutput(mlir::OpBuilder& builder, ConcreteOp operation, SmallVector<mlir::OpFoldResult>& resultSizes) {
    auto padInfo = toPadInfo(operation.getPad());
    if (!padInfo.enabled() || !operation->hasAttr(tilingStrategy)) {
        return;
    }
    auto padsByDims = padInfo.toPadByDims();
    const auto strategy =
            Shape(parseIntArrayAttr<int64_t>(mlir::cast<mlir::ArrayAttr>(operation->getAttr(tilingStrategy))));
    auto tilingDims = getNonOneDim(strategy);
    for (auto dim : tilingDims) {
        auto sizeValue = mlir::getConstantIntValue(resultSizes[dim.ind()]);
        if (sizeValue.has_value() && padsByDims.contains(dim.ind()) &&
            (padsByDims[dim.ind()].first != 0 || padsByDims[dim.ind()].second != 0)) {
            resultSizes[dim.ind()] = mlir::arith::ConstantOp::create(builder, operation->getLoc(),
                                                                     builder.getIndexAttr(sizeValue.value()))
                                             .getResult();
        }
    }
}

/** @brief Result of a side-effect-free SCF fusion feasibility check.
    Indicates whether fusion is possible and, if a producer MC strategy adjustment is needed,
    which strategy to apply. The caller must apply the adjustment only after all remaining
    fusion preconditions (e.g. multi-user checks) pass. */
struct SCFFusionPlan {
    bool canFuse = false;
    mlir::Operation* producerToAdjust = nullptr;
    std::optional<VPU::MultiClusterStrategy> newProducerStrategy = std::nullopt;

    static SCFFusionPlan rejected() {
        return {};
    }
    static SCFFusionPlan accepted() {
        SCFFusionPlan plan;
        plan.canFuse = true;
        return plan;
    }
    static SCFFusionPlan acceptedWithAdjustment(mlir::Operation* producer, VPU::MultiClusterStrategy strategy) {
        SCFFusionPlan plan;
        plan.canFuse = true;
        plan.producerToAdjust = producer;
        plan.newProducerStrategy = strategy;
        return plan;
    }
};

/** @brief Pure predicate: checks if two operations might be vertically fused.
    Does NOT mutate IR. If MC strategy adjustment is needed, the planned adjustment
    is returned in the SCFFusionPlan but not applied.

    The function checks if there are some spills already between operations
    To be extended to more complex checks
*/
SCFFusionPlan planFusion(mlir::OpOperand& consumer, mlir::OpResult producerCandidate,
                         const llvm::SetVector<mlir::Operation*>& producers, const MergeConfiguration& mergeConfig);

/** @brief Apply a planned MC strategy adjustment from a successful SCFFusionPlan.
    This is the ONLY place where the SCF fusion path mutates a producer's MC strategy; all
    feasibility helpers (planFusion/findProducerStrategyForFusion) stay side-effect-free.
    Call only after every fusion precondition (including multi-user checks) has been validated, so
    no speculative mutation is introduced. A no-op when the plan carries no adjustment. */
void applyFusionPlan(const SCFFusionPlan& plan);

/** @brief Returns the padded dim of @p op if it is a VPU.Expand with single-dim
    end-padding (`pads_begin == 0`, exactly one `pads_end[i] > 0`); `std::nullopt`
    otherwise. Such Expand may be tiled only on non-padded dims; callers must keep
    the padded dim untiled (see `isSupportedOutTile`). */
std::optional<vpux::Dim> getSinglePaddedExpandDim(mlir::Operation* op);

/** @brief Generate upper bounds for dynamic tensors
 */
mlir::LogicalResult getResultTileBounds(mlir::Operation* operation, unsigned resultNumber, DimArrRef tilingDims,
                                        ArrayRef<mlir::OpFoldResult> sizes, Bounds& resultBounds);

/** @brief Checks if an operation is an NCE operation with padding attribute
 *
 *  This function verifies if the given operation is an NCE operation
 *  that supports the MLIR tiling interface and has a padding attribute defined.
 *
 *  @param op The operation to check
 *  @return true if the operation is an NCE operation with pad attribute, false otherwise
 */
bool isNceOpWithPadAttr(mlir::Operation* op);

mlir::func::FuncOp cloneFuncOp(mlir::func::FuncOp originalFunc, const std::string& newName,
                               mlir::FunctionType newFuncType = nullptr);

mlir::RankedTensorType removeBoundsAttr(mlir::RankedTensorType type);

void moveAffineArithOpsEarly(mlir::Block& block);

void addCheckForBlockSize(mlir::OpBuilder& builder, mlir::tensor::DimOp dimOp, mlir::Value blockSize,
                          mlir::func::FuncOp funcOp, llvm::StringRef errorMsg);
mlir::LogicalResult getTensorDimOpFromIndex(mlir::OpBuilder& builder, mlir::Value tensor, int64_t dimIdx,
                                            mlir::tensor::DimOp& dimOp);

/**
 * @brief Applies index backtracking to adjust tensor slice indices based on InsertSliceOp parameters.
 *
 * This function performs index backtracking for a tensor InsertSliceOp operation, adjusting
 * the indices for specified dimensions. It returns a vector of adjusted MLIR values that
 * represent the backtracked indices.
 *
 * @param insertSliceOp The MLIR tensor InsertSliceOp operation to analyze for index backtracking
 * @param dimsToAdjust Array of dimension indices that need to be adjusted during backtracking
 * @return SmallVector<mlir::Value> A collection of MLIR values representing the adjusted indices
 *         after applying the backtracking algorithm
 */
SmallVector<mlir::Value> applyIndexBacktracking(mlir::tensor::InsertSliceOp insertSliceOp,
                                                ArrayRef<size_t> dimsToAdjust);

void restorePaddingAttribute(mlir::Operation* region, Logger log, bool deferDynamicShapeWorkloads = false);

void refreshInferredResultShapes(ArrayRef<mlir::Operation*> operations);

// @brief Re-infers shape-only result types for `targetOps`, walking `scopes` to recover their true
// program (dominance) order first - `targetOps`'s own order (e.g. a SetVector's insertion order,
// or a graph-reachability traversal order) is not guaranteed to be topological. `scopes` need not be
// loops; any ancestor operations covering all of `targetOps` work (e.g. a single enclosing module).
void refreshResultShapesInIROrder(ArrayRef<mlir::Operation*> scopes, ArrayRef<mlir::Operation*> targetOps);

inline void refreshResultShapesInIROrder(ArrayRef<mlir::LoopLikeOpInterface> loops,
                                         ArrayRef<mlir::Operation*> targetOps) {
    refreshResultShapesInIROrder(
            SmallVector<mlir::Operation*>(llvm::map_range(loops,
                                                          [](mlir::LoopLikeOpInterface loop) -> mlir::Operation* {
                                                              return loop;
                                                          })),
            targetOps);
}

// Translates workloads on NCE ops that restorePaddingAttribute() deferred because their shapes were still dynamic.
// Must run after the unrolled shapes have been fully inferred, so the concrete output tile is known.
void translateDeferredWorkloadsAfterInference(mlir::Operation* region, Logger log);

// Clamps any single workload whose output tile exceeds the op's static output to that output, recomputing the input
// tile and padding. Repairs stale bounded-shape fallback descriptors on ops not covered by padding restoration (e.g.
// unpadded Eltwise). Must run after the unrolled shapes have been fully inferred.
void clampOversizedWorkloadsToStaticOutput(mlir::Operation* region, Logger log);

/**
 * @brief Number of iterations of `loopOp`, or nullopt when the trip count is not a compile-time constant.
 *
 * Handles both scf.for and scf.forall; for scf.forall the per-dimension trip counts are multiplied, so the
 * result is the total number of iterations of the whole index space.
 *
 * Returns nullopt when any lower/upper bound or step is non-constant which happens when the loop is tiled
 * against a dimension whose extent is only known at runtime, as in the HostCompile pipeline when a step is
 * non-positive, or when the product does not fit in int64.
 *
 * @param loopOp The loop operation to analyze; any op that is neither scf.for nor scf.forall yields nullopt
 * @return The constant trip count, or nullopt when it cannot be determined
 */
std::optional<int64_t> getConstantTripCount(mlir::Operation* loopOp);

/**
 * @brief This function tries to determine if the given OpFoldResult is dependent on any of the induction variables
 * of the provided scf.forall operation.
 *
 * @param ofr OpFoldResult to check for dependency
 * @param forallOp The scf.forall operation whose induction variables are checked against the OpFoldResult
 * @return bool True if the OpFoldResult is dependent on any induction variable of the forallOp, false otherwise
 */
bool isDependentOnForallIv(mlir::OpFoldResult ofr, mlir::scf::ForallOp forallOp);

/** @brief Analyze skip-connections for SCF tile+fuse planning.
 *
 * Scans operations selected for fusion, detects skip-source operations (multiple in-graph users),
 * and compares per-branch tile requirements using `tilingStorage` (tile 0 input tiling).
 * For each skip-source, returns precomputed replacement state with:
 *   - `biggestUserOp`: branch user requiring the largest tile,
 *   - `allUsersWithTheSameTileSize`: whether all candidate branch tiles are equal.
 *
 * The returned map is consumed later by fusion control logic to defer/allow producer fusion
 * and to drive post-tiling slice replacement for smaller branches.
 */
llvm::DenseMap<mlir::Operation*, VPU::PendingSliceReplacement> analyzeSkipConnectionsForTiling(
        const llvm::SetVector<mlir::Operation*>& allOpsToFuse, const TilingOperationStorage::UPtr& tilingStorage,
        const Logger& log);

/** @brief Apply deferred ExtractSlice replacements for skip-connections after SCF tile+fuse.
 *
 * Replaces recorded branch slices using the tiled value from the selected biggest branch.
 * If slice sizes match, replacement is direct; otherwise, offsets are adjusted relative to
 * the biggest-branch slice and a new tensor.extract_slice is created.
 */
void applyDeferredSliceReplacements(
        mlir::RewriterBase& builder,
        const llvm::DenseMap<mlir::Operation*, VPU::PendingSliceReplacement>& skipConnectionMap, const Logger& log);

}  // namespace vpux::VPU
