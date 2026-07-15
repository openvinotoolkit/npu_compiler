//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/NPU40XX/dialect/VPU/IR/ops_interfaces.hpp"
#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/tiling.hpp"
#include "vpux/compiler/dialect/VPU/utils/manual_strategy_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/vf_merge_configuration.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/vertical_fusion_utils.hpp"
#include "vpux/compiler/dialect/core/IR/tensor_attr.hpp"
#include "vpux/compiler/dialect/core/types.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <llvm/ADT/StringRef.h>
#include <mlir/Dialect/Affine/Utils.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/Utils/StaticValueUtils.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/Interfaces/TilingInterface.h>

#include <array>

namespace vpux::VPU {

enum class TilePosition { MIDDLE = 0, END = 1, START = 2, FULLBLK = 3 };
constexpr size_t NUMBITS = 2;

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

    explicit SCFTileInfo(SCFShapeRef shape, BoundsRef bounds, mlir::OpBuilder& builder)
            : shape(shape),
              offsets(SCFShape(shape.size(), builder.getIndexAttr(0))),
              axis(SCFShape(shape.size(), builder.getIndexAttr(1))),
              bounds(bounds) {
    }

    void printFormat(llvm::raw_ostream& stream) const {
        printTo(stream, "SCFTile [shape = {0}, offsets = {1}, axis = {2}, bounds = {3}]", shape, offsets, axis, bounds);
    }
};

struct SCFTilingInfo {
    SCFTilingInfo(ArrayRef<SCFTileInfo> tilesValue): tiles(tilesValue) {
    }
    SCFTilingInfo(ArrayRef<SCFTileInfo> tilesValue, SCFShapeRef padsValue): tiles(tilesValue), pads(padsValue) {
    }

    SmallVector<SCFTileInfo> tiles;
    std::optional<SCFShape> pads;
};

// Tracks deferred ExtractSlice replacements for skip-connections during SCF tile+fuse.
struct PendingSliceReplacement {
    mlir::Operation* biggestUserOp;            // The user op that requires the biggest tile for this skip connection
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

/** @brief Restores input tiling from output tile data

    The function calculates input shape, offset and bounds based on
    parameters and shape and offset of output tile
*/
std::pair<std::optional<mlir::Range>, std::optional<int64_t>> solutionForOutputRange(
        mlir::Location loc, mlir::OpBuilder& builder, const SCFTileInfo& outputTile, Dim dim, const int64_t kernel,
        const int64_t stride, mlir::OpFoldResult origInputSize, int64_t origOutputSize,
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
    auto createPaddingValue = [&](mlir::Type valueType, mlir::TypedAttr attr) -> mlir::Value {
        auto constantOp = builder.create<mlir::arith::ConstantOp>(loc, attr);
        return builder.create<mlir::UnrealizedConversionCastOp>(loc, valueType, constantOp.getResult()).getResult(0);
    };

    mlir::Value paddingValue;
    auto storageType = qType.getStorageType();
    mlir::Type paddingType = storageType;

    VPUX_THROW_WHEN(mlir::isa<mlir::quant::UniformQuantizedPerAxisType>(tiledType.getElementType()),
                    "Per-channel quantization is not supported");

    if (auto intType = mlir::dyn_cast<mlir::IntegerType>(storageType)) {
        paddingType = builder.getIntegerType(intType.getWidth());
    }

    if (const auto uniformType = mlir::dyn_cast<mlir::quant::UniformQuantizedType>(qType)) {
        const auto zeroPoint = uniformType.getZeroPoint();
        auto zeroPointAttr = builder.getIntegerAttr(paddingType, zeroPoint);
        paddingValue = createPaddingValue(tiledType.getElementType(), zeroPointAttr);
    }
    return paddingValue;
}

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

    auto adjustedBounds = Bounds();
    if (auto boundedType = mlir::dyn_cast<vpux::Core::BoundedTensorType>(tiledType)) {
        adjustedBounds = boundedType.getBounds().toValues();
    }

    SmallVector<mlir::OpFoldResult> lows(tiledType.getRank(), builder.getIndexAttr(0));
    SmallVector<mlir::OpFoldResult> highs(tiledType.getRank(), builder.getIndexAttr(0));

    auto padsByDims = padInfo.toPadByDims();
    // bounds are not updated for dynamic dimensions, as the pad value is calculated at runtime based on the loop
    // index
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
        if (!adjustedBounds.raw().empty()) {
            adjustedBounds[spatialDim] += padsByDims[spatialDim.ind()].first + padsByDims[spatialDim.ind()].second;
        }
    }

    mlir::Value paddingValue;
    if (qType != nullptr) {
        paddingValue = createQuantizedPaddingValue(loc, builder, qType, tiledType);
    } else {
        paddingValue = builder.create<mlir::arith::ConstantOp>(loc, builder.getZeroAttr(tiledType.getElementType()));
    }

    tiledOperands[0] = builder.create<mlir::tensor::PadOp>(loc, /*result=*/mlir::Type(), tiledInput, lows, highs,
                                                           paddingValue, /*nofold=*/false);

    if (!mlir::cast<mlir::RankedTensorType>(tiledOperands[0].getType()).hasStaticShape() &&
        adjustedBounds.raw().empty()) {
        auto origInputType = mlir::cast<vpux::NDTypeInterface>(operation->getOperand(0).getType());
        auto boundsValue = to_small_vector(origInputType.getShape());
        adjustedBounds = vpux::Bounds(ArrayRef<int64_t>(boundsValue.data(), boundsValue.size()));
        for (auto index : irange(Dims4D::Act::numSpatialDims)) {
            const auto spatialDim = Dims4D::Act::getSpatialDim(index);
            adjustedBounds[spatialDim] += padsByDims[spatialDim.ind()].first + padsByDims[spatialDim.ind()].second;
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

    const auto castCreatedOperation = [&](ConcreteOp generatedOp) {
        SmallVector<int64_t> staticOutputShape;
        llvm::transform(outputTile.shape, std::back_inserter(staticOutputShape), [&](mlir::OpFoldResult val) {
            auto shapeDimValue = mlir::getConstantIntValue(val);
            return shapeDimValue.value();
        });
        auto firstResult = generatedOp->getResult(0);
        auto generatedType = mlir::cast<vpux::NDTypeInterface>(firstResult.getType());
        auto correctedTensorDesc = vpux::getTensorAttr(generatedType.getContext(), generatedType.getDimsOrder(),
                                                       generatedType.getMemSpace());

        mlir::Type correctedTiledOutputType =
                mlir::RankedTensorType::get(staticOutputShape, generatedType.getElementType(), correctedTensorDesc);
        return builder.create<mlir::tensor::CastOp>(generatedOp.getLoc(), correctedTiledOutputType,
                                                    generatedOp->getResult(0));
    };

    // check if next operation has static shape then we cast current dynamic shape to static shape.
    auto nextOperationIsStaticallyShaped = llvm::all_of(outputTile.shape, [](mlir::OpFoldResult ofr) {
        return mlir::getConstantIntValue(ofr).has_value();
    });

    auto nextOperationIsNotLastOperationInFusion = llvm::any_of(operation->getUsers(), [](mlir::Operation* userOp) {
        return !mlir::isa<mlir::tensor::InsertSliceOp>(userOp);
    });

    if (nextOperationIsStaticallyShaped && nextOperationIsNotLastOperationInFusion) {
        return castCreatedOperation(createOperation());
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
    if (padInfo.enabled() && operation->hasAttr(tilingStrategy)) {
        auto padsByDims = padInfo.toPadByDims();
        const auto strategy =
                Shape(parseIntArrayAttr<int64_t>(mlir::cast<mlir::ArrayAttr>(operation->getAttr(tilingStrategy))));
        auto tilingDims = getNonOneDim(strategy);
        for (auto dim : tilingDims) {
            auto sizeValue = mlir::getConstantIntValue(resultSizes[dim.ind()]);
            if (sizeValue.has_value() && padsByDims.contains(dim.ind()) &&
                (padsByDims[dim.ind()].first != 0 || padsByDims[dim.ind()].second != 0)) {
                resultSizes[dim.ind()] = builder.create<mlir::arith::ConstantOp>(
                                                        operation->getLoc(), builder.getIndexAttr(sizeValue.value()))
                                                 .getResult();
            }
        }
    }
}

/** @brief Checks if two operations might be vertically fused

    The function checks if there are some spills already between operations
    To be extended to more complex checks
*/
bool checkFusion(mlir::OpOperand& consumer, mlir::OpResult producerCandidate,
                 const llvm::SetVector<mlir::Operation*>& producers, const MergeConfiguration& mergeConfig);

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

void restorePaddingAttribute(mlir::Operation* region, Logger log);

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
