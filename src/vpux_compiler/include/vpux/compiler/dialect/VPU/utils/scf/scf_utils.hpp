//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/NPU40XX/dialect/VPU/IR/ops_interfaces.hpp"
#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/tiling.hpp"
#include "vpux/compiler/dialect/VPU/utils/manual_strategy_utils.hpp"
#include "vpux/compiler/dialect/core/IR/tensor_attr.hpp"
#include "vpux/compiler/dialect/core/types.hpp"
#include "vpux/compiler/utils/attributes.hpp"

#include <mlir/Dialect/Affine/Utils.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/Utils/StaticValueUtils.h>
#include <mlir/Interfaces/TilingInterface.h>

namespace vpux::VPU {

/** @brief Information about a tile.

    The structure incapsulates data of offsets, shape and tile axis
    for a tensor represented as mlir::OpFoldResult
*/

using SCFShape = SmallVector<mlir::OpFoldResult>;
using SCFShapeRef = ArrayRef<mlir::OpFoldResult>;

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
};

struct SCFTilingInfo {
    SCFTilingInfo(ArrayRef<SCFTileInfo> tilesValue): tiles(tilesValue) {
    }
    SCFTilingInfo(ArrayRef<SCFTileInfo> tilesValue, SCFShapeRef padsValue): tiles(tilesValue), pads(padsValue) {
    }

    SmallVector<SCFTileInfo> tiles;
    std::optional<SCFShape> pads;
};

using OpTilingOperandsFunc = std::function<void(SCFTilingInfo&)>;
using OpGeneratorFunc = std::function<mlir::Operation*()>;

// @brief Dim value of input/output/weights shape
mlir::OpFoldResult getDimValue(mlir::OpBuilder& builder, mlir::Operation* operation, int64_t dim);

// @brief Calculates tile for weights table based on output tile
SCFTileInfo getWeightsTableSCFTile(mlir::Type origWeightsTableType, mlir::OpBuilder& builder,
                                   const SCFTileInfo& outputTile);

/** @brief Restores input tiling from output tile data

    The function calculates input shape, offset and bounds based on
    parameters and shape and offset of output tile
*/
std::pair<std::optional<mlir::Range>, std::optional<int64_t>> solutionForOutputRange(
        mlir::Location loc, mlir::OpBuilder& builder, const SCFTileInfo& outputTile, Dim dim, const int64_t kernel,
        const int64_t stride, const int64_t origInputSize, const int64_t origOutputSize,
        const std::pair<int64_t, int64_t>& origPadding, mlir::OpFoldResult& padBefore, mlir::OpFoldResult& padAfter);

/** @brief Generate slice based on tiling information

    The function generates ExtractSliceOp based on offset and size in tile info
*/
mlir::Value generateTile(mlir::Location loc, mlir::OpBuilder& builder, mlir::Value origInput,
                         const SCFTileInfo& inputTileInfo);

/** @brief Return result type after tiling to new shape

    The function extracts result type of operation
    after changing shape
*/
mlir::Type extractResultType(mlir::Type origType, SCFShapeRef newShape, BoundsRef bounds);

/** @brief create operation with padding adjustment

    @note If operation has paddings which are not 0, they have to be corrected based on
    position of tile. Unfortunately, in OV based operations paddings have to be known integer attributes,
    they cannot be calculated or created as constant for each case. That's why there must be the structure
    with if-else which identifies how paddings are set.
*/
template <class ConcreteOp>
mlir::Operation* createTiledPaddedOperation(OpGeneratorFunc opGenerator, OpTilingOperandsFunc operandsGenerator,
                                            mlir::OpBuilder& builder, SCFTilingInfo& inputTiling, DimArrRef dims,
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

    auto paddingValue = builder.create<mlir::arith::ConstantOp>(loc, builder.getZeroAttr(tiledType.getElementType()));
    auto adjustedBounds = Bounds();
    if (auto boundedType = mlir::dyn_cast<vpux::Core::BoundedTensorType>(tiledType)) {
        adjustedBounds = boundedType.getBounds().toValues();
    }

    SmallVector<mlir::OpFoldResult> lows(tiledType.getRank(), builder.getIndexAttr(0));
    SmallVector<mlir::OpFoldResult> highs(tiledType.getRank(), builder.getIndexAttr(0));

    auto padsByDims = padInfo.toPadByDims();
    // bounds are not updated for dynamic dimensions, as the pad value is calculated at runtime based on the loop index
    for (auto index : irange(Dims4D::Act::numSpatialDims)) {
        const auto spatialDim = Dims4D::Act::getSpatialDim(index);
        if (llvm::find(dims, spatialDim) != dims.end()) {
            lows[spatialDim.ind()] = inputTiling.pads.value()[index];
            // the order of pads is "left, top, right, bottom"
            // so, to get padding of other side, get +2 to current index
            highs[spatialDim.ind()] = inputTiling.pads.value()[index + 2];
        } else {
            lows[spatialDim.ind()] = builder.getIndexAttr(padsByDims[spatialDim.ind()].first);
            highs[spatialDim.ind()] = builder.getIndexAttr(padsByDims[spatialDim.ind()].second);
            if (!adjustedBounds.raw().empty()) {
                adjustedBounds[spatialDim] += padsByDims[spatialDim.ind()].first + padsByDims[spatialDim.ind()].second;
            }
        }
    }

    tiledOperands[0] = builder.create<mlir::tensor::PadOp>(loc, /*result=*/mlir::Type(), tiledInput, lows, highs,
                                                           paddingValue, /*nofold=*/false);
    const auto tensorDesc = vpux::getTensorAttr(tiledType.getContext(), tiledType.getDimsOrder(),
                                                tiledType.getMemSpace(), adjustedBounds);
    SmallVector<int64_t> staticDims;
    auto rankedType = mlir::cast<mlir::RankedTensorType>(tiledOperands[0].getType());
    staticDims.reserve(rankedType.getRank());
    llvm::transform(llvm::seq<size_t>(0, rankedType.getRank()), std::back_inserter(staticDims), [&](auto i) {
        if (rankedType.isDynamicDim(i)) {
            return mlir::ShapedType::kDynamic;
        }
        return rankedType.getDimSize(i);
    });

    tiledOperands[0].setType(mlir::RankedTensorType::get(staticDims, tiledType.getElementType(), tensorDesc));
    const auto createOperation = [&]() {
        auto generatedOp = mlir::cast<ConcreteOp>(opGenerator());
        generatedOp.setPadAttr(getPaddingAttr(builder.getContext(), 0, 0, 0, 0));
        auto outputType = mlir::cast<vpux::NDTypeInterface>(generatedOp->getResult(0).getType());
        auto outputShape = to_small_vector(outputType.getShape().raw());
        for (auto staticDim : staticDims | indexed) {
            if (staticDim.value() == mlir::ShapedType::kDynamic) {
                outputShape[staticDim.index()] = mlir::ShapedType::kDynamic;
            }
        }
        outputType = outputType.changeShape(ShapeRef(outputShape));
        generatedOp->getResult(0).setType(outputType);
        return generatedOp;
    };
    return createOperation();
}

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
bool checkFusion(mlir::OpOperand& consumer, mlir::OpResult producerCandidate);

/** @brief Checks if an operation is an NCE operation with padding attribute
 *
 *  This function verifies if the given operation is an NCE operation
 *  that supports the MLIR tiling interface and has a padding attribute defined.
 *
 *  @param op The operation to check
 *  @return true if the operation is an NCE operation with pad attribute, false otherwise
 */
bool isNceOpWithPadAttr(mlir::Operation* op);

llvm::SmallVector<mlir::Operation*> collectOpsInTopologicalOrder(
        llvm::ArrayRef<mlir::Operation*> startNodes,
        llvm::function_ref<llvm::SmallSetVector<mlir::Operation*, 16>(mlir::Operation*)> getNeighbors,
        llvm::function_ref<bool(mlir::Operation*)> stopCheckFn);

/**
 * @brief Utility class for analyzing and processing affine operation chains in MLIR
 *
 * The AffineChainUtils class provides functionality to collect, and evaluate
 * chains of affine operations in MLIR. It helps with tracking dependencies between
 * affine operations and computing values from OpFoldResult objects within the context
 * of affine transformations.
 *
 * Key features:
 * - Collects chains of related affine operations from a given value
 * - Evaluates OpFoldResult values with optional bounded shape considerations
 * - Caches affine operation chains for performance optimization
 * - Provides utilities for extracting affine maps and operands from operations
 *
 */
class AffineChainUtils {
public:
    explicit AffineChainUtils(Logger log = Logger::global().nest("affine-utils"));

    llvm::SmallSetVector<mlir::Operation*, 4> collectAffineOpsChain(mlir::Value val);

    /**
     * @brief Get the value from an OpFoldResult
     * @param val The OpFoldResult to process
     * @param valueMap Map of values to their possible ranges
     * @return The computed value, or nullopt if processing failed
     */
    std::optional<int64_t> getOpFoldResultValue(mlir::OpFoldResult val,
                                                llvm::DenseMap<mlir::Value, SmallVector<int64_t>>& valueMap);

private:
    std::pair<mlir::AffineMap, mlir::ValueRange> getAffineMapAndOperands(mlir::Operation* op);
    int64_t getAffineResult(mlir::Operation* op, llvm::ArrayRef<int64_t> results);
    void updateChainCache(mlir::Value val, const llvm::SmallSetVector<mlir::Operation*, 4>& chain) const;
    std::optional<int64_t> getIntegerFromValue(mlir::Value value);
    std::optional<int64_t> processAffineCallChain(mlir::Value val,
                                                  llvm::DenseMap<mlir::Value, SmallVector<int64_t>>& valueMap);

    mutable llvm::DenseMap<mlir::Value, llvm::SmallSetVector<mlir::Operation*, 4>> _chainCache;
    Logger _log;
};

}  // namespace vpux::VPU
