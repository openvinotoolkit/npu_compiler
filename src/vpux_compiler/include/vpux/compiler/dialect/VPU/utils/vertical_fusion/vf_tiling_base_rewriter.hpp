//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/VPU/utils/cost_model/layer_vpunn_cost.hpp"
#include "vpux/compiler/dialect/VPU/utils/manual_strategy_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/vertical_fusion_utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

namespace vpux::VPU {

//
// VerticalFusionTilingRewriter
//

using TilingFunction = std::function<void(int64_t, mlir::Operation*, SmallVector<mlir::Value>&, SmallVector<Shape>&)>;

template <typename VFConfigType, typename VFSchedulingFactoryType>
class VerticalFusionTilingRewriterBase : public mlir::OpRewritePattern<VPU::VerticalFusionOp> {
public:
    VerticalFusionTilingRewriterBase(mlir::MLIRContext* ctx, bool enableVerticalFusionPipelining,
                                     const std::unique_ptr<VPU::LayerVPUNNCost>& costFunction, Logger log)
            : mlir::OpRewritePattern<VPU::VerticalFusionOp>(ctx),
              _log(log),
              _enableVerticalFusionPipelining(enableVerticalFusionPipelining),
              _vpunnCostFunction(costFunction) {
    }

    mlir::LogicalResult matchAndRewrite(VPU::VerticalFusionOp origOp, mlir::PatternRewriter& rewriter) const final;

protected:
    virtual VFConfigType createVFConfig(VPU::VerticalFusionOp vfOp) const = 0;
    virtual std::pair<DimArr, int64_t> getDimsData(ArrayRef<int64_t> strategy) const = 0;

    virtual TilingStorage restoreTilingStorage(VFConfigType& config, ArrayRef<int64_t> strategy,
                                               TilingOperationStorage::UPtr& operationStorage) const = 0;

    Logger _log;
    bool _enableVerticalFusionPipelining;

private:
    void adjustInputShape(mlir::PatternRewriter& rewriter, mlir::Operation* operation, InputTiling& inputTiling,
                          mlir::IRMapping& mapper, TilingStorage& tilingStorage,
                          const TilingOperationStorage::UPtr& opStorage, int64_t tilingIndex, DimArrRef dims) const;
    void processOffset(mlir::Value operand, const TilingOperationStorage::UPtr& opStorage, TileInfo& originalTiling,
                       int64_t tilingIndex, DimArrRef dims, ShapeRef expectedShape) const;
    bool processBlockArgument(mlir::BlockArgument blockArg, mlir::Operation* operation, TilingStorage& tilingStorage,
                              TileInfo& originalTiling, int64_t tilingIndex, DimArrRef dims) const;
    void applyLinearTiling(const int64_t numTiles, VFConfigType& config,
                           SmallVector<SmallVector<mlir::Value>>& resultTileVals,
                           SmallVector<SmallVector<Shape>>& resultTileOffsets, const TilingFunction& tilingProcedure,
                           VFLoopIndexAttr vfIndexAttr) const;
    void applyPipelinedTiling(const int64_t numTiles, VFConfigType& config,
                              SmallVector<SmallVector<mlir::Value>>& resultTileVals,
                              SmallVector<SmallVector<Shape>>& resultTileOffsets, const TilingFunction& tilingProcedure,
                              const TilingOperationStorage::UPtr& storage, VFLoopIndexAttr vfIndexAttr) const;
    const std::unique_ptr<VPU::LayerVPUNNCost>& _vpunnCostFunction;
};

template <typename VFConfigType, typename VFSchedulingFactoryType>
bool VerticalFusionTilingRewriterBase<VFConfigType, VFSchedulingFactoryType>::processBlockArgument(
        mlir::BlockArgument blockArg, mlir::Operation* operation, TilingStorage& tilingStorage,
        TileInfo& originalTiling, int64_t tilingIndex, DimArrRef dims) const {
    auto& offset = originalTiling.offsets;
    const auto storageInfo = tilingStorage.get(std::make_pair(blockArg.getArgNumber(), operation), tilingIndex);
    VPUX_THROW_WHEN(!storageInfo.has_value(), "Tiling info for argument {0} with index {1} not found", blockArg,
                    tilingIndex);

    auto tileInfo = storageInfo.value();
    VPUX_THROW_UNLESS(dims.size() < tileInfo.shape.size(), "Got invalid tiling shape size {0}", tileInfo.shape.size());

    const auto inputOffset = tileInfo.offsets;
    const auto inputDimShape = tileInfo.shape;
    const auto origDimSize = originalTiling.shape;

    _log.trace("Input Offset {0}, shape {1} ==> offset: {2}, shape: {3} ", inputOffset, inputDimShape, offset,
               origDimSize);

    for (auto dim : dims) {
        if (offset[dim] >= inputOffset[dim] &&
            (inputOffset[dim] + inputDimShape[dim]) >= (offset[dim] + origDimSize[dim])) {
            offset[dim] -= inputOffset[dim];
            continue;
        }
        _log.trace("invalid offsets: Input Offset {0}, shape {1} ==> offset: {2}, shape: {3} ", inputOffset,
                   inputDimShape, offset, origDimSize);
        return false;
    }

    return true;
}

template <typename VFConfigType, typename VFSchedulingFactoryType>
void VerticalFusionTilingRewriterBase<VFConfigType, VFSchedulingFactoryType>::processOffset(
        mlir::Value operand, const TilingOperationStorage::UPtr& opStorage, TileInfo& originalTiling,
        int64_t tilingIndex, DimArrRef dims, ShapeRef expectedShape) const {
    auto& offset = originalTiling.offsets;
    auto offsetEqualsToZero = llvm::all_of(dims, [&](Dim dim) {
        return offset[dim] == 0;
    });

    if (offsetEqualsToZero) {
        return;
    }

    auto operandOp = operand.getDefiningOp();
    if (operandOp != nullptr) {
        auto inputOutputTiling = opStorage->get(operandOp, tilingIndex);
        VPUX_THROW_UNLESS(inputOutputTiling.has_value(), "Couldn't find tiling info at {0}", operandOp->getLoc());
        const auto& inputOutputTilingPair = inputOutputTiling.value();
        auto& outTile = inputOutputTilingPair.second;
        for (auto dim : dims) {
            offset[dim] -= outTile.offsets[dim];
        }
        return;
    }

    for (auto dim : dims) {
        offset[dim] = expectedShape[dim] - originalTiling.shape[dim];
    }
}

/*
 This function slice to original tile shape in case bigger tile size was chosen
 during backpropagation process.
 In this case adjust shapes to original one by slicing
*/
template <typename VFConfigType, typename VFSchedulingFactoryType>
void VerticalFusionTilingRewriterBase<VFConfigType, VFSchedulingFactoryType>::adjustInputShape(
        mlir::PatternRewriter& rewriter, mlir::Operation* operation, InputTiling& inputTiling, mlir::IRMapping& mapper,
        TilingStorage& tilingStorage, const TilingOperationStorage::UPtr& opStorage, int64_t tilingIndex,
        DimArrRef dims) const {
    VPUX_THROW_WHEN(inputTiling.tiles.size() < operation->getOperands().size(),
                    "Number of operands {0} is more than number of operand tiles {1}", operation->getOperands().size(),
                    inputTiling.tiles.size());
    for (auto op : operation->getOperands() | indexed) {
        auto operand = op.value();
        auto opIndex = op.index();

        auto expectedOp = mapper.lookupOrNull(operand);
        if (expectedOp == nullptr) {
            continue;
        }

        auto originalTiling = inputTiling.tiles[opIndex];
        auto expectedShape = getShape(expectedOp);
        auto expectedOpSize = expectedShape.totalSize();
        const auto originalOpSize = originalTiling.shape.totalSize();
        if (expectedOpSize == originalOpSize) {
            continue;
        }

        //
        // For below pattern, the Eltwise3 may be tiled before the Eltwise2.
        // Then the Operand has been mapped to the new "SliceOp1" instead of "Eltwise1".
        // While tiling "Eltwise2", it throw exception of "expectedOpSize < originalOpSize".
        // Need to update this branch operand for this case.
        //
        // VF tilingStrategy: [1, 1, 1, 4]
        //                |                                 |
        //           Eltwise1: 1x64x72x128       Conv: 1x64x72x128
        //                |                 X               |
        //           Eltwise2: 1x64x72x128       Eltwise3: 1x64x72x128
        //                |                                 |
        //             Conv: 1x64x72x128                    |
        //                |                                 |
        //             Conv: 1x64x72x128                    |
        //                           \                     /
        //                             Eltwise4: 1x64x72x128
        //                                     |
        //
        // tiling into:
        //
        //                |                                 |
        //           Eltwise1: 1x64x72x36       Conv: 1x64x72x36
        //                |                 X               |
        //                |               /  SliceOp1    SliceOp2
        //                |             /         \         |
        //           Eltwise2: 1x64x72x36       Eltwise3: 1x64x72x32
        //                |                                 |
        //             Conv: 1x64x72x34                     |
        //                |                                 |
        //             Conv: 1x64x72x32                     |
        //                            \                    /
        //                             Eltwise4: 1x64x72x32
        //                                     |
        if (expectedOpSize < originalOpSize) {
            if (auto insertSliceOp = mlir::dyn_cast<VPU::SliceOp>(expectedOp.getDefiningOp())) {
                expectedOp = insertSliceOp.getInputs().front();
                expectedShape = getShape(expectedOp);
                expectedOpSize = expectedShape.totalSize();
            }
        }

        VPUX_THROW_WHEN(
                expectedOpSize < originalOpSize,
                "Original shape size for operand {0} is bigger than current one. Current size {1}, original size {2}",
                operand, expectedOpSize, originalOpSize);

        VPUX_THROW_WHEN(expectedShape.size() != originalTiling.shape.size(),
                        "Expected shape {0} and original one {1} must have same rank", expectedShape,
                        originalTiling.shape);

        // correct offset of operations based on offsets of block argument
        // In case the output of previous operation is bigger than expected
        // which might happen when bigger tile was chosen for same block argument
        // slice operation is needed after the output with correct offsets
        // calculated based on tiling information of current operation and previous one
        _log.trace("op {0}, Offset before {1}, shape {2}", operation->getLoc(), originalTiling.offsets,
                   originalTiling.shape);

        mlir::Value opSlice;
        const auto valName = printToString("input {0}", opIndex);
        auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(operand);
        if (blockArg != nullptr) {
            if (!processBlockArgument(blockArg, operation, tilingStorage, originalTiling, tilingIndex, dims)) {
                auto sliceOp = mlir::dyn_cast_or_null<VPU::SliceOp>(expectedOp.getDefiningOp());
                VPUX_THROW_WHEN(sliceOp == nullptr || sliceOp.getSource() == operand,
                                "Can't get the operand from Slice");

                auto inputOutputTiling = opStorage->get(operation, tilingIndex);
                VPUX_THROW_UNLESS(inputOutputTiling.has_value(), "Couldn't find tiling info at {0}",
                                  operation->getLoc());

                const auto inputTiling = inputOutputTiling.value().first.tiles[blockArg.getArgNumber()];
                opSlice = makeTile(rewriter, operation->getLoc(), sliceOp.getSource(), inputTiling, valName);
            } else {
                opSlice = makeTile(rewriter, operation->getLoc(), expectedOp, originalTiling, valName);
            }
        } else {
            processOffset(operand, opStorage, originalTiling, tilingIndex, dims, expectedShape);
            if (auto sliceOp = mlir::dyn_cast_or_null<VPU::SliceOp>(expectedOp.getDefiningOp())) {
                // correct offsets
                for (auto axis : dims) {
                    auto sliceOffset = parseIntArrayAttr<int64_t>(sliceOp.getStaticOffsets());
                    VPUX_THROW_UNLESS(originalTiling.offsets[axis] >= sliceOffset[axis.ind()],
                                      "Slice offset {0} is bigger than original one {1}", sliceOffset[axis.ind()],
                                      originalTiling.offsets[axis]);
                    originalTiling.offsets[axis] = originalTiling.offsets[axis] - sliceOffset[axis.ind()];
                }
            }
            opSlice = makeTile(rewriter, operation->getLoc(), expectedOp, originalTiling, valName);
        }

        _log.trace("Offset after {0}, shape {1} expectedOp {2}", originalTiling.offsets, originalTiling.shape,
                   expectedOp);

        mapper.map(operand, opSlice);
    }
}

template <typename VFConfigType, typename VFSchedulingFactoryType>
void VerticalFusionTilingRewriterBase<VFConfigType, VFSchedulingFactoryType>::applyLinearTiling(
        const int64_t numTiles, VFConfigType& config, SmallVector<SmallVector<mlir::Value>>& resultTileVals,
        SmallVector<SmallVector<Shape>>& resultTileOffsets, const TilingFunction& tilingProcedure,
        VFLoopIndexAttr vfIndexAttr) const {
    auto operations = config.getVFOperations();

    for (auto index : irange(numTiles)) {
        SmallVector<mlir::Value> currentResults;
        SmallVector<Shape> currentTiles;
        for (auto* op : operations) {
            currentResults.clear();
            currentTiles.clear();
            tilingProcedure(index, op, currentResults, currentTiles);
            VPUX_THROW_WHEN(currentResults.empty(), "Tiling procedure didn't return any result for operation @ {0}",
                            op);
            currentResults.front().getDefiningOp()->setAttr(VF_LOOP_INDEX_ATTR_NAME, vfIndexAttr);
            currentResults.front().getDefiningOp()->setAttr(VF_LOOP_TILE_INDEX_ATTR_NAME,
                                                            VFLoopTileIndexAttr::get(getContext(), index));
        }

        VPUX_THROW_WHEN(currentResults.size() != resultTileVals.size(),
                        "VF tiling expects {0} results but tiling procedure returned {1} for tile {2}",
                        resultTileVals.size(), currentResults.size(), index);
        VPUX_THROW_WHEN(currentTiles.size() != resultTileOffsets.size(),
                        "VF tiling expects {0} offsets but tiling procedure returned {1} for tile {2}",
                        resultTileOffsets.size(), currentTiles.size(), index);
        VPUX_THROW_WHEN(currentTiles.size() != currentResults.size(),
                        "Mismatch between tiled results ({0}) and tile offsets ({1}) for tile {2}",
                        currentResults.size(), currentTiles.size(), index);

        for (const auto& [resIdx, result] : enumerate(currentResults)) {
            resultTileVals[resIdx].push_back(result);
            resultTileOffsets[resIdx].push_back(currentTiles[resIdx]);
            _log.trace("  [applyLinearTiling] tile {0}, result[{1}]: resultTileOffsets = {2}", index, resIdx,
                       resultTileOffsets[resIdx]);
        }
    }
}

template <typename VFConfigType, typename VFSchedulingFactoryType>
void VerticalFusionTilingRewriterBase<VFConfigType, VFSchedulingFactoryType>::applyPipelinedTiling(
        const int64_t numTiles, VFConfigType& config, SmallVector<SmallVector<mlir::Value>>& resultTileVals,
        SmallVector<SmallVector<Shape>>& resultTileOffsets, const TilingFunction& tilingProcedure,
        const TilingOperationStorage::UPtr& storage, VFLoopIndexAttr vfIndexAttr) const {
    auto scheduling = config.getSubgraph().getScenario();
    VPUX_THROW_WHEN(!scheduling.has_value(), "Cannot get scheduling scenario from VF {0}", config.getSubgraph());

    VFSchedulingFactoryType costFactory(/*prefetching=*/true);
    auto scenario = costFactory.createVFScenario(scheduling.value(), _log);

    if (auto pipelinedScenario = std::dynamic_pointer_cast<IVFPipelinedScheduling<VFConfigType>>(scenario)) {
        auto pipelining = pipelinedScenario->getPipelining(config, numTiles, storage, _vpunnCostFunction);
        auto timeline = pipelining.getTimeLine();

        if (!timeline.empty()) {
            SmallVector<mlir::Value> currentResults;
            currentResults.reserve(resultTileVals.size());
            SmallVector<Shape> currentTiles;
            currentTiles.reserve(resultTileOffsets.size());
            for (auto& [index, operation] : pipelining.getTimeLine()) {
                // pipelining only records the compute ops, need to handle its view-like parents in advance
                auto viewLikeParents = VPU::getParentViewLikeOpsInVF(operation);
                for (auto viewOp : viewLikeParents) {
                    tilingProcedure(index, viewOp, currentResults, currentTiles);
                    // View ops have only one result
                    VPUX_THROW_WHEN(currentResults.size() != 1,
                                    "Tiling procedure didn't return exactly one result for view operation @ {0}",
                                    operation->getLoc());

                    auto* viewProducerOp = currentResults.back().getDefiningOp();
                    VPUX_THROW_UNLESS(viewProducerOp != nullptr,
                                      "tilingProcedure returned a non-op-defined value for viewOp at index {0};"
                                      " cannot attach VF loop attrs",
                                      index);
                    viewProducerOp->setAttr(VF_LOOP_INDEX_ATTR_NAME, vfIndexAttr);
                    viewProducerOp->setAttr(VF_LOOP_TILE_INDEX_ATTR_NAME,
                                            VFLoopTileIndexAttr::get(getContext(), index));
                }

                // currentResult and currentTiles keep result from previous call tilingProcedure
                tilingProcedure(index, operation, currentResults, currentTiles);
                VPUX_THROW_WHEN(currentResults.empty(), "Tiling procedure didn't return any result for operation @ {0}",
                                operation->getLoc());
                auto* producerOp = currentResults.front().getDefiningOp();
                VPUX_THROW_UNLESS(producerOp != nullptr,
                                  "tilingProcedure returned a non-op-defined value for operation at index {0};"
                                  " cannot attach VF loop attrs",
                                  index);
                producerOp->setAttr(VF_LOOP_INDEX_ATTR_NAME, vfIndexAttr);
                producerOp->setAttr(VF_LOOP_TILE_INDEX_ATTR_NAME, VFLoopTileIndexAttr::get(getContext(), index));

                VPUX_THROW_WHEN(currentResults.size() != resultTileVals.size(),
                                "VF tiling expects {0} results but tiling procedure returned {1} for tile {2}",
                                resultTileVals.size(), currentResults.size(), index);
                VPUX_THROW_WHEN(currentTiles.size() != resultTileOffsets.size(),
                                "VF tiling expects {0} offsets but tiling procedure returned {1} for tile {2}",
                                resultTileOffsets.size(), currentTiles.size(), index);
                VPUX_THROW_WHEN(currentTiles.size() != currentResults.size(),
                                "Mismatch between tiled results ({0}) and tile offsets ({1}) for tile {2}",
                                currentResults.size(), currentTiles.size(), index);

                if (llvm::find(config.getOutputs(), operation) != config.getOutputs().end()) {
                    for (const auto& [resIdx, result] : enumerate(currentResults)) {
                        resultTileVals[resIdx].push_back(result);
                        resultTileOffsets[resIdx].push_back(currentTiles[resIdx]);
                    }
                }
            }
            return;
        }
    }
    applyLinearTiling(numTiles, config, resultTileVals, resultTileOffsets, tilingProcedure, vfIndexAttr);
}

template <typename VFConfigType, typename VFSchedulingFactoryType>
mlir::LogicalResult VerticalFusionTilingRewriterBase<VFConfigType, VFSchedulingFactoryType>::matchAndRewrite(
        VPU::VerticalFusionOp vfOp, mlir::PatternRewriter& rewriter) const {
    const auto tilingStrategy = parseIntArrayAttr<int64_t>(mlir::cast<mlir::ArrayAttr>(vfOp.getTilingStrategy()));

    DimArr dims;
    int64_t tilesLen = 0;
    std::tie(dims, tilesLen) = getDimsData(tilingStrategy);

    if (tilesLen <= 1) {
        return mlir::failure();
    }

    _log.trace("Applying tiling for VF op at {0}: tilingStrategy = {1}", vfOp->getLoc(), tilingStrategy);

    auto vfConfig = createVFConfig(vfOp);

    auto operationStorage = std::make_unique<TilingOperationStorage>();
    auto tilingStorage = restoreTilingStorage(vfConfig, tilingStrategy, operationStorage);

    // resultTileVals = {{res_0_tile_0, res_0_tile_1, ...}, {res_1_tile_0, res_1_tile_1, ...}, ...}
    auto resultTileVals = SmallVector<SmallVector<mlir::Value>>(vfOp->getNumResults(), {});
    auto resultTileOffsets = SmallVector<SmallVector<Shape>>(vfOp->getNumResults(), {});
    for (size_t i = 0; i < vfOp->getNumResults(); ++i) {
        resultTileVals[i].reserve(tilesLen);
        resultTileOffsets[i].reserve(tilesLen);
    }

    DenseMap<int64_t, mlir::IRMapping> mappers;
    DenseMap<mlir::Operation*, DenseMap<int64_t, SmallVector<mlir::Value>>> opTileResults;

    const auto tilingProcedure = [&](int64_t index, mlir::Operation* op, SmallVector<mlir::Value>& currentResults,
                                     SmallVector<Shape>& currentTiles) {
        auto inputTiling = operationStorage->get(op, index);
        VPUX_THROW_WHEN(!inputTiling.has_value(), "Couldn't find tile information for operation {0} and tile {1}", *op,
                        index);

        currentTiles.clear();

        const auto& inputTilingPair = inputTiling.value();
        auto inputTilingInfo = inputTilingPair.first;
        auto& mapper = mappers[index];

        auto firstSameTile = operationStorage->findFirstTile(op, inputTilingPair);
        if (firstSameTile.has_value() && firstSameTile.value() < static_cast<size_t>(index)) {
            // already calculated in previous tile.
            VPUX_THROW_UNLESS(opTileResults.count(op) > 0 && opTileResults[op].count(firstSameTile.value()) > 0,
                              "Tile result not found for operation {0} and tile {1}", op->getLoc(),
                              firstSameTile.value());
            currentResults = opTileResults[op][firstSameTile.value()];
            mapper.map(op->getResults(), currentResults);

            // Also restore per-result tile offsets; they are needed later when building Concat static_offsets.
            if (auto tiledBuilderOp = mlir::dyn_cast<VPU::TilingBuilderOpInterface>(op)) {
                const auto outTiles = tiledBuilderOp.getOutputTiling(inputTilingPair.second, _log);
                currentTiles.reserve(outTiles.size());
                for (const auto& outTile : outTiles) {
                    currentTiles.push_back(outTile.offsets);
                }
            } else {
                // View-like ops are expected to have a single result.
                currentTiles.push_back(inputTilingPair.second.offsets);
            }
            return;
        }

        for (auto operand : op->getOperands()) {
            if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(operand)) {
                const auto valName = printToString("ba_input {0}", index);
                auto origInput = vfOp.getOperand(blockArg.getArgNumber());
                const auto& tileInfo = tilingStorage.get(std::make_pair(blockArg.getArgNumber(), op), index);

                VPUX_THROW_WHEN(!tileInfo.has_value(), "Couldn't find tile information for argument {0} and tile {1}",
                                blockArg.getArgNumber(), index);
                auto operandTile = VPU::makeTile(rewriter, op->getLoc(), origInput, tileInfo.value(), valName);

                mapper.map(operand, operandTile);
            }
        }

        adjustInputShape(rewriter, op, inputTilingInfo, mapper, tilingStorage, operationStorage, index, dims);

        auto* copiedOp = rewriter.clone(*op, mapper);
        extendOpLoc(copiedOp, "slice_{0}", index);
        currentResults = copiedOp->getResults();
        OutputTiling crtOpOutputTiling = {inputTilingPair.second};

        const auto baseResType = mlir::cast<NDTypeInterface>(op->getResult(0).getType());
        if (auto tiledBuilderOp = mlir::dyn_cast<VPU::TilingBuilderOpInterface>(copiedOp)) {
            tiledBuilderOp.adjustAttrs(inputTilingInfo, crtOpOutputTiling.front());
            crtOpOutputTiling = tiledBuilderOp.getOutputTiling(crtOpOutputTiling.front(), _log);
        } else if (auto tiledViewOp = mlir::dyn_cast<VPU::TilingViewLikeOpInterface>(copiedOp)) {
            tiledViewOp.adjustAttrs(inputTilingInfo, inputTilingPair.second, baseResType.getShape());
        }

        // Tile results
        const unsigned numOpResults = copiedOp->getNumResults();
        VPUX_THROW_WHEN(crtOpOutputTiling.size() != numOpResults,
                        "getOutputTiling returned {0} tiles for op '{1}' @ {2}, expected {3}", crtOpOutputTiling.size(),
                        op->getName(), op->getLoc(), numOpResults);

        for (unsigned i = 0; i < numOpResults; ++i) {
            const auto resNdType = mlir::cast<NDTypeInterface>(op->getResult(i).getType());
            const auto& tiledShape = crtOpOutputTiling[i].shape;
            const auto& tiledOffsets = crtOpOutputTiling[i].offsets;

            currentTiles.push_back(tiledOffsets);

            const auto tiledResType = resNdType.extractDenseTile(tiledOffsets, tiledShape);
            copiedOp->getResult(i).setType(tiledResType);
            mapper.map(op->getResult(i), copiedOp->getResult(i));
        }
        opTileResults[op][index] = currentResults;
    };

    VPUX_THROW_UNLESS(vfOp->hasAttrOfType<VFLoopIndexAttr>(VF_LOOP_INDEX_ATTR_NAME),
                      "Op {0} does not contain an attribute {1} of type vpux::VPUIP::VFIndexAttr", vfOp->getLoc(),
                      VF_LOOP_INDEX_ATTR_NAME);
    auto vfIndexAttr = vfOp->getAttrOfType<VFLoopIndexAttr>(VF_LOOP_INDEX_ATTR_NAME);
    assert(vfIndexAttr != nullptr && "vfIndexAttr is null");

    if (vfConfig.isPipelined()) {
        applyPipelinedTiling(tilesLen, vfConfig, resultTileVals, resultTileOffsets, tilingProcedure, operationStorage,
                             vfIndexAttr);
    } else {
        applyLinearTiling(tilesLen, vfConfig, resultTileVals, resultTileOffsets, tilingProcedure, vfIndexAttr);
    }

    _log.trace("Replacing VF at {0} with Concat over {1} tiles", vfOp->getLoc(), tilesLen);
    SmallVector<mlir::Value> replacements;
    replacements.reserve(vfOp->getNumResults());
    for (size_t i = 0; i < vfOp->getNumResults(); ++i) {
        auto concatOp =
                rewriter.create<VPU::ConcatOp>(appendLoc(vfOp.getLoc(), "concat_{0}", i), vfOp->getResult(i).getType(),
                                               mlir::ValueRange(resultTileVals[i]), ArrayRef(resultTileOffsets[i]));
        replacements.push_back(concatOp.getResult());
    }
    rewriter.replaceOp(vfOp, replacements);
    return mlir::success();
}

}  // namespace vpux::VPU
