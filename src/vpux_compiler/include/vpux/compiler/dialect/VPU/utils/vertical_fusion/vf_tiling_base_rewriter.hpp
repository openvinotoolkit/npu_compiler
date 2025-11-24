//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/vertical_fusion_utils.hpp"

namespace vpux::VPU {

//
// VerticalFusionTilingRewriter
//

typedef std::function<void(int64_t, mlir::Operation*, mlir::Value&, Shape&)> TilingFunction;

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
    virtual std::pair<DimArr, int64_t> getDimsData(ArrayRef<int64_t> strategy) const = 0;

    virtual TilingStorage restoreTilingStorage(VFConfigType& config, ArrayRef<int64_t> strategy,
                                               TilingOperationStorage::UPtr& operationStorage) const = 0;

    Logger _log;

private:
    void adjustInputShape(mlir::PatternRewriter& rewriter, mlir::Operation* operation, InputTiling& inputTiling,
                          mlir::IRMapping& mapper, TilingStorage& tilingStorage,
                          const TilingOperationStorage::UPtr& opStorage, int64_t tilingIndex, DimArrRef dims) const;
    void processOffset(mlir::Value operand, const TilingOperationStorage::UPtr& opStorage, TileInfo& originalTiling,
                       int64_t tilingIndex, DimArrRef dims, ShapeRef expectedShape) const;
    bool processBlockArgument(mlir::BlockArgument blockArg, TilingStorage& tilingStorage, TileInfo& originalTiling,
                              int64_t tilingIndex, DimArrRef dims) const;
    void applyLinearTiling(const int64_t numTiles, VFConfigType& config, SmallVector<mlir::Value>& resultTileVals,
                           SmallVector<Shape>& resultTileOffsets, const TilingFunction& tilingProcedure) const;
    void applyPipelinedTiling(const int64_t numTiles, VFConfigType& config, SmallVector<mlir::Value>& resultTileVals,
                              SmallVector<Shape>& resultTileOffsets, const TilingFunction& tilingProcedure,
                              const TilingOperationStorage::UPtr& storage) const;

    bool _enableVerticalFusionPipelining;
    const std::unique_ptr<VPU::LayerVPUNNCost>& _vpunnCostFunction;
};

template <typename VFConfigType, typename VFSchedulingFactoryType>
bool VerticalFusionTilingRewriterBase<VFConfigType, VFSchedulingFactoryType>::processBlockArgument(
        mlir::BlockArgument blockArg, TilingStorage& tilingStorage, TileInfo& originalTiling, int64_t tilingIndex,
        DimArrRef dims) const {
    auto& offset = originalTiling.offsets;
    const auto storageInfo = tilingStorage.get(blockArg.getArgNumber(), tilingIndex);
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
        const auto inputOutputTilingPair = inputOutputTiling.value();
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
            if (!processBlockArgument(blockArg, tilingStorage, originalTiling, tilingIndex, dims)) {
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
        const int64_t numTiles, VFConfigType& config, SmallVector<mlir::Value>& resultTileVals,
        SmallVector<Shape>& resultTileOffsets, const TilingFunction& tilingProcedure) const {
    auto operations = config.getVFOperations();

    for (auto index : irange(numTiles)) {
        mlir::Value currentResult;
        Shape currentTile;
        for (auto* op : operations) {
            tilingProcedure(index, op, currentResult, currentTile);
        }

        resultTileVals.push_back(currentResult);
        resultTileOffsets.push_back(currentTile);
    }
}

template <typename VFConfigType, typename VFSchedulingFactoryType>
void VerticalFusionTilingRewriterBase<VFConfigType, VFSchedulingFactoryType>::applyPipelinedTiling(
        const int64_t numTiles, VFConfigType& config, SmallVector<mlir::Value>& resultTileVals,
        SmallVector<Shape>& resultTileOffsets, const TilingFunction& tilingProcedure,
        const TilingOperationStorage::UPtr& storage) const {
    auto scheduling = config.getSubgraph().getScenario();
    VPUX_THROW_WHEN(!scheduling.has_value(), "Cannot get scheduling scenario from VF {0}", config.getSubgraph());

    VFSchedulingFactoryType costFactory(/*prefetching=*/true);
    auto scenario = costFactory.createVFScenario(scheduling.value(), _log);

    if (auto pipelinedScenario = std::dynamic_pointer_cast<IVFPipelinedScheduling<VFConfigType>>(scenario)) {
        auto pipelining = pipelinedScenario->getPipelining(config, numTiles, storage, _vpunnCostFunction);
        auto timeline = pipelining.getTimeLine();
        if (!timeline.empty()) {
            mlir::Value currentResult;
            Shape currentTile;
            for (auto& [index, operation] : pipelining.getTimeLine()) {
                // currentResult and currentTiles keep result from previous call tilingProcedure
                tilingProcedure(index, operation, currentResult, currentTile);

                if (llvm::find(config.getOutputs(), operation) != config.getOutputs().end()) {
                    resultTileVals.push_back(currentResult);
                    resultTileOffsets.push_back(currentTile);
                }
            }
            return;
        }
    }
    applyLinearTiling(numTiles, config, resultTileVals, resultTileOffsets, tilingProcedure);
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

    VFConfigType vfConfig(vfOp, _enableVerticalFusionPipelining);

    auto operationStorage = std::make_unique<TilingOperationStorage>();
    auto tilingStorage = restoreTilingStorage(vfConfig, tilingStrategy, operationStorage);

    SmallVector<mlir::Value> resultTileVals;
    resultTileVals.reserve(tilesLen);
    SmallVector<Shape> resultTileOffsets;
    DenseMap<int64_t, mlir::IRMapping> mappers;

    const auto tilingProcedure = [&](int64_t index, mlir::Operation* op, mlir::Value& currentResult,
                                     Shape& currentTile) {
        auto& mapper = mappers[index];
        for (auto operand : op->getOperands()) {
            if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(operand)) {
                const auto valName = printToString("ba_input {0}", index);
                auto origInput = vfOp.getOperand(blockArg.getArgNumber());
                auto tileInfo = tilingStorage.get(blockArg.getArgNumber(), index);

                VPUX_THROW_WHEN(!tileInfo.has_value(), "Couldn't find tile information for argument {0} and tile {1}",
                                blockArg.getArgNumber(), index);
                auto operandTile = VPU::makeTile(rewriter, op->getLoc(), origInput, tileInfo.value(), valName);

                mapper.map(operand, operandTile);
            }
        }

        auto inputTiling = operationStorage->get(op, index);

        VPUX_THROW_WHEN(!inputTiling.has_value(), "Couldn't find tile information for operation {0} and tile {1}", *op,
                        index);

        const auto inputTilingPair = inputTiling.value();
        auto inputTilingInfo = inputTilingPair.first;
        adjustInputShape(rewriter, op, inputTilingInfo, mapper, tilingStorage, operationStorage, index, dims);

        auto* copiedOp = rewriter.clone(*op, mapper);
        currentResult = copiedOp->getResult(0);

        currentTile = inputTilingPair.second.offsets;
        const auto baseResType = mlir::cast<NDTypeInterface>(op->getResult(0).getType());
        if (auto tiledBuilderOp = mlir::dyn_cast<VPU::TilingBuilderOpInterface>(copiedOp)) {
            tiledBuilderOp.adjustAttrs(inputTilingInfo, inputTilingPair.second);
        } else if (auto tiledViewOp = mlir::dyn_cast<VPU::TilingViewLikeOpInterface>(copiedOp)) {
            tiledViewOp.adjustAttrs(inputTilingInfo, inputTilingPair.second, baseResType.getShape());
        }
        const auto tiledResType =
                baseResType.extractDenseTile(inputTilingPair.second.offsets, inputTilingPair.second.shape);

        currentResult.setType(tiledResType);
        mapper.map(op->getResult(0), currentResult);
    };

    if (vfConfig.isPipelined()) {
        applyPipelinedTiling(tilesLen, vfConfig, resultTileVals, resultTileOffsets, tilingProcedure, operationStorage);
    } else {
        applyLinearTiling(tilesLen, vfConfig, resultTileVals, resultTileOffsets, tilingProcedure);
    }

    rewriter.replaceOpWithNewOp<VPU::ConcatOp>(vfOp, vfOp->getResult(0).getType(), mlir::ValueRange(resultTileVals),
                                               ArrayRef(resultTileOffsets));

    return mlir::success();
}

}  // namespace vpux::VPU
