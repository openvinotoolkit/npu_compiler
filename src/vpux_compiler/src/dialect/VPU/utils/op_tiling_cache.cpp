//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/utils/op_tiling_cache.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/generate_tiling.hpp"
#include "vpux/compiler/dialect/VPU/utils/manual_strategy_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/sparsity_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/interfaces/nce_invariant.hpp"
#include "vpux/compiler/utils/VPU/tile_utils.hpp"
#include "vpux/compiler/utils/hash.hpp"

#include <mlir/IR/OperationSupport.h>

using namespace vpux;
using namespace VPU;

void OpTilingCache::enableIfNecessary(bool enable) {
    _enableCache = enable;
}

mlir::FailureOr<OutputTiling> OpTilingCache::getHWLayerTilingStrategyWithTileDimOrder(
        mlir::Operation* op, llvm::hash_code opHash, TilingMode tilingMode, DimArrRef tileDimOrder,
        ShapeRef outputShape, const std::optional<OutputTilingCacheItem>& isolatedTiles, Logger log) {
    auto calculateTilingStrategy = [&]() -> mlir::FailureOr<OutputTiling> {
        if (tilingMode == TilingMode::ISOLATED || tilingMode == TilingMode::PREFETCHING) {
            return vpux::getHWLayerTilingStrategyWithTileDimOrderForIsolatedOrPrefetch(op, tilingMode, tileDimOrder,
                                                                                       outputShape, log);
        }
        VPUX_THROW_UNLESS(isolatedTiles.has_value(), "Isolated tiles must be provided for pipelining mode");
        if (mlir::failed(isolatedTiles.value())) {
            // If isolated tiling failed, return failure
            return mlir::failure();
        }
        return vpux::getHWLayerTilingStrategyWithTileDimOrderForPipelining(op, outputShape,
                                                                           isolatedTiles.value().value(), log);
    };

    const auto useCache = isCacheSupported();
    if (!useCache) {
        return calculateTilingStrategy();
    }

    auto opHashWithTilingMode = updateOpHashWithTilingMode(op, opHash, tilingMode);
    auto cacheResult = getOutputTiling(opHashWithTilingMode, op, outputShape);
    if (cacheResult.has_value()) {
        return std::move(cacheResult.value());
    }
    auto tilingStrategy = calculateTilingStrategy();
    updateOutputTiling(opHashWithTilingMode, op, tilingStrategy);

    return tilingStrategy;
}

std::optional<OutputTilingCacheItem> OpTilingCache::getOutputTiling(llvm::hash_code opHash, mlir::Operation* op,
                                                                    ShapeRef outputShape) {
    if (!_enableCache) {
        return std::nullopt;
    }
    _tilingAccessCount++;
    std::optional<NTilesOnDim> nTilesOnDim = std::nullopt;
    std::optional<llvm::hash_code> inputOutputModeHash = std::nullopt;
    {
        std::lock_guard<std::mutex> lock(_tilingMutex);
        auto it = _tilingCache.find(opHash);
        if (it == _tilingCache.end()) {
            return std::nullopt;
        }

        auto cachedInputOutputModeHash = _opHashToInputOutputModeHash.find(opHash);
        if (cachedInputOutputModeHash == _opHashToInputOutputModeHash.end()) {
            return std::nullopt;
        }
        inputOutputModeHash = cachedInputOutputModeHash->second;
        nTilesOnDim = it->second;
    }

    OutputTilingCacheItem tilingStrategy = mlir::failure();
    if (nTilesOnDim.has_value()) {
        tilingStrategy = fillDividedTiles(op, nTilesOnDim.value(), outputShape);
    }
    auto modeHash = calculateInputOutputModeHash(op, tilingStrategy);
    if (modeHash != inputOutputModeHash) {
        // Disitrubted output mode is changed, cache is invalid
        return std::nullopt;
    }

    _tilingHitCount++;
    return tilingStrategy;
}

std::optional<SmallVector<uint32_t>> OpTilingCache::getOpDpuCost(llvm::hash_code opHash) {
    if (!_enableCache) {
        return std::nullopt;
    }
    _dpuCostAccessCount++;
    std::lock_guard<std::mutex> lock(_dpuMutex);
    auto it = _opDpuCostCache.find(opHash);
    if (it == _opDpuCostCache.end()) {
        return std::nullopt;
    }
    _dpuCostHitCount++;
    return it->second;
}

std::optional<PerClusterShapeCacheItem> OpTilingCache::getPerClusterMemoryShapes(llvm::hash_code shapeHash) {
    if (!_enableCache) {
        return std::nullopt;
    }
    _perClusterShapeAccessCount++;
    std::lock_guard<std::mutex> lock(_perClusterShapeMutex);
    auto it = _perClusterShapeCache.find(shapeHash);
    if (it == _perClusterShapeCache.end()) {
        return std::nullopt;
    }
    _perClusterShapeHitCount++;
    return it->second;
}

std::optional<uint32_t> OpTilingCache::getVPUNNLayerCost(llvm::hash_code layerHash) {
    if (!_enableCache) {
        return std::nullopt;
    }
    _vpunnLayerCostAccessCount++;
    std::lock_guard<std::mutex> lock(_vpunnLayerMutex);
    auto it = _vpunnLayerCostCache.find(layerHash);
    if (it == _vpunnLayerCostCache.end()) {
        return std::nullopt;
    }
    _vpunnLayerCostHitCount++;
    return it->second;
}

void OpTilingCache::printStats(Logger& logger) const {
    if (!_enableCache) {
        return;
    }

    logger.info("Tiling cache hit : {0}", _tilingHitCount);
    logger.info("Tiling cache miss : {0}", _tilingAccessCount - _tilingHitCount);
    if (_tilingAccessCount != 0) {
        logger.info("Tiling cache hit rate: {0}%", _tilingHitCount * 100.0 / _tilingAccessCount);
    }

    logger.info("DPU cost cache hit : {0}", _dpuCostHitCount);
    logger.info("DPU cost cache miss : {0}", _dpuCostAccessCount - _dpuCostHitCount);
    if (_dpuCostAccessCount != 0) {
        logger.info("DPU cost cache hit rate: {0}%", _dpuCostHitCount * 100.0 / _dpuCostAccessCount);
    }

    logger.info("VPUNNLayer cost cache hit : {0}", _vpunnLayerCostHitCount);
    logger.info("VPUNNLayer cost cache miss : {0}", _vpunnLayerCostAccessCount - _vpunnLayerCostHitCount);
    if (_vpunnLayerCostAccessCount != 0) {
        logger.info("VPUNNLayer cost cache hit rate: {0}%",
                    _vpunnLayerCostHitCount * 100.0 / _vpunnLayerCostAccessCount);
    }

    logger.info("Shape with distributionInfo cache hit : {0}", _perClusterShapeHitCount);
    logger.info("Shape with distributionInfo cache miss : {0}", _perClusterShapeAccessCount - _perClusterShapeHitCount);
    if (_perClusterShapeAccessCount != 0) {
        logger.info("Shape with distributionInfo cache hit rate: {0}%",
                    _perClusterShapeHitCount * 100.0 / _perClusterShapeAccessCount);
    }
}

void OpTilingCache::updateOutputTiling(const llvm::hash_code opHash, mlir::Operation* op,
                                       const OutputTilingCacheItem& outputTiling) {
    auto outputModeHash = calculateInputOutputModeHash(op, outputTiling);
    std::lock_guard<std::mutex> lock(_tilingMutex);
    if (mlir::failed(outputTiling)) {
        _tilingCache.insert({opHash, std::nullopt});
    } else {
        const auto& outputTilingResult = outputTiling.value();
        VPUX_THROW_WHEN(outputTilingResult.empty(), "Output tiling is empty for op {0}", op->getLoc());
        _tilingCache.insert({opHash, outputTilingResult.front().axis});
    }
    _opHashToInputOutputModeHash.insert({opHash, outputModeHash});
}

void OpTilingCache::updateOpDPUCost(llvm::hash_code opHash, ArrayRef<uint32_t> dpuCosts) {
    if (!_enableCache) {
        return;
    }
    std::lock_guard<std::mutex> lock(_dpuMutex);
    _opDpuCostCache[opHash] = SmallVector<uint32_t>{dpuCosts.begin(), dpuCosts.end()};
}

void OpTilingCache::updateVPUNNLayerCost(llvm::hash_code layerHash, uint32_t cost) {
    if (!_enableCache) {
        return;
    }
    std::lock_guard<std::mutex> lock(_vpunnLayerMutex);
    _vpunnLayerCostCache[layerHash] = cost;
}

void OpTilingCache::updatePerClusterShape(llvm::hash_code shapeHash, const PerClusterShapeCacheItem& perClusterShape) {
    if (!_enableCache) {
        return;
    }
    std::lock_guard<std::mutex> lock(_perClusterShapeMutex);
    _perClusterShapeCache[shapeHash] = perClusterShape;
}

void OpTilingCache::cleanUp() {
    _tilingAccessCount = 0;
    _tilingHitCount = 0;
    _dpuCostAccessCount = 0;
    _dpuCostHitCount = 0;
    _vpunnLayerCostAccessCount = 0;
    _vpunnLayerCostHitCount = 0;

    {
        std::lock_guard<std::mutex> lock(_tilingMutex);
        _tilingCache.clear();
        _opHashToInputOutputModeHash.clear();
    }
    {
        std::lock_guard<std::mutex> lock(_dpuMutex);
        _opDpuCostCache.clear();
    }

    {
        std::lock_guard<std::mutex> lock(_vpunnLayerMutex);
        _vpunnLayerCostCache.clear();
    }
    std::lock_guard<std::mutex> lock(_perClusterShapeMutex);
    _perClusterShapeCache.clear();
}

bool OpTilingCache::isCacheSupported() {
    return _enableCache;
}

llvm::hash_code OpTilingCache::calculateOpHash(mlir::Operation* op, const std::optional<DimArrRef>& dimOrder,
                                               const std::optional<OutputTiling>& outputTiling) {
    // The hash result is composed of the hash of the op, the dim order, the output tiling and the tiling mode.
    // For the op's hash, it will be calculated based on the op's type, input/output type and its attributes.
    // If the tiling mode is PREFETCHING, the hash will also include the hash of the parent compute op since the parent
    // op will affect the decision of the tiling result.
    auto opHash = vpux::hashOperationForTiling(op);
    if (dimOrder.has_value()) {
        auto getDimArrHash = [](DimArrRef dimOrder) {
            return llvm::hash_value(llvm::formatv("{0}", dimOrder).str());
        };
        opHash = llvm::hash_combine(opHash, getDimArrHash(dimOrder.value()));
    }

    if (outputTiling.has_value()) {
        opHash = llvm::hash_combine(opHash, llvm::hash_value(llvm::formatv("{0}", outputTiling.value()).str()));
    }
    return opHash;
}

llvm::hash_code OpTilingCache::updateOpHashWithTilingMode(mlir::Operation* op, llvm::hash_code opHash,
                                                          TilingMode mode) {
    opHash = llvm::hash_combine(opHash, mode);
    if (mode == TilingMode::PREFETCHING) {
        if (auto parentOp = VPU::getParentComputeOp(op)) {
            opHash = llvm::hash_combine(
                    opHash, VPUIP::NCEInvariant::getRequiredCMXSizeForLastTile(parentOp, Logger::global()).count());
        }
    }
    return opHash;
}

llvm::hash_code OpTilingCache::calculateShapeAndDistributionHash(ShapeRef shape,
                                                                 const VPU::DistributionInfo& distribution) {
    auto hash = llvm::hash_combine_range(shape.begin(), shape.end());
    hash = llvm::hash_combine(hash, distribution.getDistributionMode(), distribution.getNumClusters(),
                              distribution.hasUniformDistributedSegments(), distribution.hasEqualMemoryAndComputeView(),
                              distribution.getNumTiles(), distribution.getKernel(), distribution.getStrides(),
                              distribution.getAlignment());
    for (auto& shape : distribution.getComputeShapes()) {
        hash = llvm::hash_combine(hash, ArrayRef<int64_t>(shape));
    }
    for (auto& offset : distribution.getComputeOffsets()) {
        hash = llvm::hash_combine(hash, ArrayRef<int64_t>(offset));
    }
    for (auto& shape : distribution.getMemoryShapes()) {
        hash = llvm::hash_combine(hash, ArrayRef<int64_t>(shape));
    }
    for (auto& offset : distribution.getMemoryOffsets()) {
        hash = llvm::hash_combine(hash, ArrayRef<int64_t>(offset));
    }
    auto pad = distribution.getPadding();
    if (pad.has_value()) {
        auto padVal = pad.value();
        hash = llvm::hash_combine(hash, padVal.getBottomPad(), padVal.getLeftPad(), padVal.getRightPad(),
                                  padVal.getTopPad());
    }
    return hash;
}

llvm::hash_code OpTilingCache::calculateVPUNNLayerHash(const VPUNN::DPULayer& vpunnLayer,
                                                       const VPUNN::VPULayerStrategy& vpunnStrategy) {
    std::ostringstream layerStream;
    layerStream << vpunnLayer;
    layerStream << vpunnStrategy;
    return llvm::hash_value(layerStream.str());
}

std::optional<llvm::hash_code> OpTilingCache::calculateInputOutputModeHash(mlir::Operation* op,
                                                                           const OutputTilingCacheItem& outputTiling) {
    if (!op->hasAttr(vpux::multiClusterStrategy)) {
        return std::nullopt;
    }
    auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(op);
    VPUX_THROW_WHEN(clusteredOp == nullptr, "Op {0} is not a clustered op but has MultiClusterStrategy attr",
                    op->getLoc());
    auto mcStrategy = clusteredOp.getMultiClusterStrategy().value();
    auto outputType = mlir::dyn_cast<vpux::NDTypeInterface>(op->getResult(0).getType());
    if (auto sparseType = mlir::dyn_cast<VPU::SparseTensorType>(outputType)) {
        outputType = mlir::cast<NDTypeInterface>(VPU::getEffectiveSparseOutputType(sparseType));
    }

    SmallVector<VPU::DistributionMode> inputOutputMode;
    inputOutputMode.push_back(getActivationTensorDistributionMode(clusteredOp, mcStrategy));
    if (mlir::succeeded(outputTiling)) {
        auto tilingResult = outputTiling.value();
        OutputTiling uniqueOutputTiling;
        std::set<Shape> uniqueShapes;
        for (auto& tile : tilingResult) {
            if (uniqueShapes.count(tile.shape) > 0) {
                continue;
            }
            uniqueOutputTiling.push_back(tile);
            uniqueShapes.insert(tile.shape);
        }
        for (auto& outputTile : uniqueOutputTiling) {
            const auto outputTileType = outputType.extractDenseTile(outputTile.offsets, outputTile.shape);
            inputOutputMode.push_back(getOutputTensorDistributionMode(clusteredOp, mcStrategy, outputTileType));
        }
    } else {
        inputOutputMode.push_back(getOutputTensorDistributionMode(clusteredOp, mcStrategy, outputType));
    }
    return llvm::hash_combine_range(inputOutputMode.begin(), inputOutputMode.end());
}
