//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/op_tiling_cache.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/generate_tiling.hpp"
#include "vpux/compiler/dialect/VPU/utils/hash_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/manual_strategy_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/sparsity_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/interfaces/nce_invariant.hpp"
#include "vpux/compiler/utils/hash.hpp"

#include <mlir/IR/OperationSupport.h>

#include <vpu/layer.h>
#include <vpu_layer_strategy.h>

#include <mutex>

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

        auto pipelineTiles = vpux::getHWLayerTilingStrategyWithTileDimOrderForPipelining(
                op, outputShape, isolatedTiles.value().value(), log);

        return mlir::failed(pipelineTiles) ? isolatedTiles.value() : pipelineTiles;
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
    _tilingAccessCount.fetch_add(1, std::memory_order_relaxed);

    std::optional<NTilesOnDim> nTilesOnDim = std::nullopt;
    std::optional<llvm::hash_code> inputOutputModeHash = std::nullopt;
    {
        auto cachedTiles = _tilingCache.find(opHash);
        if (!cachedTiles.has_value()) {
            return std::nullopt;
        }
        nTilesOnDim = cachedTiles.value();

        auto cachedInputOutputModeHash = _opHashToInputOutputModeHash.find(opHash);
        if (!cachedInputOutputModeHash.has_value()) {
            return std::nullopt;
        }
        inputOutputModeHash = cachedInputOutputModeHash.value();
    }

    OutputTilingCacheItem tilingStrategy = mlir::failure();
    if (nTilesOnDim.has_value()) {
        tilingStrategy = fillDividedTiles(op, nTilesOnDim.value(), outputShape);
    }

    auto modeHash = calculateInputOutputModeHash(op, tilingStrategy);
    if (modeHash != inputOutputModeHash) {
        // Distributed output mode is changed, cache is invalid
        return std::nullopt;
    }

    _tilingHitCount.fetch_add(1, std::memory_order_relaxed);
    return tilingStrategy;
}

std::optional<SmallVector<uint32_t>> OpTilingCache::getOpDpuCost(llvm::hash_code opHash) {
    if (!_enableCache) {
        return std::nullopt;
    }
    _dpuCostAccessCount.fetch_add(1, std::memory_order_relaxed);

    auto result = _opDpuCostCache.find(opHash);

    if (result.has_value()) {
        _dpuCostHitCount.fetch_add(1, std::memory_order_relaxed);
    }

    return result;
}

std::optional<PerClusterShapeCacheItem> OpTilingCache::getPerClusterMemoryShapes(llvm::hash_code shapeHash) {
    if (!_enableCache) {
        return std::nullopt;
    }
    _perClusterShapeAccessCount.fetch_add(1, std::memory_order_relaxed);

    auto result = _perClusterShapeCache.find(shapeHash);

    if (result.has_value()) {
        _perClusterShapeHitCount.fetch_add(1, std::memory_order_relaxed);
    }

    return result;
}

std::optional<uint32_t> OpTilingCache::getVPUNNLayerCost(llvm::hash_code layerHash) {
    if (!_enableCache) {
        return std::nullopt;
    }
    _vpunnLayerCostAccessCount.fetch_add(1, std::memory_order_relaxed);

    auto result = _vpunnLayerCostCache.find(layerHash);

    if (result.has_value()) {
        _vpunnLayerCostHitCount.fetch_add(1, std::memory_order_relaxed);
    }

    return result;
}

std::optional<size_t> OpTilingCache::getDPUWorkloadCost(llvm::hash_code opHash) {
    if (!_enableCache) {
        return std::nullopt;
    }
    _dpuTaskOpCostAccessCount.fetch_add(1, std::memory_order_relaxed);

    auto result = _dpuTaskOpCostCache.find(opHash);
    if (result.has_value()) {
        _dpuTaskOpCostHitCount.fetch_add(1, std::memory_order_relaxed);
    }

    return result;
}

std::optional<std::optional<TemporalTilingInfo>> OpTilingCache::getTemporalTilingInfo(llvm::hash_code opHash) {
    if (!_enableCache) {
        return std::nullopt;
    }
    _temporalTilingAccessCount.fetch_add(1, std::memory_order_relaxed);

    auto result = _temporalTilingCache.find(opHash);

    if (result.has_value()) {
        _temporalTilingHitCount.fetch_add(1, std::memory_order_relaxed);
    }

    return result;
}

std::optional<SmallVector<DimArr>> OpTilingCache::getValidPermutations(llvm::hash_code opHash) {
    if (!_enableCache) {
        return std::nullopt;
    }
    _validPermutationsAccessCount.fetch_add(1, std::memory_order_relaxed);

    auto result = _validPermutationsCache.find(opHash);

    if (result.has_value()) {
        _validPermutationsHitCount.fetch_add(1, std::memory_order_relaxed);
    }

    return result;
}

std::optional<DimArr> OpTilingCache::getDimOrder(llvm::hash_code opHash) {
    if (!_enableCache) {
        return std::nullopt;
    }
    _dimOrderAccessCount.fetch_add(1, std::memory_order_relaxed);

    auto result = _dimOrderCache.find(opHash);

    if (result.has_value()) {
        _dimOrderHitCount.fetch_add(1, std::memory_order_relaxed);
    }

    return result;
}

void OpTilingCache::printStats(Logger& logger) const {
    if (!_enableCache) {
        return;
    }

    auto tilingHitCount = _tilingHitCount.load(std::memory_order_relaxed);
    auto tilingAccessCount = _tilingAccessCount.load(std::memory_order_relaxed);

    auto dpuCostHitCount = _dpuCostHitCount.load(std::memory_order_relaxed);
    auto dpuCostAccessCount = _dpuCostAccessCount.load(std::memory_order_relaxed);

    auto vpunnLayerCostHitCount = _vpunnLayerCostHitCount.load(std::memory_order_relaxed);
    auto vpunnLayerCostAccessCount = _vpunnLayerCostAccessCount.load(std::memory_order_relaxed);

    auto perClusterShapeHitCount = _perClusterShapeHitCount.load(std::memory_order_relaxed);
    auto perClusterShapeAccessCount = _perClusterShapeAccessCount.load(std::memory_order_relaxed);

    auto validPermutationsHitCount = _validPermutationsHitCount.load(std::memory_order_relaxed);
    auto validPermutationsAccessCount = _validPermutationsAccessCount.load(std::memory_order_relaxed);

    auto dimOrderHitCount = _dimOrderHitCount.load(std::memory_order_relaxed);
    auto dimOrderAccessCount = _dimOrderAccessCount.load(std::memory_order_relaxed);

    auto dpuTaskOpCostHitCount = _dpuTaskOpCostHitCount.load(std::memory_order_relaxed);
    auto dpuTaskOpCostAccessCount = _dpuTaskOpCostAccessCount.load(std::memory_order_relaxed);

    auto logCacheStats = [&](const char* name, uint64_t hit, uint64_t access) {
        logger.info("{0} cache hit : {1}", name, hit);
        logger.info("{0} cache miss : {1}", name, access - hit);
        if (access != 0) {
            logger.info("{0} cache hit rate: {1}%", name, hit * 100.0 / access);
        }
    };

    logCacheStats("Tiling", tilingHitCount, tilingAccessCount);
    logCacheStats("DPU cost", dpuCostHitCount, dpuCostAccessCount);
    logCacheStats("VPUNNLayer cost", vpunnLayerCostHitCount, vpunnLayerCostAccessCount);
    logCacheStats("Shape with distributionInfo", perClusterShapeHitCount, perClusterShapeAccessCount);
    logCacheStats("Valid permutations", validPermutationsHitCount, validPermutationsAccessCount);
    logCacheStats("Dim Order", dimOrderHitCount, dimOrderAccessCount);
    logCacheStats("DPU Workload cost", dpuTaskOpCostHitCount, dpuTaskOpCostAccessCount);

    auto temporalTilingHitCount = _temporalTilingHitCount.load(std::memory_order_relaxed);
    auto temporalTilingAccessCount = _temporalTilingAccessCount.load(std::memory_order_relaxed);
    logCacheStats("Temporal tiling", temporalTilingHitCount, temporalTilingAccessCount);
}

void OpTilingCache::updateOutputTiling(const llvm::hash_code opHash, mlir::Operation* op,
                                       const OutputTilingCacheItem& outputTiling) {
    if (!_enableCache) {
        return;
    }

    auto outputModeHash = calculateInputOutputModeHash(op, outputTiling);

    // Store tiling data and mode hash atomically
    if (mlir::failed(outputTiling)) {
        _tilingCache.insert(opHash, std::nullopt);
    } else {
        const auto& outputTilingResult = outputTiling.value();
        VPUX_THROW_WHEN(outputTilingResult.empty(), "Output tiling is empty for op {0}", op->getLoc());
        _tilingCache.insert(opHash, outputTilingResult.front().axis);
    }
    _opHashToInputOutputModeHash.insert(opHash, outputModeHash);
}

void OpTilingCache::updateOpDPUCost(llvm::hash_code opHash, ArrayRef<uint32_t> dpuCosts) {
    if (!_enableCache) {
        return;
    }

    _opDpuCostCache.insert(opHash, SmallVector<uint32_t>{dpuCosts.begin(), dpuCosts.end()});
}

void OpTilingCache::updateVPUNNLayerCost(llvm::hash_code layerHash, uint32_t cost) {
    if (!_enableCache) {
        return;
    }

    _vpunnLayerCostCache.insert(layerHash, cost);
}

void OpTilingCache::updatePerClusterShape(llvm::hash_code shapeHash, const PerClusterShapeCacheItem& perClusterShape) {
    if (!_enableCache) {
        return;
    }

    _perClusterShapeCache.insert(shapeHash, perClusterShape);
}

void OpTilingCache::updateValidPermutations(llvm::hash_code opHash, const SmallVector<DimArr>& validPermutations) {
    if (!_enableCache) {
        return;
    }

    _validPermutationsCache.insert(opHash, validPermutations);
}

void OpTilingCache::updateDimOrder(llvm::hash_code opHash, const DimArr& dimOrder) {
    if (!_enableCache) {
        return;
    }

    _dimOrderCache.insert(opHash, dimOrder);
}

void OpTilingCache::updateDPUWorkloadCost(llvm::hash_code opHash, size_t cost) {
    if (!_enableCache) {
        return;
    }

    _dpuTaskOpCostCache.insert(opHash, cost);
}
void OpTilingCache::updateTemporalTilingInfo(llvm::hash_code opHash, const std::optional<TemporalTilingInfo>& info) {
    if (!_enableCache) {
        return;
    }

    _temporalTilingCache.insert(opHash, info);
}
void OpTilingCache::cleanUp() {
    _tilingAccessCount = 0;
    _tilingHitCount = 0;
    _dpuCostAccessCount = 0;
    _dpuCostHitCount = 0;
    _vpunnLayerCostAccessCount = 0;
    _vpunnLayerCostHitCount = 0;
    _validPermutationsHitCount = 0;
    _validPermutationsAccessCount = 0;
    _perClusterShapeHitCount = 0;
    _perClusterShapeAccessCount = 0;
    _dimOrderHitCount = 0;
    _dimOrderAccessCount = 0;
    _dpuTaskOpCostHitCount = 0;
    _dpuTaskOpCostAccessCount = 0;
    _temporalTilingHitCount = 0;
    _temporalTilingAccessCount = 0;

    _tilingCache.clear();
    _opHashToInputOutputModeHash.clear();
    _opDpuCostCache.clear();
    _vpunnLayerCostCache.clear();
    _validPermutationsCache.clear();
    _dimOrderCache.clear();
    _perClusterShapeCache.clear();
    _dpuTaskOpCostCache.clear();
    _temporalTilingCache.clear();
}

bool OpTilingCache::isCacheSupported() {
    return _enableCache;
}

llvm::hash_code OpTilingCache::calculateOpHash(mlir::Operation* op, const std::optional<DimArrRef>& dimOrder,
                                               const std::optional<OutputTiling>& outputTiling,
                                               const std::optional<mlir::Attribute> multiClusterStrategyAttr) {
    // The hash result is composed of the hash of the op, the dim order, the output tiling and the tiling mode.
    // For the op's hash, it will be calculated based on the op's type, input/output type and its attributes.
    // If the tiling mode is PREFETCHING, the hash will also include the hash of the parent compute op since the parent
    // op will affect the decision of the tiling result.
    if (multiClusterStrategyAttr.has_value()) {
        auto opHash = VPU::hashOperationWithCustomAttr(op, VPU::multiClusterStrategy, multiClusterStrategyAttr.value());
        hashOptionalContents(opHash, dimOrder, outputTiling);
        return opHash;
    }

    auto opHash = VPU::hashOperationForTiling(op);
    hashOptionalContents(opHash, dimOrder, outputTiling);
    return opHash;
}

llvm::hash_code OpTilingCache::calculateOpHashIncludingTilingExcludingAttr(
        mlir::Operation* op, mlir::StringRef excludedAttrName, const std::optional<DimArrRef>& dimOrder,
        const std::optional<OutputTiling>& outputTiling) {
    if (!op->hasAttr(excludedAttrName)) {
        return calculateOpHash(op, dimOrder, outputTiling);
    }
    auto opHash = hashOperationForTilingExcludingAttr(op, excludedAttrName);
    hashOptionalContents(opHash, dimOrder, outputTiling);
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

llvm::hash_code OpTilingCache::calculateVPUNNLayersHash(ArrayRef<VPUNN::DPULayer> vpunnLayers) {
    std::ostringstream layerStream;
    for (const auto& vpunnLayer : vpunnLayers) {
        layerStream << vpunnLayer;
    }
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

namespace vpux {
namespace VPU {

OpTilingCache& getGlobalOpTilingCache() {
    static OpTilingCache globalCache;
    return globalCache;
}

}  // namespace VPU
}  // namespace vpux
