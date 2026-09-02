//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/core/tiling.hpp"
#include "vpux/compiler/dialect/VPU/utils/scheduling/temporal_tiling_utils.hpp"
#include "vpux/compiler/utils/thread_safe_hash_map.hpp"

#include <atomic>
#include <optional>
#include <utility>

namespace VPUNN {
struct VPULayerStrategy;
struct DPULayer;
}  // namespace VPUNN

namespace vpux {

namespace VPU {
class DistributionInfo;
}  // namespace VPU

namespace VPU {

using OutputTilingCacheItem = mlir::FailureOr<OutputTiling>;

using NTilesOnDim = Shape;

using PerClusterShapeCacheItem = std::optional<SmallVector<Shape>>;

/*
Cache for possible tiling strategies for an operation with specified multi cluster strategy, tiling dim order and
tiling mode.
Uses ThreadSafeHashMap for automatic selection between TBB and mutex-based implementations.
*/
class OpTilingCache {
public:
    OpTilingCache() = default;
    ~OpTilingCache() = default;
    OpTilingCache(const OpTilingCache&) = delete;
    OpTilingCache& operator=(const OpTilingCache&) = delete;

    void enableIfNecessary(bool enable);

    llvm::hash_code calculateOpHash(mlir::Operation* op, const std::optional<DimArrRef>& dimOrder = std::nullopt,
                                    const std::optional<OutputTiling>& outputTile = std::nullopt,
                                    const std::optional<mlir::Attribute> multiClusterStrategyAttr = std::nullopt);

    llvm::hash_code calculateOpHashWithCustomAttr(mlir::Operation* op, mlir::StringRef customAttrName,
                                                  mlir::Attribute customAttrValue,
                                                  const std::optional<DimArrRef>& dimOrder = std::nullopt,
                                                  const std::optional<OutputTiling>& outputTile = std::nullopt);

    llvm::hash_code calculateOpHashIncludingTilingExcludingAttr(
            mlir::Operation* op, mlir::StringRef excludedAttrName,
            const std::optional<DimArrRef>& dimOrder = std::nullopt,
            const std::optional<OutputTiling>& outputTile = std::nullopt);
    llvm::hash_code updateOpHashWithTilingMode(mlir::Operation* op, llvm::hash_code opHash, TilingMode mode);

    llvm::hash_code calculateVPUNNLayerHash(const VPUNN::DPULayer& vpunnLayer,
                                            const VPUNN::VPULayerStrategy& vpunnStrategy);
    llvm::hash_code calculateVPUNNLayersHash(ArrayRef<VPUNN::DPULayer> vpunnLayer);

    llvm::hash_code calculateShapeAndDistributionHash(ShapeRef shape, const VPU::DistributionInfo& distribution);

    OutputTilingCacheItem getHWLayerTilingStrategyWithTileDimOrder(
            mlir::Operation* op, llvm::hash_code opHash, TilingMode tilingMode, DimArrRef tileDimOrder,
            ShapeRef outputShape, const std::optional<OutputTilingCacheItem>& isolatedTiles, Logger log);

    std::optional<OutputTilingCacheItem> getOutputTiling(llvm::hash_code opHash, mlir::Operation* op,
                                                         ShapeRef outputShape);

    std::optional<SmallVector<uint32_t>> getOpDpuCost(llvm::hash_code opHash);

    std::optional<PerClusterShapeCacheItem> getPerClusterMemoryShapes(llvm::hash_code shapeHash);

    std::optional<uint32_t> getVPUNNLayerCost(llvm::hash_code layerHash);

    std::optional<size_t> getDPUWorkloadCost(llvm::hash_code opHash);

    std::optional<std::optional<TemporalTilingInfo>> getTemporalTilingInfo(llvm::hash_code opHash);

    std::optional<SmallVector<DimArr>> getValidPermutations(llvm::hash_code opHash);

    std::optional<DimArr> getDimOrder(llvm::hash_code opHash);

    void updateOutputTiling(const llvm::hash_code opHash, mlir::Operation* op, const OutputTilingCacheItem& outputTile);

    void updateOpDPUCost(llvm::hash_code opHash, ArrayRef<uint32_t> cost);

    void updateVPUNNLayerCost(llvm::hash_code layerHash, uint32_t cost);

    void updatePerClusterShape(llvm::hash_code shapeHash, const PerClusterShapeCacheItem& perClusterShape);

    void updateValidPermutations(llvm::hash_code opHash, const SmallVector<DimArr>& validPermutations);

    void updateDimOrder(llvm::hash_code opHash, const DimArr& validDimOrder);

    void updateDPUWorkloadCost(llvm::hash_code opHash, size_t cost);

    void updateTemporalTilingInfo(llvm::hash_code opHash, const std::optional<TemporalTilingInfo>& info);

    bool isCacheSupported();

    void cleanUp();

    void printStats(Logger& logger) const;

private:
    std::optional<llvm::hash_code> calculateInputOutputModeHash(mlir::Operation* op,
                                                                const OutputTilingCacheItem& outputTiling);

    ThreadSafeHashMap<llvm::hash_code, std::optional<NTilesOnDim>> _tilingCache;
    ThreadSafeHashMap<llvm::hash_code, std::optional<llvm::hash_code>> _opHashToInputOutputModeHash;
    ThreadSafeHashMap<llvm::hash_code, SmallVector<uint32_t>> _opDpuCostCache;
    ThreadSafeHashMap<llvm::hash_code, uint32_t> _vpunnLayerCostCache;
    ThreadSafeHashMap<llvm::hash_code, PerClusterShapeCacheItem> _perClusterShapeCache;
    ThreadSafeHashMap<llvm::hash_code, SmallVector<DimArr>> _validPermutationsCache;
    ThreadSafeHashMap<llvm::hash_code, DimArr> _dimOrderCache;
    ThreadSafeHashMap<llvm::hash_code, size_t> _dpuTaskOpCostCache;
    ThreadSafeHashMap<llvm::hash_code, std::optional<TemporalTilingInfo>> _temporalTilingCache;

    std::atomic<bool> _enableCache{false};

    std::atomic<uint64_t> _tilingHitCount{0};
    std::atomic<uint64_t> _tilingAccessCount{0};

    std::atomic<uint64_t> _dpuCostHitCount{0};
    std::atomic<uint64_t> _dpuCostAccessCount{0};

    std::atomic<uint64_t> _vpunnLayerCostHitCount{0};
    std::atomic<uint64_t> _vpunnLayerCostAccessCount{0};

    std::atomic<uint64_t> _perClusterShapeHitCount{0};
    std::atomic<uint64_t> _perClusterShapeAccessCount{0};

    std::atomic<uint64_t> _validPermutationsHitCount{0};
    std::atomic<uint64_t> _validPermutationsAccessCount{0};

    std::atomic<uint64_t> _dimOrderHitCount{0};
    std::atomic<uint64_t> _dimOrderAccessCount{0};

    std::atomic<uint64_t> _dpuTaskOpCostHitCount{0};
    std::atomic<uint64_t> _dpuTaskOpCostAccessCount{0};

    std::atomic<uint64_t> _temporalTilingHitCount{0};
    std::atomic<uint64_t> _temporalTilingAccessCount{0};
};

// Global instance accessor
// This provides singleton-like behavior for the tiling cache
OpTilingCache& getGlobalOpTilingCache();

}  // namespace VPU
}  // namespace vpux
