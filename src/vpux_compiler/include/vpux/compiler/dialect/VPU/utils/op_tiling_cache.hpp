//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/core/tiling.hpp"
#include "vpux/utils/core/dense_map.hpp"

#include <atomic>
#include <mutex>
namespace VPUNN {
struct VPULayerStrategy;
struct DPULayer;
}  // namespace VPUNN

namespace vpux::VPU {
class DistributionInfo;
}

namespace vpux {
namespace VPU {

using OutputTilingCacheItem = mlir::FailureOr<OutputTiling>;

using NTilesOnDim = Shape;

using PerClusterShapeCacheItem = std::optional<SmallVector<Shape>>;

/*
Cache for possible tiling strategies for an operation with specified multi cluster strategy, tiling dim order and
tiling mode.
*/

class OpTilingCache {
public:
    ~OpTilingCache() = default;
    OpTilingCache(const OpTilingCache&) = delete;
    OpTilingCache& operator=(const OpTilingCache&) = delete;

    static OpTilingCache& instance() {
        static OpTilingCache cache;
        return cache;
    }

    void enableIfNecessary(bool enable);

    llvm::hash_code calculateOpHash(mlir::Operation* op, const std::optional<DimArrRef>& dimOrder = std::nullopt,
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

    std::optional<SmallVector<DimArr>> getValidPermutations(llvm::hash_code opHash);

    std::optional<DimArr> getDimOrder(llvm::hash_code opHash);

    void updateOutputTiling(const llvm::hash_code opHash, mlir::Operation* op, const OutputTilingCacheItem& outputTile);

    void updateOpDPUCost(llvm::hash_code opHash, ArrayRef<uint32_t> cost);

    void updateVPUNNLayerCost(llvm::hash_code layerHash, uint32_t cost);

    void updatePerClusterShape(llvm::hash_code shapeHash, const PerClusterShapeCacheItem& perClusterShape);

    void updateValidPermutations(llvm::hash_code opHash, const SmallVector<DimArr>& validPermutations);

    void updateDimOrder(llvm::hash_code opHash, const DimArr& validDimOrder);

    bool isCacheSupported();

    void cleanUp();

    void printStats(Logger& logger) const;

private:
    OpTilingCache() = default;

    std::optional<llvm::hash_code> calculateInputOutputModeHash(mlir::Operation* op,
                                                                const OutputTilingCacheItem& outputTiling);

    std::mutex _tilingMutex;
    std::mutex _dpuMutex;
    std::mutex _vpunnLayerMutex;
    std::mutex _perClusterShapeMutex;
    std::mutex _validPermutationsMutex;
    std::mutex _dimOrderMutex;
    DenseMap<llvm::hash_code, std::optional<NTilesOnDim>> _tilingCache;
    DenseMap<llvm::hash_code, std::optional<llvm::hash_code>> _opHashToInputOutputModeHash;
    DenseMap<llvm::hash_code, SmallVector<uint32_t>> _opDpuCostCache;
    DenseMap<llvm::hash_code, uint32_t> _vpunnLayerCostCache;
    DenseMap<llvm::hash_code, PerClusterShapeCacheItem> _perClusterShapeCache;
    DenseMap<llvm::hash_code, SmallVector<DimArr>> _validPermutationsCache;
    DenseMap<llvm::hash_code, DimArr> _dimOrderCache;

    bool _enableCache{false};

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
};
}  // namespace VPU
}  // namespace vpux
