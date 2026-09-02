//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/IR/ops/internal.hpp"
#include "vpux/compiler/dialect/VPU/utils/strategy_manager/operation_strategies.hpp"
#include "vpux/compiler/utils/thread_safe_hash_map.hpp"

#include <llvm/ADT/DenseMap.h>

#include <memory>

namespace vpux::VPU::VF::v2 {

// Pass-level cache shared by v2 VF rewrites to avoid recomputing tiled type inference,
// CMX requirements, and VPUNN costs for repeated tiling candidates.
class VFCacheAnalysis final {
public:
    explicit VFCacheAnalysis(mlir::Operation*) {
    }

    // Stores dense tile result types keyed by operation, output distribution, strategy, and tile metadata.
    const SmallVector<NDTypeInterface>* getCachedOperationTypes(llvm::hash_code hash) const;
    const SmallVector<NDTypeInterface>& cacheOperationTypes(llvm::hash_code hash, SmallVector<NDTypeInterface>&& types);

    // Stores required CMX size keyed by operation, output distribution, strategy, and full tile metadata.
    std::optional<Byte> getCachedRequiredCMX(llvm::hash_code hash) const;
    Byte cacheRequiredCMX(llvm::hash_code hash, Byte requiredCMX);

    // Stores VPUNN strategy costs keyed by scenario-specific scheduling metadata.
    std::optional<StrategyCost> getCachedStrategyCost(llvm::hash_code hash) const;
    void cacheStrategyCost(llvm::hash_code hash, StrategyCost cost);

private:
    // Tiled types produced by getTileTypes(), keyed by operation hash, output distribution, and tile metadata.
    // Stored behind shared pointers to keep references returned by VFConfig stable across cache lookups.
    ThreadSafeHashMap<llvm::hash_code, std::shared_ptr<SmallVector<NDTypeInterface>>> _tilesCache;

    // Required CMX bytes for one tiled operation, keyed by operation hash, output distribution, strategy, and tiles.
    ThreadSafeHashMap<llvm::hash_code, Byte> _requiredCMXCache;

    // VPUNN strategy costs keyed by operation hash and scheduling parameters built for one tiling candidate.
    ThreadSafeHashMap<llvm::hash_code, StrategyCost> _strategyCostCache;
};

}  // namespace vpux::VPU::VF::v2
