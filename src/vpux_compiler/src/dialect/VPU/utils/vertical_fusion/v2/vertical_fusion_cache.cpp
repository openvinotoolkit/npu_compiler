//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/vertical_fusion_cache.hpp"

using namespace vpux;
using namespace VPU;

namespace vpux::VPU::VF::v2 {

const SmallVector<NDTypeInterface>* VFCacheAnalysis::getCachedOperationTypes(llvm::hash_code hash) const {
    auto cachedTypes = _tilesCache.find(hash);
    if (cachedTypes.has_value()) {
        return cachedTypes.value().get();
    }
    return nullptr;
}

const SmallVector<NDTypeInterface>& VFCacheAnalysis::cacheOperationTypes(llvm::hash_code hash,
                                                                         SmallVector<NDTypeInterface>&& types) {
    auto cachedTypes = _tilesCache.find(hash);
    if (cachedTypes.has_value()) {
        return *cachedTypes.value();
    }

    auto insertedTypes = std::make_shared<SmallVector<NDTypeInterface>>(std::move(types));
    _tilesCache.insert(hash, insertedTypes);
    auto storedTypes = _tilesCache.find(hash);
    VPUX_THROW_UNLESS(storedTypes.has_value(), "Expected cached tile types for hash {0}", hash);
    return *storedTypes.value();
}

std::optional<Byte> VFCacheAnalysis::getCachedRequiredCMX(llvm::hash_code hash) const {
    return _requiredCMXCache.find(hash);
}

Byte VFCacheAnalysis::cacheRequiredCMX(llvm::hash_code hash, Byte requiredCMX) {
    auto cachedSize = _requiredCMXCache.find(hash);
    if (cachedSize.has_value()) {
        return cachedSize.value();
    }
    _requiredCMXCache.insert(hash, requiredCMX);
    return requiredCMX;
}

std::optional<StrategyCost> VFCacheAnalysis::getCachedStrategyCost(llvm::hash_code hash) const {
    return _strategyCostCache.find(hash);
}

void VFCacheAnalysis::cacheStrategyCost(llvm::hash_code hash, StrategyCost cost) {
    _strategyCostCache.insert(hash, cost);
}

}  // namespace vpux::VPU::VF::v2
