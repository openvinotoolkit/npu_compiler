//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#ifdef BACKGROUND_FOLDING_ENABLED

#include "vpux/compiler/dialect/const/utils/constant_folding_cache.hpp"
#include "vpux/utils/core/disable_warning.hpp"

#include <mutex>

using namespace vpux;

//
// ConstantFoldingCache
//

void Const::ConstantFoldingCache::enqueueRequest(const Const::FoldingRequest& foldingRequest) {
    _requestQueue.push(foldingRequest);
    if (_collectStatistics) {
        _statistics.updateMaxNumRequestsInQueue(
                std::max(static_cast<int64_t>(0), checked_cast<int64_t>(_requestQueue.size())));
    }
}

Const::FoldingRequest Const::ConstantFoldingCache::getRequest() {
    Const::FoldingRequest result;
    NPU_DISABLE_STRINGOP_OVERFLOW(129510)
    _requestQueue.pop(result);
    NPU_DISABLE_WARNING_END
    return result;
}

bool Const::ConstantFoldingCache::hasContent(Const::ContentAttr attr) {
    Const::details::ContentMap::accessor accessor;
    if (_cache.find(accessor, attr) && !accessor.empty()) {
        accessor->second.refCount++;
        return true;
    }
    return false;
}

void Const::ConstantFoldingCache::setMemoryUsageLimit(vpux::Byte memoryUsageLimit) {
    _memoryUsageLimit = memoryUsageLimit.count();
}

void Const::ConstantFoldingCache::setCacheCleanThreshold(double cacheCleanThreshold) {
    _cacheCleanThreshold = cacheCleanThreshold;
}

bool Const::ConstantFoldingCache::isMemoryLimitReached() const {
    return _memoryUsedCache >= _memoryUsageLimit;
}

void Const::ConstantFoldingCache::cleanUpCache() {
    std::vector<std::pair<Const::details::ContentAttrHashCode, int>> contents;
    {
        std::lock_guard<std::mutex> lock(_mutex);
        contents.reserve(_cache.size());
        for (const auto& [attr, cachedContent] : _cache) {
            contents.emplace_back(attr, cachedContent.refCount);
        }
    }
    std::sort(contents.begin(), contents.end(), [](const auto& a, const auto& b) {
        return a.second < b.second;
    });

    for (const auto& entry : contents) {
        removeContent(entry.first);
        if (_memoryUsedCache <= _memoryUsageLimit * _cacheCleanThreshold) {
            break;
        }
    }
}

void Const::ConstantFoldingCache::addContent(Const::details::ContentAttrHashCode attr, Const::Content&& content) {
    Const::details::ContentMap::accessor accessor;
    _cache.insert(accessor, attr);
    VPUX_THROW_WHEN(accessor.empty(), "Failed to add folding request to cache");
    accessor->second.content = std::move(content);
    accessor.release();

    const auto size = attr.totalAllocSize;
    _memoryUsedCache += size.count();

    if (isMemoryLimitReached()) {
        cleanUpCache();
    }

    if (_collectStatistics) {
        _statistics.numElementsAddedToCache++;

        _statistics.updateMaxMemoryUsedCache(_memoryUsedCache.load());
        _statistics.updateMaxCacheSize(_cache.size());
    }
}

void Const::ConstantFoldingCache::removeContent(Const::details::ContentAttrHashCode attr) {
    Const::details::ContentMap::accessor accessor;
    if (_cache.find(accessor, attr) && !accessor.empty()) {
        _cache.erase(accessor);
        accessor.release();

        const auto size = attr.totalAllocSize;
        _memoryUsedCache -= size.count();

        if (_collectStatistics) {
            _statistics.numElementsErasedFromCache++;
        }
    }
}

namespace {
// Note: workaround an issue with background folding keeping the real Content
// object internally and still be able to provide a "copy" to the user outside.
Const::Content referenceAnotherContent(const Const::Content& origin) {
    return Const::Content::fromRawBuffer(origin.getType(), origin.getRawStorageBuf(), origin.getStorageElemType(),
                                         origin.isSplat());
}
}  // namespace

std::optional<Const::Content> Const::ConstantFoldingCache::getContent(Const::ContentAttr attr) {
    // Return the value from the cache if it contains it
    Const::details::ContentMap::accessor accessor;
    if (_cache.find(accessor, attr) && !accessor.empty()) {
        if (_collectStatistics) {
            _statistics.numCacheHits++;
        }
        accessor->second.refCount--;
        return referenceAnotherContent(accessor->second.content);
    }

    if (_collectStatistics) {
        _statistics.numCacheMisses++;
    }
    return std::nullopt;
}

bool Const::ConstantFoldingCache::replaceContentAttr(Const::ContentAttr originalAttr, Const::ContentAttr newAttr) {
    Const::details::ContentMap::accessor originalAttrAccessor;
    if (_cache.find(originalAttrAccessor, originalAttr) && !originalAttrAccessor.empty()) {
        Const::details::ContentMap::accessor newAttrAccessor;
        _cache.insert(newAttrAccessor, newAttr);
        VPUX_THROW_WHEN(newAttrAccessor.empty(), "Failed to add folding request to cache");
        newAttrAccessor->second.content = std::move(originalAttrAccessor->second.content);
        _cache.erase(originalAttrAccessor);
        return true;
    }
    return false;
}

void Const::ConstantFoldingCache::enableStatisticsCollection() {
    _collectStatistics = true;
}

void Const::ConstantFoldingCache::disableStatisticsCollection() {
    _collectStatistics = false;
}

bool Const::ConstantFoldingCache::isStatisticsCollectionEnabled() {
    return _collectStatistics;
}

size_t Const::ConstantFoldingCache::getMemoryUsedCache() const {
    return _memoryUsedCache;
}

Const::details::CacheStatistics& Const::ConstantFoldingCache::getStatistics() {
    return _statistics;
}

//
// CacheStatistics
//

void Const::details::CacheStatistics::updateMaxNumRequestsInQueue(size_t newNumRequests) {
    std::lock_guard<std::mutex> lock(_mtx);
    _maxNumRequestsInQueue = std::max(_maxNumRequestsInQueue, newNumRequests);
}

void Const::details::CacheStatistics::updateMaxCacheSize(size_t newCacheSize) {
    std::lock_guard<std::mutex> lock(_mtx);
    _maxCacheSize = std::max(_maxCacheSize, newCacheSize);
}

void Const::details::CacheStatistics::updateMaxMemoryUsedCache(size_t newMemoryUsedCache) {
    std::lock_guard<std::mutex> lock(_mtx);
    _maxMemoryUsedCache = std::max(_maxMemoryUsedCache, newMemoryUsedCache);
}

size_t Const::details::CacheStatistics::getMaxNumRequestsInQueue() {
    std::lock_guard<std::mutex> lock(_mtx);
    return _maxNumRequestsInQueue;
}

size_t Const::details::CacheStatistics::getMaxCacheSize() {
    std::lock_guard<std::mutex> lock(_mtx);
    return _maxCacheSize;
}

size_t Const::details::CacheStatistics::getMaxMemoryUsedCache() {
    std::lock_guard<std::mutex> lock(_mtx);
    return _maxMemoryUsedCache;
}

//
// ConstantFoldingCacheManager
//

Const::ConstantFoldingCacheManager& Const::ConstantFoldingCacheManager::getInstance() {
    static Const::ConstantFoldingCacheManager instance;
    return instance;
}

bool Const::ConstantFoldingCacheManager::addCache(mlir::MLIRContext* ctx) {
    std::lock_guard<std::mutex> lock(_mtx);
    if (contains(ctx)) {
        return false;
    }
    // Create the cache object for the given context
    _caches[ctx];
    return true;
}

bool Const::ConstantFoldingCacheManager::removeCache(mlir::MLIRContext* ctx) {
    std::lock_guard<std::mutex> lock(_mtx);
    if (auto it = _caches.find(ctx); it != _caches.end()) {
        _caches.erase(it);
        return true;
    }
    return false;
}

bool Const::ConstantFoldingCacheManager::contains(mlir::MLIRContext* ctx) {
    return _caches.find(ctx) != _caches.end();
}

Const::ConstantFoldingCache& Const::ConstantFoldingCacheManager::get(mlir::MLIRContext* ctx) {
    auto it = _caches.find(ctx);
    if (it != _caches.end()) {
        return it->second;
    }
    VPUX_THROW("Unable to find cache for {0}", ctx);
}

#endif
