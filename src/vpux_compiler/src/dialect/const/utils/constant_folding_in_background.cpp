//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#ifdef BACKGROUND_FOLDING_ENABLED

#include "vpux/compiler/dialect/const/utils/constant_folding_in_background.hpp"
#include "vpux/compiler/dialect/const/utils/constant_folding_cache.hpp"

#include <mutex>

using namespace vpux;
using namespace vpux::Const;

BackgroundConstantFolding::BackgroundConstantFolding(mlir::MLIRContext* ctx, size_t maxConcurrentTasks,
                                                     bool collectStatistics, size_t memoryUsageLimit,
                                                     double cacheCleanThreshold, Logger log)
        : _ctx(ctx), _maxConcurrentTasks(maxConcurrentTasks), _log(log) {
    if (!_ctx->isMultithreadingEnabled()) {
        _log.info("Multi thread is disabled, background constant folding is disabled");
        _isEnabled = false;
        return;
    }

    auto& threadPool = _ctx->getThreadPool();

    if (threadPool.getMaxConcurrency() <= 1) {
        _log.info("The thread pool size is not greater than 1, background constant folding is disabled");
        _isEnabled = false;
        return;
    }

    auto& cacheManager = ConstantFoldingCacheManager::getInstance();
    cacheManager.addCache(_ctx);
    vpux::MB memoryUsageLimitMB(memoryUsageLimit);
    auto memoryUsageLimitBytes = memoryUsageLimitMB.to<vpux::Byte>();
    cacheManager.get(_ctx).setMemoryUsageLimit(memoryUsageLimitBytes);
    cacheManager.get(_ctx).setCacheCleanThreshold(cacheCleanThreshold);
    if (collectStatistics) {
        cacheManager.get(_ctx).enableStatisticsCollection();
    }

    _listenerThread = initFoldingListener(threadPool);
}

BackgroundConstantFolding::~BackgroundConstantFolding() {
    if (!_isEnabled) {
        return;
    }

    try {
        stopFoldingListener();

        std::unique_lock<std::mutex> lock(_mutex);
        _cv.wait(lock, [this] {
            return _activeTasks == 0;
        });
    } catch (...) {
        _log.error("Exception caught in BackgroundConstantFolding destructor");
    }
}

SmallVector<TransformAttrInterface> vpux::Const::BackgroundConstantFolding::stripTransformationsFrom(
        ArrayRef<TransformAttrInterface> transformations, TransformAttrInterface transformation) {
    auto it = std::find(transformations.rbegin(), transformations.rend(), transformation);

    if (it == transformations.rend()) {
        return {};
    }

    return SmallVector<TransformAttrInterface>(transformations.begin(), (it + 1).base());
}

SmallVector<TransformAttrInterface> vpux::Const::BackgroundConstantFolding::getLastTransformationsFrom(
        ArrayRef<TransformAttrInterface> transformations, TransformAttrInterface transformation) {
    auto it = std::find(transformations.rbegin(), transformations.rend(), transformation);

    if (it == transformations.rend()) {
        return SmallVector<TransformAttrInterface>(transformations);
    }

    return SmallVector<TransformAttrInterface>((it + 1).base(), transformations.end());
}

// Checks if the ContentAttr with the list of transformations up to the new one already has the folded result in the
// cache. If it does, it will be reused and only the new transformation (and any other that is placed after the new
// transformation in the list) will be applied.
// Returns true if this partial folding has succeeded, otherwise false.
//
// For example:
// New ContentAttr: [Transformation1, Transformation2, NewTransformation, Transformation3]
// Checks cache for presence of ContentAttr: [Transformation1, Transformation2]
// If the folded value is found in the cache, transformations [NewTransformation, Transformation3] are called over it
bool tryFoldingPartially(vpux::Const::ConstantFoldingCache& cache, Const::ContentAttr request,
                         Const::TransformAttrInterface newTransformation) {
    if (newTransformation == nullptr) {
        return false;
    }
    auto partialTransformations = vpux::Const::BackgroundConstantFolding::stripTransformationsFrom(
            request.getTransformations(), newTransformation);
    auto partialContentAttr = ContentAttr::get(request.getBaseContent(), partialTransformations);
    auto maybeFoldedPartialContent = cache.getContent(partialContentAttr);
    if (!maybeFoldedPartialContent.has_value()) {
        return false;
    }
    auto foldedPartialContent = std::move(maybeFoldedPartialContent.value());

    auto lastTransformations = vpux::Const::BackgroundConstantFolding::getLastTransformationsFrom(
            request.getTransformations(), newTransformation);
    if (lastTransformations.empty()) {
        return false;
    }

    auto partialContent =
            Const::Content::fromRawBuffer(foldedPartialContent.getType(), foldedPartialContent.getRawTempBuf(),
                                          foldedPartialContent.getStorageElemType(), foldedPartialContent.isSplat());
    for (auto tr : lastTransformations) {
        partialContent = tr.transform(partialContent);
    }

    // Create a copy of the Content which will own the referenced buffer
    // This is done since the Content object obtained after folding may reference an external object without
    // owning it. If that object is erased, the Content object from the cache would point to an invalid object
    cache.addContent(request, Const::Content::copyUnownedBuffer(std::move(partialContent)));
    cache.removeContent(partialContentAttr);

    return true;
}

class TaskCompletionNotifier {
public:
    TaskCompletionNotifier(std::atomic<size_t>& activeTasks, std::condition_variable& cv)
            : _activeTasks(&activeTasks), _cv(&cv) {
    }

    ~TaskCompletionNotifier() {
        (*_activeTasks)--;
        _cv->notify_one();
    }

private:
    std::atomic<size_t>* const _activeTasks;
    std::condition_variable* const _cv;
};

void BackgroundConstantFolding::processFoldingRequest(FoldingRequest&& req, ConstantFoldingCache& cache) {
    // As the main compilation process also utilizes the pool concurrently, _maxConcurrentTasks is introduced to
    // manually control the resources used by constant folding in background.
    std::unique_lock<std::mutex> lock(_mutex);
    _cv.wait(lock, [this] {
        return _activeTasks < _maxConcurrentTasks;
    });

    _activeTasks++;
    _ctx->getThreadPool().async([this, foldingRequest = std::move(req), &cache]() {
        TaskCompletionNotifier notifier(_activeTasks, _cv);

        Const::ContentAttr request;
        if (auto* equivalenceRequest = std::get_if<Const::EquivalenceRequest>(&foldingRequest.attr)) {
            // E#135947: instead of doing replace here (which at worst does
            // nothing and at best doesn't prevent doing too much), maintain a
            // mapping: `(old) -> (new)` and *always* give "new attr" request
            // (we expect "new" to be more optimal to compute). otherwise,
            // there's really no point in several declare op canonicalizers and,
            // in return, in the whole equivalence machinery.
            if (cache.replaceContentAttr(equivalenceRequest->originalAttr, equivalenceRequest->newAttr)) {
                return;
            }
            request = equivalenceRequest->newAttr;
        } else {
            VPUX_THROW_WHEN(!std::holds_alternative<Const::ContentAttr>(foldingRequest.attr),
                            "Invalid folding request");
            request = std::get<Const::ContentAttr>(foldingRequest.attr);
        }

        if (cache.hasContent(request)) {
            if (cache.isStatisticsCollectionEnabled()) {
                cache.getStatistics().numDuplicatedRequests++;
            }
            return;
        }

        // Try folding partially if the new transformation is added to the end of the list of transformations
        // In this case, it is likely that the previous ContentAttr (without the new transformation) is already
        // in the cache, so its folded result can be reused
        if (tryFoldingPartially(cache, request, foldingRequest.newTransformation)) {
            return;
        }

        // Create a copy of the Content which will own the referenced buffer
        // This is done since the Content object obtained after folding may reference an external object without
        // owning it. If that object is erased, the Content object from the cache would point to an invalid
        // object
        cache.addContent(request, Const::Content::copyUnownedBuffer(request.fold(/*bypassCache=*/true)));
    });
}

std::shared_future<void> BackgroundConstantFolding::initFoldingListener(llvm::ThreadPoolInterface& threadPool) {
    auto listenerThread = threadPool.async([this]() {
        auto& cacheManager = ConstantFoldingCacheManager::getInstance();
        auto& cache = cacheManager.get(_ctx);

        while (true) {
            auto foldingRequest = cache.getRequest();
            if (std::holds_alternative<FoldingRequest::TerminationToken>(foldingRequest.attr)) {
                break;
            }

            processFoldingRequest(std::move(foldingRequest), cache);
        }
    });

    return listenerThread;
}

void BackgroundConstantFolding::stopFoldingListener() {
    auto& cacheManager = ConstantFoldingCacheManager::getInstance();
    auto& cache = cacheManager.get(_ctx);
    cache.enqueueRequest(Const::FoldingRequest{Const::FoldingRequest::TerminationToken{}, nullptr});

    _listenerThread.wait();

    if (cache.isStatisticsCollectionEnabled()) {
        _log.setName("constant-folding-in-background");
        auto& statistics = cacheManager.get(_ctx).getStatistics();
        _log.info("Cache statistics");
        _log.nest().info("number of cache hits:                       {0}", statistics.numCacheHits);
        _log.nest().info("number of cache misses:                     {0}", statistics.numCacheMisses);
        _log.nest().info("maximum number of requests in queue:        {0}", statistics.getMaxNumRequestsInQueue());
        _log.nest().info("maximum number of elements in cache:        {0}", statistics.getMaxCacheSize());
        _log.nest().info("maximum memory used by cache:               {0}", statistics.getMaxMemoryUsedCache());
        _log.nest().info("current memory used by cache:               {0}",
                         cacheManager.get(_ctx).getMemoryUsedCache());
        _log.nest().info("number of duplicated requests:              {0}", statistics.numDuplicatedRequests);
        _log.nest().info("total number of elements added to cache:    {0}", statistics.numElementsAddedToCache);
        _log.nest().info("total number of elements erased from cache: {0}", statistics.numElementsErasedFromCache);
    }

    cacheManager.removeCache(_ctx);
}

#endif
