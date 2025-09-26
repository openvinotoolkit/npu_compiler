//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#ifdef BACKGROUND_FOLDING_ENABLED

#include "vpux/compiler/dialect/const/utils/constant_folding_cache.hpp"
#include "vpux/utils/core/array_ref.hpp"
#include "vpux/utils/core/small_vector.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <llvm/Support/ThreadPool.h>
#include <mlir/IR/MLIRContext.h>

#include <mutex>

namespace vpux {
namespace Const {

class BackgroundConstantFolding {
public:
    BackgroundConstantFolding(mlir::MLIRContext* ctx, size_t maxConcurrentTasks, bool collectStatistics,
                              size_t memoryUsageLimit, double cacheCleanThreshold, Logger log = Logger::global());
    ~BackgroundConstantFolding();

    BackgroundConstantFolding(const BackgroundConstantFolding&) = delete;
    BackgroundConstantFolding(BackgroundConstantFolding&&) = delete;
    BackgroundConstantFolding& operator=(const BackgroundConstantFolding&) = delete;
    BackgroundConstantFolding& operator=(BackgroundConstantFolding&&) = delete;

    // Returns a ContentAttr containing a subset of the transformations from the original ContentAttr. The subset is
    // determined by the transformation given as a parameter to the function, so that all the transformations from the
    // list are included up to the one given as a parameter.
    //
    // For example:
    // Original ContentAttr: [Transformation1, Transformation2, Transformation3, Transformation4]
    // If Transformation3 is given as a parameter, a ContentAttr containing [Transformation1, Transformation2] is
    // returned Note: in case there are multiple instances of Transformation3 in the list, which have the same
    // underlying parameters, the rightmost one will be used
    static SmallVector<TransformAttrInterface> stripTransformationsFrom(
            ArrayRef<TransformAttrInterface> transformations, TransformAttrInterface transformation);
    // Returns the list of transformations of the ContentAttr starting from the transformation given as a parameter.
    //
    // For example:
    // ContentAttr: [Transformation1, Transformation2, Transformation3, Transformation4]
    // If Transformation3 is given as a parameter, a list containing [Transformation3, Transformation4] is returned
    // Note: in case there are multiple instances of NewTransformation in the list, which have the same underlying
    // parameters, the rightmost one will be used
    static SmallVector<TransformAttrInterface> getLastTransformationsFrom(
            ArrayRef<TransformAttrInterface> transformations, TransformAttrInterface transformation);

private:
    std::shared_future<void> initFoldingListener(llvm::ThreadPoolInterface& threadPool);
    void stopFoldingListener();
    void processFoldingRequest(FoldingRequest&& foldingRequest, ConstantFoldingCache& cache);

    bool _isEnabled = true;
    mlir::MLIRContext* _ctx;
    const size_t _maxConcurrentTasks;
    std::shared_future<void> _listenerThread;
    Logger _log;

    // The goal is to prevent overloading the thread-pool queue with background folding tasks, as the main compilation
    // process also utilizes the pool concurrently. To achieve this, a limit is imposed on the number of tasks submitted
    // to the pool from the background. Currently, the pool lacks a sophisticated mechanism to prioritize tasks from the
    // main compilation over those from the background.
    std::atomic<size_t> _activeTasks{0};
    std::mutex _mutex;
    std::condition_variable _cv;
};

}  // namespace Const
}  // namespace vpux

#endif
