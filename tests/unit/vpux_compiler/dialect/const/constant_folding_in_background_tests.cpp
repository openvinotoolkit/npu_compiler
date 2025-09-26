//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#ifdef BACKGROUND_FOLDING_ENABLED

#include <chrono>
#include "vpux/compiler/dialect/const/dialect.hpp"
#include "vpux/compiler/dialect/const/utils/constant_folding_cache.hpp"
#include "vpux/compiler/dialect/const/utils/constant_folding_in_background.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"

#include "common/utils.hpp"
#include "vpux/compiler/init.hpp"

#include <mlir/IR/AsmState.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Types.h>

#include <gtest/gtest.h>
#include <thread>

using namespace vpux;
using namespace std::chrono_literals;

namespace {

using ContentAttrFunction = llvm::function_ref<std::pair<Const::ContentAttr, SmallVector<float>>(mlir::MLIRContext*)>;

std::pair<Const::ContentAttr, SmallVector<float>> createContentAttrSameTransformation(mlir::MLIRContext* ctx) {
    const size_t numElements = 100;
    const float baseValue = 1.0f;
    const auto baseType = mlir::RankedTensorType::get({numElements}, mlir::Float32Type::get(ctx));
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(baseValue));
    Const::ContentSetup contentAttrSetup(baseType);

    const size_t numTransformations = 5;
    for (size_t i = 0; i < numTransformations; ++i) {
        contentAttrSetup = contentAttrSetup.add(1.0);
    }

    const float expectedValue = baseValue + numTransformations;
    SmallVector<float> expectedFoldedResults(numElements, expectedValue);

    return std::make_pair(Const::ContentAttr::get(baseAttr, std::move(contentAttrSetup)), expectedFoldedResults);
}

std::pair<Const::ContentAttr, SmallVector<float>> createContentAttrMixedTransformations(mlir::MLIRContext* ctx) {
    const size_t numElements = 100;
    const float baseValue = 0.0f;
    const auto baseType = mlir::RankedTensorType::get({numElements}, mlir::Float32Type::get(ctx));
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(baseValue));
    auto contentAttr = Const::ContentAttr::get(
            baseAttr,
            Const::ContentSetup(baseType).padWithZero({10}, {10}).add(1.0).rescale(3.0).subview({0}, {numElements}));

    const float expectedValue = 3.0f;
    SmallVector<float> expectedFoldedResults(numElements, expectedValue);

    return std::make_pair(contentAttr, expectedFoldedResults);
}

};  // namespace

class ConstantFoldingInBackground : public testing::TestWithParam<size_t> {};

void compile(mlir::MLIRContext* ctx, size_t numFoldingThreads, std::chrono::milliseconds sleepDuration,
             ContentAttrFunction contentAttrFn) {
    const auto collectStatistics = true;
    const auto memoryUsageLimit = 3 * 1024;
    const auto cacheCleanThreshold = 0.8;
    Logger log = Logger::global();

    auto foldingListener = std::make_unique<Const::BackgroundConstantFolding>(
            ctx, numFoldingThreads, collectStatistics, memoryUsageLimit, cacheCleanThreshold, log);

    const auto [contentAttr, expectedValues] = contentAttrFn(ctx);

    std::this_thread::sleep_for(sleepDuration);

    auto result = contentAttr.fold();
    const auto resultValues = result.getValues<float>();
    EXPECT_EQ(resultValues.size(), expectedValues.size());
    for (auto values : zip(resultValues, expectedValues)) {
        const auto resultValue = std::get<0>(values);
        const auto expectedValue = std::get<1>(values);
        EXPECT_EQ(resultValue, expectedValue);
    }
}

TEST_P(ConstantFoldingInBackground, CompilationFlow) {
    auto registry = vpux::createDialectRegistry();

    const auto numFoldingThreads = GetParam();

    mlir::MLIRContext ctx(registry);
    const size_t maxThreads = ctx.getThreadPool().getMaxConcurrency();
    llvm::StdThreadPool newThreadPool(llvm::optimal_concurrency(std::max(2 * numFoldingThreads + 1, maxThreads)));

    ctx.disableMultithreading();
    ctx.setThreadPool(newThreadPool);

    ctx.loadDialect<Const::ConstDialect>();

    const auto sleepDuration = 100ms;
    compile(&ctx, numFoldingThreads, sleepDuration, createContentAttrSameTransformation);
}

TEST_P(ConstantFoldingInBackground, MultipleCompilations) {
    auto registry = vpux::createDialectRegistry();

    const auto numFoldingThreads = GetParam();

    mlir::MLIRContext ctx1(registry);
    const size_t maxThreadsCtx1 = ctx1.getThreadPool().getMaxConcurrency();
    llvm::StdThreadPool newThreadPoolCtx1(
            llvm::optimal_concurrency(std::max(2 * numFoldingThreads + 1, maxThreadsCtx1)));

    ctx1.disableMultithreading();
    ctx1.setThreadPool(newThreadPoolCtx1);

    mlir::MLIRContext ctx2(registry);
    const size_t maxThreadsCtx2 = ctx2.getThreadPool().getMaxConcurrency();
    llvm::StdThreadPool newThreadPoolCtx2(
            llvm::optimal_concurrency(std::max(2 * numFoldingThreads + 1, maxThreadsCtx2)));

    ctx2.disableMultithreading();
    ctx2.setThreadPool(newThreadPoolCtx2);

    SmallVector<mlir::MLIRContext*> contexts = {&ctx1, &ctx2};
    for (auto ctx : contexts) {
        ctx->loadDialect<Const::ConstDialect>();
    }

    const SmallVector<std::chrono::milliseconds> sleepDurations{100ms, 300ms};
    const SmallVector<ContentAttrFunction> contentAttrFns = {createContentAttrSameTransformation,
                                                             createContentAttrMixedTransformations};

    std::vector<std::thread> threads;
    for (size_t i = 0; i < contexts.size(); ++i) {
        const auto ctx = contexts[i];
        const auto sleepDuration = sleepDurations[i];
        const auto contentAttrFn = contentAttrFns[i];
        std::thread t([ctx, numFoldingThreads, sleepDuration, contentAttrFn]() {
            compile(ctx, numFoldingThreads, sleepDuration, contentAttrFn);
        });
        threads.push_back(std::move(t));
    }

    for (auto& t : threads) {
        t.join();
    }
}

std::vector<size_t> numThreads = {1, 2, 3, 4, 5};

INSTANTIATE_TEST_SUITE_P(MLIRThreading, ConstantFoldingInBackground, testing::ValuesIn(numThreads));

using ConstantFoldingInBackgroundUnit = MLIR_UnitBase;

TEST_F(ConstantFoldingInBackgroundUnit, EquivalenceRequest) {
    auto registry = vpux::createDialectRegistry();

    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<Const::ConstDialect>();

    const auto numFoldingThreads = 2;
    const auto collectStatistics = true;
    const auto memoryUsageLimit = 3 * 1024;
    const auto cacheCleanThreshold = 0.8;
    Logger log = Logger::global();

    auto foldingListener = Const::BackgroundConstantFolding(&ctx, numFoldingThreads, collectStatistics,
                                                            memoryUsageLimit, cacheCleanThreshold, log);

    const size_t numElements = 100;
    const float baseValue = 1.0f;
    const auto baseType = mlir::RankedTensorType::get({numElements}, mlir::Float32Type::get(&ctx));
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(baseValue));
    auto contentAttr = Const::ContentAttr::get(baseAttr);

    // `subview` sends folding requests to the background threads
    auto contentAttr1 = contentAttr.transform()
                                .subview(ShapeRef({0}), ShapeRef({50}))
                                .subview(ShapeRef({10}), ShapeRef({30}))
                                .get();

    // Manually creating a ContentAttr will not sent a folding request, so an equivalence request is manually sent
    auto optimizedSubviewTransformation = Const::SubViewAttr::get(getIntArrayAttr(&ctx, SmallVector<int64_t>({0})),
                                                                  getIntArrayAttr(&ctx, SmallVector<int64_t>({30})));
    auto contentAttr2 = Const::ContentAttr::get(baseAttr, {optimizedSubviewTransformation});
    auto& cacheManager = Const::ConstantFoldingCacheManager::getInstance();
    ASSERT_TRUE(cacheManager.contains(&ctx));
    auto& cache = cacheManager.get(&ctx);
    auto equivalenceRequest = Const::EquivalenceRequest{contentAttr1, contentAttr2};
    cache.enqueueRequest(Const::FoldingRequest{std::move(equivalenceRequest), /*newTransformation=*/nullptr});

    std::this_thread::sleep_for(100ms);

    // contentAttr1 may or may not still be in cache, depending on whether it was added before contentAttr2,
    // so only contentAttr2 is checked as it must be in the cache
    EXPECT_TRUE(cache.hasContent(contentAttr2));
}

#endif
