//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common/utils.hpp"

#include "vpux/compiler/dialect/VPU/utils/manual_strategy_utils.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <memory>
#include <thread>
#include <vector>

using namespace vpux;

using LoopIndexAllocatorTest = MLIR_UnitBase;

namespace {

constexpr size_t numThreads = 16;
constexpr size_t indicesPerThread = 128;
constexpr size_t numFunctions = 2;
constexpr size_t indicesPerFunction = numThreads / numFunctions * indicesPerThread;

void expectContiguousIndices(std::vector<int64_t>& indices) {
    ASSERT_EQ(indices.size(), indicesPerFunction);
    std::sort(indices.begin(), indices.end());
    EXPECT_EQ(std::unique(indices.begin(), indices.end()), indices.end());
}

}  // namespace

TEST_F(LoopIndexAllocatorTest, AllocatesIndicesMultiThread) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<mlir::func::FuncDialect>();
    auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));
    mlir::OpBuilder builder(module.getBody(), module.getBody()->begin());
    const auto functionType = builder.getFunctionType({}, {});
    const auto firstFunc = mlir::func::FuncOp::create(builder, module.getLoc(), "first", functionType);
    const auto secondFunc = mlir::func::FuncOp::create(builder, module.getLoc(), "second", functionType);
    const std::array allocators = {std::make_unique<LoopIndexAllocator>(firstFunc),
                                   std::make_unique<LoopIndexAllocator>(secondFunc)};
    std::array<std::vector<int64_t>, numFunctions> tilingIndices;
    std::array<std::vector<int64_t>, numFunctions> vfIndices;
    for (size_t functionIndex = 0; functionIndex < numFunctions; ++functionIndex) {
        tilingIndices[functionIndex].resize(indicesPerFunction);
        vfIndices[functionIndex].resize(indicesPerFunction);
    }

    std::vector<std::thread> threads;
    threads.reserve(numThreads);
    for (size_t threadIndex = 0; threadIndex < numThreads; ++threadIndex) {
        threads.emplace_back([&, threadIndex] {
            const auto functionIndex = threadIndex % numFunctions;
            const auto firstIndex = threadIndex / numFunctions * indicesPerThread;
            for (size_t index = 0; index < indicesPerThread; ++index) {
                tilingIndices[functionIndex][firstIndex + index] = allocators[functionIndex]->getNextTilingLoopIndex();
                vfIndices[functionIndex][firstIndex + index] = allocators[functionIndex]->getNextVFLoopIndex();
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    for (size_t functionIndex = 0; functionIndex < numFunctions; ++functionIndex) {
        expectContiguousIndices(tilingIndices[functionIndex]);
        expectContiguousIndices(vfIndices[functionIndex]);
    }
}

TEST_F(LoopIndexAllocatorTest, InitializesFromCurrentFunctionIndices) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<mlir::func::FuncDialect>();
    auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));
    mlir::OpBuilder builder(module.getBody(), module.getBody()->begin());
    const auto functionType = builder.getFunctionType({}, {});
    auto func = mlir::func::FuncOp::create(builder, module.getLoc(), "first", functionType);
    func->setAttr(TILING_LOOP_INDEX_ATTR_NAME, I64Attr::get(&ctx, 7));
    func->setAttr(VF_LOOP_INDEX_ATTR_NAME, I64Attr::get(&ctx, 3));

    LoopIndexAllocator allocator(func);
    EXPECT_EQ(allocator.getNextTilingLoopIndex(), 8);
    EXPECT_EQ(allocator.getNextVFLoopIndex(), 4);
}
