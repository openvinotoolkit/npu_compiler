//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common/utils.hpp"
#include "vpux/compiler/core/async_deps_info.hpp"

#include "vpux/utils/core/array_ref.hpp"
#include "vpux/utils/core/string_ref.hpp"
#include "vpux/utils/logger/logger.hpp"

#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"

#include <mlir/Dialect/Async/IR/Async.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Parser/Parser.h>

#include <gtest/gtest.h>

// Run cmd: npuUnitTests --gtest_filter="MLIR_AsyncDepsInfo.*"

using namespace vpux;
using MLIR_AsyncDepsInfo = MLIR_UnitBase;

TEST_F(MLIR_AsyncDepsInfo, AddDependencySCSPWithCircle) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<vpux::VPUIP::VPUIPDialect>();

    constexpr StringLiteral inputIR = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
        module @test {
            func.func @main() {
                %token_0, %results_0 = async.execute -> !async.value<memref<1x32x256x256xf16, {order = #NHWC}>>
                    attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 0 : i64} {
                    %0 = memref.alloc() : memref<1x32x256x256xf16, {order = #NHWC}>
                    async.yield %0 : memref<1x32x256x256xf16, {order = #NHWC}>
                }
                %token_1, %results_1 = async.execute [%token_0]
                    (%results_0 as %arg0: !async.value<memref<1x32x256x256xf16, {order = #NHWC}>>)
                    -> !async.value<memref<1x32x256x256xf16, {order = #NHWC}>>
                    attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 1 : i64} {
                    async.yield %arg0 : memref<1x32x256x256xf16, {order = #NHWC}>
                }
                %token_2, %results_2 = async.execute [%token_1]
                    (%results_1 as %arg0: !async.value<memref<1x32x256x256xf16, {order = #NHWC}>>)
                    -> !async.value<memref<1x32x256x256xf16, {order = #NHWC}>>
                    attributes {VPUIP.executor = @DPU, "async-deps-index" = 2 : i64} {
                    async.yield %arg0 : memref<1x32x256x256xf16, {order = #NHWC}>
                }
                return
            }
        }
    )";

    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    vpux::AsyncDepsInfo info(func);
    info.buildConsMap();

    // Check initial consumer relationships
    auto op0Consumers = info.getConsumerOps(0);
    EXPECT_EQ(op0Consumers.size(), 1);
    EXPECT_EQ(op0Consumers[0], 1);

    auto op1Consumers = info.getConsumerOps(1);
    EXPECT_EQ(op1Consumers.size(), 1);
    EXPECT_EQ(op1Consumers[0], 2);

    auto op2Consumers = info.getConsumerOps(2);
    EXPECT_EQ(op2Consumers.size(), 0);

    // Prime the dependency cache before adding a new edge.
    auto op2Deps = info.getOpDeps(2);
    ASSERT_EQ(op2Deps.size(), 1);
    EXPECT_EQ(op2Deps[0], 1);

    // Calling addDependency twice should not duplicate entries
    info.addDependency(0, 2);
    info.addDependency(0, 2);
    auto op0ConsumersAfter = info.getConsumerOps(0);
    EXPECT_EQ(op0ConsumersAfter.size(), 2);
    EXPECT_TRUE(std::find(op0ConsumersAfter.begin(), op0ConsumersAfter.end(), 1) != op0ConsumersAfter.end());
    EXPECT_TRUE(std::find(op0ConsumersAfter.begin(), op0ConsumersAfter.end(), 2) != op0ConsumersAfter.end());

    auto op2DepsAfter = info.getOpDeps(2);
    ASSERT_EQ(op2DepsAfter.size(), 2);
    EXPECT_EQ(op2DepsAfter[0], 0);
    EXPECT_EQ(op2DepsAfter[1], 1);

    // Introduce a cycle and verifyAcyclic should throw
    info.addDependency(2, 0);
    EXPECT_THROW(info.verifyAcyclic(), std::exception);
}

TEST_F(MLIR_AsyncDepsInfo, AddDependencyMCMP) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<vpux::VPUIP::VPUIPDialect>();

    constexpr StringLiteral inputIR = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
        module @test {
            func.func @main() {
                %token_0, %results_0 = async.execute -> !async.value<memref<1x32x256x256xf16, {order = #NHWC}>>
                    attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 0 : i64} {
                    %0 = memref.alloc() : memref<1x32x256x256xf16, {order = #NHWC}>
                    async.yield %0 : memref<1x32x256x256xf16, {order = #NHWC}>
                }
                %token_1, %results_1 = async.execute -> !async.value<memref<1x32x256x256xf16, {order = #NHWC}>>
                    attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 1 : i64} {
                    %0 = memref.alloc() : memref<1x32x256x256xf16, {order = #NHWC}>
                    async.yield %0 : memref<1x32x256x256xf16, {order = #NHWC}>
                }
                %token_2, %results_2 = async.execute [%token_0]
                    (%results_0 as %arg0: !async.value<memref<1x32x256x256xf16, {order = #NHWC}>>)
                    -> !async.value<memref<1x32x256x256xf16, {order = #NHWC}>>
                    attributes {VPUIP.executor = @DPU, "async-deps-index" = 2 : i64} {
                    async.yield %arg0 : memref<1x32x256x256xf16, {order = #NHWC}>
                }
                %token_3, %results_3 = async.execute [%token_0]
                    (%results_0 as %arg0: !async.value<memref<1x32x256x256xf16, {order = #NHWC}>>)
                    -> !async.value<memref<1x32x256x256xf16, {order = #NHWC}>>
                    attributes {VPUIP.executor = @DPU, "async-deps-index" = 3 : i64} {
                    async.yield %arg0 : memref<1x32x256x256xf16, {order = #NHWC}>
                }
                %token_4, %results_4 = async.execute [%token_0, %token_1]
                    (%results_0 as %arg0: !async.value<memref<1x32x256x256xf16, {order = #NHWC}>>,
                     %results_1 as %arg1: !async.value<memref<1x32x256x256xf16, {order = #NHWC}>>)
                    -> !async.value<memref<1x32x256x256xf16, {order = #NHWC}>>
                    attributes {VPUIP.executor = @DPU, "async-deps-index" = 4 : i64} {
                    async.yield %arg0 : memref<1x32x256x256xf16, {order = #NHWC}>
                }
                return
            }
        }
    )";

    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    vpux::AsyncDepsInfo info(func);
    info.buildConsMap();

    // One producer feeding multiple consumers
    // op0 feeds op2, op3, and op4
    auto op0Consumers = info.getConsumerOps(0);
    ASSERT_EQ(op0Consumers.size(), 3);
    EXPECT_EQ(op0Consumers[0], 2);
    EXPECT_EQ(op0Consumers[1], 3);
    EXPECT_EQ(op0Consumers[2], 4);

    // Multiple producers feeding one consumer
    // op4 consumes from both op0 and op1
    auto op4Deps = info.getOpDeps(4);
    ASSERT_EQ(op4Deps.size(), 2);
    EXPECT_EQ(op4Deps[0], 0);
    EXPECT_EQ(op4Deps[1], 1);

    // op1 has one consumer (op4)
    auto op1Consumers = info.getConsumerOps(1);
    EXPECT_EQ(op1Consumers.size(), 1);
    EXPECT_EQ(op1Consumers[0], 4);

    // Add new dependency from op1 to op2 and verify consumer map is updated
    info.addDependency(1, 2);
    info.addDependency(1, 2);

    auto op1ConsumersAfter = info.getConsumerOps(1);
    ASSERT_EQ(op1ConsumersAfter.size(), 2);
    EXPECT_EQ(op1ConsumersAfter[0], 2);
    EXPECT_EQ(op1ConsumersAfter[1], 4);

    auto op2DepsAfter = info.getOpDeps(2);
    ASSERT_EQ(op2DepsAfter.size(), 2);
    EXPECT_EQ(op2DepsAfter[0], 0);
    EXPECT_EQ(op2DepsAfter[1], 1);

    // Verify op0 consumers remain unchanged after adding op1->op2 dependency
    auto op0ConsumersAfter = info.getConsumerOps(0);
    EXPECT_EQ(op0ConsumersAfter.size(), 3);

    // Rebuilding token operands consumes the cached, sorted dependency vectors.
    // Exercise both the cache-miss and cache-hit paths and preserve deterministic order.
    const auto verifyOp2Dependencies = [&]() {
        auto op2Dependencies = info.getExecuteOpAtIndex(2).getDependencies();
        ASSERT_EQ(op2Dependencies.size(), 2);
        EXPECT_EQ(op2Dependencies[0], info.getExecuteOpAtIndex(0).getToken());
        EXPECT_EQ(op2Dependencies[1], info.getExecuteOpAtIndex(1).getToken());
    };
    info.updateTokenDependencies();
    verifyOp2Dependencies();
    info.updateTokenDependencies();
    verifyOp2Dependencies();
}

TEST_F(MLIR_AsyncDepsInfo, OptimizeDepsMap) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<vpux::VPUIP::VPUIPDialect>();

    constexpr StringLiteral inputIR = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
        module @test {
            func.func @main() {
                %token_0, %results_0 = async.execute -> !async.value<memref<1x32x256x256xf16, {order = #NHWC}>>
                    attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 0 : i64} {
                    %0 = memref.alloc() : memref<1x32x256x256xf16, {order = #NHWC}>
                    async.yield %0 : memref<1x32x256x256xf16, {order = #NHWC}>
                }
                %token_1, %results_1 = async.execute [%token_0]
                    (%results_0 as %arg0: !async.value<memref<1x32x256x256xf16, {order = #NHWC}>>)
                    -> !async.value<memref<1x32x256x256xf16, {order = #NHWC}>>
                    attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 1 : i64} {
                    async.yield %arg0 : memref<1x32x256x256xf16, {order = #NHWC}>
                }
                %token_2, %results_2 = async.execute [%token_0, %token_1]
                    (%results_0 as %arg0: !async.value<memref<1x32x256x256xf16, {order = #NHWC}>>,
                     %results_1 as %arg1: !async.value<memref<1x32x256x256xf16, {order = #NHWC}>>)
                    -> !async.value<memref<1x32x256x256xf16, {order = #NHWC}>>
                    attributes {VPUIP.executor = @DPU, "async-deps-index" = 2 : i64} {
                    async.yield %arg0 : memref<1x32x256x256xf16, {order = #NHWC}>
                }
                return
            }
        }
    )";

    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    vpux::AsyncDepsInfo info(func);
    info.buildConsMap();

    // Before optimization: op2 depends on both op0 and op1
    auto op2DepsBefore = info.getOpDeps(2);
    EXPECT_EQ(op2DepsBefore.size(), 2);

    // Prime the consumer cache before optimizeDepsMap rebuilds the consumer map.
    auto op0ConsumersBefore = info.getConsumerOps(0);
    EXPECT_EQ(op0ConsumersBefore.size(), 2);

    // Optimize: since op1 depends on op0, and op2 depends on both,
    // the dependency from op2 to op0 is redundant
    info.optimizeDepsMap();

    // After optimization: op2 should only depend on op1
    auto op2DepsAfter = info.getOpDeps(2);
    EXPECT_EQ(op2DepsAfter.size(), 1);
    EXPECT_EQ(op2DepsAfter[0], 1);

    auto op0ConsumersAfter = info.getConsumerOps(0);
    EXPECT_EQ(op0ConsumersAfter.size(), 1);
    EXPECT_EQ(op0ConsumersAfter[0], 1);
}

TEST_F(MLIR_AsyncDepsInfo, CalculateInOutDegree) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<vpux::VPUIP::VPUIPDialect>();

    constexpr StringLiteral inputIR = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
        module @test {
            func.func @main() {
                %token_0, %results_0 = async.execute -> !async.value<memref<1x32x256x256xf16, {order = #NHWC}>>
                    attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 0 : i64} {
                    %0 = memref.alloc() : memref<1x32x256x256xf16, {order = #NHWC}>
                    async.yield %0 : memref<1x32x256x256xf16, {order = #NHWC}>
                }
                %token_1, %results_1 = async.execute [%token_0]
                    (%results_0 as %arg0: !async.value<memref<1x32x256x256xf16, {order = #NHWC}>>)
                    -> !async.value<memref<1x32x256x256xf16, {order = #NHWC}>>
                    attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 1 : i64} {
                    async.yield %arg0 : memref<1x32x256x256xf16, {order = #NHWC}>
                }
                %token_2, %results_2 = async.execute [%token_1]
                    (%results_1 as %arg0: !async.value<memref<1x32x256x256xf16, {order = #NHWC}>>)
                    -> !async.value<memref<1x32x256x256xf16, {order = #NHWC}>>
                    attributes {VPUIP.executor = @DPU, "async-deps-index" = 2 : i64} {
                    async.yield %arg0 : memref<1x32x256x256xf16, {order = #NHWC}>
                }
                return
            }
        }
    )";

    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    vpux::AsyncDepsInfo info(func);

    // Calculate in-degree
    auto inDegree = info.calculateOpInDegreeTable();
    EXPECT_EQ(inDegree[0], 0);  // op0 has no dependencies
    EXPECT_EQ(inDegree[1], 1);  // op1 depends on op0
    EXPECT_EQ(inDegree[2], 1);  // op2 depends on op1

    // Build consumer map and calculate out-degree
    info.buildConsMap();
    auto outDegree = info.calculateOpOutDegreeTable();
    EXPECT_EQ(outDegree[0], 1);  // op0 has 1 consumer (op1)
    EXPECT_EQ(outDegree[1], 1);  // op1 has 1 consumer (op2)
    EXPECT_EQ(outDegree[2], 0);  // op2 has no consumers
}

TEST_F(MLIR_AsyncDepsInfo, InsertNewExecOp) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<vpux::VPUIP::VPUIPDialect>();

    constexpr StringLiteral inputIR = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
        module @test {
            func.func @main() {
                %token_0, %results_0 = async.execute -> !async.value<memref<1x32x256x256xf16, {order = #NHWC}>>
                    attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 0 : i64} {
                    %0 = memref.alloc() : memref<1x32x256x256xf16, {order = #NHWC}>
                    async.yield %0 : memref<1x32x256x256xf16, {order = #NHWC}>
                }
                return
            }
        }
    )";

    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    vpux::AsyncDepsInfo info(func);
    info.buildConsMap();

    // Prime both caches before extending the maps.
    EXPECT_TRUE(info.getOpDeps(0).empty());
    EXPECT_TRUE(info.getConsumerOps(0).empty());

    // Initial count should be 1
    EXPECT_EQ(info.getExecOpCount(), 1);

    // Reserve a batch of slots to cover the production insertion pattern.
    info.preAllocateForNewOps(2);

    // Create new async.execute operations
    mlir::OpBuilder builder(&ctx);

    auto memrefType = mlir::MemRefType::get({1, 32, 256, 256}, builder.getF16Type());
    auto asyncValueType = mlir::async::ValueType::get(memrefType);
    auto tokenType = builder.getType<mlir::async::TokenType>();

    auto createExecOp = [&](mlir::ValueRange dependencies) {
        builder.setInsertionPointToEnd(&func.getBody().front());
        auto execOp = builder.create<mlir::async::ExecuteOp>(
                builder.getUnknownLoc(), mlir::TypeRange{tokenType, asyncValueType}, dependencies, mlir::ValueRange{});

        auto& bodyBlock = execOp.getBodyRegion().emplaceBlock();
        builder.setInsertionPointToStart(&bodyBlock);
        auto allocOp = builder.create<mlir::memref::AllocOp>(builder.getUnknownLoc(), memrefType);
        builder.create<mlir::async::YieldOp>(builder.getUnknownLoc(), mlir::ValueRange{allocOp});
        return execOp;
    };

    auto newExecOp = createExecOp(mlir::ValueRange{info.getExecuteOpAtIndex(0).getToken()});

    // Insert new exec op to deps map
    size_t newIdx = info.insertNewExecOpToDepsMap(newExecOp);

    // Verify the new operation was added
    EXPECT_EQ(info.getExecOpCount(), 2);
    EXPECT_EQ(newIdx, 1);

    auto retrievedOp = info.getExecuteOpAtIndex(newIdx);
    EXPECT_EQ(retrievedOp, newExecOp);

    auto newOpDeps = info.getOpDeps(newIdx);
    ASSERT_EQ(newOpDeps.size(), 1);
    EXPECT_EQ(newOpDeps[0], 0);

    auto op0Consumers = info.getConsumerOps(0);
    ASSERT_EQ(op0Consumers.size(), 1);
    EXPECT_EQ(op0Consumers[0], newIdx);

    // Query between insertions so the second insertion must invalidate warm caches.
    EXPECT_TRUE(info.getConsumerOps(newIdx).empty());

    SmallVector<mlir::Value> secondOpDependencies{newExecOp.getToken(), info.getExecuteOpAtIndex(0).getToken()};
    auto secondExecOp = createExecOp(secondOpDependencies);
    auto secondIdx = info.insertNewExecOpToDepsMap(secondExecOp);

    EXPECT_EQ(info.getExecOpCount(), 3);
    EXPECT_EQ(secondIdx, 2);

    auto secondOpDeps = info.getOpDeps(secondIdx);
    ASSERT_EQ(secondOpDeps.size(), 2);
    EXPECT_EQ(secondOpDeps[0], 0);
    EXPECT_EQ(secondOpDeps[1], 1);

    op0Consumers = info.getConsumerOps(0);
    ASSERT_EQ(op0Consumers.size(), 2);
    EXPECT_EQ(op0Consumers[0], 1);
    EXPECT_EQ(op0Consumers[1], 2);

    auto op1Consumers = info.getConsumerOps(1);
    ASSERT_EQ(op1Consumers.size(), 1);
    EXPECT_EQ(op1Consumers[0], 2);

    // Repeat the final reads to exercise the cache-hit path.
    EXPECT_EQ(info.getOpDeps(secondIdx), secondOpDeps);
    EXPECT_EQ(info.getConsumerOps(0), op0Consumers);
    EXPECT_EQ(info.getConsumerOps(1), op1Consumers);
}
