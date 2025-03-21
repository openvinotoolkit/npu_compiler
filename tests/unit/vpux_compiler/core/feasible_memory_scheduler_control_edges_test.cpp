//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/utils/core/logger.hpp"

#include "vpux/compiler/core/async_deps_info.hpp"
#include "vpux/compiler/core/control_edge_generator.hpp"
#include "vpux/compiler/core/feasible_memory_scheduler_control_edges.hpp"

#include "common/utils.hpp"

#include <mlir/Dialect/Async/IR/Async.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Visitors.h>
#include <mlir/Parser/Parser.h>

#include <gtest/gtest.h>

using namespace vpux;

using MLIR_FeasibleMemorySchedulerControlEdges = MLIR_UnitBase;

TEST_F(MLIR_FeasibleMemorySchedulerControlEdges, OptimizeDepsIfOverlappingCycles) {
    mlir::MLIRContext ctx(registry);

    // Create an IR so that AsyncDepsInfo class can be initialized
    constexpr llvm::StringLiteral inputIR = R"(
        #NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

        !Type_DDR = memref<1x1x1x32xf16, #NCHW, @DDR>
        !Type_DDR_Half = memref<1x1x1x16xf16, #NCHW, @DDR>
        !Type_DDR_SubView = memref<1x1x1x16xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [32, 32, 32, 1]}, @DDR>

        module @test {
            func.func @main(%arg0: !Type_DDR) -> !Type_DDR {

                %buf0 = VPURT.DeclareBuffer <DDR> <0> -> !Type_DDR
                %buf1 = VPURT.DeclareBuffer <DDR> <64> -> !Type_DDR
                %buf2 = VPURT.DeclareBuffer <DDR> <128> -> !Type_DDR_Half
                %buf3 = VPURT.DeclareBuffer <DDR> <160> -> !Type_DDR_Half
                %buf4 = VPURT.DeclareBuffer <DDR> <64> -> !Type_DDR
                %buf5 = VPURT.DeclareBuffer <DDR> <192> -> !Type_DDR

                %t0, %r0 = async.execute -> !async.value<!Type_DDR> attributes {VPUIP.executor = @DMA_NN, VPUIP.executorIdx = [0], "async-deps-index" = 0 : i64, cycleBegin = 0 : i64, cycleCost = 1 : i64, cycleEnd = 1 : i64} {
                    %0 = VPUIP.Copy inputs(%arg0 : !Type_DDR) outputs(%buf0 : !Type_DDR) -> !Type_DDR
                    async.yield %0 : !Type_DDR
                }

                %t1, %r1 = async.execute (%r0 as %arg1: !async.value<!Type_DDR>) -> !async.value<!Type_DDR> attributes {VPUIP.executor = @DMA_NN, VPUIP.executorIdx = [0], "async-deps-index" = 1 : i64, cycleBegin = 1 : i64, cycleCost = 1 : i64, cycleEnd = 2 : i64} {
                    %0 = VPUIP.Copy inputs(%arg1 : !Type_DDR) outputs(%buf1 : !Type_DDR) -> !Type_DDR
                    async.yield %0 : !Type_DDR
                }

                %t2, %r2 = async.execute -> !async.value<!Type_DDR_SubView> attributes {VPUIP.executor = @DMA_NN, VPUIP.executorIdx = [0], "async-deps-index" = 2 : i64, cycleBegin = 2 : i64, cycleCost = 1 : i64, cycleEnd = 3 : i64} {
                    %view = VPUIP.SubView %buf4 [0, 0, 0, 0] [1, 1, 1, 16] : !Type_DDR to !Type_DDR_SubView
                    %0 = VPUIP.Copy inputs(%buf2 : !Type_DDR_Half) outputs(%view : !Type_DDR_SubView) -> !Type_DDR_SubView
                    async.yield %0 : !Type_DDR_SubView
                }

                %t3, %r3 = async.execute -> !async.value<!Type_DDR_SubView> attributes {VPUIP.executor = @DMA_NN, VPUIP.executorIdx = [1], "async-deps-index" = 3 : i64, cycleBegin = 2 : i64, cycleCost = 1 : i64, cycleEnd = 3 : i64} {
                    %view = VPUIP.SubView %buf4 [0, 0, 0, 16] [1, 1, 1, 16] : !Type_DDR to !Type_DDR_SubView
                    %0 = VPUIP.Copy inputs(%buf3 : !Type_DDR_Half) outputs(%view : !Type_DDR_SubView) -> !Type_DDR_SubView
                    async.yield %0 : !Type_DDR_SubView
                }

                %t4, %r4 = async.execute -> !async.value<!Type_DDR> attributes {VPUIP.executor = @DMA_NN, VPUIP.executorIdx = [0], "async-deps-index" = 4 : i64, cycleBegin = 3 : i64, cycleCost = 1 : i64, cycleEnd = 4 : i64} {
                    %0 = VPUIP.Copy inputs(%buf4 : !Type_DDR) outputs(%buf5 : !Type_DDR) -> !Type_DDR
                    async.yield %0 : !Type_DDR
                }

                %r = async.await %r4 : !async.value<!Type_DDR>
                return %r : !Type_DDR
            }
        }
    )";

    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    auto log = vpux::Logger::global();

    // Run helper utilities used by scheduler which allow it to gather data
    // represented in the IR
    AsyncDepsInfo depsInfo{func};

    // Created IR will have following ops consuming and producing given ranges:
    std::vector<ScheduledOpOneResource> scheduledOpsResources = {
            ScheduledOpOneResource(0, 0, 63, ScheduledOpOneResource::EResRelation::PRODUCER),
            ScheduledOpOneResource(1, 0, 63, ScheduledOpOneResource::EResRelation::CONSUMER),
            ScheduledOpOneResource(1, 64, 127, ScheduledOpOneResource::EResRelation::PRODUCER),
            ScheduledOpOneResource(2, 128, 159, ScheduledOpOneResource::EResRelation::CONSUMER),
            ScheduledOpOneResource(2, 64, 127, ScheduledOpOneResource::EResRelation::PRODUCER),
            ScheduledOpOneResource(3, 160, 191, ScheduledOpOneResource::EResRelation::CONSUMER),
            ScheduledOpOneResource(3, 64, 127, ScheduledOpOneResource::EResRelation::PRODUCER),
            ScheduledOpOneResource(4, 64, 127, ScheduledOpOneResource::EResRelation::CONSUMER),
            ScheduledOpOneResource(4, 192, 255, ScheduledOpOneResource::EResRelation::PRODUCER)};

    ControlEdgeSet controlEdges;
    ControlEdgeGenerator controlEdgeGenerator;
    // Generate control edges for overlapping memory regions
    controlEdgeGenerator.generateControlEdges(scheduledOpsResources.begin(), scheduledOpsResources.end(), controlEdges);

    // Above function will create edges not taking into account cycles but just order they are provided.
    // Initial edges from above function will be following:
    //  OP0 -> OP1
    //  OP1 -> OP2
    //  OP2 -> OP3
    //  OP3 -> OP4
    // This creates a linear execution:
    //  OP0 -> OP1 -> OP2 -> OP3 -> OP4
    SmallVector<ControlEdge> expectedInitialControlEdgesBeforeOptim = {{0, 1}, {1, 2}, {2, 3}, {3, 4}};

    ASSERT_EQ(controlEdges.size(), expectedInitialControlEdgesBeforeOptim.size());

    for (size_t i = 0; i < controlEdges.size(); i++) {
        ASSERT_EQ(controlEdges[i]._source, expectedInitialControlEdgesBeforeOptim[i]._source);
        ASSERT_EQ(controlEdges[i]._sink, expectedInitialControlEdgesBeforeOptim[i]._sink);
    }

    // OP2 and OP3 can run in parallel and edge OP2-> OP3 is not needed
    // Run method which updates deps info and performs optimization based on cycles
    updateControlEdgesInDepsInfo(depsInfo, controlEdges, log);

    // Below is final deps representation after optimization
    //  OP0 -> OP1 -> OP2 -> OP4
    //           |--> OP3 -->|
    SmallVector<std::set<size_t>> expectedDepsSetPerOp = {
            {},      // OP0: none
            {0},     // OP1: OP0->OP1
            {1},     // OP2: OP1->OP2
            {1},     // OP3: OP1->OP3
            {2, 3},  // OP4: OP2->OP4, OP3->OP4
    };

    size_t numOfOps = expectedDepsSetPerOp.size();

    for (size_t opIdx = 0; opIdx < numOfOps; opIdx++) {
        auto depsVec = depsInfo.getOpDeps(opIdx);

        // Check number of deps
        ASSERT_EQ(depsVec.size(), expectedDepsSetPerOp[opIdx].size());

        if (depsVec.empty()) {
            continue;
        }

        // Check content of deps
        std::set<size_t> depsSet(depsVec.begin(), depsVec.end());

        EXPECT_EQ(depsSet, expectedDepsSetPerOp[opIdx]);
    }
}
