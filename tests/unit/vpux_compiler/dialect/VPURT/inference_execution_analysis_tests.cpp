//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/cycle_cost_info.hpp"
#include "vpux/compiler/dialect/VPU/utils/cost_model/factories/cost_model_config.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPURT/IR/task.hpp"
#include "vpux/compiler/dialect/VPURT/interfaces/inference_execution_simulator.hpp"
#include "vpux/compiler/init.hpp"

#include "common/utils.hpp"

#include <mlir/IR/MLIRContext.h>
#include <mlir/Parser/Parser.h>

#include <gtest/gtest.h>

using namespace vpux;

using MLIR_InferenceExecutionAnalysis = MLIR_UnitBase;

void verifyCorrectCycles(SmallVector<vpux::VPURT::TaskConfig, 1>& tasks,
                         SmallVector<std::pair<size_t, size_t>>& cycleBeginEndPairs) {
    for (size_t i = 0; i < tasks.size(); i++) {
        auto cycleBegin = tasks[i].cycleStart;
        auto cycleEnd = tasks[i].cycleStart + tasks[i].cycleCost;

        EXPECT_EQ(cycleBegin, cycleBeginEndPairs[i].first);
        EXPECT_EQ(cycleEnd, cycleBeginEndPairs[i].second);
    }
}

TEST_F(MLIR_InferenceExecutionAnalysis, CheckCycleUpdateWith1ActShaveEngineOn1Cluster) {
    mlir::MLIRContext ctx(registry);

    // Below is an IR example with ActShave tasks
    // Its intention is to verify that if there is only 1 ActShave engine on a cluster
    // then those tasks will execute sequentially
    //
    // Below is a overview of execution
    // ACT C0_0: [----------------]
    // ACT C0_1:                   [----------------]

    constexpr StringLiteral inputIR = R"(
        module @test attributes {config.arch = #config.arch_kind<NPU40XX>, config.compilationMode = #config.compilation_mode<DefaultHW>} {
            config.PipelineOptions @Options {
                config.Option @config.UseDedicatedFifoPerShaveEngine : false
            }
            config.Resources 6 of @NCE at 1.700000e+03 MHz {
                config.MemoryResource 1327104 bytes of @CMX_NN_FragmentationAware
                config.MemoryResource 1474560 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
                config.ExecutorResource 1 of @SHAVE_ACT
                config.ExecutorResource 1 of @DPU
            }
            config.Resources 1 of @global {
                config.ExecutorResource 1 of @M2I
                config.ExecutorResource 1 of @DMA_NN
                config.MemoryResource 524288000 bytes of @DDR {config.bandwidth = 64 : i64, config.derateFactor = 6.000000e-01 : f64}
            }
            VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]

            module @VPU.SW  {
                func.func private @builtin_TanhOp(memref<*xf16>, memref<*xf16>, i64) attributes {VPU.kernel_code = "activation_tanh.cpp", VPU.kernel_entry = "activation_tanh"}
                func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
            }

            func.func @main(%arg0: memref<1x16x24x24xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @DDR>, %arg1: memref<1x16x24x24xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @DDR>) -> memref<1x16x24x24xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @DDR> {

                %netin = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x16x24x24xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @DDR>

                %buf_cmx0_1_Part0 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x12x24xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>
                %buf_cmx0_1_Part1 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x12x24xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>

                %buf_cmx0_2_Part0 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x12x24xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>
                %buf_cmx0_2_Part1 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x12x24xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>

                VPURT.Task {
                    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_TanhOp inputs(%buf_cmx0_1_Part0 as %arg2: memref<1x16x12x24xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>) outputs(%buf_cmx0_2_Part0 as %arg3: memref<1x16x12x24xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>) on tile 0 -> memref<1x16x12x24xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>{
                    VPUIP.SW.Kernel.run {attrs = [false, true, 1.0013580322265625E-5]}(%arg2, %arg3) : memref<1x16x12x24xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>, memref<1x16x12x24xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>
                    }
                }

                VPURT.Task {
                    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_TanhOp inputs(%buf_cmx0_1_Part1 as %arg2: memref<1x16x12x24xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>) outputs(%buf_cmx0_2_Part1 as %arg3: memref<1x16x12x24xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>) on tile 0 -> memref<1x16x12x24xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>{
                    VPUIP.SW.Kernel.run {attrs = [false, true, 1.0013580322265625E-5]}(%arg2, %arg3) : memref<1x16x12x24xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>, memref<1x16x12x24xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>
                    }
                }

                return %arg1 : memref<1x16x24x24xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @DDR>
            }
        }
    )";

    Logger log("inference-simulator-test", LogLevel::Info);

    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto funcOp = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(funcOp != nullptr);

    // set cost model factory
    VPU::CostModelConfig::setFactory(config::ArchKind::NPU40XX);

    CycleCostInfo cycleCostInfo(funcOp);
    VPURT::InferenceExecutionSimulator infSim(log, funcOp, cycleCostInfo);

    infSim.runSim();

    SmallVector<size_t> tasksCost;
    for (auto taskOp : funcOp.getOps<VPURT::TaskOp>()) {
        tasksCost.push_back(cycleCostInfo.getCycleCost(taskOp));
    }

    ASSERT_EQ(tasksCost.size(), 2);

    auto tasksAndCycles = infSim.getTaskCycleConfig(VPU::ExecutorKind::SHAVE_ACT);

    // Tasks execute sequentially so start time of second task is equal to cost of first task
    SmallVector<std::pair<size_t, size_t>> actShaveTasksCycleBeginEndPairs = {
            {0, tasksCost[0]},
            {tasksCost[0], tasksCost[0] + tasksCost[1]}};

    ASSERT_EQ(tasksAndCycles.size(), actShaveTasksCycleBeginEndPairs.size());

    verifyCorrectCycles(tasksAndCycles, actShaveTasksCycleBeginEndPairs);
}

TEST_F(MLIR_InferenceExecutionAnalysis, CheckCycleUpdateWith2ActShaveEngineOn1Cluster) {
    mlir::MLIRContext ctx(registry);

    // Below is an IR example with ActShave tasks
    // Its intention is to verify that if there are ActShave engine on a cluster
    // then those tasks will be distributed and executed simultaneously on independent queues
    //
    // Below is a overview of execution
    // ACT C0_0: [----------------]
    // ACT C0_1: [----------------]

    constexpr StringLiteral inputIR = R"(
        module @test attributes {config.arch = #config.arch_kind<NPU40XX>, config.compilationMode = #config.compilation_mode<DefaultHW>} {
            config.PipelineOptions @Options {
                config.Option @config.UseDedicatedFifoPerShaveEngine : false
            }
            config.Resources 6 of @NCE at 1.700000e+03 MHz {
                config.MemoryResource 1327104 bytes of @CMX_NN_FragmentationAware
                config.MemoryResource 1474560 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
                config.ExecutorResource 2 of @SHAVE_ACT
                config.ExecutorResource 1 of @DPU
            }
            config.Resources 1 of @global {
                config.ExecutorResource 1 of @M2I
                config.ExecutorResource 1 of @DMA_NN
                config.MemoryResource 524288000 bytes of @DDR {config.bandwidth = 64 : i64, config.derateFactor = 6.000000e-01 : f64}
            }

            VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]

            module @VPU.SW  {
                func.func private @builtin_TanhOp(memref<*xf16>, memref<*xf16>, i64) attributes {VPU.kernel_code = "activation_tanh.cpp", VPU.kernel_entry = "activation_tanh"}
                func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
            }

            func.func @main(%arg0: memref<1x16x24x24xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @DDR>, %arg1: memref<1x16x24x24xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @DDR>) -> memref<1x16x24x24xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @DDR> {

                %netin = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x16x24x24xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @DDR>

                %buf_cmx0_1_Part0 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x12x24xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>
                %buf_cmx0_1_Part1 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x12x24xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>

                %buf_cmx0_2_Part0 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x12x24xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>
                %buf_cmx0_2_Part1 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x12x24xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>

                VPURT.Task {
                    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_TanhOp inputs(%buf_cmx0_1_Part0 as %arg2: memref<1x16x12x24xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>) outputs(%buf_cmx0_2_Part0 as %arg3: memref<1x16x12x24xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>) on tile 0 -> memref<1x16x12x24xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>{
                    VPUIP.SW.Kernel.run {attrs = [false, true, 1.0013580322265625E-5]}(%arg2, %arg3) : memref<1x16x12x24xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>, memref<1x16x12x24xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>
                    }
                }

                VPURT.Task {
                    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_TanhOp inputs(%buf_cmx0_1_Part1 as %arg2: memref<1x16x12x24xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>) outputs(%buf_cmx0_2_Part1 as %arg3: memref<1x16x12x24xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>) on tile 0 -> memref<1x16x12x24xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>{
                    VPUIP.SW.Kernel.run {attrs = [false, true, 1.0013580322265625E-5]}(%arg2, %arg3) : memref<1x16x12x24xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>, memref<1x16x12x24xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>
                    }
                }

                return %arg1 : memref<1x16x24x24xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @DDR>
            }
        }
    )";

    Logger log("inference-simulator-test", LogLevel::Info);

    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto funcOp = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(funcOp != nullptr);

    // set cost model factory
    VPU::CostModelConfig::setFactory(config::ArchKind::NPU40XX);

    CycleCostInfo cycleCostInfo(funcOp);
    VPURT::InferenceExecutionSimulator infSim(log, funcOp, cycleCostInfo);

    infSim.runSim();

    SmallVector<size_t> tasksCost;
    for (auto taskOp : funcOp.getOps<VPURT::TaskOp>()) {
        tasksCost.push_back(cycleCostInfo.getCycleCost(taskOp));
    }

    ASSERT_EQ(tasksCost.size(), 2);

    auto tasksAndCycles = infSim.getTaskCycleConfig(VPU::ExecutorKind::SHAVE_ACT);

    // Tasks execute in parallel so start time is 0 for both and their end time equal to their cost
    SmallVector<std::pair<size_t, size_t>> actShaveTasksCycleBeginEndPairs = {{0, tasksCost[0]}, {0, tasksCost[1]}};

    ASSERT_EQ(tasksAndCycles.size(), actShaveTasksCycleBeginEndPairs.size());

    verifyCorrectCycles(tasksAndCycles, actShaveTasksCycleBeginEndPairs);
}

TEST_F(MLIR_InferenceExecutionAnalysis, CheckCycleUpdateOnMultiQueueIR) {
    mlir::MLIRContext ctx(registry);

    // Below is an IR example with DMAs, NCE and ActShave task
    // Its intention is to only verify that inference execution simulator will correctly
    // interpret it and calculate cycleBegin/End of each task as in
    // below one their are not correct, only task cycleCost is valid (cycleEnd - cycleBegin)
    //
    // Below is a overview of execution
    // DMA P0:   [--][--][--][--]
    // DMA P1:   [--][--][--][--]
    // NCE C0:                   [------]
    // NCE C1:                   [------]
    // ACT C0_0:                         [----------------]
    // ACT C0_1:                         [----------------]
    // ACT C1_0:                         [----------------]
    // ACT C1_1:                         [----------------]
    constexpr StringLiteral inputIR = R"(
        module @test attributes {config.arch = #config.arch_kind<NPU37XX>, config.compilationMode = #config.compilation_mode<DefaultHW>} {
            config.PipelineOptions @Options {
                config.Option @config.UseDedicatedFifoPerShaveEngine : false
            }
            config.Resources 6 of @NCE at 1.700000e+03 MHz {
                config.MemoryResource 1327104 bytes of @CMX_NN_FragmentationAware
                config.MemoryResource 1474560 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
                config.ExecutorResource 2 of @SHAVE_ACT
                config.ExecutorResource 1 of @DPU
            }
            config.Resources 1 of @global {
                config.ExecutorResource 1 of @DMA_NN
                config.MemoryResource 524288000 bytes of @DDR {config.bandwidth = 64 : i64, config.derateFactor = 6.000000e-01 : f64}
            }
            VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]

            module @VPU.SW  {
                func.func private @builtin_TanhOp(memref<*xf16>, memref<*xf16>, i64) attributes {VPU.kernel_code = "activation_tanh.cpp", VPU.kernel_entry = "activation_tanh"}
                func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
            }

            func.func @main(%arg0: memref<1x16x24x24xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @DDR>, %arg1: memref<1x16x24x24xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @DDR>) -> memref<1x16x24x24xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @DDR> {
                %bar0 = VPURT.ConfigureBarrier<0> -> !VPURT.Barrier
                %bar1 = VPURT.ConfigureBarrier<1> -> !VPURT.Barrier
                %bar2 = VPURT.ConfigureBarrier<2> -> !VPURT.Barrier
                %bar3 = VPURT.ConfigureBarrier<3> -> !VPURT.Barrier

                %cst_WT = const.Declare memref<16x1x1x4xsi32> = dense<2> : tensor<16x1x1x4xsi32>

                %netin = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x16x24x24xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @DDR>
                %buf_ddr_0 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x16x24x24xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @DDR>
                %buf_ddr_1 = VPURT.DeclareBuffer <DDR> <32768> -> memref<1x16x24x24xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @DDR>
                %buf_cmx0_0 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x24x24xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>
                %buf_cmx1_0 = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x16x24x24xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 1]>
                %buf_cmx0_WT = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<16x1x1x4xsi32, [@CMX_NN, 0]>
                %buf_cmx1_WT = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<16x1x1x4xsi32, [@CMX_NN, 1]>

                %buf_cmx0_1 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x24x24xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>
                %buf_cmx1_1 = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x16x24x24xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 1]>

                %buf_cmx0_1_Part0 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x12x24xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>
                %buf_cmx0_1_Part1 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x12x24xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>

                %buf_cmx1_1_Part0 = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x16x12x24xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 1]>
                %buf_cmx1_1_Part1 = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x16x12x24xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 1]>

                %buf_cmx0_2_Part0 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x12x24xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>
                %buf_cmx0_2_Part1 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x12x24xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>

                %buf_cmx1_2_Part0 = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x16x12x24xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 1]>
                %buf_cmx1_2_Part1 = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x16x12x24xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 1]>

                VPURT.Task {
                    %0 = VPUIP.NNDMA {port = 0 : i64} inputs(%netin : memref<1x16x24x24xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @DDR>) outputs(%buf_ddr_0 : memref<1x16x24x24xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @DDR>) -> memref<1x16x24x24xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @DDR>
                }

                VPURT.Task {
                    %0 = VPUIP.NNDMA {port = 1 : i64} inputs(%netin : memref<1x16x24x24xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @DDR>) outputs(%buf_ddr_1 : memref<1x16x24x24xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @DDR>) -> memref<1x16x24x24xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @DDR>
                }

                VPURT.Task {
                    %0 = VPUIP.NNDMA {port = 0 : i64} inputs(%cst_WT : memref<16x1x1x4xsi32>) outputs(%buf_cmx0_WT : memref<16x1x1x4xsi32, [@CMX_NN, 0]>) -> memref<16x1x1x4xsi32, [@CMX_NN, 0]>
                }

                VPURT.Task {
                    %0 = VPUIP.NNDMA {port = 1 : i64} inputs(%cst_WT : memref<16x1x1x4xsi32>) outputs(%buf_cmx1_WT : memref<16x1x1x4xsi32, [@CMX_NN, 1]>) -> memref<16x1x1x4xsi32, [@CMX_NN, 1]>
                }

                VPURT.Task updates(%bar0 : !VPURT.Barrier) {
                    %0 = VPUIP.NNDMA {port = 0 : i64} inputs(%buf_ddr_0 : memref<1x16x24x24xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @DDR>) outputs(%buf_cmx0_0 : memref<1x16x24x24xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>) -> memref<1x16x24x24xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>
                }

                VPURT.Task updates(%bar1 : !VPURT.Barrier) {
                    %0 = VPUIP.NNDMA {port = 1 : i64} inputs(%buf_ddr_1 : memref<1x16x24x24xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @DDR>) outputs(%buf_cmx1_0 : memref<1x16x24x24xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 1]>) -> memref<1x16x24x24xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 1]>
                }

                VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar2 : !VPURT.Barrier) {
                    %0 = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<MAXPOOL>}
                    input(%buf_cmx0_0 : memref<1x16x24x24xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>)
                    weight_table(%buf_cmx0_WT : memref<16x1x1x4xsi32, [@CMX_NN, 0]>)
                    parent_input(%buf_cmx0_0 : memref<1x16x24x24xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>)
                    parent_output(%buf_cmx0_1 : memref<1x16x24x24xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>)
                    outputs(%buf_cmx0_1 : memref<1x16x24x24xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>) -> memref<1x16x24x24xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]> variants : {
                    DPUTask {cluster_id = 0 : i64, outEnd = [23, 11, 15], mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0]}
                    } PPE : {
                    }
                }

                VPURT.Task waits(%bar1 : !VPURT.Barrier) updates(%bar3 : !VPURT.Barrier) {
                    %0 = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<MAXPOOL>}
                    input(%buf_cmx1_0 : memref<1x16x24x24xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 1]>)
                    weight_table(%buf_cmx1_WT : memref<16x1x1x4xsi32, [@CMX_NN, 1]>)
                    parent_input(%buf_cmx1_0 : memref<1x16x24x24xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 1]>)
                    parent_output(%buf_cmx1_1 : memref<1x16x24x24xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 1]>)
                    outputs(%buf_cmx1_1 : memref<1x16x24x24xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 1]>) -> memref<1x16x24x24xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 1]> variants : {
                    DPUTask {cluster_id = 1 : i64, outEnd = [23, 23, 15], mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 12, 0]}
                    } PPE : {
                    }
                }

                VPURT.Task waits(%bar2 : !VPURT.Barrier) {
                    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_TanhOp inputs(%buf_cmx0_1_Part0 as %arg2: memref<1x16x12x24xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>) outputs(%buf_cmx0_2_Part0 as %arg3: memref<1x16x12x24xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>) on tile 0 -> memref<1x16x12x24xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>{
                    VPUIP.SW.Kernel.run {attrs = [false, true, 1.0013580322265625E-5]}(%arg2, %arg3) : memref<1x16x12x24xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>, memref<1x16x12x24xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>
                    }
                }

                VPURT.Task waits(%bar2 : !VPURT.Barrier) {
                    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_TanhOp inputs(%buf_cmx0_1_Part1 as %arg2: memref<1x16x12x24xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>) outputs(%buf_cmx0_2_Part1 as %arg3: memref<1x16x12x24xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>) on tile 0 -> memref<1x16x12x24xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>{
                    VPUIP.SW.Kernel.run {attrs = [false, true, 1.0013580322265625E-5]}(%arg2, %arg3) : memref<1x16x12x24xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>, memref<1x16x12x24xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>
                    }
                }

                VPURT.Task waits(%bar3 : !VPURT.Barrier) {
                    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_TanhOp inputs(%buf_cmx1_1_Part0 as %arg2: memref<1x16x12x24xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 1]>) outputs(%buf_cmx1_2_Part0 as %arg3: memref<1x16x12x24xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 1]>) on tile 1 -> memref<1x16x12x24xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 1]>{
                    VPUIP.SW.Kernel.run {attrs = [false, true, 1.0013580322265625E-5]}(%arg2, %arg3) : memref<1x16x12x24xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 1]>, memref<1x16x12x24xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 1]>
                    }
                }

                VPURT.Task waits(%bar3 : !VPURT.Barrier) {
                    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_TanhOp inputs(%buf_cmx1_1_Part1 as %arg2: memref<1x16x12x24xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 1]>) outputs(%buf_cmx1_2_Part1 as %arg3: memref<1x16x12x24xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 1]>) on tile 1 -> memref<1x16x12x24xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 1]>{
                    VPUIP.SW.Kernel.run {attrs = [false, true, 1.0013580322265625E-5]}(%arg2, %arg3) : memref<1x16x12x24xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 1]>, memref<1x16x12x24xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 1]>
                    }
                }

                return %arg1 : memref<1x16x24x24xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @DDR>
            }
        }
    )";

    Logger log("inference-simulator-test", LogLevel::Info);

    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto funcOp = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(funcOp != nullptr);

    // set cost model factory
    VPU::CostModelConfig::setFactory(config::ArchKind::NPU37XX);

    CycleCostInfo cycleCostInfo(funcOp);
    VPURT::InferenceExecutionSimulator infSim(log, funcOp, cycleCostInfo);

    infSim.runSim();

    SmallVector<size_t> tasksCostDmaP0;
    SmallVector<size_t> tasksCostDmaP1;
    SmallVector<size_t> tasksCostNce;
    SmallVector<size_t> tasksCostSwKernel;

    for (auto taskOp : funcOp.getOps<VPURT::TaskOp>()) {
        auto cost = cycleCostInfo.getCycleCost(taskOp);

        auto* op = taskOp.getInnerTaskOp();
        if (auto dmaOp = mlir::dyn_cast<VPUIP::DMATypeOpInterface>(op)) {
            if (dmaOp.getPortVal() == 0) {
                tasksCostDmaP0.push_back(cost);
            } else {
                tasksCostDmaP1.push_back(cost);
            }
        } else if (mlir::isa<VPUIP::SwKernelOp>(op)) {
            tasksCostSwKernel.push_back(cost);
        } else {
            tasksCostNce.push_back(cost);
        }
    }

    ASSERT_EQ(tasksCostDmaP0.size(), 3);
    ASSERT_EQ(tasksCostDmaP1.size(), 3);
    ASSERT_EQ(tasksCostNce.size(), 2);
    ASSERT_EQ(tasksCostSwKernel.size(), 4);

    // Expected cycles need to be aligned with below execution
    // DMA P0:   [--][--][--][--]
    // DMA P1:   [--][--][--][--]
    // NCE C0:                   [------]
    // NCE C1:                   [------]
    // ACT C0_0:                         [----------------]
    // ACT C0_1:                         [----------------]
    // ACT C1_0:                         [----------------]
    // ACT C1_1:                         [----------------]
    // Prepare expected cycleBegin/End based on operations cost and their
    // placement in the schedule

    auto updateDmaExpectedCycleBeginEndPairs = [](SmallVector<std::pair<size_t, size_t>>& tasksCycleBeginEndPairs,
                                                  SmallVector<size_t>& tasksCost) {
        for (size_t i = 0; i < tasksCost.size(); i++) {
            size_t prevTaskCycleEnd = 0;
            if (i > 0) {
                prevTaskCycleEnd = tasksCycleBeginEndPairs.back().second;
            }
            tasksCycleBeginEndPairs.push_back(std::make_pair(prevTaskCycleEnd, prevTaskCycleEnd + tasksCost[i]));
        }
    };

    SmallVector<std::pair<size_t, size_t>> dmaTasksCycleBeginEndPairs;

    updateDmaExpectedCycleBeginEndPairs(dmaTasksCycleBeginEndPairs, tasksCostDmaP0);
    updateDmaExpectedCycleBeginEndPairs(dmaTasksCycleBeginEndPairs, tasksCostDmaP1);

    auto nceCycleBegin = dmaTasksCycleBeginEndPairs.back().second;

    SmallVector<std::pair<size_t, size_t>> nceTasksCycleBeginEndPairs = {
            {nceCycleBegin, nceCycleBegin + tasksCostNce[0]},
            {nceCycleBegin, nceCycleBegin + tasksCostNce[1]}};

    auto actShaveCycleBegin = nceTasksCycleBeginEndPairs.back().second;
    SmallVector<std::pair<size_t, size_t>> actShaveTasksCycleBeginEndPairs = {
            {actShaveCycleBegin, actShaveCycleBegin + tasksCostSwKernel[0]},
            {actShaveCycleBegin, actShaveCycleBegin + tasksCostSwKernel[1]},
            {actShaveCycleBegin, actShaveCycleBegin + tasksCostSwKernel[2]},
            {actShaveCycleBegin, actShaveCycleBegin + tasksCostSwKernel[3]}};

    auto dmaTasks = infSim.getTaskCycleConfig(VPU::ExecutorKind::DMA_NN);
    auto nceTasks = infSim.getTaskCycleConfig(VPU::ExecutorKind::DPU);
    auto actShaveTasks = infSim.getTaskCycleConfig(VPU::ExecutorKind::SHAVE_ACT);

    ASSERT_EQ(dmaTasks.size(), dmaTasksCycleBeginEndPairs.size());
    ASSERT_EQ(nceTasks.size(), nceTasksCycleBeginEndPairs.size());
    ASSERT_EQ(actShaveTasks.size(), actShaveTasksCycleBeginEndPairs.size());

    verifyCorrectCycles(dmaTasks, dmaTasksCycleBeginEndPairs);
    verifyCorrectCycles(nceTasks, nceTasksCycleBeginEndPairs);
    verifyCorrectCycles(actShaveTasks, actShaveTasksCycleBeginEndPairs);
}

TEST_F(MLIR_InferenceExecutionAnalysis, CheckBarrierConfigClass) {
    // Create a barrier config and increment producer counter to 1
    VPURT::BarrierConfigInfo barrierConf;
    barrierConf.addProducer();

    // Producer count is still 1, barrier is not released and cannot get release cycle
    EXPECT_FALSE(barrierConf.isReleased());
    EXPECT_ANY_THROW(barrierConf.getReleaseCycle());

    barrierConf.decrementAtCycle(10);

    // After decrementing producer count is 0 and barrier is released
    EXPECT_EQ(barrierConf.getReleaseCycle(), 10);
    EXPECT_TRUE(barrierConf.isReleased());
}

TEST_F(MLIR_InferenceExecutionAnalysis, CheckSubTaskStartTimesQueueCount1) {
    size_t startTime = 5;
    SmallVector<size_t> subTasksCost = {10, 10, 10, 10};
    size_t queueCount = 1;

    auto subTasksStartTime = VPURT::getSubTasksStartTime(subTasksCost, startTime, queueCount);

    ASSERT_EQ(subTasksStartTime.size(), subTasksCost.size());

    EXPECT_EQ(subTasksStartTime[0], 5);
    EXPECT_EQ(subTasksStartTime[1], 15);
    EXPECT_EQ(subTasksStartTime[2], 25);
    EXPECT_EQ(subTasksStartTime[3], 35);
}

TEST_F(MLIR_InferenceExecutionAnalysis, CheckSubTaskStartTimesQueueCount2) {
    size_t startTime = 5;
    SmallVector<size_t> subTasksCost = {10, 10, 10, 10};
    size_t queueCount = 2;

    auto subTasksStartTime = VPURT::getSubTasksStartTime(subTasksCost, startTime, queueCount);

    ASSERT_EQ(subTasksStartTime.size(), subTasksCost.size());

    EXPECT_EQ(subTasksStartTime[0], 5);
    EXPECT_EQ(subTasksStartTime[1], 5);
    EXPECT_EQ(subTasksStartTime[2], 15);
    EXPECT_EQ(subTasksStartTime[3], 15);
}
