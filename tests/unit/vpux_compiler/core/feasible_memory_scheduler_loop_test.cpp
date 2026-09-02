//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/utils/logger/logger.hpp"

#include "vpux/compiler/core/aliases_info.hpp"
#include "vpux/compiler/core/async_deps_info.hpp"
#include "vpux/compiler/core/feasible_memory_scheduler.hpp"
#include "vpux/compiler/core/linear_scan_handler.hpp"
#include "vpux/compiler/dialect/VPU/utils/cost_model/cost_model.hpp"
#include "vpux/compiler/dialect/VPU/utils/scheduling/loop_schedule_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/async_dialect_utils.hpp"
#include "vpux/compiler/utils/hw_settings.hpp"

#include "../utils/scheduler_test_utils.hpp"
#include "common/utils.hpp"
#include "feasible_memory_scheduler_test_utils.hpp"

#include <mlir/Dialect/Async/IR/Async.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Visitors.h>
#include <mlir/Parser/Parser.h>

#include <gtest/gtest.h>

#include <string>

using namespace vpux;

// Run cmd: npuUnitTests --gtest_filter="MLIR_FeasibleMemorySchedulerLoop.*"

using MLIR_FeasibleMemorySchedulerLoop = MLIR_UnitBase;

// Verifies that input buffers for loop operations are correctly reused across iterations in an tiling loop.
// Example from a convolution tiled 5 over output channels:
// Op 3 (Tile 0): Input[0]=360448(589824B), Input[1]=0(32768B), Input[2]=950272(1024B)
// 				 + Added
// Op 5 (Tile 1): Input[0]=360448(589824B), Input[1]=950272(1024B), Input[2]=180224(32768B)
// 				 + Added
// Op 7 (Tile 2): Input[0]=360448(589824B), Input[1]=950272(1024B), Input[2]=0(32768B)
//               ✓ Checked - match tile 0
// Op 9 (Tile 3): Input[0]=360448(589824B), Input[1]=950272(1024B), Input[2]=180224(32768B)
//               ✓ Checked - matches tile 1
// Op 12 (Tile 4): Input[0]=950272(1024B), Input[1]=360448(589824B), Input[2]=0(32768B)
//               ✓ Checked - match tile 0 again
bool FeasibleMemorySchedulerTest::verifyLoopInputAddress() const {
    auto log = Logger::global().nest("loop-allocator-test");
    llvm::DenseMap<size_t, llvm::DenseSet<vpux::AddressType>> inputAddresses;  // Map: bufferSize - {set of addresses}
    // Collect the buffer addresses from the first 2 loop operations
    // which should be the same for all iterations in tiling loop
    const auto opNum = _scheduler._scheduledOps.size();
    if (opNum < 2) {
        log.error("Not enough scheduled operations to verify loop input addresses");
        return false;
    }
    auto isComputeLoopOp = [&](const FeasibleMemoryScheduler::ScheduledOpInfo& op) -> bool {
        if (!op.isLoopOp()) {
            return false;
        }

        if (op.numOfInputResources() == 0 || op.numOfOutputResources() == 0) {
            return false;
        }
        return true;
    };

    // Collect all buffer addresses from loop operations
    size_t checkedLoopOpCount = 0;
    for (const auto& scheduledOp : _scheduler._scheduledOps) {
        if (!isComputeLoopOp(scheduledOp)) {
            continue;
        }
        log.trace("Op {0} ({1}), scheduledOp.numOfInputResources() {2}", scheduledOp.op_,
                  scheduledOp.opTypeName().str(), scheduledOp.numOfInputResources());
        if (checkedLoopOpCount < 2) {
            // Store the first 2 loop ops' addresses
            for (size_t i = 0; i < scheduledOp.numOfInputResources(); i++) {
                if (!scheduledOp.isActiveInputResource(i)) {
                    continue;
                }
                vpux::AddressType beginAddr = scheduledOp.beginInputResource(i);
                vpux::AddressType bufferSize = scheduledOp.endInputResource(i) - beginAddr;
                log.trace("[Add] Input[{0}]: begin={1}, size={2}", i, beginAddr, bufferSize);
                inputAddresses[bufferSize].insert(beginAddr);
            }
        } else {
            // Check if the addresses are reused from the first 2 loop ops
            for (size_t i = 0; i < scheduledOp.numOfInputResources(); i++) {
                if (!scheduledOp.isActiveInputResource(i)) {
                    continue;
                }
                vpux::AddressType beginAddr = scheduledOp.beginInputResource(i);
                vpux::AddressType bufferSize = scheduledOp.endInputResource(i) - beginAddr;
                log.trace("[Check] Input[{0}]: begin={1}, size={2}", i, beginAddr, bufferSize);
                if (inputAddresses.count(bufferSize)) {
                    if (llvm::find(inputAddresses[bufferSize], beginAddr) == inputAddresses[bufferSize].end()) {
                        log.error("Unexpected buffer address {0} for size {1}, expected one of {2}", beginAddr,
                                  bufferSize, llvm::to_vector(inputAddresses[bufferSize]));
                        return false;
                    }
                }
            }
        }
        ++checkedLoopOpCount;
    }
    return true;
}

// IR constants for the non-distributed 2x5 tiling-CH test (1x320x48x48 conv, 5 tiles over C, 2 tiles over H).
// Buffer sizes: activation 1x256x24x48xf16 (589824B), weights 64x256x1x1xf16 (32768B),
// weight table 64x1x1x4xsi32 (1024B), output 1x64x24x48xf16 (147456B).
namespace TilingCH2x5IR {

constexpr llvm::StringLiteral kPart1 = R"(

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @model_name attributes {config.compilationMode = #config.compilation_mode<DefaultHW>, config.platform = #config.platform<NPU5010>} {
func.func @main(%arg0: memref<1x48x48x256xf16, @DDR>, %arg1: memref<1x320x48x48xf16, @DDR>) -> memref<1x320x48x48xf16, @DDR> {
  %cst = const.Declare memref<64x1x1x4xsi32> = dense<1> : tensor<64x1x1x4xsi32>
  %cst_0 = const.Declare memref<64x256x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<320x256x1x1xf32>, [#const.SubView<[256, 0, 0, 0], [64, 256, 1, 1]>, #const.CastElemType<f16>, #const.Reorder<#NHWC>]
  %cst_1 = const.Declare memref<64x256x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<320x256x1x1xf32>, [#const.SubView<[192, 0, 0, 0], [64, 256, 1, 1]>, #const.CastElemType<f16>, #const.Reorder<#NHWC>]
  %cst_2 = const.Declare memref<64x256x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<320x256x1x1xf32>, [#const.SubView<[128, 0, 0, 0], [64, 256, 1, 1]>, #const.CastElemType<f16>, #const.Reorder<#NHWC>]
  %cst_3 = const.Declare memref<64x256x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<320x256x1x1xf32>, [#const.SubView<[64, 0, 0, 0], [64, 256, 1, 1]>, #const.CastElemType<f16>, #const.Reorder<#NHWC>]
  %cst_4 = const.Declare memref<64x256x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<320x256x1x1xf32>, [#const.SubView<[0, 0, 0, 0], [64, 256, 1, 1]>, #const.CastElemType<f16>, #const.Reorder<#NHWC>]
  %alloc = memref.alloc() : memref<1x256x24x48xf16, {order = #NHWC}, [@CMX_NN, 0]>
  %alloc_5 = memref.alloc() : memref<64x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
  %alloc_6 = memref.alloc() : memref<64x1x1x4xsi32, [@CMX_NN, 0]>
  %alloc_7 = memref.alloc() : memref<1x64x24x48xf16, [@CMX_NN, 0]>
  %alloc_8 = memref.alloc() : memref<64x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
  %alloc_9 = memref.alloc() : memref<1x64x24x48xf16, [@CMX_NN, 0]>
  %alloc_10 = memref.alloc() : memref<64x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
  %alloc_11 = memref.alloc() : memref<1x64x24x48xf16, [@CMX_NN, 0]>
  %alloc_12 = memref.alloc() : memref<64x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
  %alloc_13 = memref.alloc() : memref<1x64x24x48xf16, [@CMX_NN, 0]>
  %alloc_14 = memref.alloc() : memref<64x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
  %alloc_15 = memref.alloc() : memref<1x64x24x48xf16, [@CMX_NN, 0]>
  %alloc_16 = memref.alloc() : memref<1x256x24x48xf16, {order = #NHWC}, [@CMX_NN, 0]>
  %alloc_17 = memref.alloc() : memref<64x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
  %alloc_18 = memref.alloc() : memref<1x64x24x48xf16, [@CMX_NN, 0]>
  %alloc_19 = memref.alloc() : memref<64x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
  %alloc_20 = memref.alloc() : memref<1x64x24x48xf16, [@CMX_NN, 0]>
  %alloc_21 = memref.alloc() : memref<64x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
  %alloc_22 = memref.alloc() : memref<1x64x24x48xf16, [@CMX_NN, 0]>
  %alloc_23 = memref.alloc() : memref<64x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
  %alloc_24 = memref.alloc() : memref<1x64x24x48xf16, [@CMX_NN, 0]>
  %alloc_25 = memref.alloc() : memref<64x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
  %alloc_26 = memref.alloc() : memref<1x64x24x48xf16, [@CMX_NN, 0]>
  %token, %bodyResults = async.execute -> !async.value<memref<1x256x24x48xf16, {order = #NHWC}, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 0 : i64, cycleCost = 16718 : i64} {
    %11 = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NCHW} inputs(%arg0 : memref<1x48x48x256xf16, @DDR>) -> memref<1x256x48x48xf16, {order = #NHWC}, @DDR>
    %12 = VPUIP.SubView %11 [0, 0, 0, 0] [1, 256, 24, 48] : memref<1x256x48x48xf16, {order = #NHWC}, @DDR> to memref<1x256x24x48xf16, {order = #NHWC, strides = [589824, 1, 12288, 256]}, @DDR>
    %13 = VPUIP.NNDMA inputs(%12 : memref<1x256x24x48xf16, {order = #NHWC, strides = [589824, 1, 12288, 256]}, @DDR>) outputs(%alloc : memref<1x256x24x48xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x256x24x48xf16, {order = #NHWC}, [@CMX_NN, 0]>
    async.yield %13 : memref<1x256x24x48xf16, {order = #NHWC}, [@CMX_NN, 0]>
  }
  %token_27, %bodyResults_28 = async.execute -> !async.value<memref<64x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 1 : i64, cycleCost = 1482 : i64} {
    %11 = VPUIP.NNDMA inputs(%cst_4 : memref<64x256x1x1xf16, {order = #NHWC}>) outputs(%alloc_5 : memref<64x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<64x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    async.yield %11 : memref<64x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
  }
  %token_29, %bodyResults_30 = async.execute -> !async.value<memref<64x1x1x4xsi32, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 2 : i64, cycleCost = 600 : i64} {
    %11 = VPUIP.NNDMA inputs(%cst : memref<64x1x1x4xsi32>) outputs(%alloc_6 : memref<64x1x1x4xsi32, [@CMX_NN, 0]>) -> memref<64x1x1x4xsi32, [@CMX_NN, 0]>
    async.yield %11 : memref<64x1x1x4xsi32, [@CMX_NN, 0]>
  }
  %token_31, %bodyResults_32 = async.execute [%token, %token_27, %token_29] (%bodyResults as %arg2: !async.value<memref<1x256x24x48xf16, {order = #NHWC}, [@CMX_NN, 0]>>, %bodyResults_28 as %arg3: !async.value<memref<64x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>>, %bodyResults_30 as %arg4: !async.value<memref<64x1x1x4xsi32, [@CMX_NN, 0]>>) -> !async.value<memref<1x64x24x48xf16, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DPU, "async-deps-index" = 3 : i64, cycleCost = 25361 : i64} {
    %11 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 25361 : i64, tiling_loop_index = 0 : i64, resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>}  <{is_zero_offset_weights_table, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<CONV>}> input(%arg2 : memref<1x256x24x48xf16, {order = #NHWC}, [@CMX_NN, 0]>) weights(%arg3 : memref<64x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) weight_table(%arg4 : memref<64x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%arg2 : memref<1x256x24x48xf16, {order = #NHWC}, [@CMX_NN, 0]>) parent_output(%alloc_7 : memref<1x64x24x48xf16, [@CMX_NN, 0]>) outputs(%alloc_7 : memref<1x64x24x48xf16, [@CMX_NN, 0]>) -> memref<1x64x24x48xf16, [@CMX_NN, 0]> variants : {
      DPUTask {inEnd = [47, 23, 255], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [47, 23, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>}
    }
    async.yield %11 : memref<1x64x24x48xf16, [@CMX_NN, 0]>
  })";

// Continuation of 2x5 tiling-CH IR: tiles 1-4 for H-tile 0, then tiles 0-4 for H-tile 1
constexpr llvm::StringLiteral kPart2 = R"(
  %token_33, %bodyResults_34 = async.execute -> !async.value<memref<64x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 4 : i64, cycleCost = 1482 : i64} {
    %11 = VPUIP.NNDMA inputs(%cst_3 : memref<64x256x1x1xf16, {order = #NHWC}>) outputs(%alloc_8 : memref<64x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<64x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    async.yield %11 : memref<64x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
  }
  %token_35, %bodyResults_36 = async.execute [%token, %token_29, %token_33] (%bodyResults as %arg2: !async.value<memref<1x256x24x48xf16, {order = #NHWC}, [@CMX_NN, 0]>>, %bodyResults_30 as %arg3: !async.value<memref<64x1x1x4xsi32, [@CMX_NN, 0]>>, %bodyResults_34 as %arg4: !async.value<memref<64x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>>) -> !async.value<memref<1x64x24x48xf16, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DPU, "async-deps-index" = 5 : i64, cycleCost = 25361 : i64} {
    %11 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 25361 : i64, tiling_loop_index = 0 : i64, resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{is_zero_offset_weights_table, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<CONV>}> input(%arg2 : memref<1x256x24x48xf16, {order = #NHWC}, [@CMX_NN, 0]>) weights(%arg4 : memref<64x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) weight_table(%arg3 : memref<64x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%arg2 : memref<1x256x24x48xf16, {order = #NHWC}, [@CMX_NN, 0]>) parent_output(%alloc_9 : memref<1x64x24x48xf16, [@CMX_NN, 0]>) outputs(%alloc_9 : memref<1x64x24x48xf16, [@CMX_NN, 0]>) -> memref<1x64x24x48xf16, [@CMX_NN, 0]> variants : {
      DPUTask {inEnd = [47, 23, 255], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [47, 23, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>}
    }
    async.yield %11 : memref<1x64x24x48xf16, [@CMX_NN, 0]>
  }
  %token_37, %bodyResults_38 = async.execute -> !async.value<memref<64x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 6 : i64, cycleCost = 1482 : i64} {
    %11 = VPUIP.NNDMA inputs(%cst_2 : memref<64x256x1x1xf16, {order = #NHWC}>) outputs(%alloc_10 : memref<64x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<64x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    async.yield %11 : memref<64x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
  }
  %token_39, %bodyResults_40 = async.execute [%token, %token_29, %token_37] (%bodyResults as %arg2: !async.value<memref<1x256x24x48xf16, {order = #NHWC}, [@CMX_NN, 0]>>, %bodyResults_30 as %arg3: !async.value<memref<64x1x1x4xsi32, [@CMX_NN, 0]>>, %bodyResults_38 as %arg4: !async.value<memref<64x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>>) -> !async.value<memref<1x64x24x48xf16, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DPU, "async-deps-index" = 7 : i64, cycleCost = 25361 : i64} {
    %11 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 25361 : i64, tiling_loop_index = 0 : i64, resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{is_zero_offset_weights_table, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<CONV>}> input(%arg2 : memref<1x256x24x48xf16, {order = #NHWC}, [@CMX_NN, 0]>) weights(%arg4 : memref<64x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) weight_table(%arg3 : memref<64x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%arg2 : memref<1x256x24x48xf16, {order = #NHWC}, [@CMX_NN, 0]>) parent_output(%alloc_11 : memref<1x64x24x48xf16, [@CMX_NN, 0]>) outputs(%alloc_11 : memref<1x64x24x48xf16, [@CMX_NN, 0]>) -> memref<1x64x24x48xf16, [@CMX_NN, 0]> variants : {
      DPUTask {inEnd = [47, 23, 255], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [47, 23, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>}
    }
    async.yield %11 : memref<1x64x24x48xf16, [@CMX_NN, 0]>
  }
  %token_41, %bodyResults_42 = async.execute -> !async.value<memref<64x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 8 : i64, cycleCost = 1482 : i64} {
    %11 = VPUIP.NNDMA inputs(%cst_1 : memref<64x256x1x1xf16, {order = #NHWC}>) outputs(%alloc_12 : memref<64x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<64x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    async.yield %11 : memref<64x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
  }
  %token_43, %bodyResults_44 = async.execute [%token, %token_29, %token_41] (%bodyResults as %arg2: !async.value<memref<1x256x24x48xf16, {order = #NHWC}, [@CMX_NN, 0]>>, %bodyResults_30 as %arg3: !async.value<memref<64x1x1x4xsi32, [@CMX_NN, 0]>>, %bodyResults_42 as %arg4: !async.value<memref<64x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>>) -> !async.value<memref<1x64x24x48xf16, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DPU, "async-deps-index" = 9 : i64, cycleCost = 25361 : i64} {
    %11 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 25361 : i64, tiling_loop_index = 0 : i64, resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{is_zero_offset_weights_table, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<CONV>}> input(%arg2 : memref<1x256x24x48xf16, {order = #NHWC}, [@CMX_NN, 0]>) weights(%arg4 : memref<64x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) weight_table(%arg3 : memref<64x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%arg2 : memref<1x256x24x48xf16, {order = #NHWC}, [@CMX_NN, 0]>) parent_output(%alloc_13 : memref<1x64x24x48xf16, [@CMX_NN, 0]>) outputs(%alloc_13 : memref<1x64x24x48xf16, [@CMX_NN, 0]>) -> memref<1x64x24x48xf16, [@CMX_NN, 0]> variants : {
      DPUTask {inEnd = [47, 23, 255], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [47, 23, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>}
    }
    async.yield %11 : memref<1x64x24x48xf16, [@CMX_NN, 0]>
  }
  %token_45, %bodyResults_46 = async.execute -> !async.value<memref<64x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 10 : i64, cycleCost = 1482 : i64} {
    %11 = VPUIP.NNDMA inputs(%cst_0 : memref<64x256x1x1xf16, {order = #NHWC}>) outputs(%alloc_14 : memref<64x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<64x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    async.yield %11 : memref<64x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
  }
  %token_47, %bodyResults_48 = async.execute [%token, %token_29, %token_45] (%bodyResults as %arg2: !async.value<memref<1x256x24x48xf16, {order = #NHWC}, [@CMX_NN, 0]>>, %bodyResults_30 as %arg3: !async.value<memref<64x1x1x4xsi32, [@CMX_NN, 0]>>, %bodyResults_46 as %arg4: !async.value<memref<64x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>>) -> !async.value<memref<1x64x24x48xf16, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DPU, "async-deps-index" = 11 : i64, cycleCost = 25361 : i64} {
    %11 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 25361 : i64, tiling_loop_index = 0 : i64, resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{is_zero_offset_weights_table, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<CONV>}> input(%arg2 : memref<1x256x24x48xf16, {order = #NHWC}, [@CMX_NN, 0]>) weights(%arg4 : memref<64x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) weight_table(%arg3 : memref<64x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%arg2 : memref<1x256x24x48xf16, {order = #NHWC}, [@CMX_NN, 0]>) parent_output(%alloc_15 : memref<1x64x24x48xf16, [@CMX_NN, 0]>) outputs(%alloc_15 : memref<1x64x24x48xf16, [@CMX_NN, 0]>) -> memref<1x64x24x48xf16, [@CMX_NN, 0]> variants : {
      DPUTask {inEnd = [47, 23, 255], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [47, 23, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>}
    }
    async.yield %11 : memref<1x64x24x48xf16, [@CMX_NN, 0]>
  })";

// Continuation of 2x5 tiling-CH IR: H-tile 1 (5 C-tiles) and copy-out DMAs
constexpr llvm::StringLiteral kPart3 = R"(
  %token_49, %bodyResults_50 = async.execute -> !async.value<memref<1x256x24x48xf16, {order = #NHWC}, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 12 : i64, cycleCost = 16718 : i64} {
    %11 = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NCHW} inputs(%arg0 : memref<1x48x48x256xf16, @DDR>) -> memref<1x256x48x48xf16, {order = #NHWC}, @DDR>
    %12 = VPUIP.SubView %11 [0, 0, 24, 0] [1, 256, 24, 48] : memref<1x256x48x48xf16, {order = #NHWC}, @DDR> to memref<1x256x24x48xf16, {order = #NHWC, strides = [589824, 1, 12288, 256]}, @DDR>
    %13 = VPUIP.NNDMA inputs(%12 : memref<1x256x24x48xf16, {order = #NHWC, strides = [589824, 1, 12288, 256]}, @DDR>) outputs(%alloc_16 : memref<1x256x24x48xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x256x24x48xf16, {order = #NHWC}, [@CMX_NN, 0]>
    async.yield %13 : memref<1x256x24x48xf16, {order = #NHWC}, [@CMX_NN, 0]>
  }
  %token_51, %bodyResults_52 = async.execute -> !async.value<memref<64x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 13 : i64, cycleCost = 1482 : i64} {
    %11 = VPUIP.NNDMA inputs(%cst_4 : memref<64x256x1x1xf16, {order = #NHWC}>) outputs(%alloc_17 : memref<64x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<64x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    async.yield %11 : memref<64x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
  }
  %token_53, %bodyResults_54 = async.execute [%token_29, %token_49, %token_51] (%bodyResults_30 as %arg2: !async.value<memref<64x1x1x4xsi32, [@CMX_NN, 0]>>, %bodyResults_50 as %arg3: !async.value<memref<1x256x24x48xf16, {order = #NHWC}, [@CMX_NN, 0]>>, %bodyResults_52 as %arg4: !async.value<memref<64x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>>) -> !async.value<memref<1x64x24x48xf16, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DPU, "async-deps-index" = 14 : i64, cycleCost = 25361 : i64} {
    %11 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 25361 : i64, tiling_loop_index = 0 : i64, resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{is_zero_offset_weights_table, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<CONV>}> input(%arg3 : memref<1x256x24x48xf16, {order = #NHWC}, [@CMX_NN, 0]>) weights(%arg4 : memref<64x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) weight_table(%arg2 : memref<64x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%arg3 : memref<1x256x24x48xf16, {order = #NHWC}, [@CMX_NN, 0]>) parent_output(%alloc_18 : memref<1x64x24x48xf16, [@CMX_NN, 0]>) outputs(%alloc_18 : memref<1x64x24x48xf16, [@CMX_NN, 0]>) -> memref<1x64x24x48xf16, [@CMX_NN, 0]> variants : {
      DPUTask {inEnd = [47, 23, 255], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [47, 23, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>}
    }
    async.yield %11 : memref<1x64x24x48xf16, [@CMX_NN, 0]>
  }
  %token_55, %bodyResults_56 = async.execute -> !async.value<memref<64x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 15 : i64, cycleCost = 1482 : i64} {
    %11 = VPUIP.NNDMA inputs(%cst_3 : memref<64x256x1x1xf16, {order = #NHWC}>) outputs(%alloc_19 : memref<64x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<64x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    async.yield %11 : memref<64x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
  }
  %token_57, %bodyResults_58 = async.execute [%token_29, %token_49, %token_55] (%bodyResults_30 as %arg2: !async.value<memref<64x1x1x4xsi32, [@CMX_NN, 0]>>, %bodyResults_50 as %arg3: !async.value<memref<1x256x24x48xf16, {order = #NHWC}, [@CMX_NN, 0]>>, %bodyResults_56 as %arg4: !async.value<memref<64x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>>) -> !async.value<memref<1x64x24x48xf16, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DPU, "async-deps-index" = 16 : i64, cycleCost = 25361 : i64} {
    %11 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 25361 : i64, tiling_loop_index = 0 : i64, resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{is_zero_offset_weights_table, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<CONV>}> input(%arg3 : memref<1x256x24x48xf16, {order = #NHWC}, [@CMX_NN, 0]>) weights(%arg4 : memref<64x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) weight_table(%arg2 : memref<64x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%arg3 : memref<1x256x24x48xf16, {order = #NHWC}, [@CMX_NN, 0]>) parent_output(%alloc_20 : memref<1x64x24x48xf16, [@CMX_NN, 0]>) outputs(%alloc_20 : memref<1x64x24x48xf16, [@CMX_NN, 0]>) -> memref<1x64x24x48xf16, [@CMX_NN, 0]> variants : {
      DPUTask {inEnd = [47, 23, 255], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [47, 23, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>}
    }
    async.yield %11 : memref<1x64x24x48xf16, [@CMX_NN, 0]>
  }
  %token_59, %bodyResults_60 = async.execute -> !async.value<memref<64x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 17 : i64, cycleCost = 1482 : i64} {
    %11 = VPUIP.NNDMA inputs(%cst_2 : memref<64x256x1x1xf16, {order = #NHWC}>) outputs(%alloc_21 : memref<64x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<64x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    async.yield %11 : memref<64x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
  }
  %token_61, %bodyResults_62 = async.execute [%token_29, %token_49, %token_59] (%bodyResults_30 as %arg2: !async.value<memref<64x1x1x4xsi32, [@CMX_NN, 0]>>, %bodyResults_50 as %arg3: !async.value<memref<1x256x24x48xf16, {order = #NHWC}, [@CMX_NN, 0]>>, %bodyResults_60 as %arg4: !async.value<memref<64x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>>) -> !async.value<memref<1x64x24x48xf16, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DPU, "async-deps-index" = 18 : i64, cycleCost = 25361 : i64} {
    %11 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 25361 : i64, tiling_loop_index = 0 : i64, resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{is_zero_offset_weights_table, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<CONV>}> input(%arg3 : memref<1x256x24x48xf16, {order = #NHWC}, [@CMX_NN, 0]>) weights(%arg4 : memref<64x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) weight_table(%arg2 : memref<64x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%arg3 : memref<1x256x24x48xf16, {order = #NHWC}, [@CMX_NN, 0]>) parent_output(%alloc_22 : memref<1x64x24x48xf16, [@CMX_NN, 0]>) outputs(%alloc_22 : memref<1x64x24x48xf16, [@CMX_NN, 0]>) -> memref<1x64x24x48xf16, [@CMX_NN, 0]> variants : {
      DPUTask {inEnd = [47, 23, 255], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [47, 23, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>}
    }
    async.yield %11 : memref<1x64x24x48xf16, [@CMX_NN, 0]>
  }
  %token_63, %bodyResults_64 = async.execute -> !async.value<memref<64x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 19 : i64, cycleCost = 1482 : i64} {
    %11 = VPUIP.NNDMA inputs(%cst_1 : memref<64x256x1x1xf16, {order = #NHWC}>) outputs(%alloc_23 : memref<64x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<64x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    async.yield %11 : memref<64x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
  }
  %token_65, %bodyResults_66 = async.execute [%token_29, %token_49, %token_63] (%bodyResults_30 as %arg2: !async.value<memref<64x1x1x4xsi32, [@CMX_NN, 0]>>, %bodyResults_50 as %arg3: !async.value<memref<1x256x24x48xf16, {order = #NHWC}, [@CMX_NN, 0]>>, %bodyResults_64 as %arg4: !async.value<memref<64x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>>) -> !async.value<memref<1x64x24x48xf16, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DPU, "async-deps-index" = 20 : i64, cycleCost = 25361 : i64} {
    %11 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 25361 : i64, tiling_loop_index = 0 : i64, resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{is_zero_offset_weights_table, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<CONV>}> input(%arg3 : memref<1x256x24x48xf16, {order = #NHWC}, [@CMX_NN, 0]>) weights(%arg4 : memref<64x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) weight_table(%arg2 : memref<64x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%arg3 : memref<1x256x24x48xf16, {order = #NHWC}, [@CMX_NN, 0]>) parent_output(%alloc_24 : memref<1x64x24x48xf16, [@CMX_NN, 0]>) outputs(%alloc_24 : memref<1x64x24x48xf16, [@CMX_NN, 0]>) -> memref<1x64x24x48xf16, [@CMX_NN, 0]> variants : {
      DPUTask {inEnd = [47, 23, 255], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [47, 23, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>}
    }
    async.yield %11 : memref<1x64x24x48xf16, [@CMX_NN, 0]>
  }
  %token_67, %bodyResults_68 = async.execute -> !async.value<memref<64x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 21 : i64, cycleCost = 1482 : i64} {
    %11 = VPUIP.NNDMA inputs(%cst_0 : memref<64x256x1x1xf16, {order = #NHWC}>) outputs(%alloc_25 : memref<64x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<64x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    async.yield %11 : memref<64x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
  }
  %token_69, %bodyResults_70 = async.execute [%token_29, %token_49, %token_67] (%bodyResults_30 as %arg2: !async.value<memref<64x1x1x4xsi32, [@CMX_NN, 0]>>, %bodyResults_50 as %arg3: !async.value<memref<1x256x24x48xf16, {order = #NHWC}, [@CMX_NN, 0]>>, %bodyResults_68 as %arg4: !async.value<memref<64x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>>) -> !async.value<memref<1x64x24x48xf16, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DPU, "async-deps-index" = 22 : i64, cycleCost = 25361 : i64} {
    %11 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 25361 : i64, tiling_loop_index = 0 : i64, resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{is_zero_offset_weights_table, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<CONV>}> input(%arg3 : memref<1x256x24x48xf16, {order = #NHWC}, [@CMX_NN, 0]>) weights(%arg4 : memref<64x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) weight_table(%arg2 : memref<64x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%arg3 : memref<1x256x24x48xf16, {order = #NHWC}, [@CMX_NN, 0]>) parent_output(%alloc_26 : memref<1x64x24x48xf16, [@CMX_NN, 0]>) outputs(%alloc_26 : memref<1x64x24x48xf16, [@CMX_NN, 0]>) -> memref<1x64x24x48xf16, [@CMX_NN, 0]> variants : {
      DPUTask {inEnd = [47, 23, 255], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [47, 23, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>}
    }
    async.yield %11 : memref<1x64x24x48xf16, [@CMX_NN, 0]>
  })";

// Continuation of 2x5 tiling-CH IR: copy-out DMAs and return
constexpr llvm::StringLiteral kPart4 = R"(
  %token_71, %bodyResults_72 = async.execute [%token_31] (%bodyResults_32 as %arg2: !async.value<memref<1x64x24x48xf16, [@CMX_NN, 0]>>) -> !async.value<memref<1x64x24x48xf16, {order = #NCHW, strides = [737280, 2304, 48, 1]}, @DDR>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 23 : i64, cycleCost = 4619 : i64} {
    %11 = VPUIP.SubView %arg1 [0, 0, 0, 0] [1, 64, 24, 48] : memref<1x320x48x48xf16, @DDR> to memref<1x64x24x48xf16, {order = #NCHW, strides = [737280, 2304, 48, 1]}, @DDR>
    %12 = VPUIP.NNDMA inputs(%arg2 : memref<1x64x24x48xf16, [@CMX_NN, 0]>) outputs(%11 : memref<1x64x24x48xf16, {order = #NCHW, strides = [737280, 2304, 48, 1]}, @DDR>) -> memref<1x64x24x48xf16, {order = #NCHW, strides = [737280, 2304, 48, 1]}, @DDR>
    async.yield %12 : memref<1x64x24x48xf16, {order = #NCHW, strides = [737280, 2304, 48, 1]}, @DDR>
  }
  %token_73, %bodyResults_74 = async.execute [%token_35] (%bodyResults_36 as %arg2: !async.value<memref<1x64x24x48xf16, [@CMX_NN, 0]>>) -> !async.value<memref<1x64x24x48xf16, {order = #NCHW, strides = [737280, 2304, 48, 1]}, @DDR>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 24 : i64, cycleCost = 4619 : i64} {
    %11 = VPUIP.SubView %arg1 [0, 64, 0, 0] [1, 64, 24, 48] : memref<1x320x48x48xf16, @DDR> to memref<1x64x24x48xf16, {order = #NCHW, strides = [737280, 2304, 48, 1]}, @DDR>
    %12 = VPUIP.NNDMA inputs(%arg2 : memref<1x64x24x48xf16, [@CMX_NN, 0]>) outputs(%11 : memref<1x64x24x48xf16, {order = #NCHW, strides = [737280, 2304, 48, 1]}, @DDR>) -> memref<1x64x24x48xf16, {order = #NCHW, strides = [737280, 2304, 48, 1]}, @DDR>
    async.yield %12 : memref<1x64x24x48xf16, {order = #NCHW, strides = [737280, 2304, 48, 1]}, @DDR>
  }
  %token_75, %bodyResults_76 = async.execute [%token_39] (%bodyResults_40 as %arg2: !async.value<memref<1x64x24x48xf16, [@CMX_NN, 0]>>) -> !async.value<memref<1x64x24x48xf16, {order = #NCHW, strides = [737280, 2304, 48, 1]}, @DDR>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 25 : i64, cycleCost = 4619 : i64} {
    %11 = VPUIP.SubView %arg1 [0, 128, 0, 0] [1, 64, 24, 48] : memref<1x320x48x48xf16, @DDR> to memref<1x64x24x48xf16, {order = #NCHW, strides = [737280, 2304, 48, 1]}, @DDR>
    %12 = VPUIP.NNDMA inputs(%arg2 : memref<1x64x24x48xf16, [@CMX_NN, 0]>) outputs(%11 : memref<1x64x24x48xf16, {order = #NCHW, strides = [737280, 2304, 48, 1]}, @DDR>) -> memref<1x64x24x48xf16, {order = #NCHW, strides = [737280, 2304, 48, 1]}, @DDR>
    async.yield %12 : memref<1x64x24x48xf16, {order = #NCHW, strides = [737280, 2304, 48, 1]}, @DDR>
  }
  %token_77, %bodyResults_78 = async.execute [%token_43] (%bodyResults_44 as %arg2: !async.value<memref<1x64x24x48xf16, [@CMX_NN, 0]>>) -> !async.value<memref<1x64x24x48xf16, {order = #NCHW, strides = [737280, 2304, 48, 1]}, @DDR>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 26 : i64, cycleCost = 4619 : i64} {
    %11 = VPUIP.SubView %arg1 [0, 192, 0, 0] [1, 64, 24, 48] : memref<1x320x48x48xf16, @DDR> to memref<1x64x24x48xf16, {order = #NCHW, strides = [737280, 2304, 48, 1]}, @DDR>
    %12 = VPUIP.NNDMA inputs(%arg2 : memref<1x64x24x48xf16, [@CMX_NN, 0]>) outputs(%11 : memref<1x64x24x48xf16, {order = #NCHW, strides = [737280, 2304, 48, 1]}, @DDR>) -> memref<1x64x24x48xf16, {order = #NCHW, strides = [737280, 2304, 48, 1]}, @DDR>
    async.yield %12 : memref<1x64x24x48xf16, {order = #NCHW, strides = [737280, 2304, 48, 1]}, @DDR>
  }
  %token_79, %bodyResults_80 = async.execute [%token_47] (%bodyResults_48 as %arg2: !async.value<memref<1x64x24x48xf16, [@CMX_NN, 0]>>) -> !async.value<memref<1x64x24x48xf16, {order = #NCHW, strides = [737280, 2304, 48, 1]}, @DDR>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 27 : i64, cycleCost = 4619 : i64} {
    %11 = VPUIP.SubView %arg1 [0, 256, 0, 0] [1, 64, 24, 48] : memref<1x320x48x48xf16, @DDR> to memref<1x64x24x48xf16, {order = #NCHW, strides = [737280, 2304, 48, 1]}, @DDR>
    %12 = VPUIP.NNDMA inputs(%arg2 : memref<1x64x24x48xf16, [@CMX_NN, 0]>) outputs(%11 : memref<1x64x24x48xf16, {order = #NCHW, strides = [737280, 2304, 48, 1]}, @DDR>) -> memref<1x64x24x48xf16, {order = #NCHW, strides = [737280, 2304, 48, 1]}, @DDR>
    async.yield %12 : memref<1x64x24x48xf16, {order = #NCHW, strides = [737280, 2304, 48, 1]}, @DDR>
  }
  %token_81, %bodyResults_82 = async.execute [%token_53] (%bodyResults_54 as %arg2: !async.value<memref<1x64x24x48xf16, [@CMX_NN, 0]>>) -> !async.value<memref<1x64x24x48xf16, {order = #NCHW, strides = [737280, 2304, 48, 1]}, @DDR>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 28 : i64, cycleCost = 4619 : i64} {
    %11 = VPUIP.SubView %arg1 [0, 0, 24, 0] [1, 64, 24, 48] : memref<1x320x48x48xf16, @DDR> to memref<1x64x24x48xf16, {order = #NCHW, strides = [737280, 2304, 48, 1]}, @DDR>
    %12 = VPUIP.NNDMA inputs(%arg2 : memref<1x64x24x48xf16, [@CMX_NN, 0]>) outputs(%11 : memref<1x64x24x48xf16, {order = #NCHW, strides = [737280, 2304, 48, 1]}, @DDR>) -> memref<1x64x24x48xf16, {order = #NCHW, strides = [737280, 2304, 48, 1]}, @DDR>
    async.yield %12 : memref<1x64x24x48xf16, {order = #NCHW, strides = [737280, 2304, 48, 1]}, @DDR>
  }
  %token_83, %bodyResults_84 = async.execute [%token_57] (%bodyResults_58 as %arg2: !async.value<memref<1x64x24x48xf16, [@CMX_NN, 0]>>) -> !async.value<memref<1x64x24x48xf16, {order = #NCHW, strides = [737280, 2304, 48, 1]}, @DDR>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 29 : i64, cycleCost = 4619 : i64} {
    %11 = VPUIP.SubView %arg1 [0, 64, 24, 0] [1, 64, 24, 48] : memref<1x320x48x48xf16, @DDR> to memref<1x64x24x48xf16, {order = #NCHW, strides = [737280, 2304, 48, 1]}, @DDR>
    %12 = VPUIP.NNDMA inputs(%arg2 : memref<1x64x24x48xf16, [@CMX_NN, 0]>) outputs(%11 : memref<1x64x24x48xf16, {order = #NCHW, strides = [737280, 2304, 48, 1]}, @DDR>) -> memref<1x64x24x48xf16, {order = #NCHW, strides = [737280, 2304, 48, 1]}, @DDR>
    async.yield %12 : memref<1x64x24x48xf16, {order = #NCHW, strides = [737280, 2304, 48, 1]}, @DDR>
  }
  %token_85, %bodyResults_86 = async.execute [%token_61] (%bodyResults_62 as %arg2: !async.value<memref<1x64x24x48xf16, [@CMX_NN, 0]>>) -> !async.value<memref<1x64x24x48xf16, {order = #NCHW, strides = [737280, 2304, 48, 1]}, @DDR>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 30 : i64, cycleCost = 4619 : i64} {
    %11 = VPUIP.SubView %arg1 [0, 128, 24, 0] [1, 64, 24, 48] : memref<1x320x48x48xf16, @DDR> to memref<1x64x24x48xf16, {order = #NCHW, strides = [737280, 2304, 48, 1]}, @DDR>
    %12 = VPUIP.NNDMA inputs(%arg2 : memref<1x64x24x48xf16, [@CMX_NN, 0]>) outputs(%11 : memref<1x64x24x48xf16, {order = #NCHW, strides = [737280, 2304, 48, 1]}, @DDR>) -> memref<1x64x24x48xf16, {order = #NCHW, strides = [737280, 2304, 48, 1]}, @DDR>
    async.yield %12 : memref<1x64x24x48xf16, {order = #NCHW, strides = [737280, 2304, 48, 1]}, @DDR>
  }
  %token_87, %bodyResults_88 = async.execute [%token_65] (%bodyResults_66 as %arg2: !async.value<memref<1x64x24x48xf16, [@CMX_NN, 0]>>) -> !async.value<memref<1x64x24x48xf16, {order = #NCHW, strides = [737280, 2304, 48, 1]}, @DDR>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 31 : i64, cycleCost = 4619 : i64} {
    %11 = VPUIP.SubView %arg1 [0, 192, 24, 0] [1, 64, 24, 48] : memref<1x320x48x48xf16, @DDR> to memref<1x64x24x48xf16, {order = #NCHW, strides = [737280, 2304, 48, 1]}, @DDR>
    %12 = VPUIP.NNDMA inputs(%arg2 : memref<1x64x24x48xf16, [@CMX_NN, 0]>) outputs(%11 : memref<1x64x24x48xf16, {order = #NCHW, strides = [737280, 2304, 48, 1]}, @DDR>) -> memref<1x64x24x48xf16, {order = #NCHW, strides = [737280, 2304, 48, 1]}, @DDR>
    async.yield %12 : memref<1x64x24x48xf16, {order = #NCHW, strides = [737280, 2304, 48, 1]}, @DDR>
  }
  %token_89, %bodyResults_90 = async.execute [%token_69] (%bodyResults_70 as %arg2: !async.value<memref<1x64x24x48xf16, [@CMX_NN, 0]>>) -> !async.value<memref<1x64x24x48xf16, {order = #NCHW, strides = [737280, 2304, 48, 1]}, @DDR>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 32 : i64, cycleCost = 4619 : i64} {
    %11 = VPUIP.SubView %arg1 [0, 256, 24, 0] [1, 64, 24, 48] : memref<1x320x48x48xf16, @DDR> to memref<1x64x24x48xf16, {order = #NCHW, strides = [737280, 2304, 48, 1]}, @DDR>
    %12 = VPUIP.NNDMA inputs(%arg2 : memref<1x64x24x48xf16, [@CMX_NN, 0]>) outputs(%11 : memref<1x64x24x48xf16, {order = #NCHW, strides = [737280, 2304, 48, 1]}, @DDR>) -> memref<1x64x24x48xf16, {order = #NCHW, strides = [737280, 2304, 48, 1]}, @DDR>
    async.yield %12 : memref<1x64x24x48xf16, {order = #NCHW, strides = [737280, 2304, 48, 1]}, @DDR>
  }
  %0 = async.await %bodyResults_72 : !async.value<memref<1x64x24x48xf16, {order = #NCHW, strides = [737280, 2304, 48, 1]}, @DDR>>
  %1 = async.await %bodyResults_74 : !async.value<memref<1x64x24x48xf16, {order = #NCHW, strides = [737280, 2304, 48, 1]}, @DDR>>
  %2 = async.await %bodyResults_76 : !async.value<memref<1x64x24x48xf16, {order = #NCHW, strides = [737280, 2304, 48, 1]}, @DDR>>
  %3 = async.await %bodyResults_78 : !async.value<memref<1x64x24x48xf16, {order = #NCHW, strides = [737280, 2304, 48, 1]}, @DDR>>
  %4 = async.await %bodyResults_80 : !async.value<memref<1x64x24x48xf16, {order = #NCHW, strides = [737280, 2304, 48, 1]}, @DDR>>
  %5 = async.await %bodyResults_82 : !async.value<memref<1x64x24x48xf16, {order = #NCHW, strides = [737280, 2304, 48, 1]}, @DDR>>
  %6 = async.await %bodyResults_84 : !async.value<memref<1x64x24x48xf16, {order = #NCHW, strides = [737280, 2304, 48, 1]}, @DDR>>
  %7 = async.await %bodyResults_86 : !async.value<memref<1x64x24x48xf16, {order = #NCHW, strides = [737280, 2304, 48, 1]}, @DDR>>
  %8 = async.await %bodyResults_88 : !async.value<memref<1x64x24x48xf16, {order = #NCHW, strides = [737280, 2304, 48, 1]}, @DDR>>
  %9 = async.await %bodyResults_90 : !async.value<memref<1x64x24x48xf16, {order = #NCHW, strides = [737280, 2304, 48, 1]}, @DDR>>
  %10 = VPUIP.ConcatView inputs(%0, %1, %2, %3, %4, %5, %6, %7, %8, %9 : memref<1x64x24x48xf16, {order = #NCHW, strides = [737280, 2304, 48, 1]}, @DDR>, memref<1x64x24x48xf16, {order = #NCHW, strides = [737280, 2304, 48, 1]}, @DDR>, memref<1x64x24x48xf16, {order = #NCHW, strides = [737280, 2304, 48, 1]}, @DDR>, memref<1x64x24x48xf16, {order = #NCHW, strides = [737280, 2304, 48, 1]}, @DDR>, memref<1x64x24x48xf16, {order = #NCHW, strides = [737280, 2304, 48, 1]}, @DDR>, memref<1x64x24x48xf16, {order = #NCHW, strides = [737280, 2304, 48, 1]}, @DDR>, memref<1x64x24x48xf16, {order = #NCHW, strides = [737280, 2304, 48, 1]}, @DDR>, memref<1x64x24x48xf16, {order = #NCHW, strides = [737280, 2304, 48, 1]}, @DDR>, memref<1x64x24x48xf16, {order = #NCHW, strides = [737280, 2304, 48, 1]}, @DDR>, memref<1x64x24x48xf16, {order = #NCHW, strides = [737280, 2304, 48, 1]}, @DDR>) outputs(%arg1 : memref<1x320x48x48xf16, @DDR>) -> memref<1x320x48x48xf16, @DDR>
  return %10 : memref<1x320x48x48xf16, @DDR>
}
})";

}  // namespace TilingCH2x5IR

TEST_F(MLIR_FeasibleMemorySchedulerLoop, ScheduleLoopRegionTilingCH_2x5) {
    // A single convolutional layer with tiling over C and H [1, 2, 5, 1]
    // This creates a 2x5 tiling pattern (2 tiles in H dimension, 5 tiles in C dimension)
    const auto platform = config::Platform::NPU5010;
    const auto arch = config::getArch(platform);
    VPU::initializeSingletons(registry, platform);

    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<vpux::VPU::VPUDialect>();

    auto module = mlir::parseSourceString<mlir::ModuleOp>(
            (TilingCH2x5IR::kPart1 + TilingCH2x5IR::kPart2 + TilingCH2x5IR::kPart3 + TilingCH2x5IR::kPart4).str(),
            &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    auto log = vpux::Logger::global();

    // Initialize the required components for scheduling
    auto aliasesInfo = AliasesInfoMemType<VPU::MemoryKind::CMX_NN>{func};
    AsyncDepsInfo depsInfo{func};

    const auto availableCMXSize = 1473536;

    // Create loop regions from tiling over C and H [1, 2, 5, 1]
    auto computeRegionVec = getComputeRegionsFromAsyncExec(aliasesInfo, depsInfo, log);
    // With relaxed matching, all 10 C-tile iterations (5 per H-tile group) merge into a single loop.
    // The weight table DMA (shared across all 10) is factored out as a global dependency.
    // Result: 2 regions - 1 non-loop (weight table DMA) + 1 tiling loop with 10 iterations.
    EXPECT_EQ(computeRegionVec.size(), 2);
    size_t tilingRegionCount = 0;
    for (auto& region : computeRegionVec) {
        if (region.getLoopType() != LoopType::None) {
            // Single merged loop with all 10 C-tile iterations (5 from each H-tile group)
            EXPECT_EQ(region.schedulingLoop->loopBodies.size(), 10);
            ++tilingRegionCount;
        }
    }
    EXPECT_EQ(tilingRegionCount, 1);

    // Use generateLoopSchedules directly to produce predefined schedules for the merged loop.
    auto liveRangeInfo = MemLiveRangeInfoMemType<VPU::MemoryKind::CMX_NN>{func, aliasesInfo};
    const auto memKind = VPU::MemoryKind::CMX_NN;
    const auto secondLvlMemKind = VPU::MemoryKind::DDR;
    uint64_t alignment = vpux::DEFAULT_CMX_ALIGNMENT;
    LinearScan<mlir::Value, LinearScanHandler> scan(availableCMXSize, {}, alignment);
    const auto vpuDevice = VPUNN::VPUDevice::NPU_5_0;
    auto costModel = VPU::CostModelConfig::createCostModel(&ctx);
    const int64_t nceClusterCount = 3;
    const int64_t dmaCount = 2;

    auto computeRegionsSchedule =
            VPU::generateLoopSchedules(computeRegionVec, availableCMXSize, /*enableVfUndefinedScheduler=*/false, log);
    FeasibleMemoryScheduler scheduler(memKind, secondLvlMemKind, liveRangeInfo, depsInfo, log, scan, arch, vpuDevice,
                                      costModel, nceClusterCount, dmaCount,
                                      /*enableScheduleStatistics*/ false, /*optimizeFragmentation*/ false,
                                      /*activelySpillForPrefetching*/ false, std::move(computeRegionsSchedule),
                                      std::move(computeRegionVec));
    FeasibleMemorySchedulerTest testAccessor(scheduler);
    EXPECT_EQ(testAccessor.getLoopRegionSize(), 2);
    scheduler.generateSchedule();
    // Single merged loop should be scheduled as a loop region
    EXPECT_EQ(testAccessor.getScheduledLoopRegionSize(), 1);
    EXPECT_TRUE(testAccessor.verifyLoopInputAddress());
}

TEST_F(MLIR_FeasibleMemorySchedulerLoop, ScheduleLoopRegionTilingCH_WithDdr2DdrConsumer) {
    // Check for the scenario ComputeRegion -> DDR2DDR DMA consumer
    // Verify that:
    //  1. The DDR2DDR DMA consumer is not wrapped into the loop region
    //  2. Other non-DDR2DDR ops are wrapped into the loop region
    //  3. The DDR2DDR DMA is scheduled after the loop region
    const auto platform = config::Platform::NPU5010;
    const auto arch = config::getArch(platform);
    VPU::initializeSingletons(registry, platform);

    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<vpux::VPU::VPUDialect>();
    ctx.loadDialect<vpux::VPUIP::VPUIPDialect>();

    auto module = createTiledConvolutionModule(&ctx, 2, 5, platform, /*addDdr2DdrConsumers=*/true);
    ASSERT_TRUE(module);

    auto func = module->lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    auto log = vpux::Logger::global();
    auto aliasesInfo = AliasesInfoMemType<VPU::MemoryKind::CMX_NN>{func};
    AsyncDepsInfo depsInfo{func};

    size_t totalDdr2DdrOpCount = 0;
    for (size_t opIdx = 0; opIdx < depsInfo.getExecOpCount(); ++opIdx) {
        if (VPUIP::isDmaDDR2DDR(depsInfo.getExecuteOpAtIndex(opIdx))) {
            ++totalDdr2DdrOpCount;
        }
    }
    EXPECT_EQ(totalDdr2DdrOpCount, 1u);

    auto computeRegionVec = getComputeRegionsFromAsyncExec(aliasesInfo, depsInfo, log);
    size_t tilingRegionCount = 0;
    size_t loopDdr2DdrOpCount = 0;
    for (const auto& region : computeRegionVec) {
        if (region.getLoopType() == LoopType::None) {
            continue;
        }
        ++tilingRegionCount;
        ASSERT_EQ(region.schedulingLoop->loopBodies.size(), 10u);
        for (const auto& iteration : region.schedulingLoop->loopBodies) {
            EXPECT_EQ(iteration.size(), 4u);
            for (const auto& op : iteration) {
                if (VPUIP::isDmaDDR2DDR(depsInfo.getExecuteOpAtIndex(op.opIdx))) {
                    ++loopDdr2DdrOpCount;
                }
            }
        }
    }
    EXPECT_EQ(tilingRegionCount, 1u);
    EXPECT_EQ(loopDdr2DdrOpCount, 0u);

    const auto availableCMXSize = 1473536;
    auto liveRangeInfo = MemLiveRangeInfoMemType<VPU::MemoryKind::CMX_NN>{func, aliasesInfo};
    uint64_t alignment = vpux::DEFAULT_CMX_ALIGNMENT;
    LinearScan<mlir::Value, LinearScanHandler> scan(availableCMXSize, {}, alignment);
    auto costModel = VPU::CostModelConfig::createCostModel(&ctx);
    auto computeRegionsSchedule =
            VPU::generateLoopSchedules(computeRegionVec, availableCMXSize, /*enableVfUndefinedScheduler=*/false, log);
    FeasibleMemoryScheduler scheduler(VPU::MemoryKind::CMX_NN, VPU::MemoryKind::DDR, liveRangeInfo, depsInfo, log, scan,
                                      arch, VPUNN::VPUDevice::NPU_5_0, costModel,
                                      /*nceClusterCount=*/3, /*dmaCount=*/2,
                                      /*enableScheduleStatistics=*/false, /*optimizeFragmentation=*/false,
                                      /*activelySpillForPrefetching=*/false, std::move(computeRegionsSchedule),
                                      std::move(computeRegionVec));
    FeasibleMemorySchedulerTest testAccessor(scheduler);
    EXPECT_EQ(testAccessor.getLoopRegionSize(), 3);
    EXPECT_NO_THROW(scheduler.generateSchedule());
    EXPECT_EQ(testAccessor.getScheduledLoopRegionSize(), 1);

    bool loopOpSeen = false;
    bool ddr2DdrOpSeen = false;
    bool loopOpSeenAfterDdr2Ddr = false;
    size_t scheduledDdr2DdrOpCount = 0;
    for (const auto& scheduledOp : testAccessor.getScheduledOps()) {
        if (scheduledOp.isLoopOp()) {
            loopOpSeen = true;
            if (ddr2DdrOpSeen) {
                loopOpSeenAfterDdr2Ddr = true;
            }
        }
        if (VPUIP::isDmaDDR2DDR(depsInfo.getExecuteOpAtIndex(scheduledOp.op_))) {
            ++scheduledDdr2DdrOpCount;
            ddr2DdrOpSeen = true;
            EXPECT_TRUE(loopOpSeen) << "DDR2DDR DMA must be scheduled after the loop region";
        }
    }

    EXPECT_EQ(scheduledDdr2DdrOpCount, 1u);
    EXPECT_TRUE(ddr2DdrOpSeen);
    EXPECT_FALSE(loopOpSeenAfterDdr2Ddr) << "No loop operation may be scheduled after the DDR2DDR DMA";
}

// Verify that loop scheduling with tight CMX completes without crash.
// With reduced CMX, higher memory pressure forces the scheduler into spilling paths.
TEST_F(MLIR_FeasibleMemorySchedulerLoop, ScheduleLoopRegionTilingCH_WithTightCMX) {
    const auto platform = config::Platform::NPU5010;
    const auto arch = config::getArch(platform);
    VPU::initializeSingletons(registry, platform);

    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<vpux::VPU::VPUDialect>();

    auto module = mlir::parseSourceString<mlir::ModuleOp>(
            (TilingCH2x5IR::kPart1 + TilingCH2x5IR::kPart2 + TilingCH2x5IR::kPart3 + TilingCH2x5IR::kPart4).str(),
            &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    auto log = vpux::Logger::global();

    auto aliasesInfo = AliasesInfoMemType<VPU::MemoryKind::CMX_NN>{func};
    AsyncDepsInfo depsInfo{func};

    const auto tightCMXSize = 1000000;

    auto computeRegionVec = getComputeRegionsFromAsyncExec(aliasesInfo, depsInfo, log);
    auto liveRangeInfo = MemLiveRangeInfoMemType<VPU::MemoryKind::CMX_NN>{func, aliasesInfo};
    const auto memKind = VPU::MemoryKind::CMX_NN;
    const auto secondLvlMemKind = VPU::MemoryKind::DDR;
    uint64_t alignment = vpux::DEFAULT_CMX_ALIGNMENT;
    LinearScan<mlir::Value, LinearScanHandler> scan(tightCMXSize, {}, alignment);
    const auto vpuDevice = VPUNN::VPUDevice::NPU_5_0;
    auto costModel = VPU::CostModelConfig::createCostModel(&ctx);
    const int64_t nceClusterCount = 3;
    const int64_t dmaCount = 2;

    auto computeRegionsSchedule =
            VPU::generateLoopSchedules(computeRegionVec, tightCMXSize, /*enableVfUndefinedScheduler=*/false, log);
    FeasibleMemoryScheduler scheduler(memKind, secondLvlMemKind, liveRangeInfo, depsInfo, log, scan, arch, vpuDevice,
                                      costModel, nceClusterCount, dmaCount,
                                      /*enableScheduleStatistics*/ false, /*optimizeFragmentation*/ false,
                                      /*activelySpillForPrefetching*/ false, std::move(computeRegionsSchedule),
                                      std::move(computeRegionVec));
    FeasibleMemorySchedulerTest testAccessor(scheduler);

    // Must complete without throwing.
    EXPECT_NO_THROW(scheduler.generateSchedule());

    // Under tight CMX the scheduler may or may not schedule loops depending on whether
    // prepareLoopRegion can fit shared buffers + reserved block. Verify consistency:
    // either the merged loop was scheduled (positive) or gracefully skipped (zero).
    const auto scheduledLoops = testAccessor.getScheduledLoopRegionSize();
    EXPECT_TRUE(scheduledLoops == 0 || scheduledLoops == 1)
            << "Expected 0 (graceful fallback) or 1 (merged loop scheduled), got " << scheduledLoops;
}

TEST_F(MLIR_FeasibleMemorySchedulerLoop, GenerateLoopSchedules_PopulatesScheduleState) {
    // Verifies that generateLoopSchedules correctly populates the ComputeRegionsSchedule
    // structure which is consumed by FeasibleMemoryScheduler to apply predefined schedules.
    // Uses the same 5-tile-C IR from ScheduleLoopRegion test.
    constexpr llvm::StringLiteral inputIRPart1 = R"(

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
!TotalInputType = !VPUIP.DistributedBuffer<1x480x88x27xf16, {order = #NHWC}, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments, compute_shapes = [[1, 480, 30, 27], [1, 480, 29, 27], [1, 480, 29, 27]], compute_offsets = [[0, 0, 0, 0], [0, 0, 30, 0], [0, 0, 59, 0]], memory_shapes = [[1, 480, 30, 27], [1, 480, 29, 27], [1, 480, 29, 27]], memory_offsets = [[0, 0, 0, 0], [0, 0, 30, 0], [0, 0, 59, 0]]}>
!SliceWeightsType = !VPUIP.DistributedBuffer<96x480x1x1xf16, {order = #NHWC}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 3 : i64, uniform_distributed_segments, compute_shapes = [[96, 480, 1, 1], [96, 480, 1, 1], [96, 480, 1, 1]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], memory_shapes = [[96, 480, 1, 1], [96, 480, 1, 1], [96, 480, 1, 1]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
!SliceWTType = !VPUIP.DistributedBuffer<96x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 3 : i64, uniform_distributed_segments, compute_shapes = [[96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], memory_shapes = [[96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
!SliceOutputType = !VPUIP.DistributedBuffer<1x96x88x27xf16, #NCHW, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments, compute_shapes = [[1, 96, 30, 27], [1, 96, 29, 27], [1, 96, 29, 27]], compute_offsets = [[0, 0, 0, 0], [0, 0, 30, 0], [0, 0, 59, 0]], memory_shapes = [[1, 96, 30, 27], [1, 96, 29, 27], [1, 96, 29, 27]], memory_offsets = [[0, 0, 0, 0], [0, 0, 30, 0], [0, 0, 59, 0]]}>

module @model_name attributes {config.compilationMode = #config.compilation_mode<DefaultHW>, config.platform = #config.platform<NPU5010>} {
func.func @main(%arg0: memref<1x88x27x480xf16, @DDR>, %arg1: memref<1x480x88x27xf16, @DDR>) -> memref<1x480x88x27xf16, @DDR> {
  %cst = const.Declare memref<96x480x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<480x480x1x1xf32>, [#const.SubView<[0, 0, 0, 0], [96, 480, 1, 1]>, #const.CastElemType<f16>, #const.Reorder<#NHWC>]
  %cst_0 = const.Declare memref<96x480x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<480x480x1x1xf32>, [#const.SubView<[96, 0, 0, 0], [96, 480, 1, 1]>, #const.CastElemType<f16>, #const.Reorder<#NHWC>]
  %cst_1 = const.Declare memref<96x480x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<480x480x1x1xf32>, [#const.SubView<[192, 0, 0, 0], [96, 480, 1, 1]>, #const.CastElemType<f16>, #const.Reorder<#NHWC>]
  %cst_2 = const.Declare memref<96x480x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<480x480x1x1xf32>, [#const.SubView<[288, 0, 0, 0], [96, 480, 1, 1]>, #const.CastElemType<f16>, #const.Reorder<#NHWC>]
  %cst_3 = const.Declare memref<96x480x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<480x480x1x1xf32>, [#const.SubView<[384, 0, 0, 0], [96, 480, 1, 1]>, #const.CastElemType<f16>, #const.Reorder<#NHWC>]
  %cst_4 = const.Declare memref<96x1x1x4xsi32> = dense<1> : tensor<96x1x1x4xsi32>
  %0 = VPURT.AllocDistributed -> !TotalInputType
  %1 = VPURT.AllocDistributed -> !SliceWeightsType
  %2 = VPURT.AllocDistributed -> !SliceWTType
  %3 = VPURT.AllocDistributed -> !SliceOutputType
  %4 = VPURT.AllocDistributed -> !SliceWeightsType
  %5 = VPURT.AllocDistributed -> !SliceOutputType
  %6 = VPURT.AllocDistributed -> !SliceWeightsType
  %7 = VPURT.AllocDistributed -> !SliceOutputType
  %8 = VPURT.AllocDistributed -> !SliceWeightsType
  %9 = VPURT.AllocDistributed -> !SliceOutputType
  %10 = VPURT.AllocDistributed -> !SliceWeightsType
  %11 = VPURT.AllocDistributed -> !SliceOutputType
  %token, %bodyResults = async.execute -> !async.value<!TotalInputType> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 0 : i64, cycleCost = 32071 : i64} {
    %18 = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NCHW} inputs(%arg0 : memref<1x88x27x480xf16, @DDR>) -> memref<1x480x88x27xf16, {order = #NHWC}, @DDR>
    %19 = VPUIP.NNDMA inputs(%18 : memref<1x480x88x27xf16, {order = #NHWC}, @DDR>) outputs(%0 : !TotalInputType) -> !TotalInputType
    async.yield %19 : !TotalInputType
  }
  %token_5, %bodyResults_6 = async.execute -> !async.value<!SliceWeightsType> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 1 : i64, cycleCost = 3106 : i64} {
    %18 = VPUIP.NNDMA inputs(%cst : memref<96x480x1x1xf16, {order = #NHWC}>) outputs(%1 : !SliceWeightsType) -> !SliceWeightsType
    async.yield %18 : !SliceWeightsType
  }
  %token_7, %bodyResults_8 = async.execute -> !async.value<!SliceWTType> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 2 : i64, cycleCost = 607 : i64} {
    %18 = VPUIP.NNDMA inputs(%cst_4 : memref<96x1x1x4xsi32>) outputs(%2 : !SliceWTType) -> !SliceWTType
    async.yield %18 : !SliceWTType
  }
  %token_9, %bodyResults_10 = async.execute [%token, %token_5, %token_7] (%bodyResults as %arg2: !async.value<!TotalInputType>, %bodyResults_6 as %arg3: !async.value<!SliceWeightsType>, %bodyResults_8 as %arg4: !async.value<!SliceWTType>) -> !async.value<!SliceOutputType> attributes {VPUIP.executor = @DPU, "async-deps-index" = 3 : i64, cycleCost = 29731 : i64} {
    %18 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 29731 : i64, tiling_loop_index = 0 : i64, resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{is_superdense, is_zero_offset_weights_table, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<CONV>}> input(%arg2 : !TotalInputType) weights(%arg3 : !SliceWeightsType) weight_table(%arg4 : !SliceWTType) parent_input(%arg2 : !TotalInputType) parent_output(%3 : !SliceOutputType) outputs(%3 : !SliceOutputType) -> !SliceOutputType variants : {
      DPUTask {cluster_id = 0 : i64, inEnd = [26, 29, 479], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [26, 29, 95], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      DPUTask {cluster_id = 1 : i64, inEnd = [26, 28, 479], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [26, 28, 95], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      DPUTask {cluster_id = 2 : i64, inEnd = [26, 28, 479], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [26, 28, 95], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>}
    }
    async.yield %18 : !SliceOutputType
  }
  %token_11, %bodyResults_12 = async.execute [%token_9] (%bodyResults_10 as %arg2: !async.value<!SliceOutputType>) -> !async.value<memref<1x96x88x27xf16, {order = #NCHW, strides = [1140480, 2376, 27, 1]}, @DDR>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 4 : i64, cycleCost = 7116 : i64} {
    %18 = VPUIP.SubView %arg1 [0, 0, 0, 0] [1, 96, 88, 27] : memref<1x480x88x27xf16, @DDR> to memref<1x96x88x27xf16, {order = #NCHW, strides = [1140480, 2376, 27, 1]}, @DDR>
    %19 = VPUIP.NNDMA inputs(%arg2 : !SliceOutputType) outputs(%18 : memref<1x96x88x27xf16, {order = #NCHW, strides = [1140480, 2376, 27, 1]}, @DDR>) -> memref<1x96x88x27xf16, {order = #NCHW, strides = [1140480, 2376, 27, 1]}, @DDR>
    async.yield %19 : memref<1x96x88x27xf16, {order = #NCHW, strides = [1140480, 2376, 27, 1]}, @DDR>
  }
  %token_13, %bodyResults_14 = async.execute -> !async.value<!SliceWeightsType> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 5 : i64, cycleCost = 3106 : i64} {
    %18 = VPUIP.NNDMA inputs(%cst_0 : memref<96x480x1x1xf16, {order = #NHWC}>) outputs(%4 : !SliceWeightsType) -> !SliceWeightsType
    async.yield %18 : !SliceWeightsType
  }
  %token_15, %bodyResults_16 = async.execute [%token, %token_7, %token_13] (%bodyResults as %arg2: !async.value<!TotalInputType>, %bodyResults_8 as %arg3: !async.value<!SliceWTType>, %bodyResults_14 as %arg4: !async.value<!SliceWeightsType>) -> !async.value<!SliceOutputType> attributes {VPUIP.executor = @DPU, "async-deps-index" = 6 : i64, cycleCost = 29731 : i64} {
    %18 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 29731 : i64, tiling_loop_index = 0 : i64, resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{is_superdense, is_zero_offset_weights_table, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<CONV>}> input(%arg2 : !TotalInputType) weights(%arg4 : !SliceWeightsType) weight_table(%arg3 : !SliceWTType) parent_input(%arg2 : !TotalInputType) parent_output(%5 : !SliceOutputType) outputs(%5 : !SliceOutputType) -> !SliceOutputType variants : {
      DPUTask {cluster_id = 0 : i64, inEnd = [26, 29, 479], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [26, 29, 95], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      DPUTask {cluster_id = 1 : i64, inEnd = [26, 28, 479], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [26, 28, 95], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      DPUTask {cluster_id = 2 : i64, inEnd = [26, 28, 479], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [26, 28, 95], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>}
    }
    async.yield %18 : !SliceOutputType
  }
  %token_17, %bodyResults_18 = async.execute [%token_15] (%bodyResults_16 as %arg2: !async.value<!SliceOutputType>) -> !async.value<memref<1x96x88x27xf16, {order = #NCHW, strides = [1140480, 2376, 27, 1]}, @DDR>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 7 : i64, cycleCost = 7116 : i64} {
    %18 = VPUIP.SubView %arg1 [0, 96, 0, 0] [1, 96, 88, 27] : memref<1x480x88x27xf16, @DDR> to memref<1x96x88x27xf16, {order = #NCHW, strides = [1140480, 2376, 27, 1]}, @DDR>
    %19 = VPUIP.NNDMA inputs(%arg2 : !SliceOutputType) outputs(%18 : memref<1x96x88x27xf16, {order = #NCHW, strides = [1140480, 2376, 27, 1]}, @DDR>) -> memref<1x96x88x27xf16, {order = #NCHW, strides = [1140480, 2376, 27, 1]}, @DDR>
    async.yield %19 : memref<1x96x88x27xf16, {order = #NCHW, strides = [1140480, 2376, 27, 1]}, @DDR>
  })";

    constexpr llvm::StringLiteral inputIRPart2 = R"(
  %token_19, %bodyResults_20 = async.execute -> !async.value<!SliceWeightsType> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 8 : i64, cycleCost = 3106 : i64} {
    %18 = VPUIP.NNDMA inputs(%cst_1 : memref<96x480x1x1xf16, {order = #NHWC}>) outputs(%6 : !SliceWeightsType) -> !SliceWeightsType
    async.yield %18 : !SliceWeightsType
  }
  %token_21, %bodyResults_22 = async.execute [%token, %token_7, %token_19] (%bodyResults as %arg2: !async.value<!TotalInputType>, %bodyResults_8 as %arg3: !async.value<!SliceWTType>, %bodyResults_20 as %arg4: !async.value<!SliceWeightsType>) -> !async.value<!SliceOutputType> attributes {VPUIP.executor = @DPU, "async-deps-index" = 9 : i64, cycleCost = 29731 : i64} {
    %18 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 29731 : i64, tiling_loop_index = 0 : i64, resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{is_superdense, is_zero_offset_weights_table, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<CONV>}> input(%arg2 : !TotalInputType) weights(%arg4 : !SliceWeightsType) weight_table(%arg3 : !SliceWTType) parent_input(%arg2 : !TotalInputType) parent_output(%7 : !SliceOutputType) outputs(%7 : !SliceOutputType) -> !SliceOutputType variants : {
      DPUTask {cluster_id = 0 : i64, inEnd = [26, 29, 479], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [26, 29, 95], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      DPUTask {cluster_id = 1 : i64, inEnd = [26, 28, 479], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [26, 28, 95], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      DPUTask {cluster_id = 2 : i64, inEnd = [26, 28, 479], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [26, 28, 95], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>}
    }
    async.yield %18 : !SliceOutputType
  }
  %token_23, %bodyResults_24 = async.execute [%token_21] (%bodyResults_22 as %arg2: !async.value<!SliceOutputType>) -> !async.value<memref<1x96x88x27xf16, {order = #NCHW, strides = [1140480, 2376, 27, 1]}, @DDR>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 10 : i64, cycleCost = 7116 : i64} {
    %18 = VPUIP.SubView %arg1 [0, 192, 0, 0] [1, 96, 88, 27] : memref<1x480x88x27xf16, @DDR> to memref<1x96x88x27xf16, {order = #NCHW, strides = [1140480, 2376, 27, 1]}, @DDR>
    %19 = VPUIP.NNDMA inputs(%arg2 : !SliceOutputType) outputs(%18 : memref<1x96x88x27xf16, {order = #NCHW, strides = [1140480, 2376, 27, 1]}, @DDR>) -> memref<1x96x88x27xf16, {order = #NCHW, strides = [1140480, 2376, 27, 1]}, @DDR>
    async.yield %19 : memref<1x96x88x27xf16, {order = #NCHW, strides = [1140480, 2376, 27, 1]}, @DDR>
  }
  %token_25, %bodyResults_26 = async.execute -> !async.value<!SliceWeightsType> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 11 : i64, cycleCost = 3106 : i64} {
    %18 = VPUIP.NNDMA inputs(%cst_2 : memref<96x480x1x1xf16, {order = #NHWC}>) outputs(%8 : !SliceWeightsType) -> !SliceWeightsType
    async.yield %18 : !SliceWeightsType
  }
  %token_27, %bodyResults_28 = async.execute [%token, %token_7, %token_25] (%bodyResults as %arg2: !async.value<!TotalInputType>, %bodyResults_8 as %arg3: !async.value<!SliceWTType>, %bodyResults_26 as %arg4: !async.value<!SliceWeightsType>) -> !async.value<!SliceOutputType> attributes {VPUIP.executor = @DPU, "async-deps-index" = 12 : i64, cycleCost = 29731 : i64} {
    %18 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 29731 : i64, tiling_loop_index = 0 : i64, resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{is_superdense, is_zero_offset_weights_table, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<CONV>}> input(%arg2 : !TotalInputType) weights(%arg4 : !SliceWeightsType) weight_table(%arg3 : !SliceWTType) parent_input(%arg2 : !TotalInputType) parent_output(%9 : !SliceOutputType) outputs(%9 : !SliceOutputType) -> !SliceOutputType variants : {
      DPUTask {cluster_id = 0 : i64, inEnd = [26, 29, 479], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [26, 29, 95], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      DPUTask {cluster_id = 1 : i64, inEnd = [26, 28, 479], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [26, 28, 95], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      DPUTask {cluster_id = 2 : i64, inEnd = [26, 28, 479], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [26, 28, 95], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>}
    }
    async.yield %18 : !SliceOutputType
  }
  %token_29, %bodyResults_30 = async.execute [%token_27] (%bodyResults_28 as %arg2: !async.value<!SliceOutputType>) -> !async.value<memref<1x96x88x27xf16, {order = #NCHW, strides = [1140480, 2376, 27, 1]}, @DDR>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 13 : i64, cycleCost = 7116 : i64} {
    %18 = VPUIP.SubView %arg1 [0, 288, 0, 0] [1, 96, 88, 27] : memref<1x480x88x27xf16, @DDR> to memref<1x96x88x27xf16, {order = #NCHW, strides = [1140480, 2376, 27, 1]}, @DDR>
    %19 = VPUIP.NNDMA inputs(%arg2 : !SliceOutputType) outputs(%18 : memref<1x96x88x27xf16, {order = #NCHW, strides = [1140480, 2376, 27, 1]}, @DDR>) -> memref<1x96x88x27xf16, {order = #NCHW, strides = [1140480, 2376, 27, 1]}, @DDR>
    async.yield %19 : memref<1x96x88x27xf16, {order = #NCHW, strides = [1140480, 2376, 27, 1]}, @DDR>
  }
  %token_31, %bodyResults_32 = async.execute -> !async.value<!SliceWeightsType> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 14 : i64, cycleCost = 3106 : i64} {
    %18 = VPUIP.NNDMA inputs(%cst_3 : memref<96x480x1x1xf16, {order = #NHWC}>) outputs(%10 : !SliceWeightsType) -> !SliceWeightsType
    async.yield %18 : !SliceWeightsType
  }
  %token_33, %bodyResults_34 = async.execute [%token, %token_7, %token_31] (%bodyResults as %arg2: !async.value<!TotalInputType>, %bodyResults_8 as %arg3: !async.value<!SliceWTType>, %bodyResults_32 as %arg4: !async.value<!SliceWeightsType>) -> !async.value<!SliceOutputType> attributes {VPUIP.executor = @DPU, "async-deps-index" = 15 : i64, cycleCost = 29731 : i64} {
    %18 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 29731 : i64, tiling_loop_index = 0 : i64, resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{is_superdense, is_zero_offset_weights_table, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<CONV>}> input(%arg2 : !TotalInputType) weights(%arg4 : !SliceWeightsType) weight_table(%arg3 : !SliceWTType) parent_input(%arg2 : !TotalInputType) parent_output(%11 : !SliceOutputType) outputs(%11 : !SliceOutputType) -> !SliceOutputType variants : {
      DPUTask {cluster_id = 0 : i64, inEnd = [26, 29, 479], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [26, 29, 95], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      DPUTask {cluster_id = 1 : i64, inEnd = [26, 28, 479], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [26, 28, 95], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      DPUTask {cluster_id = 2 : i64, inEnd = [26, 28, 479], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [26, 28, 95], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>}
    }
    async.yield %18 : !SliceOutputType
  }
  %token_35, %bodyResults_36 = async.execute [%token_33] (%bodyResults_34 as %arg2: !async.value<!SliceOutputType>) -> !async.value<memref<1x96x88x27xf16, {order = #NCHW, strides = [1140480, 2376, 27, 1]}, @DDR>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 16 : i64, cycleCost = 7116 : i64} {
    %18 = VPUIP.SubView %arg1 [0, 384, 0, 0] [1, 96, 88, 27] : memref<1x480x88x27xf16, @DDR> to memref<1x96x88x27xf16, {order = #NCHW, strides = [1140480, 2376, 27, 1]}, @DDR>
    %19 = VPUIP.NNDMA inputs(%arg2 : !SliceOutputType) outputs(%18 : memref<1x96x88x27xf16, {order = #NCHW, strides = [1140480, 2376, 27, 1]}, @DDR>) -> memref<1x96x88x27xf16, {order = #NCHW, strides = [1140480, 2376, 27, 1]}, @DDR>
    async.yield %19 : memref<1x96x88x27xf16, {order = #NCHW, strides = [1140480, 2376, 27, 1]}, @DDR>
  }
  %12 = async.await %bodyResults_12 : !async.value<memref<1x96x88x27xf16, {order = #NCHW, strides = [1140480, 2376, 27, 1]}, @DDR>>
  %13 = async.await %bodyResults_18 : !async.value<memref<1x96x88x27xf16, {order = #NCHW, strides = [1140480, 2376, 27, 1]}, @DDR>>
  %14 = async.await %bodyResults_24 : !async.value<memref<1x96x88x27xf16, {order = #NCHW, strides = [1140480, 2376, 27, 1]}, @DDR>>
  %15 = async.await %bodyResults_30 : !async.value<memref<1x96x88x27xf16, {order = #NCHW, strides = [1140480, 2376, 27, 1]}, @DDR>>
  %16 = async.await %bodyResults_36 : !async.value<memref<1x96x88x27xf16, {order = #NCHW, strides = [1140480, 2376, 27, 1]}, @DDR>>
  %17 = VPUIP.ConcatView inputs(%12, %13, %14, %15, %16 : memref<1x96x88x27xf16, {order = #NCHW, strides = [1140480, 2376, 27, 1]}, @DDR>, memref<1x96x88x27xf16, {order = #NCHW, strides = [1140480, 2376, 27, 1]}, @DDR>, memref<1x96x88x27xf16, {order = #NCHW, strides = [1140480, 2376, 27, 1]}, @DDR>, memref<1x96x88x27xf16, {order = #NCHW, strides = [1140480, 2376, 27, 1]}, @DDR>, memref<1x96x88x27xf16, {order = #NCHW, strides = [1140480, 2376, 27, 1]}, @DDR>) outputs(%arg1 : memref<1x480x88x27xf16, @DDR>) -> memref<1x480x88x27xf16, @DDR>
  return %17 : memref<1x480x88x27xf16, @DDR>
}
})";

    const auto platform = config::Platform::NPU5010;
    VPU::initializeSingletons(registry, platform);

    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<vpux::VPU::VPUDialect>();

    auto module = mlir::parseSourceString<mlir::ModuleOp>((inputIRPart1 + inputIRPart2).str(), &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    auto log = vpux::Logger::global();

    auto aliasesInfo = AliasesInfoMemType<VPU::MemoryKind::CMX_NN>{func};
    AsyncDepsInfo depsInfo{func};
    const auto availableCMXSize = 1473536;

    auto computeRegionVec = getComputeRegionsFromAsyncExec(aliasesInfo, depsInfo, log);

    // Verify that generateLoopSchedules produces valid schedule state
    auto scheduleState =
            VPU::generateLoopSchedules(computeRegionVec, availableCMXSize, /*enableVfUndefinedScheduler=*/false, log);

    // Only tiling regions should produce schedules
    for (const auto& [idx, result] : scheduleState.scheduleResults) {
        ASSERT_LT(idx, computeRegionVec.size()) << "Schedule result index out of range";
        EXPECT_NE(computeRegionVec[idx].getLoopType(), LoopType::None)
                << "Schedule should not be generated for non-loop regions";
        EXPECT_FALSE(result.empty()) << "Schedule result at index " << idx << " should not be empty";
    }

    // loopRegionInd and loopPrefetchInd should be disjoint
    for (auto idx : scheduleState.loopRegionInd) {
        EXPECT_FALSE(scheduleState.loopPrefetchInd.contains(idx))
                << "Op index " << idx << " is in both loopRegionInd and loopPrefetchInd";
    }

    // DATA_IN ops should be categorized as prefetchable
    EXPECT_FALSE(scheduleState.loopPrefetchInd.empty()) << "DATA_IN operations should be eligible for prefetching";

    // COMPUTE ops should be in loopRegionInd
    EXPECT_FALSE(scheduleState.loopRegionInd.empty()) << "COMPUTE operations should be in loopRegionInd";
}

TEST_F(MLIR_FeasibleMemorySchedulerLoop, ScheduleVfLoopRegion) {
    // IR structure:
    // OP0 - DMA_IN (Common weight Load for VF first compute op)
    // OP1 - DMA_IN (Common weightTable Load for first compute op)
    // OP2 - DMA_IN (Common weight Load for VF second compute op)
    // OP3 - DMA_IN (Common weightTable Load for second compute op)
    // VF Loop Region 0, iteration 0 and 1:
    //  OP4, OP8 - DATA_IN
    //  OP5, OP9 - COMPUTE_OP
    //  OP6,OP10 - COMPUTE_OP
    //  OP7,OP11 - DATA_OUT

    constexpr llvm::StringLiteral inputIRPart1 = R"(

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!alloc_in_tile38 = !VPUIP.DistributedBuffer<1x32x800x38x!quant.uniform<u8:f16, 0.0:127>, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments, compute_shapes = [[1, 32, 267, 38], [1, 32, 267, 38], [1, 32, 266, 38]], compute_offsets = [[0, 0, 0, 0], [0, 0, 267, 0], [0, 0, 534, 0]], memory_shapes = [[1, 32, 268, 38], [1, 32, 269, 38], [1, 32, 267, 38]], memory_offsets = [[0, 0, 0, 0], [0, 0, 266, 0], [0, 0, 533, 0]]}>
!alloc_weights_3x3 = !VPUIP.DistributedBuffer<32x32x3x3x!quant.uniform<u8:f16:0, {4.9856147345374616E-4,4.8728018414740466E-4,4.9659171525169826E-4,5.3172853647493845E-4,4.6505197590472651E-4,5.9242686804603128E-4,6.6902386207206579E-4,4.5109075658461627E-4,5.2083187243517708E-4,5.5445976117077993E-4,4.6679894713794483E-4,5.233020759096333E-4,4.6531297996932385E-4,5.0246493489134543E-4,6.1671476738125672E-4,4.66597343192381E-4,5.4963564171510589E-4,4.6772299443974215E-4,5.0370249093747608E-4,4.5755277661716236E-4,4.7184404204873479E-4,5.5338267017813287E-4,4.7706730809866211E-4,5.503104597914453E-4,4.4678367820440553E-4,5.22200208084256E-4,6.3570731995152491E-4,5.5049143585504266E-4,6.26214111552519E-4,6.0009950516270658E-4,4.5743070396722532E-4,4.9889286359151202E-4}>, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 3 : i64, uniform_distributed_segments, compute_shapes = [[32, 32, 3, 3], [32, 32, 3, 3], [32, 32, 3, 3]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], memory_shapes = [[32, 32, 3, 3], [32, 32, 3, 3], [32, 32, 3, 3]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
!alloc_wt_table = !VPUIP.DistributedBuffer<32x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 3 : i64, uniform_distributed_segments, compute_shapes = [[32, 1, 1, 4], [32, 1, 1, 4], [32, 1, 1, 4]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], memory_shapes = [[32, 1, 1, 4], [32, 1, 1, 4], [32, 1, 1, 4]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
!alloc_conv3x3_out = !VPUIP.DistributedBuffer<1x32x800x37x!quant.uniform<u8:f16, 0.0:127>, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments, compute_shapes = [[1, 32, 267, 37], [1, 32, 267, 37], [1, 32, 266, 37]], compute_offsets = [[0, 0, 0, 0], [0, 0, 267, 0], [0, 0, 534, 0]], memory_shapes = [[1, 32, 267, 37], [1, 32, 267, 37], [1, 32, 266, 37]], memory_offsets = [[0, 0, 0, 0], [0, 0, 267, 0], [0, 0, 534, 0]]}>
!alloc_weights_1x1 = !VPUIP.DistributedBuffer<32x32x1x1x!quant.uniform<u8:f16:0, {0.0014105684616986442,0.0012884110796685313,0.001420291264851888,0.0013994499748828364,0.0013700697936263738,0.0012760597116806929,0.0013273022922815061,0.0012557416569952871,0.0013306969521092434,0.0012786817316915475,0.0012744302843131271,0.0012421463050094304,0.0013519255553974825,0.0013760386728772929,0.0013405649101032931,0.0013802739919400683,0.001243662951039333,0.0014184873478085387,0.0012636188198538387,0.0014206820843266506,0.0012609205993951535,0.0013639197630040787,0.0013960076313392789,0.0012737028739031623,0.0012145846497778798,0.0013234123295428706,0.0013260017423068777,0.0013992653173558853,0.0012951428983725753,0.0013398787554572611,0.001301735055212881,0.0012934278039371267}>, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 3 : i64, uniform_distributed_segments, compute_shapes = [[32, 32, 1, 1], [32, 32, 1, 1], [32, 32, 1, 1]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], memory_shapes = [[32, 32, 1, 1], [32, 32, 1, 1], [32, 32, 1, 1]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
!alloc_conv1x1_out = !VPUIP.DistributedBuffer<1x32x800x37x!quant.uniform<u8:f16, 0.0:127>, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments, compute_shapes = [[1, 32, 267, 37], [1, 32, 267, 37], [1, 32, 266, 37]], compute_offsets = [[0, 0, 0, 0], [0, 0, 267, 0], [0, 0, 534, 0]], memory_shapes = [[1, 32, 267, 37], [1, 32, 267, 37], [1, 32, 266, 37]], memory_offsets = [[0, 0, 0, 0], [0, 0, 267, 0], [0, 0, 534, 0]]}>
!alloc_in_tile39 = !VPUIP.DistributedBuffer<1x32x800x39x!quant.uniform<u8:f16, 0.0:127>, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments, compute_shapes = [[1, 32, 267, 39], [1, 32, 267, 39], [1, 32, 266, 39]], compute_offsets = [[0, 0, 0, 0], [0, 0, 267, 0], [0, 0, 534, 0]], memory_shapes = [[1, 32, 268, 39], [1, 32, 269, 39], [1, 32, 267, 39]], memory_offsets = [[0, 0, 0, 0], [0, 0, 266, 0], [0, 0, 533, 0]]}>

// DDR memref aliases
!in_act_ddr = memref<1x32x800x1280x!quant.uniform<u8:f16, 0.0:127>, #NHWC, @DDR>
!out_act_ddr = memref<1x32x800x1280x!quant.uniform<u8:f16, 0.0:127>, #NHWC, @DDR>
!view_in_tile38_ddr = memref<1x32x800x38x!quant.uniform<u8:f16, 0.0:127>, {order = #NHWC, strides = [32768000, 1, 40960, 32]}, @DDR>
!view_in_tile39_ddr = memref<1x32x800x39x!quant.uniform<u8:f16, 0.0:127>, {order = #NHWC, strides = [32768000, 1, 40960, 32]}, @DDR>
!view_out_tile37_ddr = memref<1x32x800x37x!quant.uniform<u8:f16, 0.0:127>, {order = #NHWC, strides = [32768000, 1, 40960, 32]}, @DDR>

// Constant (weights / weight-table) memref aliases
!wt_3x3_memref = memref<32x32x3x3x!quant.uniform<u8:f16:0, {4.9856147345374616E-4,4.8728018414740466E-4,4.9659171525169826E-4,5.3172853647493845E-4,4.6505197590472651E-4,5.9242686804603128E-4,6.6902386207206579E-4,4.5109075658461627E-4,5.2083187243517708E-4,5.5445976117077993E-4,4.6679894713794483E-4,5.233020759096333E-4,4.6531297996932385E-4,5.0246493489134543E-4,6.1671476738125672E-4,4.66597343192381E-4,5.4963564171510589E-4,4.6772299443974215E-4,5.0370249093747608E-4,4.5755277661716236E-4,4.7184404204873479E-4,5.5338267017813287E-4,4.7706730809866211E-4,5.503104597914453E-4,4.4678367820440553E-4,5.22200208084256E-4,6.3570731995152491E-4,5.5049143585504266E-4,6.26214111552519E-4,6.0009950516270658E-4,4.5743070396722532E-4,4.9889286359151202E-4}>, #NHWC>
!wt_1x1_memref = memref<32x32x1x1x!quant.uniform<u8:f16:0, {0.0014105684616986442,0.0012884110796685313,0.001420291264851888,0.0013994499748828364,0.0013700697936263738,0.0012760597116806929,0.0013273022922815061,0.0012557416569952871,0.0013306969521092434,0.0012786817316915475,0.0012744302843131271,0.0012421463050094304,0.0013519255553974825,0.0013760386728772929,0.0013405649101032931,0.0013802739919400683,0.001243662951039333,0.0014184873478085387,0.0012636188198538387,0.0014206820843266506,0.0012609205993951535,0.0013639197630040787,0.0013960076313392789,0.0012737028739031623,0.0012145846497778798,0.0013234123295428706,0.0013260017423068777,0.0013992653173558853,0.0012951428983725753,0.0013398787554572611,0.001301735055212881,0.0012934278039371267}>, #NHWC>
!wt_table_memref = memref<32x1x1x4xsi32>

module @model_name attributes {config.compilationMode = #config.compilation_mode<DefaultHW>, config.platform = #config.platform<NPU5010>} {
func.func @main(%arg0: !in_act_ddr, %arg1: !out_act_ddr) -> !out_act_ddr attributes {pure_vertical_fusion_region} {
  %cst = const.Declare !wt_3x3_memref = dense<1> : tensor<32x32x3x3xui8>, [#const.CastElemType<f16>, #const.CastElemType<!quant.uniform<u8:f16:0, {4.9856147345374616E-4,4.8728018414740466E-4,4.9659171525169826E-4,5.3172853647493845E-4,4.6505197590472651E-4,5.9242686804603128E-4,6.6902386207206579E-4,4.5109075658461627E-4,5.2083187243517708E-4,5.5445976117077993E-4,4.6679894713794483E-4,5.233020759096333E-4,4.6531297996932385E-4,5.0246493489134543E-4,6.1671476738125672E-4,4.66597343192381E-4,5.4963564171510589E-4,4.6772299443974215E-4,5.0370249093747608E-4,4.5755277661716236E-4,4.7184404204873479E-4,5.5338267017813287E-4,4.7706730809866211E-4,5.503104597914453E-4,4.4678367820440553E-4,5.22200208084256E-4,6.3570731995152491E-4,5.5049143585504266E-4,6.26214111552519E-4,6.0009950516270658E-4,4.5743070396722532E-4,4.9889286359151202E-4}>>, #const.Reorder<#NHWC>]
  %cst_0 = const.Declare !wt_1x1_memref = dense<1> : tensor<32x32x1x1xui8>, [#const.CastElemType<f16>, #const.CastElemType<!quant.uniform<u8:f16:0, {0.0014105684616986442,0.0012884110796685313,0.001420291264851888,0.0013994499748828364,0.0013700697936263738,0.0012760597116806929,0.0013273022922815061,0.0012557416569952871,0.0013306969521092434,0.0012786817316915475,0.0012744302843131271,0.0012421463050094304,0.0013519255553974825,0.0013760386728772929,0.0013405649101032931,0.0013802739919400683,0.001243662951039333,0.0014184873478085387,0.0012636188198538387,0.0014206820843266506,0.0012609205993951535,0.0013639197630040787,0.0013960076313392789,0.0012737028739031623,0.0012145846497778798,0.0013234123295428706,0.0013260017423068777,0.0013992653173558853,0.0012951428983725753,0.0013398787554572611,0.001301735055212881,0.0012934278039371267}>>, #const.Reorder<#NHWC>]
  %cst_1 = const.Declare !wt_table_memref = dense<1> : tensor<32x1x1x4xsi32>
  %cst_2 = const.Declare !wt_table_memref = dense<1> : tensor<32x1x1x4xsi32>
  %0 = VPURT.AllocDistributed -> !alloc_in_tile38
  %1 = VPURT.AllocDistributed -> !alloc_weights_3x3
  %2 = VPURT.AllocDistributed -> !alloc_wt_table
  %3 = VPURT.AllocDistributed -> !alloc_conv3x3_out
  %4 = VPURT.AllocDistributed -> !alloc_weights_1x1
  %5 = VPURT.AllocDistributed -> !alloc_wt_table
  %6 = VPURT.AllocDistributed -> !alloc_conv1x1_out
  %7 = VPURT.AllocDistributed -> !alloc_in_tile39
  %8 = VPURT.AllocDistributed -> !alloc_conv3x3_out
  %9 = VPURT.AllocDistributed -> !alloc_conv1x1_out

  %token_3, %bodyResults_4 = async.execute -> !async.value<!alloc_weights_3x3> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 1 : i64, cycleCost = 838 : i64} {
    %145 = VPUIP.NNDMA  inputs(%cst : !wt_3x3_memref) outputs(%1 : !alloc_weights_3x3) -> !alloc_weights_3x3
    async.yield %145 : !alloc_weights_3x3
  }
  %token_5, %bodyResults_6 = async.execute -> !async.value<!alloc_wt_table> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 2 : i64, cycleCost = 593 : i64} {
    %145 = VPUIP.NNDMA  inputs(%cst_1 : !wt_table_memref) outputs(%2 : !alloc_wt_table) -> !alloc_wt_table
    async.yield %145 : !alloc_wt_table
  }
  %token_9, %bodyResults_10 = async.execute -> !async.value<!alloc_weights_1x1> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 4 : i64, cycleCost = 614 : i64} {
    %145 = VPUIP.NNDMA  inputs(%cst_0 : !wt_1x1_memref) outputs(%4 : !alloc_weights_1x1) -> !alloc_weights_1x1
    async.yield %145 : !alloc_weights_1x1
  }
  %token_11, %bodyResults_12 = async.execute -> !async.value<!alloc_wt_table> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 5 : i64, cycleCost = 593 : i64} {
    %145 = VPUIP.NNDMA  inputs(%cst_2 : !wt_table_memref) outputs(%5 : !alloc_wt_table) -> !alloc_wt_table
    async.yield %145 : !alloc_wt_table
  }

  %token, %bodyResults = async.execute -> !async.value<!alloc_in_tile38> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 0 : i64, cycleCost = 14248 : i64} {
    %145 = VPUIP.SubView %arg0 [0, 0, 0, 0] [1, 32, 800, 38] : !in_act_ddr to !view_in_tile38_ddr
    %146 = VPUIP.NNDMA  inputs(%145 : !view_in_tile38_ddr) outputs(%0 : !alloc_in_tile38) -> !alloc_in_tile38
    async.yield %146 : !alloc_in_tile38
  })";

    constexpr llvm::StringLiteral inputIRPart2 = R"(
  %token_7, %bodyResults_8 = async.execute [%token, %token_3, %token_5] (%bodyResults as %arg2: !async.value<!alloc_in_tile38>, %bodyResults_4 as %arg3: !async.value<!alloc_weights_3x3>, %bodyResults_6 as %arg4: !async.value<!alloc_wt_table>) -> !async.value<!alloc_conv3x3_out> attributes {VPUIP.executor = @DPU, "async-deps-index" = 3 : i64, cycleCost = 26917 : i64} {
    %145 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 26917 : i64, vf_loop_index = 0 : i64, vf_loop_tile_index = 0 : i64, vf_tiling_predicted_cost = 1237049 : i64, vf_tiling_strategy = #VPU.vf_scenario<FULL_PREFETCHING>, resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{is_zero_offset_weights_table, kernel_padding = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 1 : i64>, kernel_size = [3, 3], kernel_strides = [1, 1], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<CONV>}> input(%arg2 : !alloc_in_tile38) weights(%arg3 : !alloc_weights_3x3) weight_table(%arg4 : !alloc_wt_table) parent_input(%arg2 : !alloc_in_tile38) parent_output(%3 : !alloc_conv3x3_out) outputs(%3 : !alloc_conv3x3_out) -> !alloc_conv3x3_out variants : {
      DPUTask {cluster_id = 0 : i64, inEnd = [37, 267, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [36, 266, 31], outStart = [0, 0, 0], pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>}
      DPUTask {cluster_id = 1 : i64, inEnd = [37, 268, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [36, 266, 31], outStart = [0, 0, 0], pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      DPUTask {cluster_id = 2 : i64, inEnd = [37, 266, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [36, 265, 31], outStart = [0, 0, 0], pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 0 : i64, bottom = 1 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -9.800000e+01 : f64, clamp_high = 1.570000e+02 : f64, prelu_alpha = [1.000000e+00], adder = 9.800000e+01 : f64>}
    }
    async.yield %145 : !alloc_conv3x3_out
  }
  %token_13, %bodyResults_14 = async.execute [%token_7, %token_9, %token_11] (%bodyResults_8 as %arg2: !async.value<!alloc_conv3x3_out>, %bodyResults_10 as %arg3: !async.value<!alloc_weights_1x1>, %bodyResults_12 as %arg4: !async.value<!alloc_wt_table>) -> !async.value<!alloc_conv1x1_out> attributes {VPUIP.executor = @DPU, "async-deps-index" = 6 : i64, cycleCost = 8016 : i64} {
    %145 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 8016 : i64, vf_loop_index = 0 : i64, vf_loop_tile_index = 0 : i64, vf_tiling_predicted_cost = 1237049 : i64, vf_tiling_strategy = #VPU.vf_scenario<FULL_PREFETCHING>, resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{is_zero_offset_weights_table, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<CONV>}> input(%arg2 : !alloc_conv3x3_out) weights(%arg3 : !alloc_weights_1x1) weight_table(%arg4 : !alloc_wt_table) parent_input(%arg2 : !alloc_conv3x3_out) parent_output(%6 : !alloc_conv1x1_out) outputs(%6 : !alloc_conv1x1_out) -> !alloc_conv1x1_out variants : {
      DPUTask {cluster_id = 0 : i64, inEnd = [36, 266, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [36, 266, 31], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      DPUTask {cluster_id = 1 : i64, inEnd = [36, 266, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [36, 266, 31], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      DPUTask {cluster_id = 2 : i64, inEnd = [36, 265, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [36, 265, 31], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -9.800000e+01 : f64, clamp_high = 1.570000e+02 : f64, prelu_alpha = [1.000000e+00], adder = 9.800000e+01 : f64>}
    }
    async.yield %145 : !alloc_conv1x1_out
  }
  %token_15, %bodyResults_16 = async.execute [%token_13] (%bodyResults_14 as %arg2: !async.value<!alloc_conv1x1_out>) -> !async.value<!view_out_tile37_ddr> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 7 : i64, cycleCost = 13831 : i64} {
    %145 = VPUIP.SubView %arg1 [0, 0, 0, 0] [1, 32, 800, 37] : !out_act_ddr to !view_out_tile37_ddr
    %146 = VPUIP.NNDMA  inputs(%arg2 : !alloc_conv1x1_out) outputs(%145 : !view_out_tile37_ddr) -> !view_out_tile37_ddr
    async.yield %146 : !view_out_tile37_ddr
  }

  %token_17, %bodyResults_18 = async.execute -> !async.value<!alloc_in_tile39> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 8 : i64, cycleCost = 14599 : i64} {
    %145 = VPUIP.SubView %arg0 [0, 0, 0, 36] [1, 32, 800, 39] : !in_act_ddr to !view_in_tile39_ddr
    %146 = VPUIP.NNDMA  inputs(%145 : !view_in_tile39_ddr) outputs(%7 : !alloc_in_tile39) -> !alloc_in_tile39
    async.yield %146 : !alloc_in_tile39
  }
  %token_19, %bodyResults_20 = async.execute [%token_3, %token_5, %token_17] (%bodyResults_4 as %arg2: !async.value<!alloc_weights_3x3>, %bodyResults_6 as %arg3: !async.value<!alloc_wt_table>, %bodyResults_18 as %arg4: !async.value<!alloc_in_tile39>) -> !async.value<!alloc_conv3x3_out> attributes {VPUIP.executor = @DPU, "async-deps-index" = 9 : i64, cycleCost = 26790 : i64} {
    %145 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 26790 : i64, vf_loop_index = 0 : i64, vf_loop_tile_index = 1 : i64, vf_tiling_predicted_cost = 1237049 : i64, vf_tiling_strategy = #VPU.vf_scenario<FULL_PREFETCHING>, resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{is_zero_offset_weights_table, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 1 : i64, bottom = 1 : i64>, kernel_size = [3, 3], kernel_strides = [1, 1], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<CONV>}> input(%arg4 : !alloc_in_tile39) weights(%arg2 : !alloc_weights_3x3) weight_table(%arg3 : !alloc_wt_table) parent_input(%arg4 : !alloc_in_tile39) parent_output(%8 : !alloc_conv3x3_out) outputs(%8 : !alloc_conv3x3_out) -> !alloc_conv3x3_out variants : {
      DPUTask {cluster_id = 0 : i64, inEnd = [38, 267, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [36, 266, 31], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>}
      DPUTask {cluster_id = 1 : i64, inEnd = [38, 268, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [36, 266, 31], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      DPUTask {cluster_id = 2 : i64, inEnd = [38, 266, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [36, 265, 31], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 1 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -9.800000e+01 : f64, clamp_high = 1.570000e+02 : f64, prelu_alpha = [1.000000e+00], adder = 9.800000e+01 : f64>}
    }
    async.yield %145 : !alloc_conv3x3_out
  }
  %token_21, %bodyResults_22 = async.execute [%token_9, %token_11, %token_19] (%bodyResults_10 as %arg2: !async.value<!alloc_weights_1x1>, %bodyResults_12 as %arg3: !async.value<!alloc_wt_table>, %bodyResults_20 as %arg4: !async.value<!alloc_conv3x3_out>) -> !async.value<!alloc_conv1x1_out> attributes {VPUIP.executor = @DPU, "async-deps-index" = 10 : i64, cycleCost = 8016 : i64} {
    %145 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 8016 : i64, vf_loop_index = 0 : i64, vf_loop_tile_index = 1 : i64, vf_tiling_predicted_cost = 1237049 : i64, vf_tiling_strategy = #VPU.vf_scenario<FULL_PREFETCHING>, resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{is_zero_offset_weights_table, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<CONV>}> input(%arg4 : !alloc_conv3x3_out) weights(%arg2 : !alloc_weights_1x1) weight_table(%arg3 : !alloc_wt_table) parent_input(%arg4 : !alloc_conv3x3_out) parent_output(%9 : !alloc_conv1x1_out) outputs(%9 : !alloc_conv1x1_out) -> !alloc_conv1x1_out variants : {
      DPUTask {cluster_id = 0 : i64, inEnd = [36, 266, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [36, 266, 31], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      DPUTask {cluster_id = 1 : i64, inEnd = [36, 266, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [36, 266, 31], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      DPUTask {cluster_id = 2 : i64, inEnd = [36, 265, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [36, 265, 31], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -9.800000e+01 : f64, clamp_high = 1.570000e+02 : f64, prelu_alpha = [1.000000e+00], adder = 9.800000e+01 : f64>}
    }
    async.yield %145 : !alloc_conv1x1_out
  }
  %token_23, %bodyResults_24 = async.execute [%token_21] (%bodyResults_22 as %arg2: !async.value<!alloc_conv1x1_out>) -> !async.value<!view_out_tile37_ddr> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 11 : i64, cycleCost = 13831 : i64} {
    %145 = VPUIP.SubView %arg1 [0, 0, 0, 37] [1, 32, 800, 37] : !out_act_ddr to !view_out_tile37_ddr
    %146 = VPUIP.NNDMA  inputs(%arg2 : !alloc_conv1x1_out) outputs(%145 : !view_out_tile37_ddr) -> !view_out_tile37_ddr
    async.yield %146 : !view_out_tile37_ddr
  }

  %109 = async.await %bodyResults_16 : !async.value<!view_out_tile37_ddr>
  %110 = async.await %bodyResults_24 : !async.value<!view_out_tile37_ddr>
  %144 = VPUIP.ConcatView inputs(%109, %110 : !view_out_tile37_ddr, !view_out_tile37_ddr) outputs(%arg1 : !out_act_ddr) -> !out_act_ddr
  return %144 : !out_act_ddr
}
})";

    const auto platform = config::Platform::NPU5010;
    const auto arch = config::getArch(platform);
    VPU::initializeSingletons(registry, platform);

    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<vpux::VPU::VPUDialect>();

    auto module = mlir::parseSourceString<mlir::ModuleOp>((inputIRPart1 + inputIRPart2).str(), &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    auto log = vpux::Logger::global();

    // Initialize the required components for scheduling
    auto aliasesInfo = AliasesInfoMemType<VPU::MemoryKind::CMX_NN>{func};
    auto liveRangeInfo = MemLiveRangeInfoMemType<VPU::MemoryKind::CMX_NN>{func, aliasesInfo};
    AsyncDepsInfo depsInfo{func};

    const auto memKind = VPU::MemoryKind::CMX_NN;
    const auto secondLvlMemKind = VPU::MemoryKind::DDR;
    uint64_t alignment = vpux::DEFAULT_CMX_ALIGNMENT;
    const auto availableCMXSize = 1473536;
    LinearScan<mlir::Value, LinearScanHandler> scan(availableCMXSize, {}, alignment);

    const auto vpuDevice = VPUNN::VPUDevice::NPU_5_0;
    auto costModel = VPU::CostModelConfig::createCostModel(&ctx);
    const int64_t nceClusterCount = 3;
    const int64_t dmaCount = 2;
    const bool enableScheduleStatistics = false;
    const bool optimizeFragmentation = false;
    const bool activelySpillForPrefetching = false;

    // Create FeasibleMemoryScheduler with empty loop regions for testing
    ComputeRegionsSchedule emptySchedule;
    ComputeRegionVec emptyRegions;
    FeasibleMemoryScheduler noLoopRegionScheduler(
            memKind, secondLvlMemKind, liveRangeInfo, depsInfo, log, scan, arch, vpuDevice, costModel, nceClusterCount,
            dmaCount, enableScheduleStatistics, optimizeFragmentation, activelySpillForPrefetching,
            std::move(emptySchedule), std::move(emptyRegions));

    // Create test fixture to access private members
    FeasibleMemorySchedulerTest testAccessor(noLoopRegionScheduler);

    // Loop region should be empty when no loopRegions is provided for scheduler
    EXPECT_EQ(testAccessor.getLoopRegionSize(), 0);
    // Scheduled loop regions should be zero as scheduling is not performed here
    EXPECT_EQ(testAccessor.getScheduledLoopRegionSize(), 0);

    // Create loop regions from tiling
    // And provide the loop for scheduler
    depsInfo = AsyncDepsInfo{func};
    auto computeRegionVec = getComputeRegionsFromAsyncExec(aliasesInfo, depsInfo, log);
    // Include 4 LoopType::None regions and 1 VF loop region with 2 loop bodies
    EXPECT_EQ(computeRegionVec.size(), 5);
    for (auto& region : computeRegionVec) {
        if (region.getLoopType() != LoopType::None) {
            EXPECT_EQ(region.schedulingLoop->loopBodies.size(), 2);
        }
    }

    auto computeRegionsSchedule =
            VPU::generateLoopSchedules(computeRegionVec, availableCMXSize, /*enableVfUndefinedScheduler=*/true, log);
    FeasibleMemoryScheduler vfLoopRegionScheduler(
            memKind, secondLvlMemKind, liveRangeInfo, depsInfo, log, scan, arch, vpuDevice, costModel, nceClusterCount,
            dmaCount, enableScheduleStatistics, optimizeFragmentation, activelySpillForPrefetching,
            std::move(computeRegionsSchedule), std::move(computeRegionVec));
    FeasibleMemorySchedulerTest testAccessor2(vfLoopRegionScheduler);
    EXPECT_EQ(testAccessor2.getLoopRegionSize(), 5);
    vfLoopRegionScheduler.generateSchedule();
    // Scheduled loop regions should not be zero as scheduler has loop region
    EXPECT_EQ(testAccessor2.getScheduledLoopRegionSize(), 1);
    EXPECT_TRUE(testAccessor2.verifyLoopInputAddress());
}

// Test if scheduleComputeOps() skips scheduling a loop-owned compute DMAs when it is already
// unblocked and in _readyDMAOps, so that scheduleLoopRegions() can still emit it as a LOOP_OP later.
// To exercise this scenario this test:
//   1. Runs only doInitSetup() so scheduler state is populated with no scheduling.
//   2. Marks the iteration-0 CMX2CMX DMA's DATA_IN dep as "already scheduled"
//      (simulating a DMA queue having picked it up).
//   3. Directly places the loop-owned CMX2CMX into _readyDMAOps (simulating the
//      race window where scheduleComputeOps() is invoked with a loop-owned DMA
//      already unblocked).
//   4. Calls scheduleComputeOps() in isolation and asserts that the guarded DMA
//      is NOT pushed into _cycleBeginHeap and remains in _readyDMAOps so that
//      scheduleLoopRegions() can still emit it as a LOOP_OP later.
//
// If the guard at scheduleComputeOps() is removed, this test fails: the CMX2CMX
// gets pushed into _cycleBeginHeap as an ORIGINAL_OP.
TEST_F(MLIR_FeasibleMemorySchedulerLoop, ScheduleComputeOpsSkipsCmx2CmxInLoopRegion) {
    // IR structure (2-iteration VF loop, each iteration starts with a CMX2CMX DMA):
    //   Shared (outside loop iteration):
    //     OP - weights_3x3 DMA (DDR->CMX)
    //     OP - wt_table_3x3 DMA (DDR->CMX)
    //     OP - weights_1x1 DMA (DDR->CMX)
    //     OP - wt_table_1x1 DMA (DDR->CMX)
    //   VF Loop Region 0, iteration 0 and 1 (in order):
    //     OP - DATA_IN (DDR->CMX input tile)
    //     OP - CMX2CMX DMA (CMX->CMX copy of input tile)  <-- exercises the fix
    //     OP - COMPUTE_OP (conv3x3 consuming CMX2CMX output)
    //     OP - COMPUTE_OP (conv1x1)
    //     OP - DATA_OUT (CMX->DDR)

    constexpr llvm::StringLiteral inputIRPart1 = R"(

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!alloc_in_tile38 = !VPUIP.DistributedBuffer<1x32x800x38x!quant.uniform<u8:f16, 0.0:127>, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments, compute_shapes = [[1, 32, 267, 38], [1, 32, 267, 38], [1, 32, 266, 38]], compute_offsets = [[0, 0, 0, 0], [0, 0, 267, 0], [0, 0, 534, 0]], memory_shapes = [[1, 32, 268, 38], [1, 32, 269, 38], [1, 32, 267, 38]], memory_offsets = [[0, 0, 0, 0], [0, 0, 266, 0], [0, 0, 533, 0]]}>
!alloc_weights_3x3 = !VPUIP.DistributedBuffer<32x32x3x3x!quant.uniform<u8:f16:0, {4.9856147345374616E-4,4.8728018414740466E-4,4.9659171525169826E-4,5.3172853647493845E-4,4.6505197590472651E-4,5.9242686804603128E-4,6.6902386207206579E-4,4.5109075658461627E-4,5.2083187243517708E-4,5.5445976117077993E-4,4.6679894713794483E-4,5.233020759096333E-4,4.6531297996932385E-4,5.0246493489134543E-4,6.1671476738125672E-4,4.66597343192381E-4,5.4963564171510589E-4,4.6772299443974215E-4,5.0370249093747608E-4,4.5755277661716236E-4,4.7184404204873479E-4,5.5338267017813287E-4,4.7706730809866211E-4,5.503104597914453E-4,4.4678367820440553E-4,5.22200208084256E-4,6.3570731995152491E-4,5.5049143585504266E-4,6.26214111552519E-4,6.0009950516270658E-4,4.5743070396722532E-4,4.9889286359151202E-4}>, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 3 : i64, uniform_distributed_segments, compute_shapes = [[32, 32, 3, 3], [32, 32, 3, 3], [32, 32, 3, 3]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], memory_shapes = [[32, 32, 3, 3], [32, 32, 3, 3], [32, 32, 3, 3]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
!alloc_wt_table = !VPUIP.DistributedBuffer<32x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 3 : i64, uniform_distributed_segments, compute_shapes = [[32, 1, 1, 4], [32, 1, 1, 4], [32, 1, 1, 4]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], memory_shapes = [[32, 1, 1, 4], [32, 1, 1, 4], [32, 1, 1, 4]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
!alloc_conv3x3_out = !VPUIP.DistributedBuffer<1x32x800x37x!quant.uniform<u8:f16, 0.0:127>, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments, compute_shapes = [[1, 32, 267, 37], [1, 32, 267, 37], [1, 32, 266, 37]], compute_offsets = [[0, 0, 0, 0], [0, 0, 267, 0], [0, 0, 534, 0]], memory_shapes = [[1, 32, 267, 37], [1, 32, 267, 37], [1, 32, 266, 37]], memory_offsets = [[0, 0, 0, 0], [0, 0, 267, 0], [0, 0, 534, 0]]}>
!alloc_weights_1x1 = !VPUIP.DistributedBuffer<32x32x1x1x!quant.uniform<u8:f16:0, {0.0014105684616986442,0.0012884110796685313,0.001420291264851888,0.0013994499748828364,0.0013700697936263738,0.0012760597116806929,0.0013273022922815061,0.0012557416569952871,0.0013306969521092434,0.0012786817316915475,0.0012744302843131271,0.0012421463050094304,0.0013519255553974825,0.0013760386728772929,0.0013405649101032931,0.0013802739919400683,0.001243662951039333,0.0014184873478085387,0.0012636188198538387,0.0014206820843266506,0.0012609205993951535,0.0013639197630040787,0.0013960076313392789,0.0012737028739031623,0.0012145846497778798,0.0013234123295428706,0.0013260017423068777,0.0013992653173558853,0.0012951428983725753,0.0013398787554572611,0.001301735055212881,0.0012934278039371267}>, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 3 : i64, uniform_distributed_segments, compute_shapes = [[32, 32, 1, 1], [32, 32, 1, 1], [32, 32, 1, 1]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], memory_shapes = [[32, 32, 1, 1], [32, 32, 1, 1], [32, 32, 1, 1]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
!alloc_conv1x1_out = !VPUIP.DistributedBuffer<1x32x800x37x!quant.uniform<u8:f16, 0.0:127>, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments, compute_shapes = [[1, 32, 267, 37], [1, 32, 267, 37], [1, 32, 266, 37]], compute_offsets = [[0, 0, 0, 0], [0, 0, 267, 0], [0, 0, 534, 0]], memory_shapes = [[1, 32, 267, 37], [1, 32, 267, 37], [1, 32, 266, 37]], memory_offsets = [[0, 0, 0, 0], [0, 0, 267, 0], [0, 0, 534, 0]]}>
!alloc_in_tile39 = !VPUIP.DistributedBuffer<1x32x800x39x!quant.uniform<u8:f16, 0.0:127>, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments, compute_shapes = [[1, 32, 267, 39], [1, 32, 267, 39], [1, 32, 266, 39]], compute_offsets = [[0, 0, 0, 0], [0, 0, 267, 0], [0, 0, 534, 0]], memory_shapes = [[1, 32, 268, 39], [1, 32, 269, 39], [1, 32, 267, 39]], memory_offsets = [[0, 0, 0, 0], [0, 0, 266, 0], [0, 0, 533, 0]]}>

// DDR memref aliases
!in_act_ddr = memref<1x32x800x1280x!quant.uniform<u8:f16, 0.0:127>, #NHWC, @DDR>
!out_act_ddr = memref<1x32x800x1280x!quant.uniform<u8:f16, 0.0:127>, #NHWC, @DDR>
!view_in_tile38_ddr = memref<1x32x800x38x!quant.uniform<u8:f16, 0.0:127>, {order = #NHWC, strides = [32768000, 1, 40960, 32]}, @DDR>
!view_in_tile39_ddr = memref<1x32x800x39x!quant.uniform<u8:f16, 0.0:127>, {order = #NHWC, strides = [32768000, 1, 40960, 32]}, @DDR>
!view_out_tile37_ddr = memref<1x32x800x37x!quant.uniform<u8:f16, 0.0:127>, {order = #NHWC, strides = [32768000, 1, 40960, 32]}, @DDR>

// Constant (weights / weight-table) memref aliases
!wt_3x3_memref = memref<32x32x3x3x!quant.uniform<u8:f16:0, {4.9856147345374616E-4,4.8728018414740466E-4,4.9659171525169826E-4,5.3172853647493845E-4,4.6505197590472651E-4,5.9242686804603128E-4,6.6902386207206579E-4,4.5109075658461627E-4,5.2083187243517708E-4,5.5445976117077993E-4,4.6679894713794483E-4,5.233020759096333E-4,4.6531297996932385E-4,5.0246493489134543E-4,6.1671476738125672E-4,4.66597343192381E-4,5.4963564171510589E-4,4.6772299443974215E-4,5.0370249093747608E-4,4.5755277661716236E-4,4.7184404204873479E-4,5.5338267017813287E-4,4.7706730809866211E-4,5.503104597914453E-4,4.4678367820440553E-4,5.22200208084256E-4,6.3570731995152491E-4,5.5049143585504266E-4,6.26214111552519E-4,6.0009950516270658E-4,4.5743070396722532E-4,4.9889286359151202E-4}>, #NHWC>
!wt_1x1_memref = memref<32x32x1x1x!quant.uniform<u8:f16:0, {0.0014105684616986442,0.0012884110796685313,0.001420291264851888,0.0013994499748828364,0.0013700697936263738,0.0012760597116806929,0.0013273022922815061,0.0012557416569952871,0.0013306969521092434,0.0012786817316915475,0.0012744302843131271,0.0012421463050094304,0.0013519255553974825,0.0013760386728772929,0.0013405649101032931,0.0013802739919400683,0.001243662951039333,0.0014184873478085387,0.0012636188198538387,0.0014206820843266506,0.0012609205993951535,0.0013639197630040787,0.0013960076313392789,0.0012737028739031623,0.0012145846497778798,0.0013234123295428706,0.0013260017423068777,0.0013992653173558853,0.0012951428983725753,0.0013398787554572611,0.001301735055212881,0.0012934278039371267}>, #NHWC>
!wt_table_memref = memref<32x1x1x4xsi32>

module @model_name attributes {config.compilationMode = #config.compilation_mode<DefaultHW>, config.platform = #config.platform<NPU5010>} {
func.func @main(%arg0: !in_act_ddr, %arg1: !out_act_ddr) -> !out_act_ddr attributes {pure_vertical_fusion_region} {
  %cst = const.Declare !wt_3x3_memref = dense<1> : tensor<32x32x3x3xui8>, [#const.CastElemType<f16>, #const.CastElemType<!quant.uniform<u8:f16:0, {4.9856147345374616E-4,4.8728018414740466E-4,4.9659171525169826E-4,5.3172853647493845E-4,4.6505197590472651E-4,5.9242686804603128E-4,6.6902386207206579E-4,4.5109075658461627E-4,5.2083187243517708E-4,5.5445976117077993E-4,4.6679894713794483E-4,5.233020759096333E-4,4.6531297996932385E-4,5.0246493489134543E-4,6.1671476738125672E-4,4.66597343192381E-4,5.4963564171510589E-4,4.6772299443974215E-4,5.0370249093747608E-4,4.5755277661716236E-4,4.7184404204873479E-4,5.5338267017813287E-4,4.7706730809866211E-4,5.503104597914453E-4,4.4678367820440553E-4,5.22200208084256E-4,6.3570731995152491E-4,5.5049143585504266E-4,6.26214111552519E-4,6.0009950516270658E-4,4.5743070396722532E-4,4.9889286359151202E-4}>>, #const.Reorder<#NHWC>]
  %cst_0 = const.Declare !wt_1x1_memref = dense<1> : tensor<32x32x1x1xui8>, [#const.CastElemType<f16>, #const.CastElemType<!quant.uniform<u8:f16:0, {0.0014105684616986442,0.0012884110796685313,0.001420291264851888,0.0013994499748828364,0.0013700697936263738,0.0012760597116806929,0.0013273022922815061,0.0012557416569952871,0.0013306969521092434,0.0012786817316915475,0.0012744302843131271,0.0012421463050094304,0.0013519255553974825,0.0013760386728772929,0.0013405649101032931,0.0013802739919400683,0.001243662951039333,0.0014184873478085387,0.0012636188198538387,0.0014206820843266506,0.0012609205993951535,0.0013639197630040787,0.0013960076313392789,0.0012737028739031623,0.0012145846497778798,0.0013234123295428706,0.0013260017423068777,0.0013992653173558853,0.0012951428983725753,0.0013398787554572611,0.001301735055212881,0.0012934278039371267}>>, #const.Reorder<#NHWC>]
  %cst_1 = const.Declare !wt_table_memref = dense<1> : tensor<32x1x1x4xsi32>
  %cst_2 = const.Declare !wt_table_memref = dense<1> : tensor<32x1x1x4xsi32>
  %0 = VPURT.AllocDistributed -> !alloc_in_tile38
  %1 = VPURT.AllocDistributed -> !alloc_weights_3x3
  %2 = VPURT.AllocDistributed -> !alloc_wt_table
  %3 = VPURT.AllocDistributed -> !alloc_conv3x3_out
  %4 = VPURT.AllocDistributed -> !alloc_weights_1x1
  %5 = VPURT.AllocDistributed -> !alloc_wt_table
  %6 = VPURT.AllocDistributed -> !alloc_conv1x1_out
  %7 = VPURT.AllocDistributed -> !alloc_in_tile39
  %8 = VPURT.AllocDistributed -> !alloc_conv3x3_out
  %9 = VPURT.AllocDistributed -> !alloc_conv1x1_out
  // Extra CMX destinations for the CMX->CMX DMAs at the start of each VF iteration
  %10 = VPURT.AllocDistributed -> !alloc_in_tile38
  %11 = VPURT.AllocDistributed -> !alloc_in_tile39

  %token_3, %bodyResults_4 = async.execute -> !async.value<!alloc_weights_3x3> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 1 : i64, cycleCost = 838 : i64} {
    %145 = VPUIP.NNDMA  inputs(%cst : !wt_3x3_memref) outputs(%1 : !alloc_weights_3x3) -> !alloc_weights_3x3
    async.yield %145 : !alloc_weights_3x3
  }
  %token_5, %bodyResults_6 = async.execute -> !async.value<!alloc_wt_table> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 2 : i64, cycleCost = 593 : i64} {
    %145 = VPUIP.NNDMA  inputs(%cst_1 : !wt_table_memref) outputs(%2 : !alloc_wt_table) -> !alloc_wt_table
    async.yield %145 : !alloc_wt_table
  }
  %token_9, %bodyResults_10 = async.execute -> !async.value<!alloc_weights_1x1> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 4 : i64, cycleCost = 614 : i64} {
    %145 = VPUIP.NNDMA  inputs(%cst_0 : !wt_1x1_memref) outputs(%4 : !alloc_weights_1x1) -> !alloc_weights_1x1
    async.yield %145 : !alloc_weights_1x1
  }
  %token_11, %bodyResults_12 = async.execute -> !async.value<!alloc_wt_table> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 5 : i64, cycleCost = 593 : i64} {
    %145 = VPUIP.NNDMA  inputs(%cst_2 : !wt_table_memref) outputs(%5 : !alloc_wt_table) -> !alloc_wt_table
    async.yield %145 : !alloc_wt_table
  }

  // ---- VF iteration 0 ----
  // DATA_IN: DDR -> CMX (input tile 38)
  %token, %bodyResults = async.execute -> !async.value<!alloc_in_tile38> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 0 : i64, cycleCost = 14248 : i64} {
    %145 = VPUIP.SubView %arg0 [0, 0, 0, 0] [1, 32, 800, 38] : !in_act_ddr to !view_in_tile38_ddr
    %146 = VPUIP.NNDMA  inputs(%145 : !view_in_tile38_ddr) outputs(%0 : !alloc_in_tile38) -> !alloc_in_tile38
    async.yield %146 : !alloc_in_tile38
  }
  // CMX2CMX DMA: copies DATA_IN result to another CMX buffer that feeds conv3x3.
  // Exercises the fix: this DMA belongs to loopRegionInd (as a VF compute-loop op)
  // and must NOT be scheduled by scheduleComputeOps() outside of scheduleLoopRegions().
  %token_cmx0, %bodyResults_cmx0 = async.execute [%token] (%bodyResults as %arg2: !async.value<!alloc_in_tile38>) -> !async.value<!alloc_in_tile38> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 12 : i64, cycleCost = 5000 : i64} {
    %145 = VPUIP.NNDMA  inputs(%arg2 : !alloc_in_tile38) outputs(%10 : !alloc_in_tile38) -> !alloc_in_tile38
    async.yield %145 : !alloc_in_tile38
  })";

    constexpr llvm::StringLiteral inputIRPart2 = R"(
  %token_7, %bodyResults_8 = async.execute [%token_cmx0, %token_3, %token_5] (%bodyResults_cmx0 as %arg2: !async.value<!alloc_in_tile38>, %bodyResults_4 as %arg3: !async.value<!alloc_weights_3x3>, %bodyResults_6 as %arg4: !async.value<!alloc_wt_table>) -> !async.value<!alloc_conv3x3_out> attributes {VPUIP.executor = @DPU, "async-deps-index" = 3 : i64, cycleCost = 26917 : i64} {
    %145 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 26917 : i64, vf_loop_index = 0 : i64, vf_loop_tile_index = 0 : i64, vf_tiling_predicted_cost = 1237049 : i64, vf_tiling_strategy = #VPU.vf_scenario<FULL_PREFETCHING>, resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{is_zero_offset_weights_table, kernel_padding = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 1 : i64>, kernel_size = [3, 3], kernel_strides = [1, 1], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<CONV>}> input(%arg2 : !alloc_in_tile38) weights(%arg3 : !alloc_weights_3x3) weight_table(%arg4 : !alloc_wt_table) parent_input(%arg2 : !alloc_in_tile38) parent_output(%3 : !alloc_conv3x3_out) outputs(%3 : !alloc_conv3x3_out) -> !alloc_conv3x3_out variants : {
      DPUTask {cluster_id = 0 : i64, inEnd = [37, 267, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [36, 266, 31], outStart = [0, 0, 0], pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>}
      DPUTask {cluster_id = 1 : i64, inEnd = [37, 268, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [36, 266, 31], outStart = [0, 0, 0], pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      DPUTask {cluster_id = 2 : i64, inEnd = [37, 266, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [36, 265, 31], outStart = [0, 0, 0], pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 0 : i64, bottom = 1 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -9.800000e+01 : f64, clamp_high = 1.570000e+02 : f64, prelu_alpha = [1.000000e+00], adder = 9.800000e+01 : f64>}
    }
    async.yield %145 : !alloc_conv3x3_out
  }
  %token_13, %bodyResults_14 = async.execute [%token_7, %token_9, %token_11] (%bodyResults_8 as %arg2: !async.value<!alloc_conv3x3_out>, %bodyResults_10 as %arg3: !async.value<!alloc_weights_1x1>, %bodyResults_12 as %arg4: !async.value<!alloc_wt_table>) -> !async.value<!alloc_conv1x1_out> attributes {VPUIP.executor = @DPU, "async-deps-index" = 6 : i64, cycleCost = 8016 : i64} {
    %145 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 8016 : i64, vf_loop_index = 0 : i64, vf_loop_tile_index = 0 : i64, vf_tiling_predicted_cost = 1237049 : i64, vf_tiling_strategy = #VPU.vf_scenario<FULL_PREFETCHING>, resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{is_zero_offset_weights_table, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<CONV>}> input(%arg2 : !alloc_conv3x3_out) weights(%arg3 : !alloc_weights_1x1) weight_table(%arg4 : !alloc_wt_table) parent_input(%arg2 : !alloc_conv3x3_out) parent_output(%6 : !alloc_conv1x1_out) outputs(%6 : !alloc_conv1x1_out) -> !alloc_conv1x1_out variants : {
      DPUTask {cluster_id = 0 : i64, inEnd = [36, 266, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [36, 266, 31], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      DPUTask {cluster_id = 1 : i64, inEnd = [36, 266, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [36, 266, 31], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      DPUTask {cluster_id = 2 : i64, inEnd = [36, 265, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [36, 265, 31], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -9.800000e+01 : f64, clamp_high = 1.570000e+02 : f64, prelu_alpha = [1.000000e+00], adder = 9.800000e+01 : f64>}
    }
    async.yield %145 : !alloc_conv1x1_out
  }
  %token_15, %bodyResults_16 = async.execute [%token_13] (%bodyResults_14 as %arg2: !async.value<!alloc_conv1x1_out>) -> !async.value<!view_out_tile37_ddr> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 7 : i64, cycleCost = 13831 : i64} {
    %145 = VPUIP.SubView %arg1 [0, 0, 0, 0] [1, 32, 800, 37] : !out_act_ddr to !view_out_tile37_ddr
    %146 = VPUIP.NNDMA  inputs(%arg2 : !alloc_conv1x1_out) outputs(%145 : !view_out_tile37_ddr) -> !view_out_tile37_ddr
    async.yield %146 : !view_out_tile37_ddr
  }

  // ---- VF iteration 1 ----
  // DATA_IN: DDR -> CMX (input tile 39)
  %token_17, %bodyResults_18 = async.execute -> !async.value<!alloc_in_tile39> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 8 : i64, cycleCost = 14599 : i64} {
    %145 = VPUIP.SubView %arg0 [0, 0, 0, 36] [1, 32, 800, 39] : !in_act_ddr to !view_in_tile39_ddr
    %146 = VPUIP.NNDMA  inputs(%145 : !view_in_tile39_ddr) outputs(%7 : !alloc_in_tile39) -> !alloc_in_tile39
    async.yield %146 : !alloc_in_tile39
  }
  // CMX2CMX DMA for iteration 1 (same purpose as the iteration-0 CMX2CMX DMA above).
  %token_cmx1, %bodyResults_cmx1 = async.execute [%token_17] (%bodyResults_18 as %arg2: !async.value<!alloc_in_tile39>) -> !async.value<!alloc_in_tile39> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 13 : i64, cycleCost = 5000 : i64} {
    %145 = VPUIP.NNDMA  inputs(%arg2 : !alloc_in_tile39) outputs(%11 : !alloc_in_tile39) -> !alloc_in_tile39
    async.yield %145 : !alloc_in_tile39
  }
  %token_19, %bodyResults_20 = async.execute [%token_3, %token_5, %token_cmx1] (%bodyResults_4 as %arg2: !async.value<!alloc_weights_3x3>, %bodyResults_6 as %arg3: !async.value<!alloc_wt_table>, %bodyResults_cmx1 as %arg4: !async.value<!alloc_in_tile39>) -> !async.value<!alloc_conv3x3_out> attributes {VPUIP.executor = @DPU, "async-deps-index" = 9 : i64, cycleCost = 26790 : i64} {
    %145 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 26790 : i64, vf_loop_index = 0 : i64, vf_loop_tile_index = 1 : i64, vf_tiling_predicted_cost = 1237049 : i64, vf_tiling_strategy = #VPU.vf_scenario<FULL_PREFETCHING>, resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{is_zero_offset_weights_table, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 1 : i64, bottom = 1 : i64>, kernel_size = [3, 3], kernel_strides = [1, 1], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<CONV>}> input(%arg4 : !alloc_in_tile39) weights(%arg2 : !alloc_weights_3x3) weight_table(%arg3 : !alloc_wt_table) parent_input(%arg4 : !alloc_in_tile39) parent_output(%8 : !alloc_conv3x3_out) outputs(%8 : !alloc_conv3x3_out) -> !alloc_conv3x3_out variants : {
      DPUTask {cluster_id = 0 : i64, inEnd = [38, 267, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [36, 266, 31], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>}
      DPUTask {cluster_id = 1 : i64, inEnd = [38, 268, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [36, 266, 31], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      DPUTask {cluster_id = 2 : i64, inEnd = [38, 266, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [36, 265, 31], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 1 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -9.800000e+01 : f64, clamp_high = 1.570000e+02 : f64, prelu_alpha = [1.000000e+00], adder = 9.800000e+01 : f64>}
    }
    async.yield %145 : !alloc_conv3x3_out
  }
  %token_21, %bodyResults_22 = async.execute [%token_9, %token_11, %token_19] (%bodyResults_10 as %arg2: !async.value<!alloc_weights_1x1>, %bodyResults_12 as %arg3: !async.value<!alloc_wt_table>, %bodyResults_20 as %arg4: !async.value<!alloc_conv3x3_out>) -> !async.value<!alloc_conv1x1_out> attributes {VPUIP.executor = @DPU, "async-deps-index" = 10 : i64, cycleCost = 8016 : i64} {
    %145 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 8016 : i64, vf_loop_index = 0 : i64, vf_loop_tile_index = 1 : i64, vf_tiling_predicted_cost = 1237049 : i64, vf_tiling_strategy = #VPU.vf_scenario<FULL_PREFETCHING>, resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{is_zero_offset_weights_table, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<CONV>}> input(%arg4 : !alloc_conv3x3_out) weights(%arg2 : !alloc_weights_1x1) weight_table(%arg3 : !alloc_wt_table) parent_input(%arg4 : !alloc_conv3x3_out) parent_output(%9 : !alloc_conv1x1_out) outputs(%9 : !alloc_conv1x1_out) -> !alloc_conv1x1_out variants : {
      DPUTask {cluster_id = 0 : i64, inEnd = [36, 266, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [36, 266, 31], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      DPUTask {cluster_id = 1 : i64, inEnd = [36, 266, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [36, 266, 31], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      DPUTask {cluster_id = 2 : i64, inEnd = [36, 265, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [36, 265, 31], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -9.800000e+01 : f64, clamp_high = 1.570000e+02 : f64, prelu_alpha = [1.000000e+00], adder = 9.800000e+01 : f64>}
    }
    async.yield %145 : !alloc_conv1x1_out
  }
  %token_23, %bodyResults_24 = async.execute [%token_21] (%bodyResults_22 as %arg2: !async.value<!alloc_conv1x1_out>) -> !async.value<!view_out_tile37_ddr> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 11 : i64, cycleCost = 13831 : i64} {
    %145 = VPUIP.SubView %arg1 [0, 0, 0, 37] [1, 32, 800, 37] : !out_act_ddr to !view_out_tile37_ddr
    %146 = VPUIP.NNDMA  inputs(%arg2 : !alloc_conv1x1_out) outputs(%145 : !view_out_tile37_ddr) -> !view_out_tile37_ddr
    async.yield %146 : !view_out_tile37_ddr
  }

  %109 = async.await %bodyResults_16 : !async.value<!view_out_tile37_ddr>
  %110 = async.await %bodyResults_24 : !async.value<!view_out_tile37_ddr>
  %144 = VPUIP.ConcatView inputs(%109, %110 : !view_out_tile37_ddr, !view_out_tile37_ddr) outputs(%arg1 : !out_act_ddr) -> !out_act_ddr
  return %144 : !out_act_ddr
}
})";

    const auto platform = config::Platform::NPU5010;
    const auto arch = config::getArch(platform);
    VPU::initializeSingletons(registry, platform);

    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<vpux::VPU::VPUDialect>();

    auto module = mlir::parseSourceString<mlir::ModuleOp>((inputIRPart1 + inputIRPart2).str(), &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    auto log = vpux::Logger::global();

    auto aliasesInfo = AliasesInfoMemType<VPU::MemoryKind::CMX_NN>{func};
    auto liveRangeInfo = MemLiveRangeInfoMemType<VPU::MemoryKind::CMX_NN>{func, aliasesInfo};
    AsyncDepsInfo depsInfo{func};

    const auto memKind = VPU::MemoryKind::CMX_NN;
    const auto secondLvlMemKind = VPU::MemoryKind::DDR;
    uint64_t alignment = vpux::DEFAULT_CMX_ALIGNMENT;
    const auto availableCMXSize = 1473536;
    LinearScan<mlir::Value, LinearScanHandler> scan(availableCMXSize, {}, alignment);

    const auto vpuDevice = VPUNN::VPUDevice::NPU_5_0;
    auto costModel = VPU::CostModelConfig::createCostModel(&ctx);
    const int64_t nceClusterCount = 3;
    const int64_t dmaCount = 2;
    const bool enableScheduleStatistics = false;
    const bool optimizeFragmentation = false;
    const bool activelySpillForPrefetching = false;

    auto computeRegionVec = getComputeRegionsFromAsyncExec(aliasesInfo, depsInfo, log);
    auto computeRegionsSchedule =
            VPU::generateLoopSchedules(computeRegionVec, availableCMXSize, /*enableVfUndefinedScheduler=*/true, log);

    // Locate the two CMX2CMX DMAs. Both must be in loopRegionInd for this test to be meaningful.
    llvm::SmallVector<size_t, 2> cmx2cmxOpIndices;
    for (size_t opIdx = 0; opIdx < depsInfo.getExecOpCount(); ++opIdx) {
        auto execOp = depsInfo.getExecuteOpAtIndex(opIdx);
        if (VPUIP::isDmaCMX2CMX(execOp)) {
            cmx2cmxOpIndices.push_back(opIdx);
        }
    }
    ASSERT_EQ(cmx2cmxOpIndices.size(), 2u) << "Expected two CMX2CMX DMAs (one per VF iteration)";

    // Exercise iteration-0's CMX2CMX. The choice of iteration is arbitrary; both are
    // in loopRegionInd and both are subject to the same guard.
    const auto cmx2cmxOpIdx = cmx2cmxOpIndices[0];
    ASSERT_TRUE(computeRegionsSchedule.loopRegionInd.count(cmx2cmxOpIdx))
            << "Precondition: iteration-0 CMX2CMX (op idx " << cmx2cmxOpIdx
            << ") must be classified as a compute-loop op";

    // Find its DATA_IN dep (the DDR->CMX DMA producing the input tile that the CMX2CMX copies).
    const auto cmx2cmxDeps = depsInfo.getOpDeps(cmx2cmxOpIdx);
    ASSERT_EQ(cmx2cmxDeps.size(), 1u) << "Precondition: iteration-0 CMX2CMX must have exactly one dep (its DATA_IN)";
    const auto dataInOpIdx = cmx2cmxDeps[0];

    FeasibleMemoryScheduler scheduler(memKind, secondLvlMemKind, liveRangeInfo, depsInfo, log, scan, arch, vpuDevice,
                                      costModel, nceClusterCount, dmaCount, enableScheduleStatistics,
                                      optimizeFragmentation, activelySpillForPrefetching,
                                      std::move(computeRegionsSchedule), std::move(computeRegionVec));
    FeasibleMemorySchedulerTest testAccessor(scheduler);

    // Populate scheduler internal state without running any scheduling activity.
    testAccessor.runInitSetup();

    // Simulate the exact race window the fix protects against:
    //   - DATA_IN has already been scheduled (its cycle-end is recorded, no longer in _readyDataOps).
    //   - The CMX2CMX DMA is now unblocked and sitting in _readyDMAOps.
    // Without the fix, scheduleComputeOps() iterates _readyDMAOps, allocates the CMX2CMX
    // buffers and pushes an ORIGINAL_OP entry to _cycleBeginHeap.
    testAccessor.markOpAsScheduled(dataInOpIdx, /*cycleEnd=*/1000);
    testAccessor.removeFromReadyDataOps(dataInOpIdx);
    testAccessor.injectReadyDMAOp(cmx2cmxOpIdx);

    ASSERT_TRUE(testAccessor.isInLoopRegionInd(cmx2cmxOpIdx))
            << "Precondition: loop-owned CMX2CMX must be in loopRegionInd so the fix's guard applies";
    ASSERT_TRUE(testAccessor.isInReadyDMAOps(cmx2cmxOpIdx))
            << "Precondition: loop-owned CMX2CMX must be sitting in _readyDMAOps for scheduleComputeOps() to see it";

    // Invoke the code path under test in isolation.
    testAccessor.runScheduleComputeOps();

    // Fix's post-condition: the loop-owned CMX2CMX must NOT have been pushed to
    // _cycleBeginHeap. If the guard is removed, scheduleComputeOps() would emit an
    // ORIGINAL_OP entry for it here.
    EXPECT_EQ(testAccessor.countInCycleBeginHeap(cmx2cmxOpIdx), 0u)
            << "Loop-owned CMX2CMX DMA (op idx " << cmx2cmxOpIdx
            << ") was scheduled by scheduleComputeOps() outside of scheduleLoopRegions(); "
               "the loopRegionInd skip check in scheduleComputeOps() is missing or broken";

    // The DMA must remain in _readyDMAOps so scheduleLoopRegions() can still emit it as LOOP_OP.
    EXPECT_TRUE(testAccessor.isInReadyDMAOps(cmx2cmxOpIdx))
            << "Loop-owned CMX2CMX DMA at op idx " << cmx2cmxOpIdx
            << " was removed from _readyDMAOps by scheduleComputeOps(); it must remain there so "
               "scheduleLoopRegions() can still schedule it as a LOOP_OP";
}
