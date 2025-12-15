//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "common/utils.hpp"
#include "vpux/utils/logger/logger.hpp"

#include "vpux/compiler/core/aliases_info.hpp"
#include "vpux/compiler/core/feasible_memory_scheduler.hpp"
#include "vpux/compiler/core/prefetch_data_ops.hpp"
#include "vpux/compiler/dialect/VPU/utils/cost_model/factories/cost_model_config.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/utils/hw_settings.hpp"

#include <mlir/IR/MLIRContext.h>
#include <mlir/Parser/Parser.h>

#include <gtest/gtest.h>

using namespace vpux;

using MLIR_FeasibleMemorySchedulerActivelySpilling = MLIR_UnitBase;

TEST_F(MLIR_FeasibleMemorySchedulerActivelySpilling, FixMemoryFragmentation) {
    mlir::MLIRContext ctx(registry);

    // Create an IR so that AsyncDepsInfo class can be initialized
    constexpr llvm::StringLiteral inputIRPart1 = R"(
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
module @Test attributes {config.arch = #config.arch_kind<NPU40XX>, config.compilationMode = #config.compilation_mode<DefaultHW>} {
    config.ExecutorResource 2 of @DMA_NN
    config.Resources 3 of @NCE at 2.100000e+03 MHz {
        config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
        config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
        config.ExecutorResource 2 of @SHAVE_ACT
        config.ExecutorResource 1 of @DPU
    }
    config.Resources 1 of @global {
        config.ExecutorResource 1 of @M2I
        config.ExecutorResource 2 of @DMA_NN
        config.MemoryResource 67108864000 bytes of @DDR {config.bandwidth = 64 : i64, config.derateFactor = 6.000000e-01 : f64}
    }
    module @VPU.SW {
        func.func private @builtin_RMS(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, memref<*xf32, @CMX_NN>, memref<*xf16, @CMX_NN>, f64) attributes {VPU.kernel_code = "rms_norm.cpp", VPU.kernel_entry = "rms_norm", VPU.task_type = @COMPUTE}
        func.func private @builtin_Convert(memref<*xf16, @CMX_NN>, memref<*xsi4, @CMX_NN>) attributes {VPU.kernel_code = "convert.cpp", VPU.kernel_entry = "convert", VPU.task_type = @COMPUTE}
        func.func private @builtin_Swish(memref<*xf16, @CMX_NN>) attributes {VPU.kernel_code = "activation_swish.cpp", VPU.kernel_entry = "activation_swish"}
    }
    func.func @main(%arg0: memref<1x64x60x60xf32, @DDR>, %arg1: memref<1x64x60x60xf32, @DDR>) -> memref<1x64x60x60xf32, @DDR> {
        %cst = const.Declare memref<64x64x3x3xf16, #NHWC> = dense<0.000000e+00> : tensor<64x64x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
        %cst_0 = const.Declare memref<64x64x5x5xf16, #NHWC> = dense<0.000000e+00> : tensor<64x64x5x5xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
        %cst_1 = const.Declare memref<64x64x3x3xf16, #NHWC> = dense<0.000000e+00> : tensor<64x64x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
        %cst_2 = const.Declare memref<64x1x1x4xsi32> = dense<0> : tensor<64x1x1x4xsi32>
        %cst_3 = const.Declare memref<64x1x1x4xsi32> = dense<0> : tensor<64x1x1x4xsi32>
        %cst_4 = const.Declare memref<32x64x7x7xf16, #NHWC> = dense<0.000000e+00> : tensor<64x64x7x7xf32>, [#const.SubView<[0, 0, 0, 0], [32, 64, 7, 7]>, #const.CastElemType<f16>, #const.Reorder<#NHWC>]
        %cst_5 = const.Declare memref<32x64x7x7xf16, #NHWC> = dense<0.000000e+00> : tensor<64x64x7x7xf32>, [#const.SubView<[32, 0, 0, 0], [32, 64, 7, 7]>, #const.CastElemType<f16>, #const.Reorder<#NHWC>]
        %cst_6 = const.Declare memref<32x1x1x4xsi32> = dense<0> : tensor<32x1x1x4xsi32>
        %alloc = memref.alloc() : memref<1x64x60x60xf16, #NHWC, [@CMX_NN, 0]>
        %alloc_7 = memref.alloc() : memref<1x64x225x16xf16, [@CMX_NN, 0]>
        %alloc_8 = memref.alloc() : memref<1x16x64x225xf16, #NWCH, [@CMX_NN, 0]>
        %alloc_9 = memref.alloc() : memref<64x64x3x3xf16, #NHWC, [@CMX_NN, 0]>
        %alloc_10 = memref.alloc() : memref<64x1x1x4xsi32, [@CMX_NN, 0]>
        %alloc_11 = memref.alloc() : memref<1x64x60x60xf16, #NHWC, [@CMX_NN, 0]>
        %alloc_12 = memref.alloc() : memref<32x64x7x7xf16, #NHWC, [@CMX_NN, 0]>
        %alloc_13 = memref.alloc() : memref<32x1x1x4xsi32, [@CMX_NN, 0]>
        %alloc_14 = memref.alloc() : memref<32x64x7x7xf16, #NHWC, [@CMX_NN, 0]>
        %alloc_15 = memref.alloc() : memref<64x64x5x5xf16, #NHWC, [@CMX_NN, 0]>
        %alloc_16 = memref.alloc() : memref<64x1x1x4xsi32, [@CMX_NN, 0]>
        %alloc_17 = memref.alloc() : memref<1x64x60x60xf16, #NHWC, [@CMX_NN, 0]>
        %alloc_18 = memref.alloc() : memref<64x64x3x3xf16, #NHWC, [@CMX_NN, 0]>
        %alloc_19 = memref.alloc() : memref<1x64x60x60xf32, [@CMX_NN, 0]>
        %token, %bodyResults = async.execute -> !async.value<memref<1x64x225x16xf16, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 0 : i64, cycleCost = 25792 : i64} {
          %1 = VPUIP.ShapeCast {shape = [1, 64, 225, 16]} inputs(%arg0 : memref<1x64x60x60xf32, @DDR>) -> memref<1x64x225x16xf32, @DDR>
          %2 = VPUIP.ConvertDMA inputs(%1 : memref<1x64x225x16xf32, @DDR>) outputs(%alloc_7 : memref<1x64x225x16xf16, [@CMX_NN, 0]>) -> memref<1x64x225x16xf16, [@CMX_NN, 0]>
          async.yield %2 : memref<1x64x225x16xf16, [@CMX_NN, 0]>
        }
        %token_20, %bodyResults_21 = async.execute [%token] (%bodyResults as %arg2: !async.value<memref<1x64x225x16xf16, [@CMX_NN, 0]>>) -> !async.value<memref<1x16x64x225xf16, #NWCH, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DPU, "async-deps-index" = 1 : i64, cycleCost = 1 : i64} {
          %1 = VPUIP.ViewOp %arg2 : memref<1x64x225x16xf16, [@CMX_NN, 0]> to memref<1x16x64x225xf16, #NHWC, [@CMX_NN, 0]>
          %2 = VPUIP.NCEClusterTask {is_permute_quantize, minimumHardwareExecutionCost = 4294967195 : i64, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<ELTWISE>} input(%1 : memref<1x16x64x225xf16, #NHWC, [@CMX_NN, 0]>) weights(%1 : memref<1x16x64x225xf16, #NHWC, [@CMX_NN, 0]>) parent_input(%1 : memref<1x16x64x225xf16, #NHWC, [@CMX_NN, 0]>) parent_output(%alloc_8 : memref<1x16x64x225xf16, #NWCH, [@CMX_NN, 0]>) outputs(%alloc_8 : memref<1x16x64x225xf16, #NWCH, [@CMX_NN, 0]>) -> memref<1x16x64x225xf16, #NWCH, [@CMX_NN, 0]> variants : {
            DPUTask {inEnd = [224, 63, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [224, 63, 15], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
          } PPE : {
            PPETask {ppe = #VPU.PPEFp<mode = <ADD>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 5.000000e-01 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>}
          }
          async.yield %2 : memref<1x16x64x225xf16, #NWCH, [@CMX_NN, 0]>
        }
        %token_22, %bodyResults_23 = async.execute -> !async.value<memref<64x64x3x3xf16, #NHWC, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 2 : i64, cycleCost = 2602 : i64} {
          %1 = VPUIP.NNDMA inputs(%cst : memref<64x64x3x3xf16, #NHWC>) outputs(%alloc_9 : memref<64x64x3x3xf16, #NHWC, [@CMX_NN, 0]>) -> memref<64x64x3x3xf16, #NHWC, [@CMX_NN, 0]>
          async.yield %1 : memref<64x64x3x3xf16, #NHWC, [@CMX_NN, 0]>
        }
        %token_24, %bodyResults_25 = async.execute -> !async.value<memref<64x1x1x4xsi32, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 3 : i64, cycleCost = 600 : i64} {
          %1 = VPUIP.NNDMA inputs(%cst_2 : memref<64x1x1x4xsi32>) outputs(%alloc_10 : memref<64x1x1x4xsi32, [@CMX_NN, 0]>) -> memref<64x1x1x4xsi32, [@CMX_NN, 0]>
          async.yield %1 : memref<64x1x1x4xsi32, [@CMX_NN, 0]>
        }
        %token_26, %bodyResults_27 = async.execute [%token_20, %token_22, %token_24] (%bodyResults_21 as %arg2: !async.value<memref<1x16x64x225xf16, #NWCH, [@CMX_NN, 0]>>, %bodyResults_23 as %arg3: !async.value<memref<64x64x3x3xf16, #NHWC, [@CMX_NN, 0]>>, %bodyResults_25 as %arg4: !async.value<memref<64x1x1x4xsi32, [@CMX_NN, 0]>>) -> !async.value<memref<1x64x60x60xf16, #NHWC, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DPU, "async-deps-index" = 4 : i64, cycleCost = 75051 : i64} {
          %1 = VPUIP.ViewOp %arg2 : memref<1x16x64x225xf16, #NWCH, [@CMX_NN, 0]> to memref<1x64x225x16xf16, #NHWC, [@CMX_NN, 0]>
          %2 = VPUIP.ShapeCast {shape = [1, 64, 60, 60]} inputs(%1 : memref<1x64x225x16xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x60x60xf16, #NHWC, [@CMX_NN, 0]>
          %3 = VPUIP.NCEClusterTask {is_zero_offset_weights_table, kernel_padding = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, kernel_size = [3, 3], kernel_strides = [1, 1], minimumHardwareExecutionCost = 75051 : i64, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<CONV>} input(%2 : memref<1x64x60x60xf16, #NHWC, [@CMX_NN, 0]>) weights(%arg3 : memref<64x64x3x3xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%arg4 : memref<64x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%2 : memref<1x64x60x60xf16, #NHWC, [@CMX_NN, 0]>) parent_output(%alloc_11 : memref<1x64x60x60xf16, #NHWC, [@CMX_NN, 0]>) outputs(%alloc_11 : memref<1x64x60x60xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x60x60xf16, #NHWC, [@CMX_NN, 0]> variants : {
            DPUTask {inEnd = [59, 59, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [59, 59, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>}
          } PPE : {
            PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>}
          }
          async.yield %3 : memref<1x64x60x60xf16, #NHWC, [@CMX_NN, 0]>
        })";

    constexpr llvm::StringLiteral inputIRPart2 = R"(
        %token_28, %bodyResults_29 = async.execute -> !async.value<memref<32x64x7x7xf16, #NHWC, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 5 : i64, cycleCost = 6075 : i64} {
          %1 = VPUIP.NNDMA inputs(%cst_4 : memref<32x64x7x7xf16, #NHWC>) outputs(%alloc_12 : memref<32x64x7x7xf16, #NHWC, [@CMX_NN, 0]>) -> memref<32x64x7x7xf16, #NHWC, [@CMX_NN, 0]>
          async.yield %1 : memref<32x64x7x7xf16, #NHWC, [@CMX_NN, 0]>
        }
        %token_30, %bodyResults_31 = async.execute -> !async.value<memref<32x1x1x4xsi32, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 6 : i64, cycleCost = 593 : i64} {
          %1 = VPUIP.NNDMA inputs(%cst_6 : memref<32x1x1x4xsi32>) outputs(%alloc_13 : memref<32x1x1x4xsi32, [@CMX_NN, 0]>) -> memref<32x1x1x4xsi32, [@CMX_NN, 0]>
          async.yield %1 : memref<32x1x1x4xsi32, [@CMX_NN, 0]>
        }
        %token_32, %bodyResults_33 = async.execute [%token_26, %token_28, %token_30] (%bodyResults_27 as %arg2: !async.value<memref<1x64x60x60xf16, #NHWC, [@CMX_NN, 0]>>, %bodyResults_29 as %arg3: !async.value<memref<32x64x7x7xf16, #NHWC, [@CMX_NN, 0]>>, %bodyResults_31 as %arg4: !async.value<memref<32x1x1x4xsi32, [@CMX_NN, 0]>>) -> !async.value<memref<1x32x60x60xf16, {order = #NHWC, strides = [230400, 1, 3840, 64]}, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DPU, "async-deps-index" = 7 : i64, cycleCost = 202445 : i64} {
          %1 = VPUIP.SubView %alloc [0, 0, 0, 0] [1, 32, 60, 60] : memref<1x64x60x60xf16, #NHWC, [@CMX_NN, 0]> to memref<1x32x60x60xf16, {order = #NHWC, strides = [230400, 1, 3840, 64]}, [@CMX_NN, 0]>
          %2 = VPUIP.NCEClusterTask {is_zero_offset_weights_table, kernel_padding = #VPU.Padding<left = 3 : i64, right = 3 : i64, top = 3 : i64, bottom = 3 : i64>, kernel_size = [7, 7], kernel_strides = [1, 1], minimumHardwareExecutionCost = 202445 : i64, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<CONV>} input(%arg2 : memref<1x64x60x60xf16, #NHWC, [@CMX_NN, 0]>) weights(%arg3 : memref<32x64x7x7xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%arg4 : memref<32x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%arg2 : memref<1x64x60x60xf16, #NHWC, [@CMX_NN, 0]>) parent_output(%1 : memref<1x32x60x60xf16, {order = #NHWC, strides = [230400, 1, 3840, 64]}, [@CMX_NN, 0]>) outputs(%1 : memref<1x32x60x60xf16, {order = #NHWC, strides = [230400, 1, 3840, 64]}, [@CMX_NN, 0]>) -> memref<1x32x60x60xf16, {order = #NHWC, strides = [230400, 1, 3840, 64]}, [@CMX_NN, 0]> variants : {
            DPUTask {inEnd = [59, 59, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [59, 59, 31], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 3 : i64, top = 3 : i64, bottom = 3 : i64>}
          } PPE : {
            PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>}
          }
          async.yield %2 : memref<1x32x60x60xf16, {order = #NHWC, strides = [230400, 1, 3840, 64]}, [@CMX_NN, 0]>
        }
        %token_34, %bodyResults_35 = async.execute -> !async.value<memref<32x64x7x7xf16, #NHWC, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 8 : i64, cycleCost = 6075 : i64} {
          %1 = VPUIP.NNDMA inputs(%cst_5 : memref<32x64x7x7xf16, #NHWC>) outputs(%alloc_14 : memref<32x64x7x7xf16, #NHWC, [@CMX_NN, 0]>) -> memref<32x64x7x7xf16, #NHWC, [@CMX_NN, 0]>
          async.yield %1 : memref<32x64x7x7xf16, #NHWC, [@CMX_NN, 0]>
        }
        %token_36, %bodyResults_37 = async.execute [%token_26, %token_30, %token_34] (%bodyResults_27 as %arg2: !async.value<memref<1x64x60x60xf16, #NHWC, [@CMX_NN, 0]>>, %bodyResults_31 as %arg3: !async.value<memref<32x1x1x4xsi32, [@CMX_NN, 0]>>, %bodyResults_35 as %arg4: !async.value<memref<32x64x7x7xf16, #NHWC, [@CMX_NN, 0]>>) -> !async.value<memref<1x32x60x60xf16, {order = #NHWC, strides = [230400, 1, 3840, 64]}, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DPU, "async-deps-index" = 9 : i64, cycleCost = 202445 : i64} {
          %1 = VPUIP.SubView %alloc [0, 32, 0, 0] [1, 32, 60, 60] : memref<1x64x60x60xf16, #NHWC, [@CMX_NN, 0]> to memref<1x32x60x60xf16, {order = #NHWC, strides = [230400, 1, 3840, 64]}, [@CMX_NN, 0]>
          %2 = VPUIP.NCEClusterTask {is_zero_offset_weights_table, kernel_padding = #VPU.Padding<left = 3 : i64, right = 3 : i64, top = 3 : i64, bottom = 3 : i64>, kernel_size = [7, 7], kernel_strides = [1, 1], minimumHardwareExecutionCost = 202445 : i64, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<CONV>} input(%arg2 : memref<1x64x60x60xf16, #NHWC, [@CMX_NN, 0]>) weights(%arg4 : memref<32x64x7x7xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%arg3 : memref<32x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%arg2 : memref<1x64x60x60xf16, #NHWC, [@CMX_NN, 0]>) parent_output(%1 : memref<1x32x60x60xf16, {order = #NHWC, strides = [230400, 1, 3840, 64]}, [@CMX_NN, 0]>) outputs(%1 : memref<1x32x60x60xf16, {order = #NHWC, strides = [230400, 1, 3840, 64]}, [@CMX_NN, 0]>) -> memref<1x32x60x60xf16, {order = #NHWC, strides = [230400, 1, 3840, 64]}, [@CMX_NN, 0]> variants : {
            DPUTask {inEnd = [59, 59, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [59, 59, 31], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 3 : i64, top = 3 : i64, bottom = 3 : i64>}
          } PPE : {
            PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>}
          }
          async.yield %2 : memref<1x32x60x60xf16, {order = #NHWC, strides = [230400, 1, 3840, 64]}, [@CMX_NN, 0]>
        }
        %token_38, %bodyResults_39 = async.execute -> !async.value<memref<64x64x5x5xf16, #NHWC, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 10 : i64, cycleCost = 6187 : i64} {
          %1 = VPUIP.NNDMA inputs(%cst_0 : memref<64x64x5x5xf16, #NHWC>) outputs(%alloc_15 : memref<64x64x5x5xf16, #NHWC, [@CMX_NN, 0]>) -> memref<64x64x5x5xf16, #NHWC, [@CMX_NN, 0]>
          async.yield %1 : memref<64x64x5x5xf16, #NHWC, [@CMX_NN, 0]>
        }
        %token_40, %bodyResults_41 = async.execute -> !async.value<memref<64x1x1x4xsi32, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 11 : i64, cycleCost = 600 : i64} {
          %1 = VPUIP.NNDMA inputs(%cst_3 : memref<64x1x1x4xsi32>) outputs(%alloc_16 : memref<64x1x1x4xsi32, [@CMX_NN, 0]>) -> memref<64x1x1x4xsi32, [@CMX_NN, 0]>
          async.yield %1 : memref<64x1x1x4xsi32, [@CMX_NN, 0]>
        }
        %token_42, %bodyResults_43 = async.execute [%token_32, %token_36, %token_38, %token_40] (%bodyResults_33 as %arg2: !async.value<memref<1x32x60x60xf16, {order = #NHWC, strides = [230400, 1, 3840, 64]}, [@CMX_NN, 0]>>, %bodyResults_37 as %arg3: !async.value<memref<1x32x60x60xf16, {order = #NHWC, strides = [230400, 1, 3840, 64]}, [@CMX_NN, 0]>>, %bodyResults_39 as %arg4: !async.value<memref<64x64x5x5xf16, #NHWC, [@CMX_NN, 0]>>, %bodyResults_41 as %arg5: !async.value<memref<64x1x1x4xsi32, [@CMX_NN, 0]>>) -> !async.value<memref<1x64x60x60xf16, #NHWC, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DPU, "async-deps-index" = 12 : i64, cycleCost = 193622 : i64} {
          %1 = VPUIP.ConcatView inputs(%arg2, %arg3 : memref<1x32x60x60xf16, {order = #NHWC, strides = [230400, 1, 3840, 64]}, [@CMX_NN, 0]>, memref<1x32x60x60xf16, {order = #NHWC, strides = [230400, 1, 3840, 64]}, [@CMX_NN, 0]>) outputs(%alloc : memref<1x64x60x60xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x60x60xf16, #NHWC, [@CMX_NN, 0]>
          %2 = VPUIP.NCEClusterTask {is_zero_offset_weights_table, kernel_padding = #VPU.Padding<left = 2 : i64, right = 2 : i64, top = 2 : i64, bottom = 2 : i64>, kernel_size = [5, 5], kernel_strides = [1, 1], minimumHardwareExecutionCost = 193622 : i64, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<CONV>} input(%1 : memref<1x64x60x60xf16, #NHWC, [@CMX_NN, 0]>) weights(%arg4 : memref<64x64x5x5xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%arg5 : memref<64x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%1 : memref<1x64x60x60xf16, #NHWC, [@CMX_NN, 0]>) parent_output(%alloc_17 : memref<1x64x60x60xf16, #NHWC, [@CMX_NN, 0]>) outputs(%alloc_17 : memref<1x64x60x60xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x60x60xf16, #NHWC, [@CMX_NN, 0]> variants : {
            DPUTask {inEnd = [59, 59, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [59, 59, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 2 : i64, right = 2 : i64, top = 2 : i64, bottom = 2 : i64>}
          } PPE : {
            PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>}
          }
          async.yield %2 : memref<1x64x60x60xf16, #NHWC, [@CMX_NN, 0]>
        }
        %token_44, %bodyResults_45 = async.execute -> !async.value<memref<64x64x3x3xf16, #NHWC, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 13 : i64, cycleCost = 2602 : i64} {
          %1 = VPUIP.NNDMA inputs(%cst_1 : memref<64x64x3x3xf16, #NHWC>) outputs(%alloc_18 : memref<64x64x3x3xf16, #NHWC, [@CMX_NN, 0]>) -> memref<64x64x3x3xf16, #NHWC, [@CMX_NN, 0]>
          async.yield %1 : memref<64x64x3x3xf16, #NHWC, [@CMX_NN, 0]>
        }
        %token_46, %bodyResults_47 = async.execute [%token_24, %token_42, %token_44] (%bodyResults_25 as %arg2: !async.value<memref<64x1x1x4xsi32, [@CMX_NN, 0]>>, %bodyResults_43 as %arg3: !async.value<memref<1x64x60x60xf16, #NHWC, [@CMX_NN, 0]>>, %bodyResults_45 as %arg4: !async.value<memref<64x64x3x3xf16, #NHWC, [@CMX_NN, 0]>>) -> !async.value<memref<1x64x60x60xf32, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DPU, "async-deps-index" = 14 : i64, cycleCost = 1 : i64} {
          %1 = VPUIP.NCEClusterTask {is_superdense, is_zero_offset_weights_table, kernel_padding = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, kernel_size = [3, 3], kernel_strides = [1, 1], minimumHardwareExecutionCost = 89942 : i64, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<CONV>} input(%arg3 : memref<1x64x60x60xf16, #NHWC, [@CMX_NN, 0]>) weights(%arg4 : memref<64x64x3x3xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%arg2 : memref<64x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%arg3 : memref<1x64x60x60xf16, #NHWC, [@CMX_NN, 0]>) parent_output(%alloc_19 : memref<1x64x60x60xf32, [@CMX_NN, 0]>) outputs(%alloc_19 : memref<1x64x60x60xf32, [@CMX_NN, 0]>) -> memref<1x64x60x60xf32, [@CMX_NN, 0]> variants : {
            DPUTask {inEnd = [59, 59, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [59, 59, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>}
          } PPE : {
            PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>}
          }
          async.yield %1 : memref<1x64x60x60xf32, [@CMX_NN, 0]>
        }
        %token_48, %bodyResults_49 = async.execute [%token_46] (%bodyResults_47 as %arg2: !async.value<memref<1x64x60x60xf32, [@CMX_NN, 0]>>) -> !async.value<memref<1x64x60x60xf32, @DDR>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 15 : i64, cycleCost = 25792 : i64} {
          %1 = VPUIP.NNDMA inputs(%arg2 : memref<1x64x60x60xf32, [@CMX_NN, 0]>) outputs(%arg1 : memref<1x64x60x60xf32, @DDR>) -> memref<1x64x60x60xf32, @DDR>
          async.yield %1 : memref<1x64x60x60xf32, @DDR>
        }
        %0 = async.await %bodyResults_49 : !async.value<memref<1x64x60x60xf32, @DDR>>
        return %0 : memref<1x64x60x60xf32, @DDR>
    }
}
    )";

    // To avoid long string literals, the IR is split into two parts and concatenated during parsing
    auto module = mlir::parseSourceString<mlir::ModuleOp>((inputIRPart1 + inputIRPart2).str(), &ctx);
    ASSERT_TRUE(module.get() != nullptr);
    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    // Prepare variables
    const auto memKind = VPU::MemoryKind::CMX_NN;
    const auto secondLvlMemKind = VPU::MemoryKind::DDR;
    auto aliasesInfo = AliasesInfoMemType<VPU::MemoryKind::CMX_NN>{func};
    AsyncDepsInfo depsInfo{func};
    auto liveRange = MemLiveRangeInfoMemType<VPU::MemoryKind::CMX_NN>(func, aliasesInfo);
    auto log = vpux::Logger::global();
    uint64_t alignment = vpux::DEFAULT_CMX_ALIGNMENT;
    auto memKindAttr = mlir::SymbolRefAttr::get(&ctx, stringifyEnum(memKind));
    auto available = config::getAvailableMemory(*module, memKindAttr);
    const auto maxSize = available.size();
    auto reservedMemVec = config::getReservedMemOffsetAndSizeVec(*module, memKindAttr);
    LinearScan<mlir::Value, LinearScanHandler> scan(maxSize.count(), reservedMemVec, alignment);
    const auto arch = config::getArch(module->getOperation());
    VPU::CostModelConfig::setFactory(arch);
    auto costModel = VPU::CostModelConfig::createCostModel(arch);
    auto tileOp = config::getTileExecutor(*module);
    ASSERT_TRUE(tileOp != nullptr);
    auto tileCount = tileOp.getCount();
    auto dmaPorts = config::getAvailableExecutor(*module, VPU::ExecutorKind::DMA_NN);
    ASSERT_TRUE(dmaPorts != nullptr);
    auto dmaCount = dmaPorts.getCount();

    // init schedule
    FeasibleMemoryScheduler initSchedule(memKind, secondLvlMemKind, liveRange, depsInfo, log, scan, arch, costModel,
                                         tileCount, dmaCount, /*enableScheduleStatistics*/ false,
                                         /*optimizeFragmentation*/ true,
                                         /*activelySpillForPrefetching*/ false);
    auto initScheduledOps = initSchedule.generateSchedule();
    const auto initRes = initScheduledOps.back().cycleEnd_;

    PrefetchDataOps prefetching(initScheduledOps, depsInfo);
    EXPECT_EQ(prefetching.enableDataOpPrefetching(), true);
    VPUIP::moveDeclarationsToTop(func);

    // prefetching - default
    depsInfo = AsyncDepsInfo{func};
    LinearScan<mlir::Value, LinearScanHandler> defaultPrefetchScan(maxSize.count(), reservedMemVec, alignment);
    auto defaultLiveRange = MemLiveRangeInfoMemType<VPU::MemoryKind::CMX_NN>(func, aliasesInfo);
    FeasibleMemoryScheduler schedulerDefault(memKind, secondLvlMemKind, defaultLiveRange, depsInfo, log,
                                             defaultPrefetchScan, arch, costModel, tileCount, dmaCount,
                                             /*enableScheduleStatistics*/ false, /*optimizeFragmentation*/ true,
                                             /*activelySpillForPrefetching*/ false);
    const auto defaultRes = schedulerDefault.generateSchedule().back().cycleEnd_;

    // prefetching - actively spilling
    depsInfo = AsyncDepsInfo{func};
    LinearScan<mlir::Value, LinearScanHandler> aggressivePrefetchScan(maxSize.count(), reservedMemVec, alignment);
    auto aggressiveLiveRange = MemLiveRangeInfoMemType<VPU::MemoryKind::CMX_NN>(func, aliasesInfo);
    FeasibleMemoryScheduler schedulerWithAggressivePrefetch(
            memKind, secondLvlMemKind, aggressiveLiveRange, depsInfo, log, aggressivePrefetchScan, arch, costModel,
            tileCount, dmaCount, /*enableScheduleStatistics*/ false, /*optimizeFragmentation*/ true,
            /*activelySpillForPrefetching*/ true);
    const auto activelySpillRes = schedulerWithAggressivePrefetch.generateSchedule().back().cycleEnd_;

    // The initial scheduler, default prefetch and actively spilling prefetch should give different results
    // In this case, the initial schedule is the worst, default prefetch is better and actively spilling prefetch is the
    // best
    EXPECT_GT(defaultRes, activelySpillRes);
    EXPECT_GT(initRes, defaultRes);
}
