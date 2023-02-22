//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
// max_pool_1x64x16x16xfp16_2x2_pads_1x0x1x0_strides_2x2_fp16

// RUN: vpux-opt --init-compiler="vpu-arch=VPUX37XX" --convert-VPUIP-to-VPUIPRegMapped %s | FileCheck %s
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @mainModule {

  IE.CNNNetwork entryPoint : @maxpool_f16_f16 inputsInfo : {
    DataInfo "input_0" : tensor<1x64x16x16xf16, {order = #NHWC}>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x64x9x8xf16, {order = #NHWC}>
  }

  func private @maxpool_f16_f16(%arg0: memref<1x64x16x16xf16, #NHWC, @DDR>, %arg1: memref<1x64x9x8xf16, #NHWC, @DDR>) -> memref<1x64x9x8xf16, #NHWC, @DDR> {
    %0 = VPURT.DeclareBuffer "CMX_NN" [0] <9216> -> memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]>
    %1 = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<1x64x9x8xf16, #NHWC, [@CMX_NN, 0]>
    %2 = VPURT.DeclareBuffer "CMX_NN" [0] <9216> -> memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]>
    %3 = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<1x64x9x8xf16, #NHWC, [@CMX_NN, 0]>
    %4 = VPURT.ConfigureBarrier<0> -> !VPURT.Barrier
    %5 = VPURT.ConfigureBarrier<1> -> !VPURT.Barrier
    VPURT.Task updates(%4 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %8 = VPUIP.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x64x16x16xf16, #NHWC, @DDR>) outputs(%0 : memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]>
    }
    %cst = const.Declare memref<1x1x1x16xui8, #NHWC, @DDR> = dense<0> : tensor<1x1x1x16xui8>, [#const.Reorder<#NHWC>]
    %6 = VPURT.DeclareBuffer "CMX_NN" [0] <41984> -> memref<1x1x1x16xui8, #NHWC, [@CMX_NN, 0]>
    VPURT.Task updates(%4 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %8 = VPUIP.NNDMA {port = 0 : i64} inputs(%cst : memref<1x1x1x16xui8, #NHWC, @DDR>) outputs(%6 : memref<1x1x1x16xui8, #NHWC, [@CMX_NN, 0]>) -> memref<1x1x1x16xui8, #NHWC, [@CMX_NN, 0]>
    }
    %cst_0 = const.Declare memref<64x1x1x4xsi32, #NHWC, @DDR> = dense<0> : tensor<64x1x1x4xsi32>, [#const.Reorder<#NHWC>]
    %7 = VPURT.DeclareBuffer "CMX_NN" [0] <42000> -> memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>
    VPURT.Task updates(%4 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %8 = VPUIP.NNDMA {port = 0 : i64} inputs(%cst_0 : memref<64x1x1x4xsi32, #NHWC, @DDR>) outputs(%7 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>) -> memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>
    }
    VPURT.Task waits(%4 : !VPURT.Barrier) updates(%5 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %8 = VPUIP.NCEClusterTask {activation_window_channel_length = 16 : i32, kernel_padding = {bottom = 1 : i64, left = 0 : i64, right = 0 : i64, top = 1 : i64}, kernel_size = [2, 2], kernel_strides = [2, 2], task_type = "MAXPOOL"} input(%0 : memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%7 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>) activation_window(%6 : memref<1x1x1x16xui8, #NHWC, [@CMX_NN, 0]>) parent_input(%2 : memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]>) parent_output(%3 : memref<1x64x9x8xf16, #NHWC, [@CMX_NN, 0]>) outputs(%1 : memref<1x64x9x8xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x9x8xf16, #NHWC, [@CMX_NN, 0]> variants : {
        DPUTask {outEnd = [7, 8, 63], mpe_mode = "CUBOID_16x16", pad = {bottom = 1 : i64, left = 0 : i64, right = 0 : i64, top = 1 : i64}, outStart = [0, 0, 0]}
      } PPE : {
        PPETask "NOOP" {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64}
      }
    }
    VPURT.Task waits(%5 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %8 = VPUIP.NNDMA {port = 0 : i64} inputs(%1 : memref<1x64x9x8xf16, #NHWC, [@CMX_NN, 0]>) outputs(%arg1 : memref<1x64x9x8xf16, #NHWC, @DDR>) -> memref<1x64x9x8xf16, #NHWC, @DDR>
    }
    return %arg1 : memref<1x64x9x8xf16, #NHWC, @DDR>
  }

}

// CHECK: func private @maxpool_f16_f16
// CHECK: VPUIPRegMapped.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 3 : ui8}<0, -1> -> !VPUIPRegMapped.Index<0>
// CHECK: VPUIPRegMapped.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<1, -1> -> !VPUIPRegMapped.Index<1>
// CHECK-NOT: VPURT.Task
// CHECK-NEXT: VPUIPRegMapped.NNDMA
// CHECK-NEXT: const.Declare
// CHECK-NEXT: VPURT.DeclareBuffer "CMX_NN"
// CHECK-NOT: VPURT.Task
// CHECK-NEXT: VPUIPRegMapped.NNDMA
// CHECK-NEXT: const.Declare
// CHECK-NEXT: VPURT.DeclareBuffer "CMX_NN"
// CHECK-NOT: VPURT.Task
// CHECK-NEXT: VPUIPRegMapped.NNDMA
// CHECK-NOT: VPURT.Task
// CHECK-NEXT: VPUIPRegMapped.DPUInvariant
// CHECK: VPUIPRegMapped.DPUVariant
// CHECK-NOT: VPURT.Task
// CHECK: VPUIPRegMapped.NNDMA
