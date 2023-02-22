//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --init-compiler="vpu-arch=VPUX37XX" --convert-VPUIP-to-VPUIPRegMapped %s | FileCheck %s

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType0 = type !quant.uniform<u8:f16, 0.013495710784313726:128>
module @mainModule {

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
module @VPU.SW {
  func private @builtin_Convert(memref<*xf16, [@CMX_NN, 0]>, memref<*xf32, [@CMX_NN, 0]>) attributes {VPU.kernel_code = "single_shave_convert.cpp", VPU.kernel_entry = "single_shave_convert"}
  func private @builtin_MemPermute(memref<*x!qElemType0, [@CMX_NN, 0]>, memref<*x!qElemType0, [@CMX_NN, 0]>, none) attributes {VPU.kernel_code = "reorder_fp16.cpp", VPU.kernel_entry = "reorder_fp16"}
  func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func private @barrier_counters(%arg0: memref<1x32x32x32xf16, #NHWC, @DDR>, %arg1: memref<1x64x16x32xf16>) -> memref<1x64x16x32xf16> {
    %b0 = VPURT.ConfigureBarrier<0> -> !VPURT.Barrier
    %b1 = VPURT.ConfigureBarrier<1> -> !VPURT.Barrier
    %b2 = VPURT.ConfigureBarrier<2> -> !VPURT.Barrier
    %b3 = VPURT.ConfigureBarrier<3> -> !VPURT.Barrier

    %cst_0 = const.Declare memref<1x1x1x3088xui8> = dense<0> : tensor<1x1x1x3088xui8>

    %m0 = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<1x1x1x3088xui8, [@CMX_NN, 0]>
    %m1 = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<1x16x112x112x!qElemType0, {order = #NHWC}, [@CMX_NN, 0]>
    %m2 = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<1x96x112x112x!qElemType0, {order = #NHWC}, [@CMX_NN, 0]>
    %m3 = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<96x16x1x1x!qElemType0, {order = #NHWC}, [@CMX_NN, 0]>
    %m4 = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<96x1x1x4xsi32, {order = #NCHW}, [@CMX_NN, 0]>
    %m5 = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<96x16x1x1x!qElemType0, #NHWC, [@CMX_NN, 0]>
    %m6 = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<96x1x1x4xsi32, [@CMX_NN, 0]>
    %m7 = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<1x1x1x16xui8, [@CMX_NN, 0]>
    %m8 = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<1x96x56x56x!qElemType0, {order = #NHWC}, [@CMX_NN, 0]>
    %m9 = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<1x1000xf16, [@CMX_NN, 0]>
    %m10 = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<1x1000xf32, [@CMX_NN, 0]>

    VPURT.Task updates(%b0 : !VPURT.Barrier) attributes {cycleBegin = 127955 : i64, cycleEnd = 128084 : i64, isTrailingSWLayer = false} {
        %t0 = VPUIP.NNDMA {port = 0 : i64} inputs(%cst_0 : memref<1x1x1x3088xui8>) outputs(%m0 : memref<1x1x1x3088xui8, [@CMX_NN, 0]>) -> memref<1x1x1x3088xui8, [@CMX_NN, 0]>
    }
    VPURT.Task waits(%b0 : !VPURT.Barrier) updates(%b1 : !VPURT.Barrier) attributes {cycleBegin = 127955 : i64, cycleEnd = 128084 : i64, isTrailingSWLayer = false} {
        %t0 = VPUIP.NNDMA {port = 0 : i64} inputs(%cst_0 : memref<1x1x1x3088xui8>) outputs(%m0 : memref<1x1x1x3088xui8, [@CMX_NN, 0]>) -> memref<1x1x1x3088xui8, [@CMX_NN, 0]>
    }
    VPURT.Task waits(%b1 : !VPURT.Barrier) updates(%b2 : !VPURT.Barrier) attributes {cycleBegin = 127955 : i64, cycleEnd = 161696 : i64, isTrailingSWLayer = false} {
        %t0 = VPUIP.NCEClusterTask {kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, kernel_size = [1, 1], kernel_strides = [1, 1], minimumHardwareExecutionCost = 33741 : i64, task_type = "CONV"} input(%m1 : memref<1x16x112x112x!qElemType0, {order = #NHWC}, [@CMX_NN, 0]>) weights(%m3 : memref<96x16x1x1x!qElemType0, {order = #NHWC}, [@CMX_NN, 0]>) weight_table(%m4 : memref<96x1x1x4xsi32, {order = #NCHW}, [@CMX_NN, 0]>) parent_input(%m1 : memref<1x16x112x112x!qElemType0, {order = #NHWC}, [@CMX_NN, 0]>) parent_output(%m2 : memref<1x96x112x112x!qElemType0, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%m2 : memref<1x96x112x112x!qElemType0, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x96x112x112x!qElemType0, {order = #NHWC}, [@CMX_NN, 0]> variants : {
        DPUTask {outEnd = [111, 111, 95], outStart = [0, 0, 0], mpe_mode = "CUBOID_8x16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}}
        } PPE : {
        PPETask "LRELUX" {clamp_high = 255 : i64, clamp_low = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64}
        }
    }
    VPURT.Task waits(%b2 : !VPURT.Barrier) updates(%b3 : !VPURT.Barrier) attributes {cycleBegin = 161696 : i64, cycleEnd = 189390 : i64, isTrailingSWLayer = false} {
        %t0 = VPUIP.NCEClusterTask {activation_window_channel_length = 51 : i64, kernel_padding = {bottom = 0 : i64, left = 1 : i64, right = 0 : i64, top = 1 : i64}, kernel_size = [3, 3], kernel_strides = [2, 2], minimumHardwareExecutionCost = 27694 : i64, task_type = "DWCONV"} input(%m2 : memref<1x96x112x112x!qElemType0, {order = #NHWC}, [@CMX_NN, 0]>) weights(%m5 : memref<96x16x1x1x!qElemType0, #NHWC, [@CMX_NN, 0]>) weight_table(%m6 : memref<96x1x1x4xsi32, [@CMX_NN, 0]>) activation_window(%m7 : memref<1x1x1x16xui8, [@CMX_NN, 0]>) parent_input(%m2 : memref<1x96x112x112x!qElemType0, {order = #NHWC}, [@CMX_NN, 0]>) parent_output(%m8 : memref<1x96x56x56x!qElemType0, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%m8 : memref<1x96x56x56x!qElemType0, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x96x56x56x!qElemType0, {order = #NHWC}, [@CMX_NN, 0]> variants : {
        DPUTask {outEnd = [55, 55, 63], outStart = [0, 0, 0], mpe_mode = "CUBOID_16x16", pad = {bottom = 0 : i64, left = 1 : i64, right = 0 : i64, top = 1 : i64}}
        DPUTask {outEnd = [55, 55, 95], outStart = [0, 0, 0], mpe_mode = "CUBOID_16x16", pad = {bottom = 0 : i64, left = 1 : i64, right = 0 : i64, top = 1 : i64}}
        } PPE : {
        PPETask "LRELUX" {clamp_high = 255 : i64, clamp_low = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64}
        }
    }
    VPURT.Task waits(%b3 : !VPURT.Barrier) attributes {cycleBegin = 510750 : i64, cycleEnd = 510752 : i64, isTrailingSWLayer = false} {
      %results = VPUIP.SW.Kernel {result_segment_sizes = dense<[1, 0]> : vector<2xi32>} @VPU.SW::@builtin_Convert inputs(%m9 as %arg2: memref<1x1000xf16, [@CMX_NN, 0]>) outputs(%m10 as %arg3: memref<1x1000xf32, [@CMX_NN, 0]>) on tile 0 -> memref<1x1000xf32, [@CMX_NN, 0]>{
        VPUIP.SW.Kernel.run(%arg2, %arg3) : memref<1x1000xf16, [@CMX_NN, 0]>, memref<1x1000xf32, [@CMX_NN, 0]>
      }
    }
  return %arg1 : memref<1x64x16x32xf16>
}
}

// CHECK: %0 = VPUIPRegMapped.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<0, -1> -> !VPUIPRegMapped.Index<0>
// CHECK: %1 = VPUIPRegMapped.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<1, -1> -> !VPUIPRegMapped.Index<1>
// CHECK: %2 = VPUIPRegMapped.ConfigureBarrier {consumer_count = 2 : ui8, producer_count = 1 : ui8}<2, -1> -> !VPUIPRegMapped.Index<2>
// CHECK: %3 = VPUIPRegMapped.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 2 : ui8}<3, -1> -> !VPUIPRegMapped.Index<3>
