//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX37XX" --convert-VPUIP-to-VPUIPRegMapped %s | FileCheck %s

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @mainModule {
IE.CNNNetwork entryPoint : @maxpool_f16_f16 inputsInfo : {
DataInfo "input_0" : tensor<1x64x16x16xf16, {order = #NHWC}>
} outputsInfo : {
DataInfo "output_0" : tensor<1x64x8x8xf16, {order = #NHWC}>
}

func private @maxpool_f16_f16(%arg0: memref<1x64x16x16xf16, #NHWC, @DDR>, %arg1: memref<1x64x8x8xf16, #NHWC, @DDR>) -> memref<1x64x8x8xf16, #NHWC, @DDR> {
  %0 = VPURT.ConfigureBarrier<0> -> !VPURT.Barrier
  %1 = VPURT.ConfigureBarrier<1> -> !VPURT.Barrier

  %cst = const.Declare memref<64x1x1x4xsi32, #NHWC, @DDR> = dense<1> : tensor<64x1x1x4xsi32>, [#const.Reorder<#NHWC>]
  %cst_0 = const.Declare memref<1x1x1x16xui8, #NHWC, @DDR> = dense<[[[[3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]]]> : tensor<1x1x1x16xui8>, [#const.Reorder<#NHWC>]

  %2 = VPURT.DeclareBuffer "CMX_NN" [0] <8192> -> memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]>
  %3 = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<1x64x8x8xf16, #NHWC, [@CMX_NN, 0]>
  %4 = VPURT.DeclareBuffer "CMX_NN" [0] <8192> -> memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]>
  %5 = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<1x64x8x8xf16, #NHWC, [@CMX_NN, 0]>
  %6 = VPURT.DeclareBuffer "CMX_NN" [0] <40960> -> memref<1x1x1x16xui8, #NHWC, [@CMX_NN, 0]>
  %7 = VPURT.DeclareBuffer "CMX_NN" [0] <40976> -> memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>

  VPURT.Task updates(%0 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %8 = VPUIP.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x64x16x16xf16, #NHWC, @DDR>) outputs(%2 : memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]>
  }
  VPURT.Task updates(%0 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %8 = VPUIP.NNDMA {port = 0 : i64} inputs(%cst_0 : memref<1x1x1x16xui8, #NHWC, @DDR>) outputs(%6 : memref<1x1x1x16xui8, #NHWC, [@CMX_NN, 0]>) -> memref<1x1x1x16xui8, #NHWC, [@CMX_NN, 0]>
  }
  VPURT.Task updates(%0 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %8 = VPUIP.NNDMA {port = 0 : i64} inputs(%cst : memref<64x1x1x4xsi32, #NHWC, @DDR>) outputs(%7 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>) -> memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>
  }
  VPURT.Task waits(%0 : !VPURT.Barrier) updates(%1 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %8 = VPUIP.NCEClusterTask {activation_window_channel_length = 16 : i32, kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, kernel_size = [2, 2], kernel_strides = [2, 2], task_type = "MAXPOOL"} input(%2 : memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%7 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>) activation_window(%6 : memref<1x1x1x16xui8, #NHWC, [@CMX_NN, 0]>) parent_input(%4 : memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]>) parent_output(%5 : memref<1x64x8x8xf16, #NHWC, [@CMX_NN, 0]>) outputs(%3 : memref<1x64x8x8xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x8x8xf16, #NHWC, [@CMX_NN, 0]> variants : {
      DPUTask {outEnd = [7, 7, 63], mpe_mode = "CUBOID_16x16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [0, 0, 0]}
      } PPE : {
      PPETask "NOOP" {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64}
      }
  }
  VPURT.Task waits(%1 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %8 = VPUIP.NNDMA {port = 0 : i64} inputs(%3 : memref<1x64x8x8xf16, #NHWC, [@CMX_NN, 0]>) outputs(%arg1 : memref<1x64x8x8xf16, #NHWC, @DDR>) -> memref<1x64x8x8xf16, #NHWC, @DDR>
  }
  return %arg1 : memref<1x64x8x8xf16, #NHWC, @DDR>
}
}


//CHECK-LABEL: @maxpool_f16_f16

//CHECK: %[[VAL0:.*]] = VPUIPRegMapped.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 3 : ui8}<0, -1> -> !VPUIPRegMapped.Index<0>
//CHECK: %[[VAL1:.*]] = VPUIPRegMapped.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<1, -1> -> !VPUIPRegMapped.Index<1>

//CHECK: %[[VALcst:.*]] = const.Declare [[TYPE_CST:.*]] = dense
//CHECK: %[[VALcst_0:.*]] = const.Declare [[TYPE_CST0:.*]] = dense

//CHECK: %[[VAL2:.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <8192> -> [[TYPE2:.*]]
//CHECK: %[[VAL3:.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> [[TYPE3:.*]]
//CHECK: %[[VAL4:.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <8192> -> [[TYPE4:.*]]
//CHECK: %[[VAL5:.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> [[TYPE5:.*]]
//CHECK: %[[VAL6:.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <40960> -> [[TYPE6:.*]]
//CHECK: %[[VAL7:.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <40976> -> [[TYPE7:.*]]


//CHECK-NOT: VPURT.Task
//CHECK: VPUIPRegMapped.NNDMA {{.*}} inputs(%arg0 : memref<1x64x16x16xf16, #NHWC, @DDR>) outputs(%[[VAL2]] : [[TYPE2]]) updates(%[[VAL0]] : !VPUIPRegMapped.Index<0>)

//CHECK-NOT: VPURT.Task
//CHECK: VPUIPRegMapped.NNDMA {{.*}} inputs(%[[VALcst]] : [[TYPE_CST]]) outputs(%[[VAL7]] : [[TYPE7]])
//CHECK-SAME: VPUIPRegMapped.Index<2>

//CHECK-NOT: VPURT.Task
//CHECK: DPUInvariant
    //CHECK-SAME task_type = "MAXPOOL"
    //CHECK-SAME input(%[[VAL2]] : [[TYPE2]])
    //CHECK-SAME weight_table(%[[VAL7]] : [[TYPE7]])
    //CHECK-SAME parent_input(%[[VAL4]] : [[TYPE4]])
    //CHECK-SAME parent_output(%[[VAL5]] : [[TYPE5]])
    //CHECK-SAME outputs(%[[VAL3]] : [[TYPE3]])
    //CHECK-SAME waits(%[[VAL0]]
    //CHECK-SAME updates(%[[VAL1]])
//CHECK-NOT: DPUTask
//CHECK-NEXT: VPUIP.PPETask "NOOP"

//CHECK: VPUIPRegMapped.DPUVariant