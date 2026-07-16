//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --convert-VPUIP-to-VPUMI37XX %s | FileCheck %s
// REQUIRES: platform-NPU3720

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @mainModule {
net.NetworkInfo entryPoint : @maxpool_f16_f16 inputsInfo : {
DataInfo "input_0" : tensor<1x64x16x16xf16>
} outputsInfo : {
DataInfo "output_0" : tensor<1x64x8x8xf16>
}

func.func nested @maxpool_f16_f16(%arg0: memref<1x64x16x16xf16, {order = #NHWC}, @DDR>, %arg1: memref<1x64x8x8xf16, {order = #NHWC}, @DDR>) -> memref<1x64x8x8xf16, {order = #NHWC}, @DDR> {
  %input = VPURT.DeclareBuffer <CMX_NN> [0] <8192> -> memref<1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>
  %output = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x64x8x8xf16, {order = #NHWC}, [@CMX_NN, 0]>
  %parent_input = VPURT.DeclareBuffer <CMX_NN> [0] <8192> -> memref<1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>
  %parent_output = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x64x8x8xf16, {order = #NHWC}, [@CMX_NN, 0]>
  %weight_table = VPURT.DeclareBuffer <CMX_NN> [0] <40976> -> memref<64x1x1x4xsi32, {order = #NHWC}, [@CMX_NN, 0]>

  VPURT.Task {
      %8 = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{
            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            kernel_size = [2, 2],
            kernel_strides = [2, 2],
            task_type = #VPUIP.nce_task_type<MAXPOOL>}>
          input(%input : memref<1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>)
          weight_table(%weight_table : memref<64x1x1x4xsi32, {order = #NHWC}, [@CMX_NN, 0]>)
          parent_input(%parent_input : memref<1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>)
          parent_output(%parent_output : memref<1x64x8x8xf16, {order = #NHWC}, [@CMX_NN, 0]>)
          outputs(%output : memref<1x64x8x8xf16, {order = #NHWC}, [@CMX_NN, 0]>)
        -> memref<1x64x8x8xf16, {order = #NHWC}, [@CMX_NN, 0]> variants : {
      DPUTask {outEnd = [7, 7, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0]}
      } PPE : { PPETask {ppe = #VPU.PPEStub<>} }
  }
  return %arg1 : memref<1x64x8x8xf16, {order = #NHWC}, @DDR>
}
}


//CHECK-LABEL: @maxpool_f16_f16

//CHECK: [[INPUT:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <8192> -> [[TYPE_INPUT:.+]]
//CHECK: [[OUTPUT:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> [[TYPE_OUTPUT:.+]]
//CHECK: [[PARENT_INPUT:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <8192> -> [[TYPE_PARENT_INPUT:.+]]
//CHECK: [[PARENT_OUTPUT:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> [[TYPE_PARENT_OUTPUT:.+]]
//CHECK: [[WEIGHT_TABLE:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <40976> -> [[TYPE_WEIGHT_TABLE:.+]]


//CHECK-NOT: VPURT.Task
//CHECK: DPUInvariant
//CHECK-SAME: task_type = #VPUIP.nce_task_type<MAXPOOL>
//CHECK-SAME: input([[INPUT]] : [[TYPE_INPUT]])
//CHECK-SAME: weight_table([[WEIGHT_TABLE]] : [[TYPE_WEIGHT_TABLE]])
//CHECK-SAME: parent_input([[PARENT_INPUT]] : [[TYPE_PARENT_INPUT]])
//CHECK-SAME: parent_output([[PARENT_OUTPUT]] : [[TYPE_PARENT_OUTPUT]])
//CHECK-SAME: outputs([[OUTPUT]] : [[TYPE_OUTPUT]])
//CHECK-NOT: DPUTask
//CHECK-NEXT: VPUMI37XX.PPETask

//CHECK: VPUMI37XX.DPUVariant
