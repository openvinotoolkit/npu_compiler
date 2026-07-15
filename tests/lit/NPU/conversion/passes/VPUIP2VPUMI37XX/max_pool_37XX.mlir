//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// max_pool_1x64x16x16xfp16_2x2_pads_1x0x1x0_strides_2x2_fp16

// RUN: vpux-opt --init-compiler="platform=%platform%" --convert-VPUIP-to-VPUMI37XX %s | FileCheck %s
// REQUIRES: platform-NPU3720

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @mainModule {

  net.NetworkInfo entryPoint : @maxpool_f16_f16 inputsInfo : {
    DataInfo "input_0" : tensor<1x64x16x16xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x64x9x8xf16>
  }

  func.func nested @maxpool_f16_f16(%arg0: memref<1x64x16x16xf16, {order = #NHWC}, @DDR>, %arg1: memref<1x64x9x8xf16, {order = #NHWC}, @DDR>) -> memref<1x64x9x8xf16, {order = #NHWC}, @DDR> {
    %0 = VPURT.DeclareBuffer <CMX_NN> [0] <9216> -> memref<1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %1 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x64x9x8xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %2 = VPURT.DeclareBuffer <CMX_NN> [0] <9216> -> memref<1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %3 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x64x9x8xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %6 = VPURT.DeclareBuffer <CMX_NN> [0] <41984> -> memref<1x1x1x16xui8, {order = #NHWC}, [@CMX_NN, 0]>
    %7 = VPURT.DeclareBuffer <CMX_NN> [0] <42000> -> memref<64x1x1x4xsi32, {order = #NHWC}, [@CMX_NN, 0]>
    VPURT.Task {
      %8 = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 1 : i64, bottom = 1 : i64>, kernel_size = [2, 2], kernel_strides = [2, 2], task_type = #VPUIP.nce_task_type<MAXPOOL>}> input(%0 : memref<1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>) weight_table(%7 : memref<64x1x1x4xsi32, {order = #NHWC}, [@CMX_NN, 0]>) parent_input(%2 : memref<1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>) parent_output(%3 : memref<1x64x9x8xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%1 : memref<1x64x9x8xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x64x9x8xf16, {order = #NHWC}, [@CMX_NN, 0]> variants : {
        DPUTask {outEnd = [7, 8, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 1 : i64, bottom = 1 : i64>, outStart = [0, 0, 0]}
      } PPE : { PPETask {ppe = #VPU.PPEStub<>} }
    }
    return %arg1 : memref<1x64x9x8xf16, {order = #NHWC}, @DDR>
  }

}

// CHECK: func.func nested @maxpool_f16_f16
// CHECK-NEXT: VPURT.DeclareBuffer <CMX_NN>
// CHECK-NEXT: VPURT.DeclareBuffer <CMX_NN>
// CHECK-NOT: VPURT.Task
// CHECK: VPUMI37XX.DPUInvariant
// CHECK: VPUMI37XX.DPUVariant
