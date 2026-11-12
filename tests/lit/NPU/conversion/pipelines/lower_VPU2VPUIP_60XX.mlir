//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --lower-VPU-to-VPUIP %s | FileCheck %s
// REQUIRES: platform-NPU6010

//
// The 'lower-VPU-to-VPUIP' pipeline:
//
//   * Fully replaces VPU Dialect with VPUIP Dialect
//   * Changes all Value types from `tensor` to `memref`
//   * Adds result arguments to Function signature
//   * Inserts `VPUIP.Copy` to store result in output buffer
//   * Uses activation SHAVE kernels `VPUIP.SW.Kernel` for software ops
//

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module {
  net.NetworkInfo entryPoint : @NCEConvSplitWeightTableFormat inputsInfo : {
    DataInfo "Parameter" : tensor<1x32x16x16xf16>
  } outputsInfo : {
    DataInfo "Result" : tensor<1x64x14x14xf16>
  }
// CHECK: func.func @NCEConvSplitWeightTableFormat([[ARG0:%.+]]: memref<1x32x16x16xf16, {order = #NHWC}, @CMX_NN>, [[ARG1:%.+]]: memref<1x64x14x14xf16, {order = #NHWC}, @CMX_NN>) -> memref<1x64x14x14xf16, {order = #NHWC}, @CMX_NN> {
func.func @NCEConvSplitWeightTableFormat(%arg0 : tensor<1x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x64x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}> {
    %weights = const.Declare tensor<64x32x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}> = dense<1.000000e+00> : tensor<64x32x3x3xf16, {mem_space = @CMX_NN}>, [#const.Reorder<#NHWC>]
    %scale_table = const.Declare tensor<64x1x1x1xf32, {mem_space = @CMX_NN}> = dense<1.0> : tensor<64x1x1x1xf32, {mem_space = @CMX_NN}>
    %bias_table = const.Declare tensor<64x1x1x1xf32, {mem_space = @CMX_NN}> = dense<2.0> : tensor<64x1x1x1xf32, {mem_space = @CMX_NN}>

    %out = VPU.NCE.Convolution(%arg0, %weights, %scale_table, %bias_table) rawFilterShape [64, 32, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            ppe = #VPU.PPEStub<>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,

            strides = [1, 1],
            operandSegmentSizes = array<i32: 1, 1, 0, 1, 1, 0, 0>
        } : tensor<1x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>, tensor<64x32x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>, tensor<64x1x1x1xf32, {mem_space = @CMX_NN}>, tensor<64x1x1x1xf32, {mem_space = @CMX_NN}> -> tensor<1x64x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}> {
            VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 32, 16, 16] pad [0, 0, 0, 0] #VPU.mpe_mode<VECTOR_FP16>
        }

    return %out : tensor<1x64x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>

    // CHECK-DAG:       [[WEIGHTS:%.+]] = const.Declare memref<64x32x3x3xf16, {order = #NHWC}, @CMX_NN> = dense<1.000000e+00>
    // CHECK-SAME:      : tensor<64x32x3x3xf16, {mem_space = @CMX_NN}>, [#const.Reorder<#NHWC>]
    // CHECK-DAG:       [[SCALE_TABLE:%.+]] = const.Declare memref<64x1x1x1xf32, @CMX_NN> = dense<1.000000e+00>
    // CHECK-DAG:       [[BIAS_TABLE:%.+]] = const.Declare memref<64x1x1x1xf32, @CMX_NN> = dense<2.000000e+00>
    // CHECK-SAME:      : tensor<64x1x1x1xf32, {mem_space = @CMX_NN}>

    // CHECK:       [[OUT_BUF:%.+]] = memref.alloc() : memref<1x64x14x14xf16, {order = #NHWC}, @CMX_NN>
    // CHECK:       [[OUT:%.+]] = VPUIP.NCEClusterTask
    // CHECK-SAME:          {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [3, 3], kernel_strides = [1, 1],
    // CHECK-SAME:          task_type = #VPUIP.nce_task_type<CONV>}
    // CHECK-SAME:          input([[ARG0]]
    // CHECK-SAME:          weights([[WEIGHTS]]
    // CHECK-SAME:          weight_table_scale([[SCALE_TABLE]]
    // CHECK-SAME:          weight_table_bias([[BIAS_TABLE]]
    // CHECK-SAME:          parent_input([[ARG0]]
    // CHECK-SAME:          parent_output([[OUT_BUF]]
    // CHECK-SAME:          outputs([[OUT_BUF]]
    // CHECK-SAME:      -> memref<1x64x14x14xf16, {order = #NHWC}, @CMX_NN> variants :  {
    // CHECK:               DPUTask {mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [15, 15, 31], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    // CHECK:           } PPE :  {
    // CHECK:           }

    // CHECK:       [[RESULT:%.+]] = VPUIP.Copy inputs([[OUT]] : memref<1x64x14x14xf16, {order = #NHWC}, @CMX_NN>) outputs([[ARG1]] : memref<1x64x14x14xf16, {order = #NHWC}, @CMX_NN>)
    // CHECK-SAME:      -> memref<1x64x14x14xf16, {order = #NHWC}, @CMX_NN>
    // CHECK:       return [[RESULT]] : memref<1x64x14x14xf16, {order = #NHWC}, @CMX_NN>
}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module {
  net.NetworkInfo entryPoint : @NCEConvWithoutAnyWeightTable inputsInfo : {
    DataInfo "Parameter" : tensor<1x32x16x16xf16>
  } outputsInfo : {
    DataInfo "Result" : tensor<1x64x14x14xf16>
  }
// CHECK: func.func @NCEConvWithoutAnyWeightTable([[ARG0:%.+]]: memref<1x32x16x16xf16, {order = #NHWC}, @CMX_NN>, [[ARG1:%.+]]: memref<1x64x14x14xf16, {order = #NHWC}, @CMX_NN>) -> memref<1x64x14x14xf16, {order = #NHWC}, @CMX_NN> {
func.func @NCEConvWithoutAnyWeightTable(%arg0 : tensor<1x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x64x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}> {
    %weights = const.Declare tensor<64x32x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}> = dense<1.000000e+00> : tensor<64x32x3x3xf16, {mem_space = @CMX_NN}>, [#const.Reorder<#NHWC>]

    %out = VPU.NCE.Convolution(%arg0, %weights) rawFilterShape [64, 32, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            ppe = #VPU.PPEStub<>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,

            strides = [1, 1]
        } : tensor<1x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>, tensor<64x32x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x64x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}> {
            VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 32, 16, 16] pad [0, 0, 0, 0] #VPU.mpe_mode<VECTOR_FP16>
        }

    return %out : tensor<1x64x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>

    // CHECK-DAG:       [[WEIGHTS:%.+]] = const.Declare memref<64x32x3x3xf16, {order = #NHWC}, @CMX_NN> = dense<1.000000e+00>
    // CHECK-SAME:      : tensor<64x32x3x3xf16, {mem_space = @CMX_NN}>, [#const.Reorder<#NHWC>]

    // CHECK:       [[OUT_BUF:%.+]] = memref.alloc() : memref<1x64x14x14xf16, {order = #NHWC}, @CMX_NN>
    // CHECK:       [[OUT:%.+]] = VPUIP.NCEClusterTask
    // CHECK-SAME:          {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [3, 3], kernel_strides = [1, 1],
    // CHECK-SAME:          task_type = #VPUIP.nce_task_type<CONV>}
    // CHECK-SAME:          input([[ARG0]]
    // CHECK-SAME:          weights([[WEIGHTS]]
    // CHECK-SAME:          parent_input([[ARG0]]
    // CHECK-SAME:          parent_output([[OUT_BUF]]
    // CHECK-SAME:          outputs([[OUT_BUF]]
    // CHECK-SAME:      -> memref<1x64x14x14xf16, {order = #NHWC}, @CMX_NN> variants :  {
    // CHECK:               DPUTask {mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [15, 15, 31], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    // CHECK:           } PPE :  {
    // CHECK:           }

    // CHECK:       [[RESULT:%.+]] = VPUIP.Copy inputs([[OUT]] : memref<1x64x14x14xf16, {order = #NHWC}, @CMX_NN>) outputs([[ARG1]] : memref<1x64x14x14xf16, {order = #NHWC}, @CMX_NN>)
    // CHECK-SAME:      -> memref<1x64x14x14xf16, {order = #NHWC}, @CMX_NN>
    // CHECK:       return [[RESULT]] : memref<1x64x14x14xf16, {order = #NHWC}, @CMX_NN>
}
}
