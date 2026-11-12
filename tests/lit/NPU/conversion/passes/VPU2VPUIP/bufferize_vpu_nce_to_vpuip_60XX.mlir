//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --verify-diagnostics --init-compiler="platform=%platform%" --one-shot-bufferize-VPU-to-VPUIP %s | FileCheck %s
// REQUIRES: platform-NPU6010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NceConvWithSplitWeightTableFormat
// CHECK-SAME: (
// CHECK-SAME: [[INPUT:%.+]]: memref<1x16x16x16xf16, {order = #NHWC}, @CMX_NN>
// CHECK-SAME: [[WEIGHTS:%.+]]: memref<16x16x1x1xf16, {order = #NHWC}, @CMX_NN>
// CHECK-SAME: [[SCALE_TABLE:%.+]]: memref<16x1x1x1xf32, @CMX_NN>, [[BIAS_TABLE:%.+]]: memref<16x1x1x1xf32, @CMX_NN>, [[ZERO_POINTS_TABLE:%.+]]: memref<32x1x1x1xi8, @CMX_NN>
// CHECK-SAME: )
// CHECK-SAME: -> memref<1x16x16x16xf16, {order = #NHWC}, @CMX_NN>
func.func @NceConvWithSplitWeightTableFormat(%arg0: tensor<1x16x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>,
                   %arg1: tensor<16x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>,
                   %arg2: tensor<16x1x1x1xf32, {mem_space = @CMX_NN}>,
                   %arg3: tensor<16x1x1x1xf32, {mem_space = @CMX_NN}>,
                   %arg4: tensor<32x1x1x1xi8, {mem_space = @CMX_NN}>)
                   -> tensor<1x16x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}> {

    %0 = VPU.NCE.Convolution(%arg0, %arg1, %arg2, %arg3, %arg4) rawFilterShape [16, 16, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
                ppe = #VPU.PPEStub<>,
                pad = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,

                strides = [1, 1],
                operandSegmentSizes = array<i32: 1, 1, 0, 1, 1, 1, 0>
            } : tensor<1x16x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>, tensor<16x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>, tensor<16x1x1x1xf32, {mem_space = @CMX_NN}>, tensor<16x1x1x1xf32, {mem_space = @CMX_NN}>, tensor<32x1x1x1xi8, {mem_space = @CMX_NN}> -> tensor<1x16x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}> {
        VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 16, 16, 16] pad [0, 0, 0, 0] #VPU.mpe_mode<VECTOR_FP16>
    }

    return %0 : tensor<1x16x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>

    // CHECK: [[OUT_BUF:%.+]] = memref.alloc() : memref<1x16x16x16xf16, {order = #NHWC}, @CMX_NN>

    // CHECK:       VPUIP.NCEClusterTask
    // CHECK-SAME:      task_type = #VPUIP.nce_task_type<CONV>
    // CHECK-SAME:  input([[INPUT]] : memref<1x16x16x16xf16, {order = #NHWC}, @CMX_NN>)
    // CHECK-SAME:  weights([[WEIGHTS]] : memref<16x16x1x1xf16, {order = #NHWC}, @CMX_NN>)
    // CHECK-SAME:  weight_table_scale([[SCALE_TABLE]] : memref<16x1x1x1xf32, @CMX_NN>)
    // CHECK-SAME:  weight_table_bias([[BIAS_TABLE]] : memref<16x1x1x1xf32, @CMX_NN>)
    // CHECK-SAME:  weight_zero_points([[ZERO_POINTS_TABLE]] : memref<32x1x1x1xi8, @CMX_NN>)
    // CHECK-SAME:  parent_input([[INPUT]] : memref<1x16x16x16xf16, {order = #NHWC}, @CMX_NN>)
    // CHECK-SAME:  parent_output([[OUT_BUF]] : memref<1x16x16x16xf16, {order = #NHWC}, @CMX_NN>)
    // CHECK-SAME:  outputs([[OUT_BUF]] : memref<1x16x16x16xf16, {order = #NHWC}, @CMX_NN>)

    // CHECK: return
    // CHECK-SAME: memref<1x16x16x16xf16, {order = #NHWC}, @CMX_NN>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NceConvWithoutAnyWeightTable
// CHECK-SAME: (
// CHECK-SAME: [[INPUT:%.+]]: memref<1x16x16x16xf16, {order = #NHWC}, @CMX_NN>
// CHECK-SAME: [[WEIGHTS:%.+]]: memref<16x16x1x1xf16, {order = #NHWC}, @CMX_NN>
// CHECK-SAME: )
// CHECK-SAME: -> memref<1x16x16x16xf16, {order = #NHWC}, @CMX_NN>
func.func @NceConvWithoutAnyWeightTable(%arg0: tensor<1x16x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>,
                   %arg1: tensor<16x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>)
                   -> tensor<1x16x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}> {

    %0 = VPU.NCE.Convolution(%arg0, %arg1) rawFilterShape [16, 16, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
                ppe = #VPU.PPEStub<>,
                pad = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,

                strides = [1, 1]
            } : tensor<1x16x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>, tensor<16x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x16x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}> {
        VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 16, 16, 16] pad [0, 0, 0, 0] #VPU.mpe_mode<VECTOR_FP16>
    }

    return %0 : tensor<1x16x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>

    // CHECK: [[OUT_BUF:%.+]] = memref.alloc() : memref<1x16x16x16xf16, {order = #NHWC}, @CMX_NN>

    // CHECK:       VPUIP.NCEClusterTask
    // CHECK-SAME:      task_type = #VPUIP.nce_task_type<CONV>
    // CHECK-SAME:  input([[INPUT]] : memref<1x16x16x16xf16, {order = #NHWC}, @CMX_NN>)
    // CHECK-SAME:  weights([[WEIGHTS]] : memref<16x16x1x1xf16, {order = #NHWC}, @CMX_NN>)
    // CHECK-SAME:  parent_input([[INPUT]] : memref<1x16x16x16xf16, {order = #NHWC}, @CMX_NN>)
    // CHECK-SAME:  parent_output([[OUT_BUF]] : memref<1x16x16x16xf16, {order = #NHWC}, @CMX_NN>)
    // CHECK-SAME:  outputs([[OUT_BUF]] : memref<1x16x16x16xf16, {order = #NHWC}, @CMX_NN>)

    // CHECK: return
    // CHECK-SAME: memref<1x16x16x16xf16, {order = #NHWC}, @CMX_NN>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<i8:f16:0, {1.000000e+00:1,1.000000e+00:1,1.000000e+00:1,1.000000e+00:1,1.000000e+00:1,1.000000e+00:1,1.000000e+00:1,1.000000e+00:1,1.000000e+00:2,1.000000e+00:2,1.000000e+00:2,1.000000e+00:2,1.000000e+00:2,1.000000e+00:2,1.000000e+00:2,1.000000e+00:3}>

// CHECK-LABEL: @NceConvWithZeroPointTable
// CHECK-SAME: (
// CHECK-SAME: [[INPUT:%.+]]: memref<1x16x16x16xf16, {order = #NHWC}, @CMX_NN>,
// CHECK-SAME: [[WEIGHTS:%.+]]: memref<16x16x1x1x!qElemType, {order = #NHWC}, @CMX_NN>,
// CHECK-SAME: [[SCALE_TABLE:%.+]]: memref<16x1x1x1xf32, @CMX_NN>,
// CHECK-SAME: [[BIAS_TABLE:%.+]]: memref<16x1x1x1xf32, @CMX_NN>
// CHECK-SAME: )
// CHECK-SAME: -> memref<1x16x16x16xf16, {order = #NHWC}, @CMX_NN>
func.func @NceConvWithZeroPointTable(%arg0: tensor<1x16x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>,
                   %arg1: tensor<16x16x1x1x!qElemType, {mem_space = @CMX_NN, order = #NHWC}>,
                   %arg2: tensor<16x1x1x1xf32, {mem_space = @CMX_NN}>,
                   %arg3: tensor<16x1x1x1xf32, {mem_space = @CMX_NN}>)
                   -> tensor<1x16x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}> {

    %zero_points = const.Declare tensor<16x1x1x1xsi8> = dense<[[[[1]]], [[[1]]], [[[1]]], [[[1]]], [[[1]]], [[[1]]], [[[1]]], [[[1]]], [[[2]]], [[[2]]], [[[2]]], [[[2]]], [[[2]]], [[[2]]], [[[2]]], [[[3]]]]> : tensor<16x1x1x1xsi8>
    %0 = VPU.ZeroPointTable(%zero_points) {workloadSizes = [16], zeroPointTableData = [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]} : tensor<16x1x1x1xsi8> -> tensor<32x1x1x1xi8>

    %2 = VPU.Copy(%0) {out_mem_space = @CMX_NN} : tensor<32x1x1x1xi8> -> tensor<32x1x1x1xi8, {mem_space = @CMX_NN}>

    %1 = VPU.NCE.Convolution(%arg0, %arg1, %arg2, %arg3, %2) rawFilterShape [16, 16, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
                ppe = #VPU.PPEStub<>,
                pad = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,

                strides = [1, 1],
                operandSegmentSizes = array<i32: 1, 1, 0, 1, 1, 1, 0>
            } : tensor<1x16x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>, tensor<16x16x1x1x!qElemType, {mem_space = @CMX_NN, order = #NHWC}>, tensor<16x1x1x1xf32, {mem_space = @CMX_NN}>, tensor<16x1x1x1xf32, {mem_space = @CMX_NN}>, tensor<32x1x1x1xi8, {mem_space = @CMX_NN}> -> tensor<1x16x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}> {
        VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 16, 16, 16] pad [0, 0, 0, 0] #VPU.mpe_mode<VECTOR_FP16>
    }

    return %1 : tensor<1x16x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>

    // Verify ZeroPointTable is converted to a constant with values: [1,1,1,1,1,1,1,1, 2,2,2,2,2,2,2,3, -1,-1,-1,-1,-1,-1,-1,-1, -1,-1,-1,-1,-1,-1,-1,-1]
    // Note: {{\[}} escapes [ to avoid FileCheck's [[ variable syntax
    // CHECK-DAG:   [[CST_ZERO_POINTS:%.+]] = const.Declare memref<32x1x1x1xi8> = dense<{{\[}}{{\[}}{{\[}}{{\[}}1]]],{{.*}}{{\[}}{{\[}}{{\[}}2]]],{{.*}}{{\[}}{{\[}}{{\[}}-1]]]]> : tensor<32x1x1x1xi8>

    // CHECK:       [[ZERO_POINTS_BUF:%.+]] = memref.alloc() : memref<32x1x1x1xi8, @CMX_NN>
    // CHECK:       [[ZERO_POINTS:%.+]] = VPUIP.Copy inputs([[CST_ZERO_POINTS]] : memref<32x1x1x1xi8>)
    // CHECK-SAME:      outputs([[ZERO_POINTS_BUF]] : memref<32x1x1x1xi8, @CMX_NN>)

    // CHECK:       [[OUT_BUF:%.+]] = memref.alloc() : memref<1x16x16x16xf16, {order = #NHWC}, @CMX_NN>

    // CHECK:       VPUIP.NCEClusterTask
    // CHECK-SAME:      task_type = #VPUIP.nce_task_type<CONV>
    // CHECK-SAME:  input([[INPUT]] : memref<1x16x16x16xf16, {order = #NHWC}, @CMX_NN>)
    // CHECK-SAME:  weights([[WEIGHTS]] : memref<16x16x1x1x!qElemType, {order = #NHWC}, @CMX_NN>)
    // CHECK-SAME:  weight_table_scale([[SCALE_TABLE]] : memref<16x1x1x1xf32, @CMX_NN>)
    // CHECK-SAME:  weight_table_bias([[BIAS_TABLE]] : memref<16x1x1x1xf32, @CMX_NN>)
    // CHECK-SAME:  weight_zero_points([[ZERO_POINTS]] : memref<32x1x1x1xi8, @CMX_NN>)
    // CHECK-SAME:  parent_input([[INPUT]] : memref<1x16x16x16xf16, {order = #NHWC}, @CMX_NN>)
    // CHECK-SAME:  parent_output([[OUT_BUF]] : memref<1x16x16x16xf16, {order = #NHWC}, @CMX_NN>)
    // CHECK-SAME:  outputs([[OUT_BUF]] : memref<1x16x16x16xf16, {order = #NHWC}, @CMX_NN>)

    // CHECK: return
    // CHECK-SAME: memref<1x16x16x16xf16, {order = #NHWC}, @CMX_NN>
}
