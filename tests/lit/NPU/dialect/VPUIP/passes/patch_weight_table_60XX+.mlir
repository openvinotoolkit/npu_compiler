//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% allow-custom-values=true" --patch-weight-table %s | FileCheck %s
// REQUIRES: platform-NPU6010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @PatchWeightTable
module @PatchWeightTable {

func.func @main() {
    %weights = VPURT.DeclareBuffer <CMX_NN> [0] <395136> -> memref<1008x64x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>

    %scale_table = VPURT.DeclareBuffer <CMX_NN> [0] <540288> -> memref<1008x1x1x1xf32, [@CMX_NN, 0]>
    %scale_table_const = const.Declare memref<1008x1x1x1xf32> = dense<1.0> : tensor<1008x1x1x1xf32>

    %bias_table = VPURT.DeclareBuffer <CMX_NN> [0] <544320> -> memref<1008x1x1x1xf32, [@CMX_NN, 0]>
    %bias_table_const = const.Declare memref<1008x1x1x1xf32> = dense<1.0> : tensor<1008x1x1x1xf32>

    %in = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x1008x14x14xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %out = VPURT.DeclareBuffer <CMX_NN> [0] <548352> -> memref<1x1008x2x2xf16, {order = #NHWC}, [@CMX_NN, 0]>


    %nndma_op_1 = VPUIP.NNDMA inputs(%scale_table_const : memref<1008x1x1x1xf32>) outputs(%scale_table : memref<1008x1x1x1xf32, [@CMX_NN, 0]>) -> memref<1008x1x1x1xf32, [@CMX_NN, 0]>
    %nndma_op_2 = VPUIP.NNDMA inputs(%bias_table_const : memref<1008x1x1x1xf32>) outputs(%bias_table : memref<1008x1x1x1xf32, [@CMX_NN, 0]>) -> memref<1008x1x1x1xf32, [@CMX_NN, 0]>

    %NCEOp = VPUIP.NCEClusterTask <{resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [7, 7], kernel_strides = [7, 7], task_type = #VPUIP.nce_task_type<CONV>}> input(%in : memref<1x1008x14x14xf16, {order = #NHWC}, [@CMX_NN, 0]>) weights(%weights : memref<1008x64x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) weight_table_scale(%scale_table : memref<1008x1x1x1xf32, [@CMX_NN, 0]>) weight_table_bias(%bias_table : memref<1008x1x1x1xf32, [@CMX_NN, 0]>) parent_input(%in : memref<1x1008x14x14xf16, {order = #NHWC}, [@CMX_NN, 0]>) parent_output(%out : memref<1x1008x2x2xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%out : memref<1x1008x2x2xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x1008x2x2xf16, {order = #NHWC}, [@CMX_NN, 0]> variants :  {
        DPUTask {outEnd = [1, 0, 1007], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0]}
    } PPE :  {
    }

    return

    // CHECK:       [[WEIGHTS_BUF:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <395136> -> memref<1008x64x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK:       [[SCALE_TABLE_BUF:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <540288> -> memref<1008x1x1x1xf32, [@CMX_NN, 0]>
    // CHECK-DAG:       [[SCALE_TABLE_CONST:%.+]] = const.Declare memref<1008x1x1x1xf32> = dense<1.000000e+00> : tensor<1008x1x1x1xf32>
    // CHECK:       [[BIAS_TABLE_BUF:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <544320> -> memref<1008x1x1x1xf32, [@CMX_NN, 0]>
    // CHECK-DAG:       [[BIAS_TABLE_CONST:%.+]] = const.Declare memref<1008x1x1x1xf32> = dense<1.000000e+00> : tensor<1008x1x1x1xf32>
    // CHECK:       [[NDMA_OP_1:.+]] = VPUIP.NNDMA inputs([[SCALE_TABLE_CONST]] : memref<1008x1x1x1xf32>) outputs([[SCALE_TABLE_BUF]] : memref<1008x1x1x1xf32, [@CMX_NN, 0]>) -> memref<1008x1x1x1xf32, [@CMX_NN, 0]>
    // CHECK:       [[NDMA_OP_2:.+]] = VPUIP.NNDMA inputs([[BIAS_TABLE_CONST]] : memref<1008x1x1x1xf32>) outputs([[BIAS_TABLE_BUF]] : memref<1008x1x1x1xf32, [@CMX_NN, 0]>) -> memref<1008x1x1x1xf32, [@CMX_NN, 0]>
    // CHECK:       [[NCE_CLUST_TASK_OP:.+]] = VPUIP.NCEClusterTask
    // CHECK-SAME:  weights([[WEIGHTS_BUF]] : memref<1008x64x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK-SAME:  weight_table_scale([[SCALE_TABLE_BUF]] : memref<1008x1x1x1xf32, [@CMX_NN, 0]>)
    // CHECK-SAME:  weight_table_bias([[BIAS_TABLE_BUF]] : memref<1008x1x1x1xf32, [@CMX_NN, 0]>)
}

}
