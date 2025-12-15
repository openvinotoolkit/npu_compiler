//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% allow-custom-values=true enable-auto-padding-odu" --patch-weight-table %s | FileCheck %s
// REQUIRES: arch-NPU50XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @PatchWeightTableWeightsOnlyAutopad
func.func @PatchWeightTableWeightsOnlyAutopad() -> memref<16x1x1x4xsi32, [@CMX_NN, 0]> {
    %weight_table = VPURT.DeclareBuffer <CMX_NN> [0] <1024> -> memref<16x1x1x4xsi32, [@CMX_NN, 0]>
    %weights = VPURT.DeclareBuffer <CMX_NN> [0] <1024> -> memref<3x4x1x1xf16, #NHWC, [@CMX_NN, 0]>
    %weight_table_const = const.Declare memref<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>

    %in = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x4x16x16xf16, #NHWC, [@CMX_NN, 0]>
    %out = VPURT.DeclareBuffer <CMX_NN> [0] <428608> -> memref<1x3x16x16xf16, #NHWC, [@CMX_NN, 0]>
    %4 = VPUIP.NNDMA inputs(%weight_table_const : memref<16x1x1x4xsi32>) outputs(%weight_table : memref<16x1x1x4xsi32, [@CMX_NN, 0]>) -> memref<16x1x1x4xsi32, [@CMX_NN, 0]>
    %5 = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>} input(%in : memref<1x4x16x16xf16, #NHWC, [@CMX_NN, 0]>) weights(%weights : memref<3x4x1x1xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%weight_table : memref<16x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%in : memref<1x4x16x16xf16, #NHWC, [@CMX_NN, 0]>) parent_output(%out : memref<1x3x16x16xf16, #NHWC, [@CMX_NN, 0]>) outputs(%out : memref<1x3x16x16xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x3x16x16xf16, #NHWC, [@CMX_NN, 0]> variants : {
      DPUTask {outEnd = [1, 1, 1], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0]}
    } PPE :  {
    }

    return %weight_table : memref<16x1x1x4xsi32, [@CMX_NN, 0]>

    // CHECK:       [[WEIGHT_TABLE_BUF:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <1024> -> memref<16x1x1x4xsi32, [@CMX_NN, 0]>
    // CHECK:       [[WEIGHTS_BUF:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <[[WEIGHTS_ADDR:[^>]+]]> -> memref<3x4x1x1xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK-DAG:       [[CONST:%.*]] = const.Declare memref<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>, [#const.RelocateWeightsTable<weightsPtr=[[[WEIGHTS_ADDR]]], sparsityPtr=16777215 : i64, offsets=[0], weightsTableSize=256 : i64, weightsElemBitSize=16 : i64, channelOffset=0 : i64, originalOC=3 : i64>]
    // CHECK:       [[NDMA_OP:.*]] = VPUIP.NNDMA inputs([[CONST]] : memref<16x1x1x4xsi32>) outputs([[WEIGHT_TABLE_BUF]] : memref<16x1x1x4xsi32, [@CMX_NN, 0]>) -> memref<16x1x1x4xsi32, [@CMX_NN, 0]>
    // CHECK:       [[NCE_CLUST_TASK_OP:.*]] = VPUIP.NCEClusterTask
    // CHECK-SAME:  weights([[WEIGHTS_BUF]] : memref<3x4x1x1xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK-SAME:  weight_table([[WEIGHT_TABLE_BUF]] : memref<16x1x1x4xsi32, [@CMX_NN, 0]>)
}
