//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --adjust-input-data-for-explicit-se-table %s | FileCheck %s
// REQUIRES: arch-NPU40XX
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @SparseConvSETable() -> memref<1x16x80x288xf16, #NHWC, [@CMX_NN, 0]> {
    %input = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x36x144xf16, #NHWC, [@CMX_NN, 0]>
    %input_sm = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x73x289xi1, #NHWC, [@CMX_NN, 0]>
    %input_se = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x1x73x289xi32, #NHWC, [@CMX_NN, 0]>
    %weights = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<16x16x2x2xf16, #NHWC, [@CMX_NN, 0]>
    %weights_table = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<16x1x1x4xsi32, [@CMX_NN, 0]>
    %output = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x80x288xf16, #NHWC, [@CMX_NN, 0]>
    %parent_input = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x36x144xf16, #NHWC, [@CMX_NN, 0]>
    %parent_input_sm = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x73x289xi1, #NHWC, [@CMX_NN, 0]>
    %parent_input_se = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x1x73x289xi32, #NHWC, [@CMX_NN, 0]>
    %parent_output = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x80x288xf16, #NHWC, [@CMX_NN, 0]>

    VPURT.Task attributes {isTrailingSWLayer = false} {
      %out = VPUIP.NCEClusterTask {input_se_size = 16 : i64, is_segmented, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [2, 2], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}
              input(%input : memref<1x16x36x144xf16, #NHWC, [@CMX_NN, 0]>)
              input_sparsity_map(%input_sm : memref<1x16x73x289xi1, #NHWC, [@CMX_NN, 0]>)
              input_storage_element_table(%input_se : memref<1x1x73x289xi32, #NHWC, [@CMX_NN, 0]>)
              weights(%weights : memref<16x16x2x2xf16, #NHWC, [@CMX_NN, 0]>)
              weight_table(%weights_table : memref<16x1x1x4xsi32, [@CMX_NN, 0]>)
              parent_input(%parent_input : memref<1x16x36x144xf16, #NHWC, [@CMX_NN, 0]>)
              parent_input_sparsity_map(%parent_input_sm : memref<1x16x73x289xi1, #NHWC, [@CMX_NN, 0]>)
              parent_input_storage_element_table(%parent_input_se : memref<1x1x73x289xi32, #NHWC, [@CMX_NN, 0]>)
              parent_output(%parent_output : memref<1x16x80x288xf16, #NHWC, [@CMX_NN, 0]>)
              outputs(%output : memref<1x16x80x288xf16, #NHWC, [@CMX_NN, 0]>)
      -> memref<1x16x80x288xf16, #NHWC, [@CMX_NN, 0]> variants : {
        DPUTask {cluster_id = 0 : i64, inEnd = [288, 80, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [287, 79, 15], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }

    return %output : memref<1x16x80x288xf16, #NHWC, [@CMX_NN, 0]>

    // CHECK:       [[INPUT:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x73x289xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK:       [[PARENT_INPUT:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x73x289xf16, #NHWC, [@CMX_NN, 0]>

    // CHECK:       VPUIP.NCEClusterTask
    // CHECK-NOT:      input({{%.+}} : memref<1x16x36x144xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK:          input([[INPUT]] : memref<1x16x73x289xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK-NOT:      parent_input({{%.+}} : memref<1x16x36x144xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK:          parent_input([[PARENT_INPUT]] : memref<1x16x73x289xf16, #NHWC, [@CMX_NN, 0]>)
}
