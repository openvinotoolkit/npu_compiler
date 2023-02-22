//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --compute-se-sizes  %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func @Conv(%input: memref<1x32x56x56xf16, #NHWC, [@CMX_NN, 0]>, %input_sm: memref<1x32x56x56xi1, #NHWC, [@CMX_NN, 0]>,
           %output: memref<1x64x56x56xf16, #NHWC, [@CMX_NN, 0]>, %output_sm: memref<1x64x56x56xi1, #NHWC, [@CMX_NN, 0]>)
        -> (memref<1x64x56x56xf16, #NHWC, [@CMX_NN, 0]>, memref<1x64x56x56xi1, #NHWC, [@CMX_NN, 0]>) {
    %weights = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>
    %weights_table = VPURT.DeclareBuffer "CMX_NN" [0] <512> -> memref<16x1x1x4xsi32, [@CMX_NN, 0]>
    VPURT.Task attributes {isTrailingSWLayer = false} {
        %2:2 = VPUIP.NCEClusterTask {kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, kernel_size = [1, 1], kernel_strides = [1, 1], out_channel_offset = 0 : i64, task_type = "CONV"}
        input(%input : memref<1x32x56x56xf16, #NHWC, [@CMX_NN, 0]>)
        input_sparsity_map(%input_sm : memref<1x32x56x56xi1, #NHWC, [@CMX_NN, 0]>)
        weights(%weights : memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>)
        weight_table(%weights_table : memref<16x1x1x4xsi32, [@CMX_NN, 0]>)
        parent_input(%input : memref<1x32x56x56xf16, #NHWC, [@CMX_NN, 0]>)
        parent_input_sparsity_map(%input_sm : memref<1x32x56x56xi1, #NHWC, [@CMX_NN, 0]>)
        parent_output(%output : memref<1x64x56x56xf16, #NHWC, [@CMX_NN, 0]>)
        parent_output_sparsity_map(%output_sm : memref<1x64x56x56xi1, #NHWC, [@CMX_NN, 0]>)
        outputs(%output : memref<1x64x56x56xf16, #NHWC, [@CMX_NN, 0]>)
        output_sparsity_map(%output_sm : memref<1x64x56x56xi1, #NHWC, [@CMX_NN, 0]>)
        -> memref<1x64x56x56xf16, #NHWC, [@CMX_NN, 0]>, memref<1x64x56x56xi1, #NHWC, [@CMX_NN, 0]> variants : {
            DPUTask {cluster_id = 0 : i64, mpe_mode = "VECTOR_FP16", outEnd = [56, 56, 63], outStart = [0, 0, 0], pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}}
        } PPE : {
        }
    }
    return %output, %output_sm : memref<1x64x56x56xf16, #NHWC, [@CMX_NN, 0]>, memref<1x64x56x56xi1, #NHWC, [@CMX_NN, 0]>

    // CHECK:       VPUIP.NCEClusterTask
    // CHECK-SAME:    input_se_size = 32 : i64
    // CHECK-SAME:    output_se_size = 64 : i64
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func @ConvMultipleVariants(%input: memref<1x32x56x56xf16, #NHWC, [@CMX_NN, 0]>, %input_sm: memref<1x32x56x56xi1, #NHWC, [@CMX_NN, 0]>,
                           %output: memref<1x64x56x56xf16, #NHWC, [@CMX_NN, 0]>, %output_sm: memref<1x64x56x56xi1, #NHWC, [@CMX_NN, 0]>)
        -> (memref<1x64x56x56xf16, #NHWC, [@CMX_NN, 0]>, memref<1x64x56x56xi1, #NHWC, [@CMX_NN, 0]>) {
    %weights = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>
    %weights_table = VPURT.DeclareBuffer "CMX_NN" [0] <512> -> memref<16x1x1x4xsi32, [@CMX_NN, 0]>
    VPURT.Task attributes {isTrailingSWLayer = false} {
        %2:2 = VPUIP.NCEClusterTask {kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, kernel_size = [1, 1], kernel_strides = [1, 1], out_channel_offset = 0 : i64, task_type = "CONV"}
        input(%input : memref<1x32x56x56xf16, #NHWC, [@CMX_NN, 0]>)
        input_sparsity_map(%input_sm : memref<1x32x56x56xi1, #NHWC, [@CMX_NN, 0]>)
        weights(%weights : memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>)
        weight_table(%weights_table : memref<16x1x1x4xsi32, [@CMX_NN, 0]>)
        parent_input(%input : memref<1x32x56x56xf16, #NHWC, [@CMX_NN, 0]>)
        parent_input_sparsity_map(%input_sm : memref<1x32x56x56xi1, #NHWC, [@CMX_NN, 0]>)
        parent_output(%output : memref<1x64x56x56xf16, #NHWC, [@CMX_NN, 0]>)
        parent_output_sparsity_map(%output_sm : memref<1x64x56x56xi1, #NHWC, [@CMX_NN, 0]>)
        outputs(%output : memref<1x64x56x56xf16, #NHWC, [@CMX_NN, 0]>)
        output_sparsity_map(%output_sm : memref<1x64x56x56xi1, #NHWC, [@CMX_NN, 0]>)
        -> memref<1x64x56x56xf16, #NHWC, [@CMX_NN, 0]>, memref<1x64x56x56xi1, #NHWC, [@CMX_NN, 0]> variants : {
            DPUTask {cluster_id = 0 : i64, mpe_mode = "VECTOR_FP16", outEnd = [56, 56, 31], outStart = [0, 0, 0], pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}}
            DPUTask {cluster_id = 0 : i64, mpe_mode = "VECTOR_FP16", outEnd = [56, 56, 63], outStart = [0, 0, 32], pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}}
        } PPE : {
        }
    }
    return %output, %output_sm : memref<1x64x56x56xf16, #NHWC, [@CMX_NN, 0]>, memref<1x64x56x56xi1, #NHWC, [@CMX_NN, 0]>

    // CHECK:       VPUIP.NCEClusterTask
    // CHECK-SAME:    input_se_size = 32 : i64
    // CHECK-SAME:    output_se_size = 32 : i64
}
