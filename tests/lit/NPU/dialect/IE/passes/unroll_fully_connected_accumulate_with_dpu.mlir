//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --unroll-fully-connected="accumulate-matmul-with-dpu=true" %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
// -----

#CN = affine_map<(d0, d1) -> (d1, d0)>

// CHECK-LABEL: @AccumulateMatmulWithDPU
// CHECK-SAME:   [[LHS_1:%arg[0-9]]]: tensor<1024x3584xf32>,
// CHECK-SAME:   [[WEIGHTS:%arg[0-9]]]: tensor<1x896x512xf32>,
// CHECK-SAME:   [[IN_PARAM:%arg[0-9]]]: tensor<1x1x1xf32>,
// CHECK-SAME:   [[OUT_PARAM:%arg[0-9]]]: tensor<1x1x512xf32>
func.func @AccumulateMatmulWithDPU(%LHS_1: tensor<1024x3584xf32>,
                        %WEIGHTS: tensor<1x896x512xf32>,
                        %IN_PARAM: tensor<1x1x1xf32>,
                        %OUT_PARAM: tensor<1x1x512xf32>) -> tensor<1024x512xf32> {
    %RHS_1 = IE.FakeQuantize(%WEIGHTS, %IN_PARAM, %IN_PARAM, %OUT_PARAM, %OUT_PARAM) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 16 : i64
    } : tensor<1x896x512xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<1x1x512xf32>, tensor<1x1x512xf32> -> tensor<1x896x512xf32>
    %RHS_2 = IE.FakeQuantize(%WEIGHTS, %IN_PARAM, %IN_PARAM, %OUT_PARAM, %OUT_PARAM) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 16 : i64
    } : tensor<1x896x512xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<1x1x512xf32>, tensor<1x1x512xf32> -> tensor<1x896x512xf32>
    %RHS_3 = IE.FakeQuantize(%WEIGHTS, %IN_PARAM, %IN_PARAM, %OUT_PARAM, %OUT_PARAM) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 16 : i64
    } : tensor<1x896x512xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<1x1x512xf32>, tensor<1x1x512xf32> -> tensor<1x896x512xf32>
    %RHS_4 = IE.FakeQuantize(%WEIGHTS, %IN_PARAM, %IN_PARAM, %OUT_PARAM, %OUT_PARAM) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 16 : i64
    } : tensor<1x896x512xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<1x1x512xf32>, tensor<1x1x512xf32> -> tensor<1x896x512xf32>
    // CHECK:   [[RHS_1:%.+]] = IE.FakeQuantize
    // CHECK:   [[RHS_2:%.+]] = IE.FakeQuantize
    // CHECK:   [[RHS_3:%.+]] = IE.FakeQuantize
    // CHECK:   [[RHS_4:%.+]] = IE.FakeQuantize

    %CONCAT_RHS = IE.Concat(%RHS_1, %RHS_2, %RHS_3, %RHS_4) {
        per_axis = #IE.Concat<axis = 0 : i64>
    } : tensor<1x896x512xf32>, tensor<1x896x512xf32>, tensor<1x896x512xf32>, tensor<1x896x512xf32> -> tensor<4x896x512xf32>
    // CHECK-NOT:   IE.Concat

    %RESHAPE_RHS = IE.AffineReshape(%CONCAT_RHS) {
        dim_mapping = [[0], [0], [1]],
        shape_value = [3584, 512]
    } : tensor<4x896x512xf32> -> tensor<3584x512xf32>
    // CHECK:   [[RESHAPE_RHS_1:%.+]] = IE.Reshape([[RHS_1]]) {
    // CHECK-SAME:      shape_value = [896, 512]
    // CHECK-SAME:  } : tensor<1x896x512xf32> -> tensor<896x512xf32>
    // CHECK:   [[RESHAPE_RHS_2:%.+]] = IE.Reshape([[RHS_2]]) {
    // CHECK-SAME:      shape_value = [896, 512]
    // CHECK-SAME:  } : tensor<1x896x512xf32> -> tensor<896x512xf32>
    // CHECK:   [[RESHAPE_RHS_3:%.+]] = IE.Reshape([[RHS_3]]) {
    // CHECK-SAME:      shape_value = [896, 512]
    // CHECK-SAME:  } : tensor<1x896x512xf32> -> tensor<896x512xf32>
    // CHECK:   [[RESHAPE_RHS_4:%.+]] = IE.Reshape([[RHS_4]]) {
    // CHECK-SAME:      shape_value = [896, 512]
    // CHECK-SAME:  } : tensor<1x896x512xf32> -> tensor<896x512xf32>

    %TRANSPOSE_RHS = IE.Transpose(%RESHAPE_RHS) {
        order_value = #CN
    } : tensor<3584x512xf32> -> tensor<512x3584xf32>

    %GEMM = IE.FullyConnected(%LHS_1, %TRANSPOSE_RHS) : tensor<1024x3584xf32>, tensor<512x3584xf32> -> tensor<1024x512xf32>
    // CHECK:   [[LHS_SLICE_1:%.+]] = IE.Slice [[LHS_1]] [0, 0] [1024, 896]
    // CHECK:   [[LHS_SLICE_2:%.+]] = IE.Slice [[LHS_1]] [0, 896] [1024, 896]
    // CHECK:   [[LHS_SLICE_3:%.+]] = IE.Slice [[LHS_1]] [0, 1792] [1024, 896]
    // CHECK:   [[LHS_SLICE_4:%.+]] = IE.Slice [[LHS_1]] [0, 2688] [1024, 896]

    // CHECK:   [[TRANSPOSE_1:%.+]] = IE.Transpose([[RESHAPE_RHS_1]])
    // CHECK:   [[GEMM_1:%.+]] = IE.FullyConnected([[LHS_SLICE_1]], [[TRANSPOSE_1]])
    // CHECK:   [[TRANSPOSE_2:%.+]] = IE.Transpose([[RESHAPE_RHS_2]])
    // CHECK:   [[GEMM_2:%.+]] = IE.FullyConnected([[LHS_SLICE_2]], [[TRANSPOSE_2]])
    // CHECK:   [[TRANSPOSE_3:%.+]] = IE.Transpose([[RESHAPE_RHS_3]])
    // CHECK:   [[GEMM_3:%.+]] = IE.FullyConnected([[LHS_SLICE_3]], [[TRANSPOSE_3]])
    // CHECK:   [[TRANSPOSE_4:%.+]] = IE.Transpose([[RESHAPE_RHS_4]])
    // CHECK:   [[GEMM_4:%.+]] = IE.FullyConnected([[LHS_SLICE_4]], [[TRANSPOSE_4]])

    // CHECK:   [[ADD_1:%.+]] = IE.Add([[GEMM_1]], [[GEMM_2]])
    // CHECK:   [[ADD_2:%.+]] = IE.Add([[ADD_1]], [[GEMM_3]])
    // CHECK:   [[ADD_3:%.+]] = IE.Add([[ADD_2]], [[GEMM_4]])

    return %GEMM : tensor<1024x512xf32>
    // CHECK:   return [[ADD_3]] : tensor<1024x512xf32>
}
