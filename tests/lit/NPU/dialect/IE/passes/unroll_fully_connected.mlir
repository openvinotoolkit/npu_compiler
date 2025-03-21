//
// Copyright (C) 2023-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --unroll-fully-connected %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
#CN = affine_map<(d0, d1) -> (d1, d0)>

// CHECK-LABEL: @UnrollMatMul
// CHECK-SAME:   [[LHS_1:%arg0]]: tensor<16x3072xf32>,
// CHECK-SAME:   [[WEIGHTS:%arg1]]: tensor<1x1024x4096xf32>,
// CHECK-SAME:   [[IN_PARAM:%arg2]]: tensor<1x1x1xf32>,
// CHECK-SAME:   [[OUT_PARAM:%arg3]]: tensor<1x1x4096xf32>
func.func @UnrollMatMul(%LHS_1: tensor<16x3072xf32>,
                        %WEIGHTS: tensor<1x1024x4096xf32>,
                        %IN_PARAM: tensor<1x1x1xf32>,
                        %OUT_PARAM: tensor<1x1x4096xf32>) -> tensor<16x4096xf32> {
    %RHS_1 = IE.FakeQuantize(%WEIGHTS, %IN_PARAM, %IN_PARAM, %OUT_PARAM, %OUT_PARAM) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 16 : i64
    } : tensor<1x1024x4096xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<1x1x4096xf32>, tensor<1x1x4096xf32> -> tensor<1x1024x4096xf32>
    %RHS_2 = IE.FakeQuantize(%WEIGHTS, %IN_PARAM, %IN_PARAM, %OUT_PARAM, %OUT_PARAM) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 16 : i64
    } : tensor<1x1024x4096xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<1x1x4096xf32>, tensor<1x1x4096xf32> -> tensor<1x1024x4096xf32>
    %RHS_3 = IE.FakeQuantize(%WEIGHTS, %IN_PARAM, %IN_PARAM, %OUT_PARAM, %OUT_PARAM) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 16 : i64
    } : tensor<1x1024x4096xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<1x1x4096xf32>, tensor<1x1x4096xf32> -> tensor<1x1024x4096xf32>
    // CHECK:   [[RHS_1:%.+]] = IE.FakeQuantize
    // CHECK:   [[RHS_2:%.+]] = IE.FakeQuantize
    // CHECK:   [[RHS_3:%.+]] = IE.FakeQuantize

    %CONCAT_RHS = IE.Concat(%RHS_1, %RHS_2, %RHS_3) {
        per_axis = #IE.Concat<axis = 0 : i64>
    } : tensor<1x1024x4096xf32>, tensor<1x1024x4096xf32>, tensor<1x1024x4096xf32> -> tensor<3x1024x4096xf32>
    // CHECK-NOT:   IE.Concat

    %RESHAPE_RHS = IE.AffineReshape(%CONCAT_RHS) {
        dim_mapping = [[0], [0], [1]],
        shape_value = [3072, 4096]
    } : tensor<3x1024x4096xf32> -> tensor<3072x4096xf32>
    // CHECK:   [[RESHAPE_RHS_1:%.+]] = IE.Reshape([[RHS_1]]) {
    // CHECK-SAME:      shape_value = [1024, 4096]
    // CHECK-SAME:  } : tensor<1x1024x4096xf32> -> tensor<1024x4096xf32>
    // CHECK:   [[RESHAPE_RHS_2:%.+]] = IE.Reshape([[RHS_2]]) {
    // CHECK-SAME:      shape_value = [1024, 4096]
    // CHECK-SAME:  } : tensor<1x1024x4096xf32> -> tensor<1024x4096xf32>
    // CHECK:   [[RESHAPE_RHS_3:%.+]] = IE.Reshape([[RHS_3]]) {
    // CHECK-SAME:      shape_value = [1024, 4096]
    // CHECK-SAME:  } : tensor<1x1024x4096xf32> -> tensor<1024x4096xf32>

    %TRANSPOSE_RHS = IE.Transpose(%RESHAPE_RHS) {
        order_value = #CN
    } : tensor<3072x4096xf32> -> tensor<4096x3072xf32>

    %GEMM = IE.FullyConnected(%LHS_1, %TRANSPOSE_RHS) : tensor<16x3072xf32>, tensor<4096x3072xf32> -> tensor<16x4096xf32>
    // CHECK:   [[LHS_SLICE_1:%.+]] = IE.Slice [[LHS_1]] [0, 0] [16, 1024]
    // CHECK:   [[LHS_SLICE_2:%.+]] = IE.Slice [[LHS_1]] [0, 1024] [16, 1024]
    // CHECK:   [[LHS_SLICE_3:%.+]] = IE.Slice [[LHS_1]] [0, 2048] [16, 1024]

    // CHECK:   [[TRANSPOSE_1:%.+]] = IE.Transpose([[RESHAPE_RHS_1]])
    // CHECK:   [[GEMM_1:%.+]] = IE.FullyConnected([[LHS_SLICE_1]], [[TRANSPOSE_1]])
    // CHECK:   [[TRANSPOSE_2:%.+]] = IE.Transpose([[RESHAPE_RHS_2]])
    // CHECK:   [[GEMM_2:%.+]] = IE.FullyConnected([[LHS_SLICE_2]], [[TRANSPOSE_2]])
    // CHECK:   [[TRANSPOSE_3:%.+]] = IE.Transpose([[RESHAPE_RHS_3]])
    // CHECK:   [[GEMM_3:%.+]] = IE.FullyConnected([[LHS_SLICE_3]], [[TRANSPOSE_3]])

    // CHECK:   [[ADD_1:%.+]] = IE.Accumulate([[GEMM_1]], [[GEMM_2]])
    // CHECK:   [[ADD_2:%.+]] = IE.Accumulate([[ADD_1]], [[GEMM_3]])

    return %GEMM : tensor<16x4096xf32>
    // CHECK:   return [[ADD_2]] : tensor<16x4096xf32>
}

// -----

#CN = affine_map<(d0, d1) -> (d1, d0)>

// CHECK-LABEL: @DontUnrollMatMulNot4bit
// CHECK-SAME:   [[LHS_1:%arg0]]: tensor<16x3072xf32>,
func.func @DontUnrollMatMulNot4bit(%LHS_1: tensor<16x3072xf32>,
                        %WEIGHTS: tensor<1x1024x4096xf32>,
                        %IN_PARAM: tensor<1x1x1xf32>,
                        %OUT_PARAM: tensor<1x1x4096xf32>) -> tensor<16x4096xf32> {
    %RHS_1 = IE.FakeQuantize(%WEIGHTS, %IN_PARAM, %IN_PARAM, %OUT_PARAM, %OUT_PARAM) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64
    } : tensor<1x1024x4096xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<1x1x4096xf32>, tensor<1x1x4096xf32> -> tensor<1x1024x4096xf32>
    %RHS_2 = IE.FakeQuantize(%WEIGHTS, %IN_PARAM, %IN_PARAM, %OUT_PARAM, %OUT_PARAM) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64
    } : tensor<1x1024x4096xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<1x1x4096xf32>, tensor<1x1x4096xf32> -> tensor<1x1024x4096xf32>
    %RHS_3 = IE.FakeQuantize(%WEIGHTS, %IN_PARAM, %IN_PARAM, %OUT_PARAM, %OUT_PARAM) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64
    } : tensor<1x1024x4096xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<1x1x4096xf32>, tensor<1x1x4096xf32> -> tensor<1x1024x4096xf32>

    %CONCAT_RHS = IE.Concat(%RHS_1, %RHS_2, %RHS_3) {
        per_axis = #IE.Concat<axis = 0 : i64>
    } : tensor<1x1024x4096xf32>, tensor<1x1024x4096xf32>, tensor<1x1024x4096xf32> -> tensor<3x1024x4096xf32>

    %RESHAPE_RHS = IE.AffineReshape(%CONCAT_RHS) {
        dim_mapping = [[0], [0], [1]],
        shape_value = [3072, 4096]
    } : tensor<3x1024x4096xf32> -> tensor<3072x4096xf32>

    %TRANSPOSE_RHS = IE.Transpose(%RESHAPE_RHS) {
        order_value = #CN
    } : tensor<3072x4096xf32> -> tensor<4096x3072xf32>

    %GEMM = IE.FullyConnected(%LHS_1, %TRANSPOSE_RHS) : tensor<16x3072xf32>, tensor<4096x3072xf32> -> tensor<16x4096xf32>

    return %GEMM : tensor<16x4096xf32>

    // CHECK:   [[CONCAT:%.+]] = IE.Concat
    // CHECK:   [[RESHAPE:%.+]] = IE.AffineReshape([[CONCAT]])
    // CHECK:   [[TRANSPOSE:%.+]] = IE.Transpose([[RESHAPE]])
    // CHECK:   [[GEMM:%.+]] = IE.FullyConnected([[LHS_1]], [[TRANSPOSE]])

    // CHECK:   return [[GEMM]]
}

// -----

#CN = affine_map<(d0, d1) -> (d1, d0)>

// CHECK-LABEL: @DontUnrollMatMulNoPerfBenefit
// CHECK-SAME:   [[LHS_1:%arg0]]: tensor<16x96xf32>,
func.func @DontUnrollMatMulNoPerfBenefit(%LHS_1: tensor<16x96xf32>,
                        %WEIGHTS: tensor<1x32x64xf32>,
                        %IN_PARAM: tensor<1x1x1xf32>,
                        %OUT_PARAM: tensor<1x1x64xf32>) -> tensor<16x64xf32> {
    %RHS_1 = IE.FakeQuantize(%WEIGHTS, %IN_PARAM, %IN_PARAM, %OUT_PARAM, %OUT_PARAM) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 16 : i64
    } : tensor<1x32x64xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<1x1x64xf32>, tensor<1x1x64xf32> -> tensor<1x32x64xf32>
    %RHS_2 = IE.FakeQuantize(%WEIGHTS, %IN_PARAM, %IN_PARAM, %OUT_PARAM, %OUT_PARAM) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 16 : i64
    } : tensor<1x32x64xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<1x1x64xf32>, tensor<1x1x64xf32> -> tensor<1x32x64xf32>
    %RHS_3 = IE.FakeQuantize(%WEIGHTS, %IN_PARAM, %IN_PARAM, %OUT_PARAM, %OUT_PARAM) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 16 : i64
    } : tensor<1x32x64xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<1x1x64xf32>, tensor<1x1x64xf32> -> tensor<1x32x64xf32>

    %CONCAT_RHS = IE.Concat(%RHS_1, %RHS_2, %RHS_3) {
        per_axis = #IE.Concat<axis = 0 : i64>
    } : tensor<1x32x64xf32>, tensor<1x32x64xf32>, tensor<1x32x64xf32> -> tensor<3x32x64xf32>

    %RESHAPE_RHS = IE.AffineReshape(%CONCAT_RHS) {
        dim_mapping = [[0], [0], [1]],
        shape_value = [96, 64]
    } : tensor<3x32x64xf32> -> tensor<96x64xf32>

    %TRANSPOSE_RHS = IE.Transpose(%RESHAPE_RHS) {
        order_value = #CN
    } : tensor<96x64xf32> -> tensor<64x96xf32>

    %GEMM = IE.FullyConnected(%LHS_1, %TRANSPOSE_RHS) : tensor<16x96xf32>, tensor<64x96xf32> -> tensor<16x64xf32>

    return %GEMM : tensor<16x64xf32>

    // CHECK:   [[CONCAT:%.+]] = IE.Concat
    // CHECK:   [[RESHAPE:%.+]] = IE.AffineReshape([[CONCAT]])
    // CHECK:   [[TRANSPOSE:%.+]] = IE.Transpose([[RESHAPE]])
    // CHECK:   [[GEMM:%.+]] = IE.FullyConnected([[LHS_1]], [[TRANSPOSE]])

    // CHECK:   return [[GEMM]]
}

// -----

// CHECK-LABEL: @SkipMatMulWithoutTranspose
// CHECK-SAME:   [[LHS:%.+]]: tensor<16x96xf32>
// CHECK-SAME:   [[RHS:%.+]]: tensor<64x96xf32>
func.func @SkipMatMulWithoutTranspose(%LHS: tensor<16x96xf32>, %RHS: tensor<64x96xf32>) -> tensor<16x64xf32> {
    %GEMM = IE.FullyConnected(%LHS, %RHS) : tensor<16x96xf32>, tensor<64x96xf32> -> tensor<16x64xf32>
    // CHECK:   [[GEMM:%.+]] = IE.FullyConnected([[LHS]], [[RHS]])

    return %GEMM : tensor<16x64xf32>
    // CHECK:   return [[GEMM]] : tensor<16x64xf32>
}

// -----

#CN = affine_map<(d0, d1) -> (d1, d0)>

// CHECK-LABEL: @SkipMatMulWithoutReshape
// CHECK-SAME:   [[LHS:%.+]]: tensor<16x96xf32>
// CHECK-SAME:   [[RHS:%.+]]: tensor<96x64xf32>
func.func @SkipMatMulWithoutReshape(%LHS: tensor<16x96xf32>, %RHS: tensor<96x64xf32>) -> tensor<16x64xf32> {
    %TRANSPOSE_RHS = IE.Transpose(%RHS) {
        order_value = #CN
    } : tensor<96x64xf32> -> tensor<64x96xf32>
    // CHECK:   [[TRANSPOSE_RHS:%.+]] = IE.Transpose([[RHS]])

    %GEMM = IE.FullyConnected(%LHS, %TRANSPOSE_RHS) : tensor<16x96xf32>, tensor<64x96xf32> -> tensor<16x64xf32>
    // CHECK:   [[GEMM:%.+]] = IE.FullyConnected([[LHS]], [[TRANSPOSE_RHS]])

    return %GEMM : tensor<16x64xf32>
    // CHECK:   return [[GEMM]] : tensor<16x64xf32>
}

// -----

#CN = affine_map<(d0, d1) -> (d1, d0)>

// CHECK-LABEL: @SkipMatMulWithUnsupportedReshape
// CHECK-SAME:   [[LHS:%.+]]: tensor<16x96xf32>
// CHECK-SAME:   [[RHS:%.+]]: tensor<2x96x32xf32>
func.func @SkipMatMulWithUnsupportedReshape(%LHS: tensor<16x96xf32>, %RHS: tensor<2x96x32xf32>) -> tensor<16x64xf32> {
    %RESHAPE_RHS = IE.AffineReshape(%RHS) {
        dim_mapping = [[0], [0], [1]],
        shape_value = [96, 64]
    } : tensor<2x96x32xf32> -> tensor<96x64xf32>
    // CHECK:   [[RESHAPE_RHS:%.+]] = IE.AffineReshape([[RHS]])
    // CHECK-SAME:      shape_value = [96, 64]
    // CHECK-SAME:  tensor<2x96x32xf32> -> tensor<96x64xf32>

    %TRANSPOSE_RHS = IE.Transpose(%RESHAPE_RHS) {
        order_value = #CN
    } : tensor<96x64xf32> -> tensor<64x96xf32>
    // CHECK:   [[TRANSPOSE_RHS:%.+]] = IE.Transpose([[RESHAPE_RHS]])

    %GEMM = IE.FullyConnected(%LHS, %TRANSPOSE_RHS) : tensor<16x96xf32>, tensor<64x96xf32> -> tensor<16x64xf32>
    // CHECK:   [[GEMM:%.+]] = IE.FullyConnected([[LHS]], [[TRANSPOSE_RHS]])

    return %GEMM : tensor<16x64xf32>
    // CHECK:   return [[GEMM]] : tensor<16x64xf32>
}

// -----

#CN = affine_map<(d0, d1) -> (d1, d0)>

// CHECK-LABEL: @SkipMatMulWithoutConcat
// CHECK-SAME:   [[LHS:%.+]]: tensor<16x96xf32>
// CHECK-SAME:   [[RHS:%.+]]: tensor<3x32x64xf32>
func.func @SkipMatMulWithoutConcat(%LHS: tensor<16x96xf32>, %RHS: tensor<3x32x64xf32>) -> tensor<16x64xf32> {
    %RESHAPE_RHS = IE.AffineReshape(%RHS) {
        dim_mapping = [[0], [0], [1]],
        shape_value = [96, 64]
    } : tensor<3x32x64xf32> -> tensor<96x64xf32>
    // CHECK:   [[RESHAPE_RHS:%.+]] = IE.AffineReshape([[RHS]])
    // CHECK-SAME:      shape_value = [96, 64]
    // CHECK-SAME:  tensor<3x32x64xf32> -> tensor<96x64xf32>

    %TRANSPOSE_RHS = IE.Transpose(%RESHAPE_RHS) {
        order_value = #CN
    } : tensor<96x64xf32> -> tensor<64x96xf32>
    // CHECK:   [[TRANSPOSE_RHS:%.+]] = IE.Transpose([[RESHAPE_RHS]])

    %GEMM = IE.FullyConnected(%LHS, %TRANSPOSE_RHS) : tensor<16x96xf32>, tensor<64x96xf32> -> tensor<16x64xf32>
    // CHECK:   [[GEMM:%.+]] = IE.FullyConnected([[LHS]], [[TRANSPOSE_RHS]])

    return %GEMM : tensor<16x64xf32>
    // CHECK:   return [[GEMM]] : tensor<16x64xf32>
}

// -----

#CN = affine_map<(d0, d1) -> (d1, d0)>

// CHECK-LABEL: @SkipMatMulWithUnsupportedConcat
// CHECK-SAME:   [[LHS:%.+]]: tensor<16x96xf32>
// CHECK-SAME:   [[RHS_0:%.+]]: tensor<3x16x64xf32>,
// CHECK-SAME:   [[RHS_1:%.+]]: tensor<3x16x64xf32>
func.func @SkipMatMulWithUnsupportedConcat(%LHS: tensor<16x96xf32>,
                                           %RHS_0: tensor<3x16x64xf32>,
                                           %RHS_1: tensor<3x16x64xf32>) -> tensor<16x64xf32> {
    %CONCAT_RHS = IE.Concat(%RHS_0, %RHS_1) {
        per_axis = #IE.Concat<axis = 1 : i64>
    } : tensor<3x16x64xf32>, tensor<3x16x64xf32> -> tensor<3x32x64xf32>
    // CHECK:   [[CONCAT_RHS:%.+]] = IE.Concat([[RHS_0]], [[RHS_1]])

    %RESHAPE_RHS = IE.AffineReshape(%CONCAT_RHS) {
        dim_mapping = [[0], [0], [1]],
        shape_value = [96, 64]
    } : tensor<3x32x64xf32> -> tensor<96x64xf32>
    // CHECK:   [[RESHAPE_RHS:%.+]] = IE.AffineReshape([[CONCAT_RHS]])
    // CHECK-SAME:      shape_value = [96, 64]
    // CHECK-SAME:  tensor<3x32x64xf32> -> tensor<96x64xf32>

    %TRANSPOSE_RHS = IE.Transpose(%RESHAPE_RHS) {
        order_value = #CN
    } : tensor<96x64xf32> -> tensor<64x96xf32>
    // CHECK:   [[TRANSPOSE_RHS:%.+]] = IE.Transpose([[RESHAPE_RHS]])

    %GEMM = IE.FullyConnected(%LHS, %TRANSPOSE_RHS) : tensor<16x96xf32>, tensor<64x96xf32> -> tensor<16x64xf32>
    // CHECK:   [[GEMM:%.+]] = IE.FullyConnected([[LHS]], [[TRANSPOSE_RHS]])

    return %GEMM : tensor<16x64xf32>
    // CHECK:   return [[GEMM]] : tensor<16x64xf32>
}

// -----

#map = affine_map<(d0, d1, d2) -> (d2, d0, d1)>

// CHECK-LABEL: @UnrollMatMulReshapeTranspose
// CHECK-SAME:   [[LHS_1:%arg0]]: tensor<16x3072xf32>,
// CHECK-SAME:   [[WEIGHTS:%arg1]]: tensor<1x1024x4096xf32>,
// CHECK-SAME:   [[IN_PARAM:%arg2]]: tensor<1x1x1xf32>,
// CHECK-SAME:   [[OUT_PARAM:%arg3]]: tensor<1x1x4096xf32>
func.func @UnrollMatMulReshapeTranspose(%LHS_1: tensor<16x3072xf32>,
                                        %WEIGHTS: tensor<1x1024x4096xf32>,
                                        %IN_PARAM: tensor<1x1x1xf32>,
                                        %OUT_PARAM: tensor<1x1x4096xf32>) -> tensor<16x4096xf32> {
    %RHS_1 = IE.FakeQuantize(%WEIGHTS, %IN_PARAM, %IN_PARAM, %OUT_PARAM, %OUT_PARAM) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 16 : i64
    } : tensor<1x1024x4096xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<1x1x4096xf32>, tensor<1x1x4096xf32> -> tensor<1x1024x4096xf32>
    %RHS_2 = IE.FakeQuantize(%WEIGHTS, %IN_PARAM, %IN_PARAM, %OUT_PARAM, %OUT_PARAM) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 16 : i64
    } : tensor<1x1024x4096xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<1x1x4096xf32>, tensor<1x1x4096xf32> -> tensor<1x1024x4096xf32>
    %RHS_3 = IE.FakeQuantize(%WEIGHTS, %IN_PARAM, %IN_PARAM, %OUT_PARAM, %OUT_PARAM) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 16 : i64
    } : tensor<1x1024x4096xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<1x1x4096xf32>, tensor<1x1x4096xf32> -> tensor<1x1024x4096xf32>
    // CHECK:   [[RHS_1:%.+]] = IE.FakeQuantize
    // CHECK:   [[RHS_2:%.+]] = IE.FakeQuantize
    // CHECK:   [[RHS_3:%.+]] = IE.FakeQuantize

    %CONCAT_RHS = IE.Concat(%RHS_1, %RHS_2, %RHS_3) {
        per_axis = #IE.Concat<axis = 0 : i64>
    } : tensor<1x1024x4096xf32>, tensor<1x1024x4096xf32>, tensor<1x1024x4096xf32> -> tensor<3x1024x4096xf32>
    // CHECK-NOT:   IE.Concat

    %TRANSPOSE_RHS = IE.Transpose(%CONCAT_RHS) {
        order_value = #map
    } : tensor<3x1024x4096xf32> -> tensor<4096x3x1024xf32>

    %RESHAPE_RHS = IE.AffineReshape(%TRANSPOSE_RHS) {
        dim_mapping = [[0], [1], [1]],
        shape_value = [4096, 3072]
    } : tensor<4096x3x1024xf32> -> tensor<4096x3072xf32>
    // CHECK:   [[RESHAPE_RHS_1:%.+]] = IE.Reshape([[RHS_1]]) {
    // CHECK-SAME:      shape_value = [1024, 4096]
    // CHECK-SAME:  } : tensor<1x1024x4096xf32> -> tensor<1024x4096xf32>
    // CHECK:   [[RESHAPE_RHS_2:%.+]] = IE.Reshape([[RHS_2]]) {
    // CHECK-SAME:      shape_value = [1024, 4096]
    // CHECK-SAME:  } : tensor<1x1024x4096xf32> -> tensor<1024x4096xf32>
    // CHECK:   [[RESHAPE_RHS_3:%.+]] = IE.Reshape([[RHS_3]]) {
    // CHECK-SAME:      shape_value = [1024, 4096]
    // CHECK-SAME:  } : tensor<1x1024x4096xf32> -> tensor<1024x4096xf32>

    %GEMM = IE.FullyConnected(%LHS_1, %RESHAPE_RHS) : tensor<16x3072xf32>, tensor<4096x3072xf32> -> tensor<16x4096xf32>
    // CHECK:   [[LHS_SLICE_1:%.+]] = IE.Slice [[LHS_1]] [0, 0] [16, 1024]
    // CHECK:   [[LHS_SLICE_2:%.+]] = IE.Slice [[LHS_1]] [0, 1024] [16, 1024]
    // CHECK:   [[LHS_SLICE_3:%.+]] = IE.Slice [[LHS_1]] [0, 2048] [16, 1024]

    // CHECK:   [[TRANSPOSE_1:%.+]] = IE.Transpose([[RESHAPE_RHS_1]])
    // CHECK:   [[GEMM_1:%.+]] = IE.FullyConnected([[LHS_SLICE_1]], [[TRANSPOSE_1]])
    // CHECK:   [[TRANSPOSE_2:%.+]] = IE.Transpose([[RESHAPE_RHS_2]])
    // CHECK:   [[GEMM_2:%.+]] = IE.FullyConnected([[LHS_SLICE_2]], [[TRANSPOSE_2]])
    // CHECK:   [[TRANSPOSE_3:%.+]] = IE.Transpose([[RESHAPE_RHS_3]])
    // CHECK:   [[GEMM_3:%.+]] = IE.FullyConnected([[LHS_SLICE_3]], [[TRANSPOSE_3]])

    // CHECK:   [[ADD_1:%.+]] = IE.Accumulate([[GEMM_1]], [[GEMM_2]])
    // CHECK:   [[ADD_2:%.+]] = IE.Accumulate([[ADD_1]], [[GEMM_3]])

    return %GEMM : tensor<16x4096xf32>
    // CHECK:   return [[ADD_2]] : tensor<16x4096xf32>
}

// -----

// CHECK-LABEL: @UnrollMatMulWithOnlyReshape
// CHECK-SAME:   [[LHS_1:%arg0]]: tensor<16x3072xf32>,
// CHECK-SAME:   [[WEIGHTS:%arg1]]: tensor<2048x1x1024xf32>,
// CHECK-SAME:   [[IN_PARAM:%arg2]]: tensor<1x1x1xf32>,
// CHECK-SAME:   [[OUT_PARAM:%arg3]]: tensor<2048x1x1xf32>
func.func @UnrollMatMulWithOnlyReshape( %LHS_1: tensor<16x3072xf32>,
                                        %WEIGHTS: tensor<2048x1x1024xf32>,
                                        %IN_PARAM: tensor<1x1x1xf32>,
                                        %OUT_PARAM: tensor<2048x1x1xf32>) -> tensor<16x2048xf32> {
    %RHS_1 = IE.FakeQuantize(%WEIGHTS, %IN_PARAM, %IN_PARAM, %OUT_PARAM, %OUT_PARAM) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 16 : i64
    } : tensor<2048x1x1024xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<2048x1x1xf32>, tensor<2048x1x1xf32> -> tensor<2048x1x1024xf32>
    %RHS_2 = IE.FakeQuantize(%WEIGHTS, %IN_PARAM, %IN_PARAM, %OUT_PARAM, %OUT_PARAM) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 16 : i64
    } : tensor<2048x1x1024xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<2048x1x1xf32>, tensor<2048x1x1xf32> -> tensor<2048x1x1024xf32>
    %RHS_3 = IE.FakeQuantize(%WEIGHTS, %IN_PARAM, %IN_PARAM, %OUT_PARAM, %OUT_PARAM) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 16 : i64
    } : tensor<2048x1x1024xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<2048x1x1xf32>, tensor<2048x1x1xf32> -> tensor<2048x1x1024xf32>
    // CHECK:   [[RHS_1:%.+]] = IE.FakeQuantize
    // CHECK:   [[RHS_2:%.+]] = IE.FakeQuantize
    // CHECK:   [[RHS_3:%.+]] = IE.FakeQuantize

    %CONCAT_RHS = IE.Concat(%RHS_1, %RHS_2, %RHS_3) {
        per_axis = #IE.Concat<axis = 1 : i64>
    } : tensor<2048x1x1024xf32>, tensor<2048x1x1024xf32>, tensor<2048x1x1024xf32> -> tensor<2048x3x1024xf32>
    // CHECK-NOT:   IE.Concat

    %RESHAPE_RHS = IE.AffineReshape(%CONCAT_RHS) {
        dim_mapping = [[0], [1], [1]],
        shape_value = [2048, 3072]
    } : tensor<2048x3x1024xf32> -> tensor<2048x3072xf32>
    // CHECK:   [[RESHAPE_RHS_1:%.+]] = IE.Reshape([[RHS_1]]) {
    // CHECK-SAME:      shape_value = [2048, 1024]
    // CHECK-SAME:  } : tensor<2048x1x1024xf32> -> tensor<2048x1024xf32>
    // CHECK:   [[RESHAPE_RHS_2:%.+]] = IE.Reshape([[RHS_2]]) {
    // CHECK-SAME:      shape_value = [2048, 1024]
    // CHECK-SAME:  } : tensor<2048x1x1024xf32> -> tensor<2048x1024xf32>
    // CHECK:   [[RESHAPE_RHS_3:%.+]] = IE.Reshape([[RHS_3]]) {
    // CHECK-SAME:      shape_value = [2048, 1024]
    // CHECK-SAME:  } : tensor<2048x1x1024xf32> -> tensor<2048x1024xf32>

    %GEMM = IE.FullyConnected(%LHS_1, %RESHAPE_RHS) : tensor<16x3072xf32>, tensor<2048x3072xf32> -> tensor<16x2048xf32>
    // CHECK:   [[LHS_SLICE_1:%.+]] = IE.Slice [[LHS_1]] [0, 0] [16, 1024]
    // CHECK:   [[LHS_SLICE_2:%.+]] = IE.Slice [[LHS_1]] [0, 1024] [16, 1024]
    // CHECK:   [[LHS_SLICE_3:%.+]] = IE.Slice [[LHS_1]] [0, 2048] [16, 1024]

    // CHECK-NOT:   IE.Transpose
    // CHECK:   [[GEMM_1:%.+]] = IE.FullyConnected([[LHS_SLICE_1]], [[RESHAPE_RHS_1]])
    // CHECK-NOT:   IE.Transpose
    // CHECK:   [[GEMM_2:%.+]] = IE.FullyConnected([[LHS_SLICE_2]], [[RESHAPE_RHS_2]])
    // CHECK-NOT:   IE.Transpose
    // CHECK:   [[GEMM_3:%.+]] = IE.FullyConnected([[LHS_SLICE_3]], [[RESHAPE_RHS_3]])

    // CHECK:   [[ADD_1:%.+]] = IE.Accumulate([[GEMM_1]], [[GEMM_2]])
    // CHECK:   [[ADD_2:%.+]] = IE.Accumulate([[ADD_1]], [[GEMM_3]])

    return %GEMM : tensor<16x2048xf32>
    // CHECK:   return [[ADD_2]] : tensor<16x2048xf32>
}

// -----

// CHECK-LABEL: @SkipMatMulWith2dReshape
// CHECK-SAME:   [[LHS:%.+]]: tensor<32xf16>, [[RHS:%.+]]: tensor<32xf16>
func.func @SkipMatMulWith2dReshape(%LHS: tensor<32xf16>, %RHS: tensor<32xf16>) -> tensor<1x1xf16> {
    %LHS_TO_2D = IE.AffineReshape(%LHS) {
        dim_mapping = [[0, 1]],
        shape_value = [1, 32]
    } : tensor<32xf16> -> tensor<1x32xf16>

    // CHECK:   [[LHS_TO_2D:%.+]] = IE.AffineReshape([[LHS]]) {
    // CHECK-SAME:      shape_value = [1, 32]
    // CHECK-SAME:  } : tensor<32xf16> -> tensor<1x32xf16>

    %RHS_TO_2D = IE.AffineReshape(%RHS) {
        dim_mapping = [[0, 1]],
        shape_value = [1, 32]
    } : tensor<32xf16> -> tensor<1x32xf16>

    // CHECK:   [[RHS_TO_2D:%.+]] = IE.AffineReshape([[RHS]]) {
    // CHECK-SAME:      shape_value = [1, 32]
    // CHECK-SAME:  } : tensor<32xf16> -> tensor<1x32xf16>

    %FC = IE.FullyConnected(%LHS_TO_2D, %RHS_TO_2D) : tensor<1x32xf16>, tensor<1x32xf16> -> tensor<1x1xf16>

    // CHECK:   [[FC:%.+]] = IE.FullyConnected([[LHS_TO_2D]], [[RHS_TO_2D]])

    return %FC : tensor<1x1xf16>
    // CHECK:   return [[FC]] : tensor<1x1xf16>
}


// -----

#CN = affine_map<(d0, d1) -> (d1, d0)>

// CHECK-LABEL: @UnrollMatMulForConvAccumulate
// CHECK-SAME:   [[LHS_1:%arg0]]: tensor<1x3072xf32>,
// CHECK-SAME:   [[WEIGHTS:%arg1]]: tensor<1x1024x3584xf32>,
// CHECK-SAME:   [[IN_PARAM:%arg2]]: tensor<1x1x1xf32>,
// CHECK-SAME:   [[OUT_PARAM:%arg3]]: tensor<1x1x3584xf32>
func.func @UnrollMatMulForConvAccumulate(%LHS_1: tensor<1x3072xf32>,
                        %WEIGHTS: tensor<1x1024x3584xf32>,
                        %IN_PARAM: tensor<1x1x1xf32>,
                        %OUT_PARAM: tensor<1x1x3584xf32>) -> tensor<1x3584xf32> {
    %RHS_1 = IE.FakeQuantize(%WEIGHTS, %IN_PARAM, %IN_PARAM, %OUT_PARAM, %OUT_PARAM) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 16 : i64
    } : tensor<1x1024x3584xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<1x1x3584xf32>, tensor<1x1x3584xf32> -> tensor<1x1024x3584xf32>
    %RHS_2 = IE.FakeQuantize(%WEIGHTS, %IN_PARAM, %IN_PARAM, %OUT_PARAM, %OUT_PARAM) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 16 : i64
    } : tensor<1x1024x3584xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<1x1x3584xf32>, tensor<1x1x3584xf32> -> tensor<1x1024x3584xf32>
    %RHS_3 = IE.FakeQuantize(%WEIGHTS, %IN_PARAM, %IN_PARAM, %OUT_PARAM, %OUT_PARAM) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 16 : i64
    } : tensor<1x1024x3584xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<1x1x3584xf32>, tensor<1x1x3584xf32> -> tensor<1x1024x3584xf32>

    %CONCAT_RHS = IE.Concat(%RHS_1, %RHS_2, %RHS_3) {
        per_axis = #IE.Concat<axis = 0 : i64>
    } : tensor<1x1024x3584xf32>, tensor<1x1024x3584xf32>, tensor<1x1024x3584xf32> -> tensor<3x1024x3584xf32>
    %RESHAPE_RHS = IE.AffineReshape(%CONCAT_RHS) {
        dim_mapping = [[0], [0], [1]],
        shape_value = [3072, 3584]
    } : tensor<3x1024x3584xf32> -> tensor<3072x3584xf32>
    %TRANSPOSE_RHS = IE.Transpose(%RESHAPE_RHS) {
        order_value = #CN
    } : tensor<3072x3584xf32> -> tensor<3584x3072xf32>
    %GEMM = IE.FullyConnected(%LHS_1, %TRANSPOSE_RHS) : tensor<1x3072xf32>, tensor<3584x3072xf32> -> tensor<1x3584xf32>
    return %GEMM : tensor<1x3584xf32>

    // CHECK:   [[RHS_1:%.+]] = IE.FakeQuantize
    // CHECK:   [[RHS_2:%.+]] = IE.FakeQuantize
    // CHECK:   [[RHS_3:%.+]] = IE.FakeQuantize
    // CHECK:   [[RESHAPE_RHS_1:%.+]] = IE.Reshape([[RHS_1]]) {shape_value = [1024, 3584]} : tensor<1x1024x3584xf32> -> tensor<1024x3584xf32>
    // CHECK:   [[RESHAPE_RHS_2:%.+]] = IE.Reshape([[RHS_2]]) {shape_value = [1024, 3584]} : tensor<1x1024x3584xf32> -> tensor<1024x3584xf32>
    // CHECK:   [[RESHAPE_RHS_3:%.+]] = IE.Reshape([[RHS_3]]) {shape_value = [1024, 3584]} : tensor<1x1024x3584xf32> -> tensor<1024x3584xf32>
    // CHECK:   [[LHS_SLICE_1:%.+]] = IE.Slice [[LHS_1]] [0, 0] [1, 1024] : tensor<1x3072xf32> to tensor<1x1024xf32>
    // CHECK:   [[LHS_SLICE_2:%.+]] = IE.Slice [[LHS_1]] [0, 1024] [1, 1024] : tensor<1x3072xf32> to tensor<1x1024xf32>
    // CHECK:   [[LHS_SLICE_3:%.+]] = IE.Slice [[LHS_1]] [0, 2048] [1, 1024] : tensor<1x3072xf32> to tensor<1x1024xf32>

    // CHECK:   [[TRANSPOSE_1:%.+]] = IE.Transpose([[RESHAPE_RHS_1]]) {order_value = #CN} : tensor<1024x3584xf32> -> tensor<3584x1024xf32>
    // CHECK:   [[GEMM_1:%.+]] = IE.FullyConnected([[LHS_SLICE_1]], [[TRANSPOSE_1]]) : tensor<1x1024xf32>, tensor<3584x1024xf32> -> tensor<1x3584xf32>
    // CHECK:   [[GEMM_RESHAPE_1:%.+]] = IE.Reshape([[GEMM_1]]) {shape_value = [1, 1, 1, 3584]} : tensor<1x3584xf32> -> tensor<1x1x1x3584xf32>

    // CHECK:   [[TRANSPOSE_2:%.+]] = IE.Transpose([[RESHAPE_RHS_2]]) {order_value = #CN} : tensor<1024x3584xf32> -> tensor<3584x1024xf32>
    // CHECK:   [[GEMM_2:%.+]] = IE.FullyConnected([[LHS_SLICE_2]], [[TRANSPOSE_2]]) : tensor<1x1024xf32>, tensor<3584x1024xf32> -> tensor<1x3584xf32>
    // CHECK:   [[GEMM_RESHAPE_2:%.+]] = IE.Reshape([[GEMM_2]]) {shape_value = [1, 1, 1, 3584]} : tensor<1x3584xf32> -> tensor<1x1x1x3584xf32>

    // CHECK:   [[TRANSPOSE_3:%.+]] = IE.Transpose([[RESHAPE_RHS_3]]) {order_value = #CN} : tensor<1024x3584xf32> -> tensor<3584x1024xf32>
    // CHECK:   [[GEMM_3:%.+]] = IE.FullyConnected([[LHS_SLICE_3]], [[TRANSPOSE_3]]) : tensor<1x1024xf32>, tensor<3584x1024xf32> -> tensor<1x3584xf32>
    // CHECK:   [[GEMM_RESHAPE_3:%.+]] = IE.Reshape([[GEMM_3]]) {shape_value = [1, 1, 1, 3584]} : tensor<1x3584xf32> -> tensor<1x1x1x3584xf32>

    // CHECK:   [[CONCAT:%.+]] = IE.Concat([[GEMM_RESHAPE_1]], [[GEMM_RESHAPE_2]], [[GEMM_RESHAPE_3]]) {per_axis = #IE.Concat<axis = 1 : i64>} : tensor<1x1x1x3584xf32>, tensor<1x1x1x3584xf32>, tensor<1x1x1x3584xf32> -> tensor<1x3x1x3584xf32>
    // CHECK:   [[REDUCE_SUM:%.+]] = IE.ReduceSum([[CONCAT]]) {axes_value = [1]} : tensor<1x3x1x3584xf32> -> tensor<1x1x3584xf32>
    // CHECK:   [[RESHAPE_OUT:%.+]] = IE.Reshape([[REDUCE_SUM]]) {shape_value = [1, 3584]} : tensor<1x1x3584xf32> -> tensor<1x3584xf32>

    // CHECK:   return  [[RESHAPE_OUT]]
}

// -----

#CN = affine_map<(d0, d1) -> (d1, d0)>

// CHECK-LABEL: @UnrollMatMulWithDPUAccumulateForLargeSize
// CHECK-SAME:   [[LHS_1:%arg0]]: tensor<1x3072xf32>,
// CHECK-SAME:   [[RHS_1:%arg1]]: tensor<1x1024x9000xf32>,
// CHECK-SAME:   [[RHS_2:%arg2]]: tensor<1x1x1xf32>,
// CHECK-SAME:   [[RHS_3:%arg3]]: tensor<1x1x9000xf32>
func.func @UnrollMatMulWithDPUAccumulateForLargeSize(
        %LHS_1: tensor<1x3072xf32>,
        %WEIGHTS: tensor<1x1024x9000xf32>,
        %IN_PARAM: tensor<1x1x1xf32>,
        %OUT_PARAM: tensor<1x1x9000xf32>) -> tensor<1x9000xf32> {
    %RHS_1 = IE.FakeQuantize(%WEIGHTS, %IN_PARAM, %IN_PARAM, %OUT_PARAM, %OUT_PARAM) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 16 : i64
    } : tensor<1x1024x9000xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<1x1x9000xf32>, tensor<1x1x9000xf32> -> tensor<1x1024x9000xf32>
    %RHS_2 = IE.FakeQuantize(%WEIGHTS, %IN_PARAM, %IN_PARAM, %OUT_PARAM, %OUT_PARAM) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 16 : i64
    } : tensor<1x1024x9000xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<1x1x9000xf32>, tensor<1x1x9000xf32> -> tensor<1x1024x9000xf32>
    %RHS_3 = IE.FakeQuantize(%WEIGHTS, %IN_PARAM, %IN_PARAM, %OUT_PARAM, %OUT_PARAM) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 16 : i64
    } : tensor<1x1024x9000xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<1x1x9000xf32>, tensor<1x1x9000xf32> -> tensor<1x1024x9000xf32>
    // CHECK:   [[RHS_1:%.+]] = IE.FakeQuantize
    // CHECK:   [[RHS_2:%.+]] = IE.FakeQuantize
    // CHECK:   [[RHS_3:%.+]] = IE.FakeQuantize

    %CONCAT_RHS = IE.Concat(%RHS_1, %RHS_2, %RHS_3) {
        per_axis = #IE.Concat<axis = 0 : i64>
    } : tensor<1x1024x9000xf32>, tensor<1x1024x9000xf32>, tensor<1x1024x9000xf32> -> tensor<3x1024x9000xf32>
    %RESHAPE_RHS = IE.AffineReshape(%CONCAT_RHS) {
        dim_mapping = [[0], [0], [1]],
        shape_value = [3072, 9000]
    } : tensor<3x1024x9000xf32> -> tensor<3072x9000xf32>
    %TRANSPOSE_RHS = IE.Transpose(%RESHAPE_RHS) {
        order_value = #CN
    } : tensor<3072x9000xf32> -> tensor<9000x3072xf32>
    %GEMM = IE.FullyConnected(%LHS_1, %TRANSPOSE_RHS) : tensor<1x3072xf32>, tensor<9000x3072xf32> -> tensor<1x9000xf32>
    return %GEMM : tensor<1x9000xf32>

    // CHECK:   [[RESHAPE_RHS_1:%.+]] = IE.Reshape([[RHS_1]]) {shape_value = [1024, 9000]} : tensor<1x1024x9000xf32> -> tensor<1024x9000xf32>
    // CHECK:   [[RESHAPE_RHS_2:%.+]] = IE.Reshape([[RHS_2]]) {shape_value = [1024, 9000]} : tensor<1x1024x9000xf32> -> tensor<1024x9000xf32>
    // CHECK:   [[RESHAPE_RHS_3:%.+]] = IE.Reshape([[RHS_3]]) {shape_value = [1024, 9000]} : tensor<1x1024x9000xf32> -> tensor<1024x9000xf32>
    // CHECK:   [[LHS_SLICE_1:%.+]] = IE.Slice [[LHS_1]] [0, 0] [1, 1024] : tensor<1x3072xf32> to tensor<1x1024xf32>
    // CHECK:   [[LHS_SLICE_2:%.+]] = IE.Slice [[LHS_1]] [0, 1024] [1, 1024] : tensor<1x3072xf32> to tensor<1x1024xf32>
    // CHECK:   [[LHS_SLICE_3:%.+]] = IE.Slice [[LHS_1]] [0, 2048] [1, 1024] : tensor<1x3072xf32> to tensor<1x1024xf32>

    // CHECK:   [[TRANSPOSE_1:%.+]] = IE.Transpose([[RESHAPE_RHS_1]]) {order_value = #CN} : tensor<1024x9000xf32> -> tensor<9000x1024xf32>
    // CHECK:   [[GEMM_1:%.+]] = IE.FullyConnected([[LHS_SLICE_1]], [[TRANSPOSE_1]]) : tensor<1x1024xf32>, tensor<9000x1024xf32> -> tensor<1x9000xf32>
    // CHECK:   [[GEMM_RESHAPE_1:%.+]] = IE.Reshape([[GEMM_1]]) {shape_value = [1, 1, 1, 9000]} : tensor<1x9000xf32> -> tensor<1x1x1x9000xf32>

    // CHECK:   [[TRANSPOSE_2:%.+]] = IE.Transpose([[RESHAPE_RHS_2]]) {order_value = #CN} : tensor<1024x9000xf32> -> tensor<9000x1024xf32>
    // CHECK:   [[GEMM_2:%.+]] = IE.FullyConnected([[LHS_SLICE_2]], [[TRANSPOSE_2]]) : tensor<1x1024xf32>, tensor<9000x1024xf32> -> tensor<1x9000xf32>
    // CHECK:   [[GEMM_RESHAPE_2:%.+]] = IE.Reshape([[GEMM_2]]) {shape_value = [1, 1, 1, 9000]} : tensor<1x9000xf32> -> tensor<1x1x1x9000xf32>

    // CHECK:   [[TRANSPOSE_3:%.+]] = IE.Transpose([[RESHAPE_RHS_3]]) {order_value = #CN} : tensor<1024x9000xf32> -> tensor<9000x1024xf32>
    // CHECK:   [[GEMM_3:%.+]] = IE.FullyConnected([[LHS_SLICE_3]], [[TRANSPOSE_3]]) : tensor<1x1024xf32>, tensor<9000x1024xf32> -> tensor<1x9000xf32>
    // CHECK:   [[GEMM_RESHAPE_3:%.+]] = IE.Reshape([[GEMM_3]]) {shape_value = [1, 1, 1, 9000]} : tensor<1x9000xf32> -> tensor<1x1x1x9000xf32>

    // CHECK:   [[CONCAT:%.+]] = IE.Concat([[GEMM_RESHAPE_1]], [[GEMM_RESHAPE_2]], [[GEMM_RESHAPE_3]]) {per_axis = #IE.Concat<axis = 1 : i64>} : tensor<1x1x1x9000xf32>, tensor<1x1x1x9000xf32>, tensor<1x1x1x9000xf32> -> tensor<1x3x1x9000xf32>
    // CHECK:   [[REDUCE_SUM:%.+]] = IE.ReduceSum([[CONCAT]]) {axes_value = [1]} : tensor<1x3x1x9000xf32> -> tensor<1x1x9000xf32>
    // CHECK:   [[RESHAPE_OUT:%.+]] = IE.Reshape([[REDUCE_SUM]]) {shape_value = [1, 9000]} : tensor<1x1x9000xf32> -> tensor<1x9000xf32>

    // CHECK:   return  [[RESHAPE_OUT]]
}



// -----

#CN = affine_map<(d0, d1) -> (d1, d0)>
!qElemType = !quant.uniform<i4:f16, 1.000000e+00>

// CHECK-LABEL: @UnrollMatMulForDynamicDequantize
// CHECK-SAME:   [[WEIGHTS:%arg0]]: tensor<1x1024x3584x!qElemType>,
// CHECK-SAME:   [[SCALE_1:%arg1]]: tensor<1x1x3584xf32>,
// CHECK-SAME:   [[SCALE_2:%arg2]]: tensor<1x1x3584xf32>,
// CHECK-SAME:   [[SCALE_3:%arg3]]: tensor<1x1x3584xf32>,
// CHECK-SAME:   [[LHS_1:%arg4]]: tensor<1x3072xf32>
func.func @UnrollMatMulForDynamicDequantize(%WEIGHTS: tensor<1x1024x3584x!qElemType>,
                        %SCALE_1: tensor<1x1x3584xf32>,
                        %SCALE_2: tensor<1x1x3584xf32>,
                        %SCALE_3: tensor<1x1x3584xf32>,
                        %LHS_1: tensor<1x3072xf32>
                        ) -> tensor<1x3584xf32> {
    %RHS_1 = IE.DynamicDequantize(%WEIGHTS, %SCALE_1) {dstElemType = f32} : tensor<1x1024x3584x!qElemType>, tensor<1x1x3584xf32> -> tensor<1x1024x3584xf32>
    %RHS_2 = IE.DynamicDequantize(%WEIGHTS, %SCALE_2) {dstElemType = f32} : tensor<1x1024x3584x!qElemType>, tensor<1x1x3584xf32> -> tensor<1x1024x3584xf32>
    %RHS_3 = IE.DynamicDequantize(%WEIGHTS, %SCALE_3) {dstElemType = f32} : tensor<1x1024x3584x!qElemType>, tensor<1x1x3584xf32> -> tensor<1x1024x3584xf32>

    %CONCAT_RHS = IE.Concat(%RHS_1, %RHS_2, %RHS_3) {
        per_axis = #IE.Concat<axis = 0 : i64>
    } : tensor<1x1024x3584xf32>, tensor<1x1024x3584xf32>, tensor<1x1024x3584xf32> -> tensor<3x1024x3584xf32>
    %RESHAPE_RHS = IE.AffineReshape(%CONCAT_RHS) {
        dim_mapping = [[0], [0], [1]],
        shape_value = [3072, 3584]
    } : tensor<3x1024x3584xf32> -> tensor<3072x3584xf32>
    %TRANSPOSE_RHS = IE.Transpose(%RESHAPE_RHS) {
        order_value = #CN
    } : tensor<3072x3584xf32> -> tensor<3584x3072xf32>
    %GEMM = IE.FullyConnected(%LHS_1, %TRANSPOSE_RHS) : tensor<1x3072xf32>, tensor<3584x3072xf32> -> tensor<1x3584xf32>
    return %GEMM : tensor<1x3584xf32>

    // CHECK:   [[RHS_1:%.+]] = IE.DynamicDequantize
    // CHECK:   [[RHS_2:%.+]] = IE.DynamicDequantize
    // CHECK:   [[RHS_3:%.+]] = IE.DynamicDequantize
    // CHECK:   [[RESHAPE_RHS_1:%.+]] = IE.Reshape([[RHS_1]]) {shape_value = [1024, 3584]} : tensor<1x1024x3584xf32> -> tensor<1024x3584xf32>
    // CHECK:   [[RESHAPE_RHS_2:%.+]] = IE.Reshape([[RHS_2]]) {shape_value = [1024, 3584]} : tensor<1x1024x3584xf32> -> tensor<1024x3584xf32>
    // CHECK:   [[RESHAPE_RHS_3:%.+]] = IE.Reshape([[RHS_3]]) {shape_value = [1024, 3584]} : tensor<1x1024x3584xf32> -> tensor<1024x3584xf32>
    // CHECK:   [[LHS_SLICE_1:%.+]] = IE.Slice [[LHS_1]] [0, 0] [1, 1024] : tensor<1x3072xf32> to tensor<1x1024xf32>
    // CHECK:   [[LHS_SLICE_2:%.+]] = IE.Slice [[LHS_1]] [0, 1024] [1, 1024] : tensor<1x3072xf32> to tensor<1x1024xf32>
    // CHECK:   [[LHS_SLICE_3:%.+]] = IE.Slice [[LHS_1]] [0, 2048] [1, 1024] : tensor<1x3072xf32> to tensor<1x1024xf32>

    // CHECK:   [[TRANSPOSE_1:%.+]] = IE.Transpose([[RESHAPE_RHS_1]]) {order_value = #CN} : tensor<1024x3584xf32> -> tensor<3584x1024xf32>
    // CHECK:   [[GEMM_1:%.+]] = IE.FullyConnected([[LHS_SLICE_1]], [[TRANSPOSE_1]]) : tensor<1x1024xf32>, tensor<3584x1024xf32> -> tensor<1x3584xf32>
    // CHECK:   [[GEMM_RESHAPE_1:%.+]] = IE.Reshape([[GEMM_1]]) {shape_value = [1, 1, 1, 3584]} : tensor<1x3584xf32> -> tensor<1x1x1x3584xf32>

    // CHECK:   [[TRANSPOSE_2:%.+]] = IE.Transpose([[RESHAPE_RHS_2]]) {order_value = #CN} : tensor<1024x3584xf32> -> tensor<3584x1024xf32>
    // CHECK:   [[GEMM_2:%.+]] = IE.FullyConnected([[LHS_SLICE_2]], [[TRANSPOSE_2]]) : tensor<1x1024xf32>, tensor<3584x1024xf32> -> tensor<1x3584xf32>
    // CHECK:   [[GEMM_RESHAPE_2:%.+]] = IE.Reshape([[GEMM_2]]) {shape_value = [1, 1, 1, 3584]} : tensor<1x3584xf32> -> tensor<1x1x1x3584xf32>

    // CHECK:   [[TRANSPOSE_3:%.+]] = IE.Transpose([[RESHAPE_RHS_3]]) {order_value = #CN} : tensor<1024x3584xf32> -> tensor<3584x1024xf32>
    // CHECK:   [[GEMM_3:%.+]] = IE.FullyConnected([[LHS_SLICE_3]], [[TRANSPOSE_3]]) : tensor<1x1024xf32>, tensor<3584x1024xf32> -> tensor<1x3584xf32>
    // CHECK:   [[GEMM_RESHAPE_3:%.+]] = IE.Reshape([[GEMM_3]]) {shape_value = [1, 1, 1, 3584]} : tensor<1x3584xf32> -> tensor<1x1x1x3584xf32>

    // CHECK:   [[CONCAT:%.+]] = IE.Concat([[GEMM_RESHAPE_1]], [[GEMM_RESHAPE_2]], [[GEMM_RESHAPE_3]]) {per_axis = #IE.Concat<axis = 1 : i64>} : tensor<1x1x1x3584xf32>, tensor<1x1x1x3584xf32>, tensor<1x1x1x3584xf32> -> tensor<1x3x1x3584xf32>
    // CHECK:   [[REDUCE_SUM:%.+]] = IE.ReduceSum([[CONCAT]]) {axes_value = [1]} : tensor<1x3x1x3584xf32> -> tensor<1x1x3584xf32>
    // CHECK:   [[RESHAPE_OUT:%.+]] = IE.Reshape([[REDUCE_SUM]]) {shape_value = [1, 3584]} : tensor<1x1x3584xf32> -> tensor<1x3584xf32>

    // CHECK:   return  [[RESHAPE_OUT]]
}


// -----

#HCW = affine_map<(d0, d1, d2) -> (d1, d0, d2)>

// CHECK-LABEL: @UnrollMatMulReshapeTranspose102
// CHECK-SAME:   [[LHS_1:%arg0]]: tensor<1x12288xf32>,
// CHECK-SAME:   [[WEIGHTS:%arg1]]: tensor<1x4608x4096xf32>,
// CHECK-SAME:   [[IN_PARAM:%arg2]]: tensor<1x1x1xf32>,
// CHECK-SAME:   [[OUT_PARAM:%arg3]]: tensor<1x4608x1xf32>
func.func @UnrollMatMulReshapeTranspose102(%LHS_1: tensor<1x12288xf32>,
                                        %WEIGHTS: tensor<1x4608x4096xf32>,
                                        %IN_PARAM: tensor<1x1x1xf32>,
                                        %OUT_PARAM: tensor<1x4608x1xf32>) -> tensor<1x4608xf32> {
    %RHS_1 = IE.FakeQuantize(%WEIGHTS, %IN_PARAM, %IN_PARAM, %OUT_PARAM, %OUT_PARAM) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 16 : i64
    } : tensor<1x4608x4096xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<1x4608x1xf32>, tensor<1x4608x1xf32> -> tensor<1x4608x4096xf32>
    %RHS_2 = IE.FakeQuantize(%WEIGHTS, %IN_PARAM, %IN_PARAM, %OUT_PARAM, %OUT_PARAM) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 16 : i64
    } : tensor<1x4608x4096xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<1x4608x1xf32>, tensor<1x4608x1xf32> -> tensor<1x4608x4096xf32>
    %RHS_3 = IE.FakeQuantize(%WEIGHTS, %IN_PARAM, %IN_PARAM, %OUT_PARAM, %OUT_PARAM) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 16 : i64
    } : tensor<1x4608x4096xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<1x4608x1xf32>, tensor<1x4608x1xf32> -> tensor<1x4608x4096xf32>
    // CHECK:   [[RHS_1:%.+]] = IE.FakeQuantize
    // CHECK:   [[RHS_2:%.+]] = IE.FakeQuantize
    // CHECK:   [[RHS_3:%.+]] = IE.FakeQuantize

    %CONCAT_RHS = IE.Concat(%RHS_1, %RHS_2, %RHS_3) {
        per_axis = #IE.Concat<axis = 0 : i64>
    } : tensor<1x4608x4096xf32>, tensor<1x4608x4096xf32>, tensor<1x4608x4096xf32> -> tensor<3x4608x4096xf32>
    // CHECK-NOT:   IE.Concat

    %TRANSPOSE_RHS = IE.Transpose(%CONCAT_RHS) {
        order_value = #HCW
    } : tensor<3x4608x4096xf32> -> tensor<4608x3x4096xf32>

    %RESHAPE_RHS = IE.AffineReshape(%TRANSPOSE_RHS) {
        dim_mapping = [[0], [1], [1]],
        shape_value = [4608, 12288]
    } : tensor<4608x3x4096xf32> -> tensor<4608x12288xf32>

    // CHECK:   [[RESHAPE_RHS_1:%.+]] = IE.Reshape([[RHS_1]]) {
    // CHECK-SAME:      shape_value = [4608, 4096]
    // CHECK-SAME:  } : tensor<1x4608x4096xf32> -> tensor<4608x4096xf32>
    // CHECK:   [[RESHAPE_RHS_2:%.+]] = IE.Reshape([[RHS_2]]) {
    // CHECK-SAME:      shape_value = [4608, 4096]
    // CHECK-SAME:  } : tensor<1x4608x4096xf32> -> tensor<4608x4096xf32>
    // CHECK:   [[RESHAPE_RHS_3:%.+]] = IE.Reshape([[RHS_3]]) {
    // CHECK-SAME:      shape_value = [4608, 4096]
    // CHECK-SAME:  } : tensor<1x4608x4096xf32> -> tensor<4608x4096xf32>

    %GEMM = IE.FullyConnected(%LHS_1, %RESHAPE_RHS) : tensor<1x12288xf32>, tensor<4608x12288xf32> -> tensor<1x4608xf32>
    // CHECK:   [[LHS_SLICE_1:%.+]] = IE.Slice [[LHS_1]] [0, 0] [1, 4096]
    // CHECK:   [[LHS_SLICE_2:%.+]] = IE.Slice [[LHS_1]] [0, 4096] [1, 4096]
    // CHECK:   [[LHS_SLICE_3:%.+]] = IE.Slice [[LHS_1]] [0, 8192] [1, 4096]

    // CHECK:   [[GEMM_1:%.+]] = IE.FullyConnected([[LHS_SLICE_1]], [[RESHAPE_RHS_1]])
    // CHECK:   [[RESHAPE_OUT_1:%.+]] = IE.Reshape([[GEMM_1]]) {shape_value = [1, 1, 1, 4608]} : tensor<1x4608xf32> -> tensor<1x1x1x4608xf32>
    // CHECK:   [[GEMM_2:%.+]] = IE.FullyConnected([[LHS_SLICE_2]], [[RESHAPE_RHS_2]])
    // CHECK:   [[RESHAPE_OUT_2:%.+]]  = IE.Reshape([[GEMM_2]]) {shape_value = [1, 1, 1, 4608]} : tensor<1x4608xf32> -> tensor<1x1x1x4608xf32>
    // CHECK:   [[GEMM_3:%.+]] = IE.FullyConnected([[LHS_SLICE_3]], [[RESHAPE_RHS_3]])
    // CHECK:   [[RESHAPE_OUT_3:%.+]] = IE.Reshape([[GEMM_3]]) {shape_value = [1, 1, 1, 4608]} : tensor<1x4608xf32> -> tensor<1x1x1x4608xf32>

    // CHECK:   [[CONCAT:%.+]] = IE.Concat([[RESHAPE_OUT_1]], [[RESHAPE_OUT_2]], [[RESHAPE_OUT_3]]) {per_axis = #IE.Concat<axis = 1 : i64>} : tensor<1x1x1x4608xf32>, tensor<1x1x1x4608xf32>, tensor<1x1x1x4608xf32> -> tensor<1x3x1x4608xf32>
    // CHECK:   [[REDUCESUM:%.+]] = IE.ReduceSum([[CONCAT]]) {axes_value = [1]} : tensor<1x3x1x4608xf32> -> tensor<1x1x4608xf32>
    // CHECK:   [[RESHAPE:%.+]] = IE.Reshape([[REDUCESUM]]) {shape_value = [1, 4608]} : tensor<1x1x4608xf32> -> tensor<1x4608xf32>

    return %GEMM : tensor<1x4608xf32>
    // CHECK:   return [[RESHAPE]] : tensor<1x4608xf32>
}
