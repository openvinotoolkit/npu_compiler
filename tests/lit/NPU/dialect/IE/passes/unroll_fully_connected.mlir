//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --unroll-fully-connected %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

#CN = affine_map<(d0, d1) -> (d1, d0)>

// CHECK-LABEL: @UnrollMatMul
// CHECK-SAME:   [[LHS_1:%arg[0-9]+]]: tensor<16x3072xf32>,
// CHECK-SAME:   [[WEIGHTS:%arg[0-9]+]]: tensor<1x1024x4096xf32>,
// CHECK-SAME:   [[IN_PARAM:%arg[0-9]+]]: tensor<1x1x1xf32>,
// CHECK-SAME:   [[OUT_PARAM:%arg[0-9]+]]: tensor<1x1x4096xf32>
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

    // CHECK:   [[ADD_1:%.+]] = IE.Add([[GEMM_1]], [[GEMM_2]])
    // CHECK:   [[ADD_2:%.+]] = IE.Add([[ADD_1]], [[GEMM_3]])

    return %GEMM : tensor<16x4096xf32>
    // CHECK:   return [[ADD_2]] : tensor<16x4096xf32>
}

// -----

#CN = affine_map<(d0, d1) -> (d1, d0)>

// CHECK-LABEL: @DontUnrollMatMulNot4bit
// CHECK-SAME:   [[LHS_1:%arg[0-9]+]]: tensor<16x3072xf32>,
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
// CHECK-SAME:   [[LHS_1:%arg[0-9]+]]: tensor<16x96xf32>,
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
// CHECK-SAME:   [[LHS_1:%arg[0-9]+]]: tensor<16x3072xf32>,
// CHECK-SAME:   [[WEIGHTS:%arg[0-9]+]]: tensor<1x1024x4096xf32>,
// CHECK-SAME:   [[IN_PARAM:%arg[0-9]+]]: tensor<1x1x1xf32>,
// CHECK-SAME:   [[OUT_PARAM:%arg[0-9]+]]: tensor<1x1x4096xf32>
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

    // CHECK:   [[ADD_1:%.+]] = IE.Add([[GEMM_1]], [[GEMM_2]])
    // CHECK:   [[ADD_2:%.+]] = IE.Add([[ADD_1]], [[GEMM_3]])

    return %GEMM : tensor<16x4096xf32>
    // CHECK:   return [[ADD_2]] : tensor<16x4096xf32>
}

// -----

// CHECK-LABEL: @UnrollMatMulWithOnlyReshape
// CHECK-SAME:   [[LHS_1:%arg[0-9]+]]: tensor<16x3072xf32>,
// CHECK-SAME:   [[WEIGHTS:%arg[0-9]+]]: tensor<2048x1x1024xf32>,
// CHECK-SAME:   [[IN_PARAM:%arg[0-9]+]]: tensor<1x1x1xf32>,
// CHECK-SAME:   [[OUT_PARAM:%arg[0-9]+]]: tensor<2048x1x1xf32>
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

    // CHECK:   [[ADD_1:%.+]] = IE.Add([[GEMM_1]], [[GEMM_2]])
    // CHECK:   [[ADD_2:%.+]] = IE.Add([[ADD_1]], [[GEMM_3]])

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
// CHECK-SAME:   [[LHS_1:%arg[0-9]+]]: tensor<1x3072xf32>,
// CHECK-SAME:   [[WEIGHTS:%arg[0-9]+]]: tensor<1x1024x3584xf32>,
// CHECK-SAME:   [[IN_PARAM:%arg[0-9]+]]: tensor<1x1x1xf32>,
// CHECK-SAME:   [[OUT_PARAM:%arg[0-9]+]]: tensor<1x1x3584xf32>
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
// CHECK-SAME:   [[LHS_1:%arg[0-9]+]]: tensor<1x3072xf32>,
// CHECK-SAME:   [[RHS_1:%arg[0-9]+]]: tensor<1x1024x9000xf32>,
// CHECK-SAME:   [[RHS_2:%arg[0-9]+]]: tensor<1x1x1xf32>,
// CHECK-SAME:   [[RHS_3:%arg[0-9]+]]: tensor<1x1x9000xf32>
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
// CHECK-SAME:   [[WEIGHTS:%arg[0-9]+]]: tensor<1x1024x3584x!qElemType>,
// CHECK-SAME:   [[SCALE_1:%arg[0-9]+]]: tensor<1x1x3584xf32>,
// CHECK-SAME:   [[SCALE_2:%arg[0-9]+]]: tensor<1x1x3584xf32>,
// CHECK-SAME:   [[SCALE_3:%arg[0-9]+]]: tensor<1x1x3584xf32>,
// CHECK-SAME:   [[LHS_1:%arg[0-9]+]]: tensor<1x3072xf32>
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

#CN = affine_map<(d0, d1) -> (d1, d0)>
!qElemType = !quant.uniform<u2:f16, 1.000000e+00>

// CHECK-LABEL: @UnrollMatMulForU2DynamicDequantize
// CHECK-SAME:   [[WEIGHTS:%arg[0-9]+]]: tensor<1x1024x3584x!qElemType>,
// CHECK-SAME:   [[SCALE_1:%arg[0-9]+]]: tensor<1x1x3584xf32>,
// CHECK-SAME:   [[SCALE_2:%arg[0-9]+]]: tensor<1x1x3584xf32>,
// CHECK-SAME:   [[SCALE_3:%arg[0-9]+]]: tensor<1x1x3584xf32>,
// CHECK-SAME:   [[LHS_1:%arg[0-9]+]]: tensor<1x3072xf32>
func.func @UnrollMatMulForU2DynamicDequantize(%WEIGHTS: tensor<1x1024x3584x!qElemType>,
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
// CHECK-SAME:   [[LHS_1:%arg[0-9]+]]: tensor<1x12288xf32>,
// CHECK-SAME:   [[WEIGHTS:%arg[0-9]+]]: tensor<1x4608x4096xf32>,
// CHECK-SAME:   [[IN_PARAM:%arg[0-9]+]]: tensor<1x1x1xf32>,
// CHECK-SAME:   [[OUT_PARAM:%arg[0-9]+]]: tensor<1x4608x1xf32>
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


// -----

#CN = affine_map<(d0, d1) -> (d1, d0)>
!qElemType = !quant.uniform<i4:f16, 1.000000e+00>

// CHECK-LABEL: @UnrollMatMulForDequantize
// CHECK-SAME:   [[INPUT:%arg[0-9]+]]: tensor<1x1024x3584x!qElemType>,
// CHECK-SAME:   [[LHS_1:%arg[0-9]+]]: tensor<1x3072xf32>
func.func @UnrollMatMulForDequantize(%INPUT: tensor<1x1024x3584x!qElemType>, %LHS_1: tensor<1x3072xf32>) -> tensor<1x3584xf32> {
    %RHS_1 = IE.Dequantize(%INPUT) {dstElemType = f32} : tensor<1x1024x3584x!qElemType> -> tensor<1x1024x3584xf32>
    %RHS_2 = IE.Dequantize(%INPUT) {dstElemType = f32} : tensor<1x1024x3584x!qElemType> -> tensor<1x1024x3584xf32>
    %RHS_3 = IE.Dequantize(%INPUT) {dstElemType = f32} : tensor<1x1024x3584x!qElemType> -> tensor<1x1024x3584xf32>

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

    // CHECK:   [[RHS_1:%.+]] = IE.Dequantize
    // CHECK:   [[RHS_2:%.+]] = IE.Dequantize
    // CHECK:   [[RHS_3:%.+]] = IE.Dequantize
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
!qElemType = !quant.uniform<i8:f16, 1.000000e+00>

// CHECK-LABEL: @DontUnrollMatMulForDequantize
// CHECK-SAME:   [[INPUT:%arg[0-9]+]]: tensor<1x1024x3584x!qElemType>,
// CHECK-SAME:   [[LHS_1:%arg[0-9]+]]: tensor<1x3072xf32>
func.func @DontUnrollMatMulForDequantize(%INPUT: tensor<1x1024x3584x!qElemType>, %LHS_1: tensor<1x3072xf32>) -> tensor<1x3584xf32> {
    %RHS_1 = IE.Dequantize(%INPUT) {dstElemType = f32} : tensor<1x1024x3584x!qElemType> -> tensor<1x1024x3584xf32>
    %RHS_2 = IE.Dequantize(%INPUT) {dstElemType = f32} : tensor<1x1024x3584x!qElemType> -> tensor<1x1024x3584xf32>
    %RHS_3 = IE.Dequantize(%INPUT) {dstElemType = f32} : tensor<1x1024x3584x!qElemType> -> tensor<1x1024x3584xf32>

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

    // CHECK:   [[RHS_1:%.+]] = IE.Dequantize
    // CHECK:   [[RHS_2:%.+]] = IE.Dequantize
    // CHECK:   [[RHS_3:%.+]] = IE.Dequantize

    // CHECK:   [[CONCAT:%.+]] = IE.Concat([[RHS_1]], [[RHS_2]], [[RHS_3]])
    // CHECK:   [[RESHAPE:%.+]] = IE.AffineReshape([[CONCAT]])
    // CHECK:   [[TRANSPOSE:%.+]] = IE.Transpose([[RESHAPE]])
    // CHECK:   [[GEMM:%.+]] = IE.FullyConnected([[LHS_1]], [[TRANSPOSE]])

    // CHECK:   return [[GEMM]]
}

// -----

#CN = affine_map<(d0, d1) -> (d1, d0)>
!qElemType = !quant.uniform<i4:f16, 1.000000e+00>

// Test case: inputChannels (3072) cannot be evenly divided by numChunks (5)
// Performance metric passes threshold but should not unroll due to indivisible channels
// CHECK-LABEL: @DontUnrollMatMulWhenInputChannelsNotDivisible
// CHECK-SAME:   [[WEIGHTS:%arg[0-9]+]]: tensor<1x1x3584x!qElemType>,
// CHECK-SAME:   [[SCALE_1:%arg[0-9]+]]: tensor<1x1x3584xf16>,
// CHECK-SAME:   [[SCALE_2:%arg[0-9]+]]: tensor<1x1x3584xf16>,
// CHECK-SAME:   [[SCALE_3:%arg[0-9]+]]: tensor<1x1x3584xf16>,
// CHECK-SAME:   [[SCALE_4:%arg[0-9]+]]: tensor<1x1x3584xf16>,
// CHECK-SAME:   [[SCALE_5:%arg[0-9]+]]: tensor<1x1x3584xf16>,
// CHECK-SAME:   [[LHS_1:%arg[0-9]+]]: tensor<1x3072xf16>
func.func @DontUnrollMatMulWhenInputChannelsNotDivisible(%WEIGHTS: tensor<1x1x3584x!qElemType>,
                        %SCALE_1: tensor<1x1x3584xf16>,
                        %SCALE_2: tensor<1x1x3584xf16>,
                        %SCALE_3: tensor<1x1x3584xf16>,
                        %SCALE_4: tensor<1x1x3584xf16>,
                        %SCALE_5: tensor<1x1x3584xf16>,
                        %LHS_1: tensor<1x3072xf16>) -> tensor<1x3584xf16> {
    %RHS_1 = IE.DynamicDequantize(%WEIGHTS, %SCALE_1) {dstElemType = f16} : tensor<1x1x3584x!qElemType>, tensor<1x1x3584xf16> -> tensor<1x1x3584xf16>
    %RHS_2 = IE.DynamicDequantize(%WEIGHTS, %SCALE_2) {dstElemType = f16} : tensor<1x1x3584x!qElemType>, tensor<1x1x3584xf16> -> tensor<1x1x3584xf16>
    %RHS_3 = IE.DynamicDequantize(%WEIGHTS, %SCALE_3) {dstElemType = f16} : tensor<1x1x3584x!qElemType>, tensor<1x1x3584xf16> -> tensor<1x1x3584xf16>
    %RHS_4 = IE.DynamicDequantize(%WEIGHTS, %SCALE_4) {dstElemType = f16} : tensor<1x1x3584x!qElemType>, tensor<1x1x3584xf16> -> tensor<1x1x3584xf16>
    %RHS_5 = IE.DynamicDequantize(%WEIGHTS, %SCALE_5) {dstElemType = f16} : tensor<1x1x3584x!qElemType>, tensor<1x1x3584xf16> -> tensor<1x1x3584xf16>

    %CONCAT_RHS = IE.Concat(%RHS_1, %RHS_2, %RHS_3, %RHS_4, %RHS_5) {
        per_axis = #IE.Concat<axis = 0 : i64>
    } : tensor<1x1x3584xf16>, tensor<1x1x3584xf16>, tensor<1x1x3584xf16>, tensor<1x1x3584xf16>, tensor<1x1x3584xf16> -> tensor<5x1x3584xf16>

    %RESHAPE_RHS = IE.AffineReshape(%CONCAT_RHS) {
        dim_mapping = [[0], [0], [1]],
        shape_value = [5, 3584]
    } : tensor<5x1x3584xf16> -> tensor<5x3584xf16>

    %TRANSPOSE_RHS = IE.Transpose(%RESHAPE_RHS) {
        order_value = #CN
    } : tensor<5x3584xf16> -> tensor<3584x5xf16>

    // input: tensor<1x3072xf16>, weights: tensor<3584x5xf16>
    // This would fail matrix multiplication (3072 != 5), but the pass checks for divisibility first
    %GEMM = IE.FullyConnected(%LHS_1, %TRANSPOSE_RHS) : tensor<1x3072xf16>, tensor<3584x5xf16> -> tensor<1x3584xf16>

    return %GEMM : tensor<1x3584xf16>

    // CHECK:   [[RHS_1:%.+]] = IE.DynamicDequantize
    // CHECK:   [[RHS_2:%.+]] = IE.DynamicDequantize
    // CHECK:   [[RHS_3:%.+]] = IE.DynamicDequantize
    // CHECK:   [[RHS_4:%.+]] = IE.DynamicDequantize
    // CHECK:   [[RHS_5:%.+]] = IE.DynamicDequantize
    // CHECK:   [[CONCAT:%.+]] = IE.Concat([[RHS_1]], [[RHS_2]], [[RHS_3]], [[RHS_4]], [[RHS_5]])
    // CHECK:   [[RESHAPE:%.+]] = IE.AffineReshape([[CONCAT]])
    // CHECK:   [[TRANSPOSE:%.+]] = IE.Transpose([[RESHAPE]])
    // CHECK:   [[GEMM:%.+]] = IE.FullyConnected([[LHS_1]], [[TRANSPOSE]])

    // CHECK:   return [[GEMM]]
}

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


// -----

!qElemType = !quant.uniform<i4:f16, 1.000000e+00>

// CHECK-LABEL: @UnrollMatMulForI4DynamicDequantize
// CHECK-SAME:   [[WEIGHTS:%arg[0-9]+]]: tensor<1x1024x128x!qElemType>,
// CHECK-SAME:   [[SCALE_1:%arg[0-9]+]]: tensor<1x1024x1xf16>,
// CHECK-SAME:   [[SCALE_2:%arg[0-9]+]]: tensor<1x1024x1xf16>,
// CHECK-SAME:   [[SCALE_3:%arg[0-9]+]]: tensor<1x1024x1xf16>,
// CHECK-SAME:   [[SCALE_4:%arg[0-9]+]]: tensor<1x1024x1xf16>,
// CHECK-SAME:   [[SCALE_5:%arg[0-9]+]]: tensor<1x1024x1xf16>,
// CHECK-SAME:   [[SCALE_6:%arg[0-9]+]]: tensor<1x1024x1xf16>,
// CHECK-SAME:   [[SCALE_7:%arg[0-9]+]]: tensor<1x1024x1xf16>,
// CHECK-SAME:   [[SCALE_8:%arg[0-9]+]]: tensor<1x1024x1xf16>,
// CHECK-SAME:   [[LHS:%arg[0-9]+]]: tensor<1x1024xf16>
func.func @UnrollMatMulForI4DynamicDequantize(
        %WEIGHTS: tensor<1x1024x128x!qElemType>,
        %SCALE_1: tensor<1x1024x1xf16>, %SCALE_2: tensor<1x1024x1xf16>,
        %SCALE_3: tensor<1x1024x1xf16>, %SCALE_4: tensor<1x1024x1xf16>,
        %SCALE_5: tensor<1x1024x1xf16>, %SCALE_6: tensor<1x1024x1xf16>,
        %SCALE_7: tensor<1x1024x1xf16>, %SCALE_8: tensor<1x1024x1xf16>,
        %LHS: tensor<1x1024xf16>
    ) -> tensor<1x1024xf16> {

    %RHS_1 = IE.DynamicDequantize(%WEIGHTS, %SCALE_1) {dstElemType = f16} : tensor<1x1024x128x!qElemType>, tensor<1x1024x1xf16> -> tensor<1x1024x128xf16>
    %RHS_2 = IE.DynamicDequantize(%WEIGHTS, %SCALE_2) {dstElemType = f16} : tensor<1x1024x128x!qElemType>, tensor<1x1024x1xf16> -> tensor<1x1024x128xf16>
    %RHS_3 = IE.DynamicDequantize(%WEIGHTS, %SCALE_3) {dstElemType = f16} : tensor<1x1024x128x!qElemType>, tensor<1x1024x1xf16> -> tensor<1x1024x128xf16>
    %RHS_4 = IE.DynamicDequantize(%WEIGHTS, %SCALE_4) {dstElemType = f16} : tensor<1x1024x128x!qElemType>, tensor<1x1024x1xf16> -> tensor<1x1024x128xf16>
    %RHS_5 = IE.DynamicDequantize(%WEIGHTS, %SCALE_5) {dstElemType = f16} : tensor<1x1024x128x!qElemType>, tensor<1x1024x1xf16> -> tensor<1x1024x128xf16>
    %RHS_6 = IE.DynamicDequantize(%WEIGHTS, %SCALE_6) {dstElemType = f16} : tensor<1x1024x128x!qElemType>, tensor<1x1024x1xf16> -> tensor<1x1024x128xf16>
    %RHS_7 = IE.DynamicDequantize(%WEIGHTS, %SCALE_7) {dstElemType = f16} : tensor<1x1024x128x!qElemType>, tensor<1x1024x1xf16> -> tensor<1x1024x128xf16>
    %RHS_8 = IE.DynamicDequantize(%WEIGHTS, %SCALE_8) {dstElemType = f16} : tensor<1x1024x128x!qElemType>, tensor<1x1024x1xf16> -> tensor<1x1024x128xf16>

    %CONCAT = IE.Concat(%RHS_1, %RHS_2, %RHS_3, %RHS_4, %RHS_5, %RHS_6, %RHS_7, %RHS_8) {
        per_axis = #IE.Concat<axis = 0 : i64>
    } : tensor<1x1024x128xf16>, tensor<1x1024x128xf16>, tensor<1x1024x128xf16>, tensor<1x1024x128xf16>,
        tensor<1x1024x128xf16>, tensor<1x1024x128xf16>, tensor<1x1024x128xf16>, tensor<1x1024x128xf16>
        -> tensor<8x1024x128xf16>

    %TRANSPOSE = IE.Transpose(%CONCAT) {order_value = affine_map<(d0, d1, d2) -> (d1, d0, d2)>} : tensor<8x1024x128xf16> -> tensor<1024x8x128xf16>

    %RESHAPE = IE.AffineReshape(%TRANSPOSE) {
        dim_mapping = [[0], [1], [1]],
        shape_value = [1024, 1024]
    } : tensor<1024x8x128xf16> -> tensor<1024x1024xf16>

    %GEMM = IE.FullyConnected(%LHS, %RESHAPE) : tensor<1x1024xf16>, tensor<1024x1024xf16> -> tensor<1x1024xf16>

    return %GEMM : tensor<1x1024xf16>

    // CHECK:   [[DDQ_1:%.+]] = IE.DynamicDequantize([[WEIGHTS]], [[SCALE_1]])
    // CHECK:   [[DDQ_2:%.+]] = IE.DynamicDequantize([[WEIGHTS]], [[SCALE_2]])
    // CHECK:   [[DDQ_3:%.+]] = IE.DynamicDequantize([[WEIGHTS]], [[SCALE_3]])
    // CHECK:   [[DDQ_4:%.+]] = IE.DynamicDequantize([[WEIGHTS]], [[SCALE_4]])
    // CHECK:   [[DDQ_5:%.+]] = IE.DynamicDequantize([[WEIGHTS]], [[SCALE_5]])
    // CHECK:   [[DDQ_6:%.+]] = IE.DynamicDequantize([[WEIGHTS]], [[SCALE_6]])
    // CHECK:   [[DDQ_7:%.+]] = IE.DynamicDequantize([[WEIGHTS]], [[SCALE_7]])
    // CHECK:   [[DDQ_8:%.+]] = IE.DynamicDequantize([[WEIGHTS]], [[SCALE_8]])

    // CHECK:   [[RESHAPE_RHS_1:%.+]] = IE.Reshape([[DDQ_1]]) {shape_value = [1024, 128]} : tensor<1x1024x128xf16> -> tensor<1024x128xf16>
    // CHECK:   [[RESHAPE_RHS_2:%.+]] = IE.Reshape([[DDQ_2]]) {shape_value = [1024, 128]} : tensor<1x1024x128xf16> -> tensor<1024x128xf16>
    // CHECK:   [[RESHAPE_RHS_3:%.+]] = IE.Reshape([[DDQ_3]]) {shape_value = [1024, 128]} : tensor<1x1024x128xf16> -> tensor<1024x128xf16>
    // CHECK:   [[RESHAPE_RHS_4:%.+]] = IE.Reshape([[DDQ_4]]) {shape_value = [1024, 128]} : tensor<1x1024x128xf16> -> tensor<1024x128xf16>
    // CHECK:   [[RESHAPE_RHS_5:%.+]] = IE.Reshape([[DDQ_5]]) {shape_value = [1024, 128]} : tensor<1x1024x128xf16> -> tensor<1024x128xf16>
    // CHECK:   [[RESHAPE_RHS_6:%.+]] = IE.Reshape([[DDQ_6]]) {shape_value = [1024, 128]} : tensor<1x1024x128xf16> -> tensor<1024x128xf16>
    // CHECK:   [[RESHAPE_RHS_7:%.+]] = IE.Reshape([[DDQ_7]]) {shape_value = [1024, 128]} : tensor<1x1024x128xf16> -> tensor<1024x128xf16>
    // CHECK:   [[RESHAPE_RHS_8:%.+]] = IE.Reshape([[DDQ_8]]) {shape_value = [1024, 128]} : tensor<1x1024x128xf16> -> tensor<1024x128xf16>

    // CHECK:   [[LHS_SLICE_1:%.+]] = IE.Slice [[LHS]] [0, 0] [1, 128] : tensor<1x1024xf16> to tensor<1x128xf16>
    // CHECK:   [[LHS_SLICE_2:%.+]] = IE.Slice [[LHS]] [0, 128] [1, 128] : tensor<1x1024xf16> to tensor<1x128xf16>
    // CHECK:   [[LHS_SLICE_3:%.+]] = IE.Slice [[LHS]] [0, 256] [1, 128] : tensor<1x1024xf16> to tensor<1x128xf16>
    // CHECK:   [[LHS_SLICE_4:%.+]] = IE.Slice [[LHS]] [0, 384] [1, 128] : tensor<1x1024xf16> to tensor<1x128xf16>
    // CHECK:   [[LHS_SLICE_5:%.+]] = IE.Slice [[LHS]] [0, 512] [1, 128] : tensor<1x1024xf16> to tensor<1x128xf16>
    // CHECK:   [[LHS_SLICE_6:%.+]] = IE.Slice [[LHS]] [0, 640] [1, 128] : tensor<1x1024xf16> to tensor<1x128xf16>
    // CHECK:   [[LHS_SLICE_7:%.+]] = IE.Slice [[LHS]] [0, 768] [1, 128] : tensor<1x1024xf16> to tensor<1x128xf16>
    // CHECK:   [[LHS_SLICE_8:%.+]] = IE.Slice [[LHS]] [0, 896] [1, 128] : tensor<1x1024xf16> to tensor<1x128xf16>

    // CHECK:   [[GEMM_1:%.+]] = IE.FullyConnected([[LHS_SLICE_1]], [[RESHAPE_RHS_1]]) : tensor<1x128xf16>, tensor<1024x128xf16> -> tensor<1x1024xf16>
    // CHECK:   [[GEMM_RESHAPE_1:%.+]] = IE.Reshape([[GEMM_1]]) {shape_value = [1, 1, 1, 1024]} : tensor<1x1024xf16> -> tensor<1x1x1x1024xf16>

    // CHECK:   [[GEMM_2:%.+]] = IE.FullyConnected([[LHS_SLICE_2]], [[RESHAPE_RHS_2]]) : tensor<1x128xf16>, tensor<1024x128xf16> -> tensor<1x1024xf16>
    // CHECK:   [[GEMM_RESHAPE_2:%.+]] = IE.Reshape([[GEMM_2]]) {shape_value = [1, 1, 1, 1024]} : tensor<1x1024xf16> -> tensor<1x1x1x1024xf16>

    // CHECK:   [[GEMM_3:%.+]] = IE.FullyConnected([[LHS_SLICE_3]], [[RESHAPE_RHS_3]]) : tensor<1x128xf16>, tensor<1024x128xf16> -> tensor<1x1024xf16>
    // CHECK:   [[GEMM_RESHAPE_3:%.+]] = IE.Reshape([[GEMM_3]]) {shape_value = [1, 1, 1, 1024]} : tensor<1x1024xf16> -> tensor<1x1x1x1024xf16>

    // CHECK:   [[GEMM_4:%.+]] = IE.FullyConnected([[LHS_SLICE_4]], [[RESHAPE_RHS_4]]) : tensor<1x128xf16>, tensor<1024x128xf16> -> tensor<1x1024xf16>
    // CHECK:   [[GEMM_RESHAPE_4:%.+]] = IE.Reshape([[GEMM_4]]) {shape_value = [1, 1, 1, 1024]} : tensor<1x1024xf16> -> tensor<1x1x1x1024xf16>

    // CHECK:   [[GEMM_5:%.+]] = IE.FullyConnected([[LHS_SLICE_5]], [[RESHAPE_RHS_5]]) : tensor<1x128xf16>, tensor<1024x128xf16> -> tensor<1x1024xf16>
    // CHECK:   [[GEMM_RESHAPE_5:%.+]] = IE.Reshape([[GEMM_5]]) {shape_value = [1, 1, 1, 1024]} : tensor<1x1024xf16> -> tensor<1x1x1x1024xf16>

    // CHECK:   [[GEMM_6:%.+]] = IE.FullyConnected([[LHS_SLICE_6]], [[RESHAPE_RHS_6]]) : tensor<1x128xf16>, tensor<1024x128xf16> -> tensor<1x1024xf16>
    // CHECK:   [[GEMM_RESHAPE_6:%.+]] = IE.Reshape([[GEMM_6]]) {shape_value = [1, 1, 1, 1024]} : tensor<1x1024xf16> -> tensor<1x1x1x1024xf16>

    // CHECK:   [[GEMM_7:%.+]] = IE.FullyConnected([[LHS_SLICE_7]], [[RESHAPE_RHS_7]]) : tensor<1x128xf16>, tensor<1024x128xf16> -> tensor<1x1024xf16>
    // CHECK:   [[GEMM_RESHAPE_7:%.+]] = IE.Reshape([[GEMM_7]]) {shape_value = [1, 1, 1, 1024]} : tensor<1x1024xf16> -> tensor<1x1x1x1024xf16>

    // CHECK:   [[GEMM_8:%.+]] = IE.FullyConnected([[LHS_SLICE_8]], [[RESHAPE_RHS_8]]) : tensor<1x128xf16>, tensor<1024x128xf16> -> tensor<1x1024xf16>
    // CHECK:   [[GEMM_RESHAPE_8:%.+]] = IE.Reshape([[GEMM_8]]) {shape_value = [1, 1, 1, 1024]} : tensor<1x1024xf16> -> tensor<1x1x1x1024xf16>

    // CHECK:   [[CONCAT_OUT:%.+]] = IE.Concat([[GEMM_RESHAPE_1]], [[GEMM_RESHAPE_2]], [[GEMM_RESHAPE_3]], [[GEMM_RESHAPE_4]], [[GEMM_RESHAPE_5]], [[GEMM_RESHAPE_6]], [[GEMM_RESHAPE_7]], [[GEMM_RESHAPE_8]])
    // CHECK-SAME:  {per_axis = #IE.Concat<axis = 1 : i64>}
    // CHECK-SAME:  -> tensor<1x8x1x1024xf16>

    // CHECK:   [[REDUCE_SUM:%.+]] = IE.ReduceSum([[CONCAT_OUT]]) {axes_value = [1]} : tensor<1x8x1x1024xf16> -> tensor<1x1x1024xf16>
    // CHECK:   [[RESHAPE_OUT:%.+]] = IE.Reshape([[REDUCE_SUM]]) {shape_value = [1, 1024]} : tensor<1x1x1024xf16> -> tensor<1x1024xf16>

    // CHECK:   return [[RESHAPE_OUT]] : tensor<1x1024xf16>
}

// -----

// Verify that U2 weights (levels=4 FakeQuantize) are always unrolled regardless of the
// cost-model threshold that would prevent unrolling for small shapes with i4 weights.
// This mirrors @DontUnrollMatMulNoPerfBenefit but with levels=4 instead of levels=16.

// CHECK-LABEL: @UnrollMatMulU2AlwaysBeneficial
// CHECK-SAME:   [[ARG0:%.+]]: tensor<1x1x64xf16>
func.func @UnrollMatMulU2AlwaysBeneficial(%arg0: tensor<1x1x64xf16>) -> tensor<1x1x64xf16> {
  %cst   = const.Declare tensor<1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1xf16>, [#const.Transpose<affine_map<(d0, d1, d2) -> (d1, d2, d0)>>]
  %cst_0 = const.Declare tensor<1x1x1xf16> = dense<3.000000e+00> : tensor<1x1x1xf16>, [#const.Transpose<affine_map<(d0, d1, d2) -> (d1, d2, d0)>>]
  %cst_1 = const.Declare tensor<1x16x64xf16> = dense<1> : tensor<64x4x16xui2>, [#const.SubView<[0, 0, 0], [64, 1, 16]>, #const.ConvertElemType<ui8>, #const.CastElemType<f16>, #const.Transpose<affine_map<(d0, d1, d2) -> (d1, d2, d0)>>]
  %cst_2 = const.Declare tensor<1x16x64xf16> = dense<1> : tensor<64x4x16xui2>, [#const.SubView<[0, 1, 0], [64, 1, 16]>, #const.ConvertElemType<ui8>, #const.CastElemType<f16>, #const.Transpose<affine_map<(d0, d1, d2) -> (d1, d2, d0)>>]
  %cst_3 = const.Declare tensor<1x16x64xf16> = dense<1> : tensor<64x4x16xui2>, [#const.SubView<[0, 2, 0], [64, 1, 16]>, #const.ConvertElemType<ui8>, #const.CastElemType<f16>, #const.Transpose<affine_map<(d0, d1, d2) -> (d1, d2, d0)>>]
  %cst_4 = const.Declare tensor<1x16x64xf16> = dense<1> : tensor<64x4x16xui2>, [#const.SubView<[0, 3, 0], [64, 1, 16]>, #const.ConvertElemType<ui8>, #const.CastElemType<f16>, #const.Transpose<affine_map<(d0, d1, d2) -> (d1, d2, d0)>>]
  %cst_5  = const.Declare tensor<1x1x64xf16> = dense<1.0> : tensor<64x4x1xf16>, [#const.SubView<[0, 0, 0], [64, 1, 1]>, #const.Transpose<affine_map<(d0, d1, d2) -> (d1, d2, d0)>>]
  %cst_6  = const.Declare tensor<1x1x64xf16> = dense<1.0> : tensor<64x4x1xf16>, [#const.SubView<[0, 1, 0], [64, 1, 1]>, #const.Transpose<affine_map<(d0, d1, d2) -> (d1, d2, d0)>>]
  %cst_7  = const.Declare tensor<1x1x64xf16> = dense<1.0> : tensor<64x4x1xf16>, [#const.SubView<[0, 2, 0], [64, 1, 1]>, #const.Transpose<affine_map<(d0, d1, d2) -> (d1, d2, d0)>>]
  %cst_8  = const.Declare tensor<1x1x64xf16> = dense<1.0> : tensor<64x4x1xf16>, [#const.SubView<[0, 3, 0], [64, 1, 1]>, #const.Transpose<affine_map<(d0, d1, d2) -> (d1, d2, d0)>>]
  %cst_9  = const.Declare tensor<1x1x64xf16> = dense<1.0> : tensor<64x4x1xf16>, [#const.SubView<[0, 0, 0], [64, 1, 1]>, #const.Transpose<affine_map<(d0, d1, d2) -> (d1, d2, d0)>>]
  %cst_10 = const.Declare tensor<1x1x64xf16> = dense<1.0> : tensor<64x4x1xf16>, [#const.SubView<[0, 1, 0], [64, 1, 1]>, #const.Transpose<affine_map<(d0, d1, d2) -> (d1, d2, d0)>>]
  %cst_11 = const.Declare tensor<1x1x64xf16> = dense<1.0> : tensor<64x4x1xf16>, [#const.SubView<[0, 2, 0], [64, 1, 1]>, #const.Transpose<affine_map<(d0, d1, d2) -> (d1, d2, d0)>>]
  %cst_12 = const.Declare tensor<1x1x64xf16> = dense<1.0> : tensor<64x4x1xf16>, [#const.SubView<[0, 3, 0], [64, 1, 1]>, #const.Transpose<affine_map<(d0, d1, d2) -> (d1, d2, d0)>>]

  %0 = IE.FakeQuantize(%cst_1, %cst, %cst_0, %cst_5,  %cst_9)  {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 4 : i64} : tensor<1x16x64xf16>, tensor<1x1x1xf16>, tensor<1x1x1xf16>, tensor<1x1x64xf16>, tensor<1x1x64xf16> -> tensor<1x16x64xf16>
  %1 = IE.FakeQuantize(%cst_2, %cst, %cst_0, %cst_6,  %cst_10) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 4 : i64} : tensor<1x16x64xf16>, tensor<1x1x1xf16>, tensor<1x1x1xf16>, tensor<1x1x64xf16>, tensor<1x1x64xf16> -> tensor<1x16x64xf16>
  %2 = IE.FakeQuantize(%cst_3, %cst, %cst_0, %cst_7,  %cst_11) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 4 : i64} : tensor<1x16x64xf16>, tensor<1x1x1xf16>, tensor<1x1x1xf16>, tensor<1x1x64xf16>, tensor<1x1x64xf16> -> tensor<1x16x64xf16>
  %3 = IE.FakeQuantize(%cst_4, %cst, %cst_0, %cst_8,  %cst_12) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 4 : i64} : tensor<1x16x64xf16>, tensor<1x1x1xf16>, tensor<1x1x1xf16>, tensor<1x1x64xf16>, tensor<1x1x64xf16> -> tensor<1x16x64xf16>

  %4 = IE.Concat(%0, %1, %2, %3) {per_axis = #IE.Concat<axis = 0 : i64>} : tensor<1x16x64xf16>, tensor<1x16x64xf16>, tensor<1x16x64xf16>, tensor<1x16x64xf16> -> tensor<4x16x64xf16>
  %5 = IE.Transpose(%4) {order_value = affine_map<(d0, d1, d2) -> (d2, d0, d1)>} : tensor<4x16x64xf16> -> tensor<64x4x16xf16>
  %6 = IE.AffineReshape(%5) {dim_mapping = [[0], [1], [1]], shape_value = [64, 64]} : tensor<64x4x16xf16> -> tensor<64x64xf16>
  %7 = IE.AffineReshape(%arg0) {dim_mapping = [[0], [0], [1]], shape_value = [1, 64]} : tensor<1x1x64xf16> -> tensor<1x64xf16>
  %8 = IE.FullyConnected(%7, %6) : tensor<1x64xf16>, tensor<64x64xf16> -> tensor<1x64xf16>
  %9 = IE.AffineReshape(%8) {dim_mapping = [[0, 1], [2]], shape_value = [1, 1, 64]} : tensor<1x64xf16> -> tensor<1x1x64xf16>
  return %9 : tensor<1x1x64xf16>


  // CHECK:   IE.FakeQuantize
  // CHECK:   IE.FakeQuantize
  // CHECK:   IE.FakeQuantize
  // CHECK:   IE.FakeQuantize
  // CHECK:   IE.FullyConnected
  // CHECK:   IE.FullyConnected
  // CHECK:   IE.FullyConnected
  // CHECK:   IE.FullyConnected
  // CHECK:   IE.ReduceSum
}
