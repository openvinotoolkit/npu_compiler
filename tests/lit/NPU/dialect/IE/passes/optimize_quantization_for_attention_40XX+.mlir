//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --optimize-quantization-for-attention  %s | FileCheck %s
// REQUIRES: platform-NPU4000 || platform-NPU5010


// CHECK-LABEL: @OptimizeQuantizationForAttention
// CHECK-SAME:  ([[INPUT0:%.+]]: tensor<1x1xsi32>, [[INPUT1:%.+]]: tensor<1x10x8192x128xsi8>, [[INPUT2:%.+]]: tensor<1x10x1x128xf16>,
// CHECK-SAME:  [[INPUT3:%.+]]: tensor<1x10x1x128xf32>, [[INPUT4:%.+]]: tensor<1x10x8192x128xf32>, [[INPUT5:%.+]]: tensor<1x8192xf32>)
func.func @OptimizeQuantizationForAttention(%past_seq_len: tensor<1x1xsi32>, %past_values: tensor<1x10x8192x128xsi8>, %input: tensor<1x10x1x128xf16>,
                                                %sdpa_in_0: tensor<1x10x1x128xf32>, %sdpa_in_1: tensor<1x10x8192x128xf32>, %sdpa_in_2: tensor<1x8192xf32> )
                    -> (tensor<1xsi64>, tensor<1x10x8192x128xf32>, tensor<1x10x8192x128xsi8>, tensor<1x10x1x128xf32>) {
    %cst = const.Declare tensor<1xsi64> = dense<1> : tensor<1xsi64>
    %cst_0 = const.Declare tensor<1x1xsi64> = dense<1> : tensor<1x1xsi64>
    %convert_0 = IE.Convert(%past_seq_len) {dstElemType = si64} : tensor<1x1xsi32> -> tensor<1x1xsi64>
    %add = IE.Add(%convert_0, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1xsi64>, tensor<1x1xsi64> -> tensor<1x1xsi64>
    %reshape = IE.AffineReshape(%add) {dim_mapping = [[0], [0]], shape_value = [1]} : tensor<1x1xsi64> -> tensor<1xsi64>
    %subtract = IE.Add(%reshape, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1xsi64>, tensor<1xsi64> -> tensor<1xsi64>

    %cst_1 = const.Declare tensor<1x10x1x128xf32> = dense<2.0> : tensor<1x10x1x128xf32>
    %convert_1 = IE.Convert(%past_values) {dstElemType = f32} : tensor<1x10x8192x128xsi8> -> tensor<1x10x8192x128xf32>
    %multiply_0 = IE.Multiply(%convert_1, %cst_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
            : tensor<1x10x8192x128xf32>, tensor<1x10x1x128xf32> -> tensor<1x10x8192x128xf32>


    %convert_2 = IE.Convert(%input) {dstElemType = f32} : tensor<1x10x1x128xf16> -> tensor<1x10x1x128xf32>

    %cst_2 = const.Declare tensor<1x10x1x128xf32> = dense<3.0> : tensor<1x10x1x128xf32>
    %scatter_update = IE.ScatterUpdate(%multiply_0, %subtract, %convert_2) {axis_value = 2 : i64}
            : tensor<1x10x8192x128xf32>, tensor<1xsi64>, tensor<1x10x1x128xf32> -> tensor<1x10x8192x128xf32>
    %multiply_1 = IE.Multiply(%scatter_update, %cst_2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
            : tensor<1x10x8192x128xf32>, tensor<1x10x1x128xf32> -> tensor<1x10x8192x128xf32>
    %round = IE.Round(%multiply_1) {mode = #IE.round_mode<HALF_TO_EVEN>} : tensor<1x10x8192x128xf32> -> tensor<1x10x8192x128xf32>
    %clamp = IE.Clamp(%round) {max = 1.270000e+02 : f64, min = -1.280000e+02 : f64} : tensor<1x10x8192x128xf32> -> tensor<1x10x8192x128xf32>
    %present_values = IE.Convert(%clamp) {dstElemType = si8} : tensor<1x10x8192x128xf32> -> tensor<1x10x8192x128xsi8>

    %sdpa = IE.SDPA(%sdpa_in_0, %scatter_update, %sdpa_in_1, %sdpa_in_2) {operandSegmentSizes = array<i32: 1, 1, 1, 1, 0, 0>} : tensor<1x10x1x128xf32>, tensor<1x10x8192x128xf32>, tensor<1x10x8192x128xf32>, tensor<1x8192xf32> -> tensor<1x10x1x128xf32>

    return %subtract, %scatter_update, %present_values, %sdpa : tensor<1xsi64>, tensor<1x10x8192x128xf32>, tensor<1x10x8192x128xsi8>, tensor<1x10x1x128xf32>

    // CHECK:    [[CST:%.+]] = const.Declare tensor<1xsi64> = dense<1> : tensor<1xsi64>
    // CHECK:    [[CST_0:%.+]] = const.Declare tensor<1x1xsi64> = dense<1> : tensor<1x1xsi64>
    // CHECK:    [[CONVERT:%.+]] = IE.Convert([[INPUT0]]) {dstElemType = si64} : tensor<1x1xsi32> -> tensor<1x1xsi64>
    // CHECK:    [[ADD:%.+]] = IE.Add([[CONVERT]], [[CST_0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK-SAME:          : tensor<1x1xsi64>, tensor<1x1xsi64> -> tensor<1x1xsi64>
    // CHECK:    [[AFFINE_RESHAPE:%.+]] = IE.AffineReshape([[ADD]])
    // CHECK-SAME{LITERAL}:          {dim_mapping = [[0], [0]], shape_value = [1]} : tensor<1x1xsi64> -> tensor<1xsi64>
    // CHECK:    [[ADD_0:%.+]] = IE.Add([[AFFINE_RESHAPE]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK-SAME:          : tensor<1xsi64>, tensor<1xsi64> -> tensor<1xsi64>

    // CHECK:    [[CST_1:%.+]] = const.Declare tensor<1x10x1x128xf32> = dense<2.000000e+00> : tensor<1x10x1x128xf32>
    // CHECK:    [[CONVERT_0:%.+]] = IE.Convert([[INPUT2]]) {dstElemType = f32} : tensor<1x10x1x128xf16> -> tensor<1x10x1x128xf32>
    // CHECK:    [[CST_2:%.+]] = const.Declare tensor<1x10x1x128xf32> = dense<3.000000e+00> : tensor<1x10x1x128xf32>
    // CHECK:    [[MULTIPLY:%.+]] = IE.Multiply([[CONVERT_0]], [[CST_2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK-SAME:          : tensor<1x10x1x128xf32>, tensor<1x10x1x128xf32> -> tensor<1x10x1x128xf32>
    // CHECK:    [[ROUND:%.+]] = IE.Round([[MULTIPLY]]) {mode = #IE.round_mode<HALF_TO_EVEN>} : tensor<1x10x1x128xf32> -> tensor<1x10x1x128xf32>
    // CHECK:    [[CLAMP:%.+]] = IE.Clamp([[ROUND]]) {max = 1.270000e+02 : f64, min = -1.280000e+02 : f64}
    // CHECK-SAME:          : tensor<1x10x1x128xf32> -> tensor<1x10x1x128xf32>
    // CHECK:    [[CONVERT_1:%.+]] = IE.Convert([[CLAMP]]) {dstElemType = si8} : tensor<1x10x1x128xf32> -> tensor<1x10x1x128xsi8>

    // CHECK:    [[SCATTER_UPDATE:%.+]] = IE.ScatterUpdate([[INPUT1]], [[ADD_0]], [[CONVERT_1]]) {axis_value = 2 : i64}
    // CHECK-SAME:          : tensor<1x10x8192x128xsi8>, tensor<1xsi64>, tensor<1x10x1x128xsi8> -> tensor<1x10x8192x128xsi8>

    // CHECK:    [[CONVERT_2:%.+]] = IE.Convert([[SCATTER_UPDATE]]) {dstElemType = f32} : tensor<1x10x8192x128xsi8> -> tensor<1x10x8192x128xf32>
    // CHECK:    [[MULTIPLY_0:%.+]] = IE.Multiply([[CONVERT_2]], [[CST_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK-SAME:          : tensor<1x10x8192x128xf32>, tensor<1x10x1x128xf32> -> tensor<1x10x8192x128xf32>

    // CHECK:    [[SDPA:%.+]] = IE.SDPA([[INPUT3]], [[MULTIPLY_0]], [[INPUT4]], [[INPUT5]]) {operandSegmentSizes = array<i32: 1, 1, 1, 1, 0, 0>} : tensor<1x10x1x128xf32>, tensor<1x10x8192x128xf32>, tensor<1x10x8192x128xf32>, tensor<1x8192xf32> -> tensor<1x10x1x128xf32>

    // CHECK:    return [[ADD_0]], [[MULTIPLY_0]], [[SCATTER_UPDATE]], [[SDPA]] : tensor<1xsi64>, tensor<1x10x8192x128xf32>, tensor<1x10x8192x128xsi8>, tensor<1x10x1x128xf32>
}
