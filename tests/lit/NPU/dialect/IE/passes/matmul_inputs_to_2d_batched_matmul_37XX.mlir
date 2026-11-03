//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --run-batch-op-processing-rewriters="enable-grouped-matmul=true rewriter=matmul-inputs-to-2d-set" %s | FileCheck %s
// REQUIRES: platform-NPU3720

// Two test below show that CMX calculations are not affected by quantization
// As currently quantization for grouped matmul is not enabled.
// Tests are valid only for NPU3720 due to dependency on CMX memory otherwise
// batched matmul related changes are not specific to NPU3720.

// CHECK-LABEL: @MatMul4dInputs4dWeightsTo2dQuantized
// CHECK-SAME:  [[ARG0:%.+]]: tensor<1x2x650x750xf16>
func.func @MatMul4dInputs4dWeightsTo2dQuantized(%arg0: tensor<1x2x650x750xf16>) -> tensor<1x2x650x512xf16> {
    %cst = const.Declare tensor<1x1x1x1xf16> = dense<-0.285366833> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
    %cst_1 = const.Declare tensor<1x1x1x1xf16> = dense<0.435113788> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
    %cst_2 = const.Declare tensor<1x1x1x1xf16> = dense<2.550000e+02> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
    %cst_3 = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
    %cst_4 = const.Declare tensor<1x2x512x750xf16> = dense<1.0> : tensor<1x2x512x750xf32>, [#const.CastElemType<f16>]
    %0 = IE.FakeQuantize(%cst_4, %cst_3, %cst_2, %cst, %cst_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels_or_low_fp_type = 256 : i64} : tensor<1x2x512x750xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x2x512x750xf16>
    %1 = IE.FakeQuantize(%arg0, %cst_3, %cst_2, %cst, %cst_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels_or_low_fp_type = 256 : i64} : tensor<1x2x650x750xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x2x650x750xf16>
    %2 = IE.MatMul(%1, %0) {transpose_b} : tensor<1x2x650x750xf16>, tensor<1x2x512x750xf16> -> tensor<1x2x650x512xf16>

    return %2 : tensor<1x2x650x512xf16>

    // CHECK:       [[CST:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<-0.285366833> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
    // CHECK:       [[CST0:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<0.435113788> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
    // CHECK:       [[CST1:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<2.550000e+02> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
    // CHECK:       [[CST2:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
    // CHECK:       [[CST3:%.+]] = const.Declare tensor<1x2x512x750xf16> = dense<1.000000e+00> : tensor<1x2x512x750xf32>, [#const.CastElemType<f16>]
    // CHECK:       [[FQ0:%.+]] = IE.FakeQuantize([[CST3]], [[CST2]], [[CST1]], [[CST]], [[CST0]])
    // CHECK-SAME:  {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels_or_low_fp_type = 256 : i64} :
    // CHECK-SAME:  tensor<1x2x512x750xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x2x512x750xf16>
    // CHECK:       [[FQ1:%.+]] = IE.FakeQuantize([[ARG0]], [[CST2]], [[CST1]], [[CST]], [[CST0]])
    // CHECK-SAME:  {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels_or_low_fp_type = 256 : i64} :
    // CHECK-SAME:  tensor<1x2x650x750xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x2x650x750xf16>
    // CHECK:       [[SLICE0:%.+]] = IE.Slice [[FQ1]] [0, 0, 0, 0] [1, 1, 650, 750] : tensor<1x2x650x750xf16> to tensor<1x1x650x750xf16>
    // CHECK:       [[RESHAPE0:%.+]] = IE.AffineReshape([[SLICE0]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [0], [0], [1]], shape_value = [650, 750]} : tensor<1x1x650x750xf16> -> tensor<650x750xf16>
    // CHECK:       [[SLICE1:%.+]] = IE.Slice [[FQ1]] [0, 1, 0, 0] [1, 1, 650, 750] : tensor<1x2x650x750xf16> to tensor<1x1x650x750xf16>
    // CHECK:       [[RESHAPE1:%.+]] = IE.AffineReshape([[SLICE1]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [0], [0], [1]], shape_value = [650, 750]} : tensor<1x1x650x750xf16> -> tensor<650x750xf16>
    // CHECK:       [[SLICE2:%.+]] = IE.Slice [[FQ0]] [0, 0, 0, 0] [1, 1, 512, 750] : tensor<1x2x512x750xf16> to tensor<1x1x512x750xf16>
    // CHECK:       [[RESHAPE2:%.+]] = IE.AffineReshape([[SLICE2]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [0], [0], [1]], shape_value = [512, 750]} : tensor<1x1x512x750xf16> -> tensor<512x750xf16>
    // CHECK:       [[SLICE3:%.+]] = IE.Slice [[FQ0]] [0, 1, 0, 0] [1, 1, 512, 750] : tensor<1x2x512x750xf16> to tensor<1x1x512x750xf16>
    // CHECK:       [[RESHAPE3:%.+]] = IE.AffineReshape([[SLICE3]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [0], [0], [1]], shape_value = [512, 750]} : tensor<1x1x512x750xf16> -> tensor<512x750xf16>
    // CHECK:       [[MATMUL0:%.+]] = IE.MatMul([[RESHAPE0]], [[RESHAPE2]]) {transpose_b} : tensor<650x750xf16>, tensor<512x750xf16> -> tensor<650x512xf16>
    // CHECK:       [[MATMUL1:%.+]] = IE.MatMul([[RESHAPE1]], [[RESHAPE3]]) {transpose_b} : tensor<650x750xf16>, tensor<512x750xf16> -> tensor<650x512xf16>
    // CHECK:       [[CONCAT0:%.+]] = IE.Concat([[MATMUL0]], [[MATMUL1]]) {per_axis = #IE.Concat<axis = 0 : i64>} : tensor<650x512xf16>,
    // CHECK-SAME:  tensor<650x512xf16> -> tensor<1300x512xf16>
    // CHECK:       [[RESHAPE4:%.+]] = IE.AffineReshape([[CONCAT0]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 2, 650, 512]} : tensor<1300x512xf16> -> tensor<1x2x650x512xf16>
    // CHECK: return [[RESHAPE4]] : tensor<1x2x650x512xf16>
}

// -----

// CHECK-LABEL: @MatMul4dInputs4dWeightsTo2dNoQuantizeDontFitCMX
// CHECK-SAME:  [[ARG0:%.+]]: tensor<1x2x650x750xf16>
// CHECK-SAME:  [[ARG1:%.+]]: tensor<1x2x512x750xf16>
func.func @MatMul4dInputs4dWeightsTo2dNoQuantizeDontFitCMX(%arg0: tensor<1x2x650x750xf16>, %arg1: tensor<1x2x512x750xf16>) -> tensor<1x2x650x512xf16> {
    %0 = IE.MatMul(%arg0, %arg1) {transpose_b} : tensor<1x2x650x750xf16>, tensor<1x2x512x750xf16> -> tensor<1x2x650x512xf16>

    return %0 : tensor<1x2x650x512xf16>


    // CHECK:       [[SLICE0:%.+]] = IE.Slice [[ARG0]] [0, 0, 0, 0] [1, 1, 650, 750] : tensor<1x2x650x750xf16> to tensor<1x1x650x750xf16>
    // CHECK:       [[RESHAPE0:%.+]] = IE.AffineReshape([[SLICE0]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [0], [0], [1]], shape_value = [650, 750]} : tensor<1x1x650x750xf16> -> tensor<650x750xf16>
    // CHECK:       [[SLICE1:%.+]] = IE.Slice [[ARG0]] [0, 1, 0, 0] [1, 1, 650, 750] : tensor<1x2x650x750xf16> to tensor<1x1x650x750xf16>
    // CHECK:       [[RESHAPE1:%.+]] = IE.AffineReshape([[SLICE1]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [0], [0], [1]], shape_value = [650, 750]} : tensor<1x1x650x750xf16> -> tensor<650x750xf16>
    // CHECK:       [[SLICE2:%.+]] = IE.Slice [[ARG1]] [0, 0, 0, 0] [1, 1, 512, 750] : tensor<1x2x512x750xf16> to tensor<1x1x512x750xf16>
    // CHECK:       [[RESHAPE2:%.+]] = IE.AffineReshape([[SLICE2]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [0], [0], [1]], shape_value = [512, 750]} : tensor<1x1x512x750xf16> -> tensor<512x750xf16>
    // CHECK:       [[SLICE3:%.+]] = IE.Slice [[ARG1]] [0, 1, 0, 0] [1, 1, 512, 750] : tensor<1x2x512x750xf16> to tensor<1x1x512x750xf16>
    // CHECK:       [[RESHAPE3:%.+]] = IE.AffineReshape([[SLICE3]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [0], [0], [1]], shape_value = [512, 750]} : tensor<1x1x512x750xf16> -> tensor<512x750xf16>
    // CHECK:       [[MATMUL0:%.+]] = IE.MatMul([[RESHAPE0]], [[RESHAPE2]]) {transpose_b} : tensor<650x750xf16>, tensor<512x750xf16> -> tensor<650x512xf16>
    // CHECK:       [[MATMUL1:%.+]] = IE.MatMul([[RESHAPE1]], [[RESHAPE3]]) {transpose_b} : tensor<650x750xf16>, tensor<512x750xf16> -> tensor<650x512xf16>
    // CHECK:       [[CONCAT:%.+]] = IE.Concat([[MATMUL0]], [[MATMUL1]]) {per_axis = #IE.Concat<axis = 0 : i64>} : tensor<650x512xf16>,
    // CHECK-SAME:  tensor<650x512xf16> -> tensor<1300x512xf16>
    // CHECK:       [[RESHAPE4:%.+]] = IE.AffineReshape([[CONCAT]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 2, 650, 512]} : tensor<1300x512xf16> -> tensor<1x2x650x512xf16>
    // CHECK:       return [[RESHAPE4]] : tensor<1x2x650x512xf16>
}

// -----

// CHECK-LABEL: @MatMul4dInputs4dWeightsTo2dOutputQuantized
// CHECK-SAME:  [[ARG0:%.+]]: tensor<1x2x650x750xf16>
func.func @MatMul4dInputs4dWeightsTo2dOutputQuantized(%arg0: tensor<1x2x650x750xf16>) -> tensor<1x2x650x512xf16> {
    %cst = const.Declare tensor<1x1x1x1xf16> = dense<-0.285366833> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
    %cst_1 = const.Declare tensor<1x1x1x1xf16> = dense<0.435113788> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
    %cst_2 = const.Declare tensor<1x1x1x1xf16> = dense<2.550000e+02> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
    %cst_3 = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
    %cst_4 = const.Declare tensor<1x2x512x750xf16> = dense<1.0> : tensor<1x2x512x750xf32>, [#const.CastElemType<f16>]
    %0 = IE.FakeQuantize(%cst_4, %cst_3, %cst_2, %cst, %cst_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels_or_low_fp_type = 256 : i64} : tensor<1x2x512x750xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x2x512x750xf16>
    %1 = IE.FakeQuantize(%arg0, %cst_3, %cst_2, %cst, %cst_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels_or_low_fp_type = 256 : i64} : tensor<1x2x650x750xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x2x650x750xf16>
    %2 = IE.MatMul(%1, %0) {transpose_b} : tensor<1x2x650x750xf16>, tensor<1x2x512x750xf16> -> tensor<1x2x650x512xf16>
    %3 = IE.FakeQuantize(%2, %cst_3, %cst_2, %cst, %cst_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels_or_low_fp_type = 256 : i64} : tensor<1x2x650x512xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x2x650x512xf16>

    return %3 : tensor<1x2x650x512xf16>

    // CHECK:       [[CST:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<-0.285366833> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
    // CHECK:       [[CST0:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<0.435113788> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
    // CHECK:       [[CST1:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<2.550000e+02> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
    // CHECK:       [[CST2:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
    // CHECK:       [[CST3:%.+]] = const.Declare tensor<1x2x512x750xf16> = dense<1.000000e+00> : tensor<1x2x512x750xf32>, [#const.CastElemType<f16>]
    // CHECK:       [[FQ0:%.+]] = IE.FakeQuantize([[CST3]], [[CST2]], [[CST1]], [[CST]], [[CST0]])
    // CHECK-SAME:  {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels_or_low_fp_type = 256 : i64} :
    // CHECK-SAME:  tensor<1x2x512x750xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x2x512x750xf16>
    // CHECK:       [[FQ1:%.+]] = IE.FakeQuantize([[ARG0]], [[CST2]], [[CST1]], [[CST]], [[CST0]])
    // CHECK-SAME:  {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels_or_low_fp_type = 256 : i64} :
    // CHECK-SAME:  tensor<1x2x650x750xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x2x650x750xf16>
    // CHECK:       [[SLICE0:%.+]] = IE.Slice [[FQ1]] [0, 0, 0, 0] [1, 1, 650, 750] : tensor<1x2x650x750xf16> to tensor<1x1x650x750xf16>
    // CHECK:       [[RESHAPE0:%.+]] = IE.AffineReshape([[SLICE0]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [0], [0], [1]], shape_value = [650, 750]} : tensor<1x1x650x750xf16> -> tensor<650x750xf16>
    // CHECK:       [[SLICE1:%.+]] = IE.Slice [[FQ1]] [0, 1, 0, 0] [1, 1, 650, 750] : tensor<1x2x650x750xf16> to tensor<1x1x650x750xf16>
    // CHECK:       [[RESHAPE1:%.+]] = IE.AffineReshape([[SLICE1]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [0], [0], [1]], shape_value = [650, 750]} : tensor<1x1x650x750xf16> -> tensor<650x750xf16>
    // CHECK:       [[SLICE2:%.+]] = IE.Slice [[FQ0]] [0, 0, 0, 0] [1, 1, 512, 750] : tensor<1x2x512x750xf16> to tensor<1x1x512x750xf16>
    // CHECK:       [[RESHAPE2:%.+]] = IE.AffineReshape([[SLICE2]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [0], [0], [1]], shape_value = [512, 750]} : tensor<1x1x512x750xf16> -> tensor<512x750xf16>
    // CHECK:       [[SLICE3:%.+]] = IE.Slice [[FQ0]] [0, 1, 0, 0] [1, 1, 512, 750] : tensor<1x2x512x750xf16> to tensor<1x1x512x750xf16>
    // CHECK:       [[RESHAPE3:%.+]] = IE.AffineReshape([[SLICE3]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [0], [0], [1]], shape_value = [512, 750]} : tensor<1x1x512x750xf16> -> tensor<512x750xf16>
    // CHECK:       [[MATMUL0:%.+]] = IE.MatMul([[RESHAPE0]], [[RESHAPE2]]) {transpose_b} : tensor<650x750xf16>, tensor<512x750xf16> -> tensor<650x512xf16>
    // CHECK:       [[MATMUL1:%.+]] = IE.MatMul([[RESHAPE1]], [[RESHAPE3]]) {transpose_b} : tensor<650x750xf16>, tensor<512x750xf16> -> tensor<650x512xf16>
    // CHECK:       [[CONCAT0:%.+]] = IE.Concat([[MATMUL0]], [[MATMUL1]]) {per_axis = #IE.Concat<axis = 0 : i64>} :
    // CHECK-SAME:  tensor<650x512xf16>, tensor<650x512xf16> -> tensor<1300x512xf16>
    // CHECK:       [[RESHAPE4:%.+]] = IE.AffineReshape([[CONCAT0]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 2, 650, 512]} : tensor<1300x512xf16> -> tensor<1x2x650x512xf16>
    // CHECK:       [[FQ2:%.+]] = IE.FakeQuantize([[RESHAPE4]], [[CST2]], [[CST1]], [[CST]], [[CST0]])
    // CHECK-SAME:  {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels_or_low_fp_type = 256 : i64} :
    // CHECK-SAME:  tensor<1x2x650x512xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x2x650x512xf16>
    // CHECK:       return [[FQ2]] : tensor<1x2x650x512xf16>
}
