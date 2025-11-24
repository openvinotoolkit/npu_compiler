//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --run-adjust-for-vpu-rewriters="rewriter=merge-weights-shared-conv" %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX


// CHECK-LABEL: @MergeWeightsSharedConv
// CHECK-SAME:      [[INPUT0:%.+]]: tensor<1x128x1x1xf16>, [[INPUT1:%.+]]: tensor<1x128x1x1xf16>, [[INPUT2:%.+]]: tensor<1x128x1x1xf16>
func.func @MergeWeightsSharedConv(%arg0: tensor<1x128x1x1xf16>, %arg1: tensor<1x128x1x1xf16>, %arg2: tensor<1x128x1x1xf16>) -> (tensor<1x16x3x1xf16>) {
    %filter = const.Declare tensor<16x128x1x1xf16> = dense<0.000000e+00> : tensor<16x128x1x1xf16>
    %0 = IE.Convolution(%arg0, %filter) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x128x1x1xf16>, tensor<16x128x1x1xf16> -> tensor<1x16x1x1xf16>
    %1 = IE.Convolution(%arg1, %filter) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x128x1x1xf16>, tensor<16x128x1x1xf16> -> tensor<1x16x1x1xf16>
    %2 = IE.Convolution(%arg2, %filter) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x128x1x1xf16>, tensor<16x128x1x1xf16> -> tensor<1x16x1x1xf16>
    %3 = IE.Concat(%0, %1, %2) {per_axis = #IE.Concat<axis = 2>} : tensor<1x16x1x1xf16>, tensor<1x16x1x1xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x3x1xf16>
    return %3 : tensor<1x16x3x1xf16>


    // CHECK-DAG:   [[FILTER:%.+]]  = const.Declare tensor<16x128x1x1xf16> = dense<0.000000e+00> : tensor<16x128x1x1xf16>
    // CHECK:       [[IN_CONCAT:%.+]]  = IE.Concat([[INPUT2]], [[INPUT1]], [[INPUT0]]) {per_axis = #IE.Concat<axis = 2 : i64>} : tensor<1x128x1x1xf16>, tensor<1x128x1x1xf16>, tensor<1x128x1x1xf16> -> tensor<1x128x3x1xf16>
    // CHECK:       [[CONV:%.+]]  = IE.Convolution([[IN_CONCAT]], [[FILTER]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x128x3x1xf16>, tensor<16x128x1x1xf16> -> tensor<1x16x3x1xf16>
    // CHECK:       [[SLICE_0:%.+]] = IE.Slice [[CONV]]  [0, 0, 0, 0] [1, 16, 1, 1] : tensor<1x16x3x1xf16> to tensor<1x16x1x1xf16>
    // CHECK:       [[SLICE_1:%.+]] = IE.Slice [[CONV]]  [0, 0, 1, 0] [1, 16, 1, 1] : tensor<1x16x3x1xf16> to tensor<1x16x1x1xf16>
    // CHECK:       [[SLICE_2:%.+]] = IE.Slice [[CONV]]  [0, 0, 2, 0] [1, 16, 1, 1] : tensor<1x16x3x1xf16> to tensor<1x16x1x1xf16>
    // CHECK:       [[OUT_CONCAT:%.+]] = IE.Concat([[SLICE_2]], [[SLICE_1]], [[SLICE_0]]) {per_axis = #IE.Concat<axis = 2 : i64>} : tensor<1x16x1x1xf16>, tensor<1x16x1x1xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x3x1xf16>
    // CHECK:       return  [[OUT_CONCAT]]
}

// -----

// CHECK-LABEL: @MergeWeightsSharedBeforeAffineReshape
// CHECK-SAME:      [[INPUT0:%.+]]: tensor<1x128x1x1xf16>, [[INPUT1:%.+]]: tensor<1x128x1x1xf16>, [[INPUT2:%.+]]: tensor<1x128x1x1xf16>, [[INPUT3:%.+]]: tensor<16x128x1xf16>
func.func @MergeWeightsSharedBeforeAffineReshape(%arg0: tensor<1x128x1x1xf16>, %arg1: tensor<1x128x1x1xf16>, %arg2: tensor<1x128x1x1xf16>, %arg3: tensor<16x128x1xf16>) -> (tensor<1x16x3x1xf16>) {
    %0 = IE.AffineReshape(%arg3) {dim_mapping = [[0], [1], [2], [2]], shape_value = [16, 128, 1, 1]} : tensor<16x128x1xf16> -> tensor<16x128x1x1xf16>
    %1 = IE.AffineReshape(%arg3) {dim_mapping = [[0], [1], [2], [2]], shape_value = [16, 128, 1, 1]} : tensor<16x128x1xf16> -> tensor<16x128x1x1xf16>
    %2 = IE.AffineReshape(%arg3) {dim_mapping = [[0], [1], [2], [2]], shape_value = [16, 128, 1, 1]} : tensor<16x128x1xf16> -> tensor<16x128x1x1xf16>
    %3 = IE.Convolution(%arg0, %0) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x128x1x1xf16>, tensor<16x128x1x1xf16> -> tensor<1x16x1x1xf16>
    %4 = IE.Convolution(%arg1, %1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x128x1x1xf16>, tensor<16x128x1x1xf16> -> tensor<1x16x1x1xf16>
    %5 = IE.Convolution(%arg2, %2) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x128x1x1xf16>, tensor<16x128x1x1xf16> -> tensor<1x16x1x1xf16>
    %6 = IE.Concat(%3, %4, %5) {per_axis = #IE.Concat<axis = 2>} : tensor<1x16x1x1xf16>, tensor<1x16x1x1xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x3x1xf16>
    return %6 : tensor<1x16x3x1xf16>


    // CHECK:       [[RESHAPE:%.+]]  = IE.AffineReshape([[INPUT3]])
    // CHECK-SAME{LITERAL}:        {dim_mapping = [[0], [1], [2], [2]], shape_value = [16, 128, 1, 1]} : tensor<16x128x1xf16> -> tensor<16x128x1x1xf16>
    // CHECK:       [[IN_CONCAT:%.+]]  = IE.Concat([[INPUT2]], [[INPUT1]], [[INPUT0]]) {per_axis = #IE.Concat<axis = 2 : i64>} : tensor<1x128x1x1xf16>, tensor<1x128x1x1xf16>, tensor<1x128x1x1xf16> -> tensor<1x128x3x1xf16>
    // CHECK:       [[CONV:%.+]]  = IE.Convolution([[IN_CONCAT]], [[RESHAPE]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x128x3x1xf16>, tensor<16x128x1x1xf16> -> tensor<1x16x3x1xf16>
    // CHECK:       [[SLICE_0:%.+]] = IE.Slice [[CONV]]  [0, 0, 0, 0] [1, 16, 1, 1] : tensor<1x16x3x1xf16> to tensor<1x16x1x1xf16>
    // CHECK:       [[SLICE_1:%.+]] = IE.Slice [[CONV]]  [0, 0, 1, 0] [1, 16, 1, 1] : tensor<1x16x3x1xf16> to tensor<1x16x1x1xf16>
    // CHECK:       [[SLICE_2:%.+]] = IE.Slice [[CONV]]  [0, 0, 2, 0] [1, 16, 1, 1] : tensor<1x16x3x1xf16> to tensor<1x16x1x1xf16>
    // CHECK:       [[OUT_CONCAT:%.+]] = IE.Concat([[SLICE_2]], [[SLICE_1]], [[SLICE_0]]) {per_axis = #IE.Concat<axis = 2 : i64>} : tensor<1x16x1x1xf16>, tensor<1x16x1x1xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x3x1xf16>
    // CHECK:       return  [[OUT_CONCAT]]
}

// -----

// CHECK-LABEL: @NotMergeForDifferentParameter
// CHECK-SAME:      [[INPUT0:%.+]]: tensor<1x128x1x1xf16>, [[INPUT1:%.+]]: tensor<1x128x1x1xf16>
func.func @NotMergeForDifferentParameter(%arg0: tensor<1x128x1x1xf16>, %arg1: tensor<1x128x1x1xf16>) -> (tensor<1x16x3x1xf16>) {
    %filter = const.Declare tensor<16x128x1x1xf16> = dense<0.000000e+00> : tensor<16x128x1x1xf16>
    %0 = IE.Convolution(%arg0, %filter) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x128x1x1xf16>, tensor<16x128x1x1xf16> -> tensor<1x16x1x1xf16>
    %1 = IE.Convolution(%arg1, %filter) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [1, 0], strides = [1, 1]} : tensor<1x128x1x1xf16>, tensor<16x128x1x1xf16> -> tensor<1x16x2x1xf16>
    %2 = IE.Concat(%0, %1) {per_axis = #IE.Concat<axis = 2>} : tensor<1x16x1x1xf16>, tensor<1x16x2x1xf16> -> tensor<1x16x3x1xf16>
    return %2 : tensor<1x16x3x1xf16>

    // CHECK-DAG:   [[FILTER:%.+]]  = const.Declare tensor<16x128x1x1xf16>
    // CHECK:       [[CONV_0:%.+]]  = IE.Convolution
    // CHECK:       [[CONV_1:%.+]]  = IE.Convolution
    // CHECK:       [[CONCAT:%.+]]  = IE.Concat([[CONV_0]], [[CONV_1]])
    // CHECK:       return  [[CONCAT]]
}


// -----

// CHECK-LABEL: @NotMergeHeightHasPadding
// CHECK-SAME:      [[INPUT0:%.+]]: tensor<1x128x1x1xf16>, [[INPUT1:%.+]]: tensor<1x128x1x1xf16>
func.func @NotMergeHeightHasPadding(%arg0: tensor<1x128x1x1xf16>, %arg1: tensor<1x128x1x1xf16>) -> (tensor<1x16x4x1xf16>) {
    %filter = const.Declare tensor<16x128x1x1xf16> = dense<0.000000e+00> : tensor<16x128x1x1xf16>
    %0 = IE.Convolution(%arg0, %filter) {dilations = [1, 1], pads_begin = [1, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x128x1x1xf16>, tensor<16x128x1x1xf16> -> tensor<1x16x2x1xf16>
    %1 = IE.Convolution(%arg1, %filter) {dilations = [1, 1], pads_begin = [1, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x128x1x1xf16>, tensor<16x128x1x1xf16> -> tensor<1x16x2x1xf16>
    %2 = IE.Concat(%0, %1) {per_axis = #IE.Concat<axis = 2>} : tensor<1x16x2x1xf16>, tensor<1x16x2x1xf16> -> tensor<1x16x4x1xf16>
    return %2 : tensor<1x16x4x1xf16>

    // CHECK-DAG:   [[FILTER:%.+]]  = const.Declare tensor<16x128x1x1xf16>
    // CHECK:       [[CONV_0:%.+]]  = IE.Convolution
    // CHECK:       [[CONV_1:%.+]]  = IE.Convolution
    // CHECK:       [[CONCAT:%.+]]  = IE.Concat([[CONV_0]], [[CONV_1]])
    // CHECK:       return  [[CONCAT]]
}

// -----

// CHECK-LABEL: @NotMergeDonotHaveSameUser
// CHECK-SAME:      [[INPUT0:%.+]]: tensor<1x128x1x1xf16>, [[INPUT1:%.+]]: tensor<1x128x1x1xf16>
func.func @NotMergeDonotHaveSameUser(%arg0: tensor<1x128x1x1xf16>, %arg1: tensor<1x128x1x1xf16>) -> (tensor<1x8x1x1xf16>, tensor<1x8x1x1xf16>) {
    %filter = const.Declare tensor<16x128x1x1xf16> = dense<0.000000e+00> : tensor<16x128x1x1xf16>
    %0 = IE.Convolution(%arg0, %filter) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x128x1x1xf16>, tensor<16x128x1x1xf16> -> tensor<1x16x1x1xf16>
    %1 = IE.Slice %0 [0, 0, 0, 0] [1, 8, 1, 1] : tensor<1x16x1x1xf16> to tensor<1x8x1x1xf16>
    %2 = IE.Convolution(%arg1, %filter) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x128x1x1xf16>, tensor<16x128x1x1xf16> -> tensor<1x16x1x1xf16>
    %3 = IE.Slice %2 [0, 0, 0, 0] [1, 8, 1, 1] : tensor<1x16x1x1xf16> to tensor<1x8x1x1xf16>
    return %1, %3 :  tensor<1x8x1x1xf16>, tensor<1x8x1x1xf16>

    // CHECK-DAG:   [[FILTER:%.+]]  = const.Declare tensor<16x128x1x1xf16>
    // CHECK:       [[CONV_0:%.+]]  = IE.Convolution
    // CHECK:       [[SLICE_0:%.+]]  = IE.Slice
    // CHECK:       [[CONV_1:%.+]]  = IE.Convolution
    // CHECK:       [[SLICE_1:%.+]]  = IE.Slice
    // CHECK:       return  [[SLICE_0]], [[SLICE_1]]
}

// -----

// CHECK-LABEL: @MergeWeightsSharedBeforeAffineReshapeNewCase
// CHECK-SAME:      [[INPUT0:%.+]]: tensor<1x128x1x1xf16>, [[INPUT1:%.+]]: tensor<1x128x1x1xf16>, [[INPUT2:%.+]]: tensor<1x16x128x1xf16>
func.func @MergeWeightsSharedBeforeAffineReshapeNewCase(%arg0: tensor<1x128x1x1xf16>, %arg1: tensor<1x128x1x1xf16>, %arg2: tensor<1x16x128x1xf16>) -> (tensor<1x16x128x1xf16>, tensor<1x16x2x1xf16>) {
    %0 = IE.AffineReshape(%arg2) {dim_mapping = [[0], [0], [1], [2, 3]], shape_value = [16, 128, 1, 1]} : tensor<1x16x128x1xf16> -> tensor<16x128x1x1xf16>
    %1 = IE.AffineReshape(%arg2) {dim_mapping = [[0], [0], [1], [2, 3]], shape_value = [16, 128, 1, 1]} : tensor<1x16x128x1xf16> -> tensor<16x128x1x1xf16>
    %2 = IE.ReLU(%arg2) : tensor<1x16x128x1xf16> -> tensor<1x16x128x1xf16>
    %3 = IE.Convolution(%arg0, %0) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x128x1x1xf16>, tensor<16x128x1x1xf16> -> tensor<1x16x1x1xf16>
    %4 = IE.Convolution(%arg1, %1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x128x1x1xf16>, tensor<16x128x1x1xf16> -> tensor<1x16x1x1xf16>
    %5 = IE.Concat(%3, %4) {per_axis = #IE.Concat<axis = 2>} : tensor<1x16x1x1xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x2x1xf16>
    return %2, %5 : tensor<1x16x128x1xf16>, tensor<1x16x2x1xf16>

    // CHECK:       [[AFFINERESHAPE:%.+]]  = IE.AffineReshape([[INPUT2]])
    // CHECK-SAME{LITERAL}:        {dim_mapping = [[0], [0], [1], [2, 3]], shape_value = [16, 128, 1, 1]} : tensor<1x16x128x1xf16> -> tensor<16x128x1x1xf16>
    // CHECK:       [[RELU:%.+]]  = IE.ReLU([[INPUT2]]) : tensor<1x16x128x1xf16> -> tensor<1x16x128x1xf16>
    // CHECK:       [[IN_CONCAT:%.+]] = IE.Concat([[INPUT1]], [[INPUT0]]) {per_axis = #IE.Concat<axis = 2 : i64>} : tensor<1x128x1x1xf16>, tensor<1x128x1x1xf16> -> tensor<1x128x2x1xf16>
    // CHECK:       [[CONV:%.+]]  = IE.Convolution([[IN_CONCAT]], [[AFFINERESHAPE]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x128x2x1xf16>, tensor<16x128x1x1xf16> -> tensor<1x16x2x1xf16>
    // CHECK:       [[SLICE_0:%.+]] = IE.Slice [[CONV]]  [0, 0, 0, 0] [1, 16, 1, 1] : tensor<1x16x2x1xf16> to tensor<1x16x1x1xf16>
    // CHECK:       [[SLICE_1:%.+]] = IE.Slice [[CONV]]  [0, 0, 1, 0] [1, 16, 1, 1] : tensor<1x16x2x1xf16> to tensor<1x16x1x1xf16>
    // CHECK:       [[OUT_CONCAT:%.+]] = IE.Concat([[SLICE_1]], [[SLICE_0]]) {per_axis = #IE.Concat<axis = 2 : i64>} : tensor<1x16x1x1xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x2x1xf16>
    // CHECK:       return  [[RELU]], [[OUT_CONCAT]]
}
