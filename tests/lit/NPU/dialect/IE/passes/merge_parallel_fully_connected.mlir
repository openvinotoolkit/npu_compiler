//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --merge-parallel-fully-connected %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010


// CHECK-LABEL: @MergeParallelFCWithReshapeTransposeOrderInput
// CHECK-SAME:      [[INPUT0:%.+]]: tensor<1x6xf32>
func.func @MergeParallelFCWithReshapeTransposeOrderInput(%arg0: tensor<1x6xf32>) -> (tensor<1x2xf32>, tensor<1x3xf32>) {
    %cst0_in = const.Declare tensor<2x3x2xf32> = dense<[[[1.0, 2.0], [1.0, 2.0], [2.0, 3.0]], [[3.0, 4.0], [1.0, 3.0], [2.0, 3.0]]]> : tensor<2x3x2xf32>
    %cst0_inlow = const.Declare tensor<1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1xf32>
    %cst0_inhigh = const.Declare tensor<1x1x1xf32> = dense<1.500000e+01> : tensor<1x1x1xf32>
    %cst0_outlow = const.Declare tensor<2x1x2xf32> = dense<[[[-1.0, -2.0]], [[-3.0, -4.0]]]> : tensor<2x1x2xf32>
    %cst0_outhigh = const.Declare tensor<2x1x2xf32> = dense<[[[1.0, 2.0]], [[3.0, 4.0]]]> : tensor<2x1x2xf32>

    %0 = IE.FakeQuantize(%cst0_in, %cst0_inlow, %cst0_inhigh, %cst0_outlow, %cst0_outhigh) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 16 : i64}
            : tensor<2x3x2xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<2x1x2xf32>, tensor<2x1x2xf32> -> tensor<2x3x2xf32>
    %1 = IE.AffineReshape(%0) {dim_mapping = [[0], [0], [1]], shape_value = [6, 2]} : tensor<2x3x2xf32> -> tensor<6x2xf32>
    %2 = IE.Transpose(%1) {order_value = affine_map<(d0, d1) -> (d1, d0)>} : tensor<6x2xf32> -> tensor<2x6xf32>
    %3 = IE.FullyConnected(%arg0, %2) : tensor<1x6xf32>, tensor<2x6xf32> -> tensor<1x2xf32>

    %cst1_in = const.Declare tensor<2x3x3xf32> = dense<[[[1.0, 2.0, 3.0], [1.0, 2.0, 1.0], [2.0, 3.0, 2.0]], [[3.0, 4.0, 2.0], [1.0, 3.0, 2.0], [2.0, 3.0, 4.0]]]> : tensor<2x3x3xf32>
    %cst1_inlow = const.Declare tensor<1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1xf32>
    %cst1_inhigh = const.Declare tensor<1x1x1xf32> = dense<1.500000e+01> : tensor<1x1x1xf32>
    %cst1_outlow = const.Declare tensor<2x1x3xf32> = dense<[[[-1.0, -2.0, -3.0]], [[-3.0, -4.0, -1.0]]]> : tensor<2x1x3xf32>
    %cst1_outhigh = const.Declare tensor<2x1x3xf32> = dense<[[[1.0, 2.0, 3.0]], [[3.0, 4.0, 1.0]]]> : tensor<2x1x3xf32>

    %4 = IE.FakeQuantize(%cst1_in, %cst1_inlow, %cst1_inhigh, %cst1_outlow, %cst1_outhigh) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 16 : i64}
            : tensor<2x3x3xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<2x1x3xf32>, tensor<2x1x3xf32> -> tensor<2x3x3xf32>
    %5 = IE.AffineReshape(%4) {dim_mapping = [[0], [0], [1]], shape_value = [6, 3]} : tensor<2x3x3xf32> -> tensor<6x3xf32>
    %6 = IE.Transpose(%5) {order_value = affine_map<(d0, d1) -> (d1, d0)>} : tensor<6x3xf32> -> tensor<3x6xf32>
    %7 = IE.FullyConnected(%arg0, %6) : tensor<1x6xf32>, tensor<3x6xf32> -> tensor<1x3xf32>

    return  %3, %7 : tensor<1x2xf32>, tensor<1x3xf32>

    // CHECK-DAG:   [[CST_INLOW:%.+]]  = const.Declare tensor<1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1xf32>
    // CHECK-DAG:   [[CST_INHIGH:%.+]]  = const.Declare tensor<1x1x1xf32> = dense<1.500000e+01> : tensor<1x1x1xf32>
    // CHECK-DAG:   [[CST_IN:%.+]]  = const.Declare tensor<2x3x5xf32>
    // CHECK-SAME:      #const.Concat
    // CHECK-SAME{LITERAL}:      dense<[[[1.000000e+00, 2.000000e+00, 3.000000e+00], [1.000000e+00, 2.000000e+00, 1.000000e+00], [2.000000e+00, 3.000000e+00, 2.000000e+00]], [[3.000000e+00, 4.000000e+00, 2.000000e+00], [1.000000e+00, 3.000000e+00, 2.000000e+00], [2.000000e+00, 3.000000e+00, 4.000000e+00]]]>
    // CHECK-SAME{LITERAL}:      dense<[[[1.000000e+00, 2.000000e+00], [1.000000e+00, 2.000000e+00], [2.000000e+00, 3.000000e+00]], [[3.000000e+00, 4.000000e+00], [1.000000e+00, 3.000000e+00], [2.000000e+00, 3.000000e+00]]]>
    // CHECK-DAG:   [[CST_OUTLOW:%.+]]  = const.Declare tensor<2x1x5xf32>
    // CHECK-SAME:      #const.Concat
    // CHECK-SAME{LITERAL}:      dense<[[[-1.000000e+00, -2.000000e+00, -3.000000e+00]], [[-3.000000e+00, -4.000000e+00, -1.000000e+00]]]>
    // CHECK-SAME{LITERAL}:      dense<[[[-1.000000e+00, -2.000000e+00]], [[-3.000000e+00, -4.000000e+00]]]>
    // CHECK-DAG:   [[CST_OUTHIGH:%.+]]  = const.Declare tensor<2x1x5xf32>
    // CHECK-SAME:      #const.Concat
    // CHECK-SAME{LITERAL}:      dense<[[[1.000000e+00, 2.000000e+00, 3.000000e+00]], [[3.000000e+00, 4.000000e+00, 1.000000e+00]]]>
    // CHECK-SAME{LITERAL}:      dense<[[[1.000000e+00, 2.000000e+00]], [[3.000000e+00, 4.000000e+00]]]>
    // CHECK-DAG:   [[FQ:%.+]]  = IE.FakeQuantize([[CST_IN:%.+]], [[CST_INLOW:%.+]], [[CST_INHIGH:%.+]], [[CST_OUTLOW:%.+]], [[CST_OUTHIGH:%.+]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 16 : i64} : tensor<2x3x5xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<2x1x5xf32>, tensor<2x1x5xf32> -> tensor<2x3x5xf32>
    // CHECK:       [[AFFINERESHAPE:%.+]]  = IE.AffineReshape([[FQ]])
    // CHECK-SAME{LITERAL}:         {dim_mapping = [[0], [0], [1]], shape_value = [6, 5]} : tensor<2x3x5xf32> -> tensor<6x5xf32>
    // CHECK:       [[TRANSPOSE:%.+]] = IE.Transpose([[AFFINERESHAPE]]) {order_value = #CN} : tensor<6x5xf32> -> tensor<5x6xf32>
    // CHECK:       [[FULLYCONNECTED:%.+]] = IE.FullyConnected([[INPUT0]], [[TRANSPOSE]]) : tensor<1x6xf32>, tensor<5x6xf32> -> tensor<1x5xf32>
    // CHECK:       [[SLICE0:%.+]] = IE.Slice [[FULLYCONNECTED]] [0, 0] [1, 3] : tensor<1x5xf32> to tensor<1x3xf32>
    // CHECK:       [[SLICE1:%.+]] = IE.Slice [[FULLYCONNECTED]] [0, 3] [1, 2] : tensor<1x5xf32> to tensor<1x2xf32>
    // CHECK:       return [[SLICE1]], [[SLICE0]]
}

// -----

#map = affine_map<(d0, d1, d2) -> (d2, d0, d1)>

// CHECK-LABEL: @MergeParallelFCWithTransposeReshapeOrderInput
// CHECK-SAME:      [[INPUT0:%.+]]: tensor<1x6xf32>
func.func @MergeParallelFCWithTransposeReshapeOrderInput(%arg0: tensor<1x6xf32>) -> (tensor<1x2xf32>, tensor<1x3xf32>) {
    %cst0_in = const.Declare tensor<2x3x2xf32> = dense<[[[1.0, 2.0], [1.0, 2.0], [2.0, 3.0]], [[3.0, 4.0], [1.0, 3.0], [2.0, 3.0]]]> : tensor<2x3x2xf32>
    %cst0_inlow = const.Declare tensor<1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1xf32>
    %cst0_inhigh = const.Declare tensor<1x1x1xf32> = dense<1.500000e+01> : tensor<1x1x1xf32>
    %cst0_outlow = const.Declare tensor<2x1x2xf32> = dense<[[[-1.0, -2.0]], [[-3.0, -4.0]]]> : tensor<2x1x2xf32>
    %cst0_outhigh = const.Declare tensor<2x1x2xf32> = dense<[[[1.0, 2.0]], [[3.0, 4.0]]]> : tensor<2x1x2xf32>

    %0 = IE.FakeQuantize(%cst0_in, %cst0_inlow, %cst0_inhigh, %cst0_outlow, %cst0_outhigh) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 16 : i64}
            : tensor<2x3x2xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<2x1x2xf32>, tensor<2x1x2xf32> -> tensor<2x3x2xf32>
    %1 = IE.Transpose(%0) {order_value = #map} : tensor<2x3x2xf32> -> tensor<2x2x3xf32>
    %2 = IE.AffineReshape(%1) {dim_mapping = [[0], [1], [1]], shape_value = [2, 6]} : tensor<2x2x3xf32> -> tensor<2x6xf32>
    %3 = IE.FullyConnected(%arg0, %2) : tensor<1x6xf32>, tensor<2x6xf32> -> tensor<1x2xf32>

    %cst1_in = const.Declare tensor<2x3x3xf32> = dense<[[[1.0, 2.0, 3.0], [1.0, 2.0, 1.0], [2.0, 3.0, 2.0]], [[3.0, 4.0, 2.0], [1.0, 3.0, 2.0], [2.0, 3.0, 4.0]]]> : tensor<2x3x3xf32>
    %cst1_inlow = const.Declare tensor<1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1xf32>
    %cst1_inhigh = const.Declare tensor<1x1x1xf32> = dense<1.500000e+01> : tensor<1x1x1xf32>
    %cst1_outlow = const.Declare tensor<2x1x3xf32> = dense<[[[-1.0, -2.0, -3.0]], [[-3.0, -4.0, -1.0]]]> : tensor<2x1x3xf32>
    %cst1_outhigh = const.Declare tensor<2x1x3xf32> = dense<[[[1.0, 2.0, 3.0]], [[3.0, 4.0, 1.0]]]> : tensor<2x1x3xf32>

    %4 = IE.FakeQuantize(%cst1_in, %cst1_inlow, %cst1_inhigh, %cst1_outlow, %cst1_outhigh) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 16 : i64}
            : tensor<2x3x3xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<2x1x3xf32>, tensor<2x1x3xf32> -> tensor<2x3x3xf32>
    %5 = IE.Transpose(%4) {order_value = #map} : tensor<2x3x3xf32> -> tensor<3x2x3xf32>
    %6 = IE.AffineReshape(%5) {dim_mapping = [[0], [1], [1]], shape_value = [3, 6]} : tensor<3x2x3xf32> -> tensor<3x6xf32>
    %7 = IE.FullyConnected(%arg0, %6) : tensor<1x6xf32>, tensor<3x6xf32> -> tensor<1x3xf32>

    return  %3, %7 : tensor<1x2xf32>, tensor<1x3xf32>

    // CHECK-DAG:   [[CST_INLOW:%.+]]  = const.Declare tensor<1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1xf32>
    // CHECK-DAG:   [[CST_INHIGH:%.+]]  = const.Declare tensor<1x1x1xf32> = dense<1.500000e+01> : tensor<1x1x1xf32>
    // CHECK-DAG:   [[CST_IN:%.+]]  = const.Declare tensor<2x3x5xf32>
    // CHECK-SAME:      #const.Concat
    // CHECK-SAME{LITERAL}:      dense<[[[1.000000e+00, 2.000000e+00, 3.000000e+00], [1.000000e+00, 2.000000e+00, 1.000000e+00], [2.000000e+00, 3.000000e+00, 2.000000e+00]], [[3.000000e+00, 4.000000e+00, 2.000000e+00], [1.000000e+00, 3.000000e+00, 2.000000e+00], [2.000000e+00, 3.000000e+00, 4.000000e+00]]]>
    // CHECK-SAME{LITERAL}:      dense<[[[1.000000e+00, 2.000000e+00], [1.000000e+00, 2.000000e+00], [2.000000e+00, 3.000000e+00]], [[3.000000e+00, 4.000000e+00], [1.000000e+00, 3.000000e+00], [2.000000e+00, 3.000000e+00]]]>
    // CHECK-DAG:   [[CST_OUTLOW:%.+]]  = const.Declare tensor<2x1x5xf32>
    // CHECK-SAME:      #const.Concat
    // CHECK-SAME{LITERAL}:      dense<[[[-1.000000e+00, -2.000000e+00, -3.000000e+00]], [[-3.000000e+00, -4.000000e+00, -1.000000e+00]]]>
    // CHECK-SAME{LITERAL}:      dense<[[[-1.000000e+00, -2.000000e+00]], [[-3.000000e+00, -4.000000e+00]]]>
    // CHECK-DAG:   [[CST_OUTHIGH:%.+]]  = const.Declare tensor<2x1x5xf32>
    // CHECK-SAME:      #const.Concat
    // CHECK-SAME{LITERAL}:      dense<[[[1.000000e+00, 2.000000e+00, 3.000000e+00]], [[3.000000e+00, 4.000000e+00, 1.000000e+00]]]>
    // CHECK-SAME{LITERAL}:      dense<[[[1.000000e+00, 2.000000e+00]], [[3.000000e+00, 4.000000e+00]]]>
    // CHECK-DAG:   [[FQ:%.+]]  = IE.FakeQuantize([[CST_IN:%.+]], [[CST_INLOW:%.+]], [[CST_INHIGH:%.+]], [[CST_OUTLOW:%.+]], [[CST_OUTHIGH:%.+]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 16 : i64} : tensor<2x3x5xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<2x1x5xf32>, tensor<2x1x5xf32> -> tensor<2x3x5xf32>
    // CHECK:       [[TRANSPOSE:%.+]] = IE.Transpose([[FQ]]) {order_value = #map} : tensor<2x3x5xf32> -> tensor<5x2x3xf32>
    // CHECK:       [[AFFINERESHAPE:%.+]]  = IE.AffineReshape([[TRANSPOSE]])
    // CHECK-SAME{LITERAL}:         {dim_mapping = [[0], [1], [1]], shape_value = [5, 6]} : tensor<5x2x3xf32> -> tensor<5x6xf32>
    // CHECK:       [[FULLYCONNECTED:%.+]] = IE.FullyConnected([[INPUT0]], [[AFFINERESHAPE]]) : tensor<1x6xf32>, tensor<5x6xf32> -> tensor<1x5xf32>
    // CHECK:       [[SLICE0:%.+]] = IE.Slice [[FULLYCONNECTED]] [0, 0] [1, 3] : tensor<1x5xf32> to tensor<1x3xf32>
    // CHECK:       [[SLICE1:%.+]] = IE.Slice [[FULLYCONNECTED]] [0, 3] [1, 2] : tensor<1x5xf32> to tensor<1x2xf32>
    // CHECK:       return [[SLICE1]], [[SLICE0]]
}

// -----

// CHECK-LABEL: @NotMergeDifferentZeroPoint
// CHECK-SAME:      [[INPUT0:%.+]]: tensor<1x6xf32>
func.func @NotMergeDifferentZeroPoint(%arg0: tensor<1x6xf32>) -> (tensor<1x2xf32>, tensor<1x3xf32>) {
    %cst0_in = const.Declare tensor<2x3x2xf32> = dense<[[[1.0, 2.0], [1.0, 2.0], [2.0, 3.0]], [[3.0, 4.0], [1.0, 3.0], [2.0, 3.0]]]> : tensor<2x3x2xf32>
    %cst0_inlow = const.Declare tensor<1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1xf32>
    %cst0_inhigh = const.Declare tensor<1x1x1xf32> = dense<1.500000e+01> : tensor<1x1x1xf32>
    %cst0_outlow = const.Declare tensor<2x1x2xf32> = dense<[[[-1.0, -2.0]], [[-3.0, -4.0]]]> : tensor<2x1x2xf32>
    %cst0_outhigh = const.Declare tensor<2x1x2xf32> = dense<[[[1.0, 2.0]], [[3.0, 5.0]]]> : tensor<2x1x2xf32>

    %0 = IE.FakeQuantize(%cst0_in, %cst0_inlow, %cst0_inhigh, %cst0_outlow, %cst0_outhigh) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 16 : i64}
            : tensor<2x3x2xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<2x1x2xf32>, tensor<2x1x2xf32> -> tensor<2x3x2xf32>
    %1 = IE.AffineReshape(%0) {dim_mapping = [[0], [0], [1]], shape_value = [6, 2]} : tensor<2x3x2xf32> -> tensor<6x2xf32>
    %2 = IE.Transpose(%1) {order_value = affine_map<(d0, d1) -> (d1, d0)>} : tensor<6x2xf32> -> tensor<2x6xf32>
    %3 = IE.FullyConnected(%arg0, %2) : tensor<1x6xf32>, tensor<2x6xf32> -> tensor<1x2xf32>

    %cst1_in = const.Declare tensor<2x3x3xf32> = dense<[[[1.0, 2.0, 3.0], [1.0, 2.0, 1.0], [2.0, 3.0, 2.0]], [[3.0, 4.0, 2.0], [1.0, 3.0, 2.0], [2.0, 3.0, 4.0]]]> : tensor<2x3x3xf32>
    %cst1_inlow = const.Declare tensor<1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1xf32>
    %cst1_inhigh = const.Declare tensor<1x1x1xf32> = dense<1.500000e+01> : tensor<1x1x1xf32>
    %cst1_outlow = const.Declare tensor<2x1x3xf32> = dense<[[[-1.0, -2.0, -3.0]], [[-3.0, -4.0, -1.0]]]> : tensor<2x1x3xf32>
    %cst1_outhigh = const.Declare tensor<2x1x3xf32> = dense<[[[1.0, 2.0, 3.0]], [[3.0, 4.0, 1.0]]]> : tensor<2x1x3xf32>

    %4 = IE.FakeQuantize(%cst1_in, %cst1_inlow, %cst1_inhigh, %cst1_outlow, %cst1_outhigh) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 16 : i64}
            : tensor<2x3x3xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<2x1x3xf32>, tensor<2x1x3xf32> -> tensor<2x3x3xf32>
    %5 = IE.AffineReshape(%4) {dim_mapping = [[0], [0], [1]], shape_value = [6, 3]} : tensor<2x3x3xf32> -> tensor<6x3xf32>
    %6 = IE.Transpose(%5) {order_value = affine_map<(d0, d1) -> (d1, d0)>} : tensor<6x3xf32> -> tensor<3x6xf32>
    %7 = IE.FullyConnected(%arg0, %6) : tensor<1x6xf32>, tensor<3x6xf32> -> tensor<1x3xf32>

    return  %3, %7 : tensor<1x2xf32>, tensor<1x3xf32>


    // CHECK-DAG:   [[CST0:%.+]]  = const.Declare tensor<2x1x3xf32>
    // CHECK-DAG:   [[CST1:%.+]]  = const.Declare tensor<2x1x3xf32>
    // CHECK-DAG:   [[CST2:%.+]]  = const.Declare tensor<2x3x3xf32>
    // CHECK-DAG:   [[CST3:%.+]]  = const.Declare tensor<2x3x2xf32>
    // CHECK-DAG:   [[CST4:%.+]]  = const.Declare tensor<1x1x1xf32>
    // CHECK-DAG:   [[CST5:%.+]]  = const.Declare tensor<1x1x1xf32>
    // CHECK-DAG:   [[CST6:%.+]]  = const.Declare tensor<2x1x2xf32>
    // CHECK-DAG:   [[CST7:%.+]]  = const.Declare tensor<2x1x2xf32>
    // CHECK:       [[FQ0:%.+]] = IE.FakeQuantize([[CST3]], [[CST4]], [[CST5]], [[CST6]], [[CST7]])
    // CHECK:       [[AFFINERESHAPE0:%.+]] = IE.AffineReshape([[FQ0]])
    // CHECK:       [[TRANSPOSE0:%.+]] = IE.Transpose([[AFFINERESHAPE0]])
    // CHECK:       [[FULLYCONNECTED0:%.+]] = IE.FullyConnected([[INPUT0]], [[TRANSPOSE0]])
    // CHECK:       [[FQ1:%.+]] = IE.FakeQuantize([[CST2]], [[CST4]], [[CST5]], [[CST1]], [[CST0]])
    // CHECK:       [[AFFINERESHAPE1:%.+]] = IE.AffineReshape([[FQ1]])
    // CHECK:       [[TRANSPOSE1:%.+]] = IE.Transpose([[AFFINERESHAPE1]])
    // CHECK:       [[FULLYCONNECTED1:%.+]] = IE.FullyConnected([[INPUT0]], [[TRANSPOSE1]])
    // CHECK:       return [[FULLYCONNECTED0]], [[FULLYCONNECTED1]]
}

// -----

// CHECK-LABEL: @NotMergeWithDifferentInput
// CHECK-SAME:      [[INPUT0:%.+]]: tensor<1x6xf32>, [[INPUT1:%.+]]: tensor<1x6xf32>
func.func @NotMergeWithDifferentInput(%arg0: tensor<1x6xf32>, %arg1: tensor<1x6xf32>) -> (tensor<1x2xf32>, tensor<1x3xf32>) {
    %cst0_in = const.Declare tensor<2x3x2xf32> = dense<[[[1.0, 2.0], [1.0, 2.0], [2.0, 3.0]], [[3.0, 4.0], [1.0, 3.0], [2.0, 3.0]]]> : tensor<2x3x2xf32>
    %cst0_inlow = const.Declare tensor<1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1xf32>
    %cst0_inhigh = const.Declare tensor<1x1x1xf32> = dense<1.500000e+01> : tensor<1x1x1xf32>
    %cst0_outlow = const.Declare tensor<2x1x2xf32> = dense<[[[-1.0, -2.0]], [[-3.0, -4.0]]]> : tensor<2x1x2xf32>
    %cst0_outhigh = const.Declare tensor<2x1x2xf32> = dense<[[[1.0, 2.0]], [[3.0, 4.0]]]> : tensor<2x1x2xf32>

    %0 = IE.FakeQuantize(%cst0_in, %cst0_inlow, %cst0_inhigh, %cst0_outlow, %cst0_outhigh) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 16 : i64}
            : tensor<2x3x2xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<2x1x2xf32>, tensor<2x1x2xf32> -> tensor<2x3x2xf32>
    %1 = IE.AffineReshape(%0) {dim_mapping = [[0], [0], [1]], shape_value = [6, 2]} : tensor<2x3x2xf32> -> tensor<6x2xf32>
    %2 = IE.Transpose(%1) {order_value = affine_map<(d0, d1) -> (d1, d0)>} : tensor<6x2xf32> -> tensor<2x6xf32>
    %3 = IE.FullyConnected(%arg0, %2) : tensor<1x6xf32>, tensor<2x6xf32> -> tensor<1x2xf32>

    %cst1_in = const.Declare tensor<2x3x3xf32> = dense<[[[1.0, 2.0, 3.0], [1.0, 2.0, 1.0], [2.0, 3.0, 2.0]], [[3.0, 4.0, 2.0], [1.0, 3.0, 2.0], [2.0, 3.0, 4.0]]]> : tensor<2x3x3xf32>
    %cst1_inlow = const.Declare tensor<1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1xf32>
    %cst1_inhigh = const.Declare tensor<1x1x1xf32> = dense<1.500000e+01> : tensor<1x1x1xf32>
    %cst1_outlow = const.Declare tensor<2x1x3xf32> = dense<[[[-1.0, -2.0, -3.0]], [[-3.0, -4.0, -1.0]]]> : tensor<2x1x3xf32>
    %cst1_outhigh = const.Declare tensor<2x1x3xf32> = dense<[[[1.0, 2.0, 3.0]], [[3.0, 4.0, 1.0]]]> : tensor<2x1x3xf32>

    %4 = IE.FakeQuantize(%cst1_in, %cst1_inlow, %cst1_inhigh, %cst1_outlow, %cst1_outhigh) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 16 : i64}
            : tensor<2x3x3xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<2x1x3xf32>, tensor<2x1x3xf32> -> tensor<2x3x3xf32>
    %5 = IE.AffineReshape(%4) {dim_mapping = [[0], [0], [1]], shape_value = [6, 3]} : tensor<2x3x3xf32> -> tensor<6x3xf32>
    %6 = IE.Transpose(%5) {order_value = affine_map<(d0, d1) -> (d1, d0)>} : tensor<6x3xf32> -> tensor<3x6xf32>
    %7 = IE.FullyConnected(%arg1, %6) : tensor<1x6xf32>, tensor<3x6xf32> -> tensor<1x3xf32>

    return  %3, %7 : tensor<1x2xf32>, tensor<1x3xf32>


    // CHECK-DAG:   [[CST0:%.+]]  = const.Declare tensor<2x1x3xf32>
    // CHECK-DAG:   [[CST1:%.+]]  = const.Declare tensor<2x1x3xf32>
    // CHECK-DAG:   [[CST2:%.+]]  = const.Declare tensor<2x3x3xf32>
    // CHECK-DAG:   [[CST3:%.+]]  = const.Declare tensor<2x3x2xf32>
    // CHECK-DAG:   [[CST4:%.+]]  = const.Declare tensor<1x1x1xf32>
    // CHECK-DAG:   [[CST5:%.+]]  = const.Declare tensor<1x1x1xf32>
    // CHECK-DAG:   [[CST6:%.+]]  = const.Declare tensor<2x1x2xf32>
    // CHECK-DAG:   [[CST7:%.+]]  = const.Declare tensor<2x1x2xf32>
    // CHECK:       [[FQ0:%.+]] = IE.FakeQuantize([[CST3]], [[CST4]], [[CST5]], [[CST6]], [[CST7]])
    // CHECK:       [[AFFINERESHAPE0:%.+]] = IE.AffineReshape([[FQ0]])
    // CHECK:       [[TRANSPOSE0:%.+]] = IE.Transpose([[AFFINERESHAPE0]])
    // CHECK:       [[FULLYCONNECTED0:%.+]] = IE.FullyConnected([[INPUT0]], [[TRANSPOSE0]])
    // CHECK:       [[FQ1:%.+]] = IE.FakeQuantize([[CST2]], [[CST4]], [[CST5]], [[CST1]], [[CST0]])
    // CHECK:       [[AFFINERESHAPE1:%.+]] = IE.AffineReshape([[FQ1]])
    // CHECK:       [[TRANSPOSE1:%.+]] = IE.Transpose([[AFFINERESHAPE1]])
    // CHECK:       [[FULLYCONNECTED1:%.+]] = IE.FullyConnected([[INPUT1]], [[TRANSPOSE1]])
    // CHECK:       return [[FULLYCONNECTED0]], [[FULLYCONNECTED1]]
}

// -----

// CHECK-LABEL: @NotMergeWithMoreUsers
// CHECK-SAME:      [[INPUT0:%.+]]: tensor<1x6xf32>
func.func @NotMergeWithMoreUsers(%arg0: tensor<1x6xf32>) -> (tensor<1x2xf32>, tensor<1x3xf32>, tensor<3x6xf32>) {
    %cst0_in = const.Declare tensor<2x3x2xf32> = dense<[[[1.0, 2.0], [1.0, 2.0], [2.0, 3.0]], [[3.0, 4.0], [1.0, 3.0], [2.0, 3.0]]]> : tensor<2x3x2xf32>
    %cst0_inlow = const.Declare tensor<1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1xf32>
    %cst0_inhigh = const.Declare tensor<1x1x1xf32> = dense<1.500000e+01> : tensor<1x1x1xf32>
    %cst0_outlow = const.Declare tensor<2x1x2xf32> = dense<[[[-1.0, -2.0]], [[-3.0, -4.0]]]> : tensor<2x1x2xf32>
    %cst0_outhigh = const.Declare tensor<2x1x2xf32> = dense<[[[1.0, 2.0]], [[3.0, 4.0]]]> : tensor<2x1x2xf32>

    %0 = IE.FakeQuantize(%cst0_in, %cst0_inlow, %cst0_inhigh, %cst0_outlow, %cst0_outhigh) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 16 : i64}
            : tensor<2x3x2xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<2x1x2xf32>, tensor<2x1x2xf32> -> tensor<2x3x2xf32>
    %1 = IE.AffineReshape(%0) {dim_mapping = [[0], [0], [1]], shape_value = [6, 2]} : tensor<2x3x2xf32> -> tensor<6x2xf32>
    %2 = IE.Transpose(%1) {order_value = affine_map<(d0, d1) -> (d1, d0)>} : tensor<6x2xf32> -> tensor<2x6xf32>
    %3 = IE.FullyConnected(%arg0, %2) {transpose_b} : tensor<1x6xf32>, tensor<2x6xf32> -> tensor<1x2xf32>

    %cst1_in = const.Declare tensor<2x3x3xf32> = dense<[[[1.0, 2.0, 3.0], [1.0, 2.0, 1.0], [2.0, 3.0, 2.0]], [[3.0, 4.0, 2.0], [1.0, 3.0, 2.0], [2.0, 3.0, 4.0]]]> : tensor<2x3x3xf32>
    %cst1_inlow = const.Declare tensor<1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1xf32>
    %cst1_inhigh = const.Declare tensor<1x1x1xf32> = dense<1.500000e+01> : tensor<1x1x1xf32>
    %cst1_outlow = const.Declare tensor<2x1x3xf32> = dense<[[[-1.0, -2.0, -3.0]], [[-3.0, -4.0, -1.0]]]> : tensor<2x1x3xf32>
    %cst1_outhigh = const.Declare tensor<2x1x3xf32> = dense<[[[1.0, 2.0, 3.0]], [[3.0, 4.0, 1.0]]]> : tensor<2x1x3xf32>

    %4 = IE.FakeQuantize(%cst1_in, %cst1_inlow, %cst1_inhigh, %cst1_outlow, %cst1_outhigh) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 16 : i64}
            : tensor<2x3x3xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<2x1x3xf32>, tensor<2x1x3xf32> -> tensor<2x3x3xf32>
    %5 = IE.AffineReshape(%4) {dim_mapping = [[0], [0], [1]], shape_value = [6, 3]} : tensor<2x3x3xf32> -> tensor<6x3xf32>
    %6 = IE.Transpose(%5) {order_value = affine_map<(d0, d1) -> (d1, d0)>} : tensor<6x3xf32> -> tensor<3x6xf32>
    %7 = IE.FullyConnected(%arg0, %6) {transpose_b} : tensor<1x6xf32>, tensor<3x6xf32> -> tensor<1x3xf32>

    return  %3, %7, %6 : tensor<1x2xf32>, tensor<1x3xf32>, tensor<3x6xf32>


    // CHECK-DAG:   [[CST0:%.+]]  = const.Declare tensor<2x1x3xf32>
    // CHECK-DAG:   [[CST1:%.+]]  = const.Declare tensor<2x1x3xf32>
    // CHECK-DAG:   [[CST2:%.+]]  = const.Declare tensor<2x3x3xf32>
    // CHECK-DAG:   [[CST3:%.+]]  = const.Declare tensor<2x3x2xf32>
    // CHECK-DAG:   [[CST4:%.+]]  = const.Declare tensor<1x1x1xf32>
    // CHECK-DAG:   [[CST5:%.+]]  = const.Declare tensor<1x1x1xf32>
    // CHECK-DAG:   [[CST6:%.+]]  = const.Declare tensor<2x1x2xf32>
    // CHECK-DAG:   [[CST7:%.+]]  = const.Declare tensor<2x1x2xf32>
    // CHECK:       [[FQ0:%.+]] = IE.FakeQuantize([[CST3]], [[CST4]], [[CST5]], [[CST6]], [[CST7]])
    // CHECK:       [[AFFINERESHAPE0:%.+]] = IE.AffineReshape([[FQ0]])
    // CHECK:       [[TRANSPOSE0:%.+]] = IE.Transpose([[AFFINERESHAPE0]])
    // CHECK:       [[FULLYCONNECTED0:%.+]] = IE.FullyConnected([[INPUT0]], [[TRANSPOSE0]])
    // CHECK:       [[FQ1:%.+]] = IE.FakeQuantize([[CST2]], [[CST4]], [[CST5]], [[CST1]], [[CST0]])
    // CHECK:       [[AFFINERESHAPE1:%.+]] = IE.AffineReshape([[FQ1]])
    // CHECK:       [[TRANSPOSE1:%.+]] = IE.Transpose([[AFFINERESHAPE1]])
    // CHECK:       [[FULLYCONNECTED1:%.+]] = IE.FullyConnected([[INPUT0]], [[TRANSPOSE1]])
    // CHECK:       return [[FULLYCONNECTED0]], [[FULLYCONNECTED1]], [[TRANSPOSE1]]
}

// -----

// CHECK-LABEL: @NotMergeForNonGPTQ
// CHECK-SAME:      [[INPUT0:%.+]]: tensor<1x6xf32>
func.func @NotMergeForNonGPTQ(%arg0: tensor<1x6xf32>) -> (tensor<1x2xf32>, tensor<1x3xf32>) {
    %cst0_in = const.Declare tensor<2x3x2xf32> = dense<[[[1.0, 2.0], [1.0, 2.0], [2.0, 3.0]], [[3.0, 4.0], [1.0, 3.0], [2.0, 3.0]]]> : tensor<2x3x2xf32>
    %cst0_inlow = const.Declare tensor<1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1xf32>
    %cst0_inhigh = const.Declare tensor<1x1x1xf32> = dense<1.500000e+01> : tensor<1x1x1xf32>
    %cst0_outlow = const.Declare tensor<1x1x2xf32> = dense<[[[-1.0, -2.0]]]> : tensor<1x1x2xf32>
    %cst0_outhigh = const.Declare tensor<1x1x2xf32> = dense<[[[1.0, 2.0]]]> : tensor<1x1x2xf32>

    %0 = IE.FakeQuantize(%cst0_in, %cst0_inlow, %cst0_inhigh, %cst0_outlow, %cst0_outhigh) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 16 : i64}
            : tensor<2x3x2xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<1x1x2xf32>, tensor<1x1x2xf32> -> tensor<2x3x2xf32>
    %1 = IE.AffineReshape(%0) {dim_mapping = [[0], [0], [1]], shape_value = [6, 2]} : tensor<2x3x2xf32> -> tensor<6x2xf32>
    %2 = IE.Transpose(%1) {order_value = affine_map<(d0, d1) -> (d1, d0)>} : tensor<6x2xf32> -> tensor<2x6xf32>
    %3 = IE.FullyConnected(%arg0, %2) : tensor<1x6xf32>, tensor<2x6xf32> -> tensor<1x2xf32>

    %cst1_in = const.Declare tensor<2x3x3xf32> = dense<[[[1.0, 2.0, 3.0], [1.0, 2.0, 1.0], [2.0, 3.0, 2.0]], [[3.0, 4.0, 2.0], [1.0, 3.0, 2.0], [2.0, 3.0, 4.0]]]> : tensor<2x3x3xf32>
    %cst1_inlow = const.Declare tensor<1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1xf32>
    %cst1_inhigh = const.Declare tensor<1x1x1xf32> = dense<1.500000e+01> : tensor<1x1x1xf32>
    %cst1_outlow = const.Declare tensor<2x1x3xf32> = dense<[[[-1.0, -2.0, -3.0]], [[-3.0, -4.0, -1.0]]]> : tensor<2x1x3xf32>
    %cst1_outhigh = const.Declare tensor<2x1x3xf32> = dense<[[[1.0, 2.0, 3.0]], [[3.0, 4.0, 1.0]]]> : tensor<2x1x3xf32>

    %4 = IE.FakeQuantize(%cst1_in, %cst1_inlow, %cst1_inhigh, %cst1_outlow, %cst1_outhigh) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 16 : i64}
            : tensor<2x3x3xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<2x1x3xf32>, tensor<2x1x3xf32> -> tensor<2x3x3xf32>
    %5 = IE.AffineReshape(%4) {dim_mapping = [[0], [0], [1]], shape_value = [6, 3]} : tensor<2x3x3xf32> -> tensor<6x3xf32>
    %6 = IE.Transpose(%5) {order_value = affine_map<(d0, d1) -> (d1, d0)>} : tensor<6x3xf32> -> tensor<3x6xf32>
    %7 = IE.FullyConnected(%arg0, %6) : tensor<1x6xf32>, tensor<3x6xf32> -> tensor<1x3xf32>

    return  %3, %7 : tensor<1x2xf32>, tensor<1x3xf32>


    // CHECK-DAG:   [[CST0:%.+]]  = const.Declare tensor<2x1x3xf32>
    // CHECK-DAG:   [[CST1:%.+]]  = const.Declare tensor<2x1x3xf32>
    // CHECK-DAG:   [[CST2:%.+]]  = const.Declare tensor<2x3x3xf32>
    // CHECK-DAG:   [[CST3:%.+]]  = const.Declare tensor<2x3x2xf32>
    // CHECK-DAG:   [[CST4:%.+]]  = const.Declare tensor<1x1x1xf32>
    // CHECK-DAG:   [[CST5:%.+]]  = const.Declare tensor<1x1x1xf32>
    // CHECK-DAG:   [[CST6:%.+]]  = const.Declare tensor<1x1x2xf32>
    // CHECK-DAG:   [[CST7:%.+]]  = const.Declare tensor<1x1x2xf32>
    // CHECK:       [[FQ0:%.+]] = IE.FakeQuantize([[CST3]], [[CST4]], [[CST5]], [[CST6]], [[CST7]])
    // CHECK:       [[AFFINERESHAPE0:%.+]] = IE.AffineReshape([[FQ0]])
    // CHECK:       [[TRANSPOSE0:%.+]] = IE.Transpose([[AFFINERESHAPE0]])
    // CHECK:       [[FULLYCONNECTED0:%.+]] = IE.FullyConnected([[INPUT0]], [[TRANSPOSE0]])
    // CHECK:       [[FQ1:%.+]] = IE.FakeQuantize([[CST2]], [[CST4]], [[CST5]], [[CST1]], [[CST0]])
    // CHECK:       [[AFFINERESHAPE1:%.+]] = IE.AffineReshape([[FQ1]])
    // CHECK:       [[TRANSPOSE1:%.+]] = IE.Transpose([[AFFINERESHAPE1]])
    // CHECK:       [[FULLYCONNECTED1:%.+]] = IE.FullyConnected([[INPUT0]], [[TRANSPOSE1]])
    // CHECK:       return [[FULLYCONNECTED0]], [[FULLYCONNECTED1]]
}

// -----

!qElemType = !quant.uniform<u2:f16, 1.000000e+00>

// CHECK-LABEL: @MergeParallelFCWithDynamicDequantizeReshapeOrderInput
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x1x8192xf16>
func.func @MergeParallelFCWithDynamicDequantizeReshapeOrderInput(%input: tensor<1x1x8192xf16>) -> (tensor<1x1x3072xf32>, tensor<1x1x3072xf32>) {
    %cst = const.Declare tensor<3072x128x1xf16> = dense<1.0> : tensor<3072x128x1xf32>, [#const.CastElemType<f16>]
    %cst_0 = const.Declare tensor<3072x128xf16> = dense<1.0> : tensor<3072x128x1xf32>, [#const.Rescale<Content<dense<2> : tensor<3072x128x1xui2>, [#const.ConvertElemType<ui8>, #const.CastElemType<f32>]>>, #const.Reshape<[3072, 128]>, #const.CastElemType<f16>]
    %cst_1 = const.Declare tensor<3072x128xf16> = dense<1.0> : tensor<3072x128x1xf32>, [#const.Rescale<Content<dense<3> : tensor<3072x128x1xui2>, [#const.ConvertElemType<ui8>, #const.CastElemType<f32>]>>, #const.Reshape<[3072, 128]>, #const.CastElemType<f16>]
    %cst_2 = const.Declare tensor<3072x128x64x!qElemType> = dense<1> : tensor<3072x128x64xui2>, [#const.CastElemType<!qElemType>]
    %cst_3 = const.Declare tensor<3072x128x64x!qElemType> = dense<1> : tensor<3072x128x64xui2>, [#const.CastElemType<!qElemType>]
    %cst_4 = const.Declare tensor<3072x128x1xf16> = dense<1.0> : tensor<3072x128x1xf32>, [#const.CastElemType<f16>]
    %0 = IE.DynamicDequantize(%cst_3, %cst_4) {dstElemType = f16} : tensor<3072x128x64x!qElemType>, tensor<3072x128x1xf16> -> tensor<3072x128x64xf16>
    %1 = IE.DynamicDequantize(%cst_2, %cst) {dstElemType = f16} : tensor<3072x128x64x!qElemType>, tensor<3072x128x1xf16> -> tensor<3072x128x64xf16>
    %2 = IE.AffineReshape(%1) {dim_mapping = [[0], [1], [1]], shape_value = [3072, 8192]} : tensor<3072x128x64xf16> -> tensor<3072x8192xf16>
    %3 = IE.AffineReshape(%input) {dim_mapping = [[0], [0], [1]], shape_value = [1, 8192]} : tensor<1x1x8192xf16> -> tensor<1x8192xf16>
    %4 = IE.FullyConnected(%3, %2) : tensor<1x8192xf16>, tensor<3072x8192xf16> -> tensor<1x3072xf16>
    %5 = IE.AffineReshape(%input) {dim_mapping = [[0], [0], [1, 2]], shape_value = [1, 128, 64]} : tensor<1x1x8192xf16> -> tensor<1x128x64xf16>
    %6 = IE.ReduceSum(%5) {axes_value = [2]} : tensor<1x128x64xf16> -> tensor<1x128xf16>
    %7 = IE.FullyConnected(%6, %cst_1) : tensor<1x128xf16>, tensor<3072x128xf16> -> tensor<1x3072xf16>
    %8 = IE.Subtract(%4, %7) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3072xf16>, tensor<1x3072xf16> -> tensor<1x3072xf16>
    %9 = IE.AffineReshape(%8) {dim_mapping = [[0, 1], [2]], shape_value = [1, 1, 3072]} : tensor<1x3072xf16> -> tensor<1x1x3072xf16>
    %10 = IE.AffineReshape(%0) {dim_mapping = [[0], [1], [1]], shape_value = [3072, 8192]} : tensor<3072x128x64xf16> -> tensor<3072x8192xf16>
    %11 = IE.FullyConnected(%3, %10) : tensor<1x8192xf16>, tensor<3072x8192xf16> -> tensor<1x3072xf16>
    %12 = IE.FullyConnected(%6, %cst_0) : tensor<1x128xf16>, tensor<3072x128xf16> -> tensor<1x3072xf16>
    %13 = IE.Subtract(%11, %12) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3072xf16>, tensor<1x3072xf16> -> tensor<1x3072xf16>
    %14 = IE.AffineReshape(%13) {dim_mapping = [[0, 1], [2]], shape_value = [1, 1, 3072]} : tensor<1x3072xf16> -> tensor<1x1x3072xf16>
    %15 = IE.Convert(%14) {dstElemType = f32} : tensor<1x1x3072xf16> -> tensor<1x1x3072xf32>
    %16 = IE.Convert(%9) {dstElemType = f32} : tensor<1x1x3072xf16> -> tensor<1x1x3072xf32>
    return %15, %16 : tensor<1x1x3072xf32>, tensor<1x1x3072xf32>

    // CHECK:    [[CST_SCALE:%.+]] = const.Declare tensor<6144x128x1xf16>
    // CHECK:    [[CST_WEIGHTS:%.+]] = const.Declare tensor<6144x128x64x!qElemType>
    // CHECK:    [[CST_ZP0:%.+]] = const.Declare tensor<3072x128xf16>
    // CHECK:    [[CST_ZP1:%.+]] = const.Declare tensor<3072x128xf16>

    // CHECK:    [[AFFINE_RESHAPE0:%.+]] = IE.AffineReshape([[INPUT]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [0], [1]], shape_value = [1, 8192]} : tensor<1x1x8192xf16> -> tensor<1x8192xf16>
    // CHECK:    [[DYN_DEQUANT:%.+]] = IE.DynamicDequantize([[CST_WEIGHTS]], [[CST_SCALE]])
    // CHECK:    [[AFFINE_RESHAPE1:%.+]] = IE.AffineReshape([[DYN_DEQUANT]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [1], [1]], shape_value = [6144, 8192]} : tensor<6144x128x64xf16> -> tensor<6144x8192xf16>

    // CHECK:    [[FULLY_CONNECTED0:%.+]] = IE.FullyConnected([[AFFINE_RESHAPE0]], [[AFFINE_RESHAPE1]])
    // CHECK:    [[SLICE0:%.+]] = IE.Slice [[FULLY_CONNECTED0]] [0, 0] [1, 3072]
    // CHECK:    [[SLICE1:%.+]] = IE.Slice [[FULLY_CONNECTED0]] [0, 3072] [1, 3072]
    // CHECK:    [[AFFINE_RESHAPE2:%.+]] = IE.AffineReshape([[INPUT]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [0], [1, 2]], shape_value = [1, 128, 64]} : tensor<1x1x8192xf16> -> tensor<1x128x64xf16>
    // CHECK:    [[REDUCE_SUM:%.+]] = IE.ReduceSum([[AFFINE_RESHAPE2]])

    // CHECK:    [[FULLY_CONNECTED1:%.+]] = IE.FullyConnected([[REDUCE_SUM]], [[CST_ZP1]])
    // CHECK:    [[SUBTRACT0:%.+]] = IE.Subtract([[SLICE1]], [[FULLY_CONNECTED1]])
    // CHECK:    [[AFFINE_RESHAPE3:%.+]] = IE.AffineReshape([[SUBTRACT0]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0, 1], [2]], shape_value = [1, 1, 3072]} : tensor<1x3072xf16> -> tensor<1x1x3072xf16>

    // CHECK:    [[FULLY_CONNECTED2:%.+]] = IE.FullyConnected([[REDUCE_SUM]], [[CST_ZP0]])
    // CHECK:    [[SUBTRACT1:%.+]] = IE.Subtract([[SLICE0]], [[FULLY_CONNECTED2]])
    // CHECK:    [[AFFINE_RESHAPE4:%.+]] = IE.AffineReshape([[SUBTRACT1]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0, 1], [2]], shape_value = [1, 1, 3072]} : tensor<1x3072xf16> -> tensor<1x1x3072xf16>
    // CHECK:    [[CONVERT0:%.+]] = IE.Convert([[AFFINE_RESHAPE4]])
    // CHECK:    [[CONVERT1:%.+]] = IE.Convert([[AFFINE_RESHAPE3]])

    // CHECK:    return [[CONVERT0]], [[CONVERT1]] : tensor<1x1x3072xf32>, tensor<1x1x3072xf32>
}

// -----

!qElemType = !quant.uniform<u2:f16, 1.000000e+00>

// CHECK-LABEL: @NotMergeParallelFCWithDynamicDequantizeNonConstInput
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x1x8192xf16>,
// CHECK-SAME:      [[WEIGHTS0:%.+]]: tensor<3072x128x64x!qElemType>
func.func @NotMergeParallelFCWithDynamicDequantizeNonConstInput(%input: tensor<1x1x8192xf16>, %weights0: tensor<3072x128x64x!qElemType>) -> (tensor<1x1x3072xf32>, tensor<1x1x3072xf32>) {
    %cst = const.Declare tensor<3072x128x1xf16> = dense<1.0> : tensor<3072x128x1xf32>, [#const.CastElemType<f16>]
    %cst_0 = const.Declare tensor<3072x128xf16> = dense<1.0> : tensor<3072x128x1xf32>, [#const.Rescale<Content<dense<2> : tensor<3072x128x1xui2>, [#const.ConvertElemType<ui8>, #const.CastElemType<f32>]>>, #const.Reshape<[3072, 128]>, #const.CastElemType<f16>]
    %cst_1 = const.Declare tensor<3072x128xf16> = dense<1.0> : tensor<3072x128x1xf32>, [#const.Rescale<Content<dense<3> : tensor<3072x128x1xui2>, [#const.ConvertElemType<ui8>, #const.CastElemType<f32>]>>, #const.Reshape<[3072, 128]>, #const.CastElemType<f16>]
    %cst_2 = const.Declare tensor<3072x128x64x!qElemType> = dense<1> : tensor<3072x128x64xui2>, [#const.CastElemType<!qElemType>]
    %cst_4 = const.Declare tensor<3072x128x1xf16> = dense<1.0> : tensor<3072x128x1xf32>, [#const.CastElemType<f16>]
    %0 = IE.DynamicDequantize(%weights0, %cst_4) {dstElemType = f16} : tensor<3072x128x64x!qElemType>, tensor<3072x128x1xf16> -> tensor<3072x128x64xf16>
    %1 = IE.DynamicDequantize(%cst_2, %cst) {dstElemType = f16} : tensor<3072x128x64x!qElemType>, tensor<3072x128x1xf16> -> tensor<3072x128x64xf16>
    %2 = IE.AffineReshape(%1) {dim_mapping = [[0], [1], [1]], shape_value = [3072, 8192]} : tensor<3072x128x64xf16> -> tensor<3072x8192xf16>
    %3 = IE.AffineReshape(%input) {dim_mapping = [[0], [0], [1]], shape_value = [1, 8192]} : tensor<1x1x8192xf16> -> tensor<1x8192xf16>
    %4 = IE.FullyConnected(%3, %2) : tensor<1x8192xf16>, tensor<3072x8192xf16> -> tensor<1x3072xf16>
    %5 = IE.AffineReshape(%input) {dim_mapping = [[0], [0], [1, 2]], shape_value = [1, 128, 64]} : tensor<1x1x8192xf16> -> tensor<1x128x64xf16>
    %6 = IE.ReduceSum(%5) {axes_value = [2]} : tensor<1x128x64xf16> -> tensor<1x128xf16>
    %7 = IE.FullyConnected(%6, %cst_1) : tensor<1x128xf16>, tensor<3072x128xf16> -> tensor<1x3072xf16>
    %8 = IE.Subtract(%4, %7) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3072xf16>, tensor<1x3072xf16> -> tensor<1x3072xf16>
    %9 = IE.AffineReshape(%8) {dim_mapping = [[0, 1], [2]], shape_value = [1, 1, 3072]} : tensor<1x3072xf16> -> tensor<1x1x3072xf16>
    %10 = IE.AffineReshape(%0) {dim_mapping = [[0], [1], [1]], shape_value = [3072, 8192]} : tensor<3072x128x64xf16> -> tensor<3072x8192xf16>
    %11 = IE.FullyConnected(%3, %10) : tensor<1x8192xf16>, tensor<3072x8192xf16> -> tensor<1x3072xf16>
    %12 = IE.FullyConnected(%6, %cst_0) : tensor<1x128xf16>, tensor<3072x128xf16> -> tensor<1x3072xf16>
    %13 = IE.Subtract(%11, %12) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3072xf16>, tensor<1x3072xf16> -> tensor<1x3072xf16>
    %14 = IE.AffineReshape(%13) {dim_mapping = [[0, 1], [2]], shape_value = [1, 1, 3072]} : tensor<1x3072xf16> -> tensor<1x1x3072xf16>
    %15 = IE.Convert(%14) {dstElemType = f32} : tensor<1x1x3072xf16> -> tensor<1x1x3072xf32>
    %16 = IE.Convert(%9) {dstElemType = f32} : tensor<1x1x3072xf16> -> tensor<1x1x3072xf32>
    return %15, %16 : tensor<1x1x3072xf32>, tensor<1x1x3072xf32>

    // CHECK:    [[CST_SCALE:%.+]] = const.Declare tensor<3072x128x1xf16>
    // CHECK:    [[CST_ZP0:%.+]] = const.Declare tensor<3072x128xf16>
    // CHECK:    [[CST_ZP1:%.+]] = const.Declare tensor<3072x128xf16>
    // CHECK:    [[CST_WEIGHTS:%.+]] = const.Declare tensor<3072x128x64x!qElemType>
    // CHECK:    [[DYN_DEQUANT0:%.+]] = IE.DynamicDequantize([[WEIGHTS0]], [[CST_SCALE]])
    // CHECK:    [[DYN_DEQUANT1:%.+]] = IE.DynamicDequantize([[CST_WEIGHTS]], [[CST_SCALE]])
    // CHECK:    [[AFFINE_RESHAPE0:%.+]] = IE.AffineReshape([[DYN_DEQUANT1]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [1], [1]], shape_value = [3072, 8192]} : tensor<3072x128x64xf16> -> tensor<3072x8192xf16>
    // CHECK:    [[AFFINE_RESHAPE1:%.+]] = IE.AffineReshape([[INPUT]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [0], [1]], shape_value = [1, 8192]} : tensor<1x1x8192xf16> -> tensor<1x8192xf16>

    // CHECK:    [[FULLY_CONNECTED0:%.+]] = IE.FullyConnected([[AFFINE_RESHAPE1]], [[AFFINE_RESHAPE0]])
    // CHECK:    [[AFFINE_RESHAPE2:%.+]] = IE.AffineReshape([[INPUT]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [0], [1, 2]], shape_value = [1, 128, 64]} : tensor<1x1x8192xf16> -> tensor<1x128x64xf16>
    // CHECK:    [[REDUCE_SUM:%.+]] = IE.ReduceSum([[AFFINE_RESHAPE2]])

    // CHECK:    [[FULLY_CONNECTED1:%.+]] = IE.FullyConnected([[REDUCE_SUM]], [[CST_ZP1]])
    // CHECK:    [[SUBTRACT0:%.+]] = IE.Subtract([[FULLY_CONNECTED0]], [[FULLY_CONNECTED1]])
    // CHECK:    [[AFFINE_RESHAPE3:%.+]] = IE.AffineReshape([[SUBTRACT0]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0, 1], [2]], shape_value = [1, 1, 3072]} : tensor<1x3072xf16> -> tensor<1x1x3072xf16>
    // CHECK:    [[AFFINE_RESHAPE4:%.+]] = IE.AffineReshape([[DYN_DEQUANT0]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [1], [1]], shape_value = [3072, 8192]} : tensor<3072x128x64xf16> -> tensor<3072x8192xf16>

    // CHECK:    [[FULLY_CONNECTED2:%.+]] = IE.FullyConnected([[AFFINE_RESHAPE1]], [[AFFINE_RESHAPE4]])
    // CHECK:    [[FULLY_CONNECTED3:%.+]] = IE.FullyConnected([[REDUCE_SUM]], [[CST_ZP0]])
    // CHECK:    [[SUBTRACT1:%.+]] = IE.Subtract([[FULLY_CONNECTED2]], [[FULLY_CONNECTED3]])
    // CHECK:    [[AFFINE_RESHAPE5:%.+]] = IE.AffineReshape([[SUBTRACT1]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0, 1], [2]], shape_value = [1, 1, 3072]} : tensor<1x3072xf16> -> tensor<1x1x3072xf16>
    // CHECK:    [[CONVERT0:%.+]] = IE.Convert([[AFFINE_RESHAPE5]])
    // CHECK:    [[CONVERT1:%.+]] = IE.Convert([[AFFINE_RESHAPE3]])

    // CHECK:    return [[CONVERT0]], [[CONVERT1]] : tensor<1x1x3072xf32>, tensor<1x1x3072xf32>
}

// -----

// Parallel FC ops with direct constant weights whose outputs are stacked via
// AffineReshape → Concat. The Concat is eliminated and replaced by a Reshape.
//
// CHECK-LABEL: @MergeDirectConstWeightsWithConcatElim
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x4xf16>)
func.func @MergeDirectConstWeightsWithConcatElim(%x: tensor<1x4xf16>) -> tensor<2x1x3xf16> {
    %w1 = const.Declare tensor<3x4xf16> = dense<1.0> : tensor<3x4xf16>
    %w2 = const.Declare tensor<3x4xf16> = dense<2.0> : tensor<3x4xf16>
    %0 = IE.FullyConnected(%x, %w1) : tensor<1x4xf16>, tensor<3x4xf16> -> tensor<1x3xf16>
    %1 = IE.AffineReshape(%0) {dim_mapping = [[0, 1], [2]], shape_value = [1, 1, 3]}
            : tensor<1x3xf16> -> tensor<1x1x3xf16>
    %2 = IE.FullyConnected(%x, %w2) : tensor<1x4xf16>, tensor<3x4xf16> -> tensor<1x3xf16>
    %3 = IE.AffineReshape(%2) {dim_mapping = [[0, 1], [2]], shape_value = [1, 1, 3]}
            : tensor<1x3xf16> -> tensor<1x1x3xf16>
    %4 = IE.Concat(%1, %3) {static_offsets = [[0, 0, 0], [1, 0, 0]]}
            : tensor<1x1x3xf16>, tensor<1x1x3xf16> -> tensor<2x1x3xf16>
    return %4 : tensor<2x1x3xf16>

    // CHECK:     [[MERGED_W:%.+]] = const.Declare tensor<6x4xf16>
    // CHECK:     [[FC:%.+]] = IE.FullyConnected([[ARG0]], [[MERGED_W]]) : tensor<1x4xf16>, tensor<6x4xf16> -> tensor<1x6xf16>
    // CHECK-NOT: IE.Slice
    // CHECK-NOT: IE.Concat
    // CHECK:     [[RESHAPED:%.+]] = IE.Reshape([[FC]]) {shape_value = [2, 1, 3]} : tensor<1x6xf16> -> tensor<2x1x3xf16>
    // CHECK:     return [[RESHAPED]] : tensor<2x1x3xf16>
}

// -----

// One FC weight is a direct Const::DeclareOp; the other passes through an AffineReshape,
// so isDirectConst = false and the direct-const Concat-elimination path must NOT fire.
//
// CHECK-LABEL: @DoNotMergeDirectConstWhenNotAllWeightsAreConst
func.func @DoNotMergeDirectConstWhenNotAllWeightsAreConst(%x: tensor<1x4xf16>, %w_runtime: tensor<3x4xf16>)
        -> tensor<2x1x3xf16> {
    %w1 = const.Declare tensor<3x4xf16> = dense<1.0> : tensor<3x4xf16>
    // Weight of second FC is not a direct const (comes from a function argument via AffineReshape).
    %w2 = IE.AffineReshape(%w_runtime) {dim_mapping = [[0], [1]], shape_value = [3, 4]}
            : tensor<3x4xf16> -> tensor<3x4xf16>
    %0 = IE.FullyConnected(%x, %w1) : tensor<1x4xf16>, tensor<3x4xf16> -> tensor<1x3xf16>
    %1 = IE.AffineReshape(%0) {dim_mapping = [[0, 1], [2]], shape_value = [1, 1, 3]}
            : tensor<1x3xf16> -> tensor<1x1x3xf16>
    %2 = IE.FullyConnected(%x, %w2) : tensor<1x4xf16>, tensor<3x4xf16> -> tensor<1x3xf16>
    %3 = IE.AffineReshape(%2) {dim_mapping = [[0, 1], [2]], shape_value = [1, 1, 3]}
            : tensor<1x3xf16> -> tensor<1x1x3xf16>
    %4 = IE.Concat(%1, %3) {static_offsets = [[0, 0, 0], [1, 0, 0]]}
            : tensor<1x1x3xf16>, tensor<1x1x3xf16> -> tensor<2x1x3xf16>
    return %4 : tensor<2x1x3xf16>

    // CHECK-NOT: IE.Reshape
    // CHECK:     IE.FullyConnected
    // CHECK:     IE.FullyConnected
    // CHECK:     IE.Concat
}

// -----

// One FC output has two uses (fed into Concat AND returned directly).
// getConcatThroughReshape returns nullptr for that FC, so directConstWithConcat = false.
//
// CHECK-LABEL: @DoNotMergeDirectConstWhenFCHasMultipleUsers
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x4xf16>)
func.func @DoNotMergeDirectConstWhenFCHasMultipleUsers(%x: tensor<1x4xf16>) -> (tensor<2x1x3xf16>, tensor<1x3xf16>) {
    %w1 = const.Declare tensor<3x4xf16> = dense<1.0> : tensor<3x4xf16>
    %w2 = const.Declare tensor<3x4xf16> = dense<2.0> : tensor<3x4xf16>
    %0 = IE.FullyConnected(%x, %w1) : tensor<1x4xf16>, tensor<3x4xf16> -> tensor<1x3xf16>
    %1 = IE.AffineReshape(%0) {dim_mapping = [[0, 1], [2]], shape_value = [1, 1, 3]}
            : tensor<1x3xf16> -> tensor<1x1x3xf16>
    %2 = IE.FullyConnected(%x, %w2) : tensor<1x4xf16>, tensor<3x4xf16> -> tensor<1x3xf16>
    %3 = IE.AffineReshape(%2) {dim_mapping = [[0, 1], [2]], shape_value = [1, 1, 3]}
            : tensor<1x3xf16> -> tensor<1x1x3xf16>
    %4 = IE.Concat(%1, %3) {static_offsets = [[0, 0, 0], [1, 0, 0]]}
            : tensor<1x1x3xf16>, tensor<1x1x3xf16> -> tensor<2x1x3xf16>
    return %4, %0 : tensor<2x1x3xf16>, tensor<1x3xf16>

    // CHECK-NOT: IE.Reshape
    // CHECK:     IE.FullyConnected
    // CHECK:     IE.FullyConnected
    // CHECK:     IE.Concat
}

// -----

// Concat has an extra input from a non-FC source (3 inputs, 2 FCs).
// allFeedSameConcat becomes false because input count != FC group size.
//
// CHECK-LABEL: @DoNotMergeDirectConstWhenConcatHasExtraInput
func.func @DoNotMergeDirectConstWhenConcatHasExtraInput(%x: tensor<1x4xf16>, %extra: tensor<1x1x3xf16>) -> tensor<3x1x3xf16> {
    %w1 = const.Declare tensor<3x4xf16> = dense<1.0> : tensor<3x4xf16>
    %w2 = const.Declare tensor<3x4xf16> = dense<2.0> : tensor<3x4xf16>
    %0 = IE.FullyConnected(%x, %w1) : tensor<1x4xf16>, tensor<3x4xf16> -> tensor<1x3xf16>
    %1 = IE.AffineReshape(%0) {dim_mapping = [[0, 1], [2]], shape_value = [1, 1, 3]}
            : tensor<1x3xf16> -> tensor<1x1x3xf16>
    %2 = IE.FullyConnected(%x, %w2) : tensor<1x4xf16>, tensor<3x4xf16> -> tensor<1x3xf16>
    %3 = IE.AffineReshape(%2) {dim_mapping = [[0, 1], [2]], shape_value = [1, 1, 3]}
            : tensor<1x3xf16> -> tensor<1x1x3xf16>
    %4 = IE.Concat(%1, %3, %extra) {static_offsets = [[0, 0, 0], [1, 0, 0], [2, 0, 0]]}
            : tensor<1x1x3xf16>, tensor<1x1x3xf16>, tensor<1x1x3xf16> -> tensor<3x1x3xf16>
    return %4 : tensor<3x1x3xf16>

    // CHECK-NOT: IE.Reshape
    // CHECK:     IE.FullyConnected
    // CHECK:     IE.FullyConnected
    // CHECK:     IE.Concat
}
