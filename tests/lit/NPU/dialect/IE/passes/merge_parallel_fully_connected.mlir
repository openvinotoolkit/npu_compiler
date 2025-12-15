//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//


// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --merge-parallel-fully-connected %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX


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
    // CHECK-DAG:   [[CST_IN:%.+]]  = const.Declare tensor<2x3x5xf32> =
    // CHECK-SAME{LITERAL}:         dense<[[[1.000000e+00, 2.000000e+00, 3.000000e+00, 1.000000e+00, 2.000000e+00], [1.000000e+00, 2.000000e+00, 1.000000e+00, 1.000000e+00, 2.000000e+00], [2.000000e+00, 3.000000e+00, 2.000000e+00, 2.000000e+00, 3.000000e+00]], [[3.000000e+00, 4.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00], [1.000000e+00, 3.000000e+00, 2.000000e+00, 1.000000e+00, 3.000000e+00], [2.000000e+00, 3.000000e+00, 4.000000e+00, 2.000000e+00, 3.000000e+00]]]> : tensor<2x3x5xf32>
    // CHECK-DAG:   [[CST_OUTLOW:%.+]]  = const.Declare tensor<2x1x5xf32> =
    // CHECK-SAME{LITERAL}:         dense<[[[-1.000000e+00, -2.000000e+00, -3.000000e+00, -1.000000e+00, -2.000000e+00]], [[-3.000000e+00, -4.000000e+00, -1.000000e+00, -3.000000e+00, -4.000000e+00]]]> : tensor<2x1x5xf32>
    // CHECK-DAG:   [[CST_OUTHIGH:%.+]]  = const.Declare tensor<2x1x5xf32> =
    // CHECK-SAME{LITERAL}:         dense<[[[1.000000e+00, 2.000000e+00, 3.000000e+00, 1.000000e+00, 2.000000e+00]], [[3.000000e+00, 4.000000e+00, 1.000000e+00, 3.000000e+00, 4.000000e+00]]]> : tensor<2x1x5xf32>
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
    // CHECK-DAG:   [[CST_IN:%.+]]  = const.Declare tensor<2x3x5xf32> =
    // CHECK-SAME{LITERAL}:         dense<[[[1.000000e+00, 2.000000e+00, 3.000000e+00, 1.000000e+00, 2.000000e+00], [1.000000e+00, 2.000000e+00, 1.000000e+00, 1.000000e+00, 2.000000e+00], [2.000000e+00, 3.000000e+00, 2.000000e+00, 2.000000e+00, 3.000000e+00]], [[3.000000e+00, 4.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00], [1.000000e+00, 3.000000e+00, 2.000000e+00, 1.000000e+00, 3.000000e+00], [2.000000e+00, 3.000000e+00, 4.000000e+00, 2.000000e+00, 3.000000e+00]]]> : tensor<2x3x5xf32>
    // CHECK-DAG:   [[CST_OUTLOW:%.+]]  = const.Declare tensor<2x1x5xf32> =
    // CHECK-SAME{LITERAL}:         dense<[[[-1.000000e+00, -2.000000e+00, -3.000000e+00, -1.000000e+00, -2.000000e+00]], [[-3.000000e+00, -4.000000e+00, -1.000000e+00, -3.000000e+00, -4.000000e+00]]]> : tensor<2x1x5xf32>
    // CHECK-DAG:   [[CST_OUTHIGH:%.+]]  = const.Declare tensor<2x1x5xf32> =
    // CHECK-SAME{LITERAL}:         dense<[[[1.000000e+00, 2.000000e+00, 3.000000e+00, 1.000000e+00, 2.000000e+00]], [[3.000000e+00, 4.000000e+00, 1.000000e+00, 3.000000e+00, 4.000000e+00]]]> : tensor<2x1x5xf32>
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
