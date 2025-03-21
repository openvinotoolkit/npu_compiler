//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --fuse-scales-to-accumulate %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
#CN = affine_map<(d0, d1) -> (d1, d0)>

// CHECK:   @FuseScales
// CHECK-SAME:  [[BLOCK_ARG:%arg0]]: tensor<3x64xf32>
func.func @FuseScales(%arg0: tensor<3x64xf32>) -> tensor<3x16xf32> {
    %WEIGHTS_0 = const.Declare tensor<1x32x16xf32> = dense<1.125> : tensor<1x32x16xf32>
    // CHECK-DAG:   [[WEIGHTS_0:%.+]] = const.Declare tensor<1x32x16xf32> = dense<1.125000e+00>
    %WEIGHTS_1 = const.Declare tensor<1x32x16xf32> = dense<2.0625> : tensor<1x32x16xf32>
    // CHECK-DAG:   [[WEIGHTS_1:%.+]] = const.Declare tensor<1x32x16xf32> = dense<2.062500e+00>

    %FQ_OUT_LO_0 = const.Declare tensor<1x1x16xf32> = dense<[[[
        -1.280000e+02, -2.560000e+02, -3.840000e+02, -5.120000e+02,
        -6.400000e+02, -7.680000e+02, -8.960000e+02, -1.280000e+02,
        -2.560000e+02, -3.840000e+02, -5.120000e+02, -6.400000e+02,
        -7.680000e+02, -8.960000e+02, -1.280000e+02, -2.560000e+02
    ]]]> : tensor<1x1x16xf32>

    %FQ_OUT_LO_1 = const.Declare tensor<1x1x16xf32> = dense<[[[
        -3.840000e+02, -5.120000e+02, -6.400000e+02, -7.680000e+02,
        -8.960000e+02, -1.280000e+02, -2.560000e+02, -3.840000e+02,
        -5.120000e+02, -6.400000e+02, -7.680000e+02, -8.960000e+02,
        -1.280000e+02, -2.560000e+02, -3.840000e+02, -5.120000e+02
    ]]]> : tensor<1x1x16xf32>

    %FQ_OUT_HI_0 = const.Declare tensor<1x1x16xf32> = dense<[[[
        1.260000e+02, 2.520000e+02, 3.780000e+02, 5.040000e+02,
        6.300000e+02, 7.560000e+02, 8.820000e+02, 1.260000e+02,
        2.520000e+02, 3.780000e+02, 5.040000e+02, 6.300000e+02,
        7.560000e+02, 8.820000e+02, 1.260000e+02, 2.520000e+02
    ]]]> : tensor<1x1x16xf32>

    %FQ_OUT_HI_1 = const.Declare tensor<1x1x16xf32> = dense<[[[
        3.780000e+02, 5.040000e+02, 6.300000e+02, 7.560000e+02,
        8.820000e+02, 1.260000e+02, 2.520000e+02, 3.780000e+02,
        5.040000e+02, 6.300000e+02, 7.560000e+02, 8.820000e+02,
        1.260000e+02, 2.520000e+02, 3.780000e+02, 5.040000e+02
    ]]]> : tensor<1x1x16xf32>
    // Note that FakeQuantize output parameters are collapsed in a splat value here.
    // This happens because the pass divided them all by the scale.
    // For instance, consider the following out low and out high:
    // FQ_OUT_LO_0 = [-384, -512, -640]
    // FQ_OUT_HI_0 = [ 378,  504,  630]
    // To get the scales, the pass must subtract high from low and divide by the number of levels - 1:
    // SCALES_0 = [ {378 - (-384)} / 254, {504 - (-512)} / 254, {630 - (-640)} / 254]
    // SCALES_0 = [ 3, 4, 5 ]
    // Now, the pass must divide all the FQ_OUT_LO and FQ_OUT_HI by their corresponding scales:
    // FQ_OUT_LO = [-384 / 3, -512 / 4, -640 / 5] = [-128, -128, -128]
    // FQ_OUT_HI = [ 378 / 3,  504 / 4,  630 / 5] = [ 126,  126,  126]
    // CHECK-DAG:   [[FQ_OUT_LO:%.+]] = const.Declare tensor<1x1x16xf32> = dense<-1.280000e+02>
    // CHECK-DAG:   [[FQ_OUT_HI:%.+]] = const.Declare tensor<1x1x16xf32> = dense<1.260000e+02>

    // CHECK-DAG:   [[SCALE_0:%.+]] = const.Declare tensor<16xf32> = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 1.000000e+00, 2.000000e+00]> : tensor<16xf32>
    // CHECK-DAG:   [[SCALE_1:%.+]] = const.Declare tensor<16xf32> = dense<[3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]> : tensor<16xf32>

    %FQ_IN_LO = const.Declare tensor<1x1x1xf32> = dense<-1.270000e+02> : tensor<1x1x1xf32>
    // CHECK-DAG:   [[FQ_IN_LO:%.+]] = const.Declare tensor<1x1x1xf32> = dense<-1.270000e+02>
    %FQ_IN_HI = const.Declare tensor<1x1x1xf32> = dense<1.270000e+02> : tensor<1x1x1xf32>
    // CHECK-DAG:   [[FQ_IN_HI:%.+]] = const.Declare tensor<1x1x1xf32> = dense<1.270000e+02>

    %FQ_0 = IE.FakeQuantize(%WEIGHTS_0, %FQ_IN_LO, %FQ_IN_HI, %FQ_OUT_LO_0, %FQ_OUT_HI_0) {
          auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
          levels = 255 : i64
    } : tensor<1x32x16xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<1x1x16xf32>, tensor<1x1x16xf32> -> tensor<1x32x16xf32>
    // CHECK:   [[FQ_0:%.+]] = IE.FakeQuantize([[WEIGHTS_0]], [[FQ_IN_LO]], [[FQ_IN_HI]], [[FQ_OUT_LO]], [[FQ_OUT_HI]])

    %FQ_1 = IE.FakeQuantize(%WEIGHTS_1, %FQ_IN_LO, %FQ_IN_HI, %FQ_OUT_LO_1, %FQ_OUT_HI_1) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 255 : i64
    } : tensor<1x32x16xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<1x1x16xf32>, tensor<1x1x16xf32> -> tensor<1x32x16xf32>
    // CHECK:   [[FQ_1:%.+]] = IE.FakeQuantize([[WEIGHTS_1]], [[FQ_IN_LO]], [[FQ_IN_HI]], [[FQ_OUT_LO]], [[FQ_OUT_HI]])

    %RESHAPE_FQ_0 = IE.Reshape(%FQ_0) {
        shape_value = [32, 16]
    } : tensor<1x32x16xf32> -> tensor<32x16xf32>
    // CHECK:   [[RESHAPE_FQ_0:%.+]] = IE.Reshape([[FQ_0]]) {
    // CHECK-SAME:      shape_value = [32, 16]
    // CHECK-SAME:  } : tensor<1x32x16xf32> -> tensor<32x16xf32>

    %RESHAPE_FQ_1 = IE.Reshape(%FQ_1) {
        shape_value = [32, 16]
    } : tensor<1x32x16xf32> -> tensor<32x16xf32>
    // CHECK:   [[RESHAPE_FQ_1:%.+]] = IE.Reshape([[FQ_1]]) {
    // CHECK-SAME:      shape_value = [32, 16]
    // CHECK-SAME:  } : tensor<1x32x16xf32> -> tensor<32x16xf32>

    %SLICE_0 = IE.Slice %arg0 [0, 0] [3, 32] : tensor<3x64xf32> to tensor<3x32xf32>
    // CHECK:   [[SLICE_0:%.+]] = IE.Slice [[BLOCK_ARG]] [0, 0] [3, 32] : tensor<3x64xf32> to tensor<3x32xf32>

    %SLICE_1 = IE.Slice %arg0 [0, 32] [3, 32] : tensor<3x64xf32> to tensor<3x32xf32>
    // CHECK:   [[SLICE_1:%.+]] = IE.Slice [[BLOCK_ARG]] [0, 32] [3, 32] : tensor<3x64xf32> to tensor<3x32xf32>

    %TRANSPOSE_0 = IE.Transpose(%RESHAPE_FQ_0) {
        order_value = #CN
    } : tensor<32x16xf32> -> tensor<16x32xf32>
    // CHECK:   [[TRANSPOSE_0:%.+]] = IE.Transpose([[RESHAPE_FQ_0]])

    %MATMUL_0 = IE.FullyConnected(%SLICE_0, %TRANSPOSE_0) : tensor<3x32xf32>, tensor<16x32xf32> -> tensor<3x16xf32>
    // CHECK:   [[MATMUL_0:%.+]] = IE.FullyConnected([[SLICE_0]], [[TRANSPOSE_0]]) : tensor<3x32xf32>, tensor<16x32xf32> -> tensor<3x16xf32>

    %TRANSPOSE_1 = IE.Transpose(%RESHAPE_FQ_1) {
        order_value = #CN
    } : tensor<32x16xf32> -> tensor<16x32xf32>
    // CHECK:   [[TRANSPOSE_1:%.+]] = IE.Transpose([[RESHAPE_FQ_1]])

    %MATMUL_1 = IE.FullyConnected(%SLICE_1, %TRANSPOSE_1) : tensor<3x32xf32>, tensor<16x32xf32> -> tensor<3x16xf32>
    // CHECK:   [[MATMUL_1:%.+]] = IE.FullyConnected([[SLICE_1]], [[TRANSPOSE_1]]) : tensor<3x32xf32>, tensor<16x32xf32> -> tensor<3x16xf32>

    %ACCUMULATE = IE.Accumulate(%MATMUL_0, %MATMUL_1) {
        operandSegmentSizes = array<i32: 1, 1, 0, 0>
    } : tensor<3x16xf32>, tensor<3x16xf32> -> tensor<3x16xf32>
    // CHECK:   [[ACCUMULATE:%.+]] = IE.Accumulate([[MATMUL_0]], [[MATMUL_1]], [[SCALE_0]], [[SCALE_1]]) {
    // CHECK-SAME:      operandSegmentSizes = array<i32: 1, 1, 1, 1>
    // CHECK-SAME:  } : tensor<3x16xf32>, tensor<3x16xf32>, tensor<16xf32>, tensor<16xf32> -> tensor<3x16xf32>

    return %ACCUMULATE : tensor<3x16xf32>
    // CHECK:   [[ACCUMULATE]] : tensor<3x16xf32>
}

// -----

#CN = affine_map<(d0, d1) -> (d1, d0)>

// CHECK:   @FuseScalesNoTranspose
// CHECK-SAME:  [[BLOCK_ARG:%arg0]]: tensor<3x64xf32>
func.func @FuseScalesNoTranspose(%arg0: tensor<3x64xf32>) -> tensor<3x16xf32> {
    %WEIGHTS_0 = const.Declare tensor<16x1x32xf32> = dense<1.125> : tensor<16x1x32xf32>
    // CHECK-DAG:   [[WEIGHTS_0:%.+]] = const.Declare tensor<16x1x32xf32> = dense<1.125000e+00>
    %WEIGHTS_1 = const.Declare tensor<16x1x32xf32> = dense<2.0625> : tensor<16x1x32xf32>
    // CHECK-DAG:   [[WEIGHTS_1:%.+]] = const.Declare tensor<16x1x32xf32> = dense<2.062500e+00>

    %FQ_OUT_LO_0 = const.Declare tensor<16x1x1xf32> = dense<[
        [[-1.280000e+02]], [[-2.560000e+02]], [[-3.840000e+02]], [[-5.120000e+02]],
        [[-6.400000e+02]], [[-7.680000e+02]], [[-8.960000e+02]], [[-1.280000e+02]],
        [[-2.560000e+02]], [[-3.840000e+02]], [[-5.120000e+02]], [[-6.400000e+02]],
        [[-7.680000e+02]], [[-8.960000e+02]], [[-1.280000e+02]], [[-2.560000e+02]]
    ]> : tensor<16x1x1xf32>

    %FQ_OUT_LO_1 = const.Declare tensor<16x1x1xf32> = dense<[
        [[-3.840000e+02]], [[-5.120000e+02]], [[-6.400000e+02]], [[-7.680000e+02]],
        [[-8.960000e+02]], [[-1.280000e+02]], [[-2.560000e+02]], [[-3.840000e+02]],
        [[-5.120000e+02]], [[-6.400000e+02]], [[-7.680000e+02]], [[-8.960000e+02]],
        [[-1.280000e+02]], [[-2.560000e+02]], [[-3.840000e+02]], [[-5.120000e+02]]
    ]> : tensor<16x1x1xf32>

    %FQ_OUT_HI_0 = const.Declare tensor<16x1x1xf32> = dense<[
        [[1.260000e+02]], [[2.520000e+02]], [[3.780000e+02]], [[5.040000e+02]],
        [[6.300000e+02]], [[7.560000e+02]], [[8.820000e+02]], [[1.260000e+02]],
        [[2.520000e+02]], [[3.780000e+02]], [[5.040000e+02]], [[6.300000e+02]],
        [[7.560000e+02]], [[8.820000e+02]], [[1.260000e+02]], [[2.520000e+02]]
    ]> : tensor<16x1x1xf32>

    %FQ_OUT_HI_1 = const.Declare tensor<16x1x1xf32> = dense<[
        [[3.780000e+02]], [[5.040000e+02]], [[6.300000e+02]], [[7.560000e+02]],
        [[8.820000e+02]], [[1.260000e+02]], [[2.520000e+02]], [[3.780000e+02]],
        [[5.040000e+02]], [[6.300000e+02]], [[7.560000e+02]], [[8.820000e+02]],
        [[1.260000e+02]], [[2.520000e+02]], [[3.780000e+02]], [[5.040000e+02]]
    ]> : tensor<16x1x1xf32>
    // CHECK-DAG:   [[FQ_OUT_LO:%.+]] = const.Declare tensor<16x1x1xf32> = dense<-1.280000e+02>
    // CHECK-DAG:   [[FQ_OUT_HI:%.+]] = const.Declare tensor<16x1x1xf32> = dense<1.260000e+02>

    // CHECK-DAG:   [[SCALE_0:%.+]] = const.Declare tensor<16xf32> = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 1.000000e+00, 2.000000e+00]> : tensor<16xf32>
    // CHECK-DAG:   [[SCALE_1:%.+]] = const.Declare tensor<16xf32> = dense<[3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]> : tensor<16xf32>

    %FQ_IN_LO = const.Declare tensor<1x1x1xf32> = dense<-1.270000e+02> : tensor<1x1x1xf32>
    // CHECK-DAG:   [[FQ_IN_LO:%.+]] = const.Declare tensor<1x1x1xf32> = dense<-1.270000e+02>
    %FQ_IN_HI = const.Declare tensor<1x1x1xf32> = dense<1.270000e+02> : tensor<1x1x1xf32>
    // CHECK-DAG:   [[FQ_IN_HI:%.+]] = const.Declare tensor<1x1x1xf32> = dense<1.270000e+02>

    %FQ_0 = IE.FakeQuantize(%WEIGHTS_0, %FQ_IN_LO, %FQ_IN_HI, %FQ_OUT_LO_0, %FQ_OUT_HI_0) {
          auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
          levels = 255 : i64
    } : tensor<16x1x32xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<16x1x1xf32>, tensor<16x1x1xf32> -> tensor<16x1x32xf32>
    // CHECK:   [[FQ_0:%.+]] = IE.FakeQuantize([[WEIGHTS_0]], [[FQ_IN_LO]], [[FQ_IN_HI]], [[FQ_OUT_LO]], [[FQ_OUT_HI]])

    %FQ_1 = IE.FakeQuantize(%WEIGHTS_1, %FQ_IN_LO, %FQ_IN_HI, %FQ_OUT_LO_1, %FQ_OUT_HI_1) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 255 : i64
    } : tensor<16x1x32xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<16x1x1xf32>, tensor<16x1x1xf32> -> tensor<16x1x32xf32>
    // CHECK:   [[FQ_1:%.+]] = IE.FakeQuantize([[WEIGHTS_1]], [[FQ_IN_LO]], [[FQ_IN_HI]], [[FQ_OUT_LO]], [[FQ_OUT_HI]])

    %RESHAPE_FQ_0 = IE.Reshape(%FQ_0) {
        shape_value = [16, 32]
    } : tensor<16x1x32xf32> -> tensor<16x32xf32>
    // CHECK:   [[RESHAPE_FQ_0:%.+]] = IE.Reshape([[FQ_0]]) {
    // CHECK-SAME:      shape_value = [16, 32]
    // CHECK-SAME:  } : tensor<16x1x32xf32> -> tensor<16x32xf32>

    %RESHAPE_FQ_1 = IE.Reshape(%FQ_1) {
        shape_value = [16, 32]
    } : tensor<16x1x32xf32> -> tensor<16x32xf32>
    // CHECK:   [[RESHAPE_FQ_1:%.+]] = IE.Reshape([[FQ_1]]) {
    // CHECK-SAME:      shape_value = [16, 32]
    // CHECK-SAME:  } : tensor<16x1x32xf32> -> tensor<16x32xf32>

    %SLICE_0 = IE.Slice %arg0 [0, 0] [3, 32] : tensor<3x64xf32> to tensor<3x32xf32>
    // CHECK:   [[SLICE_0:%.+]] = IE.Slice [[BLOCK_ARG]] [0, 0] [3, 32] : tensor<3x64xf32> to tensor<3x32xf32>

    %SLICE_1 = IE.Slice %arg0 [0, 32] [3, 32] : tensor<3x64xf32> to tensor<3x32xf32>
    // CHECK:   [[SLICE_1:%.+]] = IE.Slice [[BLOCK_ARG]] [0, 32] [3, 32] : tensor<3x64xf32> to tensor<3x32xf32>

    %MATMUL_0 = IE.FullyConnected(%SLICE_0, %RESHAPE_FQ_0) : tensor<3x32xf32>, tensor<16x32xf32> -> tensor<3x16xf32>
    // CHECK:   [[MATMUL_0:%.+]] = IE.FullyConnected([[SLICE_0]], [[RESHAPE_FQ_0]]) : tensor<3x32xf32>, tensor<16x32xf32> -> tensor<3x16xf32>

    %MATMUL_1 = IE.FullyConnected(%SLICE_1, %RESHAPE_FQ_1) : tensor<3x32xf32>, tensor<16x32xf32> -> tensor<3x16xf32>
    // CHECK:   [[MATMUL_1:%.+]] = IE.FullyConnected([[SLICE_1]], [[RESHAPE_FQ_1]]) : tensor<3x32xf32>, tensor<16x32xf32> -> tensor<3x16xf32>

    %ACCUMULATE = IE.Accumulate(%MATMUL_0, %MATMUL_1) {
        operandSegmentSizes = array<i32: 1, 1, 0, 0>
    } : tensor<3x16xf32>, tensor<3x16xf32> -> tensor<3x16xf32>
    // CHECK:   [[ACCUMULATE:%.+]] = IE.Accumulate([[MATMUL_0]], [[MATMUL_1]], [[SCALE_0]], [[SCALE_1]]) {
    // CHECK-SAME:      operandSegmentSizes = array<i32: 1, 1, 1, 1>
    // CHECK-SAME:  } : tensor<3x16xf32>, tensor<3x16xf32>, tensor<16xf32>, tensor<16xf32> -> tensor<3x16xf32>

    return %ACCUMULATE : tensor<3x16xf32>
    // CHECK:   [[ACCUMULATE]] : tensor<3x16xf32>
}

// -----

#CN = affine_map<(d0, d1) -> (d1, d0)>

// CHECK:   @FuseScalesNoTransposeWeightsAsParameters
// CHECK-SAME:  [[INPUT:%.+]]: tensor<3x64xf32>,
// CHECK-SAME:  [[WEIGHTS_0:%.+]]: tensor<16x1x32xf32>,
// CHECK-SAME:  [[WEIGHTS_1:%.+]]: tensor<16x1x32xf32>
func.func @FuseScalesNoTransposeWeightsAsParameters(%input: tensor<3x64xf32>, %weights_0: tensor<16x1x32xf32>, %weights_1: tensor<16x1x32xf32>) -> tensor<3x16xf32> {
    %FQ_OUT_LO_0 = const.Declare tensor<16x1x1xf32> = dense<[
        [[-1.280000e+02]], [[-2.560000e+02]], [[-3.840000e+02]], [[-5.120000e+02]],
        [[-6.400000e+02]], [[-7.680000e+02]], [[-8.960000e+02]], [[-1.280000e+02]],
        [[-2.560000e+02]], [[-3.840000e+02]], [[-5.120000e+02]], [[-6.400000e+02]],
        [[-7.680000e+02]], [[-8.960000e+02]], [[-1.280000e+02]], [[-2.560000e+02]]
    ]> : tensor<16x1x1xf32>

    %FQ_OUT_LO_1 = const.Declare tensor<16x1x1xf32> = dense<[
        [[-3.840000e+02]], [[-5.120000e+02]], [[-6.400000e+02]], [[-7.680000e+02]],
        [[-8.960000e+02]], [[-1.280000e+02]], [[-2.560000e+02]], [[-3.840000e+02]],
        [[-5.120000e+02]], [[-6.400000e+02]], [[-7.680000e+02]], [[-8.960000e+02]],
        [[-1.280000e+02]], [[-2.560000e+02]], [[-3.840000e+02]], [[-5.120000e+02]]
    ]> : tensor<16x1x1xf32>

    %FQ_OUT_HI_0 = const.Declare tensor<16x1x1xf32> = dense<[
        [[1.260000e+02]], [[2.520000e+02]], [[3.780000e+02]], [[5.040000e+02]],
        [[6.300000e+02]], [[7.560000e+02]], [[8.820000e+02]], [[1.260000e+02]],
        [[2.520000e+02]], [[3.780000e+02]], [[5.040000e+02]], [[6.300000e+02]],
        [[7.560000e+02]], [[8.820000e+02]], [[1.260000e+02]], [[2.520000e+02]]
    ]> : tensor<16x1x1xf32>

    %FQ_OUT_HI_1 = const.Declare tensor<16x1x1xf32> = dense<[
        [[3.780000e+02]], [[5.040000e+02]], [[6.300000e+02]], [[7.560000e+02]],
        [[8.820000e+02]], [[1.260000e+02]], [[2.520000e+02]], [[3.780000e+02]],
        [[5.040000e+02]], [[6.300000e+02]], [[7.560000e+02]], [[8.820000e+02]],
        [[1.260000e+02]], [[2.520000e+02]], [[3.780000e+02]], [[5.040000e+02]]
    ]> : tensor<16x1x1xf32>
    // CHECK-DAG:   [[FQ_OUT_LO:%.+]] = const.Declare tensor<16x1x1xf32> = dense<-1.280000e+02>
    // CHECK-DAG:   [[FQ_OUT_HI:%.+]] = const.Declare tensor<16x1x1xf32> = dense<1.260000e+02>

    // CHECK-DAG:   [[SCALE_0:%.+]] = const.Declare tensor<16xf32> = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 1.000000e+00, 2.000000e+00]> : tensor<16xf32>
    // CHECK-DAG:   [[SCALE_1:%.+]] = const.Declare tensor<16xf32> = dense<[3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]> : tensor<16xf32>

    %FQ_IN_LO = const.Declare tensor<1x1x1xf32> = dense<-1.270000e+02> : tensor<1x1x1xf32>
    // CHECK-DAG:   [[FQ_IN_LO:%.+]] = const.Declare tensor<1x1x1xf32> = dense<-1.270000e+02>
    %FQ_IN_HI = const.Declare tensor<1x1x1xf32> = dense<1.270000e+02> : tensor<1x1x1xf32>
    // CHECK-DAG:   [[FQ_IN_HI:%.+]] = const.Declare tensor<1x1x1xf32> = dense<1.270000e+02>

    %FQ_0 = IE.FakeQuantize(%weights_0, %FQ_IN_LO, %FQ_IN_HI, %FQ_OUT_LO_0, %FQ_OUT_HI_0) {
          auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
          levels = 255 : i64
    } : tensor<16x1x32xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<16x1x1xf32>, tensor<16x1x1xf32> -> tensor<16x1x32xf32>
    // CHECK:   [[FQ_0:%.+]] = IE.FakeQuantize([[WEIGHTS_0]], [[FQ_IN_LO]], [[FQ_IN_HI]], [[FQ_OUT_LO]], [[FQ_OUT_HI]])

    %FQ_1 = IE.FakeQuantize(%weights_1, %FQ_IN_LO, %FQ_IN_HI, %FQ_OUT_LO_1, %FQ_OUT_HI_1) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 255 : i64
    } : tensor<16x1x32xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<16x1x1xf32>, tensor<16x1x1xf32> -> tensor<16x1x32xf32>
    // CHECK:   [[FQ_1:%.+]] = IE.FakeQuantize([[WEIGHTS_1]], [[FQ_IN_LO]], [[FQ_IN_HI]], [[FQ_OUT_LO]], [[FQ_OUT_HI]])

    %RESHAPE_FQ_0 = IE.Reshape(%FQ_0) {
        shape_value = [16, 32]
    } : tensor<16x1x32xf32> -> tensor<16x32xf32>
    // CHECK:   [[RESHAPE_FQ_0:%.+]] = IE.Reshape([[FQ_0]]) {
    // CHECK-SAME:      shape_value = [16, 32]
    // CHECK-SAME:  } : tensor<16x1x32xf32> -> tensor<16x32xf32>

    %RESHAPE_FQ_1 = IE.Reshape(%FQ_1) {
        shape_value = [16, 32]
    } : tensor<16x1x32xf32> -> tensor<16x32xf32>
    // CHECK:   [[RESHAPE_FQ_1:%.+]] = IE.Reshape([[FQ_1]]) {
    // CHECK-SAME:      shape_value = [16, 32]
    // CHECK-SAME:  } : tensor<16x1x32xf32> -> tensor<16x32xf32>

    %SLICE_0 = IE.Slice %input [0, 0] [3, 32] : tensor<3x64xf32> to tensor<3x32xf32>
    // CHECK:   [[SLICE_0:%.+]] = IE.Slice [[INPUT]] [0, 0] [3, 32] : tensor<3x64xf32> to tensor<3x32xf32>

    %SLICE_1 = IE.Slice %input [0, 32] [3, 32] : tensor<3x64xf32> to tensor<3x32xf32>
    // CHECK:   [[SLICE_1:%.+]] = IE.Slice [[INPUT]] [0, 32] [3, 32] : tensor<3x64xf32> to tensor<3x32xf32>

    %MATMUL_0 = IE.FullyConnected(%SLICE_0, %RESHAPE_FQ_0) : tensor<3x32xf32>, tensor<16x32xf32> -> tensor<3x16xf32>
    // CHECK:   [[MATMUL_0:%.+]] = IE.FullyConnected([[SLICE_0]], [[RESHAPE_FQ_0]]) : tensor<3x32xf32>, tensor<16x32xf32> -> tensor<3x16xf32>

    %MATMUL_1 = IE.FullyConnected(%SLICE_1, %RESHAPE_FQ_1) : tensor<3x32xf32>, tensor<16x32xf32> -> tensor<3x16xf32>
    // CHECK:   [[MATMUL_1:%.+]] = IE.FullyConnected([[SLICE_1]], [[RESHAPE_FQ_1]]) : tensor<3x32xf32>, tensor<16x32xf32> -> tensor<3x16xf32>

    %ACCUMULATE = IE.Accumulate(%MATMUL_0, %MATMUL_1) {
        operandSegmentSizes = array<i32: 1, 1, 0, 0>
    } : tensor<3x16xf32>, tensor<3x16xf32> -> tensor<3x16xf32>
    // CHECK:   [[ACCUMULATE:%.+]] = IE.Accumulate([[MATMUL_0]], [[MATMUL_1]], [[SCALE_0]], [[SCALE_1]]) {
    // CHECK-SAME:      operandSegmentSizes = array<i32: 1, 1, 1, 1>
    // CHECK-SAME:  } : tensor<3x16xf32>, tensor<3x16xf32>, tensor<16xf32>, tensor<16xf32> -> tensor<3x16xf32>

    return %ACCUMULATE : tensor<3x16xf32>
    // CHECK:   [[ACCUMULATE]] : tensor<3x16xf32>
}

// -----

#CN = affine_map<(d0, d1) -> (d1, d0)>

// CHECK:   @FuseScalesF16
// CHECK-SAME:  [[BLOCK_ARG:%arg0]]: tensor<3x64xf16>
func.func @FuseScalesF16(%arg0: tensor<3x64xf16>) -> tensor<3x16xf16> {
    %WEIGHTS_0 = const.Declare tensor<1x32x16xf16> = dense<1.125> : tensor<1x32x16xf16>, [#const.CastElemType<f16>]
    // CHECK-DAG:   [[WEIGHTS_0:%.+]] = const.Declare tensor<1x32x16xf16> = dense<1.125000e+00>
    %WEIGHTS_1 = const.Declare tensor<1x32x16xf16> = dense<2.0625> : tensor<1x32x16xf16>, [#const.CastElemType<f16>]
    // CHECK-DAG:   [[WEIGHTS_1:%.+]] = const.Declare tensor<1x32x16xf16> = dense<2.062500e+00>

    %FQ_OUT_LO_0 = const.Declare tensor<1x1x16xf16> = dense<[[[
        -1.280000e+02, -2.560000e+02, -3.840000e+02, -5.120000e+02,
        -6.400000e+02, -7.680000e+02, -8.960000e+02, -1.280000e+02,
        -2.560000e+02, -3.840000e+02, -5.120000e+02, -6.400000e+02,
        -7.680000e+02, -8.960000e+02, -1.280000e+02, -2.560000e+02
    ]]]> : tensor<1x1x16xf32>, [#const.CastElemType<f16>]

    %FQ_OUT_LO_1 = const.Declare tensor<1x1x16xf16> = dense<[[[
        -3.840000e+02, -5.120000e+02, -6.400000e+02, -7.680000e+02,
        -8.960000e+02, -1.280000e+02, -2.560000e+02, -3.840000e+02,
        -5.120000e+02, -6.400000e+02, -7.680000e+02, -8.960000e+02,
        -1.280000e+02, -2.560000e+02, -3.840000e+02, -5.120000e+02
    ]]]> : tensor<1x1x16xf32>, [#const.CastElemType<f16>]

    %FQ_OUT_HI_0 = const.Declare tensor<1x1x16xf16> = dense<[[[
        1.260000e+02, 2.520000e+02, 3.780000e+02, 5.040000e+02,
        6.300000e+02, 7.560000e+02, 8.820000e+02, 1.260000e+02,
        2.520000e+02, 3.780000e+02, 5.040000e+02, 6.300000e+02,
        7.560000e+02, 8.820000e+02, 1.260000e+02, 2.520000e+02
    ]]]> : tensor<1x1x16xf32>, [#const.CastElemType<f16>]

    %FQ_OUT_HI_1 = const.Declare tensor<1x1x16xf16> = dense<[[[
        3.780000e+02, 5.040000e+02, 6.300000e+02, 7.560000e+02,
        8.820000e+02, 1.260000e+02, 2.520000e+02, 3.780000e+02,
        5.040000e+02, 6.300000e+02, 7.560000e+02, 8.820000e+02,
        1.260000e+02, 2.520000e+02, 3.780000e+02, 5.040000e+02
    ]]]> : tensor<1x1x16xf32>, [#const.CastElemType<f16>]

    // CHECK-DAG:   [[FQ_OUT_LO:%.+]] = const.Declare tensor<1x1x16xf16> = dense<-1.280000e+02>
    // CHECK-DAG:   [[FQ_OUT_HI:%.+]] = const.Declare tensor<1x1x16xf16> = dense<1.260000e+02>

    // CHECK-DAG:   [[SCALE_0:%.+]] = const.Declare tensor<16xf16> = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 1.000000e+00, 2.000000e+00]> : tensor<16xf32>, [#const.CastElemType<f16>]
    // CHECK-DAG:   [[SCALE_1:%.+]] = const.Declare tensor<16xf16> = dense<[3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]> : tensor<16xf32>, [#const.CastElemType<f16>]

    %FQ_IN_LO = const.Declare tensor<1x1x1xf16> = dense<-1.270000e+02> : tensor<1x1x1xf32>, [#const.CastElemType<f16>]
    // CHECK-DAG:   [[FQ_IN_LO:%.+]] = const.Declare tensor<1x1x1xf16> = dense<-1.270000e+02>
    %FQ_IN_HI = const.Declare tensor<1x1x1xf16> = dense<1.270000e+02> : tensor<1x1x1xf32>, [#const.CastElemType<f16>]
    // CHECK-DAG:   [[FQ_IN_HI:%.+]] = const.Declare tensor<1x1x1xf16> = dense<1.270000e+02>

    %FQ_0 = IE.FakeQuantize(%WEIGHTS_0, %FQ_IN_LO, %FQ_IN_HI, %FQ_OUT_LO_0, %FQ_OUT_HI_0) {
          auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
          levels = 255 : i64
    } : tensor<1x32x16xf16>, tensor<1x1x1xf16>, tensor<1x1x1xf16>, tensor<1x1x16xf16>, tensor<1x1x16xf16> -> tensor<1x32x16xf16>
    // CHECK:   [[FQ_0:%.+]] = IE.FakeQuantize([[WEIGHTS_0]], [[FQ_IN_LO]], [[FQ_IN_HI]], [[FQ_OUT_LO]], [[FQ_OUT_HI]])

    %FQ_1 = IE.FakeQuantize(%WEIGHTS_1, %FQ_IN_LO, %FQ_IN_HI, %FQ_OUT_LO_1, %FQ_OUT_HI_1) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 255 : i64
    } : tensor<1x32x16xf16>, tensor<1x1x1xf16>, tensor<1x1x1xf16>, tensor<1x1x16xf16>, tensor<1x1x16xf16> -> tensor<1x32x16xf16>
    // CHECK:   [[FQ_1:%.+]] = IE.FakeQuantize([[WEIGHTS_1]], [[FQ_IN_LO]], [[FQ_IN_HI]], [[FQ_OUT_LO]], [[FQ_OUT_HI]])

    %RESHAPE_FQ_0 = IE.Reshape(%FQ_0) {
        shape_value = [32, 16]
    } : tensor<1x32x16xf16> -> tensor<32x16xf16>
    // CHECK:   [[RESHAPE_FQ_0:%.+]] = IE.Reshape([[FQ_0]]) {
    // CHECK-SAME:      shape_value = [32, 16]
    // CHECK-SAME:  } : tensor<1x32x16xf16> -> tensor<32x16xf16>

    %RESHAPE_FQ_1 = IE.Reshape(%FQ_1) {
        shape_value = [32, 16]
    } : tensor<1x32x16xf16> -> tensor<32x16xf16>
    // CHECK:   [[RESHAPE_FQ_1:%.+]] = IE.Reshape([[FQ_1]]) {
    // CHECK-SAME:      shape_value = [32, 16]
    // CHECK-SAME:  } : tensor<1x32x16xf16> -> tensor<32x16xf16>

    %SLICE_0 = IE.Slice %arg0 [0, 0] [3, 32] : tensor<3x64xf16> to tensor<3x32xf16>
    // CHECK:   [[SLICE_0:%.+]] = IE.Slice [[BLOCK_ARG]] [0, 0] [3, 32] : tensor<3x64xf16> to tensor<3x32xf16>

    %SLICE_1 = IE.Slice %arg0 [0, 32] [3, 32] : tensor<3x64xf16> to tensor<3x32xf16>
    // CHECK:   [[SLICE_1:%.+]] = IE.Slice [[BLOCK_ARG]] [0, 32] [3, 32] : tensor<3x64xf16> to tensor<3x32xf16>

    %TRANSPOSE_0 = IE.Transpose(%RESHAPE_FQ_0) {
        order_value = #CN
    } : tensor<32x16xf16> -> tensor<16x32xf16>
    // CHECK:   [[TRANSPOSE_0:%.+]] = IE.Transpose([[RESHAPE_FQ_0]])

    %MATMUL_0 = IE.FullyConnected(%SLICE_0, %TRANSPOSE_0) : tensor<3x32xf16>, tensor<16x32xf16> -> tensor<3x16xf16>
    // CHECK:   [[MATMUL_0:%.+]] = IE.FullyConnected([[SLICE_0]], [[TRANSPOSE_0]]) : tensor<3x32xf16>, tensor<16x32xf16> -> tensor<3x16xf16>

    %TRANSPOSE_1 = IE.Transpose(%RESHAPE_FQ_1) {
        order_value = #CN
    } : tensor<32x16xf16> -> tensor<16x32xf16>
    // CHECK:   [[TRANSPOSE_1:%.+]] = IE.Transpose([[RESHAPE_FQ_1]])

    %MATMUL_1 = IE.FullyConnected(%SLICE_1, %TRANSPOSE_1) : tensor<3x32xf16>, tensor<16x32xf16> -> tensor<3x16xf16>
    // CHECK:   [[MATMUL_1:%.+]] = IE.FullyConnected([[SLICE_1]], [[TRANSPOSE_1]]) : tensor<3x32xf16>, tensor<16x32xf16> -> tensor<3x16xf16>

    %ACCUMULATE = IE.Accumulate(%MATMUL_0, %MATMUL_1) {
        operandSegmentSizes = array<i32: 1, 1, 0, 0>
    } : tensor<3x16xf16>, tensor<3x16xf16> -> tensor<3x16xf16>
    // CHECK:   [[ACCUMULATE:%.+]] = IE.Accumulate([[MATMUL_0]], [[MATMUL_1]], [[SCALE_0]], [[SCALE_1]]) {
    // CHECK-SAME:      operandSegmentSizes = array<i32: 1, 1, 1, 1>
    // CHECK-SAME:  } : tensor<3x16xf16>, tensor<3x16xf16>, tensor<16xf16>, tensor<16xf16> -> tensor<3x16xf16>

    return %ACCUMULATE : tensor<3x16xf16>
    // CHECK:   [[ACCUMULATE]] : tensor<3x16xf16>
}

// -----

#CN = affine_map<(d0, d1) -> (d1, d0)>

// CHECK:   @SkipAccumulateWithScales
// CHECK-SAME:  [[BLOCK_ARG:%arg0]]: tensor<3x64xf32>
func.func @SkipAccumulateWithScales(%arg0: tensor<3x64xf32>) -> tensor<3x16xf32> {
    %WEIGHTS_0 = const.Declare tensor<1x32x16xf32> = dense<1.125> : tensor<1x32x16xf32>
    // CHECK-DAG:   [[WEIGHTS_0:%.+]] = const.Declare tensor<1x32x16xf32> = dense<1.125000e+00>
    %WEIGHTS_1 = const.Declare tensor<1x32x16xf32> = dense<2.0625> : tensor<1x32x16xf32>
    // CHECK-DAG:   [[WEIGHTS_1:%.+]] = const.Declare tensor<1x32x16xf32> = dense<2.062500e+00>

    %FQ_OUT_LO_0 = const.Declare tensor<1x1x16xf32> = dense<-256.0> : tensor<1x1x16xf32>
    // CHECK-DAG:   [[FQ_OUT_LO_0:%.+]] = const.Declare tensor<1x1x16xf32> = dense<-2.560000e+02> : tensor<1x1x16xf32>

    %FQ_OUT_LO_1 = const.Declare tensor<1x1x16xf32> = dense<-384.0> : tensor<1x1x16xf32>
    // CHECK-DAG:   [[FQ_OUT_LO_1:%.+]] = const.Declare tensor<1x1x16xf32> = dense<-3.840000e+02> : tensor<1x1x16xf32>

    %FQ_OUT_HI_0 = const.Declare tensor<1x1x16xf32> = dense<256.0> : tensor<1x1x16xf32>
    // CHECK-DAG:   [[FQ_OUT_HI_0:%.+]] = const.Declare tensor<1x1x16xf32> = dense<2.560000e+02> : tensor<1x1x16xf32>

    %FQ_OUT_HI_1 = const.Declare tensor<1x1x16xf32> = dense<384.0> : tensor<1x1x16xf32>
    // CHECK-DAG:   [[FQ_OUT_HI_1:%.+]] = const.Declare tensor<1x1x16xf32> = dense<3.840000e+02> : tensor<1x1x16xf32>

    %FQ_IN_LO = const.Declare tensor<1x1x1xf32> = dense<-1.270000e+02> : tensor<1x1x1xf32>
    // CHECK-DAG:   [[FQ_IN_LO:%.+]] = const.Declare tensor<1x1x1xf32> = dense<-1.270000e+02>
    %FQ_IN_HI = const.Declare tensor<1x1x1xf32> = dense<1.270000e+02> : tensor<1x1x1xf32>
    // CHECK-DAG:   [[FQ_IN_HI:%.+]] = const.Declare tensor<1x1x1xf32> = dense<1.270000e+02>

    %SCALE_0 = const.Declare tensor<16xf32> = dense<3.0> : tensor<16xf32>
    // CHECK-DAG:   [[SCALE_0:%.+]] = const.Declare tensor<16xf32> = dense<3.000000e+00> : tensor<16xf32>

    %SCALE_1 = const.Declare tensor<16xf32> = dense<2.0> : tensor<16xf32>
    // CHECK-DAG:   [[SCALE_1:%.+]] = const.Declare tensor<16xf32> = dense<2.000000e+00> : tensor<16xf32>


    %FQ_0 = IE.FakeQuantize(%WEIGHTS_0, %FQ_IN_LO, %FQ_IN_HI, %FQ_OUT_LO_0, %FQ_OUT_HI_0) {
          auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
          levels = 255 : i64
    } : tensor<1x32x16xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<1x1x16xf32>, tensor<1x1x16xf32> -> tensor<1x32x16xf32>
    // CHECK:   [[FQ_0:%.+]] = IE.FakeQuantize([[WEIGHTS_0]], [[FQ_IN_LO]], [[FQ_IN_HI]], [[FQ_OUT_LO_0]], [[FQ_OUT_HI_0]])

    %FQ_1 = IE.FakeQuantize(%WEIGHTS_1, %FQ_IN_LO, %FQ_IN_HI, %FQ_OUT_LO_1, %FQ_OUT_HI_1) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 255 : i64
    } : tensor<1x32x16xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<1x1x16xf32>, tensor<1x1x16xf32> -> tensor<1x32x16xf32>
    // CHECK:   [[FQ_1:%.+]] = IE.FakeQuantize([[WEIGHTS_1]], [[FQ_IN_LO]], [[FQ_IN_HI]], [[FQ_OUT_LO_1]], [[FQ_OUT_HI_1]])

    %RESHAPE_FQ_0 = IE.Reshape(%FQ_0) {
        shape_value = [32, 16]
    } : tensor<1x32x16xf32> -> tensor<32x16xf32>
    // CHECK:   [[RESHAPE_FQ_0:%.+]] = IE.Reshape([[FQ_0]]) {
    // CHECK-SAME:      shape_value = [32, 16]
    // CHECK-SAME:  } : tensor<1x32x16xf32> -> tensor<32x16xf32>

    %RESHAPE_FQ_1 = IE.Reshape(%FQ_1) {
        shape_value = [32, 16]
    } : tensor<1x32x16xf32> -> tensor<32x16xf32>
    // CHECK:   [[RESHAPE_FQ_1:%.+]] = IE.Reshape([[FQ_1]]) {
    // CHECK-SAME:      shape_value = [32, 16]
    // CHECK-SAME:  } : tensor<1x32x16xf32> -> tensor<32x16xf32>

    %SLICE_0 = IE.Slice %arg0 [0, 0] [3, 32] : tensor<3x64xf32> to tensor<3x32xf32>
    // CHECK:   [[SLICE_0:%.+]] = IE.Slice [[BLOCK_ARG]] [0, 0] [3, 32] : tensor<3x64xf32> to tensor<3x32xf32>

    %SLICE_1 = IE.Slice %arg0 [0, 32] [3, 32] : tensor<3x64xf32> to tensor<3x32xf32>
    // CHECK:   [[SLICE_1:%.+]] = IE.Slice [[BLOCK_ARG]] [0, 32] [3, 32] : tensor<3x64xf32> to tensor<3x32xf32>

    %TRANSPOSE_0 = IE.Transpose(%RESHAPE_FQ_0) {
        order_value = #CN
    } : tensor<32x16xf32> -> tensor<16x32xf32>
    // CHECK:   [[TRANSPOSE_0:%.+]] = IE.Transpose([[RESHAPE_FQ_0]])

    %MATMUL_0 = IE.FullyConnected(%SLICE_0, %TRANSPOSE_0) : tensor<3x32xf32>, tensor<16x32xf32> -> tensor<3x16xf32>
    // CHECK:   [[MATMUL_0:%.+]] = IE.FullyConnected([[SLICE_0]], [[TRANSPOSE_0]]) : tensor<3x32xf32>, tensor<16x32xf32> -> tensor<3x16xf32>

    %TRANSPOSE_1 = IE.Transpose(%RESHAPE_FQ_1) {
        order_value = #CN
    } : tensor<32x16xf32> -> tensor<16x32xf32>
    // CHECK:   [[TRANSPOSE_1:%.+]] = IE.Transpose([[RESHAPE_FQ_1]])

    %MATMUL_1 = IE.FullyConnected(%SLICE_1, %TRANSPOSE_1) : tensor<3x32xf32>, tensor<16x32xf32> -> tensor<3x16xf32>
    // CHECK:   [[MATMUL_1:%.+]] = IE.FullyConnected([[SLICE_1]], [[TRANSPOSE_1]]) : tensor<3x32xf32>, tensor<16x32xf32> -> tensor<3x16xf32>

    %ACCUMULATE = IE.Accumulate(%MATMUL_0, %MATMUL_1, %SCALE_0, %SCALE_1) {
        operandSegmentSizes = array<i32: 1, 1, 1, 1>
    } : tensor<3x16xf32>, tensor<3x16xf32>, tensor<16xf32>, tensor<16xf32> -> tensor<3x16xf32>
    // CHECK:   [[ACCUMULATE:%.+]] = IE.Accumulate([[MATMUL_0]], [[MATMUL_1]], [[SCALE_0]], [[SCALE_1]]) {
    // CHECK-SAME:      operandSegmentSizes = array<i32: 1, 1, 1, 1>
    // CHECK-SAME:  } : tensor<3x16xf32>, tensor<3x16xf32>, tensor<16xf32>, tensor<16xf32> -> tensor<3x16xf32>

    return %ACCUMULATE : tensor<3x16xf32>
    // CHECK:   [[ACCUMULATE]] : tensor<3x16xf32>
}

// -----

// CHECK:   @SkipAccumulateWithBlockArg
// CHECK-SAME:  [[BLOCK_ARG_0:%arg0]]: tensor<3x16xf32>
// CHECK-SAME:  [[BLOCK_ARG_1:%arg1]]: tensor<3x16xf32>
func.func @SkipAccumulateWithBlockArg(%arg0: tensor<3x16xf32>, %arg1: tensor<3x16xf32>) -> tensor<3x16xf32> {
    %ACCUMULATE = IE.Accumulate(%arg0, %arg1) {
        operandSegmentSizes = array<i32: 1, 1, 0, 0>
    } : tensor<3x16xf32>, tensor<3x16xf32> -> tensor<3x16xf32>
    // CHECK:   [[ACCUMULATE:%.+]] = IE.Accumulate([[BLOCK_ARG_0]], [[BLOCK_ARG_1]]) {
    // CHECK-SAME:      operandSegmentSizes = array<i32: 1, 1, 0, 0>
    // CHECK-SAME:  } : tensor<3x16xf32>, tensor<3x16xf32> -> tensor<3x16xf32>

    return %ACCUMULATE : tensor<3x16xf32>
    // CHECK:   [[ACCUMULATE]] : tensor<3x16xf32>
}

// -----

#CN = affine_map<(d0, d1) -> (d1, d0)>

// CHECK:   @SkipAccumulateWithScales
// CHECK-SAME:  [[BLOCK_ARG_0:%arg0]]: tensor<3x64xf32>
func.func @SkipAccumulateWithScales(%arg0: tensor<3x64xf32>) -> tensor<3x16xf32> {
    %WEIGHTS_0 = const.Declare tensor<1x16x32xf32> = dense<1.125> : tensor<1x16x32xf32>
    // CHECK-DAG:   [[WEIGHTS_0:%.+]] = const.Declare tensor<1x16x32xf32> = dense<1.125000e+00>
    %WEIGHTS_1 = const.Declare tensor<1x16x32xf32> = dense<2.0625> : tensor<1x16x32xf32>
    // CHECK-DAG:   [[WEIGHTS_1:%.+]] = const.Declare tensor<1x16x32xf32> = dense<2.062500e+00>

    %FQ_OUT_LO_0 = const.Declare tensor<1x16x1xf32> = dense<-256.0> : tensor<1x16x1xf32>
    // CHECK-DAG:   [[FQ_OUT_LO_0:%.+]] = const.Declare tensor<1x16x1xf32> = dense<-2.560000e+02> : tensor<1x16x1xf32>

    %FQ_OUT_LO_1 = const.Declare tensor<1x16x1xf32> = dense<-384.0> : tensor<1x16x1xf32>
    // CHECK-DAG:   [[FQ_OUT_LO_1:%.+]] = const.Declare tensor<1x16x1xf32> = dense<-3.840000e+02> : tensor<1x16x1xf32>

    %FQ_OUT_HI_0 = const.Declare tensor<1x16x1xf32> = dense<256.0> : tensor<1x16x1xf32>
    // CHECK-DAG:   [[FQ_OUT_HI_0:%.+]] = const.Declare tensor<1x16x1xf32> = dense<2.560000e+02> : tensor<1x16x1xf32>

    %FQ_OUT_HI_1 = const.Declare tensor<1x16x1xf32> = dense<384.0> : tensor<1x16x1xf32>
    // CHECK-DAG:   [[FQ_OUT_HI_1:%.+]] = const.Declare tensor<1x16x1xf32> = dense<3.840000e+02> : tensor<1x16x1xf32>

    %FQ_IN_LO = const.Declare tensor<1x1x1xf32> = dense<-1.270000e+02> : tensor<1x1x1xf32>
    // CHECK-DAG:   [[FQ_IN_LO:%.+]] = const.Declare tensor<1x1x1xf32> = dense<-1.270000e+02>
    %FQ_IN_HI = const.Declare tensor<1x1x1xf32> = dense<1.270000e+02> : tensor<1x1x1xf32>
    // CHECK-DAG:   [[FQ_IN_HI:%.+]] = const.Declare tensor<1x1x1xf32> = dense<1.270000e+02>

    %SCALE_0 = const.Declare tensor<16xf32> = dense<3.0> : tensor<16xf32>
    // CHECK-DAG:   [[SCALE_0:%.+]] = const.Declare tensor<16xf32> = dense<3.000000e+00> : tensor<16xf32>

    %SCALE_1 = const.Declare tensor<16xf32> = dense<2.0> : tensor<16xf32>
    // CHECK-DAG:   [[SCALE_1:%.+]] = const.Declare tensor<16xf32> = dense<2.000000e+00> : tensor<16xf32>


    %FQ_0 = IE.FakeQuantize(%WEIGHTS_0, %FQ_IN_LO, %FQ_IN_HI, %FQ_OUT_LO_0, %FQ_OUT_HI_0) {
          auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
          levels = 255 : i64
    } : tensor<1x16x32xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<1x16x1xf32>, tensor<1x16x1xf32> -> tensor<1x16x32xf32>
    // CHECK:   [[FQ_0:%.+]] = IE.FakeQuantize([[WEIGHTS_0]], [[FQ_IN_LO]], [[FQ_IN_HI]], [[FQ_OUT_LO_0]], [[FQ_OUT_HI_0]])

    %FQ_1 = IE.FakeQuantize(%WEIGHTS_1, %FQ_IN_LO, %FQ_IN_HI, %FQ_OUT_LO_1, %FQ_OUT_HI_1) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 255 : i64
    } : tensor<1x16x32xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<1x16x1xf32>, tensor<1x16x1xf32> -> tensor<1x16x32xf32>
    // CHECK:   [[FQ_1:%.+]] = IE.FakeQuantize([[WEIGHTS_1]], [[FQ_IN_LO]], [[FQ_IN_HI]], [[FQ_OUT_LO_1]], [[FQ_OUT_HI_1]])

    %RESHAPE_FQ_0 = IE.Reshape(%FQ_0) {
        shape_value = [16, 32]
    } : tensor<1x16x32xf32> -> tensor<16x32xf32>
    // CHECK:   [[RESHAPE_FQ_0:%.+]] = IE.Reshape([[FQ_0]]) {
    // CHECK-SAME:      shape_value = [16, 32]
    // CHECK-SAME:  } : tensor<1x16x32xf32> -> tensor<16x32xf32>

    %RESHAPE_FQ_1 = IE.Reshape(%FQ_1) {
        shape_value = [16, 32]
    } : tensor<1x16x32xf32> -> tensor<16x32xf32>
    // CHECK:   [[RESHAPE_FQ_1:%.+]] = IE.Reshape([[FQ_1]]) {
    // CHECK-SAME:      shape_value = [16, 32]
    // CHECK-SAME:  } : tensor<1x16x32xf32> -> tensor<16x32xf32>

    %SLICE_0 = IE.Slice %arg0 [0, 0] [3, 32] : tensor<3x64xf32> to tensor<3x32xf32>
    // CHECK:   [[SLICE_0:%.+]] = IE.Slice [[BLOCK_ARG]] [0, 0] [3, 32] : tensor<3x64xf32> to tensor<3x32xf32>

    %SLICE_1 = IE.Slice %arg0 [0, 32] [3, 32] : tensor<3x64xf32> to tensor<3x32xf32>
    // CHECK:   [[SLICE_1:%.+]] = IE.Slice [[BLOCK_ARG]] [0, 32] [3, 32] : tensor<3x64xf32> to tensor<3x32xf32>

    %MATMUL_0 = IE.FullyConnected(%SLICE_0, %RESHAPE_FQ_0) : tensor<3x32xf32>, tensor<16x32xf32> -> tensor<3x16xf32>
    // CHECK:   [[MATMUL_0:%.+]] = IE.FullyConnected([[SLICE_0]], [[RESHAPE_FQ_0]])

    %MATMUL_1 = IE.FullyConnected(%SLICE_1, %RESHAPE_FQ_1) : tensor<3x32xf32>, tensor<16x32xf32> -> tensor<3x16xf32>
    // CHECK:   [[MATMUL_1:%.+]] = IE.FullyConnected([[SLICE_1]], [[RESHAPE_FQ_1]])

    %ACCUMULATE = IE.Accumulate(%MATMUL_0, %MATMUL_1, %SCALE_0, %SCALE_1) {
        operandSegmentSizes = array<i32: 1, 1, 1, 1>
    } : tensor<3x16xf32>, tensor<3x16xf32>, tensor<16xf32>, tensor<16xf32> -> tensor<3x16xf32>
    // CHECK:   [[ACCUMULATE:%.+]] = IE.Accumulate([[MATMUL_0]], [[MATMUL_1]], [[SCALE_0]], [[SCALE_1]]) {
    // CHECK-SAME:      operandSegmentSizes = array<i32: 1, 1, 1, 1>
    // CHECK-SAME:  } : tensor<3x16xf32>, tensor<3x16xf32>, tensor<16xf32>, tensor<16xf32> -> tensor<3x16xf32>

    return %ACCUMULATE : tensor<3x16xf32>
    // CHECK:   [[ACCUMULATE]] : tensor<3x16xf32>
}

// -----

// CHECK:   @SkipAccumulateWithDifferentShapes
// CHECK-SAME:  [[BLOCK_ARG:%arg0]]: tensor<3x64xf32>
func.func @SkipAccumulateWithDifferentShapes(%arg0: tensor<3x64xf32>) -> tensor<3x16xf32> {
    %WEIGHTS_0 = const.Declare tensor<1x32x16xf32> = dense<1.125> : tensor<1x32x16xf32>
    // CHECK-DAG:   [[WEIGHTS_0:%.+]] = const.Declare tensor<1x32x16xf32> = dense<1.125000e+00>
    %WEIGHTS_1 = const.Declare tensor<1x32x16xf32> = dense<2.0625> : tensor<1x32x16xf32>
    // CHECK-DAG:   [[WEIGHTS_1:%.+]] = const.Declare tensor<1x32x16xf32> = dense<2.062500e+00>

    %FQ_OUT_LO_0 = const.Declare tensor<1x1x16xf32> = dense<-256.0> : tensor<1x1x16xf32>
    // CHECK-DAG:   [[FQ_OUT_LO_0:%.+]] = const.Declare tensor<1x1x16xf32> = dense<-2.560000e+02> : tensor<1x1x16xf32>

    %FQ_OUT_LO_1 = const.Declare tensor<1x1x16xf32> = dense<-384.0> : tensor<1x1x16xf32>
    // CHECK-DAG:   [[FQ_OUT_LO_1:%.+]] = const.Declare tensor<1x1x16xf32> = dense<-3.840000e+02> : tensor<1x1x16xf32>

    %FQ_OUT_HI_0 = const.Declare tensor<1x1x16xf32> = dense<256.0> : tensor<1x1x16xf32>
    // CHECK-DAG:   [[FQ_OUT_HI_0:%.+]] = const.Declare tensor<1x1x16xf32> = dense<2.560000e+02> : tensor<1x1x16xf32>

    %FQ_OUT_HI_1 = const.Declare tensor<1x1x16xf32> = dense<384.0> : tensor<1x1x16xf32>
    // CHECK-DAG:   [[FQ_OUT_HI_1:%.+]] = const.Declare tensor<1x1x16xf32> = dense<3.840000e+02> : tensor<1x1x16xf32>

    %FQ_IN_LO = const.Declare tensor<1x1x1xf32> = dense<-1.270000e+02> : tensor<1x1x1xf32>
    // CHECK-DAG:   [[FQ_IN_LO:%.+]] = const.Declare tensor<1x1x1xf32> = dense<-1.270000e+02>
    %FQ_IN_HI = const.Declare tensor<1x1x1xf32> = dense<1.270000e+02> : tensor<1x1x1xf32>
    // CHECK-DAG:   [[FQ_IN_HI:%.+]] = const.Declare tensor<1x1x1xf32> = dense<1.270000e+02>

    %FQ_0 = IE.FakeQuantize(%WEIGHTS_0, %FQ_IN_LO, %FQ_IN_HI, %FQ_OUT_LO_0, %FQ_OUT_HI_0) {
          auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
          levels = 255 : i64
    } : tensor<1x32x16xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<1x1x16xf32>, tensor<1x1x16xf32> -> tensor<1x32x16xf32>
    // CHECK:   [[FQ_0:%.+]] = IE.FakeQuantize([[WEIGHTS_0]], [[FQ_IN_LO]], [[FQ_IN_HI]], [[FQ_OUT_LO_0]], [[FQ_OUT_HI_0]])

    %FQ_1 = IE.FakeQuantize(%WEIGHTS_1, %FQ_IN_LO, %FQ_IN_HI, %FQ_OUT_LO_1, %FQ_OUT_HI_1) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 255 : i64
    } : tensor<1x32x16xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<1x1x16xf32>, tensor<1x1x16xf32> -> tensor<1x32x16xf32>
    // CHECK:   [[FQ_1:%.+]] = IE.FakeQuantize([[WEIGHTS_1]], [[FQ_IN_LO]], [[FQ_IN_HI]], [[FQ_OUT_LO_1]], [[FQ_OUT_HI_1]])

    %RESHAPE_FQ_0 = IE.Reshape(%FQ_0) {
        shape_value = [16, 32]
    } : tensor<1x32x16xf32> -> tensor<16x32xf32>
    // CHECK:   [[RESHAPE_FQ_0:%.+]] = IE.Reshape([[FQ_0]]) {
    // CHECK-SAME:      shape_value = [16, 32]
    // CHECK-SAME:  } : tensor<1x32x16xf32> -> tensor<16x32xf32>

    %RESHAPE_FQ_1 = IE.Reshape(%FQ_1) {
        shape_value = [16, 32]
    } : tensor<1x32x16xf32> -> tensor<16x32xf32>
    // CHECK:   [[RESHAPE_FQ_1:%.+]] = IE.Reshape([[FQ_1]]) {
    // CHECK-SAME:      shape_value = [16, 32]
    // CHECK-SAME:  } : tensor<1x32x16xf32> -> tensor<16x32xf32>

    %SLICE_0 = IE.Slice %arg0 [0, 0] [3, 32] : tensor<3x64xf32> to tensor<3x32xf32>
    // CHECK:   [[SLICE_0:%.+]] = IE.Slice [[BLOCK_ARG]] [0, 0] [3, 32] : tensor<3x64xf32> to tensor<3x32xf32>

    %SLICE_1 = IE.Slice %arg0 [0, 32] [3, 32] : tensor<3x64xf32> to tensor<3x32xf32>
    // CHECK:   [[SLICE_1:%.+]] = IE.Slice [[BLOCK_ARG]] [0, 32] [3, 32] : tensor<3x64xf32> to tensor<3x32xf32>

    %MATMUL_0 = IE.FullyConnected(%SLICE_0, %RESHAPE_FQ_0) : tensor<3x32xf32>, tensor<16x32xf32> -> tensor<3x16xf32>
    // CHECK:   [[MATMUL_0:%.+]] = IE.FullyConnected([[SLICE_0]], [[RESHAPE_FQ_0]]) : tensor<3x32xf32>, tensor<16x32xf32> -> tensor<3x16xf32>

    %MATMUL_1 = IE.FullyConnected(%SLICE_1, %RESHAPE_FQ_1) : tensor<3x32xf32>, tensor<16x32xf32> -> tensor<3x16xf32>
    // CHECK:   [[MATMUL_1:%.+]] = IE.FullyConnected([[SLICE_1]], [[RESHAPE_FQ_1]]) : tensor<3x32xf32>, tensor<16x32xf32> -> tensor<3x16xf32>

    %ACCUMULATE = IE.Accumulate(%MATMUL_0, %MATMUL_1) {
        operandSegmentSizes = array<i32: 1, 1, 0, 0>
    } : tensor<3x16xf32>, tensor<3x16xf32> -> tensor<3x16xf32>
    // CHECK:   [[ACCUMULATE:%.+]] = IE.Accumulate([[MATMUL_0]], [[MATMUL_1]]) {
    // CHECK-SAME:      operandSegmentSizes = array<i32: 1, 1, 0, 0>
    // CHECK-SAME:  } : tensor<3x16xf32>, tensor<3x16xf32> -> tensor<3x16xf32>

    return %ACCUMULATE : tensor<3x16xf32>
    // CHECK:   [[ACCUMULATE]] : tensor<3x16xf32>
}

// -----

#CN = affine_map<(d0, d1) -> (d1, d0)>

// CHECK:   @SkipAccumulateWithFQOverHeight
// CHECK-SAME:  [[BLOCK_ARG:%arg0]]: tensor<3x64xf32>
func.func @SkipAccumulateWithFQOverHeight(%arg0: tensor<3x64xf32>) -> tensor<3x16xf32> {
    %WEIGHTS_0 = const.Declare tensor<1x32x16xf32> = dense<1.125> : tensor<1x32x16xf32>
    // CHECK-DAG:   [[WEIGHTS_0:%.+]] = const.Declare tensor<1x32x16xf32> = dense<1.125000e+00>
    %WEIGHTS_1 = const.Declare tensor<1x32x16xf32> = dense<2.0625> : tensor<1x32x16xf32>
    // CHECK-DAG:   [[WEIGHTS_1:%.+]] = const.Declare tensor<1x32x16xf32> = dense<2.062500e+00>

    %FQ_OUT_LO_0 = const.Declare tensor<1x32x1xf32> = dense<-256.0> : tensor<1x32x1xf32>
    // CHECK-DAG:   [[FQ_OUT_LO_0:%.+]] = const.Declare tensor<1x32x1xf32> = dense<-2.560000e+02> : tensor<1x32x1xf32>

    %FQ_OUT_LO_1 = const.Declare tensor<1x32x1xf32> = dense<-384.0> : tensor<1x32x1xf32>
    // CHECK-DAG:   [[FQ_OUT_LO_1:%.+]] = const.Declare tensor<1x32x1xf32> = dense<-3.840000e+02> : tensor<1x32x1xf32>

    %FQ_OUT_HI_0 = const.Declare tensor<1x32x1xf32> = dense<256.0> : tensor<1x32x1xf32>
    // CHECK-DAG:   [[FQ_OUT_HI_0:%.+]] = const.Declare tensor<1x32x1xf32> = dense<2.560000e+02> : tensor<1x32x1xf32>

    %FQ_OUT_HI_1 = const.Declare tensor<1x32x1xf32> = dense<384.0> : tensor<1x32x1xf32>
    // CHECK-DAG:   [[FQ_OUT_HI_1:%.+]] = const.Declare tensor<1x32x1xf32> = dense<3.840000e+02> : tensor<1x32x1xf32>

    %FQ_IN_LO = const.Declare tensor<1x1x1xf32> = dense<-1.270000e+02> : tensor<1x1x1xf32>
    // CHECK-DAG:   [[FQ_IN_LO:%.+]] = const.Declare tensor<1x1x1xf32> = dense<-1.270000e+02>
    %FQ_IN_HI = const.Declare tensor<1x1x1xf32> = dense<1.270000e+02> : tensor<1x1x1xf32>
    // CHECK-DAG:   [[FQ_IN_HI:%.+]] = const.Declare tensor<1x1x1xf32> = dense<1.270000e+02>

    %FQ_0 = IE.FakeQuantize(%WEIGHTS_0, %FQ_IN_LO, %FQ_IN_HI, %FQ_OUT_LO_0, %FQ_OUT_HI_0) {
          auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
          levels = 255 : i64
    } : tensor<1x32x16xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<1x32x1xf32>, tensor<1x32x1xf32> -> tensor<1x32x16xf32>
    // CHECK:   [[FQ_0:%.+]] = IE.FakeQuantize([[WEIGHTS_0]], [[FQ_IN_LO]], [[FQ_IN_HI]], [[FQ_OUT_LO_0]], [[FQ_OUT_HI_0]])

    %FQ_1 = IE.FakeQuantize(%WEIGHTS_1, %FQ_IN_LO, %FQ_IN_HI, %FQ_OUT_LO_1, %FQ_OUT_HI_1) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 255 : i64
    } : tensor<1x32x16xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<1x32x1xf32>, tensor<1x32x1xf32> -> tensor<1x32x16xf32>
    // CHECK:   [[FQ_1:%.+]] = IE.FakeQuantize([[WEIGHTS_1]], [[FQ_IN_LO]], [[FQ_IN_HI]], [[FQ_OUT_LO_1]], [[FQ_OUT_HI_1]])

    %RESHAPE_FQ_0 = IE.Reshape(%FQ_0) {
        shape_value = [32, 16]
    } : tensor<1x32x16xf32> -> tensor<32x16xf32>
    // CHECK:   [[RESHAPE_FQ_0:%.+]] = IE.Reshape([[FQ_0]]) {
    // CHECK-SAME:      shape_value = [32, 16]
    // CHECK-SAME:  } : tensor<1x32x16xf32> -> tensor<32x16xf32>

    %RESHAPE_FQ_1 = IE.Reshape(%FQ_1) {
        shape_value = [32, 16]
    } : tensor<1x32x16xf32> -> tensor<32x16xf32>
    // CHECK:   [[RESHAPE_FQ_1:%.+]] = IE.Reshape([[FQ_1]]) {
    // CHECK-SAME:      shape_value = [32, 16]
    // CHECK-SAME:  } : tensor<1x32x16xf32> -> tensor<32x16xf32>

    %SLICE_0 = IE.Slice %arg0 [0, 0] [3, 32] : tensor<3x64xf32> to tensor<3x32xf32>
    // CHECK:   [[SLICE_0:%.+]] = IE.Slice [[BLOCK_ARG]] [0, 0] [3, 32] : tensor<3x64xf32> to tensor<3x32xf32>

    %SLICE_1 = IE.Slice %arg0 [0, 32] [3, 32] : tensor<3x64xf32> to tensor<3x32xf32>
    // CHECK:   [[SLICE_1:%.+]] = IE.Slice [[BLOCK_ARG]] [0, 32] [3, 32] : tensor<3x64xf32> to tensor<3x32xf32>

    %TRANSPOSE_0 = IE.Transpose(%RESHAPE_FQ_0) {
        order_value = #CN
    } : tensor<32x16xf32> -> tensor<16x32xf32>
    // CHECK:   [[TRANSPOSE_0:%.+]] = IE.Transpose([[RESHAPE_FQ_0]])

    %MATMUL_0 = IE.FullyConnected(%SLICE_0, %TRANSPOSE_0) : tensor<3x32xf32>, tensor<16x32xf32> -> tensor<3x16xf32>
    // CHECK:   [[MATMUL_0:%.+]] = IE.FullyConnected([[SLICE_0]], [[TRANSPOSE_0]]) : tensor<3x32xf32>, tensor<16x32xf32> -> tensor<3x16xf32>

    %TRANSPOSE_1 = IE.Transpose(%RESHAPE_FQ_1) {
        order_value = #CN
    } : tensor<32x16xf32> -> tensor<16x32xf32>
    // CHECK:   [[TRANSPOSE_1:%.+]] = IE.Transpose([[RESHAPE_FQ_1]])

    %MATMUL_1 = IE.FullyConnected(%SLICE_1, %TRANSPOSE_1) : tensor<3x32xf32>, tensor<16x32xf32> -> tensor<3x16xf32>
    // CHECK:   [[MATMUL_1:%.+]] = IE.FullyConnected([[SLICE_1]], [[TRANSPOSE_1]]) : tensor<3x32xf32>, tensor<16x32xf32> -> tensor<3x16xf32>

    %ACCUMULATE = IE.Accumulate(%MATMUL_0, %MATMUL_1) {
        operandSegmentSizes = array<i32: 1, 1, 0, 0>
    } : tensor<3x16xf32>, tensor<3x16xf32> -> tensor<3x16xf32>
    // CHECK:   [[ACCUMULATE:%.+]] = IE.Accumulate([[MATMUL_0]], [[MATMUL_1]]) {
    // CHECK-SAME:      operandSegmentSizes = array<i32: 1, 1, 0, 0>
    // CHECK-SAME:  } : tensor<3x16xf32>, tensor<3x16xf32> -> tensor<3x16xf32>

    return %ACCUMULATE : tensor<3x16xf32>
    // CHECK:   [[ACCUMULATE]] : tensor<3x16xf32>
}

// -----

#CN = affine_map<(d0, d1) -> (d1, d0)>

// CHECK:   @SkipAccumulateWithBlockArgumentReshape
// CHECK-SAME:  [[BLOCK_ARG_0:%arg0]]: tensor<3x64xf32>
// CHECK-SAME:  [[BLOCK_ARG_1:%arg1]]: tensor<1x32x16xf32>
func.func @SkipAccumulateWithBlockArgumentReshape(%arg0: tensor<3x64xf32>, %arg1: tensor<1x32x16xf32>) -> tensor<3x16xf32> {
    %RESHAPE_FQ_0 = IE.Reshape(%arg1) {
        shape_value = [32, 16]
    } : tensor<1x32x16xf32> -> tensor<32x16xf32>
    // CHECK:   [[RESHAPE_FQ_0:%.+]] = IE.Reshape([[BLOCK_ARG_1]]) {
    // CHECK-SAME:      shape_value = [32, 16]
    // CHECK-SAME:  } : tensor<1x32x16xf32> -> tensor<32x16xf32>

    %RESHAPE_FQ_1 = IE.Reshape(%arg1) {
        shape_value = [32, 16]
    } : tensor<1x32x16xf32> -> tensor<32x16xf32>
    // CHECK:   [[RESHAPE_FQ_1:%.+]] = IE.Reshape([[BLOCK_ARG_1]]) {
    // CHECK-SAME:      shape_value = [32, 16]
    // CHECK-SAME:  } : tensor<1x32x16xf32> -> tensor<32x16xf32>

    %SLICE_0 = IE.Slice %arg0 [0, 0] [3, 32] : tensor<3x64xf32> to tensor<3x32xf32>
    // CHECK:   [[SLICE_0:%.+]] = IE.Slice [[BLOCK_ARG]] [0, 0] [3, 32] : tensor<3x64xf32> to tensor<3x32xf32>

    %SLICE_1 = IE.Slice %arg0 [0, 32] [3, 32] : tensor<3x64xf32> to tensor<3x32xf32>
    // CHECK:   [[SLICE_1:%.+]] = IE.Slice [[BLOCK_ARG]] [0, 32] [3, 32] : tensor<3x64xf32> to tensor<3x32xf32>

    %TRANSPOSE_0 = IE.Transpose(%RESHAPE_FQ_0) {
        order_value = #CN
    } : tensor<32x16xf32> -> tensor<16x32xf32>
    // CHECK:   [[TRANSPOSE_0:%.+]] = IE.Transpose([[RESHAPE_FQ_0]])

    %MATMUL_0 = IE.FullyConnected(%SLICE_0, %TRANSPOSE_0) : tensor<3x32xf32>, tensor<16x32xf32> -> tensor<3x16xf32>
    // CHECK:   [[MATMUL_0:%.+]] = IE.FullyConnected([[SLICE_0]], [[TRANSPOSE_0]]) : tensor<3x32xf32>, tensor<16x32xf32> -> tensor<3x16xf32>

    %TRANSPOSE_1 = IE.Transpose(%RESHAPE_FQ_1) {
        order_value = #CN
    } : tensor<32x16xf32> -> tensor<16x32xf32>
    // CHECK:   [[TRANSPOSE_1:%.+]] = IE.Transpose([[RESHAPE_FQ_1]])

    %MATMUL_1 = IE.FullyConnected(%SLICE_1, %TRANSPOSE_1) : tensor<3x32xf32>, tensor<16x32xf32> -> tensor<3x16xf32>
    // CHECK:   [[MATMUL_1:%.+]] = IE.FullyConnected([[SLICE_1]], [[TRANSPOSE_1]]) : tensor<3x32xf32>, tensor<16x32xf32> -> tensor<3x16xf32>

    %ACCUMULATE = IE.Accumulate(%MATMUL_0, %MATMUL_1) {
        operandSegmentSizes = array<i32: 1, 1, 0, 0>
    } : tensor<3x16xf32>, tensor<3x16xf32> -> tensor<3x16xf32>
    // CHECK:   [[ACCUMULATE:%.+]] = IE.Accumulate([[MATMUL_0]], [[MATMUL_1]]) {
    // CHECK-SAME:      operandSegmentSizes = array<i32: 1, 1, 0, 0>
    // CHECK-SAME:  } : tensor<3x16xf32>, tensor<3x16xf32> -> tensor<3x16xf32>

    return %ACCUMULATE : tensor<3x16xf32>
    // CHECK:   [[ACCUMULATE]] : tensor<3x16xf32>
}

// -----

#CN = affine_map<(d0, d1) -> (d1, d0)>

// CHECK:   @SkipAccumulateWithMultipleZeroPoints
// CHECK-SAME:  [[BLOCK_ARG:%arg0]]: tensor<3x64xf32>
func.func @SkipAccumulateWithMultipleZeroPoints(%arg0: tensor<3x64xf32>) -> tensor<3x16xf32> {
    %WEIGHTS_0 = const.Declare tensor<1x32x16xf32> = dense<1.125> : tensor<1x32x16xf32>
    // CHECK-DAG:   [[WEIGHTS_0:%.+]] = const.Declare tensor<1x32x16xf32> = dense<1.125000e+00>
    %WEIGHTS_1 = const.Declare tensor<1x32x16xf32> = dense<2.0625> : tensor<1x32x16xf32>
    // CHECK-DAG:   [[WEIGHTS_1:%.+]] = const.Declare tensor<1x32x16xf32> = dense<2.062500e+00>

    %FQ_OUT_LO_0 = const.Declare tensor<1x1x16xf32> = dense<[[[
        -256.0, -254.0, -252.0, -250.0,
        -256.0, -254.0, -252.0, -250.0,
        -256.0, -254.0, -252.0, -250.0,
        -256.0, -254.0, -252.0, -250.0
    ]]]> : tensor<1x1x16xf32>
    // CHECK-DAG:   [[FQ_OUT_LO_0:%.+]] = const.Declare tensor<1x1x16xf32> = dense<
    // CHECK-DAG-SAME:      -2.560000e+02, -2.540000e+02, -2.520000e+02, -2.500000e+02,
    // CHECK-DAG-SAME:      -2.560000e+02, -2.540000e+02, -2.520000e+02, -2.500000e+02,
    // CHECK-DAG-SAME:      -2.560000e+02, -2.540000e+02, -2.520000e+02, -2.500000e+02,
    // CHECK-DAG-SAME:      -2.560000e+02, -2.540000e+02, -2.520000e+02, -2.500000e+02
    // CHECK-DAG-SAME:  > : tensor<1x1x16xf32>

    %FQ_OUT_LO_1 = const.Declare tensor<1x1x16xf32> = dense<-384.0> : tensor<1x1x16xf32>
    // CHECK-DAG:   [[FQ_OUT_LO_1:%.+]] = const.Declare tensor<1x1x16xf32> = dense<-3.840000e+02> : tensor<1x1x16xf32>

    %FQ_OUT_HI_0 = const.Declare tensor<1x1x16xf32> = dense<256.0> : tensor<1x1x16xf32>
    // CHECK-DAG:   [[FQ_OUT_HI_0:%.+]] = const.Declare tensor<1x1x16xf32> = dense<2.560000e+02> : tensor<1x1x16xf32>

    %FQ_OUT_HI_1 = const.Declare tensor<1x1x16xf32> = dense<384.0> : tensor<1x1x16xf32>
    // CHECK-DAG:   [[FQ_OUT_HI_1:%.+]] = const.Declare tensor<1x1x16xf32> = dense<3.840000e+02> : tensor<1x1x16xf32>

    %FQ_IN_LO = const.Declare tensor<1x1x1xf32> = dense<-1.270000e+02> : tensor<1x1x1xf32>
    // CHECK-DAG:   [[FQ_IN_LO:%.+]] = const.Declare tensor<1x1x1xf32> = dense<-1.270000e+02>
    %FQ_IN_HI = const.Declare tensor<1x1x1xf32> = dense<1.270000e+02> : tensor<1x1x1xf32>
    // CHECK-DAG:   [[FQ_IN_HI:%.+]] = const.Declare tensor<1x1x1xf32> = dense<1.270000e+02>

    %FQ_0 = IE.FakeQuantize(%WEIGHTS_0, %FQ_IN_LO, %FQ_IN_HI, %FQ_OUT_LO_0, %FQ_OUT_HI_0) {
          auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
          levels = 255 : i64
    } : tensor<1x32x16xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<1x1x16xf32>, tensor<1x1x16xf32> -> tensor<1x32x16xf32>
    // CHECK:   [[FQ_0:%.+]] = IE.FakeQuantize([[WEIGHTS_0]], [[FQ_IN_LO]], [[FQ_IN_HI]], [[FQ_OUT_LO_0]], [[FQ_OUT_HI_0]])

    %FQ_1 = IE.FakeQuantize(%WEIGHTS_1, %FQ_IN_LO, %FQ_IN_HI, %FQ_OUT_LO_1, %FQ_OUT_HI_1) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 255 : i64
    } : tensor<1x32x16xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<1x1x16xf32>, tensor<1x1x16xf32> -> tensor<1x32x16xf32>
    // CHECK:   [[FQ_1:%.+]] = IE.FakeQuantize([[WEIGHTS_1]], [[FQ_IN_LO]], [[FQ_IN_HI]], [[FQ_OUT_LO_1]], [[FQ_OUT_HI_1]])

    %RESHAPE_FQ_0 = IE.Reshape(%FQ_0) {
        shape_value = [32, 16]
    } : tensor<1x32x16xf32> -> tensor<32x16xf32>
    // CHECK:   [[RESHAPE_FQ_0:%.+]] = IE.Reshape([[FQ_0]]) {
    // CHECK-SAME:      shape_value = [32, 16]
    // CHECK-SAME:  } : tensor<1x32x16xf32> -> tensor<32x16xf32>

    %RESHAPE_FQ_1 = IE.Reshape(%FQ_1) {
        shape_value = [32, 16]
    } : tensor<1x32x16xf32> -> tensor<32x16xf32>
    // CHECK:   [[RESHAPE_FQ_1:%.+]] = IE.Reshape([[FQ_1]]) {
    // CHECK-SAME:      shape_value = [32, 16]
    // CHECK-SAME:  } : tensor<1x32x16xf32> -> tensor<32x16xf32>

    %SLICE_0 = IE.Slice %arg0 [0, 0] [3, 32] : tensor<3x64xf32> to tensor<3x32xf32>
    // CHECK:   [[SLICE_0:%.+]] = IE.Slice [[BLOCK_ARG]] [0, 0] [3, 32] : tensor<3x64xf32> to tensor<3x32xf32>

    %SLICE_1 = IE.Slice %arg0 [0, 32] [3, 32] : tensor<3x64xf32> to tensor<3x32xf32>
    // CHECK:   [[SLICE_1:%.+]] = IE.Slice [[BLOCK_ARG]] [0, 32] [3, 32] : tensor<3x64xf32> to tensor<3x32xf32>

    %TRANSPOSE_0 = IE.Transpose(%RESHAPE_FQ_0) {
        order_value = #CN
    } : tensor<32x16xf32> -> tensor<16x32xf32>
    // CHECK:   [[TRANSPOSE_0:%.+]] = IE.Transpose([[RESHAPE_FQ_0]])

    %MATMUL_0 = IE.FullyConnected(%SLICE_0, %TRANSPOSE_0) : tensor<3x32xf32>, tensor<16x32xf32> -> tensor<3x16xf32>
    // CHECK:   [[MATMUL_0:%.+]] = IE.FullyConnected([[SLICE_0]], [[TRANSPOSE_0]]) : tensor<3x32xf32>, tensor<16x32xf32> -> tensor<3x16xf32>

    %TRANSPOSE_1 = IE.Transpose(%RESHAPE_FQ_1) {
        order_value = #CN
    } : tensor<32x16xf32> -> tensor<16x32xf32>
    // CHECK:   [[TRANSPOSE_1:%.+]] = IE.Transpose([[RESHAPE_FQ_1]])

    %MATMUL_1 = IE.FullyConnected(%SLICE_1, %TRANSPOSE_1) : tensor<3x32xf32>, tensor<16x32xf32> -> tensor<3x16xf32>
    // CHECK:   [[MATMUL_1:%.+]] = IE.FullyConnected([[SLICE_1]], [[TRANSPOSE_1]]) : tensor<3x32xf32>, tensor<16x32xf32> -> tensor<3x16xf32>

    %ACCUMULATE = IE.Accumulate(%MATMUL_0, %MATMUL_1) {
        operandSegmentSizes = array<i32: 1, 1, 0, 0>
    } : tensor<3x16xf32>, tensor<3x16xf32> -> tensor<3x16xf32>
    // CHECK:   [[ACCUMULATE:%.+]] = IE.Accumulate([[MATMUL_0]], [[MATMUL_1]]) {
    // CHECK-SAME:      operandSegmentSizes = array<i32: 1, 1, 0, 0>
    // CHECK-SAME:  } : tensor<3x16xf32>, tensor<3x16xf32> -> tensor<3x16xf32>

    return %ACCUMULATE : tensor<3x16xf32>
    // CHECK:   [[ACCUMULATE]] : tensor<3x16xf32>
}
