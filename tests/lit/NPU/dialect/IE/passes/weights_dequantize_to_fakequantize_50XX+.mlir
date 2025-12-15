//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --weights-dequantize-to-fake-quantize %s | FileCheck %s
// REQUIRES: arch-NPU50XX

// CHECK-LABEL: @WeightsMultToFakeQuantizeF8E4M3
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x8x3000xf32>
// CHECK-SAME: -> tensor<1x2x3000xf32>
func.func @WeightsMultToFakeQuantizeF8E4M3(%input: tensor<1x8x3000xf32>) -> tensor<1x2x3000xf32> {
    %cst_0 = const.Declare tensor<1x1x1xf32> = dense<158.881897> : tensor<1x1x1xf32>
    %cst = const.Declare tensor<1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1xf32>
    %0 = IE.FakeQuantize(%input, %cst, %cst_0, %cst, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E4M3FN} : tensor<1x8x3000xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32> -> tensor<1x8x3000xf32>
    %cst_1 = const.Declare tensor<2x8x3xf32> = dense<[[[1.500000e+01, 2.250000e+00, -2.800000e+01], [3.000000e+01, 4.000000e+00, -2.600000e+01], [6.000000e+01, 2.000000e+01, -1.500000e+01], [4.000000e+01, 6.000000e+00, -1.800000e+01], [2.600000e+01, -3.250000e+00, 1.000000e+01], [-1.875000e+00, -7.500000e+00, 6.000000e+01], [-7.000000e+00, -5.500000e+00, 8.000000e+01], [1.400000e+01, -5.000000e-01, 4.800000e+01]], [[4.800000e+01, 2.800000e+01, 8.000000e+01], [3.600000e+01, -4.400000e+01, -4.800000e+01], [-3.600000e+01, -1.120000e+02, -7.200000e+01], [-4.800000e+01, -1.920000e+02, -2.400000e+02], [-1.600000e+02, -3.200000e+02, -3.840000e+02], [-1.920000e+02, -3.200000e+02, -3.840000e+02], [-2.400000e+02, -2.880000e+02, -3.200000e+02], [-2.240000e+02, -2.560000e+02, -3.520000e+02]]]> : tensor<2x8x3xf8E4M3FN>, [#const.CastElemType<f32>]
    %cst_2 = const.Declare tensor<2x1x1xf32> = dense<[[[8.68524875E-5]], [[2.0912716E-4]]]> : tensor<2x1x1xf32>
    %1 = IE.Multiply(%cst_1, %cst_2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<2x8x3xf32>, tensor<2x1x1xf32> -> tensor<2x8x3xf32>
    %2 = IE.Convolution(%0, %1) {dilations = [1], pads_begin = [1], pads_end = [1], strides = [1]} : tensor<1x8x3000xf32>, tensor<2x8x3xf32> -> tensor<1x2x3000xf32>

    return %2 : tensor<1x2x3000xf32>

    // CHECK: [[ACT_LOW:%.+]] = const.Declare tensor<1x1x1xf32> = dense<0.000000e+00>
    // CHECK: [[ACT_HIGH:%.+]] = const.Declare tensor<1x1x1xf32> = dense<158.881897>

    // CHECK: [[WT_IN_LOW:%.+]] = const.Declare tensor<1x1x1xf32> = dense<-4.480000e+02>
    // CHECK: [[WT_IN_HIGH:%.+]] = const.Declare tensor<1x1x1xf32> = dense<4.480000e+02>
    // CHECK: [[WT_OUT_LOW:%.+]] = const.Declare tensor<2x1x1xf32>
    // CHECK-SAME{LITERAL}: dense<[[[-0.0389099158]], [[-0.0936889648]]]>
    // CHECK: [[WT_OUT_HIGH:%.+]] = const.Declare tensor<2x1x1xf32>
    // CHECK-SAME{LITERAL}: dense<[[[0.0389099158]], [[0.0936889648]]]>

    // CHECK: [[DATA:%.+]] = const.Declare tensor<2x8x3xf32>
    // CHECK-SAME{LITERAL}: dense<[[[1.500000e+01, 2.250000e+00, -2.800000e+01], [3.000000e+01, 4.000000e+00, -2.600000e+01], [6.000000e+01, 2.000000e+01, -1.500000e+01], [4.000000e+01, 6.000000e+00, -1.800000e+01], [2.600000e+01, -3.250000e+00, 1.000000e+01], [-1.875000e+00, -7.500000e+00, 6.000000e+01], [-7.000000e+00, -5.500000e+00, 8.000000e+01], [1.400000e+01, -5.000000e-01, 4.800000e+01]], [[4.800000e+01, 2.800000e+01, 8.000000e+01], [3.600000e+01, -4.400000e+01, -4.800000e+01], [-3.600000e+01, -1.120000e+02, -7.200000e+01], [-4.800000e+01, -1.920000e+02, -2.400000e+02], [-1.600000e+02, -3.200000e+02, -3.840000e+02], [-1.920000e+02, -3.200000e+02, -3.840000e+02], [-2.400000e+02, -2.880000e+02, -3.200000e+02], [-2.240000e+02, -2.560000e+02, -3.520000e+02]]]>

    // CHECK: [[WT_FQ:%.+]] = IE.FakeQuantize([[DATA]], [[WT_IN_LOW]], [[WT_IN_HIGH]], [[WT_OUT_LOW]], [[WT_OUT_HIGH]])
    // CHECK-SAME: {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E4M3FN}
    // CHECK-SAME: -> tensor<2x8x3xf32>

    // CHECK: [[ACT_FQ:%.+]] = IE.FakeQuantize([[INPUT]], [[ACT_LOW]], [[ACT_HIGH]], [[ACT_LOW]], [[ACT_HIGH]])
    // CHECK-SAME: {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E4M3FN}
    // CHECK-SAME: -> tensor<1x8x3000xf32>

    // CHECK: [[CONV:%.+]] = IE.Convolution([[ACT_FQ]], [[WT_FQ]])
    // CHECK-SAME: {dilations = [1], pads_begin = [1], pads_end = [1], strides = [1]}

    // CHECK: return [[CONV]]

}

// -----

// CHECK-LABEL: @WeightsSubtractMultToFakeQuantizeF8E5M2
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x4x30xf32>
// CHECK-SAME: -> tensor<1x2x30xf32>
func.func @WeightsSubtractMultToFakeQuantizeF8E5M2(%input: tensor<1x4x30xf32>) -> tensor<1x2x30xf32> {
    %cst_0 = const.Declare tensor<1x1x1xf32> = dense<62.314334> : tensor<1x1x1xf32>
    %cst = const.Declare tensor<1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1xf32>
    %0 = IE.FakeQuantize(%input, %cst, %cst_0, %cst, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E5M2} : tensor<1x4x30xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32> -> tensor<1x4x30xf32>
    %cst_1 = const.Declare tensor<2x4x3xf32> = dense<[
    [[4.800000e+01, 2.800000e+01, 8.000000e+01],[3.600000e+01, -4.400000e+01, -4.800000e+01],[-3.600000e+01, -1.120000e+02, -7.200000e+01],[-4.800000e+01, -1.920000e+02, -2.400000e+02]],
    [[-1.600000e+02, -3.200000e+02, -3.840000e+02], [-1.920000e+02, -3.200000e+02, -3.840000e+02], [-2.400000e+02, -2.880000e+02, -3.200000e+02], [-2.240000e+02, -2.560000e+02, -3.520000e+02]]]> : tensor<2x4x3xf8E5M2>, [#const.CastElemType<f32>]
    %cst_2 = const.Declare tensor<2x1x1xf32> = dense<[[[1.000000e+00]], [[2.000000e+00]]]> : tensor<2x1x1xf32>
    %cst_3 = const.Declare tensor<2x1x1xf32> = dense<[[[1.14941406]], [[-1.44335938]]]> : tensor<2x1x1xf32>
    %1 = IE.Subtract(%cst_1, %cst_2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<2x4x3xf32>, tensor<2x1x1xf32> -> tensor<2x4x3xf32>
    %2 = IE.Multiply(%1, %cst_3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<2x4x3xf32>, tensor<2x1x1xf32> -> tensor<2x4x3xf32>
    %3 = IE.Convolution(%0, %2) {dilations = [1], pads_begin = [1], pads_end = [1], strides = [1]} : tensor<1x4x30xf32>, tensor<2x4x3xf32> -> tensor<1x2x30xf32>

    return %3 : tensor<1x2x30xf32>

    // CHECK: [[ACT_LOW:%.+]] = const.Declare tensor<1x1x1xf32> = dense<0.000000e+00>
    // CHECK: [[ACT_HIGH:%.+]] = const.Declare tensor<1x1x1xf32> = dense<62.3143349>

    // CHECK: [[WT_IN_LOW:%.+]] = const.Declare tensor<1x1x1xf32> = dense<-5.734400e+04>
    // CHECK: [[WT_IN_HIGH:%.+]] = const.Declare tensor<1x1x1xf32> = dense<5.734400e+04>
    // CHECK: [[WT_OUT_LOW:%.+]] = const.Declare tensor<2x1x1xf32>
    // CHECK-SAME{LITERAL}: dense<[[[-65913.1484]], [[82770.8906]]]>
    // CHECK: [[WT_OUT_HIGH:%.+]] = const.Declare tensor<2x1x1xf32>
    // CHECK-SAME{LITERAL}: dense<[[[65910.8515]], [[-82765.1093]]]>

    // CHECK: [[DATA:%.+]] = const.Declare tensor<2x4x3xf32>
    // CHECK-SAME{LITERAL}: dense<[[[4.800000e+01, 2.800000e+01, 8.000000e+01], [3.200000e+01, -4.800000e+01, -4.800000e+01], [-3.200000e+01, -1.120000e+02, -6.400000e+01], [-4.800000e+01, -1.920000e+02, -2.560000e+02]], [[-1.600000e+02, -3.200000e+02, -3.840000e+02], [-1.920000e+02, -3.200000e+02, -3.840000e+02], [-2.560000e+02, -2.560000e+02, -3.200000e+02], [-2.240000e+02, -2.560000e+02, -3.840000e+02]]]>

    // CHECK: [[WT_FQ:%.+]] = IE.FakeQuantize([[DATA]], [[WT_IN_LOW]], [[WT_IN_HIGH]], [[WT_OUT_LOW]], [[WT_OUT_HIGH]])
    // CHECK-SAME: {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E5M2}
    // CHECK-SAME: -> tensor<2x4x3xf32>

    // CHECK: [[ACT_FQ:%.+]] = IE.FakeQuantize([[INPUT]], [[ACT_LOW]], [[ACT_HIGH]], [[ACT_LOW]], [[ACT_HIGH]])
    // CHECK-SAME: {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E5M2}
    // CHECK-SAME: -> tensor<1x4x30xf32>

    // CHECK: [[CONV:%.+]] = IE.Convolution([[ACT_FQ]], [[WT_FQ]])
    // CHECK-SAME: {dilations = [1], pads_begin = [1], pads_end = [1], strides = [1]}

    // CHECK: return [[CONV]]
}

// CHECK-LABEL: @DontBlockArgMultToFakeQuantize
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x4x28x28xf8E4M3FN>
// CHECK-SAME: -> tensor<1x4x28x28xf32>
func.func @DontBlockArgMultToFakeQuantize(%input: tensor<1x4x28x28xf8E4M3FN>) -> tensor<1x4x28x28xf32> {
    %cst = const.Declare tensor<1x1x1x1xf32> = dense<0.407326102> : tensor<1x1x1x1xf32>
    %scale = const.Declare tensor<1x1x1x1xf32> = dense<0.5> : tensor<1x1x1x1xf32>

    %convert = IE.Convert(%input) { dstElemType = f32 } : tensor<1x4x28x28xf8E4M3FN> -> tensor<1x4x28x28xf32>
    %1 = IE.Multiply(%convert, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x28x28xf32>, tensor<1x1x1x1xf32> -> tensor<1x4x28x28xf32>
    %2 = IE.Add(%1, %cst) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } : tensor<1x4x28x28xf32>, tensor<1x1x1x1xf32> -> tensor<1x4x28x28xf32>

    return %2 : tensor<1x4x28x28xf32>

    // CHECK-DAG: [[CST:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<0.407326102>
    // CHECK-DAG: [[CST_0:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<5.000000e-01>
    // CHECK: [[CONV:%.+]] = IE.Convert([[INPUT]]) {dstElemType = f32} : tensor<1x4x28x28xf8E4M3FN> -> tensor<1x4x28x28xf32>
    // CHECK: [[MULTI:%.+]] = IE.Multiply([[CONV]], [[CST_0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x28x28xf32>, tensor<1x1x1x1xf32> -> tensor<1x4x28x28xf32>
    // CHECK: [[ADD:%.+]] = IE.Add([[MULTI]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}

    // CHECK: return [[ADD]]
}
