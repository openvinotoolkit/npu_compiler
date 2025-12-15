//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

// CHECK-LABEL: @FuseScaleAndBias
// CHECK-SAME:     ([[ARG0:%.+]]: tensor<1x3x300x300xf32>)
func.func @FuseScaleAndBias(%arg0: tensor<1x3x300x300xf32>) -> tensor<1x3x300x300xf32> {
    %weights = const.Declare tensor<1x3x1x1xf32> = dense<2.0> : tensor<1x3x1x1xf32>
    %0 = IE.ScaleShift(%arg0, %weights)
        {operandSegmentSizes = array<i32: 1, 1, 0>} :
        tensor<1x3x300x300xf32>, tensor<1x3x1x1xf32> -> tensor<1x3x300x300xf32>

    %bias = const.Declare tensor<1x3x1x1xf32> = dense<3.0> : tensor<1x3x1x1xf32>
    %1 = IE.ScaleShift(%0, %bias)
        {operandSegmentSizes = array<i32: 1, 0, 1>} :
        tensor<1x3x300x300xf32>, tensor<1x3x1x1xf32> -> tensor<1x3x300x300xf32>

    return %1 : tensor<1x3x300x300xf32>

    // CHECK-DAG:   [[WEIGHTS:%.+]] = const.Declare tensor<1x3x1x1xf32> = dense<2.000000e+00> : tensor<1x3x1x1xf32>
    // CHECK-DAG:   [[BIAS:%.+]] = const.Declare tensor<1x3x1x1xf32> = dense<3.000000e+00> : tensor<1x3x1x1xf32>
    // CHECK:       [[SCALE_SHIFT:%.+]] = IE.ScaleShift([[ARG0]], [[WEIGHTS]], [[BIAS]])
    // CHECK:       return [[SCALE_SHIFT]]
}

// -----

// Fuse ScaleShift and Bias should fail
// CHECK-LABEL: @FuseScaleAndBias
// CHECK-SAME:     ([[ARG0:%.+]]: tensor<1x3x300x300xf32>)
func.func @FuseScaleAndBias(%arg0: tensor<1x3x300x300xf32>) -> tensor<1x3x300x300xf32> {
    %weights = const.Declare tensor<1x3x1x1xf32> = dense<2.0> : tensor<1x3x1x1xf32>
    %bias0 = const.Declare tensor<1x3x1x1xf32> = dense<3.0> : tensor<1x3x1x1xf32>
    %0 = IE.ScaleShift(%arg0, %weights, %bias0)
        {operandSegmentSizes = array<i32: 1, 1, 1>}:
        tensor<1x3x300x300xf32>, tensor<1x3x1x1xf32>, tensor<1x3x1x1xf32> -> tensor<1x3x300x300xf32>

    %bias1 = const.Declare tensor<1x3x1x1xf32> = dense<4.0> : tensor<1x3x1x1xf32>
    %1 = IE.ScaleShift(%0, %bias1)
        {operandSegmentSizes = array<i32: 1, 0, 1>} :
        tensor<1x3x300x300xf32>, tensor<1x3x1x1xf32> -> tensor<1x3x300x300xf32>

    return %1 : tensor<1x3x300x300xf32>

    // CHECK-DAG:   [[WEIGHTS:%.+]] = const.Declare tensor<1x3x1x1xf32> = dense<2.000000e+00> : tensor<1x3x1x1xf32>
    // CHECK-DAG:   [[BIAS_0:%.+]] = const.Declare tensor<1x3x1x1xf32> = dense<3.000000e+00> : tensor<1x3x1x1xf32>
    // CHECK-DAG:   [[BIAS_1:%.+]] = const.Declare tensor<1x3x1x1xf32> = dense<4.000000e+00> : tensor<1x3x1x1xf32>
    // CHECK:       [[SCALE_SHIFT_0:%.+]] = IE.ScaleShift([[ARG0]], [[WEIGHTS]], [[BIAS_0]])
    // CHECK:       [[SCALE_SHIFT_1:%.+]] = IE.ScaleShift([[SCALE_SHIFT_0]], [[BIAS_1]])
    // CHECK:       return [[SCALE_SHIFT_1]]
}

// -----

// Fuse Scale and ScaleShift should fail
// CHECK-LABEL: @FuseScaleAndBias
// CHECK-SAME:     ([[ARG0:%.+]]: tensor<1x3x300x300xf32>)
func.func @FuseScaleAndBias(%arg0: tensor<1x3x300x300xf32>) -> tensor<1x3x300x300xf32> {
    %weights = const.Declare tensor<1x3x1x1xf32> = dense<2.0> : tensor<1x3x1x1xf32>
    %0 = IE.ScaleShift(%arg0, %weights)
        {operandSegmentSizes = array<i32: 1, 1, 0>}:
        tensor<1x3x300x300xf32>, tensor<1x3x1x1xf32> -> tensor<1x3x300x300xf32>

    %weights1 = const.Declare tensor<1x3x1x1xf32> = dense<3.0> : tensor<1x3x1x1xf32>
    %bias = const.Declare tensor<1x3x1x1xf32> = dense<4.0> : tensor<1x3x1x1xf32>
    %1 = IE.ScaleShift(%0, %weights1, %bias)
        {operandSegmentSizes = array<i32: 1, 1, 1>} :
        tensor<1x3x300x300xf32>, tensor<1x3x1x1xf32>, tensor<1x3x1x1xf32> -> tensor<1x3x300x300xf32>

    return %1 : tensor<1x3x300x300xf32>

    // CHECK-DAG:   [[WEIGHTS_0:%.+]] = const.Declare tensor<1x3x1x1xf32> = dense<2.000000e+00> : tensor<1x3x1x1xf32>
    // CHECK-DAG:   [[WEIGHTS_1:%.+]] = const.Declare tensor<1x3x1x1xf32> = dense<3.000000e+00> : tensor<1x3x1x1xf32>
    // CHECK-DAG:   [[BIAS:%.+]] = const.Declare tensor<1x3x1x1xf32> = dense<4.000000e+00> : tensor<1x3x1x1xf32>
    // CHECK:       [[SCALE_SHIFT_0:%.+]] = IE.ScaleShift([[ARG0]], [[WEIGHTS_0]])
    // CHECK:       [[SCALE_SHIFT_1:%.+]] = IE.ScaleShift([[SCALE_SHIFT_0]], [[WEIGHTS_1]], [[BIAS]])
    // CHECK:       return [[SCALE_SHIFT_1]]
}

// -----

// CHECK-LABEL: @FuseScaleShifts
// CHECK-SAME:     ([[ARG0:%.+]]: tensor<1x3x300x300xf32>)
func.func @FuseScaleShifts(%arg0: tensor<1x3x300x300xf32>) -> tensor<1x3x300x300xf32> {
    %weights_0 = const.Declare tensor<1x3x1x1xf32> = dense<2.0> : tensor<1x3x1x1xf32>
    %bias_0 = const.Declare tensor<1x3x1x1xf32> = dense<3.0> : tensor<1x3x1x1xf32>
    %0 = IE.ScaleShift(%arg0, %weights_0, %bias_0)
        {operandSegmentSizes = array<i32: 1, 1, 1>} :
        tensor<1x3x300x300xf32>, tensor<1x3x1x1xf32>, tensor<1x3x1x1xf32> -> tensor<1x3x300x300xf32>

    %weights_1 = const.Declare tensor<1x3x1x1xf32> = dense<2.0> : tensor<1x3x1x1xf32>
    %bias_1 = const.Declare tensor<1x3x1x1xf32> = dense<3.0> : tensor<1x3x1x1xf32>
    %1 = IE.ScaleShift(%0, %weights_1, %bias_1)
        {operandSegmentSizes = array<i32: 1, 1, 1>} :
        tensor<1x3x300x300xf32>, tensor<1x3x1x1xf32>, tensor<1x3x1x1xf32> -> tensor<1x3x300x300xf32>

    return %1 : tensor<1x3x300x300xf32>

    // CHECK-DAG:   [[WEIGHTS:%.+]] = const.Declare tensor<1x3x1x1xf32> = dense<4.000000e+00> : tensor<1x3x1x1xf32>
    // CHECK-DAG:   [[BIAS:%.+]] = const.Declare tensor<1x3x1x1xf32> = dense<9.000000e+00> : tensor<1x3x1x1xf32>
    // CHECK:       [[SCALE_SHIFT:%.+]] = IE.ScaleShift([[ARG0]], [[WEIGHTS]], [[BIAS]])
    // CHECK:       return [[SCALE_SHIFT]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @FuseScaleShiftsDynamicInput
// CHECK-SAME:     ([[ARG0:%.+]]: tensor<1x3x?x300xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 300, 300]> : tensor<4xsi64>, order = #NHWC}>)
func.func @FuseScaleShiftsDynamicInput(%arg0: tensor<1x3x?x300xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 300, 300]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x3x?x300xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 300, 300]> : tensor<4xsi64>, order = #NHWC}> {
    %weights_0 = const.Declare tensor<1x3x1x1xf32> = dense<2.0> : tensor<1x3x1x1xf32>
    %bias_0 = const.Declare tensor<1x3x1x1xf32> = dense<3.0> : tensor<1x3x1x1xf32>
    %0 = IE.ScaleShift(%arg0, %weights_0, %bias_0)
        {operandSegmentSizes = array<i32: 1, 1, 1>} :
        tensor<1x3x?x300xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 300, 300]> : tensor<4xsi64>, order = #NHWC}>, tensor<1x3x1x1xf32>, tensor<1x3x1x1xf32> -> tensor<1x3x?x300xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 300, 300]> : tensor<4xsi64>, order = #NHWC}>

    %weights_1 = const.Declare tensor<1x3x1x1xf32> = dense<2.0> : tensor<1x3x1x1xf32>
    %bias_1 = const.Declare tensor<1x3x1x1xf32> = dense<3.0> : tensor<1x3x1x1xf32>
    %1 = IE.ScaleShift(%0, %weights_1, %bias_1)
        {operandSegmentSizes = array<i32: 1, 1, 1>} :
        tensor<1x3x?x300xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 300, 300]> : tensor<4xsi64>, order = #NHWC}>, tensor<1x3x1x1xf32>, tensor<1x3x1x1xf32> -> tensor<1x3x?x300xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 300, 300]> : tensor<4xsi64>, order = #NHWC}>

    return %1 : tensor<1x3x?x300xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 300, 300]> : tensor<4xsi64>, order = #NHWC}>

    // CHECK-DAG:   [[WEIGHTS:%.+]] = const.Declare tensor<1x3x1x1xf32> = dense<4.000000e+00> : tensor<1x3x1x1xf32>
    // CHECK-DAG:   [[BIAS:%.+]] = const.Declare tensor<1x3x1x1xf32> = dense<9.000000e+00> : tensor<1x3x1x1xf32>
    // CHECK:       [[SCALE_SHIFT:%.+]] = IE.ScaleShift([[ARG0]], [[WEIGHTS]], [[BIAS]])
    // CHECK:       return [[SCALE_SHIFT]]
}

// -----

// CHECK-LABEL: @FuseScaleShiftsWithMultipleConstTransformations
// CHECK-SAME:     ([[ARG0:%.+]]: tensor<1x64x1x9216xf16>)
func.func @FuseScaleShiftsWithMultipleConstTransformations(%arg0: tensor<1x64x1x9216xf16>) -> tensor<1x64x1x9216xf16> {
    %scale_0 = const.Declare tensor<1x64x1x1xf16> = dense<2.0> : tensor<1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>, #const.Broadcast<1 : i64, 64 : i64>]
    %bias_0 = const.Declare tensor<1x64x1x1xf16> = dense<5.0> : tensor<1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>, #const.Broadcast<1 : i64, 64 : i64>]
    %scale = const.Declare tensor<1x64x1x1xf16> = dense<4.0> : tensor<1x1x1xf16>, [#const.Reshape<[1, 1, 1, 1]>, #const.Broadcast<1 : i64, 64 : i64>]
    %bias = const.Declare tensor<1x64x1x1xf16> = dense<20.0> : tensor<1x1x1xf16>, [#const.Reshape<[1, 1, 1, 1]>, #const.Broadcast<1 : i64, 64 : i64>]
    %0 = IE.ScaleShift(%arg0, %scale_0) {operandSegmentSizes = array<i32: 1, 1, 0>} : tensor<1x64x1x9216xf16>, tensor<1x64x1x1xf16> -> tensor<1x64x1x9216xf16>
    %1 = IE.ScaleShift(%0, %bias_0) {operandSegmentSizes = array<i32: 1, 0, 1>} : tensor<1x64x1x9216xf16>, tensor<1x64x1x1xf16> -> tensor<1x64x1x9216xf16>
    %2 = IE.ScaleShift(%1, %scale) {operandSegmentSizes = array<i32: 1, 1, 0>} : tensor<1x64x1x9216xf16>, tensor<1x64x1x1xf16> -> tensor<1x64x1x9216xf16>
    %3 = IE.ScaleShift(%2, %bias) {operandSegmentSizes = array<i32: 1, 0, 1>} : tensor<1x64x1x9216xf16>, tensor<1x64x1x1xf16> -> tensor<1x64x1x9216xf16>
    return %3 : tensor<1x64x1x9216xf16>
    // CHECK-DAG:   [[SCALE:%.+]] = const.Declare tensor<1x64x1x1xf16> = dense<8.000000e+00> : tensor<1x64x1x1xf32>, [#const.CastElemType<f16>]
    // CHECK-DAG:   [[BIAS:%.+]] = const.Declare tensor<1x64x1x1xf16> = dense<4.000000e+01> : tensor<1x64x1x1xf32>, [#const.CastElemType<f16>]
    // CHECK:       [[SCALE_SHIFT:%.+]] = IE.ScaleShift([[ARG0]], [[SCALE]], [[BIAS]]) {operandSegmentSizes = array<i32: 1, 1, 1>} : tensor<1x64x1x9216xf16>, tensor<1x64x1x1xf16>, tensor<1x64x1x1xf16> -> tensor<1x64x1x9216xf16>
    // CHECK:       return [[SCALE_SHIFT]]
}

// -----

// CHECK-LABEL: @FoldIdentity
// CHECK-SAME:     ([[ARG0:%.+]]: tensor<1x3x300x300xf32>)
func.func @FoldIdentity(%arg0: tensor<1x3x300x300xf32>) -> tensor<1x3x300x300xf32> {
    %add = IE.Add(%arg0, %arg0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x300x300xf32>, tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32>
    %weights = const.Declare tensor<1x3x1x1xf32> = dense<1.0> : tensor<1x3x1x1xf32>
    %bias = const.Declare tensor<1x3x1x1xf32> = dense<0.0> : tensor<1x3x1x1xf32>
    %scale_shift = IE.ScaleShift(%add, %weights, %bias)
        {operandSegmentSizes = array<i32: 1, 1, 1>} :
        tensor<1x3x300x300xf32>, tensor<1x3x1x1xf32>, tensor<1x3x1x1xf32> -> tensor<1x3x300x300xf32>

    return %scale_shift : tensor<1x3x300x300xf32>

    // CHECK: [[ADD:%.+]] = IE.Add([[ARG0]], [[ARG0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x300x300xf32>, tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32>
    // CHECK: return [[ADD]]
}

// -----

// CHECK-LABEL: @FuseAndFoldIdentity
// CHECK-SAME:     ([[ARG0:%.+]]: tensor<1x1x64x3072xf16>)
func.func @FuseAndFoldIdentity(%arg0: tensor<1x1x64x3072xf16>) -> tensor<1x1x64x3072xf16> {
    %wights1_cst = const.Declare tensor<1x1x1x1xf16> = dense<1.83503522E-4> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
    %bias1_cst = const.Declare tensor<1x1x1x1xf16> = dense<-6.07873774> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
    %wights0_cst = const.Declare tensor<1x1x1x1xf16> = dense<5449.48633> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
    %bias0_cst = const.Declare tensor<1x1x1x1xf16> = dense<33125.9961> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>] loc(unknown)
    %add = IE.Add(%arg0, %arg0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x64x3072xf16>, tensor<1x1x64x3072xf16> -> tensor<1x1x64x3072xf16>
    %sf0 = IE.ScaleShift(%add, %wights0_cst) {operandSegmentSizes = array<i32: 1, 1, 0>} : tensor<1x1x64x3072xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x64x3072xf16>
    %sf1 = IE.ScaleShift(%sf0, %bias0_cst) {operandSegmentSizes = array<i32: 1, 0, 1>} : tensor<1x1x64x3072xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x64x3072xf16>
    %sf2 = IE.ScaleShift(%sf1, %wights1_cst) {operandSegmentSizes = array<i32: 1, 1, 0>} : tensor<1x1x64x3072xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x64x3072xf16>
    %sf3 = IE.ScaleShift(%sf2, %bias1_cst) {operandSegmentSizes = array<i32: 1, 0, 1>} : tensor<1x1x64x3072xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x64x3072xf16>
    return %sf3 : tensor<1x1x64x3072xf16>

    // CHECK: [[ADD:%.+]] = IE.Add([[ARG0]], [[ARG0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x64x3072xf16>, tensor<1x1x64x3072xf16> -> tensor<1x1x64x3072xf16>
    // CHECK: return [[ADD]]
}
