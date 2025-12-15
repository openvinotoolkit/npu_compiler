//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-min-max-to-clamp %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

// -----

// CHECK-LABEL: @EltwiseMinToClamp
// CHECK-SAME:  [[ARG0:%.+]]: tensor<1x64x28x28xf16>
func.func @EltwiseMinToClamp(%arg0: tensor<1x64x28x28xf16>)
        -> tensor<1x64x28x28xf16> {
    %cst = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf16>
    %0 = IE.Minimum(%cst, %arg0) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
        tensor<1x1x1x1xf16>, tensor<1x64x28x28xf16>
        -> tensor<1x64x28x28xf16>

    return %0 : tensor<1x64x28x28xf16>

    // CHECK:       [[OUT:%.+]] = IE.Clamp([[ARG0]]) {max = 0.000000e+00 : f64, min = -6.550400e+04 : f64} : tensor<1x64x28x28xf16> -> tensor<1x64x28x28xf16>
    // CHECK:       return [[OUT]] : tensor<1x64x28x28xf16>
}

// -----

// CHECK-LABEL: @EltwiseMaxToClamp
// CHECK-SAME:  [[ARG0:%.+]]: tensor<1x64x28x28xf16>
func.func @EltwiseMaxToClamp(%arg0: tensor<1x64x28x28xf16>)
        -> tensor<1x64x28x28xf16> {
    %cst = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf16>
    %0 = IE.Maximum(%cst, %arg0) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
        tensor<1x1x1x1xf16>, tensor<1x64x28x28xf16>
        -> tensor<1x64x28x28xf16>

    return %0 : tensor<1x64x28x28xf16>

    // CHECK:       [[OUT:%.+]] = IE.Clamp([[ARG0]]) {max = 6.550400e+04 : f64, min = 0.000000e+00 : f64} : tensor<1x64x28x28xf16> -> tensor<1x64x28x28xf16>
    // CHECK:       return [[OUT]] : tensor<1x64x28x28xf16>
}

// -----

// CHECK-LABEL: @EltwiseMaxWithLargeMinValueToClamp
// CHECK-SAME:  [[ARG0:%.+]]: tensor<1x1024x1xf32>
func.func @EltwiseMaxWithLargeMinValueToClamp(%arg0: tensor<1x1024x1xf32>)
        -> tensor<1x1024x1xf32> {
    %cst = const.Declare tensor<1x1x1xf32> = dense<8454144.0> : tensor<1x1x1xf32>
    %0 = IE.Maximum(%arg0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1024x1xf32>, tensor<1x1x1xf32> -> tensor<1x1024x1xf32>

    return %0 : tensor<1x1024x1xf32>

    // CHECK:       [[OUT:%.+]] = IE.Clamp([[ARG0]]) {max = 6.550400e+04 : f64, min = 6.550400e+04 : f64} : tensor<1x1024x1xf32> -> tensor<1x1024x1xf32>
    // CHECK:       return [[OUT]] : tensor<1x1024x1xf32>
}

// -----

// CHECK-LABEL: @MaximumNotScalarInput
// CHECK-SAME:  [[ARG0:%.+]]: tensor<1x64x28x28xf16>
func.func @MaximumNotScalarInput(%arg0: tensor<1x64x28x28xf16>)
        -> tensor<1x64x28x28xf16> {
    %cst = const.Declare tensor<1x64x28x28xf16> = dense<0.000000e+00> : tensor<1x64x28x28xf16>
    %0 = IE.Maximum(%cst, %arg0) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
        tensor<1x64x28x28xf16>, tensor<1x64x28x28xf16>
        -> tensor<1x64x28x28xf16>

    return %0 : tensor<1x64x28x28xf16>

    // CHECK:       [[CST:%.+]] = const.Declare tensor<1x64x28x28xf16> = dense<0.000000e+00> : tensor<1x64x28x28xf16>
    // CHECK:       [[OUT:%.+]] = IE.Maximum([[CST]], [[ARG0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x64x28x28xf16>, tensor<1x64x28x28xf16> -> tensor<1x64x28x28xf16>
    // CHECK:       return [[OUT]] : tensor<1x64x28x28xf16>
}

// -----

// CHECK-LABEL: @MinimumNotScalarInput
// CHECK-SAME:  [[ARG0:%.+]]: tensor<1x64x28x28xf16>
func.func @MinimumNotScalarInput(%arg0: tensor<1x64x28x28xf16>)
        -> tensor<1x64x28x28xf16> {
    %cst = const.Declare tensor<1x64x28x28xf16> = dense<0.000000e+00> : tensor<1x64x28x28xf16>
    %0 = IE.Minimum(%cst, %arg0) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
        tensor<1x64x28x28xf16>, tensor<1x64x28x28xf16>
        -> tensor<1x64x28x28xf16>

    return %0 : tensor<1x64x28x28xf16>

    // CHECK:       [[CST:%.+]] = const.Declare tensor<1x64x28x28xf16> = dense<0.000000e+00> : tensor<1x64x28x28xf16>
    // CHECK:       [[OUT:%.+]] = IE.Minimum([[CST]], [[ARG0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x64x28x28xf16>, tensor<1x64x28x28xf16> -> tensor<1x64x28x28xf16>
    // CHECK:       return [[OUT]] : tensor<1x64x28x28xf16>
}

// -----

// CHECK-LABEL: @MinimumWithSecondInputScalar
// CHECK-SAME:  [[ARG0:%.+]]: tensor<1x1x1x3xf16>
func.func @MinimumWithSecondInputScalar(%arg0: tensor<1x1x1x3xf16>)
        -> tensor<1x1x1x3xf16> {
    %cst = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf16>
    %0 = IE.Minimum(%arg0, %cst) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
        tensor<1x1x1x3xf16>, tensor<1x1x1x1xf16>
        -> tensor<1x1x1x3xf16>

    return %0 : tensor<1x1x1x3xf16>

    // CHECK:       [[OUT:%.+]] = IE.Clamp([[ARG0]]) {max = 0.000000e+00 : f64, min = -6.550400e+04 : f64} : tensor<1x1x1x3xf16> -> tensor<1x1x1x3xf16>
    // CHECK:       return [[OUT]] : tensor<1x1x1x3xf16>
}

// -----

// CHECK-LABEL: @MaximumNoInputAsScalar
// CHECK-SAME:  [[ARG0:%.+]]: tensor<1x1x40x1xf32>
// CHECK-SAME:  [[ARG1:%.+]]: tensor<f32>
func.func @MaximumNoInputAsScalar(%arg0: tensor<1x1x40x1xf32>, %arg1: tensor<f32>)
        -> tensor<1x1x40x1xf32> {
    %0 = IE.Maximum(%arg0, %arg1) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
        tensor<1x1x40x1xf32>, tensor<f32>
        -> tensor<1x1x40x1xf32>

    return %0 : tensor<1x1x40x1xf32>

    // CHECK:       [[OUT:%.+]] = IE.Maximum([[ARG0]], [[ARG1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x40x1xf32>, tensor<f32> -> tensor<1x1x40x1xf32>
    // CHECK:       return [[OUT]] : tensor<1x1x40x1xf32>
}

// -----

// CHECK-LABEL: @MinimumNoInputAsScalar
// CHECK-SAME:  [[ARG0:%.+]]: tensor<1x1x40x1xf32>
// CHECK-SAME:  [[ARG1:%.+]]: tensor<f32>
func.func @MinimumNoInputAsScalar(%arg0: tensor<1x1x40x1xf32>, %arg1: tensor<f32>)
        -> tensor<1x1x40x1xf32> {
    %0 = IE.Minimum(%arg0, %arg1) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
        tensor<1x1x40x1xf32>, tensor<f32>
        -> tensor<1x1x40x1xf32>

    return %0 : tensor<1x1x40x1xf32>

    // CHECK:       [[OUT:%.+]] = IE.Minimum([[ARG0]], [[ARG1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x40x1xf32>, tensor<f32> -> tensor<1x1x40x1xf32>
    // CHECK:       return [[OUT]] : tensor<1x1x40x1xf32>
}

// -----

// CHECK-LABEL: @MinimumWithSecondInputFakeQuantizeScalar
// CHECK-SAME:  [[ARG0:%.+]]: tensor<1x1x1x3xf32>
func.func @MinimumWithSecondInputFakeQuantizeScalar(%arg0: tensor<1x1x1x3xf32>)
        -> tensor<1x1x1x3xf32> {
    %cst_0 = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
    %cst_1 = const.Declare tensor<1x1x1x1xf32> = dense<2.550000e+02> : tensor<1x1x1x1xf32>
    %cst_2 = const.Declare tensor<1x1x1x1xf32> = dense<-1.41549385> : tensor<1x1x1x1xf32>
    %cst_3 = const.Declare tensor<1x1x1x1xf32> = dense<13.0225439> : tensor<1x1x1x1xf32>
    %cst_4 = const.Declare tensor<1x1x1x1xf32> = dense<20> : tensor<1x1x1x1xui8>, [#const.CastElemType<f32>]
    %0 = IE.FakeQuantize(%cst_4, %cst_0, %cst_1, %cst_2, %cst_3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x1x1x1xf32>
    %1 = IE.Minimum(%arg0, %0) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
        tensor<1x1x1x3xf32>, tensor<1x1x1x1xf32>
        -> tensor<1x1x1x3xf32>

    return %1 : tensor<1x1x1x3xf32>

    // CHECK:       [[OUT:%.+]] = IE.Clamp([[ARG0]]) {max = -0.28309872804903513 : f64, min = -6.550400e+04 : f64} : tensor<1x1x1x3xf32> -> tensor<1x1x1x3xf32>
    // CHECK:       return [[OUT]] : tensor<1x1x1x3xf32>
}

// -----

// CHECK-LABEL: @MaximumWithFirstInputFakeQuantizeScalar
// CHECK-SAME:  [[ARG0:%.+]]: tensor<1x1x1x3xf32>
func.func @MaximumWithFirstInputFakeQuantizeScalar(%arg0: tensor<1x1x1x3xf32>)
        -> tensor<1x1x1x3xf32> {
    %cst_0 = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
    %cst_1 = const.Declare tensor<1x1x1x1xf32> = dense<2.550000e+02> : tensor<1x1x1x1xf32>
    %cst_2 = const.Declare tensor<1x1x1x1xf32> = dense<-1.41549385> : tensor<1x1x1x1xf32>
    %cst_3 = const.Declare tensor<1x1x1x1xf32> = dense<13.0225439> : tensor<1x1x1x1xf32>
    %cst_4 = const.Declare tensor<1x1x1x1xf32> = dense<20> : tensor<1x1x1x1xui8>, [#const.CastElemType<f32>]
    %0 = IE.FakeQuantize(%cst_4, %cst_0, %cst_1, %cst_2, %cst_3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x1x1x1xf32>
    %1 = IE.Maximum(%0, %arg0) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
        tensor<1x1x1x1xf32>, tensor<1x1x1x3xf32>
        -> tensor<1x1x1x3xf32>

    return %1 : tensor<1x1x1x3xf32>

    // CHECK:       [[OUT:%.+]] = IE.Clamp([[ARG0]]) {max = 6.550400e+04 : f64, min = -0.28309872804903513 : f64} : tensor<1x1x1x3xf32> -> tensor<1x1x1x3xf32>
    // CHECK:       return [[OUT]] : tensor<1x1x1x3xf32>
}

// -----

!qElemType = !quant.uniform<u8:f32, 5.000000e-01:100>

// CHECK-LABEL: @MaximumWithSecondInputDequantizeScalar
// CHECK-SAME:  [[ARG0:%.+]]: tensor<1x1x1x3xf32>
func.func @MaximumWithSecondInputDequantizeScalar(%arg0: tensor<1x1x1x3xf32>)
        -> tensor<1x1x1x3xf32> {
    %cst = const.Declare tensor<1x1x1x1x!qElemType> = dense<20> : tensor<1x1x1x1xui8>, [#const.CastElemType<!qElemType>]
    %0 = IE.Dequantize(%cst) {dstElemType = f32} : tensor<1x1x1x1x!qElemType> -> tensor<1x1x1x1xf32>
    %1 = IE.Maximum(%arg0, %0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
        tensor<1x1x1x3xf32>, tensor<1x1x1x1xf32>
        -> tensor<1x1x1x3xf32>

    return %1 : tensor<1x1x1x3xf32>

    // CHECK:       [[OUT:%.+]] = IE.Clamp([[ARG0]]) {max = 6.550400e+04 : f64, min = -4.000000e+01 : f64} : tensor<1x1x1x3xf32> -> tensor<1x1x1x3xf32>
    // CHECK:       return [[OUT]] : tensor<1x1x1x3xf32>
}

// -----

!qElemType = !quant.uniform<u8:f32, 5.000000e-01:100>

// CHECK-LABEL: @MinimumWithFirstInputDequantizeScalar
// CHECK-SAME:  [[ARG0:%.+]]: tensor<1x1x1x3xf32>
func.func @MinimumWithFirstInputDequantizeScalar(%arg0: tensor<1x1x1x3xf32>)
        -> tensor<1x1x1x3xf32> {
    %cst = const.Declare tensor<1x1x1x1x!qElemType> = dense<20> : tensor<1x1x1x1xui8>, [#const.CastElemType<!qElemType>]
    %0 = IE.Dequantize(%cst) {dstElemType = f32} : tensor<1x1x1x1x!qElemType> -> tensor<1x1x1x1xf32>
    %1 = IE.Minimum(%0, %arg0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
        tensor<1x1x1x1xf32>, tensor<1x1x1x3xf32>
        -> tensor<1x1x1x3xf32>

    return %1 : tensor<1x1x1x3xf32>

    // CHECK:       [[OUT:%.+]] = IE.Clamp([[ARG0]]) {max = -4.000000e+01 : f64, min = -6.550400e+04 : f64} : tensor<1x1x1x3xf32> -> tensor<1x1x1x3xf32>
    // CHECK:       return [[OUT]] : tensor<1x1x1x3xf32>
}
