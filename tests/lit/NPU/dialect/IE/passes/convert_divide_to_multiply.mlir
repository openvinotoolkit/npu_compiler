//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-divide-to-multiply --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
!qElemType = !quant.uniform<u8:f16, 0.01013327205882353>

// CHECK-LABEL: @DoNotConvertNonConstDivide
// CHECK-SAME: ([[ARG:%.+]]: tensor<1x12x512x512xf16>) -> tensor<1x12x512x512xf16>
func.func @DoNotConvertNonConstDivide(%arg: tensor<1x12x512x512xf16>) -> tensor<1x12x512x512xf16> {
    %divisor = const.Declare tensor<1x1x1x1x!qElemType> = dense<2> : tensor<1x1x1x1xui8>, [#const.CastElemType<!qElemType>]
    %nonCst = IE.Dequantize(%divisor) {dstElemType = f16} : tensor<1x1x1x1x!qElemType> -> tensor<1x1x1x1xf16>
    %0 = IE.Divide(%arg, %nonCst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
        : tensor<1x12x512x512xf16>, tensor<1x1x1x1xf16> -> tensor<1x12x512x512xf16>
    return %0 : tensor<1x12x512x512xf16>

    // CHECK: [[CONST:%.+]] = const.Declare tensor<1x1x1x1x!qElemType>
    // CHECK: [[NON_CONST:%.+]] = IE.Dequantize([[CONST]])
    // CHECK: [[DIVIDE:%.+]] = IE.Divide([[ARG]], [[NON_CONST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK: return [[DIVIDE]]
}

// -----

// CHECK-LABEL: @DoNotConvertIntegerDivide
// CHECK-SAME: ([[ARG:%.+]]: tensor<1x12x512x512xsi32>) -> tensor<1x12x512x512xsi32>
func.func @DoNotConvertIntegerDivide(%arg: tensor<1x12x512x512xsi32>) -> tensor<1x12x512x512xsi32> {
    %divisor = const.Declare tensor<1x1x1x1xsi32> = dense<2> : tensor<1x1x1x1xsi32>
    %0 = IE.Divide(%arg, %divisor) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
        : tensor<1x12x512x512xsi32>, tensor<1x1x1x1xsi32> -> tensor<1x12x512x512xsi32>
    return %0 : tensor<1x12x512x512xsi32>

    // CHECK: [[CONST:%.+]] = const.Declare tensor<1x1x1x1xsi32>
    // CHECK: [[DIVIDE:%.+]] = IE.Divide([[ARG]], [[CONST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK: return [[DIVIDE]]
}

// -----

// CHECK-LABEL: @DoNotConvertConstDividend
// CHECK-SAME: ([[ARG:%.+]]: tensor<1x12x512x512xf16>) -> tensor<1x12x512x512xf16>
func.func @DoNotConvertConstDividend(%arg: tensor<1x12x512x512xf16>) -> tensor<1x12x512x512xf16> {
    %cst = const.Declare tensor<1x1x1x1xf16> = dense<2.0> : tensor<1x1x1x1xf16>
    %0 = IE.Divide(%cst, %arg) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
        : tensor<1x1x1x1xf16>, tensor<1x12x512x512xf16> -> tensor<1x12x512x512xf16>
    return %0 : tensor<1x12x512x512xf16>

    // CHECK: [[CONST:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<2.000000e+00> : tensor<1x1x1x1xf16>
    // CHECK: [[DIVIDE:%.+]] = IE.Divide([[CONST]], [[ARG]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK: return [[DIVIDE]]
}

// -----

// CHECK-LABEL: @ConvertConstDivisor
// CHECK-SAME: ([[ARG:%.+]]: tensor<1x12x512x512xf16>) -> tensor<1x12x512x512xf16>
func.func @ConvertConstDivisor(%arg: tensor<1x12x512x512xf16>) -> tensor<1x12x512x512xf16> {
    %divisor = const.Declare tensor<1x1x1x1xf16> = dense<2.0> : tensor<1x1x1x1xf16>
    %0 = IE.Divide(%arg, %divisor) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
        : tensor<1x12x512x512xf16>, tensor<1x1x1x1xf16> -> tensor<1x12x512x512xf16>
    return %0 : tensor<1x12x512x512xf16>

    // CHECK: [[CONST:%.+]] = const.Declare tensor<1x1x1x1xf16> {{.*}} [#const.ScalarMultInverse]
    // CHECK: [[MULTIPLY:%.+]] = IE.Multiply([[ARG]], [[CONST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK: return [[MULTIPLY]]
}

// -----

// CHECK-LABEL: @ConvertConstDivisor_NonScalar
// CHECK-SAME: ([[ARG:%.+]]: tensor<1x12x512x512xf16>) -> tensor<1x12x512x512xf16>
func.func @ConvertConstDivisor_NonScalar(%arg: tensor<1x12x512x512xf16>) -> tensor<1x12x512x512xf16> {
    %divisor = const.Declare tensor<1x12x512x512xf16> = dense<2.0> : tensor<1x12x512x512xf16>
    %0 = IE.Divide(%arg, %divisor) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>}
        : tensor<1x12x512x512xf16>, tensor<1x12x512x512xf16> -> tensor<1x12x512x512xf16>
    return %0 : tensor<1x12x512x512xf16>

    // CHECK: [[CONST:%.+]] = const.Declare tensor<1x12x512x512xf16> {{.*}} [#const.ScalarMultInverse]
    // CHECK: [[MULTIPLY:%.+]] = IE.Multiply([[ARG]], [[CONST]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>}
    // CHECK: return [[MULTIPLY]]
}

// -----

// CHECK-LABEL: @ConvertMultipleDivideOps
// CHECK-SAME: ([[ARG:%.+]]: tensor<1x12x512x512xf16>) -> (tensor<1x12x512x512xf16>, tensor<1x12x512x512xf16>)
func.func @ConvertMultipleDivideOps(%arg: tensor<1x12x512x512xf16>) -> (tensor<1x12x512x512xf16>, tensor<1x12x512x512xf16>) {
    %divisor = const.Declare tensor<1x12x512x512xf16> = dense<2.0> : tensor<1x12x512x512xf16>
    %0 = IE.Divide(%arg, %divisor) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>}
        : tensor<1x12x512x512xf16>, tensor<1x12x512x512xf16> -> tensor<1x12x512x512xf16>
    %1 = IE.Divide(%arg, %divisor) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>}
        : tensor<1x12x512x512xf16>, tensor<1x12x512x512xf16> -> tensor<1x12x512x512xf16>
    return %0, %1 : tensor<1x12x512x512xf16>, tensor<1x12x512x512xf16>

    // CHECK:     [[CONST:%.+]] = const.Declare tensor<1x12x512x512xf16> = dense<2.000000e+00> : tensor<1x12x512x512xf16>, [#const.ScalarMultInverse]
    // CHECK-DAG: [[MULTIPLY0:%.+]] = IE.Multiply([[ARG]], [[CONST]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>}
    // CHECK-DAG: [[MULTIPLY1:%.+]] = IE.Multiply([[ARG]], [[CONST]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>}
    // CHECK:     return [[MULTIPLY1]], [[MULTIPLY0]]
}

// -----

// CHECK-LABEL: @DoNotConvertMultipleDivideOps_NotSecondInput
// CHECK-SAME: ([[ARG:%.+]]: tensor<1x12x512x512xf16>) -> (tensor<1x12x512x512xf16>, tensor<1x12x512x512xf16>)
func.func @DoNotConvertMultipleDivideOps_NotSecondInput(%arg: tensor<1x12x512x512xf16>) -> (tensor<1x12x512x512xf16>, tensor<1x12x512x512xf16>) {
    %divisor = const.Declare tensor<1x12x512x512xf16> = dense<2.0> : tensor<1x12x512x512xf16>
    %0 = IE.Divide(%arg, %divisor) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>}
        : tensor<1x12x512x512xf16>, tensor<1x12x512x512xf16> -> tensor<1x12x512x512xf16>
    %1 = IE.Divide(%divisor, %arg) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>}
        : tensor<1x12x512x512xf16>, tensor<1x12x512x512xf16> -> tensor<1x12x512x512xf16>
    return %0, %1 : tensor<1x12x512x512xf16>, tensor<1x12x512x512xf16>

    // CHECK: [[CONST:%.+]] = const.Declare tensor<1x12x512x512xf16> = dense<2.000000e+00> : tensor<1x12x512x512xf16>
    // CHECK: [[DIVIDE0:%.+]] = IE.Divide([[ARG]], [[CONST]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>}
    // CHECK: [[DIVIDE1:%.+]] = IE.Divide([[CONST]], [[ARG]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>}
    // CHECK: return [[DIVIDE0]], [[DIVIDE1]]
}

// -----

// CHECK-LABEL: @DoNotConvert_NonDivideUsers
// CHECK-SAME: ([[ARG:%.+]]: tensor<1x12x512x512xf16>) -> (tensor<1x12x512x512xf16>, tensor<1x12x512x512xf16>)
func.func @DoNotConvert_NonDivideUsers(%arg: tensor<1x12x512x512xf16>) -> (tensor<1x12x512x512xf16>, tensor<1x12x512x512xf16>) {
    %cst = const.Declare tensor<1x12x512x512xf16> = dense<2.0> : tensor<1x12x512x512xf16>
    %0 = IE.Divide(%arg, %cst) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>}
        : tensor<1x12x512x512xf16>, tensor<1x12x512x512xf16> -> tensor<1x12x512x512xf16>
    %1 = IE.Add(%arg, %cst) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>}
        : tensor<1x12x512x512xf16>, tensor<1x12x512x512xf16> -> tensor<1x12x512x512xf16>
    return %0, %1 : tensor<1x12x512x512xf16>, tensor<1x12x512x512xf16>

    // CHECK: [[CONST:%.+]] = const.Declare tensor<1x12x512x512xf16> = dense<2.000000e+00> : tensor<1x12x512x512xf16>
    // CHECK: [[DIVIDE:%.+]] = IE.Divide([[ARG]], [[CONST]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>}
    // CHECK: [[ADD:%.+]] = IE.Add([[ARG]], [[CONST]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>}
    // CHECK: return [[DIVIDE]], [[ADD]]
}

// -----

// CHECK-LABEL: @ConvertDivideWithConstQuantizedDivisor
// CHECK-SAME: ([[ARG:%.+]]: tensor<1x3x768x1152xf16>) -> tensor<1x3x768x1152xf16>
func.func @ConvertDivideWithConstQuantizedDivisor(%arg0: tensor<1x3x768x1152xf16>) -> tensor<1x3x768x1152xf16> {
    %weights = const.Declare tensor<1x3x1x1xf16> = dense<[[[223]], [[217]], [[219]]]> : tensor<3x1x1xui8>, [#const.Reshape<[1, 3, 1, 1]>, #const.CastElemType<f16>]
    %input_low = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
    %input_high = const.Declare tensor<1x1x1x1xf16> = dense<2.550000e+02> : tensor<1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
    %output_low = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
    %output_high = const.Declare tensor<1x1x1x1xf16> = dense<60.6930428> : tensor<1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]

    %0 = IE.FakeQuantize(%weights, %input_low, %input_high, %output_low, %output_high)
        {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} :
        tensor<1x3x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x3x1x1xf16>

    %1 = IE.Divide(%arg0, %0)
        {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
        tensor<1x3x768x1152xf16>, tensor<1x3x1x1xf16> -> tensor<1x3x768x1152xf16>

    return %1 : tensor<1x3x768x1152xf16>

    // CHECK-DAG:       [[WEIGHTS:%.+]] = const.Declare tensor<1x3x1x1xf16>
    // CHECK-SAME{LITERAL}: = dense<[[[223]], [[217]], [[219]]]> : tensor<3x1x1xui8>, [#const.Reshape<[1, 3, 1, 1]>, #const.CastElemType<f16>, #const.ScalarMultInverse]
    // CHECK-DAG:       [[IN_LOW:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf16>
    // CHECK-DAG:       [[IN_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<4.608150e-03> : tensor<1x1x1x1xf16>
    // CHECK-DAG:       [[OUT_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<1.936340e-02> : tensor<1x1x1x1xf16>
    // CHECK:           [[FAKE_QUANTIZE:%.+]] = IE.FakeQuantize([[WEIGHTS]], [[IN_LOW]], [[IN_HIGH]], [[IN_LOW]], [[OUT_HIGH]])
    // CHECK-SAME:          {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64}
    // CHECK:           [[MULTIPLY:%.+]] = IE.Multiply([[ARG]], [[FAKE_QUANTIZE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK:           return [[MULTIPLY]]
}

// -----

// CHECK-LABEL: @ConvertMultipleDivideOpsWithConstQuantizedDivisor
// CHECK-SAME: ([[ARG:%.+]]: tensor<1x3x768x1152xf16>) -> (tensor<1x3x768x1152xf16>, tensor<1x3x768x1152xf16>)
func.func @ConvertMultipleDivideOpsWithConstQuantizedDivisor(%arg0: tensor<1x3x768x1152xf16>) -> (tensor<1x3x768x1152xf16>, tensor<1x3x768x1152xf16>) {
    %weights = const.Declare tensor<1x3x1x1xf16> = dense<[[[223]], [[217]], [[219]]]> : tensor<3x1x1xui8>, [#const.Reshape<[1, 3, 1, 1]>, #const.CastElemType<f16>]
    %input_low = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
    %input_high = const.Declare tensor<1x1x1x1xf16> = dense<2.550000e+02> : tensor<1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
    %output_low = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
    %output_high = const.Declare tensor<1x1x1x1xf16> = dense<60.6930428> : tensor<1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]

    %0 = IE.FakeQuantize(%weights, %input_low, %input_high, %output_low, %output_high)
        {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} :
        tensor<1x3x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x3x1x1xf16>

    %1 = IE.Divide(%arg0, %0)
        {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
        tensor<1x3x768x1152xf16>, tensor<1x3x1x1xf16> -> tensor<1x3x768x1152xf16>

    %2 = IE.Divide(%arg0, %0)
        {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
        tensor<1x3x768x1152xf16>, tensor<1x3x1x1xf16> -> tensor<1x3x768x1152xf16>

    return %1, %2 : tensor<1x3x768x1152xf16>, tensor<1x3x768x1152xf16>

    // CHECK-DAG:       [[WEIGHTS:%.+]] = const.Declare tensor<1x3x1x1xf16>
    // CHECK-SAME{LITERAL}: = dense<[[[223]], [[217]], [[219]]]> : tensor<3x1x1xui8>, [#const.Reshape<[1, 3, 1, 1]>, #const.CastElemType<f16>, #const.ScalarMultInverse]
    // CHECK-DAG:       [[IN_LOW:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf16>
    // CHECK-DAG:       [[IN_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<4.608150e-03> : tensor<1x1x1x1xf16>
    // CHECK-DAG:       [[OUT_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<1.936340e-02> : tensor<1x1x1x1xf16>
    // CHECK:           [[FAKE_QUANTIZE:%.+]] = IE.FakeQuantize([[WEIGHTS]], [[IN_LOW]], [[IN_HIGH]], [[IN_LOW]], [[OUT_HIGH]])
    // CHECK-SAME:          {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64}
    // CHECK:           [[MULTIPLY0:%.+]] = IE.Multiply([[ARG]], [[FAKE_QUANTIZE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK:           [[MULTIPLY1:%.+]] = IE.Multiply([[ARG]], [[FAKE_QUANTIZE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK:           return [[MULTIPLY0]], [[MULTIPLY1]]
}

// -----

// CHECK-LABEL: @DoNotConvertMultipleDivideOpsWithQuantizedDivisor_NotSecondInput
// CHECK-SAME: ([[ARG:%.+]]: tensor<1x3x768x1152xf16>) -> (tensor<1x3x768x1152xf16>, tensor<1x3x768x1152xf16>)
func.func @DoNotConvertMultipleDivideOpsWithQuantizedDivisor_NotSecondInput(%arg0: tensor<1x3x768x1152xf16>) -> (tensor<1x3x768x1152xf16>, tensor<1x3x768x1152xf16>) {
    %weights = const.Declare tensor<1x3x1x1xf16> = dense<[[[223]], [[217]], [[219]]]> : tensor<3x1x1xui8>, [#const.Reshape<[1, 3, 1, 1]>, #const.CastElemType<f16>]
    %input_low = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
    %input_high = const.Declare tensor<1x1x1x1xf16> = dense<2.550000e+02> : tensor<1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
    %output_low = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
    %output_high = const.Declare tensor<1x1x1x1xf16> = dense<60.6930428> : tensor<1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]

    %0 = IE.FakeQuantize(%weights, %input_low, %input_high, %output_low, %output_high)
        {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} :
        tensor<1x3x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x3x1x1xf16>

    %1 = IE.Divide(%0, %arg0)
        {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
        tensor<1x3x1x1xf16>, tensor<1x3x768x1152xf16> -> tensor<1x3x768x1152xf16>

    %2 = IE.Divide(%arg0, %0)
        {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
        tensor<1x3x768x1152xf16>, tensor<1x3x1x1xf16> -> tensor<1x3x768x1152xf16>

    return %1, %2 : tensor<1x3x768x1152xf16>, tensor<1x3x768x1152xf16>

    // CHECK:           [[WEIGHTS:%.+]] = const.Declare tensor<1x3x1x1xf16> {{.*}} [#const.Reshape<[1, 3, 1, 1]>, #const.CastElemType<f16>]
    // CHECK:           [[IN_LOW:%.+]] = const.Declare tensor<1x1x1x1xf16> {{.*}} [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
    // CHECK:           [[IN_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf16> {{.*}} [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
    // CHECK:           [[OUT_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf16> {{.*}} [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
    // CHECK:           [[FAKE_QUANTIZE:%.+]] = IE.FakeQuantize([[WEIGHTS]], [[IN_LOW]], [[IN_HIGH]], [[IN_LOW]], [[OUT_HIGH]])
    // CHECK-SAME:          {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64}
    // CHECK:           [[DIVIDE0:%.+]] = IE.Divide([[FAKE_QUANTIZE]], [[ARG]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK:           [[DIVIDE1:%.+]] = IE.Divide([[ARG]], [[FAKE_QUANTIZE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK:           return [[DIVIDE0]], [[DIVIDE1]]
}
