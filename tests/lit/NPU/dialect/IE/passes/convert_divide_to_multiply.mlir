//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-divide-to-multiply --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX


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
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x12x512x512xf16>, [[ARG1:%.+]]: tensor<1x12x512x512xf16>) -> (tensor<1x12x512x512xf16>, tensor<1x12x512x512xf16>)
func.func @ConvertMultipleDivideOps(%arg0: tensor<1x12x512x512xf16>, %arg1: tensor<1x12x512x512xf16>) -> (tensor<1x12x512x512xf16>, tensor<1x12x512x512xf16>) {
    %divisor = const.Declare tensor<1x12x512x512xf16> = dense<2.0> : tensor<1x12x512x512xf16>
    %0 = IE.Divide(%arg0, %divisor) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>}
        : tensor<1x12x512x512xf16>, tensor<1x12x512x512xf16> -> tensor<1x12x512x512xf16>
    %1 = IE.Divide(%arg1, %divisor) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>}
        : tensor<1x12x512x512xf16>, tensor<1x12x512x512xf16> -> tensor<1x12x512x512xf16>
    return %0, %1 : tensor<1x12x512x512xf16>, tensor<1x12x512x512xf16>

    // CHECK:     [[CONST:%.+]] = const.Declare tensor<1x12x512x512xf16> = dense<2.000000e+00> : tensor<1x12x512x512xf16>, [#const.ScalarMultInverse]
    // CHECK-DAG: [[MULTIPLY0:%.+]] = IE.Multiply([[ARG0]], [[CONST]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>}
    // CHECK-DAG: [[MULTIPLY1:%.+]] = IE.Multiply([[ARG1]], [[CONST]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>}
    // CHECK:     return [[MULTIPLY0]], [[MULTIPLY1]]
}

// -----

// CHECK-LABEL: @ConvertMultipleDivideOps_NotSecondInput
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x12x512x512xf16>, [[ARG1:%.+]]: tensor<1x12x512x512xf16>) -> (tensor<1x12x512x512xf16>, tensor<1x12x512x512xf16>)
func.func @ConvertMultipleDivideOps_NotSecondInput(%arg0: tensor<1x12x512x512xf16>, %arg1: tensor<1x12x512x512xf16>) -> (tensor<1x12x512x512xf16>, tensor<1x12x512x512xf16>) {
    %divisor = const.Declare tensor<1x12x512x512xf16> = dense<2.0> : tensor<1x12x512x512xf16>
    %0 = IE.Divide(%arg0, %divisor) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>}
        : tensor<1x12x512x512xf16>, tensor<1x12x512x512xf16> -> tensor<1x12x512x512xf16>
    %1 = IE.Divide(%divisor, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>}
        : tensor<1x12x512x512xf16>, tensor<1x12x512x512xf16> -> tensor<1x12x512x512xf16>
    return %0, %1 : tensor<1x12x512x512xf16>, tensor<1x12x512x512xf16>

    // CHECK: [[CONST0:%.+]] = const.Declare tensor<1x12x512x512xf16> = dense<2.000000e+00> : tensor<1x12x512x512xf16>, [#const.ScalarMultInverse]
    // CHECK: [[CONST1:%.+]] = const.Declare tensor<1x12x512x512xf16> = dense<2.000000e+00> : tensor<1x12x512x512xf16>
    // CHECK: [[MULTIPLY:%.+]] = IE.Multiply([[ARG0]], [[CONST0]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>}
    // CHECK: [[DIVIDE:%.+]] = IE.Divide([[CONST1]], [[ARG1]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>}
    // CHECK: return [[MULTIPLY]], [[DIVIDE]]
}

// -----

// CHECK-LABEL: @ConvertNotOnlyDivideUsers
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x12x512x512xf16>, [[ARG1:%.+]]: tensor<1x12x512x512xf16>) -> (tensor<1x12x512x512xf16>, tensor<1x12x512x512xf16>)
func.func @ConvertNotOnlyDivideUsers(%arg0: tensor<1x12x512x512xf16>, %arg1: tensor<1x12x512x512xf16>) -> (tensor<1x12x512x512xf16>, tensor<1x12x512x512xf16>) {
    %cst = const.Declare tensor<1x12x512x512xf16> = dense<2.0> : tensor<1x12x512x512xf16>
    %0 = IE.Divide(%arg0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>}
        : tensor<1x12x512x512xf16>, tensor<1x12x512x512xf16> -> tensor<1x12x512x512xf16>
    %1 = IE.Add(%arg1, %cst) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>}
        : tensor<1x12x512x512xf16>, tensor<1x12x512x512xf16> -> tensor<1x12x512x512xf16>
    return %0, %1 : tensor<1x12x512x512xf16>, tensor<1x12x512x512xf16>

    // CHECK: [[CONST0:%.+]] = const.Declare tensor<1x12x512x512xf16> = dense<2.000000e+00> : tensor<1x12x512x512xf16>, [#const.ScalarMultInverse]
    // CHECK: [[CONST1:%.+]] = const.Declare tensor<1x12x512x512xf16> = dense<2.000000e+00> : tensor<1x12x512x512xf16>
    // CHECK: [[MULTIPLY:%.+]] = IE.Multiply([[ARG0]], [[CONST0]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>}
    // CHECK: [[ADD:%.+]] = IE.Add([[ARG1]], [[CONST1]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>}
    // CHECK: return [[MULTIPLY]], [[ADD]]
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

!qElemType = !quant.uniform<ui8:f16, 0.57450980392156858:128>
// CHECK-DAG: [[ORIGINAL_QELEMTYPE:!.+]] = !quant.uniform<u8:f16, 0.57450980392156858:128>
// CHECK-DAG: [[NEW_QELEMTYPE:!.+]] = !quant.uniform<u8:f16, 0.0067992747440273043:64>

// CHECK: @ConvertDivideWithConstDequantizedDivisor
// CHECK-SAME: ([[ARG:%.+]]: tensor<1x3x768x1152xf16>) -> tensor<1x3x768x1152xf16>
func.func @ConvertDivideWithConstDequantizedDivisor(%arg0: tensor<1x3x768x1152xf16>) -> tensor<1x3x768x1152xf16> {
    %input = const.Declare tensor<1x3x1x1x!qElemType> = dense<132.0> : tensor<1x3x1x1xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]
    %0 = IE.Dequantize(%input) {dstElemType = f16} : tensor<1x3x1x1x!qElemType> -> tensor<1x3x1x1xf16>

    %1 = IE.Divide(%arg0, %0)
        {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
        tensor<1x3x768x1152xf16>, tensor<1x3x1x1xf16> -> tensor<1x3x768x1152xf16>

    return %1 : tensor<1x3x768x1152xf16>

    // CHECK-DAG:       [[CST:%.+]] = const.Declare tensor<1x3x1x1x[[NEW_QELEMTYPE]]> = dense<1.320000e+02> : tensor<1x3x1x1xf16>, [#const.CastElemType<ui8>, #const.CastElemType<[[ORIGINAL_QELEMTYPE]]>, #const.Add<-1.280000e+02 : f64>, #const.CastElemType<f32>, #const.ScalarMultInverse, #const.Rescale<2.560000e+02 : f64>, #const.Add<6.400000e+01 : f64>, #const.CastElemType<ui8>]
    // CHECK-DAG:       [[DQ:%.+]] = IE.Dequantize([[CST]]) {dstElemType = f16} : tensor<1x3x1x1x[[NEW_QELEMTYPE]]> -> tensor<1x3x1x1xf16>
    // CHECK-DAG:       [[MULTIPLY:%.+]] = IE.Multiply([[ARG]], [[DQ]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x768x1152xf16>, tensor<1x3x1x1xf16> -> tensor<1x3x768x1152xf16>
    // CHECK-DAG:       return [[MULTIPLY]] : tensor<1x3x768x1152xf16>
}

// -----

!qElemType = !quant.uniform<i4:f16, 1.000000e+00:8>
// CHECK-DAG: [[QELEMTYPE:!.+]] = !quant.uniform<i4:f16, 1.000000e+00:8>

// CHECK: @DoNotConvertDivideWithConstDequantizedDivisorI4
// CHECK-SAME: ([[ARG:%.+]]: tensor<1x3x768x1152xf16>) -> tensor<1x3x768x1152xf16>
func.func @DoNotConvertDivideWithConstDequantizedDivisorI4(%arg0: tensor<1x3x768x1152xf16>) -> tensor<1x3x768x1152xf16> {
    %input = const.Declare tensor<1x3x1x1x!qElemType> = dense<132.0> : tensor<1x3x1x1xf16>, [#const.CastElemType<i4>, #const.CastElemType<!qElemType>]
    %0 = IE.Dequantize(%input) {dstElemType = f16} : tensor<1x3x1x1x!qElemType> -> tensor<1x3x1x1xf16>

    %1 = IE.Divide(%arg0, %0)
        {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
        tensor<1x3x768x1152xf16>, tensor<1x3x1x1xf16> -> tensor<1x3x768x1152xf16>

    return %1 : tensor<1x3x768x1152xf16>

    // CHECK-DAG:       [[CST:%.+]] = const.Declare tensor<1x3x1x1x[[QELEMTYPE]]> = dense<1.320000e+02> : tensor<1x3x1x1xf16>, [#const.CastElemType<i4>, #const.CastElemType<[[QELEMTYPE]]>]
    // CHECK-DAG:       [[DQ:%.+]] = IE.Dequantize([[CST]]) {dstElemType = f16} : tensor<1x3x1x1x[[QELEMTYPE]]> -> tensor<1x3x1x1xf16>
    // CHECK-DAG:       [[DIVIDE:%.+]] = IE.Divide([[ARG]], [[DQ]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x768x1152xf16>, tensor<1x3x1x1xf16> -> tensor<1x3x768x1152xf16>
    // CHECK-DAG:       return [[DIVIDE]] : tensor<1x3x768x1152xf16>
}

// -----

!qElemType = !quant.uniform<f8E4M3FN:f16, 1.000000e+00>
// CHECK-DAG: [[QELEMTYPE:!.+]] = !quant.uniform<f8E4M3FN:f16, 1.000000e+00>

// CHECK: @DoNotConvertDivideWithConstDequantizedDivisorFP8
// CHECK-SAME: ([[ARG:%.+]]: tensor<1x3x768x1152xf16>) -> tensor<1x3x768x1152xf16>
func.func @DoNotConvertDivideWithConstDequantizedDivisorFP8(%arg0: tensor<1x3x768x1152xf16>) -> tensor<1x3x768x1152xf16> {
    %input = const.Declare tensor<1x3x1x1x!qElemType> = dense<132.0> : tensor<1x3x1x1xf16>, [#const.CastElemType<f8E4M3FN>, #const.CastElemType<!qElemType>]
    %0 = IE.Dequantize(%input) {dstElemType = f16} : tensor<1x3x1x1x!qElemType> -> tensor<1x3x1x1xf16>

    %1 = IE.Divide(%arg0, %0)
        {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
        tensor<1x3x768x1152xf16>, tensor<1x3x1x1xf16> -> tensor<1x3x768x1152xf16>

    return %1 : tensor<1x3x768x1152xf16>

    // CHECK-DAG:       [[CST:%.+]] = const.Declare tensor<1x3x1x1x[[QELEMTYPE]]> = dense<1.320000e+02> : tensor<1x3x1x1xf16>, [#const.CastElemType<f8E4M3FN>, #const.CastElemType<[[QELEMTYPE]]>]
    // CHECK-DAG:       [[DQ:%.+]] = IE.Dequantize([[CST]]) {dstElemType = f16} : tensor<1x3x1x1x[[QELEMTYPE]]> -> tensor<1x3x1x1xf16>
    // CHECK-DAG:       [[DIVIDE:%.+]] = IE.Divide([[ARG]], [[DQ]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x768x1152xf16>, tensor<1x3x1x1xf16> -> tensor<1x3x768x1152xf16>
    // CHECK-DAG:       return [[DIVIDE]] : tensor<1x3x768x1152xf16>
}

// -----

!qElemType = !quant.uniform<i8:f32:1, {0.10000000149011612:100, 0.20000000298023224:100, 0.30000001192092896:100}>
// CHECK-DAG: [[QELEMTYPE:!.+]] = !quant.uniform<i8:f32:1, {0.10000000149011612:100,0.20000000298023224:100,0.30000001192092896:100}>

// CHECK: @DoNotConvertDivideWithConstDequantizedDivisorPerAxisQuant
// CHECK-SAME: ([[ARG:%.+]]: tensor<1x3x768x1152xf16>) -> tensor<1x3x768x1152xf16>
func.func @DoNotConvertDivideWithConstDequantizedDivisorPerAxisQuant(%arg0: tensor<1x3x768x1152xf16>) -> tensor<1x3x768x1152xf16> {
    %input = const.Declare tensor<1x3x1x1x!qElemType> = dense<132.0> : tensor<1x3x1x1xf16>, [#const.CastElemType<i8>, #const.CastElemType<!qElemType>]
    %0 = IE.Dequantize(%input) {dstElemType = f16} : tensor<1x3x1x1x!qElemType> -> tensor<1x3x1x1xf16>

    %1 = IE.Divide(%arg0, %0)
        {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
        tensor<1x3x768x1152xf16>, tensor<1x3x1x1xf16> -> tensor<1x3x768x1152xf16>

    return %1 : tensor<1x3x768x1152xf16>

    // CHECK-DAG:       [[CST:%.+]] = const.Declare tensor<1x3x1x1x[[QELEMTYPE]]> = dense<1.320000e+02> : tensor<1x3x1x1xf16>, [#const.CastElemType<i8>, #const.CastElemType<[[QELEMTYPE]]>]
    // CHECK-DAG:       [[DQ:%.+]] = IE.Dequantize([[CST]]) {dstElemType = f16} : tensor<1x3x1x1x[[QELEMTYPE]]> -> tensor<1x3x1x1xf16>
    // CHECK-DAG:       [[DIVIDE:%.+]] = IE.Divide([[ARG]], [[DQ]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x768x1152xf16>, tensor<1x3x1x1xf16> -> tensor<1x3x768x1152xf16>
    // CHECK-DAG:       return [[DIVIDE]] : tensor<1x3x768x1152xf16>
}

// -----

// CHECK-LABEL: @ConvertMultipleDivideOpsWithConstQuantizedDivisor
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x3x768x1152xf16>, [[ARG1:%.+]]: tensor<1x3x768x1152xf16>) -> (tensor<1x3x768x1152xf16>, tensor<1x3x768x1152xf16>)
func.func @ConvertMultipleDivideOpsWithConstQuantizedDivisor(%arg0: tensor<1x3x768x1152xf16>, %arg1: tensor<1x3x768x1152xf16>) -> (tensor<1x3x768x1152xf16>, tensor<1x3x768x1152xf16>) {
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

    %2 = IE.Divide(%arg1, %0)
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
    // CHECK:           [[MULTIPLY0:%.+]] = IE.Multiply([[ARG0]], [[FAKE_QUANTIZE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK:           [[MULTIPLY1:%.+]] = IE.Multiply([[ARG1]], [[FAKE_QUANTIZE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK:           return [[MULTIPLY0]], [[MULTIPLY1]]
}

// -----

!qElemType = !quant.uniform<ui8:f16, 0.57450980392156858:128>
// CHECK-DAG: [[ORIGINAL_QELEMTYPE:!.+]] = !quant.uniform<u8:f16, 0.57450980392156858:128>
// CHECK-DAG: [[NEW_QELEMTYPE:!.+]] = !quant.uniform<u8:f16, 0.0067992747440273043:-2>

// CHECK: @ConvertMultipleDivideOpsWithConstDequantizedDivisor
// CHECK-SAME: ([[ARG_0:%.+]]: tensor<1x3x768x1152xf16>, [[ARG_1:%.+]]: tensor<1x3x768x1152xf16>) -> (tensor<1x3x768x1152xf16>, tensor<1x3x768x1152xf16>)
func.func @ConvertMultipleDivideOpsWithConstDequantizedDivisor(%arg0: tensor<1x3x768x1152xf16>, %arg1: tensor<1x3x768x1152xf16>) -> (tensor<1x3x768x1152xf16>, tensor<1x3x768x1152xf16>) {
    %input = const.Declare tensor<1x3x1x1x!qElemType> = dense<1.0> : tensor<1x3x1x1xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]
    %0 = IE.Dequantize(%input) {dstElemType = f16} : tensor<1x3x1x1x!qElemType> -> tensor<1x3x1x1xf16>

    %1 = IE.Divide(%arg0, %0)
        {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
        tensor<1x3x768x1152xf16>, tensor<1x3x1x1xf16> -> tensor<1x3x768x1152xf16>

    %2 = IE.Divide(%arg1, %0)
        {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
        tensor<1x3x768x1152xf16>, tensor<1x3x1x1xf16> -> tensor<1x3x768x1152xf16>

    return %1, %2 : tensor<1x3x768x1152xf16>, tensor<1x3x768x1152xf16>

    // CHECK-DAG:       [[CST:%.+]] = const.Declare tensor<1x3x1x1x[[NEW_QELEMTYPE]]> = dense<1.000000e+00> : tensor<1x3x1x1xf16>, [#const.CastElemType<ui8>, #const.CastElemType<[[ORIGINAL_QELEMTYPE]]>, #const.Add<-1.280000e+02 : f64>, #const.CastElemType<f32>, #const.ScalarMultInverse, #const.Rescale<2.560000e+02 : f64>, #const.Add<-2.000000e+00 : f64>, #const.CastElemType<ui8>]
    // CHECK-DAG:       [[DQ:%.+]] = IE.Dequantize([[CST]]) {dstElemType = f16} : tensor<1x3x1x1x[[NEW_QELEMTYPE]]> -> tensor<1x3x1x1xf16>
    // CHECK-DAG:       [[MULTIPLY_0:%.+]] = IE.Multiply([[ARG_0]], [[DQ]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x768x1152xf16>, tensor<1x3x1x1xf16> -> tensor<1x3x768x1152xf16>
    // CHECK-DAG:       [[MULTIPLY_1:%.+]] = IE.Multiply([[ARG_1]], [[DQ]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x768x1152xf16>, tensor<1x3x1x1xf16> -> tensor<1x3x768x1152xf16>
    // CHECK-DAG:       return [[MULTIPLY_0]], [[MULTIPLY_1]] : tensor<1x3x768x1152xf16>, tensor<1x3x768x1152xf16>
}

// -----

// CHECK-LABEL: @ConvertMultipleDivideOpsWithConstQuantizedDivisor_NotOnlyDivideUsers
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x3x768x1152xf16>, [[ARG1:%.+]]: tensor<1x3x768x1152xf16>) -> (tensor<1x3x768x1152xf16>, tensor<1x3x768x1152xf16>)
func.func @ConvertMultipleDivideOpsWithConstQuantizedDivisor_NotOnlyDivideUsers(%arg0: tensor<1x3x768x1152xf16>, %arg1: tensor<1x3x768x1152xf16>) -> (tensor<1x3x768x1152xf16>, tensor<1x3x768x1152xf16>) {
    %weights = const.Declare tensor<1x3x1x1xf16> = dense<[[[223]], [[217]], [[219]]]> : tensor<3x1x1xui8>, [#const.Reshape<[1, 3, 1, 1]>, #const.CastElemType<f16>]
    %input_low = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
    %input_high = const.Declare tensor<1x1x1x1xf16> = dense<2.550000e+02> : tensor<1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
    %output_low = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
    %output_high = const.Declare tensor<1x1x1x1xf16> = dense<60.6930428> : tensor<1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]

    %0 = IE.FakeQuantize(%weights, %input_low, %input_high, %output_low, %output_high)
        {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} :
        tensor<1x3x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x3x1x1xf16>

    %1 = IE.Add(%arg0, %0)
        {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
        tensor<1x3x768x1152xf16>, tensor<1x3x1x1xf16> -> tensor<1x3x768x1152xf16>

    %2 = IE.Divide(%arg1, %0)
        {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
        tensor<1x3x768x1152xf16>, tensor<1x3x1x1xf16> -> tensor<1x3x768x1152xf16>

    return %1, %2 : tensor<1x3x768x1152xf16>, tensor<1x3x768x1152xf16>

    // CHECK-DAG:       [[CONST0:%.+]] = const.Declare tensor<1x3x1x1xf16>
    // CHECK-SAME{LITERAL}: = dense<[[[223]], [[217]], [[219]]]> : tensor<3x1x1xui8>, [#const.Reshape<[1, 3, 1, 1]>, #const.CastElemType<f16>, #const.ScalarMultInverse]
    // CHECK-DAG:       [[CONST1:%.+]] = const.Declare tensor<1x3x1x1xf16>
    // CHECK-SAME{LITERAL}: = dense<[[[223]], [[217]], [[219]]]> : tensor<3x1x1xui8>, [#const.Reshape<[1, 3, 1, 1]>, #const.CastElemType<f16>]

    // CHECK-DAG:       [[IN_LOW1:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
    // CHECK-DAG:       [[IN_HIGH1:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<2.550000e+02> : tensor<1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
    // CHECK-DAG:       [[OUT_HIGH1:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<60.6930428> : tensor<1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]

    // CHECK-DAG:       [[IN_LOW0:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf16>
    // CHECK-DAG:       [[IN_HIGH0:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<4.608150e-03> : tensor<1x1x1x1xf16>
    // CHECK-DAG:       [[OUT_HIGH0:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<1.936340e-02> : tensor<1x1x1x1xf16>

    // CHECK:           [[FAKE_QUANTIZE0:%.+]] = IE.FakeQuantize([[CONST0]], [[IN_LOW0]], [[IN_HIGH0]], [[IN_LOW0]], [[OUT_HIGH0]])
    // CHECK-SAME:          {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64}
    // CHECK:           [[FAKE_QUANTIZE1:%.+]] = IE.FakeQuantize([[CONST1]], [[IN_LOW1]], [[IN_HIGH1]], [[IN_LOW1]], [[OUT_HIGH1]])
    // CHECK-SAME:          {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64}

    // CHECK:           [[ADD:%.+]] = IE.Add([[ARG0]], [[FAKE_QUANTIZE1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK:           [[MULTIPLY:%.+]] = IE.Multiply([[ARG1]], [[FAKE_QUANTIZE0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK:           return [[ADD]], [[MULTIPLY]]
}

// -----

!qElemType = !quant.uniform<ui8:f16, 0.57450980392156858:128>
//CHECK-DAG: [[ORIGINAL_QELEMTYPE:!.+]] = !quant.uniform<u8:f16, 0.57450980392156858:128>
//CHECK-DAG: [[NEW_QELEMTYPE:!.+]] = !quant.uniform<u8:f16, 0.0067992747440273043:-2>

// CHECK: @ConvertMultipleDivideOpsWithConstDequantizedDivisor_NotOnlyDivideUsers
// CHECK-SAME: ([[ARG_0:%.+]]: tensor<1x3x768x1152xf16>, [[ARG_1:%.+]]: tensor<1x3x768x1152xf16>) -> (tensor<1x3x768x1152xf16>, tensor<1x3x768x1152xf16>)
func.func @ConvertMultipleDivideOpsWithConstDequantizedDivisor_NotOnlyDivideUsers(%arg0: tensor<1x3x768x1152xf16>, %arg1: tensor<1x3x768x1152xf16>) -> (tensor<1x3x768x1152xf16>, tensor<1x3x768x1152xf16>) {
    %input = const.Declare tensor<1x3x1x1x!qElemType> = dense<0.0> : tensor<1x3x1x1xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]
    %0 = IE.Dequantize(%input) {dstElemType = f16} : tensor<1x3x1x1x!qElemType> -> tensor<1x3x1x1xf16>

    %1 = IE.Add(%arg0, %0)
        {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
        tensor<1x3x768x1152xf16>, tensor<1x3x1x1xf16> -> tensor<1x3x768x1152xf16>

    %2 = IE.Divide(%arg1, %0)
        {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
        tensor<1x3x768x1152xf16>, tensor<1x3x1x1xf16> -> tensor<1x3x768x1152xf16>

    return %1, %2 : tensor<1x3x768x1152xf16>, tensor<1x3x768x1152xf16>

    // CHECK-DAG:       [[CST_0:%.+]] = const.Declare tensor<1x3x1x1x[[NEW_QELEMTYPE]]> = dense<0.000000e+00> : tensor<1x3x1x1xf16>, [#const.CastElemType<ui8>, #const.CastElemType<[[ORIGINAL_QELEMTYPE]]>, #const.Add<-1.280000e+02 : f64>, #const.CastElemType<f32>, #const.ScalarMultInverse, #const.Rescale<2.560000e+02 : f64>, #const.Add<-2.000000e+00 : f64>, #const.CastElemType<ui8>]
    // CHECK-DAG:       [[CST_1:%.+]] = const.Declare tensor<1x3x1x1x[[ORIGINAL_QELEMTYPE]]> = dense<0.000000e+00> : tensor<1x3x1x1xf16>, [#const.CastElemType<ui8>, #const.CastElemType<[[ORIGINAL_QELEMTYPE]]>]
    // CHECK-DAG:       [[DQ_0:%.+]] = IE.Dequantize([[CST_0]]) {dstElemType = f16} : tensor<1x3x1x1x[[NEW_QELEMTYPE]]> -> tensor<1x3x1x1xf16>
    // CHECK-DAG:       [[DQ_1:%.+]] = IE.Dequantize([[CST_1]]) {dstElemType = f16} : tensor<1x3x1x1x[[ORIGINAL_QELEMTYPE]]> -> tensor<1x3x1x1xf16>
    // CHECK-DAG:       [[ADD:%.+]] = IE.Add([[ARG_0]], [[DQ_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x768x1152xf16>, tensor<1x3x1x1xf16> -> tensor<1x3x768x1152xf16>
    // CHECK-DAG:       [[MULTIPLY:%.+]] = IE.Multiply([[ARG_1]], [[DQ_0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x768x1152xf16>, tensor<1x3x1x1xf16> -> tensor<1x3x768x1152xf16>
    // CHECK-DAG:       return [[ADD]], [[MULTIPLY]] : tensor<1x3x768x1152xf16>, tensor<1x3x768x1152xf16>
}

// -----

// CHECK-LABEL: @ConvertMultipleDivideOpsWithQuantizedDivisor_NotSecondInput
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x3x768x1152xf16>, [[ARG1:%.+]]: tensor<1x3x768x1152xf16>) -> (tensor<1x3x768x1152xf16>, tensor<1x3x768x1152xf16>)
func.func @ConvertMultipleDivideOpsWithQuantizedDivisor_NotSecondInput(%arg0: tensor<1x3x768x1152xf16>, %arg1: tensor<1x3x768x1152xf16>) -> (tensor<1x3x768x1152xf16>, tensor<1x3x768x1152xf16>) {
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

    %2 = IE.Divide(%arg1, %0)
        {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
        tensor<1x3x768x1152xf16>, tensor<1x3x1x1xf16> -> tensor<1x3x768x1152xf16>

    return %1, %2 : tensor<1x3x768x1152xf16>, tensor<1x3x768x1152xf16>

    // CHECK-DAG:       [[CONST0:%.+]] = const.Declare tensor<1x3x1x1xf16>
    // CHECK-SAME{LITERAL}: = dense<[[[223]], [[217]], [[219]]]> : tensor<3x1x1xui8>, [#const.Reshape<[1, 3, 1, 1]>, #const.CastElemType<f16>, #const.ScalarMultInverse]
    // CHECK-DAG:       [[CONST1:%.+]] = const.Declare tensor<1x3x1x1xf16>
    // CHECK-SAME{LITERAL}: = dense<[[[223]], [[217]], [[219]]]> : tensor<3x1x1xui8>, [#const.Reshape<[1, 3, 1, 1]>, #const.CastElemType<f16>]

    // CHECK:           [[IN_LOW1:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
    // CHECK:           [[IN_HIGH1:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<2.550000e+02> : tensor<1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
    // CHECK:           [[OUT_HIGH1:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<60.6930428> : tensor<1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]

    // CHECK:           [[IN_LOW0:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf16>
    // CHECK:           [[IN_HIGH0:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<4.608150e-03> : tensor<1x1x1x1xf16>
    // CHECK:           [[OUT_HIGH0:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<1.936340e-02> : tensor<1x1x1x1xf16>

    // CHECK:           [[FAKE_QUANTIZE0:%.+]] = IE.FakeQuantize([[CONST0]], [[IN_LOW0]], [[IN_HIGH0]], [[IN_LOW0]], [[OUT_HIGH0]])
    // CHECK-SAME:          {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64}
    // CHECK:           [[FAKE_QUANTIZE1:%.+]] = IE.FakeQuantize([[CONST1]], [[IN_LOW1]], [[IN_HIGH1]], [[IN_LOW1]], [[OUT_HIGH1]])
    // CHECK-SAME:          {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64}

    // CHECK:           [[DIVIDE:%.+]] = IE.Divide([[FAKE_QUANTIZE1]], [[ARG0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK:           [[MULTIPLY:%.+]] = IE.Multiply([[ARG1]], [[FAKE_QUANTIZE0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK:           return [[DIVIDE]], [[MULTIPLY]]
}

// -----

!qElemType = !quant.uniform<ui8:f16, 0.57450980392156858:128>
// CHECK-DAG: [[ORIGINAL_QELEMTYPE:!.+]] = !quant.uniform<u8:f16, 0.57450980392156858:128>
// CHECK-DAG: [[NEW_QELEMTYPE:!.+]] = !quant.uniform<u8:f16, 0.0067992747440273043:2>

// CHECK: @ConvertMultipleDivideOpsWithDequantizedDivisor_NotSecondInput
// CHECK-SAME: ([[ARG_0:%.+]]: tensor<1x3x768x1152xf16>, [[ARG_1:%.+]]: tensor<1x3x768x1152xf16>) -> (tensor<1x3x768x1152xf16>, tensor<1x3x768x1152xf16>)
func.func @ConvertMultipleDivideOpsWithDequantizedDivisor_NotSecondInput(%arg0: tensor<1x3x768x1152xf16>, %arg1: tensor<1x3x768x1152xf16>) -> (tensor<1x3x768x1152xf16>, tensor<1x3x768x1152xf16>) {
    %input = const.Declare tensor<1x3x1x1x!qElemType> = dense<256.0> : tensor<1x3x1x1xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]
    %0 = IE.Dequantize(%input) {dstElemType = f16} : tensor<1x3x1x1x!qElemType> -> tensor<1x3x1x1xf16>

    %1 = IE.Divide(%0, %arg0)
        {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
        tensor<1x3x1x1xf16>, tensor<1x3x768x1152xf16> -> tensor<1x3x768x1152xf16>

    %2 = IE.Divide(%arg1, %0)
        {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
        tensor<1x3x768x1152xf16>, tensor<1x3x1x1xf16> -> tensor<1x3x768x1152xf16>

    return %1, %2 : tensor<1x3x768x1152xf16>, tensor<1x3x768x1152xf16>

    // CHECK-DAG:       [[CST_0:%.+]] = const.Declare tensor<1x3x1x1x[[NEW_QELEMTYPE]]> = dense<2.560000e+02> : tensor<1x3x1x1xf16>, [#const.CastElemType<ui8>, #const.CastElemType<[[ORIGINAL_QELEMTYPE]]>, #const.Add<-1.280000e+02 : f64>, #const.CastElemType<f32>, #const.ScalarMultInverse, #const.Rescale<2.560000e+02 : f64>, #const.Add<2.000000e+00 : f64>, #const.CastElemType<ui8>]
    // CHECK-DAG:       [[CST_1:%.+]] = const.Declare tensor<1x3x1x1x[[ORIGINAL_QELEMTYPE]]> = dense<2.560000e+02> : tensor<1x3x1x1xf16>, [#const.CastElemType<ui8>, #const.CastElemType<[[ORIGINAL_QELEMTYPE]]>]
    // CHECK-DAG:       [[DQ_0:%.+]] = IE.Dequantize([[CST_0]]) {dstElemType = f16} : tensor<1x3x1x1x[[NEW_QELEMTYPE]]> -> tensor<1x3x1x1xf16>
    // CHECK-DAG:       [[DQ_1:%.+]] = IE.Dequantize([[CST_1]]) {dstElemType = f16} : tensor<1x3x1x1x[[ORIGINAL_QELEMTYPE]]> -> tensor<1x3x1x1xf16>
    // CHECK-DAG:       [[DIVIDE:%.+]] = IE.Divide([[DQ_1]], [[ARG_0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x1x1xf16>, tensor<1x3x768x1152xf16> -> tensor<1x3x768x1152xf16>
    // CHECK-DAG:       [[MULTIPLY:%.+]] = IE.Multiply([[ARG_1]], [[DQ_0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x768x1152xf16>, tensor<1x3x1x1xf16> -> tensor<1x3x768x1152xf16>
    // CHECK-DAG:       return [[DIVIDE]], [[MULTIPLY]] : tensor<1x3x768x1152xf16>, tensor<1x3x768x1152xf16>
}

// -----

// CHECK-LABEL: @NonConstDivisorConvert
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x12x512x512xf32>, [[ARG1:%.+]]: tensor<1x1x1x1xf32>) -> tensor<1x12x512x512xf32>
func.func @NonConstDivisorConvert(%arg0: tensor<1x12x512x512xf32>, %arg1: tensor<1x1x1x1xf32>) -> tensor<1x12x512x512xf32> {
    %0 = IE.Divide(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
        : tensor<1x12x512x512xf32>, tensor<1x1x1x1xf32> -> tensor<1x12x512x512xf32>
    return %0 : tensor<1x12x512x512xf32>

    // CHECK: [[CST:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<1.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK: [[DIVIDE:%.+]] = IE.Divide([[CST]], [[ARG1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x1x1x1xf32>

    // CHECK: [[MULTIPLY:%.+]] = IE.Multiply([[ARG0]], [[DIVIDE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x12x512x512xf32>, tensor<1x1x1x1xf32> -> tensor<1x12x512x512xf32>
    // CHECK:   return   [[MULTIPLY]]
}

// -----

// CHECK-LABEL: @NotConvertForDividendIsSmall
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x1x1x512xf32>, [[ARG1:%.+]]: tensor<1x1x1x1xf32>) -> tensor<1x1x1x512xf32>
func.func @NotConvertForDividendIsSmall(%arg0: tensor<1x1x1x512xf32>, %arg1: tensor<1x1x1x1xf32>) -> tensor<1x1x1x512xf32> {
    %0 = IE.Divide(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
        : tensor<1x1x1x512xf32>, tensor<1x1x1x1xf32> -> tensor<1x1x1x512xf32>
    return %0 : tensor<1x1x1x512xf32>

    // CHECK: [[DIVIDE:%.+]] = IE.Divide
    // CHECK:   return   [[DIVIDE]]
}

// -----

// CHECK-LABEL: @ConvertWhenDivisorNeedsBroadcast
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x1x8192x2048xf16>, [[ARG1:%.+]]: tensor<1x1x8192x1xf16>) -> tensor<1x1x8192x2048xf16>
func.func @ConvertWhenDivisorNeedsBroadcast(%arg0: tensor<1x1x8192x2048xf16>, %arg1: tensor<1x1x8192x1xf16>) -> tensor<1x1x8192x2048xf16> {
    %0 = IE.Divide(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
        : tensor<1x1x8192x2048xf16>, tensor<1x1x8192x1xf16> -> tensor<1x1x8192x2048xf16>

    return %0 : tensor<1x1x8192x2048xf16>

    // CHECK: [[CST:%.+]] = const.Declare tensor<1x1x8192x1xf16> = dense<1.000000e+00> : tensor<1x1x8192x1xf16>
    // CHECK: [[DIVIDE:%.+]] = IE.Divide([[CST]], [[ARG1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x8192x1xf16>, tensor<1x1x8192x1xf16> -> tensor<1x1x8192x1xf16>

    // CHECK: [[MULTIPLY:%.+]] = IE.Multiply([[ARG0]], [[DIVIDE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x8192x2048xf16>, tensor<1x1x8192x1xf16> -> tensor<1x1x8192x2048xf16>
    // CHECK:   return   [[MULTIPLY]]
}
