//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-to-scale-shift %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

// CHECK-LABEL: @ConvertAddToScaleShift
func.func @ConvertAddToScaleShift(%arg0: tensor<1x3x300x300xf16>) -> tensor<1x3x300x300xf16> {
    %bias = const.Declare tensor<1x3x1x1xf16> = dense<2.0> : tensor<1x3x1x1xf16>
    %0 = IE.Add(%arg0, %bias)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
        tensor<1x3x300x300xf16>, tensor<1x3x1x1xf16> -> tensor<1x3x300x300xf16>

    return %0 : tensor<1x3x300x300xf16>

    // CHECK-DAG:       [[BIAS:%.+]] = const.Declare tensor<1x3x1x1xf16> = dense<2.000000e+00> : tensor<1x3x1x1xf16>
    // CHECK:       [[VAL0:%.+]] = IE.ScaleShift(%arg0, [[BIAS]]) {operandSegmentSizes = array<i32: 1, 0, 1>} : tensor<1x3x300x300xf16>, tensor<1x3x1x1xf16> -> tensor<1x3x300x300xf16>
    // CHECK:       return [[VAL0]]
}

// -----

// CHECK-LABEL: @ConvertAddToScaleShiftWithReshape
func.func @ConvertAddToScaleShiftWithReshape(%arg0: tensor<1x16x64x64xf16>) -> tensor<1x16x64x64xf16> {
    %bias = const.Declare tensor<1x1x1x16xf16> = dense<2.0> : tensor<1x1x1x16xf16>
    %reshape = IE.Reshape(%bias) { shape_value = [1, 16, 1, 1] } : tensor<1x1x1x16xf16> -> tensor<1x16x1x1xf16>
    %0 = IE.Add(%arg0, %reshape)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
        tensor<1x16x64x64xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x64x64xf16>

    return %0 : tensor<1x16x64x64xf16>

    // CHECK-DAG:       %[[BIAS:.*]] = const.Declare tensor<1x16x1x1xf16> = dense<2.000000e+00> : tensor<1x1x1x16xf16>, [#const.Reshape<[1, 16, 1, 1]>]
    // CHECK:       %[[VAL0:.*]] = IE.ScaleShift(%arg0, %[[BIAS]]) {operandSegmentSizes = array<i32: 1, 0, 1>} : tensor<1x16x64x64xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x64x64xf16>
    // CHECK:       return %[[VAL0]]
}

// -----

// CHECK-LABEL: @ConvertAddWithNegativeConstToScaleShift
func.func @ConvertAddWithNegativeConstToScaleShift(%arg0: tensor<1x3x224x224xf16>) -> tensor<1x3x224x224xf16> {
    %cst = const.Declare tensor<1x3x1x1xf16> = dense<2.0> : tensor<1x3x1x1xf16>
    %0 = IE.Negative(%cst) : tensor<1x3x1x1xf16> -> tensor<1x3x1x1xf16>
    %1 = IE.Add(%arg0, %0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x224x224xf16>, tensor<1x3x1x1xf16> -> tensor<1x3x224x224xf16>

    return %1 : tensor<1x3x224x224xf16>

    // CHECK:       [[CST:%.+]] = const.Declare tensor<1x3x1x1xf16> = dense<2.000000e+00> : tensor<1x3x1x1xf16>, [#const.Rescale<-1.000000e+00 : f64>]
    // CHECK:       [[VAL0:%.+]] = IE.ScaleShift(%arg0, [[CST]]) {operandSegmentSizes = array<i32: 1, 0, 1>} : tensor<1x3x224x224xf16>, tensor<1x3x1x1xf16> -> tensor<1x3x224x224xf16>
    // CHECK:       return [[VAL0]]
}

// -----

// CHECK-LABEL: @ConvertAddToScaleShiftBroadcastChannels
func.func @ConvertAddToScaleShiftBroadcastChannels(%arg0: tensor<1x3x300x300xf16>) -> tensor<1x3x300x300xf16> {
    %bias = const.Declare tensor<1x1x1x1xf16> = dense<2.0> : tensor<1x1x1x1xf16>
    %0 = IE.Add(%arg0, %bias)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
        tensor<1x3x300x300xf16>, tensor<1x1x1x1xf16> -> tensor<1x3x300x300xf16>

    return %0 : tensor<1x3x300x300xf16>

    // CHECK-DAG:       [[BIAS:%.+]] = const.Declare tensor<1x3x1x1xf16> = dense<2.000000e+00> : tensor<1x1x1x1xf16>, [#const.Broadcast<1 : i64, 3 : i64>]
    // CHECK:       [[VAL0:%.+]] = IE.ScaleShift(%arg0, [[BIAS]]) {operandSegmentSizes = array<i32: 1, 0, 1>} : tensor<1x3x300x300xf16>, tensor<1x3x1x1xf16> -> tensor<1x3x300x300xf16>
    // CHECK:       return [[VAL0]]
}

// -----

// CHECK-LABEL: @ConvertAddWithConstFQToScaleShiftBroadcastChannels
func.func @ConvertAddWithConstFQToScaleShiftBroadcastChannels(%arg0: tensor<1x3x300x300xf16>) -> tensor<1x3x300x300xf16> {
    %bias = const.Declare tensor<1x1x1x1xf16> = dense<2.0> : tensor<1x1x1x1xf16>
    %input_low = const.Declare tensor<1x1x1x1xf16> = dense<0.0> : tensor<1x1x1x1xf16>
    %input_high = const.Declare tensor<1x1x1x1xf16> = dense<255.0> : tensor<1x1x1x1xf16>
    %output_low = const.Declare tensor<1x1x1x1xf16> = dense<0.0> : tensor<1x1x1x1xf16>
    %output_high = const.Declare tensor<1x1x1x1xf16> = dense<255.0> : tensor<1x1x1x1xf16>

    %0 = IE.FakeQuantize(%bias, %input_low, %input_high, %output_low, %output_high)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 } :
        tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x1x1xf16>

    %1 = IE.Multiply(%arg0, %0)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
        tensor<1x3x300x300xf16>, tensor<1x1x1x1xf16> -> tensor<1x3x300x300xf16>

    return %1 : tensor<1x3x300x300xf16>

    // CHECK-DAG:       [[BIAS:%.+]] = const.Declare tensor<1x3x1x1xf16> = dense<2.000000e+00> : tensor<1x1x1x1xf16>, [#const.Broadcast<1 : i64, 3 : i64>]
    // CHECK-DAG:       [[CONST_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<2.550000e+02> : tensor<1x1x1x1xf16>
    // CHECK-DAG:       [[CONST_LOW:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf16>
    // CHECK:       [[VAL0:%.+]] = IE.FakeQuantize([[BIAS]], [[CONST_LOW]], [[CONST_HIGH]], [[CONST_LOW]], [[CONST_HIGH]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x3x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x3x1x1xf16>
    // CHECK:       [[VAL1:%.+]] = IE.ScaleShift(%arg0, [[VAL0]]) {operandSegmentSizes = array<i32: 1, 1, 0>} : tensor<1x3x300x300xf16>, tensor<1x3x1x1xf16> -> tensor<1x3x300x300xf16>
    // CHECK:       return [[VAL1]]
}

// -----

// CHECK-LABEL: @ConvertAddToScaleShiftReversedInputs
func.func @ConvertAddToScaleShiftReversedInputs(%arg0: tensor<1x3x300x300xf16>) -> tensor<1x3x300x300xf16> {
    %bias = const.Declare tensor<1x3x1x1xf16> = dense<2.0> : tensor<1x3x1x1xf16>
    %0 = IE.Add(%bias, %arg0)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
        tensor<1x3x1x1xf16>, tensor<1x3x300x300xf16> -> tensor<1x3x300x300xf16>

    return %0 : tensor<1x3x300x300xf16>

    // CHECK-DAG:       [[BIAS:%.+]] = const.Declare tensor<1x3x1x1xf16> = dense<2.000000e+00> : tensor<1x3x1x1xf16>
    // CHECK:       [[VAL0:%.+]] = IE.ScaleShift(%arg0, [[BIAS]]) {operandSegmentSizes = array<i32: 1, 0, 1>} : tensor<1x3x300x300xf16>, tensor<1x3x1x1xf16> -> tensor<1x3x300x300xf16>
    // CHECK:       return [[VAL0]]
}

// -----

// CHECK-LABEL: @cannotConvertAddToScaleShift
func.func @cannotConvertAddToScaleShift(%arg0: tensor<1x3x1x1xf16>, %arg1: tensor<1x3x300x300xf16>) -> tensor<1x3x300x300xf16> {
    %0 = IE.Add(%arg0, %arg1)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
        tensor<1x3x1x1xf16>, tensor<1x3x300x300xf16> -> tensor<1x3x300x300xf16>

    return %0 : tensor<1x3x300x300xf16>

    // CHECK:       [[VAL0:%.+]] = IE.Add(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x1x1xf16>, tensor<1x3x300x300xf16> -> tensor<1x3x300x300xf16>
    // CHECK:       return [[VAL0]]
}

// -----

// CHECK-LABEL: @ConvertMultiplyToScaleShift
func.func @ConvertMultiplyToScaleShift(%arg0: tensor<1x3x300x300xf16>) -> tensor<1x3x300x300xf16> {
    %weights = const.Declare tensor<1x3x1x1xf16> = dense<3.0> : tensor<1x3x1x1xf16>
    %0 = IE.Multiply(%arg0, %weights)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
        tensor<1x3x300x300xf16>, tensor<1x3x1x1xf16> -> tensor<1x3x300x300xf16>

    return %0 : tensor<1x3x300x300xf16>

    // CHECK-DAG:       [[WEIGHTS:%.+]] = const.Declare tensor<1x3x1x1xf16> = dense<3.000000e+00> : tensor<1x3x1x1xf16>
    // CHECK:       [[VAL0:%.+]] = IE.ScaleShift(%arg0, [[WEIGHTS]]) {operandSegmentSizes = array<i32: 1, 1, 0>} : tensor<1x3x300x300xf16>, tensor<1x3x1x1xf16> -> tensor<1x3x300x300xf16>
    // CHECK:       return [[VAL0]]
}

// -----

// CHECK-LABEL: @ConvertMultiplyToScaleShiftBroadcastChannels
func.func @ConvertMultiplyToScaleShiftBroadcastChannels(%arg0: tensor<1x3x300x300xf16>) -> tensor<1x3x300x300xf16> {
    %weights = const.Declare tensor<1x1x1x1xf16> = dense<3.0> : tensor<1x1x1x1xf16>
    %0 = IE.Multiply(%arg0, %weights)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
        tensor<1x3x300x300xf16>, tensor<1x1x1x1xf16> -> tensor<1x3x300x300xf16>

    return %0 : tensor<1x3x300x300xf16>

    // CHECK-DAG:       [[WEIGHTS:%.+]] = const.Declare tensor<1x3x1x1xf16> = dense<3.000000e+00> : tensor<1x1x1x1xf16>, [#const.Broadcast<1 : i64, 3 : i64>]
    // CHECK:       [[VAL0:%.+]] = IE.ScaleShift(%arg0, [[WEIGHTS]]) {operandSegmentSizes = array<i32: 1, 1, 0>} : tensor<1x3x300x300xf16>, tensor<1x3x1x1xf16> -> tensor<1x3x300x300xf16>
    // CHECK:       return [[VAL0]]
}

// -----

// CHECK-LABEL: @ConvertMultiplyToScaleShiftReversedInputs
func.func @ConvertMultiplyToScaleShiftReversedInputs(%arg0: tensor<1x3x1x1xf16>, %arg1: tensor<1x3x300x300xf16>) -> tensor<1x3x300x300xf16> {
    %0 = IE.Multiply(%arg0, %arg1)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
        tensor<1x3x1x1xf16>, tensor<1x3x300x300xf16> -> tensor<1x3x300x300xf16>

    return %0 : tensor<1x3x300x300xf16>

    // CHECK:       [[VAL0:%.+]] = IE.ScaleShift(%arg1, %arg0) {operandSegmentSizes = array<i32: 1, 1, 0>} : tensor<1x3x300x300xf16>, tensor<1x3x1x1xf16> -> tensor<1x3x300x300xf16>
    // CHECK:       return [[VAL0]]
}

// -----

// CHECK-LABEL: @ConvertMultiplyWithConstFQToScaleShiftBroadcastChannels
func.func @ConvertMultiplyWithConstFQToScaleShiftBroadcastChannels(%arg0: tensor<1x3x300x300xf16>) -> tensor<1x3x300x300xf16> {
    %weights = const.Declare tensor<1x1x1x1xf16> = dense<3.0> : tensor<1x1x1x1xf16>
    %input_low = const.Declare tensor<1x1x1x1xf16> = dense<0.0> : tensor<1x1x1x1xf16>
    %input_high = const.Declare tensor<1x1x1x1xf16> = dense<255.0> : tensor<1x1x1x1xf16>
    %output_low = const.Declare tensor<1x1x1x1xf16> = dense<0.0> : tensor<1x1x1x1xf16>
    %output_high = const.Declare tensor<1x1x1x1xf16> = dense<255.0> : tensor<1x1x1x1xf16>

    %0 = IE.FakeQuantize(%weights, %input_low, %input_high, %output_low, %output_high)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 } :
        tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x1x1xf16>

    %1 = IE.Reshape(%0) { shape_value = [1, 1, 1] } : tensor<1x1x1x1xf16> -> tensor<1x1x1xf16>
    %2 = IE.Reshape(%1) { shape_value = [1, 1, 1, 1] } : tensor<1x1x1xf16> -> tensor<1x1x1x1xf16>

    %3 = IE.Multiply(%arg0, %2)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
        tensor<1x3x300x300xf16>, tensor<1x1x1x1xf16> -> tensor<1x3x300x300xf16>

    return %3 : tensor<1x3x300x300xf16>

    // CHECK-DAG:       [[WEIGHTS:%.+]] = const.Declare tensor<1x3x1x1xf16> = dense<3.000000e+00> : tensor<1x1x1x1xf16>, [#const.Broadcast<1 : i64, 3 : i64>]
    // CHECK-DAG:       [[CONST_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<2.550000e+02> : tensor<1x1x1x1xf16>
    // CHECK-DAG:       [[CONST_LOW:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf16>
    // CHECK:       [[VAL0:%.+]] = IE.FakeQuantize([[WEIGHTS]], [[CONST_LOW]], [[CONST_HIGH]], [[CONST_LOW]], [[CONST_HIGH]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x3x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x3x1x1xf16>
    // CHECK:       [[VAL1:%.+]] = IE.Reshape([[VAL0]]) {shape_value = [1, 1, 1]} : tensor<1x3x1x1xf16> -> tensor<1x1x1xf16>
    // CHECK:       [[VAL2:%.+]] = IE.Reshape([[VAL1]]) {shape_value = [1, 1, 1, 1]} : tensor<1x1x1xf16> -> tensor<1x1x1x1xf16>
    // CHECK:       [[VAL3:%.+]] = IE.ScaleShift(%arg0, [[VAL2]]) {operandSegmentSizes = array<i32: 1, 1, 0>} : tensor<1x3x300x300xf16>, tensor<1x1x1x1xf16> -> tensor<1x3x300x300xf16>
    // CHECK:       return [[VAL3]]
}

// -----

// CHECK-LABEL: @ConvertAddWithConstFQToScaleShiftSameShape
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x16x1x1xf16>
func.func @ConvertAddWithConstFQToScaleShiftSameShape(%arg0: tensor<1x16x1x1xf16>) -> tensor<1x16x1x1xf16> {
    %bias = const.Declare tensor<1x16x1x1xf16> = dense<2.0> : tensor<1x16x1x1xf16>
    %input_low = const.Declare tensor<1x1x1x1xf16> = dense<0.0> : tensor<1x1x1x1xf16>
    %input_high = const.Declare tensor<1x1x1x1xf16> = dense<255.0> : tensor<1x1x1x1xf16>
    %output_low = const.Declare tensor<1x1x1x1xf16> = dense<0.0> : tensor<1x1x1x1xf16>
    %output_high = const.Declare tensor<1x1x1x1xf16> = dense<255.0> : tensor<1x1x1x1xf16>

    %0 = IE.FakeQuantize(%bias, %input_low, %input_high, %output_low, %output_high)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 } :
        tensor<1x16x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x16x1x1xf16>

    %1 = IE.Add(%arg0, %0)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
        tensor<1x16x1x1xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x1x1xf16>

    return %1 : tensor<1x16x1x1xf16>

    // CHECK-DAG:       [[BIAS:%.+]] = const.Declare tensor<1x16x1x1xf16> = dense<2.000000e+00> : tensor<1x16x1x1xf16>
    // CHECK-DAG:       [[CONST_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<2.550000e+02> : tensor<1x1x1x1xf16>
    // CHECK-DAG:       [[CONST_LOW:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf16>
    // CHECK:       [[VAL0:%.+]] = IE.FakeQuantize([[BIAS]], [[CONST_LOW]], [[CONST_HIGH]], [[CONST_LOW]], [[CONST_HIGH]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x16x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x16x1x1xf16>
    // CHECK:       [[VAL1:%.+]] = IE.ScaleShift([[INPUT]], [[VAL0]]) {operandSegmentSizes = array<i32: 1, 0, 1>} : tensor<1x16x1x1xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x1x1xf16>
    // CHECK:       return [[VAL1]]
}

// -----

// CHECK-LABEL: @NoConvertAddToScaleShift
func.func @NoConvertAddToScaleShift(%arg0: tensor<1x3x300x300xsi32>) -> tensor<1x3x300x300xsi32> {
    %bias = const.Declare tensor<1x3x1x1xsi32> = dense<2> : tensor<1x3x1x1xsi32>
    %0 = IE.Add(%arg0, %bias)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
        tensor<1x3x300x300xsi32>, tensor<1x3x1x1xsi32> -> tensor<1x3x300x300xsi32>

    return %0 : tensor<1x3x300x300xsi32>

    // CHECK-DAG:       [[BIAS:%.+]] = const.Declare tensor<1x3x1x1xsi32> = dense<2> : tensor<1x3x1x1xsi32>
    // CHECK:       [[VAL0:%.+]] = IE.Add(%arg0, [[BIAS]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x300x300xsi32>, tensor<1x3x1x1xsi32> -> tensor<1x3x300x300xsi32>
    // CHECK:       return [[VAL0]]
}

// -----

// CHECK-LABEL: @NoConvertAddToScaleShiftF32
func.func @NoConvertAddToScaleShiftF32(%arg0: tensor<1x3x300x300xf32>) -> tensor<1x3x300x300xf32> {
    %bias = const.Declare tensor<1x3x1x1xf32> = dense<2.0> : tensor<1x3x1x1xf32>
    %0 = IE.Add(%arg0, %bias)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
        tensor<1x3x300x300xf32>, tensor<1x3x1x1xf32> -> tensor<1x3x300x300xf32>

    return %0 : tensor<1x3x300x300xf32>

    // CHECK-DAG:       [[BIAS:%.+]] = const.Declare tensor<1x3x1x1xf32> = dense<2.000000e+00> : tensor<1x3x1x1xf32>
    // CHECK:       [[VAL0:%.+]] = IE.Add(%arg0, [[BIAS]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x300x300xf32>, tensor<1x3x1x1xf32> -> tensor<1x3x300x300xf32>
    // CHECK:       return [[VAL0]]
}


// -----

// CHECK-LABEL: @NoConvertMultiplyToScaleShift
func.func @NoConvertMultiplyToScaleShift(%arg0: tensor<1x3x300x300xsi32>) -> tensor<1x3x300x300xsi32> {
    %weights = const.Declare tensor<1x3x1x1xsi32> = dense<3> : tensor<1x3x1x1xsi32>
    %0 = IE.Multiply(%arg0, %weights)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
        tensor<1x3x300x300xsi32>, tensor<1x3x1x1xsi32> -> tensor<1x3x300x300xsi32>

    return %0 : tensor<1x3x300x300xsi32>

    // CHECK-DAG:       [[WEIGHTS:%.+]] = const.Declare tensor<1x3x1x1xsi32> = dense<3> : tensor<1x3x1x1xsi32>
    // CHECK:       [[VAL0:%.+]] = IE.Multiply(%arg0, [[WEIGHTS]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x300x300xsi32>, tensor<1x3x1x1xsi32> -> tensor<1x3x300x300xsi32>
    // CHECK:       return [[VAL0]]
}

// -----

// CHECK-LABEL: @NoConvertMultiplyToScaleShiftF32
func.func @NoConvertMultiplyToScaleShiftF32(%arg0: tensor<1x3x300x300xf32>) -> tensor<1x3x300x300xf32> {
    %weights = const.Declare tensor<1x3x1x1xf32> = dense<3.0> : tensor<1x3x1x1xf32>
    %0 = IE.Multiply(%arg0, %weights)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
        tensor<1x3x300x300xf32>, tensor<1x3x1x1xf32> -> tensor<1x3x300x300xf32>

    return %0 : tensor<1x3x300x300xf32>

    // CHECK-DAG:       [[WEIGHTS:%.+]] = const.Declare tensor<1x3x1x1xf32> = dense<3.000000e+00> : tensor<1x3x1x1xf32>
    // CHECK:       [[VAL0:%.+]] = IE.Multiply(%arg0, [[WEIGHTS]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x300x300xf32>, tensor<1x3x1x1xf32> -> tensor<1x3x300x300xf32>
    // CHECK:       return [[VAL0]]
}

// -----

// CHECK-LABEL: @NoConvertMultiplyToScaleShiftWithInconsistentActShape
func.func @NoConvertMultiplyToScaleShiftWithInconsistentActShape(%arg0: tensor<1x256x1x1xf16>) -> tensor<1x256x1x768xf16> {
    %weights = const.Declare tensor<1x1x1x768xf16> = dense<3.0> : tensor<1x1x1x768xf16>
    %0 = IE.Multiply(%arg0, %weights)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
        tensor<1x256x1x1xf16>, tensor<1x1x1x768xf16> -> tensor<1x256x1x768xf16>

    return %0 : tensor<1x256x1x768xf16>

    // CHECK-DAG:       [[WEIGHTS:%.+]] = const.Declare tensor<1x1x1x768xf16> = dense<3.000000e+00> : tensor<1x1x1x768xf16>
    // CHECK:       [[VAL0:%.+]] = IE.Multiply(%arg0, [[WEIGHTS]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x256x1x1xf16>, tensor<1x1x1x768xf16> -> tensor<1x256x1x768xf16>
    // CHECK:       return [[VAL0]]
}

// -----

// CHECK-LABEL: @ConvertSubtractToScaleShift
func.func @ConvertSubtractToScaleShift(%arg0: tensor<1x3x300x300xf16>) -> tensor<1x3x300x300xf16> {
    %bias = const.Declare tensor<1x3x1x1xf16> = dense<2.0> : tensor<1x3x1x1xf16>
    %0 = IE.Subtract(%arg0, %bias)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
        tensor<1x3x300x300xf16>, tensor<1x3x1x1xf16> -> tensor<1x3x300x300xf16>

    return %0 : tensor<1x3x300x300xf16>

    // CHECK-DAG:       [[VAL0:%.+]] = const.Declare tensor<1x3x1x1xf16> = dense<2.000000e+00> : tensor<1x3x1x1xf16>, [#const.Rescale<-1.000000e+00 : f64>]
    // CHECK:       [[VAL1:%.+]] = IE.ScaleShift(%arg0, [[VAL0]]) {operandSegmentSizes = array<i32: 1, 0, 1>} : tensor<1x3x300x300xf16>, tensor<1x3x1x1xf16> -> tensor<1x3x300x300xf16>
    // CHECK:       return [[VAL1]]
}

// -----

// CHECK-LABEL: @NotConvertSubtractToScaleShiftLhsIsNotActivation
func.func @NotConvertSubtractToScaleShiftLhsIsNotActivation(%arg0: tensor<1x3x300x300xf16>) -> tensor<1x3x300x300xf16> {
    %0 = const.Declare tensor<1x3x1x1xf16> = dense<2.0> : tensor<1x3x1x1xf16>
    %1 = IE.Subtract(%0, %arg0)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
        tensor<1x3x1x1xf16>, tensor<1x3x300x300xf16> -> tensor<1x3x300x300xf16>

    return %1 : tensor<1x3x300x300xf16>

    // CHECK-DAG:       [[VAL0:%.+]] = const.Declare tensor<1x3x1x1xf16> = dense<2.000000e+00> : tensor<1x3x1x1xf16>
    // CHECK:       [[VAL1:%.+]] = IE.Subtract([[VAL0]], %arg0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x1x1xf16>, tensor<1x3x300x300xf16> -> tensor<1x3x300x300xf16>
    // CHECK:       return [[VAL1]]
}

// -----

// CHECK-LABEL: @NotConvertMultiplyToScaleShift
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x42840x1x4xf16>)
func.func @NotConvertMultiplyToScaleShift(%arg0: tensor<1x42840x1x4xf16>) -> tensor<1x42840x1x4xf16> {
    %0 = const.Declare tensor<1x42840x1x1xf16> = dense<8.0> : tensor<1x42840x1x1xf16>
    %1 = IE.Multiply(%arg0, %0)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
        tensor<1x42840x1x4xf16>, tensor<1x42840x1x1xf16> -> tensor<1x42840x1x4xf16>

    return %1 : tensor<1x42840x1x4xf16>

    // CHECK-DAG:       [[VAL0:%.+]] = const.Declare tensor<1x42840x1x1xf16> = dense<8.000000e+00> : tensor<1x42840x1x1xf16>
    // CHECK:       [[VAL1:%.+]] = IE.Multiply([[ARG0]], [[VAL0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x42840x1x4xf16>, tensor<1x42840x1x1xf16> -> tensor<1x42840x1x4xf16>
    // CHECK:       return [[VAL1]]
}

// -----

// CHECK-LABEL: @ConvertToScaleShiftFakeQuantizeMultipleUse
func.func @ConvertToScaleShiftFakeQuantizeMultipleUse(%arg0: tensor<1x2x128x128xf16>) -> tensor<1x2x128x128xf16> {
    %weights = const.Declare tensor<1x1x1x1xf16> = dense<3.0> : tensor<1x1x1x1xf16>
    %input_low = const.Declare tensor<1x1x1x1xf16> = dense<0.0> : tensor<1x1x1x1xf16>
    %input_high = const.Declare tensor<1x1x1x1xf16> = dense<255.0> : tensor<1x1x1x1xf16>
    %output_low = const.Declare tensor<1x1x1x1xf16> = dense<0.0> : tensor<1x1x1x1xf16>
    %output_high = const.Declare tensor<1x1x1x1xf16> = dense<255.0> : tensor<1x1x1x1xf16>

    %0 = IE.FakeQuantize(%arg0, %input_low, %input_high, %output_low, %output_high)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 } :
        tensor<1x2x128x128xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x2x128x128xf16>
    %1 = IE.FakeQuantize(%weights, %input_low, %input_high, %output_low, %output_high)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 } :
        tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x1x1xf16>
    %2 = IE.Multiply(%0, %1)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
        tensor<1x2x128x128xf16>, tensor<1x1x1x1xf16> -> tensor<1x2x128x128xf16>
    %3 = IE.FakeQuantize(%2, %input_low, %input_high, %output_low, %output_high)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 } :
        tensor<1x2x128x128xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x2x128x128xf16>
    %4 = IE.Multiply(%3, %1)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
        tensor<1x2x128x128xf16>, tensor<1x1x1x1xf16> -> tensor<1x2x128x128xf16>

    return %4 : tensor<1x2x128x128xf16>

    // CHECK-DAG:       [[WEIGHTS:%.+]] = const.Declare tensor<1x2x1x1xf16> = dense<3.000000e+00> : tensor<1x1x1x1xf16>, [#const.Broadcast<1 : i64, 2 : i64>]
    // CHECK-DAG:       [[CONST_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<2.550000e+02> : tensor<1x1x1x1xf16>
    // CHECK-DAG:       [[CONST_LOW:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf16>
    // CHECK:       [[VAL0:%.+]] = IE.FakeQuantize(%arg0, [[CONST_LOW]], [[CONST_HIGH]], [[CONST_LOW]], [[CONST_HIGH]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} :  tensor<1x2x128x128xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x2x128x128xf16>
    // CHECK:       [[VAL1:%.+]] = IE.FakeQuantize([[WEIGHTS]], [[CONST_LOW]], [[CONST_HIGH]], [[CONST_LOW]], [[CONST_HIGH]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x2x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x2x1x1xf16>
    // CHECK:       [[VAL2:%.+]] = IE.ScaleShift([[VAL0]], [[VAL1]]) {operandSegmentSizes = array<i32: 1, 1, 0>} : tensor<1x2x128x128xf16>, tensor<1x2x1x1xf16> -> tensor<1x2x128x128xf16>
    // CHECK:       [[VAL3:%.+]] = IE.FakeQuantize([[VAL2]], [[CONST_LOW]], [[CONST_HIGH]], [[CONST_LOW]], [[CONST_HIGH]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x2x128x128xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x2x128x128xf16>
    // CHECK:       [[VAL4:%.+]] = IE.ScaleShift([[VAL3]], [[VAL1]]) {operandSegmentSizes = array<i32: 1, 1, 0>} : tensor<1x2x128x128xf16>, tensor<1x2x1x1xf16> -> tensor<1x2x128x128xf16>
    // CHECK:       return [[VAL4]]
}

// -----

// CHECK-LABEL: @ConvertSubtractToScaleShiftBroadcastChannels
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x500x1x2xf16>
func.func @ConvertSubtractToScaleShiftBroadcastChannels(%arg0: tensor<1x500x1x2xf16>) -> tensor<1x500x1x2xf16> {
    %bias = const.Declare tensor<1x1x1x1xf16> = dense<1.000000e+00> : tensor<1x1x1x1xf16>
    %0 = IE.Subtract(%arg0, %bias)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
        tensor<1x500x1x2xf16>, tensor<1x1x1x1xf16> -> tensor<1x500x1x2xf16>

    return %0 : tensor<1x500x1x2xf16>

    // CHECK-DAG:   [[BIAS:%.+]] = const.Declare tensor<1x500x1x1xf16> = dense<1.000000e+00> : tensor<1x1x1x1xf16>, [#const.Broadcast<1 : i64, 500 : i64>, #const.Rescale<-1.000000e+00 : f64>]
    // CHECK:       [[SUB:%.+]] = IE.ScaleShift([[INPUT]], [[BIAS]]) {operandSegmentSizes = array<i32: 1, 0, 1>} : tensor<1x500x1x2xf16>, tensor<1x500x1x1xf16> -> tensor<1x500x1x2xf16>
    // CHECK:       return [[SUB]]
}

// -----

// CHECK-LABEL: @CopyInputChainWhenUserHasAutoBroadcast
// CHECK-SAME:  [[INPUT1:%.+]]: tensor<64x8x49x32xf16>
// CHECK-SAME:  [[INPUT2:%.+]]: tensor<16x16x49x32xf16>
func.func @CopyInputChainWhenUserHasAutoBroadcast(%arg0: tensor<64x8x49x32xf16>, %arg1: tensor<16x16x49x32xf16>) -> (tensor<64x8x49x32xf16>, tensor<16x16x49x32xf16>) {
    %weights = const.Declare tensor<1x1x1x1xf16> = dense<3.0> : tensor<1x1x1x1xf16>
    %input_low = const.Declare tensor<1x1x1x1xf16> = dense<0.0> : tensor<1x1x1x1xf16>
    %input_high = const.Declare tensor<1x1x1x1xf16> = dense<255.0> : tensor<1x1x1x1xf16>
    %output_low = const.Declare tensor<1x1x1x1xf16> = dense<0.0> : tensor<1x1x1x1xf16>
    %output_high = const.Declare tensor<1x1x1x1xf16> = dense<255.0> : tensor<1x1x1x1xf16>

    %0 = IE.FakeQuantize(%weights, %input_low, %input_high, %output_low, %output_high)
        {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} :
        tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x1x1xf16>
    %1 = IE.Multiply(%arg0, %0)
        {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
        tensor<64x8x49x32xf16>, tensor<1x1x1x1xf16> -> tensor<64x8x49x32xf16>
    %2 = IE.Multiply(%arg1, %0)
        {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
        tensor<16x16x49x32xf16>, tensor<1x1x1x1xf16> -> tensor<16x16x49x32xf16>

    return %1, %2 : tensor<64x8x49x32xf16>, tensor<16x16x49x32xf16>

    // CHECK-DAG: [[WEIGHTS:%.+]] = const.Declare tensor<1x16x1x1xf16> = dense<3.000000e+00> : tensor<1x1x1x1xf16>, [#const.Broadcast<1 : i64, 16 : i64>]
    // CHECK:     [[WEIGHTS_COPY:%.+]] = const.Declare tensor<1x8x1x1xf16> = dense<3.000000e+00> : tensor<1x1x1x1xf16>, [#const.Broadcast<1 : i64, 8 : i64>]
    // CHECK:     [[CONST_LOW:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf16>
    // CHECK:     [[CONST_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<2.550000e+02> : tensor<1x1x1x1xf16>
    // CHECK:     [[FQ:%.+]] = IE.FakeQuantize([[WEIGHTS]], [[CONST_LOW]], [[CONST_HIGH]], [[CONST_LOW]], [[CONST_HIGH]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x16x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x16x1x1xf16>
    // CHECK:     [[FQ_COPY:%.+]] = IE.FakeQuantize([[WEIGHTS_COPY]], [[CONST_LOW]], [[CONST_HIGH]], [[CONST_LOW]], [[CONST_HIGH]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x8x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x8x1x1xf16>
    // CHECK:     [[MUL1:%.+]] = IE.ScaleShift([[INPUT1]], [[FQ_COPY]]) {operandSegmentSizes = array<i32: 1, 1, 0>} : tensor<64x8x49x32xf16>, tensor<1x8x1x1xf16> -> tensor<64x8x49x32xf16>
    // CHECK:     [[MUL2:%.+]] = IE.ScaleShift([[INPUT2]], [[FQ]]) {operandSegmentSizes = array<i32: 1, 1, 0>} : tensor<16x16x49x32xf16>, tensor<1x16x1x1xf16> -> tensor<16x16x49x32xf16>
    // CHECK:     return [[MUL1]], [[MUL2]]
}

// -----

// CHECK-LABEL: @CopyInputChainWhenUserDoesNotHaveAutoBroadcast
// CHECK-SAME:  [[INPUT:%.+]]: tensor<64x8x49x32xf16>
func.func @CopyInputChainWhenUserDoesNotHaveAutoBroadcast(%arg0: tensor<64x8x49x32xf16>) -> (tensor<64x8x49x32xf16>, tensor<1x1x1x1xf16>) {
    %weights = const.Declare tensor<1x1x1x1xf16> = dense<3.0> : tensor<1x1x1x1xf16>
    %input_low = const.Declare tensor<1x1x1x1xf16> = dense<0.0> : tensor<1x1x1x1xf16>
    %input_high = const.Declare tensor<1x1x1x1xf16> = dense<255.0> : tensor<1x1x1x1xf16>
    %output_low = const.Declare tensor<1x1x1x1xf16> = dense<0.0> : tensor<1x1x1x1xf16>
    %output_high = const.Declare tensor<1x1x1x1xf16> = dense<255.0> : tensor<1x1x1x1xf16>

    %0 = IE.FakeQuantize(%weights, %input_low, %input_high, %output_low, %output_high)
        {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} :
        tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x1x1xf16>
    %1 = IE.FakeQuantize(%0, %input_low, %input_high, %output_low, %output_high)
        {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} :
        tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x1x1xf16>
    %2 = IE.Multiply(%arg0, %1)
        {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
        tensor<64x8x49x32xf16>, tensor<1x1x1x1xf16> -> tensor<64x8x49x32xf16>
    %3 = IE.Transpose(%0) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>} : tensor<1x1x1x1xf16> -> tensor<1x1x1x1xf16>

    return %2, %3 : tensor<64x8x49x32xf16>, tensor<1x1x1x1xf16>

    // CHECK-DAG: [[WEIGHTS:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<3.000000e+00> : tensor<1x1x1x1xf16>
    // CHECK-DAG: [[WEIGHTS_COPY:%.+]] = const.Declare tensor<1x8x1x1xf16> = dense<3.000000e+00> : tensor<1x1x1x1xf16>, [#const.Broadcast<1 : i64, 8 : i64>]
    // CHECK:     [[CONST_LOW:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf16>
    // CHECK:     [[CONST_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<2.550000e+02> : tensor<1x1x1x1xf16>
    // CHECK:     [[FQ:%.+]] = IE.FakeQuantize([[WEIGHTS]], [[CONST_LOW]], [[CONST_HIGH]], [[CONST_LOW]], [[CONST_HIGH]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x1x1xf16>
    // CHECK:     [[FQ_COPY:%.+]] = IE.FakeQuantize([[WEIGHTS_COPY]], [[CONST_LOW]], [[CONST_HIGH]], [[CONST_LOW]], [[CONST_HIGH]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x8x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x8x1x1xf16>
    // CHECK:     [[MUL:%.+]] = IE.ScaleShift([[INPUT]], [[FQ_COPY]]) {operandSegmentSizes = array<i32: 1, 1, 0>} : tensor<64x8x49x32xf16>, tensor<1x8x1x1xf16> -> tensor<64x8x49x32xf16>
    // CHECK:     [[TRANSPOSE:%.+]] = IE.Transpose([[FQ]]) {order_value = #NWHC} : tensor<1x1x1x1xf16> -> tensor<1x1x1x1xf16>
    // CHECK:     return [[MUL]], [[TRANSPOSE]]
}

// -----

// CHECK-LABEL: @DoNotCopyInputChainWhenUsersAreBroadcastable
// CHECK-SAME:  [[INPUT1:%.+]]: tensor<64x8x49x32xf16>
// CHECK-SAME:  [[INPUT2:%.+]]: tensor<16x8x49x32xf16>
func.func @DoNotCopyInputChainWhenUsersAreBroadcastable(%arg0: tensor<64x8x49x32xf16>, %arg1: tensor<16x8x49x32xf16>) -> (tensor<64x8x49x32xf16>, tensor<16x8x49x32xf16>) {
    %weights = const.Declare tensor<1x1x1x1xf16> = dense<3.0> : tensor<1x1x1x1xf16>
    %input_low = const.Declare tensor<1x1x1x1xf16> = dense<0.0> : tensor<1x1x1x1xf16>
    %input_high = const.Declare tensor<1x1x1x1xf16> = dense<255.0> : tensor<1x1x1x1xf16>
    %output_low = const.Declare tensor<1x1x1x1xf16> = dense<0.0> : tensor<1x1x1x1xf16>
    %output_high = const.Declare tensor<1x1x1x1xf16> = dense<255.0> : tensor<1x1x1x1xf16>

    %0 = IE.FakeQuantize(%weights, %input_low, %input_high, %output_low, %output_high)
        {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} :
        tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x1x1xf16>
    %1 = IE.Multiply(%arg0, %0)
        {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
        tensor<64x8x49x32xf16>, tensor<1x1x1x1xf16> -> tensor<64x8x49x32xf16>
    %2 = IE.Multiply(%arg1, %0)
        {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
        tensor<16x8x49x32xf16>, tensor<1x1x1x1xf16> -> tensor<16x8x49x32xf16>

    return %1, %2 : tensor<64x8x49x32xf16>, tensor<16x8x49x32xf16>

    // CHECK-DAG: [[WEIGHTS:%.+]] = const.Declare tensor<1x8x1x1xf16> = dense<3.000000e+00> : tensor<1x1x1x1xf16>, [#const.Broadcast<1 : i64, 8 : i64>]
    // CHECK:     [[CONST_LOW:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf16>
    // CHECK:     [[CONST_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<2.550000e+02> : tensor<1x1x1x1xf16>
    // CHECK:     [[FQ:%.+]] = IE.FakeQuantize([[WEIGHTS]], [[CONST_LOW]], [[CONST_HIGH]], [[CONST_LOW]], [[CONST_HIGH]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x8x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x8x1x1xf16>
    // CHECK:     [[MUL1:%.+]] = IE.ScaleShift([[INPUT1]], [[FQ]]) {operandSegmentSizes = array<i32: 1, 1, 0>} : tensor<64x8x49x32xf16>, tensor<1x8x1x1xf16> -> tensor<64x8x49x32xf16>
    // CHECK:     [[MUL2:%.+]] = IE.ScaleShift([[INPUT2]], [[FQ]]) {operandSegmentSizes = array<i32: 1, 1, 0>} : tensor<16x8x49x32xf16>, tensor<1x8x1x1xf16> -> tensor<16x8x49x32xf16>
    // CHECK:     return [[MUL1]], [[MUL2]]
}

// -----

// CHECK-LABEL: @CopyInputChainIncludingReshape
// CHECK-SAME:  [[INPUT1:%.+]]: tensor<64x3x300x300xf16>
func.func @CopyInputChainIncludingReshape(%arg0: tensor<64x3x300x300xf16>) -> (tensor<1x1x1x1xf16>, tensor<64x3x300x300xf16>) {
    %weights = const.Declare tensor<1x1x1x1xf16> = dense<3.0> : tensor<1x1x1x1xf16>
    %input_low = const.Declare tensor<1x1x1x1xf16> = dense<0.0> : tensor<1x1x1x1xf16>
    %input_high = const.Declare tensor<1x1x1x1xf16> = dense<255.0> : tensor<1x1x1x1xf16>
    %output_low = const.Declare tensor<1x1x1x1xf16> = dense<0.0> : tensor<1x1x1x1xf16>
    %output_high = const.Declare tensor<1x1x1x1xf16> = dense<255.0> : tensor<1x1x1x1xf16>

    %0 = IE.Reshape(%weights) { shape_value = [1, 1, 1] } : tensor<1x1x1x1xf16> -> tensor<1x1x1xf16>
    %1 = IE.Reshape(%0) { shape_value = [1, 1, 1, 1] } : tensor<1x1x1xf16> -> tensor<1x1x1x1xf16>

    %2 = IE.FakeQuantize(%1, %input_low, %input_high, %output_low, %output_high)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 } :
        tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x1x1xf16>

    %3 = IE.Transpose(%2) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>} : tensor<1x1x1x1xf16> -> tensor<1x1x1x1xf16>

    %4 = IE.Add(%arg0, %2)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
        tensor<64x3x300x300xf16>, tensor<1x1x1x1xf16> -> tensor<64x3x300x300xf16>

    return %3, %4 : tensor<1x1x1x1xf16>, tensor<64x3x300x300xf16>

    // CHECK-DAG:       [[WEIGHTS:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<3.000000e+00> : tensor<1x1x1x1xf16>, [#const.Reshape<[1, 1, 1, 1]>]
    // CHECK-DAG:       [[WEIGHTS_COPY:%.+]] = const.Declare tensor<1x3x1x1xf16> = dense<3.000000e+00> : tensor<1x1x1x1xf16>, [#const.Reshape<[1, 1, 1, 1]>, #const.Broadcast<1 : i64, 3 : i64>]
    // CHECK-DAG:       [[CONST_LOW:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf16>
    // CHECK-DAG:       [[CONST_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<2.550000e+02> : tensor<1x1x1x1xf16>
    // CHECK:       [[FQ:%.+]] = IE.FakeQuantize([[WEIGHTS]], [[CONST_LOW]], [[CONST_HIGH]], [[CONST_LOW]], [[CONST_HIGH]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x1x1xf16>
    // CHECK:       [[TRANSPOSE:%.+]] = IE.Transpose([[FQ]]) {order_value = #NWHC} : tensor<1x1x1x1xf16> -> tensor<1x1x1x1xf16>
    // CHECK:       [[FQ_COPY:%.+]] = IE.FakeQuantize([[WEIGHTS_COPY]], [[CONST_LOW]], [[CONST_HIGH]], [[CONST_LOW]], [[CONST_HIGH]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x3x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x3x1x1xf16>
    // CHECK:       [[ADD:%.+]] = IE.ScaleShift([[INPUT1]], [[FQ_COPY]]) {operandSegmentSizes = array<i32: 1, 0, 1>} : tensor<64x3x300x300xf16>, tensor<1x3x1x1xf16> -> tensor<64x3x300x300xf16>
    // CHECK:       return [[TRANSPOSE]], [[ADD]]
}

// -----

// CHECK-LABEL: @ConvertMultiplyToScaleShiftWithHSplatWeights
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x2400x12x2xf16>
func.func @ConvertMultiplyToScaleShiftWithHSplatWeights(%arg0: tensor<1x2400x12x2xf16>) -> tensor<1x2400x12x2xf16> {
    %weights = const.Declare tensor<1x1x12x1xf16> = dense<0.17> : tensor<1x1x12x1xf16> isSplat
    %0 = IE.Multiply(%arg0, %weights)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
        tensor<1x2400x12x2xf16>, tensor<1x1x12x1xf16> -> tensor<1x2400x12x2xf16>

    return %0 : tensor<1x2400x12x2xf16>

    // CHECK-DAG:       [[WEIGHTS:%.+]] = const.Declare tensor<1x2400x1x1xf16> = dense<1.700440e-01> : tensor<1x1x12x1xf16>, [#const.SubView<[0, 0, 0, 0], [1, 1, 1, 1]>, #const.Broadcast<1 : i64, 2400 : i64>]
    // CHECK:       [[SCALESHIFT:%.+]] = IE.ScaleShift([[INPUT]], [[WEIGHTS]]) {operandSegmentSizes = array<i32: 1, 1, 0>} : tensor<1x2400x12x2xf16>, tensor<1x2400x1x1xf16> -> tensor<1x2400x12x2xf16>

    // CHECK:       return [[SCALESHIFT]]
}

// -----

// CHECK-LABEL: @ConvertMultiplyToScaleShiftWithWSplatWeights
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x2400x12x2xf16>
func.func @ConvertMultiplyToScaleShiftWithWSplatWeights(%arg0: tensor<1x2400x12x2xf16>) -> tensor<1x2400x12x2xf16> {
    %weights = const.Declare tensor<1x1x1x2xf16> = dense<0.17> : tensor<1x1x1x2xf16> isSplat
    %0 = IE.Multiply(%arg0, %weights)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
        tensor<1x2400x12x2xf16>, tensor<1x1x1x2xf16> -> tensor<1x2400x12x2xf16>

    return %0 : tensor<1x2400x12x2xf16>

    // CHECK-DAG:       [[WEIGHTS:%.+]] = const.Declare tensor<1x2400x1x1xf16> = dense<1.700440e-01> : tensor<1x1x1x2xf16>, [#const.SubView<[0, 0, 0, 0], [1, 1, 1, 1]>, #const.Broadcast<1 : i64, 2400 : i64>]
    // CHECK:       [[SCALESHIFT:%.+]] = IE.ScaleShift([[INPUT]], [[WEIGHTS]]) {operandSegmentSizes = array<i32: 1, 1, 0>} : tensor<1x2400x12x2xf16>, tensor<1x2400x1x1xf16> -> tensor<1x2400x12x2xf16>

    // CHECK:       return [[SCALESHIFT]]
}

// -----

// CHECK-LABEL: @ConvertMultiplyToScaleShiftWithWeightsFront
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x16x1x1xf16>
func.func @ConvertMultiplyToScaleShiftWithWeightsFront(%arg0: tensor<1x16x1x1xf16>) -> tensor<1x16x14x64xf16> {
    %weights = const.Declare tensor<1x16x14x64xf16> = dense<1.25> : tensor<1x16x14x64xf16>
    %0 = IE.Multiply(%arg0, %weights)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
        tensor<1x16x1x1xf16>, tensor<1x16x14x64xf16> -> tensor<1x16x14x64xf16>

    return %0 : tensor<1x16x14x64xf16>

    // CHECK-DAG:   [[WEIGHTS:%.+]] = const.Declare tensor<1x16x14x64xf16> = dense<1.250000e+00> : tensor<1x16x14x64xf16>
    // CHECK:       [[SCALESHIFT:%.+]] = IE.ScaleShift([[WEIGHTS]], [[INPUT]]) {operandSegmentSizes = array<i32: 1, 1, 0>} : tensor<1x16x14x64xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x14x64xf16>
    // CHECK:       return [[SCALESHIFT]]
}

// -----

#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

// CHECK-LABEL: @ConvertMultiplyToScaleShiftWithHExpandedWeightsBroadcast
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x1x64x1xf16>
func.func @ConvertMultiplyToScaleShiftWithHExpandedWeightsBroadcast(%arg0: tensor<1x1x64x1xf16>) -> tensor<1x8x64x16xf16> {
    %weights = const.Declare tensor<1x8x64x16xf16> = dense<0.75> : tensor<1x8x64x16xf16>
    %0 = IE.Multiply(%arg0, %weights)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
        tensor<1x1x64x1xf16>, tensor<1x8x64x16xf16> -> tensor<1x8x64x16xf16>

    return %0 : tensor<1x8x64x16xf16>

    // CHECK-DAG:   [[WEIGHTS:%.+]] = const.Declare tensor<1x64x8x16xf16> = dense<7.500000e-01> : tensor<1x8x64x16xf16>, [#const.Transpose<#NHCW>]
    // CHECK:       [[TRANSPOSE_IN:%.+]] = IE.Transpose([[INPUT]]) {order_value = #NHCW} : tensor<1x1x64x1xf16> -> tensor<1x64x1x1xf16>
    // CHECK:       [[SCALESHIFT:%.+]] = IE.ScaleShift([[WEIGHTS]], [[TRANSPOSE_IN]]) {operandSegmentSizes = array<i32: 1, 1, 0>} : tensor<1x64x8x16xf16>, tensor<1x64x1x1xf16> -> tensor<1x64x8x16xf16>
    // CHECK:       [[TRANSPOSE_OUT:%.+]] = IE.Transpose([[SCALESHIFT]]) {order_value = #NHCW} : tensor<1x64x8x16xf16> -> tensor<1x8x64x16xf16>
    // CHECK:       return [[TRANSPOSE_OUT]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @ConvertMultiplyToScaleShiftWithWExpandedWeightsBroadcast
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x1x1x64xf16>
func.func @ConvertMultiplyToScaleShiftWithWExpandedWeightsBroadcast(%arg0: tensor<1x1x1x64xf16>) -> tensor<1x16x14x64xf16> {
    %weights = const.Declare tensor<1x16x14x64xf16> = dense<1.25> : tensor<1x16x14x64xf16>
    %0 = IE.Multiply(%arg0, %weights)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
        tensor<1x1x1x64xf16>, tensor<1x16x14x64xf16> -> tensor<1x16x14x64xf16>

    return %0 : tensor<1x16x14x64xf16>

    // CHECK-DAG:   [[WEIGHTS:%.+]] = const.Declare tensor<1x64x16x14xf16> = dense<1.250000e+00> : tensor<1x16x14x64xf16>, [#const.Transpose<#NWCH>]
    // CHECK:       [[TRANSPOSE_IN:%.+]] = IE.Transpose([[INPUT]]) {order_value = #NWCH} : tensor<1x1x1x64xf16> -> tensor<1x64x1x1xf16>
    // CHECK:       [[SCALESHIFT:%.+]] = IE.ScaleShift([[WEIGHTS]], [[TRANSPOSE_IN]]) {operandSegmentSizes = array<i32: 1, 1, 0>} : tensor<1x64x16x14xf16>, tensor<1x64x1x1xf16> -> tensor<1x64x16x14xf16>
    // CHECK:       [[TRANSPOSE_OUT:%.+]] = IE.Transpose([[SCALESHIFT]]) {order_value = #NHWC} : tensor<1x64x16x14xf16> -> tensor<1x16x14x64xf16>
    // CHECK:       return [[TRANSPOSE_OUT]]
}

// -----

// CHECK-LABEL: @NoConvertMultiplyToScaleShiftWithHExpanded
// CHECK-SAME:  ([[INPUT_0:%.+]]: tensor<1x1x64x1xf16>, [[INPUT_1:%.+]]: tensor<1x8x64x16xf16>)
func.func @NoConvertMultiplyToScaleShiftWithHExpanded(%arg0: tensor<1x1x64x1xf16>, %arg1: tensor<1x8x64x16xf16>) -> tensor<1x8x64x16xf16> {
    %0 = IE.Multiply(%arg0, %arg1)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
        tensor<1x1x64x1xf16>, tensor<1x8x64x16xf16> -> tensor<1x8x64x16xf16>

    return %0 : tensor<1x8x64x16xf16>

    // CHECK:       [[MULTIPLY:%.+]] = IE.Multiply([[INPUT_0]], [[INPUT_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x64x1xf16>, tensor<1x8x64x16xf16> -> tensor<1x8x64x16xf16>
    // CHECK:       return [[MULTIPLY]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @ConvertMultiplyToScaleShiftWithWExpandedAndTransposeInput
// CHECK-SAME:  ([[INPUT_0:%.+]]: tensor<1x1x1x1536xf16>, [[INPUT_1:%.+]]: tensor<1x1536x1024x1xf16>)
func.func @ConvertMultiplyToScaleShiftWithWExpandedAndTransposeInput(%arg0: tensor<1x1x1x1536xf16>, %arg1: tensor<1x1536x1024x1xf16>) -> tensor<1x1024x1x1536xf16> {
    %0 = IE.Transpose(%arg1) {order_value = #NHWC} : tensor<1x1536x1024x1xf16> -> tensor<1x1024x1x1536xf16>
    %1 = IE.Multiply(%arg0, %0)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
        tensor<1x1x1x1536xf16>, tensor<1x1024x1x1536xf16> -> tensor<1x1024x1x1536xf16>

    return %1 : tensor<1x1024x1x1536xf16>

    // CHECK:       [[TRANSPOSE_ACTIVATION_0:%.+]] = IE.Transpose([[INPUT_1]]) {order_value = #NHWC} : tensor<1x1536x1024x1xf16> -> tensor<1x1024x1x1536xf16>
    // CHECK:       [[TRANSPOSE_ACTIVATION_1:%.+]] = IE.Transpose([[TRANSPOSE_ACTIVATION_0]]) {order_value = #NWCH} : tensor<1x1024x1x1536xf16> -> tensor<1x1536x1024x1xf16>
    // CHECK:       [[TRANSPOSE_WEIGHTS:%.+]] = IE.Transpose([[INPUT_0]]) {order_value = #NWCH} : tensor<1x1x1x1536xf16> -> tensor<1x1536x1x1xf16>
    // CHECK:       [[SCALESHIFT:%.+]] = IE.ScaleShift([[TRANSPOSE_ACTIVATION_1]], [[TRANSPOSE_WEIGHTS]]) {operandSegmentSizes = array<i32: 1, 1, 0>} : tensor<1x1536x1024x1xf16>, tensor<1x1536x1x1xf16> -> tensor<1x1536x1024x1xf16>
    // CHECK:       [[TRANSPOSE_OUT:%.+]] = IE.Transpose([[SCALESHIFT]]) {order_value = #NHWC} : tensor<1x1536x1024x1xf16> -> tensor<1x1024x1x1536xf16>
    // CHECK:       return [[TRANSPOSE_OUT]]
}
