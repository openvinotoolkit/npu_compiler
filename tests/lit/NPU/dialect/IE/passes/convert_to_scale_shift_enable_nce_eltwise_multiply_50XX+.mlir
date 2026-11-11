//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --convert-to-scale-shift="enable-nce-eltwise-multiply=true" %s | FileCheck %s
// REQUIRES: platform-NPU5010 || platform-NPU6010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @DontConvertCOnlyMultiplyToScaleShift
// CHECK-SAME:      [[INPUT0:%.+]]: tensor<1x6144x1x1xf16>,
// CHECK-SAME:      [[INPUT1:%.+]]: tensor<1x6144x1x1xf16>
func.func @DontConvertCOnlyMultiplyToScaleShift(%arg0: tensor<1x6144x1x1xf16>, %arg1: tensor<1x6144x1x1xf16>) -> tensor<1x6144x1x1xf16> {
    %multiply = IE.Multiply(%arg0, %arg1)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
        tensor<1x6144x1x1xf16>, tensor<1x6144x1x1xf16> -> tensor<1x6144x1x1xf16>

    return %multiply : tensor<1x6144x1x1xf16>

    // Multiply with C-only shape is NOT converted to ScaleShift
    // when it is supported to be converted to NCEEltwise Multiply
    // The Multiply is changed to NHWC layout to make it convertible to NCEEltwise Multiply
    // CHECK-DAG:       [[PERMUTE0:%.+]] = IE.PermuteCast([[INPUT0]])
    // CHECK-DAG:       [[PERMUTE1:%.+]] = IE.PermuteCast([[INPUT1]])
    // CHECK:           [[MULTIPLY:%.+]] = IE.Multiply([[PERMUTE0]], [[PERMUTE1]])
    // CHECK-SAME:          tensor<1x6144x1x1xf16, {order = #NHWC}>
    // CHECK-NOT:       [[MULTIPLY:%.+]] = IE.ScaleShift
    // CHECK:           [[PERMUTE2:%.+]] = IE.PermuteCast([[MULTIPLY]])
    // CHECK:           return [[PERMUTE2]]
}

// -----

// CHECK-LABEL: @ConvertCOnlyMultiplyToScaleShiftForFusion
func.func @ConvertCOnlyMultiplyToScaleShiftForFusion(%arg0: tensor<1x6144x1x1xf16>, %arg1: tensor<1x6144x1x1xf16>, %arg2: tensor<1x6144x1x1xf16>) -> tensor<1x6144x1x1xf16> {
    %multiply = IE.Multiply(%arg0, %arg1)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
        tensor<1x6144x1x1xf16>, tensor<1x6144x1x1xf16> -> tensor<1x6144x1x1xf16>
    %add = IE.Add(%multiply, %arg2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
            : tensor<1x6144x1x1xf16>, tensor<1x6144x1x1xf16> -> tensor<1x6144x1x1xf16>

    return %add : tensor<1x6144x1x1xf16>

    // Multiply with C-only shape is converted to ScaleShift
    // when it is connected to another operation that can be changed to ScaleShift
    // The connected ScaleShift can be fused as one op for efficiency
    // CHECK-NOT:       [[MULTIPLY:%.+]] = IE.Multiply
    // CHECK:           [[SCALE_SHIFT:%.+]] = IE.ScaleShift
    // CHECK:           [[ADD:%.+]] = IE.Add
    // CHECK:           return [[ADD]]
}

// -----

// CHECK-LABEL: @ConvertCOnlyMultiplyToScaleShiftWithFakeQuantize
// CHECK-SAME:      [[INPUT0:%.+]]: tensor<1x6144x1x1xf16>,
// CHECK-SAME:      [[INPUT1:%.+]]: tensor<1x6144x1x1xf16>
func.func @ConvertCOnlyMultiplyToScaleShiftWithFakeQuantize(%arg0: tensor<1x6144x1x1xf16>, %arg1: tensor<1x6144x1x1xf16>) -> tensor<1x6144x1x1xf16> {
    %cst_low = const.Declare tensor<1x1x1x1xf16> = dense<-1.0> : tensor<1x1x1x1xf16>
    %cst_high = const.Declare tensor<1x1x1x1xf16> = dense<1.0> : tensor<1x1x1x1xf16>
    %multiply = IE.Multiply(%arg0, %arg1)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
        tensor<1x6144x1x1xf16>, tensor<1x6144x1x1xf16> -> tensor<1x6144x1x1xf16>
    %fq = IE.FakeQuantize(%multiply, %cst_low, %cst_high, %cst_low, %cst_high)
        {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels_or_low_fp_type = 256 : i64} :
        tensor<1x6144x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>
        -> tensor<1x6144x1x1xf16>

    return %fq : tensor<1x6144x1x1xf16>

    // CHECK-NOT:       IE.Multiply
    // CHECK:           [[SS:%.+]] = IE.ScaleShift([[INPUT0]], [[INPUT1]])
    // CHECK:           [[FQ:%.+]] = IE.FakeQuantize([[SS]]
    // CHECK:           return [[FQ]]
}

// -----

// CHECK-LABEL: @DontConvertLargeCOnlyMultiplyToScaleShiftWithFakeQuantize
// CHECK-SAME:      [[INPUT0:%.+]]: tensor<1x16384x1x1xf16>,
// CHECK-SAME:      [[INPUT1:%.+]]: tensor<1x16384x1x1xf16>
func.func @DontConvertLargeCOnlyMultiplyToScaleShiftWithFakeQuantize(%arg0: tensor<1x16384x1x1xf16>, %arg1: tensor<1x16384x1x1xf16>) -> tensor<1x16384x1x1xf16> {
    %cst_low = const.Declare tensor<1x1x1x1xf16> = dense<-1.0> : tensor<1x1x1x1xf16>
    %cst_high = const.Declare tensor<1x1x1x1xf16> = dense<1.0> : tensor<1x1x1x1xf16>
    %multiply = IE.Multiply(%arg0, %arg1)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
        tensor<1x16384x1x1xf16>, tensor<1x16384x1x1xf16> -> tensor<1x16384x1x1xf16>
    %fq = IE.FakeQuantize(%multiply, %cst_low, %cst_high, %cst_low, %cst_high)
        {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels_or_low_fp_type = 256 : i64} :
        tensor<1x16384x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>
        -> tensor<1x16384x1x1xf16>

    return %fq : tensor<1x16384x1x1xf16>

    // CHECK-NOT:       IE.ScaleShift
    // CHECK:           [[MUL:%.+]] = IE.Multiply
    // CHECK:           [[FQ:%.+]] = IE.FakeQuantize([[MUL]]
    // CHECK:           return [[FQ]]
}

// -----

// CHECK-LABEL: @ConvertCOnlyMultiplyToScaleShiftWithFakeQuantizeViaViewOp
// CHECK-SAME:      [[INPUT0:%.+]]: tensor<1x6144x1x1xf16>,
// CHECK-SAME:      [[INPUT1:%.+]]: tensor<1x6144x1x1xf16>
func.func @ConvertCOnlyMultiplyToScaleShiftWithFakeQuantizeViaViewOp(%arg0: tensor<1x6144x1x1xf16>, %arg1: tensor<1x6144x1x1xf16>) -> tensor<1x1x6144x1xf16> {
    %cst_low = const.Declare tensor<1x1x1x1xf16> = dense<-1.0> : tensor<1x1x1x1xf16>
    %cst_high = const.Declare tensor<1x1x1x1xf16> = dense<1.0> : tensor<1x1x1x1xf16>
    %multiply = IE.Multiply(%arg0, %arg1)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
        tensor<1x6144x1x1xf16>, tensor<1x6144x1x1xf16> -> tensor<1x6144x1x1xf16>
    %reshape = IE.Reshape(%multiply) { shape_value = [1, 1, 6144, 1] } :
        tensor<1x6144x1x1xf16> -> tensor<1x1x6144x1xf16>
    %fq = IE.FakeQuantize(%reshape, %cst_low, %cst_high, %cst_low, %cst_high)
        {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels_or_low_fp_type = 256 : i64} :
        tensor<1x1x6144x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>
        -> tensor<1x1x6144x1xf16>

    return %fq : tensor<1x1x6144x1xf16>

    // CHECK-NOT:       IE.Multiply
    // CHECK:           [[SS:%.+]] = IE.ScaleShift([[INPUT0]], [[INPUT1]])
    // CHECK:           [[RS:%.+]] = IE.Reshape([[SS]])
    // CHECK:           IE.FakeQuantize([[RS]]
}

// -----

// CHECK-LABEL: @DontConvertCOnlyMultiplyWithMultipleUsersBeforeFakeQuantize
// CHECK-SAME:      [[INPUT0:%.+]]: tensor<1x6144x1x1xf16>,
// CHECK-SAME:      [[INPUT1:%.+]]: tensor<1x6144x1x1xf16>
func.func @DontConvertCOnlyMultiplyWithMultipleUsersBeforeFakeQuantize(%arg0: tensor<1x6144x1x1xf16>, %arg1: tensor<1x6144x1x1xf16>) -> (tensor<1x6144x1x1xf16>, tensor<1x6144x1x1xf16>) {
    %cst_low = const.Declare tensor<1x1x1x1xf16> = dense<-1.0> : tensor<1x1x1x1xf16>
    %cst_high = const.Declare tensor<1x1x1x1xf16> = dense<1.0> : tensor<1x1x1x1xf16>
    %multiply = IE.Multiply(%arg0, %arg1)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
        tensor<1x6144x1x1xf16>, tensor<1x6144x1x1xf16> -> tensor<1x6144x1x1xf16>
    %fq = IE.FakeQuantize(%multiply, %cst_low, %cst_high, %cst_low, %cst_high)
        {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels_or_low_fp_type = 256 : i64} :
        tensor<1x6144x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>
        -> tensor<1x6144x1x1xf16>

    return %multiply, %fq : tensor<1x6144x1x1xf16>, tensor<1x6144x1x1xf16>

    // CHECK-NOT:       IE.ScaleShift
    // CHECK:           IE.Multiply
    // CHECK:           IE.FakeQuantize
}
