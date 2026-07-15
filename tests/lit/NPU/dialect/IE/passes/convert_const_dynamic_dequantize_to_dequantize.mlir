//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --convert-const-dynamic-dequantize-to-dequantize %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

// -----
// Per-tensor case: single constant scale, no zero-point.
// Expect: IE.QuantizeCast embeds the scale, IE.Dequantize replaces IE.DynamicDequantize.

!qElemType = !quant.uniform<i8:f16, 1.000000e+00>

// CHECK: !qElemType = !quant.uniform<i8:f16, 1.000000e+00>
// CHECK: !qElemType1 = !quant.uniform<i8:f16, 5.000000e-01>
// CHECK-LABEL: @PerTensorScaleNoZP
// CHECK-SAME:      [[INPUT:%.+]]: tensor<4x8x!qElemType>
func.func @PerTensorScaleNoZP(%arg0: tensor<4x8x!qElemType>) -> tensor<4x8xf16> {
    %scale = const.Declare tensor<1x1xf16> = dense<0.5> : tensor<1x1xf32>, [#const.CastElemType<f16>]
    %0 = IE.DynamicDequantize(%arg0, %scale) {dstElemType = f16}
            : tensor<4x8x!qElemType>, tensor<1x1xf16> -> tensor<4x8xf16>
    return %0 : tensor<4x8xf16>

    // CHECK:       [[CAST:%.+]] = IE.QuantizeCast(%arg0)
    // CHECK-SAME:      tensor<4x8x!qElemType> -> tensor<4x8x!qElemType1>
    // CHECK:       [[DEQU:%.+]] = IE.Dequantize([[CAST]]) {dstElemType = f16}
    // CHECK-SAME:      tensor<4x8x!qElemType1> -> tensor<4x8xf16>
    // CHECK:       return [[DEQU]] : tensor<4x8xf16>
    // CHECK-NOT:   IE.DynamicDequantize
}

// -----
// Per-axis case: one scale per output channel (axis 0), no zero-point.
// Expect: IE.QuantizeCast with per-channel uniform type, IE.Dequantize.

!qElemType = !quant.uniform<i8:f16, 1.000000e+00>

// CHECK: !qElemType = !quant.uniform<i8:f16, 1.000000e+00>
// CHECK: !qElemType1 = !quant.uniform<i8:f16:0, {2.500000e-01,5.000000e-01,7.500000e-01,1.000000e+00}>
// CHECK-LABEL: @PerAxisScaleNoZP
// CHECK-SAME:      [[INPUT:%.+]]: tensor<4x8x!qElemType>
func.func @PerAxisScaleNoZP(%arg0: tensor<4x8x!qElemType>) -> tensor<4x8xf16> {
    // 4 per-row scales along axis 0
    %scale = const.Declare tensor<4xf16> = dense<[0.25, 0.5, 0.75, 1.0]> : tensor<4xf32>, [#const.CastElemType<f16>]
    %0 = IE.DynamicDequantize(%arg0, %scale) {dstElemType = f16}
            : tensor<4x8x!qElemType>, tensor<4xf16> -> tensor<4x8xf16>
    return %0 : tensor<4x8xf16>

    // CHECK:       [[CAST:%.+]] = IE.QuantizeCast([[INPUT]])
    // CHECK-SAME:      tensor<4x8x!qElemType> -> tensor<4x8x!qElemType1>
    // CHECK:       [[DEQU:%.+]] = IE.Dequantize([[CAST]]) {dstElemType = f16}
    // CHECK-SAME:      tensor<4x8x!qElemType1> -> tensor<4x8xf16>
    // CHECK:       return [[DEQU]] : tensor<4x8xf16>
    // CHECK-NOT:   IE.DynamicDequantize
}

// -----
// Per-tensor case with constant scale AND constant zero-point.
// Expect: ZP values are embedded into the quant type, IE.DynamicDequantize is removed.

!qElemType = !quant.uniform<i8:f16, 1.000000e+00>

// CHECK: !qElemType = !quant.uniform<i8:f16, 1.000000e+00>
// CHECK: !qElemType1 = !quant.uniform<i8:f16, 5.000000e-01:-10>
// CHECK-LABEL: @PerTensorScaleAndZP
// CHECK-SAME:      [[INPUT:%.+]]: tensor<4x8x!qElemType>
func.func @PerTensorScaleAndZP(%arg0: tensor<4x8x!qElemType>) -> tensor<4x8xf16> {
    %scale = const.Declare tensor<1xf16> = dense<0.5>   : tensor<1xf32>, [#const.CastElemType<f16>]
    %zp    = const.Declare tensor<4x8xsi8>  = dense<-10>   : tensor<4x8xsi8>
    %0 = IE.DynamicDequantize(%arg0, %scale, %zp) {dstElemType = f16}
            : tensor<4x8x!qElemType>, tensor<1xf16>, tensor<4x8xsi8> -> tensor<4x8xf16>
    return %0 : tensor<4x8xf16>

    // CHECK:       [[CAST:%.+]] = IE.QuantizeCast([[INPUT]])
    // CHECK-SAME:      tensor<4x8x!qElemType> -> tensor<4x8x!qElemType1>
    // CHECK:       [[DEQU:%.+]] = IE.Dequantize([[CAST]]) {dstElemType = f16}
    // CHECK-SAME:      tensor<4x8x!qElemType1> -> tensor<4x8xf16>
    // CHECK:       return [[DEQU]] : tensor<4x8xf16>
    // CHECK-NOT:   IE.DynamicDequantize
}

// -----
// Per-axis case: scales along the inner axis (axis 1), no zero-point.
// Exercises axis detection on a non-zero quantization dimension.

!qElemType = !quant.uniform<i8:f16, 1.000000e+00>

// CHECK: !qElemType = !quant.uniform<i8:f16, 1.000000e+00>
// CHECK: !qElemType1 = !quant.uniform<i8:f16:1, {{.*}}>
// CHECK-LABEL: @PerAxisScaleAxis1
// CHECK-SAME:      [[INPUT:%.+]]: tensor<4x8x!qElemType>
func.func @PerAxisScaleAxis1(%arg0: tensor<4x8x!qElemType>) -> tensor<4x8xf16> {
    // 8 scales along axis 1; axis 0 is trivial (size 1)
    %scale = const.Declare tensor<1x8xf16> = dense<[[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]]> : tensor<1x8xf32>, [#const.CastElemType<f16>]
    %0 = IE.DynamicDequantize(%arg0, %scale) {dstElemType = f16}
            : tensor<4x8x!qElemType>, tensor<1x8xf16> -> tensor<4x8xf16>
    return %0 : tensor<4x8xf16>

    // CHECK:       [[CAST:%.+]] = IE.QuantizeCast([[INPUT]])
    // CHECK-SAME:      tensor<4x8x!qElemType> -> tensor<4x8x!qElemType1>
    // CHECK:       [[DEQU:%.+]] = IE.Dequantize([[CAST]]) {dstElemType = f16}
    // CHECK-SAME:      tensor<4x8x!qElemType1> -> tensor<4x8xf16>
    // CHECK:       return [[DEQU]] : tensor<4x8xf16>
    // CHECK-NOT:   IE.DynamicDequantize
}

// -----
// Per-axis case: one scale and one zero-point per channel (non-splat ZP).
// Exercises the zpVals.size() == scales.size() branch.

!qElemType = !quant.uniform<i8:f16, 1.000000e+00>

// CHECK: !qElemType = !quant.uniform<i8:f16, 1.000000e+00>
// CHECK: !qElemType1 = !quant.uniform<i8:f16:0, {2.500000e-01,5.000000e-01:1,7.500000e-01:2,1.000000e+00:3}>
// CHECK-LABEL: @PerAxisScaleAndPerAxisZP
// CHECK-SAME:      [[INPUT:%.+]]: tensor<4x8x!qElemType>
func.func @PerAxisScaleAndPerAxisZP(%arg0: tensor<4x8x!qElemType>) -> tensor<4x8xf16> {
    %scale = const.Declare tensor<4x1xf16> = dense<[[0.25], [0.5], [0.75], [1.0]]> : tensor<4x1xf32>, [#const.CastElemType<f16>]
    %zp    = const.Declare tensor<4x1xsi8>  = dense<[[0], [1], [2], [3]]>           : tensor<4x1xsi8>
    %0 = IE.DynamicDequantize(%arg0, %scale, %zp) {dstElemType = f16}
            : tensor<4x8x!qElemType>, tensor<4x1xf16>, tensor<4x1xsi8> -> tensor<4x8xf16>
    return %0 : tensor<4x8xf16>

    // CHECK:       [[CAST:%.+]] = IE.QuantizeCast([[INPUT]])
    // CHECK-SAME:      tensor<4x8x!qElemType> -> tensor<4x8x!qElemType1>
    // CHECK:       [[DEQU:%.+]] = IE.Dequantize([[CAST]]) {dstElemType = f16}
    // CHECK-SAME:      tensor<4x8x!qElemType1> -> tensor<4x8xf16>
    // CHECK:       return [[DEQU]] : tensor<4x8xf16>
    // CHECK-NOT:   IE.DynamicDequantize
}

// -----
// Negative: zero-point is a function argument (not a constant).
// Scale is constant so the input-type guard passes, but ZP guard must reject.

!qElemType = !quant.uniform<i8:f16, 1.000000e+00>

// CHECK: !qElemType = !quant.uniform<i8:f16, 1.000000e+00>
// CHECK-LABEL: @DynamicZPNotConverted
// CHECK-SAME:      [[INPUT:%.+]]: tensor<4x8x!qElemType>,
// CHECK-SAME:      [[ZP:%.+]]: tensor<4x8xsi8>
func.func @DynamicZPNotConverted(%arg0: tensor<4x8x!qElemType>, %arg1: tensor<4x8xsi8>) -> tensor<4x8xf16> {
    %scale = const.Declare tensor<1x1xf16> = dense<0.5> : tensor<1x1xf32>, [#const.CastElemType<f16>]
    %0 = IE.DynamicDequantize(%arg0, %scale, %arg1) {dstElemType = f16}
            : tensor<4x8x!qElemType>, tensor<1x1xf16>, tensor<4x8xsi8> -> tensor<4x8xf16>
    return %0 : tensor<4x8xf16>

    // CHECK:       [[OUT:%.+]] = IE.DynamicDequantize([[INPUT]], {{%.+}}, [[ZP]]) {dstElemType = f16}
    // CHECK-SAME:      tensor<4x8x!qElemType>, tensor<1x1xf16>, tensor<4x8xsi8> -> tensor<4x8xf16>
    // CHECK:       return [[OUT]] : tensor<4x8xf16>
    // CHECK-NOT:   IE.QuantizeCast
    // CHECK-NOT:   IE.Dequantize
}

// -----
// Negative: scale has two non-trivial axes (sub-channel quantization).
// Expect: IE.DynamicDequantize is preserved unchanged.

!qElemType = !quant.uniform<i8:f16, 1.000000e+00>

// CHECK: !qElemType = !quant.uniform<i8:f16, 1.000000e+00>
// CHECK-LABEL: @MultiAxisScaleNotConverted
// CHECK-SAME:      [[INPUT:%.+]]: tensor<4x8x!qElemType>
func.func @MultiAxisScaleNotConverted(%arg0: tensor<4x8x!qElemType>) -> tensor<4x8xf16> {
    // Both dims are > 1, so the multi-axis guard must reject this.
    %scale = const.Declare tensor<4x8xf16> = dense<1.0> : tensor<4x8xf32>, [#const.CastElemType<f16>]
    %0 = IE.DynamicDequantize(%arg0, %scale) {dstElemType = f16}
            : tensor<4x8x!qElemType>, tensor<4x8xf16> -> tensor<4x8xf16>
    return %0 : tensor<4x8xf16>

    // CHECK:       [[OUT:%.+]] = IE.DynamicDequantize([[INPUT]], {{%.+}}) {dstElemType = f16}
    // CHECK-SAME:      tensor<4x8x!qElemType>, tensor<4x8xf16> -> tensor<4x8xf16>
    // CHECK:       return [[OUT]] : tensor<4x8xf16>
    // CHECK-NOT:   IE.QuantizeCast
    // CHECK-NOT:   IE.Dequantize
}

// -----
// Negative: scale is a function argument (not a constant).
// Expect: IE.DynamicDequantize is preserved unchanged.

!qElemType = !quant.uniform<i8:f16, 1.000000e+00>

// CHECK: !qElemType = !quant.uniform<i8:f16, 1.000000e+00>
// CHECK-LABEL: @DynamicScaleNotConverted
// CHECK-SAME:      [[INPUT:%.+]]: tensor<4x8x!qElemType>,
// CHECK-SAME:      [[SCALE:%.+]]: tensor<1x1xf16>
func.func @DynamicScaleNotConverted(%arg0: tensor<4x8x!qElemType>, %arg1: tensor<1x1xf16>) -> tensor<4x8xf16> {
    %0 = IE.DynamicDequantize(%arg0, %arg1) {dstElemType = f16}
            : tensor<4x8x!qElemType>, tensor<1x1xf16> -> tensor<4x8xf16>
    return %0 : tensor<4x8xf16>

    // CHECK:       [[OUT:%.+]] = IE.DynamicDequantize([[INPUT]], [[SCALE]]) {dstElemType = f16}
    // CHECK-SAME:      tensor<4x8x!qElemType>, tensor<1x1xf16> -> tensor<4x8xf16>
    // CHECK:       return [[OUT]] : tensor<4x8xf16>
    // CHECK-NOT:   IE.QuantizeCast
    // CHECK-NOT:   IE.Dequantize
}
