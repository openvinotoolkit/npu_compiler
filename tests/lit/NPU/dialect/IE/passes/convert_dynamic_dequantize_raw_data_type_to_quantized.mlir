//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --convert-dq-raw-data-type-to-quantized %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

// -----
// SI4 raw input: pass inserts an identity QuantizeCast with quant.uniform<i4:f16, 1.0>.

// CHECK: !qElemType = !quant.uniform<i4:f16, 1.000000e+00>
// CHECK-LABEL: @SI4RawInput
// CHECK-SAME: ([[INPUT:%.+]]: tensor<4x8xsi4>)
func.func @SI4RawInput(%arg0: tensor<4x8xsi4>) -> tensor<4x8xf16> {
    %scale = const.Declare tensor<1x1xf16> = dense<0.5> : tensor<1x1xf32>, [#const.CastElemType<f16>]
    %0 = IE.DynamicDequantize(%arg0, %scale) {dstElemType = f16, vpux.synthetic_dyn_dequant}
            : tensor<4x8xsi4>, tensor<1x1xf16> -> tensor<4x8xf16>
    return %0 : tensor<4x8xf16>

    // CHECK-DAG:   [[SCALE:%.+]] = const.Declare tensor<1x1xf16> = dense<5.000000e-01> : tensor<1x1xf32>, [#const.CastElemType<f16>]
    // CHECK:       [[CAST:%.+]] = IE.QuantizeCast([[INPUT]])
    // CHECK-SAME:      {dstElemType = !qElemType}
    // CHECK-SAME:      tensor<4x8xsi4> -> tensor<4x8x!qElemType>
    // CHECK:       [[OUT:%.+]] = IE.DynamicDequantize([[CAST]], [[SCALE]])
    // CHECK-SAME:      {dstElemType = f16, vpux.synthetic_dyn_dequant}
    // CHECK-SAME:      tensor<4x8x!qElemType>, tensor<1x1xf16> -> tensor<4x8xf16>
    // CHECK:       return [[OUT]] : tensor<4x8xf16>
}

// -----
// UI4 raw input: pass inserts an identity QuantizeCast with quant.uniform<u4:f16, 1.0>.

// CHECK: !qElemType = !quant.uniform<u4:f16, 1.000000e+00>
// CHECK-LABEL: @UI4RawInput
// CHECK-SAME: ([[INPUT:%.+]]: tensor<4x8xui4>)
func.func @UI4RawInput(%arg0: tensor<4x8xui4>) -> tensor<4x8xf16> {
    %scale = const.Declare tensor<1x1xf16> = dense<1.0> : tensor<1x1xf32>, [#const.CastElemType<f16>]
    %0 = IE.DynamicDequantize(%arg0, %scale) {dstElemType = f16, vpux.synthetic_dyn_dequant}
            : tensor<4x8xui4>, tensor<1x1xf16> -> tensor<4x8xf16>
    return %0 : tensor<4x8xf16>

    // CHECK-DAG:   [[SCALE:%.+]] = const.Declare tensor<1x1xf16> = dense<1.000000e+00> : tensor<1x1xf32>, [#const.CastElemType<f16>]
    // CHECK:       [[CAST:%.+]] = IE.QuantizeCast([[INPUT]])
    // CHECK-SAME:      {dstElemType = !qElemType}
    // CHECK-SAME:      tensor<4x8xui4> -> tensor<4x8x!qElemType>
    // CHECK:       [[OUT:%.+]] = IE.DynamicDequantize([[CAST]], [[SCALE]])
    // CHECK-SAME:      {dstElemType = f16, vpux.synthetic_dyn_dequant}
    // CHECK-SAME:      tensor<4x8x!qElemType>, tensor<1x1xf16> -> tensor<4x8xf16>
    // CHECK:       return [[OUT]] : tensor<4x8xf16>
}

// -----
// SI8 raw input: pass inserts an identity QuantizeCast with quant.uniform<i8:f16, 1.0>.

// CHECK: !qElemType = !quant.uniform<i8:f16, 1.000000e+00>
// CHECK-LABEL: @SI8RawInput
// CHECK-SAME: ([[INPUT:%.+]]: tensor<16x64xsi8>)
func.func @SI8RawInput(%arg0: tensor<16x64xsi8>) -> tensor<16x64xf16> {
    %scale = const.Declare tensor<1x1xf16> = dense<0.25> : tensor<1x1xf32>, [#const.CastElemType<f16>]
    %0 = IE.DynamicDequantize(%arg0, %scale) {dstElemType = f16, vpux.synthetic_dyn_dequant}
            : tensor<16x64xsi8>, tensor<1x1xf16> -> tensor<16x64xf16>
    return %0 : tensor<16x64xf16>

    // CHECK-DAG:   [[SCALE:%.+]] = const.Declare tensor<1x1xf16> = dense<2.500000e-01> : tensor<1x1xf32>, [#const.CastElemType<f16>]
    // CHECK:       [[CAST:%.+]] = IE.QuantizeCast([[INPUT]])
    // CHECK-SAME:      {dstElemType = !qElemType}
    // CHECK-SAME:      tensor<16x64xsi8> -> tensor<16x64x!qElemType>
    // CHECK:       [[OUT:%.+]] = IE.DynamicDequantize([[CAST]], [[SCALE]])
    // CHECK-SAME:      {dstElemType = f16, vpux.synthetic_dyn_dequant}
    // CHECK-SAME:      tensor<16x64x!qElemType>, tensor<1x1xf16> -> tensor<16x64xf16>
    // CHECK:       return [[OUT]] : tensor<16x64xf16>
}

// -----
// UI8 raw input with f32 output: pass inserts an identity QuantizeCast with quant.uniform<u8:f32, 1.0>.

// CHECK: !qElemType = !quant.uniform<u8:f32, 1.000000e+00>
// CHECK-LABEL: @UI8RawInputF32Output
// CHECK-SAME: ([[INPUT:%.+]]: tensor<4x4x3x3xui8>)
func.func @UI8RawInputF32Output(%arg0: tensor<4x4x3x3xui8>) -> tensor<4x4x3x3xf32> {
    %scale = const.Declare tensor<1x1x1x1xf32> = dense<1.0> : tensor<1x1x1x1xf32>
    %0 = IE.DynamicDequantize(%arg0, %scale) {dstElemType = f32, vpux.synthetic_dyn_dequant}
            : tensor<4x4x3x3xui8>, tensor<1x1x1x1xf32> -> tensor<4x4x3x3xf32>
    return %0 : tensor<4x4x3x3xf32>

    // CHECK-DAG:   [[SCALE:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<1.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK:       [[CAST:%.+]] = IE.QuantizeCast([[INPUT]])
    // CHECK-SAME:      {dstElemType = !qElemType}
    // CHECK-SAME:      tensor<4x4x3x3xui8> -> tensor<4x4x3x3x!qElemType>
    // CHECK:       [[OUT:%.+]] = IE.DynamicDequantize([[CAST]], [[SCALE]])
    // CHECK-SAME:      {dstElemType = f32, vpux.synthetic_dyn_dequant}
    // CHECK-SAME:      tensor<4x4x3x3x!qElemType>, tensor<1x1x1x1xf32> -> tensor<4x4x3x3xf32>
    // CHECK:       return [[OUT]] : tensor<4x4x3x3xf32>
}

// -----
// SI4 raw input with an explicit zero-point operand.
// The zero-point operand does not affect whether the QuantizeCast is inserted.

// CHECK: !qElemType = !quant.uniform<i4:f16, 1.000000e+00>
// CHECK-LABEL: @SI4RawInputWithZeroPoint
// CHECK-SAME: ([[INPUT:%.+]]: tensor<4x8xsi4>)
func.func @SI4RawInputWithZeroPoint(%arg0: tensor<4x8xsi4>) -> tensor<4x8xf16> {
    %scale = const.Declare tensor<1x1xf16> = dense<0.5> : tensor<1x1xf32>, [#const.CastElemType<f16>]
    %zp    = const.Declare tensor<1x1xsi4> = dense<2> : tensor<1x1xsi8>, [#const.CastElemType<si4>]
    %0 = IE.DynamicDequantize(%arg0, %scale, %zp) {dstElemType = f16, vpux.synthetic_dyn_dequant}
            : tensor<4x8xsi4>, tensor<1x1xf16>, tensor<1x1xsi4> -> tensor<4x8xf16>
    return %0 : tensor<4x8xf16>

    // CHECK-DAG:   [[SCALE:%.+]] = const.Declare tensor<1x1xf16> = dense<5.000000e-01> : tensor<1x1xf32>, [#const.CastElemType<f16>]
    // CHECK-DAG:   [[ZP:%.+]]    = const.Declare tensor<1x1xsi4> = dense<2> : tensor<1x1xsi8>, [#const.CastElemType<si4>]
    // CHECK:       [[CAST:%.+]] = IE.QuantizeCast([[INPUT]])
    // CHECK-SAME:      {dstElemType = !qElemType}
    // CHECK-SAME:      tensor<4x8xsi4> -> tensor<4x8x!qElemType>
    // CHECK:       [[OUT:%.+]] = IE.DynamicDequantize([[CAST]], [[SCALE]], [[ZP]])
    // CHECK-SAME:      {dstElemType = f16, vpux.synthetic_dyn_dequant}
    // CHECK-SAME:      tensor<4x8x!qElemType>, tensor<1x1xf16>, tensor<1x1xsi4> -> tensor<4x8xf16>
    // CHECK:       return [[OUT]] : tensor<4x8xf16>
}

// -----
// Per-channel scale (one scale value per output row along axis 0).
// The pass transforms based on the input element type only; scale shape does not affect it.

// CHECK: !qElemType = !quant.uniform<i8:f16, 1.000000e+00>
// CHECK-LABEL: @SI8RawInputPerChannelScale
// CHECK-SAME: ([[INPUT:%.+]]: tensor<4x8xsi8>)
func.func @SI8RawInputPerChannelScale(%arg0: tensor<4x8xsi8>) -> tensor<4x8xf16> {
    %scale = const.Declare tensor<4x1xf16> = dense<[[0.25], [0.5], [0.75], [1.0]]> : tensor<4x1xf32>, [#const.CastElemType<f16>]
    %0 = IE.DynamicDequantize(%arg0, %scale) {dstElemType = f16, vpux.synthetic_dyn_dequant}
            : tensor<4x8xsi8>, tensor<4x1xf16> -> tensor<4x8xf16>
    return %0 : tensor<4x8xf16>

    // CHECK-DAG:   [[SCALE:%.+]] = const.Declare tensor<4x1xf16>
    // CHECK:       [[CAST:%.+]] = IE.QuantizeCast([[INPUT]])
    // CHECK-SAME:      {dstElemType = !qElemType}
    // CHECK-SAME:      tensor<4x8xsi8> -> tensor<4x8x!qElemType>
    // CHECK:       [[OUT:%.+]] = IE.DynamicDequantize([[CAST]], [[SCALE]])
    // CHECK-SAME:      {dstElemType = f16, vpux.synthetic_dyn_dequant}
    // CHECK-SAME:      tensor<4x8x!qElemType>, tensor<4x1xf16> -> tensor<4x8xf16>
    // CHECK:       return [[OUT]] : tensor<4x8xf16>
}

// -----
// Already-quantized input: the DynamicDequantize input carries a quant.uniform type.
// The pass skips this op entirely; no QuantizeCast is inserted.

!qElemType = !quant.uniform<i8:f16, 5.000000e-01>

// CHECK: !qElemType = !quant.uniform<i8:f16, 5.000000e-01>
// CHECK-LABEL: @AlreadyQuantizedInputIsSkipped
// CHECK-SAME: ([[INPUT:%.+]]: tensor<4x8x!qElemType>)
func.func @AlreadyQuantizedInputIsSkipped(%arg0: tensor<4x8x!qElemType>) -> tensor<4x8xf16> {
    %scale = const.Declare tensor<1x1xf16> = dense<0.5> : tensor<1x1xf32>, [#const.CastElemType<f16>]
    %0 = IE.DynamicDequantize(%arg0, %scale) {dstElemType = f16, vpux.synthetic_dyn_dequant}
            : tensor<4x8x!qElemType>, tensor<1x1xf16> -> tensor<4x8xf16>
    return %0 : tensor<4x8xf16>

    // CHECK-NOT:   IE.QuantizeCast
    // CHECK:       [[OUT:%.+]] = IE.DynamicDequantize([[INPUT]], {{%.+}})
    // CHECK-SAME:      {dstElemType = f16, vpux.synthetic_dyn_dequant}
    // CHECK-SAME:      tensor<4x8x!qElemType>, tensor<1x1xf16> -> tensor<4x8xf16>
    // CHECK:       return [[OUT]] : tensor<4x8xf16>
}

// -----
// Multiple DynamicDequantize ops in one function: each raw-type op receives its own QuantizeCast.

// CHECK: !qElemType = !quant.uniform<i4:f16, 1.000000e+00>
// CHECK: !qElemType1 = !quant.uniform<u8:f16, 1.000000e+00>
// CHECK-LABEL: @MultipleOpsEachGetsQuantizeCast
// CHECK-SAME: ([[A:%.+]]: tensor<4x8xsi4>, [[B:%.+]]: tensor<4x8xui8>)
func.func @MultipleOpsEachGetsQuantizeCast(%arg0: tensor<4x8xsi4>, %arg1: tensor<4x8xui8>) -> (tensor<4x8xf16>, tensor<4x8xf16>) {
    %scale = const.Declare tensor<1x1xf16> = dense<0.5> : tensor<1x1xf32>, [#const.CastElemType<f16>]
    %0 = IE.DynamicDequantize(%arg0, %scale) {dstElemType = f16, vpux.synthetic_dyn_dequant}
            : tensor<4x8xsi4>, tensor<1x1xf16> -> tensor<4x8xf16>
    %1 = IE.DynamicDequantize(%arg1, %scale) {dstElemType = f16, vpux.synthetic_dyn_dequant}
            : tensor<4x8xui8>, tensor<1x1xf16> -> tensor<4x8xf16>
    return %0, %1 : tensor<4x8xf16>, tensor<4x8xf16>

    // CHECK-DAG:   [[SCALE:%.+]] = const.Declare
    // CHECK:       [[CAST_A:%.+]] = IE.QuantizeCast([[A]])
    // CHECK-SAME:      {dstElemType = !qElemType}
    // CHECK-SAME:      tensor<4x8xsi4> -> tensor<4x8x!qElemType>
    // CHECK:       [[OUT_A:%.+]] = IE.DynamicDequantize([[CAST_A]], [[SCALE]])
    // CHECK-SAME:      tensor<4x8x!qElemType>
    // CHECK:       [[CAST_B:%.+]] = IE.QuantizeCast([[B]])
    // CHECK-SAME:      {dstElemType = !qElemType1}
    // CHECK-SAME:      tensor<4x8xui8> -> tensor<4x8x!qElemType1>
    // CHECK:       [[OUT_B:%.+]] = IE.DynamicDequantize([[CAST_B]], [[SCALE]])
    // CHECK-SAME:      tensor<4x8x!qElemType1>
    // CHECK:       return [[OUT_A]], [[OUT_B]]
}
