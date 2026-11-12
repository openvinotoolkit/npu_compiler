//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --convert-quantize-ops-to-nce-ops %s | FileCheck %s
// REQUIRES: platform-NPU6010

!qElemType = !quant.uniform<u16:f16, 1.000000e+00>

// CHECK:       !qElemType = !quant.uniform<u16:f16, 1.000000e+00>
// CHECK-NEXT:  !qElemType1 = !quant.uniform<u8:f16, 1.000000e+00>
// CHECK-LABEL: func.func @DequantizeU16ToConvDCIMEligible
// CHECK-SAME:  ([[INPUT:%.+]]: tensor<1x128x8x8x!qElemType>)
func.func @DequantizeU16ToConvDCIMEligible(%arg0 : tensor<1x128x8x8x!qElemType>) -> tensor<1x128x8x8xf16> {
    %0 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x128x8x8x!qElemType> -> tensor<1x128x8x8xf16>
    return %0 : tensor<1x128x8x8xf16>

    // CHECK:      [[WEIGHTS:%.+]] = const.Declare tensor<128x128x1x1x!qElemType1> = dense{{.+}} : tensor<128x128x1x1xf32>, [#const.CastElemType<!qElemType1>]
    // CHECK-NEXT: [[CONV:%.+]] = IE.Convolution([[INPUT]], [[WEIGHTS]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x128x8x8x!qElemType>, tensor<128x128x1x1x!qElemType1> -> tensor<1x128x8x8xf16>
    // CHECK:      return [[CONV]] : tensor<1x128x8x8xf16>
}

// -----

!qElemType = !quant.uniform<i16:f16, 1.000000e+00>

// CHECK:       !qElemType = !quant.uniform<i16:f16, 1.000000e+00>
// CHECK-NEXT:  !qElemType1 = !quant.uniform<i8:f16, 1.000000e+00>
// CHECK-LABEL: func.func @DequantizeI16ToConvDCIMEligible
// CHECK-SAME:  ([[INPUT:%.+]]: tensor<1x128x8x8x!qElemType>)
func.func @DequantizeI16ToConvDCIMEligible(%arg0 : tensor<1x128x8x8x!qElemType>) -> tensor<1x128x8x8xf16> {
    %0 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x128x8x8x!qElemType> -> tensor<1x128x8x8xf16>
    return %0 : tensor<1x128x8x8xf16>

    // CHECK:      [[WEIGHTS:%.+]] = const.Declare tensor<128x128x1x1x!qElemType1> = dense{{.+}} : tensor<128x128x1x1xf32>, [#const.CastElemType<!qElemType1>]
    // CHECK-NEXT: [[CONV:%.+]] = IE.Convolution([[INPUT]], [[WEIGHTS]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x128x8x8x!qElemType>, tensor<128x128x1x1x!qElemType1> -> tensor<1x128x8x8xf16>
    // CHECK:      return [[CONV]] : tensor<1x128x8x8xf16>
}

// -----

// The following tests verify that IE.Dequantize ops are NOT lowered to IE.Convolution when the
// prospective identity convolution fails DCIM SW or HW heuristics. Unconditionally lowering to
// a non-DCIM convolution would be more expensive than keeping the dequantize, regressing
// performance relative to pre-NPU60XX architectures where the DCIM gate did not exist.

!qElemType = !quant.uniform<u16:f32, 1.000000e+00>

// CHECK:       !qElemType = !quant.uniform<u16:f32, 1.000000e+00>
// CHECK-LABEL:    func.func @KeepDequantizeU16
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x4x28x28x!qElemType>)
func.func @KeepDequantizeU16(%arg0 : tensor<1x4x28x28x!qElemType>) -> tensor<1x4x28x28xf32> {
    %1 = IE.Dequantize(%arg0) {dstElemType = f32} : tensor<1x4x28x28x!qElemType> -> tensor<1x4x28x28xf32>
    return %1 : tensor<1x4x28x28xf32>

    // CHECK:       [[DEQUANT:%.+]] = IE.Dequantize([[INPUT]]) {dstElemType = f32} : tensor<1x4x28x28x!qElemType> -> tensor<1x4x28x28xf32>
    // CHECK:       return [[DEQUANT]] : tensor<1x4x28x28xf32>
}

// -----

!qElemType = !quant.uniform<u16:f16, 1.000000e+00>

// CHECK:       !qElemType = !quant.uniform<u16:f16, 1.000000e+00>
// CHECK-LABEL:    func.func @KeepDequantizeU16WithFP16Precision
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x4x28x28x!qElemType>)
func.func @KeepDequantizeU16WithFP16Precision(%arg0 : tensor<1x4x28x28x!qElemType>) -> tensor<1x4x28x28xf16> {
    %1 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x4x28x28x!qElemType> -> tensor<1x4x28x28xf16>
    return %1 : tensor<1x4x28x28xf16>

    // CHECK:       [[DEQUANT:%.+]] = IE.Dequantize([[INPUT]]) {dstElemType = f16} : tensor<1x4x28x28x!qElemType> -> tensor<1x4x28x28xf16>
    // CHECK:       return [[DEQUANT]] : tensor<1x4x28x28xf16>
}

// -----

!qElemType = !quant.uniform<i16:f32, 1.000000e+00>

// CHECK:       !qElemType = !quant.uniform<i16:f32, 1.000000e+00>
// CHECK-LABEL: func.func @KeepDequantizeI16
// CHECK:       ([[INPUT:%.+]]: tensor<1x4x28x28x!qElemType>)
func.func @KeepDequantizeI16(%arg0 : tensor<1x4x28x28x!qElemType>) -> tensor<1x4x28x28xf32> {
    %1 = IE.Dequantize(%arg0) {dstElemType = f32} : tensor<1x4x28x28x!qElemType> -> tensor<1x4x28x28xf32>
    return %1 : tensor<1x4x28x28xf32>

    // CHECK:       [[DEQUANT:%.+]] = IE.Dequantize([[INPUT]]) {dstElemType = f32} : tensor<1x4x28x28x!qElemType> -> tensor<1x4x28x28xf32>
    // CHECK:       return [[DEQUANT]] : tensor<1x4x28x28xf32>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!qElemType = !quant.uniform<i16:f32, 1.000000e+00>
!dynType = tensor<1x1x?x2xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 500, 2]> : tensor<4xsi64>, order = #NCHW}>
!dynQuantType = tensor<1x1x?x2x!qElemType, {bounds = #const.OpaqueI64Elements<[1, 1, 500, 2]> : tensor<4xsi64>, order = #NCHW}>

// CHECK:  !qElemType = !quant.uniform<i16:f32, 1.000000e+00>

// CHECK-LABEL:  func.func @DynamicShapeKeepDequantizeI16
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x1x?x2x!qElemType, {bounds = #const.OpaqueI64Elements<[1, 1, 500, 2]> : tensor<4xsi64>, order = #NCHW}>
func.func @DynamicShapeKeepDequantizeI16(%arg0 : !dynQuantType) -> !dynType {
    %0 = IE.Dequantize(%arg0) {dstElemType = f16} : !dynQuantType -> !dynType
    return %0 : !dynType

    // CHECK:       [[DEQUANT:%.+]] = IE.Dequantize([[INPUT]]) {dstElemType = f16} : tensor<1x1x?x2x!qElemType, {bounds = #const.OpaqueI64Elements<[1, 1, 500, 2]> : tensor<4xsi64>, order = #NCHW}> -> tensor<1x1x?x2xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 500, 2]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK:  return [[DEQUANT]] : tensor<1x1x?x2xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 500, 2]> : tensor<4xsi64>, order = #NCHW}>
}

// -----

// A u16 Quantize whose output scale is too small (rescale 1 / scale would need an unsupported post-shift in the
// legacy weight table) must remain a SW IE.Quantize instead of being lowered to an identity depthwise
// GroupConvolution, which would otherwise throw during weights table generation.

!qElemType = !quant.uniform<u16:f16, 3.0072000000000001E-5>

// CHECK-LABEL: func.func @NotConvertQuantizeToDwConvUnrepresentableScale
// CHECK-SAME:  ([[INPUT:%.+]]: tensor<1x320x64x64xf16>)
func.func @NotConvertQuantizeToDwConvUnrepresentableScale(%arg0 : tensor<1x320x64x64xf16>) -> tensor<1x320x64x64x!qElemType> {
    %0 = IE.Quantize(%arg0) {dstElemType = !qElemType } : tensor<1x320x64x64xf16> -> tensor<1x320x64x64x!qElemType>
    return %0 : tensor<1x320x64x64x!qElemType>

    // CHECK:  [[QUANTIZE:%.+]] = IE.Quantize([[INPUT]]) {dstElemType = !qElemType} : tensor<1x320x64x64xf16> -> tensor<1x320x64x64x!qElemType>
    // CHECK:  return [[QUANTIZE]] : tensor<1x320x64x64x!qElemType>
}

// -----

// Boundary case: scale exactly 2^-15 gives rescale exactly 2^15, whose exponent still requires a post-shift,
// so the Quantize must remain a SW IE.Quantize rather than being lowered to a DW GroupConvolution.

!qElemType = !quant.uniform<u16:f16, 3.0517578125E-5>

// CHECK-LABEL: func.func @NotConvertQuantizeToDwConvScaleAtBoundary
// CHECK-SAME:  ([[INPUT:%.+]]: tensor<1x320x64x64xf16>)
func.func @NotConvertQuantizeToDwConvScaleAtBoundary(%arg0 : tensor<1x320x64x64xf16>) -> tensor<1x320x64x64x!qElemType> {
    %0 = IE.Quantize(%arg0) {dstElemType = !qElemType } : tensor<1x320x64x64xf16> -> tensor<1x320x64x64x!qElemType>
    return %0 : tensor<1x320x64x64x!qElemType>

    // CHECK:  [[QUANTIZE:%.+]] = IE.Quantize([[INPUT]]) {dstElemType = !qElemType} : tensor<1x320x64x64xf16> -> tensor<1x320x64x64x!qElemType>
    // CHECK:  return [[QUANTIZE]] : tensor<1x320x64x64x!qElemType>
}

// -----

// On 60XX+ the dequantize path for small dynamic shapes is lowered to
// QuantizeCast + Add instead of AvgPool used on NPU3720/NPU4000/NPU5010.

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!qElemType = !quant.uniform<u8:f16, 0.013961060374390846:109>
!dynType = tensor<1x1x?x2xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 500, 2]> : tensor<4xsi64>, order = #NCHW}>
!dynQuantType = tensor<1x1x?x2x!qElemType, {bounds = #const.OpaqueI64Elements<[1, 1, 500, 2]> : tensor<4xsi64>, order = #NCHW}>

// CHECK:  !qElemType = !quant.uniform<u8:f16, 0.027922120748781691:109>
// CHECK:  !qElemType1 = !quant.uniform<u8:f16, 0.013961060374390846:109>
// CHECK:  !qElemType2 = !quant.uniform<u8:f16, 0.0069805301871954228:109>

// CHECK-LABEL:  func.func @DynamicShapeToAdd
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x1x?x2xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 500, 2]> : tensor<4xsi64>, order = #NCHW}>)
func.func @DynamicShapeToAdd(%arg0 : !dynType) -> !dynType {
    %0 = IE.Quantize(%arg0) {dstElemType = !qElemType} : !dynType -> !dynQuantType
    %1 = IE.Dequantize(%0) {dstElemType = f16} : !dynQuantType -> !dynType
    return %1 : !dynType

    // CHECK:       [[Q_ADD:%.+]] = IE.Add([[INPUT]], [[INPUT]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} :
    // CHECK-SAME:    tensor<1x1x?x2xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 500, 2]> : tensor<4xsi64>, order = #NCHW}>, tensor<1x1x?x2xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 500, 2]> : tensor<4xsi64>, order = #NCHW}> -> tensor<1x1x?x2x!qElemType, {bounds = #const.OpaqueI64Elements<[1, 1, 500, 2]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK:       [[Q_CAST:%.+]] = IE.QuantizeCast([[Q_ADD]]) {dstElemType = !qElemType1} :
    // CHECK-SAME:    tensor<1x1x?x2x!qElemType, {bounds = #const.OpaqueI64Elements<[1, 1, 500, 2]> : tensor<4xsi64>, order = #NCHW}> -> tensor<1x1x?x2x!qElemType1, {bounds = #const.OpaqueI64Elements<[1, 1, 500, 2]> : tensor<4xsi64>, order = #NCHW}>

    // CHECK:       [[D_CAST:%.+]] = IE.QuantizeCast([[Q_CAST]]) {dstElemType = !qElemType2} :
    // CHECK-SAME:    tensor<1x1x?x2x!qElemType1, {bounds = #const.OpaqueI64Elements<[1, 1, 500, 2]> : tensor<4xsi64>, order = #NCHW}> -> tensor<1x1x?x2x!qElemType2, {bounds = #const.OpaqueI64Elements<[1, 1, 500, 2]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK:       [[D_ADD:%.+]] = IE.Add([[D_CAST]], [[D_CAST]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} :
    // CHECK-SAME:    tensor<1x1x?x2x!qElemType2, {bounds = #const.OpaqueI64Elements<[1, 1, 500, 2]> : tensor<4xsi64>, order = #NCHW}>, tensor<1x1x?x2x!qElemType2, {bounds = #const.OpaqueI64Elements<[1, 1, 500, 2]> : tensor<4xsi64>, order = #NCHW}> -> tensor<1x1x?x2xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 500, 2]> : tensor<4xsi64>, order = #NCHW}>

    // CHECK:  return [[D_ADD]] : tensor<1x1x?x2xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 500, 2]> : tensor<4xsi64>, order = #NCHW}>
}
