//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% revision-id=3" --convert-quantize-ops-to-nce-ops %s | FileCheck %s
// REQUIRES: platform-NPU6010
// Baseline: REVISION_B via revision-id=3.

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

!qElemType = !quant.uniform<u16:f32, 8.0507534851041623E-5:3325>

// CHECK-LABEL: @ConvertToGroupConvAsymmetricU16Quant
// CHECK:       ([[INPUT:%.+]]: tensor<1x320x64x64xf16>)
func.func @ConvertToGroupConvAsymmetricU16Quant(%arg0 : tensor<1x320x64x64xf16>) -> tensor<1x320x64x64x!qElemType> {
    %0 = IE.Quantize(%arg0) {dstElemType = !qElemType } : tensor<1x320x64x64xf16> -> tensor<1x320x64x64x!qElemType>
    return %0 : tensor<1x320x64x64x!qElemType>

    // CHECK:  [[CST:%.+]] = const.Declare tensor<320x1x1x1xf16> = dense<1.000000e+00> : tensor<320x1x1x1xf16>
    // CHECK:  [[GCONV:%.+]] = IE.GroupConvolution([[INPUT]], [[CST]]) {dilations = [1, 1], groups = 320 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x320x64x64xf16>, tensor<320x1x1x1xf16> -> tensor<1x320x64x64x!qElemType>
    // CHECK:  return [[GCONV]] : tensor<1x320x64x64x!qElemType>
}

// -----

!qElemType = !quant.uniform<u16:f32, 8.0507534851041623E-5:0>

// CHECK-LABEL: func.func @ConvertToGroupConvSymmetricU16Quant
// CHECK:       ([[INPUT:%.+]]: tensor<1x320x64x64xf16>)
func.func @ConvertToGroupConvSymmetricU16Quant(%arg0 : tensor<1x320x64x64xf16>) -> tensor<1x320x64x64x!qElemType> {
    %0 = IE.Quantize(%arg0) {dstElemType = !qElemType } : tensor<1x320x64x64xf16> -> tensor<1x320x64x64x!qElemType>
    return %0 : tensor<1x320x64x64x!qElemType>

    // CHECK:  [[CST:%.+]] = const.Declare tensor<320x1x1x1xf16> = dense<1.000000e+00> : tensor<320x1x1x1xf16>
    // CHECK:  [[CONV:%.+]] = IE.GroupConvolution([[INPUT]], [[CST]]) {dilations = [1, 1], groups = 320 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x320x64x64xf16>, tensor<320x1x1x1xf16> -> tensor<1x320x64x64x!qElemType>
    // CHECK:  return [[CONV]] : tensor<1x320x64x64x!qElemType>
}
