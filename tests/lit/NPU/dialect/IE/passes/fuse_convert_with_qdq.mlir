//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --fuse-convert-with-qdq %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

// CHECK: !qElemType = !quant.uniform<u8:f16, 0.95599999999999996:128>
!qElemType = !quant.uniform<u8:f16, 0.956:128>

// CHECK-LABEL: @PerTensor
// CHECK-SAME: [[ARG:%.+]]: tensor<1x3x16x16xui8>
func.func @PerTensor(%arg0: tensor<1x3x16x16xui8>) -> tensor<1x3x16x16xf16> {
    %0 = IE.Convert(%arg0) {dstElemType = f16} : tensor<1x3x16x16xui8> -> tensor<1x3x16x16xf16>
    %1 = IE.Quantize(%0) {dstElemType = !qElemType} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!qElemType>
    %2 = IE.Dequantize(%1) {dstElemType = f16} : tensor<1x3x16x16x!qElemType> -> tensor<1x3x16x16xf16>

    return %2 : tensor<1x3x16x16xf16>

    // CHECK: [[VAL0:%.+]] = IE.QuantizeCast([[ARG]]) {dstElemType = !qElemType} :
    // CHECK-SAME:   tensor<1x3x16x16xui8> -> tensor<1x3x16x16x!qElemType>
    // CHECK: [[VAL1:%.+]] = IE.Dequantize([[VAL0]]) {dstElemType = f16} :
    // CHECK-SAME:   tensor<1x3x16x16x!qElemType> -> tensor<1x3x16x16xf16>

    // CHECK: return [[VAL1]] : tensor<1x3x16x16xf16>
}

// -----

// CHECK: !qElemType = !quant.uniform<u8:f16:1, {0.95599999999999996:128,7.850000e-01:128,5.670000e-01:128}>
!qElemType = !quant.uniform<u8:f16:1, {0.956:128, 0.785:128, 0.567:128}>

// CHECK-LABEL: @PerAxis
// CHECK-SAME: [[ARG:%.+]]: tensor<1x3x16x16xui8>
func.func @PerAxis(%arg0: tensor<1x3x16x16xui8>) -> tensor<1x3x16x16xf16> {
    %0 = IE.Convert(%arg0) {dstElemType = f16} : tensor<1x3x16x16xui8> -> tensor<1x3x16x16xf16>
    %1 = IE.Quantize(%0) {dstElemType = !qElemType} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!qElemType>
    %2 = IE.Dequantize(%1) {dstElemType = f16} : tensor<1x3x16x16x!qElemType> -> tensor<1x3x16x16xf16>

    return %2 : tensor<1x3x16x16xf16>

    // CHECK: [[VAL0:%.+]] = IE.QuantizeCast([[ARG]]) {dstElemType = !qElemType} :
    // CHECK-SAME:   tensor<1x3x16x16xui8> -> tensor<1x3x16x16x!qElemType>
    // CHECK: [[VAL1:%.+]] = IE.Dequantize([[VAL0]]) {dstElemType = f16} :
    // CHECK-SAME:   tensor<1x3x16x16x!qElemType> -> tensor<1x3x16x16xf16>

    // CHECK: return [[VAL1]] : tensor<1x3x16x16xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16, 5.000000e-01:0>

// CHECK-LABEL: @PerTensorDequantizeConvert
// CHECK-SAME: [[ARG:%.+]]: tensor<1x3x16x16x!qElemType>
func.func @PerTensorDequantizeConvert(%arg0: tensor<1x3x16x16x!qElemType>) -> tensor<1x3x16x16xui8> {
    %0 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x3x16x16x!qElemType> -> tensor<1x3x16x16xf16>
    %1 = IE.Convert(%0) {dstElemType = ui8} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16xui8>

    return %1 : tensor<1x3x16x16xui8>

    // CHECK-NOT: IE.QuantizeCast
    // CHECK: [[DEQUANT:%.+]] = IE.Dequantize([[ARG]]) {dstElemType = f16} :
    // CHECK-SAME:   tensor<1x3x16x16x!qElemType> -> tensor<1x3x16x16xf16>
    // CHECK: [[CONVERT:%.+]] = IE.Convert([[DEQUANT]]) {dstElemType = ui8} :
    // CHECK-SAME:   tensor<1x3x16x16xf16> -> tensor<1x3x16x16xui8>

    // CHECK: return [[CONVERT]] : tensor<1x3x16x16xui8>
}


// -----

!qElemType = !quant.uniform<u8:f16:1, {5.000000e-01:0, 7.500000e-01:0, 1.250000e+00:0}>

// CHECK-LABEL: @PerAxisDequantizeConvert
// CHECK-SAME: [[ARG:%.+]]: tensor<1x3x16x16x!qElemType>
func.func @PerAxisDequantizeConvert(%arg0: tensor<1x3x16x16x!qElemType>) -> tensor<1x3x16x16xui8> {
    %0 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x3x16x16x!qElemType> -> tensor<1x3x16x16xf16>
    %1 = IE.Convert(%0) {dstElemType = ui8} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16xui8>

    return %1 : tensor<1x3x16x16xui8>

    // CHECK-NOT: IE.QuantizeCast
    // CHECK: [[DEQUANT:%.+]] = IE.Dequantize([[ARG]]) {dstElemType = f16} :
    // CHECK-SAME:   tensor<1x3x16x16x!qElemType> -> tensor<1x3x16x16xf16>
    // CHECK: [[CONVERT:%.+]] = IE.Convert([[DEQUANT]]) {dstElemType = ui8} :
    // CHECK-SAME:   tensor<1x3x16x16xf16> -> tensor<1x3x16x16xui8>

    // CHECK: return [[CONVERT]] : tensor<1x3x16x16xui8>
}


// -----

!qElemType = !quant.uniform<u8:f16, 1.000000e+00:128>

// CHECK-LABEL: @PerTensorSI8
// CHECK-SAME: [[ARG:%.+]]: tensor<1x12x19x19x!qElemType>
func.func @PerTensorSI8(%arg0: tensor<1x12x19x19x!qElemType>) -> tensor<1x12x19x19xsi8> {
   %0 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x12x19x19x!qElemType> -> tensor<1x12x19x19xf16>
   %1 = IE.Convert(%0) {dstElemType = si8} : tensor<1x12x19x19xf16> -> tensor<1x12x19x19xsi8>

   return %1 : tensor<1x12x19x19xsi8>

   // CHECK: [[VAL0:%.+]] = IE.Dequantize([[ARG]]) {dstElemType = f16} :
   // CHECK-SAME: tensor<1x12x19x19x!qElemType> -> tensor<1x12x19x19xf16>
   // CHECK: [[VAL1:%.+]] = IE.Convert([[VAL0]]) {dstElemType = si8} :
   // CHECK-SAME: tensor<1x12x19x19xf16> -> tensor<1x12x19x19xsi8>
   // CHECK: return [[VAL1]] :  tensor<1x12x19x19xsi8>

}

// -----

!qElemType = !quant.uniform<u8:f16, 1.000000e+00:49>

// CHECK-LABEL: @WithQuantizeCastNonZeroZeroPoint
// CHECK-SAME: [[ARG:%.+]]: tensor<1x3x16x16xui8>
func.func @WithQuantizeCastNonZeroZeroPoint(%arg0: tensor<1x3x16x16xui8>) -> tensor<1x3x16x16xui8> {
    %0 = IE.QuantizeCast(%arg0) {dstElemType = !qElemType} : tensor<1x3x16x16xui8> -> tensor<1x3x16x16x!qElemType>
    %1 = IE.Dequantize(%0) {dstElemType = f16} : tensor<1x3x16x16x!qElemType> -> tensor<1x3x16x16xf16>
    %2 = IE.Convert(%1) {dstElemType = ui8} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16xui8>

    return %2 : tensor<1x3x16x16xui8>

    // CHECK: [[VAL0:%.+]] = IE.QuantizeCast([[ARG]]) {dstElemType = !qElemType} :
    // CHECK-SAME:   tensor<1x3x16x16xui8> -> tensor<1x3x16x16x!qElemType>
    // CHECK: [[VAL1:%.+]] = IE.Dequantize([[VAL0]]) {dstElemType = f16} :
    // CHECK-SAME:   tensor<1x3x16x16x!qElemType> -> tensor<1x3x16x16xf16>
    // CHECK: [[VAL2:%.+]] = IE.Convert([[VAL1]]) {dstElemType = ui8} :
    // CHECK-SAME:   tensor<1x3x16x16xf16> -> tensor<1x3x16x16xui8>

    // CHECK: return [[VAL2]] : tensor<1x3x16x16xui8>
}

// -----

!qElemType = !quant.uniform<u8:f16, 1.000000e+00:0>

// CHECK-LABEL: @WithQuantizeCastZeroZeroPoint
// CHECK-SAME: [[ARG:%.+]]: tensor<1x3x16x16xui8>
func.func @WithQuantizeCastZeroZeroPoint(%arg0: tensor<1x3x16x16xui8>) -> tensor<1x3x16x16xui8> {
    %0 = IE.QuantizeCast(%arg0) {dstElemType = !qElemType} : tensor<1x3x16x16xui8> -> tensor<1x3x16x16x!qElemType>
    %1 = IE.Dequantize(%0) {dstElemType = f16} : tensor<1x3x16x16x!qElemType> -> tensor<1x3x16x16xf16>
    %2 = IE.Convert(%1) {dstElemType = ui8} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16xui8>

    return %2 : tensor<1x3x16x16xui8>

    // CHECK: [[VAL0:%.+]] = IE.QuantizeCast([[ARG]]) {dstElemType = !qElemType} :
    // CHECK-SAME:   tensor<1x3x16x16xui8> -> tensor<1x3x16x16x!qElemType>
    // CHECK: [[VAL1:%.+]] = IE.QuantizeCast([[VAL0]]) {dstElemType = ui8} :
    // CHECK-SAME:   tensor<1x3x16x16x!qElemType> -> tensor<1x3x16x16xui8>

    // CHECK: return [[VAL1]] : tensor<1x3x16x16xui8>
}

// -----

!qElemType_mismatch = !quant.uniform<u8:f16, 0.956:128>

// CHECK-LABEL: @NotFuseConvertQuantizeSignednessMismatch
// CHECK-SAME: [[ARG:%.+]]: tensor<1x3x16x16xsi8>
func.func @NotFuseConvertQuantizeSignednessMismatch(%arg0: tensor<1x3x16x16xsi8>) -> tensor<1x3x16x16xf16> {
    %0 = IE.Convert(%arg0) {dstElemType = f16} : tensor<1x3x16x16xsi8> -> tensor<1x3x16x16xf16>
    %1 = IE.Quantize(%0) {dstElemType = !qElemType_mismatch} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!qElemType_mismatch>
    %2 = IE.Dequantize(%1) {dstElemType = f16} : tensor<1x3x16x16x!qElemType_mismatch> -> tensor<1x3x16x16xf16>

    return %2 : tensor<1x3x16x16xf16>

    // CHECK: [[CONVERT:%.+]] = IE.Convert([[ARG]]) {dstElemType = f16} :
    // CHECK-SAME:   tensor<1x3x16x16xsi8> -> tensor<1x3x16x16xf16>
    // CHECK: [[QUANT:%.+]] = IE.Quantize([[CONVERT]]) {dstElemType = !qElemType} :
    // CHECK-SAME:   tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!qElemType>
    // CHECK: [[DEQUANT:%.+]] = IE.Dequantize([[QUANT]]) {dstElemType = f16} :
    // CHECK-SAME:   tensor<1x3x16x16x!qElemType> -> tensor<1x3x16x16xf16>

    // CHECK: return [[DEQUANT]] : tensor<1x3x16x16xf16>
}

// -----

!qElemType_mismatch_axis = !quant.uniform<u8:f16:1, {0.956:128, 0.785:128, 0.567:128}>

// CHECK-LABEL: @NotFuseConvertQuantizeSignednessMismatchPerAxis
// CHECK-SAME: [[ARG:%.+]]: tensor<1x3x16x16xsi8>
func.func @NotFuseConvertQuantizeSignednessMismatchPerAxis(%arg0: tensor<1x3x16x16xsi8>) -> tensor<1x3x16x16xf16> {
    %0 = IE.Convert(%arg0) {dstElemType = f16} : tensor<1x3x16x16xsi8> -> tensor<1x3x16x16xf16>
    %1 = IE.Quantize(%0) {dstElemType = !qElemType_mismatch_axis} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!qElemType_mismatch_axis>
    %2 = IE.Dequantize(%1) {dstElemType = f16} : tensor<1x3x16x16x!qElemType_mismatch_axis> -> tensor<1x3x16x16xf16>

    return %2 : tensor<1x3x16x16xf16>

    // CHECK: [[CONVERT:%.+]] = IE.Convert([[ARG]]) {dstElemType = f16} :
    // CHECK-SAME:   tensor<1x3x16x16xsi8> -> tensor<1x3x16x16xf16>
    // CHECK: [[QUANT:%.+]] = IE.Quantize([[CONVERT]]) {dstElemType = !qElemType} :
    // CHECK-SAME:   tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!qElemType>
    // CHECK: [[DEQUANT:%.+]] = IE.Dequantize([[QUANT]]) {dstElemType = f16} :
    // CHECK-SAME:   tensor<1x3x16x16x!qElemType> -> tensor<1x3x16x16xf16>

    // CHECK: return [[DEQUANT]] : tensor<1x3x16x16xf16>
}

// -----

!qElemType_signed = !quant.uniform<i8:f16, 0.956:0>

// CHECK: !qElemType = !quant.uniform<i8:f16, 0.95599999999999996>

// CHECK-LABEL: @FuseConvertQuantizeSignednessMatch
// CHECK-SAME: [[ARG:%.+]]: tensor<1x3x16x16xsi8>
func.func @FuseConvertQuantizeSignednessMatch(%arg0: tensor<1x3x16x16xsi8>) -> tensor<1x3x16x16xf16> {
    %0 = IE.Convert(%arg0) {dstElemType = f16} : tensor<1x3x16x16xsi8> -> tensor<1x3x16x16xf16>
    %1 = IE.Quantize(%0) {dstElemType = !qElemType_signed} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!qElemType_signed>
    %2 = IE.Dequantize(%1) {dstElemType = f16} : tensor<1x3x16x16x!qElemType_signed> -> tensor<1x3x16x16xf16>

    return %2 : tensor<1x3x16x16xf16>

    // CHECK: [[QUANTIZE_CAST:%.+]] = IE.QuantizeCast([[ARG]]) {dstElemType = !qElemType} :
    // CHECK-SAME:   tensor<1x3x16x16xsi8> -> tensor<1x3x16x16x!qElemType>
    // CHECK: [[DEQUANT:%.+]] = IE.Dequantize([[QUANTIZE_CAST]]) {dstElemType = f16} :
    // CHECK-SAME:   tensor<1x3x16x16x!qElemType> -> tensor<1x3x16x16xf16>

    // CHECK: return [[DEQUANT]] : tensor<1x3x16x16xf16>
}
