//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --fuse-convert-with-quantize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
// CHECK: !qElemType = !quant.uniform<u8:f16, 1.000000e+00>
!qElemType = !quant.uniform<u8:f16, 0.956:128>

// CHECK-LABEL: @PerTensor
func.func @PerTensor(%arg0: tensor<1x3x16x16xui8>) -> tensor<1x3x16x16xf16> {
    %0 = IE.Convert(%arg0) {dstElemType = f16} : tensor<1x3x16x16xui8> -> tensor<1x3x16x16xf16>
    %1 = IE.Quantize(%0) {dstElemType = !qElemType} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!qElemType>
    %2 = IE.Dequantize(%1) {dstElemType = f16} : tensor<1x3x16x16x!qElemType> -> tensor<1x3x16x16xf16>

    return %2 : tensor<1x3x16x16xf16>

    //CHECK: [[VAL0:%.*]] = IE.QuantizeCast(%arg0) {dstElemType = !qElemType} :
    //CHECK-SAME:   tensor<1x3x16x16xui8> -> tensor<1x3x16x16x!qElemType>
    //CHECK: [[VAL1:%.*]] = IE.Dequantize([[VAL0]]) {dstElemType = f16} :
    //CHECK-SAME:   tensor<1x3x16x16x!qElemType> -> tensor<1x3x16x16xf16>

    //CHECK: return [[VAL1]] : tensor<1x3x16x16xf16>
}

// -----

// CHECK: !qElemType = !quant.uniform<u8:f16:1, {1.000000e+00,1.000000e+00,1.000000e+00}>
!qElemType = !quant.uniform<u8:f16:1, {0.956:128, 0.785:128, 0.567:128}>

// CHECK-LABEL: @PerAxis
func.func @PerAxis(%arg0: tensor<1x3x16x16xui8>) -> tensor<1x3x16x16xf16> {
    %0 = IE.Convert(%arg0) {dstElemType = f16} : tensor<1x3x16x16xui8> -> tensor<1x3x16x16xf16>
    %1 = IE.Quantize(%0) {dstElemType = !qElemType} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!qElemType>
    %2 = IE.Dequantize(%1) {dstElemType = f16} : tensor<1x3x16x16x!qElemType> -> tensor<1x3x16x16xf16>

    return %2 : tensor<1x3x16x16xf16>

    //CHECK: [[VAL0:%.*]] = IE.QuantizeCast(%arg0) {dstElemType = !qElemType} :
    //CHECK-SAME:   tensor<1x3x16x16xui8> -> tensor<1x3x16x16x!qElemType>
    //CHECK: [[VAL1:%.*]] = IE.Dequantize([[VAL0]]) {dstElemType = f16} :
    //CHECK-SAME:   tensor<1x3x16x16x!qElemType> -> tensor<1x3x16x16xf16>

    //CHECK: return [[VAL1]] : tensor<1x3x16x16xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16, 1.0:128>

// CHECK-LABEL: @PerTensorDequantizeConvert
func.func @PerTensorDequantizeConvert(%arg0: tensor<1x3x16x16x!qElemType>) -> tensor<1x3x16x16xui8> {
    %0 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x3x16x16x!qElemType> -> tensor<1x3x16x16xf16>
    %1 = IE.Convert(%0) {dstElemType = ui8} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16xui8>

    return %1 : tensor<1x3x16x16xui8>

    //CHECK: [[VAL0:%.*]] = IE.QuantizeCast(%arg0) {dstElemType = ui8} :
    //CHECK-SAME:   tensor<1x3x16x16x!qElemType> -> tensor<1x3x16x16xui8>

    //CHECK: return [[VAL0]] : tensor<1x3x16x16xui8>
}


// -----

!qElemType = !quant.uniform<u8:f16:1, {0.956:128, 0.785:128, 0.567:128}>

// CHECK-LABEL: @PerAxisDequantizeConvert
func.func @PerAxisDequantizeConvert(%arg0: tensor<1x3x16x16x!qElemType>) -> tensor<1x3x16x16xui8> {
    %0 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x3x16x16x!qElemType> -> tensor<1x3x16x16xf16>
    %1 = IE.Convert(%0) {dstElemType = ui8} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16xui8>

    return %1 : tensor<1x3x16x16xui8>

    //CHECK: [[VAL0:%.*]] = IE.QuantizeCast(%arg0) {dstElemType = ui8} :
    //CHECK-SAME:   tensor<1x3x16x16x!qElemType> -> tensor<1x3x16x16xui8>

    //CHECK: return [[VAL0]] : tensor<1x3x16x16xui8>
}


// -----

!qElemType = !quant.uniform<u8:f16, 1.000000e+00:128>

// CHECK-LABEL: @PerTensorSI8
func.func @PerTensorSI8(%arg0: tensor<1x12x19x19x!qElemType>) -> tensor<1x12x19x19xsi8> {
   %0 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x12x19x19x!qElemType> -> tensor<1x12x19x19xf16>
   %1 = IE.Convert(%0) {dstElemType = si8} : tensor<1x12x19x19xf16> -> tensor<1x12x19x19xsi8>

   return %1 : tensor<1x12x19x19xsi8>

   //CHECK: [[VAL0:%.*]] IE.Dequantize(%arg0) {dstElemType = f16} :
   //CHECK-SAME: tensor<1x12x19x19x!qElemType> -> tensor<1x12x19x19xf16>
   //CHECK: [[VAL1:%.*]]  = IE.Convert(%0) {dstElemType = si8} :
   //CHECK-SAME: tensor<1x12x19x19xf16> -> tensor<1x12x19x19xsi8>
   //CHECK: return [[VAL1]] :  tensor<1x12x19x19xsi8>

}
