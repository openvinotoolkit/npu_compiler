//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

// CHECK: !qElemType = !quant.uniform<u8:f16, 0.0173492431640625:32>
// CHECK: !qElemType1 = !quant.uniform<u8:f16, 0.0064682904411764702:128>
!qElemType = !quant.uniform<u8:f16, 0.0173492431640625:32>
!qElemType1 = !quant.uniform<u8:f16, 0.01293658088235294:64>
!qElemType2 = !quant.uniform<u8:f16, 0.0064682904411764702:128>

// CHECK-LABEL: @FuseQuantizeCasts
func.func @FuseQuantizeCasts(%arg0: tensor<1x16x32x64x!qElemType>) -> tensor<1x16x32x64x!qElemType2> {
    %FIRST_QUANT_CAST = IE.QuantizeCast(%arg0) {
        dstElemType = !qElemType1
    } : tensor<1x16x32x64x!qElemType> -> tensor<1x16x32x64x!qElemType1>

    %SECOND_QUANT_CAST = IE.QuantizeCast(%FIRST_QUANT_CAST) {
        dstElemType = !qElemType2
    } : tensor<1x16x32x64x!qElemType1> -> tensor<1x16x32x64x!qElemType2>

    return %SECOND_QUANT_CAST : tensor<1x16x32x64x!qElemType2>

    // CHECK:       [[QUANT_CAST:%.*]] = IE.QuantizeCast(%arg0) {
    // CHECK-SAME:      dstElemType = !qElemType1
    // CHECK-SAME:  } : tensor<1x16x32x64x!qElemType> -> tensor<1x16x32x64x!qElemType1>

    // CHECK:       return [[QUANT_CAST]] : tensor<1x16x32x64x!qElemType1>
}

// -----

// CHECK: !qElemType = !quant.uniform<u8:f16, 0.0173492431640625:32>
// CHECK: !qElemType1 = !quant.uniform<u8:f16, 0.0064682904411764702:128>
// CHECK: !qElemType2 = !quant.uniform<u8:f16, 0.01293658088235294:64>
!qElemType = !quant.uniform<u8:f16, 0.0173492431640625:32>
!qElemType1 = !quant.uniform<u8:f16, 0.01293658088235294:64>
!qElemType2 = !quant.uniform<u8:f16, 0.0064682904411764702:128>

// CHECK-LABEL: @FuseQuantCastsMultipleConsumers
func.func @FuseQuantCastsMultipleConsumers(%arg0: tensor<1x16x32x64x!qElemType>) -> tensor<1x16x32x64x!qElemType2> {
    %FIRST_QUANT_CAST = IE.QuantizeCast(%arg0) {
        dstElemType = !qElemType1
    } : tensor<1x16x32x64x!qElemType> -> tensor<1x16x32x64x!qElemType1>

    %ADD = IE.Add(%FIRST_QUANT_CAST, %FIRST_QUANT_CAST) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    } : tensor<1x16x32x64x!qElemType1>, tensor<1x16x32x64x!qElemType1> -> tensor<1x16x32x64x!qElemType1>

    %ADD_QUANT_CAST = IE.QuantizeCast(%ADD) {
        dstElemType = !qElemType2
    } : tensor<1x16x32x64x!qElemType1> -> tensor<1x16x32x64x!qElemType2>

    %SECOND_QUANT_CAST = IE.QuantizeCast(%FIRST_QUANT_CAST) {
        dstElemType = !qElemType2
    } : tensor<1x16x32x64x!qElemType1> -> tensor<1x16x32x64x!qElemType2>

    %MUL = IE.Multiply(%SECOND_QUANT_CAST, %ADD_QUANT_CAST) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    } : tensor<1x16x32x64x!qElemType2>, tensor<1x16x32x64x!qElemType2> -> tensor<1x16x32x64x!qElemType2>

    return %MUL : tensor<1x16x32x64x!qElemType2>

    // CHECK:       [[FIRST_QUANT_CAST:%.*]] = IE.QuantizeCast(%arg0) {
    // CHECK-SAME:      dstElemType = !qElemType2
    // CHECK-SAME:  } : tensor<1x16x32x64x!qElemType> -> tensor<1x16x32x64x!qElemType2>

    // CHECK:       [[ADD:%.*]] = IE.Add([[FIRST_QUANT_CAST]], [[FIRST_QUANT_CAST]]) {
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    // CHECK-SAME:  } : tensor<1x16x32x64x!qElemType2>, tensor<1x16x32x64x!qElemType2>
    // CHECK-SAME:  -> tensor<1x16x32x64x!qElemType2>

    // CHECK:       [[ADD_QUANT_CAST:%.*]] = IE.QuantizeCast([[ADD]]) {
    // CHECK-SAME:      dstElemType = !qElemType1
    // CHECK-SAME:  } : tensor<1x16x32x64x!qElemType2> -> tensor<1x16x32x64x!qElemType1>

    // Note that the second IE.QuantizeCast accepts arg0, not FIRST_QUANT_CAST
    // CHECK:       [[SECOND_QUANT_CAST:%.*]] = IE.QuantizeCast(%arg0) {
    // CHECK-SAME:      dstElemType = !qElemType1
    // CHECK-SAME:  } : tensor<1x16x32x64x!qElemType> -> tensor<1x16x32x64x!qElemType1>

    // CHECK:       [[MUL:%.*]] = IE.Multiply([[SECOND_QUANT_CAST]], [[ADD_QUANT_CAST]]) {
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    // CHECK-SAME:  } : tensor<1x16x32x64x!qElemType1>, tensor<1x16x32x64x!qElemType1>
    // CHECK-SAME:  -> tensor<1x16x32x64x!qElemType1>

    // CHECK:       return [[MUL]] : tensor<1x16x32x64x!qElemType1>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0173492431640625:32>

// CHECK-LABEL: @ConstantFolding
func.func @ConstantFolding() -> tensor<2x2x!qElemType> {
    %cst = const.Declare tensor<2x2xui8> = dense<[[1, 2], [3, 4]]> : tensor<2x2xui8>
    %quant_cast = IE.QuantizeCast(%cst) { dstElemType = !qElemType } : tensor<2x2xui8> -> tensor<2x2x!qElemType>
    return %quant_cast : tensor<2x2x!qElemType>
    // CHECK: [[CST:%.+]] = const.Declare tensor<2x2x!qElemType> = dense<{{\[\[}}1, 2], [3, 4]]> : tensor<2x2xui8>, [#const.CastElemType<!qElemType>]
    // CHECK: return [[CST]] : tensor<2x2x!qElemType>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0173492431640625:32>

// CHECK-LABEL: @ConstantFoldingNoOp
func.func @ConstantFoldingNoOp() -> tensor<2x2x!qElemType> {
    %cst = const.Declare tensor<2x2x!qElemType> = dense<[[1, 2], [3, 4]]> : tensor<2x2xui8>, [#const.CastElemType<!qElemType>]
    %quant_cast = IE.QuantizeCast(%cst) { dstElemType = !qElemType } : tensor<2x2x!qElemType> -> tensor<2x2x!qElemType>
    return %quant_cast : tensor<2x2x!qElemType>
    // CHECK: [[CST:%.+]] = const.Declare tensor<2x2x!qElemType> = dense<{{\[\[}}1, 2], [3, 4]]> : tensor<2x2xui8>, [#const.CastElemType<!qElemType>]
    // CHECK: return [[CST]] : tensor<2x2x!qElemType>
}

// -----

!quantileFloatType = !QuantileFloat.quantileFloat<ui4:f16, {-1.000000e+00,-0.69619280099868774,-0.52507305145263672,-0.39491748809814453,-0.28444138169288635,-0.18477343022823334,-0.091050036251544952,0.000000e+00,0.07958029955625534,0.16093020141124725,0.24611230194568634,0.33791524171829224,0.44070982933044434,0.56261700391769409,0.72295683622360229,1.000000e+00}>
!qElemType = !quant.quantile<u4:f16:f16, {-1.000000e+00,-0.69619280099868774,-0.52507305145263672,-0.39491748809814453,-0.28444138169288635,-0.18477343022823334,-0.091050036251544952,0.000000e+00,0.07958029955625534,0.16093020141124725,0.24611230194568634,0.33791524171829224,0.44070982933044434,0.56261700391769409,0.72295683622360229,1.000000e+00}:0.07874348958333334>

// CHECK-LABEL: @QuantizeCastNF4
// CHECK-SAME:  [[INPUT0:%.+]]: tensor<16x32x1x1x!QuantileFloat.quantileFloat<ui4:f16, {-1.000000e+00,-0.69619280099868774,-0.52507305145263672,-0.39491748809814453,-0.28444138169288635,-0.18477343022823334,-0.091050036251544952,0.000000e+00,0.07958029955625534,0.16093020141124725,0.24611230194568634,0.33791524171829224,0.44070982933044434,0.56261700391769409,0.72295683622360229,1.000000e+00}>>
// CHECK-SAME:  [[INPUT1:%.+]]: tensor<16x32x1x1x!qElemType>
func.func @QuantizeCastNF4(%arg0: tensor<16x32x1x1x!quantileFloatType>, %arg1: tensor<16x32x1x1x!qElemType>) -> (tensor<16x32x1x1x!qElemType>, tensor<16x32x1x1x!quantileFloatType>) {
    %0 = IE.QuantizeCast(%arg0) {dstElemType = !qElemType} : tensor<16x32x1x1x!quantileFloatType> -> tensor<16x32x1x1x!qElemType>
    %1 = IE.QuantizeCast(%arg1) {dstElemType = !quantileFloatType} : tensor<16x32x1x1x!qElemType> -> tensor<16x32x1x1x!quantileFloatType>

    return %0, %1 : tensor<16x32x1x1x!qElemType>, tensor<16x32x1x1x!quantileFloatType>

    // CHECK:   [[FIRST_QUANT_CAST:%.+]] = IE.QuantizeCast([[INPUT0]]) {dstElemType = !qElemType} : tensor<16x32x1x1x!QuantileFloat.quantileFloat<ui4:f16, {-1.000000e+00,-0.69619280099868774,-0.52507305145263672,-0.39491748809814453,-0.28444138169288635,-0.18477343022823334,-0.091050036251544952,0.000000e+00,0.07958029955625534,0.16093020141124725,0.24611230194568634,0.33791524171829224,0.44070982933044434,0.56261700391769409,0.72295683622360229,1.000000e+00}>> -> tensor<16x32x1x1x!qElemType>
    // CHECK:   [[SECOND_QUANT_CAST:%.+]] = IE.QuantizeCast([[INPUT1]]) {dstElemType = !QuantileFloat.quantileFloat<ui4:f16, {-1.000000e+00,-0.69619280099868774,-0.52507305145263672,-0.39491748809814453,-0.28444138169288635,-0.18477343022823334,-0.091050036251544952,0.000000e+00,0.07958029955625534,0.16093020141124725,0.24611230194568634,0.33791524171829224,0.44070982933044434,0.56261700391769409,0.72295683622360229,1.000000e+00}>} : tensor<16x32x1x1x!qElemType> -> tensor<16x32x1x1x!QuantileFloat.quantileFloat<ui4:f16, {-1.000000e+00,-0.69619280099868774,-0.52507305145263672,-0.39491748809814453,-0.28444138169288635,-0.18477343022823334,-0.091050036251544952,0.000000e+00,0.07958029955625534,0.16093020141124725,0.24611230194568634,0.33791524171829224,0.44070982933044434,0.56261700391769409,0.72295683622360229,1.000000e+00}>>
    // CHECK:   return [[FIRST_QUANT_CAST]], [[SECOND_QUANT_CAST]]
}
