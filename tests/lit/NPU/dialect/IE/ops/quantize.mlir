//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% allow-custom-values=true" --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

!qElemType = !quant.uniform<u8:f16, 2.4627450980392158>
!qElemType2 = !quant.uniform<u8:f16, 1.23423>

func.func @ConstFoldWithRealQuantize() -> tensor<1x8x4x4x!qElemType> {
    %0 = const.Declare tensor<1x8x4x4xf16> = dense<5.0> : tensor<1x8x4x4xf32>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType2>, #const.Dequantize]
    %1 = IE.Quantize(%0) {dstElemType = !qElemType}: tensor<1x8x4x4xf16> -> tensor<1x8x4x4x!qElemType>
    return %1 : tensor<1x8x4x4x!qElemType>

    // CHECK:       [[VAL0:%.*]] = const.Declare tensor<1x8x4x4x!qElemType> = dense<5.000000e+00> : tensor<1x8x4x4xf32>,
    // CHECK-SAME:          [#const.CastElemType<ui8>, #const.CastElemType<!qElemType1>,
    // CHECK-SAME:           #const.Dequantize, #const.Quantize<!qElemType>, #const.CastElemType<!qElemType>]
    // CHECK-NOT:   IE.Quantize
    // CHECK:       return [[VAL0]]
}

// -----

!qElemType = !quant.uniform<u8:f16, 2.4627450980392158>

func.func @ConstFold() -> tensor<1x8x4x4x!qElemType> {
    %0 = const.Declare tensor<1x8x4x4xf32> = dense<5.0> : tensor<1x8x4x4xf32>
    %1 = IE.Quantize(%0) {dstElemType = !qElemType}: tensor<1x8x4x4xf32> -> tensor<1x8x4x4x!qElemType>
    return %1 : tensor<1x8x4x4x!qElemType>

    // CHECK:       [[VAL0:%.*]] = const.Declare tensor<1x8x4x4x!qElemType> =
    // CHECK-SAME:       dense<5.000000e+00> : tensor<1x8x4x4xf32>,
    // CHECK-SAME:       [#const.Quantize<!qElemType>, #const.CastElemType<!qElemType>]
    // CHECK-NOT:   IE.Quantize
    // CHECK:       return [[VAL0]]
}

// -----

!qElemType = !quant.uniform<u8:f16, 2.4627450980392158>

func.func @FuseDequantQuantWithSeveralUses(%arg0: tensor<1x8x4x4x!qElemType>) -> (tensor<1x8x4x4xf16>, tensor<1x8x4x4x!qElemType>) {
    %0 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x8x4x4x!qElemType> -> tensor<1x8x4x4xf16>
    %1 = IE.ReLU(%0) : tensor<1x8x4x4xf16> -> tensor<1x8x4x4xf16>
    %2 = IE.Quantize(%0) {dstElemType = !qElemType}: tensor<1x8x4x4xf16> -> tensor<1x8x4x4x!qElemType>
    return %1, %2 : tensor<1x8x4x4xf16>, tensor<1x8x4x4x!qElemType>

    // CHECK:   [[VAL0:%.*]] = IE.Dequantize(%arg0)
    // CHECK-NOT:   IE.Quantize
    // CHECK:       [[VAL1:%.*]] = IE.ReLU([[VAL0]])
    // CHECK:       return [[VAL1]], %arg0 : tensor<1x8x4x4xf16>, tensor<1x8x4x4x!qElemType>
}

// -----

!qElemType = !quant.uniform<u8:f16, 2.4627450980392158>

func.func @FuseDequantQuant(%arg0: tensor<1x8x4x4x!qElemType>) -> tensor<1x8x4x4x!qElemType> {
    %1 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x8x4x4x!qElemType> -> tensor<1x8x4x4xf16>
    %2 = IE.Quantize(%1) {dstElemType = !qElemType}: tensor<1x8x4x4xf16> -> tensor<1x8x4x4x!qElemType>
    return %2 : tensor<1x8x4x4x!qElemType>

    // CHECK-NOT:   IE.Dequantize
    // CHECK-NOT:   IE.Quantize
    // CHECK:       return %arg0 : tensor<1x8x4x4x!qElemType>
}

// -----

!qElemType = !quant.uniform<u8<0:254>:f16:1, {8.7179349163385824E-4:127,5.2096149114173233E-4:127}>

func.func @FuseDequantQuantPerAxis(%arg0: tensor<1x2x4x4x!qElemType>) -> tensor<1x2x4x4x!qElemType> {
    %1 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x2x4x4x!qElemType> -> tensor<1x2x4x4xf16>
    %2 = IE.Quantize(%1) {dstElemType = !qElemType}: tensor<1x2x4x4xf16> -> tensor<1x2x4x4x!qElemType>
    return %2 : tensor<1x2x4x4x!qElemType>

    // CHECK-NOT:   IE.Dequantize
    // CHECK-NOT:   IE.Quantize
    // CHECK:       return %arg0 : tensor<1x2x4x4x!qElemType>
}

// -----

!qElemType = !quant.uniform<u8:f16, 1.4627450980392158>
!qElemType1 = !quant.uniform<u8:f16, 2.3463457356746546>

func.func @DifferentQuantizationParams(%arg0: tensor<1x8x4x4x!qElemType>) -> tensor<1x8x4x4x!qElemType1> {
    %0 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x8x4x4x!qElemType> -> tensor<1x8x4x4xf16>
    %1 = IE.Quantize(%0) {dstElemType = !qElemType1}: tensor<1x8x4x4xf16> -> tensor<1x8x4x4x!qElemType1>
    return %1 : tensor<1x8x4x4x!qElemType1>

    // CHECK-DAG:   IE.Dequantize
    // CHECK-DAG:   IE.Quantize
    // CHECK:   return %1 : tensor<1x8x4x4x!qElemType1>
}

// -----

!qElemType = !quant.uniform<u8<0:254>:f16:1, {8.7179349163385824E-4:127,5.2096149114173233E-4:127}>
!qElemType1 = !quant.uniform<u8<0:254>:f16:1, {3.4678567856785681E-4:127,9.5675698679264696E-4:127}>

func.func @DifferentQuantizationParams(%arg0: tensor<1x2x4x4x!qElemType>) -> tensor<1x2x4x4x!qElemType1> {
    %0 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x2x4x4x!qElemType> -> tensor<1x2x4x4xf16>
    %1 = IE.Quantize(%0) {dstElemType = !qElemType1}: tensor<1x2x4x4xf16> -> tensor<1x2x4x4x!qElemType1>
    return %1 : tensor<1x2x4x4x!qElemType1>

    // CHECK-DAG:   IE.Dequantize
    // CHECK-DAG:   IE.Quantize
    // CHECK:   return %1 : tensor<1x2x4x4x!qElemType1>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.015704019396912818:153>
!qElemType1 = !quant.uniform<u8:f16, 0.015704022669324687:153>

// CHECK-LABEL:  func.func @FuseReshapeQuantizationParamsDifferent
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x12x512x8xf16>)
module {
config.PipelineOptions @Options {
    config.Option @config.EnableAdaptiveStripping : true
}

func.func @FuseReshapeQuantizationParamsDifferent(%input: tensor<1x12x512x8xf16>) -> tensor<1x12288x4x1xf16> {
    %0 = IE.Quantize(%input) {dstElemType = !qElemType} : tensor<1x12x512x8xf16> -> tensor<1x12x512x8x!qElemType>
    %1 = IE.Dequantize(%0) {dstElemType = f16} : tensor<1x12x512x8x!qElemType> -> tensor<1x12x512x8xf16>
    %2 = IE.Reshape(%1) {shape_value = [1, 12288, 4, 1]} : tensor<1x12x512x8xf16> -> tensor<1x12288x4x1xf16>
    %3 = IE.Quantize(%2) {dstElemType = !qElemType1} : tensor<1x12288x4x1xf16> -> tensor<1x12288x4x1x!qElemType1>
    %4 = IE.Dequantize(%3) {dstElemType = f16} : tensor<1x12288x4x1x!qElemType1> -> tensor<1x12288x4x1xf16>
    return %4 : tensor<1x12288x4x1xf16>

    // CHECK: [[QUANTIZE:%.+]] = IE.Quantize([[INPUT]])
    // CHECK-SAME:   {dstElemType = !qElemType} : tensor<1x12x512x8xf16> -> tensor<1x12x512x8x!qElemType>
    // CHECK-NOT:   IE.Dequantize
    // CHECK:      [[RESHAPE:%.+]] = IE.Reshape([[QUANTIZE]])
    // CHECK-SAME:   {shape_value = [1, 12288, 4, 1]} : tensor<1x12x512x8x!qElemType> -> tensor<1x12288x4x1x!qElemType>
    // CHECK-NOT:   IE.Quantize
    // CHECK: [[DEQUANTIZE:%.+]] = IE.Dequantize([[RESHAPE]])
    // CHECK-SAME:   {dstElemType = f16} : tensor<1x12288x4x1x!qElemType> -> tensor<1x12288x4x1xf16>
    // CHECK: return [[DEQUANTIZE]]
}
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.015704019396912818:153>
!qElemType1 = !quant.uniform<u8:f16, 0.015704022669324687:153>

// CHECK-LABEL:  func.func @FuseAffineReshapeQuantizationParamsDifferent
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x12x512x8xf16>)
module {
config.PipelineOptions @Options {
    config.Option @config.EnableAdaptiveStripping : true
}

func.func @FuseAffineReshapeQuantizationParamsDifferent(%input: tensor<1x12x512x8xf16>) -> tensor<1x12288x4x1xf16> {
    %0 = IE.Quantize(%input) {dstElemType = !qElemType} : tensor<1x12x512x8xf16> -> tensor<1x12x512x8x!qElemType>
    %1 = IE.Dequantize(%0) {dstElemType = f16} : tensor<1x12x512x8x!qElemType> -> tensor<1x12x512x8xf16>
    %2 = IE.AffineReshape(%1) {shape_value = [1, 12288, 4, 1], dim_mapping = [[0], [1], [2], [3]]} : tensor<1x12x512x8xf16> -> tensor<1x12288x4x1xf16>
    %3 = IE.Quantize(%2) {dstElemType = !qElemType1} : tensor<1x12288x4x1xf16> -> tensor<1x12288x4x1x!qElemType1>
    %4 = IE.Dequantize(%3) {dstElemType = f16} : tensor<1x12288x4x1x!qElemType1> -> tensor<1x12288x4x1xf16>
    return %4 : tensor<1x12288x4x1xf16>

    // CHECK: [[QUANTIZE:%.+]] = IE.Quantize([[INPUT]])
    // CHECK-SAME:   {dstElemType = !qElemType} : tensor<1x12x512x8xf16> -> tensor<1x12x512x8x!qElemType>
    // CHECK-NOT:   IE.Dequantize
    // CHECK:      [[AFFINERESHAPE:%.+]] = IE.AffineReshape([[QUANTIZE]])
    // CHECK-SAME{LITERAL}:   {dim_mapping = [[0], [1], [2], [3]], shape_value = [1, 12288, 4, 1]} : tensor<1x12x512x8x!qElemType> -> tensor<1x12288x4x1x!qElemType>
    // CHECK-NOT:   IE.Quantize
    // CHECK: [[DEQUANTIZE:%.+]] = IE.Dequantize([[AFFINERESHAPE]])
    // CHECK-SAME:   {dstElemType = f16} : tensor<1x12288x4x1x!qElemType> -> tensor<1x12288x4x1xf16>
    // CHECK: return [[DEQUANTIZE]]
}
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.015704019396912818:153>
!qElemType1 = !quant.uniform<u8:f16, 0.015704022669324687:153>

// CHECK-LABEL:  func.func @FuseSliceQuantizationParamsDifferent
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x12x256x2xf16>)
module {
config.PipelineOptions @Options {
    config.Option @config.EnableAdaptiveStripping : true
}

func.func @FuseSliceQuantizationParamsDifferent(%input: tensor<1x12x256x2xf16>) -> tensor<1x12x256x1xf16> {
    %0 = IE.Quantize(%input) {dstElemType = !qElemType} : tensor<1x12x256x2xf16> -> tensor<1x12x256x2x!qElemType>
    %1 = IE.Dequantize(%0) {dstElemType = f16} : tensor<1x12x256x2x!qElemType> -> tensor<1x12x256x2xf16>
    %2 = IE.Slice %1 [0, 0, 0, 0] [1, 12, 256, 1] : tensor<1x12x256x2xf16> to tensor<1x12x256x1xf16>
    %3 = IE.Quantize(%2) {dstElemType = !qElemType1} : tensor<1x12x256x1xf16> -> tensor<1x12x256x1x!qElemType1>
    %4 = IE.Dequantize(%3) {dstElemType = f16} : tensor<1x12x256x1x!qElemType1> -> tensor<1x12x256x1xf16>
    return %4 : tensor<1x12x256x1xf16>

    // CHECK: [[QUANTIZE:%.+]] = IE.Quantize([[INPUT]])
    // CHECK-SAME:   {dstElemType = !qElemType} : tensor<1x12x256x2xf16> -> tensor<1x12x256x2x!qElemType>
    // CHECK-NOT:   IE.Dequantize
    // CHECK:      [[SLICE:%.+]] = IE.Slice [[QUANTIZE]]
    // CHECK-SAME{LITERAL}:   [0, 0, 0, 0] [1, 12, 256, 1] : tensor<1x12x256x2x!qElemType> to tensor<1x12x256x1x!qElemType>
    // CHECK-NOT:   IE.Quantize
    // CHECK: [[DEQUANTIZE:%.+]] = IE.Dequantize([[SLICE]])
    // CHECK-SAME:   {dstElemType = f16} : tensor<1x12x256x1x!qElemType> -> tensor<1x12x256x1xf16>
    // CHECK: return [[DEQUANTIZE]]
}
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.015704019396912818:153>
!qElemType1 = !quant.uniform<u8:f16, 0.015704022669324687:153>

// CHECK-LABEL:  func.func @FuseTileQuantizationParamsDifferent
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x12x256x2xf16>)
module {
config.PipelineOptions @Options {
    config.Option @config.EnableAdaptiveStripping : true
}

func.func @FuseTileQuantizationParamsDifferent(%input: tensor<1x12x256x2xf16>) -> tensor<1x12x256x4xf16> {
    %0 = IE.Quantize(%input) {dstElemType = !qElemType} : tensor<1x12x256x2xf16> -> tensor<1x12x256x2x!qElemType>
    %1 = IE.Dequantize(%0) {dstElemType = f16} : tensor<1x12x256x2x!qElemType> -> tensor<1x12x256x2xf16>
    %2 = IE.Tile(%1) {repeats_values = [1, 1, 1, 2]} : tensor<1x12x256x2xf16> -> tensor<1x12x256x4xf16>
    %3 = IE.Quantize(%2) {dstElemType = !qElemType1} : tensor<1x12x256x4xf16> -> tensor<1x12x256x4x!qElemType1>
    %4 = IE.Dequantize(%3) {dstElemType = f16} : tensor<1x12x256x4x!qElemType1> -> tensor<1x12x256x4xf16>
    return %4 : tensor<1x12x256x4xf16>

    // CHECK: [[QUANTIZE:%.+]] = IE.Quantize([[INPUT]])
    // CHECK-SAME:   {dstElemType = !qElemType} : tensor<1x12x256x2xf16> -> tensor<1x12x256x2x!qElemType>
    // CHECK-NOT:   IE.Dequantize
    // CHECK:      [[TILE:%.+]] = IE.Tile([[QUANTIZE]])
    // CHECK-SAME{LITERAL}:   {repeats_values = [1, 1, 1, 2]} : tensor<1x12x256x2x!qElemType> -> tensor<1x12x256x4x!qElemType>
    // CHECK-NOT:   IE.Quantize
    // CHECK: [[DEQUANTIZE:%.+]] = IE.Dequantize([[TILE]])
    // CHECK-SAME:   {dstElemType = f16} : tensor<1x12x256x4x!qElemType> -> tensor<1x12x256x4xf16>
    // CHECK: return [[DEQUANTIZE]]
}
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.015704019396912818:153>
!qElemType1 = !quant.uniform<u8:f16, 0.015704022669324687:153>

#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

// CHECK-LABEL:  func.func @FuseTransposeQuantizationParamsDifferent
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x512x12x8xf16>)
module {
config.PipelineOptions @Options {
    config.Option @config.EnableAdaptiveStripping : true
}

func.func @FuseTransposeQuantizationParamsDifferent(%input: tensor<1x512x12x8xf16>) -> tensor<1x12x512x8xf16> {
    %0 = IE.Quantize(%input) {dstElemType = !qElemType} : tensor<1x512x12x8xf16> -> tensor<1x512x12x8x!qElemType>
    %1 = IE.Dequantize(%0) {dstElemType = f16} : tensor<1x512x12x8x!qElemType> -> tensor<1x512x12x8xf16>
    %2 = IE.Transpose(%1) {order_value = #NHCW} : tensor<1x512x12x8xf16> -> tensor<1x12x512x8xf16>
    %3 = IE.Quantize(%2) {dstElemType = !qElemType1} : tensor<1x12x512x8xf16> -> tensor<1x12x512x8x!qElemType1>
    %4 = IE.Dequantize(%3) {dstElemType = f16} : tensor<1x12x512x8x!qElemType1> -> tensor<1x12x512x8xf16>
    return %4 : tensor<1x12x512x8xf16>

    // CHECK: [[QUANTIZE:%.+]] = IE.Quantize([[INPUT]])
    // CHECK-SAME:   {dstElemType = !qElemType} : tensor<1x512x12x8xf16> -> tensor<1x512x12x8x!qElemType>
    // CHECK-NOT:   IE.Dequantize
    // CHECK:      [[TRANSPOSE:%.+]] = IE.Transpose([[QUANTIZE]])
    // CHECK-SAME:   {order_value = #NHCW} : tensor<1x512x12x8x!qElemType> -> tensor<1x12x512x8x!qElemType>
    // CHECK-NOT:   IE.Quantize
    // CHECK: [[DEQUANTIZE:%.+]] = IE.Dequantize([[TRANSPOSE]])
    // CHECK-SAME:   {dstElemType = f16} : tensor<1x12x512x8x!qElemType> -> tensor<1x12x512x8xf16>
    // CHECK: return [[DEQUANTIZE]]
}
}
