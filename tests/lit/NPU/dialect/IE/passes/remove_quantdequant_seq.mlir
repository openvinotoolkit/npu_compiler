//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --remove-quantdequant-seq %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

!qElemType = !quant.uniform<u8<1:255>:f16:0, {0.010680671751968504:128,0.0081200787401574797:128,0.010596087598425197:128}>
!qElemType1 = !quant.uniform<u8:f16, 1.1534313725490195:128>
!qElemType2 = !quant.uniform<u8:f16, 2.4627450980392158>

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @RemoveQuantDequantSequence
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x3x16x16xf16>)
func.func @RemoveQuantDequantSequence(%arg0: tensor<1x3x16x16xf16>) -> (tensor<1x3x14x14xf16>, tensor<1x3x14x14xf16>)  {
  %1 = IE.Quantize(%arg0) {dstElemType = !qElemType1} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!qElemType1>
  %2 = IE.Dequantize(%1) {dstElemType = f16} : tensor<1x3x16x16x!qElemType1> -> tensor<1x3x16x16xf16>
  %weights = const.Declare tensor<3x3x3x3x!qElemType> = dense<1.0> : tensor<3x3x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]
  %3 = IE.Dequantize(%weights) {dstElemType = f16} : tensor<3x3x3x3x!qElemType> -> tensor<3x3x3x3xf16>
  %4 = IE.Convolution(%2, %3) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x16x16xf16>, tensor<3x3x3x3xf16> -> tensor<1x3x14x14xf16>
  %5 = IE.Quantize(%4) {dstElemType = !qElemType2}: tensor<1x3x14x14xf16> -> tensor<1x3x14x14x!qElemType2>
  %6 = IE.Dequantize(%5) {dstElemType = f16} : tensor<1x3x14x14x!qElemType2> -> tensor<1x3x14x14xf16>

  return %6, %4 : tensor<1x3x14x14xf16>, tensor<1x3x14x14xf16>

  // CHECK: [[VAL1:%.+]] = IE.Quantize([[ARG_0]]) {dstElemType = !qElemType} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!qElemType>
  // CHECK-DAG: [[WEIGHTS:%.+]] = const.Declare tensor<3x3x3x3x!qElemType1> =
  // CHECK-SAME:                 dense<1.000000e+00> : tensor<3x3x3x3xf16>,
  // CHECK-SAME:                 [#const.CastElemType<ui8>, #const.CastElemType<!qElemType1>]
  // CHECK: [[VAL2:%.+]] = IE.Dequantize([[WEIGHTS]]) {dstElemType = f16} : tensor<3x3x3x3x!qElemType1> -> tensor<3x3x3x3xf16>

  // CHECK: [[VAL3:%.+]] = IE.Convolution([[ARG_0]], [[VAL2]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x16x16xf16>, tensor<3x3x3x3xf16> -> tensor<1x3x14x14xf16>
  // CHECK: return [[VAL3]], [[VAL3]]
}

// -----

!qElemType = !quant.uniform<u8:f16, 1.1534313725490195:128>

// CHECK-LABEL: @RemoveQuantReshapeDequantSequence
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x4420x1x2xf16>, [[ARG_1:%[^:]+]]: tensor<1x4420x1x2xf16>)
func.func @RemoveQuantReshapeDequantSequence(%arg0: tensor<1x4420x1x2xf16>, %arg1: tensor<1x4420x1x2xf16>) -> tensor<1x4420x1x2xf16> {
  %1 = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x4420x1x2xf16> -> tensor<1x4420x1x2x!qElemType>
  %2 = IE.AffineReshape(%1) { dim_mapping = [[0, 1], [2], [2], [3]], shape_value = [1, 1, 4420, 2] } : tensor<1x4420x1x2x!qElemType> -> tensor<1x1x4420x2x!qElemType>
  %3 = IE.AffineReshape(%2) { dim_mapping = [[0], [0], [1, 2], [3]], shape_value = [1, 4420, 1, 2] } : tensor<1x1x4420x2x!qElemType> -> tensor<1x4420x1x2x!qElemType>
  %4 = IE.Dequantize(%3) {dstElemType = f16} : tensor<1x4420x1x2x!qElemType> -> tensor<1x4420x1x2xf16>
  %5 = IE.Add(%4, %arg1)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
        tensor<1x4420x1x2xf16>, tensor<1x4420x1x2xf16> -> tensor<1x4420x1x2xf16>
  return %5 : tensor<1x4420x1x2xf16>

  // CHECK: [[VAL0:%.+]] = IE.AffineReshape([[ARG_0]])
  // CHECK-SAME{LITERAL}: {dim_mapping = [[0, 1], [2], [2], [3]], shape_value = [1, 1, 4420, 2]} : tensor<1x4420x1x2xf16> -> tensor<1x1x4420x2xf16>
  // CHECK: [[VAL1:%.+]] = IE.AffineReshape([[VAL0]])
  // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [0], [1, 2], [3]], shape_value = [1, 4420, 1, 2]} : tensor<1x1x4420x2xf16> -> tensor<1x4420x1x2xf16>
  // CHECK: [[VAL2:%.+]] = IE.Add([[VAL1]], [[ARG_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4420x1x2xf16>, tensor<1x4420x1x2xf16> -> tensor<1x4420x1x2xf16>
  // CHECK: return [[VAL2]]
}

// -----

!qElemType = !quant.uniform<u8:f16, 1.1534313725490195:128>

// CHECK-LABEL: @RemoveQuantReshapeDequantMaxPool
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x64x40x112x112xf16>)
func.func @RemoveQuantReshapeDequantMaxPool(%arg0: tensor<1x64x40x112x112xf16>) -> tensor<1x64x40x112x112xf16> {
  %0 = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x64x40x112x112xf16> -> tensor<1x64x40x112x112x!qElemType>
  %1 = IE.MaxPool(%0) {
        kernel_size = [3, 3, 3],
        pads_begin = [1, 1, 1],
        pads_end = [1, 1, 1],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 1, 1]} : tensor<1x64x40x112x112x!qElemType> -> tensor<1x64x40x112x112x!qElemType>
  %2 = IE.Dequantize(%1) {dstElemType = f16} : tensor<1x64x40x112x112x!qElemType> -> tensor<1x64x40x112x112xf16>
  return %2 : tensor<1x64x40x112x112xf16>

    // CHECK: [[MAXPOOL:%.+]] = IE.MaxPool([[ARG_0]]) {kernel_size = [3, 3, 3], pads_begin = [1, 1, 1], pads_end = [1, 1, 1], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1, 1]} : tensor<1x64x40x112x112xf16> -> tensor<1x64x40x112x112xf16>
    // CHECK: return [[MAXPOOL]] : tensor<1x64x40x112x112xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0039215686274509803>

// // CHECK-LABEL: @RemoveQuantConcatDequantSeqNoElemTypeInfoOpInterface
// // CHECK-SAME: ([[INPUT0:%.+]]: tensor<1x12800x2x1xf16>, [[INPUT1:%.+]]: tensor<1x3200x2x1xf16>, [[INPUT2:%.+]]: tensor<1x800x2x1xf16>)
func.func @RemoveQuantConcatDequantSeqNoElemTypeInfoOpInterface(%arg0: tensor<1x12800x2x1xf16>, %arg1: tensor<1x3200x2x1xf16>, %arg2: tensor<1x800x2x1xf16>) -> tensor<1x16800x2x1xf16> {
  %0 = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x12800x2x1xf16> -> tensor<1x12800x2x1x!qElemType>
  %1 = IE.Quantize(%arg1) {dstElemType = !qElemType} : tensor<1x3200x2x1xf16> -> tensor<1x3200x2x1x!qElemType>
  %2 = IE.Quantize(%arg2) {dstElemType = !qElemType} : tensor<1x800x2x1xf16> -> tensor<1x800x2x1x!qElemType>
  %3 = IE.Concat(%0, %1, %2) {static_offsets = [[0, 0, 0, 0], [0, 12800, 0, 0], [0, 16000, 0, 0]]} : tensor<1x12800x2x1x!qElemType>, tensor<1x3200x2x1x!qElemType>, tensor<1x800x2x1x!qElemType> -> tensor<1x16800x2x1x!qElemType>
  %4 = IE.Dequantize(%3) {dstElemType = f16} : tensor<1x16800x2x1x!qElemType> -> tensor<1x16800x2x1xf16>

  return %4 : tensor<1x16800x2x1xf16>

  // CHECK: [[VAL0:%.+]] = IE.Concat([[INPUT0]], [[INPUT1]], [[INPUT2]])
  // CHECK-SAME{LITERAL}: {static_offsets = [[0, 0, 0, 0], [0, 12800, 0, 0], [0, 16000, 0, 0]]} : tensor<1x12800x2x1xf16>, tensor<1x3200x2x1xf16>, tensor<1x800x2x1xf16> -> tensor<1x16800x2x1xf16>
  // CHECK: return [[VAL0]]
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0039215686274509803>

// CHECK-LABEL: @RemoveQuantConcatDequantSeq
// CHECK-SAME: ([[INPUT0:%.+]]: tensor<1x12800x2x1xf16>, [[INPUT1:%.+]]: tensor<1x3200x2x1xf16>, [[INPUT2:%.+]]: tensor<1x800x2x1xf16>)
func.func @RemoveQuantConcatDequantSeq(%arg0: tensor<1x12800x2x1xf16>, %arg1: tensor<1x3200x2x1xf16>, %arg2: tensor<1x800x2x1xf16>) -> tensor<1x16800x2x1xf16> {
  %0 = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x12800x2x1xf16> -> tensor<1x12800x2x1x!qElemType>
  %1 = IE.AffineReshape(%0) {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 1, 12800, 2]} : tensor<1x12800x2x1x!qElemType> -> tensor<1x1x12800x2x!qElemType>
  %2 = IE.Quantize(%arg1) {dstElemType = !qElemType} : tensor<1x3200x2x1xf16> -> tensor<1x3200x2x1x!qElemType>
  %3 = IE.AffineReshape(%2) {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 1, 3200, 2]} : tensor<1x3200x2x1x!qElemType> -> tensor<1x1x3200x2x!qElemType>
  %4 = IE.Quantize(%arg2) {dstElemType = !qElemType} : tensor<1x800x2x1xf16> -> tensor<1x800x2x1x!qElemType>
  %5 = IE.AffineReshape(%4) {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 1, 800, 2]} : tensor<1x800x2x1x!qElemType> -> tensor<1x1x800x2x!qElemType>
  %6 = IE.Concat(%1, %3, %5) {static_offsets = [[0, 0, 0, 0], [0, 0, 12800, 0], [0, 0, 16000, 0]]} : tensor<1x1x12800x2x!qElemType>, tensor<1x1x3200x2x!qElemType>, tensor<1x1x800x2x!qElemType> -> tensor<1x1x16800x2x!qElemType>
  %7 = IE.AffineReshape(%6) {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 16800, 2, 1]} : tensor<1x1x16800x2x!qElemType> -> tensor<1x16800x2x1x!qElemType>
  %8 = IE.Dequantize(%7) {dstElemType = f16} : tensor<1x16800x2x1x!qElemType> -> tensor<1x16800x2x1xf16>
  return %8 : tensor<1x16800x2x1xf16>

  // CHECK: [[VAL0:%.+]] = IE.AffineReshape([[INPUT0]])
  // CHECK-SAME{LITERAL}: {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 1, 12800, 2]} : tensor<1x12800x2x1xf16> -> tensor<1x1x12800x2xf16>
  // CHECK: [[VAL1:%.+]] = IE.AffineReshape([[INPUT1]])
  // CHECK-SAME{LITERAL}: {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 1, 3200, 2]} : tensor<1x3200x2x1xf16> -> tensor<1x1x3200x2xf16>
  // CHECK: [[VAL2:%.+]] = IE.AffineReshape([[INPUT2]])
  // CHECK-SAME{LITERAL}: {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 1, 800, 2]} : tensor<1x800x2x1xf16> -> tensor<1x1x800x2xf16>
  // CHECK: [[VAL3:%.+]] = IE.Concat([[VAL0]], [[VAL1]], [[VAL2]])
  // CHECK-SAME{LITERAL}: {static_offsets = [[0, 0, 0, 0], [0, 0, 12800, 0], [0, 0, 16000, 0]]} : tensor<1x1x12800x2xf16>, tensor<1x1x3200x2xf16>, tensor<1x1x800x2xf16> -> tensor<1x1x16800x2xf16>
  // CHECK: [[VAL4:%.+]] = IE.AffineReshape([[VAL3]])
  // CHECK-SAME{LITERAL}: {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 16800, 2, 1]} : tensor<1x1x16800x2xf16> -> tensor<1x16800x2x1xf16>
  // CHECK: return [[VAL4]]
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0039215686274509803>

// CHECK-LABEL: @DontRemoveQuantConcatElemTypeConcatDequantSeq
// CHECK-SAME: ([[INPUT0:%.+]]: tensor<1x12800x2x1xf16>, [[INPUT1:%.+]]: tensor<1x3200x2x1xf16>, [[INPUT2:%.+]]: tensor<1x800x2x1xf16>)
func.func @DontRemoveQuantConcatElemTypeConcatDequantSeq(%arg0: tensor<1x12800x2x1xf16>, %arg1: tensor<1x3200x2x1xf16>, %arg2: tensor<1x800x2x1xf16>) -> tensor<1x1x17600x2xf16> {
  %0 = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x12800x2x1xf16> -> tensor<1x12800x2x1x!qElemType>
  %1 = IE.AffineReshape(%0) {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 1, 12800, 2]} : tensor<1x12800x2x1x!qElemType> -> tensor<1x1x12800x2x!qElemType>
  %2 = IE.Quantize(%arg1) {dstElemType = !qElemType} : tensor<1x3200x2x1xf16> -> tensor<1x3200x2x1x!qElemType>
  %3 = IE.AffineReshape(%2) {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 1, 3200, 2]} : tensor<1x3200x2x1x!qElemType> -> tensor<1x1x3200x2x!qElemType>
  %4 = IE.Quantize(%arg2) {dstElemType = !qElemType} : tensor<1x800x2x1xf16> -> tensor<1x800x2x1x!qElemType>
  %5 = IE.AffineReshape(%4) {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 1, 800, 2]} : tensor<1x800x2x1x!qElemType> -> tensor<1x1x800x2x!qElemType>
  %6 = IE.Concat(%1, %3, %5) {static_offsets = [[0, 0, 0, 0], [0, 0, 12800, 0], [0, 0, 16000, 0]]} : tensor<1x1x12800x2x!qElemType>, tensor<1x1x3200x2x!qElemType>, tensor<1x1x800x2x!qElemType> -> tensor<1x1x16800x2x!qElemType>
  %7 = IE.AffineReshape(%6) {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 16800, 2, 1]} : tensor<1x1x16800x2x!qElemType> -> tensor<1x16800x2x1x!qElemType>
  %8 = IE.AffineReshape(%7) {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 1, 16800, 2]} : tensor<1x16800x2x1x!qElemType> -> tensor<1x1x16800x2x!qElemType>
  %9 = IE.Concat(%5, %8) {static_offsets = [[0, 0, 0, 0], [0, 0, 800, 0]]} : tensor<1x1x800x2x!qElemType>, tensor<1x1x16800x2x!qElemType> -> tensor<1x1x17600x2x!qElemType>
  %10 = IE.Dequantize(%9) {dstElemType = f16} : tensor<1x1x17600x2x!qElemType> -> tensor<1x1x17600x2xf16>
  return %10 : tensor<1x1x17600x2xf16>

  // CHECK: [[VAL0:%.+]] = IE.Quantize([[INPUT0]])
  // CHECK-SAME{LITERAL}: {dstElemType = !qElemType} : tensor<1x12800x2x1xf16> -> tensor<1x12800x2x1x!qElemType>
  // CHECK: [[VAL1:%.+]] = IE.AffineReshape([[VAL0]])
  // CHECK-SAME{LITERAL}: {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 1, 12800, 2]} : tensor<1x12800x2x1x!qElemType> -> tensor<1x1x12800x2x!qElemType>
  // CHECK: [[VAL2:%.+]] = IE.Quantize([[INPUT1]])
  // CHECK-SAME{LITERAL}: {dstElemType = !qElemType} : tensor<1x3200x2x1xf16> -> tensor<1x3200x2x1x!qElemType>
  // CHECK: [[VAL3:%.+]] = IE.AffineReshape([[VAL2]])
  // CHECK-SAME{LITERAL}: {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 1, 3200, 2]} : tensor<1x3200x2x1x!qElemType> -> tensor<1x1x3200x2x!qElemType>
  // CHECK: [[VAL4:%.+]] = IE.Quantize([[INPUT2]])
  // CHECK-SAME{LITERAL}: {dstElemType = !qElemType} : tensor<1x800x2x1xf16> -> tensor<1x800x2x1x!qElemType>
  // CHECK: [[VAL5:%.+]] = IE.AffineReshape([[VAL4]])
  // CHECK-SAME{LITERAL}: {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 1, 800, 2]} : tensor<1x800x2x1x!qElemType> -> tensor<1x1x800x2x!qElemType>
  // CHECK: [[VAL6:%.+]] = IE.Concat([[VAL1]], [[VAL3]], [[VAL5]])
  // CHECK-SAME{LITERAL}: {static_offsets = [[0, 0, 0, 0], [0, 0, 12800, 0], [0, 0, 16000, 0]]} : tensor<1x1x12800x2x!qElemType>, tensor<1x1x3200x2x!qElemType>, tensor<1x1x800x2x!qElemType> -> tensor<1x1x16800x2x!qElemType>
  // CHECK: [[VAL7:%.+]] = IE.AffineReshape([[VAL6]])
  // CHECK-SAME{LITERAL}: {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 16800, 2, 1]} : tensor<1x1x16800x2x!qElemType> -> tensor<1x16800x2x1x!qElemType>
  // CHECK: [[VAL8:%.+]] = IE.AffineReshape([[VAL7]])
  // CHECK-SAME{LITERAL}: {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 1, 16800, 2]} : tensor<1x16800x2x1x!qElemType> -> tensor<1x1x16800x2x!qElemType>
  // CHECK: [[VAL9:%.+]] = IE.Concat([[VAL5]], [[VAL8]])
  // CHECK-SAME{LITERAL}: {static_offsets = [[0, 0, 0, 0], [0, 0, 800, 0]]} : tensor<1x1x800x2x!qElemType>, tensor<1x1x16800x2x!qElemType> -> tensor<1x1x17600x2x!qElemType>
  // CHECK: [[VAL10:%.+]] = IE.Dequantize([[VAL9]])
  // CHECK-SAME{LITERAL}: {dstElemType = f16} : tensor<1x1x17600x2x!qElemType> -> tensor<1x1x17600x2xf16>
  // CHECK: return [[VAL10]]
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0039215686274509803>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @RemoveQuantConcatMultipleElemTypeDequantSeq
// CHECK-SAME: ([[INPUT0:%.+]]: tensor<1x12800x2x1xf16>, [[INPUT1:%.+]]: tensor<1x3200x2x1xf16>, [[INPUT2:%.+]]: tensor<1x800x2x1xf16>)
func.func @RemoveQuantConcatMultipleElemTypeDequantSeq(%arg0: tensor<1x12800x2x1xf16>, %arg1: tensor<1x3200x2x1xf16>, %arg2: tensor<1x800x2x1xf16>) -> tensor<1x16800x2x1xf16> {
  %0 = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x12800x2x1xf16> -> tensor<1x12800x2x1x!qElemType>
  %1 = IE.AffineReshape(%0) {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 1, 12800, 2]} : tensor<1x12800x2x1x!qElemType> -> tensor<1x1x12800x2x!qElemType>
  %2 = IE.Reorder(%1) {dstOrder = #NCHW} : tensor<1x1x12800x2x!qElemType> -> tensor<1x1x12800x2x!qElemType>
  %3 = IE.Quantize(%arg1) {dstElemType = !qElemType} : tensor<1x3200x2x1xf16> -> tensor<1x3200x2x1x!qElemType>
  %4 = IE.AffineReshape(%3) {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 1, 3200, 2]} : tensor<1x3200x2x1x!qElemType> -> tensor<1x1x3200x2x!qElemType>
  %5 = IE.Reorder(%4) {dstOrder = #NCHW} : tensor<1x1x3200x2x!qElemType> -> tensor<1x1x3200x2x!qElemType>
  %6 = IE.Quantize(%arg2) {dstElemType = !qElemType} : tensor<1x800x2x1xf16> -> tensor<1x800x2x1x!qElemType>
  %7 = IE.AffineReshape(%6) {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 1, 800, 2]} : tensor<1x800x2x1x!qElemType> -> tensor<1x1x800x2x!qElemType>
  %8 = IE.Reorder(%7) {dstOrder = #NCHW} : tensor<1x1x800x2x!qElemType> -> tensor<1x1x800x2x!qElemType>
  %9 = IE.Concat(%2, %5, %8) {static_offsets = [[0, 0, 0, 0], [0, 0, 12800, 0], [0, 0, 16000, 0]]} : tensor<1x1x12800x2x!qElemType>, tensor<1x1x3200x2x!qElemType>, tensor<1x1x800x2x!qElemType> -> tensor<1x1x16800x2x!qElemType>
  %10 = IE.AffineReshape(%9) {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 16800, 2, 1]} : tensor<1x1x16800x2x!qElemType> -> tensor<1x16800x2x1x!qElemType>
  %11 = IE.Dequantize(%10) {dstElemType = f16} : tensor<1x16800x2x1x!qElemType> -> tensor<1x16800x2x1xf16>
  return %11 : tensor<1x16800x2x1xf16>

  // CHECK: [[VAL0:%.+]] = IE.AffineReshape([[INPUT0]])
  // CHECK-SAME{LITERAL}: {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 1, 12800, 2]} : tensor<1x12800x2x1xf16> -> tensor<1x1x12800x2xf16>
  // CHECK: [[VAL1:%.+]] = IE.Reorder([[VAL0]])
  // CHECK-SAME{LITERAL}: {dstOrder = #NCHW} : tensor<1x1x12800x2xf16> -> tensor<1x1x12800x2xf16>
  // CHECK: [[VAL2:%.+]] = IE.AffineReshape([[INPUT1]])
  // CHECK-SAME{LITERAL}: {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 1, 3200, 2]} : tensor<1x3200x2x1xf16> -> tensor<1x1x3200x2xf16>
  // CHECK: [[VAL3:%.+]] = IE.Reorder([[VAL2]])
  // CHECK-SAME{LITERAL}: {dstOrder = #NCHW} : tensor<1x1x3200x2xf16> -> tensor<1x1x3200x2xf16>
  // CHECK: [[VAL4:%.+]] = IE.AffineReshape([[INPUT2]])
  // CHECK-SAME{LITERAL}: {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 1, 800, 2]} : tensor<1x800x2x1xf16> -> tensor<1x1x800x2xf16>
  // CHECK: [[VAL5:%.+]] = IE.Reorder([[VAL4]])
  // CHECK-SAME{LITERAL}: {dstOrder = #NCHW} : tensor<1x1x800x2xf16> -> tensor<1x1x800x2xf16>
  // CHECK: [[VAL6:%.+]] = IE.Concat([[VAL1]], [[VAL3]], [[VAL5]])
  // CHECK-SAME{LITERAL}: {static_offsets = [[0, 0, 0, 0], [0, 0, 12800, 0], [0, 0, 16000, 0]]} : tensor<1x1x12800x2xf16>, tensor<1x1x3200x2xf16>, tensor<1x1x800x2xf16> -> tensor<1x1x16800x2xf16>
  // CHECK: [[VAL7:%.+]] = IE.AffineReshape([[VAL6]])
  // CHECK-SAME{LITERAL}: {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 16800, 2, 1]} : tensor<1x1x16800x2xf16> -> tensor<1x16800x2x1xf16>
  // CHECK: return [[VAL7]]
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0038988489730685367:128>

// CHECK-LABEL: @Remove_Quant_Input_SideTap_Concat_Dequant_Seq
func.func @Remove_Quant_Input_SideTap_Concat_Dequant_Seq(%arg0: tensor<1x1x40x64xf16>, %arg1: tensor<1x100x40x64xf16>) -> (tensor<1x101x40x64xf16>, tensor<1x1x40x64xf16>) {
    %0 = IE.Quantize(%arg1) {dstElemType = !qElemType} : tensor<1x100x40x64xf16> -> tensor<1x100x40x64x!qElemType>
    %1 = IE.Clamp(%0) {max = 0.49297687411308289 : f64, min = -0.49685856699943542 : f64} : tensor<1x100x40x64x!qElemType> -> tensor<1x100x40x64x!qElemType>
    %2 = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x1x40x64xf16> -> tensor<1x1x40x64x!qElemType>
    %3 = IE.Clamp(%2) {max = 0.4860345721244812 : f64, min = -0.48986160755157471 : f64} : tensor<1x1x40x64x!qElemType> -> tensor<1x1x40x64x!qElemType>
    %sidetap = IE.Dequantize(%3) {dstElemType = f16} : tensor<1x1x40x64x!qElemType> -> tensor<1x1x40x64xf16>
    %5 = IE.Concat(%1, %3) {static_offsets = [[0, 0, 0, 0], [0, 100, 0, 0]]} : tensor<1x100x40x64x!qElemType>, tensor<1x1x40x64x!qElemType> -> tensor<1x101x40x64x!qElemType>
    %6 = IE.Dequantize(%5) {dstElemType = f16} : tensor<1x101x40x64x!qElemType> -> tensor<1x101x40x64xf16>
    return %6, %sidetap : tensor<1x101x40x64xf16>, tensor<1x1x40x64xf16>
    // CHECK-NOT:  IE.Quantize
    // CHECK-NOT:  IE.Dequantize
    // CHECK: return
}

// -----

!qElemType = !quant.uniform<u8:f16, 1.9664713541666667:255>

// CHECK-LABEL:  @Handle_Multiple_Users_Quant_Dequant_Seq
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<4x64x199x63xf16>)
func.func @Handle_Multiple_Users_Quant_Dequant_Seq(%arg0: tensor<4x64x199x63xf16>) -> (tensor<1x64x199x63xf16>, tensor<1x64x25x9xf16>) {
  %0 = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<4x64x199x63xf16> -> tensor<4x64x199x63x!qElemType>
  %1 = IE.Slice %0 [2, 0, 0, 0] [1, 64, 199, 63] : tensor<4x64x199x63x!qElemType> to tensor<1x64x199x63x!qElemType>
  %2 = IE.Dequantize(%1) {dstElemType = f16} : tensor<1x64x199x63x!qElemType> -> tensor<1x64x199x63xf16>
  %3 = IE.Slice %1 [0, 0, 0, 0] [1, 64, 1, 63] : tensor<1x64x199x63x!qElemType> to tensor<1x64x1x63x!qElemType>
  %4 = IE.Concat(%1, %3) {static_offsets = [[0, 0, 0, 0], [0, 0, 199, 0]]} : tensor<1x64x199x63x!qElemType>, tensor<1x64x1x63x!qElemType> -> tensor<1x64x200x63x!qElemType>
  %5 = IE.MaxPool(%4) {kernel_size = [8, 7], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [8, 7]} : tensor<1x64x200x63x!qElemType> -> tensor<1x64x25x9x!qElemType>
  %6 = IE.Dequantize(%5) {dstElemType = f16} : tensor<1x64x25x9x!qElemType> -> tensor<1x64x25x9xf16>

  return %2, %6 : tensor<1x64x199x63xf16>, tensor<1x64x25x9xf16>

  // CHECK: [[VAL0:%.+]] = IE.Slice [[ARG_0]] [2, 0, 0, 0] [1, 64, 199, 63] : tensor<4x64x199x63xf16> to tensor<1x64x199x63xf16>
  // CHECK: [[VAL1:%.+]] = IE.Slice [[VAL0]] [0, 0, 0, 0] [1, 64, 1, 63] : tensor<1x64x199x63xf16> to tensor<1x64x1x63xf16>
  // CHECK: [[VAL2:%.+]] = IE.Concat([[VAL0]], [[VAL1]])
  // CHECK-SAME{LITERAL}: {static_offsets = [[0, 0, 0, 0], [0, 0, 199, 0]]} : tensor<1x64x199x63xf16>, tensor<1x64x1x63xf16> -> tensor<1x64x200x63xf16>
  // CHECK: [[VAL3:%.+]] = IE.MaxPool([[VAL2]]) {kernel_size = [8, 7], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [8, 7]} : tensor<1x64x200x63xf16> -> tensor<1x64x25x9xf16>
  // CHECK: return [[VAL0]], [[VAL3]] : tensor<1x64x199x63xf16>, tensor<1x64x25x9xf16>
}

// -----

// CHECK-LABEL: @Ignore_Pattern_FuncOpArg_ElemTypeInfoOp_ConcatOp
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x1x1440x320xf16>, [[ARG_1:%[^:]+]]: tensor<1x2x1x1xf16>)
func.func @Ignore_Pattern_FuncOpArg_ElemTypeInfoOp_ConcatOp(%arg0: tensor<1x1x1440x320xf16>, %arg1: tensor<1x2x1x1xf16>) -> tensor<1x2x1440x160xf16> {
  %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 1, 1440, 160] : tensor<1x1x1440x320xf16> to tensor<1x1x1440x160xf16>
  %1 = IE.Slice %arg0 [0, 0, 0, 160] [1, 1, 1440, 160] : tensor<1x1x1440x320xf16> to tensor<1x1x1440x160xf16>
  %2 = IE.Concat(%0, %1) {static_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]]} : tensor<1x1x1440x160xf16>, tensor<1x1x1440x160xf16> -> tensor<1x2x1440x160xf16>
  return %2 : tensor<1x2x1440x160xf16>

  // CHECK: [[VAL0:%.+]] = IE.Slice [[ARG_0]]
  // CHECK: [[VAL1:%.+]] = IE.Slice [[ARG_0]]
  // CHECK: [[VAL2:%.+]] = IE.Concat([[VAL0]], [[VAL1]])
  // CHECK: return [[VAL2]]
}

// -----

!qElemType = !quant.uniform<u8:f16, 1.1534313725490195:128>

// CHECK-LABEL: @RemoveQuantDequantSeqWithSameInputConcat
func.func @RemoveQuantDequantSeqWithSameInputConcat(%arg0: tensor<1x16x32x32xf16>) -> tensor<1x32x32x32xf16> {
  %0 = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x16x32x32xf16> -> tensor<1x16x32x32x!qElemType>
  %1 = IE.Concat(%0, %0) {static_offsets = [[0, 0, 0, 0], [0, 16, 0, 0]]} : tensor<1x16x32x32x!qElemType>, tensor<1x16x32x32x!qElemType> -> tensor<1x32x32x32x!qElemType>
  %2 = IE.Dequantize(%1) {dstElemType = f16} : tensor<1x32x32x32x!qElemType> -> tensor<1x32x32x32xf16>
  return %2 : tensor<1x32x32x32xf16>

  // CHECK: [[VAL0:%.+]] = IE.Concat(%arg0, %arg0)
  // CHECK-SAME{LITERAL}: {static_offsets = [[0, 0, 0, 0], [0, 16, 0, 0]]} : tensor<1x16x32x32xf16>, tensor<1x16x32x32xf16> -> tensor<1x32x32x32xf16>
  // CHECK: return [[VAL0]]
}

// -----

!qElemType = !quant.uniform<u8:f16, 1.1534313725490195:128>

// CHECK-LABEL: @DontRemove_QuantInterpolateDequant_SESupported
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x16x8x8xf16>)
func.func @DontRemove_QuantInterpolateDequant_SESupported(%arg0: tensor<1x16x8x8xf16>) -> tensor<1x16x16x16xf16> {
  %0 = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x16x8x8xf16> -> tensor<1x16x8x8x!qElemType>
  %1 = IE.Interpolate(%0) {
        attr = #IE.Interpolate<antialias = false, coord_mode = <ASYMMETRIC>, cube_coeff = -7.500000e-01 : f64,
                               mode = <NEAREST>, nearest_mode = <FLOOR>,
                               pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0],
                               shape_calc_mode = <SCALES>>,
        axes_attr = [2, 3],
        operandSegmentSizes = array<i32: 1, 0, 0, 0>,
        scales_attr = [2.000000e+00, 2.000000e+00],
        sizes_attr = [16, 16]
      } : tensor<1x16x8x8x!qElemType> -> tensor<1x16x16x16x!qElemType>
  %2 = IE.Dequantize(%1) {dstElemType = f16} : tensor<1x16x16x16x!qElemType> -> tensor<1x16x16x16xf16>
  return %2 : tensor<1x16x16x16xf16>

  // CHECK: [[QUANT:%.+]] = IE.Quantize([[ARG_0]]) {dstElemType = !qElemType} : tensor<1x16x8x8xf16> -> tensor<1x16x8x8x!qElemType>
  // CHECK: [[INTERP:%.+]] = IE.Interpolate([[QUANT]])
  // CHECK-SAME: tensor<1x16x8x8x!qElemType> -> tensor<1x16x16x16x!qElemType>
  // CHECK: [[DEQUANT:%.+]] = IE.Dequantize([[INTERP]]) {dstElemType = f16} : tensor<1x16x16x16x!qElemType> -> tensor<1x16x16x16xf16>
  // CHECK: return [[DEQUANT]]
}

// -----

!qElemType = !quant.uniform<u8:f16, 1.1534313725490195:128>

// CHECK-LABEL: @Remove_QuantInterpolateDequant_SENotSupported
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x16x8x8xf16>)
func.func @Remove_QuantInterpolateDequant_SENotSupported(%arg0: tensor<1x16x8x8xf16>) -> tensor<1x16x16x16xf16> {
  %0 = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x16x8x8xf16> -> tensor<1x16x8x8x!qElemType>
  %1 = IE.Interpolate(%0) {
        attr = #IE.Interpolate<antialias = true, coord_mode = <ASYMMETRIC>, cube_coeff = -7.500000e-01 : f64,
                               mode = <LINEAR_ONNX>, nearest_mode = <FLOOR>,
                               pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0],
                               shape_calc_mode = <SCALES>>,
        axes_attr = [2, 3],
        operandSegmentSizes = array<i32: 1, 0, 0, 0>,
        scales_attr = [2.000000e+00, 2.000000e+00],
        sizes_attr = [16, 16]
      } : tensor<1x16x8x8x!qElemType> -> tensor<1x16x16x16x!qElemType>
  %2 = IE.Dequantize(%1) {dstElemType = f16} : tensor<1x16x16x16x!qElemType> -> tensor<1x16x16x16xf16>
  return %2 : tensor<1x16x16x16xf16>

  // CHECK-NOT: IE.Quantize
  // CHECK-NOT: IE.Dequantize
  // CHECK: [[INTERP:%.+]] = IE.Interpolate([[ARG_0]])
  // CHECK-SAME: tensor<1x16x8x8xf16> -> tensor<1x16x16x16xf16>
  // CHECK: return [[INTERP]]
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.15016976150811887:128>

// CHECK-LABEL: @SkipConcatPatternWithMultiResultConsumer
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x256x1x1xf16>, [[ARG_1:%[^:]+]]: tensor<1x256x1x1xf16>)
func.func @SkipConcatPatternWithMultiResultConsumer(%arg0: tensor<1x256x1x1xf16>, %arg1: tensor<1x256x1x1xf16>) -> (tensor<1x256x1x1xf16>, tensor<1x256x1x1x!qElemType>) {
  %0 = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x256x1x1xf16> -> tensor<1x256x1x1x!qElemType>
  %1 = IE.Quantize(%arg1) {dstElemType = !qElemType} : tensor<1x256x1x1xf16> -> tensor<1x256x1x1x!qElemType>
  %2 = IE.Concat(%0, %1) {static_offsets = [[0, 0, 0, 0], [0, 256, 0, 0]]} : tensor<1x256x1x1x!qElemType>, tensor<1x256x1x1x!qElemType> -> tensor<1x512x1x1x!qElemType>
  %3 = IE.AffineReshape(%2) {dim_mapping = [[0], [1], [1], [1]], shape_value = [1, 512]} : tensor<1x512x1x1x!qElemType> -> tensor<1x512x!qElemType>
  %4:2 = IE.Split(%3) {axis_value = 1 : i64, num_splits = 2 : i64} : tensor<1x512x!qElemType> -> tensor<1x256x!qElemType>, tensor<1x256x!qElemType>
  %5 = IE.AffineReshape(%4#0) {dim_mapping = [[0], [1, 2, 3]], shape_value = [1, 256, 1, 1]} : tensor<1x256x!qElemType> -> tensor<1x256x1x1x!qElemType>
  %6 = IE.Dequantize(%5) {dstElemType = f16} : tensor<1x256x1x1x!qElemType> -> tensor<1x256x1x1xf16>
  %7 = IE.AffineReshape(%4#1) {dim_mapping = [[0], [1, 2, 3]], shape_value = [1, 256, 1, 1]} : tensor<1x256x!qElemType> -> tensor<1x256x1x1x!qElemType>
  return %6, %7 : tensor<1x256x1x1xf16>, tensor<1x256x1x1x!qElemType>

  // CHECK: [[VAL0:%.+]] = IE.Quantize([[ARG_0]]) {dstElemType = !qElemType} : tensor<1x256x1x1xf16> -> tensor<1x256x1x1x!qElemType>
  // CHECK: [[VAL1:%.+]] = IE.Quantize([[ARG_1]]) {dstElemType = !qElemType} : tensor<1x256x1x1xf16> -> tensor<1x256x1x1x!qElemType>
  // CHECK: [[VAL2:%.+]] = IE.Concat([[VAL0]], [[VAL1]])
  // CHECK-SAME{LITERAL}: {static_offsets = [[0, 0, 0, 0], [0, 256, 0, 0]]} : tensor<1x256x1x1x!qElemType>, tensor<1x256x1x1x!qElemType> -> tensor<1x512x1x1x!qElemType>
  // CHECK: [[VAL3:%.+]] = IE.AffineReshape([[VAL2]])
  // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1], [1], [1]], shape_value = [1, 512]} : tensor<1x512x1x1x!qElemType> -> tensor<1x512x!qElemType>
  // CHECK: [[VAL4:%.+]]:2 = IE.Split([[VAL3]]) {axis_value = 1 : i64, num_splits = 2 : i64} : tensor<1x512x!qElemType> -> tensor<1x256x!qElemType>, tensor<1x256x!qElemType>
  // CHECK: [[VAL5:%.+]] = IE.AffineReshape([[VAL4]]#0)
  // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1, 2, 3]], shape_value = [1, 256, 1, 1]} : tensor<1x256x!qElemType> -> tensor<1x256x1x1x!qElemType>
  // CHECK: [[VAL6:%.+]] = IE.Dequantize([[VAL5]]) {dstElemType = f16} : tensor<1x256x1x1x!qElemType> -> tensor<1x256x1x1xf16>
  // CHECK: [[VAL7:%.+]] = IE.AffineReshape([[VAL4]]#1)
  // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1, 2, 3]], shape_value = [1, 256, 1, 1]} : tensor<1x256x!qElemType> -> tensor<1x256x1x1x!qElemType>
  // CHECK: return [[VAL6]], [[VAL7]] : tensor<1x256x1x1xf16>, tensor<1x256x1x1x!qElemType>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.15016976150811887:128>

// CHECK-LABEL: @SkipConcatPatternWithMultiResultProducer
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x512x1x1xf16>)
func.func @SkipConcatPatternWithMultiResultProducer(%arg0: tensor<1x512x1x1xf16>) -> (tensor<1x512x1x1xf16>, tensor<1x256x1x1x!qElemType>) {
  %0 = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x512x1x1xf16> -> tensor<1x512x1x1x!qElemType>
  %1 = IE.AffineReshape(%0) {dim_mapping = [[0], [1], [1], [1]], shape_value = [1, 512]} : tensor<1x512x1x1x!qElemType> -> tensor<1x512x!qElemType>
  %2:2 = IE.Split(%1) {axis_value = 1 : i64, num_splits = 2 : i64} : tensor<1x512x!qElemType> -> tensor<1x256x!qElemType>, tensor<1x256x!qElemType>
  %3 = IE.AffineReshape(%2#0) {dim_mapping = [[0], [1, 2, 3]], shape_value = [1, 256, 1, 1]} : tensor<1x256x!qElemType> -> tensor<1x256x1x1x!qElemType>
  %4 = IE.AffineReshape(%2#1) {dim_mapping = [[0], [1, 2, 3]], shape_value = [1, 256, 1, 1]} : tensor<1x256x!qElemType> -> tensor<1x256x1x1x!qElemType>
  %5 = IE.Concat(%3, %3) {static_offsets = [[0, 0, 0, 0], [0, 256, 0, 0]]} : tensor<1x256x1x1x!qElemType>, tensor<1x256x1x1x!qElemType> -> tensor<1x512x1x1x!qElemType>
  %6 = IE.Dequantize(%5) {dstElemType = f16} : tensor<1x512x1x1x!qElemType> -> tensor<1x512x1x1xf16>
  return %6, %4 : tensor<1x512x1x1xf16>, tensor<1x256x1x1x!qElemType>

  // CHECK: [[VAL0:%.+]] = IE.Quantize([[ARG_0]]) {dstElemType = !qElemType} : tensor<1x512x1x1xf16> -> tensor<1x512x1x1x!qElemType>
  // CHECK: [[VAL1:%.+]] = IE.AffineReshape([[VAL0]])
  // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1], [1], [1]], shape_value = [1, 512]} : tensor<1x512x1x1x!qElemType> -> tensor<1x512x!qElemType>
  // CHECK: [[VAL2:%.+]]:2 = IE.Split([[VAL1]]) {axis_value = 1 : i64, num_splits = 2 : i64} : tensor<1x512x!qElemType> -> tensor<1x256x!qElemType>, tensor<1x256x!qElemType>
  // CHECK: [[VAL3:%.+]] = IE.AffineReshape([[VAL2]]#0)
  // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1, 2, 3]], shape_value = [1, 256, 1, 1]} : tensor<1x256x!qElemType> -> tensor<1x256x1x1x!qElemType>
  // CHECK: [[VAL4:%.+]] = IE.AffineReshape([[VAL2]]#1)
  // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1, 2, 3]], shape_value = [1, 256, 1, 1]} : tensor<1x256x!qElemType> -> tensor<1x256x1x1x!qElemType>
  // CHECK: [[VAL5:%.+]] = IE.Concat([[VAL3]], [[VAL3]])
  // CHECK-SAME{LITERAL}: {static_offsets = [[0, 0, 0, 0], [0, 256, 0, 0]]} : tensor<1x256x1x1x!qElemType>, tensor<1x256x1x1x!qElemType> -> tensor<1x512x1x1x!qElemType>
  // CHECK: [[VAL6:%.+]] = IE.Dequantize([[VAL5]]) {dstElemType = f16} : tensor<1x512x1x1x!qElemType> -> tensor<1x512x1x1xf16>
  // CHECK: return [[VAL6]], [[VAL4]] : tensor<1x512x1x1xf16>, tensor<1x256x1x1x!qElemType>
}
