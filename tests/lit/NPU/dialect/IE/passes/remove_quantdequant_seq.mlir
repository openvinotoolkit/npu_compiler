//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --remove-quantdequant-seq %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
!qElemType = !quant.uniform<u8<1:255>:f16:0, {0.010680671751968504:128,0.0081200787401574797:128,0.010596087598425197:128}>
!qElemType1 = !quant.uniform<u8:f16, 1.1534313725490195:128>
!qElemType2 = !quant.uniform<u8:f16, 2.4627450980392158>
!qElemType3 = !quant.uniform<u8:f16, 0.0039215686274509803>

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @RemoveQuantDequantSequence
func.func @RemoveQuantDequantSequence(%arg0: tensor<1x3x16x16xf16>) -> (tensor<1x3x14x14xf16>, tensor<1x3x14x14xf16>)  {
  %1 = IE.Quantize(%arg0) {dstElemType = !qElemType1} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!qElemType1>
  %2 = IE.Dequantize(%1) {dstElemType = f16} : tensor<1x3x16x16x!qElemType1> -> tensor<1x3x16x16xf16>
  %weights = const.Declare tensor<3x3x3x3x!qElemType> = dense<1.0> : tensor<3x3x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]
  %3 = IE.Dequantize(%weights) {dstElemType = f16} : tensor<3x3x3x3x!qElemType> -> tensor<3x3x3x3xf16>
  %4 = IE.Convolution(%2, %3) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x16x16xf16>, tensor<3x3x3x3xf16> -> tensor<1x3x14x14xf16>
  %5 = IE.Quantize(%4) {dstElemType = !qElemType2}: tensor<1x3x14x14xf16> -> tensor<1x3x14x14x!qElemType2>
  %6 = IE.Dequantize(%5) {dstElemType = f16} : tensor<1x3x14x14x!qElemType2> -> tensor<1x3x14x14xf16>

  return %6, %4 : tensor<1x3x14x14xf16>, tensor<1x3x14x14xf16>

  //CHECK: [[VAL1:%.*]] = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!qElemType>
  //CHECK: [[VAL0:%.*]] = const.Declare tensor<3x3x3x3x!qElemType1> =
  //CHECK-SAME:                 dense<1.000000e+00> : tensor<3x3x3x3xf16>,
  //CHECK-SAME:                 [#const.CastElemType<ui8>, #const.CastElemType<!qElemType1>]
  //CHECK: [[VAL2:%.*]] = IE.Dequantize(%cst) {dstElemType = f16} : tensor<3x3x3x3x!qElemType1> -> tensor<3x3x3x3xf16>

  //CHECK: [[VAL3:%.*]] = IE.Convolution(%arg0, [[VAL2]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x16x16xf16>, tensor<3x3x3x3xf16> -> tensor<1x3x14x14xf16>
  //CHECK: return [[VAL3]], [[VAL3]]
}

// CHECK-LABEL: @RemoveQuantReshapeDequantSequence
func.func @RemoveQuantReshapeDequantSequence(%arg0: tensor<1x4420x1x2xf16>, %arg1: tensor<1x4420x1x2xf16>) -> tensor<1x4420x1x2xf16> {
  %1 = IE.Quantize(%arg0) {dstElemType = !qElemType1} : tensor<1x4420x1x2xf16> -> tensor<1x4420x1x2x!qElemType1>
  %2 = IE.AffineReshape(%1) { dim_mapping = [[0, 1], [2], [2], [3]], shape_value = [1, 1, 4420, 2] } : tensor<1x4420x1x2x!qElemType1> -> tensor<1x1x4420x2x!qElemType1>
  %3 = IE.AffineReshape(%2) { dim_mapping = [[0], [0], [1, 2], [3]], shape_value = [1, 4420, 1, 2] } : tensor<1x1x4420x2x!qElemType1> -> tensor<1x4420x1x2x!qElemType1>
  %4 = IE.Dequantize(%3) {dstElemType = f16} : tensor<1x4420x1x2x!qElemType1> -> tensor<1x4420x1x2xf16>
  %5 = IE.Add(%4, %arg1)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
        tensor<1x4420x1x2xf16>, tensor<1x4420x1x2xf16> -> tensor<1x4420x1x2xf16>
  return %5 : tensor<1x4420x1x2xf16>

  //CHECK: [[VAL0:%.*]] = IE.AffineReshape(%arg0)
  //CHECK-SAME{LITERAL}: {dim_mapping = [[0, 1], [2], [2], [3]], shape_value = [1, 1, 4420, 2]} : tensor<1x4420x1x2xf16> -> tensor<1x1x4420x2xf16>
  //CHECK: [[VAL1:%.*]] = IE.AffineReshape([[VAL0]])
  //CHECK-SAME{LITERAL}: {dim_mapping = [[0], [0], [1, 2], [3]], shape_value = [1, 4420, 1, 2]} : tensor<1x1x4420x2xf16> -> tensor<1x4420x1x2xf16>
  //CHECK: [[VAL2:%.*]] = IE.Add([[VAL1]], %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4420x1x2xf16>, tensor<1x4420x1x2xf16> -> tensor<1x4420x1x2xf16>
  //CHECK: return [[VAL2]]
}

// CHECK-LABEL: @RemoveQuantReshapeDequantMaxPool
func.func @RemoveQuantReshapeDequantMaxPool(%arg0: tensor<1x64x40x112x112xf16>) -> tensor<1x64x40x112x112xf16> {
  %0 = IE.Quantize(%arg0) {dstElemType = !qElemType1} : tensor<1x64x40x112x112xf16> -> tensor<1x64x40x112x112x!qElemType1>
  %1 = IE.MaxPool(%0) {
        kernel_size = [3, 3, 3],
        pads_begin = [1, 1, 1],
        pads_end = [1, 1, 1],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 1, 1]} : tensor<1x64x40x112x112x!qElemType1> -> tensor<1x64x40x112x112x!qElemType1>
  %2 = IE.Dequantize(%1) {dstElemType = f16} : tensor<1x64x40x112x112x!qElemType1> -> tensor<1x64x40x112x112xf16>
  return %2 : tensor<1x64x40x112x112xf16>

    //CHECK: [[MAXPOOL:%.*]] = IE.MaxPool(%arg0) {kernel_size = [3, 3, 3], pads_begin = [1, 1, 1], pads_end = [1, 1, 1], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1, 1]} : tensor<1x64x40x112x112xf16> -> tensor<1x64x40x112x112xf16>
    //CHECK: return [[MAXPOOL]] : tensor<1x64x40x112x112xf16>
}

// // CHECK-LABEL: @RemoveQuantConcatDequantSeqNoElemTypeInfoOpInterface
// // CHECK-SAME: ([[INPUT0:%.+]]: tensor<1x12800x2x1xf16>, [[INPUT1:%.+]]: tensor<1x3200x2x1xf16>, [[INPUT2:%.+]]: tensor<1x800x2x1xf16>)
func.func @RemoveQuantConcatDequantSeqNoElemTypeInfoOpInterface(%arg0: tensor<1x12800x2x1xf16>, %arg1: tensor<1x3200x2x1xf16>, %arg2: tensor<1x800x2x1xf16>) -> tensor<1x16800x2x1xf16> {
  %0 = IE.Quantize(%arg0) {dstElemType = !quant.uniform<u8:f16, 0.0039215686274509803>} : tensor<1x12800x2x1xf16> -> tensor<1x12800x2x1x!quant.uniform<u8:f16, 0.0039215686274509803>>
  %1 = IE.Quantize(%arg1) {dstElemType = !quant.uniform<u8:f16, 0.0039215686274509803>} : tensor<1x3200x2x1xf16> -> tensor<1x3200x2x1x!quant.uniform<u8:f16, 0.0039215686274509803>>
  %2 = IE.Quantize(%arg2) {dstElemType = !quant.uniform<u8:f16, 0.0039215686274509803>} : tensor<1x800x2x1xf16> -> tensor<1x800x2x1x!quant.uniform<u8:f16, 0.0039215686274509803>>
  %3 = IE.Concat(%0, %1, %2) {static_offsets = [[0, 0, 0, 0], [0, 12800, 0, 0], [0, 16000, 0, 0]]} : tensor<1x12800x2x1x!quant.uniform<u8:f16, 0.0039215686274509803>>, tensor<1x3200x2x1x!quant.uniform<u8:f16, 0.0039215686274509803>>, tensor<1x800x2x1x!quant.uniform<u8:f16, 0.0039215686274509803>> -> tensor<1x16800x2x1x!quant.uniform<u8:f16, 0.0039215686274509803>>
  %4 = IE.Dequantize(%3) {dstElemType = f16} : tensor<1x16800x2x1x!quant.uniform<u8:f16, 0.0039215686274509803>> -> tensor<1x16800x2x1xf16>

  return %4 : tensor<1x16800x2x1xf16>

  //CHECK: [[VAL0:%.+]] = IE.Concat([[INPUT0]], [[INPUT1]], [[INPUT2]])
  //CHECK-SAME{LITERAL}: {static_offsets = [[0, 0, 0, 0], [0, 12800, 0, 0], [0, 16000, 0, 0]]} : tensor<1x12800x2x1xf16>, tensor<1x3200x2x1xf16>, tensor<1x800x2x1xf16> -> tensor<1x16800x2x1xf16>
  //CHECK: return [[VAL0]]
}

// CHECK-LABEL: @RemoveQuantConcatDequantSeq
// CHECK-SAME: ([[INPUT0:%.+]]: tensor<1x12800x2x1xf16>, [[INPUT1:%.+]]: tensor<1x3200x2x1xf16>, [[INPUT2:%.+]]: tensor<1x800x2x1xf16>)
func.func @RemoveQuantConcatDequantSeq(%arg0: tensor<1x12800x2x1xf16>, %arg1: tensor<1x3200x2x1xf16>, %arg2: tensor<1x800x2x1xf16>) -> tensor<1x16800x2x1xf16> {
  %0 = IE.Quantize(%arg0) {dstElemType = !quant.uniform<u8:f16, 0.0039215686274509803>} : tensor<1x12800x2x1xf16> -> tensor<1x12800x2x1x!quant.uniform<u8:f16, 0.0039215686274509803>>
  %1 = IE.AffineReshape(%0) {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 1, 12800, 2]} : tensor<1x12800x2x1x!quant.uniform<u8:f16, 0.0039215686274509803>> -> tensor<1x1x12800x2x!quant.uniform<u8:f16, 0.0039215686274509803>>
  %2 = IE.Quantize(%arg1) {dstElemType = !quant.uniform<u8:f16, 0.0039215686274509803>} : tensor<1x3200x2x1xf16> -> tensor<1x3200x2x1x!quant.uniform<u8:f16, 0.0039215686274509803>>
  %3 = IE.AffineReshape(%2) {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 1, 3200, 2]} : tensor<1x3200x2x1x!quant.uniform<u8:f16, 0.0039215686274509803>> -> tensor<1x1x3200x2x!quant.uniform<u8:f16, 0.0039215686274509803>>
  %4 = IE.Quantize(%arg2) {dstElemType = !quant.uniform<u8:f16, 0.0039215686274509803>} : tensor<1x800x2x1xf16> -> tensor<1x800x2x1x!quant.uniform<u8:f16, 0.0039215686274509803>>
  %5 = IE.AffineReshape(%4) {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 1, 800, 2]} : tensor<1x800x2x1x!quant.uniform<u8:f16, 0.0039215686274509803>> -> tensor<1x1x800x2x!quant.uniform<u8:f16, 0.0039215686274509803>>
  %6 = IE.Concat(%1, %3, %5) {static_offsets = [[0, 0, 0, 0], [0, 0, 12800, 0], [0, 0, 16000, 0]]} : tensor<1x1x12800x2x!quant.uniform<u8:f16, 0.0039215686274509803>>, tensor<1x1x3200x2x!quant.uniform<u8:f16, 0.0039215686274509803>>, tensor<1x1x800x2x!quant.uniform<u8:f16, 0.0039215686274509803>> -> tensor<1x1x16800x2x!quant.uniform<u8:f16, 0.0039215686274509803>>
  %7 = IE.AffineReshape(%6) {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 16800, 2, 1]} : tensor<1x1x16800x2x!quant.uniform<u8:f16, 0.0039215686274509803>> -> tensor<1x16800x2x1x!quant.uniform<u8:f16, 0.0039215686274509803>>
  %8 = IE.Dequantize(%7) {dstElemType = f16} : tensor<1x16800x2x1x!quant.uniform<u8:f16, 0.0039215686274509803>> -> tensor<1x16800x2x1xf16>
  return %8 : tensor<1x16800x2x1xf16>

  //CHECK: [[VAL0:%.+]] = IE.AffineReshape([[INPUT0]])
  //CHECK-SAME{LITERAL}: {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 1, 12800, 2]} : tensor<1x12800x2x1xf16> -> tensor<1x1x12800x2xf16>
  //CHECK: [[VAL1:%.+]] = IE.AffineReshape([[INPUT1]])
  //CHECK-SAME{LITERAL}: {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 1, 3200, 2]} : tensor<1x3200x2x1xf16> -> tensor<1x1x3200x2xf16>
  //CHECK: [[VAL2:%.+]] = IE.AffineReshape([[INPUT2]])
  //CHECK-SAME{LITERAL}: {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 1, 800, 2]} : tensor<1x800x2x1xf16> -> tensor<1x1x800x2xf16>
  //CHECK: [[VAL3:%.+]] = IE.Concat([[VAL0]], [[VAL1]], [[VAL2]])
  //CHECK-SAME{LITERAL}: {static_offsets = [[0, 0, 0, 0], [0, 0, 12800, 0], [0, 0, 16000, 0]]} : tensor<1x1x12800x2xf16>, tensor<1x1x3200x2xf16>, tensor<1x1x800x2xf16> -> tensor<1x1x16800x2xf16>
  //CHECK: [[VAL4:%.+]] = IE.AffineReshape([[VAL3]])
  //CHECK-SAME{LITERAL}: {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 16800, 2, 1]} : tensor<1x1x16800x2xf16> -> tensor<1x16800x2x1xf16>
  //CHECK: return [[VAL4]]
}

// CHECK-LABEL: @DontRemoveQuantConcatElemTypeConcatDequantSeq
// CHECK-SAME: ([[INPUT0:%.+]]: tensor<1x12800x2x1xf16>, [[INPUT1:%.+]]: tensor<1x3200x2x1xf16>, [[INPUT2:%.+]]: tensor<1x800x2x1xf16>)
func.func @DontRemoveQuantConcatElemTypeConcatDequantSeq(%arg0: tensor<1x12800x2x1xf16>, %arg1: tensor<1x3200x2x1xf16>, %arg2: tensor<1x800x2x1xf16>) -> tensor<1x1x17600x2xf16> {
  %0 = IE.Quantize(%arg0) {dstElemType = !quant.uniform<u8:f16, 0.0039215686274509803>} : tensor<1x12800x2x1xf16> -> tensor<1x12800x2x1x!quant.uniform<u8:f16, 0.0039215686274509803>>
  %1 = IE.AffineReshape(%0) {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 1, 12800, 2]} : tensor<1x12800x2x1x!quant.uniform<u8:f16, 0.0039215686274509803>> -> tensor<1x1x12800x2x!quant.uniform<u8:f16, 0.0039215686274509803>>
  %2 = IE.Quantize(%arg1) {dstElemType = !quant.uniform<u8:f16, 0.0039215686274509803>} : tensor<1x3200x2x1xf16> -> tensor<1x3200x2x1x!quant.uniform<u8:f16, 0.0039215686274509803>>
  %3 = IE.AffineReshape(%2) {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 1, 3200, 2]} : tensor<1x3200x2x1x!quant.uniform<u8:f16, 0.0039215686274509803>> -> tensor<1x1x3200x2x!quant.uniform<u8:f16, 0.0039215686274509803>>
  %4 = IE.Quantize(%arg2) {dstElemType = !quant.uniform<u8:f16, 0.0039215686274509803>} : tensor<1x800x2x1xf16> -> tensor<1x800x2x1x!quant.uniform<u8:f16, 0.0039215686274509803>>
  %5 = IE.AffineReshape(%4) {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 1, 800, 2]} : tensor<1x800x2x1x!quant.uniform<u8:f16, 0.0039215686274509803>> -> tensor<1x1x800x2x!quant.uniform<u8:f16, 0.0039215686274509803>>
  %6 = IE.Concat(%1, %3, %5) {static_offsets = [[0, 0, 0, 0], [0, 0, 12800, 0], [0, 0, 16000, 0]]} : tensor<1x1x12800x2x!quant.uniform<u8:f16, 0.0039215686274509803>>, tensor<1x1x3200x2x!quant.uniform<u8:f16, 0.0039215686274509803>>, tensor<1x1x800x2x!quant.uniform<u8:f16, 0.0039215686274509803>> -> tensor<1x1x16800x2x!quant.uniform<u8:f16, 0.0039215686274509803>>
  %7 = IE.AffineReshape(%6) {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 16800, 2, 1]} : tensor<1x1x16800x2x!quant.uniform<u8:f16, 0.0039215686274509803>> -> tensor<1x16800x2x1x!quant.uniform<u8:f16, 0.0039215686274509803>>
  %8 = IE.AffineReshape(%7) {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 1, 16800, 2]} : tensor<1x16800x2x1x!quant.uniform<u8:f16, 0.0039215686274509803>> -> tensor<1x1x16800x2x!quant.uniform<u8:f16, 0.0039215686274509803>>
  %9 = IE.Concat(%5, %8) {static_offsets = [[0, 0, 0, 0], [0, 0, 800, 0]]} : tensor<1x1x800x2x!quant.uniform<u8:f16, 0.0039215686274509803>>, tensor<1x1x16800x2x!quant.uniform<u8:f16, 0.0039215686274509803>> -> tensor<1x1x17600x2x!quant.uniform<u8:f16, 0.0039215686274509803>>
  %10 = IE.Dequantize(%9) {dstElemType = f16} : tensor<1x1x17600x2x!quant.uniform<u8:f16, 0.0039215686274509803>> -> tensor<1x1x17600x2xf16>
  return %10 : tensor<1x1x17600x2xf16>

  //CHECK: [[VAL0:%.+]] = IE.Quantize([[INPUT0]])
  //CHECK-SAME{LITERAL}: {dstElemType = !qElemType3} : tensor<1x12800x2x1xf16> -> tensor<1x12800x2x1x!qElemType3>
  //CHECK: [[VAL1:%.+]] = IE.AffineReshape([[VAL0]])
  //CHECK-SAME{LITERAL}: {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 1, 12800, 2]} : tensor<1x12800x2x1x!qElemType3> -> tensor<1x1x12800x2x!qElemType3>
  //CHECK: [[VAL2:%.+]] = IE.Quantize([[INPUT1]])
  //CHECK-SAME{LITERAL}: {dstElemType = !qElemType3} : tensor<1x3200x2x1xf16> -> tensor<1x3200x2x1x!qElemType3>
  //CHECK: [[VAL3:%.+]] = IE.AffineReshape([[VAL2]])
  //CHECK-SAME{LITERAL}: {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 1, 3200, 2]} : tensor<1x3200x2x1x!qElemType3> -> tensor<1x1x3200x2x!qElemType3>
  //CHECK: [[VAL4:%.+]] = IE.Quantize([[INPUT2]])
  //CHECK-SAME{LITERAL}: {dstElemType = !qElemType3} : tensor<1x800x2x1xf16> -> tensor<1x800x2x1x!qElemType3>
  //CHECK: [[VAL5:%.+]] = IE.AffineReshape([[VAL4]])
  //CHECK-SAME{LITERAL}: {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 1, 800, 2]} : tensor<1x800x2x1x!qElemType3> -> tensor<1x1x800x2x!qElemType3>
  //CHECK: [[VAL6:%.+]] = IE.Concat([[VAL1]], [[VAL3]], [[VAL5]])
  //CHECK-SAME{LITERAL}: {static_offsets = [[0, 0, 0, 0], [0, 0, 12800, 0], [0, 0, 16000, 0]]} : tensor<1x1x12800x2x!qElemType3>, tensor<1x1x3200x2x!qElemType3>, tensor<1x1x800x2x!qElemType3> -> tensor<1x1x16800x2x!qElemType3>
  //CHECK: [[VAL7:%.+]] = IE.AffineReshape([[VAL6]])
  //CHECK-SAME{LITERAL}: {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 16800, 2, 1]} : tensor<1x1x16800x2x!qElemType3> -> tensor<1x16800x2x1x!qElemType3>
  //CHECK: [[VAL8:%.+]] = IE.AffineReshape([[VAL7]])
  //CHECK-SAME{LITERAL}: {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 1, 16800, 2]} : tensor<1x16800x2x1x!qElemType3> -> tensor<1x1x16800x2x!qElemType3>
  //CHECK: [[VAL9:%.+]] = IE.Concat([[VAL5]], [[VAL8]])
  //CHECK-SAME{LITERAL}: {static_offsets = [[0, 0, 0, 0], [0, 0, 800, 0]]} : tensor<1x1x800x2x!qElemType3>, tensor<1x1x16800x2x!qElemType3> -> tensor<1x1x17600x2x!qElemType3>
  //CHECK: [[VAL10:%.+]] = IE.Dequantize([[VAL9]])
  //CHECK-SAME{LITERAL}: {dstElemType = f16} : tensor<1x1x17600x2x!qElemType3> -> tensor<1x1x17600x2xf16>
  //CHECK: return [[VAL10]]
}

// CHECK-LABEL: @RemoveQuantConcatMultipleElemTypeDequantSeq
// CHECK-SAME: ([[INPUT0:%.+]]: tensor<1x12800x2x1xf16>, [[INPUT1:%.+]]: tensor<1x3200x2x1xf16>, [[INPUT2:%.+]]: tensor<1x800x2x1xf16>)
func.func @RemoveQuantConcatMultipleElemTypeDequantSeq(%arg0: tensor<1x12800x2x1xf16>, %arg1: tensor<1x3200x2x1xf16>, %arg2: tensor<1x800x2x1xf16>) -> tensor<1x16800x2x1xf16> {
  %0 = IE.Quantize(%arg0) {dstElemType = !quant.uniform<u8:f16, 0.0039215686274509803>} : tensor<1x12800x2x1xf16> -> tensor<1x12800x2x1x!quant.uniform<u8:f16, 0.0039215686274509803>>
  %1 = IE.AffineReshape(%0) {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 1, 12800, 2]} : tensor<1x12800x2x1x!quant.uniform<u8:f16, 0.0039215686274509803>> -> tensor<1x1x12800x2x!quant.uniform<u8:f16, 0.0039215686274509803>>
  %2 = IE.Reorder(%1) {dstOrder = #NCHW} : tensor<1x1x12800x2x!quant.uniform<u8:f16, 0.0039215686274509803>> -> tensor<1x1x12800x2x!quant.uniform<u8:f16, 0.0039215686274509803>>
  %3 = IE.Quantize(%arg1) {dstElemType = !quant.uniform<u8:f16, 0.0039215686274509803>} : tensor<1x3200x2x1xf16> -> tensor<1x3200x2x1x!quant.uniform<u8:f16, 0.0039215686274509803>>
  %4 = IE.AffineReshape(%3) {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 1, 3200, 2]} : tensor<1x3200x2x1x!quant.uniform<u8:f16, 0.0039215686274509803>> -> tensor<1x1x3200x2x!quant.uniform<u8:f16, 0.0039215686274509803>>
  %5 = IE.Reorder(%4) {dstOrder = #NCHW} : tensor<1x1x3200x2x!quant.uniform<u8:f16, 0.0039215686274509803>> -> tensor<1x1x3200x2x!quant.uniform<u8:f16, 0.0039215686274509803>>
  %6 = IE.Quantize(%arg2) {dstElemType = !quant.uniform<u8:f16, 0.0039215686274509803>} : tensor<1x800x2x1xf16> -> tensor<1x800x2x1x!quant.uniform<u8:f16, 0.0039215686274509803>>
  %7 = IE.AffineReshape(%6) {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 1, 800, 2]} : tensor<1x800x2x1x!quant.uniform<u8:f16, 0.0039215686274509803>> -> tensor<1x1x800x2x!quant.uniform<u8:f16, 0.0039215686274509803>>
  %8 = IE.Reorder(%7) {dstOrder = #NCHW} : tensor<1x1x800x2x!quant.uniform<u8:f16, 0.0039215686274509803>> -> tensor<1x1x800x2x!quant.uniform<u8:f16, 0.0039215686274509803>>
  %9 = IE.Concat(%2, %5, %8) {static_offsets = [[0, 0, 0, 0], [0, 0, 12800, 0], [0, 0, 16000, 0]]} : tensor<1x1x12800x2x!quant.uniform<u8:f16, 0.0039215686274509803>>, tensor<1x1x3200x2x!quant.uniform<u8:f16, 0.0039215686274509803>>, tensor<1x1x800x2x!quant.uniform<u8:f16, 0.0039215686274509803>> -> tensor<1x1x16800x2x!quant.uniform<u8:f16, 0.0039215686274509803>>
  %10 = IE.AffineReshape(%9) {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 16800, 2, 1]} : tensor<1x1x16800x2x!quant.uniform<u8:f16, 0.0039215686274509803>> -> tensor<1x16800x2x1x!quant.uniform<u8:f16, 0.0039215686274509803>>
  %11 = IE.Dequantize(%10) {dstElemType = f16} : tensor<1x16800x2x1x!quant.uniform<u8:f16, 0.0039215686274509803>> -> tensor<1x16800x2x1xf16>
  return %11 : tensor<1x16800x2x1xf16>

  //CHECK: [[VAL0:%.+]] = IE.AffineReshape([[INPUT0]])
  //CHECK-SAME{LITERAL}: {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 1, 12800, 2]} : tensor<1x12800x2x1xf16> -> tensor<1x1x12800x2xf16>
  //CHECK: [[VAL1:%.+]] = IE.Reorder([[VAL0]])
  //CHECK-SAME{LITERAL}: {dstOrder = #NCHW} : tensor<1x1x12800x2xf16> -> tensor<1x1x12800x2xf16>
  //CHECK: [[VAL2:%.+]] = IE.AffineReshape([[INPUT1]])
  //CHECK-SAME{LITERAL}: {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 1, 3200, 2]} : tensor<1x3200x2x1xf16> -> tensor<1x1x3200x2xf16>
  //CHECK: [[VAL3:%.+]] = IE.Reorder([[VAL2]])
  //CHECK-SAME{LITERAL}: {dstOrder = #NCHW} : tensor<1x1x3200x2xf16> -> tensor<1x1x3200x2xf16>
  //CHECK: [[VAL4:%.+]] = IE.AffineReshape([[INPUT2]])
  //CHECK-SAME{LITERAL}: {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 1, 800, 2]} : tensor<1x800x2x1xf16> -> tensor<1x1x800x2xf16>
  //CHECK: [[VAL5:%.+]] = IE.Reorder([[VAL4]])
  //CHECK-SAME{LITERAL}: {dstOrder = #NCHW} : tensor<1x1x800x2xf16> -> tensor<1x1x800x2xf16>
  //CHECK: [[VAL6:%.+]] = IE.Concat([[VAL1]], [[VAL3]], [[VAL5]])
  //CHECK-SAME{LITERAL}: {static_offsets = [[0, 0, 0, 0], [0, 0, 12800, 0], [0, 0, 16000, 0]]} : tensor<1x1x12800x2xf16>, tensor<1x1x3200x2xf16>, tensor<1x1x800x2xf16> -> tensor<1x1x16800x2xf16>
  //CHECK: [[VAL7:%.+]] = IE.AffineReshape([[VAL6]])
  //CHECK-SAME{LITERAL}: {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 16800, 2, 1]} : tensor<1x1x16800x2xf16> -> tensor<1x16800x2x1xf16>
  //CHECK: return [[VAL7]]
}
