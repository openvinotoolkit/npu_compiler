//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --expand-activation-channels %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: func.func @ExpandICMatMul([[INPUT:%.+]]: tensor<48x4x50x49xf16>) ->  tensor<48x4x50x32xf16> {
func.func @ExpandICMatMul(%input: tensor<48x4x50x49xf16>) -> tensor<48x4x50x32xf16> {
    %cst = const.Declare tensor<48x4x49x32xf16> =
        dense<1.0> : tensor<48x4x49x32xf16>
    %matmul = IE.MatMul(%input, %cst) : tensor<48x4x50x49xf16>, tensor<48x4x49x32xf16> -> tensor<48x4x50x32xf16>

    return %matmul : tensor<48x4x50x32xf16>
    // CHECK:       [[EXPAND:%.+]] = IE.Expand([[INPUT]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 15]}
    // CHECK-SAME:  : tensor<48x4x50x49xf16> -> tensor<48x4x50x64xf16>
    // CHECK:       [[CST:%.+]] = const.Declare tensor<48x4x64x32xf16> = dense<1.000000e+00> :
    // CHECK-SAME:  tensor<48x4x49x32xf16>, [#const.PadWithZero<[0, 0, 0, 0], [0, 0, 15, 0]>]
    // CHECK:       [[MATMUL:%.+]] = IE.MatMul([[EXPAND]], [[CST]])
    // CHECK-SAME:  : tensor<48x4x50x64xf16>, tensor<48x4x64x32xf16> -> tensor<48x4x50x32xf16>
    // CHECK:       [[SLICE:%.+]] = IE.Slice [[MATMUL]] [0, 0, 0, 0] [48, 4, 50, 32]
    // CHECK-SAME:  : tensor<48x4x50x32xf16> to tensor<48x4x50x32xf16>
    // CHECK:       return [[SLICE]] : tensor<48x4x50x32xf16>
}

// -----

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

// CHECK:       func.func @ExpandICMatMulWithTranspose([[INPUT1:%.+]]: tensor<48x4x50x49xf16>,
// CHECK-SAME:  [[INPUT2:%.+]]: tensor<48x4x49x32xf16>) ->  tensor<48x4x50x32xf16> {
func.func @ExpandICMatMulWithTranspose(%input1: tensor<48x4x50x49xf16>, %input2: tensor<48x4x49x32xf16>) -> tensor<48x4x50x32xf16> {
    %transpose = IE.Transpose(%input2) {order_value = #NCWH} : tensor<48x4x49x32xf16> -> tensor<48x4x32x49xf16>
    %matmul = IE.MatMul(%input1, %transpose) {transpose_b} : tensor<48x4x50x49xf16>, tensor<48x4x32x49xf16> -> tensor<48x4x50x32xf16>

    return %matmul : tensor<48x4x50x32xf16>
    // CHECK:       [[TRANSPOSE:%.+]] = IE.Transpose([[INPUT2]]) {order_value = #NCWH}
    // CHECK-SAME:  : tensor<48x4x49x32xf16> -> tensor<48x4x32x49xf16>
    // CHECK:       [[EXPAND_IN1:%.+]] = IE.Expand([[INPUT1]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 15]}
    // CHECK-SAME:  : tensor<48x4x50x49xf16> -> tensor<48x4x50x64xf16>
    // CHECK:       [[EXPAND_IN2:%.+]] = IE.Expand([[TRANSPOSE]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 15]}
    // CHECK-SAME:  : tensor<48x4x32x49xf16> -> tensor<48x4x32x64xf16>
    // CHECK:       [[MATMUL:%.+]] = IE.MatMul([[EXPAND_IN1]], [[EXPAND_IN2]]) {transpose_b}
    // CHECK-SAME:  : tensor<48x4x50x64xf16>, tensor<48x4x32x64xf16> -> tensor<48x4x50x32xf16>
    // CHECK:       [[SLICE:%.+]] = IE.Slice [[MATMUL]] [0, 0, 0, 0] [48, 4, 50, 32]
    // CHECK-SAME:  : tensor<48x4x50x32xf16> to tensor<48x4x50x32xf16>
    // CHECK:       return [[SLICE]] : tensor<48x4x50x32xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

// CHECK:       func.func @ExpandOCMatMulWithTranspose([[INPUT1:%.+]]: tensor<16x8x49x32xf16>,
// CHECK-SAME:  [[INPUT2:%.+]]: tensor<16x8x32x50xf16>) ->  tensor<16x8x49x50xf16> {
func.func @ExpandOCMatMulWithTranspose(%input1: tensor<16x8x49x32xf16>, %input2: tensor<16x8x32x50xf16>) -> tensor<16x8x49x50xf16>  {
    %transpose = IE.Transpose(%input2) {order_value = #NCWH} : tensor<16x8x32x50xf16> -> tensor<16x8x50x32xf16>
    %matmul = IE.MatMul(%input1, %transpose) {transpose_b} : tensor<16x8x49x32xf16>, tensor<16x8x50x32xf16> -> tensor<16x8x49x50xf16>

    return %matmul : tensor<16x8x49x50xf16>
    // CHECK:       [[TRANSPOSE:%.+]] = IE.Transpose([[INPUT2]]) {order_value = #NCWH}
    // CHECK-SAME:  : tensor<16x8x32x50xf16> -> tensor<16x8x50x32xf16>
    // CHECK:       [[EXPAND:%.+]] = IE.Expand([[TRANSPOSE]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 14, 0]}
    // CHECK-SAME:  : tensor<16x8x50x32xf16> -> tensor<16x8x64x32xf16>
    // CHECK:       [[MATMUL:%.+]] = IE.MatMul([[INPUT1]], [[EXPAND]]) {transpose_b}
    // CHECK-SAME:  : tensor<16x8x49x32xf16>, tensor<16x8x64x32xf16> -> tensor<16x8x49x64xf16>
    // CHECK:       [[SLICE:%.+]] = IE.Slice [[MATMUL]] [0, 0, 0, 0] [16, 8, 49, 50]
    // CHECK-SAME:  : tensor<16x8x49x64xf16> to tensor<16x8x49x50xf16>
    // CHECK:       return [[SLICE]] : tensor<16x8x49x50xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

// CHECK:       func.func @ExpandICOCMatMulWithTranspose([[INPUT1:%.+]]:  tensor<16x8x49x25xf16>,
// CHECK-SAME:  [[INPUT2:%.+]]: tensor<16x8x25x49xf16>) ->  tensor<16x8x49x49xf16> {
func.func @ExpandICOCMatMulWithTranspose(%input1: tensor<16x8x49x25xf16>, %input2: tensor<16x8x25x49xf16>) -> tensor<16x8x49x49xf16>  {
    %transpose = IE.Transpose(%input2) {order_value = #NCWH} : tensor<16x8x25x49xf16> -> tensor<16x8x49x25xf16>
    %matmul = IE.MatMul(%input1, %transpose) {transpose_b} : tensor<16x8x49x25xf16>, tensor<16x8x49x25xf16> -> tensor<16x8x49x49xf16>

    return %matmul : tensor<16x8x49x49xf16>
    // CHECK:       [[TRANSPOSE:%.+]] = IE.Transpose([[INPUT2]]) {order_value = #NCWH}
    // CHECK-SAME:  : tensor<16x8x25x49xf16> -> tensor<16x8x49x25xf16>
    // CHECK:       [[EXPAND_IC_IN1:%.+]] = IE.Expand([[INPUT1]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 7]}
    // CHECK-SAME:  : tensor<16x8x49x25xf16> -> tensor<16x8x49x32xf16>
    // CHECK:       [[EXPAND_IC_IN2:%.+]] = IE.Expand([[TRANSPOSE]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 7]}
    // CHECK-SAME:  : tensor<16x8x49x25xf16> -> tensor<16x8x49x32xf16>
    // CHECK:       [[EXPAND_OC_IN2:%.+]] = IE.Expand([[EXPAND_IC_IN2]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 15, 0]}
    // CHECK-SAME:  : tensor<16x8x49x32xf16> -> tensor<16x8x64x32xf16>
    // CHECK:       [[MATMUL:%.+]] = IE.MatMul([[EXPAND_IC_IN1]], [[EXPAND_OC_IN2]]) {transpose_b}
    // CHECK-SAME:  : tensor<16x8x49x32xf16>, tensor<16x8x64x32xf16> -> tensor<16x8x49x64xf16>
    // CHECK:       [[SLICE:%.+]] = IE.Slice [[MATMUL]] [0, 0, 0, 0] [16, 8, 49, 49]
    // CHECK-SAME:  : tensor<16x8x49x64xf16> to tensor<16x8x49x49xf16>
    // CHECK:       return [[SLICE]] : tensor<16x8x49x49xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK:       func.func @ExpandICOCMatMul([[INPUT:%.+]]:  tensor<16x8x49x25xf16>
// CHECK-SAME:  ->  tensor<16x8x49x49xf16> {
func.func @ExpandICOCMatMul(%input: tensor<16x8x49x25xf16>) -> tensor<16x8x49x49xf16>  {
    %cst = const.Declare tensor<16x8x49x25xf16> =
        dense<1.0> : tensor<16x8x49x25xf16>
    %matmul = IE.MatMul(%input, %cst) {transpose_b} : tensor<16x8x49x25xf16>, tensor<16x8x49x25xf16> -> tensor<16x8x49x49xf16>

    return %matmul : tensor<16x8x49x49xf16>
    // CHECK:           [[EXPAND:%.+]] = IE.Expand([[INPUT]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 7]}
    // CHECK-SAME: :    tensor<16x8x49x25xf16> -> tensor<16x8x49x32xf16>
    // CHECK:           [[CST:%.+]] = const.Declare tensor<16x8x64x32xf16> = dense<1.000000e+00> : tensor<16x8x49x25xf16>,
    // CHECK-SAME:      [#const.PadWithZero<[0, 0, 0, 0], [0, 0, 0, 7]>, #const.PadWithZero<[0, 0, 0, 0], [0, 0, 15, 0]>]
    // CHECK:           [[MATMUL:%.+]] = IE.MatMul([[EXPAND]], [[CST]]) {transpose_b}
    // CHECK-SAME:      : tensor<16x8x49x32xf16>, tensor<16x8x64x32xf16> -> tensor<16x8x49x64xf16>
    // CHECK:           [[SLICE:%.+]] = IE.Slice [[MATMUL]] [0, 0, 0, 0] [16, 8, 49, 49]
    // CHECK-SAME:      : tensor<16x8x49x64xf16> to tensor<16x8x49x49xf16>
    // CHECK:           return [[SLICE]] : tensor<16x8x49x49xf16>
}

// -----

!input1ElemType = !quant.uniform<u8:f16, 0.1:1>
!input2ElemType = !quant.uniform<u8:f16, 0.2:2>
!outputElemType = !quant.uniform<u8:f16, 0.3:3>
// CHECK: [[INPUT1_ELEM_TYPE:!.+]] = !quant.uniform<u8:f16, 1.000000e-01:1>
// CHECK: [[INPUT2_ELEM_TYPE:!.+]] = !quant.uniform<u8:f16, 2.000000e-01:2>
// CHECK: [[OUTPUT_ELEM_TYPE:!.+]] = !quant.uniform<u8:f16, 3.000000e-01:3>

// CHECK:       func.func @ExpandOCMatMulQuant([[INPUT1:%.+]]: tensor<1x32x75x16x[[INPUT1_ELEM_TYPE]]>,
// CHECK-SAME:  [[INPUT2:%.+]]: tensor<1x32x75x16x[[INPUT2_ELEM_TYPE]]>) ->  tensor<1x32x75x75x[[OUTPUT_ELEM_TYPE]]> {
func.func @ExpandOCMatMulQuant(%input1: tensor<1x32x75x16x!input1ElemType>, %input2: tensor<1x32x75x16x!input2ElemType>) -> tensor<1x32x75x75x!outputElemType> {
    %matmul = IE.MatMul(%input1, %input2) {transpose_b} : tensor<1x32x75x16x!input1ElemType>, tensor<1x32x75x16x!input2ElemType> -> tensor<1x32x75x75x!outputElemType>

    return %matmul : tensor<1x32x75x75x!outputElemType>
    // CHECK:           [[EXPAND:%.+]] = IE.Expand([[INPUT2]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 5, 0]}
    // CHECK-SAME:      : tensor<1x32x75x16x[[INPUT2_ELEM_TYPE]]> -> tensor<1x32x80x16x[[INPUT2_ELEM_TYPE]]>
    // CHECK:           [[MATMUL:%.+]] = IE.MatMul([[INPUT1]], [[EXPAND]]) {transpose_b}
    // CHECK-SAME:      : tensor<1x32x75x16x[[INPUT1_ELEM_TYPE]]>, tensor<1x32x80x16x[[INPUT2_ELEM_TYPE]]> -> tensor<1x32x75x80x[[OUTPUT_ELEM_TYPE]]>
    // CHECK:           [[SLICE:%.+]] = IE.Slice [[MATMUL]] [0, 0, 0, 0] [1, 32, 75, 75]
    // CHECK-SAME:      : tensor<1x32x75x80x[[OUTPUT_ELEM_TYPE]]> to tensor<1x32x75x75x[[OUTPUT_ELEM_TYPE]]>
    // CHECK:           return [[SLICE]] : tensor<1x32x75x75x[[OUTPUT_ELEM_TYPE]]>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.010420070909986309:128>
!qElemType1 = !quant.uniform<u8:f16, 0.02042007090998631:128>

// CHECK:       func.func @ExpandQuantizedICMatMul([[INPUT_0:%.+]]: tensor<1x16x64x49x!qElemType>,
// CHECK-SAME:  [[INPUT_1:%.+]]: tensor<1x16x49x128x!qElemType1>) ->  tensor<1x16x64x128xf16> {
func.func @ExpandQuantizedICMatMul(%arg0: tensor<1x16x64x49x!qElemType>, %arg1: tensor<1x16x49x128x!qElemType1>) -> tensor<1x16x64x128xf16> {
    %matmul = IE.MatMul(%arg0, %arg1) : tensor<1x16x64x49x!qElemType>, tensor<1x16x49x128x!qElemType1> -> tensor<1x16x64x128xf16>

    return %matmul : tensor<1x16x64x128xf16>

    // CHECK-DAG:       [[CST:%.+]] = const.Declare tensor<1x16x64x15x!qElemType> = dense<128> : tensor<1x16x64x15xui8>
    // CHECK:           [[CONCAT_0:%.+]] = IE.Concat([[INPUT_0]], [[CST]]) {per_axis = #IE.Concat<axis = 3 : i64>}
    // CHECK-SAME:          : tensor<1x16x64x49x!qElemType>, tensor<1x16x64x15x!qElemType> -> tensor<1x16x64x64x!qElemType>
    // CHECK-DAG:       [[CST_0:%.+]] = const.Declare tensor<1x16x15x128x!qElemType1> = dense<128> : tensor<1x16x15x128xui8>
    // CHECK:           [[CONCAT_1:%.+]] = IE.Concat([[INPUT_1]], [[CST_0]]) {per_axis = #IE.Concat<axis = 2 : i64>}
    // CHECK-SAME:          : tensor<1x16x49x128x!qElemType1>, tensor<1x16x15x128x!qElemType1> -> tensor<1x16x64x128x!qElemType1>
    // CHECK:           [[MATMUL:%.+]] = IE.MatMul([[CONCAT_0]], [[CONCAT_1]])
    // CHECK-SAME:          : tensor<1x16x64x64x!qElemType>, tensor<1x16x64x128x!qElemType1> -> tensor<1x16x64x128xf16>
    // CHECK:           [[SLICE:%.+]] = IE.Slice [[MATMUL]] [0, 0, 0, 0] [1, 16, 64, 128]
    // CHECK-SAME:          : tensor<1x16x64x128xf16> to tensor<1x16x64x128xf16>
    // CHECK:           return [[SLICE]] : tensor<1x16x64x128xf16>
}
